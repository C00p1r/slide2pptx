"""Score saved predictions against YOLO labels, with and without detector.postprocess.

    python -m detector.predict runs/ratio/r0.50/weights/last.pt data/real/yolo/images/test \\
        --conf 0.001 --no-draw --out results/pred_eval/r0.50
    python -m detector.evalpred results/pred_eval/r0.50 --labels data/real/yolo/labels/test

Reports mAP@0.5 and mAP@0.5:0.95 (needs predictions at a low --conf, like YOLO's val),
plus precision / recall / F1 at a working threshold (--conf, default 0.25), and how
many overlapping pairs remain (two boxes of one group with IoU >= 0.5).
"""
import argparse
import glob
import json
import os

import numpy as np

from . import CLASSES
from .containers import find_cards
from .postprocess import GROUP, clean, iou

IOUS = np.linspace(0.5, 0.95, 10)


def load_gt(path, W, H):
    out = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                k, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                out.append(dict(cls=CLASSES[k], bbox=[(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]))
    return out


def match(preds, gts):
    """-> bool array (len(preds), 10): pred is a true positive at each IoU threshold (YOLO's matching)."""
    tp = np.zeros((len(preds), len(IOUS)), bool)
    if not preds or not gts:
        return tp
    m = np.array([[iou(p["bbox"], g["bbox"]) if p["cls"] == g["cls"] else 0.0 for g in gts] for p in preds])
    for t, thr in enumerate(IOUS):
        pi, gi = np.nonzero(m >= thr)
        if not len(pi):
            continue
        order = np.argsort(-m[pi, gi])
        pi, gi = pi[order], gi[order]
        _, first = np.unique(pi, return_index=True)
        pi, gi = pi[np.sort(first)], gi[np.sort(first)]
        _, first = np.unique(gi, return_index=True)
        tp[pi[first], t] = True
    return tp


SIZES = (("small", 0.0, 0.05), ("medium", 0.05, 0.15), ("large", 0.15, 9.0))


def size_of(box, W, H):
    """Bucket by sqrt(box area / slide area): small < 5%, medium 5-15%, large > 15% of the slide side."""
    r = (box[2] * box[3] / (W * H)) ** 0.5
    return next(n for n, lo, hi in SIZES if lo <= r < hi)


def why_missed(preds, gts):
    """For every ground-truth box at IoU 0.5: 'found', or why not.
    wrong class: a box of another class covers it (IoU >= 0.5)
    loose box:   a box of the right class overlaps it, but only 0.3 <= IoU < 0.5
    nothing:     no box near it"""
    out, used = [], set()
    pairs = sorted(((iou(p["bbox"], g["bbox"]), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)
                    if p["cls"] == g["cls"]), reverse=True)
    found = set()
    for o, i, j in pairs:
        if o < 0.5:
            break
        if i not in used and j not in found:
            used.add(i)
            found.add(j)
    for j, g in enumerate(gts):
        if j in found:
            out.append("found")
            continue
        best_same = max((iou(p["bbox"], g["bbox"]) for p in preds if p["cls"] == g["cls"]), default=0)
        best_other = max((iou(p["bbox"], g["bbox"]) for p in preds if p["cls"] != g["cls"]), default=0)
        out.append("wrong class" if best_other >= 0.5 else "loose box" if best_same >= 0.3 else "nothing")
    return out


def diagnose(slides, conf):
    """Recall at conf, by class and by size, and why the rest were missed."""
    keys = ["found", "wrong class", "loose box", "nothing"]
    rows = {}
    for preds, gts, W, H in slides:
        work = [p for p in preds if p["conf"] >= conf]
        for g, w in zip(gts, why_missed(work, gts)):
            for key in (g["cls"], "size: " + size_of(g["bbox"], W, H), "all"):
                rows.setdefault(key, dict.fromkeys(keys, 0))[w] += 1
    order = CLASSES + ["size: " + n for n, _, _ in SIZES] + ["all"]
    md = [f"#### Why boxes are missed (conf >= {conf}, IoU 0.5)", "",
          "| labels | n | recall | wrong class | loose box (IoU 0.3-0.5) | nothing there |", "|---|---|---|---|---|---|"]
    for k in order:
        if k in rows:
            r, n = rows[k], sum(rows[k].values())
            md.append(f"| {k} | {n} | {r['found'] / n:.2f} | {r['wrong class']} | {r['loose box']} | {r['nothing']} |")
    return md, rows


def sweep(slides, confs):
    md = ["#### Working threshold", "", "| conf | boxes | P | R | F1 |", "|---|---|---|---|---|"]
    for c in confs:
        r = score([(p, g) for p, g, _, _ in slides], c)
        md.append(f"| {c:g} | {r['boxes']} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} |")
    return md


def overlaps(dets):
    return sum(1 for i, a in enumerate(dets) for b in dets[i + 1:]
               if GROUP[a["cls"]] == GROUP[b["cls"]] and iou(a["bbox"], b["bbox"]) >= 0.5)


def score(slides, conf):
    from ultralytics.utils.metrics import ap_per_class

    tps, confs, pcls, tcls = [], [], [], []
    n_tp = n_pred = n_gt = n_ov = 0
    for preds, gts in slides:
        t = match(preds, gts)
        tps.append(t)
        confs += [p["conf"] for p in preds]
        pcls += [CLASSES.index(p["cls"]) for p in preds]
        tcls += [CLASSES.index(g["cls"]) for g in gts]
        work = [i for i, p in enumerate(preds) if p["conf"] >= conf]
        n_tp += int(t[work, 0].sum()) if work else 0
        n_pred += len(work)
        n_gt += len(gts)
        n_ov += overlaps([preds[i] for i in work])
    res = ap_per_class(np.concatenate(tps), np.array(confs), np.array(pcls), np.array(tcls),
                       names={i: c for i, c in enumerate(CLASSES)})
    ap, classes = res[5], res[6]
    per = {CLASSES[int(c)]: dict(mAP50=float(ap[i, 0]), mAP50_95=float(ap[i].mean())) for i, c in enumerate(classes)}
    P, R = n_tp / max(n_pred, 1), n_tp / max(n_gt, 1)
    return dict(mAP50=float(ap[:, 0].mean()), mAP50_95=float(ap.mean()), P=P, R=R,
                F1=2 * P * R / max(P + R, 1e-9), boxes=n_pred, overlaps=n_ov, per_class=per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", help="folder of <stem>.json from detector.predict")
    ap.add_argument("--labels", default="data/real/yolo/labels/test")
    ap.add_argument("--conf", type=float, default=0.25, help="working threshold for P / R / F1")
    ap.add_argument("--same-iou", type=float, default=0.5)
    ap.add_argument("--cross-iou", type=float, default=0.6)
    ap.add_argument("--contain", type=float, default=0.9)
    ap.add_argument("--cards", action="store_true", help="also score detector.containers (shapes found behind text)")
    ap.add_argument("--sweep", default="0.05,0.1,0.15,0.2,0.25,0.35,0.5", help="conf values for the P/R/F1 table")
    ap.add_argument("--out", default=None, help="default: <pred>/evalpred.md")
    a = ap.parse_args()
    from PIL import Image

    raw, post, full, cards, both = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(a.pred, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(a.labels, stem + ".txt")
        if not os.path.exists(lab):
            continue
        W, H = Image.open(d["image"]).size
        gts = load_gt(lab, W, H)
        raw.append((d["detections"], gts))
        full.append((d["detections"], gts, W, H))
        post.append((clean(d["detections"], a.same_iou, a.cross_iou, a.contain), gts))
        if a.cards:
            new = find_cards(d["image"], [p for p in d["detections"] if p["conf"] >= a.conf])
            cards.append((d["detections"] + new, gts))
            both.append((clean(d["detections"] + new, a.same_iou, a.cross_iou, a.contain), gts))
    if not raw:
        raise SystemExit(f"no predictions in {a.pred} have a label in {a.labels}")
    rows = {"raw (YOLO NMS only)": score(raw, a.conf), "postprocess": score(post, a.conf)}
    if a.cards:
        rows["cards"] = score(cards, a.conf)
        rows["cards + postprocess"] = score(both, a.conf)
    f = lambda v: f"{v:.3f}"   # noqa: E731
    md = [f"### {a.pred}: {len(raw)} slides (P / R / F1 / boxes / overlaps at conf >= {a.conf})", "",
          "| | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | boxes | overlapping pairs | "
          + " | ".join(CLASSES) + " |", "|---|---|---|---|---|---|---|---|" + "---|" * len(CLASSES)]
    for name, r in rows.items():
        md.append(f"| {name} | {f(r['mAP50'])} | {f(r['mAP50_95'])} | {f(r['P'])} | {f(r['R'])} | {f(r['F1'])} | "
                  f"{r['boxes']} | {r['overlaps']} | "
                  + " | ".join(f(r['per_class'][c]['mAP50_95']) if c in r["per_class"] else "–" for c in CLASSES) + " |")
    md += ["", f"Postprocess: same class IoU >= {a.same_iou}, same group IoU >= {a.cross_iou}, "
           f"text inside text >= {a.contain}. Per-class columns are mAP@0.5:0.95.", ""]
    md += sweep(full, [float(x) for x in a.sweep.split(",")]) + [""]
    diag_md, diag = diagnose(full, a.conf)
    if a.cards:
        cdiag_md, rows["why_missed_cards"] = diagnose([(c[0],) + f[1:] for c, f in zip(cards, full)], a.conf)
        diag_md += ["", "With cards:", ""] + cdiag_md[2:]
    md += diag_md
    rows["why_missed"] = diag
    text = "\n".join(md) + "\n"
    out = a.out or os.path.join(a.pred, "evalpred.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(text)


if __name__ == "__main__":
    main()

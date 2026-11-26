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
    ap.add_argument("--out", default=None, help="default: <pred>/evalpred.md")
    a = ap.parse_args()
    from PIL import Image

    raw, post = [], []
    for p in sorted(glob.glob(os.path.join(a.pred, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(a.labels, stem + ".txt")
        if not os.path.exists(lab):
            continue
        W, H = Image.open(d["image"]).size
        gts = load_gt(lab, W, H)
        raw.append((d["detections"], gts))
        post.append((clean(d["detections"], a.same_iou, a.cross_iou, a.contain), gts))
    if not raw:
        raise SystemExit(f"no predictions in {a.pred} have a label in {a.labels}")
    rows = {"raw (YOLO NMS only)": score(raw, a.conf), "postprocess": score(post, a.conf)}
    f = lambda v: f"{v:.3f}"   # noqa: E731
    md = [f"### {a.pred}: {len(raw)} slides (P / R / F1 / boxes / overlaps at conf >= {a.conf})", "",
          "| | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | boxes | overlapping pairs | "
          + " | ".join(CLASSES) + " |", "|---|---|---|---|---|---|---|---|" + "---|" * len(CLASSES)]
    for name, r in rows.items():
        md.append(f"| {name} | {f(r['mAP50'])} | {f(r['mAP50_95'])} | {f(r['P'])} | {f(r['R'])} | {f(r['F1'])} | "
                  f"{r['boxes']} | {r['overlaps']} | "
                  + " | ".join(f(r['per_class'][c]['mAP50_95']) if c in r["per_class"] else "–" for c in CLASSES) + " |")
    md += ["", f"Postprocess: same class IoU >= {a.same_iou}, same group IoU >= {a.cross_iou}, "
           f"text inside text >= {a.contain}. Per-class columns are mAP@0.5:0.95."]
    text = "\n".join(md) + "\n"
    out = a.out or os.path.join(a.pred, "evalpred.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(text)


if __name__ == "__main__":
    main()

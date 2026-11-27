"""Evaluate Level 1 (boxes) and Level 2 (groups) on a fixed set of synthetic test scenes.

    python -m boxseg.evaluate --seg runs/boxseg/seg.pt --group runs/boxseg/group.pt --n 200
    python -m boxseg.evaluate --seg runs/boxseg/seg.pt --only-styles hand sticky   # held-out styles

Reports, for the U-Net and for the classical contour baseline:
  * box precision / recall / F1 at IoU 0.5 and 0.75
  * recall per style (hand, sticky, glow, shadow, outline_only, fill_only, ...)
and for Level 2, pair accuracy and the share of multi-box groups recovered exactly,
both on ground-truth boxes (Level 2 alone) and on the U-Net's boxes (end to end).
Writes a markdown table next to the checkpoint (or to --report).
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from .boxes import contour_baseline, masks_to_boxes, match
from .data import candidate_pairs, pair_input, to_tensor
from .model import PairNet, UNet
from .synth import scene


def load_unet(path, dev):
    ck = torch.load(path, map_location="cpu")
    m = UNet(base=ck.get("base", 32))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck


@torch.no_grad()
def predict_maps(model, img, dev):
    p = torch.sigmoid(model(to_tensor(img)[None].to(dev)))[0].cpu().numpy()
    return p[0], p[1]


@torch.no_grad()
def group_boxes(pairnet, img, boxes, dev, thr=0.5):
    """Level 2 inference: score candidate pairs, then union-find over pairs above thr."""
    H, W = img.shape[:2]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = candidate_pairs(boxes, W, H)
    probs = {}
    if pairs:
        xs, gs = zip(*[pair_input(img, boxes[i], boxes[j]) for i, j in pairs])
        p = torch.sigmoid(pairnet(torch.from_numpy(np.stack(xs)).to(dev),
                                  torch.from_numpy(np.stack(gs)).to(dev))).cpu().numpy()
        for (i, j), q in zip(pairs, p):
            probs[(i, j)] = float(q)
            if q > thr:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(boxes))], probs


def group_scores(pred_lab, gt_lab, probs):
    """Pair accuracy on candidate pairs and exact recovery of multi-box groups."""
    correct = total = 0
    for (i, j), q in probs.items():
        correct += (q > 0.5) == (gt_lab[i] == gt_lab[j])
        total += 1
    exact = n_groups = 0
    for g in set(gt_lab):
        members = {i for i, x in enumerate(gt_lab) if x == g}
        if len(members) < 2:
            continue
        n_groups += 1
        cl = {i for i, x in enumerate(pred_lab) if x == pred_lab[next(iter(members))]}
        exact += cl == members
    return correct, total, exact, n_groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345, help="test scenes use a seed never used in training")
    ap.add_argument("--only-styles", nargs="*", default=[],
                    help="count only GT boxes that have at least one of these styles")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ck = load_unet(a.seg, dev)
    W, H = ck.get("size", [512, 288])
    pairnet = None
    if a.group:
        pairnet = PairNet()
        pairnet.load_state_dict(torch.load(a.group, map_location="cpu")["model"])
        pairnet.to(dev).eval()

    rng = np.random.default_rng(a.seed)
    stats = {m: defaultdict(float) for m in ("unet", "contours")}
    by_style = {m: defaultdict(lambda: [0, 0]) for m in ("unet", "contours")}
    grp = {k: np.zeros(4) for k in ("gt_boxes", "unet_boxes")}
    for _ in range(a.n):
        s = scene(rng, W, H)
        gt = np.array([b["box"] for b in s["boxes"]], np.float32).reshape(-1, 4)
        relevant = [i for i, b in enumerate(s["boxes"])
                    if not a.only_styles or set(b["style"]) & set(a.only_styles)]
        preds = {"unet": masks_to_boxes(*predict_maps(unet, s["image"], dev)),
                 "contours": contour_baseline(s["image"])}
        for m, pb in preds.items():
            for thr in (0.5, 0.75):
                mt = match(pb, gt, thr)
                hit = {j for _, j in mt}
                stats[m][f"tp{thr}"] += len([j for j in hit if j in relevant])
                stats[m][f"fp{thr}"] += len(pb) - len(mt)
                if thr == 0.5:
                    for j in relevant:
                        for st in s["boxes"][j]["style"]:
                            by_style[m][st][0] += j in hit
                            by_style[m][st][1] += 1
            stats[m]["gt"] += len(relevant)
        if pairnet is not None and len(gt) > 1:
            lab = [b["group"] for b in s["boxes"]]
            pl, probs = group_boxes(pairnet, s["image"], gt.tolist(), dev)
            grp["gt_boxes"] += group_scores(pl, lab, probs)
            pb = preds["unet"]
            mt = match(pb, gt, 0.5)
            if len(mt) > 1:                         # end to end: only boxes Level 1 found
                pi = [i for i, _ in mt]
                pl, probs = group_boxes(pairnet, s["image"], pb[pi, :4].tolist(), dev)
                grp["unet_boxes"] += group_scores(pl, [lab[j] for _, j in mt], probs)

    lines = [f"# boxseg evaluation ({a.n} synthetic test scenes, {W}x{H}"
             + (f", only styles {a.only_styles}" if a.only_styles else "") + ")", "",
             f"checkpoint: `{a.seg}` (step {ck.get('step')}, trained without {ck.get('exclude') or 'nothing'})", "",
             "## Level 1: boxes", "",
             "| method | P@0.5 | R@0.5 | F1@0.5 | P@0.75 | R@0.75 |", "|---|---|---|---|---|---|"]
    for m in stats:
        st = stats[m]
        row = []
        for thr in (0.5, 0.75):
            tp, fp, n = st[f"tp{thr}"], st[f"fp{thr}"], st["gt"]
            p, r = tp / max(1, tp + fp), tp / max(1, n)
            row.append((p, r, 2 * p * r / max(1e-6, p + r)))
        lines.append(f"| {m} | {row[0][0]:.3f} | {row[0][1]:.3f} | {row[0][2]:.3f} | {row[1][0]:.3f} | {row[1][1]:.3f} |")
    lines += ["", "Recall@0.5 by style (a box can have several styles):", "",
              "| style | n | unet | contours |", "|---|---|---|---|"]
    for stl in sorted(by_style["unet"], key=lambda k: -by_style["unet"][k][1]):
        u, c = by_style["unet"][stl], by_style["contours"][stl]
        lines.append(f"| {stl} | {u[1]} | {u[0] / max(1, u[1]):.3f} | {c[0] / max(1, c[1]):.3f} |")
    if pairnet is not None:
        lines += ["", "## Level 2: groups", "", "| boxes used | pair accuracy | multi-box groups exact |",
                  "|---|---|---|"]
        for k, (c, t, e, g) in grp.items():
            lines.append(f"| {k} | {c / max(1, t):.3f} ({int(t)} pairs) | {e / max(1, g):.3f} ({int(g)} groups) |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = a.report or os.path.splitext(a.seg)[0] + "_eval.md"
    with open(out, "w") as f:
        f.write(text)
    print("wrote", out)


if __name__ == "__main__":
    main()

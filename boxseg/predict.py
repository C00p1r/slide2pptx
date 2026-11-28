"""Run both levels on real slide images and draw the result.

    python -m boxseg.predict runs/boxseg/seg.pt data/real/images --group runs/boxseg/group.pt \
        --out results/boxseg_real [--baseline]

Each output image shows, left to right: the slide, the U-Net's inside/edge maps, and the
boxes (thin) with each predicted group in one colour (thick hull around multi-box groups).
--baseline adds a panel with the classical contour boxes for comparison.
Boxes are also written to <name>.json (in original image pixels).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from slidegen.content import hsv

from .boxes import contour_baseline, masks_to_boxes
from .evaluate import group_boxes, load_pairnet, load_unet, predict_maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("images")
    ap.add_argument("--group", default=None)
    ap.add_argument("--out", default="results/boxseg_pred")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ck = load_unet(a.seg, dev)
    W, H = ck.get("size", [512, 288])
    pairnet = None
    if a.group:
        pairnet = load_pairnet(a.group, dev)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for f in files:
        orig = Image.open(f).convert("RGB")
        img = np.asarray(orig.resize((W, H), Image.BILINEAR))
        inside, edge = predict_maps(unet, img, dev)
        boxes = masks_to_boxes(inside, edge)
        labels = list(range(len(boxes)))
        if pairnet is not None and len(boxes) > 1:
            labels, _ = group_boxes(pairnet, img, boxes[:, :4].tolist(), dev)
        vis = img.copy()
        for g in set(labels):
            col = tuple(int(c) for c in hsv(hash(g) % 97 / 97, 0.9, 0.9))
            idx = [i for i, l in enumerate(labels) if l == g]
            for i in idx:
                x0, y0, x1, y1 = boxes[i, :4].astype(int)
                cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), col, 1)
            if len(idx) > 1:
                b = boxes[idx, :4]
                cv2.rectangle(vis, (int(b[:, 0].min()) - 2, int(b[:, 1].min()) - 2),
                              (int(b[:, 2].max()) + 1, int(b[:, 3].max()) + 1), col, 2)
        maps = np.stack([edge * 255, inside * 255, np.zeros_like(inside)], -1).astype(np.uint8)
        panels = [img, maps, vis]
        if a.baseline:
            cb = img.copy()
            for x0, y0, x1, y1, _ in contour_baseline(img).astype(int):
                cv2.rectangle(cb, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 1)
            panels.append(cb)
        name = os.path.splitext(os.path.basename(f))[0]
        Image.fromarray(np.concatenate(panels, 1)).save(os.path.join(a.out, f"{name}.png"))
        sx, sy = orig.width / W, orig.height / H
        with open(os.path.join(a.out, f"{name}.json"), "w") as fh:
            json.dump([dict(box=[float(b[0] * sx), float(b[1] * sy), float(b[2] * sx), float(b[3] * sy)],
                            score=float(b[4]), group=int(l)) for b, l in zip(boxes, labels)], fh)
        print(name, len(boxes), "boxes,", len(set(labels)), "groups")


if __name__ == "__main__":
    main()

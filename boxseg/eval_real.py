"""Class-agnostic evaluation of Level 1 on the hand-labelled real slides (YOLO format).

    python -m boxseg.eval_real runs/boxseg/seg.pt data/real/yolo --size 768x432

For each labelled class, the share of its boxes that some predicted box matches at IoU 0.5
(predictions have no class, so this is recall only). `matches any label` is the share of
predicted boxes that match a labelled box of any class: a rough precision, since the labelling
rules leave backdrops and textures unboxed. The contour baseline is reported alongside.
"""
import argparse
import collections
import glob

import numpy as np
from PIL import Image

from .boxes import contour_baseline, masks_to_boxes, match
from .evaluate import load_unet, predict_maps

CLASSES = ["title", "text", "shape", "picture", "icon"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("yolo_dir", help="folder with images/test and labels/test")
    ap.add_argument("--size", default=None, help="WxH to run at (default: training size); the U-Net is fully convolutional")
    a = ap.parse_args()
    model, ck = load_unet(a.seg, "cpu")
    W, H = map(int, a.size.split("x")) if a.size else ck.get("size", [512, 288])
    res = {k: collections.defaultdict(lambda: [0, 0]) for k in ("unet", "contours")}
    prec = {k: [0, 0] for k in res}
    for f in sorted(glob.glob(f"{a.yolo_dir}/images/test/*.png")):
        lab = np.loadtxt(f.replace("images", "labels").rsplit(".", 1)[0] + ".txt", ndmin=2)
        cls = lab[:, 0].astype(int)
        gt = np.stack([(lab[:, 1] - lab[:, 3] / 2) * W, (lab[:, 2] - lab[:, 4] / 2) * H,
                       (lab[:, 1] + lab[:, 3] / 2) * W, (lab[:, 2] + lab[:, 4] / 2) * H], 1)
        img = np.array(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR))
        preds = {"unet": masks_to_boxes(*predict_maps(model, img, "cpu"), min_side=int(8 * W / 512)),
                 "contours": contour_baseline(img)}
        for k, pb in preds.items():
            for name, sel in [(c, cls == i) for i, c in enumerate(CLASSES)] + [("shape+picture+icon", cls >= 2)]:
                idx = np.where(sel)[0]
                if len(idx):
                    res[k][name][0] += len(match(pb, gt[idx], 0.5))
                    res[k][name][1] += len(idx)
            prec[k][0] += len(match(pb, gt, 0.5))
            prec[k][1] += len(pb)
    names = CLASSES + ["shape+picture+icon"]
    print(f"| method ({W}x{H}) | " + " | ".join(f"{n} R" for n in names) + " | matches any label |")
    print("|---" * (len(names) + 2) + "|")
    for k in res:
        cells = [f"{res[k][n][0] / max(1, res[k][n][1]):.3f}" for n in names]
        print(f"| {k} | " + " | ".join(cells) + f" | {prec[k][0] / max(1, prec[k][1]):.3f} of {prec[k][1]} |")


if __name__ == "__main__":
    main()

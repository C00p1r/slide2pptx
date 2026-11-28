"""mAP@0.5 of the whole v3 pipeline (U-Net + pair CNN + object CNN), like the YOLO evaluation.

    python -m boxseg.eval_pipeline --seg runs/boxseg/seg_v3.pt --group runs/boxseg/group_v3.pt \
        --cls runs/boxseg/cls.pt --real data/real/yolo --size 768x432
    python -m boxseg.eval_pipeline ... --synthetic 200      # generated test scenes instead

Classes are the YOLO ones (title, text, shape, picture, icon), so the table can be put next to
results/detector/*_real.md. Writes markdown to --report if given.
"""
import argparse
import glob

import numpy as np
import torch
from PIL import Image

from .boxes import ap_per_class
from .data import objects
from .evaluate import load_pairnet, load_unet
from .model import OBJ_CLASSES
from .pipeline import load_clsnet, run
from .synth import scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--cls", default=None)
    ap.add_argument("--real", default=None, help="YOLO-format folder with images/test and labels/test")
    ap.add_argument("--synthetic", type=int, default=0, help="number of generated test scenes")
    ap.add_argument("--size", default=None, help="WxH to run at (default: the U-Net's training size)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    unet, ck = load_unet(a.seg, dev)
    W, H = map(int, a.size.split("x")) if a.size else ck.get("size", [512, 288])
    pairnet = load_pairnet(a.group, dev) if a.group else None
    clsnet = load_clsnet(a.cls, dev) if a.cls else None

    samples = []                                   # (image, gt boxes, gt classes)
    if a.real:
        for f in sorted(glob.glob(f"{a.real}/images/test/*.png")):
            lab = np.loadtxt(f.replace("images", "labels").rsplit(".", 1)[0] + ".txt", ndmin=2)
            gt = np.stack([(lab[:, 1] - lab[:, 3] / 2) * W, (lab[:, 2] - lab[:, 4] / 2) * H,
                           (lab[:, 1] + lab[:, 3] / 2) * W, (lab[:, 2] + lab[:, 4] / 2) * H], 1)
            img = np.array(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR))
            samples.append((img, gt, lab[:, 0].astype(int)))
    rng = np.random.default_rng(a.seed)
    for _ in range(a.synthetic):
        s = scene(rng, W, H)
        obs = objects(s)
        samples.append((s["image"], np.array([o[0] for o in obs], np.float32).reshape(-1, 4),
                        np.array([o[1] for o in obs], int)))

    per_image = []
    for img, gt, gc in samples:
        res = run(img, unet, pairnet, clsnet, dev, min_side=int(8 * W / 512))
        pb = np.array([r["box"] for r in res], np.float32).reshape(-1, 4)
        pc = np.array([OBJ_CLASSES.index(r["cls"]) for r in res], int)
        ps = np.array([r["score"] for r in res], np.float32)
        per_image.append((pb, pc, ps, gt, gc))
    aps = ap_per_class(per_image, len(OBJ_CLASSES))
    n_gt = [int(sum((g[4] == c).sum() for g in per_image)) for c in range(len(OBJ_CLASSES))]
    src = f"{a.real} (real)" if a.real else f"{a.synthetic} synthetic scenes"
    lines = [f"### boxseg pipeline on {src}, {W}x{H}", "",
             "| class | instances | AP@0.5 |", "|---|---|---|"]
    for c, name in enumerate(OBJ_CLASSES):
        lines.append(f"| {name} | {n_gt[c]} | {aps[c]:.3f} |")
    lines.append(f"| **all** | {sum(n_gt)} | **{np.nanmean(aps):.3f}** |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.report:
        with open(a.report, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()

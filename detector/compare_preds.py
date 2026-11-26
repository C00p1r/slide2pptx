"""Compare several detectors on unlabelled slides (no ground truth needed).

First run detector.predict once per model, each into its own folder:
    python -m detector.predict runs/ratio/r0.20/weights/last.pt data/real/images --out results/pred_llm/r0.20
Then:
    python -m detector.compare_preds results/pred_llm

Writes into results/pred_llm/compare/:
  summary.md       per model: boxes per slide and mean confidence, by class; and how often
                   each pair of models agree (F1 of matched boxes: same class, IoU >= 0.5)
  summary.json     the same numbers
  grids/<slide>.jpg  the original slide and every model's boxes side by side, for eyeballing
Without labels these numbers say how the models *differ*, not which one is right.
"""
import argparse
import glob
import json
import os
from itertools import combinations

from PIL import Image, ImageDraw

from . import CLASSES


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def agreement(da, db, thr=0.5):
    """F1 between two lists of detections, greedy matching by IoU within each class."""
    if not da and not db:
        return 1.0
    pairs = sorted(((iou(a["bbox"], b["bbox"]), i, j) for i, a in enumerate(da) for j, b in enumerate(db)
                    if a["cls"] == b["cls"]), reverse=True)
    used_a, used_b, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            tp += 1
    return 2 * tp / (len(da) + len(db))


def load(root, runs):
    preds = {}
    for r in runs:
        preds[r] = {os.path.splitext(os.path.basename(p))[0]: json.load(open(p))
                    for p in glob.glob(os.path.join(root, r, "*.json"))}
    return preds


def grid(stems, preds, runs, root, out, tile_w=800):
    os.makedirs(out, exist_ok=True)
    for stem in stems:
        src = next(iter(preds[r][stem]["image"] for r in runs if stem in preds[r]))
        tiles = [("original", Image.open(src).convert("RGB"))]
        for r in runs:
            p = os.path.join(root, r, f"{stem}.jpg")
            if os.path.exists(p):
                tiles.append((f"{r}: {len(preds[r][stem]['detections'])} boxes", Image.open(p).convert("RGB")))
        th = int(tile_w * tiles[0][1].height / tiles[0][1].width)
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, rows * (th + 28)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (label, im) in enumerate(tiles):
            x, y = (k % cols) * tile_w, (k // cols) * (th + 28)
            d.text((x + 8, y + 6), label, fill=(20, 20, 20))
            sheet.paste(im.resize((tile_w, th)), (x, y + 28))
        sheet.save(os.path.join(out, f"{stem}.jpg"), quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one prediction folder per model")
    ap.add_argument("--runs", nargs="*", help="model folders to compare (default: all subfolders)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-grids", action="store_true")
    a = ap.parse_args()
    runs = a.runs or sorted(d for d in os.listdir(a.root)
                            if d != "compare" and glob.glob(os.path.join(a.root, d, "*.json")))
    out = a.out or os.path.join(a.root, "compare")
    os.makedirs(out, exist_ok=True)
    preds = load(a.root, runs)
    stems = sorted(set.intersection(*(set(p) for p in preds.values())))
    n = len(stems)
    if not n:
        raise SystemExit("no slide was predicted by every model")

    stats = {}
    for r in runs:
        per_cls = {c: [] for c in CLASSES}
        for s in stems:
            for det in preds[r][s]["detections"]:
                per_cls[det["cls"]].append(det["conf"])
        stats[r] = dict(boxes_per_slide={c: len(v) / n for c, v in per_cls.items()},
                        mean_conf={c: (sum(v) / len(v) if v else None) for c, v in per_cls.items()},
                        total_per_slide=sum(len(v) for v in per_cls.values()) / n)
    agree = {f"{x}|{y}": sum(agreement(preds[x][s]["detections"], preds[y][s]["detections"]) for s in stems) / n
             for x, y in combinations(runs, 2)}
    json.dump(dict(slides=n, runs=runs, stats=stats, agreement_f1=agree), open(os.path.join(out, "summary.json"), "w"),
              indent=2)

    f = lambda v: "–" if v is None else f"{v:.2f}"   # noqa: E731
    md = [f"Unlabelled slides: {n}. Confidence threshold as used in detector.predict.", "",
          "#### Boxes per slide", "", "| model | all | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * (len(CLASSES) + 1)]
    md += [f"| {r} | {stats[r]['total_per_slide']:.1f} | " + " | ".join(f(stats[r]["boxes_per_slide"][c]) for c in CLASSES)
           + " |" for r in runs]
    md += ["", "#### Mean confidence of predicted boxes", "", "| model | " + " | ".join(CLASSES) + " |",
           "|---|" + "---|" * len(CLASSES)]
    md += [f"| {r} | " + " | ".join(f(stats[r]["mean_conf"][c]) for c in CLASSES) + " |" for r in runs]
    md += ["", "#### Agreement between models (F1 of matched boxes, same class, IoU ≥ 0.5)", "",
           "| | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for x in runs:
        cells = ["1.00" if x == y else f"{agree.get(f'{x}|{y}', agree.get(f'{y}|{x}')):.2f}" for y in runs]
        md.append(f"| {x} | " + " | ".join(cells) + " |")
    text = "\n".join(md) + "\n"
    open(os.path.join(out, "summary.md"), "w").write(text)
    print(text)
    if not a.no_grids:
        grid(stems, preds, runs, a.root, os.path.join(out, "grids"))
        print(f"wrote {n} side-by-side images to {out}/grids")


if __name__ == "__main__":
    main()

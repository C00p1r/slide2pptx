"""Pull the slide images out of NotebookLM .pptx exports.

NotebookLM's .pptx keeps each slide as one big picture. This saves that picture
for every slide, at its original resolution, and starts a CSV for tagging.

    python tools/extract_slides.py path/to/pptx_folder --out data/real/images

Then fill in data/real/real_slides.csv (language, layout, style, hard_case).
"""
import argparse
import csv
import glob
import os

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def pictures(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from pictures(sh.shapes)
        elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            yield sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with .pptx files (or one .pptx)")
    ap.add_argument("--out", default="data/real/images")
    ap.add_argument("--csv", default="data/real/real_slides.csv")
    a = ap.parse_args()
    files = [a.src] if a.src.endswith(".pptx") else sorted(glob.glob(os.path.join(a.src, "*.pptx")))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in files:
        deck = os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
        prs = Presentation(f)
        slide_area = prs.slide_width * prs.slide_height
        for i, slide in enumerate(prs.slides, 1):
            pics = list(pictures(slide.shapes))
            if not pics:
                print(f"  {deck} slide {i}: no picture, skipped")
                continue
            big = max(pics, key=lambda p: p.width * p.height)
            cover = big.width * big.height / slide_area
            others = len(slide.shapes) - 1
            if cover < 0.9 or others:
                print(f"  {deck} slide {i}: picture covers {cover:.0%}, {others} other shapes, check by hand")
            name = f"{deck}_{i:02d}.{big.image.ext}"
            with open(os.path.join(a.out, name), "wb") as fh:
                fh.write(big.image.blob)
            rows.append(dict(file=name, deck=deck, slide=i, size=f"{big.image.size[0]}x{big.image.size[1]}",
                             language="", layout="", style="", hard_case=""))
        print(f"{deck}: {len(prs.slides)} slides")
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} images to {a.out}, tags in {a.csv}")


if __name__ == "__main__":
    main()

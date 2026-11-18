"""Draw labels on top of generated slides to check they line up.

    python -m slidegen.visualize data/synth --n 12 --out data/preview
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageDraw

from . import CLASSES

COLORS = {"title": (230, 40, 40), "text": (40, 120, 255), "shape": (40, 180, 60),
          "picture": (255, 150, 0), "icon": (200, 40, 200)}


def draw(root, meta_path, lines=False):
    m = json.load(open(meta_path))
    img = Image.open(os.path.join(root, m["image"])).convert("RGB")
    d = ImageDraw.Draw(img)
    for el in m["elements"]:
        x, y, w, h = el["bbox"]
        c = COLORS[el["cls"]]
        d.rectangle([x, y, x + w, y + h], outline=c, width=2)
        d.text((x + 2, y + 1), el["cls"], fill=c)
        if lines and el["kind"] == "text":
            for lx, ly, lw, lh in el["lines"]:
                d.rectangle([lx, ly, lx + lw, ly + lh], outline=(0, 200, 200), width=1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lines", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "preview")
    os.makedirs(out, exist_ok=True)
    metas = sorted(glob.glob(os.path.join(a.root, "meta", "*.json")))[: a.n]
    tiles = [draw(a.root, p, a.lines) for p in metas]
    for p, t in zip(metas, tiles):
        t.save(os.path.join(out, os.path.basename(p).replace(".json", ".png")))
    if tiles:   # contact sheet, 3 per row at half size
        tw, th = 640, 360
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * th), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        sheet.save(os.path.join(out, "contact_sheet.png"))
    print(f"wrote {len(tiles)} previews to {out}")


if __name__ == "__main__":
    main()

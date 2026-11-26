"""Detect elements on slide images: drawn previews + JSON boxes for the next steps.

    python -m detector.predict runs/detect/slides/weights/best.pt data/real/images --out results/pred_real
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageDraw

from . import CLASSES
from .postprocess import clean

COLORS = {"title": (230, 40, 40), "text": (40, 120, 255), "shape": (40, 180, 60),
          "picture": (255, 150, 0), "icon": (200, 40, 200)}


def detect(model, path, imgsz=1280, conf=0.25, augment=False):
    """-> list of dict(cls, conf, bbox=[x, y, w, h]) in pixels of the original image."""
    r = model.predict(path, imgsz=imgsz, conf=conf, augment=augment, verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        out.append(dict(cls=CLASSES[int(k)], conf=round(c, 3),
                        bbox=[round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("src", help="image or folder")
    ap.add_argument("--out", default="results/pred")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--augment", action="store_true", help="test-time augmentation (multi-scale)")
    ap.add_argument("--post", action="store_true", help="merge overlapping boxes (detector/postprocess.py)")
    ap.add_argument("--no-draw", action="store_true", help="only write the JSON")
    ap.add_argument("--yolo-txt", action="store_true",
                    help="also write YOLO .txt files (pre-labels to correct by hand, see docs/labelling.md)")
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    files = [a.src] if os.path.isfile(a.src) else sorted(
        f for e in ("png", "jpg", "jpeg", "webp") for f in glob.glob(os.path.join(a.src, f"*.{e}")))
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        dets = detect(model, f, a.imgsz, a.conf, a.augment)
        if a.post:
            dets = clean(dets)
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(os.path.join(a.out, f"{stem}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(image=os.path.abspath(f), detections=dets), fh, indent=1, ensure_ascii=False)
        print(f"{stem}: {len(dets)} boxes")
        if a.no_draw and not a.yolo_txt:
            continue
        img = Image.open(f).convert("RGB")
        if a.yolo_txt:
            W, H = img.size
            with open(os.path.join(a.out, f"{stem}.txt"), "w") as fh:
                for det in dets:
                    x, y, w, h = det["bbox"]
                    fh.write(f"{CLASSES.index(det['cls'])} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}\n")
        d, lw = ImageDraw.Draw(img), max(2, img.width // 640)
        for det in dets:
            x, y, w, h = det["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=COLORS[det["cls"]], width=lw)
            d.text((x + 3, y + 2), f"{det['cls']} {det['conf']:.2f}", fill=COLORS[det["cls"]])
        if not a.no_draw:
            img.save(os.path.join(a.out, f"{stem}.jpg"), quality=90)


if __name__ == "__main__":
    main()

"""Hand labels for the real (NotebookLM) test set. See docs/labelling.md.

1) Pre-label with the detector, packed for CVAT ("YOLO 1.1" format):
    python tools/real_labels.py prelabel runs/detect/slides/weights/best.pt
   -> data/real/cvat_prelabels.zip   (upload in CVAT: Actions > Upload annotations > YOLO 1.1)

2) After correcting the boxes, export from CVAT as "YOLO 1.1" (or any YOLO export
   that has obj.names / classes.txt / data.yaml) and build the test set:
    python tools/real_labels.py build path/to/export.zip
   -> data/real/yolo/{images,labels}/test + data/real/yolo/data.yaml

Class ids are remapped by NAME, so the class order in the labelling tool does not matter.
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import CLASSES  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def images_in(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMG_EXT))


def prelabel(a):
    from ultralytics import YOLO

    from detector.predict import detect

    model = YOLO(a.weights)
    imgs = images_in(a.images)
    if not imgs:
        sys.exit(f"no images in {a.images}")
    work = tempfile.mkdtemp()
    os.makedirs(os.path.join(work, "obj_train_data"))
    from PIL import Image
    for f in imgs:
        W, H = Image.open(f).size
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(os.path.join(work, "obj_train_data", stem + ".txt"), "w") as fh:
            for d in detect(model, f, a.imgsz, a.conf):
                x, y, w, h = d["bbox"]
                fh.write(f"{CLASSES.index(d['cls'])} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}\n")
    with open(os.path.join(work, "obj.names"), "w") as fh:
        fh.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(work, "obj.data"), "w") as fh:
        fh.write(f"classes = {len(CLASSES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
    with open(os.path.join(work, "train.txt"), "w") as fh:
        fh.write("".join(f"data/obj_train_data/{os.path.basename(f)}\n" for f in imgs))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with zipfile.ZipFile(a.out, "w") as z:
        for root, _, files in os.walk(work):
            for name in files:
                p = os.path.join(root, name)
                z.write(p, os.path.relpath(p, work))
    shutil.rmtree(work)
    print(f"pre-labels for {len(imgs)} images -> {a.out}")


def read_names(root):
    for pat in ("obj.names", "classes.txt"):
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return [l.strip() for l in open(hits[0], encoding="utf-8") if l.strip()]
    hits = glob.glob(os.path.join(root, "**", "data.yaml"), recursive=True)
    if hits:
        import yaml
        names = yaml.safe_load(open(hits[0]))["names"]
        return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    sys.exit("export has no obj.names, classes.txt or data.yaml: cannot tell class names")


def build(a):
    src = a.export
    if src.endswith(".zip"):
        tmp = tempfile.mkdtemp()
        zipfile.ZipFile(src).extractall(tmp)
        src = tmp
    names = [n.lower() for n in read_names(src)]
    unknown = [n for n in names if n not in CLASSES]
    if unknown:
        sys.exit(f"unknown classes in export: {unknown}; expected names from {CLASSES}")
    remap = {i: CLASSES.index(n) for i, n in enumerate(names)}
    labels = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(os.path.join(src, "**", "*.txt"), recursive=True)
              if os.path.basename(p) not in ("classes.txt", "train.txt", "valid.txt", "test.txt")}
    out = a.out
    for d in ("images/test", "labels/test"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    n_img = n_box = 0
    missing = []
    counts = [0] * len(CLASSES)
    for f in images_in(a.images):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in labels:
            missing.append(stem)
            continue
        lines = []
        for line in open(labels[stem]):
            parts = line.split()
            if len(parts) != 5:
                continue
            k = remap[int(parts[0])]
            counts[k] += 1
            lines.append(" ".join([str(k)] + parts[1:]))
        shutil.copy(f, os.path.join(out, "images/test", os.path.basename(f)))
        with open(os.path.join(out, "labels/test", stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        n_img, n_box = n_img + 1, n_box + len(lines)
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(out)}\ntrain: images/test\nval: images/test\ntest: images/test\n"
                 "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"{n_img} labelled slides, {n_box} boxes -> {out}")
    print("  " + ", ".join(f"{c}: {n}" for c, n in zip(CLASSES, counts)))
    if missing:
        print(f"  WARNING: {len(missing)} images have no label file (not labelled yet?): {missing[:5]}...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prelabel")
    p.add_argument("weights")
    p.add_argument("--images", default="data/real/images")
    p.add_argument("--out", default="data/real/cvat_prelabels.zip")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.3)
    b = sub.add_parser("build")
    b.add_argument("export", help="YOLO export folder or .zip from the labelling tool")
    b.add_argument("--images", default="data/real/images")
    b.add_argument("--out", default="data/real/yolo")
    a = ap.parse_args()
    prelabel(a) if a.cmd == "prelabel" else build(a)


if __name__ == "__main__":
    main()

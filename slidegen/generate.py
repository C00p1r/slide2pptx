"""Generate a synthetic slide dataset.

    python -m slidegen.generate --out data/synth --n 2000 --workers 4

Output layout (YOLO-ready):
    images/{train,val,test}/000123.png   rendered slide
    labels/{train,val,test}/000123.txt   YOLO boxes (class cx cy w h, normalised)
    meta/000123.json                     full labels: text, font, size, colour, lines...
    masks/000123.png                     instance map (0 = background, k = element k)
    clean/000123.png                     same slide with all text removed (inpainting target)
    data.yaml                            YOLO dataset config
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from . import CLASSES
from .build import build_deck
from .labels import decode_idmap, make_labels
from .layout import random_slide
from .render import render


def split_of(idx, ratios):
    r = (idx * 0.6180339887) % 1.0          # deterministic, well spread
    return "train" if r < ratios[0] else ("val" if r < ratios[0] + ratios[1] else "test")


def _strip(spec):
    s = json.loads(json.dumps(spec, default=float))
    s["background"].pop("src", None)
    for el in s["elements"]:
        el.pop("src", None)
    return s


def run_batch(job):
    start, count, a = job
    work = tempfile.mkdtemp(prefix=f"slidegen_{start}_", dir=a["tmp"])
    try:
        assets = os.path.join(work, "assets")
        os.makedirs(assets)
        specs = []
        for idx in range(start, start + count):
            rng = np.random.default_rng([a["seed"], idx])
            specs.append(random_slide(rng, assets, a["photo_dir"], a["zh_ratio"]))
        profile = os.path.join(a["tmp"], f"lo_profile_{os.getpid()}")
        modes = ["full", "idmap"] + ([] if a["no_clean"] else ["clean"])
        pages = {}
        for mode in modes:
            path = os.path.join(work, f"{mode}.pptx")
            build_deck(specs, path, mode)
            pages[mode] = render(path, os.path.join(work, mode), profile, antialias=(mode != "idmap"))
            if len(pages[mode]) != count:
                raise RuntimeError(f"{mode}: expected {count} pages, got {len(pages[mode])}")
        out, ext = a["out"], a["ext"]
        n_el = 0
        for j, idx in enumerate(range(start, start + count)):
            name, split = f"{idx:06d}", split_of(idx, a["ratios"])
            spec = specs[j]
            idmap = np.asarray(Image.open(pages["idmap"][j]).convert("RGB"))
            inst = decode_idmap(idmap, len(spec["elements"]))
            records, yolo = make_labels(spec, inst)
            n_el += len(records)
            img = Image.open(pages["full"][j]).convert("RGB")
            img.save(os.path.join(out, "images", split, f"{name}.{ext}"), quality=95)
            with open(os.path.join(out, "labels", split, f"{name}.txt"), "w") as f:
                f.write("\n".join(yolo) + ("\n" if yolo else ""))
            meta = dict(name=name, split=split, layout=spec["layout"], theme=_strip(spec)["theme"],
                        background=_strip(spec)["background"], elements=records,
                        image=f"images/{split}/{name}.{ext}")
            with open(os.path.join(out, "meta", f"{name}.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, default=float)
            Image.fromarray(inst).save(os.path.join(out, "masks", f"{name}.png"))
            if "clean" in pages:
                Image.open(pages["clean"][j]).convert("RGB").save(
                    os.path.join(out, "clean", f"{name}.{ext}"), quality=95)
        return count, n_el
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=200, help="number of slides")
    ap.add_argument("--start", type=int, default=0, help="first slide index (to extend a dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=25, help="slides per LibreOffice call")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--photo-dir", default=None, help="optional folder of real photos to use as pictures")
    ap.add_argument("--zh-ratio", type=float, default=0.3, help="share of slides with Chinese text")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test")
    ap.add_argument("--no-clean", action="store_true", help="skip the text-free render (faster)")
    ap.add_argument("--jpg", action="store_true", help="save JPEG instead of PNG (smaller)")
    a = ap.parse_args()

    out = a.out
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    for d in ("meta", "masks") + (() if a.no_clean else ("clean",)):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    tmp = tempfile.mkdtemp(prefix="slidegen_")
    args = dict(out=out, seed=a.seed, photo_dir=a.photo_dir, zh_ratio=a.zh_ratio,
                ratios=[float(x) for x in a.ratios.split(",")], no_clean=a.no_clean,
                ext="jpg" if a.jpg else "png", tmp=tmp)
    jobs = [(s, min(a.batch, a.start + a.n - s), args) for s in range(a.start, a.start + a.n, a.batch)]
    t0, done, n_el = time.time(), 0, 0
    try:
        with Pool(a.workers) as pool:
            for c, e in pool.imap_unordered(run_batch, jobs):
                done, n_el = done + c, n_el + e
                rate = done / (time.time() - t0)
                print(f"{done}/{a.n} slides, {n_el} elements, {rate:.1f} slides/s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

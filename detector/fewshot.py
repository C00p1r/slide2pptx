"""Few-shot: fine-tune a synthetic-trained detector on a few hand-labelled real slides.

    python -m detector.fewshot runs/ratio/r0.50/weights/last.pt --real data/real/yolo \\
        --synth path/to/pool_tpl path/to/pool_rand --device cpu

The labelled slides are split into k folds by deck (default 2), so no deck is in both
train and test. Each fold fine-tunes on the other folds and tests on its own slides, so
every labelled slide is tested exactly once. Per fold it evaluates, on the same slides:
  base  the model as it is (no fine-tuning)
  real  fine-tuned on the real training slides only
  mix   fine-tuned on the real slides plus the same number of synthetic slides
        (guards against forgetting; --synth pools, sampled evenly)
Writes <out>/fewshot.md and fewshot.json (mean over folds, and each fold). Finished
evaluations are skipped on a re-run, so an interrupted run picks up where it stopped.
"""
import argparse
import glob
import json
import os
import random
import re
import shutil

import yaml

from . import CLASSES
from .evaluate import evaluate
from .train import SLIDE_AUG

EXT = (".png", ".jpg", ".jpeg", ".webp")
VARIANTS = ("base", "real", "mix")


def deck_of(stem):
    return re.sub(r"[_-]\d+$", "", stem)


def folds_by_deck(images, k, seed):
    decks = {}
    for f in images:
        decks.setdefault(deck_of(os.path.splitext(os.path.basename(f))[0]), []).append(f)
    names = sorted(decks)
    random.Random(seed).shuffle(names)
    folds = [[] for _ in range(k)]
    for d in sorted(names, key=lambda d: -len(decks[d])):      # biggest decks first, to the emptiest fold
        min(folds, key=len).extend(decks[d])
    return [sorted(f) for f in folds]


def sample_synth(pools, n, seed, dst):
    """Copy n synthetic train slides (spread evenly over the pools) with labels into dst."""
    rng, out = random.Random(seed), []
    for i, pool in enumerate(pools):
        files = sorted(f for f in glob.glob(os.path.join(pool, "images", "train", "*")) if f.lower().endswith(EXT))
        if not files:
            raise SystemExit(f"no images in {pool}/images/train")
        take = n // len(pools) + (1 if i < n % len(pools) else 0)
        for f in rng.sample(files, min(take, len(files))):
            stem = os.path.splitext(os.path.basename(f))[0]
            name = f"p{i}_{os.path.basename(f)}"
            for sub, src in (("images", f), ("labels", os.path.join(pool, "labels", "train", stem + ".txt"))):
                os.makedirs(os.path.join(dst, sub), exist_ok=True)
                target = os.path.join(dst, sub, name if sub == "images" else f"p{i}_{stem}.txt")
                if not os.path.exists(target):
                    shutil.copy(src, target)
            out.append(os.path.abspath(os.path.join(dst, "images", name)))
    return out


def write_yaml(d, name, train, test):
    with open(os.path.join(d, f"{name}_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test) + "\n")
    p = os.path.join(d, f"{name}.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(path=d, train=f"{name}_train.txt", val="test.txt", test="test.txt",
                            names={i: c for i, c in enumerate(CLASSES)}), f, sort_keys=False, allow_unicode=True)
    return p


def finetune(base, data, run_dir, a):
    from ultralytics import YOLO

    from .augment import sim2real

    w = os.path.join(run_dir, "weights", "last.pt")
    if os.path.exists(w) and os.path.exists(os.path.join(run_dir, "done")):
        return w
    YOLO(base).train(
        data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        project=os.path.dirname(run_dir), name=os.path.basename(run_dir), exist_ok=True, seed=a.seed,
        patience=0, val=False, plots=False, freeze=a.freeze, optimizer="AdamW", lr0=a.lr0, warmup_epochs=1,
        mosaic=a.mosaic, close_mosaic=max(1, a.epochs // 3), **SLIDE_AUG, augmentations=sim2real(),
    )
    open(os.path.join(run_dir, "done"), "w").close()
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="synthetic-trained model to start from")
    ap.add_argument("--real", default="data/real/yolo", help="output of tools/real_labels.py build")
    ap.add_argument("--synth", nargs="*", default=[], help="synthetic pool roots for the mix variant")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--variants", default="base,real,mix")
    ap.add_argument("--repeat", type=int, default=0, help="copies of each real slide per epoch (0: ~40 per epoch)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (10 = backbone)")
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--mosaic", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/fewshot")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    variants = [v for v in a.variants.split(",") if v]
    if "mix" in variants and not a.synth:
        ap.error("the mix variant needs --synth pool folders (or pass --variants base,real)")
    out = os.path.abspath(a.out)
    images = sorted(os.path.abspath(f) for f in glob.glob(os.path.join(a.real, "images", "test", "*"))
                    if f.lower().endswith(EXT))
    if not images:
        raise SystemExit(f"no images in {a.real}/images/test (run tools/real_labels.py build first)")
    folds = folds_by_deck(images, a.folds, a.seed)
    results = {v: [] for v in variants}
    for k, test in enumerate(folds):
        train = [f for j, fold in enumerate(folds) if j != k for f in fold]
        rep = a.repeat or max(1, round(40 / len(train)))
        d = os.path.join(out, f"fold{k}")
        os.makedirs(d, exist_ok=True)
        print(f"fold {k}: train {len(train)} slides x{rep}, test {len(test)} slides "
              f"({', '.join(sorted({deck_of(os.path.splitext(os.path.basename(f))[0]) for f in test}))})")
        for v in variants:
            tag = f"fold{k}_{v}"
            js = os.path.join(out, "eval", f"{tag}.json")
            if not os.path.exists(js):
                if v == "base":
                    data, w = write_yaml(d, "base", train, test), a.weights
                else:
                    tr = train * rep
                    if v == "mix":
                        tr += sample_synth(a.synth, len(tr), a.seed + k, os.path.join(d, "synth"))
                    data = write_yaml(d, v, tr, test)
                    w = finetune(a.weights, data, os.path.join(out, "runs", tag), a)
                evaluate(w, data, "test", a.imgsz, a.device, os.path.join(out, "eval"), tag)
            results[v].append(dict(json.load(open(js)), fold=k, test=[os.path.basename(f) for f in test]))
    report(results, out)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def report(results, out):
    fmt = lambda v: "–" if v is None else f"{v:.3f}"   # noqa: E731
    summary = {}
    for v, rs in results.items():
        summary[v] = dict(
            mAP50=mean([r["all"]["mAP50"] for r in rs]), mAP50_95=mean([r["all"]["mAP50_95"] for r in rs]),
            per_class={c: mean([r["per_class"][i]["mAP50_95"] for r in rs]) for i, c in enumerate(CLASSES)},
            recall=mean([r["all"]["R"] for r in rs]), precision=mean([r["all"]["P"] for r in rs]))
    md = ["## Few-shot fine-tuning on real slides (mean over folds)", "",
          "| model | mAP@0.5 | mAP@0.5:0.95 | P | R | " + " | ".join(CLASSES) + " |",
          "|---|---|---|---|---|" + "---|" * len(CLASSES)]
    for v, s in summary.items():
        md.append(f"| {v} | {fmt(s['mAP50'])} | {fmt(s['mAP50_95'])} | {fmt(s['precision'])} | {fmt(s['recall'])} | "
                  + " | ".join(fmt(s["per_class"][c]) for c in CLASSES) + " |")
    md += ["", "Per-class columns are mAP@0.5:0.95.", "", "### Per fold", "",
           "| fold | model | test slides | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    for v, rs in results.items():
        for r in rs:
            md.append(f"| {r['fold']} | {v} | {len(r['test'])} | {fmt(r['all']['mAP50'])} | {fmt(r['all']['mAP50_95'])} |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(out, "fewshot.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "fewshot.json"), "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, folds=results), f, indent=2, default=float, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()

"""Build training sets with a chosen share of random-layout slides.

Generate two pools once, then mix them for any ratio R (no re-rendering):
    python -m slidegen.generate --out data/pool_tpl  --n 2500 --random-ratio 0 --seed 0 --no-clean --jpg
    python -m slidegen.generate --out data/pool_rand --n 2500 --random-ratio 1 --seed 1 --no-clean --jpg
    python -m detector.mix --tpl data/pool_tpl --rand data/pool_rand --ratios 0,0.2,0.5,0.8,1 --out data/mix

For ratio R, data/mix/r0.20/data.yaml trains on round(R*N) random slides plus
N - round(R*N) template slides (same for val). Every ratio has the same N, and the
sets are nested (taking prefixes of one fixed shuffle), so the only thing that changes
between ratios is the mix. Use different --seed for the two pools so slide i of each
pool does not share a theme.
"""
import argparse
import glob
import os
import random

import yaml

from . import CLASSES

EXT = (".png", ".jpg", ".jpeg")


def pool_files(root, split, seed):
    files = sorted(os.path.abspath(f) for f in glob.glob(os.path.join(root, "images", split, "*"))
                   if f.lower().endswith(EXT))
    if not files:
        raise SystemExit(f"no images in {root}/images/{split}")
    random.Random(seed).shuffle(files)
    return files


def take(tpl, rand, ratio, n):
    k = round(ratio * n)
    if k > len(rand) or n - k > len(tpl):
        raise SystemExit(f"ratio {ratio}: need {n - k} template + {k} random, pools have {len(tpl)} + {len(rand)}")
    return tpl[: n - k] + rand[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tpl", required=True, help="pool generated with --random-ratio 0")
    ap.add_argument("--rand", required=True, help="pool generated with --random-ratio 1")
    ap.add_argument("--ratios", default="0,0.2,0.5,0.8,1")
    ap.add_argument("--n-train", type=int, default=None, help="default: the smaller pool's train split")
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--out", default="data/mix")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    split = {s: (pool_files(a.tpl, s, a.seed), pool_files(a.rand, s, a.seed)) for s in ("train", "val")}
    n = {"train": a.n_train or min(map(len, split["train"])), "val": a.n_val or min(map(len, split["val"]))}
    for r in (float(x) for x in a.ratios.split(",")):
        d = os.path.abspath(os.path.join(a.out, f"r{r:.2f}"))
        os.makedirs(d, exist_ok=True)
        for s in ("train", "val"):
            with open(os.path.join(d, f"{s}.txt"), "w") as f:
                f.write("\n".join(take(*split[s], r, n[s])) + "\n")
        with open(os.path.join(d, "data.yaml"), "w") as f:
            yaml.safe_dump(dict(path=d, train="train.txt", val="val.txt",
                                names={i: c for i, c in enumerate(CLASSES)}), f, sort_keys=False)
        print(f"R={r:.2f}: train {n['train']} ({round(r * n['train'])} random), "
              f"val {n['val']} ({round(r * n['val'])} random) -> {d}/data.yaml")


if __name__ == "__main__":
    main()

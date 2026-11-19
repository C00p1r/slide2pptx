"""Table and plot for the random-ratio experiment (docs/experiments.md).

    python -m detector.sweep results/detector --prefix ratio

Reads <prefix>_r<R>_<testset>.json written by detector.evaluate (testset = tpl, rand
or real) and writes <prefix>_sweep.md and <prefix>_sweep.png into the same folder.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from . import CLASSES

TESTSETS = [("tpl", "template slides (synthetic)"), ("rand", "random slides (synthetic)"),
            ("real", "NotebookLM slides (real)")]
COLORS = {"tpl": "#2a78d6", "rand": "#eb6834", "real": "#1baf7a"}   # dataviz categorical slots 1-3
MARKERS = {"tpl": "o", "rand": "s", "real": "D"}


def load(folder, prefix):
    res = defaultdict(dict)                      # res[testset][ratio] = summary
    pat = re.compile(rf"^{re.escape(prefix)}_r([0-9.]+)_(\w+)\.json$")
    for p in glob.glob(os.path.join(folder, f"{prefix}_r*_*.json")):
        m = pat.match(os.path.basename(p))
        if m:
            res[m.group(2)][float(m.group(1))] = json.load(open(p))
    return res


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def tables(res):
    ratios = sorted({r for t in res.values() for r in t})
    sets = [t for t, _ in TESTSETS if t in res]
    out = ["| R (random share) | " + " | ".join(f"{t} mAP@0.5 | {t} mAP@0.5:0.95" for t in sets) + " |",
           "|---|" + "---|---|" * len(sets)]
    for r in ratios:
        cells = []
        for t in sets:
            s = res[t].get(r)
            cells += [fmt(s and s["all"]["mAP50"]), fmt(s and s["all"]["mAP50_95"])]
        out.append(f"| {r:.2f} | " + " | ".join(cells) + " |")
    for t in sets:                                # per-class detail, mAP@0.5:0.95
        out += ["", f"Per class on **{t}**, mAP@0.5:0.95", "",
                "| R | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for r in ratios:
            s = res[t].get(r)
            per = {c["cls"]: c["mAP50_95"] for c in s["per_class"]} if s else {}
            out.append(f"| {r:.2f} | " + " | ".join(fmt(per.get(c)) for c in CLASSES) + " |")
    return "\n".join(out) + "\n"


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, key, name in zip(axes, ("mAP50", "mAP50_95"), ("mAP@0.5", "mAP@0.5:0.95")):
        for t, label in TESTSETS:
            if t not in res:
                continue
            rs = sorted(res[t])
            ys = [res[t][r]["all"][key] for r in rs]
            ax.plot(rs, ys, color=COLORS[t], marker=MARKERS[t], markersize=7, linewidth=2, label=label)
            ax.annotate(f"{ys[-1]:.2f}", (rs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#52514e")
        ax.set_title(name, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("share of random-layout slides in training (R)", color="#52514e")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="results/detector")
    ap.add_argument("--prefix", default="ratio")
    a = ap.parse_args()
    res = load(a.folder, a.prefix)
    if not res:
        raise SystemExit(f"no {a.prefix}_r*_*.json in {a.folder}")
    text = tables(res)
    print(text)
    with open(os.path.join(a.folder, f"{a.prefix}_sweep.md"), "w") as f:
        f.write(text)
    plot(res, os.path.join(a.folder, f"{a.prefix}_sweep.png"))
    print(f"wrote {a.prefix}_sweep.md and {a.prefix}_sweep.png in {a.folder}")


if __name__ == "__main__":
    main()

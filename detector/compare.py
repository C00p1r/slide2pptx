"""One table comparing several evaluation runs (for the report).

    python -m detector.compare results/detector/*_real.json --out results/detector/compare_real.md

Reads the .json files written by detector.evaluate. One row per run, with mAP@0.5 and
mAP@0.5:0.95 overall and per class.
"""
import argparse
import json
import os

from . import CLASSES


def table(paths, metric):
    head = ["run", "all"] + CLASSES
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in paths:
        s = json.load(open(p))
        per = {r["cls"]: r[metric] for r in s["per_class"]}
        cells = [os.path.splitext(os.path.basename(p))[0], s["all"][metric]] + [per.get(c) for c in CLASSES]
        rows.append("| " + " | ".join(c if isinstance(c, str) else ("–" if c is None else f"{c:.3f}")
                                      for c in cells) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="evaluate.py .json files")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text = (f"#### mAP@0.5\n\n{table(a.results, 'mAP50')}\n\n"
            f"#### mAP@0.5:0.95\n\n{table(a.results, 'mAP50_95')}\n")
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()

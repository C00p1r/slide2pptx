"""Per-class mAP for the results table.

    python -m detector.evaluate runs/detect/slides/weights/best.pt --data data/synth/data.yaml --split test
    python -m detector.evaluate runs/detect/slides/weights/best.pt --data data/real/data.yaml --split test --tag real

Writes <out>/<tag>.json and <tag>.md (mAP@0.5 and mAP@0.5:0.95 per class).
"""
import argparse
import json
import os

from . import CLASSES
from .data import local_yaml


def evaluate(weights, data, split="test", imgsz=1280, device=None, out="results/detector", tag=None, conf=0.001):
    from ultralytics import YOLO

    tag = tag or f"synth_{split}"
    res = YOLO(weights).val(data=local_yaml(data), split=split, imgsz=imgsz, device=device, conf=conf,
                            rect=True, plots=True, project=os.path.abspath(out), name=tag, exist_ok=True)
    box = res.box
    present = set(int(c) for c in box.ap_class_index)
    rows = []
    for i, name in enumerate(CLASSES):
        if i in present:
            p, r, ap50, ap = box.class_result(list(box.ap_class_index).index(i))
            n = int(res.nt_per_class[i]) if hasattr(res, "nt_per_class") else None
            rows.append(dict(cls=name, instances=n, P=p, R=r, mAP50=ap50, mAP50_95=ap))
        else:
            rows.append(dict(cls=name, instances=0, P=None, R=None, mAP50=None, mAP50_95=None))
    summary = dict(weights=os.path.abspath(weights), data=os.path.abspath(data), split=split, imgsz=imgsz,
                   images=int(getattr(res, "seen", 0)) or None,
                   all=dict(P=box.mp, R=box.mr, mAP50=box.map50, mAP50_95=box.map), per_class=rows)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    fmt = lambda v: "–" if v is None else f"{v:.3f}"   # noqa: E731
    md = [f"### {tag} ({split}, imgsz {imgsz})", "", "| class | instances | P | R | mAP@0.5 | mAP@0.5:0.95 |",
          "|---|---|---|---|---|---|"]
    for r in rows + [dict(cls="**all**", instances=sum(r["instances"] or 0 for r in rows), **summary["all"])]:
        md.append(f"| {r['cls']} | {r['instances']} | {fmt(r['P'])} | {fmt(r['R'])} | "
                  f"{fmt(r['mAP50'])} | {fmt(r['mAP50_95'])} |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(out, f"{tag}.md"), "w") as f:
        f.write(text)
    print(text)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/detector")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    evaluate(a.weights, a.data, a.split, a.imgsz, a.device, a.out, a.tag)


if __name__ == "__main__":
    main()

"""Train the slide element detector.

    python -m detector.train --data data/synth/data.yaml --model yolo11s.pt --epochs 60
    python -m detector.train --data data/synth/data.yaml --no-sim2real --name no_aug   # ablation

Defaults are tuned for slides: large images (small text and icons), no flips
(text has a direction), little mosaic (slides have a fixed layout), and extra
JPEG / blur / downscale augmentation for the sim-to-real gap.
"""
import argparse
import os

from .data import local_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data.yaml from slidegen.generate (not needed with --resume)")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n/s/m.pt, yolov8s.pt, or a .pt to fine-tune")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto (60%% of GPU memory); T4 at 1280 ~ 8")
    ap.add_argument("--rect", action="store_true", help="16:9 batches (less padding, but no shuffle/mosaic)")
    ap.add_argument("--mosaic", type=float, default=0.2)
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--no-sim2real", action="store_true", help="turn off the extra augmentation (ablation)")
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="slides")
    ap.add_argument("--device", default=None, help="0, cpu, ... (default: auto)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=False, help="ram / disk / False")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction of the data (data-size study)")
    ap.add_argument("--resume", action="store_true", help="resume runs/<project>/<name>/weights/last.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO

    if a.resume:
        YOLO(f"{os.path.abspath(a.project)}/{a.name}/weights/last.pt").train(resume=True)
        return

    if not a.data:
        ap.error("--data is required unless --resume")
    from .augment import sim2real

    data = local_yaml(a.data)
    YOLO(a.model).train(
        data=data, epochs=a.epochs, imgsz=a.imgsz, batch=int(a.batch) if a.batch >= 1 else a.batch,
        rect=a.rect, device=a.device, workers=a.workers, cache=a.cache, patience=a.patience,
        project=os.path.abspath(a.project), name=a.name, seed=a.seed, fraction=a.fraction, exist_ok=False,
        # geometry: slides are upright, left-to-right, and fill the frame
        fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.05, scale=0.3, mosaic=a.mosaic, close_mosaic=a.close_mosaic, mixup=0.0,
        # colour: themes vary a lot, so push hue/saturation more than the default
        hsv_h=0.1, hsv_s=0.6, hsv_v=0.4,
        augmentations=None if a.no_sim2real else sim2real(a.aug_strength),
    )


if __name__ == "__main__":
    main()

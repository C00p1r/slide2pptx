"""Sim-to-real augmentation.

Synthetic slides are perfectly sharp PNGs; real NotebookLM slides are upscaled,
JPEG-compressed and softer, and their colours differ. These extra transforms run
on top of YOLO's own colour / scale / translate augmentation (see train.py).
"""


def sim2real(strength=1.0):
    import albumentations as A

    p = lambda x: min(1.0, x * strength)   # noqa: E731
    return [
        A.ImageCompression(quality_range=(35, 95), p=p(0.5)),
        A.Downscale(scale_range=(0.4, 0.9), p=p(0.3)),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5))], p=p(0.25)),
        A.GaussNoise(std_range=(0.01, 0.04), p=p(0.2)),
        A.Sharpen(alpha=(0.1, 0.3), p=p(0.1)),
        A.RandomBrightnessContrast(0.15, 0.15, p=p(0.3)),
        A.ToGray(p=p(0.03)),
    ]

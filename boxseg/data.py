"""Torch datasets. Scenes are generated on the fly, so every batch is new data."""
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .synth import scene


def _rng(seed):
    wi = get_worker_info()
    return np.random.default_rng([seed, wi.id if wi else 0, np.random.SeedSequence().entropy % 2**32])


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255


class SegStream(IterableDataset):
    """(image, inside, edge) triples for Level 1."""

    def __init__(self, W=512, H=288, exclude=(), bg_dir=None, seed=0):
        self.W, self.H, self.exclude, self.seed = W, H, tuple(exclude), seed
        self.bg = sorted(glob.glob(f"{bg_dir}/*.png") + glob.glob(f"{bg_dir}/*.jpg")) if bg_dir else []

    def __iter__(self):
        rng = _rng(self.seed)
        while True:
            s = scene(rng, self.W, self.H, self.exclude, self.bg)
            yield (to_tensor(s["image"]), torch.from_numpy(s["inside"]).float(),
                   torch.from_numpy(s["edge"]).float())


# ------------------------------------------------------------------ Level 2 pairs

PAIR_SIZE = 64


def candidate_pairs(boxes, W, H):
    """Pairs of boxes close enough that grouping them is a real question."""
    if len(boxes) < 2:
        return []
    b = np.asarray(boxes, np.float32).reshape(len(boxes), -1)[:, :4]
    out = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            gx = max(0, max(b[i, 0], b[j, 0]) - min(b[i, 2], b[j, 2]))
            gy = max(0, max(b[i, 1], b[j, 1]) - min(b[i, 3], b[j, 3]))
            if np.hypot(gx, gy) < 0.12 * W:
                out.append((i, j))
    return out


def pair_input(img, a, b, size=PAIR_SIZE):
    """5-channel crop (RGB + mask A + mask B) around two boxes, plus 8 geometry numbers."""
    H, W = img.shape[:2]
    a, b = np.asarray(a[:4], np.float32), np.asarray(b[:4], np.float32)
    u = np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])
    m = 0.15 * max(u[2] - u[0], u[3] - u[1]) + 4
    x0, y0 = int(max(0, u[0] - m)), int(max(0, u[1] - m))
    x1, y1 = int(min(W, u[2] + m)), int(min(H, u[3] + m))
    crop = cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sx, sy = size / max(1, x1 - x0), size / max(1, y1 - y0)
    masks = np.zeros((2, size, size), np.float32)
    for k, r in enumerate((a, b)):
        masks[k, int((r[1] - y0) * sy):int(np.ceil((r[3] - y0) * sy)),
              int((r[0] - x0) * sx):int(np.ceil((r[2] - x0) * sx))] = 1
    x = np.concatenate([crop.transpose(2, 0, 1), masks], 0)

    area = lambda r: max(1.0, (r[2] - r[0]) * (r[3] - r[1]))
    sa, sb = np.sqrt(area(a)), np.sqrt(area(b))
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    geo = np.array([
        ((b[0] + b[2]) - (a[0] + a[2])) / 2 / sa, ((b[1] + b[3]) - (a[1] + a[3])) / 2 / sa,
        np.log(sb / sa), inter / area(a), inter / area(b), np.hypot(gx, gy) / min(sa, sb),
        np.log(sa / W * 10), np.log(sb / W * 10)], np.float32)
    return x, geo


def jitter(rng, box, f=0.03):
    w, h = box[2] - box[0], box[3] - box[1]
    return [box[0] + rng.normal(0, f) * w, box[1] + rng.normal(0, f) * h,
            box[2] + rng.normal(0, f) * w, box[3] + rng.normal(0, f) * h]


class PairStream(IterableDataset):
    """(crop, geo, same_group) for Level 2, from ground-truth boxes with small jitter."""

    def __init__(self, W=512, H=288, exclude=(), seed=1, per_scene=16):
        self.W, self.H, self.exclude, self.seed, self.k = W, H, tuple(exclude), seed, per_scene

    def __iter__(self):
        rng = _rng(self.seed)
        while True:
            s = scene(rng, self.W, self.H, self.exclude)
            bx = s["boxes"]
            pairs = candidate_pairs([b["box"] for b in bx], self.W, self.H)
            if not pairs:
                continue
            pos = [p for p in pairs if bx[p[0]]["group"] == bx[p[1]]["group"]]
            neg = [p for p in pairs if bx[p[0]]["group"] != bx[p[1]]["group"]]
            pick = []                                       # roughly balanced classes
            for lst in (pos, neg):
                if lst:
                    pick += [lst[i] for i in rng.choice(len(lst), min(len(lst), self.k // 2), replace=False)]
            for i, j in pick:
                if rng.random() < 0.5:
                    i, j = j, i
                x, geo = pair_input(s["image"], jitter(rng, bx[i]["box"]), jitter(rng, bx[j]["box"]))
                yield (torch.from_numpy(x), torch.from_numpy(geo),
                       torch.tensor(float(bx[i]["group"] == bx[j]["group"])))

"""Find the card behind detected text: recover `shape` boxes from `text` boxes.

YOLO predicts every box on its own, so it often finds the text on a card but not the card
(on the 20 real slides, 34 of 98 text-in-shape pairs had only the text found). A card is a
region of one fill colour that differs from the slide background. For each text box this:
  1. reads the colour of a thin ring just outside the text box,
  2. skips it when that colour matches the slide background, because the text sits on the slide,
  3. grows the connected region of that colour from the ring (cv2.floodFill),
  4. keeps the region's bounding box as a `shape` if it contains the text, is clearly bigger
     than it, is not most of the slide, is mostly filled (card-like, not a thin stroke), and
     has an edge: the colour just outside it jumps (a background gradient fades instead).
Boxes that a detected shape/picture already covers are left alone, and several texts on one
card give one shape.
"""
import numpy as np

from .postprocess import inside, iou

WORK_W = 960   # analyse a downscaled copy for speed


def _bg_colour(img):
    """Median colour of the slide's outer 3% frame."""
    h, w = img.shape[:2]
    b = max(2, int(0.03 * min(h, w)))
    frame = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                            img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def _ring(img, x0, y0, x1, y1, m):
    h, w = img.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    mask = np.zeros((h, w), bool)
    mask[Y0:Y1, X0:X1] = True
    mask[max(0, y0):y1, max(0, x0):x1] = False
    return mask


def find_cards(image, dets, conf=0.3, bg_tol=8.0, fill_tol=18, min_grow=1.15, max_frac=0.6, min_fill=0.55):
    """image: path or RGB array. dets: list of dict(cls, conf, bbox=[x, y, w, h]) in pixels.
    -> list of new shape dets (not including dets)."""
    import cv2

    if isinstance(image, str):
        # cv2.imread cannot open non-ASCII paths on Windows (most of our slide names are Chinese)
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(image)
    H, W = img.shape[:2]
    s = min(1.0, WORK_W / W)
    small = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    small = cv2.GaussianBlur(small, (3, 3), 0)   # flatten JPEG noise and gradients a little
    h, w = small.shape[:2]
    bg = _bg_colour(small)
    covers = [d["bbox"] for d in dets if d["cls"] in ("shape", "picture")]
    found = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if d["cls"] not in ("text", "title"):
            continue
        tb = d["bbox"]
        if any(inside(tb, c) >= 0.9 for c in covers + [f["bbox"] for f in found]):
            continue
        x0, y0 = int(tb[0] * s), int(tb[1] * s)
        x1, y1 = int((tb[0] + tb[2]) * s) + 1, int((tb[1] + tb[3]) * s) + 1
        m = max(3, int(0.25 * min(y1 - y0, x1 - x0)))
        ring = _ring(small, x0, y0, x1, y1, m)
        if ring.sum() < 20:
            continue
        px = small[ring].astype(float)
        fill = np.median(px, axis=0)
        gap = float(np.linalg.norm(fill - bg))
        if gap < bg_tol:
            continue                                  # text sits on the slide background
        tol = int(np.clip(0.4 * gap, 3, fill_tol))    # never wide enough to leak into the background
        # the ring must be mostly that one colour, not a busy picture
        if (np.linalg.norm(px - fill, axis=1) < 2 * tol).mean() < 0.6:
            continue
        seeds = np.argwhere(ring & (np.linalg.norm(small.astype(float) - fill, axis=2) < tol))
        if not len(seeds):
            continue
        flood = np.zeros((h + 2, w + 2), np.uint8)
        work = small.copy()
        for sy, sx in seeds[:: max(1, len(seeds) // 8)]:   # several seeds: the ring may be cut by the text
            if flood[sy + 1, sx + 1]:
                continue
            cv2.floodFill(work, flood, (int(sx), int(sy)), (0, 0, 0), (tol,) * 3, (tol,) * 3,
                          flags=4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
        region = flood[1:-1, 1:-1] > 0
        ys, xs = np.nonzero(region)
        if not len(xs):
            continue
        cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        cx0, cy0, cx1, cy1 = min(cx0, x0), min(cy0, y0), max(cx1, x1), max(cy1, y1)
        cw, ch = cx1 - cx0, cy1 - cy0
        if cw * ch > max_frac * w * h:
            continue                                  # that is the slide, not a card
        if cw * ch < (min_grow ** 2) * (x1 - x0) * (y1 - y0):
            continue                                  # barely bigger than the text: a text highlight
        # a card has an edge: just outside it the colour jumps. A gradient band fades out instead.
        edges = sum((cx0 <= 1, cy0 <= 1, cx1 >= w - 1, cy1 >= h - 1))
        if edges >= 3:
            continue
        outer = _ring(small, cx0, cy0, cx1, cy1, 3) & ~region
        if outer.sum() >= 20 and np.linalg.norm(np.median(small[outer].astype(float), axis=0) - fill) < max(bg_tol, 2 * tol):
            continue
        text_area = (x1 - x0) * (y1 - y0)
        if (region[cy0:cy1, cx0:cx1].sum() + text_area) / (cw * ch) < min_fill:
            continue                                  # thin strokes or a ring, not a filled card
        box = [cx0 / s, cy0 / s, cw / s, ch / s]
        if any(iou(box, c) >= 0.5 for c in covers):
            continue
        same = next((f for f in found if iou(box, f["bbox"]) >= 0.7), None)
        if same is None:
            found.append(dict(cls="shape", conf=round(conf, 3), bbox=[round(v, 1) for v in box], source="card"))
    return found

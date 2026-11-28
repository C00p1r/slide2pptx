"""Style-randomised synthetic slides for class-agnostic box segmentation.

Unlike slidegen/ (python-pptx + LibreOffice, real PowerPoint styles), this draws
slides directly with numpy/OpenCV so that every box can get styles PowerPoint
cannot render: hand-drawn wobbly outlines, sketchy double strokes, sticky notes,
glow, soft shadows, dashed lines, hatching, paper textures. The aim is that the
network cannot rely on style and has to learn geometry: "this is a closed region".

One call to `scene(rng)` returns:
    image   HxWx3 uint8
    inside  HxW   uint8  1 where a pixel belongs to any box (outline included)
    edge    HxW   uint8  1 on a box/background boundary, 2 on a box/box boundary
    inst    HxW   int32  visible instance map, 0 = background, k = boxes[k-1]
    boxes   list of dict(box=[x0,y0,x1,y1], group=g, kind=..., style=[...])

`group` is the id of the widget the box was drawn as part of (a card and its
header strip, a number badge and its label box, a stack of cards, ...). Level 2
(grouping) learns to recover it.
"""
import glob
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slidegen import content

# every style a box can be drawn in; --exclude-styles removes some for held-out tests
STYLES = ["plain", "rounded", "ellipse", "hand", "dashed", "double", "sticky",
          "shadow", "glow", "gradient", "hatch", "outline_only", "fill_only"]
MIN_BOX = 10          # px at the output size; smaller closed shapes are not labelled


# ------------------------------------------------------------------ fonts / text

@lru_cache(None)
def _font_files():
    pats = ["/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf", "/usr/share/fonts/**/*.ttc",
            os.path.expanduser("~/.fonts/**/*.ttf")]
    files = sorted({f for p in pats for f in glob.glob(p, recursive=True)})
    bad = ("emoji", "symbol", "mono", "math")
    latin = [f for f in files if not any(b in f.lower() for b in bad) and "cjk" not in f.lower()]
    cjk = [f for f in files if "cjk" in f.lower() or "wqy" in f.lower()]
    return latin or files, cjk


@lru_cache(512)
def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _text_layer(rng, W, H, items):
    """items: list of (x, y, w, h, color, ...). Draw random text lines into each area.

    Returns colour, alpha, a mask of the drawn text lines (the text target), and per item
    the tight box of the whole block (None if nothing was drawn).
    """
    latin, cjk = _font_files()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    line_mask = np.zeros((H, W), np.uint8)
    blocks = []
    for x, y, w, h, col, *_ in items:
        if w < 8 or h < 6:
            blocks.append(None)
            continue
        blk = [np.inf, np.inf, -np.inf, -np.inf]
        zh = bool(cjk) and rng.random() < 0.25
        path = str(rng.choice(cjk if zh else latin))
        size = int(np.clip(rng.uniform(0.25, 0.5) * h, 7, max(8, H * 0.09)))
        f = _font(path, size)
        n_lines = max(1, min(int(rng.integers(1, 4)), int(h // (size * 1.3))))
        yy = y + (h - n_lines * size * 1.3) / 2
        for _ in range(n_lines):
            words = content.sentence(rng, int(rng.integers(1, 6)), "zh" if zh else "en")
            while len(words) > 1 and d.textlength(words, font=f) > w:
                words = words[:-1]
            tw = d.textlength(words, font=f)
            if tw > w * 1.05:                       # one long word that does not fit: skip the line
                yy += size * 1.3
                continue
            xx = x + rng.choice([0.05 * w, (w - tw) / 2])
            d.text((xx, yy), words, font=f, fill=tuple(int(c) for c in col) + (255,))
            bb = d.textbbox((xx, yy), words, font=f)
            if bb[2] > bb[0] and bb[3] > bb[1]:
                x0, y0 = int(max(0, bb[0])), int(max(0, bb[1]))
                x1, y1 = int(min(W, np.ceil(bb[2]))), int(min(H, np.ceil(bb[3])))
                line_mask[y0:y1, x0:x1] = 1
                blk = [min(blk[0], x0), min(blk[1], y0), max(blk[2], x1), max(blk[3], y1)]
            yy += size * 1.3
        blocks.append([float(v) for v in blk] if blk[2] > blk[0] and blk[3] > blk[1] else None)
    a = np.asarray(layer).astype(np.float32) / 255
    return a[..., :3] * 255, a[..., 3], line_mask, blocks


# ------------------------------------------------------------------ geometry

def _densify(poly, step=3.0):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = max(1, int(np.hypot(*(b - a)) / step))
        t = np.arange(n)[:, None] / n
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def _rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def _rounded(x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    pts = []
    for cx, cy, a0 in [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                       (x + r, y + h - r, 90), (x + r, y + r, 180)]:
        a = np.deg2rad(np.linspace(a0, a0 + 90, 8))
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
    return np.concatenate(pts).astype(np.float32)


def _ellipse(x, y, w, h):
    a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    return np.stack([x + w / 2 + w / 2 * np.cos(a), y + h / 2 + h / 2 * np.sin(a)], 1).astype(np.float32)


def _rotate(poly, deg):
    c = poly.mean(0)
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    return (poly - c) @ R.T + c


def _wobble(rng, poly, amp):
    """Low-frequency noise along the outline: looks hand-drawn, stays closed."""
    p = _densify(poly, 3.0)
    n = len(p)
    t = np.arange(n) / n * 2 * np.pi
    off = np.zeros((n, 2), np.float32)
    for k in rng.integers(2, 9, 3):
        off += rng.normal(0, amp / 2, 2) * np.sin(k * t + rng.uniform(0, 2 * np.pi))[:, None]
    return p + off + rng.normal(0, amp * 0.1, p.shape)


def _shrink(poly, d):
    c = poly.mean(0)
    span = np.maximum(np.abs(poly - c).max(0), 1)
    return c + (poly - c) * np.maximum(span - d, 1) / span


# ------------------------------------------------------------------ drawing primitives

S = 4  # cv2 sub-pixel shift bits


def _ipts(poly):
    return [np.round(poly * (1 << S)).astype(np.int32).reshape(-1, 1, 2)]


def _fill_alpha(shape, poly):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, _ipts(poly), 255, cv2.LINE_AA, shift=S)
    return m.astype(np.float32) / 255


def _stroke_alpha(shape, poly, t, closed=True, dashed=False):
    m = np.zeros(shape, np.uint8)
    t = max(1, int(round(t)))
    if dashed:
        p = _densify(poly, 2.0)
        seg = max(3, int(t * 3))
        for i in range(0, len(p) - seg, 2 * seg):
            cv2.polylines(m, _ipts(p[i:i + seg]), False, 255, t, cv2.LINE_AA, shift=S)
    else:
        cv2.polylines(m, _ipts(poly), closed, 255, t, cv2.LINE_AA, shift=S)
    return m.astype(np.float32) / 255


def _over(img, color, alpha):
    a = alpha[..., None]
    img *= (1 - a)
    img += np.asarray(color, np.float32) * a


def _rand_color(rng, avoid=None, min_dist=60):
    for _ in range(20):
        c = np.array(content.hsv(rng.random(), rng.uniform(0, 0.9), rng.uniform(0.15, 1.0)), np.float32)
        if avoid is None or np.abs(c - avoid).sum() > min_dist:
            return c
    return 255 - np.asarray(avoid, np.float32)


# ------------------------------------------------------------------ backgrounds

def _background(rng, W, H, bg_files=()):
    if bg_files and rng.random() < 0.3:         # a real slide/photo, heavily blurred
        im = Image.open(str(rng.choice(bg_files))).convert("RGB").resize((W, H))
        arr = cv2.GaussianBlur(np.asarray(im, np.float32), (0, 0), rng.uniform(8, 25))
        return arr, arr.reshape(-1, 3).mean(0)
    c1 = _rand_color(rng)
    if rng.random() < 0.6:                      # most slides: light, near-flat
        c1 = np.array(content.hsv(rng.random(), rng.uniform(0, 0.15), rng.uniform(0.9, 1)), np.float32)
    c2 = np.clip(c1 + rng.normal(0, 25, 3), 0, 255)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, np.pi)
    t = (np.cos(a) * xx / W + np.sin(a) * yy / H)
    t = (t - t.min()) / max(np.ptp(t), 1e-6)
    img = c1 * (1 - t[..., None]) + c2 * t[..., None]
    k = rng.random()
    if k < 0.2:                                 # paper / grunge texture
        n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), rng.uniform(1, 6))
        img += (n / (n.std() + 1e-6) * rng.uniform(3, 12))[..., None]
    elif k < 0.3:                               # grid / dot paper (open lines: not boxes)
        step = int(rng.integers(12, 40))
        col = np.clip(c1 - rng.uniform(15, 50), 0, 255)
        g = np.zeros((H, W), np.float32)
        g[::step, :] = 1
        g[:, ::step] = 1
        _over(img, col, g * rng.uniform(0.3, 0.8))
    img += rng.normal(0, rng.uniform(0, 4), img.shape)
    return img, c1


# ------------------------------------------------------------------ one box

def _pick_style(rng, allowed, kind):
    s = set()
    shape = rng.choice([x for x in ["plain", "rounded", "ellipse"] if x in allowed] or ["plain"],
                       p=None)
    if kind == "sticky" and "sticky" in allowed:
        return {"sticky", "fill_only", "shadow"} | ({"hand"} if "hand" in allowed and rng.random() < 0.2 else set())
    s.add(str(shape))
    for name, p in [("hand", 0.25), ("dashed", 0.08), ("double", 0.06), ("shadow", 0.25),
                    ("glow", 0.12), ("gradient", 0.15), ("hatch", 0.06)]:
        if name in allowed and rng.random() < p:
            s.add(name)
    fill_mode = rng.choice(["both", "outline_only", "fill_only"], p=[0.45, 0.25, 0.30])
    if fill_mode != "both" and fill_mode in allowed:
        s.add(str(fill_mode))
    if "hatch" in s:
        s.discard("fill_only")
    return s


def _draw_box(rng, img, inst_alpha_out, x, y, w, h, style, bg_col, fill_col=None, line_col=None, t=None):
    """Draw one box on img. Returns the box's solid mask (for the instance map).
    fill_col / line_col / t override the random choices (tables, icons share them)."""
    H, W = img.shape[:2]
    shp = (H, W)
    if "ellipse" in style:
        poly = _ellipse(x, y, w, h)
    elif "rounded" in style or "sticky" in style and rng.random() < 0.3:
        poly = _rounded(x, y, w, h, rng.uniform(0.08, 0.35) * min(w, h))
    else:
        poly = _rect(x, y, w, h)
    if "sticky" in style:
        poly = _rotate(_densify(poly), rng.uniform(-6, 6))
    if "hand" in style:
        poly = _wobble(rng, poly, rng.uniform(0.8, 2.5) * max(1.0, min(w, h) / 80))

    fixed = (fill_col, line_col, t)
    fill_col = _rand_color(rng, bg_col, 70)
    if "sticky" in style:
        fill_col = np.array(content.hsv(rng.choice([0.14, 0.95, 0.33, 0.55, 0.08]) + rng.normal(0, 0.02),
                                        rng.uniform(0.25, 0.6), rng.uniform(0.9, 1.0)), np.float32)
    line_col = _rand_color(rng, bg_col, 120) if rng.random() < 0.6 else np.clip(fill_col * 0.6, 0, 255)
    t = float(np.clip(rng.lognormal(0.5, 0.5), 1, 7)) * (1.5 if "hand" in style else 1)
    fill_col = fill_col if fixed[0] is None else np.asarray(fixed[0], np.float32)
    line_col = line_col if fixed[1] is None else np.asarray(fixed[1], np.float32)
    t = t if fixed[2] is None else fixed[2]

    has_fill = "outline_only" not in style
    has_line = "fill_only" not in style
    fill_a = _fill_alpha(shp, poly)

    if "shadow" in style:
        dx, dy = rng.uniform(-2, 6, 2) * max(1, t / 2)
        sh = cv2.GaussianBlur(np.roll(fill_a, (int(dy), int(dx)), (0, 1)), (0, 0), rng.uniform(1.5, 6))
        _over(img, np.clip(bg_col * 0.35, 0, 255), sh * rng.uniform(0.25, 0.6))
    if "glow" in style:
        gl = cv2.GaussianBlur(cv2.dilate(fill_a, np.ones((5, 5))), (0, 0), rng.uniform(4, 12))
        _over(img, _rand_color(rng), np.clip(gl * rng.uniform(0.6, 1.2), 0, 1))

    if has_fill:
        alpha = rng.uniform(0.55, 1.0) if rng.random() < 0.3 else 1.0
        if "gradient" in style:
            c2 = _rand_color(rng)
            yy = (np.arange(H, dtype=np.float32)[:, None] - y) / max(h, 1)
            tt = np.clip(yy, 0, 1)[..., None] * np.ones((1, W, 1), np.float32)
            layer = fill_col * (1 - tt) + c2 * tt
            a = (fill_a * alpha)[..., None]
            img *= 1 - a
            img += layer * a
        else:
            _over(img, fill_col, fill_a * alpha)
        if "sticky" in style:                      # curl: darker bottom corner
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            curl = np.clip(1 - np.hypot(xx - (x + w), yy - (y + h)) / (0.4 * min(w, h)), 0, 1)
            _over(img, fill_col * 0.75, fill_a * curl * 0.6)
    if "hatch" in style:
        hl = np.zeros(shp, np.float32)
        step = int(rng.integers(4, 10))
        ang = rng.choice([1, -1])
        for k in range(-H, W, step):
            cv2.line(hl, (k, 0 if ang > 0 else H), (k + H, H if ang > 0 else 0), 1.0, 1, cv2.LINE_AA)
        _over(img, line_col, hl * fill_a * 0.8)

    line_a = np.zeros(shp, np.float32)
    if has_line or not has_fill and "hatch" not in style:
        if "hand" in style:
            p = _densify(poly, 3)
            for _ in range(int(rng.integers(1, 3))):  # sketchy: 1-2 overlapping strokes, open ends
                q = p + rng.normal(0, 0.6, p.shape)
                s0 = int(rng.integers(0, len(q)))
                q = np.concatenate([q[s0:], q[:s0], q[:int(rng.integers(0, max(2, len(q) // 20)))]])
                line_a = np.maximum(line_a, _stroke_alpha(shp, q, t * rng.uniform(0.6, 1.1), closed=False))
        else:
            line_a = _stroke_alpha(shp, poly, t, dashed="dashed" in style)
        if "double" in style:
            line_a = np.maximum(line_a, _stroke_alpha(shp, _shrink(poly, t + 3), max(1, t / 2)))
        _over(img, line_col, line_a)

    solid = (fill_a > 0.5) | (cv2.dilate(_stroke_alpha(shp, poly, t), np.ones((2, 2))) > 0.5)
    ys, xs = np.nonzero(solid)
    return solid, ([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                   if len(xs) else None)


# ------------------------------------------------------------------ widgets (= groups)

def _widget(rng, x, y, w, h, allowed, extras=True):
    """Return a list of (x, y, w, h, kind) boxes making one widget, drawn back to front.

    With extras (v3 data): tables of touching cells, pictures, illustrations and small
    icons built from 2-4 simple shapes; these return special kinds handled in scene().
    """
    kinds = ["single", "card", "badge", "stack", "sticky", "pills", "table", "picture", "illustration", "icon"]
    p = np.array([0.30, 0.25, 0.15, 0.08, 0.12, 0.10] + ([0.12, 0.08, 0.06, 0.08] if extras else [0, 0, 0, 0]))
    if "sticky" not in allowed:
        p[kinds.index("sticky")] = 0
    k = rng.choice(kinds, p=p / p.sum())
    if k == "single" or min(w, h) < 4 * MIN_BOX:
        return [(x, y, w, h, "box")]
    if k == "table":                   # grid of cells that share their borders
        rows, cols = int(rng.integers(2, 6)), int(rng.integers(2, 5))
        ch = h * rng.uniform(0.6, 1.0) / rows
        xs = np.cumsum(np.r_[0, rng.dirichlet(np.ones(cols) * 3)]) * w + x
        out = []
        for r in range(rows):
            for c in range(cols):
                out.append((xs[c], y + r * ch, xs[c + 1] - xs[c], ch, "cell_head" if r == 0 else "cell"))
        return out
    if k in ("picture", "illustration"):
        return [(x, y, w, h, k)]
    if k == "icon":                    # an icon next to a short label (label text added in scene)
        s = min(h, w) * rng.uniform(0.25, 0.5)
        return [(x, y + (h - s) / 2, s, s, "icon")]
    if k == "sticky":
        s = min(w, h) * rng.uniform(0.7, 1.0)
        return [(x + (w - s) / 2, y + (h - s) / 2, s, s, "sticky")]
    if k == "card":                    # outer box + header strip and/or inner boxes
        out = [(x, y, w, h, "card")]
        m = rng.uniform(0.05, 0.12) * min(w, h)
        if rng.random() < 0.5:
            out.append((x, y, w, h * rng.uniform(0.18, 0.3), "header"))
        if rng.random() < 0.6:
            n = int(rng.integers(1, 3))
            iw = (w - (n + 1) * m) / n
            ih = h * rng.uniform(0.3, 0.5)
            iy = y + h - m - ih
            for i in range(n):
                out.append((x + m + i * (iw + m), iy, iw, ih, "inner"))
        if len(out) == 1:
            s = min(w, h) * rng.uniform(0.2, 0.35)
            out.append((x + m, y + m, s, s, "icon" if extras and rng.random() < 0.5 else "icon_circle"))
        elif extras and rng.random() < 0.25:
            s = min(w, h) * rng.uniform(0.15, 0.25)
            out.append((x + w - m - s, y + m, s, s, "icon"))
        return out
    if k == "badge":                   # number circle overlapping a label box
        s = min(h, w * 0.4) * rng.uniform(0.6, 0.9)
        bx = x + s * 0.5
        return [(bx, y + (h - s * 0.9) / 2, w - s * 0.5, s * 0.9, "label"),
                (x, y + (h - s) / 2, s, s, "badge")]
    if k == "stack":                   # cards stacked with an offset
        n = int(rng.integers(2, 4))
        d = rng.uniform(0.04, 0.1) * min(w, h)
        cw, ch = w - (n - 1) * d, h - (n - 1) * d
        return [(x + i * d, y + (n - 1 - i) * d, cw, ch, "stacked") for i in range(n)]
    # pills: a long bar split into touching segments (progress bar / segmented control)
    n = int(rng.integers(2, 5))
    bh = min(h, rng.uniform(0.25, 0.5) * w / n)
    by = y + (h - bh) / 2
    return [(x + i * w / n, by, w / n, bh, "segment") for i in range(n)]


def _layout(rng, W, H):
    """Areas for widgets: a grid with jitter, or random non-overlapping rectangles."""
    top = H * rng.uniform(0.15, 0.25) if rng.random() < 0.7 else H * 0.05
    areas = []
    if rng.random() < 0.6:
        rows, cols = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        gx, gy = rng.uniform(0.01, 0.05) * W, rng.uniform(0.02, 0.06) * H
        m = W * rng.uniform(0.03, 0.1)
        cw = (W - 2 * m - (cols - 1) * gx) / cols
        ch = (H - top - H * 0.05 - (rows - 1) * gy) / rows
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.9:
                    areas.append((m + c * (cw + gx), top + r * (ch + gy), cw, ch))
    else:
        for _ in range(60):
            if len(areas) >= rng.integers(2, 9):
                break
            w = rng.uniform(0.1, 0.45) * W
            h = rng.uniform(0.1, 0.45) * H
            x, y = rng.uniform(0, W - w), rng.uniform(top, max(top + 1, H - h))
            if all(x + w + 4 < a[0] or a[0] + a[2] + 4 < x or y + h + 4 < a[1] or a[1] + a[3] + 4 < y
                   for a in areas):
                areas.append((x, y, w, h))
    return top, areas


def _distractors(rng, img, W, H, centers):
    """Open strokes that are NOT boxes: connectors, arrows, underlines, scribbles."""
    col = _rand_color(rng)
    for _ in range(int(rng.integers(0, 5))):
        a = np.zeros((H, W), np.float32)
        t = int(rng.integers(1, 4))
        if len(centers) >= 2 and rng.random() < 0.6:          # connector between two widgets
            i, j = rng.choice(len(centers), 2, replace=False)
            p0, p1 = np.array(centers[i]), np.array(centers[j])
            if rng.random() < 0.5:                             # elbow connector
                mid = np.array([p1[0], p0[1]])
                pts = np.stack([p0, mid, p1])
            else:
                pts = np.stack([p0, p1])
            cv2.polylines(a, _ipts(pts.astype(np.float32)), False, 1.0, t, cv2.LINE_AA, shift=S)
            if rng.random() < 0.6:
                d = pts[-1] - pts[-2]
                d = d / (np.linalg.norm(d) + 1e-6) * rng.uniform(6, 12)
                nrm = np.array([-d[1], d[0]]) * 0.6
                tri = np.stack([pts[-1], pts[-1] - d + nrm, pts[-1] - d - nrm]).astype(np.float32)
                cv2.fillPoly(a, _ipts(tri), 1.0, cv2.LINE_AA, shift=S)
        else:                                                  # scribble / underline / open bracket
            n = int(rng.integers(2, 6))
            p = np.cumsum(rng.normal(0, 30, (n, 2)), 0) + rng.uniform([0, 0], [W, H])
            cv2.polylines(a, _ipts(p.astype(np.float32)), False, 1.0, t, cv2.LINE_AA, shift=S)
        _over(img, col, a)


# ------------------------------------------------------------------ icons, pictures, illustrations

ICON_TEMPLATES = {    # parts as (x, y, w, h, shape) in a unit square
    "bulb":   [(0.15, 0.0, 0.7, 0.7, "ellipse"), (0.32, 0.68, 0.36, 0.3, "rounded")],
    "person": [(0.3, 0.0, 0.4, 0.4, "ellipse"), (0.1, 0.45, 0.8, 0.55, "rounded")],
    "doc":    [(0.15, 0.0, 0.7, 1.0, "plain"), (0.27, 0.2, 0.46, 0.1, "plain"), (0.27, 0.42, 0.46, 0.1, "plain"),
               (0.27, 0.64, 0.3, 0.1, "plain")],
    "target": [(0.0, 0.0, 1.0, 1.0, "ellipse"), (0.25, 0.25, 0.5, 0.5, "ellipse"), (0.42, 0.42, 0.16, 0.16, "ellipse")],
    "chat":   [(0.0, 0.05, 1.0, 0.65, "rounded"), (0.15, 0.62, 0.22, 0.3, "plain")],
    "chart":  [(0.05, 0.55, 0.2, 0.45, "plain"), (0.4, 0.3, 0.2, 0.7, "plain"), (0.75, 0.05, 0.2, 0.95, "plain")],
    "monitor": [(0.0, 0.0, 1.0, 0.7, "rounded"), (0.1, 0.1, 0.8, 0.5, "plain"), (0.35, 0.72, 0.3, 0.12, "plain"),
                (0.2, 0.85, 0.6, 0.12, "rounded")],
}


def _icon_parts(rng, x, y, s):
    name = list(ICON_TEMPLATES)[int(rng.integers(len(ICON_TEMPLATES)))]
    return [(x + px * s, y + py * s, pw * s, ph * s, shp) for px, py, pw, ph, shp in ICON_TEMPLATES[name]]


def _paste_picture(rng, img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(W, x + w)), int(min(H, y + h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    pic = np.asarray(content.photo_like(rng, x1 - x0, y1 - y0).convert("RGB"), np.float32)
    m = np.zeros((H, W), bool)
    if rng.random() < 0.3:                      # rounded corners
        a = _fill_alpha((H, W), _rounded(x0, y0, x1 - x0, y1 - y0, 0.1 * min(x1 - x0, y1 - y0))) > 0.5
        sub = a[y0:y1, x0:x1]
        img[y0:y1, x0:x1][sub] = pic[sub]
        m = a
    else:
        img[y0:y1, x0:x1] = pic
        m[y0:y1, x0:x1] = True
    return m


def _illustration(rng, img, x, y, w, h, allowed, bg_col):
    """Many small multi-colour blobs drawn on top of each other: one picture, not many boxes."""
    union = np.zeros(img.shape[:2], bool)
    for _ in range(int(rng.integers(5, 16))):
        bw, bh = rng.uniform(0.15, 0.5) * w, rng.uniform(0.15, 0.5) * h
        bx, by = x + rng.uniform(0, w - bw), y + rng.uniform(0, h - bh)
        st = _pick_style(rng, allowed, "box") - {"glow", "sticky"}
        st.discard("outline_only")
        solid, _ = _draw_box(rng, img, None, bx, by, bw, bh, st, bg_col)
        union |= solid
    return union


# ------------------------------------------------------------------ scene

def scene(rng, W=512, H=288, exclude=(), bg_files=(), extras=True):
    """extras=False reproduces the v1/v2 data (no tables, pictures, illustrations, icons)."""
    allowed = [s for s in STYLES if s not in set(exclude)]
    img, bg_col = _background(rng, W, H, bg_files)
    top, areas = _layout(rng, W, H)
    text_items = []
    if top > H * 0.1:                                          # unboxed title
        text_items.append((W * 0.05, H * 0.02, W * 0.9, top * 0.8, _rand_color(rng, bg_col, 200), "title", -1))

    widgets = []
    for g, (x, y, w, h) in enumerate(areas):
        s = rng.uniform(0.7, 1.0)                              # not every widget fills its cell
        w2, h2 = w * s, h * rng.uniform(0.6, 1.0)
        widgets.append(_widget(rng, x + (w - w2) * rng.random(), y + (h - h2) * rng.random(), w2, h2, allowed,
                               extras))
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in areas]
    if rng.random() < 0.5:
        _distractors(rng, img, W, H, centers)

    masks, boxes = [], []
    n_obj = 0

    def add(solid, bb, g, kind, st, cls, obj):
        masks.append(solid)
        boxes.append(dict(box=bb, group=g, obj=obj, cls=cls, kind=kind, style=sorted(st)))

    def mask_box(m):
        ys, xs = np.nonzero(m)
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else None

    for g, parts in enumerate(widgets):
        wstyle = _pick_style(rng, allowed, "box")              # siblings in a widget often share style
        if parts and parts[0][4] in ("cell", "cell_head"):     # table: shared colours and line width
            tst = {"plain"} | ({"outline_only"} if rng.random() < 0.3 else set())
            line = _rand_color(rng, bg_col, 120)
            t_line = float(rng.uniform(1, 2.5))
            fills = [_rand_color(rng, bg_col, 40) for _ in range(3)]
            zebra = rng.random() < 0.5
        for r_i, (x, y, w, h, kind) in enumerate(parts):
            if kind == "picture":
                m = _paste_picture(rng, img, x, y, w, h)
                if m is not None and m.any():
                    add(m, mask_box(m), g, kind, ["photo"], "picture", n_obj)
                    n_obj += 1
                continue
            if kind == "illustration":
                m = _illustration(rng, img, x, y, w, h, allowed, bg_col)
                if m.any():
                    add(m, mask_box(m), g, kind, ["illustration"], "picture", n_obj)
                    n_obj += 1
                continue
            if kind == "icon":                                 # 2-4 parts, one object
                # The instance target follows what is visible: a one-colour filled icon is one
                # silhouette (one instance); outlined or multi-colour parts are separate instances
                # that share the object id, for the pair CNN to merge ("same object").
                col = _rand_color(rng, bg_col, 120)
                mode = rng.choice(["silhouette", "outline", "multicolour"], p=[0.4, 0.3, 0.3])
                ist = {"outline_only"} if mode == "outline" else {"fill_only"}
                lw = float(rng.uniform(1, 2))
                union, n_parts = np.zeros((H, W), bool), 0
                for i, (px, py, pw, ph, shp) in enumerate(_icon_parts(rng, x, y, w)):
                    if pw < 2 or ph < 2:
                        continue
                    pcol = col if mode != "multicolour" or i == 0 else _rand_color(rng, col, 150)
                    if mode == "silhouette" and i > 0 and px > x + 0.1 * w and px + pw < x + 0.9 * w \
                            and py > y + 0.05 * w and py + ph < y + 0.95 * w:
                        pcol = bg_col                              # holes (doc lines, target rings)
                    solid, bb = _draw_box(rng, img, None, px, py, pw, ph, ist | {shp}, bg_col,
                                          fill_col=pcol, line_col=pcol, t=lw)
                    if bb is None:
                        continue
                    if mode == "silhouette":
                        union |= solid
                    else:
                        add(solid, bb, g, "icon_part", ist | {shp}, "icon", n_obj)
                        n_parts += 1
                if mode == "silhouette" and union.any():
                    add(union, mask_box(union), g, "icon", ["silhouette"], "icon", n_obj)
                n_obj += 1
                if len(parts) == 1:                            # label next to a standalone icon
                    text_items.append((x + w * 1.15, y, w * 3, w, _rand_color(rng, bg_col, 150), "text", g))
                continue
            if kind in ("cell", "cell_head"):
                row = int(round((y - parts[0][1]) / max(h, 1)))
                fc = fills[2] if kind == "cell_head" else (fills[row % 2] if zebra else fills[0])
                solid, bb = _draw_box(rng, img, None, x, y, w, h, tst, bg_col, fill_col=fc, line_col=line, t=t_line)
                if bb is not None:
                    add(solid, bb, g, kind, tst, "shape", n_obj)
                    n_obj += 1
                    if rng.random() < 0.75:
                        text_items.append((x + 0.08 * w, y + 0.1 * h, 0.84 * w, 0.8 * h,
                                           _rand_color(rng, bg_col, 150), "text", g))
                continue
            if w < MIN_BOX or h < MIN_BOX:
                continue
            st = _pick_style(rng, allowed, "sticky" if kind == "sticky" else "box")
            if kind in ("stacked", "segment") or rng.random() < 0.3:
                st = set(wstyle)
            if kind in ("badge", "icon_circle") and "ellipse" in allowed:
                st = (st - {"plain", "rounded"}) | {"ellipse"}
            if kind in ("header", "segment", "inner", "label", "badge") and "outline_only" in st:
                st.discard("outline_only")                     # keep children visibly distinct
            solid, bb = _draw_box(rng, img, None, x, y, w, h, st, bg_col)
            if bb is None:
                continue
            add(solid, bb, g, kind, st, "shape", n_obj)
            n_obj += 1
            if rng.random() < 0.7 and kind not in ("segment",):
                text_items.append((x + 0.08 * w, y + 0.15 * h, 0.84 * w, 0.7 * h,
                                   _rand_color(rng, bg_col, 150), "text", g))
    for _ in range(int(rng.integers(0, 4))):                   # loose text outside boxes
        tw, th = rng.uniform(0.15, 0.4) * W, rng.uniform(0.05, 0.12) * H
        text_items.append((rng.uniform(0, W - tw), rng.uniform(0, H - th), tw, th, _rand_color(rng, bg_col, 150),
                           "text", -1))

    tc, ta, text_mask, blocks = _text_layer(rng, W, H, text_items)
    img = img * (1 - ta[..., None]) + tc * ta[..., None]
    n_groups = len(widgets)
    texts = []
    for it, bb in zip(text_items, blocks):
        if bb is None:
            continue
        g = it[6] if it[6] >= 0 else n_groups + len(texts)    # loose text is its own group
        texts.append(dict(box=bb, group=int(g), obj=-1, cls=it[5]))

    inst = np.zeros((H, W), np.int32)                          # later boxes are on top
    for k, m in enumerate(masks):
        inst[m] = k + 1
    keep = [k for k in range(len(masks)) if (inst == k + 1).sum() >= MIN_BOX * MIN_BOX / 4]
    remap = np.zeros(len(masks) + 1, np.int32)
    remap[[k + 1 for k in keep]] = np.arange(1, len(keep) + 1)
    inst = remap[inst]
    boxes = [boxes[k] for k in keep]
    for t in texts:                                            # text objects get ids after the boxes
        t["obj"] = n_obj
        n_obj += 1

    edge = _instance_edges(inst)
    img = _degrade(rng, np.clip(img, 0, 255).astype(np.uint8))
    return dict(image=img, inside=(inst > 0).astype(np.uint8), edge=edge, inst=inst, boxes=boxes,
                text=text_mask, texts=texts)


def _instance_edges(inst, width=2):
    """1 = boundary between a box and the background, 2 = boundary between two boxes.

    Box/box boundaries (a card and its header, a badge on its label) are what splits
    nested boxes apart, and they are rare and low-contrast, so the loss weights them up
    (the same idea as the border weight map in the original U-Net paper).
    """
    def diff(a, b):
        return a != b, (a != b) & (a > 0) & (b > 0)
    e, inner = np.zeros(inst.shape, bool), np.zeros(inst.shape, bool)
    for a, b, sa, sb in [(inst[:, 1:], inst[:, :-1], np.s_[:, 1:], np.s_[:, :-1]),
                         (inst[1:, :], inst[:-1, :], np.s_[1:, :], np.s_[:-1, :])]:
        d, di = diff(a, b)
        e[sa] |= d
        e[sb] |= d
        inner[sa] |= di
        inner[sb] |= di
    if width > 1:
        k = np.ones((width, width), np.uint8)
        e = cv2.dilate(e.astype(np.uint8), k) > 0
        inner = cv2.dilate(inner.astype(np.uint8), k) > 0
    return (e.astype(np.uint8) + (inner & e)).astype(np.uint8)


def _degrade(rng, img):
    """Same sim-to-real noise as the YOLO pipeline: blur, JPEG, downscale."""
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        H, W = img.shape[:2]
        f = rng.uniform(0.5, 0.9)
        img = cv2.resize(cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA), (W, H))
    if rng.random() < 0.4:
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img)


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Preview a grid of synthetic box scenes.")
    ap.add_argument("--out", default="docs/img/boxseg_samples.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    tiles = []
    for _ in range(a.n):
        s = scene(rng)
        vis = s["image"].copy()
        vis[s["edge"] > 0] = (255, 0, 0)
        vis[(s["text"] > 0) & (s["edge"] == 0)] = vis[(s["text"] > 0) & (s["edge"] == 0)] // 2 + (0, 0, 120)
        cls_col = {"shape": (0, 160, 0), "picture": (255, 140, 0), "icon": (160, 0, 200),
                   "text": (0, 90, 255), "title": (220, 0, 0)}
        for b in s["boxes"] + s["texts"]:
            x0, y0, x1, y1 = map(int, b["box"])
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), cls_col[b["cls"]], 1)
        tiles.append(np.concatenate([s["image"], vis], 1))
    print(f"{(time.time() - t0) / a.n * 1000:.0f} ms / scene")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(tiles, 0)).save(a.out)
    print("wrote", a.out)

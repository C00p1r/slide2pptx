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
    """items: list of (x, y, w, h, color). Draw random text lines into each area."""
    latin, cjk = _font_files()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x, y, w, h, col in items:
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
            xx = x + rng.choice([0.05 * w, (w - tw) / 2])
            d.text((xx, yy), words, font=f, fill=tuple(int(c) for c in col) + (255,))
            yy += size * 1.3
    a = np.asarray(layer).astype(np.float32) / 255
    return a[..., :3] * 255, a[..., 3]


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


def _draw_box(rng, img, inst_alpha_out, x, y, w, h, style, bg_col):
    """Draw one box on img. Returns the box's solid mask (for the instance map)."""
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

    fill_col = _rand_color(rng, bg_col, 70)
    if "sticky" in style:
        fill_col = np.array(content.hsv(rng.choice([0.14, 0.95, 0.33, 0.55, 0.08]) + rng.normal(0, 0.02),
                                        rng.uniform(0.25, 0.6), rng.uniform(0.9, 1.0)), np.float32)
    line_col = _rand_color(rng, bg_col, 120) if rng.random() < 0.6 else np.clip(fill_col * 0.6, 0, 255)
    t = float(np.clip(rng.lognormal(0.5, 0.5), 1, 7)) * (1.5 if "hand" in style else 1)

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

def _widget(rng, x, y, w, h, allowed):
    """Return a list of (x, y, w, h, kind) boxes making one widget, drawn back to front."""
    kinds = ["single", "card", "badge", "stack", "sticky", "pills"]
    p = np.array([0.30, 0.25, 0.15, 0.08, 0.12, 0.10])
    if "sticky" not in allowed:
        p[kinds.index("sticky")] = 0
    k = rng.choice(kinds, p=p / p.sum())
    if k == "single" or min(w, h) < 4 * MIN_BOX:
        return [(x, y, w, h, "box")]
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
            out.append((x + m, y + m, s, s, "icon_circle"))
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


# ------------------------------------------------------------------ scene

def scene(rng, W=512, H=288, exclude=(), bg_files=()):
    allowed = [s for s in STYLES if s not in set(exclude)]
    img, bg_col = _background(rng, W, H, bg_files)
    top, areas = _layout(rng, W, H)
    text_items = []
    if top > H * 0.1:                                          # unboxed title
        text_items.append((W * 0.05, H * 0.02, W * 0.9, top * 0.8, _rand_color(rng, bg_col, 200)))

    widgets = []
    for g, (x, y, w, h) in enumerate(areas):
        s = rng.uniform(0.7, 1.0)                              # not every widget fills its cell
        w2, h2 = w * s, h * rng.uniform(0.6, 1.0)
        widgets.append(_widget(rng, x + (w - w2) * rng.random(), y + (h - h2) * rng.random(), w2, h2, allowed))
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in areas]
    if rng.random() < 0.5:
        _distractors(rng, img, W, H, centers)

    masks, boxes = [], []
    for g, parts in enumerate(widgets):
        wstyle = _pick_style(rng, allowed, "box")              # siblings in a widget often share style
        for (x, y, w, h, kind) in parts:
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
            masks.append(solid)
            boxes.append(dict(box=bb, group=g, kind=kind, style=sorted(st)))
            if rng.random() < 0.7 and kind not in ("segment",):
                text_items.append((x + 0.08 * w, y + 0.15 * h, 0.84 * w, 0.7 * h,
                                   _rand_color(rng, bg_col, 150)))
    for _ in range(int(rng.integers(0, 4))):                   # loose text outside boxes
        tw, th = rng.uniform(0.15, 0.4) * W, rng.uniform(0.05, 0.12) * H
        text_items.append((rng.uniform(0, W - tw), rng.uniform(0, H - th), tw, th, _rand_color(rng, bg_col, 150)))

    tc, ta = _text_layer(rng, W, H, text_items)
    img = img * (1 - ta[..., None]) + tc * ta[..., None]

    inst = np.zeros((H, W), np.int32)                          # later boxes are on top
    for k, m in enumerate(masks):
        inst[m] = k + 1
    keep = [k for k in range(len(masks)) if (inst == k + 1).sum() >= MIN_BOX * MIN_BOX / 4]
    remap = np.zeros(len(masks) + 1, np.int32)
    remap[[k + 1 for k in keep]] = np.arange(1, len(keep) + 1)
    inst = remap[inst]
    boxes = [boxes[k] for k in keep]

    edge = _instance_edges(inst)
    img = _degrade(rng, np.clip(img, 0, 255).astype(np.uint8))
    return dict(image=img, inside=(inst > 0).astype(np.uint8), edge=edge, inst=inst, boxes=boxes)


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
        for b in s["boxes"]:
            x0, y0, x1, y1 = map(int, b["box"])
            col = tuple(int(c) for c in content.hsv(b["group"] * 0.13, 0.9, 0.9))
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), col, 1)
        tiles.append(np.concatenate([s["image"], vis], 1))
    print(f"{(time.time() - t0) / a.n * 1000:.0f} ms / scene")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(tiles, 0)).save(a.out)
    print("wrote", a.out)

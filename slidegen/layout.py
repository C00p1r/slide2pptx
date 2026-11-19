"""Random slide layouts. A slide spec is plain JSON-able data (+ picture files).

Coordinates are pixels on a 1280x720 canvas; build.py converts them to EMU.
"""
import os
import uuid

import numpy as np

from . import CANVAS_H as H, CANVAS_W as W, PX_PER_PT, content, fonts

INSET = 5  # text frame inner margin, px (same on all sides)

ROLES = {
    "title":    dict(cls="title", pt=(28, 46), words=(2, 7),  paras=(1, 1), max_lines=2),
    "subtitle": dict(cls="text",  pt=(16, 24), words=(4, 12), paras=(1, 1), max_lines=2),
    "heading":  dict(cls="text",  pt=(15, 22), words=(1, 4),  paras=(1, 1), max_lines=2, bold=True),
    "body":     dict(cls="text",  pt=(13, 20), words=(6, 22), paras=(1, 3)),
    "bullets":  dict(cls="text",  pt=(15, 24), words=(4, 14), paras=(2, 6), bullet=True),
    "caption":  dict(cls="text",  pt=(11, 15), words=(2, 8),  paras=(1, 1), max_lines=2),
    "stat":     dict(cls="text",  pt=(34, 58), number=True,   paras=(1, 1), max_lines=1, bold=True),
    "label":    dict(cls="text",  pt=(16, 28), number=True,   paras=(1, 1), max_lines=1, bold=True),
}


# ------------------------------------------------------------------ text

def _measure(paras, fam, bold, size_px, width, lang, bullet):
    indent = size_px * 1.2 if bullet else 0
    lh = fonts.line_height(fam, bold, size_px)
    gap = size_px * 0.35 if len(paras) > 1 else 0
    wrapped = [fonts.wrap(p, fam, bold, size_px, width - indent, lang) for p in paras]
    n_lines = [len(w) for w in wrapped]
    h = sum(n_lines) * lh + gap * (len(paras) - 1)
    # a single word wider than the box would be broken mid-word by PowerPoint: treat as not fitting
    if any(fonts.text_width(line, fam, bold, size_px) > width - indent for w in wrapped for line in w):
        h = float("inf")
    return h, n_lines


def make_text(rng, th, role, x, y, w, max_h, align="l", color=None, bold=None):
    r = ROLES[role]
    lang = th["lang"]
    latin = th["title_font"] if role == "title" else th["body_font"]
    if r.get("number"):
        lang = "en"
    measure_font = th["cjk_font"] if lang == "zh" else latin
    if bold is None:
        bold = th["title_bold"] if role == "title" else r.get("bold", bool(rng.random() < 0.08))
    width = w - 2 * INSET
    for attempt in range(10):
        pt = float(rng.uniform(*r["pt"])) * (0.9 ** attempt)
        size_px = pt * PX_PER_PT
        if r.get("number"):
            paras = [content.number_text(rng)] if role == "stat" else [str(int(rng.integers(1, 10))).zfill(2)]
        else:
            n = int(rng.integers(r["paras"][0], r["paras"][1] + 1))
            paras = [content.sentence(rng, int(rng.integers(*r["words"]) ), lang) for _ in range(n)]
            if role in ("title", "heading") and lang == "en" and rng.random() < 0.6:
                paras = [content.title_case(p) for p in paras]
        while True:
            h, nl = _measure(paras, measure_font, bold, size_px, width, lang, r.get("bullet"))
            too_many = r.get("max_lines") and sum(nl) > r["max_lines"]
            if (h + 2 * INSET <= max_h and not too_many) or len(paras) <= 1:
                break
            paras.pop()
        if h + 2 * INSET <= max_h and not too_many:
            lh = fonts.line_height(measure_font, bold, size_px)
            frame_h = h + 2 * INSET + 0.25 * lh
            return dict(
                kind="text", cls=r["cls"], role=role, box=[x, y, w, frame_h],
                paras=paras, bullet=bool(r.get("bullet")), font=latin,
                ea_font=th["cjk_font"] if lang == "zh" else None, lang=lang,
                size_pt=round(pt, 2), bold=bool(bold),
                italic=bool(role in ("caption", "subtitle") and rng.random() < 0.15),
                color=list(color or (th["text"] if role != "caption" else th["muted"])),
                align=align,
            )
    return None


# --------------------------------------------------------------- helpers

class Builder:
    def __init__(self, rng, th, assets_dir, photo_dir):
        self.rng, self.th, self.assets, self.photo_dir = rng, th, assets_dir, photo_dir
        self.els = []

    def add(self, el):
        if el is not None:
            self.els.append(el)
        return el

    def text(self, role, x, y, w, max_h, **kw):
        return self.add(make_text(self.rng, self.th, role, x, y, w, max_h, **kw))

    def shape(self, kind, box, fill, alpha=None, line=None, line_w=0):
        el = dict(kind="shape", cls="shape", shape=kind, box=list(map(float, box)),
                  fill=list(fill) if fill is not None else None, alpha=alpha,
                  line=list(line) if line else None, line_w=line_w)
        if kind == "rounded":
            el["radius"] = float(self.rng.uniform(0.03, 0.15))
        return self.add(el)

    def _save(self, img, prefix):
        # uuid: names must never collide across slides in the same batch
        path = os.path.join(self.assets, f"{prefix}_{uuid.uuid4().hex}.png")
        img.save(path)
        return path

    def picture(self, box):
        x, y, w, h = box
        img = content.picture(self.rng, w, h, self.th, self.photo_dir)
        return self.add(dict(kind="picture", cls="picture", box=list(map(float, box)),
                             src=self._save(img, "pic")))

    def icon(self, cx, cy, size):
        img = content.icon(self.rng, self.th)
        s = size / max(img.width, img.height)
        w, h = img.width * s, img.height * s
        return self.add(dict(kind="icon", cls="icon", box=[cx - w / 2, cy - h / 2, w, h],
                             src=self._save(img, "icon")))

    def card_kind(self):
        return "rounded" if self.th["rounded"] else "rect"


def _title(b, m, top=None, align="l", max_h=0.24):
    y = top if top is not None else m * 0.7
    t = b.text("title", m, y, W - 2 * m, H * max_h, align=align,
               color=b.th["accent"] if b.rng.random() < 0.3 else None)
    bottom = (t["box"][1] + t["box"][3]) if t else y
    if b.rng.random() < 0.35:   # accent bar under the title
        bw = b.rng.uniform(60, 200)
        bx = m if align == "l" else (W - bw) / 2
        b.shape("rect", [bx, bottom + 4, bw, b.rng.uniform(4, 8)], b.th["accent"])
        bottom += 14
    return bottom + b.rng.uniform(14, 34)


def _footer(b, m):
    if b.rng.random() < 0.3:
        b.text("caption", W - m - 60, H - m * 0.6 - 18, 60, 40, align="r",
               color=b.th["muted"])
    if b.rng.random() < 0.15:
        b.text("caption", m, H - m * 0.6 - 18, 300, 40, color=b.th["muted"])


# --------------------------------------------------------------- layouts

def t_title(b, m):
    rng, th = b.rng, b.th
    split = rng.random() < 0.4
    tw = (W - 2 * m) * (0.5 if split else 1.0)
    align = "l" if split or rng.random() < 0.5 else "c"
    y = H * rng.uniform(0.28, 0.4)
    t = b.text("title", m, y, tw, H * 0.35, align=align)
    if t:
        b.text("subtitle", m, t["box"][1] + t["box"][3] + 10, tw, H * 0.2, align=align,
               color=th["muted"])
    if split:
        px = m + tw + 30
        b.picture([px, m, W - px - m * (0 if rng.random() < 0.5 else 1), H - 2 * m])
    elif rng.random() < 0.5:
        b.shape("rect", [0, H - rng.uniform(10, 40), W, 40], th["accent"])


def t_bullets(b, m):
    rng = b.rng
    y = _title(b, m)
    w = (W - 2 * m) * b.rng.uniform(0.7, 1.0)
    b.text("bullets", m, y, w, H - y - m)
    if w < (W - 2 * m) * 0.8 and rng.random() < 0.6:     # decorative icon in the free space
        free = W - m - (m + w)
        size = min(free * 0.7, rng.uniform(80, 180))
        b.icon(m + w + free / 2, y + (H - y - m) / 2, size)
    _footer(b, m)


def t_split(b, m):
    rng = b.rng
    y = _title(b, m)
    frac = rng.uniform(0.4, 0.58)
    gap = rng.uniform(24, 60)
    tw = (W - 2 * m - gap) * frac
    pw = W - 2 * m - gap - tw
    left_text = rng.random() < 0.6
    tx = m if left_text else m + pw + gap
    px = m + tw + gap if left_text else m
    role = "bullets" if rng.random() < 0.6 else "body"
    b.text(role, tx, y, tw, H - y - m)
    ph = min(H - y - m, pw * rng.uniform(0.55, 1.0))
    b.picture([px, y, pw, ph])
    _footer(b, m)


def t_cards(b, m):
    rng, th = b.rng, b.th
    y = _title(b, m)
    k = int(rng.integers(2, 5))
    grid = k == 4 and rng.random() < 0.4
    cols, rows = (2, 2) if grid else (k, 1)
    gap = rng.uniform(16, 36)
    cw = (W - 2 * m - gap * (cols - 1)) / cols
    ch = (H - y - m - gap * (rows - 1)) / rows
    ch = min(ch, rng.uniform(0.7, 1.0) * ch + 40)
    outline = th["muted"] if rng.random() < 0.3 else None
    with_icon = rng.random() < 0.85
    fit_cards = rows == 1 and rng.random() < 0.6
    cards, bottoms = [], []
    for i in range(k):
        cx = m + (i % cols) * (cw + gap)
        cy = y + (i // cols) * (ch + gap)
        card = b.shape(b.card_kind(), [cx, cy, cw, ch], th["card"], line=outline, line_w=1)
        pad = rng.uniform(14, 26)
        iy = cy + pad
        if with_icon:
            size = min(56, ch * 0.22)
            icx = cx + pad + size / 2 if rng.random() < 0.6 else cx + cw / 2
            if rng.random() < 0.4:
                b.shape("oval", [icx - size * 0.7, iy, size * 1.4, size * 1.4], th["accent"])
                b.icon(icx, iy + size * 0.7, size * 0.8)
                iy += size * 1.4 + 10
            else:
                b.icon(icx, iy + size / 2, size)
                iy += size + 10
        hd = b.text("heading", cx + pad, iy, cw - 2 * pad, ch * 0.3)
        if hd:
            iy = hd["box"][1] + hd["box"][3] + 4
        body = b.text("body", cx + pad, iy, cw - 2 * pad, cy + ch - iy - pad)
        cards.append(card)
        bottoms.append((body["box"][1] + body["box"][3] if body else iy) + pad)
    if fit_cards:                   # shrink all cards to the tallest content
        for card in cards:
            card["box"][3] = min(card["box"][3], max(bottoms) - card["box"][1])


def t_hero(b, m):
    rng, th = b.rng, b.th
    full = rng.random() < 0.6
    if full:
        b.picture([0, 0, W, H])
        dark = (10, 10, 20)
        if rng.random() < 0.5:
            b.shape("rect", [0, 0, W, H], dark, alpha=rng.uniform(0.35, 0.65))
            box = [m, H * 0.3, W * 0.6, H * 0.5]
        else:
            box = [m, H * rng.uniform(0.25, 0.45), W * rng.uniform(0.45, 0.6), H * 0.42]
            b.shape(b.card_kind(), [box[0] - 20, box[1] - 20, box[2] + 40, box[3] + 40],
                    dark, alpha=rng.uniform(0.5, 0.8))
        white = (250, 250, 250)
        t = b.text("title", box[0], box[1], box[2], box[3] * 0.6, color=white)
        if t:
            ty = t["box"][1] + t["box"][3] + 8
            b.text("subtitle", box[0], ty, box[2], box[1] + box[3] - ty, color=(220, 220, 220))
    else:
        side = rng.random() < 0.5
        pw = W * rng.uniform(0.38, 0.5)
        b.picture([0 if side else W - pw, 0, pw, H])
        tx = pw + m if side else m
        tw = W - pw - 2 * m
        t = b.text("title", tx, H * 0.25, tw, H * 0.3)
        if t:
            ty = t["box"][1] + t["box"][3] + 12
            b.text("body", tx, ty, tw, H - ty - m)


def t_gallery(b, m):
    rng = b.rng
    y = _title(b, m)
    k = int(rng.integers(2, 7))
    cols = k if k <= 3 else int(np.ceil(k / 2))
    rows = int(np.ceil(k / cols))
    gap = rng.uniform(16, 30)
    cw = (W - 2 * m - gap * (cols - 1)) / cols
    cap = rng.random() < 0.7
    cell_h = (H - y - m - gap * (rows - 1)) / rows
    ph = cell_h - (36 if cap else 0)
    for i in range(k):
        cx = m + (i % cols) * (cw + gap)
        cy = y + (i // cols) * (cell_h + gap)
        b.picture([cx, cy, cw, ph])
        if cap:
            b.text("caption", cx, cy + ph + 4, cw, 34, align=rng.choice(["l", "c"]))


def t_stats(b, m):
    rng, th = b.rng, b.th
    y = _title(b, m)
    k = int(rng.integers(2, 5))
    gap = rng.uniform(20, 40)
    cw = (W - 2 * m - gap * (k - 1)) / k
    on_card = rng.random() < 0.5
    stat_icons = rng.random() < 0.5
    ch = min(H - y - m, 360 if stat_icons else 300)
    for i in range(k):
        cx = m + i * (cw + gap)
        if on_card:
            b.shape(b.card_kind(), [cx, y, cw, ch], th["card"])
        sy0 = y + 20
        if stat_icons:
            size = rng.uniform(36, 56)
            b.icon(cx + cw / 2, sy0 + size / 2, size)
            sy0 += size + 8
        st = b.text("stat", cx + 10, sy0, cw - 20, 110, align="c", color=th["accent"])
        if st:
            sy = st["box"][1] + st["box"][3] + 6
            b.text("caption" if rng.random() < 0.5 else "body", cx + 14, sy, cw - 28,
                   y + ch - sy - 10, align="c")


def t_process(b, m):
    rng, th = b.rng, b.th
    y = _title(b, m)
    k = int(rng.integers(3, 6))
    gap = rng.uniform(16, 36)
    cw = (W - 2 * m - gap * (k - 1)) / k
    kind = rng.choice(["oval", "chevron", "rounded"])
    sh = min(90, cw * 0.6) if kind == "oval" else 70
    sw = sh if kind == "oval" else cw
    step_icons = rng.random() < 0.4
    if kind == "oval" and rng.random() < 0.7:     # connector line behind circles
        b.shape("rect", [m + cw / 2, y + sh / 2 - 2, W - 2 * m - cw, 4], th["muted"])
    for i in range(k):
        cx = m + i * (cw + gap)
        sx = cx + (cw - sw) / 2
        b.shape(kind, [sx, y, sw, sh], th["accent"] if i % 2 == 0 or rng.random() < 0.5 else th["accent2"])
        if step_icons:
            b.icon(sx + sw / 2, y + sh / 2, sh * 0.55)
        else:
            b.text("label", sx + 6, y + sh / 2 - 26, sw - 12, 52, align="c", color=(255, 255, 255))
        hd = b.text("heading", cx, y + sh + 16, cw, 70, align="c")
        ty = (hd["box"][1] + hd["box"][3] + 4) if hd else y + sh + 16
        b.text("body", cx, ty, cw, H - ty - m, align="c")


def t_iconlist(b, m):
    """Rows of icon + heading + text: very common in AI-generated decks."""
    rng, th = b.rng, b.th
    y = _title(b, m)
    k = int(rng.integers(3, 7))
    cols = 2 if k >= 4 and rng.random() < 0.6 else 1
    rows = int(np.ceil(k / cols))
    gap = rng.uniform(12, 24)
    cw = (W - 2 * m - 40 * (cols - 1)) / cols
    rh = (H - y - m - gap * (rows - 1)) / rows
    size = min(rh * 0.7, rng.uniform(36, 60))
    circle = rng.random() < 0.4
    for i in range(k):
        cx = m + (i % cols) * (cw + 40)
        cy = y + (i // cols) * (rh + gap)
        icx, icy = cx + size * 0.7, cy + size * 0.7
        if circle:
            b.shape("oval", [icx - size * 0.7, icy - size * 0.7, size * 1.4, size * 1.4],
                    th["accent"] if rng.random() < 0.7 else th["card"])
            b.icon(icx, icy, size * 0.8)
        else:
            b.icon(icx, icy, size)
        tx = cx + size * 1.4 + 16
        hd = b.text("heading", tx, cy, cx + cw - tx, rh * 0.45)
        ty = (hd["box"][1] + hd["box"][3]) if hd else cy
        if rh - (ty - cy) > 30:
            b.text("body", tx, ty, cx + cw - tx, cy + rh - ty)


def _overlaps(box, placed, pad=8):
    x, y, w, h = box
    for px, py, pw, ph in placed:
        if x < px + pw + pad and px < x + w + pad and y < py + ph + pad and py < y + h + pad:
            return True
    return False


def t_random(b, m):
    """Domain randomisation: element types scattered at random, no overlaps.

    Not chosen by template weight; used only through --random-ratio.
    """
    rng, th = b.rng, b.th
    placed = []
    if rng.random() < 0.5:                       # keep a normal title half the time
        y0 = _title(b, m)
        placed.append([0, 0, W, y0 - 10])
    else:                                        # or put it anywhere
        w = rng.uniform(300, W - 2 * m)
        el = make_text(rng, th, "title", rng.uniform(0, W - w), rng.uniform(0, H * 0.8), w, H * 0.3)
        if el:
            b.add(el)
            placed.append(el["box"])
    n = int(rng.integers(4, 13))
    for _ in range(n * 6):                       # attempts
        if len(placed) > n:
            break
        kind = rng.choice(["text", "text", "shape", "picture", "icon", "card"])
        if kind == "icon":
            size = rng.uniform(28, 120)
            box = [rng.uniform(0, W - size), rng.uniform(0, H - size), size, size]
            if not _overlaps(box, placed):
                b.icon(box[0] + size / 2, box[1] + size / 2, size)
                placed.append(box)
            continue
        w, h = rng.uniform(60, 600), rng.uniform(40, 360)
        box = [rng.uniform(0, W - w), rng.uniform(0, H - h), w, h]
        if kind == "text":
            role = str(rng.choice(["body", "bullets", "heading", "caption", "stat", "subtitle"]))
            el = make_text(rng, th, role, box[0], box[1], max(w, 120), h,
                           align=str(rng.choice(["l", "c", "r"])))
            if el and not _overlaps(el["box"], placed):
                b.add(el)
                placed.append(el["box"])
        elif _overlaps(box, placed):
            continue
        elif kind == "shape":
            b.shape(str(rng.choice(list(("rect", "rounded", "oval", "chevron", "pentagon")))), box,
                    th["accent"] if rng.random() < 0.5 else th["accent2"])
            placed.append(box)
        elif kind == "picture":
            b.picture(box)
            placed.append(box)
        else:                                    # card: shape with text on top
            b.shape(b.card_kind(), box, th["card"])
            b.text("body", box[0] + 12, box[1] + 12, w - 24, h - 24)
            placed.append(box)


LAYOUTS = {
    "title": (t_title, 1.0), "bullets": (t_bullets, 1.2), "split": (t_split, 1.6),
    "cards": (t_cards, 1.6), "hero": (t_hero, 0.8), "gallery": (t_gallery, 0.8),
    "stats": (t_stats, 0.8), "process": (t_process, 1.0), "iconlist": (t_iconlist, 1.2),
}


def random_slide(rng, assets_dir, photo_dir=None, zh_ratio=0.3, random_ratio=0.0):
    th = content.random_theme(rng, zh_ratio)
    names = list(LAYOUTS)
    p = np.array([LAYOUTS[n][1] for n in names]); p /= p.sum()
    name = str(rng.choice(names, p=p))
    # only draw the extra random number when asked, so ratio 0 reproduces old datasets exactly
    if random_ratio > 0 and rng.random() < random_ratio:
        name = "random"
    b = Builder(rng, th, assets_dir, photo_dir)
    m = float(rng.uniform(44, 96))
    (t_random if name == "random" else LAYOUTS[name][0])(b, m)
    bg = dict(style=th["bg_style"], color=list(th["bg"]))
    if th["bg_style"] != "solid":
        img = content.background_image(rng, th, W, H)
        bg["src"] = b._save(img, "bg")
    # keep everything on the canvas
    for el in b.els:
        x, y, w, h = el["box"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + w), min(float(H), y + h)
        el["box"] = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
    return dict(layout=name, theme={k: v for k, v in th.items()}, background=bg, elements=b.els)

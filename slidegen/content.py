"""Random content: words, colors/themes, and procedurally generated pictures."""
import colorsys
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import fonts

EN_WORDS = """
data model network learning image system design research method result
analysis process value growth market energy future team project impact
cloud vision signal neural layer feature training accuracy test score
student school city health water climate power design user product
strategy goal plan review summary overview key insight trend risk cost
time speed scale quality safety support service platform tool app
history culture science nature space ocean forest planet human brain
memory language story music art color shape light sound motion change
build create improve measure share connect explore discover reduce
increase protect deliver launch scale understand optimize compare
fast simple smart clear strong global local digital modern open new
better stable secure flexible efficient real early final main core
three four five ten first next step phase stage level part section
question answer problem solution example case study evidence source
""".split()

ZH_CHARS = (
    "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動"
    "同工也能下過子說產種面而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自"
    "二理起小物現實加量都兩體制機當使點從業本去把性好應開它合還因由其些然前外天政四日"
    "那社義事平形相全表間樣與關各重新線內數正心反你明看原又麼利比或但質氣第向道命此變"
    "條只沒結解問意建月公無系軍很情者最立代想已通並提直題黨程展五果料象員革位入常文總"
    "次品式活設及管特件長求老頭基資邊流路級少圖山統接知較將組見計別她手角期根論運農指"
    "幾九區強放決西被幹做必戰先回則任取據處理世界未來智慧科技數據分析模型網路學習影像"
)

EMOJI = list(
    "🚀📊📈📉💡🔍🧠🌍🌱⚡🔋💧🔥⭐🎯🏆📚🎓🧪🔬🛰️💻📱🖥️⚙️🔧🛠️🧩🔒🔑"
    "📦🚚✈️🚗🏠🏢🏥🌳🌊☀️🌙☁️❄️🎨🎵🎬📷🗺️🧭⏱️📅✅❌⚠️💬📣🤝👥👤❤️"
)
EMOJI = [e for e in EMOJI if e not in ("️",)]


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def random_theme(rng, zh_ratio=0.3):
    dark = rng.random() < 0.3
    hue = rng.random()
    accent = hsv(hue, rng.uniform(0.45, 0.9), rng.uniform(0.6, 0.95))
    accent2 = hsv(hue + rng.uniform(0.08, 0.5), rng.uniform(0.4, 0.85), rng.uniform(0.6, 0.95))
    if dark:
        bg = hsv(hue, rng.uniform(0.2, 0.6), rng.uniform(0.07, 0.2))
        text = hsv(hue, rng.uniform(0, 0.1), rng.uniform(0.88, 1.0))
        muted = hsv(hue, rng.uniform(0, 0.15), rng.uniform(0.65, 0.8))
        card = hsv(hue, rng.uniform(0.2, 0.5), rng.uniform(0.22, 0.34))
        accent = hsv(hue, rng.uniform(0.3, 0.7), rng.uniform(0.8, 1.0))
    else:
        bg = hsv(hue, rng.uniform(0, 0.08), rng.uniform(0.95, 1.0))
        text = hsv(hue, rng.uniform(0.1, 0.5), rng.uniform(0.08, 0.25))
        muted = hsv(hue, rng.uniform(0.05, 0.2), rng.uniform(0.35, 0.5))
        card = (255, 255, 255) if rng.random() < 0.5 else hsv(hue, rng.uniform(0.05, 0.18), rng.uniform(0.93, 1.0))
    lat = fonts.latin_fonts()
    cjk = fonts.cjk_fonts()
    lang = "zh" if (cjk and rng.random() < zh_ratio) else "en"
    title_font = rng.choice(lat)
    body_font = title_font if rng.random() < 0.5 else rng.choice(lat)
    return dict(
        dark=dark, hue=hue, bg=bg, text=text, muted=muted, card=card,
        accent=accent, accent2=accent2, lang=lang,
        title_font=title_font, body_font=body_font,
        cjk_font=(rng.choice(cjk) if cjk else None),
        title_bold=bool(rng.random() < 0.75),
        bg_style=rng.choice(["solid", "solid", "gradient", "texture"]),
        rounded=bool(rng.random() < 0.7),
    )


def sentence(rng, n_words, lang):
    if lang == "zh":
        return "".join(rng.choice(list(ZH_CHARS), size=max(2, int(n_words * 1.6))))
    words = list(rng.choice(EN_WORDS, size=max(1, n_words)))
    words[0] = words[0].capitalize()
    return " ".join(words)


def title_case(s):
    return " ".join(w.capitalize() if len(w) > 3 else w for w in s.split(" "))


def number_text(rng):
    k = rng.integers(0, 4)
    v = rng.integers(1, 1000)
    return [f"{v}%", f"{v}K", f"${v}M", f"{rng.uniform(1, 99):.1f}x"][k]


# ---------------------------------------------------------------- pictures

def _gradient(w, h, c1, c2, angle):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (np.cos(angle) * xx / max(w, 1) + np.sin(angle) * yy / max(h, 1))
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    c1, c2 = np.array(c1, np.float32), np.array(c2, np.float32)
    return (c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None])


def background_image(rng, theme, w, h):
    c1 = np.array(theme["bg"])
    shift = 25 if theme["dark"] else -18
    c2 = np.clip(c1 + shift + rng.integers(-15, 15, 3), 0, 255)
    arr = _gradient(w, h, c1, c2, rng.uniform(0, np.pi))
    if theme["bg_style"] == "texture":
        arr += rng.normal(0, 3, (h, w, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:          # soft decorative blobs, as in many AI decks
        d = ImageDraw.Draw(img, "RGBA")
        for _ in range(rng.integers(1, 4)):
            r = rng.integers(h // 4, h)
            cx, cy = rng.integers(-r // 2, w + r // 2), rng.integers(-r // 2, h + r // 2)
            col = theme["accent"] + (int(rng.integers(15, 45)),)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(20, 60)))
    return img


def photo_like(rng, w, h):
    """Abstract 'photo': gradient + blurred blobs + noise."""
    base = [tuple(int(x) for x in rng.integers(0, 256, 3)) for _ in range(2)]
    arr = _gradient(w, h, base[0], base[1], rng.uniform(0, np.pi))
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.integers(3, 12)):
        col = tuple(int(x) for x in rng.integers(0, 256, 3)) + (int(rng.integers(80, 220)),)
        x0, y0 = rng.integers(-w // 4, w), rng.integers(-h // 4, h)
        x1, y1 = x0 + rng.integers(w // 10, w), y0 + rng.integers(h // 10, h)
        if rng.random() < 0.5:
            d.ellipse([x0, y0, x1, y1], fill=col)
        else:
            pts = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(rng.integers(3, 6))]
            d.polygon(pts, fill=col)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1, 12)))
    noise = rng.normal(0, rng.uniform(2, 10), (h, w, 1))
    arr = np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def chart(rng, w, h, theme):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig, ax = plt.subplots(figsize=(max(w, 50) / dpi, max(h, 50) / dpi), dpi=dpi)
    fg = np.array(theme["text"]) / 255
    cols = [np.array(theme["accent"]) / 255, np.array(theme["accent2"]) / 255, fg * 0.6 + 0.2]
    kind = rng.choice(["bar", "line", "pie", "scatter"])
    n = int(rng.integers(3, 8))
    if kind == "bar":
        ax.bar(range(n), rng.uniform(1, 10, n), color=[cols[i % 2] for i in range(n)])
    elif kind == "line":
        for i in range(int(rng.integers(1, 3))):
            ax.plot(np.cumsum(rng.normal(0.5, 1, n * 3)), color=cols[i], lw=2.5)
    elif kind == "pie":
        ax.pie(rng.uniform(1, 5, min(n, 5)), colors=[cols[i % 3] for i in range(min(n, 5))])
    else:
        ax.scatter(rng.normal(size=40), rng.normal(size=40), color=cols[0], s=18)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg, labelsize=8)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


def picture(rng, w, h, theme, photo_dir=None):
    w, h = max(int(w), 16), max(int(h), 16)
    if photo_dir and rng.random() < 0.8:
        files = _photo_list(photo_dir)
        if files:
            im = Image.open(files[int(rng.integers(len(files)))]).convert("RGB")
            return _cover(im, w, h)
    if rng.random() < 0.3:
        return chart(rng, w, h, theme)
    return photo_like(rng, w, h)


_PHOTO_CACHE = {}


def _photo_list(d):
    if d not in _PHOTO_CACHE:
        ext = (".jpg", ".jpeg", ".png", ".webp")
        _PHOTO_CACHE[d] = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(ext)]
    return _PHOTO_CACHE[d]


def _cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, int(im.width * s + 1)), max(h, int(im.height * s + 1))))
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def icon(rng, theme):
    """An emoji glyph (or a simple flat glyph if no emoji font) on transparency."""
    path = fonts.emoji_file()
    if path and rng.random() < 0.8:
        from PIL import ImageFont
        f = ImageFont.truetype(path, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), str(rng.choice(EMOJI)), font=f, embedded_color=True)
        bb = im.getbbox()
        if bb:
            return im.crop(bb)
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = theme["accent"] + (255,)
    k = rng.integers(0, 3)
    if k == 0:
        d.ellipse([8, 8, 120, 120], outline=col, width=12)
        d.line([40, 64, 58, 84, 92, 44], fill=col, width=12)
    elif k == 1:
        for i, hh in enumerate([40, 70, 100]):
            d.rectangle([14 + i * 38, 120 - hh, 40 + i * 38, 120], fill=col)
    else:
        d.polygon([(64, 6), (80, 48), (124, 50), (90, 78), (102, 122), (64, 96), (26, 122), (38, 78), (4, 50), (48, 48)], fill=col)
    return im

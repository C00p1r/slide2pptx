"""Find installed fonts and measure text so generated text fits its box."""
import subprocess
from functools import lru_cache

from PIL import ImageFont

# Families we try to use. Only the ones actually installed are picked, so
# LibreOffice never silently substitutes a different font (which would make
# the font label wrong).
LATIN_CANDIDATES = [
    "Liberation Sans", "Liberation Serif", "DejaVu Sans", "DejaVu Serif",
    "Carlito", "Caladea", "Inter", "FreeSans", "FreeSerif", "Noto Sans",
    "Noto Serif", "Roboto", "Open Sans", "Lato", "Montserrat", "Poppins",
    "Source Sans 3", "Merriweather",
]
CJK_CANDIDATES = [
    "Noto Sans CJK TC", "Noto Serif CJK TC", "Noto Sans CJK SC",
    "WenQuanYi Zen Hei", "Source Han Sans TC",
]
EMOJI_FAMILY = "Noto Color Emoji"


def _fc(args):
    return subprocess.check_output(["fc-match", "-f"] + args).decode()


@lru_cache(None)
def installed_families():
    out = subprocess.check_output(["fc-list", ":", "family"]).decode()
    fams = set()
    for line in out.splitlines():
        for name in line.split(","):
            fams.add(name.strip().replace("\\", ""))
    return fams


@lru_cache(None)
def latin_fonts():
    fams = installed_families()
    found = [f for f in LATIN_CANDIDATES if f in fams]
    if not found:
        raise RuntimeError("No usable Latin fonts installed (see README).")
    return found


@lru_cache(None)
def cjk_fonts():
    fams = installed_families()
    return [f for f in CJK_CANDIDATES if f in fams]


@lru_cache(None)
def font_file(family, bold=False):
    pattern = family + (":bold" if bold else "")
    return _fc(["%{file}", pattern])


@lru_cache(None)
def emoji_file():
    if EMOJI_FAMILY not in installed_families():
        return None
    return font_file(EMOJI_FAMILY)


@lru_cache(512)
def pil_font(family, bold, size_px):
    return ImageFont.truetype(font_file(family, bold), max(1, int(round(size_px))))


def line_height(family, bold, size_px):
    asc, desc = pil_font(family, bold, size_px).getmetrics()
    return asc + desc


def text_width(text, family, bold, size_px):
    return pil_font(family, bold, size_px).getlength(text)


def wrap(text, family, bold, size_px, max_w, lang="en"):
    """Greedy word wrap (character wrap for Chinese). Returns list of lines."""
    tokens = list(text) if lang == "zh" else text.split(" ")
    joiner = "" if lang == "zh" else " "
    lines, cur = [], ""
    for tok in tokens:
        cand = tok if not cur else cur + joiner + tok
        if text_width(cand, family, bold, size_px) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return lines

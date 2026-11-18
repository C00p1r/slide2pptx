"""Turn slide specs into a .pptx deck.

mode="full"   : the real slide.
mode="clean"  : same slide without any text (ground truth for text removal).
mode="idmap"  : every element drawn as a flat, unique colour on black, no
                anti-aliasing at render time -> exact per-pixel instance map.
"""
import copy

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

from . import CANVAS_H, CANVAS_W, EMU_PER_PX
from .layout import INSET

SHAPES = {
    "rect": MSO_SHAPE.RECTANGLE, "rounded": MSO_SHAPE.ROUNDED_RECTANGLE,
    "oval": MSO_SHAPE.OVAL, "chevron": MSO_SHAPE.CHEVRON, "pentagon": MSO_SHAPE.PENTAGON,
}
ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}


def E(px):
    return Emu(int(round(px * EMU_PER_PX)))


def id_color(i):
    """Unique colour for element i (0-based). Black is reserved for background."""
    i += 1
    return ((i * 37) % 200 + 40, ((i // 6) * 53) % 200 + 40, ((i // 36) * 71) % 200 + 40 + (i % 6) * 2)


def new_presentation():
    prs = Presentation()
    prs.slide_width, prs.slide_height = E(CANVAS_W), E(CANVAS_H)
    return prs


def _strip_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _fill(shape, rgb, alpha=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*[int(c) for c in rgb])
    if alpha is not None:
        clr = shape._element.spPr.find(qn("a:solidFill"))[0]
        etree.SubElement(clr, qn("a:alpha")).set("val", str(int(alpha * 100000)))


def add_shape(slide, el, color=None, alpha="keep"):
    x, y, w, h = el["box"]
    sp = slide.shapes.add_shape(SHAPES[el["shape"]], E(x), E(y), E(w), E(h))
    _strip_style(sp)
    if el.get("radius") is not None:
        sp.adjustments[0] = el["radius"]
    fill = color or el["fill"]
    a = el.get("alpha") if alpha == "keep" else None
    if fill is None:
        sp.fill.background()
    else:
        _fill(sp, fill, a)
    if el.get("line") and color is None:
        sp.line.color.rgb = RGBColor(*el["line"])
        sp.line.width = Pt(el.get("line_w", 1))
    else:
        sp.line.fill.background()
    return sp


def add_text(slide, el, color=None):
    x, y, w, h = el["box"]
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = E(INSET)
    size_px = el["size_pt"] * 96 / 72
    for i, para in enumerate(el["paras"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ALIGN[el["align"]]
        if len(el["paras"]) > 1:
            p.space_after = Pt(el["size_pt"] * 0.35)
        run = p.add_run()
        run.text = para
        f = run.font
        f.size = Pt(el["size_pt"])
        f.bold = el["bold"]
        f.italic = el["italic"]
        f.color.rgb = RGBColor(*[int(c) for c in (color or el["color"])])
        f.name = el["font"]
        if el.get("ea_font"):
            rpr = run._r.get_or_add_rPr()
            latin = rpr.find(qn("a:latin"))
            ea = etree.Element(qn("a:ea"))
            ea.set("typeface", el["ea_font"])
            latin.addnext(ea)
        if el["bullet"]:
            ppr = p._p.get_or_add_pPr()
            ind = size_px * 1.2
            ppr.set("marL", str(int(E(ind))))
            ppr.set("indent", str(-int(E(ind))))
            bf = etree.SubElement(ppr, qn("a:buFont"))
            bf.set("typeface", "Arial")
            etree.SubElement(ppr, qn("a:buChar")).set("char", "•")
    return tb


def add_picture(slide, el):
    x, y, w, h = el["box"]
    return slide.shapes.add_picture(el["src"], E(x), E(y), E(w), E(h))


def add_slide(prs, spec, mode="full"):
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    bgfill = slide.background.fill
    bgfill.solid()
    if mode == "idmap":
        bgfill.fore_color.rgb = RGBColor(0, 0, 0)
    else:
        bgfill.fore_color.rgb = RGBColor(*spec["background"]["color"])
        if spec["background"].get("src"):
            slide.shapes.add_picture(spec["background"]["src"], 0, 0, prs.slide_width, prs.slide_height)
    for i, el in enumerate(spec["elements"]):
        k = el["kind"]
        if mode == "idmap":
            c = id_color(i)
            if k == "text":
                add_text(slide, el, color=c)
            else:   # pictures/icons become solid boxes of their id colour
                box_el = el if k == "shape" else dict(el, shape="rect", fill=c)
                add_shape(slide, box_el, color=c, alpha=None)
        else:
            if k == "text":
                if mode == "full":
                    add_text(slide, el)
            elif k == "shape":
                add_shape(slide, el)
            else:
                add_picture(slide, el)
    return slide


def build_deck(specs, path, mode="full"):
    prs = new_presentation()
    for s in specs:
        add_slide(prs, s, mode)
    prs.save(path)

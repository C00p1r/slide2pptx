"""Turn the rendered id-map into exact labels (tight boxes, text lines, masks)."""
import numpy as np

from . import CANVAS_H, CANVAS_W, CLASSES, PX_PER_PT
from .build import id_color


def decode_idmap(idmap_rgb, n_elements, tol=4):
    """RGB id-map -> uint8 instance map (0 = background, i+1 = element i)."""
    arr = idmap_rgb.astype(np.int16)
    inst = np.zeros(arr.shape[:2], np.uint8)
    for i in range(n_elements):
        c = np.array(id_color(i), np.int16)
        m = np.all(np.abs(arr - c) <= tol, axis=-1)
        inst[m] = i + 1
    return inst


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _lines(mask):
    rows = mask.any(axis=1)
    out, start = [], None
    for y, r in enumerate(list(rows) + [False]):
        if r and start is None:
            start = y
        elif not r and start is not None:
            xs = np.nonzero(mask[start:y].any(axis=0))[0]
            out.append([int(xs.min()), start, int(xs.max() - xs.min() + 1), y - start])
            start = None
    return out


def make_labels(spec, inst, min_text_px=8):
    """Returns (records, yolo_lines). Records hold everything needed to rebuild the slide."""
    records, yolo = [], []
    for i, el in enumerate(spec["elements"]):
        mask = inst == (i + 1)
        rec = {k: v for k, v in el.items() if k != "src"}
        rec["id"] = i + 1
        rec["visible_px"] = int(mask.sum())
        if el["kind"] == "text":
            if rec["visible_px"] < min_text_px:
                continue                    # text fell off the slide or was hidden
            rec["bbox"] = _bbox(mask)
            rec["lines"] = _lines(mask)
            rec["text"] = "\n".join(el["paras"])
            rec["size_px"] = round(el["size_pt"] * PX_PER_PT, 2)
        else:
            x, y, w, h = el["box"]
            rec["bbox"] = [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]
        x, y, w, h = rec["bbox"]
        if w < 2 or h < 2:
            continue
        rec["class_id"] = CLASSES.index(el["cls"])
        records.append(rec)
        yolo.append(f"{rec['class_id']} {(x + w / 2) / CANVAS_W:.6f} {(y + h / 2) / CANVAS_H:.6f} "
                    f"{w / CANVAS_W:.6f} {h / CANVAS_H:.6f}")
    return records, yolo


def class_map(inst, records):
    """Instance map -> semantic map (0 = background, class_id + 1). For a U-Net later."""
    lut = np.zeros(256, np.uint8)
    for r in records:
        lut[r["id"]] = r["class_id"] + 1
    return lut[inst]

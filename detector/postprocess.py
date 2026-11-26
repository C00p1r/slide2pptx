"""Clean up overlapping detections after YOLO's own NMS.

YOLO's NMS works per class with IoU 0.7, so it keeps
  - two boxes of the same class that overlap less than 0.7 (e.g. a text block and one
    of its lines), and
  - boxes of *different* classes on the same object (a sticky note as shape and icon).
Overlap across groups is normal on slides (text sits on cards), so this only merges
within a group:
  text group     title, text
  graphic group  shape, picture, icon
Rules, applied from the most confident box down:
  1. same class, IoU >= same_iou                     -> drop the less confident box
  2. same group, different class, IoU >= cross_iou   -> drop the less confident box
  3. text group: a box lying >= contain inside a more confident text-group box
     (a line inside its block)                       -> drop it
Graphic boxes inside other graphic boxes are kept (an icon on a card, a card on a panel).
"""
GROUP = {"title": "text", "text": "text", "shape": "graphic", "picture": "graphic", "icon": "graphic"}


def _inter(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return max(w, 0) * max(h, 0)


def iou(a, b):
    i = _inter(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def inside(a, b):
    """Share of box a's area that lies inside box b."""
    area = a[2] * a[3]
    return _inter(a, b) / area if area > 0 else 0.0


def clean(dets, same_iou=0.5, cross_iou=0.6, contain=0.9):
    """dets: list of dict(cls, conf, bbox=[x, y, w, h]) -> filtered list, most confident first."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        drop = False
        for k in kept:
            if GROUP[k["cls"]] != GROUP[d["cls"]]:
                continue
            o = iou(d["bbox"], k["bbox"])
            if (d["cls"] == k["cls"] and o >= same_iou) or (d["cls"] != k["cls"] and o >= cross_iou) or \
                    (GROUP[d["cls"]] == "text" and inside(d["bbox"], k["bbox"]) >= contain):
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept

"""Turning maps into boxes, matching boxes, and the classical-CV baseline."""
import cv2
import numpy as np


def masks_to_boxes(inside, edge, thr=0.5, edge_thr=0.5, min_side=8, grow=2):
    """Level 1 post-processing.

    Pixels that are inside a box and not on an edge split into one connected
    component per visible box (nested and touching boxes are separated by their
    shared edge). Each component's bounding box, grown back by the edge width, is a box.
    Returns an (N, 5) array: x0, y0, x1, y1, score.
    """
    core = ((inside > thr) & (edge < edge_thr)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    H, W = inside.shape
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if w + 2 * grow < min_side or h + 2 * grow < min_side or area < min_side * min_side / 4:
            continue
        score = float(inside[lab == k].mean())
        out.append([max(0, x - grow), max(0, y - grow), min(W, x + w + grow), min(H, y + h + grow), score])
    return nms(np.array(out, np.float32).reshape(-1, 5), 0.9)


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32)[:, :4], np.asarray(b, np.float32)[:, :4]
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


def nms(boxes, thr):
    if len(boxes) == 0:
        return boxes
    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]
    iou = iou_matrix(boxes, boxes)
    keep = []
    for i in range(len(boxes)):
        if all(iou[i, j] < thr for j in keep):
            keep.append(i)
    return boxes[keep]


def match(pred, gt, thr=0.5):
    """Greedy one-to-one matching by IoU. Returns list of (pred_idx, gt_idx)."""
    iou = iou_matrix(pred, gt)
    pairs = []
    if iou.size == 0:
        return pairs
    for i, j in sorted(zip(*np.nonzero(iou >= thr)), key=lambda p: -iou[p]):
        if all(i != a and j != b for a, b in pairs):
            pairs.append((int(i), int(j)))
    return pairs


def contour_baseline(img, min_side=8, canny=(50, 150)):
    """Classical baseline: Canny edges -> closing -> contour tree -> boxes of closed contours.

    Every contour that encloses a reasonable area relative to its bounding box counts as a
    closed region. No learning; style changes (glow, hand-drawn gaps, low contrast fills)
    are exactly where it should fail.
    """
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), *canny)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(e, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g.shape
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_side or h < min_side or w * h > 0.9 * W * H:
            continue
        fill = cv2.contourArea(c) / (w * h)
        if fill > 0.5:                                   # closed and box-like enough
            out.append([x, y, x + w, y + h, fill])
    return nms(np.array(out, np.float32).reshape(-1, 5), 0.8)

"""The whole v3 pipeline on one image: boxes -> objects -> classes -> groups.

    U-Net        inside / edge / text maps  ->  region boxes + text blocks
    pair CNN     each nearby pair: unrelated / same object / same group
                 -> merge "same object" boxes into objects (icon parts -> one icon)
    object CNN   each object, seen with its surroundings -> title / text / shape / picture / icon
                 -> groups = objects joined by "same object" or "same group"

Returns objects as dicts: box (x0, y0, x1, y1 at the image size given), cls, score, group, parts.
"""
import numpy as np
import torch

from .boxes import masks_to_boxes, text_to_boxes
from .data import candidate_pairs, obj_input, pair_input
from .evaluate import predict_maps
from .model import ClsNet, OBJ_CLASSES


def load_clsnet(path, dev):
    m = ClsNet()
    m.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return m.to(dev).eval()


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)

    def labels(self):
        return [self.find(i) for i in range(len(self.p))]


@torch.no_grad()
def run(img, unet, pairnet=None, clsnet=None, dev="cpu", min_side=8):
    H, W = img.shape[:2]
    inside, edge, text = predict_maps(unet, img, dev, with_text=True)
    regions = masks_to_boxes(inside, edge, min_side=min_side)
    texts = text_to_boxes(text)
    boxes = np.concatenate([regions, texts], 0) if len(texts) else regions
    is_text = np.r_[np.zeros(len(regions), bool), np.ones(len(texts), bool)]
    n = len(boxes)
    objs, grps = UnionFind(n), UnionFind(n)
    if pairnet is not None and n > 1:
        pairs = candidate_pairs(boxes[:, :4].tolist(), W, H)
        if pairs:
            xs, gs = zip(*[pair_input(img, boxes[i], boxes[j]) for i, j in pairs])
            out = pairnet(torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(gs)).to(dev))
            p = torch.softmax(out, 1).cpu().numpy() if out.dim() == 2 else \
                np.stack([1 - torch.sigmoid(out).cpu().numpy(), np.zeros(len(pairs)),
                          torch.sigmoid(out).cpu().numpy()], 1)
            for (i, j), q in zip(pairs, p):
                if q[1] > 0.5 and not (is_text[i] or is_text[j]):      # text is never part of an icon
                    objs.union(i, j)
                if q[1] + q[2] > 0.5:
                    grps.union(i, j)
    ol, gl = objs.labels(), grps.labels()
    result = []
    for o in sorted(set(ol)):
        idx = [i for i in range(n) if ol[i] == o]
        b = boxes[idx]
        box = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]
        result.append(dict(box=box, parts=idx, group=gl[idx[0]], region_score=float(b[:, 4].mean()),
                           text=bool(is_text[idx].all())))
    if clsnet is not None and result:
        xs, gs = zip(*[obj_input(img, r["box"]) for r in result])
        prob = torch.softmax(clsnet(torch.from_numpy(np.stack(xs)).to(dev),
                                    torch.from_numpy(np.stack(gs)).to(dev)), 1).cpu().numpy()
        for r, q in zip(result, prob):
            r["probs"] = q
            r["cls"] = OBJ_CLASSES[int(q.argmax())]
            r["score"] = float(q.max() * r["region_score"])
    else:
        for r in result:
            r["cls"] = "text" if r["text"] else "shape"
            r["score"] = r["region_score"]
    return result

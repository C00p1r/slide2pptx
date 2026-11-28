"""Level 1 network: a small U-Net that predicts two maps per pixel.

    inside  probability the pixel belongs to some closed box (outline included)
    edge    probability the pixel is on a box boundary (box/box or box/background)

Two maps are needed because "inside" alone merges touching or nested boxes into one
blob; the edge map cuts them apart again (as in DCAN / boundary-aware U-Nets).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, depth=5, out=2):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]          # 32 64 128 256 512
        self.down = nn.ModuleList([block(3 if i == 0 else ch[i - 1], ch[i]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([block(ch[i - 1] * 2, ch[i - 1]) for i in range(depth - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], out, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:])
            x = dec(torch.cat([x, s], 1))
        return self.head(x)                                  # logits: [inside, edge]


def seg_loss(logits, inside, edge, edge_w=None, text=None, edge_weight=5.0):
    """BCE + soft Dice on each map. Edges are rare, so they get a larger positive weight;
    edge_w (per pixel) further boosts box/box boundaries, which separate nested boxes.
    A third output channel, when present, is the text map."""
    loss = 0
    maps = [(inside, 1.0, None), (edge, edge_weight, edge_w)]
    if text is not None and logits.shape[1] > 2:
        maps.append((text, 2.0, None))
    for k, (y, pw, w) in enumerate(maps):
        z = logits[:, k]
        loss = loss + F.binary_cross_entropy_with_logits(z, y, weight=w, pos_weight=torch.tensor(pw, device=z.device))
        p = torch.sigmoid(z)
        inter = (p * y).sum((1, 2))
        loss = loss + (1 - (2 * inter + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()
    return loss


# ------------------------------------------------------------------ Level 2

class PairNet(nn.Module):
    """Same-group classifier for a pair of boxes.

    Input: a crop around the union of the two boxes, resized to S x S, with 5 channels:
    RGB + a mask of box A + a mask of box B. A small CNN reads the visual evidence
    (shared background panel, touching borders, same style, connectors), and a few
    geometric numbers (relative offset, size ratio, overlap, containment) are added
    before the final layers.
    """

    def __init__(self, n_geo=8, base=32, n_out=1):
        super().__init__()
        self.n_out = n_out
        c = [base, base * 2, base * 4, base * 4]
        layers, cin = [], 5
        for co in c:
            layers += [nn.Conv2d(cin, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = co
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(c[-1] + n_geo, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, n_out))

    def forward(self, x, geo):
        out = self.mlp(torch.cat([self.cnn(x), geo], 1))
        return out.squeeze(1) if self.n_out == 1 else out


PAIR_CLASSES = ["unrelated", "same_object", "same_group"]
OBJ_CLASSES = ["title", "text", "shape", "picture", "icon"]   # same ids as the YOLO labels


class ClsNet(nn.Module):
    """Object classifier with context.

    Input: a crop 2.5x the size of the object (so it sees the neighbours: is this circle part
    of an icon, is this block of text the slide title?), resized to S x S, with 4 channels:
    RGB + the object's mask. Plus 6 geometry numbers (size and position on the slide).
    """

    def __init__(self, n_geo=6, base=32, n_out=len(OBJ_CLASSES)):
        super().__init__()
        c = [base, base * 2, base * 4, base * 4]
        layers, cin = [], 4
        for co in c:
            layers += [nn.Conv2d(cin, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                       nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = co
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(c[-1] + n_geo, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, n_out))

    def forward(self, x, geo):
        return self.mlp(torch.cat([self.cnn(x), geo], 1))

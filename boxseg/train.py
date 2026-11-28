"""Train Level 1 (U-Net segmentation) or Level 2 (pair grouping CNN).

    python -m boxseg.train seg   --steps 6000 --batch 16 --out runs/boxseg
    python -m boxseg.train group --steps 3000 --batch 64 --out runs/boxseg

Checkpoints (with optimiser state) are written every --save-every steps; running the
same command again continues where it stopped, so a Colab disconnect costs minutes, not
the run. Data is generated on the fly by boxseg.synth, so there is no dataset to prepare.
`--exclude-styles hand sticky` trains without those styles, for the held-out-style
experiment (does the network generalise to a style it never saw?).
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import PairStream, SegStream
from .data import ObjStream
from .model import ClsNet, PairNet, UNet, seg_loss, OBJ_CLASSES, PAIR_CLASSES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=["seg", "group", "cls"])
    ap.add_argument("--out", default="runs/boxseg")
    ap.add_argument("--name", default=None, help="checkpoint name (default: level name)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--size", default="512x288", help="WxH of generated slides")
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--exclude-styles", nargs="*", default=[])
    ap.add_argument("--bg-dir", default=None, help="optional folder of real slide images used as blurred backgrounds")
    ap.add_argument("--inner-weight", type=float, default=5.0,
                    help="extra loss weight on box/box boundaries (0 = plain edge loss, as in the first run)")
    ap.add_argument("--resume", default="auto",
                    help="'auto' (default): continue from <out>/<name>.pt if it exists, skip if it is finished; "
                         "'no': start over; or a path to a checkpoint to start from (weights only)")
    ap.add_argument("--v2", action="store_true",
                    help="v2 data and models: no tables/pictures/icons, no text channel, 2-way grouping")
    ap.add_argument("--init-from", default=None,
                    help="start a new run from another checkpoint's weights; layers whose shape changed "
                         "(e.g. the U-Net head growing a text channel) keep the overlapping part")
    ap.add_argument("--save-every", type=int, default=250, help="checkpoint every N steps (Colab disconnects)")
    a = ap.parse_args()

    W, H = map(int, a.size.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v3 = not a.v2
    if a.level == "seg":
        model = UNet(base=a.base, out=3 if v3 else 2)
        ds = SegStream(W, H, a.exclude_styles, a.bg_dir, inner_weight=a.inner_weight, extras=v3)
    elif a.level == "group":
        model = PairNet(n_out=len(PAIR_CLASSES) if v3 else 1)
        ds = PairStream(W, H, a.exclude_styles, three_way=v3)
    else:
        model = ClsNet()
        ds = ObjStream(W, H, a.exclude_styles)
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"{a.name or a.level}.pt")
    state = None
    if a.resume == "auto" and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        if state.get("step", 0) >= a.steps:
            print(f"{ckpt} already finished ({state['step']} steps), nothing to do")
            return
        model.load_state_dict(state["model"])
        print(f"resuming {ckpt} from step {state['step']}")
    elif a.resume not in ("auto", "no"):
        model.load_state_dict(torch.load(a.resume, map_location="cpu")["model"])
    if state is None and a.init_from:
        load_partial(model, torch.load(a.init_from, map_location="cpu")["model"])
        print(f"initialised from {a.init_from}")
    model.to(dev)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    start = 0
    if state is not None and "opt" in state:
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]

    def save(step):
        tmp = ckpt + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        level=a.level, base=a.base, size=[W, H], exclude=a.exclude_styles, v3=v3,
                        step=step, steps=a.steps), tmp)
        os.replace(tmp, ckpt)            # never leave a half-written checkpoint behind

    cls_w = None    # class balance for the object CNN comes from ObjStream's per-class sampling
    model.train()
    t0, run = time.time(), 0.0
    for step, batch in enumerate(dl, start + 1):
        batch = [t.to(dev, non_blocking=True) for t in batch]
        with torch.autocast(dev, enabled=dev == "cuda"):
            if a.level == "seg":
                img, inside, edge, edge_w, text = batch
                loss = seg_loss(model(img).float(), inside, edge, edge_w, text if v3 else None)
            elif a.level == "group" and not v3:
                x, geo, y = batch
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, geo).float(), y)
            else:
                x, geo, y = batch
                loss = torch.nn.functional.cross_entropy(model(x, geo).float(), y, weight=cls_w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > start + 1 else loss.item()
        if step % 50 == 0 or step == a.steps:
            print(f"step {step}/{a.steps}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save(step)
        if step >= a.steps:
            break
    print("saved", ckpt)


def load_partial(model, sd):
    """Copy weights by name; for a tensor whose shape changed, copy the overlapping slice."""
    own = model.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif own[k].dim() == v.dim():
            sl = tuple(slice(0, min(a, b)) for a, b in zip(own[k].shape, v.shape))
            own[k][sl] = v[sl]
    model.load_state_dict(own)


if __name__ == "__main__":
    main()

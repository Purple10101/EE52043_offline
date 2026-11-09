"""Trains the learned band reader (post.bands.learned), on the CPU.

1. `crops`: cuts every labelled body of a synthetic set, whose bands are known exactly, into
   data/bands_synth/<set>.npz. Synthetic frames are already rectified, so crops come from those.
2. `train`: trains on those crops, then adjusts on real crops of measured parts. A real part's bands
   are known from its value, but not which way round it lies, so the loss takes the better way.

    python -m robot_offline.training.band_reader crops --set synth_bands
    python -m robot_offline.training.band_reader train --set synth_bands --real real.npz --out models/band_reader

real.npz holds images (N × H × W × 3 uint8, BGR, already SIZE), whites (N) and slots (N × SLOTS, colour
indices in reading order).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from robot_offline import config, manifest
from robot_offline.post.bands import crops as band_crops
from robot_offline.post.bands import learned

SYNTH_WHITE = 200.0         # the plate's level in synthetic frames
CROP_PX_PER_MM = band_crops.CROP_PX_PER_MM
RECT_PX_PER_MM = 10.0


def cut(image, polygon):
    """A body cut from a rectified frame as crops.cut does from a raw one: (crop, centre, axis)."""
    centre, axis, normal, length, width = band_crops.body_frame(polygon)
    half_u = length / 2 + band_crops.MARGIN_MM[0] * RECT_PX_PER_MM
    half_v = width / 2 + band_crops.MARGIN_MM[1] * RECT_PX_PER_MM
    step = RECT_PX_PER_MM / CROP_PX_PER_MM
    u = (np.arange(int(np.ceil(2 * half_u / step))) + 0.5) * step - half_u
    v = (np.arange(int(np.ceil(2 * half_v / step))) + 0.5) * step - half_v
    uu, vv = np.meshgrid(u, v)
    points = (centre + uu[..., None] * axis + vv[..., None] * normal).astype(np.float32)
    crop = cv2.remap(image, points[..., 0], points[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop, centre, axis


def slot_indices(names, reversed_):
    slots = learned.slots_of(names)
    return [learned.COLOURS.index(c) for c in (slots[::-1] if reversed_ else slots)]


def make_crops(name):
    samples = manifest.load(config.load("dataset")["labels"]["rules_version"])["samples"]
    images, slots = [], []
    for sample in samples:
        if sample["session"] != name:
            continue
        frame = cv2.imread(str(config.DATA_DIR / sample["image"]))
        for inst in sample["instances"]:
            # Only resistors have bands to learn; other objects (synth_obj_* sets) carry kind and no bands.
            if inst.get("kind", "resistor") != "resistor" or not inst.get("bands") or "first_end" not in inst:
                continue
            crop, centre, axis = cut(frame, np.array(inst["polygon"]))
            images.append(cv2.resize(crop, learned.SIZE, interpolation=cv2.INTER_AREA))
            slots.append(slot_indices(inst["bands"], (np.array(inst["first_end"]) - centre) @ axis > 0))
    out = config.DATA_DIR / "bands_synth"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", images=np.array(images), slots=np.array(slots))
    print(f"{len(images)} crops from {name}")


def augment(x, y, generator):
    """Light changes a real frame shows: exposure, tint, focus, noise, and which way round it lies."""
    n = len(x)
    x = x * (1 + 0.2 * (torch.rand(n, 1, 1, 1, generator=generator) - 0.5))
    x = x * (1 + 0.08 * (torch.rand(n, 3, 1, 1, generator=generator) - 0.5))
    x = x + 0.01 * torch.randn(x.shape, generator=generator)
    flip = torch.rand(n, generator=generator) < 0.5
    x[flip] = x[flip].flip(3)
    y = y.clone()
    y[flip] = y[flip].flip(1)
    return x.clamp(0, 1.5), y


def _loss(logits, y):
    return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape).sum(1)


def fit(net, x, y, epochs, lr, generator, either_way=False, batch=64):
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, total_steps=epochs * max(1, len(x) // batch))
    net.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for i in range(0, len(x) - batch + 1, batch):
            xb, yb = augment(x[order[i:i + batch]], y[order[i:i + batch]], generator)
            logits = net(xb)
            loss = _loss(logits, yb)
            if either_way:
                loss = torch.minimum(loss, _loss(logits, yb.flip(1)))
            optimiser.zero_grad()
            loss.mean().backward()
            optimiser.step()
            schedule.step()
            total += float(loss.mean())
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / max(1, len(x) // batch):.3f}", flush=True)
    net.eval()


def tensors(images, whites):
    rgb = images[..., ::-1].astype(np.float32).transpose(0, 3, 1, 2) / np.asarray(whites, np.float32).reshape(-1, 1, 1, 1)
    return torch.from_numpy(np.clip(rgb, 0, 1.5))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.training.band_reader")
    parser.add_argument("command", choices=("crops", "train"))
    parser.add_argument("--set", default="synth_bands")
    parser.add_argument("--real", type=Path, help="real crops of measured parts, for the last stage")
    parser.add_argument("--out", type=Path, default=config.ROOT / "models" / "band_reader")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--real-epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--start", type=Path, help="carry on from these weights, skipping the synthetic stage")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "crops":
        make_crops(args.set)
        return

    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    if args.start:
        net = learned.load(args.start)
    else:
        data = np.load(config.DATA_DIR / "bands_synth" / f"{args.set}.npz")
        x, y = tensors(data["images"], np.full(len(data["images"]), SYNTH_WHITE)), torch.from_numpy(data["slots"]).long()
        net = learned.BandNet(args.width)
        print(f"{len(x)} synthetic crops, {sum(p.numel() for p in net.parameters())} parameters", flush=True)
        fit(net, x, y, args.epochs, 3e-3, generator)
    if args.real:
        real = np.load(args.real)
        print(f"{len(real['images'])} real crops", flush=True)
        fit(net, tensors(real["images"], real["whites"]), torch.from_numpy(real["slots"]).long(),
            args.real_epochs, 3e-4, generator, either_way=True, batch=8)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"width": net.head.in_features // 40, "state": net.state_dict()}, args.out / "band_reader.pt")
    print(f"Wrote {args.out / 'band_reader.pt'}")


if __name__ == "__main__":
    main()

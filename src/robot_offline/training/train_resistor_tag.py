"""Trains the resistor tag (post/resistor_tag.py) on synthetic crops, and scores it on real ones.

    python -m robot_offline.training.train_resistor_tag --name resistor_tag_v1
    python -m robot_offline.training.train_resistor_tag --name resistor_tag_v1 --train synth_obj_v1 --val synth_obj_v1_val

Every labelled object in the synthetic sets (labelled "objects", each with its kind) is cut out
(resistor_tag.cutout) from its frame's colour model view (rectify.model_image). Resistor or not is
the target. Then the real test frames whose every object is outlined (test_sessions, labelled
"objects") are scored the same way, held out: none of them was trained on.

Writes runs/<name>/resistor_tag.pt and runs/<name>/scores.json: per set and per kind, the share tagged
resistor / unsure / not. The headline is the share of non-resistors tagged "resistor" on real parts
(a confident band value on a non-resistor is the failure to avoid).
"""
import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch import nn

from robot_offline import config, manifest
from robot_offline.post import resistor_tag as rt
from robot_offline.pre import rectify


def crops_of(samples, rs):
    """(crops, is_resistor, kinds, sample index) for every labelled object of the samples."""
    crops, y, kinds, where = [], [], [], []
    for k, s in enumerate(samples):
        view = rectify.model_image(s, cv2.imread(str(config.DATA_DIR / s["image"])), rs)
        for inst in s["instances"]:
            kind = inst.get("kind", "resistor")
            crops.append(rt.cutout(view, inst["polygon"]))
            y.append(kind == "resistor")
            kinds.append(kind)
            where.append(k)
    return np.stack(crops) if crops else np.zeros((0, rt.SIZE[1], rt.SIZE[0], 3), np.uint8), \
        np.array(y, np.float32), kinds, where


def augment(x, gen):
    """Light changes a real crop may show that a synthetic one doesn't: either end first, mirrored,
    light and white balance a little off, soft focus, sensor noise, the outline a little off-centre."""
    n = x.shape[0]
    flip = torch.rand(n, generator=gen, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, (2, 3)), x)
    mirror = torch.rand(n, generator=gen, device=x.device) < 0.5
    x = torch.where(mirror[:, None, None, None], torch.flip(x, (2,)), x)
    mask = (x.sum(1, keepdim=True) > 0).float()
    gain = 1 + 0.15 * (2 * torch.rand(n, 1, 1, 1, generator=gen, device=x.device) - 1)
    tint = 1 + 0.06 * (2 * torch.rand(n, 3, 1, 1, generator=gen, device=x.device) - 1)
    x = x * gain * tint
    soft = torch.rand(n, generator=gen, device=x.device) < 0.3
    blurred = nn.functional.avg_pool2d(x, 3, 1, 1)
    x = torch.where(soft[:, None, None, None], blurred, x)
    x = x + 0.02 * torch.randn(x.shape, generator=gen, device=x.device)
    dx, dy = (int(v) for v in torch.randint(-6, 7, (2,), generator=gen, device=x.device))
    x = torch.roll(x * mask, (dy, dx), (2, 3))
    return x.clamp(0, 1)


def scores(p, kinds, unsure=rt.UNSURE):
    """{kind: {"n", "resistor", "unsure", "not"}} shares, plus "all resistors" and "all others"."""
    groups = defaultdict(list)
    for v, k in zip(p, kinds):
        groups[k].append(v)
        groups["all resistors" if k == "resistor" else "all others"].append(v)
    out = {}
    for k, vs in sorted(groups.items()):
        tags = [rt.tag(v, unsure) for v in vs]
        out[k] = {"n": len(vs), **{t: round(tags.count(t) / len(vs), 4) for t in ("resistor", "unsure", "not")}}
    return out


def main(argv=None):
    ds = config.load("dataset")
    parser = argparse.ArgumentParser(prog="python -m robot_offline.training.train_resistor_tag")
    parser.add_argument("--name", default="resistor_tag_v1")
    parser.add_argument("--train", nargs="+", default=["synth_obj_v1"])
    parser.add_argument("--val", nargs="+", default=["synth_obj_v1_val"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if there is one)")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rs = rectify.Settings.from_config(ds)
    samples = manifest.load(ds["labels"]["rules_version"])["samples"]
    pick = lambda sets: [s for s in samples if s["session"] in sets and s.get("labelled") == "objects"]
    real = [s for s in samples if s["session"] in ds["splits"]["test_sessions"] and s.get("labelled") == "objects"]

    t0 = time.perf_counter()
    xtr, ytr, ktr, _ = crops_of(pick(args.train), rs)
    xva, yva, kva, _ = crops_of(pick(args.val), rs)
    xre, yre, kre, _ = crops_of(real, rs)
    print(f"crops: train {len(xtr)} ({int(ytr.sum())} resistors), val {len(xva)}, real {len(xre)} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)

    net = rt.Net(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(xtr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    pos_weight = torch.tensor([(1 - ytr.mean()) / ytr.mean()], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        net.train()
        order, total = rng.permutation(len(xtr)), 0.0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            x = augment(rt.to_tensor(list(xtr[idx])).to(device), gen)
            loss = loss_fn(net(x), torch.from_numpy(ytr[idx]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        net.eval()
        pva = rt.probability(rt.logits(net, list(xva)))
        acc = float(((pva >= 0.5) == (yva > 0.5)).mean())
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(xtr):.4f}, synthetic val accuracy {acc:.4f}", flush=True)

    out = config.ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.cpu().state_dict(), "width": args.width, "window_mm": rt.WINDOW_MM,
                "keep_mm": rt.KEEP_MM, "train": args.train}, out / "resistor_tag.pt")
    net = rt.load(out / "resistor_tag.pt")
    result = {"train": args.train, "val": args.val, "epochs": args.epochs, "unsure": rt.UNSURE,
              "synthetic_val": scores(rt.probability(rt.logits(net, list(xva))), kva),
              "real_test": scores(rt.probability(rt.logits(net, list(xre))), kre)}
    (out / "scores.json").write_text(json.dumps(result, indent=1))
    for name in ("synthetic_val", "real_test"):
        print(f"\n{name}: share tagged resistor / unsure / not")
        for k, v in result[name].items():
            print(f"  {k:>20} n {v['n']:5d}   {v['resistor']:.3f} / {v['unsure']:.3f} / {v['not']:.3f}")
    print(f"\nSaved {out / 'resistor_tag.pt'}")


if __name__ == "__main__":
    main()

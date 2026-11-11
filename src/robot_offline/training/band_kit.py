"""Trains the band reader (post.bands.learned) on the resistor kit's real crops together with the synthetic
ones, and scores it on values it never saw.

    python -m robot_offline.training.band_kit data                     # crops -> data/bands_kit/kit.npz
    python -m robot_offline.training.band_kit train --out runs/band_kit_v1 [--device cpu|cuda]

`data` reads every session in configs/band_reader.toml [kit] (labels.jsonl, written by
tools/collect_bands.py) and splits it: the test values (never trained on, nor used to choose anything),
the validation values (never trained on: they choose the epoch and set the confidence limit, so both are
judged on values the reader hasn't seen), one placement of every other value (validation on new
placements of trained values) and the rest for training. Splits are by placement, never by crop, so a body's first
capture and its close-up are always on the same side. Bodies on a kit plate are all the one value, so every
crop is labelled; a part tagged "not a resistor" at capture time is still one and is kept.

`train` fits the network to synthetic crops (data/bands_synth/synth_bands.npz, bands known with their
direction) and the kit's training crops (bands known from the value, but not which way round: the slot
loss takes the better way, and the direction is learned from that choice, except for values that read
the same both ways). Each epoch is scored on the validation values; the best epoch is kept. The confidence
limit is then the lowest at which neither validation set gives a wrong answer, and the test is scored at
it and at 0.95 and 0.99. --model picks the network: ctc (BandCTC, reading each column along the body; the
default) or slots (BandNet, the first, which memorised values). Writes band_reader.pt and report.json.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from robot_offline import config
from robot_offline.post.bands import learned
from robot_offline.training.band_reader import SYNTH_WHITE, tensors

KIT_DIR = config.DATA_DIR / "bands" / "kit"
OUT = config.DATA_DIR / "bands_kit" / "kit.npz"
DIGITS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white")
MULTIPLIER = {-2: "silver", -1: "gold", **{i: c for i, c in enumerate(DIGITS)}}


def kit_bands(ohms):
    """A 1% kit part's bands in reading order: 3 digits, multiplier, brown; a 0 ohm link one black band."""
    if ohms == 0:
        return ("black",)
    for power in range(-2, 10):
        significant = ohms / 10 ** power
        if abs(significant - round(significant)) < 1e-6 and 100 <= round(significant) < 1000:
            return tuple(DIGITS[int(c)] for c in str(int(round(significant)))) + (MULTIPLIER[power], "brown")
    raise ValueError(f"{ohms} ohm needs more than three digits")


def slot_targets(colours):
    return [learned.COLOURS.index(c) for c in learned.slots_of(colours)]


def split_of(rows, test_ohms, per_value=1, val_ohms=()):
    """{capture: "train" | "val" | "val_values" | "test"}: test values and validation values whole; of the
    others, per_value placements each for validation, rotating through the placement order with the
    value's place in the list."""
    by_value = {}
    for r in rows:
        by_value.setdefault(r["value_ohm"], set()).add(r["capture"])
    split = {}
    for i, (ohms, caps) in enumerate(sorted(by_value.items())):
        caps = sorted(caps)
        if any(abs(ohms - t) < 1e-6 for t in test_ohms):
            split.update({c: "test" for c in caps})
            continue
        if any(abs(ohms - t) < 1e-6 for t in val_ohms):
            split.update({c: "val_values" for c in caps})
            continue
        val = {caps[(i + k) % len(caps)] for k in range(min(per_value, len(caps) - 1))}
        split.update({c: "val" if c in val else "train" for c in caps})
    return split


def make_data(cfg=None, out=OUT, final=False):
    """final: no values held out (the model for the robot); only the validation placements, to set the limit."""
    from robot_offline.tools.collect_bands import parse_ohms
    cfg = dict(cfg or config.load("band_reader")["kit"])
    if final:
        cfg["test_values"], cfg["validation_values"] = [], []
    test_ohms = [parse_ohms(v) for v in cfg["test_values"]]
    rows = []
    for session in cfg["sessions"]:
        for line in (KIT_DIR / session / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("value_ohm") is None:          # the diodes: not resistors, no bands
                continue
            r["capture"] = f"{session}/{r['capture']}"
            r["path"] = KIT_DIR / session / r["image"]
            rows.append(r)
    val_ohms = [parse_ohms(v) for v in cfg.get("validation_values", [])]
    split = split_of(rows, test_ohms, cfg["validation_placements"], val_ohms)
    images, keep = [], []
    for r in rows:
        img = cv2.imread(str(r["path"]))
        if img is None:
            continue
        images.append(cv2.resize(img, learned.SIZE, interpolation=cv2.INTER_AREA))
        keep.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, images=np.array(images), whites=np.array([r["white"] for r in keep], np.float32),
        slots=np.array([slot_targets(kit_bands(r["value_ohm"])) for r in keep]),
        ohms=np.array([r["value_ohm"] for r in keep]), split=np.array([split[r["capture"]] for r in keep]),
        kind=np.array([r["kind"] for r in keep]), capture=np.array([r["capture"] for r in keep]),
        body=np.array([r["body"] for r in keep]), sharpness=np.array([r["sharpness"] for r in keep], np.float32))
    counts = {s: int(sum(split[r["capture"]] == s for r in keep)) for s in ("train", "val", "val_values", "test")}
    print(f"{len(keep)} kit crops -> {out}: {counts}")
    return out


def synth_direction(slots):
    """For synthetic slots (image order): 1 if they read left to right as a value in scope, 0 if right to
    left, -1 if both or neither (no direction to learn)."""
    forward = {tuple(r) for r in learned._FORWARD}
    out = []
    for row in np.asarray(slots):
        f, b = tuple(row) in forward, tuple(row[::-1]) in forward
        out.append(1 if f and not b else 0 if b and not f else -1)
    return np.array(out)


def _blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k).repeat(3, 1, 1, 1)
    x = nn.functional.conv2d(nn.functional.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g, groups=3)
    return nn.functional.conv2d(nn.functional.pad(x, (0, 0, k // 2, k // 2), mode="replicate"),
                                g.transpose(2, 3), groups=3)


def augment(x, y, d, either, generator):
    """What changes between real frames: exposure, tint, noise, sharpness (soft first captures, sharp
    close-ups), and which way round the part lies. Flipping a synthetic crop flips its slots (image order)
    and direction; a kit crop's slots stay in reading order, and the slot loss finds the way it lies."""
    n, dev = len(x), x.device
    rand = lambda *shape: torch.rand(*shape, generator=generator).to(dev)      # noqa: E731
    x = x * (1 + 0.2 * (rand(n, 1, 1, 1) - 0.5))
    x = x * (1 + 0.08 * (rand(n, 3, 1, 1) - 0.5))
    third = n // 3
    soft = _blur(x[:third], 0.4 + 1.2 * float(rand(1)))
    sharp = x[third:2 * third] + (0.3 + 0.9 * float(rand(1))) * (x[third:2 * third] - _blur(x[third:2 * third], 1.5))
    x = torch.cat([soft, sharp, x[2 * third:]])
    x = x + 0.01 * torch.randn(x.shape, generator=generator).to(dev)
    flip = rand(n) < 0.5
    x[flip] = x[flip].flip(3)
    y, d = y.clone(), d.clone()
    image_order = flip & ~either
    y[image_order] = y[image_order].flip(1)
    d[image_order & (d >= 0)] = 1 - d[image_order & (d >= 0)]
    return x.clamp(0, 1.5), y, d


def _slot_loss(logits, y):
    return nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape).sum(1)


def _ctc_nll(columns, y):
    """CTC negative log-likelihood of each row's bands over the columns: y is 5 slots, "none" (the blank)
    for padding, its bands in the order the columns should read them."""
    lp = torch.log_softmax(columns.float(), -1).transpose(0, 1)
    t, n = lp.shape[:2]
    return nn.functional.ctc_loss(lp, y[y != 0], torch.full((n,), t, dtype=torch.long, device=y.device),
                                  (y != 0).sum(1), blank=0, reduction="none", zero_infinity=True)


def _same_both_ways(y):
    """Rows whose bands read the same both ways round (100R, 1k, a 0 ohm link): no direction to learn."""
    return torch.tensor([bool((r[r != 0] == r[r != 0].flip(0)).all()) for r in y], device=y.device)


def step_loss(net, x, y, d, either, direction_weight):
    """The bands (slots for BandNet, CTC over the columns for BandCTC; for kit crops, whose y is in reading
    order, the better of the two ways: that also sets their direction, 1 if the image shows the reading
    order left to right) plus the direction where it is known."""
    out, dlogit = net.both(x)
    band_loss = _ctc_nll if getattr(net, "kind", "") == "ctc" else _slot_loss
    forward, backward = band_loss(out, y), band_loss(out, y.flip(1))
    use_back = either & (backward < forward)
    loss = torch.where(use_back, backward, forward)
    d = torch.where(either & (d >= 0), (~use_back).long(), d)          # kit: left to right unless flipped
    palindrome = _same_both_ways(y)
    known = (d >= 0) & ~palindrome
    if known.any():
        loss = loss + direction_weight * known * nn.functional.binary_cross_entropy_with_logits(
            dlogit, d.clamp(min=0).float(), reduction="none")
    return loss.mean()


@torch.no_grad()
def log_probs_batch(net, x, batch=256):
    """What learned.log_probs gives, for many crops at once (decode takes each row)."""
    out = []
    for i in range(0, len(x), batch):
        slots, d = net.both(x[i:i + batch])
        if getattr(net, "kind", "") == "ctc":
            out.append(learned.ctc_scores(slots, d))
            continue
        lp = torch.log_softmax(slots, -1).cpu().numpy()
        ways = np.zeros((len(lp), 1, lp.shape[2]))
        ways[:, 0, 0] = nn.functional.logsigmoid(d).cpu().numpy()
        ways[:, 0, 1] = nn.functional.logsigmoid(-d).cpu().numpy()
        out.append(np.concatenate([lp, ways], 1))
    return np.concatenate(out)


def score(lps, ohms, limits=(0.95,)):
    """Per crop: best-guess accuracy, and at each confidence limit the share answered and the answers wrong."""
    readings = [learned.decode(lp, 1.0) for lp in lps]
    right = np.array([abs(r.best - o) < 1e-6 for r, o in zip(readings, ohms)])
    conf = np.array([r.confidence for r in readings])
    out = dict(n=len(ohms), best_right=float(right.mean()) if len(ohms) else 0.0)
    for t in limits:
        answered = conf >= t
        out[f"at_{t:g}"] = dict(answered=float(answered.mean()), wrong=int((answered & ~right).sum()),
                                right=int((answered & right).sum()))
    return out, right, conf


@torch.no_grad()
def greedy_right(net, x, slots, batch=256):
    """BandCTC only: the share of crops whose columns, read greedily (best colour per column, repeats and
    blanks dropped), give exactly the true bands one way round: colour reading with no list of values."""
    right = []
    for i in range(0, len(x), batch):
        cols = net.both(x[i:i + batch])[0].argmax(-1).cpu().numpy()
        for c, y in zip(cols, slots[i:i + batch]):
            seq = [int(v) for j, v in enumerate(c) if v != 0 and (j == 0 or v != c[j - 1])]
            truth = [int(v) for v in y if v != 0]
            right.append(seq == truth or seq == truth[::-1])
    return float(np.mean(right)) if right else 0.0


def true_share(scores, ohms):
    """Per row of value scores (decode's 1-D input), the share the true value gets: its probability."""
    s = np.asarray(scores, np.float64)
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    cand = np.concatenate([learned._OHMS, learned._OHMS])
    return np.array([row[np.abs(cand - o) < 1e-6].sum() for row, o in zip(p, ohms)])


def fit_temperature(scores, ohms, grid=np.geomspace(0.5, 100, 80)):
    """The temperature that makes the value scores honest on these crops: the one (from grid) with the
    lowest mean negative log-probability of the true value. Above 1 when the network is overconfident."""
    nll = [-np.log(np.clip(true_share(np.asarray(scores) / t, ohms), 1e-12, 1)).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def safe_limit(right, conf, floor=0.5):
    """The lowest confidence limit with no wrong answer at or above it (at least floor)."""
    wrong = conf[~right]
    return float(max(floor, np.nextafter(wrong.max(), 1)) if len(wrong) else floor)


def by_body(lps, data, mask):
    """First capture and close-up of each body averaged (as Views does), with its value."""
    groups = {}
    for i in np.flatnonzero(mask):
        groups.setdefault((data["capture"][i], int(data["body"][i])), []).append(i)
    lp = np.array([np.mean(lps[idx], 0) for idx in groups.values()])
    ohms = np.array([data["ohms"][idx[0]] for idx in groups.values()])
    return lp, ohms


def train(out, device="cpu", data_path=OUT, cfg=None, epochs=None, limit_synth=None, model="ctc", seed=0,
          start=None, values=None):
    """start: weights to begin from (a run past the CTC plateau). values: the only values its decoder may
    answer, saved with the model (learned.use_values); None for every value in scope."""
    cfg = cfg or config.load("band_reader")["train"]
    epochs = epochs or cfg["epochs"]
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    kit = np.load(data_path)
    synth = np.load(config.DATA_DIR / "bands_synth" / "synth_bands.npz")
    s_n = len(synth["images"]) if limit_synth is None else min(limit_synth, len(synth["images"]))
    sx = tensors(synth["images"][:s_n], np.full(s_n, SYNTH_WHITE))
    sy = torch.from_numpy(synth["slots"][:s_n]).long()
    sd = torch.from_numpy(synth_direction(synth["slots"][:s_n])).long()
    kx_all, ky_all = tensors(kit["images"], kit["whites"]), torch.from_numpy(kit["slots"]).long()
    tr, va, vv, te = (kit["split"] == s for s in ("train", "val", "val_values", "test"))
    choose = vv if vv.any() else va             # what picks the epoch: unseen values when there are any
    reps = cfg["kit_repeat"]
    x = torch.cat([sx] + [kx_all[tr]] * reps)
    y = torch.cat([sy] + [ky_all[tr]] * reps)
    d = torch.cat([sd] + [torch.full((int(tr.sum()),), 1)] * reps)            # kit: set by the slot loss
    either = torch.cat([torch.zeros(s_n, dtype=torch.bool)] + [torch.ones(int(tr.sum()), dtype=torch.bool)] * reps)
    learned.use_values(values)
    net = (learned.BandCTC if model == "ctc" else learned.BandNet)(16)
    if start:
        net.load_state_dict(torch.load(start, map_location="cpu", weights_only=True)["state"])
    net = net.to(device)
    print(f"{model}: {s_n} synthetic + {int(tr.sum())} kit crops (x{reps}); validation placements {int(va.sum())}, "
          f"validation values {int(vv.sum())}, test {int(te.sum())}; "
          f"{sum(p.numel() for p in net.parameters())} parameters, {device}", flush=True)
    batch = cfg["batch"]
    optimiser = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = epochs * max(1, len(x) // batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, cfg["lr"], total_steps=steps)
    kx_va, kx_vv = kx_all[va].to(device), kx_all[choose].to(device)
    best, history = None, []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        net.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for i in range(0, len(x) - batch + 1, batch):
            idx = order[i:i + batch]
            eb = either[idx].to(device)
            xb, yb, db = augment(x[idx].to(device), y[idx].to(device), d[idx].to(device), eb, generator)
            loss = step_loss(net, xb, yb, db, eb, cfg["direction_weight"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        net.eval()
        val = score(log_probs_batch(net, kx_va), kit["ohms"][va])[0]
        chosen = score(log_probs_batch(net, kx_vv), kit["ohms"][choose])[0]
        history.append(dict(epoch=epoch + 1, loss=total / max(1, len(x) // batch), val_best_right=val["best_right"],
                            val_at_095=val["at_0.95"], val_values_best_right=chosen["best_right"],
                            val_values_at_095=chosen["at_0.95"]))
        print(f"  epoch {epoch + 1}/{epochs}  loss {history[-1]['loss']:.3f}  placements: right {val['best_right']:.1%}  "
              f"| unseen values: right {chosen['best_right']:.1%}, at 0.95 answered {chosen['at_0.95']['answered']:.1%} "
              f"wrong {chosen['at_0.95']['wrong']}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        if best is None or chosen["best_right"] > best[0]:
            best = (chosen["best_right"], epoch + 1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    net.eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(model=model, seed=seed, start=str(start) if start else None,
                  values=None if values is None else len(values), best_epoch=best[1], chosen_on="val_values" if vv.any() else "val", history=history,
                  data=str(data_path), synth=s_n)
    lps = log_probs_batch(net, kx_all.to(device))
    temperature = 1.0
    if model == "ctc":
        # Fitted on the values it hasn't seen (validation values), the only honest guide to its confidence on
        # new values; the confidence limit below is then set on the scaled scores.
        temperature = fit_temperature(lps[choose], kit["ohms"][choose])
        lps = lps / temperature
    report["temperature"] = temperature
    torch.save({"width": 16, "kind": getattr(net, "kind", "slots"), "temperature": temperature, "state": best[2],
                "values": None if values is None else [float(v) for v in values]}, out / "band_reader.pt")
    _, v_right, v_conf = score(lps[va | vv], kit["ohms"][va | vv])
    limit = safe_limit(v_right, v_conf)
    report["confidence_limit"] = limit
    for name, mask in (("val", va), ("val_values", vv), ("test", te)):
        if not mask.any():
            continue
        crops, _, _ = score(lps[mask], kit["ohms"][mask], (limit, 0.95, 0.99))
        b_lp, b_ohms = by_body(lps, kit, mask)
        bodies, _, _ = score(b_lp, b_ohms, (limit, 0.95, 0.99))
        per_kind = {k: score(lps[mask & (kit["kind"] == k)], kit["ohms"][mask & (kit["kind"] == k)], (limit,))[0]
                    for k in ("first", "closeup")}
        per_value = {}
        for o in sorted(set(kit["ohms"][mask])):
            m = mask & (kit["ohms"] == o)
            per_value[learned.ohms_text(o)] = score(lps[m], kit["ohms"][m], (limit,))[0]
        report[name] = dict(crops=crops, bodies=bodies, by_kind=per_kind, by_value=per_value)
        if model == "ctc":
            report[name]["colours_read_right"] = greedy_right(net, kx_all[mask].to(device), kit["slots"][mask])
    (out / "report.json").write_text(json.dumps(report, indent=1))
    t = report.get("test", {"by_value": {}})
    print(f"best epoch {best[1]} (chosen on {report['chosen_on']}); temperature {temperature:.2f}; "
          f"confidence limit from validation: {limit:.4f}")
    for name in ("val", "val_values", "test"):
        if name not in report:
            continue
        r = report[name]
        print(f"{name}: crops best-guess right {r['crops']['best_right']:.1%}, at the limit answered "
              f"{r['crops'][f'at_{limit:g}']['answered']:.1%} with {r['crops'][f'at_{limit:g}']['wrong']} wrong; "
              f"bodies (first + close-up) best-guess right {r['bodies']['best_right']:.1%}, answered "
              f"{r['bodies'][f'at_{limit:g}']['answered']:.1%} with {r['bodies'][f'at_{limit:g}']['wrong']} wrong"
              + (f"; colours read right {r['colours_read_right']:.1%}" if "colours_read_right" in r else ""))
    print(f"wrote {out / 'band_reader.pt'} and report.json" + ("; test per value: " + ", ".join(
        f"{v} {s['best_right']:.0%}" for v, s in t["by_value"].items()) if t["by_value"] else ""))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m robot_offline.training.band_kit", description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("data", "train"))
    p.add_argument("--final", action="store_true",
                   help="data: hold no values out (the robot's model); train: on that data (data/bands_kit/kit_final.npz), "
                        "decoding only the kit's values (configs/resistor_kit.toml)")
    p.add_argument("--start", type=Path, help="begin from these weights")
    p.add_argument("--out", type=Path, default=config.ROOT / "runs" / "band_kit_v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--limit-synth", type=int, help="use only this many synthetic crops (a quick check)")
    p.add_argument("--model", choices=("ctc", "slots"), default="ctc")
    p.add_argument("--seed", type=int, default=0, help="CTC runs differ a lot by seed: escaping the all-blank start is partly luck")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    final_path = OUT.with_name("kit_final.npz")
    if args.command == "data":
        make_data(out=final_path if args.final else OUT, final=args.final)
    else:
        values = None
        if args.final:
            from robot_offline.tools.collect_bands import part_ohms
            values = [v for v in (part_ohms(m) for m in config.load("resistor_kit")["values"]) if v is not None]
            known = {round(o, 4) for o, _ in learned.ALL_CANDIDATES}
            for v in [v for v in values if round(v, 4) not in known]:
                # The card's "23" values aren't E24: its 23R strip was 33R (checked 2026-11-10), so a 23k
                # strip is taken to be 33k, the value the robot would then meet.
                fix = round(v * 33 / 23, 4) if round(v / 10 ** np.floor(np.log10(v)), 4) == 2.3 else None
                print(f"kit value {v:g} ohm isn't a standard value; decoding {fix:g} ohm in its place" if fix
                      else f"kit value {v:g} ohm isn't a standard value; left out of decoding")
                values = [fix if (u == v and fix) else u for u in values if u != v or fix]
        train(args.out, args.device, data_path=final_path if args.final else OUT, epochs=args.epochs,
              limit_synth=args.limit_synth, model=args.model, seed=args.seed, start=args.start, values=values)


if __name__ == "__main__":
    main()

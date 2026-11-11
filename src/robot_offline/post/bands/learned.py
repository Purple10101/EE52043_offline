"""The learned band reader: a small network that reads a body's bands, on the CPU.

A straightened crop goes in and, for each of five slots along the body, left to right, a colour comes
out (or none: a 4-band part leaves one end slot empty). The network says nothing about values: every
value in scope is scored from the slots, both ways round, so only a real value can come out, and its
share of the total is the reading's confidence. Below MIN_CONFIDENCE the part is unknown, for the
robot to measure instead.

Some values read as another value backwards: on 1% parts 10k is brown-black-black-red-brown, and
120R is that reversed. The slots alone can't tell them apart, so the network also says which way
round the part lies (the direction head: is the tolerance band, set a little apart, on the right?),
and the decoder weighs the two readings by it. Weights from before the direction head still load and
decode as before (the two ways equally likely).

Two networks. BandNet (the first) gives five slot colours from a layer over the whole body; trained
on the kit's 26 values it learned each value's look, not its colours, and failed on values it hadn't
seen (experiments/2026-11-11_band_reader_kit). BandCTC reads each column along the body on its own,
with the same weights everywhere: this colour, or no band (the CTC blank). A value is scored by the
CTC likelihood of its band sequence over the columns, both ways round, so any order of familiar
colours can be read. The decoder and the confidence are the same for both: each value's share of
all values in scope. log_probs gives what each network's decoding needs; Views averages either.

Scope (SCOPE_OHM): 0 ohm links and E24 values from 1 ohm to 10 Mohm, any multiplier band. Wider than
the sorter's 10 ohm to 10 kohm (docs/classification.md) on purpose: a reader that only knew that range
would read a 15k part as 1k5. Values outside the sorter's range are for the robot to reject.

Crops must come from a frame with the plate at about grey 215, see docs/classification.md. The network
is trained by `training.band_reader`.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from robot_offline.post.bands import value

COLOURS = ("none",) + tuple(value.PAINT_LAB)
SLOTS = 5
SIZE = (160, 48)            # the crop as the network sees it: width, height
MIN_CONFIDENCE = 0.95    # at 0.9, 1 of 5 answers on parts never seen was wrong (2026-11-06)
READING_LEVEL = (205, 225)  # the plate's grey to read at: below it dark bands on blue run together
SCOPE_OHM = (1.0, 10e6)     # E24 values the reader knows, plus 0 ohm links (one black band)
COLUMNS = 20                # BandCTC's columns along the body: the features' width at SIZE


def _block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class BandNet(nn.Module):
    """About 150 k parameters at width 16."""

    def __init__(self, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(3, w), _block(w, w, 2), _block(w, 2 * w), _block(2 * w, 2 * w, 2),
                                      _block(2 * w, 4 * w), _block(4 * w, 4 * w, 2), _block(4 * w, 4 * w))
        self.pool = nn.AdaptiveAvgPool2d((1, 10))    # keeps the order along the body
        self.head = nn.Linear(4 * w * 10, SLOTS * len(COLOURS))
        self.direction = nn.Linear(4 * w * 10, 1)       # logit: the reading order runs left to right

    def forward(self, x):
        return self.both(x)[0]

    def both(self, x):
        """(slot logits N x SLOTS x colours, direction logits N): the colours left to right, and whether
        that is the reading order (the tolerance band on the right)."""
        f = self.pool(self.features(x)).flatten(1)
        return self.head(f).view(-1, SLOTS, len(COLOURS)), self.direction(f)[:, 0]


class BandCTC(nn.Module):
    """The columns reader: the same convolutions as BandNet, then for each of COLUMNS columns along the
    body (height averaged) colour logits from a narrow window round it (a 3-wide 1-D convolution, the
    same at every column), "none" being the CTC blank. The direction (does the reading order run left
    to right?) comes from a small summary of every column, which keeps their positions: the tolerance
    band's gap is at one end. About 120 k parameters at width 16."""

    kind = "ctc"

    def __init__(self, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(3, w), _block(w, w, 2), _block(w, 2 * w), _block(2 * w, 2 * w, 2),
                                      _block(2 * w, 4 * w), _block(4 * w, 4 * w, 2), _block(4 * w, 4 * w))
        self.columns = nn.Sequential(nn.Conv1d(4 * w, 4 * w, 3, padding=1), nn.ReLU(inplace=True),
                                     nn.Conv1d(4 * w, len(COLOURS), 1))
        self.squeeze = nn.Conv1d(4 * w, 4, 1)
        self.direction = nn.Linear(4 * COLUMNS, 1)

    def forward(self, x):
        return self.both(x)[0]

    def both(self, x):
        """(column logits N x COLUMNS x colours, left to right; direction logits N)."""
        f = nn.functional.adaptive_avg_pool2d(self.features(x), (1, COLUMNS))[:, :, 0]     # N x C x COLUMNS
        return self.columns(f).transpose(1, 2), self.direction(self.squeeze(f).flatten(1))[:, 0]


def prepare(image, white):
    """A crop (BGR) as the network's input: 3 × H × W float RGB, the plate's level at 1."""
    small = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1].astype(np.float32) / np.asarray(white, np.float32)
    return np.ascontiguousarray(np.clip(rgb, 0, 1.5).transpose(2, 0, 1))


def slots_of(colours):
    """Band colours in reading order as the five slots read that way: a 4-band part's fifth is empty."""
    return tuple(colours) + ("none",) * (SLOTS - len(colours))


_INDEX = {c: i for i, c in enumerate(COLOURS)}
ALL_CANDIDATES = [(0.0, ("black",))] + [(round(ohms, 4), colours)       # rounded: 22 x 0.1 is 220 x 0.01
                                        for ohms, colours in value.candidates(SCOPE_OHM, value.ALL_MULTIPLIERS)]
_CANDIDATES = ALL_CANDIDATES
_FORWARD = np.array([[_INDEX[c] for c in slots_of(colours)] for _, colours in _CANDIDATES])
_OHMS = np.array([ohms for ohms, _ in _CANDIDATES], float)
_SEQ = [[_INDEX[c] for c in colours] for _, colours in _CANDIDATES]          # band sequences, reading order


def _ctc_targets():
    seqs = _SEQ + [q[::-1] for q in _SEQ]                                   # read left to right, then right to left
    padded = torch.zeros(len(seqs), SLOTS, dtype=torch.long)
    for i, q in enumerate(seqs):
        padded[i, :len(q)] = torch.tensor(q)
    return padded, torch.tensor([len(q) for q in seqs])


_CTC_TARGETS, _CTC_LENGTHS = _ctc_targets()


def use_values(ohms=None):
    """Decode only these values (each as both its 4-band and 5-band code), or every value in scope (None).
    The robot only meets the kit's values, so a model trained on the kit saves them and load() limits the
    decoder to them: a misread then has to land on another kit value to come out wrong, and most become
    low confidence instead (experiments/2026-11-11_band_reader_kit, live test). Module-wide: the last
    model loaded sets it."""
    global _CANDIDATES, _FORWARD, _OHMS, _SEQ, _CTC_TARGETS, _CTC_LENGTHS
    if ohms is None:
        _CANDIDATES = ALL_CANDIDATES
    else:
        keep = [round(float(o), 4) for o in ohms]
        _CANDIDATES = [(o, c) for o, c in ALL_CANDIDATES if any(abs(o - k) < 1e-6 for k in keep)]
        missing = [k for k in keep if not any(abs(o - k) < 1e-6 for o, _ in _CANDIDATES)]
        if missing:
            raise ValueError(f"not values the reader knows: {missing}")
    _FORWARD = np.array([[_INDEX[c] for c in slots_of(colours)] for _, colours in _CANDIDATES])
    _OHMS = np.array([o for o, _ in _CANDIDATES], float)
    _SEQ = [[_INDEX[c] for c in colours] for _, colours in _CANDIDATES]
    _CTC_TARGETS, _CTC_LENGTHS = _ctc_targets()


def ctc_scores(column_logits, direction_logits=None, batch=32):
    """Each value in scope scored per crop: N x 2K numpy, the log-likelihood of its bands read left to
    right (first K) and right to left (last K) over the columns, plus the log-probability of that way
    round when there is a direction. The scores decode() takes as a 1-D row."""
    lp = torch.log_softmax(column_logits.float(), -1)
    n, t, _ = lp.shape
    k2 = len(_CTC_LENGTHS)
    targets, lengths = _CTC_TARGETS.to(lp.device), _CTC_LENGTHS.to(lp.device)
    out = []
    for i in range(0, n, batch):
        part = lp[i:i + batch]
        m = len(part)
        nll = nn.functional.ctc_loss(part.transpose(0, 1).repeat_interleave(k2, 1), targets.repeat(m, 1),
                                     torch.full((m * k2,), t, dtype=torch.long, device=lp.device),
                                     lengths.repeat(m), blank=0, reduction="none", zero_infinity=False)
        out.append(-nll.view(m, k2))
    scores = torch.cat(out)
    if direction_logits is not None:
        k = k2 // 2
        scores[:, :k] += nn.functional.logsigmoid(direction_logits)[:, None]
        scores[:, k:] += nn.functional.logsigmoid(-direction_logits)[:, None]
    return scores.cpu().numpy()


@dataclass(frozen=True)
class Reading:
    value: float | None     # ohms; None for unknown, to be measured
    best: float             # the most likely value, whether or not it is trusted
    confidence: float       # its share of all values in scope
    colours: tuple          # its bands, in reading order


def decode(log_probs, min_confidence=MIN_CONFIDENCE):
    """The Reading from what log_probs gives: BandNet's slot log-probabilities (SLOTS x colours, or with a
    direction row after them: [log P(left to right), log P(right to left), 0, ...]), or BandCTC's scores
    per value and way round (1-D, ctc_scores)."""
    lp = np.asarray(log_probs, np.float64)
    if lp.ndim == 1:
        scores = lp
    else:
        ways = lp[SLOTS, :2] if len(lp) > SLOTS else np.zeros(2)
        lp = lp[:SLOTS]
        rows = np.arange(SLOTS)
        scores = np.concatenate([lp[rows, _FORWARD].sum(1) + ways[0], lp[rows, _FORWARD[:, ::-1]].sum(1) + ways[1]])
    if not np.isfinite(scores).any():
        scores = np.zeros_like(scores)
    share = np.exp(scores - scores.max())
    share /= share.sum()
    both = share[:len(_OHMS)] + share[len(_OHMS):]
    totals = {}
    for ohms, p in zip(_OHMS, both):
        totals[ohms] = totals.get(ohms, 0.0) + p
    best = max(totals, key=totals.get)
    colours = _CANDIDATES[int(np.argmax(scores)) % len(_OHMS)][1]
    return Reading(best if totals[best] >= min_confidence else None, best, totals[best], colours)


def ohms_text(ohms):
    """470, 4k7, 10k, 1M5: a value as printed on a parts list."""
    for scale, unit in ((1e6, "M"), (1e3, "k")):
        if ohms >= scale:
            whole, part = divmod(round(ohms / scale * 10), 10)
            return f"{whole}{unit}{part if part else ''}"
    return f"{ohms:g}"


def load(path):
    """The reader in eval mode. Weights from before the direction head load without one (has_direction
    False), and their readings weigh both ways equally, as they always did."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    net = (BandCTC if saved.get("kind") == "ctc" else BandNet)(saved["width"])
    net.temperature = float(saved.get("temperature", 1.0))     # BandCTC: scores divided by it (training fits it)
    use_values(saved.get("values"))                             # the values it may answer; None for all in scope
    missing, _ = net.load_state_dict(saved["state"], strict=False)
    net.has_direction = not any(k.startswith("direction.") for k in missing)
    return net.eval()


@torch.no_grad()
def log_probs(net, image, white):
    """Log-probabilities for a straightened crop (BGR, white balanced, the plate at `white`): SLOTS x
    colours, and with a direction head one row more (see decode). Rows average across views (Views)."""
    x = torch.from_numpy(prepare(image, white))[None]
    slots, direction = net.both(x)
    if getattr(net, "kind", "") == "ctc":
        return ctc_scores(slots, direction)[0] / getattr(net, "temperature", 1.0)
    lp = torch.log_softmax(slots, -1)[0].numpy()
    if not getattr(net, "has_direction", True):
        return lp
    ways = np.zeros((1, len(COLOURS)))
    ways[0, :2] = torch.nn.functional.logsigmoid(torch.stack([direction, -direction]))[:, 0].numpy()
    return np.vstack([lp, ways])


def read(net, image, white, min_confidence=MIN_CONFIDENCE):
    """The Reading of a straightened crop (BGR, white balanced, the plate at `white`)."""
    return decode(log_probs(net, image, white), min_confidence)


class Views:
    """Readings over a body's recent views, for a live camera: each body's slot log-probabilities are
    averaged over its last `window` views and decoded once. Averaged, not summed: frames a moment apart
    are nearly the same picture, not independent evidence, so they mustn't add up to more confidence
    than one good view gives; the average only smooths the noise. Bodies are named by the caller's ids."""

    def __init__(self, window=5, min_confidence=MIN_CONFIDENCE):
        self.window, self.min_confidence = window, min_confidence
        self.seen = {}

    def add(self, body, lp):
        """One view's log-probabilities for body (any hashable id). Returns its Reading over its views."""
        views = self.seen.setdefault(body, deque(maxlen=self.window))
        views.append(np.asarray(lp, np.float64))
        return decode(np.mean(views, axis=0), self.min_confidence)

    def keep_only(self, bodies):
        """Forgets every body but these (the ones still in view)."""
        self.seen = {b: v for b, v in self.seen.items() if b in set(bodies)}

"""The learned band reader: a small network that reads a body's bands, on the CPU.

A straightened crop goes in and, for each of five slots along the body, left to right, a colour comes
out (or none: a 4-band part leaves one end slot empty). The network says nothing about values: every
value in scope is scored from the slots, both ways round, so only a real value can come out, and its
share of the total is the reading's confidence. Below MIN_CONFIDENCE the part is unknown, for the
robot to measure instead.

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

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1)).view(-1, SLOTS, len(COLOURS))


def prepare(image, white):
    """A crop (BGR) as the network's input: 3 × H × W float RGB, the plate's level at 1."""
    small = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1].astype(np.float32) / np.asarray(white, np.float32)
    return np.ascontiguousarray(np.clip(rgb, 0, 1.5).transpose(2, 0, 1))


def slots_of(colours):
    """Band colours in reading order as the five slots read that way: a 4-band part's fifth is empty."""
    return tuple(colours) + ("none",) * (SLOTS - len(colours))


_INDEX = {c: i for i, c in enumerate(COLOURS)}
_CANDIDATES = value.candidates()
_FORWARD = np.array([[_INDEX[c] for c in slots_of(colours)] for _, colours in _CANDIDATES])
_OHMS = np.array([ohms for ohms, _ in _CANDIDATES], float)


@dataclass(frozen=True)
class Reading:
    value: float | None     # ohms; None for unknown, to be measured
    best: float             # the most likely value, whether or not it is trusted
    confidence: float       # its share of all values in scope
    colours: tuple          # its bands, in reading order


def decode(log_probs, min_confidence=MIN_CONFIDENCE):
    """The Reading from slot log-probabilities (SLOTS × colours)."""
    lp = np.asarray(log_probs, np.float64)
    rows = np.arange(SLOTS)
    scores = np.concatenate([lp[rows, _FORWARD].sum(1), lp[rows, _FORWARD[:, ::-1]].sum(1)])
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
    net = BandNet(torch.load(path, map_location="cpu", weights_only=True)["width"])
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
    return net.eval()


@torch.no_grad()
def log_probs(net, image, white):
    """Slot log-probabilities (SLOTS × colours) for a straightened crop (BGR, white balanced, the plate at
    `white`)."""
    x = torch.from_numpy(prepare(image, white))[None]
    return torch.log_softmax(net(x), -1)[0].numpy()


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

"""Resistor or not: a small classifier on each segmented object's colour crop (the "resistor tag").

The segmentation model (seg_objects_*) outlines every pickable object as one class, from greyscale.
Whether an outline is a resistor is decided here, from colour and shape: a resistor's bands and body
paint, a diode's single band, a capacitor's print, a screw's metal, and the lead stubs at its ends.
Design: experiments/2026-11-09_object_seg_design ("Resistor or not"); results:
experiments/2026-11-09_resistor_tag.

Input: the model view in colour, i.e. what the segmentation model is given before it is made grey
(rectify.model_input's first value; rectify.model_image for a manifest frame): rectified at 10 px/mm,
white-balanced with the plate at the training level, off-plate masked. So the crop doesn't drift with
the room's light or show whatever lies past the plate. Each object is cut out (cutout) turned so its
long side lies along x, at its true scale (size is a cue), with 1 mm round its outline kept so lead
stubs show, black beyond.

What the robot does with each tag (agreed with band reading, robot-offline-01): "resistor", read its
bands; "unsure", read them but never trust the value (measure it instead); "not", skip it. The
failure to avoid is a confident band value on a non-resistor.

    net = resistor_tag.load("models/resistor_tag_v1/resistor_tag.pt")       # CPU
    for p, tag in resistor_tag.classify(net, view, polygons): ...
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

PX_PER_MM = 10.0             # the rectified scale (configs/dataset.toml [rectify] px_per_mm)
WINDOW_MM = (24.0, 12.0)     # crop size along and across the object's long side; longer is cut short
SIZE = (int(WINDOW_MM[0] * PX_PER_MM), int(WINDOW_MM[1] * PX_PER_MM))   # (240, 120) px, w × h
KEEP_MM = 1.0                # kept round the outline: lead stubs, the outline's own error
UNSURE = (0.2, 0.8)          # P(resistor) between these: "unsure"


def cutout(view, polygon):
    """The object as the classifier sees it: SIZE[1] × SIZE[0] × 3 uint8 BGR, long side along x,
    centred, black beyond KEEP_MM round the outline. view: the colour model view (module doc)."""
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(poly)
    if w < h:
        angle += 90.0
    rot = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
    rot[:, 2] += (SIZE[0] / 2 - cx, SIZE[1] / 2 - cy)
    crop = cv2.warpAffine(view, rot, SIZE, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    pts = poly @ rot[:, :2].T + rot[:, 2]
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    r = int(round(KEEP_MM * PX_PER_MM))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return crop * mask[:, :, None]


class Net(nn.Module):
    """A small CNN, light enough for the CPU on every outline of every live frame: a stride-2 stem,
    four conv blocks, global average pool, one logit (resistor)."""

    def __init__(self, width=16):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)]

        layers = [nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        chans = (width, 2 * width, 4 * width, 4 * width)
        c_in = width
        for c in chans:
            layers += block(c_in, c)
            c_in = c
        self.width = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def to_tensor(crops):
    """List of cutouts (BGR uint8) to an N × 3 × H × W float tensor in [0, 1], RGB."""
    a = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def load(path, device="cpu"):
    """The trained classifier, in eval mode (on the CPU unless told otherwise)."""
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    net = Net(saved.get("width", 16))
    net.load_state_dict(saved["state"])
    return net.to(device).eval()


@torch.no_grad()
def logits(net, crops, batch=64, both_ways=True):
    """The resistor logit for each cutout, numpy. both_ways: averaged with the crop turned 180°, since
    which end comes first is arbitrary. Logits average across views (log-odds), for Steady ids."""
    if not len(crops):
        return np.zeros(0)
    device = next(net.parameters()).device
    out = []
    for i in range(0, len(crops), batch):
        x = to_tensor(crops[i:i + batch]).to(device)
        z = net(x)
        if both_ways:
            z = (z + net(torch.flip(x, (2, 3)))) / 2
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def probability(logit):
    return 1.0 / (1.0 + np.exp(-np.asarray(logit, float)))


def tag(p, unsure=UNSURE):
    """"resistor", "not" or "unsure" for one P(resistor)."""
    return "resistor" if p >= unsure[1] else "not" if p <= unsure[0] else "unsure"


def classify(net, view, polygons, unsure=UNSURE):
    """[(P(resistor), tag)] for each polygon, in order. view: the colour model view (module doc)."""
    p = probability(logits(net, [cutout(view, q) for q in polygons]))
    return [(float(v), tag(v, unsure)) for v in p]

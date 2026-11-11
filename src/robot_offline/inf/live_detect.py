"""Live detection: the model run on the camera, with the body outlines drawn on the image.

Opens the camera with configs/camera.toml, finds the plate, and a few times a second rectifies the
newest frame, white-balances it with the plate brought to the brightness it has in training,
masks it to the plate (exactly what the model saw in training), runs the model and maps each
predicted outline back into the camera's view, where it is drawn over the resistor with its spine
and centroid (skeleton.py) in white. The plate is found again every few seconds in the
background (plate_search.Tracker), so a nudged camera doesn't
spoil it and the picture doesn't wait for it. The exposure follows the light, metered on the
plate's white top (exposure.py); E turns that off, back to the fixed exposure in camera.toml.

Detections are held steady from run to run (steady.py): a body shows once it passes the threshold in 2
of the last 5 runs and stays while its mean score holds, so one whose score swings round the threshold
doesn't flicker; its score shown is that mean. H shows single runs instead, to compare.

Each body's value is read from its bands (post/bands/learned.py, --reader: models/band_reader_kit_v1,
the reader trained on the resistor kit, decoding only the kit's values), averaged over its last few
runs: "4k7 1.00" is a reading at or above the reader's own limit (0.998 for band_reader_kit_v1, saved
with it), "? 1k 0.90" a best guess below it, which the robot would measure instead. U takes a close-up of
every body on the plate, each at the focus its distance needs (post/bands/closeup.py, about 1.5 s a
body; the picture waits), and folds each close-up's reading in with the live ones, half and half, while
the body stays in view: "4k7 1.00 +c". With values on, the exposure keeps the plate at the reading level (grey 205 to 225),
where the bands read best; B turns values off, and the exposure goes back to the model's band.

Each body is also tagged resistor or not (post/resistor_tag.py, --tagger), its P(resistor) averaged over
the same runs, since an object model outlines screws, diodes and the rest too. Bands are read only on
a body that isn't tagged "not". On one tagged "unsure" the reading is shown as "unsure 0.55: ~4k7, measure":
never a value to use, whatever the reader's confidence. Without the tagger's weights every body counts
as a resistor.

Each body's label on the picture says what the robot should do with it: green "#3  4k7", a value to use
(at or above the reader's limit); amber "#3  4k7?", a best guess to measure instead; grey, not a resistor
or unsure. On the right, with values on, the readings panel: a row per body, numbered and coloured like
its outline, with the band crop the reader read (its close-up's once U has taken one), the value or
"measure", its confidence against the limit, the bands it read as colour chips in reading order, and the
detection score, resistor tag and "+c" when a close-up is folded in. Values outside the sorter's range
(10 ohm to 10 kohm, docs/classification.md) are marked. T swaps the panel for the shapes panel.

The shapes panel shows one tile per detected body with only its segmented pixels: taken from the
top-down image (the same 10 px/mm for every body, free of the camera's angle), turned so the body lies
horizontally, enlarged, and black outside the predicted outline. On each, skeleton.py's results: the
thinned skeleton (cyan pixels), the spine from end to end (white) and the centroid (white dot), labelled
with the body's length × width and yaw along its spine and its centroid in scene mm on the plate. K hides
and shows all of these.

    python -m robot_offline.inf.live_detect
    python -m robot_offline.inf.live_detect --weights models/seg_grey_v2/best.pt --fps 3 --conf 0.5
    python -m robot_offline.inf.live_detect --last-fit       # the plate where it was last found, no search

Keys:  q / Esc  quit      v  camera view / top-down view      m  top-down: what the model sees (masked)
       + / -  confidence threshold up / down      c  find the plate again      s  save the shown image
       e  exposure following the light, or fixed      k  skeleton, spine and centroid shown or hidden
       h  detections held steady over runs, or each run on its own      b  band values on or off
       u  close-up of every body, read and folded into its value      t  readings panel / shapes panel
       [ / ]  confidence a value needs, down / up
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.inf.steady import FLOOR, Steady
from robot_offline.post import resistor_tag, skeleton
from robot_offline.post.bands import closeup, crops, learned, value
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.camera import label, open_camera, show

WINDOW = "robot_offline live detection"
CAMERA_SCALE = 0.4          # the camera view is shown at this share of 4608 × 2592
COLOURS = ((0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 200, 80))
TILE = (340, 150)           # px, each body's cut-out tile
ZOOM = 4.0                  # cut-outs are shown this much larger than the 10 px/mm top-down image
PANEL_COLUMNS = 2
SKELETON_COLOUR = (255, 255, 0)
# What the robot does with a body, as the colour of its label (BGR): use the value, measure it, or skip it
STATUS_COLOUR = {"value": (60, 150, 40), "measure": (0, 140, 230), "unsure": (0, 140, 230),
                 "not": (120, 120, 120), "off": (120, 120, 120)}
SORTER_OHM = (10.0, 10_000.0)   # the values the sorter sorts (docs/classification.md); others it rejects
PANEL_W = 600                   # the readings panel's width, px
ROW_H = 132                     # one body's row in it
HEADER_H = 62                   # the panel's title and legend
PANEL_BG, PANEL_INK, PANEL_MUTED = (28, 28, 28), (235, 235, 235), (150, 150, 150)
# The confidence a value needs comes from learned.load: configs/band_reader.toml [deploy] min_confidence
# (provisionally 0.7 since 2026-11-11) or else the model's own limit (0.998 for band_reader_kit_v1). Below
# the model's own it is shown as provisional. [ and ] step through LIMIT_STEPS; --min-confidence sets it.
LIMIT_STEPS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)


def _text(img, text, org, colour, scale=0.5):
    """Text with a dark shadow drawn at the text's own thickness, offset, so the two keep the same spacing."""
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw(img, polygons, scores, scale=1.0, axes=(), notes=()):
    """Numbered outlines, a light fill and the confidence on img (BGR), in place. Polygons in
    img px / scale. axes: (spine, centroid) per body, in the same px, drawn as a line and a dot.
    notes: text after each body's confidence (its value)."""
    fill = img.copy()
    for i, poly in enumerate(polygons):
        cv2.fillPoly(fill, [np.round(poly * scale).astype(np.int32)], COLOURS[i % len(COLOURS)], cv2.LINE_AA)
    cv2.addWeighted(fill, 0.25, img, 0.75, 0, dst=img)
    for i, (poly, score) in enumerate(zip(polygons, scores)):
        pts = np.round(poly * scale).astype(np.int32)
        colour = COLOURS[i % len(COLOURS)]
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        x, y = pts.min(0)
        note = notes[i] if i < len(notes) else None
        if isinstance(note, BodyReading):
            chip(img, f"#{i + 1}  {note.label}", (int(x), int(y) - 6), STATUS_COLOUR[note.status], colour)
        else:
            note = f"   {note}" if note else ""
            _text(img, f"#{i + 1} {score:.2f}{note}", (int(x), int(y) - 4), colour)
    for spine, centroid in axes:
        cv2.polylines(img, [np.round(spine * scale * 16).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA, 4)
        cv2.circle(img, tuple(int(v) for v in np.round(centroid * scale * 16)), 3 * 16, (255, 255, 255), -1,
                   cv2.LINE_AA, 4)


def chip(img, text, bottom_left, fill, edge=None, scale=0.55):
    """A filled label: text in white on fill, with a thin edge in the body's colour, its bottom left at
    bottom_left (kept inside img)."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(bottom_left[0], 0, max(0, img.shape[1] - w - 10)))
    y = int(np.clip(bottom_left[1], h + 8, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y + 2), fill, -1)
    if edge is not None:
        cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y + 2), edge, 1)
    cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def cutout(image, poly, size=TILE, zoom=ZOOM, body=None):
    """Only the pixels inside poly (image px), turned so the body's long axis is horizontal,
    enlarged by zoom and centred in a size tile; black elsewhere. With body (its
    skeleton.BodySkeleton), its skeleton, spine and centroid are drawn on."""
    m = cv2.moments(poly.astype(np.float32))
    if m["m00"] <= 0:
        return np.zeros((size[1], size[0], 3), np.uint8)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))   # long axis
    rot = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    rot[:, 2] += (size[0] / 2 - cx, size[1] / 2 - cy)
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    pixels = cv2.warpAffine(image, rot, size, flags=cv2.INTER_CUBIC)
    keep = cv2.warpAffine(inside, rot, size, flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    tile = (pixels * keep).astype(np.uint8)
    if body is not None:
        to_tile = lambda p: np.asarray(p, float).reshape(-1, 2) @ rot[:, :2].T + rot[:, 2]
        for x, y in np.round(to_tile(body.skeleton)).astype(int):
            cv2.circle(tile, (int(x), int(y)), 1, SKELETON_COLOUR, -1)
        cv2.polylines(tile, [np.round(to_tile(body.spine) * 16).astype(np.int32)], False, (255, 255, 255), 1,
                      cv2.LINE_AA, 4)
        centre = tuple(int(v) for v in np.round(to_tile(body.centroid)[0] * 16))
        cv2.circle(tile, centre, 5 * 16, (0, 0, 0), -1, cv2.LINE_AA, 4)
        cv2.circle(tile, centre, 4 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
    return tile


def cutout_panel(image, polygons, scores, height, bodies=(), settings=None):
    """A panel of numbered cut-outs, PANEL_COLUMNS wide and height tall. With bodies
    (skeleton.BodySkeleton or None per polygon) and the rectify settings, each tile shows its body's
    skeleton, spine and centroid, with its length × width, yaw and centroid in scene mm."""
    cell_w, cell_h = TILE[0] + 20, TILE[1] + 34
    panel = np.full((height, PANEL_COLUMNS * cell_w, 3), 24, np.uint8)
    rows = max(1, (height - 10) // cell_h)
    _text(panel, "segmented pixels, top-down, 4x", (10, 20), (220, 220, 220))
    for i, (poly, score) in enumerate(zip(polygons, scores)):
        if i >= rows * PANEL_COLUMNS:
            _text(panel, f"+{len(polygons) - i} more", (10, height - 10), (220, 220, 220))
            break
        col, row = divmod(i, rows)
        x, y = 10 + col * cell_w, 30 + row * cell_h
        colour = COLOURS[i % len(COLOURS)]
        body = bodies[i] if i < len(bodies) and settings is not None else None
        s = settings.px_per_mm if body is not None else 1.0
        size = f"   {body.length / s:.1f} x {body.width / s:.1f} mm  {body.yaw_deg:.0f} deg" if body else ""
        _text(panel, f"#{i + 1}  {score:.2f}{size}", (x, y + 16), colour)
        tile = cutout(image, poly, body=body)
        if body is not None:
            cx, cy = rectify.rect_to_mm(body.centroid, settings)[0]
            _text(tile, f"centroid x {cx:.1f}  y {cy:.1f} mm", (6, TILE[1] - 8), (220, 220, 220), 0.45)
        panel[y + 22:y + 22 + TILE[1], x:x + TILE[0]] = tile
        cv2.rectangle(panel, (x - 1, y + 21), (x + TILE[0], y + 22 + TILE[1]), colour, 1)
    return panel


class TagViews:
    """Each body's resistor logit averaged over its last `window` runs (log-odds average, like
    learned.Views), so one run's odd crop doesn't flip its tag. Bodies are named by steady.Steady's ids."""

    def __init__(self, window=5):
        self.window, self.seen = window, {}

    def add(self, body, logit):
        """One run's logit for body. Returns (P(resistor), tag) over its recent runs."""
        runs = self.seen.setdefault(body, deque(maxlen=self.window))
        runs.append(float(logit))
        p = float(resistor_tag.probability(np.mean(runs)))
        return p, resistor_tag.tag(p)

    def keep_only(self, bodies):
        self.seen = {b: v for b, v in self.seen.items() if b in set(bodies)}


def tag_bodies(tagger, tag_views, model_view, polygons, ids=None):
    """[(P(resistor), tag)] per polygon, from the colour model view; averaged per id when ids are given.
    Without a tagger every body is (None, None): counted as a resistor."""
    if tagger is None:
        return [(None, None)] * len(polygons)
    z = resistor_tag.logits(tagger, [resistor_tag.cutout(model_view, q) for q in polygons])
    if ids is None:
        return [(float(p), resistor_tag.tag(p)) for p in resistor_tag.probability(z)]
    tags = [tag_views.add(i, v) for i, v in zip(ids, z)]
    tag_views.keep_only(ids)
    return tags


def tag_text(p, tag):
    return "" if tag is None else "not a resistor" if tag == "not" else f"{tag} {p:.2f}"


@dataclass
class BodyReading:
    """What was read of one body, for its label and its row in the readings panel."""
    status: str                 # "value" (use it), "measure" (below the limit), "unsure", "not" (not a resistor)
    reading: object = None      # its learned.Reading over its views; None if not read
    crop: object = None         # the band crop read this run (BGR), for the panel
    p: float | None = None      # P(resistor), from the resistor tag
    tag: str | None = None
    closeup: bool = False       # a close-up's reading is folded in
    score: float | None = None  # the detection's score

    @property
    def label(self):
        """Short, for the picture: "4k7", "4k7?" (measure), "not a resistor", "unsure"."""
        if self.status == "not":
            return "not a resistor"
        if self.status == "unsure":
            return "unsure"
        text = learned.ohms_text(self.reading.best)
        return text if self.status == "value" else text + "?"

    @property
    def note(self):
        """The longer text form: "4k7 0.99", "? 1k 0.50", "not a resistor", "unsure 0.55: ~4k7, measure"."""
        if self.status == "not":
            return tag_text(self.p, self.tag)
        r = self.reading
        if self.status == "unsure":
            return f"{tag_text(self.p, self.tag)}: ~{learned.ohms_text(r.best)}, measure"
        text = (f"{learned.ohms_text(r.value)} {r.confidence:.2f}" if r.value is not None
                else f"? {learned.ohms_text(r.best)} {r.confidence:.2f}")
        return text + (" +c" if self.closeup else "")


def read_values(net, views, frame, cal, plain, polygons, ids=None, tags=None):
    """Each body's value as text (BodyReading.note); see read_bodies."""
    return [b.note for b in read_bodies(net, views, frame, cal, plain, polygons, ids, tags)]


def read_bodies(net, views, frame, cal, plain, polygons, ids=None, tags=None, scores=None):
    """Each body read as tools.read_plate reads it: a BodyReading per polygon. ids: the bodies' ids
    (steady.Steady), to average each one's reading over its recent runs (and its pinned close-up); None
    reads this run alone. tags: (P(resistor), tag) per body: a body tagged "not" isn't read, and one tagged
    "unsure" is read but never a value to use."""
    tags = tags or [(None, None)] * len(polygons)
    scores = list(scores) if scores is not None else [None] * len(polygons)
    _, gains = rectify.white_balance(plain, cal.settings)          # crops keep the frame's own level
    balanced = np.clip(plain.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    white = float(np.percentile(cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)[150:1050, 150:1050], 70))
    out = []
    for i, (poly, (p, tag)) in enumerate(zip(polygons, tags)):
        if tag == "not":
            out.append(BodyReading("not", p=p, tag=tag, score=scores[i]))
            continue
        crop = crops.cut(frame, cal, poly, gains).image
        lp = learned.log_probs(net, crop, white)
        r = views.add(ids[i], lp) if ids is not None else learned.decode(lp, views.min_confidence)
        status = "unsure" if tag == "unsure" else "value" if r.value is not None else "measure"
        out.append(BodyReading(status, r, crop, p, tag, ids is not None and ids[i] in views.pinned, scores[i]))
    if ids is not None:
        views.keep_only(ids)
    return out


def readings_panel(readings, height, limit, close_crops=None, ids=None, model_limit=None):
    """The readings panel, PANEL_W wide and height tall: a row per body (BodyReading), numbered and
    coloured like its outline. close_crops: {id: close-up crop} to show in place of the live one."""
    panel = np.full((height, PANEL_W, 3), PANEL_BG, np.uint8)
    cv2.putText(panel, "Readings", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, PANEL_INK, 2, cv2.LINE_AA)
    below = model_limit is not None and limit < model_limit
    cv2.putText(panel, f"a value is used at confidence >= {limit:g}" + (
                f" (provisional; the model's own: {model_limit:g})" if below else "; below it, the part is measured"),
                (130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 230) if below else PANEL_MUTED, 1, cv2.LINE_AA)
    lx = 12
    for name, key in (("use the value", "value"), ("measure it", "measure"), ("not a resistor / unsure", "not")):
        cv2.rectangle(panel, (lx, 38), (lx + 16, 52), STATUS_COLOUR[key], -1)
        cv2.putText(panel, name, (lx + 22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_INK, 1, cv2.LINE_AA)
        lx += 34 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    rows = max(1, (height - HEADER_H - 4) // ROW_H)
    for i, b in enumerate(readings):
        if i >= rows:
            cv2.putText(panel, f"+{len(readings) - i} more", (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        PANEL_MUTED, 1, cv2.LINE_AA)
            break
        y = HEADER_H + i * ROW_H
        colour = COLOURS[i % len(COLOURS)]
        cv2.line(panel, (8, y), (PANEL_W - 8, y), (60, 60, 60), 1)
        cv2.circle(panel, (26, y + 26), 15, colour, -1, cv2.LINE_AA)
        cv2.putText(panel, str(i + 1), (26 - 6 * len(str(i + 1)), y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        crop = (close_crops or {}).get(ids[i]) if ids is not None and i < len(ids) else None
        crop = crop if crop is not None else b.crop
        if crop is not None:
            tile = cv2.resize(crop, (240, 80), interpolation=cv2.INTER_AREA)
            panel[y + 8:y + 88, 50:290] = tile
            cv2.rectangle(panel, (49, y + 7), (290, y + 88), colour, 1)
        x = 304
        status = STATUS_COLOUR[b.status]
        if b.status in ("value", "measure", "unsure") and b.reading is not None:
            r = b.reading
            big = learned.ohms_text(r.best) + ("" if b.status == "value" else "?")
            cv2.putText(panel, big, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, status, 2, cv2.LINE_AA)
            word = {"value": "use", "measure": "measure", "unsure": "unsure: measure"}[b.status]
            cv2.putText(panel, f"{word}  -  conf {r.confidence:.3f} (needs {limit:g})", (x, y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PANEL_INK, 1, cv2.LINE_AA)
            if not SORTER_OHM[0] <= r.best <= SORTER_OHM[1]:
                cv2.putText(panel, "outside the sorter's 10R-10k", (x, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 140, 230), 1, cv2.LINE_AA)
            cx = 50
            for c in r.colours:                         # the bands it read, in reading order
                rgb = value.paint_rgb(c).clip(0, 1) * 255
                cv2.rectangle(panel, (cx, y + 94), (cx + 40, y + 112), (int(rgb[2]), int(rgb[1]), int(rgb[0])), -1)
                cv2.rectangle(panel, (cx, y + 94), (cx + 40, y + 112), (90, 90, 90), 1)
                ink = (0, 0, 0) if rgb.mean() > 110 else (255, 255, 255)
                cv2.putText(panel, c[:3], (cx + 6, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink, 1, cv2.LINE_AA)
                cx += 46
        else:
            cv2.putText(panel, "not a resistor", (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status, 2, cv2.LINE_AA)
            cv2.putText(panel, "not read", (x, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PANEL_MUTED, 1, cv2.LINE_AA)
        details = []
        if b.score is not None:
            details.append(f"det {b.score:.2f}")
        if b.tag is not None:
            details.append(f"{b.tag} {b.p:.2f}")
        if b.closeup:
            details.append("+c close-up")
        cv2.putText(panel, "   ".join(details), (x, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_MUTED, 1,
                    cv2.LINE_AA)
    return panel


def take_closeups(cap, cal, net, views, polygons, ids, tags, seg, grey, camera):
    """A close-up of each body (not tagged "not"), each at its own focus, read and pinned to its id in
    views. Returns (bodies done, seconds, {id: close-up crop})."""
    t0 = time.perf_counter()
    want = [i for i, (_, tag) in enumerate(tags or [(None, None)] * len(polygons)) if tag != "not"]
    if not want:
        return 0, 0.0, {}
    taken = {}
    for i, k in zip(want, closeup.take(cap, cal, [polygons[i] for i in want], seg=seg, grey=grey, camera=camera)):
        views.pin(ids[i], learned.log_probs(net, k.crop.image, k.white))
        taken[ids[i]] = k.crop.image
    return len(want), time.perf_counter() - t0, taken


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.inf.live_detect",
                                     description="Run the model live on the camera and draw the body outlines.")
    parser.add_argument("--weights", type=Path, default=config.ROOT / "models" / "seg_grey_v2" / "best.pt")
    parser.add_argument("--fps", type=float, default=3.0, help="detections per second (3)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0.5)")
    parser.add_argument("--reader", type=Path, default=config.ROOT / "models" / "band_reader_kit_v1" / "band_reader.pt",
                        help="band reader weights (values aren't shown if missing)")
    parser.add_argument("--tagger", type=Path,
                        default=config.ROOT / "models" / "resistor_tag_v1" / "resistor_tag.pt",
                        help="resistor-or-not weights (every body counts as a resistor if missing)")
    parser.add_argument("--min-confidence", type=float,
                        help="confidence a value needs (default: configs/band_reader.toml [deploy], provisionally 0.7, "
                             "else the reader's own limit, 0.998 for band_reader_kit_v1). [ and ] change it live")
    parser.add_argument("--last-fit", action="store_true",
                        help="use where the plate was last found (data/plate/last_rectify.json) without searching, "
                             "for a rig that hasn't moved; c searches again")
    args = parser.parse_args(argv)
    if not args.weights.exists():
        parser.error(f"no weights at {args.weights}; train first, e.g. python -m robot_offline.training.train_synth")
    if args.weights.stat().st_size < 1_000_000:
        parser.error(f"{args.weights} is a Git LFS pointer, not the model; run git lfs pull")

    from ultralytics import YOLO

    ds = config.load("dataset")
    settings = rectify.Settings.from_config(ds)
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    print(f"Loading {args.weights} ...")
    model = YOLO(str(args.weights))
    grey = tiling.wants_grey(args.weights)
    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    camera = config.load("camera")
    cap = open_camera(camera)
    fixed = camera["camera"]["exposure"], camera["camera"]["gain"]
    print(__doc__.split("Keys:", 1)[1].rstrip())
    net = learned.load(args.reader) if args.reader.exists() else None
    if net is None:
        print(f"No band reader at {args.reader}: no values")
    values = net is not None
    model_limit = net.saved_min_confidence if net is not None else learned.MIN_CONFIDENCE
    views = learned.Views(min_confidence=args.min_confidence if args.min_confidence is not None
                          else net.min_confidence if net is not None else learned.MIN_CONFIDENCE)
    if net is not None:
        print(f"Band reader {args.reader}: values answered at confidence {views.min_confidence:g} or more"
              + (f" (provisional: below the reader's own limit, {model_limit:g})" if views.min_confidence < model_limit
                 else ""))
    tagger = resistor_tag.load(args.tagger) if args.tagger.exists() else None
    if tagger is None:
        print(f"No resistor tag at {args.tagger}: every body counts as a resistor")
    tag_views = TagViews()
    last = rectify.Calibration.load(plate_search.LAST) if args.last_fit and plate_search.LAST.exists() else None
    fitter = plate_search.Tracker(cap, settings, camera=camera, cal=last, locked=last is not None,
                                  level=learned.READING_LEVEL if values else None)   # a hand over an edge: it keeps the fit
    message, message_until = "", 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    top_down, masked, conf, skeletons, hold = False, False, args.conf, True, True
    readings_view, close_crops, tags = True, {}, []
    steady = Steady(conf)
    polygons, scores, notes, view, shown = [], [], [], None, None
    next_run, latency, post, n_runs, t_start = 0.0, 0.0, 0.0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()                             # keep reading so the newest frame is used
        if not ok:
            continue
        cal = fitter.update(frame)
        now = time.perf_counter()

        if now >= next_run:
            next_run = now + 1.0 / args.fps
            t0 = time.perf_counter()
            if cal is not None:
                plain = rectify.rectify(frame, cal)
                model_view, rect = rectify.model_input(frame, cal, plain)
                tiles = tiling.tiles(work_area, (model_view.shape[1], model_view.shape[0]),
                                     tiles_cfg["size"], tiles_cfg["overlap"])
                polygons, scores = tiling.predict_frame(model, model_view, tiles, conf=FLOOR, grey=grey)
                held = steady.update(polygons, scores)         # followed always, so h switches cleanly
                if hold:
                    polygons, scores = held
                else:
                    polygons, scores = [p for p, c in zip(polygons, scores) if c >= conf], [c for c in scores if c >= conf]
                ids = steady.ids if hold else None
                tags = tag_bodies(tagger, tag_views, model_view, polygons, ids)
                notes = (read_bodies(net, views, frame, cal, plain, polygons, ids, tags, scores) if values
                         else [tag_text(p, t) for p, t in tags])
                close_crops = {k: v for k, v in close_crops.items() if ids is not None and k in ids}
                t1 = time.perf_counter()
                bodies = [skeleton.body_skeleton(p) for p in polygons] if skeletons else []
                post = time.perf_counter() - t1
                axes = [(b.spine, b.centroid) for b in bodies if b is not None]
                if top_down:
                    view = (model_view if masked else rectify.draw_grid(rect, cal.settings)).copy()
                    draw(view, polygons, scores, axes=axes, notes=notes)
                else:
                    view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                    cv2.polylines(view, [np.round(rectify.plate_outline_px(cal) * CAMERA_SCALE).astype(np.int32)],
                                  True, (255, 255, 255), 1, cv2.LINE_AA)
                    draw(view, [rectify.to_full(p, cal) for p in polygons], scores, CAMERA_SCALE,
                         [(rectify.to_full(s, cal), rectify.to_full(c, cal)[0]) for s, c in axes], notes)
                if values and readings_view:
                    panel = readings_panel(notes, view.shape[0], views.min_confidence, close_crops, ids, model_limit)
                else:
                    panel = cutout_panel(rect, polygons, scores, view.shape[0], bodies, cal.settings)
                view = np.hstack([view, panel])
            else:
                view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                view = np.hstack([view, readings_panel([], view.shape[0], views.min_confidence) if values and readings_view
                                  else cutout_panel(view, [], [], view.shape[0])])
            latency = time.perf_counter() - t0
            n_runs += 1
            rate = n_runs / max(time.perf_counter() - t_start, 1e-6)
            mode = ("top-down, as the model sees it" if masked else "top-down") if top_down else "camera view"
            plate = fitter.status()
            exposure = fitter.control.status() if fitter.follow else f"exposure fixed at {fixed[0]}, gain {fixed[1]}"
            shape = f"skeletons {post * 1000:.0f} ms" if skeletons else "skeletons off"
            if cal is None or tagger is None:
                found = f"{len(polygons)} bodies"
            else:
                found = f"{len(polygons)} bodies, {sum(t == 'resistor' for _, t in tags)} resistors"
            reading = (f"   values at >= {views.min_confidence:g}"
                       + (f" (provisional; model's {model_limit:g})" if views.min_confidence < model_limit else "")
                       if values else "   values off")
            label(view, [f"{found}   conf >= {conf:.2f} "
                         f"({'held over runs' if hold else 'each run alone'})   {mode}{reading}",
                         f"{rate:.1f} detections/s, {latency * 1000:.0f} ms each ({shape})   {plate}", exposure]
                  + ([message] if message and time.perf_counter() < message_until else [])
                  + ["q quit  v view  m model's input  +/- confidence  c find plate  e exposure  "
                     "k skeletons  h hold  b values  u close-ups  t panel  [ ] limit  s save"])
            shown = show(view, shown, WINDOW)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("v"):
            top_down, next_run = not top_down, 0.0
        elif key == ord("m"):
            masked, top_down, next_run = not masked, True, 0.0
        elif key in (ord("+"), ord("=")):
            conf, next_run = min(0.95, conf + 0.05), 0.0
            steady.conf = conf
        elif key in (ord("-"), ord("_")):
            conf, next_run = max(0.05, conf - 0.05), 0.0
            steady.conf = conf
        elif key == ord("c"):
            fitter.restart()
            steady.clear()
        elif key == ord("h"):
            hold, next_run = not hold, 0.0
        elif key == ord("b") and net is not None:
            values, next_run = not values, 0.0
            fitter.level = learned.READING_LEVEL if values else None
        elif key == ord("k"):
            skeletons, next_run = not skeletons, 0.0
        elif key in (ord("["), ord("]")) and values:
            now_at = min(range(len(LIMIT_STEPS)), key=lambda j: abs(LIMIT_STEPS[j] - views.min_confidence))
            step = max(0, now_at - 1) if key == ord("[") else min(len(LIMIT_STEPS) - 1, now_at + 1)
            views.min_confidence, next_run = LIMIT_STEPS[step], 0.0
            message, message_until = f"a value now needs confidence {views.min_confidence:g}", time.perf_counter() + 4
        elif key == ord("t"):
            readings_view, next_run = not readings_view, 0.0
        elif key == ord("u"):
            if not (values and hold and cal is not None and polygons):
                message = "close-ups need values on (b), detections held (h) and bodies in view"
            else:
                n, secs, taken = take_closeups(cap, cal, net, views, polygons, list(steady.ids), tags, model, grey,
                                               camera)
                close_crops.update(taken)
                message = f"close-ups: {n} bodies in {secs:.1f} s, folded into their values (+c)"
            message_until, next_run = time.perf_counter() + 8, 0.0
        elif key == ord("e"):
            fitter.follow = not fitter.follow
            if not fitter.follow:
                cap.set(cv2.CAP_PROP_EXPOSURE, fixed[0])
                cap.set(cv2.CAP_PROP_GAIN, fixed[1])
        elif key == ord("s") and view is not None:
            out = config.DATA_DIR / "live"
            out.mkdir(parents=True, exist_ok=True)
            path = out / datetime.now().strftime("detect_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(path), view)
            print(f"Saved {path}")

    fitter.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

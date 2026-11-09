"""Reading a resistor's value from its bands, on any body colour.

A crop from `bands.crops` becomes a colour profile along the body's centre line, the bands are found
as runs that differ from the body paint, and every value in scope is scored against them. Whole
values are scored rather than single bands, because single dark bands can't be told apart reliably,
while a value must fit all its bands, in the standard series, in one of two directions.

Band paint is thin, so a band shows its paint darkened by the body under it: yellow on a blue body
looks green, orange looks brown and red looks maroon, while on a beige body they look like
themselves. The camera's MJPG also keeps colour coarser than brightness, so the body's colour bleeds
into a thin band. predicted() models both from the body colour measured on each crop, so a body never
seen before is still handled. The model is too simple for blue bodies, whose bands are measured
instead (ON_BLUE).

Dark bands need light. With the plate at about grey 215 (exposure -4), brown and black on a blue body
separate well (measured: L 17, a* +20 against L 4, a* 0); with it at about 160 they overlap, and
values on blue bodies can't be read. Adding gain beyond that weakens the colour again.

A reading is given only if the best value clearly beats every other: Reading.value is None
otherwise, for the "unknown" bin. See docs/classification.md.
"""

from dataclasses import dataclass

import cv2
import numpy as np

DIGITS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white")
# Band paints in Lab with plate white at L 100. Black, brown, red and gold were measured on beige bodies
# at exposure -4 (2026-11-06, values measured with a multimeter), where bands show nearly pure; orange
# is scaled up from a dimmer frame; the rest are estimates, not yet measured on a body they'd show on.
PAINT_LAB = {
    "black": (10, 1, 8), "brown": (31, 27, 33), "red": (46, 65, 40), "orange": (58, 40, 50),
    "gold": (50, 5, 50), "yellow": (85, -5, 75), "green": (45, -40, 30), "blue": (35, 10, -50),
    "violet": (40, 45, -50), "grey": (55, 0, 0), "white": (90, 0, 3), "silver": (70, 0, 0),
}
# Bands as they show on blue metal film at exposure -4, in the same Lab. Black, brown, yellow and violet
# were measured on parts of known value (2026-11-06). The others are estimates: no part with those
# bands has been seen at this exposure, so blue 3 k and 10 k are not yet shown to read correctly.
ON_BLUE = {
    "black": (4, 0, -6), "brown": (16, 18, -4), "yellow": (58, -42, 58), "violet": (24, 27, -50),
    "red": (22, 32, -6), "orange": (30, 30, 0), "green": (32, -30, 5), "blue": (30, 5, -45),
    "grey": (32, -2, -15), "white": (60, -8, -15), "gold": (34, 0, 12), "silver": (42, -3, -12),
}
BLUE_HUE_DEG = (-140, -90)   # a body with its hue, atan2(b*, a*), in here...
BLUE_MIN_CHROMA = 22.0       # ...and this much colour is blue metal film
E24 = (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
       33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91)
MULTIPLIERS = {"gold": 0.1, "black": 1, "brown": 10, "red": 100, "orange": 1000}
VALUE_RANGE_OHM = (10.0, 10000.0)      # docs/classification.md

BODY_SHOWS = 0.3            # how much the body under a band darkens it: 0 opaque paint, 1 a pure filter
BLEED = 0.15                # share of a band's a* and b* that is the body's, from MJPG's coarse colour
LIGHTNESS_WEIGHT = 0.5      # against a* and b*: shading and highlights mostly change lightness
STRIP = 0.36                # of the body's width, sampled around its centre line
HIGHLIGHT_DROP = 25         # percent of each column's brightest pixels
END_TRIM = 0.055            # of the length: runs centred inside this at either end are the cap's edge
SAME_PAINT_L = 14.0         # a run this close in lightness to the paint, of its hue, may be shading or
SAME_PAINT_HUE_DEG = 25.0   # ringing, not a band; but yellow on beige is close too, so a run as wide as
NARROW_RUN = 0.7            # the bands and more colourful than the paint stays. Narrower than this share
STRONGER_COLOUR = 1.25      # of the usual width, or with less than this share more chroma, it goes
FIND_LIGHTNESS_WEIGHT = 1.5  # finding bands, lightness counts for more: most bands are darker than paint
BAND_DIFFERENCE = 14.0      # distance from the paint, so weighted, from which a column counts as band
MIN_BAND = 0.02             # of the length, the narrowest run kept
MISSING_BAND = 14.0         # cost of a band of the value that wasn't found; tolerance bands: see below
MISSING_TOLERANCE = 4.0     # gold on beige hardly shows
WRONG_END = 4.0             # a missing tolerance band is at the end with the larger gap; added if not
EXTRA_BAND = 12.0           # cost of a band found that the value doesn't have
MIN_MARGIN = 2.0            # the best value's cost must be this far below the next value's
# Blue bodies are held back: the same brown measured L 17, a* +25 on one part and L 9, a* +1 on another,
# so a multiplier can read one step low with a good margin (470 for 4.7 k). A factor of ten is the worst
# error a sorter can make, so blue stays unknown until parts of known value show it is safe.
BLUE_MIN_MARGIN = 3.0


@dataclass(frozen=True)
class Band:
    position: float         # middle of the band along the body, 0 to 1
    width: float            # share of the body's length
    lab: np.ndarray         # L 0-100, a, b


@dataclass(frozen=True)
class Reading:
    value: float | None     # ohms; None for unknown
    best: float             # the best-scoring value, whether or not it is trusted
    margin: float           # how far it beats the next value
    colours: tuple          # the best value's bands, in reading order
    reversed: bool          # read from the crop's right end
    bands: tuple            # the Bands found, left to right
    body_lab: np.ndarray


def _lab(rgb):
    """Lab of float RGB in 0-1, any shape ending in 3."""
    rgb = np.clip(np.asarray(rgb, np.float32), 0, 1)
    return cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(rgb.shape)


def _rgb(lab):
    lab = np.asarray(lab, np.float32)
    return cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_Lab2RGB).reshape(lab.shape)


def _is_blue(body_lab):
    hue = np.degrees(np.arctan2(body_lab[2], body_lab[1]))
    return BLUE_HUE_DEG[0] <= hue <= BLUE_HUE_DEG[1] and np.hypot(body_lab[1], body_lab[2]) >= BLUE_MIN_CHROMA


def paint_rgb(colour):
    """A band paint's RGB in 0-1, plate white = 1."""
    return _rgb(PAINT_LAB[colour])


def predicted(colour, body_rgb, body_shows=BODY_SHOWS, bleed=BLEED):
    """Lab that a band of this colour should show over this body (RGB in 0-1, plate white = 1)."""
    body_rgb = np.clip(np.asarray(body_rgb, np.float32), 0, 1)
    if _is_blue(_lab(body_rgb)):
        return np.array(ON_BLUE[colour], np.float32)
    lab = _lab(_rgb(PAINT_LAB[colour]) * ((1 - body_shows) + body_shows * body_rgb))
    lab[1:] = (1 - bleed) * lab[1:] + bleed * _lab(body_rgb)[1:]
    return lab


def centre_line(image, mask, white):
    """Columns × 3 RGB along the body's middle (plate white = 1), highlights left out.

    Columns narrower than half the body (lead stubs, the tips of the outline) are not included.
    """
    rgb = image[..., ::-1].astype(np.float32) / np.asarray(white, np.float32)
    inside = mask > 127
    heights = inside.sum(0)
    xs = np.flatnonzero(heights > 0.5 * heights.max())
    line = []
    for x in range(xs.min(), xs.max() + 1):
        ys = np.flatnonzero(inside[:, x])
        if len(ys) == 0:
            line.append(line[-1] if line else np.zeros(3, np.float32))
            continue
        centre, half = (ys.min() + ys.max()) / 2, max(2.0, STRIP * (ys.max() - ys.min()) / 2)
        px = rgb[int(centre - half):int(centre + half) + 1, x]
        grey = px.mean(1)
        line.append(np.median(px[grey <= np.percentile(grey, 100 - HIGHLIGHT_DROP)], axis=0))
    return np.array(line, np.float32)


def body_colour(line):
    """The paint's RGB: the median of the brighter half of the line, which leaves the bands out."""
    grey = line.mean(1)
    return np.median(line[grey >= np.median(grey)], axis=0)


def _distance(lab, target, lightness_weight=LIGHTNESS_WEIGHT):
    d = np.asarray(lab) - target
    return np.sqrt((lightness_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _split(run, lightness, width):
    """A run holding several bands, cut at its brightest gaps, as (start, end) pairs."""
    start, end = run
    count = int(round((end - start) / width))
    if count < 2:
        return [run]
    cuts = []
    for i in range(1, count):
        expected = start + (end - start) * i // count
        lo, hi = max(start + 1, expected - width // 2), min(end - 1, expected + width // 2)
        cuts.append(lo + int(np.argmax(lightness[lo:hi + 1])))
    edges = [start] + cuts + [end]
    return list(zip(edges, edges[1:]))


def _shade_of(lab, paint, relative_width):
    """Whether a run is the paint in shade or ringing, not a band. relative_width: to the usual band."""
    hue = np.degrees(np.arctan2(lab[2], lab[1]) - np.arctan2(paint[2], paint[1]))
    if abs(lab[0] - paint[0]) >= SAME_PAINT_L or abs((hue + 180) % 360 - 180) >= SAME_PAINT_HUE_DEG:
        return False
    return relative_width < NARROW_RUN or np.hypot(lab[1], lab[2]) < STRONGER_COLOUR * np.hypot(paint[1], paint[2])


def find_bands(line):
    """The Bands along a centre line, and the body's RGB."""
    kernel = np.ones(5, np.float32) / 5
    smooth = np.stack([np.convolve(np.pad(line[:, c], 2, mode="edge"), kernel, "valid") for c in range(3)], 1)
    body = body_colour(smooth)
    lab = _lab(smooth)
    on = _distance(lab, _lab(body), FIND_LIGHTNESS_WEIGHT) > BAND_DIFFERENCE
    n = len(line)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= MIN_BAND * n and END_TRIM <= (a + b) / 2 / n <= 1 - END_TRIM]
    if not runs:
        return (), body
    width = max(int(np.median([b - a for a, b in runs])), 3)
    paint = _lab(body)
    runs = [r for r in runs if not _shade_of(np.median(lab[r[0]:r[1]], axis=0), paint, (r[1] - r[0]) / width)]
    if not runs:
        return (), body
    bands = []
    for run in runs:
        for a, b in _split(run, lab[:, 0], width):
            core = lab[a + (b - a) // 4:b - (b - a) // 4 or b]
            bands.append(Band((a + b) / 2 / n, (b - a) / n, np.median(core, axis=0)))
    return tuple(bands), body


def candidates(value_range=VALUE_RANGE_OHM):
    """Every (ohms, band colours) in scope: 4-band E24 with gold, 5-band E24 with brown."""
    out = []
    for significant in E24:
        a, b = divmod(significant, 10)
        for name, factor in MULTIPLIERS.items():
            if value_range[0] <= significant * factor <= value_range[1]:
                out.append((significant * factor, (DIGITS[a], DIGITS[b], name, "gold")))
            if value_range[0] <= significant * 10 * factor <= value_range[1]:
                out.append((significant * 10 * factor, (DIGITS[a], DIGITS[b], "black", name, "brown")))
    return out


def _fit(colours, bands, expected):
    """Cost of a value's bands against those found, in order: an alignment that may skip either.

    expected: {colour: Lab it should show on this body}.
    """
    n, m = len(colours), len(bands)
    want = [expected[c] for c in colours]
    miss = [MISSING_TOLERANCE if i == n - 1 else MISSING_BAND for i in range(n)]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i and cost[i - 1, j] + miss[i - 1] < cost[i, j]:
                cost[i, j] = cost[i - 1, j] + miss[i - 1]
            if j and cost[i, j - 1] + EXTRA_BAND < cost[i, j]:
                cost[i, j] = cost[i, j - 1] + EXTRA_BAND
            if i and j:
                pair = cost[i - 1, j - 1] + _distance(bands[j - 1].lab, want[i - 1])
                if pair < cost[i, j]:
                    cost[i, j] = pair
    total = float(cost[n, m])
    if m == n - 1:              # the tolerance band unseen: it should lie beyond the larger gap
        first, last = bands[0].position, bands[-1].position
        if min(first, 1 - first) > min(last, 1 - last):
            total += WRONG_END
    return total


def read(image, mask, white=255.0, body_shows=BODY_SHOWS, min_margin=MIN_MARGIN, bleed=BLEED):
    """The Reading of a straightened crop. image: BGR, white balanced; mask: the body; white: the
    plate's level in this image. Reading.best is the best guess even when Reading.value is None."""
    line = centre_line(image, mask, white)
    bands, body = find_bands(line)
    body_lab = _lab(body)
    if len(bands) < 2:
        return Reading(None, float("nan"), 0.0, (), False, bands, body_lab)
    backwards = bands[::-1]
    expected = {c: predicted(c, body, body_shows, bleed) for c in PAINT_LAB}
    scored = {}
    for ohms, colours in candidates():
        for flipped, found in ((False, bands), (True, backwards)):
            cost = _fit(colours, found, expected) / len(colours)
            if ohms not in scored or cost < scored[ohms][0]:
                scored[ohms] = (cost, colours, flipped)
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0])
    (best, (cost, colours, flipped)), (_, (second, _, _)) = ranked[0], ranked[1]
    margin = second - cost
    needed = max(min_margin, BLUE_MIN_MARGIN) if _is_blue(body_lab) else min_margin
    return Reading(best if margin >= needed else None, best, margin, colours, flipped, bands, body_lab)

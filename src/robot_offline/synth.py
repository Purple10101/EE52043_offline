"""Synthetic frames: resistors drawn onto a table, with exact body polygons.

Each frame is a white backdrop (procedural, or an empty-table frame from a capture session) with
resistors drawn on it: a shaded body with colour bands, tinned leads (straight, a few with one
gentle bend), soft shadows, and unlabelled clutter (lead offcuts, crumbs). Then whole-frame
lighting, blur and noise, saved as JPEG like the camera's MJPG stream. Cut-outs of real resistors
are not used yet; that comes once `cutouts` exists. Settings: [synthetic] in configs/dataset.toml.

Resistors, leads included, lie wholly inside the work area box from configs/dataset.toml, every
body keeps a minimum gap from the others, and no lead or clutter touches a body. Optionally a
share of bodies touch a neighbour (pushed into contact) or lie partly over one, in pairs only.
Labels follow docs/labelling.md: one polygon per body, leads excluded, a lower body clipped to its
visible part. Placements that would leave a body in two visible pieces, or with a hole, are
rejected, so every label is a single polygon.

Polygons use the pixel-centre convention (pixel row r, column c is the point (c, r)), the same
as cv2 contours and fillPoly.

    python -m robot_offline.synth --frames 20 --name synth_try
    python -m robot_offline.synth --backgrounds data/backgrounds/2026-10-14_a

Writes data/synthetic/<name>/NNNNN.jpg, outlined previews of the first few frames in preview/,
and that set's samples in data/manifest.json. Re-running with the same name replaces the set;
other sets stay until removed with `python -m robot_offline.clean_synth`.
"""

import argparse
import math
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from robot_offline import config, manifest, visualise

SS = 4                       # supersampling per axis when drawing bodies
OUTLINE_TOLERANCE_PX = 0.1   # simplification of the exact outlines before they are stored
MIN_VISIBLE_FRACTION = 0.4   # an overlapped body keeps at least this much of its area visible
MIN_LABEL_AREA_PX = 4.0      # smaller slivers at the frame edge are not labelled
MIN_PIECE_AREA_PX = 1.0      # geometry slivers below this are numerical noise
ATTEMPTS = 30                # placement tries per body before skipping it


@dataclass(frozen=True)
class Package:
    name: str
    length_mm: tuple[float, float]
    diameter_mm: tuple[float, float]  # across the end caps
    waist: float                      # diameter mid-body as a fraction of the end caps
    body_rgb: tuple[int, int, int]
    bands: int                        # 4: two digits; 5: three digits; then multiplier, tolerance
    tolerance: str
    lead_mm: float


PACKAGES = (
    Package("carbon_quarter_w", (5.8, 6.5), (2.2, 2.5), 0.82, (214, 188, 148), 4, "gold", 0.6),
    Package("carbon_half_w", (8.5, 9.5), (3.0, 3.4), 0.82, (214, 188, 148), 4, "gold", 0.7),
    Package("metal_quarter_w", (5.9, 6.5), (2.2, 2.5), 0.90, (84, 146, 205), 5, "brown", 0.6),
    Package("metal_half_w", (8.5, 9.5), (3.0, 3.4), 0.90, (84, 146, 205), 5, "brown", 0.7),
)

DIGITS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white")
BAND_RGB = {
    name: np.array(rgb, np.float32) / 255
    for name, rgb in {
        "black": (25, 25, 25), "brown": (110, 60, 30), "red": (200, 30, 30),
        "orange": (235, 120, 20), "yellow": (240, 210, 30), "green": (30, 140, 60),
        "blue": (40, 70, 190), "violet": (130, 60, 170), "grey": (130, 130, 130),
        "white": (235, 235, 235), "gold": (190, 150, 60), "silver": (190, 190, 195),
    }.items()
}
# Band centres as a fraction of the body length from the band-1 end, and band width.
BAND_POSITIONS = {4: (0.20, 0.33, 0.46, 0.80), 5: (0.17, 0.29, 0.41, 0.53, 0.82)}
BAND_WIDTH = 0.075
E24 = (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
       33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91)

LEAD_RGB = (0.72, 0.72, 0.70)                # tinned copper
LEAD_CLEARANCE_MM = 0.5                      # leads and clutter keep at least this off every body
BACKDROPS = {
    "white": (244, 244, 242), "warm_white": (242, 238, 230), "cool_white": (236, 240, 244),
    "off_white": (232, 230, 224), "grey_white": (224, 224, 222),
}
CRUMB_RGB = ((60, 40, 25), (30, 30, 30), (225, 220, 210), (200, 175, 135), (150, 150, 150))


@dataclass(frozen=True)
class Settings:
    width: int
    height: int
    work_area: tuple[int, int, int, int]   # x0, y0, x1, y1; resistors and leads stay inside
    px_per_mm: float
    bodies_per_frame: tuple[int, int]
    touching_fraction: float
    overlap_fraction: float
    min_gap_mm: float                      # between bodies that are not touching on purpose
    lead_length_mm: tuple[float, float]
    bent_lead_fraction: float
    max_bend_deg: float
    empty_fraction: float
    distractors_per_frame: tuple[int, int]

    @classmethod
    def from_config(cls, camera, dataset):
        area, syn = dataset["work_area"], dataset["synthetic"]
        width, height = camera["camera"]["width"], camera["camera"]["height"]
        x0, y0 = area["x0"], area["y0"]
        x1, y1 = x0 + area["width"], y0 + area["height"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise config.ConfigError(
                f"dataset.toml: work_area ({x0}, {y0}) to ({x1}, {y1}) does not fit the "
                f"{width} x {height} frame in camera.toml"
            )
        return cls(
            width=width,
            height=height,
            work_area=(x0, y0, x1, y1),
            px_per_mm=dataset["scale"]["px_per_mm"],
            bodies_per_frame=tuple(syn["bodies_per_frame"]),
            touching_fraction=syn["touching_fraction"],
            overlap_fraction=syn["overlap_fraction"],
            min_gap_mm=syn["min_gap_mm"],
            lead_length_mm=tuple(syn["lead_length_mm"]),
            bent_lead_fraction=syn["bent_lead_fraction"],
            max_bend_deg=syn["max_bend_deg"],
            empty_fraction=syn["empty_fraction"],
            distractors_per_frame=tuple(syn["distractors_per_frame"]),
        )


@dataclass(frozen=True)
class Look:
    """Lighting for one frame. Vectors are x right, y down, z towards the camera."""
    light: np.ndarray          # unit vector towards the light
    half: np.ndarray           # unit half-vector between light and view, for highlights
    ambient: float
    diffuse: float
    flat: float                # brightness of the table; colours are relative to it
    shininess: float
    shadow_offset: np.ndarray  # px
    shadow_sigma: float        # px
    shadow_strength: float


@dataclass
class Resistor:
    package: Package
    length: float              # px
    radius: float              # px, half the end-cap diameter
    yaw: float                 # rad, direction of the body axis in image coordinates
    centre: np.ndarray         # px, (x, y)
    bands: list[str]
    flipped: bool              # band 1 at the +u end instead of the -u end
    body_rgb: np.ndarray
    gloss: float
    leads: list = field(default_factory=list)  # (points N×2, width px, rgb)

    @property
    def end(self):
        """Fraction of the half-length taken by the rounded end."""
        return min(0.5 * self.radius / (self.length / 2), 0.3)

    def outline(self):
        local = _body_outline(self.length, self.radius, self.package.waist, self.end)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.centre + local[:, :1] * (c, s) + local[:, 1:] * (-s, c)

    def polygon(self):
        return Polygon(self.outline())


@dataclass
class Frame:
    image: np.ndarray          # H×W×3 uint8 RGB
    polygons: list             # visible body outlines, N×2, full-frame px
    ids: list                  # each polygon's value in instance_map
    instance_map: np.ndarray   # H×W uint16: 0 background, i+1 where body i is on top


# --- Body geometry -----------------------------------------------------------------------------

def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _profile(t, waist, end):
    """Body half-width over the cap half-width at t = |u| / half-length (0 middle, 1 tip)."""
    r = waist + (1 - waist) * _smoothstep(0.40, 0.62, t)
    k = np.clip((t - (1 - end)) / end, 0.0, 1.0)
    return r * np.sqrt(1 - k * k)


def _body_outline(length, radius, waist, end, n=40):
    """Closed outline in the body's frame (u along the length, v across), N×2."""
    t = np.concatenate([
        np.linspace(0.0, 1 - end, n, endpoint=False),
        (1 - end) + end * np.sin(np.linspace(0.0, np.pi / 2, n // 3)),  # dense around the tip
    ])
    r = radius * _profile(t, waist, end)
    u = np.concatenate([-t[::-1], t[1:]]) * length / 2
    v = np.concatenate([r[::-1], r[1:]])
    upper = np.stack([u, v], axis=1)
    lower = np.stack([u[::-1], -v[::-1]], axis=1)[1:-1]
    return np.vstack([upper, lower])


def _rot(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _pieces(geom):
    """The polygons in a shapely result, ignoring slivers."""
    parts = getattr(geom, "geoms", [geom])
    return [g for g in parts if g.geom_type == "Polygon" and g.area >= MIN_PIECE_AREA_PX]


def _one_piece(geom):
    """The geometry as a single hole-free polygon, or None if it is in pieces or has a hole."""
    parts = _pieces(geom)
    if len(parts) != 1:
        return None
    holes = [Polygon(ring) for ring in parts[0].interiors]
    if any(h.area >= MIN_PIECE_AREA_PX for h in holes):
        return None
    return parts[0]


# --- Placement ---------------------------------------------------------------------------------

def _bands(package, rng):
    value = E24[rng.integers(len(E24))]
    digits = [value // 10, value % 10] + ([0] if package.bands == 5 else [])
    multiplier = "gold" if rng.random() < 0.1 else DIGITS[rng.integers(0, 6)]
    return [DIGITS[d] for d in digits] + [multiplier, package.tolerance]


def _sample_resistor(st, rng):
    pkg = PACKAGES[rng.integers(len(PACKAGES))]
    x0, y0, x1, y1 = st.work_area
    # Brightness varies more than hue: band colours sit against this body colour.
    rgb = np.array(pkg.body_rgb, np.float32) / 255 * rng.uniform(0.85, 1.1) * rng.uniform(0.96, 1.04, 3)
    return Resistor(
        package=pkg,
        length=rng.uniform(*pkg.length_mm) * st.px_per_mm,
        radius=rng.uniform(*pkg.diameter_mm) * st.px_per_mm / 2,
        yaw=rng.uniform(0, math.pi),
        centre=np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]),
        bands=_bands(pkg, rng),
        flipped=bool(rng.random() < 0.5),
        body_rgb=np.clip(rgb, 0, 1).astype(np.float32),
        gloss=rng.uniform(0.08, 0.3),
    )


def _contact(anchor, res, base, direction):
    """Smallest s for which res centred at base + s * direction just clears anchor (bisection).

    None if res does not overlap anchor at s = 0.
    """
    reach = (anchor.length + res.length) / 2 + 1
    anchor_poly = anchor.polygon()

    def hits(s):
        res.centre = base + direction * s
        return res.polygon().intersects(anchor_poly)

    if not hits(0.0) or hits(reach):
        return None
    lo, hi = 0.0, reach
    for _ in range(24):
        mid = (lo + hi) / 2
        if hits(mid):
            lo = mid
        else:
            hi = mid
    return hi


def _candidate(st, rng, placed, single):
    """A new resistor, its placement mode and its anchor's index.

    Modes: free (anchor None), touch (against the anchor) or overlap (partly over the anchor).
    Anchors come from single, the indices of placed bodies not yet in a pair.
    """
    res = _sample_resistor(st, rng)
    mode = "free"
    if single:
        r = rng.random()
        if r < st.overlap_fraction:
            mode = "overlap"
        elif r < st.overlap_fraction + st.touching_fraction:
            mode = "touch"
    if mode == "free":
        return res, mode, None

    index = single[rng.integers(len(single))]
    anchor = placed[index]
    axis = np.array([math.cos(anchor.yaw), math.sin(anchor.yaw)])
    normal = np.array([-axis[1], axis[0]])
    side = rng.choice([-1.0, 1.0])
    kind = rng.choice(3, p=(0.5, 0.25, 0.25))
    if kind == 0:    # side by side
        res.yaw = anchor.yaw + rng.normal(0, math.radians(4))
        direction = normal * side
        base = anchor.centre + axis * rng.uniform(-0.4, 0.4) * anchor.length
    elif kind == 1:  # end to end
        res.yaw = anchor.yaw + rng.normal(0, math.radians(8))
        direction = axis * side
        base = anchor.centre + normal * rng.uniform(-0.5, 0.5) * anchor.radius
    else:            # any angle
        a = rng.uniform(0, 2 * math.pi)
        direction = np.array([math.cos(a), math.sin(a)])
        base = anchor.centre.copy()

    s = _contact(anchor, res, base, direction)
    if s is None:
        return None
    if mode == "overlap":
        s *= rng.uniform(0.5, 0.85)
    res.centre = base + direction * s
    return res, mode, index


def _inside(pts, area, margin=0.0):
    x0, y0, x1, y1 = area
    return bool(
        (pts[:, 0] >= x0 + margin).all() and (pts[:, 0] <= x1 - 1 - margin).all()
        and (pts[:, 1] >= y0 + margin).all() and (pts[:, 1] <= y1 - 1 - margin).all()
    )


def _fits(res, mode, anchor, polys, min_gap):
    """Whether res can go on top of the bodies already placed (polys, bottom first).

    res keeps min_gap px from every body except its anchor, so groups are pairs at most. Over the
    anchor, it must leave the anchor's visible part in one piece and big enough.
    """
    poly = res.polygon()
    if any(poly.distance(p) < min_gap for i, p in enumerate(polys) if i != anchor):
        return False
    if mode == "touch":
        return poly.intersection(polys[anchor]).area <= 0.5
    if mode == "overlap":
        p = polys[anchor]
        visible = _one_piece(p.difference(poly))
        return visible is not None and visible.area >= MIN_VISIBLE_FRACTION * p.area
    return True


def _place_bodies(st, rng):
    """Resistors with leads, in drawing order (bottom first), and their body polygons.

    Every resistor, leads included, lies wholly inside the work area, and no lead crosses a body.
    Bodies touching or overlapping on purpose come in pairs; everything else keeps the minimum gap.
    """
    lo, hi = st.bodies_per_frame
    n = 0 if rng.random() < st.empty_fraction else int(rng.integers(lo, hi + 1))
    min_gap = st.min_gap_mm * st.px_per_mm
    bodies, polys, wires, single = [], [], [], []
    for _ in range(n):
        for _ in range(ATTEMPTS):
            cand = _candidate(st, rng, bodies, single)
            if cand is None or not _inside(cand[0].outline(), st.work_area):
                continue
            res, mode, anchor = cand
            if not _fits(res, mode, anchor, polys, min_gap):
                continue
            if any(res.polygon().intersects(w) for w in wires):
                continue
            wire = _fit_leads(res, rng, st, polys)
            if wire is None:
                continue
            if anchor is None:
                single.append(len(bodies))
            else:
                single.remove(anchor)
            bodies.append(res)
            polys.append(res.polygon())
            wires.append(wire)
            break
    return bodies, polys


def _wire_shape(lines, clearance):
    """The area covered by wires, grown by clearance px, for keeping them off bodies."""
    return unary_union([LineString(p).buffer(w / 2 + clearance) for p, w, _ in lines])


def _fit_leads(res, rng, st, polys, tries=5):
    """Gives res leads inside the work area and clear of every body in polys.

    Returns the leads' shape for later bodies to keep clear of, or None if no try fits.
    """
    clearance = LEAD_CLEARANCE_MM * st.px_per_mm
    for _ in range(tries):
        res.leads = _leads(res, rng, st)
        if not all(_inside(p, st.work_area, w / 2) for p, w, _ in res.leads):
            continue
        wire = _wire_shape(res.leads, clearance)
        if not any(wire.intersects(p) for p in polys):
            return wire
    return None


def _leads(res, rng, st):
    """Two leads, straight out of the end caps, sometimes with one gentle bend.

    Each is (points N×2, width px, rgb).
    """
    s = st.px_per_mm
    width = res.package.lead_mm * s
    rgb = np.array(LEAD_RGB, np.float32) * rng.uniform(0.85, 1.1)
    leads = []
    for sign in (-1.0, 1.0):
        axis = sign * np.array([math.cos(res.yaw), math.sin(res.yaw)])
        start = res.centre + axis * (res.length / 2 - 0.4 * res.radius)  # hidden inside the cap
        length = rng.uniform(*st.lead_length_mm) * s
        if rng.random() < st.bent_lead_fraction:
            straight = rng.uniform(0.2, 0.6) * length
            bend = start + axis * straight
            turn = rng.uniform(-1, 1) * math.radians(st.max_bend_deg)
            pts = [start, bend, bend + _rot(axis, turn) * (length - straight)]
        else:
            pts = [start, start + axis * length]
        leads.append((np.array(pts), width, rgb))
    return leads


def _distractors(st, rng, polys):
    """Unlabelled clutter clear of the bodies: lead offcuts and crumbs, as kwargs for _draw."""
    s = st.px_per_mm
    x0, y0, x1, y1 = st.work_area
    lo, hi = st.distractors_per_frame
    bodies = unary_union(polys)
    clearance = LEAD_CLEARANCE_MM * s
    out = []
    for _ in range(int(rng.integers(lo, hi + 1))):
        for _ in range(10):
            p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if rng.random() < 0.6:
                a = rng.uniform(0, 2 * math.pi)
                pts = [p]
                for _ in range(int(rng.integers(1, 3))):
                    a += rng.normal(0, 0.6)
                    step = np.array([math.cos(a), math.sin(a)]) * rng.uniform(2, 12) * s
                    pts.append(pts[-1] + step)
                rgb = np.array(LEAD_RGB, np.float32) * rng.uniform(0.85, 1.1)
                item = {"lines": [(np.array(pts), 0.6 * s, rgb)]}
                shape = _wire_shape(item["lines"], clearance)
            else:
                k = int(rng.integers(7, 13))
                angles = np.sort(rng.uniform(0, 2 * math.pi, k))
                radii = rng.uniform(0.3, 1.2) * s * rng.uniform(0.6, 1.0, k)
                pts = p + np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
                rgb = np.array(CRUMB_RGB[rng.integers(len(CRUMB_RGB))], np.float32) / 255
                item = {"blobs": [(pts, np.clip(rgb * rng.uniform(0.8, 1.2), 0, 1))]}
                shape = Polygon(pts).buffer(clearance)
            if not shape.intersects(bodies):
                out.append(item)
                break
    return out


# --- Drawing -----------------------------------------------------------------------------------

def _look(st, rng):
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(45, 85))
    light = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    half = light + (0, 0, 1)
    ambient = rng.uniform(0.35, 0.7)
    diffuse = 1 - ambient
    height = 1.2 * st.px_per_mm  # roughly how far a body stands off the table
    return Look(
        light=light,
        half=half / np.linalg.norm(half),
        ambient=ambient,
        diffuse=diffuse,
        flat=ambient + diffuse * light[2],
        shininess=rng.uniform(12, 40),
        shadow_offset=-light[:2] / np.linalg.norm(light[:2]) * height / math.tan(el),
        shadow_sigma=rng.uniform(0.3, 1.2) * st.px_per_mm,
        shadow_strength=rng.uniform(0.15, 0.5),
    )


def _stroke(pts, width, shape, closed=False, fill=False):
    """Anti-aliased polyline (or filled polygon) as alpha in 0..1."""
    canvas = np.zeros(shape, np.uint8)
    fixed = [np.round(pts * 16).astype(np.int32)]
    if fill:
        cv2.fillPoly(canvas, fixed, 255, cv2.LINE_AA, shift=4)
    else:
        cv2.polylines(canvas, fixed, closed, 255, max(1, round(width)), cv2.LINE_AA, shift=4)
    return canvas.astype(np.float32) / 255


def _line_layer(lines, origin, shape):
    """Wires as alpha and premultiplied colour, with a highlight stripe along each."""
    alpha = np.zeros(shape, np.float32)
    premul = np.zeros(shape + (3,), np.float32)
    for pts, width, rgb in lines:
        a = _stroke(pts - origin, width, shape)
        stripe = _stroke(pts - origin, width / 3, shape)
        colour = rgb + 0.3 * stripe[..., None]
        premul = premul * (1 - a)[..., None] + colour * a[..., None]
        alpha = alpha * (1 - a) + a
    return alpha, premul


def _blob_layer(blobs, origin, shape):
    alpha = np.zeros(shape, np.float32)
    premul = np.zeros(shape + (3,), np.float32)
    for pts, rgb in blobs:
        a = _stroke(pts - origin, 0, shape, fill=True)
        premul = premul * (1 - a)[..., None] + rgb * a[..., None]
        alpha = alpha * (1 - a) + a
    return alpha, premul


def _body_layer(res, look, rect):
    """The body supersampled over rect (x0, y0, x1, y1): alpha and premultiplied colour."""
    x0, y0, x1, y1 = rect
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    premul = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    pts = res.outline()
    bx0 = max(x0, math.floor(pts[:, 0].min()) - 1)
    bx1 = min(x1, math.ceil(pts[:, 0].max()) + 2)
    by0 = max(y0, math.floor(pts[:, 1].min()) - 1)
    by1 = min(y1, math.ceil(pts[:, 1].max()) + 2)
    if bx0 >= bx1 or by0 >= by1:
        return alpha, premul
    bh, bw = by1 - by0, bx1 - bx0

    offsets = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (np.arange(bx0, bx1)[:, None] + offsets).ravel() - res.centre[0]
    ys = (np.arange(by0, by1)[:, None] + offsets).ravel() - res.centre[1]
    X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(res.yaw), math.sin(res.yaw)
    u = X * c + Y * s
    v = -X * s + Y * c

    half, end = res.length / 2, res.end
    t = np.abs(u) / half
    r = res.radius * _profile(np.minimum(t, 1), res.package.waist, end)
    inside = (t <= 1) & (np.abs(v) < r)

    # Surface normal of a cylinder with rounded ends, then Lambert plus a highlight.
    k = np.clip((t - (1 - end)) / end, 0, 1)
    n = np.clip(v / np.maximum(r, 1e-6), -1, 1)
    q = np.sqrt(1 - k * k)
    nu, nv, nz = np.sign(u) * k, n * q, np.sqrt(1 - n * n) * q

    def local(vec):
        return vec[0] * c + vec[1] * s, -vec[0] * s + vec[1] * c, vec[2]

    lu, lv, lz = local(look.light)
    hu, hv, hz = local(look.half)
    ndl = np.clip(nu * lu + nv * lv + nz * lz, 0, None)
    ndh = np.clip(nu * hu + nv * hv + nz * hz, 0, None)

    pos = (u + half) / res.length
    if res.flipped:
        pos = 1 - pos
    albedo = np.broadcast_to(res.body_rgb, u.shape + (3,)).copy()
    gloss = np.full(u.shape, res.gloss)
    for p, name in zip(BAND_POSITIONS[len(res.bands)], res.bands):
        m = np.abs(pos - p) < BAND_WIDTH / 2
        albedo[m] = BAND_RGB[name]
        if name in ("gold", "silver"):
            gloss[m] = 0.6

    shade = (look.ambient + look.diffuse * ndl) / look.flat
    colour = albedo * shade[..., None] + (gloss * ndh ** look.shininess)[..., None]
    colour *= inside[..., None]
    sub = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
    alpha[sub] = inside.reshape(bh, SS, bw, SS).mean(axis=(1, 3))
    premul[sub] = colour.reshape(bh, SS, bw, SS, 3).mean(axis=(1, 3))
    return alpha, premul


def _draw(img, look, lines=(), body=None, blobs=()):
    """Draws one object (blobs, then wires, then body) and its shadow onto img, in place.

    Returns (x0, y0, body alpha) over the patch drawn, or None if the object is off the frame.
    """
    height, width = img.shape[:2]
    groups = [p for p, *_ in lines] + [p for p, _ in blobs]
    if body is not None:
        groups.append(body.outline())
    pts = np.vstack(groups)
    widest = max((w for _, w, _ in lines), default=0)
    margin = widest / 2 + np.abs(look.shadow_offset).max() + 3 * look.shadow_sigma + 2
    x0 = max(0, int(pts[:, 0].min() - margin))
    x1 = min(width, int(pts[:, 0].max() + margin) + 1)
    y0 = max(0, int(pts[:, 1].min() - margin))
    y1 = min(height, int(pts[:, 1].max() + margin) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    shape = (y1 - y0, x1 - x0)
    origin = np.array([x0, y0], float)

    layers = []
    if blobs:
        layers.append(_blob_layer(blobs, origin, shape))
    if lines:
        layers.append(_line_layer(lines, origin, shape))
    body_alpha = None
    if body is not None:
        layers.append(_body_layer(body, look, (x0, y0, x1, y1)))
        body_alpha = layers[-1][0]

    silhouette = np.maximum.reduce([a for a, _ in layers])
    shift = np.float32([[1, 0, look.shadow_offset[0]], [0, 1, look.shadow_offset[1]]])
    shadow = cv2.warpAffine(silhouette, shift, (shape[1], shape[0]))
    shadow = cv2.GaussianBlur(shadow, (0, 0), look.shadow_sigma)
    patch = img[y0:y1, x0:x1]
    patch *= (1 - look.shadow_strength * shadow)[..., None]
    for a, premul in layers:
        patch *= (1 - a)[..., None]
        patch += premul
    return x0, y0, body_alpha


def _noise(rng, shape, step):
    """Smooth noise with features about step px across, roughly unit variance."""
    h, w = shape
    small = rng.standard_normal((int(h / step) + 2, int(w / step) + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _backdrop(st, rng):
    names = list(BACKDROPS)
    base = np.array(BACKDROPS[names[rng.integers(len(names))]], np.float32) / 255
    base *= np.float32(rng.uniform(0.95, 1.02)) * rng.uniform(0.99, 1.01, 3).astype(np.float32)
    shape = (st.height, st.width)
    texture = 0.025 * _noise(rng, shape, st.height / 4) + 0.01 * _noise(rng, shape, 3.0)
    return np.clip(base * (1 + texture)[..., None], 0, 1)


def _camera(img, rng):
    """Uneven lighting, exposure, white balance, focus blur and sensor noise; returns uint8."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    field = (1 - rng.uniform(0, 0.08) * r2.astype(np.float32) / 2)
    field *= 1 + rng.uniform(0, 0.03) * _noise(rng, (h, w), h / 2)
    img *= field[..., None]
    img *= (rng.uniform(0.94, 1.03) * rng.uniform(0.985, 1.015, 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.9))
    img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(rng.uniform(0.004, 0.012))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _labels(polys, st):
    """Visible part of each body, clipped to the frame: (polygons, instance ids)."""
    frame = box(-0.5, -0.5, st.width - 0.5, st.height - 0.5)
    polygons, ids = [], []
    for i, p in enumerate(polys):
        above = [q for q in polys[i + 1:] if q.intersects(p)]
        visible = p.difference(unary_union(above)) if above else p
        parts = _pieces(visible.intersection(frame))
        if not parts:
            continue
        part = max(parts, key=lambda g: g.area)
        if part.area < MIN_LABEL_AREA_PX:
            continue
        part = part.simplify(OUTLINE_TOLERANCE_PX)
        polygons.append(np.asarray(part.exterior.coords)[:-1])
        ids.append(i + 1)
    return polygons, ids


def render_frame(st, rng, background=None):
    """One synthetic frame. background: H×W×3 uint8 RGB empty-table frame, or None."""
    look = _look(st, rng)
    bodies, polys = _place_bodies(st, rng)

    if background is None:
        img = _backdrop(st, rng)
    else:
        if background.shape != (st.height, st.width, 3):
            raise ValueError(f"background is {background.shape}, frame is {(st.height, st.width, 3)}")
        img = background.astype(np.float32) / 255

    for clutter in _distractors(st, rng, polys):
        _draw(img, look, **clutter)
    instance_map = np.zeros((st.height, st.width), np.uint16)
    for i, body in enumerate(bodies, start=1):
        drawn = _draw(img, look, body.leads, body)
        if drawn is not None:
            x0, y0, a = drawn
            instance_map[y0:y0 + a.shape[0], x0:x0 + a.shape[1]][a > 0.5] = i

    polygons, ids = _labels(polys, st)
    return Frame(_camera(img, rng), polygons, ids, instance_map)


# --- Writing a set -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _Job:
    index: int
    seed: int
    settings: Settings
    out_dir: Path
    name: str
    backgrounds: tuple
    jpeg_quality: tuple
    preview: bool


def _read_rgb(path):
    bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_jpg(path, bgr, quality):
    # imencode + write_bytes, because cv2.imwrite cannot handle non-ASCII paths on Windows.
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for {path}")
    path.write_bytes(buf.tobytes())


def _make_frame(job):
    rng = np.random.default_rng([job.seed, job.index])
    background = None
    if job.backgrounds:
        background = _read_rgb(job.backgrounds[rng.integers(len(job.backgrounds))])
    frame = render_frame(job.settings, rng, background)

    file = f"{job.index:05d}.jpg"
    bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    _write_jpg(job.out_dir / file, bgr, rng.integers(job.jpeg_quality[0], job.jpeg_quality[1] + 1))
    if job.preview:
        outlined = visualise.draw_polygons(bgr, frame.polygons)
        small = cv2.resize(outlined, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _write_jpg(job.out_dir / "preview" / file, small, 90)

    return {
        "image": f"synthetic/{job.name}/{file}",
        "session": job.name,
        "source": "synthetic",
        "width": job.settings.width,
        "height": job.settings.height,
        "instances": [
            {"polygon": manifest.polygon_to_json(p), "label_source": "synthetic"}
            for p in frame.polygons
        ],
    }


def main(argv=None):
    camera_cfg, dataset_cfg = config.load("camera"), config.load("dataset")
    syn = dataset_cfg["synthetic"]

    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.synth",
        description="Generate synthetic resistor frames with exact body polygons.",
    )
    parser.add_argument("--frames", type=int, default=syn["frames"])
    parser.add_argument("--name", default="synth",
                        help="set name: frames go to data/synthetic/<name>/, also the session name")
    parser.add_argument("--seed", type=int, default=syn["seed"])
    parser.add_argument("--backgrounds", type=Path,
                        help="folder of empty-table frames to draw on instead of procedural backdrops")
    parser.add_argument("--workers", type=int, default=4, help="processes; 1 runs in this process")
    parser.add_argument("--preview", type=int, default=5,
                        help="write outlined previews of the first N frames to preview/")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name) or set(args.name) == {"."}:
        parser.error(f"--name {args.name!r}: use letters, digits, '_', '-' and '.' only")
    settings = Settings.from_config(camera_cfg, dataset_cfg)

    backgrounds = ()
    if args.backgrounds is not None:
        backgrounds = tuple(sorted(
            p for p in args.backgrounds.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ))
        if not backgrounds:
            parser.error(f"no .png or .jpg images in {args.backgrounds}")

    # Check the manifest before spending time on frames.
    labels = manifest.load(dataset_cfg["labels"]["rules_version"])

    out_dir = config.DATA_DIR / "synthetic" / args.name
    if out_dir.exists():
        print(f"Replacing the existing set {args.name}")
        shutil.rmtree(out_dir)
    (out_dir / "preview").mkdir(parents=True)

    jobs = [
        _Job(i, args.seed, settings, out_dir, args.name, backgrounds,
             tuple(syn["jpeg_quality"]), i < args.preview)
        for i in range(args.frames)
    ]
    print(f"Writing {args.frames} frames ({settings.width} x {settings.height}) to {out_dir}")
    samples, t0 = [], time.perf_counter()
    if args.workers > 1:
        pool = ProcessPoolExecutor(args.workers)
        results = pool.map(_make_frame, jobs)
    else:
        pool, results = None, map(_make_frame, jobs)
    try:
        for i, sample in enumerate(results, start=1):
            samples.append(sample)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} frames  {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    manifest.replace_samples(labels, f"synthetic/{args.name}/", samples)
    manifest.save(labels)
    bodies = sum(len(s["instances"]) for s in samples)
    print(f"Done: {len(samples)} frames, {bodies} bodies, manifest updated ({manifest.PATH})")


if __name__ == "__main__":
    main()

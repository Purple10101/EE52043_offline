"""Synthetic top-down frames of the pick plate, with exact body polygons.

Frames are drawn directly in the rectified space of `rectify`: the pick plate seen from straight
above at [rectify] px_per_mm, with its margin (1200 × 1200 px with the defaults), like real frames
after `rectify apply`. Real frames are never used here; their look was measured and copied:

- The plate: light neutral grey with rounded corners, a gentle lighting gradient, faint pencil
  marks and smudges, and around it the robot base (blue), the table (wood) and the plate's own
  front face and legs on the near side, each side's surround chosen at random.
- Resistors: mostly turquoise metal film, some blue metal film and beige carbon, with colour bands;
  long tinned leads that curve gently, sometimes kink near the body, and may cross each other.
- The camera's view: the camera looks at the plate from one side ([synthetic] camera_xy_mm and
  camera_height_mm), so a body, standing a radius or so off the plate, appears wider and shifted
  away from the camera. Bodies are drawn and labelled as they appear.
- Clutter, never labelled: lead offcuts and crumbs, and in some frames screws and wood shavings,
  sometimes with no resistors at all.
- The camera: faint soft shadows, focus blur that grows towards the far side of the plate, low
  noise, saved as JPEG.

Resistors, leads included, lie wholly inside the work area (the plate), every body keeps a minimum
gap from the others, and no lead or clutter touches a body. Optionally a share of bodies touch a
neighbour (pushed into contact) or lie partly over one, in pairs only, and a share are laid
parallel close beside another. Labels follow docs/labelling.md: one polygon per body, leads
excluded, a lower body clipped to its visible part. Placements that would leave a body in two
visible pieces, or with a hole, are rejected, so every label is a single polygon.

Polygons use the pixel-centre convention (pixel row r, column c is the point (c, r)), the same
as cv2 contours and fillPoly.

    python -m robot_offline.synth --frames 20 --name synth_try
    python -m robot_offline.synth --backgrounds <folder of rectified empty-plate images>

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
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from robot_offline import config, manifest, rectify, visualise

SS = 4                       # supersampling per axis when drawing bodies
OUTLINE_TOLERANCE_PX = 0.1   # simplification of the exact outlines before they are stored
MIN_VISIBLE_FRACTION = 0.4   # an overlapped body keeps at least this much of its area visible
MIN_LABEL_AREA_PX = 4.0      # smaller slivers at the frame edge are not labelled
MIN_PIECE_AREA_PX = 1.0      # geometry slivers below this are numerical noise
ATTEMPTS = 30                # placement tries per body before skipping it
LEAD_TRIES = 8               # lead shapes tried per body before giving up on it


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
    weight: float                     # how often it is drawn


# Turquoise metal film is what the rig sorts most; the others keep the model general. Quarter-watt sizes are
# the measured real bodies (6.3 to 7.0 mm long, 2.4 to 2.7 mm across the caps); half-watt is rare (2%).
PACKAGES = (
    Package("metal_teal_quarter_w", (6.3, 7.0), (2.4, 2.7), 0.86, (38, 161, 146), 5, "brown", 0.6, 0.60),
    Package("metal_teal_half_w", (8.5, 9.5), (3.0, 3.4), 0.88, (38, 161, 146), 5, "brown", 0.65, 0.01),
    Package("metal_blue_quarter_w", (6.3, 7.0), (2.4, 2.7), 0.90, (84, 146, 205), 5, "brown", 0.6, 0.17),
    Package("metal_blue_half_w", (8.5, 9.5), (3.0, 3.4), 0.90, (84, 146, 205), 5, "brown", 0.65, 0.005),
    Package("carbon_quarter_w", (6.3, 7.0), (2.4, 2.7), 0.82, (214, 188, 148), 4, "gold", 0.6, 0.21),
    Package("carbon_half_w", (8.5, 9.5), (3.0, 3.4), 0.82, (214, 188, 148), 4, "gold", 0.65, 0.005),
)
PACKAGE_P = np.array([p.weight for p in PACKAGES]) / sum(p.weight for p in PACKAGES)

DIGITS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white")
BAND_RGB = {
    name: np.array(rgb, np.float32) / 255
    for name, rgb in {
        "black": (48, 44, 42), "brown": (138, 82, 46), "red": (208, 48, 54),
        "orange": (232, 128, 46), "yellow": (238, 212, 72), "green": (62, 146, 82),
        "blue": (62, 98, 196), "violet": (142, 92, 186), "grey": (140, 140, 140),
        "white": (226, 226, 222), "gold": (188, 158, 86), "silver": (190, 190, 195),
    }.items()
}
# Band centres as a fraction of the body length from the band-1 end, and band width.
BAND_POSITIONS = {4: (0.27, 0.40, 0.53, 0.75), 5: (0.24, 0.36, 0.48, 0.60, 0.77)}
BAND_WIDTH = 0.052
E24 = (10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
       33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91)

LEAD_RGB = (0.51, 0.52, 0.52)                # tinned copper, as bright as on the real plate
LEAD_CLEARANCE_MM = 0.5                      # leads and clutter keep at least this off every body
PLATE_RGB = (0.777, 0.788, 0.788)            # the plate after `rectify` white balance (about 200)
CRUMB_RGB = ((60, 40, 25), (30, 30, 30), (225, 220, 210), (200, 175, 135), (150, 150, 150))
ZINC_RGB = (0.64, 0.68, 0.74)                # zinc-plated screws, slightly blue
WOOD_RGB = (0.86, 0.73, 0.50)                # shavings
# What lies around the plate. Sides are in the rectified image: top is the robot-base side, left the
# near side, right the far side. Part of the time a side shows what the real rig has there; the rest
# of the time any surface, so the model learns nothing from the surround.
REAL_SURROUNDS = {"top": "plastic", "left": "dark", "right": "wood", "bottom": "wood"}
SURROUND_KINDS = ("wood", "plastic", "dark", "mat", "metal", "cardboard", "fabric", "gradient")
REAL_SURROUND_FRACTION = 0.4
SPLIT_FRACTION = 0.3          # chance a side shows two surfaces, divided by a straight line
FACE_FRACTION = 0.85          # chance the plate's own front face and legs show on the near side
EPOXY_RGB = (0.085, 0.085, 0.095)           # black moulded packages
TIN_RGB = (0.60, 0.61, 0.62)                 # their tinned legs
TAB_RGB = (0.70, 0.71, 0.72)                 # a TO-220's metal tab
PART_NUMBERS = {                              # printed on the packages, faintly
    "to220": ("IRF540N", "IRFZ44N", "IRF520N", "IRLZ44N", "P55NF06", "IRF9540N"),
    "to92": ("2N7000", "BS170", "BS250", "VN2222"),
    "dip8": ("LM358P", "NE5532P", "TL072CP", "LM741CN", "OP07CP", "TL082CP"),
}
CABLE_RGB = ((0.05, 0.05, 0.06), (0.75, 0.1, 0.08), (0.85, 0.85, 0.82), (0.15, 0.25, 0.7), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class Settings:
    width: int
    height: int
    work_area: tuple[int, int, int, int]   # x0, y0, x1, y1; resistors and leads stay inside
    plate: tuple[int, int, int, int]       # x0, y0, x1, y1 of the plate's top surface, px
    plate_radius: float                    # px, its rounded corners
    px_per_mm: float
    camera: tuple[float, float]            # px, the point on the plate's plane below the lens
    camera_height: float                   # px, lens above the plate
    bodies_per_frame: tuple[int, int]
    touching_fraction: float
    overlap_fraction: float
    cluster_fraction: float                # chance a body is laid parallel close beside another
    min_gap_mm: float                      # between bodies that are not touching on purpose
    lead_length_mm: tuple[float, float]
    lead_curl_deg: float                   # typical sweep of a lead along its whole length
    lead_bends: float                      # hand bends per lead, on average
    kink_fraction: float                   # share of leads with a sharp bend near the body
    empty_fraction: float
    distractors_per_frame: tuple[int, int]  # small clutter: offcuts, crumbs, marks
    objects_fraction: float                # chance a frame has screws or shavings
    objects_per_frame: tuple[int, int]
    # The camera simulation (_through_camera). Without camera_corners, frames get the simpler
    # rectified-space camera effects of _camera instead.
    camera_corners: tuple | None = None     # plate corners in the raw camera frame, image order
    plate_corners_rect: tuple | None = None  # the same corners in rectified px
    render_scale: float = 1.0              # frames are drawn this much finer, then go through the camera
    defocus: float = 0.0                   # blur sigma in raw px per unit of |distance - focus| / distance
    focus_shift: float = 0.0               # focus distance minus the distance to the plate centre, px
    mjpg_quality: tuple[int, int] = (85, 92)
    sharpen: tuple[float, float] = (0.5, 1.0)   # the camera's sharpening, unsharp-mask amount

    def scaled(self, k):
        """The same frame drawn k times finer: every px quantity times k (pixel-centre convention)."""
        if k == 1:
            return self
        edge = lambda box_: tuple(int(round(v * k)) for v in box_)          # pixel edges
        centre = lambda p: tuple((v + 0.5) * k - 0.5 for v in p)            # pixel centres
        return replace(self, width=int(round(self.width * k)), height=int(round(self.height * k)),
                       work_area=edge(self.work_area), plate=edge(self.plate),
                       plate_radius=self.plate_radius * k, px_per_mm=self.px_per_mm * k,
                       camera=centre(self.camera), camera_height=self.camera_height * k)

    @classmethod
    def from_config(cls, dataset):
        area, syn = dataset["work_area"], dataset["synthetic"]
        rect = rectify.Settings.from_config(dataset)
        width, height = rect.size
        x0, y0 = area["x0"], area["y0"]
        x1, y1 = x0 + area["width"], y0 + area["height"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise config.ConfigError(
                f"dataset.toml: work_area ({x0}, {y0}) to ({x1}, {y1}) does not fit the "
                f"{width} x {height} rectified frame"
            )
        s = rect.px_per_mm
        corners = rectify.mm_to_rect(rect.plate_corners_mm, rect) + 0.5   # pixel edges
        cam = rectify.mm_to_rect([syn["camera_xy_mm"]], rect)[0]
        return cls(
            width=width,
            height=height,
            work_area=(x0, y0, x1, y1),
            plate=tuple(int(round(v)) for v in (*corners.min(0), *corners.max(0))),
            plate_radius=rect.corner_radius_mm * s,
            px_per_mm=s,
            camera=(float(cam[0]), float(cam[1])),
            camera_height=syn["camera_height_mm"] * s,
            bodies_per_frame=tuple(syn["bodies_per_frame"]),
            touching_fraction=syn["touching_fraction"],
            overlap_fraction=syn["overlap_fraction"],
            cluster_fraction=syn["cluster_fraction"],
            min_gap_mm=syn["min_gap_mm"],
            lead_length_mm=tuple(syn["lead_length_mm"]),
            lead_curl_deg=syn["lead_curl_deg"],
            lead_bends=syn["lead_bends"],
            kink_fraction=syn["kink_fraction"],
            empty_fraction=syn["empty_fraction"],
            distractors_per_frame=tuple(syn["distractors_per_frame"]),
            objects_fraction=syn["objects_fraction"],
            objects_per_frame=tuple(syn["objects_per_frame"]),
            camera_corners=tuple(map(tuple, syn["camera_corners_px"])),
            plate_corners_rect=tuple(map(tuple, rectify.mm_to_rect(rect.plate_corners_mm, rect))),
            render_scale=syn["render_scale"],
            defocus=syn["defocus"],
            focus_shift=syn["focus_shift_mm"] * s,
            mjpg_quality=tuple(syn["mjpg_quality"]),
            sharpen=tuple(syn["sharpen"]),
        )


@dataclass(frozen=True)
class Look:
    """Lighting for one frame. Vectors are x right, y down, z towards the viewer."""
    light: np.ndarray          # unit vector towards the light
    half: np.ndarray           # unit half-vector between light and view, for highlights
    ambient: float
    diffuse: float
    flat: float                # brightness of the plate; colours are relative to it
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
    centre: np.ndarray         # px, (x, y), where the body lies on the plate
    bands: list[str]
    flipped: bool              # band 1 at the +u end instead of the -u end
    body_rgb: np.ndarray
    gloss: float
    camera: tuple[float, float]
    camera_height: float
    leads: list = field(default_factory=list)  # (points N×2, width px, rgb)
    band_style: np.ndarray | None = None  # per band: width scale, edge wobble, wobble phase, colour gain
    texture: np.ndarray | None = None     # phases of the paint's fine texture

    @property
    def lean(self):
        """Apparent shift per px of height, away from the camera."""
        return (self.centre - self.camera) / self.camera_height

    @property
    def seen_centre(self):
        """Where the body's axis appears: it sits a radius above the plate."""
        return self.centre + self.radius * self.lean

    @property
    def seen_radius(self):
        """Apparent half-width: seen at an angle, a cylinder shows some of its side."""
        normal = np.array([-math.sin(self.yaw), math.cos(self.yaw)])
        t = float(self.lean @ normal)
        return self.radius * math.sqrt(1 + t * t)

    @property
    def end(self):
        """Fraction of the half-length taken by the rounded end."""
        return min(0.5 * self.radius / (self.length / 2), 0.3)

    def outline(self):
        """The body as it appears, which is also its label."""
        local = _body_outline(self.length, self.seen_radius, self.package.waist, self.end)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.seen_centre + local[:, :1] * (c, s) + local[:, 1:] * (-s, c)

    def polygon(self):
        return Polygon(self.outline())


@dataclass
class Frame:
    image: np.ndarray          # H×W×3 uint8 RGB
    polygons: list             # visible body outlines, N×2, px
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
    pkg = PACKAGES[rng.choice(len(PACKAGES), p=PACKAGE_P)]
    x0, y0, x1, y1 = st.work_area
    # Brightness varies more than hue: band colours sit against this body colour.
    rgb = np.array(pkg.body_rgb, np.float32) / 255 * rng.uniform(0.88, 1.08) * rng.uniform(0.96, 1.04, 3)
    n = pkg.bands
    # Bands are painted by machine but not perfectly: uneven widths, wavy edges, varying shade, and
    # thin enough that the body colour shows through a little.
    band_style = np.column_stack([rng.uniform(0.75, 1.3, n), rng.uniform(0, 0.012, n),
                                  rng.uniform(0, 2 * math.pi, n), rng.uniform(0.85, 1.12, n),
                                  rng.uniform(0.8, 0.97, n)])
    return Resistor(
        band_style=band_style,
        texture=rng.uniform(0, 2 * math.pi, 4),
        package=pkg,
        length=rng.uniform(*pkg.length_mm) * st.px_per_mm,
        radius=rng.uniform(*pkg.diameter_mm) * st.px_per_mm / 2,
        yaw=rng.uniform(0, math.pi),
        centre=np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]),
        bands=_bands(pkg, rng),
        flipped=bool(rng.random() < 0.5),
        body_rgb=np.clip(rgb, 0, 1).astype(np.float32),
        gloss=rng.uniform(0.05, 0.2),
        camera=np.array(st.camera, float),
        camera_height=st.camera_height,
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
    Anchors come from single, the indices of placed bodies not yet in a pair. A free body may
    also be laid parallel a few mm beside any placed body (a cluster, as people line them up).
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
        if placed and rng.random() < st.cluster_fraction:
            anchor = placed[rng.integers(len(placed))]
            axis = np.array([math.cos(anchor.yaw), math.sin(anchor.yaw)])
            res.yaw = anchor.yaw + rng.normal(0, math.radians(6))
            direction = np.array([-axis[1], axis[0]]) * rng.choice([-1.0, 1.0])
            base = anchor.centre + axis * rng.uniform(-0.3, 0.3) * anchor.length
            s = _contact(anchor, res, base, direction)
            if s is None:
                return None
            gap = rng.uniform(st.min_gap_mm, max(st.min_gap_mm, 6.0)) + 0.1
            res.centre = base + direction * (s + gap * st.px_per_mm)
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
    Leads may cross other leads.
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
            elif anchor in single:
                single.remove(anchor)
            bodies.append(res)
            polys.append(res.polygon())
            wires.append(wire)
            break
    return bodies, polys


def _wire_shape(lines, clearance):
    """The area covered by wires, grown by clearance px, for keeping them off bodies."""
    return unary_union([LineString(p).buffer(w / 2 + clearance) for p, w, *_ in lines])


def _fit_leads(res, rng, st, polys):
    """Gives res leads inside the work area and clear of every body in polys.

    Returns the leads' shape for later bodies to keep clear of, or None if no try fits.
    """
    clearance = LEAD_CLEARANCE_MM * st.px_per_mm
    for _ in range(LEAD_TRIES):
        res.leads = _leads(res, rng, st)
        if not all(_inside(p, st.work_area, w / 2) for p, w, _ in res.leads):
            continue
        wire = _wire_shape(res.leads, clearance)
        if not any(wire.intersects(p) for p in polys):
            return wire
    return None


def _lead_path(start, heading, length, rng, st, kink):
    """A wire from start, setting off along heading (rad); N×2 px.

    Mostly straight, as handled leads lie: a slight sweep along the whole length, a few hand bends
    (lead_bends on average, each a turn over 1 to 3 mm), and with kink a sharper bend a few mm out.
    """
    step = 0.5 * st.px_per_mm
    n = max(3, int(length / step))
    s = np.linspace(0.0, 1.0, n)
    turn = rng.normal(0, math.radians(st.lead_curl_deg)) * s
    for _ in range(rng.poisson(st.lead_bends)):
        at = rng.uniform(0.1, 0.9)
        half = rng.uniform(1.0, 2.0) * st.px_per_mm / length       # each bend spread over 2 to 4 mm
        turn = turn + rng.normal(0, math.radians(20)) * _smoothstep(at - half, at + half, s)
    if kink:
        at = rng.uniform(2, 6) * st.px_per_mm / length
        half = rng.uniform(0.6, 1.2) * st.px_per_mm / length
        turn = turn + rng.choice([-1, 1]) * math.radians(rng.uniform(15, 45)) * _smoothstep(at - half, at + half, s)
    ang = heading + turn
    steps = np.stack([np.cos(ang), np.sin(ang)], 1) * step
    return start + np.vstack([[0.0, 0.0], np.cumsum(steps[:-1], 0)])


def _leads(res, rng, st):
    """Two leads out of the end caps. Each is (points N×2, width px, rgb)."""
    width = res.package.lead_mm * st.px_per_mm
    rgb = np.array(LEAD_RGB, np.float32) * rng.uniform(0.9, 1.1)
    leads = []
    for sign in (-1.0, 1.0):
        axis = sign * np.array([math.cos(res.yaw), math.sin(res.yaw)])
        start = res.seen_centre + axis * (res.length / 2 - 0.4 * res.radius)  # hidden inside the cap
        length = rng.uniform(*st.lead_length_mm) * st.px_per_mm
        pts = _lead_path(start, math.atan2(axis[1], axis[0]), length, rng, st, rng.random() < st.kink_fraction)
        leads.append((pts, width, rgb))
    return leads


def _screw(st, rng, at, yaw):
    """A pan-head screw lying on its side: dict for _screw_layer, and its outline for clearance."""
    s = st.px_per_mm
    scr = {
        "at": np.asarray(at, float), "yaw": yaw,
        "head_r": rng.uniform(2.6, 3.0) * s, "head_len": rng.uniform(1.6, 2.2) * s,
        "shaft_r": rng.uniform(1.35, 1.55) * s, "shaft_len": rng.uniform(5.5, 13.0) * s,
        "pitch": 0.5 * s, "rgb": np.array(ZINC_RGB, np.float32) * rng.uniform(0.9, 1.08),
        "lean": (np.asarray(at, float) - st.camera) / st.camera_height,
    }
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    lift = scr["head_r"] * scr["lean"]
    half = np.array([-axis[1], axis[0]])
    head_w = scr["head_r"] * 1.15
    end = scr["at"] + lift + axis * (scr["head_len"] + scr["shaft_len"])
    outline = Polygon([scr["at"] + lift + half * head_w, end + half * head_w,
                       end - half * head_w, scr["at"] + lift - half * head_w])
    return scr, outline


def _shaving(st, rng, at):
    """A curled wood shaving: a crinkled strip along an arc, as a filled polygon N×2 px."""
    s = st.px_per_mm
    radius = rng.uniform(1.5, 4.5) * s
    sweep = math.radians(rng.uniform(60, 220))
    width = rng.uniform(1.2, 3.5) * s
    a0 = rng.uniform(0, 2 * math.pi)
    t = np.linspace(0, 1, 40)
    ang = a0 + sweep * t
    crinkle = 0.2 * s * np.sin(t * rng.uniform(12, 30) + rng.uniform(0, 6))
    inner = np.maximum(radius - width / 2 + crinkle, 0.3 * s)
    outer = radius + width / 2 + crinkle * 0.7
    c = np.asarray(at, float) - radius * np.array([math.cos(a0 + sweep / 2), math.sin(a0 + sweep / 2)])
    a = np.stack([np.cos(ang), np.sin(ang)], 1)
    return np.vstack([c + a * outer[:, None], (c + a * inner[:, None])[::-1]])


def _circle(c, r, n=24):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.asarray(c, float) + r * np.stack([np.cos(t), np.sin(t)], 1)


def _component(st, rng, at, kind):
    """A MOSFET (TO-220 or TO-92) or op-amp (DIP-8) lying on the plate: a dict for _component_layer,
    and its outline for keeping clear of bodies.

    Each part is a footprint (N×2 px) with a height; drawn, its top face appears shifted away from
    the camera by the lean and the side facing the camera shows. Legs are tinned wire on the plate.
    """
    s = st.px_per_mm
    at = np.asarray(at, float)
    a = rng.uniform(0, 2 * math.pi)
    ax = np.array([math.cos(a), math.sin(a)])
    sd = np.array([-ax[1], ax[0]])

    def p(u, v):
        return at + ax * u * s + sd * v * s

    def box(u0, u1, v0, v1):
        return np.array([p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)])

    tin = np.array(TIN_RGB, np.float32) * rng.uniform(0.9, 1.1)
    epoxy = np.array(EPOXY_RGB, np.float32) * rng.uniform(0.8, 1.4)
    ink = np.full(3, rng.uniform(0.45, 0.7), np.float32)
    name = PART_NUMBERS[kind][rng.integers(len(PART_NUMBERS[kind]))]
    parts, legs = [], []

    def leg(u0, v, length, width, bend):
        start = p(u0, v)
        kink = start - ax * length * rng.uniform(0.2, 0.5) * s
        return np.array([start, kink, kink + _rot(-ax, bend) * length * 0.6 * s]), width * s, tin

    if kind == "to220":
        metal = np.array(TAB_RGB, np.float32) * rng.uniform(0.9, 1.08)
        parts.append({"poly": box(2.0, 11.0, -5.0, 5.0), "height": 1.3 * s, "top": metal, "side": metal * 0.65,
                      "holes": [_circle(p(8.2, 0), 1.8 * s)]})
        parts.append({"poly": box(-4.5, 4.5, -5.0, 5.0), "height": 4.5 * s, "top": epoxy, "side": epoxy * 0.6,
                      "text": (name, sd, 1.0 * s, ink)})
        bend = rng.normal(0, math.radians(15)) if rng.random() < 0.5 else 0.0
        for v in (-2.54, 0.0, 2.54):
            legs.append(leg(-4.5, v, rng.uniform(11, 14), 0.8, bend + rng.normal(0, 0.04)))
    elif kind == "to92":
        outline = np.array([p(-2.6, 2.4), p(2.6, 2.4), p(2.6, -2.4), p(-2.6, -2.4)])
        parts.append({"poly": outline, "height": 3.8 * s, "top": epoxy * 1.1, "side": epoxy * 0.6,
                      "text": (name, sd, 0.7 * s, ink)})
        bend = rng.normal(0, math.radians(20)) if rng.random() < 0.6 else 0.0
        for v in (-1.27, 0.0, 1.27):
            legs.append(leg(-2.6, v, rng.uniform(11, 14), 0.45, bend + rng.normal(0, 0.06)))
    else:                                               # DIP-8, lying legs down
        body = box(-4.8, 4.8, -3.2, 3.2)
        marks = [(_circle(p(4.8, 0.0), 0.9 * s, 16), epoxy * 0.45), (_circle(p(3.4, 2.0), 0.45 * s, 12), epoxy * 0.55)]
        parts.append({"poly": body, "height": 3.8 * s, "top": epoxy, "side": epoxy * 0.6, "marks": marks,
                      "text": (name, ax, 1.0 * s, ink)})
        for u in (-3.81, -1.27, 1.27, 3.81):
            for side in (-1, 1):
                legs.append((np.array([p(u, side * 3.1), p(u, side * 3.7)]), 1.4 * s, tin))
                legs.append((np.array([p(u, side * 3.6), p(u, side * 4.6)]), 0.5 * s, tin))
    lean = (at - st.camera) / st.camera_height
    pts = np.vstack([q["poly"] for q in parts] + [q["poly"] + lean * q["height"] for q in parts]
                    + [l[0] for l in legs])
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    return {"parts": parts, "legs": legs, "lean": lean, "extent": pts}, Polygon(hull)


def _distractors(st, rng, polys, objects, wires=None):
    """Unlabelled clutter clear of the bodies, as kwargs for _draw.

    Small clutter (lead offcuts, crumbs, pencil marks, smudges) in every frame; with objects,
    screws and wood shavings too.
    """
    s = st.px_per_mm
    x0, y0, x1, y1 = st.work_area
    bodies = unary_union(polys)
    # Big objects keep off the resistors' leads too, so no lead is drawn across a package.
    big_avoid = bodies if wires is None else unary_union([bodies, wires])
    clearance = LEAD_CLEARANCE_MM * s
    kinds = []
    lo, hi = st.distractors_per_frame
    kinds += list(rng.choice(["offcut", "crumb", "mark", "smudge"], int(rng.integers(lo, hi + 1)),
                             p=(0.35, 0.25, 0.2, 0.2)))
    if objects:
        lo, hi = st.objects_per_frame
        kinds += list(rng.choice(["screw", "shaving", "to220", "to92", "dip8"], int(rng.integers(lo, hi + 1)),
                                 p=(0.25, 0.15, 0.25, 0.15, 0.2)))
    out, taken = [], []
    for kind in kinds:
        for _ in range(15):
            p = np.array([rng.uniform(x0 + 8 * s, x1 - 8 * s), rng.uniform(y0 + 8 * s, y1 - 8 * s)])
            if kind == "offcut":
                a = rng.uniform(0, 2 * math.pi)
                pts = [p]
                for _ in range(int(rng.integers(1, 3))):
                    a += rng.normal(0, 0.6)
                    pts.append(pts[-1] + np.array([math.cos(a), math.sin(a)]) * rng.uniform(2, 12) * s)
                rgb = np.array(LEAD_RGB, np.float32) * rng.uniform(0.9, 1.1)
                item = {"lines": [(np.array(pts), 0.6 * s, rgb)]}
                shape = _wire_shape(item["lines"], clearance)
            elif kind == "crumb":
                k = int(rng.integers(7, 13))
                angles = np.sort(rng.uniform(0, 2 * math.pi, k))
                radii = rng.uniform(0.3, 1.0) * s * rng.uniform(0.6, 1.0, k)
                pts = p + np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
                rgb = np.array(CRUMB_RGB[rng.integers(len(CRUMB_RGB))], np.float32) / 255
                item = {"blobs": [(pts, np.clip(rgb * rng.uniform(0.8, 1.2), 0, 1))]}
                shape = Polygon(pts).buffer(clearance)
            elif kind == "mark":
                a = rng.uniform(0, 2 * math.pi)
                d = np.array([math.cos(a), math.sin(a)]) * rng.uniform(1.5, 5) * s
                pts = [p - d, p + d] if rng.random() < 0.6 else [p - d, p + d, p, p + _rot(d, math.pi / 2)]
                item = {"marks": [(np.array(pts, float), rng.uniform(0.25, 0.5) * s, rng.uniform(0.15, 0.4))]}
                shape = LineString(np.array(pts, float)).buffer(clearance)
            elif kind == "smudge":
                item = {"smudges": [(p, rng.uniform(1.5, 5) * s, rng.uniform(0.04, 0.12))]}
                shape = None
            elif kind == "screw":
                scr, outline = _screw(st, rng, p, rng.uniform(0, 2 * math.pi))
                item = {"screw": scr}
                shape = outline.buffer(clearance)
            elif kind in ("to220", "to92", "dip8"):
                comp, outline = _component(st, rng, p, kind)
                item = {"component": comp}
                shape = outline.buffer(clearance)
            else:
                pts = _shaving(st, rng, p)
                rgb = np.array(WOOD_RGB, np.float32) * rng.uniform(0.85, 1.08) * rng.uniform(0.96, 1.04, 3)
                item = {"blobs": [(pts, np.clip(rgb, 0, 1))], "edge": 0.75}
                shape = Polygon(pts).buffer(clearance)
            if shape is None:
                out.append(item)
                break
            if not _inside(np.reshape(shape.bounds, (2, 2)), st.work_area):
                continue
            big = kind in ("screw", "shaving", "to220", "to92", "dip8")
            if shape.intersects(big_avoid if big else bodies) or any(shape.intersects(t) for t in taken):
                continue
            if big:
                taken.append(shape)
            out.append(item)
            break
    return out


# --- Drawing -----------------------------------------------------------------------------------

def _look(st, rng):
    az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(55, 85))
    light = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    half = light + (0, 0, 1)
    ambient = rng.uniform(0.6, 0.85)          # the room light is soft: shading is gentle
    diffuse = 1 - ambient
    height = 1.2 * st.px_per_mm  # roughly how far a body stands off the plate
    return Look(
        light=light,
        half=half / np.linalg.norm(half),
        ambient=ambient,
        diffuse=diffuse,
        flat=ambient + diffuse * light[2],
        shininess=rng.uniform(12, 40),
        shadow_offset=-light[:2] / np.linalg.norm(light[:2]) * height / math.tan(el),
        shadow_sigma=rng.uniform(0.5, 1.2) * st.px_per_mm,
        shadow_strength=rng.uniform(0.08, 0.25),
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


def _offset(pts, d):
    """A polyline moved sideways by d px (to the left of its direction of travel)."""
    seg = np.diff(pts, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    tangent = np.vstack([seg[:1], (seg[:-1] + seg[1:]) / 2, seg[-1:]])
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    return pts + d * normal


def _line_layer(lines, origin, shape):
    """Wires as alpha and premultiplied colour: darker edges and a highlight stripe, like a
    shiny round wire."""
    alpha = np.zeros(shape, np.float32)
    premul = np.zeros(shape + (3,), np.float32)
    for pts, width, rgb in lines:
        a = _stroke(pts - origin, width, shape)
        middle = _stroke(pts - origin, width * 0.55, shape)
        stripe = _stroke(_offset(pts, 0.18 * width) - origin, max(1.0, width / 6), shape)
        # The highlight is uneven: dents and bends catch the light in glints.
        seed = int(abs(float(pts[:4].sum())) * 1000) % 2 ** 32
        glint = np.clip(1 + 0.6 * _noise(np.random.default_rng(seed), shape, 20), 0.5, 2.0)
        colour = rgb * (0.92 + 0.08 * middle)[..., None] + 0.16 * (stripe * a * glint)[..., None]
        premul = premul * (1 - a)[..., None] + colour * a[..., None]
        alpha = alpha * (1 - a) + a
    return alpha, premul


def _blob_layer(blobs, origin, shape, edge=None):
    """Filled shapes; with edge, their rim is darkened to that share of the colour."""
    alpha = np.zeros(shape, np.float32)
    premul = np.zeros(shape + (3,), np.float32)
    for pts, rgb in blobs:
        a = _stroke(pts - origin, 0, shape, fill=True)
        colour = np.broadcast_to(rgb, shape + (3,)).astype(np.float32)
        if edge is not None:
            rim = _stroke(pts - origin, 2, shape, closed=True)
            colour = colour * (1 - (1 - edge) * rim)[..., None]
        premul = premul * (1 - a)[..., None] + colour * a[..., None]
        alpha = alpha * (1 - a) + a
    return alpha, premul


def _rod(xs, ys, centre, yaw, radius_of, look, albedo_of, gloss_of):
    """Shading of a round object lying along yaw: (inside mask, colour) at sample points.

    radius_of(u) gives the half-width and the end rounding at each position u along the axis
    (negative u back from the centre); albedo_of and gloss_of give surface colour and gloss.
    """
    X, Y = np.meshgrid(xs - centre[0], ys - centre[1])
    c, s = math.cos(yaw), math.sin(yaw)
    u = X * c + Y * s
    v = -X * s + Y * c
    r, k = radius_of(u)
    inside = (r > 0) & (np.abs(v) < r)
    n = np.clip(v / np.maximum(r, 1e-6), -1, 1)
    q = np.sqrt(1 - k * k)
    nu, nv, nz = np.sign(u) * k, n * q, np.sqrt(1 - n * n) * q

    def local(vec):
        return vec[0] * c + vec[1] * s, -vec[0] * s + vec[1] * c, vec[2]

    lu, lv, lz = local(look.light)
    hu, hv, hz = local(look.half)
    ndl = np.clip(nu * lu + nv * lv + nz * lz, 0, None)
    ndh = np.clip(nu * hu + nv * hv + nz * hz, 0, None)
    shade = (look.ambient + look.diffuse * ndl) / look.flat
    colour = albedo_of(u, v) * shade[..., None] + (gloss_of(u) * ndh ** look.shininess)[..., None]
    return inside, colour * inside[..., None]


def _supersampled(rect, pts, draw):
    """Runs draw(xs, ys) on a SS-times supersampled grid over pts' box within rect and averages it
    down: (alpha, premultiplied colour) over rect."""
    x0, y0, x1, y1 = rect
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    premul = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    bx0 = max(x0, math.floor(pts[:, 0].min()) - 1)
    bx1 = min(x1, math.ceil(pts[:, 0].max()) + 2)
    by0 = max(y0, math.floor(pts[:, 1].min()) - 1)
    by1 = min(y1, math.ceil(pts[:, 1].max()) + 2)
    if bx0 >= bx1 or by0 >= by1:
        return alpha, premul
    bh, bw = by1 - by0, bx1 - bx0
    offsets = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (np.arange(bx0, bx1)[:, None] + offsets).ravel()
    ys = (np.arange(by0, by1)[:, None] + offsets).ravel()
    inside, colour = draw(xs, ys)
    sub = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
    alpha[sub] = inside.reshape(bh, SS, bw, SS).mean(axis=(1, 3))
    premul[sub] = colour.reshape(bh, SS, bw, SS, 3).mean(axis=(1, 3))
    return alpha, premul


def _body_layer(res, look, rect):
    """The body as it appears, supersampled over rect (x0, y0, x1, y1): alpha and premultiplied
    colour."""
    half, end, radius = res.length / 2, res.end, res.seen_radius

    def radius_of(u):
        t = np.abs(u) / half
        r = radius * _profile(np.minimum(t, 1), res.package.waist, end)
        return np.where(t <= 1, r, 0.0), np.clip((t - (1 - end)) / end, 0, 1)

    axis = np.array([math.cos(res.yaw), math.sin(res.yaw)])
    lean_along = float(res.lean @ axis)

    def albedo_of(u, v):
        # Seen at an angle, a painted ring is an arc: the top of the body stands higher than its
        # axis, so it appears shifted further along the lean. Undo that to find where on the
        # body each point is.
        height = res.radius * np.sqrt(1 - np.clip(v / radius, -1, 1) ** 2)
        pos = (u - lean_along * height + half) / res.length
        if res.flipped:
            pos = 1 - pos
        albedo = np.broadcast_to(res.body_rgb, u.shape + (3,)).copy()
        grain = np.zeros(u.shape, np.float32)
        if res.texture is not None:                 # fine, uneven paint
            ph = res.texture
            grain = (np.sin(1.1 * u + 1.9 * v + ph[0]) + np.sin(2.6 * u - 0.7 * v + ph[1])
                     + np.sin(0.45 * u + 3.1 * v + ph[2]) + np.sin(3.7 * u + 1.3 * v + ph[3])) / 4
        style = (res.band_style if res.band_style is not None
                 else np.tile([1.0, 0.0, 0.0, 1.0, 1.0], (len(res.bands), 1)))
        for (p, name), (scale, wobble, phase, gain, cover) in zip(zip(BAND_POSITIONS[len(res.bands)], res.bands), style):
            edge = pos - p + wobble * np.sin(3 * v / max(radius, 1.0) + phase)
            m = np.abs(edge) < BAND_WIDTH * scale / 2
            albedo[m] = (1 - cover) * res.body_rgb + cover * np.clip(BAND_RGB[name] * gain, 0, 1)
        return albedo * (1 + 0.035 * grain)[..., None]

    def gloss_of(u):
        pos = (u + half) / res.length
        if res.flipped:
            pos = 1 - pos
        gloss = np.full(u.shape, res.gloss)
        for p, name in zip(BAND_POSITIONS[len(res.bands)], res.bands):
            if name in ("gold", "silver"):
                gloss[np.abs(pos - p) < BAND_WIDTH / 2] = 0.6
        return gloss

    return _supersampled(rect, res.outline(), lambda xs, ys: _rod(
        xs, ys, res.seen_centre, res.yaw, radius_of, look, albedo_of, gloss_of))


def _screw_layer(scr, look, rect):
    """A screw lying on its side: pan head and threaded shaft, seen at the camera's angle."""
    axis = np.array([math.cos(scr["yaw"]), math.sin(scr["yaw"])])
    lean = scr["lean"]
    normal = np.array([-axis[1], axis[0]])
    widen = math.sqrt(1 + float(lean @ normal) ** 2)
    head_r, head_len = scr["head_r"] * widen, scr["head_len"]
    shaft_r, shaft_len = scr["shaft_r"] * widen, scr["shaft_len"]
    total = head_len + shaft_len
    centre = scr["at"] + scr["head_r"] * lean + axis * total / 2   # middle of the screw, as seen

    def radius_of(u):
        a = u + total / 2                                        # 0 at the head's face
        dome = np.clip(1 - a / (0.45 * head_len), 0, 1)           # rounded pan head
        r = np.where(a < head_len, head_r * np.sqrt(1 - 0.6 * dome ** 2), shaft_r)
        tip = np.clip((a - (total - 0.4 * shaft_r)) / (0.4 * shaft_r), 0, 1)
        r = np.where((a >= 0) & (a <= total), r * np.sqrt(1 - tip ** 2), 0.0)
        return r, np.maximum(dome * 0.6, tip)

    def albedo_of(u, v):
        a = u + total / 2
        thread = 1 + 0.22 * np.sin(2 * math.pi * (a + 0.35 * v) / scr["pitch"])
        tone = np.where(a < head_len, 1.0, thread)
        return scr["rgb"] * tone[..., None]

    def gloss_of(u):
        return np.full(u.shape, 0.55)

    corners = np.array([centre + axis * total / 2 + normal * head_r, centre + axis * total / 2 - normal * head_r,
                        centre - axis * total / 2 - normal * head_r, centre - axis * total / 2 + normal * head_r])
    return _supersampled(rect, corners, lambda xs, ys: _rod(
        xs, ys, centre, scr["yaw"], radius_of, look, albedo_of, gloss_of))


def _component_layer(comp, origin, shape):
    """A packaged part (from _component): each part's camera-facing side, then its top face with
    holes, marks and faint printing. Alpha and premultiplied colour."""
    alpha = np.zeros(shape, np.float32)
    premul = np.zeros(shape + (3,), np.float32)

    def paint(poly, colour, cover=None):
        nonlocal alpha, premul
        a = _stroke(poly - origin, 0, shape, fill=True) if cover is None else cover
        premul = premul * (1 - a)[..., None] + np.asarray(colour, np.float32) * a[..., None]
        alpha = alpha * (1 - a) + a

    for part in comp["parts"]:
        lift = comp["lean"] * part["height"]
        top = part["poly"] + lift
        paint(cv2.convexHull(np.vstack([part["poly"], top]).astype(np.float32)).reshape(-1, 2), part["side"])
        paint(top, part["top"])
        for hole in part.get("holes", ()):
            a = _stroke(hole + lift - origin, 0, shape, fill=True)
            alpha, premul = alpha * (1 - a), premul * (1 - a)[..., None]
        for poly, colour in part.get("marks", ()):
            paint(poly + lift, colour)
        if "text" in part:
            text, along, cap, ink = part["text"]
            reach = part["poly"] @ along
            room = 0.85 * (reach.max() - reach.min())          # the face's length along the text
            scale = cap / 22
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw > room:
                scale *= room / tw
                (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            canvas = np.zeros((th + base + 4, tw + 4), np.uint8)
            cv2.putText(canvas, text, (2, th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
            perp = np.array([-along[1], along[0]])
            h, w = canvas.shape
            t = top.mean(0) - origin - along * w / 2 - perp * h / 2
            m = np.float32([[along[0], perp[0], t[0]], [along[1], perp[1], t[1]]])
            mark = cv2.warpAffine(canvas, m, (shape[1], shape[0])).astype(np.float32) / 255 * 0.7
            premul = premul * (1 - mark)[..., None] + ink * mark[..., None]
    return alpha, premul


def _draw(img, look, lines=(), body=None, blobs=(), edge=None, screw=None, marks=(), smudges=(), component=None):
    """Draws one object and its shadow onto img, in place. Marks and smudges are faint, flat and
    cast no shadow.

    Returns (x0, y0, body alpha) over the patch drawn, or None if the object is off the frame.
    """
    height, width = img.shape[:2]
    for p, size, strength in smudges:
        r = int(3 * size)
        x0, y0 = max(0, int(p[0]) - r), max(0, int(p[1]) - r)
        x1, y1 = min(width, int(p[0]) + r + 1), min(height, int(p[1]) + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        blob = np.exp(-(((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * size * size))).astype(np.float32)
        img[y0:y1, x0:x1] *= (1 - strength * blob)[..., None]
    for pts, w, strength in marks:
        a = _stroke(pts, w, (height, width))
        img *= (1 - strength * a)[..., None]
    if not (lines or body is not None or blobs or screw is not None or component is not None):
        return None

    groups = [p for p, *_ in lines] + [p for p, _ in blobs]
    if body is not None:
        groups.append(body.outline())
    if component is not None:
        groups.append(component["extent"])
    if screw is not None:
        span = screw["head_len"] + screw["shaft_len"] + 2 * screw["head_r"]
        groups.append(screw["at"] + np.array([[-span, -span], [span, span]]))
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
        layers.append(_blob_layer(blobs, origin, shape, edge))
    if lines:
        layers.append(_line_layer(lines, origin, shape))
    if screw is not None:
        layers.append(_screw_layer(screw, look, (x0, y0, x1, y1)))
    if component is not None:
        layers.append(_line_layer(component["legs"], origin, shape))
        layers.append(_component_layer(component, origin, shape))
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


def _hsv(h, s, v):
    """RGB in 0..1 from hue in degrees, saturation and value in 0..1."""
    hsv = np.uint8([[[int(h / 2) % 180, int(s * 255), int(v * 255)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255


def _streaks(rng, shape, along, across, angle):
    """Noise stretched into streaks about along × across px, turned to angle (degrees)."""
    h, w = shape
    side = int(1.5 * max(h, w))
    small = rng.standard_normal((max(2, int(side / across)), max(2, int(side / along)))).astype(np.float32)
    big = cv2.resize(small, (side, side), interpolation=cv2.INTER_CUBIC)
    turn = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    turn[:, 2] -= ((side - w) / 2, (side - h) / 2)
    return cv2.warpAffine(big, turn, (w, h))


def _stripes(shape, period, angle, rng):
    """A sine pattern in 0..1 across stripes period px apart, at angle (degrees)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    a = math.radians(angle)
    return 0.5 + 0.5 * np.sin(2 * math.pi * (xx * math.cos(a) + yy * math.sin(a)) / period + rng.uniform(0, 6.3))


def _surround(kind, shape, rng, st, real=False):
    """One surround surface over the whole frame (it is masked to its side later), H×W×3.

    real: the surface as on the rig (blue robot base, oak bench) rather than any colour.
    """
    s = st.px_per_mm
    if kind == "wood":
        base = (_hsv(rng.uniform(25, 35), rng.uniform(0.55, 0.7), rng.uniform(0.55, 0.7)) if real
                else _hsv(rng.uniform(15, 40), rng.uniform(0.3, 0.8), rng.uniform(0.25, 0.92)))
        angle = rng.uniform(0, 180)
        grain = (rng.uniform(0.03, 0.08) * _streaks(rng, shape, rng.uniform(20, 60), 2, angle)
                 + rng.uniform(0.04, 0.12) * _streaks(rng, shape, 300, rng.uniform(25, 60), angle))
        return base * (1 + grain)[..., None]
    if kind == "plastic":
        hue = rng.uniform(222, 232) if real or rng.random() < 0.4 else rng.uniform(0, 360)
        base = _hsv(hue, rng.uniform(0.75, 0.95) if real else rng.uniform(0.2, 0.95), rng.uniform(0.35, 0.9))
        tone = 1 + rng.uniform(0.05, 0.2) * _noise(rng, shape, 200) + 0.03 * _noise(rng, shape, 4)
        if rng.random() < 0.5:                       # 3D-print layer lines
            tone = tone + rng.uniform(0.02, 0.06) * (_stripes(shape, rng.uniform(1.5, 3.0), rng.uniform(0, 180), rng) - 0.5)
        return base * tone[..., None]
    if kind == "mat":                                # cutting mat with a 10 mm grid
        base = [_hsv(rng.uniform(120, 160), rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.5)),
                np.full(3, rng.uniform(0.15, 0.4), np.float32),
                _hsv(rng.uniform(200, 230), rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.5))][rng.integers(3)]
        angle = rng.choice([0.0, 90.0]) + rng.normal(0, 3)
        lines = (_stripes(shape, 10 * s, angle, rng) > 0.998) | (_stripes(shape, 10 * s, angle + 90, rng) > 0.998)
        img = base * (1 + 0.04 * _noise(rng, shape, 3))[..., None]
        img[lines] = img[lines] * 0.4 + 0.6 * rng.uniform(0.6, 0.9)
        return img
    if kind == "metal":                              # brushed aluminium or steel
        base = np.full(3, rng.uniform(0.4, 0.8), np.float32) * rng.uniform(0.97, 1.03, 3).astype(np.float32)
        brush = 0.05 * _streaks(rng, shape, rng.uniform(80, 300), 1.5, rng.uniform(0, 180))
        return base * (1 + brush + 0.08 * _noise(rng, shape, 300))[..., None]
    if kind == "cardboard":
        base = _hsv(rng.uniform(25, 38), rng.uniform(0.35, 0.6), rng.uniform(0.45, 0.75))
        flutes = 0.04 * (_stripes(shape, rng.uniform(3, 5) * s, rng.uniform(0, 180), rng) - 0.5)
        return base * (1 + flutes + 0.05 * _noise(rng, shape, 6))[..., None]
    if kind == "fabric":
        base = _hsv(rng.uniform(0, 360), rng.uniform(0.0, 0.8), rng.uniform(0.15, 0.85))
        angle = rng.uniform(0, 180)
        weave = (_stripes(shape, rng.uniform(1.5, 3), angle, rng) * _stripes(shape, rng.uniform(1.5, 3), angle + 90, rng))
        return base * (1 + 0.1 * (weave - 0.25) + 0.08 * _noise(rng, shape, 2))[..., None]
    if kind == "gradient":
        a, b = (_hsv(rng.uniform(0, 360), rng.uniform(0, 0.9), rng.uniform(0.1, 0.9)) for _ in range(2))
        t = np.clip(0.5 + 0.5 * _stripes(shape, rng.uniform(1, 3) * max(shape), rng.uniform(0, 360), rng) - 0.25, 0, 1)
        return a * (1 - t[..., None]) + b * t[..., None]
    level = rng.uniform(0.02, 0.12) if real else rng.uniform(0.02, 0.3)    # dark: the gap under the plate
    return np.full(shape + (3,), level, np.float32) * (1 + 0.1 * _noise(rng, shape, 40))[..., None]


def _margin_clutter(img, st, rng):
    """Things lying around the plate (drawn before the plate, which covers any that reach it):
    cables, screw heads, a scrap of circuit board, and soft shadows."""
    s = st.px_per_mm
    h, w = img.shape[:2]
    for _ in range(int(rng.choice(3, p=(0.5, 0.35, 0.15)))):        # cables
        p = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        a = rng.uniform(0, 2 * math.pi)
        pts = [p]
        for _ in range(12):
            a += rng.normal(0, 0.25)
            pts.append(pts[-1] + np.array([math.cos(a), math.sin(a)]) * 15 * s)
        pts = np.array(pts)
        width = rng.uniform(2, 5) * s
        rgb = np.array(CABLE_RGB[rng.integers(len(CABLE_RGB))], np.float32) * rng.uniform(0.8, 1.15)
        for frac, gain in ((1.0, 0.8), (0.6, 1.0), (0.15, 1.6)):
            cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], False,
                          tuple(float(v) for v in np.clip(rgb * gain, 0, 1)), max(1, int(width * frac)),
                          cv2.LINE_AA, shift=4)
    for _ in range(int(rng.integers(0, 4))):                        # screw heads
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        r = int(rng.uniform(2.5, 3.5) * s)
        grey = rng.uniform(0.5, 0.8)
        cv2.circle(img, c, r, (grey,) * 3, -1, cv2.LINE_AA)
        a = rng.uniform(0, math.pi)
        for t in (a, a + math.pi / 2):
            d = (int(0.6 * r * math.cos(t)), int(0.6 * r * math.sin(t)))
            cv2.line(img, (c[0] - d[0], c[1] - d[1]), (c[0] + d[0], c[1] + d[1]), (grey * 0.5,) * 3,
                     max(1, int(0.25 * r)), cv2.LINE_AA)
    if rng.random() < 0.2:                                          # a scrap of circuit board
        c = (rng.uniform(0, w), rng.uniform(0, h))
        size = (rng.uniform(15, 40) * s, rng.uniform(10, 30) * s)
        box_pts = cv2.boxPoints((c, size, rng.uniform(0, 180)))
        cv2.fillPoly(img, [np.round(box_pts).astype(np.int32)], tuple(float(v) for v in _hsv(140, 0.7, 0.35)),
                     cv2.LINE_AA)
        for _ in range(int(rng.integers(3, 10))):
            q = box_pts.mean(0) + rng.uniform(-0.35, 0.35, 2) * size
            cv2.circle(img, (int(q[0]), int(q[1])), int(0.8 * s), (0.85, 0.7, 0.35), -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(0, 3))):                        # soft shadows
        blob = np.zeros(img.shape[:2], np.float32)
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        cv2.ellipse(blob, c, (int(rng.uniform(10, 60) * s), int(rng.uniform(10, 40) * s)), rng.uniform(0, 180),
                    0, 360, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(3, 10) * s)
        img *= (1 - rng.uniform(0.2, 0.5) * blob)[..., None]


def _scene(st, rng):
    """The plate and what lies around it, before any object: H×W×3 float RGB."""
    shape = (st.height, st.width)
    px0, py0, px1, py1 = st.plate
    s = st.px_per_mm
    yy, xx = np.mgrid[:st.height, :st.width]
    # Which side of the plate each pixel is on: bands above and below the plate run the full
    # width, as the robot base and table edge do; the left and right strips lie beside it.
    cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
    side = np.where(yy < py0, 0, np.where(yy >= py1, 3, np.where(xx < cx, 1, 2)))
    img = np.zeros(shape + (3,), np.float32)
    for i, name in enumerate(("top", "left", "right", "bottom")):
        region = side == i
        if rng.random() < REAL_SURROUND_FRACTION:
            tex = _surround(REAL_SURROUNDS[name], shape, rng, st, real=True)
        else:
            tex = _surround(SURROUND_KINDS[rng.integers(len(SURROUND_KINDS))], shape, rng, st)
        if rng.random() < SPLIT_FRACTION:              # a second surface past a straight line
            other = _surround(SURROUND_KINDS[rng.integers(len(SURROUND_KINDS))], shape, rng, st)
            a = rng.uniform(0, 2 * math.pi)
            ys, xs = np.nonzero(region)
            k = rng.integers(len(xs))
            beyond = (xx - xs[k]) * math.cos(a) + (yy - ys[k]) * math.sin(a) > 0
            tex = np.where(beyond[..., None], other, tex)
        img[region] = tex[region]
    _margin_clutter(img, st, rng)
    # The plate's front face (a grey strip just below its near edge) and its legs.
    if rng.random() < FACE_FRACTION:
        grey = rng.uniform(0.4, 0.65)
        strip = rng.uniform(2.0, 4.5) * s
        img[(xx < px0) & (xx >= px0 - strip) & (yy > py0) & (yy < py1)] = grey
        for corner_y in (py0, py1):
            leg = np.array([px0 - strip - rng.uniform(1, 3) * s, corner_y + rng.choice([-1, 1]) * rng.uniform(2, 6) * s])
            cv2.ellipse(img, (int(leg[0]), int(leg[1])), (int(rng.uniform(2.5, 3.5) * s), int(rng.uniform(3, 6) * s)),
                        rng.uniform(0, 180), 0, 360, (grey * 0.9,) * 3, -1, cv2.LINE_AA)
    # The plate: rounded square, neutral, with a gentle lighting gradient and fine texture.
    mask = np.zeros(shape, np.uint8)
    r = int(round(st.plate_radius))
    cv2.rectangle(mask, (px0 + r, py0), (px1 - 1 - r, py1 - 1), 255, -1)
    cv2.rectangle(mask, (px0, py0 + r), (px1 - 1, py1 - 1 - r), 255, -1)
    for x, y in ((px0 + r, py0 + r), (px1 - 1 - r, py0 + r), (px0 + r, py1 - 1 - r), (px1 - 1 - r, py1 - 1 - r)):
        cv2.circle(mask, (x, y), r, 255, -1, cv2.LINE_AA)
    a = rng.uniform(0, 2 * math.pi)
    slope = rng.uniform(0.0, 0.14)              # brightness change across the plate, as a share
    ramp = ((xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)) / (px1 - px0)
    plate = np.array(PLATE_RGB, np.float32) * rng.uniform(0.97, 1.03) * rng.uniform(0.995, 1.005, 3)
    field = 1 + slope * ramp + 0.008 * _noise(rng, shape, 150) + 0.0015 * _noise(rng, shape, 6)
    alpha = (mask.astype(np.float32) / 255)[..., None]
    return img * (1 - alpha) + (plate * field[..., None]) * alpha


def _camera(img, rng, st):
    """Focus blur growing towards the far side (right), the camera's sharpening (faint halos at
    edges), compression's softer colour detail, exposure, a slight tint, and low sensor noise;
    returns uint8."""
    img = img.astype(np.float32, copy=False)
    near = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.2, 0.35))
    far = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.45, 0.7))
    t = np.clip(np.linspace(-0.2, 1.2, st.width, dtype=np.float32), 0, 1)[None, :, None]
    img = near * (1 - t) + far * t
    img = img + np.float32(rng.uniform(0.3, 0.8)) * (img - cv2.GaussianBlur(img, (0, 0), 1.2))
    ycc = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2YCrCb)
    ycc[..., 1:] = cv2.GaussianBlur(np.ascontiguousarray(ycc[..., 1:]), (0, 0), 0.8)
    img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    img *= (rng.uniform(0.97, 1.03) * rng.uniform(0.99, 1.01, 3)).astype(np.float32)
    img += rng.standard_normal(img.shape, dtype=np.float32) * np.float32(rng.uniform(0.0015, 0.003))
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


def _homography_raw_from_rect(st, rng):
    """Raw camera frame px from rectified px, with the camera jittered a little as on the rig."""
    corners = np.array(st.camera_corners, np.float64)
    corners += rng.normal(0, 10, 2) + rng.normal(0, 2, (4, 2))
    h = cv2.getPerspectiveTransform(corners.astype(np.float32), np.array(st.plate_corners_rect, np.float32))
    return np.linalg.inv(h)


def _variable_blur(img, sigma):
    """img blurred by a per-pixel sigma (H×W), by blending a few fixed blurs."""
    levels = np.array([0.0, 0.6, 1.2, 2.0, 3.0, 4.5], np.float32)
    s = np.clip(sigma, levels[0], levels[-1])
    out = np.zeros_like(img)
    blurred = [img if lv == 0 else cv2.GaussianBlur(img, (0, 0), float(lv)) for lv in levels]
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        w = np.clip((s - lo) / (hi - lo), 0, 1)
        inside = ((s >= lo) & (s <= hi))[..., None]
        out = np.where(inside, blurred[i] * (1 - w)[..., None] + blurred[i + 1] * w[..., None], out)
    return out


def _through_camera(img, st, sr, rng):
    """The drawn scene as the real camera and `rectify` would deliver it; returns uint8 RGB.

    img is the scene drawn top-down at sr (st drawn render_scale finer). It is warped into the
    camera's raw view, blurred by distance from the focus plane, sampled through the colour
    filter with sensor noise and demosaiced, sharpened as the camera does, compressed as MJPG,
    warped back to the rectified view with bilinear interpolation like `rectify`, and white
    balanced on the plate.
    """
    k = sr.width / st.width
    to_rect = np.array([[1 / k, 0, 0.5 / k - 0.5], [0, 1 / k, 0.5 / k - 0.5], [0, 0, 1]])  # render px -> rect px
    raw_from_rect = _homography_raw_from_rect(st, rng)
    frame = np.array([[-0.5, -0.5], [st.width - 0.5, -0.5], [st.width - 0.5, st.height - 0.5], [-0.5, st.height - 0.5]])
    seen = cv2.perspectiveTransform(frame.reshape(-1, 1, 2), raw_from_rect).reshape(-1, 2)
    x0, y0 = np.floor(seen.min(0)).astype(int) - 8
    x1, y1 = np.ceil(seen.max(0)).astype(int) + 8
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2                       # keep the colour filter's phase
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    roi_from_rect = shift @ raw_from_rect
    size = (int(x1 - x0), int(y1 - y0))

    # Into the camera's view: soften first so the far side doesn't alias when it shrinks.
    soft = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.25 * k)
    raw = cv2.warpPerspective(soft, roi_from_rect @ to_rect, size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    # Focus: blur grows with distance from the focus plane (a coarse map, then interpolated).
    gy, gx = np.mgrid[0:size[1]:16, 0:size[0]:16].astype(np.float32)
    on_plate = cv2.perspectiveTransform(np.stack([gx, gy], -1).reshape(-1, 1, 2).astype(np.float64),
                                        np.linalg.inv(roi_from_rect)).reshape(gx.shape + (2,))
    centre = np.array([(st.plate[0] + st.plate[2]) / 2, (st.plate[1] + st.plate[3]) / 2])
    cam = np.array(st.camera)
    dist = np.sqrt(st.camera_height ** 2 + ((on_plate - cam) ** 2).sum(-1))
    focus = math.sqrt(st.camera_height ** 2 + ((centre - cam) ** 2).sum()) + st.focus_shift
    sigma = 0.1 + st.defocus * rng.uniform(0.85, 1.15) * np.abs(dist - focus) / dist
    sigma = cv2.resize(sigma.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    raw = _variable_blur(raw, sigma)

    # The sensor: a red-green-green-blue colour filter, a residual tint, shot and read noise.
    raw = np.clip(raw * rng.uniform(0.97, 1.03, 3).astype(np.float32), 0, 1)
    mosaic = np.empty(raw.shape[:2], np.float32)
    mosaic[0::2, 0::2] = raw[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = raw[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = raw[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = raw[1::2, 1::2, 2]
    shot, read = rng.uniform(0.7e-6, 2.2e-6), rng.uniform(0.0003, 0.0006)   # after the camera's denoising
    mosaic += rng.standard_normal(mosaic.shape).astype(np.float32) * np.sqrt(shot * mosaic + read ** 2)
    bayer = (np.clip(mosaic, 0, 1) * 65535).astype(np.uint16)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB_EA).astype(np.float32) / 65535   # BG2RGB decodes RGGB

    # The camera's processing: sharpening, then MJPG compression.
    rgb = rgb + np.float32(rng.uniform(*st.sharpen)) * (rgb - cv2.GaussianBlur(rgb, (0, 0), 0.8))
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*st.mjpg_quality))])
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Back to the rectified view, as `rectify` does, then the plate white balance.
    rect = cv2.warpPerspective(bgr, np.linalg.inv(roi_from_rect), (st.width, st.height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB).astype(np.float32)
    inset = int(round(5 * st.px_per_mm))
    px0, py0, px1, py1 = st.plate
    plate = rect[py0 + inset:py1 - inset, px0 + inset:px1 - inset].reshape(-1, 3)
    lum = plate.mean(1)
    white = np.median(plate[lum >= np.median(lum)], axis=0)
    rect *= white.mean() / np.maximum(white, 1.0)
    return np.clip(rect + 0.5, 0, 255).astype(np.uint8)


def render_frame(st, rng, background=None):
    """One synthetic frame. background: H×W×3 uint8 RGB rectified empty-plate image, or None.

    The scene is drawn render_scale times finer than the frame; with the camera simulation it then
    goes through _through_camera, else it is shrunk and given _camera's simpler effects. Labels
    are in the frame's px either way.
    """
    sr = st.scaled(st.render_scale)
    look = _look(sr, rng)
    bodies, polys = _place_bodies(sr, rng)

    if background is None:
        img = _scene(sr, rng)
    else:
        if background.shape != (st.height, st.width, 3):
            raise ValueError(f"background is {background.shape}, frame is {(st.height, st.width, 3)}")
        img = cv2.resize(background, (sr.width, sr.height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255

    wires = unary_union([_wire_shape(b.leads, 0.0) for b in bodies]) if bodies else None
    for clutter in _distractors(sr, rng, polys, rng.random() < sr.objects_fraction, wires):
        _draw(img, look, **clutter)
    instance_map = np.zeros((sr.height, sr.width), np.uint16)
    for i, body in enumerate(bodies, start=1):
        drawn = _draw(img, look, body.leads, body)
        if drawn is not None:
            x0, y0, a = drawn
            instance_map[y0:y0 + a.shape[0], x0:x0 + a.shape[1]][a > 0.5] = i

    polygons, ids = _labels(polys, sr)
    if st.camera_corners is not None:
        image = _through_camera(img, st, sr, rng)
    else:
        if sr is not st:
            img = cv2.resize(img, (st.width, st.height), interpolation=cv2.INTER_AREA)
        image = _camera(img, rng, st)
    if sr is not st:
        k = st.render_scale
        polygons = [(p + 0.5) / k - 0.5 for p in polygons]
        instance_map = cv2.resize(instance_map, (st.width, st.height), interpolation=cv2.INTER_NEAREST)
    return Frame(image, polygons, ids, instance_map)


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
        _write_jpg(job.out_dir / "preview" / file, visualise.draw_polygons(bgr, frame.polygons), 90)

    return {
        "image": f"synthetic/{job.name}/{file}",
        "session": job.name,
        "source": "synthetic",
        "space": "rectified",
        "width": job.settings.width,
        "height": job.settings.height,
        "instances": [
            {"polygon": manifest.polygon_to_json(p), "label_source": "synthetic"}
            for p in frame.polygons
        ],
    }


def main(argv=None):
    dataset_cfg = config.load("dataset")
    syn = dataset_cfg["synthetic"]

    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.synth",
        description="Generate synthetic top-down plate frames with exact body polygons.",
    )
    parser.add_argument("--frames", type=int, default=syn["frames"])
    parser.add_argument("--name", default="synth",
                        help="set name: frames go to data/synthetic/<name>/, also the session name")
    parser.add_argument("--seed", type=int, default=syn["seed"])
    parser.add_argument("--backgrounds", type=Path,
                        help="folder of rectified empty-plate images to draw on instead of the drawn plate")
    parser.add_argument("--workers", type=int, default=4, help="processes; 1 runs in this process")
    parser.add_argument("--preview", type=int, default=5,
                        help="write outlined previews of the first N frames to preview/")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name) or set(args.name) == {"."}:
        parser.error(f"--name {args.name!r}: use letters, digits, '_', '-' and '.' only")
    settings = Settings.from_config(dataset_cfg)

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

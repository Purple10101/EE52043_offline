"""Rectification: an angled camera frame to a straight-down view of the pick plate.

The camera looks at the plate from the tower at about 36° from vertical (camera_tower/README.md).
The plate's top surface is a known 100 × 100 mm square in the rig scene frame, so its four edges fix
a homography from the frame to a top-down image at a fixed scale: the rectified image. The model,
`synth` and tiling work on rectified images; labels stay in full-frame px and convert both ways.

1. Undistort (optional). With a lens calibration from `intrinsics`, the frame is undistorted
   first, because a homography assumes straight lines stay straight.
2. Find the plate. The bright, unsaturated blob gives a rough outline. Along each side, profiles
   across the edge find where the brightness first drops and stays low, and a robust line is
   fitted to those points. The four lines meet at the plate's corners, so its rounded corners don't
   matter. On the near side the first drop is the edge of the top surface; the plate's front face
   below it is in shadow.
3. Warp. One remap per frame, from rectified px straight back to raw frame px (homography, then
   lens distortion), so the frame is interpolated only once.

The rectified image shows the scene from above, scene +x to the right and +y up: the plate plus a
margin at px_per_mm (configs/dataset.toml [rectify]). It is exact only for the plate surface. A
body's centre sits 0.9 to 2.5 mm above it and appears shifted away from the camera (docs/plan.md).

The camera's manual white balance is not repeatable, so frames are taken with auto white balance
and `white_balance` scales the channels so the plate comes out neutral.

    python -m robot_offline.rectify calibrate <empty-plate frame> --out <session>/rectify.json
    python -m robot_offline.rectify apply <session>/rectify.json <frames...> --out <dir>
    python -m robot_offline.rectify intrinsics <checkerboard photos...> --square-mm 10 --out lens.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config

PROFILE_INSIDE_PX = 200     # each profile across a rough edge starts this far inside the plate, since
                            # the rough outline can reach down to the legs below the near edge
PROFILE_OUTSIDE_PX = 100    # and ends this far outside
PLATE_LEVEL_PX = 60         # the plate's brightness is the median of a profile's innermost part
PROFILE_STEP_PX = 0.5
EDGE_DROP = 0.75            # the edge is where brightness first falls below this share of the plate's
EDGE_HOLD_PX = 12           # and stays below it for this long, so a lead crossing the profile is skipped,
EDGE_BEYOND_PX = 40         # and is mostly below it (this share) for this long, so a lead lying at a
EDGE_BEYOND_DARK = 0.9      # shallow angle near the edge is skipped too
NEAR_BAND_PX = 60           # with the previous corners known, the edge is looked for this close to them
SIDE_SPAN = (0.2, 0.8)      # part of each side that is profiled, clear of the rounded corners
PROFILES_PER_SIDE = 120
MIN_INLIERS = 0.5           # share of a side's edge points that must agree on one line
WHITE_INSET_MM = 5.0        # plate border left out when measuring the plate's colour


class PlateNotFound(Exception):
    pass


@dataclass(frozen=True, eq=False)
class Settings:
    plate_corners_mm: np.ndarray    # 4 × 2 scene mm, in image order: top-left, top-right, bottom-right, bottom-left
    px_per_mm: float
    margin_mm: float
    corner_radius_mm: float = 4.0   # the plate's rounded corners

    @classmethod
    def from_config(cls, dataset=None):
        r = (dataset or config.load("dataset"))["rectify"]
        return cls(np.array(r["plate_corners_mm"], float), float(r["px_per_mm"]), float(r["margin_mm"]),
                   float(r["plate_corner_radius_mm"]))

    @property
    def origin_mm(self):
        """Scene (x, y) of the rectified image's top-left edge."""
        c = self.plate_corners_mm
        return float(c[:, 0].min() - self.margin_mm), float(c[:, 1].max() + self.margin_mm)

    @property
    def size(self):
        """Rectified image (width, height) in px."""
        span = self.plate_corners_mm.max(0) - self.plate_corners_mm.min(0) + 2 * self.margin_mm
        return int(round(span[0] * self.px_per_mm)), int(round(span[1] * self.px_per_mm))


def mm_to_rect(points_mm, settings):
    """Scene mm (N × 2) to rectified px. Pixel centres are at integer coordinates."""
    p = np.asarray(points_mm, float).reshape(-1, 2)
    x0, y1 = settings.origin_mm
    s = settings.px_per_mm
    return np.stack([(p[:, 0] - x0) * s - 0.5, (y1 - p[:, 1]) * s - 0.5], 1)


def rect_to_mm(points_px, settings):
    """Rectified px (N × 2) to scene mm on the plate surface."""
    p = np.asarray(points_px, float).reshape(-1, 2)
    x0, y1 = settings.origin_mm
    s = settings.px_per_mm
    return np.stack([x0 + (p[:, 0] + 0.5) / s, y1 - (p[:, 1] + 0.5) / s], 1)


@dataclass(eq=False)
class Calibration:
    """One session's mapping between raw frames and the rectified image."""

    homography: np.ndarray          # 3 × 3, undistorted frame px to rectified px
    corners_px: np.ndarray          # 4 × 2, plate corners found in the undistorted frame, image order
    frame_size: tuple[int, int]     # (width, height) of the raw frames
    settings: Settings
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None

    def save(self, path):
        data = {
            "version": 1,
            "frame_size": list(self.frame_size),
            "corners_px": self.corners_px.tolist(),
            "homography": self.homography.tolist(),
            "plate_corners_mm": self.settings.plate_corners_mm.tolist(),
            "px_per_mm": self.settings.px_per_mm,
            "margin_mm": self.settings.margin_mm,
            "plate_corner_radius_mm": self.settings.corner_radius_mm,
            "camera_matrix": None if self.camera_matrix is None else self.camera_matrix.tolist(),
            "dist_coeffs": None if self.dist_coeffs is None else self.dist_coeffs.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        settings = Settings(np.array(d["plate_corners_mm"], float), d["px_per_mm"], d["margin_mm"],
                            d.get("plate_corner_radius_mm", 4.0))
        k, dist = d["camera_matrix"], d["dist_coeffs"]
        return cls(np.array(d["homography"], float), np.array(d["corners_px"], float),
                   tuple(d["frame_size"]), settings,
                   None if k is None else np.array(k, float), None if dist is None else np.array(dist, float))


# --- Lens -----------------------------------------------------------------------------------------

def load_intrinsics(path):
    """(camera_matrix, dist_coeffs) from a file written by `intrinsics`."""
    d = json.loads(Path(path).read_text())
    return np.array(d["camera_matrix"], float), np.array(d["dist_coeffs"], float)


def calibrate_intrinsics(images, pattern, square_mm):
    """Lens calibration from photos of a flat checkerboard.

    pattern: inner corners (columns, rows). Returns (camera_matrix, dist_coeffs, rms_px, images_used).
    """
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    object_points, image_points, size = [], [], None
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
        if found:
            object_points.append(board)
            image_points.append(corners)
    if len(image_points) < 8:
        raise ValueError(f"the checkerboard was found in only {len(image_points)} images; need at least 8")
    rms, k, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    return k, dist.ravel(), float(rms), len(image_points)


def _undistort_points(points, cal):
    if cal.camera_matrix is None:
        return points
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    src = points.reshape(-1, 1, 2)
    k, dist = cal.camera_matrix, cal.dist_coeffs
    if hasattr(cv2, "undistortPointsIter"):        # OpenCV 4
        out = cv2.undistortPointsIter(src, k, dist, None, k, criteria)
    else:                                          # OpenCV 5 merged it into undistortPoints
        out = cv2.undistortPoints(src, k, dist, R=None, P=k, criteria=criteria)
    return out.reshape(-1, 2)


def _distort_points(points, cal):
    if cal.camera_matrix is None:
        return points
    k = cal.camera_matrix
    normalised = (points - (k[0, 2], k[1, 2])) / (k[0, 0], k[1, 1])
    rays = np.column_stack([normalised, np.ones(len(points))])
    out, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, cal.dist_coeffs)
    return out.reshape(-1, 2)


# --- Points ---------------------------------------------------------------------------------------

def to_rectified(points_px, cal):
    """Raw full-frame px (N × 2) to rectified px."""
    p = _undistort_points(np.asarray(points_px, np.float64).reshape(-1, 2), cal)
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), cal.homography).reshape(-1, 2)


def to_full(points_px, cal):
    """Rectified px (N × 2) to raw full-frame px."""
    p = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(p, np.linalg.inv(cal.homography)).reshape(-1, 2)
    return _distort_points(p, cal)


def plate_outline_px(cal):
    """The plate's corners in raw full-frame px (4 × 2, image order)."""
    return _distort_points(cal.corners_px, cal)


def to_scene_mm(points_px, cal):
    """Raw full-frame px (N × 2) to scene mm on the plate surface."""
    return rect_to_mm(to_rectified(points_px, cal), cal.settings)


# --- Finding the plate ----------------------------------------------------------------------------

def _image_order(quad):
    """Four points as top-left, top-right, bottom-right, bottom-left."""
    c = quad.mean(0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
    return np.roll(quad, -int(np.argmin(quad.sum(1))), 0)


def _rough_outline(frame):
    """Rough plate corners from the largest bright, unsaturated blob."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    bright = ((v > 0.6 * np.percentile(v, 99)) & (s < 60)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise PlateNotFound("no bright, unsaturated region in the frame")
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    for eps in np.linspace(0.01, 0.1, 19):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            return _image_order(quad.reshape(4, 2).astype(float))
    raise PlateNotFound("the bright region is not a quadrilateral")


def _edge_points(gray, a, b, centre, band=None):
    """Points on the plate edge near the rough side a to b, from profiles across it.

    band: if given, the edge must lie within this many px of the rough side.
    """
    side = (b - a) / np.linalg.norm(b - a)
    out = np.array([-side[1], side[0]])
    if np.dot(out, (a + b) / 2 - centre) < 0:
        out = -out
    base = a + np.linspace(*SIDE_SPAN, PROFILES_PER_SIDE)[:, None] * (b - a)
    d = np.arange(-PROFILE_INSIDE_PX, PROFILE_OUTSIDE_PX + PROFILE_STEP_PX / 2, PROFILE_STEP_PX)
    xy = base[:, None, :] + d[None, :, None] * out
    profiles = cv2.remap(gray, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hold = int(EDGE_HOLD_PX / PROFILE_STEP_PX)
    beyond = int(EDGE_BEYOND_PX / PROFILE_STEP_PX)
    inside = d < -PROFILE_INSIDE_PX + PLATE_LEVEL_PX
    allowed = np.ones(len(d), bool) if band is None else np.abs(d) <= band
    points = []
    for p0, row in zip(base, profiles):
        below = row < EDGE_DROP * np.median(row[inside])
        # Dark for EDGE_HOLD_PX, and mostly dark for EDGE_BEYOND_PX: a lead lying across the
        # profile at a shallow angle can pass the first test but the plate shows again after it.
        runs = np.convolve(below, np.ones(hold, int), "valid")[:len(d) - beyond] == hold
        after = np.convolve(below, np.ones(beyond), "valid")[:len(runs)] / beyond >= EDGE_BEYOND_DARK
        hits = np.nonzero(runs & after & allowed[:len(runs)])[0]
        if len(hits) == 0:
            continue
        i = hits[0]
        lo, hi = max(i - 8, 1), min(i + 8, len(row) - 2)
        grad = np.gradient(row)
        k = lo + int(np.argmin(grad[lo:hi + 1]))
        g0, g1, g2 = grad[k - 1], grad[k], grad[k + 1]
        denom = g0 - 2 * g1 + g2
        offset = 0.5 * (g0 - g2) / denom if denom > 0 else 0.0
        points.append(p0 + (d[k] + offset * PROFILE_STEP_PX) * out)
    return np.array(points)


def _fit_line(points):
    """Robust line through edge points: (point, unit direction, inlier share, inlier rms px)."""
    if len(points) < PROFILES_PER_SIDE * MIN_INLIERS:
        raise PlateNotFound(f"only {len(points)} of {PROFILES_PER_SIDE} profiles found an edge")
    # A Huber fit first, so profiles that hit a resistor body don't drag the line, then least squares
    # on the points within 3 sigma (at least 1 px) of it.
    keep = np.ones(len(points), bool)
    kind = cv2.DIST_HUBER
    for _ in range(5):
        vx, vy, x0, y0 = cv2.fitLine(points[keep].astype(np.float32), kind, 0, 0.01, 0.01).ravel()
        r = (points - (x0, y0)) @ np.array([-vy, vx])
        spread = 1.4826 * np.median(np.abs(r - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < max(3 * spread, 1.0)
        if keep.sum() < 2:
            raise PlateNotFound("a side's edge points don't lie on a line")
        kind = cv2.DIST_L2
    share = keep.sum() / PROFILES_PER_SIDE
    if share < MIN_INLIERS:
        raise PlateNotFound(f"only {share:.0%} of a side's edge points lie on one line")
    return np.array([x0, y0], float), np.array([vx, vy], float), share, float(np.sqrt(np.mean(r[keep] ** 2)))


def _intersect(l1, l2):
    (p1, d1), (p2, d2) = l1[:2], l2[:2]
    t, _ = np.linalg.solve(np.column_stack([d1, -d2]), p2 - p1)
    return p1 + t * d1


def find_plate_corners(frame, near=None):
    """The plate's top-surface corners in a BGR frame, in image order, and per-side fit details.

    near: corners where the plate was last found (4 × 2 px). The edges are then looked for around
    them instead of around the bright blob, whose outline leads hanging off the plate can distort.
    Returns (corners 4 × 2 px, sides), sides being (inlier share, rms px) for top, right, bottom, left.
    """
    rough = _rough_outline(frame) if near is None else np.asarray(near, float)
    band = None if near is None else NEAR_BAND_PX
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    centre = rough.mean(0)
    lines = [_fit_line(_edge_points(gray, rough[i], rough[(i + 1) % 4], centre, band)) for i in range(4)]
    top, right, bottom, left = lines
    corners = np.array([_intersect(left, top), _intersect(top, right),
                        _intersect(right, bottom), _intersect(bottom, left)])
    return corners, [(line[2], line[3]) for line in lines]


def calibrate(frame, settings=None, intrinsics=None):
    """Calibration from one frame with the plate clearly visible (ideally empty).

    intrinsics: (camera_matrix, dist_coeffs) or None. Returns (Calibration, sides) with sides as
    from find_plate_corners.
    """
    settings = settings or Settings.from_config()
    k, dist = intrinsics if intrinsics is not None else (None, None)
    undistorted = frame if k is None else cv2.undistort(frame, k, dist)
    corners, sides = find_plate_corners(undistorted)
    target = mm_to_rect(settings.plate_corners_mm, settings)
    h = cv2.getPerspectiveTransform(corners.astype(np.float32), target.astype(np.float32))
    return Calibration(h.astype(float), corners, (frame.shape[1], frame.shape[0]), settings, k, dist), sides


def with_corners(cal, corners):
    """A calibration like cal (scale, lens) but with the plate at other corners (undistorted px).

    For a frame taken after the camera moved: its own plate corners give its own homography.
    """
    target = mm_to_rect(cal.settings.plate_corners_mm, cal.settings)
    h = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), target.astype(np.float32))
    return Calibration(h.astype(float), np.asarray(corners, float), cal.frame_size, cal.settings,
                       cal.camera_matrix, cal.dist_coeffs)


def recalibrate(frame, cal):
    """The calibration for one frame: the plate found again near cal's corners. Raises PlateNotFound.

    Returns (Calibration, sides).
    """
    k, dist = cal.camera_matrix, cal.dist_coeffs
    undistorted = frame if k is None else cv2.undistort(frame, k, dist)
    corners, sides = find_plate_corners(undistorted, near=cal.corners_px)
    return with_corners(cal, corners), sides


# --- Images ---------------------------------------------------------------------------------------

def _maps(cal):
    """Remap tables from rectified px to raw frame px, built once per calibration."""
    cached = getattr(cal, "_remap", None)
    if cached is None:
        w, h = cal.settings.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        src = to_full(np.column_stack([u.ravel(), v.ravel()]), cal)
        cached = (src[:, 0].reshape(h, w).astype(np.float32), src[:, 1].reshape(h, w).astype(np.float32))
        cal._remap = cached
    return cached


def rectify(frame, cal, interpolation=cv2.INTER_LINEAR):
    """The rectified image of a raw frame."""
    if (frame.shape[1], frame.shape[0]) != tuple(cal.frame_size):
        raise ValueError(f"frame is {frame.shape[1]} × {frame.shape[0]}, calibration is for "
                         f"{cal.frame_size[0]} × {cal.frame_size[1]}")
    mx, my = _maps(cal)
    return cv2.remap(frame, mx, my, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


_MASKS = {}


def plate_mask(settings):
    """H×W bool: the rectified pixels whose centres lie on the plate's top surface (rounded corners
    included). Cached per geometry."""
    key = (settings.size, tuple(settings.plate_corners_mm.ravel()), settings.px_per_mm,
           settings.margin_mm, settings.corner_radius_mm)
    if key not in _MASKS:
        w, h = settings.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        xy = rect_to_mm(np.column_stack([u.ravel(), v.ravel()]), settings)
        c = settings.plate_corners_mm
        centre, half = (c.min(0) + c.max(0)) / 2, (c.max(0) - c.min(0)) / 2
        r = settings.corner_radius_mm
        q = np.abs(xy - centre) - (half - r)            # rounded-rectangle distance, outside > 0
        outside = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - r
        _MASKS[key] = (outside <= 0).reshape(h, w)
    return _MASKS[key]


def mask_outside_plate(rectified, settings):
    """A copy of a rectified (and white-balanced) image with everything off the plate replaced by
    the plate's own grey, so a model given it can take nothing from the surroundings.

    The fill is the median colour of the plate's brighter half (resistors and marks left out), so
    the plate's edge disappears into it. Training and inference both use this.
    """
    inside = plate_mask(settings)
    pixels = rectified[inside].reshape(-1, rectified.shape[2]).astype(np.float32)
    lum = pixels.mean(1)
    fill = np.median(pixels[lum >= np.median(lum)], axis=0)
    out = rectified.copy()
    out[~inside] = np.round(fill).astype(rectified.dtype)
    return out


def model_input(frame, cal):
    """What the model sees for a raw camera frame: rectified, white-balanced, off-plate masked."""
    rect, _ = white_balance(rectify(frame, cal), cal.settings)
    return mask_outside_plate(rect, cal.settings)


def white_balance(rectified, settings):
    """Channels scaled so the plate is neutral. Returns (image, gains as B, G, R).

    The plate's colour is the median of its brighter half, which leaves out resistors and marks.
    """
    c = settings.plate_corners_mm
    inner = np.array([[c[:, 0].min() + WHITE_INSET_MM, c[:, 1].max() - WHITE_INSET_MM],
                      [c[:, 0].max() - WHITE_INSET_MM, c[:, 1].min() + WHITE_INSET_MM]])
    (u0, v0), (u1, v1) = mm_to_rect(inner, settings).round().astype(int)
    region = rectified[v0:v1, u0:u1].reshape(-1, 3).astype(np.float32)
    lum = region.mean(1)
    plate = np.median(region[lum >= np.median(lum)], axis=0)
    gains = plate.mean() / np.maximum(plate, 1.0)
    out = np.clip(rectified.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return out, gains


# --- Command line ---------------------------------------------------------------------------------

def _draw_check(frame, cal, stem):
    """Two images to check a calibration by eye: the fitted plate on the frame, and the rectified
    image with a 10 mm grid in scene coordinates (the plate's edges should sit on the red outline)."""
    shown = frame.copy()
    corners = plate_outline_px(cal)
    cv2.polylines(shown, [corners.round().astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
    for (x, y) in corners:
        cv2.circle(shown, (int(round(x)), int(round(y))), 12, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(f"{stem}_frame.jpg", cv2.resize(shown, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    rect = white_balance(rectify(frame, cal), cal.settings)[0]
    cv2.imwrite(f"{stem}_rectified.jpg", draw_grid(rect, cal.settings))


def draw_grid(rectified, settings, step_mm=10.0):
    """A copy of a rectified image with a grid every step_mm in scene coordinates and the plate's
    outline in red. The plate's straight edges should sit on the outline."""
    out = rectified.copy()
    c = settings.plate_corners_mm
    lo, hi = c.min(0) - settings.margin_mm, c.max(0) + settings.margin_mm
    for x in np.arange(np.ceil(lo[0] / step_mm) * step_mm, hi[0] + 1e-6, step_mm):
        (u, _), = mm_to_rect([[x, 0.0]], settings)
        cv2.line(out, (int(round(u)), 0), (int(round(u)), out.shape[0]), (255, 160, 0), 1)
    for y in np.arange(np.ceil(lo[1] / step_mm) * step_mm, hi[1] + 1e-6, step_mm):
        (_, v), = mm_to_rect([[0.0, y]], settings)
        cv2.line(out, (0, int(round(v))), (out.shape[1], int(round(v))), (255, 160, 0), 1)
    cv2.polylines(out, [mm_to_rect(c, settings).round().astype(np.int32)], True, (0, 0, 255), 1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Straight-down view of the pick plate.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate", help="find the plate in a frame and save the session's calibration")
    p.add_argument("frame", help="raw frame with the plate clearly visible, ideally empty")
    p.add_argument("--out", required=True, help="calibration file to write (.json)")
    p.add_argument("--intrinsics", help="lens calibration from `intrinsics`")
    p = sub.add_parser("apply", help="rectify frames with a saved calibration")
    p.add_argument("calibration")
    p.add_argument("frames", nargs="+")
    p.add_argument("--out", required=True, help="folder for the rectified PNGs")
    p.add_argument("--no-white-balance", action="store_true", help="keep the camera's colours")
    p = sub.add_parser("intrinsics", help="lens calibration from checkerboard photos")
    p.add_argument("images", nargs="+")
    p.add_argument("--pattern", default="9x6", help="inner corners, columns x rows (default 9x6)")
    p.add_argument("--square-mm", type=float, required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        frame = cv2.imread(args.frame)
        if frame is None:
            raise SystemExit(f"cannot read {args.frame}")
        intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics else None
        cal, sides = calibrate(frame, intrinsics=intrinsics)
        cal.save(args.out)
        stem = str(Path(args.out).with_suffix(""))
        _draw_check(frame, cal, stem)
        print(f"Plate corners (px): {np.round(cal.corners_px, 1).tolist()}")
        for name, (share, rms) in zip(("top", "right", "bottom", "left"), sides):
            flag = "" if share >= 0.8 and rms < 1.5 else "   <- check this edge"
            print(f"  {name:6s} edge: {share:4.0%} of points on the line, rms {rms:.2f} px{flag}")
        print(f"Wrote {args.out}; check {stem}_frame.jpg and {stem}_rectified.jpg")

    elif args.command == "apply":
        cal = Calibration.load(args.calibration)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in args.frames:
            frame = cv2.imread(name)
            if frame is None:
                print(f"skipped {name}: cannot read")
                continue
            rect = rectify(frame, cal)
            if not args.no_white_balance:
                rect, _ = white_balance(rect, cal.settings)
            cv2.imwrite(str(out / (Path(name).stem + ".png")), rect)
        print(f"Wrote {len(args.frames)} rectified images ({cal.settings.size[0]} × {cal.settings.size[1]}) to {out}")

    else:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
        images = [cv2.imread(n) for n in args.images]
        k, dist, rms, used = calibrate_intrinsics([i for i in images if i is not None], (cols, rows), args.square_mm)
        h, w = images[0].shape[:2]
        Path(args.out).write_text(json.dumps({"camera_matrix": k.tolist(), "dist_coeffs": dist.tolist(),
                                              "frame_size": [w, h], "rms_px": rms, "images_used": used}, indent=2))
        print(f"Lens calibration from {used} of {len(args.images)} images, rms {rms:.2f} px: wrote {args.out}")


if __name__ == "__main__":
    main()

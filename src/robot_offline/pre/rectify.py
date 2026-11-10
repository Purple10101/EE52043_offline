"""Rectification: an angled camera frame to a straight-down view of the pick plate.

The camera looks at the plate from the tower at about 36° from vertical (camera_tower/README.md).
The plate is a known solid (the rig model, ../arm_and_rig_scene/EduSCARA_resistorrig.obj, body
Body1:1): a 100 × 100 mm top in the rig scene frame, a 5 mm front face below it, and a crosshair
groove at its centre. Fitting where the camera must be for that solid to look as it does fixes a
homography from the frame to a top-down image at a fixed scale: the rectified image. The model,
`synth` and tiling work on rectified images; labels stay in full-frame px and convert both ways.

1. Find the plate. A rough outline (the largest bright blob, after a grey-world colour balance)
   gives a first camera pose. Profiles across the far edge and the two side edges of the top find
   where the plate falls away to its darker surroundings, and the pose is fitted to them.
2. Fix the near side. The near edge of the top is unreliable: the front face below it can be
   lit like the top or darker, in another colour. So the near side is fixed by the crosshair,
   found as the midpoint between each arm's two lips (which part of the groove is lit depends on
   the light, the lips don't). If something covers the crosshair, the bottom edge of the front face
   is used instead: the face against the dark gap under the plate, 5 mm below the top. The camera
   pose and focal length are fitted to all of it, searched again around the fitted model and
   refitted. Whichever of the crosshair, the face bottom and the near top edge is not used is
   reported as a check.
3. Warp. One remap per frame, from rectified px straight back to raw frame px (homography, then
   lens distortion), so the frame is interpolated only once. With a lens calibration from
   `intrinsics`, its focal length and distortion are used as they are and only the pose is fitted.

The rectified image shows the scene from above, scene +x to the right and +y up: the plate plus a
margin at px_per_mm (configs/dataset.toml [rectify]). It is exact only for the plate surface. A
body's centre sits 0.9 to 2.5 mm above it and appears shifted away from the camera (docs/plan.md).

The camera's manual white balance is not repeatable, so frames are taken with auto white balance
and `white_balance` scales the channels so the plate comes out neutral.

    python -m robot_offline.pre.rectify calibrate <empty-plate frame> --out <session>/rectify.json
    python -m robot_offline.pre.rectify apply <session>/rectify.json <frames...> --out <dir>
    python -m robot_offline.pre.rectify intrinsics <checkerboard photos...> --square-mm 10 --out lens.json
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config

PROFILE_STEP_PX = 0.5
EDGE_DROP = 0.75            # an edge: the dark side is below this share of the lit side's brightness,
EDGE_LIT_PX = 24            # the lit side being the profile over this length just inside the edge,
EDGE_HALF_WIDTH_PX = 6      # this far from the edge's middle (the camera blurs an edge over about 10 px)
EDGE_SLOPE_PX = 1.5         # an edge's slope is measured with this much smoothing (Gaussian sigma)
EDGE_MIN_FALL = 0.1         # an edge falls by at least this share of the plate's brightness
EDGE_NEAREST_BAND_PX = 20   # searching no wider than this, the fall nearest the model is the edge; wider,
EDGE_LINE_PX = 3.0          # the straight line of falls most profiles agree on, to within this
EDGE_HOLD_PX = 12           # the dark side stays below the midway level for this long, so a lead crossing
EDGE_BEYOND_PX = 40         # the profile is skipped, and is mostly dark for this long, so a lead lying at
EDGE_BEYOND_DARK = 0.8      # a shallow angle near the edge is skipped too (share of the way to dark)
FIRST_BAND_PX = 80          # first search for the top edges, around the pose from the rough outline
NEAR_BAND_PX = 40           # or around the pose from the previous calibration
REFINE_BAND_PX = 10         # later searches, around the fitted model
EDGE_SPAN_MM = 40.0         # profiles cover each top edge to this far either side of its middle, clear of
                            # the rounded corners
FACE_SPAN_MM = 25.0         # and the face bottom to this far, clear of the legs and their fillets
PROFILES_PER_EDGE = 120
NEAR_EDGE_CONTRAST = 10.0   # Lab difference across the near top edge for it to count as found (check only)
CROSS_SPAN_MM = 2.0         # profiles across a crosshair arm reach this far either side of its middle line
CROSS_CLEAR_MM = 1.1        # (the plate's colour is measured beyond this, the groove looked for within it),
CROSS_ALONG_MM = (0.9, 2.0) # and lie this far along the arm from the centre: clear of the other arm and
CROSS_PROFILES = 6          # the arm's end, this many per half arm
CROSS_LIP_LEVEL = 0.3       # a lip: where the colour is this share of the way to the groove's strongest
CROSS_MIN_CONTRAST = 6.0    # and at least this far (Lab) from the plate's
CROSS_WIDTH_MM = (0.5, 1.8) # lip to lip, for a 1 mm groove seen through blur
CROSS_MIN_FOUND = 0.6       # share of profiles in every half arm that must find the groove, else it is
CROSS_MAX_RMS_PX = 1.5      # taken as covered; so is a crosshair that fits the model worse than this
CROSS_SEARCH_MM = 6.0       # with a top edge missing, the crosshair is searched for this far round the model,
CROSS_SEARCH_STEP_MM = 1.0  # in steps of this (under the groove's search width, 2 * CROSS_CLEAR_MM),
CROSS_REFITS = 6            # then found again round each fit, at most this many times, until the model
CROSS_SETTLED_PX = 0.2      # moves the crosshair less than this
MIN_INLIERS = 0.5           # share of an edge's profiles that must find it on the fitted model
MAX_RMS_PX = 2.0            # an edge further than this (rms) from the fitted model means a bad fit
FIRST_MIN_FOUND = 0.3       # share of each top edge the first search must find (the rough pose can be off)
FOCAL_RANGE = 0.2           # a fitted focal length further than this share from Settings.focal_px is rejected
WHITE_INSET_MM = 5.0        # plate border left out when measuring the plate's colour
MODEL_PLATE_LEVEL = 200.0   # the plate's grey in the model's input: synth draws it there (synth.PLATE_RGB)


class PlateNotFound(Exception):
    pass


@dataclass(frozen=True, eq=False)
class Settings:
    plate_corners_mm: np.ndarray    # 4 × 2 scene mm, in image order: top-left, top-right, bottom-right, bottom-left
    px_per_mm: float
    margin_mm: float
    corner_radius_mm: float = 5.0   # the plate's rounded corners
    top_z_mm: float = -22.0         # height of the plate's top in the scene frame
    face_mm: float = 5.0            # height of its front face: the slab's underside is this far below the top
    cross_mm: tuple = (0.0, -130.0) # centre of the crosshair groove in the top
    cross_arm_mm: float = 5.0       # length of each of its two arms
    camera_mm: tuple = (-145.0, -145.0, 178.0)  # the lens by design, where the fit starts
    focal_px: float = 3600.0        # the lens's focal length in raw px at the camera.toml focus, where the fit starts

    @classmethod
    def from_config(cls, dataset=None):
        return cls.from_dict((dataset or config.load("dataset"))["rectify"])

    def to_dict(self):
        return {"plate_corners_mm": self.plate_corners_mm.tolist(), "px_per_mm": self.px_per_mm,
                "margin_mm": self.margin_mm, "plate_corner_radius_mm": self.corner_radius_mm,
                "plate_top_z_mm": self.top_z_mm, "plate_face_mm": self.face_mm,
                "crosshair_mm": list(self.cross_mm), "crosshair_arm_mm": self.cross_arm_mm,
                "camera_mm": list(self.camera_mm), "focal_px": self.focal_px}

    @classmethod
    def from_dict(cls, d):
        """Settings from configs/dataset.toml [rectify] or saved with a calibration. Calibrations from
        before the plate model have only the first four keys; the rest take the defaults above (and
        the 4 mm corner radius they were made with stays)."""
        defaults = cls(np.zeros((4, 2)), 0.0, 0.0)
        return cls(np.array(d["plate_corners_mm"], float), float(d["px_per_mm"]), float(d["margin_mm"]),
                   float(d.get("plate_corner_radius_mm", 4.0)), float(d.get("plate_top_z_mm", defaults.top_z_mm)),
                   float(d.get("plate_face_mm", defaults.face_mm)),
                   tuple(map(float, d.get("crosshair_mm", defaults.cross_mm))),
                   float(d.get("crosshair_arm_mm", defaults.cross_arm_mm)),
                   tuple(map(float, d.get("camera_mm", defaults.camera_mm))), float(d.get("focal_px", defaults.focal_px)))

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
    corners_px: np.ndarray          # 4 × 2, plate corners in the undistorted frame, image order
    frame_size: tuple[int, int]     # (width, height) of the raw frames
    settings: Settings
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    fit: dict | None = None         # how the plate was found (PlateFit.summary), for the record

    def save(self, path):
        data = {
            "version": 2,
            "frame_size": list(self.frame_size),
            "corners_px": self.corners_px.tolist(),
            "homography": self.homography.tolist(),
            **self.settings.to_dict(),
            "camera_matrix": None if self.camera_matrix is None else self.camera_matrix.tolist(),
            "dist_coeffs": None if self.dist_coeffs is None else self.dist_coeffs.tolist(),
            "fit": self.fit,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        k, dist = d["camera_matrix"], d["dist_coeffs"]
        return cls(np.array(d["homography"], float), np.array(d["corners_px"], float),
                   tuple(d["frame_size"]), Settings.from_dict(d),
                   None if k is None else np.array(k, float), None if dist is None else np.array(dist, float),
                   d.get("fit"))


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


# Masks for the rough outline, tried in turn: (brightness above this share of the 99th percentile,
# saturation below this). The strict one suits even light; the loose one keeps a strip of the plate
# lit by a different light, which the strict one can cut off.
ROUGH_MASKS = ((0.6, 60), (0.5, 100))


def _rough_outlines(frame):
    """Rough plate corners, one set per ROUGH_MASKS entry that gives any: from the largest bright,
    unsaturated blob.

    The frame is first balanced so its brightest tenth is grey: under warm light the white plate is
    too saturated to pass as unsaturated otherwise.
    """
    f = frame.astype(np.float32)
    lum = f.mean(2)
    bright = f[lum >= np.percentile(lum, 90)].mean(0)
    balanced = np.clip(f * (bright.mean() / np.maximum(bright, 1.0)), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    out = []
    for level, saturation in ROUGH_MASKS:
        mask = ((v > level * np.percentile(v, 99)) & (s < saturation)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        for eps in np.linspace(0.01, 0.1, 19):
            quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(quad) == 4:
                out.append(_image_order(quad.reshape(4, 2).astype(float)))
                break
    if not out:
        raise PlateNotFound("no bright, unsaturated four-sided region in the frame")
    return out


@dataclass
class _Feature:
    """Where one model feature was found: profiles across it, and the points they found."""
    name: str
    model: np.ndarray       # N × 3 scene mm: the points on the feature that the found points belong to
    along: np.ndarray       # N × 3 unit directions along the feature there
    found: np.ndarray       # N × 2 raw frame px
    tried: int              # profiles looked at
    keep: np.ndarray = None # points that agree with the fitted model

    def __post_init__(self):
        if self.keep is None:
            self.keep = np.ones(len(self.found), bool)

    @property
    def share(self):
        return self.keep.sum() / self.tried


class _Camera:
    """Pose and focal length as one parameter vector for fitting: p = (rvec 3, tvec 3, focal px).

    With intrinsics (a lens calibration) the focal length and distortion are that calibration's.
    Without, the principal point is the frame's centre and the lens is taken as distortion-free.
    """

    def __init__(self, frame_size, focal, intrinsics=None):
        self.centre = (frame_size[0] / 2, frame_size[1] / 2)
        self.intrinsics = intrinsics
        self.p = np.array([0, 0, 0, 0, 0, 0, focal], float)

    def matrices(self, p=None):
        p = self.p if p is None else p
        if self.intrinsics is not None:
            return self.intrinsics
        return np.array([[p[6], 0, self.centre[0]], [0, p[6], self.centre[1]], [0, 0, 1]]), np.zeros(5)

    def project(self, points, p=None):
        p = self.p if p is None else p
        k, dist = self.matrices(p)
        out, _ = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 1, 3), p[:3], p[3:6], k, dist)
        return out.reshape(-1, 2)

    @property
    def position_mm(self):
        r = cv2.Rodrigues(self.p[:3])[0]
        return -r.T @ self.p[3:6]


def _top_square(st, z=None):
    return np.column_stack([st.plate_corners_mm, np.full(4, st.top_z_mm if z is None else z)])


def _start_pose(cam, corners_px, st):
    """The camera pose that puts the plate's top corners at corners_px (raw px, image order).

    Of the two poses a square allows, the one with the camera above the plate and nearer its
    design position.
    """
    k, dist = cam.matrices()
    local = np.column_stack([st.plate_corners_mm, np.zeros(4)])    # solvePnP's planar solver wants z = 0
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(local, np.asarray(corners_px, np.float64).reshape(-1, 1, 2),
                                             k, dist, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        r = cv2.Rodrigues(rvec)[0]
        tvec = tvec.ravel() - r @ (0, 0, st.top_z_mm)               # back to scene z
        position = -r.T @ tvec
        score = np.linalg.norm(position - st.camera_mm) + (1e6 if position[2] < st.top_z_mm else 0)
        if best is None or score < best[0]:
            best = (score, rvec.ravel(), tvec)
    cam.p[:6] = np.concatenate([best[1], best[2]])


def _sides(st):
    """The top's four sides in image order (top, right, bottom, left) as (middle, along, outward)
    unit vectors in scene mm, and the index of the side nearest the camera: the one whose face shows."""
    c, centre = st.plate_corners_mm, st.plate_corners_mm.mean(0)
    out = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        mid, along = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
        outward = np.array([-along[1], along[0]])
        if np.dot(outward, mid - centre) < 0:
            outward = -outward
        out.append((mid, along, outward))
    near = int(np.argmin([np.linalg.norm(mid - st.camera_mm[:2]) for mid, _, _ in out]))
    return out, near


def _profiles(image, cam, model, along, outward, half_px):
    """Image profiles across a model feature: one per model point, along the projected outward
    direction, from half_px inside to half_px outside. Returns (offsets px, profiles, image normals)."""
    q = cam.project(model)
    t = cam.project(model + 0.5 * along) - q
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    flip = np.sum(n * (cam.project(model + 0.5 * outward) - q), 1) < 0
    n[flip] = -n[flip]
    d = np.arange(-half_px, half_px + PROFILE_STEP_PX / 2, PROFILE_STEP_PX)
    xy = q[:, None, :] + d[None, :, None] * n[:, None, :]
    rows = cv2.remap(image, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return d, rows, q, n


def _windows(rows, start, length):
    """Mean of rows[:, k + start : k + start + length] for every k (NaN where it runs off the end)."""
    c = np.concatenate([np.zeros((rows.shape[0], 1)), np.cumsum(rows, 1)], 1)
    k = np.arange(rows.shape[1])
    a, b = k + start, k + start + length
    ok = (a >= 0) & (b <= rows.shape[1])
    out = np.full(rows.shape, np.nan)
    out[:, ok] = (c[:, b[ok]] - c[:, a[ok]]) / length
    return out


def _falls(rows, d, band, lit_px=EDGE_LIT_PX):
    """For each profile, where within band a lit surface falls to a dark one and stays dark (so not a
    lead crossing it): its position along d, or NaN.

    A profile can cross several such falls: a shadow cast on the plate inside its edge, the plate's
    own shadow on the bench outside it. What sets the edge apart is that it is straight all along the
    side, so with a wide band (the first search, the model still rough) the falls are chosen as the
    straight line most profiles agree on; with a narrow one (the model close) the fall nearest the
    model is taken.

    Only the steepest points of falls are candidates, and only those at least half as steep as their
    own step blurred over 2 * EDGE_HALF_WIDTH_PX: else noise just before an edge would pass (its dark
    side being the edge's), and so would a shadow darkening the plate gradually. The slope is measured
    over EDGE_SLOPE_PX, so that noise in a dim, high-gain frame doesn't make a gradual slope look steep
    in places. A fall must also be at least EDGE_MIN_FALL of the plate's brightness (the far, inner end
    of the profile), so that noise in a dark background doesn't count."""
    lit_n, gap = int(lit_px / PROFILE_STEP_PX), int(EDGE_HALF_WIDTH_PX / PROFILE_STEP_PX)
    hold, beyond = int(EDGE_HOLD_PX / PROFILE_STEP_PX), int(EDGE_BEYOND_PX / PROFILE_STEP_PX)
    lit = _windows(rows, -gap - lit_n, lit_n)
    dark = _windows(rows, gap, hold)
    after = _windows(rows, gap, beyond)
    peak = cv2.dilate(rows.astype(np.float32), np.ones((1, hold), np.uint8), anchor=(0, 0))
    hold_max = np.full(rows.shape, np.nan)
    hold_max[:, :rows.shape[1] - gap] = peak[:, gap:]
    sigma = EDGE_SLOPE_PX / PROFILE_STEP_PX
    kernel = cv2.getGaussianKernel(2 * int(3 * sigma) + 1, sigma).T
    g = np.gradient(cv2.filter2D(rows.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE), axis=1)
    steepest = np.zeros(rows.shape, bool)
    steepest[:, 1:-1] = (g[:, 1:-1] <= g[:, :-2]) & (g[:, 1:-1] < g[:, 2:]) & (g[:, 1:-1] < 0)
    plate = np.median(rows[:, :lit_n], axis=1)[:, None]
    with np.errstate(invalid="ignore"):
        mid = (lit + dark) / 2
        steep = g < -(lit - dark) * PROFILE_STEP_PX / (4 * EDGE_HALF_WIDTH_PX)
        ok = (steepest & steep & (dark < EDGE_DROP * lit) & (lit - dark >= EDGE_MIN_FALL * plate)
              & (hold_max < mid) & (after < dark + (1 - EDGE_BEYOND_DARK) * (lit - dark))
              & (np.abs(d) <= band)[None, :])
    profile, k = np.nonzero(ok)
    g0, g1, g2 = g[profile, k - 1], g[profile, k], g[profile, k + 1]
    denom = g0 - 2 * g1 + g2
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = np.where(denom > 0, 0.5 * (g0 - g2) / denom, 0.0)
    position = d[0] + (k + offset) * PROFILE_STEP_PX
    out = np.full(len(rows), np.nan)
    if len(position) == 0:
        return out
    if band <= EDGE_NEAREST_BAND_PX:
        line = np.zeros(2)
    else:
        line = _straightest(profile, position)
    residual = np.abs(position - (line[0] + line[1] * profile))
    for i in np.unique(profile):
        mine = profile == i
        j = int(np.argmin(np.where(mine, residual, np.inf)))
        if band <= EDGE_NEAREST_BAND_PX or residual[j] <= EDGE_LINE_PX:
            out[i] = position[j]
    return out


def _straightest(profile, position, tries=300):
    """(a, b): the line position = a + b * profile through candidate falls (one profile may have
    several) that the most profiles have a fall within EDGE_LINE_PX of. RANSAC, seeded, so a frame
    always gives the same answer."""
    rng = np.random.default_rng(0)
    best, best_line = (-1, np.inf), np.zeros(2)
    if len(np.unique(profile)) < 2:
        return np.array([position[0], 0.0])
    for _ in range(tries):
        i, j = rng.choice(len(profile), 2, replace=False)
        if profile[i] == profile[j]:
            continue
        b = (position[j] - position[i]) / (profile[j] - profile[i])
        a = position[i] - b * profile[i]
        residual = np.abs(position - (a + b * profile))
        near = residual <= EDGE_LINE_PX
        score = (len(np.unique(profile[near])), -float(residual[near].mean()))
        if score > best:
            best, best_line = score, np.array([a, b])
    return best_line


def _colour_steps(rows_lab, d, band):
    """For each profile, the strongest colour change within band (Lab, any direction): its position
    along d, or NaN where the change is weaker than NEAR_EDGE_CONTRAST."""
    w = int(3 / PROFILE_STEP_PX)
    before = np.stack([_windows(rows_lab[..., c], -2 * w, w) for c in range(3)], -1)
    after = np.stack([_windows(rows_lab[..., c], w, w) for c in range(3)], -1)
    with np.errstate(invalid="ignore"):
        contrast = np.linalg.norm(after - before, axis=-1)
    contrast[:, np.abs(d) > band] = np.nan
    out = np.full(len(rows_lab), np.nan)
    for i in range(len(rows_lab)):
        if np.all(np.isnan(contrast[i])) or np.nanmax(contrast[i]) < NEAR_EDGE_CONTRAST:
            continue
        k = int(np.nanargmax(contrast[i]))
        lo, hi = max(k - 1, 0), min(k + 1, len(d) - 1)
        c0, c1, c2 = contrast[i, lo], contrast[i, k], contrast[i, hi]
        denom = c0 - 2 * c1 + c2
        off = 0.5 * (c0 - c2) / denom if lo < k < hi and denom < 0 else 0.0
        out[i] = d[0] + (k + off) * PROFILE_STEP_PX
    return out


def _edge(name, gray, cam, st, mid, along, outward, span_mm, z, band, finder):
    s = np.linspace(-span_mm, span_mm, PROFILES_PER_EDGE)
    model = np.column_stack([mid + s[:, None] * along, np.full(len(s), z)])
    along3, out3 = np.append(along, 0.0), np.append(outward, 0.0)
    half = band + EDGE_LIT_PX + EDGE_BEYOND_PX + 2 * EDGE_HALF_WIDTH_PX
    d, rows, q, n = _profiles(gray, cam, model, along3, out3, half)
    pos = finder(rows, d, band)
    hit = ~np.isnan(pos)
    return _Feature(name, model[hit], np.tile(along3, (hit.sum(), 1)), q[hit] + pos[hit, None] * n[hit],
                    PROFILES_PER_EDGE)


def _crosshair(lab, cam, st):
    """The crosshair's arm middle lines, and the share found in its worst half arm.

    Across each arm, the groove is where the colour departs from the plate's: its lit inner wall
    can be as bright as the plate but rarely the same colour. The middle line is halfway between
    the two lips, where it starts and ends; that doesn't depend on which part of the groove is lit.
    Profiles are sampled at model points across the arm (mm), so its projected width and direction
    come out right. found offsets (mm across each arm from the model) are kept for the checks.
    """
    centre = np.array([*st.cross_mm, st.top_z_mm])
    step = 0.05
    o = np.arange(-CROSS_SPAN_MM, CROSS_SPAN_MM + step / 2, step)
    outer, inner = np.abs(o) > CROSS_CLEAR_MM, np.abs(o) < CROSS_CLEAR_MM
    models, alongs, found, offsets, tried, halves = [], [], [], [], 0, []
    for along, across in (((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 0, 0))):
        along, across = np.array(along, float), np.array(across, float)
        for sign in (-1, 1):
            hits = 0
            for s in sign * np.linspace(*CROSS_ALONG_MM, CROSS_PROFILES):
                tried += 1
                xy = cam.project(centre + s * along + o[:, None] * across).astype(np.float32)
                row = cv2.remap(lab, xy[:, 0:1], xy[:, 1:2], cv2.INTER_LINEAR)[:, 0].astype(float)
                dev = np.linalg.norm(row - np.median(row[outer], 0), axis=1)
                noise = 1.4826 * np.median(dev[outer])
                threshold = max(CROSS_MIN_CONTRAST, CROSS_LIP_LEVEL * dev[inner].max(), 4 * noise)
                groove = np.nonzero((dev > threshold) & inner)[0]
                if len(groove) == 0 or groove[0] == 0 or groove[-1] == len(o) - 1:
                    continue
                a, b = groove[0], groove[-1]
                with np.errstate(divide="ignore", invalid="ignore"):     # a groove running into the clear
                    lip_a = o[a] - (dev[a] - threshold) / (dev[a] - dev[a - 1]) * step   # band has no lip,
                    lip_b = o[b] + (dev[b] - threshold) / (dev[b] - dev[b + 1]) * step   # and fails the width
                if not CROSS_WIDTH_MM[0] <= lip_b - lip_a <= CROSS_WIDTH_MM[1]:
                    continue
                middle = (lip_a + lip_b) / 2
                models.append(centre + s * along)
                alongs.append(along)
                found.append(cam.project((centre + s * along + middle * across)[None])[0])
                offsets.append(middle * across)
                hits += 1
            halves.append(hits / CROSS_PROFILES)
    feature = _Feature("crosshair", np.array(models).reshape(-1, 3), np.array(alongs).reshape(-1, 3),
                       np.array(found).reshape(-1, 2), tried)
    feature.offsets_mm = np.array(offsets).reshape(-1, 3)
    return feature, min(halves)


def _residuals(cam, features, p, all_points=False):
    """Signed px distances of found points from their projected model features."""
    out = []
    for f in features:
        sel = slice(None) if all_points else f.keep
        q = cam.project(f.model[sel], p)
        t = cam.project(f.model[sel] + 0.5 * f.along[sel], p) - q
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        out.append(np.sum(np.column_stack([-t[:, 1], t[:, 0]]) * (f.found[sel] - q), 1))
    return out


def _fit(cam, features, free):
    """Levenberg-Marquardt on the free parameters, then drop each feature's outliers (beyond 3
    sigma, at least 1 px) and fit again, three times."""
    steps = np.array([1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-1])
    for _ in range(3):
        lam = 1e-3
        for _ in range(100):
            r0 = np.concatenate(_residuals(cam, features, cam.p))
            jac = np.column_stack([(np.concatenate(_residuals(cam, features, cam.p + np.eye(7)[i] * steps[i])) - r0)
                                   / steps[i] for i in free])
            a = jac.T @ jac
            step = np.zeros(7)
            step[free] = np.linalg.solve(a + lam * np.diag(np.diag(a)), -jac.T @ r0)
            if np.sum(np.concatenate(_residuals(cam, features, cam.p + step)) ** 2) < np.sum(r0 ** 2):
                cam.p = cam.p + step
                lam = max(lam / 3, 1e-9)
                if np.max(np.abs(step[free] / steps[free])) < 1e-3:
                    break
            else:
                lam *= 5
                if lam > 1e8:
                    break
        for f, r in zip(features, _residuals(cam, features, cam.p, all_points=True)):
            med = np.median(r[f.keep]) if f.keep.any() else 0.0
            spread = 1.4826 * np.median(np.abs(r[f.keep] - med)) if f.keep.any() else 0.0
            f.keep = np.abs(r - med) < max(3 * spread, 1.0)


def _rms(cam, f):
    r = _residuals(cam, [f], cam.p)[0]
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("inf")


@dataclass
class PlateFit:
    """The camera found from the plate: pose, lens, and how well each feature fitted."""
    rvec: np.ndarray
    tvec: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    sides: list                 # (share, rms px) for the top, right, bottom and left of the image;
                                # the near side's is the feature that fixed it
    near_by: str                # "crosshair", "face bottom" or "near edge"
    checks: dict                # where the found near-side features that were not fitted are against
                                # the fitted model: the face bottom and near edge as their median offset
                                # outward (mm), the crosshair as its (x, y) offset (mm)
    summary: dict               # all of it, for saving with the calibration

    def homography_mm(self, st):
        """3 × 3 from scene mm on the plate's top (x, y, 1) to undistorted frame px."""
        r = cv2.Rodrigues(self.rvec)[0]
        return self.camera_matrix @ np.column_stack([r[:, 0], r[:, 1], r[:, 2] * st.top_z_mm + self.tvec])


def find_plate(frame, settings=None, start=None, intrinsics=None):
    """The camera, from the plate as it appears in a BGR frame. Raises PlateNotFound.

    start: plate corners (raw px, image order) where it was last found, or None to start from the
    frame's bright blob. intrinsics: (camera_matrix, dist_coeffs) to keep, so that only the pose is
    fitted; or None, in which case the focal length is fitted too (and the lens taken as
    distortion-free: the plate's edges, all near the middle of the frame, can't measure it).

    One of the top's far and side edges may be missing, as when the bench behind it is as bright as
    the plate (a pale lab bench behind the far edge). The crosshair must then be found and fit, and the
    face bottom and near edge are fitted with it where they are found, to make up for the edge.
    """
    st = settings or Settings.from_config()
    cam = _Camera((frame.shape[1], frame.shape[0]), st.focal_px, intrinsics)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    lab = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.0)
    sides, near = _sides(st)
    z_top, z_face = st.top_z_mm, st.top_z_mm - st.face_mm
    names = ("top", "right", "bottom", "left")
    pose_only = [0, 1, 2, 3, 4, 5]
    free = pose_only if intrinsics is not None else pose_only + [6]

    def measure(name, band):
        i = names.index(name)
        return _edge(name, gray, cam, st, *sides[i], EDGE_SPAN_MM, z_top, band, _falls)

    def short_of(f):
        return PlateNotFound(f"only {f.share:.0%} of the plate's {f.name} edge was found; is something over it?")

    def top_edges(band, min_found):
        """The top's edges, but for the near side's and the missing one. One falling short of min_found
        when none is missing yet becomes the missing one: a pale bench can show part of an edge, enough
        for the first search but not for a fit."""
        nonlocal missing
        edges = [measure(names[i], band) for i in range(4) if i != near and names[i] != missing]
        short = [f for f in edges if f.share < min_found]
        if len(short) == 1 and missing is None:
            missing = short[0].name
            return [f for f in edges if f.name != missing]
        if short:
            raise short_of(short[0])
        return edges

    # 1. The far and side edges of the top, around the starting pose; the pose alone fitted to them.
    #    From the bright blob, each rough outline in turn until one leads to all three edges, or to two
    #    (the third is then missing, and the crosshair has to make up for it).
    starts = _rough_outlines(frame) if start is None else [start]
    for i, corners in enumerate(starts):
        _start_pose(cam, corners, st)
        edges = [measure(names[j], FIRST_BAND_PX if start is None else NEAR_BAND_PX) for j in range(4) if j != near]
        short = [f for f in edges if f.share < FIRST_MIN_FOUND]
        if len(short) <= 1:
            break
        if i == len(starts) - 1:
            raise short_of(short[0])
    missing = short[0].name if short else None
    edges = [f for f in edges if f.name != missing]
    _fit(cam, edges, pose_only)

    # 2. The near side: the crosshair, else the face bottom, else (if the face is lit or dark enough
    #    to show it) the near edge of the top; each searched within reach of the model.
    mid, along, outward = sides[near]

    def scale(direction, z):
        """Image px per mm at the middle of the near side, along a scene direction."""
        a = np.append(mid, z)
        return float(np.linalg.norm(cam.project([a + direction])[0] - cam.project([a])[0]))

    def near_features(band):
        face_px = scale((0, 0, st.face_mm), z_top)              # the face's height in the image
        face = _edge("face bottom", gray, cam, st, mid, along, outward, FACE_SPAN_MM, z_face, band,
                     lambda rows, d, b: _falls(rows, d, b, min(EDGE_LIT_PX, 0.5 * face_px)))
        top = _edge("near edge", lab, cam, st, mid, along, outward, EDGE_SPAN_MM, z_top, band, _colour_steps)
        cross, cross_share = _crosshair(lab, cam, st)
        return {"crosshair": (cross, cross_share >= CROSS_MIN_FOUND),
                "face bottom": (face, face.share >= MIN_INLIERS), "near edge": (top, top.share >= MIN_INLIERS)}

    def search_crosshair():
        """The crosshair looked for round the model, up to CROSS_SEARCH_MM out on the plate, and the
        pose moved to put it where it was found best. It is only looked for within CROSS_SPAN_MM of the
        model, and two edges can leave the model further off than that. Returns (feature, found)."""
        steps = np.arange(-CROSS_SEARCH_MM, CROSS_SEARCH_MM + CROSS_SEARCH_STEP_MM / 2, CROSS_SEARCH_STEP_MM)
        best = None
        for dx in steps:
            for dy in steps:
                f, share = _crosshair(lab, cam, replace(st, cross_mm=(st.cross_mm[0] + dx, st.cross_mm[1] + dy)))
                score = (share, f.share, -np.hypot(dx, dy))
                if best is None or score > best[0]:
                    best = (score, dx, dy)
        (share, _, _), dx, dy = best
        if share < CROSS_MIN_FOUND:
            return None, False
        # The plate's crosshair is seen where the model's point (dx, dy) away projects: move the camera so
        # that the crosshair itself projects there.
        cam.p[3:6] += cv2.Rodrigues(cam.p[:3])[0] @ np.array([dx, dy, 0.0])
        cross, share = _crosshair(lab, cam, st)
        return cross, share >= CROSS_MIN_FOUND

    def fit_near(edges, found, excluded):
        """Fits the camera to the edges and the first usable near-side feature. A crosshair that then
        fits badly is taken as covered (a lead or a body on it) and the next one used instead. With an
        edge missing, only the crosshair will do, and the focal length is held: two edges and the
        crosshair can't tell it from the camera's distance. The face bottom and near edge stay checks."""
        if missing is not None:
            cross, ok = found["crosshair"]
            if not ok:
                cross, ok = search_crosshair()
            if ok:
                # Looked for again round each fit, until the model stops moving: a crosshair found
                # off the model's middle draws it only part of the way each time.
                for _ in range(CROSS_REFITS):
                    _fit(cam, edges + [cross], pose_only)
                    before = cam.project([[*st.cross_mm, z_top]])[0]
                    again, share = _crosshair(lab, cam, st)
                    if share < CROSS_MIN_FOUND:
                        break
                    cross = again
                    _fit(cam, edges + [cross], pose_only)
                    if np.linalg.norm(cam.project([[*st.cross_mm, z_top]])[0] - before) < CROSS_SETTLED_PX:
                        break
                found["crosshair"] = (cross, True)      # the one fitted, for the checks below
                if _rms(cam, cross) <= CROSS_MAX_RMS_PX and cross.share >= CROSS_MIN_FOUND:
                    return "crosshair", excluded
            raise PlateNotFound(f"the plate's {missing} edge wasn't found, and the crosshair, which would make up "
                                "for it, is covered or doesn't fit: is the bench behind that edge as bright as the plate?")
        for name in ("crosshair", "face bottom", "near edge"):
            if name in excluded or not found[name][1]:
                continue
            before = cam.p.copy()
            _fit(cam, edges + [found[name][0]], free)
            if name == "crosshair" and (_rms(cam, found[name][0]) > CROSS_MAX_RMS_PX
                                        or found[name][0].share < CROSS_MIN_FOUND):
                cam.p = before
                excluded = excluded + ("crosshair",)
                continue
            return name, excluded
        raise PlateNotFound("the near side of the plate was not found: the crosshair is covered and "
                            "the bottom of the plate's front face is not visible")

    # The first search must tell the face's bottom from its top: half the face's height either way.
    found = near_features(0.4 * scale((0, 0, st.face_mm), z_top))
    used, excluded = fit_near(edges, found, ())

    # 3. Everything again around the fitted model, and refitted.
    for _ in range(2):
        edges = top_edges(REFINE_BAND_PX, MIN_INLIERS)
        found = near_features(REFINE_BAND_PX)
        used, excluded = fit_near(edges, found, excluded)

    for f in edges + [found[used][0]]:
        rms = _rms(cam, f)
        if f.share < MIN_INLIERS or rms > MAX_RMS_PX:
            raise PlateNotFound(f"the plate's {f.name} doesn't fit the plate's model "
                                f"({f.share:.0%} of it found, {rms:.1f} px rms)")
    k, dist = cam.matrices()
    if abs(k[0, 0] / st.focal_px - 1) > FOCAL_RANGE:
        raise PlateNotFound(f"the fitted focal length, {k[0, 0]:.0f} px, is far from the expected {st.focal_px:.0f}")
    position = cam.position_mm
    if position[2] < st.top_z_mm + 50:
        raise PlateNotFound("the fitted camera is not above the plate")

    # The near-side features not fitted, as checks: where they are against the fitted model, in mm.
    checks = {}
    for name, (f, ok) in found.items():
        if name == used or not ok:
            continue
        if name == "crosshair":
            f, _ = _crosshair(lab, cam, st)
            off = f.offsets_mm
            checks[name] = [round(float(np.median(off[off[:, i] != 0, i])), 2) if (off[:, i] != 0).any() else None
                            for i in (0, 1)]
        else:
            f.keep = np.ones(len(f.found), bool)
            out_px = scale(np.append(outward, 0.0), z_top if name == "near edge" else z_face)
            checks[name] = round(float(np.median(_residuals(cam, [f], cam.p)[0]) / out_px), 2)
    by_side = {f.name: (round(float(f.share), 3), round(_rms(cam, f), 2)) for f in edges}
    by_side[names[near]] = (round(float(found[used][0].share), 3), round(_rms(cam, found[used][0]), 2))
    if missing is not None:     # for the record: how much of it shows on the fitted model
        f = measure(missing, REFINE_BAND_PX)
        by_side[missing] = (round(float(f.share), 3), round(_rms(cam, f), 2) if len(f.found) else None)
    r = cv2.Rodrigues(cam.p[:3])[0]
    summary = {"near_side_by": used, "missing_edge": missing, "edges": by_side, "checks_mm": checks,
               "focal_px": round(float(k[0, 0]), 1), "camera_mm": np.round(position, 1).tolist(),
               "tilt_deg": round(float(np.degrees(np.arccos(-r[2, 2]))), 2)}
    return PlateFit(cam.p[:3].copy(), cam.p[3:6].copy(), k, np.asarray(dist, float),
                    [by_side[n] for n in names], used, checks, summary)


def _calibration(fit, frame_size, st):
    h_mm = fit.homography_mm(st)
    corners = cv2.perspectiveTransform(st.plate_corners_mm.reshape(-1, 1, 2), h_mm).reshape(-1, 2)
    to_rect = cv2.getPerspectiveTransform(st.plate_corners_mm.astype(np.float32),
                                          mm_to_rect(st.plate_corners_mm, st).astype(np.float32))
    return Calibration(to_rect @ np.linalg.inv(h_mm), corners, frame_size, st,
                       fit.camera_matrix, fit.dist_coeffs, fit.summary)


def find_plate_corners(frame, near=None, settings=None):
    """The plate's top-surface corners in a BGR frame (raw px, image order), and per-side fit details.

    near: corners where the plate was last found (4 × 2 px), to start from instead of the bright blob.
    Returns (corners 4 × 2 px, sides), sides being (share found, rms px) for top, right, bottom, left.
    """
    st = settings or Settings.from_config()
    fit = find_plate(frame, st, start=near)
    cal = _calibration(fit, (frame.shape[1], frame.shape[0]), st)
    return plate_outline_px(cal), fit.sides


def calibrate(frame, settings=None, intrinsics=None):
    """Calibration from one frame with the plate clearly visible (ideally empty).

    intrinsics: (camera_matrix, dist_coeffs) or None, in which case the focal length is fitted too.
    Returns (Calibration, sides) with sides as from find_plate_corners.
    """
    st = settings or Settings.from_config()
    fit = find_plate(frame, st, intrinsics=intrinsics)
    return _calibration(fit, (frame.shape[1], frame.shape[0]), st), fit.sides


def with_corners(cal, corners):
    """A calibration like cal (scale, lens) but with the plate at other corners (undistorted px).

    For a frame taken after the camera moved: its own plate corners give its own homography.
    """
    target = mm_to_rect(cal.settings.plate_corners_mm, cal.settings)
    h = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), target.astype(np.float32))
    return Calibration(h.astype(float), np.asarray(corners, float), cal.frame_size, cal.settings,
                       cal.camera_matrix, cal.dist_coeffs, cal.fit)


def recalibrate(frame, cal):
    """The calibration for one frame: the plate found again near cal's, with cal's lens kept and only
    the camera's pose fitted. Raises PlateNotFound. Returns (Calibration, sides)."""
    st = cal.settings
    start = plate_outline_px(cal)
    if cal.camera_matrix is None:           # a calibration from before the plate model: no lens yet
        fit = find_plate(frame, st, start=start)
    else:
        fit = find_plate(frame, st, start=start, intrinsics=(cal.camera_matrix, cal.dist_coeffs))
    return _calibration(fit, cal.frame_size, st), fit.sides


def plate_model_px(cal, steps=8):
    """The plate's model as the calibration's camera sees it, for drawing over a raw frame: polylines
    (N × 2 raw px) for the top's outline (rounded corners included), the bottom edge of its front
    face, and the crosshair's two arms. The camera's pose is taken from the calibration's corners,
    so it suits calibrations from with_corners too."""
    st = cal.settings
    intrinsics = None if cal.camera_matrix is None else (cal.camera_matrix, cal.dist_coeffs)
    cam = _Camera(cal.frame_size, st.focal_px, intrinsics)
    _start_pose(cam, plate_outline_px(cal), st)
    lo, hi, r, z = st.plate_corners_mm.min(0), st.plate_corners_mm.max(0), st.corner_radius_mm, st.top_z_mm
    outline = []
    for centre, start in (((hi[0] - r, hi[1] - r), 0), ((lo[0] + r, hi[1] - r), 90),
                          ((lo[0] + r, lo[1] + r), 180), ((hi[0] - r, lo[1] + r), 270)):
        a = np.radians(start + np.linspace(0, 90, steps))
        outline += [(centre[0] + r * np.cos(t), centre[1] + r * np.sin(t), z) for t in a]
    sides, near = _sides(st)
    mid, along, _ = sides[near]
    half = np.linalg.norm(st.plate_corners_mm[near] - st.plate_corners_mm[(near + 1) % 4]) / 2 - r
    face = [(*(mid + s * along), z - st.face_mm) for s in (-half, half)]
    (cx, cy), arm = st.cross_mm, st.cross_arm_mm / 2
    return {"top": cam.project(outline), "face bottom": cam.project(face),
            "crosshair": [cam.project([(cx - arm, cy, z), (cx + arm, cy, z)]),
                          cam.project([(cx, cy - arm, z), (cx, cy + arm, z)])]}


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


def plate_mask(settings, lift_mm=0.0):
    """H×W bool: the rectified pixels whose centres lie on the plate's top surface (rounded corners
    included). Cached per geometry.

    lift_mm: also keep what the plate would cover lifted by up to this much towards the camera
    (settings.camera_mm). The rectified image is true only at the plate's surface; anything standing
    on it appears pushed away from the lens, by about 2.4 mm for a 2.5 mm body at the far (+x) edge
    and much less elsewhere. So a body lying at the far edge seems to overhang it, and a plain mask
    cuts off its top (experiments/2026-11-09_seg_objects_v1_test, edge_evidence.jpg). A lift keeps
    that overhang, widening the mask only where parallax pushes outwards: about lift_mm at the far
    edge, a few tenths at the sides, nothing at the near edge."""
    key = (settings.size, tuple(settings.plate_corners_mm.ravel()), settings.px_per_mm,
           settings.margin_mm, settings.corner_radius_mm, float(lift_mm),
           tuple(settings.camera_mm) if lift_mm else None, settings.top_z_mm if lift_mm else None)
    if key not in _MASKS:
        w, h = settings.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        xy = rect_to_mm(np.column_stack([u.ravel(), v.ravel()]), settings)
        c = settings.plate_corners_mm
        centre, half = (c.min(0) + c.max(0)) / 2, (c.max(0) - c.min(0)) / 2
        r = settings.corner_radius_mm

        def outside(pts):                                # rounded-rectangle distance, outside > 0
            q = np.abs(pts - centre) - (half - r)
            return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - r

        keep = outside(xy) <= 0
        if lift_mm > 0:
            cam = np.asarray(settings.camera_mm, float)
            above = cam[2] - settings.top_z_mm
            # A point at height z over footprint p appears at p + (p - cam) * z / (above - z); so the
            # footprint under an apparent point x is (x + k * cam) / (1 + k), k = z / (above - z).
            for z in np.linspace(lift_mm / 8, lift_mm, 8):
                k = z / (above - z)
                keep |= outside((xy + k * cam[:2]) / (1 + k)) <= 0
        _MASKS[key] = keep.reshape(h, w)
    return _MASKS[key]


def mask_outside_plate(rectified, settings, lift_mm=0.0):
    """A copy of a rectified (and white-balanced) image with everything off the plate replaced by
    the plate's own grey, so a model given it can take nothing from the surroundings.

    The fill is the median colour of the plate's brighter half (resistors and marks left out), so
    the plate's edge disappears into it. Training and inference both use this. lift_mm: see
    plate_mask; the fill still comes from the plate alone.
    """
    inside = plate_mask(settings)
    pixels = rectified[inside].reshape(-1, rectified.shape[2]).astype(np.float32)
    lum = pixels.mean(1)
    fill = np.median(pixels[lum >= np.median(lum)], axis=0)
    out = rectified.copy()
    out[~plate_mask(settings, lift_mm)] = np.round(fill).astype(rectified.dtype)
    return out


def model_input(frame, cal, rect=None, lift_mm=0.0):
    """What the model sees for a raw camera frame: rectified, white-balanced with the plate brought to
    MODEL_PLATE_LEVEL (as in the training frames, whatever the exposure), off-plate masked.

    rect: the frame already rectified, to save doing it again. Returns (model input, the rectified
    image white-balanced to that level, unmasked)."""
    balanced, _ = white_balance(rectify(frame, cal) if rect is None else rect, cal.settings, MODEL_PLATE_LEVEL)
    return mask_outside_plate(balanced, cal.settings, lift_mm), balanced


def model_image(sample, image, rect_settings, lift_mm=0.0):
    """A manifest sample's image as the model must see it: a real frame's plate at MODEL_PLATE_LEVEL,
    as model_input does (synthetic frames are drawn there), and off-plate masked. Raises ValueError
    for raw frames."""
    if sample.get("space") != "rectified":
        raise ValueError(f"{sample['image']}: not a rectified frame, so it cannot be masked to the plate. "
                         "Regenerate old synthetic sets with `synth`; real frames need rectifying first.")
    if sample.get("source") != "synthetic":
        image, _ = white_balance(image, rect_settings, MODEL_PLATE_LEVEL)
    return mask_outside_plate(image, rect_settings, lift_mm)


def white_balance(rectified, settings, level=None):
    """Channels scaled so the plate is neutral. Returns (image, gains as B, G, R).

    The plate's colour is the median of its brighter half, which leaves out resistors and marks.
    level: if given, the plate is also brought to this brightness; else it keeps its own.
    """
    c = settings.plate_corners_mm
    inner = np.array([[c[:, 0].min() + WHITE_INSET_MM, c[:, 1].max() - WHITE_INSET_MM],
                      [c[:, 0].max() - WHITE_INSET_MM, c[:, 1].min() + WHITE_INSET_MM]])
    (u0, v0), (u1, v1) = mm_to_rect(inner, settings).round().astype(int)
    region = rectified[v0:v1, u0:u1].reshape(-1, 3).astype(np.float32)
    lum = region.mean(1)
    plate = np.median(region[lum >= np.median(lum)], axis=0)
    gains = (plate.mean() if level is None else level) / np.maximum(plate, 1.0)
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
        fit = cal.fit
        print(f"Plate corners (px): {np.round(cal.corners_px, 1).tolist()}")
        print(f"Camera at {fit['camera_mm']} mm, {fit['tilt_deg']}° from vertical, focal length {fit['focal_px']} px")
        for name, (share, rms) in zip(("top", "right", "bottom", "left"), sides):
            what = f"near side, by the {fit['near_side_by']}" if name == "bottom" else "edge"
            flag = "" if share >= 0.8 and rms < 1.5 else "   <- check this"
            print(f"  {name:6s} {what}: {share:4.0%} found on the model, rms {rms:.2f} px{flag}")
        for name, offset in fit["checks_mm"].items():
            print(f"  check: the {name} is {offset} mm from where the fit puts it")
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

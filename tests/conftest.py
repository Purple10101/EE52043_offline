import cv2
import numpy as np
import pytest

from robot_offline.pre import rectify

FRAME = (4608, 2592)
# The tower camera as fitted on a real frame (2026-10-29, after the tower was stiffened): lens about
# (-145, -146, 181) mm in the scene frame, 35.5° from vertical, focal length 3590 px.
RVEC = np.array([1.93576, -1.76165, 0.54307])
TVEC = np.array([-127.539, -22.92, 242.077])
FOCAL = 3589.6
K = np.array([[FOCAL, 0.0, FRAME[0] / 2], [0.0, FOCAL, FRAME[1] / 2], [0.0, 0.0, 1.0]])
WOOD, PLATE, DARK_FACE, GAP, LEG = (60, 110, 150), (200, 200, 200), (45, 45, 45), (15, 15, 15), (185, 185, 185)
PALE_BENCH = (190, 208, 214)    # the lab bench (2026-11-09): about 1.04 times the plate's grey, and yellower


def _project(points_mm, rvec, tvec):
    out, _ = cv2.projectPoints(np.asarray(points_mm, np.float64).reshape(-1, 1, 3), rvec, tvec, K, None)
    return out.reshape(-1, 2)


def _poly(img, points, colour):
    cv2.fillPoly(img, [np.round(np.asarray(points) * 16).astype(np.int32)], colour, cv2.LINE_AA, 4)


def _rounded_square(st, steps=8):
    """The plate's top outline (scene mm, z = top), rounded corners included."""
    c, r = st.plate_corners_mm, st.corner_radius_mm
    lo, hi = c.min(0), c.max(0)
    out = []
    for centre, start in (((hi[0] - r, hi[1] - r), 0), ((lo[0] + r, hi[1] - r), 90),
                          ((lo[0] + r, lo[1] + r), 180), ((hi[0] - r, lo[1] + r), 270)):
        a = np.radians(start + np.linspace(0, 90, steps))
        out += list(np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]))
    return np.column_stack([np.array(out), np.full(len(out), st.top_z_mm)])


def render_plate(rvec=RVEC, tvec=TVEC, face=DARK_FACE, cover_cross=False, far_bench=None, far_bench_y=(-230, -30),
                 seed=0):
    """A camera frame of the plate, drawn from its model through a pinhole camera: the plate's top
    with its crosshair groove, its front face (dark by default, or lit), the dark gap under it, the
    legs at its near corners, a screw by a far corner, leads and a resistor body near the edges;
    then blur and noise. far_bench: a colour for the bench behind the plate's far edge, as the pale
    lab bench (PALE_BENCH) is, over far_bench_y (scene mm). Returns (frame, the top's corners in px, image order)."""
    st = rectify.Settings.from_config()
    p = lambda pts: _project(pts, rvec, tvec)
    z, zf, zb = st.top_z_mm, st.top_z_mm - st.face_mm, st.top_z_mm - 28.0
    img = np.full((FRAME[1], FRAME[0], 3), WOOD, np.uint8)
    if far_bench is not None:
        y0, y1 = far_bench_y
        _poly(img, p([[45, y0, zb], [140, y0, zb], [140, y1, zb], [45, y1, zb]]), far_bench)
    _poly(img, p([[-50, -178, zf], [-50, -82, zf], [-50, -82, zb], [-50, -178, zb]]), GAP)
    for y0, y1 in ((-180, -172), (-88, -80)):                                # legs, seen from the front
        _poly(img, p([[-50, y0, zf], [-50, y1, zf], [-50, y1, zb], [-50, y0, zb]]), LEG)
    _poly(img, p([[-50, -180, z], [-50, -80, z], [-50, -80, zf], [-50, -180, zf]]), face)
    _poly(img, p(_rounded_square(st)), PLATE)
    cx, cy = st.cross_mm
    half = st.cross_arm_mm / 2
    for (x0, x1), (y0, y1) in (((cx - half, cx + half), (cy - 0.5, cy + 0.5)), ((cx - 0.5, cx + 0.5), (cy - half, cy + half))):
        _poly(img, p([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]), (95, 95, 95))
    screw = p([[56 + 3 * np.cos(a), -77 + 3 * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, 24)])
    _poly(img, screw, (215, 215, 215))
    for a, b in (((35, -95), (10, -115)), ((15, -165), (25, -175)), ((-30, -105), (-46, -85))):  # leads
        cv2.line(img, *(tuple(np.round(q).astype(int)) for q in p([[*a, z], [*b, z]])), (70, 70, 70), 7, cv2.LINE_AA)
    _poly(img, p([[-5, -177, z], [3, -177, z], [3, -174, z], [-5, -174, z]]), (60, 160, 120))  # a body near the right edge
    if cover_cross:                                                          # a lead lying over the crosshair
        cv2.line(img, *(tuple(np.round(q).astype(int)) for q in p([[cx - 6, cy - 4, z], [cx + 6, cy + 3, z]])),
                 (70, 70, 70), 9, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = np.random.default_rng(seed).normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8), p(np.column_stack([st.plate_corners_mm, np.full(4, z)]))


def camera_position(rvec=RVEC, tvec=TVEC):
    r = cv2.Rodrigues(np.asarray(rvec, float))[0]
    return -r.T @ np.asarray(tvec, float)


def moved_pose(dx_mm, dy_mm, dz_mm=0.0, rvec=RVEC, tvec=TVEC):
    """The camera moved by (dx, dy, dz) mm in the scene frame, still pointing the same way."""
    r = cv2.Rodrigues(np.asarray(rvec, float))[0]
    return np.asarray(rvec, float), np.asarray(tvec, float) - r @ np.array([dx_mm, dy_mm, dz_mm])


@pytest.fixture(scope="session")
def render():
    return render_plate


@pytest.fixture(scope="session")
def moved_plate():
    """(frame, corners) as seen after the camera moved by (dx, dy, dz) mm."""
    return lambda dx, dy, dz=0.0: render_plate(*moved_pose(dx, dy, dz))


@pytest.fixture(scope="session")
def _plate():
    return render_plate()


@pytest.fixture(scope="session")
def plate_corners(_plate):
    return _plate[1].copy()


@pytest.fixture(scope="session")
def plate_frame(_plate):
    """A camera frame like the real one, drawn from the plate's model (see render_plate)."""
    return _plate[0]

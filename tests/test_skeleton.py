import cv2
import numpy as np
import pytest

from robot_offline import evaluate, skeleton, synth, tiling

RECT = np.array([[-33, -12], [33, -12], [33, 12], [-33, 12]], float)   # 1/4 W body at 10 px/mm
BODY = synth._body_outline(66, 13, 0.86, 0.15)                          # caps 26 px, waist 22.4 px


def _place(local, cx, cy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + (cx, cy)


def _yaw_error(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _poses(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return [(x, y, a) for (x, y), a in zip(rng.uniform(100, 1100, (n, 2)), rng.uniform(0, np.pi, n))]


def test_thin_leaves_a_one_pixel_centre_line():
    mask = np.zeros((40, 120), np.uint8)
    mask[10:31, 10:111] = 1
    skel = skeleton.thin(mask)
    assert skel.any()
    assert not (skel[:-1, :-1] & skel[1:, :-1] & skel[:-1, 1:] & skel[1:, 1:]).any()   # no 2 × 2 block
    rows, cols = np.nonzero(skel)
    assert np.all(rows[(cols > 30) & (cols < 90)] == 20)


@pytest.mark.parametrize("local, width", [(RECT, 24.0), (BODY, 22.4)])
@pytest.mark.parametrize("pose", _poses())
def test_straight_body_geometry(local, width, pose):
    cx, cy, yaw = pose
    poly = _place(local, cx, cy, yaw)
    b = skeleton.body_skeleton(poly)
    assert np.allclose(b.centroid, (cx, cy), atol=0.05)
    assert np.allclose(b.midpoint, (cx, cy), atol=0.3)
    assert _yaw_error(b.yaw_deg, np.degrees(yaw)) < 1.0
    assert _yaw_error(b.yaw_deg, evaluate.body_geometry(poly)[4]) < 1.0
    assert abs(b.length - 66) < 0.5
    assert abs(b.width - width) < 0.3
    u = np.array([np.cos(yaw), np.sin(yaw)])
    off_axis = (b.spine - (cx, cy)) @ np.array([-u[1], u[0]])
    assert np.abs(off_axis).max() < 0.5
    assert len(b.spine) == pytest.approx(66 / skeleton.STEP_PX + 1, abs=1)


@pytest.mark.parametrize("pose", _poses(seed=1))
def test_outline_from_a_predicted_mask(pose):
    """The real input: a pixel mask outlined and simplified as tiling.predict_frame does. The
    simplified outline's sides can be a few degrees off parallel, and the spine follows them."""
    cx, cy, yaw = pose
    mask = np.zeros((1200, 1200), np.uint8)
    cv2.fillPoly(mask, [np.round(_place(BODY, cx, cy, yaw) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    b = skeleton.body_skeleton(tiling.mask_to_polygon(mask, epsilon_px=1.0))
    assert np.hypot(*(b.centroid - (cx, cy))) < 0.5
    assert _yaw_error(b.yaw_deg, np.degrees(yaw)) < 3.0          # the docs/plan.md yaw target
    assert abs(b.length - 66) < 1.5
    assert abs(b.width - 22.4) < 1.0


def test_spine_runs_the_way_yaw_points():
    b = skeleton.body_skeleton(_place(RECT, 500, 500, np.radians(150)))
    d = b.spine[-1] - b.spine[0]
    assert d[1] > 0
    assert _yaw_error(np.degrees(np.arctan2(d[1], d[0])), b.yaw_deg) < 0.01


def test_spine_follows_a_bent_outline_in_the_middle():
    radius, half_angle = 80.0, 0.45
    t = np.linspace(-half_angle, half_angle, 60)
    arc = np.column_stack([radius * np.sin(t), radius * (1 - np.cos(t))])
    normal = np.column_stack([-np.sin(t), np.cos(t)])
    b = skeleton.body_skeleton(np.vstack([arc + 12 * normal, (arc - 12 * normal)[::-1]]) + 200)
    middle = b.spine[np.abs(b.spine[:, 0] - 200) < 15]
    assert np.abs(np.hypot(*(middle - (200, 200 + radius)).T) - radius).max() < 0.2
    assert abs(b.width - 24) < 0.3


def test_a_squat_outline_falls_back_to_its_long_axis():
    b = skeleton.body_skeleton(np.array([[0, 0], [14, 0], [14, 10], [0, 10]], float) + 50)
    assert np.allclose(b.centroid, (57, 55))
    assert b.length == pytest.approx(14, abs=1.0)
    assert _yaw_error(b.yaw_deg, 0) < 1.0


def test_a_self_crossing_outline_uses_its_largest_piece():
    bow_tie = np.array([[0, 0], [60, 24], [60, 0], [0, 24]], float) + 100
    assert skeleton.body_skeleton(bow_tie) is not None


@pytest.mark.parametrize("poly", [np.zeros((3, 2)), np.array([[0, 0], [5, 5], [10, 10]], float)])
def test_no_area_gives_none(poly):
    assert skeleton.body_skeleton(poly) is None

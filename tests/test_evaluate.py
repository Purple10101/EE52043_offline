import numpy as np
import pytest

from robot_offline import evaluate

PX_PER_MM = 10.0


def _body(cx, cy, length=60, width=24, yaw_deg=0.0):
    yaw = np.radians(yaw_deg)
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy], float)
    return np.array([c - u * length / 2 - v * width / 2, c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2, c - u * length / 2 + v * width / 2])


@pytest.mark.parametrize("yaw", [0.0, 37.0, 90.0, 179.0])
def test_body_geometry(yaw):
    cx, cy, length, width, angle = evaluate.body_geometry(_body(100, 200, 60, 24, yaw))
    assert (cx, cy) == pytest.approx((100, 200), abs=1e-3)
    assert (length, width) == pytest.approx((60, 24), abs=1e-3)
    d = abs(angle - yaw) % 180
    assert min(d, 180 - d) < 1e-3


def test_perfect_predictions_score_zero_error():
    labels = [_body(100, 100), _body(300, 100, yaw_deg=45)]
    r = evaluate.score_frame(labels, labels, PX_PER_MM)
    assert r["misses"] == r["false_detections"] == r["merges"] == r["splits"] == 0
    assert all(m["iou"] == pytest.approx(1) and m["centre_mm"] < 1e-6 for m in r["matched"])


def test_errors_are_in_mm_and_degrees():
    label = _body(100, 100, yaw_deg=10)
    pred = _body(103, 104, length=70, yaw_deg=178)   # 5 px off, 12 deg across the 0/180 wrap
    r = evaluate.score_frame([label], [pred], PX_PER_MM)
    (m,) = r["matched"]
    assert m["centre_mm"] == pytest.approx(0.5, abs=1e-3)
    assert m["yaw_deg"] == pytest.approx(12, abs=1e-3)
    assert m["length_mm"] == pytest.approx(1.0, abs=1e-3)


def test_misses_false_detections_and_merges():
    a, b = _body(100, 100), _body(100, 130)
    merged = np.array([[70, 88], [130, 88], [130, 142], [70, 142]], float)
    stray = _body(500, 500)
    r = evaluate.score_frame([a, b], [merged, stray], PX_PER_MM)
    assert r["merges"] == 1
    assert r["misses"] + len(r["matched"]) == 2
    assert r["false_detections"] == 2 - len(r["matched"])


def test_empty_frame():
    r = evaluate.score_frame([], [_body(10, 10)], PX_PER_MM)
    assert r["false_detections"] == 1 and r["ious"] == []
    s = evaluate.summarise([r])
    assert s["false_det_on_empty"] == 1

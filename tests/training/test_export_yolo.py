import numpy as np
import pytest
from shapely.geometry import Polygon

from robot_offline.inf import tiling
from robot_offline.training import export_yolo


def _body(cx, cy, length=90, width=30, yaw=0.0):
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy], float)
    return np.array([c - u * length / 2 - v * width / 2, c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2, c - u * length / 2 + v * width / 2])


def test_clip_keeps_mostly_inside_bodies_only():
    t = tiling.Tile(0, 0, 1024, (0, 0, 1024, 1024))
    inside = _body(500, 500)
    half_out = _body(1023, 500)          # half its length past the right edge
    mostly_in = _body(1023 - 25, 500)    # 70 of 90 px inside
    clipped = export_yolo.clip_polygons([inside, half_out, mostly_in], t, 0.6)
    assert len(clipped) == 2
    assert Polygon(clipped[1]).bounds[2] == pytest.approx(1023.5)


def test_yolo_line_rounds_to_the_right_pixel_after_ultralytics_truncates():
    points = np.array([[0.0, 0.0], [1022.6, 511.4]])
    line = export_yolo.yolo_line(points, 1024)
    cls, *vals = line.split()
    assert cls == "0"
    # ultralytics: coordinates * size, then astype(int32)
    pixels = (np.array(vals, float).reshape(-1, 2) * 1024).astype(np.int32)
    np.testing.assert_array_equal(pixels, np.round(points))

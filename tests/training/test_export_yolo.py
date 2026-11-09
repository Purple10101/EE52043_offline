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


def _sample(tmp_path, monkeypatch, name, labelled):
    from robot_offline import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    img = np.zeros((1200, 1200, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 60, 140, 220            # a coloured frame
    (tmp_path / "synthetic" / name).mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(tmp_path / "synthetic" / name / "00000.png"), img)
    return {"image": f"synthetic/{name}/00000.png", "session": name, "source": "synthetic", "space": "rectified",
            "labelled": labelled, "width": 1200, "height": 1200,
            "instances": [{"polygon": _body(600, 600).tolist(), "label_source": "synthetic"}]}


def _export(samples, splits, out, class_names, grey=False):
    return export_yolo.export(samples, splits, out, (100, 100, 1000, 1000), 1024, 128, 0.6,
                              class_names=class_names, grey=grey)


def test_object_class_refuses_frames_with_only_resistors_labelled(tmp_path, monkeypatch):
    partial = _sample(tmp_path, monkeypatch, "old", "resistors")
    full = _sample(tmp_path, monkeypatch, "new", "objects")
    with pytest.raises(ValueError, match="old"):
        _export([partial, full], {"old": "train", "new": "val"}, tmp_path / "y", ("object",))
    _export([partial, full], {"old": "test", "new": "train"}, tmp_path / "y", ("object",))   # scoring only: fine
    _export([partial], {"old": "train"}, tmp_path / "y", ("resistor_body",))                 # the old class: fine


def test_grey_export_writes_grey_tiles(tmp_path, monkeypatch):
    import cv2
    s = _sample(tmp_path, monkeypatch, "new", "objects")
    _export([s], {"new": "train"}, tmp_path / "grey", ("object",), grey=True)
    tile = cv2.imread(str(next((tmp_path / "grey" / "images" / "train").glob("*"))))
    inside = tile[400:600, 400:600].astype(int)
    assert np.abs(inside[..., 0] - inside[..., 1]).max() <= 2 and np.abs(inside[..., 1] - inside[..., 2]).max() <= 2
    _export([s], {"new": "train"}, tmp_path / "colour", ("object",))
    tile = cv2.imread(str(next((tmp_path / "colour" / "images" / "train").glob("*"))))
    assert np.abs(tile[400:600, 400:600, 0].astype(int) - tile[400:600, 400:600, 2]).max() > 50

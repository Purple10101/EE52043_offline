import numpy as np
import pytest
from shapely.geometry import Polygon

from robot_offline import export_yolo, tiling

WORK_AREA = (504, 396, 3600, 1800)
FRAME = (4608, 2592)


def _body(cx, cy, length=90, width=30, yaw=0.0):
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy], float)
    return np.array([c - u * length / 2 - v * width / 2, c + u * length / 2 - v * width / 2,
                     c + u * length / 2 + v * width / 2, c - u * length / 2 + v * width / 2])


def test_tiles_cover_the_work_area_inside_the_frame():
    ts = tiling.tiles(WORK_AREA, FRAME, 1024, 128)
    assert len(ts) == 8
    x0, y0, w, h = WORK_AREA
    assert min(t.x0 for t in ts) == x0 and max(t.x0 + t.size for t in ts) == x0 + w
    assert min(t.y0 for t in ts) == y0 and max(t.y0 + t.size for t in ts) == y0 + h
    xs = sorted({t.x0 for t in ts})
    assert all(b - a <= 1024 - 128 for a, b in zip(xs, xs[1:]))


def test_cores_partition_the_plane():
    ts = tiling.tiles(WORK_AREA, FRAME, 1024, 128)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform((0, 0), FRAME, (500, 2)):
        owners = [t for t in ts if t.core[0] <= x < t.core[2] and t.core[1] <= y < t.core[3]]
        assert len(owners) == 1


@pytest.mark.parametrize("yaw", [0.0, 0.7, np.pi / 2])
def test_a_body_centred_in_a_core_is_whole_in_that_tile(yaw):
    ts = tiling.tiles(WORK_AREA, FRAME, 1024, 128)
    for t in ts:
        # Worst case: on the core boundary, widened by the merge margin.
        for cx in (t.core[0] - tiling.CORE_MARGIN_PX, t.core[2] + tiling.CORE_MARGIN_PX):
            cy = t.y0 + 512
            if not (WORK_AREA[0] + 50 < cx < WORK_AREA[0] + WORK_AREA[2] - 50):
                continue
            body = _body(cx, cy, length=93, width=33, yaw=yaw)
            clipped = tiling.clip_polygons([body], t, 0.999)
            assert len(clipped) == 1


def test_clip_keeps_mostly_inside_bodies_only():
    t = tiling.Tile(0, 0, 1024, (0, 0, 1024, 1024))
    inside = _body(500, 500)
    half_out = _body(1023, 500)          # half its length past the right edge
    mostly_in = _body(1023 - 25, 500)    # 70 of 90 px inside
    clipped = tiling.clip_polygons([inside, half_out, mostly_in], t, 0.6)
    assert len(clipped) == 2
    assert Polygon(clipped[1]).bounds[2] == pytest.approx(1023.5)


def test_merge_keeps_one_copy_of_a_body_seen_by_two_tiles():
    ts = tiling.tiles(WORK_AREA, FRAME, 1024, 128)
    a, b = ts[0], ts[1]
    boundary = a.core[2]
    body = _body(boundary + 2, a.y0 + 400)
    preds = [
        (a, [body - (a.x0, a.y0)], [0.9]),
        (b, [body + (0.7, -0.4) - (b.x0, b.y0)], [0.8]),
    ]
    polygons, scores = tiling.merge(preds)
    assert len(polygons) == 1 and scores == [0.9]
    np.testing.assert_allclose(polygons[0], body)


def test_merge_drops_clipped_predictions_at_tile_edges():
    ts = tiling.tiles(WORK_AREA, FRAME, 1024, 128)
    a = ts[0]
    clipped = _body(a.size - 20, 400, length=40) + 0  # a stub against a's right edge, in tile px
    polygons, _ = tiling.merge([(a, [clipped], [0.9])])
    assert polygons == []


def test_mask_to_polygon_follows_the_blob():
    mask = np.zeros((100, 100), bool)
    mask[20:40, 30:80] = True
    points = tiling.mask_to_polygon(mask, 0.0)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    assert (xmin, ymin, xmax, ymax) == (30, 20, 79, 39)


def test_yolo_line_rounds_to_the_right_pixel_after_ultralytics_truncates():
    points = np.array([[0.0, 0.0], [1022.6, 511.4]])
    line = export_yolo.yolo_line(points, 1024)
    cls, *vals = line.split()
    assert cls == "0"
    # ultralytics: coordinates * size, then astype(int32)
    pixels = (np.array(vals, float).reshape(-1, 2) * 1024).astype(np.int32)
    np.testing.assert_array_equal(pixels, np.round(points))

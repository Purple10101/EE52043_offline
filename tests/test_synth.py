import itertools

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from robot_offline import clean_synth, config, manifest
from robot_offline.synth import Settings, _place_bodies, render_frame

SMALL = Settings(
    width=1600,
    height=1000,
    work_area=(200, 150, 1400, 850),
    px_per_mm=9.8,
    bodies_per_frame=(6, 10),
    touching_fraction=0.4,
    overlap_fraction=0.3,
    min_gap_mm=5,
    lead_length_mm=(10, 20),
    bent_lead_fraction=0.3,
    max_bend_deg=20,
    empty_fraction=0.0,
    distractors_per_frame=(2, 4),
)


def _with(**changes):
    return Settings(**{**SMALL.__dict__, **changes})


def _raster(polygon, shape):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon * 256).astype(np.int32)], 1, cv2.LINE_8, shift=8)
    return mask.astype(bool)


@pytest.mark.parametrize("seed", range(5))
def test_labels_are_single_disjoint_polygons_inside_the_work_area(seed):
    frame = render_frame(SMALL, np.random.default_rng(seed))
    assert frame.image.shape == (1000, 1600, 3) and frame.image.dtype == np.uint8
    assert frame.polygons
    shapes = [Polygon(p) for p in frame.polygons]
    x0, y0, x1, y1 = SMALL.work_area
    for p, shape in zip(frame.polygons, shapes):
        assert len(p) >= 3 and shape.is_valid
        assert p[:, 0].min() >= x0 and p[:, 0].max() <= x1 - 1
        assert p[:, 1].min() >= y0 and p[:, 1].max() <= y1 - 1
    # Touching bodies share an edge; simplifying each outline moves it by up to 0.1 px.
    for a, b in itertools.combinations(shapes, 2):
        assert a.intersection(b).area < 2.0


@pytest.mark.parametrize("seed", range(5))
def test_labels_match_the_drawn_bodies(seed):
    frame = render_frame(SMALL, np.random.default_rng(seed))
    for polygon, inst in zip(frame.polygons, frame.ids):
        if Polygon(polygon).area < 300:
            continue
        label = _raster(polygon, frame.instance_map.shape)
        drawn = frame.instance_map == inst
        iou = (label & drawn).sum() / (label | drawn).sum()
        assert iou > 0.85, f"body {inst}: IoU {iou:.3f}"


@pytest.mark.parametrize("seed", range(5))
def test_bodies_keep_the_gap_except_within_a_pair(seed):
    settings = _with(touching_fraction=0.8, overlap_fraction=0.0)
    _, polys = _place_bodies(settings, np.random.default_rng(seed))
    gap = settings.min_gap_mm * settings.px_per_mm
    near = [sum(a is not b and a.distance(b) < gap for b in polys) for a in polys]
    assert len(polys) >= 2
    assert max(near) <= 1
    assert 1 in near


@pytest.mark.parametrize("seed", range(3))
def test_resistors_and_leads_stay_inside_the_work_area(seed):
    bodies, _ = _place_bodies(SMALL, np.random.default_rng(seed))
    assert bodies
    x0, y0, x1, y1 = SMALL.work_area
    for body in bodies:
        assert len(body.leads) == 2
        for pts in [body.outline()] + [p for p, _, _ in body.leads]:
            assert pts[:, 0].min() >= x0 and pts[:, 0].max() <= x1 - 1
            assert pts[:, 1].min() >= y0 and pts[:, 1].max() <= y1 - 1


@pytest.mark.parametrize("seed", range(5))
def test_leads_never_touch_another_body(seed):
    bodies, polys = _place_bodies(SMALL, np.random.default_rng(seed))
    for i, body in enumerate(bodies):
        for pts, width, _ in body.leads:
            wire = LineString(pts).buffer(width / 2)
            assert not any(wire.intersects(p) for j, p in enumerate(polys) if j != i)


def test_touching_bodies_are_in_contact():
    frame = render_frame(_with(touching_fraction=1.0, overlap_fraction=0.0),
                         np.random.default_rng(0))
    shapes = [Polygon(p) for p in frame.polygons]
    gaps = [a.distance(b) for a, b in itertools.combinations(shapes, 2)]
    assert min(gaps) < 0.5


def test_same_seed_same_frame():
    a = render_frame(SMALL, np.random.default_rng(7))
    b = render_frame(SMALL, np.random.default_rng(7))
    assert np.array_equal(a.image, b.image)
    assert all(np.array_equal(p, q) for p, q in zip(a.polygons, b.polygons))


def test_empty_frame_has_no_labels():
    frame = render_frame(_with(empty_fraction=1.0), np.random.default_rng(0))
    assert frame.polygons == [] and not frame.instance_map.any()


def test_background_must_match_frame_size():
    with pytest.raises(ValueError):
        render_frame(SMALL, np.random.default_rng(0), np.zeros((10, 10, 3), np.uint8))


def test_work_area_must_fit_the_frame():
    camera = {"camera": {"width": 1000, "height": 800}}
    dataset = {
        "work_area": {"x0": 100, "y0": 100, "width": 950, "height": 500},
        "scale": {"px_per_mm": 9.8},
        "synthetic": {"bodies_per_frame": [1, 2], "touching_fraction": 0, "overlap_fraction": 0,
                      "min_gap_mm": 5, "empty_fraction": 0, "distractors_per_frame": [0, 0]},
    }
    with pytest.raises(config.ConfigError):
        Settings.from_config(camera, dataset)


def _sample(name, source="synthetic"):
    return {"image": f"{'synthetic/' + name if source == 'synthetic' else 'captures/' + name}/0.jpg",
            "session": name, "source": source, "width": 1, "height": 1, "instances": []}


def test_clean_removes_only_synthetic_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(manifest, "PATH", tmp_path / "manifest.json")
    for name in ("a", "b"):
        (tmp_path / "synthetic" / name).mkdir(parents=True)
    (tmp_path / "captures" / "real").mkdir(parents=True)
    rules = config.load("dataset")["labels"]["rules_version"]
    data = manifest.new(rules)
    data["samples"] = [_sample("a"), _sample("b"), _sample("real", source="real")]
    manifest.save(data, manifest.PATH)

    assert clean_synth.clean(["a"]) == ["a"]
    assert not (tmp_path / "synthetic" / "a").exists() and (tmp_path / "synthetic" / "b").exists()
    assert [s["session"] for s in manifest.load(rules, manifest.PATH)["samples"]] == ["b", "real"]

    assert clean_synth.clean() == ["b"]
    assert [s["session"] for s in manifest.load(rules, manifest.PATH)["samples"]] == ["real"]
    assert (tmp_path / "captures" / "real").exists()

    with pytest.raises(SystemExit):
        clean_synth.clean(["nope"])


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    data = manifest.load(1, path)
    sample = {"image": "synthetic/x/00000.jpg", "session": "x", "source": "synthetic",
              "width": 640, "height": 480,
              "instances": [{"polygon": [[1.5, 2.0], [10.0, 2.0], [10.0, 8.25]],
                             "label_source": "synthetic"}]}
    manifest.replace_samples(data, "synthetic/x/", [sample])
    manifest.save(data, path)
    again = manifest.load(1, path)
    assert again["samples"] == [sample]
    manifest.replace_samples(again, "synthetic/x/", [])
    assert again["samples"] == []
    with pytest.raises(ValueError):
        manifest.load(2, path)

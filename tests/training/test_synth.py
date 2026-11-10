import itertools

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from robot_offline import config, manifest
from robot_offline.training import clean_synth
from robot_offline.training.synth import Settings, _place_bodies, _sample_resistor, render_frame

# A small rectified frame: a 100 mm plate at 5 px/mm with a 10 mm margin, the camera off to the
# lower left as on the rig.
SMALL = Settings(
    width=600,
    height=600,
    work_area=(50, 50, 550, 550),
    plate=(50, 50, 550, 550),
    plate_radius=20.0,
    px_per_mm=5.0,
    camera=(-425.0, 375.0),
    camera_height=1000.0,
    bodies_per_frame=(6, 10),
    touching_fraction=0.4,
    overlap_fraction=0.3,
    cluster_fraction=0.0,
    min_gap_mm=5,
    lead_length_mm=(10, 20),
    lead_curl_deg=12,
    lead_bends=1.0,
    kink_fraction=0.3,
    empty_fraction=0.0,
    distractors_per_frame=(2, 4),
    objects_fraction=1.0,
    objects_per_frame=(1, 3),
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
    assert frame.image.shape == (600, 600, 3) and frame.image.dtype == np.uint8
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
    dataset = {
        "work_area": {"x0": 100, "y0": 100, "width": 1150, "height": 500},
        "rectify": {"plate_corners_mm": [[50, -80], [50, -180], [-50, -180], [-50, -80]],
                    "px_per_mm": 10.0, "margin_mm": 10.0, "plate_corner_radius_mm": 4.0},
        "synthetic": {},
    }
    with pytest.raises(config.ConfigError):
        Settings.from_config(dataset)


def test_settings_from_the_config_draw_the_rectified_frame():
    st = Settings.from_config(config.load("dataset"))
    assert (st.width, st.height) == (1200, 1200) and st.plate == (100, 100, 1100, 1100)
    frame = render_frame(st, np.random.default_rng(3))
    assert frame.image.shape == (1200, 1200, 3) and frame.polygons


def test_bodies_appear_wider_and_shifted_away_from_the_camera():
    rng = np.random.default_rng(0)
    res = _sample_resistor(SMALL, rng)
    res.centre = np.array([450.0, 150.0])
    away = (res.centre - SMALL.camera) / np.linalg.norm(res.centre - SMALL.camera)
    res.yaw = np.arctan2(away[1], away[0]) + np.pi / 2   # the axis across the line of sight
    assert res.seen_radius > res.radius * 1.05
    assert (res.seen_centre - res.centre) @ away > 0.5 * res.radius * np.linalg.norm(res.lean)


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


def test_any_body_varies_size_and_paint():
    st = _with(random_body_fraction=1.0, random_size_fraction=1.0)
    rng = np.random.default_rng(0)
    bodies = [_sample_resistor(st, rng) for _ in range(200)]
    lengths = np.array([b.length for b in bodies]) / st.px_per_mm
    assert lengths.min() < 4 and lengths.max() > 9
    assert all(1.5 <= 2 * b.radius / st.px_per_mm <= 4.5 for b in bodies)
    assert len({tuple(b.package.body_rgb) for b in bodies}) > 150


def test_room_light_changes_the_frame_but_not_its_labels():
    plain = render_frame(_with(), np.random.default_rng(3))
    lit = _with(random_body_fraction=1.0, random_size_fraction=1.0, hard_light_fraction=1.0,
                shade_fraction=1.0, tint_fraction=1.0, exposure_fraction=1.0)
    frame = render_frame(lit, np.random.default_rng(3))
    assert frame.image.shape == plain.image.shape
    assert frame.polygons
    for polygon in frame.polygons:
        assert Polygon(polygon).is_valid


def test_each_label_knows_its_bands_and_which_end_is_band_one():
    frame = render_frame(_with(), np.random.default_rng(4))
    assert len(frame.bands) == len(frame.polygons)
    for polygon, (names, end) in zip(frame.polygons, frame.bands):
        assert len(names) in (4, 5)
        assert np.linalg.norm(polygon - end, axis=1).min() < 12    # at the outline's tip, in px


# --- The objects profile ([synthetic_objects]) ---------------------------------------------------

OBJECTS = dict(objects_profile=True, labelled_objects_per_frame=(3, 6), object_size_mm=(2.0, 25.0),
               edge_fraction=0.25, edge_gap_mm=(0.5, 8.0), leads_off_plate=True)


@pytest.mark.parametrize("seed", range(4))
def test_objects_profile_labels_every_object_with_its_kind(seed):
    from robot_offline.training.synth import OBJECT_KINDS
    frame = render_frame(_with(**OBJECTS), np.random.default_rng(seed))
    assert len(frame.kinds) == len(frame.polygons) == len(frame.bands) == len(frame.leads)
    known = {"resistor"} | {k for k, _ in OBJECT_KINDS}
    assert set(frame.kinds) <= known
    assert any(k != "resistor" for k in frame.kinds)
    for kind, (names, end), polygon in zip(frame.kinds, frame.bands, frame.polygons):
        assert Polygon(polygon).is_valid and len(polygon) >= 3
        if kind == "resistor":
            assert len(names) in (4, 5) and end is not None
        else:
            assert names == [] and end is None             # the band reader skips these
    shapes = [Polygon(p) for p in frame.polygons]
    for a, b in itertools.combinations(shapes, 2):
        assert a.intersection(b).area < 2.0


def test_every_object_kind_can_be_drawn_and_labelled():
    from robot_offline.training.synth import OBJECT_KINDS, ROD_KINDS, _object, _sample_rod
    st = _with(**OBJECTS)
    rng = np.random.default_rng(0)
    for kind, _ in OBJECT_KINDS:
        if kind in ROD_KINDS:
            res = _sample_rod(kind, st, rng)
            assert res.kind == kind and res.bands == [] and Polygon(res.outline()).is_valid
        else:
            draw, label, clear = _object(kind, st, rng, (300.0, 300.0), 0.7)
            assert label.is_valid and label.area > 4 and draw
            assert clear.buffer(1).contains(label)          # what is kept clear includes the object


def test_the_same_object_is_drawn_again_from_the_same_seed():
    from robot_offline.training.synth import _object
    st = _with(**OBJECTS)
    for kind in ("dip8", "blob", "screw", "capacitor_ceramic"):
        a = _object(kind, st, np.random.default_rng(5), (300.0, 300.0), 1.0)[1]
        b = _object(kind, st, np.random.default_rng(5), (300.0, 300.0), 1.0)[1]
        assert a.equals(b)


@pytest.mark.parametrize("seed", range(3))
def test_edge_placement_puts_bodies_and_objects_against_the_plate_edge(seed):
    st = _with(**{**OBJECTS, "edge_fraction": 1.0})
    frame = render_frame(st, np.random.default_rng(seed))
    x0, y0, x1, y1 = st.plate
    assert frame.polygons
    for p in frame.polygons:
        gap = min(p[:, 0].min() - x0, x1 - 1 - p[:, 0].max(), p[:, 1].min() - y0, y1 - 1 - p[:, 1].max())
        assert -1.0 <= gap <= 8.0 * st.px_per_mm + 1.0


def test_leads_may_run_off_the_plate_but_bodies_stay_on_it():
    from robot_offline.training.synth import _plate_shape
    st = _with(**{**OBJECTS, "edge_fraction": 1.0})
    plate = _plate_shape(st)
    off = 0
    for seed in range(4):
        bodies, polys = _place_bodies(st, np.random.default_rng(seed))
        for body, poly in zip(bodies, polys):
            assert plate.contains(poly)
            off += any(not plate.contains(LineString(p)) for p, _, _ in body.leads)
    assert off > 0


def test_back_of_the_plate_is_darker_with_back_dark():
    st = _with(bodies_per_frame=(0, 0), distractors_per_frame=(0, 0), objects_fraction=0.0, empty_fraction=1.0)
    plain = render_frame(st, np.random.default_rng(1)).image.astype(float).mean(2)
    dark = render_frame(_with(**{**st.__dict__, "back_dark": (0.3, 0.3)}), np.random.default_rng(1)).image.astype(float).mean(2)
    # The camera is off to the left: the plate's right side is the back.
    near, far = (slice(100, 500), slice(70, 120)), (slice(100, 500), slice(480, 530))
    ratio_near = dark[near].mean() / plain[near].mean()
    ratio_far = dark[far].mean() / plain[far].mean()
    assert ratio_far < ratio_near - 0.15


def test_instances_carry_kind_leads_and_order_only_in_the_objects_profile():
    from robot_offline.training.synth import instances_json
    plain = instances_json(render_frame(_with(), np.random.default_rng(2)))
    assert plain and all("kind" not in i and "leads" not in i for i in plain)
    assert all(len(i["bands"]) in (4, 5) and len(i["first_end"]) == 2 for i in plain)
    objects = instances_json(render_frame(_with(**OBJECTS), np.random.default_rng(2)))
    assert all("kind" in i for i in objects)
    for inst in objects:
        if inst["kind"] == "resistor":
            assert inst["bands"] and "first_end" in inst and len(inst["leads"]) == 2 and "z" in inst
        else:
            assert inst["bands"] == [] and "first_end" not in inst
    assert len({i["z"] for i in objects if "z" in i}) == sum("z" in i for i in objects)


def test_objects_profile_comes_from_the_config():
    st = Settings.from_config(config.load("dataset"), "objects")
    assert st.objects_profile and st.leads_off_plate and st.labelled_objects_per_frame[1] > 0
    plain = Settings.from_config(config.load("dataset"))
    assert not plain.objects_profile and plain.labelled_objects_per_frame == (0, 0) and plain.edge_fraction == 0
    with pytest.raises(config.ConfigError):
        Settings.from_config(config.load("dataset"), "nonesuch")


# --- Cut-off bodies ([synthetic_objects_cut]) ------------------------------------------------------

CUT = {**OBJECTS, "edge_fraction": 1.0, "edge_cut_fraction": 1.0, "edge_cut_gap_mm": (0.0, 0.5)}


def test_cut_bodies_lie_on_the_plate_but_appear_past_its_far_edge():
    from robot_offline.training.synth import _plate_shape
    st = _with(**CUT)
    plate = _plate_shape(st)
    over = []
    for seed in range(6):
        bodies, polys = _place_bodies(st, np.random.default_rng(seed))
        for body, poly in zip(bodies, polys):
            assert body.cut and plate.buffer(0.5).contains(Polygon(body.footprint()))
            if not plate.contains(poly):
                over.append(poly.centroid.x)
    # Parallax pushes tops away from the camera (off to the left): the overhangs are on the right.
    assert over and np.mean(over) > (st.plate[0] + st.plate[2]) / 2


def test_cut_body_labels_are_what_the_plate_mask_leaves():
    from robot_offline.training.synth import _plate_shape
    st = _with(**CUT)
    plate = _plate_shape(st).buffer(1.0)
    cut_short = 0
    for seed in range(4):
        frame = render_frame(st, np.random.default_rng(seed))
        assert frame.polygons
        for p in frame.polygons:
            assert plate.contains(Polygon(p))
        cut_short += sum(Polygon(p).bounds[2] >= st.plate[2] - 2 for p in frame.polygons)
    assert cut_short > 0                                    # some labels end at the far edge


def test_without_cut_fraction_edge_bodies_are_placed_as_before():
    st = _with(**{**OBJECTS, "edge_fraction": 1.0})
    a = render_frame(st, np.random.default_rng(3))
    b = render_frame(_with(**{**OBJECTS, "edge_fraction": 1.0, "edge_cut_gap_mm": (0.0, 0.1)}), np.random.default_rng(3))
    assert all(np.array_equal(p, q) for p, q in zip(a.polygons, b.polygons)) and np.array_equal(a.image, b.image)


def test_a_profile_can_build_on_another():
    st = Settings.from_config(config.load("dataset"), "objects_cut")
    objects = Settings.from_config(config.load("dataset"), "objects")
    assert st.objects_profile and st.edge_cut_fraction > 0 and objects.edge_cut_fraction == 0
    assert st.labelled_objects_per_frame == objects.labelled_objects_per_frame
    assert st.edge_fraction > objects.edge_fraction

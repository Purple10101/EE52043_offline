import cv2
import numpy as np
import pytest

from robot_offline import config
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.plate_search import outlines

PLATE = np.array([[1739, 803], [3023, 870], [3070, 2085], [1453, 1965]], np.int32)


@pytest.fixture(autouse=True)
def own_last(tmp_path, monkeypatch):
    """Every test remembers the plate in its own folder, never in data/plate."""
    monkeypatch.setattr(plate_search, "LAST", tmp_path / "plate" / "last_rectify.json")


def frame_with(*shapes):
    frame = np.full((2592, 4608, 3), 40, np.uint8)
    cv2.fillPoly(frame, [PLATE], (210, 210, 210))
    for draw in shapes:
        draw(frame)
    return frame


def nearest(quads):
    return min(np.abs(q - PLATE).max() for q in quads)


def calibration(corners):
    return rectify.Calibration(np.eye(3), np.asarray(corners, float), (4608, 2592), rectify.Settings.from_config(),
                               fit={"near_side_by": "crosshair", "missing_edge": None, "edges": {"top": (1.0, 0.3)}})


def test_the_plate_is_among_the_outlines():
    assert nearest(outlines(frame_with())) < 15


def test_a_lead_across_the_plate_doesnt_hide_it():
    lead = lambda f: cv2.line(f, (1900, 1000), (2900, 1900), (60, 60, 60), 8, cv2.LINE_AA)
    assert nearest(outlines(frame_with(lead))) < 15


def test_a_frame_with_no_long_lines_gives_no_outlines():
    assert outlines(np.full((2592, 4608, 3), 40, np.uint8)) == []


def fake_fits(monkeypatch, fits_at):
    """find_plate passing only from a start within 15 px of fits_at ("blob": only from no start; None:
    never), recording each start."""
    tried = []

    def find_plate(frame, settings, start):
        tried.append(None if start is None else np.asarray(start, float))
        if isinstance(fits_at, str):
            fits = start is None
        else:
            fits = fits_at is not None and start is not None and np.abs(np.asarray(start) - fits_at).max() < 15
        if not fits:
            raise rectify.PlateNotFound("no")
        return np.asarray(PLATE if start is None else start, float)

    monkeypatch.setattr(rectify, "find_plate", find_plate)
    monkeypatch.setattr(rectify, "_calibration", lambda fit, size, settings: calibration(fit))
    return tried


def test_where_the_plate_was_last_is_tried_before_the_lines(monkeypatch):
    tried = fake_fits(monkeypatch, PLATE + 20)
    assert np.array_equal(plate_search.find(frame_with(), None, last=PLATE + 20).corners_px, PLATE + 20)
    assert len(tried) == 1


def test_the_lines_are_searched_when_the_plate_isnt_where_it_was(monkeypatch):
    tried = fake_fits(monkeypatch, PLATE)
    assert plate_search.find(frame_with(), None, last=PLATE + 300) is not None
    assert np.array_equal(tried[0], PLATE + 300) and len(tried) > 1


def test_outlines_are_tried_nearest_the_start_first_and_only_so_many(monkeypatch):
    tried = fake_fits(monkeypatch, None)
    quads = [PLATE + d for d in range(0, 400, 20)]
    monkeypatch.setattr(plate_search, "outlines", lambda frame: quads[::-1])
    with pytest.raises(rectify.PlateNotFound):
        plate_search.find(frame_with(), None, last=PLATE + 200)
    from_lines = tried[1:-1]
    assert len(from_lines) == plate_search.MAX_OUTLINES
    assert np.array_equal(from_lines[0], PLATE + 200)


def test_the_brightest_region_is_the_last_resort(monkeypatch):
    tried = fake_fits(monkeypatch, "blob")
    assert plate_search.find(frame_with(), None) is not None
    assert tried[-1] is None


def test_nothing_fitting_says_what_was_tried(monkeypatch):
    fake_fits(monkeypatch, None)
    with pytest.raises(rectify.PlateNotFound, match="not where it was last found"):
        plate_search.find(frame_with(), None, last=PLATE)


def test_with_nothing_remembered_the_search_starts_at_the_rigs_place():
    assert np.array_equal(plate_search.start_corners(), plate_search.rig_corners())
    assert plate_search.rig_corners().shape == (4, 2)


def test_a_find_is_remembered_as_the_next_start(monkeypatch):
    fake_fits(monkeypatch, PLATE + 20)
    plate_search.find(frame_with(), None, last=PLATE + 20)
    assert np.array_equal(plate_search.start_corners(), PLATE + 20)
    tried = fake_fits(monkeypatch, PLATE + 20)
    plate_search.find(frame_with(), None)           # no start given: where it was last found
    assert np.array_equal(tried[0], PLATE + 20)


class FakeCap:
    """A camera whose plate level follows its exposure: 255 * (light * 2^exposure)^(1/2.2), clipped."""

    def __init__(self, light):
        self.light, self.props = light, {cv2.CAP_PROP_EXPOSURE: -5.0, cv2.CAP_PROP_GAIN: 0.0}

    def set(self, prop, value):
        self.props[prop] = value

    def level(self):
        linear = self.light * 2 ** self.props[cv2.CAP_PROP_EXPOSURE] * (1 + self.props[cv2.CAP_PROP_GAIN] / 32)
        return 255 * min(linear, 1.0) ** (1 / 2.2)

    def frame(self):
        f = np.full((2592, 4608, 3), 30, np.uint8)
        cv2.fillPoly(f, [PLATE], (round(self.level()),) * 3)
        return f


def run_tracker(tracker, cap, until, frames=400):
    for _ in range(frames):
        tracker.update(cap.frame())
        if until():
            return True
        import time
        time.sleep(0.005)
    return False


def test_the_tracker_looks_for_the_plate_dim_then_brings_it_up(monkeypatch):
    levels_searched = []

    def find(frame, settings, last, keep=True):
        levels_searched.append(float(np.median(frame[1400, 2200])))
        return calibration(PLATE)

    monkeypatch.setattr(plate_search, "find", find)
    monkeypatch.setattr(plate_search, "start_corners", lambda camera=None: PLATE.astype(float))
    monkeypatch.setattr(rectify, "recalibrate", lambda frame, cal: (cal, None))
    monkeypatch.setattr(rectify, "rectify", lambda frame, cal: None)
    cap = FakeCap(light=40.0)                       # bright: the plate clips at the configured exposure
    tracker = plate_search.Tracker(cap)
    try:
        assert run_tracker(tracker, cap, lambda: not tracker.searching)
        low, high = config.load("camera")["plate_search"]["search_level"]
        assert levels_searched and all(low <= v <= high for v in levels_searched)
        working = config.load("camera")["exposure_control"]
        assert run_tracker(tracker, cap, lambda: working["low"] <= cap.level() <= working["high"]
                           and tracker.control.settled)
        tracker.level = (205, 225)                  # reading wants the plate brighter
        assert run_tracker(tracker, cap, lambda: 205 <= cap.level() <= 225 and tracker.control.settled)
    finally:
        tracker.close()


def test_a_tracker_given_a_calibration_and_locked_keeps_it(monkeypatch):
    monkeypatch.setattr(plate_search, "find", lambda *a: pytest.fail("searched"))
    monkeypatch.setattr(rectify, "recalibrate", lambda *a: pytest.fail("refitted"))
    given = calibration(PLATE)
    tracker = plate_search.Tracker(None, cal=given, locked=True)
    try:
        for _ in range(20):
            assert tracker.update(np.zeros((2592, 4608, 3), np.uint8)) is given
    finally:
        tracker.close()
    assert "locked" in tracker.status()


def test_after_failing_where_expected_it_tries_every_exposure(monkeypatch):
    def find(frame, settings, last, keep=True):
        raise rectify.PlateNotFound("no")

    monkeypatch.setattr(plate_search, "find", find)
    cap = FakeCap(light=1.0)
    tracker = plate_search.Tracker(cap)
    seen = set()
    try:
        run_tracker(tracker, cap, lambda: seen.add(cap.props[cv2.CAP_PROP_EXPOSURE]) or len(seen) >= 6, frames=3000)
    finally:
        tracker.close()
    assert len(seen) >= 6


def test_frames_not_from_the_rig_are_not_remembered(monkeypatch):
    fake_fits(monkeypatch, PLATE + 20)
    plate_search.find(frame_with(), None, last=PLATE + 20, keep=False)
    assert np.array_equal(plate_search.start_corners(), plate_search.rig_corners())

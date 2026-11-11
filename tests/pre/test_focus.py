import cv2
import numpy as np
import pytest

from robot_offline.post.bands import closeup
from robot_offline.pre import focus, rectify


class FakeLens:
    """A camera whose sharpness peaks at `best` (the lens's best focus now); reads return its focus."""

    def __init__(self, best, flat=False):
        self.best, self.flat, self.focus, self.sets = best, flat, 306.0, []

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOCUS:
            self.focus = float(value)
            self.sets.append(float(value))

    def get(self, prop):
        return self.focus

    def read(self):
        return True, None


@pytest.fixture
def lens(monkeypatch):
    """Sharpness = a peak round the lens's best (flat when the cross is covered)."""
    def measure(frame, cal, box):
        cap = measure.cap
        if cap.flat and box == measure.cross_box:
            return 100.0
        return 100.0 + 1000.0 * np.exp(-((cap.focus - cap.best) / 12.0) ** 2)

    monkeypatch.setattr(closeup, "mean_frame", lambda cap, n: None)
    monkeypatch.setattr(focus, "measure", measure)
    return measure


def _cal():
    st = rectify.Settings.from_config()
    corners = np.array([[1000, 500], [2600, 500], [2600, 2100], [1000, 2100]], float)
    h = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                    rectify.mm_to_rect(st.plate_corners_mm, st).astype(np.float32))
    return rectify.Calibration(h.astype(float), corners, (4608, 2592), st, None, None,
                               {"camera_mm": [-145.0, -147.0, 182.0]})


def _tracker(cap, tmp_path):
    model = closeup.FocusModel([[233.0, 322.0], [275.0, 294.0]])
    return focus.FocusTracker(cap, _cal(), camera={"camera": {"focus": 306}}, model=model, path=tmp_path / "f.json")


def test_the_first_check_searches_fully_from_below_and_sets_the_best(lens, tmp_path):
    cap = FakeLens(best=331.0)
    lens.cap, lens.cross_box = cap, None
    t = _tracker(cap, tmp_path)
    r = t.check()
    assert r.kind == "full" and r.reference == "cross"
    assert abs(r.focus - 331) <= 1 and cap.focus == r.focus
    assert cap.sets[0] == focus.FULL[0] - focus.BELOW            # it starts under the sweep
    assert cap.sets[-2:] == [r.focus - focus.BELOW, r.focus]     # and arrives at the best from below
    cross_distance = focus.cross_target(t.cal)[1]
    assert r.offset == pytest.approx(r.peak - t.model.focus(cross_distance), abs=0.01)


def test_later_checks_are_narrow_and_follow_the_drift(lens, tmp_path):
    cap = FakeLens(best=320.0)
    lens.cap, lens.cross_box = cap, None
    t = _tracker(cap, tmp_path)
    t.check()
    cap.best, n = 324.0, len(cap.sets)
    r = t.check()
    assert r.kind == "narrow" and abs(r.focus - 324) <= 1
    steps = 2 * focus.NARROW_HALF // focus.NARROW_STEP + 1
    assert len(cap.sets) - n == steps + 1 + 2                  # the steps, the start from below, the arrival


def test_a_peak_at_the_narrow_windows_edge_widens_to_a_full_search(lens, tmp_path):
    cap = FakeLens(best=320.0)
    lens.cap, lens.cross_box = cap, None
    t = _tracker(cap, tmp_path)
    t.check()
    cap.best = 345.0
    r = t.check()
    assert r.kind == "full" and abs(r.focus - 345) <= 1


def test_a_covered_cross_uses_a_body_else_keeps_the_last(lens, tmp_path):
    cap = FakeLens(best=330.0)
    lens.cap = cap
    t = _tracker(cap, tmp_path)
    lens.cross_box = None
    first = t.check()
    lens.cross_box = focus.cross_target(t.cal)[0]
    cap.flat = True
    body = [np.array([[580, 560], [640, 560], [640, 585], [580, 585]], float)]
    r = t.check(polygons=body)
    assert r.reference == "body" and r.clear
    kept = t.check()                                           # no bodies given: nothing to measure on
    assert kept.reference == "kept" and kept.focus == r.focus and "kept" in t.status()
    assert first.reference == "cross"


def test_results_are_saved_and_read_back_while_fresh(lens, tmp_path):
    cap = FakeLens(best=328.0)
    lens.cap, lens.cross_box = cap, None
    t = _tracker(cap, tmp_path)
    r = t.check()
    got = focus.current(path=tmp_path / "f.json")
    assert got.focus == r.focus and got.offset == r.offset
    assert focus.current(max_age_s=-1, path=tmp_path / "f.json") is None
    assert focus.current(path=tmp_path / "missing.json") is None


def test_peak_of_interpolates_and_flags_edges_and_flat_curves():
    best, edge, clear = focus.peak_of([300, 305, 310], [10.0, 20.0, 10.0])
    assert best == pytest.approx(305) and not edge and clear
    assert focus.peak_of([300, 305, 310], [30.0, 20.0, 10.0])[1]
    assert not focus.peak_of([300, 305, 310], [10.0, 10.5, 10.2])[2]

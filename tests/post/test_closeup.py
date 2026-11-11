import cv2
import numpy as np
import pytest

from robot_offline import config
from robot_offline.post.bands import closeup, crops
from robot_offline.pre import rectify


def test_the_focus_model_follows_the_inverse_of_distance():
    a, b = 137.0, 43000.0
    m = closeup.FocusModel([[d, a + b / d] for d in (230.0, 250.0, 275.0)])
    assert abs(m.a - a) < 1e-6 and abs(m.b - b) < 1e-3
    assert m.focus(240.0) == round(a + b / 240.0)
    assert m.focus(230.0) > m.focus(275.0)          # nearer: a higher setting, as the sweep found


def test_points_at_one_distance_give_one_focus_everywhere():
    m = closeup.FocusModel([[250.0, 300.0], [251.0, 304.0]])
    assert m.focus(200.0) == m.focus(300.0) == 302


def test_no_points_says_how_to_get_some():
    with pytest.raises(ValueError, match="calibrate"):
        closeup.FocusModel([])


def test_the_config_has_a_model_that_suits_the_rig():
    # Sweeps on 2026-11-09 put the front of the plate (about 233 mm) at 322 to 326, the back (275) at 290 to 298.
    m = closeup.FocusModel.from_config()
    assert 318 <= m.focus(233.0) <= 330 and 288 <= m.focus(275.0) <= 302


def test_the_peak_is_found_between_samples_and_not_at_the_ends():
    focuses = [290, 294, 298, 302, 306]
    curve = [-(f - 299.0) ** 2 for f in focuses]
    assert abs(closeup.best_focus(focuses, curve) - 299.0) < 1e-6
    assert closeup.best_focus(focuses, [1, 2, 3, 4, 5]) is None
    assert closeup.best_focus(focuses, [5, 4, 3, 2, 1]) is None


def test_saving_points_changes_only_that_line(tmp_path):
    src = (config.CONFIG_DIR / "camera.toml").read_text(encoding="utf-8")
    path = tmp_path / "camera.toml"
    path.write_text(src, encoding="utf-8")
    closeup.save_points([(240.04, 316.66), (270.0, 297.0)], path)
    out = path.read_text(encoding="utf-8")
    assert "points = [[240.0, 316.7], [270.0, 297.0]]" in out
    assert [l for l in out.splitlines() if not l.startswith("points =")] == \
        [l for l in src.splitlines() if not l.startswith("points =")]


def test_a_body_further_back_is_further_from_the_lens():
    cal = rectify.Calibration.load(config.DATA_DIR / "captures" / "2026-11-09_c" / "rectify.json") \
        if (config.DATA_DIR / "captures" / "2026-11-09_c" / "rectify.json").exists() else None
    if cal is None:
        pytest.skip("needs a real calibration")
    st = cal.settings
    # The lens is off the plate's -x side (camera_mm about (-147, -145, 182)), so front to back runs along x.
    front = rectify.mm_to_rect(np.array([[st.cross_mm[0] - 30, st.cross_mm[1]]]), st)[0]
    back = rectify.mm_to_rect(np.array([[st.cross_mm[0] + 30, st.cross_mm[1]]]), st)[0]
    square = np.array([[-20, -8], [20, -8], [20, 8], [-20, 8]], float)
    d_front, d_back = closeup.distance_mm(cal, square + front), closeup.distance_mm(cal, square + back)
    assert 200 < d_front and d_back < 320 and d_back - d_front > 30


class FakeCap:
    """Frames whose content says the focus they were taken at; records every focus set."""

    def __init__(self):
        self.focus, self.set_log = 306.0, []

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOCUS:
            self.focus = value
            self.set_log.append(value)

    def read(self):
        return True, np.full((64, 64, 3), int(self.focus) % 256, np.uint8)


def test_one_closeup_per_body_in_focus_order_and_the_focus_put_back(monkeypatch):
    distances = {0: 275.0, 1: 233.0, 2: 250.0}       # body index -> mm from the lens
    monkeypatch.setattr(closeup, "distance_mm", lambda cal, p: distances[int(p[0, 0])])
    monkeypatch.setattr(closeup, "white_of", lambda cal, frame: (frame, (1.0, 1.0, 1.0), 200.0))
    seen = []

    def cut(frame, cal, polygon, gains):
        seen.append(int(frame[0, 0, 0]))
        return crops.Crop(frame, np.zeros(frame.shape[:2], np.uint8), np.zeros(2), 0.0, 6.0, 2.5)

    monkeypatch.setattr(crops, "cut", cut)
    cap = FakeCap()
    model = closeup.FocusModel([[233.0, 322.0], [275.0, 294.0]])
    polygons = [np.full((4, 2), i, float) for i in range(3)]
    out = closeup.take(cap, None, polygons, model=model, camera={"camera": {"focus": 306}}, offset=0.0, restore=306)
    assert [c.body for c in out] == [0, 1, 2]
    assert [c.focus for c in out] == [model.focus(275.0), model.focus(233.0), model.focus(250.0)]
    assert cap.set_log[:-2] == sorted(cap.set_log[:-2])     # the lens moves one way only
    assert cap.set_log[-2:] == [296, 306]                   # and back to the focus given, from below
    assert seen == [f % 256 for f in sorted(c.focus for c in out)]   # each crop from its own capture
    assert not any(c.found_again for c in out)              # no segmentation model given


def test_the_focus_plan_shifts_every_body_by_the_measured_offset(monkeypatch):
    distances = {0: 275.0, 1: 233.0}
    monkeypatch.setattr(closeup, "distance_mm", lambda cal, p: distances[int(p[0, 0])])
    model = closeup.FocusModel([[233.0, 322.0], [275.0, 294.0]])
    polygons = [np.full((4, 2), i, float) for i in range(2)]
    plain = closeup.focus_plan(None, polygons, model)
    shifted = closeup.focus_plan(None, polygons, model, offset=9.0)
    assert [f for f, _ in shifted] == [f + 9 for f, _ in plain]
    assert [i for _, i in shifted] == [0, 1]                 # lowest focus first

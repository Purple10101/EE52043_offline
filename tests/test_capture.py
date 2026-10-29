import tomllib

import cv2
import numpy as np
import pytest

from robot_offline import capture, config, rectify


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return capture.Session("2026-10-23_a", "room lights", "test", config.load("camera"))


def _meta(session):
    with open(session.dir / "session.toml", "rb") as f:
        return tomllib.load(f)


def test_empty_plate_saves_the_background_and_calibration(session, plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    assert session.background.exists()
    assert (session.dir / "rectify.json").exists()
    meta = _meta(session)
    assert meta["session"] == "2026-10-23_a" and meta["lighting"] == "room lights"
    assert meta["camera"]["focus"] == config.load("camera")["camera"]["focus"]
    assert np.allclose(meta["plate"]["corners_px"], cal.corners_px, atol=0.01)


def test_scenes_are_numbered_recorded_and_undone(session, plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    assert session.save_scene(plate_frame, "resistors", cal, True) == "0001.png"
    assert session.save_scene(plate_frame, "mixed", cal, False) == "0002.png"
    assert (session.dir / "rectified" / "0002.png").exists()
    frames = _meta(session)["frames"]
    assert [(f["file"], f["kind"], f["plate_found"]) for f in frames] == [
        ("0001.png", "resistors", True), ("0002.png", "mixed", False)]
    assert session.count("resistors") == 1 and session.count("mixed") == 1

    assert session.undo() == "0002.png"
    assert not (session.dir / "0002.png").exists() and not (session.dir / "rectified" / "0002.png").exists()
    assert len(_meta(session)["frames"]) == 1


def test_a_frame_from_a_moved_camera_gets_its_own_calibration(session, plate_frame, moved_plate):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    moved, truth = moved_plate(1.5, -1.0)
    frame_cal, _ = rectify.recalibrate(moved, session.last_cal)
    assert np.abs(frame_cal.corners_px - truth).max() < 1.5
    session.save_scene(moved, "resistors", frame_cal, True)
    entry = _meta(session)["frames"][0]
    assert entry["plate_shift_px"] == pytest.approx(np.abs(truth - cal.corners_px).max(), abs=1.5)
    assert np.allclose(session.frame_cal(entry).corners_px, frame_cal.corners_px, atol=0.01)
    # The rectified copy matches the unmoved frame's, because it used the frame's own corners.
    saved = cv2.imread(str(session.dir / "rectified" / "0001.png")).astype(float)
    expected = rectify.white_balance(rectify.rectify(plate_frame, cal), cal.settings)[0].astype(float)
    assert np.median(np.abs(saved - expected)) < 3


def test_the_plan_is_recorded_and_kept_when_continuing(tmp_path, monkeypatch, plate_frame):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    s = capture.Session("2026-10-23_b", "", "", config.load("camera"), plan="cutouts")
    s.save_empty(plate_frame, rectify.calibrate(plate_frame)[0])
    assert _meta(s)["plan"] == "cutouts"
    assert capture.Session("2026-10-23_b", "", "", config.load("camera")).plan == "cutouts"


def test_objects_are_outlined_separately_and_edge_ones_flagged():
    st = rectify.Settings.from_config()
    empty = np.full((1200, 1200, 3), 200, np.uint8)
    rect = empty.copy()
    cv2.rectangle(rect, (400, 400), (470, 425), (60, 150, 40), -1)      # a body, well inside
    cv2.line(rect, (300, 600), (500, 620), (130, 130, 130), 6)          # a lead on its own
    cv2.rectangle(rect, (105, 700), (170, 725), (60, 150, 40), -1)      # a body at the plate's edge
    out, found, near = capture.outline_objects(rect, empty, st, threshold=25)
    assert (found, near) == (3, 1)
    assert out.shape == rect.shape and not np.array_equal(out, rect)
    together = rect.copy()
    cv2.rectangle(together, (470, 400), (540, 425), (60, 150, 40), -1)  # touching the first body
    assert capture.outline_objects(together, empty, st, threshold=25)[1] == 3


@pytest.mark.parametrize("text, ohms", [("470", 470), ("47R", 47), ("10R5", 10.5), ("4k7", 4700),
                                         ("4.7k", 4700), ("1K5", 1500), ("10k", 10_000), ("1M", 1e6),
                                         (" 220 ohm", 220)])
def test_values_are_read_as_written_on_bags_and_schematics(text, ohms):
    assert capture.parse_ohms(text) == pytest.approx(ohms)


@pytest.mark.parametrize("text", ["", "k", "4.7k5", "47x", "1..2"])
def test_nonsense_values_are_refused(text):
    with pytest.raises(ValueError):
        capture.parse_ohms(text)


def test_reference_frames_record_their_value(session, plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    session.save_scene(plate_frame, "reference", cal, True, {"value_ohm": 4700.0})
    assert _meta(session)["frames"][0]["value_ohm"] == 4700.0
    assert capture.format_ohms(4700.0) == "4.7k" and capture.format_ohms(47.0) == "47R"


def test_reference_frames_are_saved_with_value_calibration_and_preview(tmp_path, plate_frame):
    from robot_offline.bands import reference

    cal, sides = rectify.calibrate(plate_frame)
    rect, _ = rectify.white_balance(rectify.rectify(plate_frame, cal), cal.settings)
    body = np.array([[500, 500], [570, 500], [570, 525], [500, 525]], float)
    reference.save(tmp_path, plate_frame, cal, 1000, rect, [body], [0.9], sides)
    entry = reference.save(tmp_path, plate_frame, cal, 3000, rect, [], [], sides)
    assert entry["file"] == "0002.png" and entry["bodies"] == 0
    for name in ["0001.png", "0001_rectify.json", "previews/0001.jpg", "previews/0002.jpg"]:
        assert (tmp_path / name).exists()
    index = reference.load_index(tmp_path)
    assert [e["value_ohm"] for e in index] == [1000.0, 3000.0]
    assert np.allclose(index[0]["polygons_rect"], body)


def test_a_session_continues_where_it_stopped(session, plate_frame, moved_plate):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    moved, truth = moved_plate(0.8, 0.4)
    session.save_scene(moved, "resistors", rectify.recalibrate(moved, cal)[0], True)
    again = capture.Session("2026-10-23_a", "", "", config.load("camera"))
    assert again.cal is not None and again.count("resistors") == 1 and again.lighting == "room lights"
    assert np.abs(again.last_cal.corners_px - truth).max() < 1.5
    assert again.save_scene(plate_frame, "resistors", cal, True) == "0002.png"

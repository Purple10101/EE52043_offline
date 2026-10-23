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


def _moved(frame, dx, dy):
    """The frame as if the camera had moved: shifted by (dx, dy) px."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)


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


def test_a_frame_from_a_moved_camera_gets_its_own_calibration(session, plate_frame, plate_corners):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    moved = _moved(plate_frame, 18, -11)
    frame_cal, _ = rectify.recalibrate(moved, session.last_cal)
    assert np.abs(frame_cal.corners_px - (plate_corners + (18, -11))).max() < 1.0
    session.save_scene(moved, "resistors", frame_cal, True)
    entry = _meta(session)["frames"][0]
    assert entry["plate_shift_px"] == pytest.approx(18, abs=1)
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


def test_a_session_continues_where_it_stopped(session, plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    session.save_empty(plate_frame, cal)
    moved = _moved(plate_frame, 10, 5)
    session.save_scene(moved, "resistors", rectify.recalibrate(moved, cal)[0], True)
    again = capture.Session("2026-10-23_a", "", "", config.load("camera"))
    assert again.cal is not None and again.count("resistors") == 1 and again.lighting == "room lights"
    assert np.abs(again.last_cal.corners_px - (cal.corners_px + (10, 5))).max() < 1.0
    assert again.save_scene(plate_frame, "resistors", cal, True) == "0002.png"

import json

import cv2
import numpy as np
import pytest

from robot_offline.pre import rectify

from conftest import camera_position

# The configured plate: the test frames are drawn from it (conftest.render_plate), crosshair included.
SETTINGS = rectify.Settings.from_config()
FRAME = (4608, 2592)
K = np.array([[3386.0, 0.0, 2304.0], [0.0, 3386.0, 1296.0], [0.0, 0.0, 1.0]])
CORNER_PX = 1.5         # corners found to this, about 0.1 mm on the near side


def test_finds_plate_corners_to_a_pixel(plate_frame, plate_corners):
    corners, sides = rectify.find_plate_corners(plate_frame)
    assert np.abs(corners - plate_corners).max() < CORNER_PX
    assert all(share > 0.8 and rms < 0.5 for share, rms in sides)


def test_the_crosshair_fixes_the_near_side_and_the_face_agrees(plate_frame):
    fit = rectify.find_plate(plate_frame, SETTINGS)
    assert fit.near_by == "crosshair"
    assert abs(fit.checks["face bottom"]) < 0.15                        # mm


def test_the_camera_is_found_where_it_is(plate_frame):
    fit = rectify.find_plate(plate_frame, SETTINGS)
    assert np.abs(np.array(fit.summary["camera_mm"]) - camera_position()).max() < 1.0
    assert abs(fit.summary["focal_px"] / 3589.6 - 1) < 0.01


def test_a_lit_front_face_does_not_move_the_near_edge(render):
    # The old failure: a face lit like the top made the face's bottom look like the top's edge.
    frame, corners = render(face=(205, 205, 205))
    found, _ = rectify.find_plate_corners(frame)
    assert np.abs(found - corners).max() < CORNER_PX


def test_a_covered_crosshair_hands_over_to_the_face_bottom(render):
    for face in ((45, 45, 45), (205, 205, 205)):
        frame, corners = render(face=face, cover_cross=True)
        fit = rectify.find_plate(frame, SETTINGS)
        assert fit.near_by == "face bottom"
        found, _ = rectify.find_plate_corners(frame)
        assert np.abs(found - corners).max() < CORNER_PX


def test_calibration_puts_the_plate_on_its_square(plate_frame, plate_corners):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    expected = rectify.mm_to_rect(SETTINGS.plate_corners_mm, SETTINGS)
    assert np.abs(rectify.to_rectified(plate_corners, cal) - expected).max() < CORNER_PX
    assert np.allclose(expected[0], (1099.5, 99.5))       # (50, -80) mm: far edge, robot side


def test_rectified_image_shows_the_plate_inside_its_outline(plate_frame):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    rect = rectify.rectify(plate_frame, cal)
    assert rect.shape == (1200, 1200, 3)
    (u, v), = rectify.mm_to_rect([[0.0, -100.0]], SETTINGS).round().astype(int)
    assert rect[v, u].mean() > 180                         # on the plate
    (u, v), = rectify.mm_to_rect([[53.0, -130.0]], SETTINGS).round().astype(int)
    assert rect[v, u].mean() < 150                         # 3 mm beyond the far edge


def test_points_round_trip_with_lens_distortion(plate_corners):
    h = cv2.getPerspectiveTransform(plate_corners.astype(np.float32),
                                    rectify.mm_to_rect(SETTINGS.plate_corners_mm, SETTINGS).astype(np.float32))
    cal = rectify.Calibration(h, plate_corners, FRAME, SETTINGS, K, np.array([-0.08, 0.05, 0.001, -0.001, 0.0]))
    points = np.random.default_rng(1).uniform((1500, 500), (3100, 1850), (50, 2))
    back = rectify.to_full(rectify.to_rectified(points, cal), cal)
    assert np.abs(back - points).max() < 0.01


def test_mm_and_rectified_px_round_trip():
    points = np.array([[0.0, -130.0], [-60.0, -70.0], [12.3, -171.9]])
    assert np.allclose(rectify.rect_to_mm(rectify.mm_to_rect(points, SETTINGS), SETTINGS), points)
    assert SETTINGS.size == (1200, 1200)


def test_white_balance_makes_the_plate_neutral(plate_frame):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    tinted = np.clip(plate_frame * np.array([0.8, 1.0, 0.9]), 0, 255).astype(np.uint8)
    balanced, gains = rectify.white_balance(rectify.rectify(tinted, cal), SETTINGS)
    (u, v), = rectify.mm_to_rect([[-20.0, -150.0]], SETTINGS).round().astype(int)
    b, g, r = balanced[v - 20:v + 20, u - 20:u + 20].reshape(-1, 3).mean(0)
    assert abs(b - g) < 2 and abs(r - g) < 2
    assert gains[0] > gains[2] > gains[1]


def test_calibration_saves_and_loads(tmp_path, plate_frame):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    cal.save(tmp_path / "rectify.json")
    loaded = rectify.Calibration.load(tmp_path / "rectify.json")
    assert np.allclose(loaded.homography, cal.homography)
    assert loaded.frame_size == FRAME and loaded.settings.size == SETTINGS.size
    assert np.array_equal(rectify.rectify(plate_frame, loaded), rectify.rectify(plate_frame, cal))


def test_the_plate_is_found_near_its_last_corners_after_a_move(moved_plate, plate_corners):
    moved, truth = moved_plate(2.0, -1.5)                                 # about 25 px
    corners, _ = rectify.find_plate_corners(moved, near=plate_corners)
    assert np.abs(corners - truth).max() < CORNER_PX


def test_recalibrating_keeps_the_lens_and_follows_the_camera(plate_frame, moved_plate):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    moved, truth = moved_plate(-1.0, 2.0, 1.0)
    again, _ = rectify.recalibrate(moved, cal)
    assert np.array_equal(again.camera_matrix, cal.camera_matrix)
    assert np.abs(rectify.plate_outline_px(again) - truth).max() < CORNER_PX


def test_the_model_sees_the_plate_at_its_training_brightness_whatever_the_exposure(plate_frame):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    (u, v), = rectify.mm_to_rect([[-20.0, -150.0]], SETTINGS).round().astype(int)
    for scale in (0.45, 1.0, 1.2):                                     # a dim room, as drawn, brighter
        frame = np.clip(plate_frame * scale, 0, 255).astype(np.uint8)
        model_view, balanced = rectify.model_input(frame, cal)
        assert abs(model_view[v - 20:v + 20, u - 20:u + 20].mean() - rectify.MODEL_PLATE_LEVEL) < 3
        assert np.array_equal(model_view[v, u], balanced[v, u])
    kept, _ = rectify.white_balance(rectify.rectify(np.clip(plate_frame * 0.45, 0, 255).astype(np.uint8), cal), SETTINGS)
    assert kept[v - 20:v + 20, u - 20:u + 20].mean() < 110                # without a level, brightness is kept


def test_the_plate_model_is_drawn_where_the_plate_is(plate_frame, plate_corners):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    model = rectify.plate_model_px(cal)
    # The top's outline passes its corners' rounding: each true corner is about r(√2 - 1) mm outside it.
    for corner in plate_corners:
        assert 2 < np.linalg.norm(model["top"] - corner, axis=1).min() < 40
    # The face's bottom is below the near edge, and the crosshair's arms cross at the plate's middle.
    tl, tr, br, bl = plate_corners
    assert model["face bottom"][:, 1].min() > min(bl[1], br[1])
    (a, b), (c, d) = model["crosshair"]
    assert np.linalg.norm((a + b) / 2 - (c + d) / 2) < 0.5


def test_calibrations_from_before_the_plate_model_still_load(tmp_path, plate_frame):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    old = {"version": 1, "frame_size": list(FRAME), "corners_px": cal.corners_px.tolist(),
           "homography": cal.homography.tolist(), "plate_corners_mm": SETTINGS.plate_corners_mm.tolist(),
           "px_per_mm": 10.0, "margin_mm": 10.0, "camera_matrix": None, "dist_coeffs": None}
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = rectify.Calibration.load(tmp_path / "old.json")
    assert loaded.settings.corner_radius_mm == 4.0 and loaded.settings.top_z_mm == -22.0
    assert loaded.fit is None and loaded.camera_matrix is None
    again, _ = rectify.recalibrate(plate_frame, loaded)                  # fits the lens it lacks
    assert again.camera_matrix is not None


def test_a_lead_lying_along_an_edge_is_not_taken_for_it(plate_frame, plate_corners):
    frame = plate_frame.copy()
    tl, _, _, bl = plate_corners
    inward = np.array([45.0, 0.0])        # a lead 45 px inside the left edge, almost parallel to it
    a, b = tl + (0.15 * (bl - tl)) + inward, tl + (0.85 * (bl - tl)) + inward + (25, 0)
    cv2.line(frame, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (70, 70, 70), 9, cv2.LINE_AA)
    for near in (None, plate_corners):
        corners, _ = rectify.find_plate_corners(frame, near=near)
        assert np.abs(corners - plate_corners).max() < CORNER_PX


def test_nothing_off_the_plate_survives_the_mask():
    st = rectify.Settings.from_config()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (1200, 1200, 3), dtype=np.uint8)       # busy surroundings
    plate = rectify.plate_mask(st)
    img[plate] = (198, 201, 203)
    out = rectify.mask_outside_plate(img, st)
    assert np.array_equal(out[plate], img[plate])                         # the plate is untouched
    assert (out[~plate] == (198, 201, 203)).all()                        # the rest is the plate's grey
    # The mask follows the plate: its edge at px 100..1099, rounded corners cut away.
    assert plate[600, 100] and not plate[600, 99] and plate[600, 1099] and not plate[600, 1100]
    assert not plate[100, 100] and plate[140, 140]


def test_a_frame_without_the_plate_is_rejected():
    dark = np.full((FRAME[1], FRAME[0], 3), (60, 110, 150), np.uint8)
    with pytest.raises(rectify.PlateNotFound):
        rectify.find_plate_corners(dark)

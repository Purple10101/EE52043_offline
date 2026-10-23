import cv2
import numpy as np
import pytest

from robot_offline import rectify

SETTINGS = rectify.Settings(np.array([[50.0, -80.0], [50.0, -180.0], [-50.0, -180.0], [-50.0, -80.0]]), 10.0, 10.0)
FRAME = (4608, 2592)
K = np.array([[3386.0, 0.0, 2304.0], [0.0, 3386.0, 1296.0], [0.0, 0.0, 1.0]])


def test_finds_plate_corners_to_a_pixel(plate_frame, plate_corners):
    corners, sides = rectify.find_plate_corners(plate_frame)
    assert np.abs(corners - plate_corners).max() < 1.0
    assert all(share > 0.8 for share, _ in sides)


def test_calibration_puts_the_plate_on_its_square(plate_frame, plate_corners):
    cal, _ = rectify.calibrate(plate_frame, SETTINGS)
    expected = rectify.mm_to_rect(SETTINGS.plate_corners_mm, SETTINGS)
    assert np.abs(rectify.to_rectified(plate_corners, cal) - expected).max() < 1.0
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


def test_the_plate_is_found_near_its_last_corners_after_a_move(plate_frame, plate_corners):
    moved = cv2.warpAffine(plate_frame, np.float32([[1, 0, -25], [0, 1, 16]]), FRAME, borderMode=cv2.BORDER_REPLICATE)
    corners, _ = rectify.find_plate_corners(moved, near=plate_corners)
    assert np.abs(corners - (plate_corners + (-25, 16))).max() < 1.0


def test_a_lead_lying_along_an_edge_is_not_taken_for_it(plate_frame, plate_corners):
    frame = plate_frame.copy()
    tl, _, _, bl = plate_corners
    inward = np.array([45.0, 0.0])        # a lead 45 px inside the left edge, almost parallel to it
    a, b = tl + (0.15 * (bl - tl)) + inward, tl + (0.85 * (bl - tl)) + inward + (25, 0)
    cv2.line(frame, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (70, 70, 70), 9, cv2.LINE_AA)
    for near in (None, plate_corners):
        corners, _ = rectify.find_plate_corners(frame, near=near)
        assert np.abs(corners - plate_corners).max() < 1.0


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

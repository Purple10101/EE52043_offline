import cv2
import numpy as np

from robot_offline.pre import rectify
from robot_offline.post.bands import crops


def _rotated_rect(centre, length, width, angle_deg):
    return cv2.boxPoints((tuple(centre), (length, width), angle_deg)).astype(np.float64)


def test_body_frame_finds_the_long_axis_and_size():
    poly = _rotated_rect((600, 500), 66, 28, 30)
    centre, axis, normal, length, width = crops.body_frame(poly)
    assert np.allclose(centre, (600, 500), atol=0.5)
    assert abs(abs(axis @ np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])) - 1) < 1e-3
    assert abs(length - 66) < 0.5 and abs(width - 28) < 0.5


def test_a_body_is_cut_straight_at_the_crop_scale(plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    # A striped body drawn in the rectified view, then put into a raw frame through the calibration.
    rect = np.full((1200, 1200, 3), 200, np.uint8)
    poly = _rotated_rect((600, 600), 66, 28, 35)
    cv2.fillPoly(rect, [np.round(poly).astype(np.int32)], (60, 160, 40))
    centre, axis, _, _, _ = crops.body_frame(poly)
    for k in (-1, 0, 1):                               # three dark bands across the body
        band = _rotated_rect(centre + axis * 14 * k, 5, 28, 35)
        cv2.fillPoly(rect, [np.round(band).astype(np.int32)], (20, 20, 160))
    raw = cv2.warpPerspective(rect, np.linalg.inv(cal.homography), cal.frame_size, flags=cv2.INTER_LINEAR)
    crop = crops.cut(raw, cal, poly)
    s = crops.CROP_PX_PER_MM
    assert abs(crop.length_mm - 6.6) < 0.1 and abs(crop.width_mm - 2.8) < 0.1
    assert abs(crop.image.shape[1] - (6.6 + 2 * crops.MARGIN_MM[0]) * s) <= 1
    # The body lies along the crop: its mask is long and thin horizontally, and the bands are upright,
    # so the red band columns repeat down the crop's middle rows.
    ys, xs = np.nonzero(crop.mask > 127)
    assert np.ptp(xs) > 2 * np.ptp(ys)
    red = crop.image[..., 2].astype(int) - crop.image[..., 1].astype(int) > 60
    mid = crop.image.shape[0] // 2
    rows = red[mid - 8:mid + 9]
    assert rows.any(axis=0).sum() > 0 and (rows.sum(axis=0)[rows.any(axis=0)] >= 15).mean() > 0.8

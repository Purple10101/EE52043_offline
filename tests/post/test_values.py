import numpy as np
import torch

from robot_offline.post.bands import learned, values
from robot_offline.pre import rectify


def _cal():
    st = rectify.Settings.from_config()
    corners = np.array([[1000, 500], [2600, 500], [2600, 2100], [1000, 2100]], float)
    target = rectify.mm_to_rect(st.plate_corners_mm, st)
    import cv2
    h = cv2.getPerspectiveTransform(corners.astype(np.float32), target.astype(np.float32))
    return rectify.Calibration(h.astype(float), corners, (4608, 2592), st, None, None, None)


def test_a_reading_per_body_in_order(tmp_path):
    net = learned.BandCTC(8)
    torch.save({"width": 8, "kind": "ctc", "min_confidence": 0.9, "state": net.state_dict()}, tmp_path / "r.pt")
    try:
        reader = values.load(tmp_path / "r.pt")
        frame = np.full((2592, 4608, 3), 200, np.uint8)
        polys = [np.array([[500, 500], [570, 500], [570, 525], [500, 525]], float),
                 np.array([[700, 600], [770, 600], [770, 625], [700, 625]], float)]
        out = values.read_frame(reader, frame, _cal(), polys)
        assert len(out) == 2 and all(isinstance(r, learned.Reading) for r in out)
        assert all(r.value is None or r.confidence >= reader.min_confidence for r in out)
        assert values.read_frame(reader, frame, _cal(), []) == []
    finally:
        learned.use_values(None)


def test_the_sorter_range():
    assert values.in_sorter_range(4700.0) and values.in_sorter_range(10.0)
    assert not values.in_sorter_range(15000.0) and not values.in_sorter_range(None)

import cv2
import numpy as np
import torch

from robot_offline.post import resistor_tag as rt


def _view_with_bar(centre, length_px, width_px, angle_deg, colour=(40, 90, 200)):
    view = np.full((600, 600, 3), 200, np.uint8)
    box = cv2.boxPoints(((centre[0], centre[1]), (length_px, width_px), angle_deg))
    cv2.fillPoly(view, [np.round(box).astype(np.int32)], colour)
    return view, box


def test_cutout_turns_the_long_side_along_x_at_true_scale():
    view, box = _view_with_bar((300, 300), 70, 25, 35)
    crop = rt.cutout(view, box)
    assert crop.shape == (rt.SIZE[1], rt.SIZE[0], 3)
    body = (crop == (40, 90, 200)).all(2)
    ys, xs = np.nonzero(body)
    assert 64 <= xs.max() - xs.min() + 1 <= 74          # 7 mm long at 10 px/mm, along x
    assert 20 <= ys.max() - ys.min() + 1 <= 29
    cy, cx = rt.SIZE[1] / 2, rt.SIZE[0] / 2
    assert abs(xs.mean() - cx) < 3 and abs(ys.mean() - cy) < 3


def test_cutout_keeps_a_ring_round_the_outline_and_blacks_out_the_rest():
    view, box = _view_with_bar((300, 300), 70, 25, 0)
    crop = rt.cutout(view, box)
    plate = (crop == 200).all(2)
    assert plate.any()                                  # the 1 mm ring (lead stubs) is kept
    ring = int(round(rt.KEEP_MM * rt.PX_PER_MM))
    ys, xs = np.nonzero(crop.sum(2) > 0)
    assert xs.max() - xs.min() + 1 <= 70 + 2 * ring + 4
    assert (crop[:5] == 0).all() and (crop[:, :20] == 0).all()


def test_classify_gives_one_tag_per_polygon_in_order():
    net = rt.Net(8).eval()
    torch.nn.init.zeros_(net.head[-1].weight)
    torch.nn.init.constant_(net.head[-1].bias, 3.0)     # every crop: P(resistor) = 0.95
    view, box = _view_with_bar((300, 300), 70, 25, 10)
    out = rt.classify(net, view, [box, box + 100])
    assert len(out) == 2 and all(t == "resistor" and p > 0.9 for p, t in out)
    assert rt.classify(net, view, []) == []
    assert rt.tag(0.5) == "unsure" and rt.tag(0.1) == "not" and rt.tag(0.9) == "resistor"


def test_load_round_trip(tmp_path):
    net = rt.Net(8)
    torch.save({"state": net.state_dict(), "width": 8}, tmp_path / "t.pt")
    again = rt.load(tmp_path / "t.pt")
    x = rt.to_tensor([np.zeros((rt.SIZE[1], rt.SIZE[0], 3), np.uint8)])
    with torch.no_grad():
        assert torch.allclose(net.eval()(x), again(x))

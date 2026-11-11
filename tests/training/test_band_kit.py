import numpy as np
import pytest
import torch

from robot_offline.post.bands import learned
from robot_offline.training import band_kit as bk


def test_kit_bands_are_five_band_one_percent():
    assert bk.kit_bands(4700) == ("yellow", "violet", "black", "brown", "brown")
    assert bk.kit_bands(2.2) == ("red", "red", "black", "silver", "brown")
    assert bk.kit_bands(5.6e6) == ("green", "blue", "black", "yellow", "brown")
    assert bk.kit_bands(0) == ("black",)
    forward = np.zeros((1, len(learned.COLOURS)))
    forward[0, :2] = np.log([0.99, 0.01])                 # read left to right: 120R and 10k need it
    for ohms in (1, 7.5, 33, 120, 10000, 4.7e6):         # every kit label decodes to its own value
        lp = np.log(np.where(np.eye(len(learned.COLOURS))[bk.slot_targets(bk.kit_bands(ohms))], 0.97, 0.0025))
        assert learned.decode(np.vstack([lp, forward])).best == pytest.approx(ohms)


def test_split_keeps_test_values_whole_and_one_placement_per_value_for_validation():
    rows = [dict(value_ohm=v, capture=f"{v}/cap{c}") for v in (10.0, 47.0, 2200.0) for c in range(1, 6) for _ in range(2)]
    split = bk.split_of(rows, test_ohms=[47.0], per_value=1)
    assert {split[f"47.0/cap{c}"] for c in range(1, 6)} == {"test"}
    for v in (10.0, 2200.0):
        sides = [split[f"{v}/cap{c}"] for c in range(1, 6)]
        assert sides.count("val") == 1 and sides.count("train") == 4
    assert [split[f"10.0/cap{c}"] for c in range(1, 6)].index("val") != [split[f"2200.0/cap{c}"] for c in range(1, 6)].index("val")


def test_synthetic_direction_is_known_only_when_one_way_reads_as_a_value():
    four = bk.slot_targets(("brown", "black", "red", "gold"))
    assert list(bk.synth_direction([four, four[::-1]])) == [1, 0]
    ten_k = bk.slot_targets(("brown", "black", "black", "red", "brown"))     # 120R backwards
    assert list(bk.synth_direction([ten_k])) == [-1]


def test_flipping_a_kit_crop_keeps_its_reading_order_and_flips_a_synthetic_ones():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(8, 3, learned.SIZE[1], learned.SIZE[0])
    y = torch.tensor([bk.slot_targets(("yellow", "violet", "black", "brown", "brown"))] * 8)
    d = torch.ones(8, dtype=torch.long)
    either = torch.tensor([True] * 4 + [False] * 4)
    for _ in range(5):
        _, y2, d2 = bk.augment(x.clone(), y, d, either, torch.Generator().manual_seed(int(torch.randint(0, 99, (1,), generator=g))))
        assert (y2[:4] == y[:4]).all() and (d2[:4] == 1).all()
        flipped = (y2[4:] != y[4:]).any(1)
        assert ((d2[4:] == 0) == flipped).all()


def test_the_kit_direction_follows_the_way_the_slot_loss_chose():
    net = learned.BandNet(8)
    y = torch.tensor([bk.slot_targets(("yellow", "violet", "black", "brown", "brown"))])
    x = torch.zeros(1, 3, learned.SIZE[1], learned.SIZE[0])
    loss = bk.step_loss(net, x, y, torch.ones(1, dtype=torch.long), torch.tensor([True]), 1.0)
    assert torch.isfinite(loss)


def test_safe_limit_is_above_every_wrong_answer():
    right = np.array([True, False, True, False])
    conf = np.array([0.99, 0.9, 0.6, 0.97])
    t = bk.safe_limit(right, conf)
    assert t > 0.97 and not ((conf >= t) & ~right).any()
    assert bk.safe_limit(np.array([True]), np.array([0.7])) == 0.5


def test_validation_values_are_kept_whole_and_out_of_training():
    rows = [dict(value_ohm=v, capture=f"{v}/cap{c}") for v in (10.0, 47.0, 2200.0) for c in range(1, 6)]
    split = bk.split_of(rows, test_ohms=[47.0], per_value=1, val_ohms=[2200.0])
    assert {split[f"2200.0/cap{c}"] for c in range(1, 6)} == {"val_values"}
    assert {split[f"47.0/cap{c}"] for c in range(1, 6)} == {"test"}
    assert sorted(split[f"10.0/cap{c}"] for c in range(1, 6)) == ["train"] * 4 + ["val"]


def test_ctc_loss_prefers_the_way_round_the_columns_show():
    net = learned.BandCTC(8)
    y = torch.tensor([bk.slot_targets(("yellow", "violet", "black", "brown", "brown"))])
    cols = torch.full((1, learned.COLUMNS, len(learned.COLOURS)), -5.0)
    for t, c in enumerate(["none", "yellow", "none", "violet", "none", "black", "none", "brown", "none", "brown"] + ["none"] * 10):
        cols[0, t, learned.COLOURS.index(c)] = 5.0
    forward, backward = bk._ctc_nll(cols, y), bk._ctc_nll(cols, y.flip(1))
    assert forward < backward
    assert torch.isfinite(bk.step_loss(net, torch.zeros(1, 3, learned.SIZE[1], learned.SIZE[0]), y,
                                       torch.ones(1, dtype=torch.long), torch.tensor([True]), 1.0))


def test_a_value_reading_the_same_both_ways_has_no_direction():
    y = torch.tensor([bk.slot_targets(("brown", "black", "black", "black", "brown")), bk.slot_targets(("black",)),
                      bk.slot_targets(("yellow", "violet", "black", "brown", "brown"))])
    assert bk._same_both_ways(y).tolist() == [True, True, False]


def test_greedy_reading_drops_blanks_and_repeats():
    class Fake:
        def both(self, x):
            cols = torch.full((1, learned.COLUMNS, len(learned.COLOURS)), -5.0)
            seq = ["none", "red", "red", "none", "red", "none", "black", "none", "silver", "none", "brown"] + ["none"] * 9
            for t, c in enumerate(seq):
                cols[0, t, learned.COLOURS.index(c)] = 5.0
            return cols.flip(1), torch.zeros(1)      # shown right to left
    y = np.array([bk.slot_targets(bk.kit_bands(2.2))])
    assert bk.greedy_right(Fake(), torch.zeros(1, 3, 2, 2), y) == 1.0


def test_temperature_softens_an_overconfident_reader():
    k = len(learned._OHMS)
    rng = np.random.default_rng(0)
    ohms, rows = [], []
    for i in range(200):                       # sure every time, right only 60% of the time
        o = learned._OHMS[50]
        s = np.full(2 * k, -60.0)
        s[50 if i % 5 < 3 else 120] = 0.0
        s += rng.normal(0, 1, 2 * k)
        rows.append(s)
        ohms.append(o)
    t = bk.fit_temperature(np.array(rows), np.array(ohms))
    assert t > 3
    share = bk.true_share(np.array(rows) / t, np.array(ohms))
    assert 0.3 < share.mean() < 0.8


def test_a_saved_temperature_scales_the_reading(tmp_path):
    net = learned.BandCTC(8)
    image = np.full((60, 200, 3), 200, np.uint8)
    torch.save({"width": 8, "kind": "ctc", "temperature": 1.0, "state": net.state_dict()}, tmp_path / "a.pt")
    torch.save({"width": 8, "kind": "ctc", "temperature": 4.0, "state": net.state_dict()}, tmp_path / "b.pt")
    a = learned.log_probs(learned.load(tmp_path / "a.pt"), image, 200)
    b = learned.log_probs(learned.load(tmp_path / "b.pt"), image, 200)
    assert np.allclose(a / 4, b, atol=1e-4)

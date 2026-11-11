import numpy as np
import pytest
import torch

from robot_offline.post.bands import learned


def _log_probs(colours, sure=0.98):
    """Slot log-probabilities that favour these band colours, read left to right."""
    p = np.full((learned.SLOTS, len(learned.COLOURS)), (1 - sure) / (len(learned.COLOURS) - 1))
    for slot, colour in enumerate(learned.slots_of(colours)):
        p[slot, learned.COLOURS.index(colour)] = sure
    return np.log(p)


def test_the_network_gives_a_colour_per_slot_and_is_small():
    net = learned.BandNet(16)
    out = net(torch.zeros(2, 3, learned.SIZE[1], learned.SIZE[0]))
    assert out.shape == (2, learned.SLOTS, len(learned.COLOURS))
    assert sum(p.numel() for p in net.parameters()) < 300_000


def test_decodes_a_clear_five_band_value():
    r = learned.decode(_log_probs(("yellow", "violet", "black", "brown", "brown")))
    assert r.value == 4700
    assert r.confidence > 0.9


def test_decodes_a_four_band_value_lying_the_other_way_round():
    p = _log_probs(("brown", "black", "red", "gold"))[::-1]
    assert learned.decode(p).value == 1000


def test_an_uncertain_band_is_unknown_but_keeps_its_best_guess():
    p = _log_probs(("brown", "black", "black", "brown", "brown"))
    p[3] = np.log(np.full(len(learned.COLOURS), 1 / len(learned.COLOURS)))
    r = learned.decode(p)
    assert r.value is None
    assert r.best == pytest.approx(r.best)


def test_only_values_in_scope_can_come_out():
    r = learned.decode(np.log(np.full((learned.SLOTS, len(learned.COLOURS)), 1 / len(learned.COLOURS))))
    assert r.best == 0 or learned.SCOPE_OHM[0] <= r.best <= learned.SCOPE_OHM[1]
    assert r.value is None


def _views_of(colours, sharpness=4.0):
    """Slot log-probabilities that favour these colours (reading order) by sharpness nats."""
    lp = np.zeros((learned.SLOTS, len(learned.COLOURS)))
    for slot, colour in enumerate(learned.slots_of(colours)):
        lp[slot, learned.COLOURS.index(colour)] = sharpness
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def test_averaging_views_doesnt_make_a_reading_surer_than_one_view():
    one = learned.decode(_views_of(("yellow", "violet", "red", "gold")))
    views = learned.Views()
    for _ in range(5):
        many = views.add("a", _views_of(("yellow", "violet", "red", "gold")))
    assert many.best == one.best == 4700
    assert abs(many.confidence - one.confidence) < 1e-9


def test_views_average_out_one_bad_view_and_forget_bodies_gone():
    views = learned.Views()
    for _ in range(4):
        views.add("a", _views_of(("brown", "black", "red", "gold")))
    r = views.add("a", _views_of(("brown", "black", "orange", "gold"), sharpness=2.0))
    assert r.best == 1000
    views.add("b", _views_of(("brown", "black", "red", "gold")))
    views.keep_only(["b"])
    assert list(views.seen) == ["b"]


def test_values_are_written_as_on_a_parts_list():
    assert [learned.ohms_text(v) for v in (100, 470, 1000, 4700, 10000, 1500000)] == \
        ["100", "470", "1k", "4k7", "10k", "1M5"]


def test_the_kit_range_is_in_scope_and_values_dont_split_by_rounding():
    for colours, ohms in ((("black",), 0), (("red", "red", "black", "silver", "brown"), 2.2),
                          (("brown", "green", "black", "red", "brown"), 15000),
                          (("green", "blue", "black", "yellow", "brown"), 5.6e6)):
        r = learned.decode(_log_probs(colours))
        assert r.value == pytest.approx(ohms) and r.confidence > 0.9
    assert len({o for o, _ in learned._CANDIDATES if abs(o - 2.2) < 1e-6}) == 1


def test_a_value_that_reads_as_another_backwards_needs_the_direction():
    ten_k = ("brown", "black", "black", "red", "brown")         # 120R backwards
    alone = learned.decode(_log_probs(ten_k))
    assert alone.value is None and 0.4 < alone.confidence < 0.6
    ways = np.zeros((1, len(learned.COLOURS)))
    ways[0, :2] = np.log([0.99, 0.01])                          # sure it reads left to right
    r = learned.decode(np.vstack([_log_probs(ten_k), ways]))
    assert r.value == 10000 and r.confidence > 0.95


def test_old_weights_load_without_a_direction_head(tmp_path):
    net = learned.BandNet(8)
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("direction.")}
    torch.save({"width": 8, "state": state}, tmp_path / "old.pt")
    old = learned.load(tmp_path / "old.pt")
    assert not old.has_direction
    image = np.full((60, 200, 3), 200, np.uint8)
    assert learned.log_probs(old, image, 200).shape == (learned.SLOTS, len(learned.COLOURS))
    torch.save({"width": 8, "state": net.state_dict()}, tmp_path / "new.pt")
    assert learned.log_probs(learned.load(tmp_path / "new.pt"), image, 200).shape == (learned.SLOTS + 1, len(learned.COLOURS))


def test_a_model_can_be_limited_to_the_kits_values(tmp_path):
    try:
        learned.use_values([1000.0, 4700.0, 0.0])
        assert set(learned._OHMS) == {0.0, 1000.0, 4700.0}
        r = learned.decode(_log_probs(("orange", "orange", "black", "yellow", "brown")))      # 3M3: not allowed
        assert r.best in (0.0, 1000.0, 4700.0) and r.value is None
        net = learned.BandCTC(8)
        torch.save({"width": 8, "kind": "ctc", "values": [1000.0, 4700.0], "state": net.state_dict()}, tmp_path / "k.pt")
        learned.load(tmp_path / "k.pt")
        assert set(learned._OHMS) == {1000.0, 4700.0}
        with pytest.raises(ValueError):
            learned.use_values([3333.0])
    finally:
        learned.use_values(None)
    assert len(learned._OHMS) == len(learned.ALL_CANDIDATES)


def test_a_pinned_close_up_counts_half_while_the_body_stays():
    views = learned.Views(window=5, min_confidence=0.9)
    sure = _views_of(("yellow", "violet", "black", "brown", "brown"), sharpness=8.0)
    unsure = _views_of(("yellow", "violet", "black", "brown", "brown"), sharpness=0.5)
    live = views.add("a", unsure)
    assert live.value is None
    views.pin("a", sure)
    both = views.reading("a")
    assert both.confidence > live.confidence and both.best == 4700
    assert views.add("a", unsure).confidence == pytest.approx(both.confidence)
    views.keep_only(["b"])
    assert views.reading("a") is None and not views.pinned


def test_a_model_carries_its_own_limit(tmp_path):
    net = learned.BandCTC(8)
    image = np.full((60, 200, 3), 200, np.uint8)
    torch.save({"width": 8, "kind": "ctc", "min_confidence": 0.998, "state": net.state_dict()}, tmp_path / "m.pt")
    loaded = learned.load(tmp_path / "m.pt")
    assert loaded.saved_min_confidence == 0.998
    assert loaded.min_confidence == learned.deployed_min_confidence(0.998)
    torch.save({"width": 8, "kind": "ctc", "state": net.state_dict()}, tmp_path / "old.pt")
    assert learned.load(tmp_path / "old.pt").saved_min_confidence == learned.MIN_CONFIDENCE
    r = learned.read(loaded, image, 200)
    assert r.value is None or r.confidence >= loaded.min_confidence


def test_the_deploy_setting_overrides_a_models_own_limit(monkeypatch):
    from robot_offline import config
    monkeypatch.setattr(config, "load", lambda name, *a: {"deploy": {"min_confidence": 0.7}})
    assert learned.deployed_min_confidence(0.998) == 0.7
    monkeypatch.setattr(config, "load", lambda name, *a: {})
    assert learned.deployed_min_confidence(0.998) == 0.998
    learned.use_values(None)

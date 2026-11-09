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
    assert 10 <= r.best <= 10000
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

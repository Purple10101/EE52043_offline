import numpy as np
import pytest

from robot_offline.inf import live_detect as ld
from robot_offline.post import resistor_tag as rt
from robot_offline.post.bands import learned


def test_tag_views_average_log_odds_per_body():
    tv = ld.TagViews(window=3)
    assert ld.TagViews().add("a", 3.0)[1] == "resistor"
    tv.add("a", 3.0)
    p, tag = tv.add("a", -3.0)                  # the mean logit is 0: P 0.5, unsure
    assert p == pytest.approx(0.5) and tag == "unsure"
    tv.add("a", -3.0)
    tv.add("a", -3.0)                           # the window has dropped the 3.0
    assert tv.add("a", -3.0)[1] == "not"
    tv.add("b", 1.0)
    tv.keep_only(["b"])
    assert list(tv.seen) == ["b"]


def test_without_a_tagger_every_body_counts_as_a_resistor():
    assert ld.tag_bodies(None, ld.TagViews(), None, [np.zeros((4, 2))] * 2) == [(None, None), (None, None)]


def test_tag_bodies_one_per_polygon_and_per_id(monkeypatch):
    monkeypatch.setattr(rt, "logits", lambda net, crops: np.array([4.0, -4.0, 0.0])[:len(crops)])
    view = np.zeros((300, 300, 3), np.uint8)
    polys = [np.array([[100, 100], [170, 100], [170, 125], [100, 125]], float)] * 3
    tags = ld.tag_bodies(object(), ld.TagViews(), view, polys)
    assert [t for _, t in tags] == ["resistor", "not", "unsure"]
    tv = ld.TagViews()
    ld.tag_bodies(object(), tv, view, polys, ids=[7, 8, 9])
    assert sorted(tv.seen) == [7, 8, 9]


def test_read_values_skips_not_and_never_gives_unsure_a_value(monkeypatch):
    read = []
    monkeypatch.setattr(ld.rectify, "white_balance", lambda plain, s: (plain, np.ones(3)))
    monkeypatch.setattr(ld.crops, "cut", lambda frame, cal, poly, gains: type("C", (), {"image": poly})())
    monkeypatch.setattr(learned, "log_probs", lambda net, image, white: read.append(image) or "lp")
    monkeypatch.setattr(learned, "decode", lambda lp, *limit: learned.Reading(4700.0, 4700.0, 0.99, ("yellow", "violet")))
    cal = type("Cal", (), {"settings": None})()
    plain = np.full((1200, 1200, 3), 200, np.uint8)
    polys = ["a", "b", "c", "d"]
    tags = [(0.95, "resistor"), (0.05, "not"), (0.5, "unsure"), (None, None)]
    notes = ld.read_values(None, learned.Views(), None, cal, plain, polys, tags=tags)
    assert read == ["a", "c", "d"]              # the "not" body isn't read
    assert notes[0] == "4k7 0.99" and notes[1] == "not a resistor"
    assert notes[2] == "unsure 0.50: ~4k7, measure" and notes[3] == "4k7 0.99"

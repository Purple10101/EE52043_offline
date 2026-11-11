import json

import pytest

from robot_offline.tools import collect_bands as cb


@pytest.mark.parametrize("text, ohms", [("0R22", 0.22), ("4R7", 4.7), ("10R", 10), ("220R", 220), ("4k7", 4700),
                                        ("47K", 47000), ("1.5k", 1500), ("1M", 1e6), ("2M2", 2.2e6), ("4700", 4700),
                                        ("1k37", 1370), (100, 100)])
def test_parse_ohms_reads_strip_markings(text, ohms):
    assert cb.parse_ohms(text) == pytest.approx(ohms)


@pytest.mark.parametrize("text", ["", "k", "4k7k", "abc", "1.5k7"])
def test_parse_ohms_refuses_what_isnt_a_value(text):
    with pytest.raises(ValueError):
        cb.parse_ohms(text)


def test_bands_four_and_five():
    assert cb.bands_of(4700, 4) == ("yellow", "violet", "red", "gold")
    assert cb.bands_of(4700, 5) == ("yellow", "violet", "black", "brown", "brown")
    assert cb.bands_of(0.22, 4) == ("red", "red", "silver", "gold")
    assert cb.bands_of(1e6, 4) == ("brown", "black", "green", "gold")
    assert cb.bands_of(1370, 4) is None and cb.bands_of(1370, 5)[:3] == ("brown", "orange", "violet")


def test_folders_are_exact_so_values_never_share_one():
    assert cb.folder_name(4700) == "4700R" and cb.folder_name(0.22) == "0_22R"
    assert cb.folder_name(1370) != cb.folder_name(1400)


def test_labels_list_every_crop_with_its_value(tmp_path):
    v = tmp_path / "4700R"
    (v / "cap01").mkdir(parents=True)
    (v / "value.json").write_text(json.dumps(dict(marking="4k7", ohms=4700.0)))
    bodies = [dict(body=1, tag="resistor", p_resistor=0.99, sharpness=7000,
                   closeup=dict(focus=320, distance_mm=236.0, found_again=True, sharpness=11000, white=214.0)),
              dict(body=2, tag="not", p_resistor=0.01, sharpness=5000)]
    (v / "cap01" / "capture.json").write_text(json.dumps(dict(camera=dict(focus=306), white=212.0, bodies=bodies)))
    assert cb.write_labels(tmp_path) == 3
    rows = [json.loads(x) for x in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert {r["value_ohm"] for r in rows} == {4700.0}
    assert [(r["body"], r["kind"]) for r in rows] == [(1, "first"), (1, "closeup"), (2, "first")]
    assert rows[1]["image"] == "4700R/cap01/closeups/body1.png" and rows[1]["focus"] == 320
    assert rows[2]["tag"] == "not"
    assert cb.done_captures(v) == [v / "cap01"]


def test_zero_ohm_links_and_diodes():
    assert cb.bands_of(0.0, 5) == ("black",) and cb.folder_name(0.0) == "0R"
    assert cb.part_ohms("diode") is None and cb.folder_name(None, "diode") == "diode"
    assert cb.bands_of(None, 5) is None


def test_the_kit_config_parses_and_names_every_value_apart():
    from robot_offline import config
    markings = config.load("resistor_kit")["values"]
    values = [cb.part_ohms(m) for m in markings]
    assert len({cb.folder_name(v, m) for v, m in zip(values, markings)}) == len(markings)


def test_a_new_placement_never_reuses_a_kept_number(tmp_path):
    v = tmp_path / "7500R"
    for n in (2, 3, 4, 5):
        (v / f"cap{n:02d}").mkdir(parents=True)
        (v / f"cap{n:02d}" / "capture.json").write_text("{}")
    (v / "cap06").mkdir()                       # stopped part way: no capture.json
    assert len(cb.done_captures(v)) == 4
    assert cb.next_capture(v) == v / "cap06" and not (v / "cap06").exists()
    assert (v / "cap05" / "capture.json").exists()


def test_excluded_placements_leave_the_labels(tmp_path):
    v = tmp_path / "7500R"
    (v / "cap01").mkdir(parents=True)
    (v / "value.json").write_text(json.dumps(dict(marking="7k5", ohms=7500.0)))
    (v / "cap01" / "capture.json").write_text(json.dumps(dict(camera=dict(focus=306), white=200.0,
                                                               bodies=[dict(body=1, tag="resistor", p_resistor=1.0, sharpness=1)])))
    assert cb.write_labels(tmp_path) == 1
    dest = cb.exclude(v / "cap01", tmp_path, "wrong value")
    assert cb.write_labels(tmp_path) == 0 and cb.done_captures(v) == []
    assert (dest / "why.txt").read_text().startswith("wrong value") and (dest / "capture.json").exists()


def test_only_picks_kit_values_in_the_order_given():
    markings = ["10k", "56k", "100k", "1M", "diode"]
    values = [cb.part_ohms(m) for m in markings]
    m, v = cb.select(markings, values, "56K, 1M,100000")
    assert m == ["56k", "1M", "100k"] and v == [56000.0, 1e6, 100000.0]
    assert cb.select(markings, values, "diode")[0] == ["diode"]
    with pytest.raises(ValueError):
        cb.select(markings, values, "47k")

from robot_offline.post.bands import learned
from robot_offline.tools import read_plate


def test_a_value_is_trusted_only_on_a_body_tagged_resistor():
    sure = learned.Reading(4700.0, 4700.0, 0.99, ())
    guess = learned.Reading(None, 4700.0, 0.6, ())
    assert read_plate.trusted(sure, "resistor") and read_plate.trusted(sure, None)
    assert not read_plate.trusted(sure, "unsure") and not read_plate.trusted(sure, "not")
    assert not read_plate.trusted(guess, "resistor")
    assert read_plate.label(sure, "resistor") == "4k7 0.99"
    assert read_plate.label(sure, "unsure") == "[unsure] 4k7 0.99"

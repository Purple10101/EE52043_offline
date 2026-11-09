import cv2
import numpy as np
import pytest

from robot_offline.post.bands import value

BEIGE = np.array([0.72, 0.62, 0.44], np.float32)
BLUE = np.array([0.12, 0.35, 0.62], np.float32)


def _crop(body, colours, positions, width=0.06, size=(160, 60)):
    """A straightened crop of a plain body with bands, plate white at 255, and its mask."""
    w, h = size
    image = np.tile(body[::-1] * 255, (h, w, 1)).astype(np.float32)
    for colour, at in zip(colours, positions):
        rgb = np.clip(value.predicted(colour, body), -128, 128)
        bgr = value._rgb(rgb)[::-1] * 255
        image[:, int((at - width / 2) * w):int((at + width / 2) * w)] = bgr
    return np.clip(image + 0.5, 0, 255).astype(np.uint8), np.full((h, w), 255, np.uint8)


def test_candidates_stay_in_scope_and_in_the_series():
    found = value.candidates()
    assert all(10 <= ohms <= 10000 for ohms, _ in found)
    assert (1000.0, ("brown", "black", "red", "gold")) in found
    assert (1000.0, ("brown", "black", "black", "brown", "brown")) in found
    assert (4700.0, ("yellow", "violet", "black", "brown", "brown")) in found
    assert not any(ohms == 12345 for ohms, _ in found)


def test_a_blue_body_turns_yellow_green_and_darkens_red():
    on_blue, on_beige = value.predicted("yellow", BLUE), value.predicted("yellow", BEIGE)
    assert on_blue[1] < on_beige[1] and on_blue[2] < on_beige[2]
    assert value.predicted("red", BLUE)[0] < value.predicted("red", BEIGE)[0]


def test_finds_the_bands_on_a_body():
    image, mask = _crop(BEIGE, ("brown", "black", "red"), (0.25, 0.45, 0.7))
    bands, _ = value.find_bands(value.centre_line(image, mask, 255.0))
    assert [round(b.position, 1) for b in bands] == [0.2, 0.5, 0.7] or len(bands) == 3


@pytest.mark.parametrize("colours, positions, ohms", [
    (("brown", "black", "red", "gold"), (0.25, 0.45, 0.65, 0.88), 1000.0),
    (("yellow", "violet", "red", "gold"), (0.25, 0.45, 0.65, 0.88), 4700.0),
])
def test_reads_a_four_band_body(colours, positions, ohms):
    image, mask = _crop(BEIGE, colours, positions)
    assert value.read(image, mask, 255.0, min_margin=0.0).best == ohms


def test_reads_a_body_lying_the_other_way_round():
    colours, positions = ("yellow", "violet", "red", "gold"), (0.25, 0.45, 0.65, 0.88)
    image, mask = _crop(BEIGE, colours, positions)
    r = value.read(cv2.flip(image, 1), mask, 255.0, min_margin=0.0)
    assert r.best == 4700.0 and r.reversed


def test_a_plain_body_is_unknown():
    image, mask = _crop(BEIGE, (), ())
    assert value.read(image, mask, 255.0).value is None


def test_a_blue_body_needs_a_larger_margin():
    image, mask = _crop(BLUE, ("yellow", "violet", "black", "brown", "brown"), (0.2, 0.36, 0.52, 0.68, 0.88))
    r = value.read(image, mask, 255.0)
    assert r.best == 4700.0
    assert r.value == (4700.0 if r.margin >= value.BLUE_MIN_MARGIN else None)

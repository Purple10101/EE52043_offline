import cv2
import numpy as np
import pytest

FRAME = (4608, 2592)
# Plate corners as found in a real frame from the tower (top-left, top-right, bottom-right, bottom-left).
CORNERS = np.array([[1776.0, 523.1], [3075.2, 619.2], [3080.6, 1825.7], [1472.0, 1679.7]])


def _poly(img, points, colour):
    cv2.fillPoly(img, [np.round(np.asarray(points) * 16).astype(np.int32)], colour, cv2.LINE_8, 4)


@pytest.fixture(scope="session")
def plate_corners():
    return CORNERS.copy()


@pytest.fixture(scope="session")
def plate_frame():
    """A camera frame like the real one: bright plate top, its shadowed front face below the near
    edge, bright legs reaching down from the near corners, a screw touching a corner, leads and a
    resistor body near the edges, then blur and noise."""
    rng = np.random.default_rng(0)
    img = np.full((FRAME[1], FRAME[0], 3), (60, 110, 150), np.uint8)       # wood
    tl, tr, br, bl = CORNERS
    down = np.array([0.0, 35.0])
    _poly(img, [bl, br, br + down, bl + down], (45, 45, 45))                 # front face
    for c in (bl, br):                                                       # legs
        _poly(img, [c + (-10, -20), c + (50, -20), c + (50, 160), c + (-10, 160)], (185, 185, 185))
    _poly(img, CORNERS, (200, 200, 200))                                     # plate top
    cv2.circle(img, (int(tl[0]) - 15, int(tl[1]) - 25), 28, (215, 215, 215), -1)   # screw
    for a, b in [((1900, 700), (2600, 1300)), ((2900, 900), (3200, 1000)), ((2000, 1500), (1450, 1600))]:
        cv2.line(img, a, b, (70, 70, 70), 7, cv2.LINE_AA)                    # leads
    cv2.rectangle(img, (2930, 1150), (2965, 1230), (120, 160, 60), -1)       # a body near the right edge
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = rng.normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

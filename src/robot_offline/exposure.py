"""Exposure that follows the light, metered on the plate.

The plate's top is white and always in view, so it serves as a grey card: the camera's exposure is
stepped until the top's brightness sits in a band around a target, whatever the room's light does,
and only once the exposure is at its longest is gain added (gain adds noise). The camera's own auto
exposure meters the whole frame, so a dark bench or a bright sleeve would move the plate's level;
this only looks at the plate.

Exposure is DirectShow's log2 seconds, so a step of 1 doubles the light. A step up is only taken
if the level it predicts lies inside the band, so it can't oscillate; a plate that is too bright
(or clipping) always steps down. The level's change per step is learnt from the steps taken, since
the camera's tone curve makes a doubling of light less than a doubling of level. After each change
SETTLE_FRAMES frames are skipped, as the camera takes a few frames to apply it.

Settings in configs/camera.toml [exposure_control].
"""

import cv2
import numpy as np

from robot_offline import rectify

METER_SHARE = 0.7           # the plate's top is metered over its middle, this share of it across
CLIP_LEVEL = 250            # grey at or above this counts as clipped
FRAME_PERCENTILE = 99.5     # before the plate is found, the frame's brightest part (the plate) is metered
GAMMA = 2.2                 # first guess at the tone curve: a step changes the level by 2^(step / GAMMA)


def plate_level(frame, cal):
    """(level, clipped share) of the plate's top in a raw frame: the median grey of the brighter half
    of its middle (resistors and marks left out), and the share of that half that is clipped."""
    outline = rectify.plate_outline_px(cal)
    centre = outline.mean(0)
    inner = centre + METER_SHARE * (outline - centre)
    h, w = frame.shape[:2]
    (x0, y0), (x1, y1) = np.clip(np.floor(inner.min(0)), 0, None).astype(int), np.ceil(inner.max(0)).astype(int)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return frame_level(frame)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(inner - (x0, y0)).astype(np.int32)], 1)
    grey = cv2.cvtColor(frame[y0:y1:2, x0:x1:2], cv2.COLOR_BGR2GRAY)[mask[::2, ::2].astype(bool)]
    bright = grey[grey >= np.median(grey)]
    return float(np.median(bright)), float(np.mean(bright >= CLIP_LEVEL))


def frame_level(frame):
    """(level, clipped share) of the frame's brightest part, for before the plate is found."""
    grey = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)
    top = grey[grey >= np.percentile(grey, FRAME_PERCENTILE)]
    return float(np.median(top)), float(np.mean(top >= CLIP_LEVEL))


class ExposureControl:
    """Steps a camera's exposure and gain so the plate's level stays in [low, high].

    settings: configs/camera.toml [exposure_control]. exposure, gain: what the camera is set to now.
    cap: the cv2.VideoCapture to set them on, or None (for tests: read .exposure and .gain).
    """

    def __init__(self, settings, exposure, gain, cap=None):
        self.s = settings
        self.exposure, self.gain = float(exposure), float(gain)
        self.cap = cap
        self.ratio = 2 ** (settings["exposure_step"] / GAMMA)   # level change per exposure step, learnt
        self.wait = 0
        self.level, self.clipped = None, 0.0
        self.pending = None         # (level before, exposure direction) of the last step, to learn from

    def update(self, level, clipped):
        """One metering of the plate. Changes the exposure or gain if needed; returns True if it did."""
        if self.wait > 0:
            self.wait -= 1
            return False
        s = self.s
        self.level, self.clipped = level, clipped
        if self.pending is not None:
            before, direction = self.pending
            self.pending = None
            unclipped = 10 < min(before, level) and max(before, level) < CLIP_LEVEL
            if direction and unclipped:
                seen = (level / before) ** direction
                self.ratio = float(np.clip(0.5 * self.ratio + 0.5 * seen, 1.05, 4.0))

        step, exposure, gain = s["exposure_step"], self.exposure, self.gain
        direction = 0
        if level > s["high"] or clipped > s["max_clipped"]:
            if gain > 0:
                gain = max(0.0, gain - s["gain_step"])
            elif exposure - step >= s["min_exposure"]:
                exposure, direction = exposure - step, -1
        elif level < s["low"]:
            if exposure + step <= s["max_exposure"]:
                if level * self.ratio <= s["high"]:
                    exposure, direction = exposure + step, 1
            elif gain + s["gain_step"] <= s["max_gain"]:
                gain += s["gain_step"]
        if (exposure, gain) == (self.exposure, self.gain):
            return False
        self.exposure, self.gain = exposure, gain
        self.pending = (level, direction)
        self.wait = s["settle_frames"]
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
        return True

    def meter(self, frame, cal=None):
        """Meters a frame (on the plate if cal is given, else on its brightest part) and updates."""
        return self.update(*(plate_level(frame, cal) if cal is not None else frame_level(frame)))

    def status(self):
        level = "?" if self.level is None else f"{self.level:.0f}"
        clipped = f", {self.clipped:.0%} clipped" if self.clipped >= 0.005 else ""
        return (f"exposure {self.exposure:g} (1/{2 ** -self.exposure:.0f} s), gain {self.gain:g}: "
                f"plate at {level}{clipped} (aim {self.s['low']}-{self.s['high']})")

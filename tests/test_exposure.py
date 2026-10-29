import numpy as np

from robot_offline import config, exposure, rectify

SETTINGS = config.load("camera")["exposure_control"]


class Camera:
    """A camera on the plate: its level from the light, exposure (log2 s) and gain, through a
    gamma 2.2 tone curve, clipped at 255. Changes take effect settle_frames frames later."""

    def __init__(self, light, gain_per_step=1.0):
        self.light, self.gain_per_step = light, gain_per_step

    def level(self, control):
        linear = self.light * 2 ** control.exposure * (1 + control.gain / 32 * self.gain_per_step)
        level = 255 * min(linear, 1.0) ** (1 / 2.2)
        return level, float(linear >= (250 / 255) ** 2.2)


def run(camera, control, frames):
    levels = []
    for _ in range(frames):
        control.update(*camera.level(control))
        levels.append(camera.level(control)[0])
    return levels


def settled(levels, control, frames=40):
    """The level ends in the band and stays put."""
    tail = levels[-frames:]
    return SETTINGS["low"] <= tail[-1] <= SETTINGS["high"] and max(tail) - min(tail) < 1e-9


def test_a_dim_room_is_brought_up_to_the_band():
    control = exposure.ExposureControl(SETTINGS, -5, 0)
    levels = run(Camera(light=5.0), control, 200)        # the plate at ~110 at exposure -5, like today
    assert levels[0] < SETTINGS["low"] and settled(levels, control)


def test_a_bright_room_is_brought_down_without_clipping():
    control = exposure.ExposureControl(SETTINGS, -5, 0)
    camera = Camera(light=400.0)                         # clipped at -5
    levels = run(camera, control, 200)
    assert camera.level(control)[1] == 0 and settled(levels, control)


def test_it_follows_the_light_when_it_changes():
    control = exposure.ExposureControl(SETTINGS, -5, 0)
    camera = Camera(light=40.0)
    run(camera, control, 100)
    for light in (3.0, 150.0, 20.0):                     # lights off, a lamp on, back
        camera.light = light
        assert settled(run(camera, control, 200), control)


def test_gain_is_only_used_once_the_exposure_is_at_its_longest():
    s = dict(SETTINGS, max_gain=64)
    control = exposure.ExposureControl(s, -5, 0)
    camera = Camera(light=0.15)                          # very dim: even max_exposure is not enough
    levels = run(camera, control, 400)
    assert control.exposure == s["max_exposure"] and control.gain > 0
    assert settled(levels, control) or control.gain == s["max_gain"]
    camera.light = 40.0                                  # much brighter: the gain comes off first
    run(camera, control, 400)
    assert control.gain == 0


def test_without_gain_too_dim_just_waits_at_the_longest_exposure():
    s = dict(SETTINGS, max_gain=0)
    control = exposure.ExposureControl(s, -5, 0)
    levels = run(Camera(light=0.05), control, 200)
    assert control.exposure == s["max_exposure"] and control.gain == 0
    assert max(levels[-40:]) - min(levels[-40:]) < 1e-9


def test_the_plate_is_metered_on_its_top(plate_frame):
    cal, _ = rectify.calibrate(plate_frame)
    level, clipped = exposure.plate_level(plate_frame, cal)
    assert abs(level - 200) < 3 and clipped == 0         # the drawn plate's top is 200; leads left out
    dark = (plate_frame * 0.5).astype(np.uint8)
    assert abs(exposure.plate_level(dark, cal)[0] - 100) < 3

"""The overhead camera: opened with configs/camera.toml, and the on-screen helpers every live window
uses (capture, live, live_plate, live_detect)."""

import cv2
import numpy as np

from robot_offline import config
from robot_offline.pre import rectify

MAX_INDEX = 6               # DirectShow indices tried when looking for the camera
RETRY_FRAMES = 20           # while the plate isn't found for the live view, look again this often
RAW_SCALE = 0.25

# camera.toml image controls and the OpenCV properties they set.
_CONTROLS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gamma": cv2.CAP_PROP_GAMMA,
}


def open_camera(camera=None):
    """The overhead camera, opened with the settings in configs/camera.toml and warmed up.

    DirectShow indices move when devices are plugged in or out, so the configured index is tried
    first and then the others, taking the first device that accepts the full-resolution mode.
    """
    camera = camera or config.load("camera")
    c = camera["camera"]
    size = (c["width"], c["height"])
    for index in [c["index"]] + [i for i in range(MAX_INDEX) if i != c["index"]]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == size:
            break
        cap.release()
    else:
        raise RuntimeError(f"no camera accepts {size[0]} × {size[1]}; is the {c['device_name']} plugged in?")

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c["autofocus"] else 0)
    if not c["autofocus"]:
        cap.set(cv2.CAP_PROP_FOCUS, c["focus"])
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if c["auto_exposure"] else 0.25)
    if not c["auto_exposure"]:
        cap.set(cv2.CAP_PROP_EXPOSURE, c["exposure"])
        cap.set(cv2.CAP_PROP_GAIN, c["gain"])
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if c["auto_white_balance"] else 0)
    for key, prop in _CONTROLS.items():
        cap.set(prop, c[key])

    for _ in range(camera["capture"]["warmup_frames"]):
        cap.read()
    return cap


# --- On-screen helpers ----------------------------------------------------------------------------

def label(img, lines):
    """Text lines in the top-left corner, white with a dark outline so they read on the plate too."""
    for i, text in enumerate(lines):
        y = 26 + 26 * i
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def find_plate(frame, settings):
    """(Calibration or None, status text) for a frame."""
    try:
        cal, sides = rectify.calibrate(frame, settings)
    except rectify.PlateNotFound as e:
        return None, f"plate not found: {e}"
    worst = min(share for share, _ in sides)
    return cal, f"plate found (near side by the {cal.fit['near_side_by']}; {worst:.0%} of the worst edge found)"


def view_of(frame, cal, raw, grid):
    """What the window shows: the top-down view, or the raw frame with the plate outlined."""
    if raw or cal is None:
        view = cv2.resize(frame, None, fx=RAW_SCALE, fy=RAW_SCALE, interpolation=cv2.INTER_AREA)
        if cal is not None:
            corners = rectify.plate_outline_px(cal) * RAW_SCALE
            cv2.polylines(view, [corners.round().astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        return view
    rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
    return rectify.draw_grid(rect, cal.settings) if grid else rect


def show(view, shown_size, window):
    """Shows a view, fitting the window to it (at most 900 px tall) when its size changes."""
    size = view.shape[1], view.shape[0]
    if size != shown_size:
        scale = min(1.0, 900 / size[1])
        cv2.resizeWindow(window, int(size[0] * scale), int(size[1] * scale))
    cv2.imshow(window, view)
    return size

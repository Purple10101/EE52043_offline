"""Live plate check: the camera's view with the fitted plate drawn on it, next to the rectified plate.

Opens the camera with configs/camera.toml and finds the plate as `rectify calibrate` does. From then
on the camera's pose is fitted again to the newest frame, over and over in the background (about a
second each), keeping the focal length from the first fit, so the picture never waits for it.

Left, the camera's view of the plate with its model drawn where the fit puts it: the top's outline
(green), the bottom edge of its front face (cyan) and the crosshair (magenta). They should sit on
the real edges. Right, the newest frame rectified and white-balanced, with the 10 mm grid, the
plate's outline in red and a magenta ring where the crosshair should be.

The text says which feature fixed the near side (the crosshair, or the face's bottom when the
crosshair is covered), how far the unused ones are from the fit, and how far the plate has moved in
the frame since the first fit: with the tower steady that should stay well under a pixel.

The exposure follows the light (exposure.py): it is metered on the plate's white top and stepped
to keep it at about 200, whatever the room does. E turns that off, back to the fixed exposure in
configs/camera.toml, to compare.

    python -m robot_offline.live_plate

Keys:  q / Esc  quit      g  grid on or off      z  whole camera frame or zoomed on the plate
       l  lock the fit (stop refitting, to see the plate drift against it)
       c  start again: find the plate from scratch, focal length included
       e  exposure following the light, or fixed
       s  save the raw frame, the rectified image and the calibration to --save-dir
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, rectify
from robot_offline.capture import label, open_camera, show
from robot_offline.exposure import ExposureControl

WINDOW = "robot_offline live plate"
HEIGHT = 900                # px, both views are shown this tall
ZOOM_MARGIN = 0.15          # zoomed camera view: the plate plus this share of its size around it
COLOURS = {"top": (0, 220, 0), "face bottom": (255, 255, 0), "crosshair": (255, 0, 255)}


class Fitter(threading.Thread):
    """Fits the plate to the newest frame, again and again, in the background (also used by
    live_detect). interval: seconds between fits once the plate is found; 0 for as often as it goes.
    Until it is found, and after `restart`, it tries every frame."""

    def __init__(self, settings, interval=0.0):
        super().__init__(daemon=True)
        self.settings = settings
        self.interval = interval
        self.frame = None           # the newest frame, set by the display loop
        self.cal = None             # the latest good fit
        self.first = None           # the first good fit, for how far the plate has moved since
        self.error = None           # why the latest try failed, or None
        self.seconds = 0.0          # how long the latest fit took
        self.fitted_at = 0.0
        self.locked = False
        self.restart = False
        self.stop = False

    def run(self):
        done, tried_at = None, 0.0
        while not self.stop:
            frame = self.frame
            waiting = self.cal is not None and not self.restart and time.perf_counter() - tried_at < self.interval
            if frame is None or frame is done or self.locked or waiting:
                time.sleep(0.02)
                continue
            done = frame
            t0 = tried_at = time.perf_counter()
            try:
                if self.cal is None or self.restart:
                    self.restart = False
                    cal, _ = rectify.calibrate(frame, self.settings)
                    self.first = cal
                else:
                    cal, _ = rectify.recalibrate(frame, self.cal)
                rectify.rectify(frame, cal)             # builds its remap tables here, not in the display loop
                self.cal, self.error = cal, None
            except rectify.PlateNotFound as e:
                self.error = str(e)
            self.seconds, self.fitted_at = time.perf_counter() - t0, time.perf_counter()


def camera_view(frame, cal, zoom):
    """The camera's view with the plate's fitted model drawn on it, HEIGHT tall."""
    if cal is None:
        x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
    else:
        model = rectify.plate_model_px(cal)
        pts = np.vstack([model["top"], model["face bottom"]])
        if zoom:
            size = pts.max(0) - pts.min(0)
            (x0, y0), (x1, y1) = (np.maximum(pts.min(0) - ZOOM_MARGIN * size, 0).astype(int),
                                  np.minimum(pts.max(0) + ZOOM_MARGIN * size, frame.shape[1::-1]).astype(int))
        else:
            x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
    scale = HEIGHT / (y1 - y0)
    view = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if cal is not None:
        to_view = lambda p: np.round((np.asarray(p) - (x0, y0)) * scale * 16).astype(np.int32)
        cv2.polylines(view, [to_view(model["top"])], True, COLOURS["top"], 1, cv2.LINE_AA, 4)
        cv2.polylines(view, [to_view(model["face bottom"])], False, COLOURS["face bottom"], 1, cv2.LINE_AA, 4)
        cv2.polylines(view, [to_view(arm) for arm in model["crosshair"]], False, COLOURS["crosshair"], 1,
                      cv2.LINE_AA, 4)
    return view


def rectified_view(frame, cal, grid):
    """The rectified, white-balanced frame with the grid and the crosshair's place, HEIGHT tall."""
    rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
    if grid:
        rect = rectify.draw_grid(rect, cal.settings)
    (u, v), = rectify.mm_to_rect([cal.settings.cross_mm], cal.settings)
    ring = int(round(cal.settings.cross_arm_mm * cal.settings.px_per_mm * 0.8))
    cv2.circle(rect, (int(round(u)), int(round(v))), ring, COLOURS["crosshair"], 1, cv2.LINE_AA)
    scale = HEIGHT / rect.shape[0]
    return cv2.resize(rect, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def status_lines(fitter):
    """What the fit says, for the top of the window."""
    cal = fitter.cal
    if cal is None:
        return [f"looking for the plate: {fitter.error}" if fitter.error else "looking for the plate..."]
    fit = cal.fit
    age = time.perf_counter() - fitter.fitted_at
    lines = []
    if fitter.locked:
        lines.append("LOCKED: the fit is held (l to refit)")
    elif fitter.error:
        lines.append(f"NOT FOUND in the latest frame, showing the last fit: {fitter.error}")
    else:
        lines.append(f"fitted {age:.1f} s ago, in {fitter.seconds:.1f} s")
    checks = ", ".join(f"{name} {offset} mm" for name, offset in fit["checks_mm"].items()) or "none"
    lines.append(f"near side by the {fit['near_side_by']}; checks: {checks}")
    edges = "  ".join(f"{name} {share:.0%} {rms:.2f}px" for name, (share, rms) in fit["edges"].items())
    lines.append(f"found, rms: {edges}")
    moved = np.abs(cal.corners_px - fitter.first.corners_px).max() if fitter.first is not None else 0.0
    lines.append(f"camera {fit['camera_mm']} mm, {fit['tilt_deg']} deg, f {cal.camera_matrix[0, 0]:.0f} px;"
                 f" plate moved {moved:.2f} px since the first fit")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.live_plate",
                                     description="Live view of the fitted plate next to the rectified plate.")
    parser.add_argument("--save-dir", default=str(config.DATA_DIR / "live"),
                        help="where `s` saves frames (default data/live)")
    args = parser.parse_args(argv)

    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    camera = config.load("camera")
    cap = open_camera(camera)
    fixed = camera["camera"]["exposure"], camera["camera"]["gain"]
    control = ExposureControl(camera["exposure_control"], *fixed, cap)
    print(__doc__.split("Keys:", 1)[1].rstrip())
    fitter = Fitter(rectify.Settings.from_config())
    fitter.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    grid, zoom, follow, shown, times = True, True, True, None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fitter.frame = frame
        cal = fitter.cal
        if follow:
            control.meter(frame, cal)
        times = (times + [time.perf_counter()])[-10:]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

        left = camera_view(frame, cal, zoom)
        right = rectified_view(frame, cal, grid) if cal is not None else np.zeros((HEIGHT, HEIGHT, 3), np.uint8)
        exposure = control.status() if follow else f"exposure fixed at {fixed[0]}, gain {fixed[1]} (e to follow the light)"
        label(left, status_lines(fitter) + [exposure, f"{fps:.1f} fps   q quit  g grid  z zoom  l lock  "
                                                      "c start again  e exposure  s save"])
        shown = show(np.hstack([left, right]), shown, WINDOW)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("g"):
            grid = not grid
        elif key == ord("z"):
            zoom = not zoom
        elif key == ord("l"):
            fitter.locked = not fitter.locked
        elif key == ord("c"):
            fitter.restart = True
            fitter.locked = False
        elif key == ord("e"):
            follow = not follow
            if not follow:
                cap.set(cv2.CAP_PROP_EXPOSURE, fixed[0])
                cap.set(cv2.CAP_PROP_GAIN, fixed[1])
                control = ExposureControl(camera["exposure_control"], *fixed, cap)
        elif key == ord("s"):
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = out / datetime.now().strftime("plate_%Y%m%d_%H%M%S")
            cv2.imwrite(f"{stem}.png", frame)
            if cal is not None:
                cv2.imwrite(f"{stem}_rectified.png", rectify.white_balance(rectify.rectify(frame, cal), cal.settings)[0])
                cal.save(f"{stem}_rectify.json")
            print(f"Saved {stem}.png" + (", _rectified.png and _rectify.json" if cal is not None else ""))

    fitter.stop = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

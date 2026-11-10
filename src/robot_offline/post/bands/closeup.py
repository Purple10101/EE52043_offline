"""Close-ups: after segmentation, one capture per resistor at the focus that suits where it lies, to read
its bands from.

One focus can't suit the whole plate: the lens is about 230 mm from the front of the plate and 275 mm
from the back, and a sweep on 2026-11-09 found the front sharpest at focus 322 and the back at 290 to
298 (experiments/2026-11-09_front_back). The config focus is a compromise for segmentation, which
doesn't mind. The bands do. So once segmentation has found the bodies, each gets a capture of its own:

1. Its distance from the lens: its centre on the plate (rectify.rect_to_mm) to where the plate fit put
   the lens (cal.fit camera_mm; else the tower's design position).
2. Its focus, from that distance: FocusModel, focus = a + b / distance (a lens's focus setting goes with
   the inverse of the distance), fitted to the points in configs/camera.toml [closeup], which `calibrate`
   measures and adds to.
3. The capture: the focus set, SETTLE frames read (a change shows on the 5th or 6th frame read, measured
   2026-11-09), AVERAGE frames averaged. Bodies go in order of focus, so the lens moves least.
4. The body found again in its close-up: a focus change scales the picture slightly, so the segmentation
   model is run again on the tiles round it and the outline nearest the first within MATCH_PX taken;
   without a model, or if none is near, the first outline is used.
5. Its band crop, cut from the close-up as read_plate cuts it, with that capture's white balance.

Afterwards the focus goes back to camera.toml's. The exposure isn't changed: set it for reading first
(plate_search.find_now(..., level=learned.READING_LEVEL)), as tools.read_plate does.

    python -m robot_offline.post.bands.closeup calibrate [--save]   # sweep focus over the bodies on the plate

calibrate needs bodies both near the front and near the back of the plate, else it can't fit a slope.
"""

import argparse
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post.bands import crops
from robot_offline.pre import rectify

SETTLE = 7                  # frames read after a focus change before using any (it shows on the 5th or 6th)
AVERAGE = 4                 # frames averaged per close-up
MATCH_PX = 30.0             # rectified px (3 mm): the same body in the close-up as in the first pass
SWEEP = tuple(range(278, 347, 4))   # calibrate: the focus settings tried
MIN_SPREAD_MM = 15.0        # calibrate: bodies' distances must span at least this to fit a slope
FOCUS_LIMITS = (200, 450)   # whatever the fit says, the setting stays in here


def sharpness(image):
    """Mean squared gradient of a crop's grey: higher is sharper (only comparable for one body)."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2))


def lens_mm(cal):
    """Where the plate fit put the lens (scene mm), else the tower's design position."""
    if cal.fit and "camera_mm" in cal.fit:
        return np.asarray(cal.fit["camera_mm"], float)
    return np.asarray(cal.settings.camera_mm, float)


def distance_mm(cal, polygon):
    """From the lens to the middle of a body outline (rectified px), on the plate's top."""
    x, y = rectify.rect_to_mm(np.asarray(polygon, float).mean(0)[None], cal.settings)[0]
    return float(np.linalg.norm(np.array([x, y, cal.settings.top_z_mm]) - lens_mm(cal)))


@dataclass
class FocusModel:
    """focus = a + b / distance, least squares over points [(distance mm, best focus), ...]."""
    points: list

    def __post_init__(self):
        p = np.asarray(self.points, float)
        if len(p) == 0:
            raise ValueError("no focus points: run `python -m robot_offline.post.bands.closeup calibrate --save`")
        if np.ptp(p[:, 0]) < 5.0:           # one distance: no slope to fit, so the same focus everywhere
            self.a, self.b = float(p[:, 1].mean()), 0.0
        else:
            design = np.column_stack([np.ones(len(p)), 1.0 / p[:, 0]])
            (self.a, self.b), *_ = np.linalg.lstsq(design, p[:, 1], rcond=None)

    @classmethod
    def from_config(cls, camera=None):
        return cls((camera or config.load("camera"))["closeup"]["points"])

    def focus(self, distance):
        return int(round(np.clip(self.a + self.b / distance, *FOCUS_LIMITS)))


@dataclass
class Closeup:
    body: int                   # index into the polygons given
    distance_mm: float
    focus: int
    polygon: np.ndarray         # the body's outline in the close-up (rectified px)
    found_again: bool           # False: segmentation didn't find it again there, so the first outline was used
    crop: crops.Crop
    white: float                # the plate's level in the crop's white balance, for learned.read
    sharpness: float
    frame: np.ndarray = None    # the averaged close-up, if keep_frames
    seconds: dict = None        # how long each step took: settle, average, balance, find_again, cut


def mean_frame(cap, n):
    """The mean of the next n frames (up to 257), rounded, exact halves to even. Summed in uint16 and scaled
    by OpenCV: about 0.55 s for 4 full frames, where a float mean took 1.9 s."""
    total, got = None, 0
    while got < n:
        ok, f = cap.read()
        if not ok:
            continue
        total = f.astype(np.uint16) if total is None else cv2.add(total, f, dtype=cv2.CV_16U)
        got += 1
    return cv2.convertScaleAbs(total, alpha=1.0 / n)


def white_of(cal, frame):
    """(gains, white) for cutting crops from a frame: crops.cut's gains and learned.read's white."""
    plain = rectify.rectify(frame, cal)
    _, gains = rectify.white_balance(plain, cal.settings)
    balanced = np.clip(plain.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return plain, gains, float(np.percentile(cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)[150:1050, 150:1050], 70))


def _found_again(seg, grey, frame, cal, plain, polygon):
    """The body's outline in a close-up, re-segmented on the tiles round it, or None."""
    ds = config.load("dataset")
    a, tc = ds["work_area"], ds["tiles"]
    view, _ = rectify.model_input(frame, cal, plain)
    centre = np.asarray(polygon, float).mean(0)
    near = [t for t in tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (view.shape[1], view.shape[0]),
                                    tc["size"], tc["overlap"])
            if t.core[0] <= centre[0] < t.core[2] and t.core[1] <= centre[1] < t.core[3]]
    polys, _ = tiling.predict_frame(seg, view, near, conf=0.25, grey=grey)
    if not polys:
        return None
    d = [np.linalg.norm(np.asarray(p).mean(0) - centre) for p in polys]
    i = int(np.argmin(d))
    return np.asarray(polys[i], float) if d[i] <= MATCH_PX else None


def take(cap, cal, polygons, seg=None, grey=False, model=None, settle=SETTLE, average=AVERAGE, keep_frames=False,
         camera=None):
    """One close-up per body outline (rectified px, from segmentation on this plate), in the bodies' order.

    cap: the live camera, exposure already set for reading. seg, grey: the segmentation model and whether
    it takes grey, to find each body again in its close-up (None: use the outlines given). model: the
    FocusModel, default from camera.toml. The focus is set back to camera.toml's afterwards."""
    camera = camera or config.load("camera")
    model = model or FocusModel.from_config(camera)
    plan = sorted(((model.focus(distance_mm(cal, p)), i) for i, p in enumerate(polygons)))
    out = {}
    try:
        for focus, i in plan:
            ticks = [time.perf_counter()]
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            for _ in range(settle):
                cap.read()
            ticks.append(time.perf_counter())
            frame = mean_frame(cap, average)
            ticks.append(time.perf_counter())
            plain, gains, white = white_of(cal, frame)
            ticks.append(time.perf_counter())
            again = _found_again(seg, grey, frame, cal, plain, polygons[i]) if seg is not None else None
            ticks.append(time.perf_counter())
            polygon = again if again is not None else np.asarray(polygons[i], float)
            crop = crops.cut(frame, cal, polygon, gains)
            ticks.append(time.perf_counter())
            steps = ("settle", "average", "balance", "find_again", "cut")
            out[i] = Closeup(i, distance_mm(cal, polygons[i]), focus, polygon, again is not None, crop, white,
                             sharpness(crop.image), frame if keep_frames else None,
                             {s: round(b - a, 3) for s, a, b in zip(steps, ticks, ticks[1:])})
    finally:
        cap.set(cv2.CAP_PROP_FOCUS, camera["camera"]["focus"])
    return [out[i] for i in range(len(polygons))]


# --- calibrate ----------------------------------------------------------------------------------------

def best_focus(focuses, values):
    """The focus where a sharpness curve peaks, between samples (a parabola through the top three); None if
    the peak is at either end of the sweep (it may lie beyond)."""
    k = int(np.argmax(values))
    if k in (0, len(values) - 1):
        return None
    y0, y1, y2 = values[k - 1], values[k], values[k + 1]
    denom = y0 - 2 * y1 + y2
    step = focuses[k + 1] - focuses[k]
    return float(focuses[k] + (0.5 * (y0 - y2) / denom * step if denom < 0 else 0.0))


def save_points(points, path=config.CONFIG_DIR / "camera.toml"):
    """Writes [closeup] points into camera.toml, rounded, leaving everything else as it is."""
    text = path.read_text(encoding="utf-8")
    line = "points = [" + ", ".join(f"[{d:.1f}, {f:.1f}]" for d, f in points) + "]"
    new, n = re.subn(r"(?m)^points = \[.*\]$", line, text, count=1)
    if n != 1:
        raise ValueError(f"no [closeup] points line in {path}")
    path.write_text(new, encoding="utf-8")


def calibrate(cap, cal, polygons, focuses=SWEEP, settle=SETTLE, camera=None):
    """Each body's best focus, by sweeping: [(distance mm, best focus or None), ...] in the bodies' order."""
    camera = camera or config.load("camera")
    curves = [[] for _ in polygons]
    try:
        for focus in focuses:
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            for _ in range(settle):
                cap.read()
            frame = mean_frame(cap, 2)
            _, gains, _ = white_of(cal, frame)
            for curve, p in zip(curves, polygons):
                curve.append(sharpness(crops.cut(frame, cal, p, gains).image))
    finally:
        cap.set(cv2.CAP_PROP_FOCUS, camera["camera"]["focus"])
    return [(distance_mm(cal, p), best_focus(list(focuses), c)) for p, c in zip(polygons, curves)], curves


def main(argv=None):
    from ultralytics import YOLO

    from robot_offline.post.bands import learned
    from robot_offline.pre import plate_search
    from robot_offline.pre.camera import open_camera

    parser = argparse.ArgumentParser(prog="python -m robot_offline.post.bands.closeup")
    sub = parser.add_subparsers(dest="command", required=True)
    cal_p = sub.add_parser("calibrate", help="sweep focus over the bodies on the plate and fit focus to distance")
    cal_p.add_argument("--weights", default=str(config.ROOT / "models" / "seg_grey_v2" / "best.pt"))
    cal_p.add_argument("--save", action="store_true", help="add the points to configs/camera.toml [closeup]")
    cal_p.add_argument("--fresh", action="store_true", help="with --save: replace the points rather than add to them")
    args = parser.parse_args(argv)

    camera = config.load("camera")
    ds = config.load("dataset")
    a, tc = ds["work_area"], ds["tiles"]
    cap = open_camera(camera)
    try:
        cal = plate_search.find_now(cap, camera=camera, level=learned.READING_LEVEL)
        frame = mean_frame(cap, 4)
        view, _ = rectify.model_input(frame, cal)
        polys, _ = tiling.predict_frame(YOLO(args.weights), view,
                                        tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]),
                                                     (view.shape[1], view.shape[0]), tc["size"], tc["overlap"]),
                                        conf=0.5, grey=tiling.wants_grey(args.weights))
        print(f"{len(polys)} bodies; sweeping focus {SWEEP[0]} to {SWEEP[-1]} ...", flush=True)
        found, _ = calibrate(cap, cal, polys, camera=camera)
    finally:
        cap.release()
    old = FocusModel.from_config(camera)
    for (d, f), p in zip(found, polys):
        c = np.asarray(p).mean(0)
        print(f"  body at ({c[0]:.0f}, {c[1]:.0f}) px, {d:.1f} mm from the lens: best focus "
              + (f"{f:.1f} (model said {old.focus(d)})" if f is not None else "at the end of the sweep: left out"))
    points = [(d, f) for d, f in found if f is not None]
    if not points:
        raise SystemExit("no body had a peak inside the sweep")
    if np.ptp([d for d, _ in points]) < MIN_SPREAD_MM:
        print(f"note: these bodies' distances span under {MIN_SPREAD_MM:.0f} mm, so on their own they fix no slope")
    merged = points if args.fresh else [tuple(p) for p in camera["closeup"]["points"]] + points
    new = FocusModel(merged)
    print(f"fit over {len(merged)} points: focus = {new.a:.1f} + {new.b:.0f} / distance;"
          f" residuals {[round(f - new.focus(d), 1) for d, f in merged]}")
    if args.save:
        save_points(merged)
        print("saved to configs/camera.toml [closeup]")


if __name__ == "__main__":
    main()

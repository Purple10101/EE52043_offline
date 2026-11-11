"""The camera's focus, measured on the plate's centre cross before each capture.

The IMX708's focus is a voice-coil motor run open loop: the focus number sets the coil's current, not
where the lens is, so the same number lands differently as the camera warms (from cold the best focus
climbs from about 299 to about 337, time constant about 3 minutes), each time it is opened (±2–3) and by
which way the lens arrives (about 3 settings of hysteresis). About 15 settings off, a body is half as
sharp. Measured in experiments/2026-11-11_focus_drift (band-reading PC). So no fixed number (camera.toml's
focus) is trusted: before each capture, with the plate in view and the arm parked, the focus is checked
on the plate's centre cross, a fixed target at a known distance.

    tracker = FocusTracker(cap, cal)     # once the plate is found
    tracker.check()                      # before each capture: sets the camera to the best focus
    tracker.focus, tracker.offset, tracker.status()

The first check is a full search (FULL, about 19 s); each later one a narrow re-check round the last
best (NARROW, about 3.5 s), widened to a full search if its peak lands at an edge. Every search approaches
from below (BELOW under its first step), so the hysteresis is always the same way round.

offset: the best focus minus the close-up rule's prediction (camera.toml [closeup], closeup.FocusModel) at
the reference's distance. One offset measured on the cross corrects the rule for bodies at every distance
to within about ±5, so closeup.take adds it to each body's focus.

If the cross is covered (no clear peak: the curve's top is less than MIN_PEAK_RATIO above its bottom), a
body near the centre is the reference, when polygons are given; failing that the last result is kept and
the status says so. Each result is saved to FOCUS_FILE (with its time), which closeup.take and the tools
read while it is fresh (current()).
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.pre import rectify

FOCUS_FILE = config.DATA_DIR / "plate" / "focus.json"
FULL = tuple(range(270, 381, 5))     # the first search
NARROW_HALF, NARROW_STEP = 6, 3      # a re-check: the last best ± NARROW_HALF in NARROW_STEP
SETTLE = 6                           # frames dropped after a focus change (it shows on the 5th or 6th)
AVERAGE = 1                          # frames measured per step
BELOW = 10                           # each search starts this far under its first step, from below
CROSS_HALF_MM = 6.0                  # the window round the cross whose sharpness is measured
BODY_PAD_PX = 12                     # round a body used as the reference instead
MIN_PEAK_RATIO = 1.15                # a curve whose top is under this × its bottom has no clear peak
FRESH_S = 120.0                      # a saved result older than this isn't used by current()


@dataclass
class FocusResult:
    focus: int                  # the focus set for the capture
    offset: float               # minus the close-up rule at the reference's distance
    reference: str              # "cross", "body", or "kept" (no clear peak: the last result kept)
    kind: str                   # "full" or "narrow"
    seconds: float
    time: float                 # time.time() when measured
    peak: float = None          # the interpolated best, before rounding
    clear: bool = True


def window_sharpness(grey, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    p = grey[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].astype(np.float32)
    if p.size == 0:
        return 0.0
    return float((cv2.Sobel(p, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(p, cv2.CV_32F, 0, 1) ** 2).mean())


def measure(frame, cal, box):
    """Sharpness in box (rectified px) of a raw frame."""
    return window_sharpness(cv2.cvtColor(rectify.rectify(frame, cal), cv2.COLOR_BGR2GRAY), box)


def cross_target(cal):
    """(box in rectified px, distance from the lens mm) of the plate's centre cross."""
    st = cal.settings
    c = rectify.mm_to_rect(np.array([st.cross_mm], float), st)[0]
    r = CROSS_HALF_MM * st.px_per_mm
    return (c[0] - r, c[1] - r, c[0] + r, c[1] + r), _distance(cal, st.cross_mm)


def body_target(cal, polygons):
    """(box, distance) of the body nearest the cross, or None."""
    if not polygons:
        return None
    st = cal.settings
    c = rectify.mm_to_rect(np.array([st.cross_mm], float), st)[0]
    p = min((np.asarray(q, float) for q in polygons), key=lambda q: np.linalg.norm(q.mean(0) - c))
    lo, hi = p.min(0) - BODY_PAD_PX, p.max(0) + BODY_PAD_PX
    xy = rectify.rect_to_mm(p.mean(0)[None], st)[0]
    return (*lo, *hi), _distance(cal, xy)


def _distance(cal, xy_mm):
    from robot_offline.post.bands.closeup import lens_mm
    return float(np.linalg.norm(np.array([xy_mm[0], xy_mm[1], cal.settings.top_z_mm]) - lens_mm(cal)))


def peak_of(focuses, values):
    """(interpolated best, at an edge of the sweep, clear peak)."""
    values = np.asarray(values, float)
    k = int(np.argmax(values))
    clear = values.max() >= MIN_PEAK_RATIO * max(values.min(), 1e-9)
    if k in (0, len(values) - 1):
        return float(focuses[k]), True, clear
    y0, y1, y2 = values[k - 1:k + 2]
    denom = y0 - 2 * y1 + y2
    step = focuses[k + 1] - focuses[k]
    return float(focuses[k] + (0.5 * (y0 - y2) / denom * step if denom < 0 else 0.0)), False, clear


def sweep(cap, cal, box, focuses, settle=SETTLE, average=AVERAGE):
    """Sharpness at each focus, approached from below."""
    from robot_offline.post.bands.closeup import mean_frame
    cap.set(cv2.CAP_PROP_FOCUS, focuses[0] - BELOW)
    out = []
    for f in focuses:
        cap.set(cv2.CAP_PROP_FOCUS, f)
        for _ in range(settle):
            cap.read()
        out.append(measure(mean_frame(cap, average), cal, box))
    return out


def set_focus(cap, focus):
    """Sets focus, arriving from below like the searches."""
    cap.set(cv2.CAP_PROP_FOCUS, focus - BELOW)
    cap.set(cv2.CAP_PROP_FOCUS, focus)


class FocusTracker:
    """Checks focus before each capture; see the module doc."""

    def __init__(self, cap, cal, camera=None, model=None, path=FOCUS_FILE, start=None):
        """model: the close-up rule (closeup.FocusModel, from camera.toml if None). start: a FocusResult to
        re-check from (the first check is otherwise a full search)."""
        from robot_offline.post.bands.closeup import FocusModel
        self.cap, self.cal, self.path = cap, cal, Path(path)
        self.camera = camera or config.load("camera")
        self.model = model or FocusModel.from_config(self.camera)
        self.last = start
        self.note = ""

    @property
    def focus(self):
        return self.last.focus if self.last else int(self.camera["camera"]["focus"])

    @property
    def offset(self):
        return self.last.offset if self.last else 0.0

    def check(self, polygons=None, full=False):
        """Measures focus and sets the camera to it. polygons (rectified px): bodies to use as the reference
        if the cross is covered. Returns the FocusResult (also saved to the focus file)."""
        t0 = time.perf_counter()
        target = cross_target(self.cal)
        kind = "full" if full or self.last is None or self.last.reference == "kept" and self.last.peak is None \
            else "narrow"
        result = self._search(target, kind, "cross")
        if result is None and polygons:
            body = body_target(self.cal, polygons)
            if body is not None:
                result = self._search(body, kind, "body")
        if result is None:
            self.note = "no clear peak (cross covered?): last focus kept"
            kept = self.last or FocusResult(self.focus, 0.0, "kept", kind, 0.0, time.time(), None, False)
            result = FocusResult(kept.focus, kept.offset, "kept", kind, 0.0, time.time(), kept.peak, False)
        else:
            self.note = ""
        result.seconds = round(time.perf_counter() - t0, 2)
        set_focus(self.cap, result.focus)
        self.last = result
        self._save(result)
        return result

    def _search(self, target, kind, reference):
        box, distance = target
        if kind == "narrow":
            centre = int(round(self.last.peak if self.last.peak is not None else self.last.focus))
            focuses = list(range(centre - NARROW_HALF, centre + NARROW_HALF + 1, NARROW_STEP))
            values = sweep(self.cap, self.cal, box, focuses)
            best, edge, clear = peak_of(focuses, values)
            if clear and edge:                       # moved further than the window: search it all
                kind = "full"
        if kind == "full":
            focuses = list(FULL)
            values = sweep(self.cap, self.cal, box, focuses)
            best, edge, clear = peak_of(focuses, values)
        if not clear:
            return None
        offset = best - self.model.focus(distance)
        if reference == "body":                      # the capture's focus: the rule at the cross, shifted
            focus = self.model.focus(cross_target(self.cal)[1]) + offset
        else:
            focus = best
        return FocusResult(int(round(focus)), round(float(offset), 2), reference, kind, 0.0, time.time(),
                           round(float(best), 2), True)

    def status(self):
        if self.last is None:
            return f"focus {self.focus} (camera.toml; not checked yet)"
        age = time.time() - self.last.time
        text = f"focus {self.last.focus} ({self.last.offset:+.0f} on the rule, by the {self.last.reference}, " \
               f"{self.last.kind} check {age:.0f} s ago)"
        return text + (f": {self.note}" if self.note else "")

    def _save(self, result):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(result), indent=1))
        tmp.replace(self.path)


def current(max_age_s=FRESH_S, path=FOCUS_FILE):
    """The saved FocusResult if it is younger than max_age_s, else None."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if time.time() - d.get("time", 0) > max_age_s:
        return None
    return FocusResult(**d)

"""Finding the plate: the one place every tool, here and in the robot repo (EE52043), gets it from.

Use one of:

- Tracker: a live camera. Call update(frame) on every frame; .cal is the plate, or None until it is found.
  It sets the exposure too: dim while it looks for the plate, then the plate at about 200 for the model.
- find_now(cap): the same, once, for a script; returns the Calibration.
- find(frame): one frame already taken (saved frames, the simulator's twin). No exposure control.

Why the exposure: the camera's tone curve squeezes the brights, so a bright plate barely stands out from
a pale bench (with the plate at grey 183 the bench beyond its far edge read 135; with it at 78, 36). Over
every raw frame there was on 2026-11-09, the fit found the plate in 37 of 38 with the plate below about
170, and failed in most of those at 180 to 210 on pale benches. So the plate is looked for at
[plate_search] search_level, metered where it is expected (the rig is rigid: the plate stays within some
40 px of there in the frame), not on the frame's brightest part, which on a bench by a window is the
bench. Once found, the exposure goes back to [exposure_control] low..high and the plate is refitted there
every refit_seconds; where a pale bench makes that fail, the last fit is kept.

The fit (rectify.find_plate) needs a start within about 40 px. Starts, the first the fit passes giving
the plate:

1. Where the plate was last found: remembered by every find, in LAST (one file, shared by both repos),
   else the rig's own place, configs/camera.toml [plate_search] rig_corners_px, so a fresh clone works
   anywhere the rig is set up. On a pale bench this is the start that matters: the fit can do without
   the one edge the bench hides (it uses the crosshair instead), but the outlines below can't.
2. Four-sided outlines from the frame's long straight lines, of about the plate's size, nearest the
   start first, at most MAX_OUTLINES of them.
3. rectify's bright blob, for frames with no usable lines.

If it still isn't found after sweep_after tries, the Tracker steps through every exposure, one per try,
in case the plate isn't where it is expected.

Saved calibrations (`s` in the live tools, the robot's camera_calibrate) are records, and a tool can be
given one to use as it is; the search's start is only ever LAST or the rig's place.
"""

import itertools
import threading
import time

import cv2
import numpy as np

from robot_offline import config
from robot_offline.pre import rectify
from robot_offline.pre.exposure import ExposureControl

LAST = config.DATA_DIR / "plate" / "last_rectify.json"     # where the plate was last found, by any tool
SCALE = 0.25                # lines are looked for at this share of the frame's size
CANNY = (20, 60)
HOUGH_VOTES = 60
MIN_LINE_PX = 120           # at SCALE: about a third of the plate's side
MAX_GAP_PX = 25             # at SCALE: a lead lying over an edge breaks it by less than this
SAME_ANGLE_RAD = 0.05       # lines this alike, and SAME_OFFSET_PX apart, are one line
SAME_OFFSET_PX = 25
LEAN_RAD = np.radians(35)   # a side lies within this of the frame's axes
MAX_LINES = 14              # of each kind, longest first
SIDE_PX = (800, 2200)       # the plate's sides in a raw frame are about 1200 to 1600
MAX_SIDE_RATIO = 1.6
MAX_OUTLINES = 12           # outlines tried, nearest the start first: a failed fit takes up to a second


def _lines(frame):
    """Long straight lines in a frame as (angle, offset), raw px, longest first and without repeats."""
    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    found = cv2.HoughLinesP(cv2.Canny(gray, *CANNY), 1, np.pi / 360, threshold=HOUGH_VOTES,
                            minLineLength=MIN_LINE_PX, maxLineGap=MAX_GAP_PX)
    if found is None:
        return []
    keep = []
    segments = found.reshape(-1, 4) / SCALE
    for x0, y0, x1, y1 in sorted(segments.tolist(), key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        angle = np.arctan2(y1 - y0, x1 - x0) % np.pi
        offset = np.array([-np.sin(angle), np.cos(angle)]) @ (x0, y0)
        if not any(min(abs(angle - a), np.pi - abs(angle - a)) < SAME_ANGLE_RAD and abs(offset - d) < SAME_OFFSET_PX
                   for a, d in keep):
            keep.append((angle, offset))
    return keep


def _crossing(p, q):
    normals = [[-np.sin(p[0]), np.cos(p[0])], [-np.sin(q[0]), np.cos(q[0])]]
    return np.linalg.solve(np.array(normals), np.array([p[1], q[1]]))


def outlines(frame):
    """Four-sided outlines (4 x 2 raw px, image order) the plate could be, largest first."""
    lines = _lines(frame)
    across = [n for n in lines if min(n[0], np.pi - n[0]) < LEAN_RAD][:MAX_LINES]
    down = [n for n in lines if abs(n[0] - np.pi / 2) < LEAN_RAD][:MAX_LINES]
    quads = []
    for (top, bottom), (left, right) in itertools.product(itertools.combinations(across, 2),
                                                          itertools.combinations(down, 2)):
        try:
            corners = [_crossing(top, left), _crossing(top, right), _crossing(bottom, right), _crossing(bottom, left)]
        except np.linalg.LinAlgError:
            continue
        quad = rectify._image_order(np.array(corners))
        sides = np.linalg.norm(quad - np.roll(quad, -1, 0), axis=1)
        if SIDE_PX[0] <= sides.min() and sides.max() <= SIDE_PX[1] and sides.max() / sides.min() <= MAX_SIDE_RATIO:
            quads.append(quad)
    return sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.float32)))


def _search_settings(camera=None):
    return (camera or config.load("camera"))["plate_search"]


def rig_corners(camera=None):
    """Where the plate sits in the frame on the rig (4 × 2 raw px, image order), from camera.toml."""
    return np.array(_search_settings(camera)["rig_corners_px"], float)


def start_corners(camera=None):
    """Where to start looking: where the plate was last found (LAST), else the rig's place."""
    try:
        return rectify.Calibration.load(LAST).corners_px
    except (OSError, ValueError, KeyError):
        return rig_corners(camera)


def remember(cal, path=None):
    """Keeps cal's plate as where the next search starts."""
    path = LAST if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    cal.save(tmp)
    tmp.replace(path)           # whole or not at all: another tool may be reading it


def find(frame, settings=None, last=None, keep=True):
    """The plate's Calibration from one frame, the focal length fitted too. last: corners to start from
    (raw px, image order); default start_corners(). keep: remember the plate found as the next start; False
    for frames not from the rig's camera now (the simulator's twin, saved frames). Raises rectify.PlateNotFound."""
    settings = settings or rectify.Settings.from_config()
    last = start_corners() if last is None else np.asarray(last, float)

    def fitted(start):
        fit = rectify.find_plate(frame, settings, start=start)
        cal = rectify._calibration(fit, (frame.shape[1], frame.shape[0]), settings)
        if keep:
            remember(cal)
        return cal

    try:
        return fitted(last)
    except rectify.PlateNotFound:
        pass
    quads = sorted(outlines(frame), key=lambda q: np.abs(q - last).max())[:MAX_OUTLINES]
    for quad in quads:
        try:
            return fitted(quad)
        except rectify.PlateNotFound:
            continue
    try:
        return fitted(None)
    except rectify.PlateNotFound as e:
        raise rectify.PlateNotFound(f"not where it was last found, nor at any of {len(quads)} outlines from the "
                                    f"frame's straight lines, nor from the brightest region ({e})") from None


def describe(cal):
    """One line on how the plate was found, for status text."""
    missing = cal.fit.get("missing_edge")
    worst = min(share for name, (share, _) in cal.fit["edges"].items() if name != missing)
    hidden = f", its {missing} edge hidden" if missing else ""
    return f"plate found (near side by the {cal.fit['near_side_by']}{hidden}; {worst:.0%} of the worst edge found)"


class _Fitter(threading.Thread):
    """The fits, in the background, so the caller's loop never waits for them."""

    def __init__(self, tracker):
        super().__init__(daemon=True)
        self.t = tracker
        self.frame = None
        self.stop = False

    def run(self):
        t, done, tried_at = self.t, None, 0.0
        while not self.stop:
            frame = self.frame
            waiting = not t.searching and time.perf_counter() - tried_at < t.refit_seconds
            if frame is None or frame is done or t.locked or waiting:
                time.sleep(0.02)
                continue
            done = frame
            t0 = tried_at = time.perf_counter()
            searching = t.searching
            try:
                if searching:
                    cal = find(frame, t.settings, t.last, keep=t.control is not None)
                else:
                    cal, _ = rectify.recalibrate(frame, t.cal)
                rectify.rectify(frame, cal)             # builds its remap tables here, not in the caller's loop
                if searching or t.first is None:
                    t.first = cal
                t.cal, t.error, t.last, t.failures = cal, None, cal.corners_px, 0
                t.searching = False
            except rectify.PlateNotFound as e:
                t.error = str(e)
                if searching:
                    t.failures += 1
            t.seconds, t.fitted_at = time.perf_counter() - t0, time.perf_counter()


class Tracker:
    """The plate on a live camera, found and kept found, with the exposure to go with it.

    cap: the cv2.VideoCapture whose exposure it sets, or None for frames it can't control (the twin, saved
    frames); those aren't remembered as where the plate is.
    cal: a calibration to start with (a saved one), refitted from then on unless locked. follow: False
    holds the exposure where it is once the plate is found (the live tools' E key). refit_seconds: how
    often the found plate is fitted again, default [plate_search]'s; 0 for as often as it goes. level:
    (low, high), the plate's grey to keep once found, default [exposure_control]'s band; settable.
    Call update(frame) for every frame read, and close() at the end.
    """

    def __init__(self, cap=None, settings=None, cal=None, camera=None, follow=True, locked=False, refit_seconds=None,
                 level=None):
        camera = camera or config.load("camera")
        search = _search_settings(camera)
        self.settings = settings or rectify.Settings.from_config()
        self.search_level = tuple(search["search_level"])
        self.refit_seconds = search["refit_seconds"] if refit_seconds is None else refit_seconds
        self.sweep_after = search["sweep_after"]
        c = camera["camera"]
        self.control = None if cap is None else ExposureControl(camera["exposure_control"], c["exposure"], c["gain"], cap)
        self.follow, self.locked, self.level = follow, locked, level
        self.cal, self.first = cal, cal
        self.last = start_corners(camera) if cal is None else cal.corners_px
        self.searching = cal is None
        self.error, self.seconds, self.fitted_at, self.failures = None, 0.0, 0.0, 0
        self._swept, self._settle = 0, 0
        self._fitter = _Fitter(self)
        self._fitter.start()

    def update(self, frame):
        """One frame from the camera: meters it and hands it to the fit. Returns the plate's Calibration,
        or None while it hasn't been found."""
        c = self.control
        if c is not None:
            if self.searching:
                c.aim(*self.search_level)
                if self.failures >= self.sweep_after:
                    # Not found where it was expected, dim: perhaps it isn't there. The exposures in turn.
                    if self.failures > self._swept:
                        self._sweep()
                    elif self._settle > 0:
                        self._settle -= 1
                    steady = self._settle == 0
                else:
                    changed = c.meter(frame, outline=self.last)
                    steady = not changed and c.wait == 0 and c.pending is None
                if not steady:
                    return self.cal     # the exposure is changing: don't spend a search on this frame
            elif self.follow:
                c.aim(*(self.level or (None, None)))
                c.meter(frame, self.cal)
        self._fitter.frame = frame
        return self.cal

    def _sweep(self):
        c, s = self.control, self.control.s
        steps = int(s["max_exposure"] - s["min_exposure"]) + 1
        c.exposure = float(s["max_exposure"] - (self.failures - self.sweep_after) % steps)
        c.gain, c.pending = 0.0, None
        c.cap.set(cv2.CAP_PROP_EXPOSURE, c.exposure)
        c.cap.set(cv2.CAP_PROP_GAIN, c.gain)
        self._swept, self._settle = self.failures, s["settle_frames"]

    def restart(self):
        """Looks for the plate afresh (the camera or plate moved), keeping the last fit until it is found."""
        self.failures, self._swept, self.locked = 0, 0, False
        self.searching = True

    def close(self):
        self._fitter.stop = True

    def status(self):
        """One line: found or looking, and why not."""
        if self.searching:
            why = f": {self.error}" if self.error else ""
            return f"looking for the plate{why}" + (" (showing the last fit)" if self.cal is not None else "")
        line = f"plate: near side by the {self.cal.fit['near_side_by']}" if self.cal.fit else "plate: given calibration"
        if self.locked:
            line += " (locked)"
        elif self.error:
            # Usual on a pale bench: at the working exposure its edges barely show. The fit stands; c looks again.
            line += " (held: not refound at this exposure)"
        return line


def find_now(cap, settings=None, camera=None, timeout_s=120, level=None):
    """The plate's Calibration from a live camera, found as the Tracker does. Leaves the exposure with the
    plate in level, (low, high), default [exposure_control]'s band. Raises rectify.PlateNotFound on timeout."""
    tracker = Tracker(cap, settings, camera=camera)
    end = time.perf_counter() + timeout_s
    try:
        while tracker.searching:
            if time.perf_counter() > end:
                raise rectify.PlateNotFound(f"not found in {timeout_s} s: {tracker.error}")
            ok, frame = cap.read()
            if ok:
                tracker.update(frame)
    finally:
        tracker.close()
    control = tracker.control
    control.aim(*(level or (None, None)))
    while time.perf_counter() < end:
        ok, frame = cap.read()
        if ok and not control.meter(frame, tracker.cal) and control.wait == 0 and control.pending is None:
            break               # in the band, or as near as the camera goes
    return tracker.cal

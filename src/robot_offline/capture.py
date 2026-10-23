"""Captures a session of full-resolution frames from the overhead camera, with a live view.

Opens the camera with the settings in configs/camera.toml and shows the plate from above. Space
captures. A session goes:

1. Empty plate: one frame of the clear plate, saved to data/backgrounds/<session>/0001.png. The
   plate is found in it and the session's calibration saved as data/captures/<session>/rectify.json.
2. The frames of the session's plan, saved as data/captures/<session>/0001.png and on. These raw
   frames are the record. A white-balanced top-down copy of each goes in rectified/, for looking
   at; it can always be regenerated with `rectify apply`.
   - `--plan scenes` (default): 15 frames of resistors arranged realistically, then 5 with
     non-resistor objects mixed in.
   - `--plan cutouts`: 12 frames of resistors spread out, then 6 of other objects spread out, for
     the cut-out library. The live view outlines every object it finds against the empty plate,
     each in its own colour: two objects inside one outline are touching, and an object outlined in
     red is too near the plate's edge. Move them until each has its own outline.
   `--count` and `--mixed` change the number of frames in the plan's first and second part.

Before saving a scene the plate is found again, near where it was last seen, and the frame keeps its
own plate corners: the camera sits loose in its pocket and a knock moves it, so one calibration per
session isn't enough. If an edge is hidden (a hand or the arm in view) the frame isn't saved and the
window says why; F saves it anyway, with the corners from the last frame. If the camera has moved
more than CAMERA_MOVED_PX since the last frame the frame is saved with a warning to check the mount.
session.toml records the camera settings, the mount, and each frame's kind and plate corners, and
is rewritten after every capture, so quitting early leaves a usable session. Running again with the
same --session continues it.

    python -m robot_offline.capture                         # 15 + 5 frames, new session for today
    python -m robot_offline.capture --plan cutouts          # 12 + 6 spread-out frames for cut-outs
    python -m robot_offline.capture --count 15 --mixed 5 --lighting "room lights" --notes "..."
    python -m robot_offline.capture --session 2026-10-23_a  # continue a session

Keys:  SPACE capture    F capture even if the plate isn't found    U undo the last scene frame
       E redo the empty plate (before any scene frame)    R raw / top-down    G grid    Q / Esc quit
"""

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, rectify

MAX_INDEX = 6               # DirectShow indices tried when looking for the camera
CAMERA_MOVED_PX = 3.0       # plate corner movement since the last frame that is reported as the camera moving
RETRY_FRAMES = 20           # while the plate isn't found for the live view, look again this often
WINDOW = "robot_offline capture"
RAW_SCALE = 0.25
EDGE_MARGIN_MM = 3.0        # cut-out objects closer than this to the plate's edge are outlined in red
MIN_OBJECT_PX = 40          # smaller differences from the empty plate are dust or noise, not objects
OUTLINE_BGR = ((255, 140, 0), (0, 200, 0), (200, 0, 200), (0, 170, 255), (255, 0, 120),
               (0, 140, 140), (140, 90, 0), (90, 0, 255))

# What a session captures after the empty plate: (kind, frames, prompt) for each part, in order.
PLANS = {
    "scenes": (("resistors", 15, "Arrange the resistors realistically"),
               ("mixed", 5, "Arrange resistors with some non-resistor objects mixed in")),
    # With only a few resistors, each frame should show them somewhere new: moved, turned, some
    # rolled over to show the other side of the bands, some leads re-bent.
    "cutouts": (("cutout_resistors", 12, "Move every resistor somewhere new, turned; roll some, re-bend some leads; keep them apart"),
                ("cutout_objects", 6, "Spread screws, shavings and other objects out, apart, away from the edges")),
}

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


# --- On-screen helpers (also used by `live`) ------------------------------------------------------

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
    return cal, f"plate found ({worst:.0%} of edge points on the worst edge)"


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


def outline_objects(rect, empty, settings, threshold):
    """Outlines what differs from the empty plate in a rectified BGR image, one colour per object.

    empty: the rectified empty plate. Objects within EDGE_MARGIN_MM of the plate's edge are outlined
    in thick red. Returns (image with outlines, objects found, objects too near the edge).
    """
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab0 = cv2.cvtColor(empty, cv2.COLOR_BGR2Lab).astype(np.float32)
    differs = (np.linalg.norm(lab - lab0, axis=2) > threshold).astype(np.uint8)
    c = rectify.mm_to_rect(settings.plate_corners_mm, settings)
    (x0, y0), (x1, y1) = np.floor(c.min(0)).astype(int) + 1, np.ceil(c.max(0)).astype(int)
    plate = np.zeros_like(differs)
    plate[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = 1            # the plate, short of its rim
    differs = cv2.morphologyEx(differs * plate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    differs = cv2.morphologyEx(differs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab_map, stats, _ = cv2.connectedComponentsWithStats(differs)
    margin = int(round(EDGE_MARGIN_MM * settings.px_per_mm))
    out, found, near = rect.copy(), 0, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_OBJECT_PX:
            continue
        edge = x < x0 + margin or y < y0 + margin or x + w > x1 - margin or y + h > y1 - margin
        contours, _ = cv2.findContours((lab_map == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = (0, 0, 255) if edge else OUTLINE_BGR[found % len(OUTLINE_BGR)]
        cv2.drawContours(out, contours, -1, colour, 4 if edge else 2, cv2.LINE_AA)
        found += 1
        near += edge
    return out, found, near


def show(view, shown_size, window=WINDOW):
    """Shows a view, fitting the window to it (at most 900 px tall) when its size changes."""
    size = view.shape[1], view.shape[0]
    if size != shown_size:
        scale = min(1.0, 900 / size[1])
        cv2.resizeWindow(window, int(size[0] * scale), int(size[1] * scale))
    cv2.imshow(window, view)
    return size


# --- Sessions -------------------------------------------------------------------------------------

def _toml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


class Session:
    """A capture session's folders, calibration and frame list, kept in step with session.toml."""

    def __init__(self, name, lighting, notes, camera, plan="scenes"):
        self.name = name
        self.dir = config.DATA_DIR / "captures" / name
        self.background = config.DATA_DIR / "backgrounds" / name / "0001.png"
        self.camera = camera
        self.lighting, self.notes, self.frames, self.cal, self.plan = lighting, notes, [], None, plan
        meta = self.dir / "session.toml"
        if meta.exists():
            with open(meta, "rb") as f:
                old = tomllib.load(f)
            self.frames = list(old.get("frames", []))
            self.lighting = lighting or old.get("lighting", "")
            self.notes = notes or old.get("notes", "")
            self.plan = old.get("plan", "scenes")
        if (self.dir / "rectify.json").exists():
            self.cal = rectify.Calibration.load(self.dir / "rectify.json")

    @property
    def last_cal(self):
        """Calibration for where the plate was last seen: the last frame's own, or the session's."""
        if self.cal is None:
            return None
        corners = next((f["plate_corners_px"] for f in reversed(self.frames) if "plate_corners_px" in f), None)
        return self.cal if corners is None else rectify.with_corners(self.cal, corners)

    def frame_cal(self, entry):
        """A saved frame's own calibration (session.toml [[frames]] entry)."""
        return rectify.with_corners(self.cal, entry["plate_corners_px"]) if "plate_corners_px" in entry else self.cal

    def count(self, kind):
        return sum(f["kind"] == kind for f in self.frames)

    def save_empty(self, frame, cal):
        self.background.parent.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.background), frame)
        cal.save(self.dir / "rectify.json")
        self.cal = cal
        self.write()

    def save_scene(self, frame, kind, cal, found):
        """Saves a scene frame with its own calibration. found: whether the plate was found in this
        frame (else cal is where it was last seen)."""
        (self.dir / "rectified").mkdir(parents=True, exist_ok=True)
        number = max([int(Path(f["file"]).stem) for f in self.frames], default=0) + 1
        name = f"{number:04d}.png"
        cv2.imwrite(str(self.dir / name), frame)
        rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
        cv2.imwrite(str(self.dir / "rectified" / name), rect)
        self.frames.append({"file": name, "kind": kind, "captured": datetime.now().isoformat(timespec="seconds"),
                            "plate_found": found,
                            "plate_shift_px": round(float(np.abs(cal.corners_px - self.cal.corners_px).max()), 2),
                            "plate_corners_px": np.round(cal.corners_px, 2).tolist()})
        self.write()
        return name

    def undo(self):
        if not self.frames:
            return None
        last = self.frames.pop()
        for path in (self.dir / last["file"], self.dir / "rectified" / last["file"]):
            path.unlink(missing_ok=True)
        self.write()
        return last["file"]

    def write(self):
        c, m = self.camera["camera"], self.camera["mount"]
        lines = [
            f"# Capture session, written by robot_offline.capture. Frames are raw {c['width']} x {c['height']} PNG.",
            f"session = {_toml(self.name)}",
            f"date = {_toml(self.name.split('_')[0])}",
            f"plan = {_toml(self.plan)}   # scenes: realistic arrangements; cutouts: spread out, for the cut-out library",
            f"backdrop = {_toml('pick plate (white)')}",
            f"lighting = {_toml(self.lighting)}",
            f"notes = {_toml(self.notes)}",
            f"calibration = {_toml('rectify.json')}",
            f"background = {_toml(f'backgrounds/{self.name}/0001.png')}   # relative to data/",
            "",
            "[camera]",
            *(f"{k} = {_toml(c[k])}" for k in ("device_name", "width", "height", "focus", "exposure", "gain",
                                               "auto_white_balance")),
            "",
            "[mount]",
            *(f"{k} = {_toml(m[k])}" for k in m),
        ]
        if self.cal is not None:
            lines += ["", "[plate]", f"corners_px = {_toml(np.round(self.cal.corners_px, 2).tolist())}"]
        for f in self.frames:
            lines += ["", "[[frames]]", *(f"{k} = {_toml(v)}" for k, v in f.items())]
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "session.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _new_session_name():
    today = datetime.now().strftime("%Y-%m-%d")
    for letter in "abcdefghijklmnopqrstuvwxyz":
        name = f"{today}_{letter}"
        if not (config.DATA_DIR / "captures" / name).exists():
            return name
    raise SystemExit(f"26 sessions already exist for {today}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture a session of frames with a live view; SPACE captures.")
    parser.add_argument("--session", help="session name; default today's date and the next free letter")
    parser.add_argument("--plan", choices=sorted(PLANS), default="scenes",
                        help="scenes: realistic arrangements (15 + 5); cutouts: spread out for cut-outs (12 + 6)")
    parser.add_argument("--count", type=int, help="frames in the plan's first part")
    parser.add_argument("--mixed", type=int, help="frames in the plan's second part")
    parser.add_argument("--lighting", default="", help="lighting, for session.toml")
    parser.add_argument("--notes", default="", help="anything else worth recording, for session.toml")
    args = parser.parse_args(argv)

    camera = config.load("camera")
    session = Session(args.session or _new_session_name(), args.lighting, args.notes, camera, args.plan)
    if session.frames and session.plan != args.plan:
        raise SystemExit(f"session {session.name} was captured with --plan {session.plan}; use that plan to continue it")
    session.plan = args.plan
    phases = [list(p) for p in PLANS[args.plan]]
    for part, count in enumerate((args.count, args.mixed)):
        if count is not None:
            phases[part][1] = count
    settings = rectify.Settings.from_config()
    threshold = config.load("dataset")["autolabel"]["diff_threshold"]
    empty = None                        # the rectified empty plate, for outlining cut-out objects
    print(f"Session {session.name}: {session.dir}")
    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    cap = open_camera(camera)
    print(__doc__.split("Keys:", 1)[1].rstrip())

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    view_cal, raw, grid, shown, n = session.last_cal, False, True, None, 0
    message = "ready" if session.cal is None else f"continuing: {len(session.frames)} frames so far"
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        if view_cal is None and n % RETRY_FRAMES == 1:
            view_cal, _ = find_plate(frame, settings)
        if args.plan == "cutouts" and empty is None and session.cal is not None and session.background.exists():
            empty, _ = rectify.white_balance(rectify.rectify(cv2.imread(str(session.background)), session.cal),
                                             session.cal.settings)

        phase = next(((kind, total, prompt) for kind, total, prompt in phases if session.count(kind) < total), None)
        if session.cal is None:
            steps = ["EMPTY PLATE", "Clear the plate. Keep hands and the arm out of view, then press SPACE."]
        elif phase is None:
            steps = [f"DONE: all {sum(t for _, t, _ in phases)} frames captured", "Press Q to finish."]
        else:
            kind, total, prompt = phase
            steps = [f"{kind.upper()}: frame {session.count(kind) + 1} of {total}",
                     f"{prompt}, move your hand away, then press SPACE."]
        if empty is not None and phase is not None and view_cal is not None and not raw:
            rect, _ = rectify.white_balance(rectify.rectify(frame, view_cal), view_cal.settings)
            rect, found, near = outline_objects(rect, empty, view_cal.settings, threshold)
            view = rectify.draw_grid(rect, view_cal.settings) if grid else rect
            steps.append(f"{found} objects outlined" + (f", {near} too near the edge (red)" if near else "")
                         + ": one outline per object")
        else:
            view = view_of(frame, view_cal, raw, grid)
        label(view, [f"Session {session.name}"] + steps + [message,
                     "SPACE capture  F force  U undo  E redo empty  R raw  G grid  Q quit"])
        shown = show(view, shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("g"):
            grid = not grid
        elif key == ord("r"):
            raw = not raw
        elif key == ord("u"):
            undone = session.undo()
            message = f"removed {undone}" if undone else "nothing to undo"
        elif key == ord("e"):
            if session.frames:
                message = "scene frames already use this calibration: quit and start a new session instead"
            else:
                session.cal, empty = None, None
                message = "redo the empty plate"
        elif key in (ord(" "), ord("f")):
            if session.cal is None:
                cal, status = find_plate(frame, settings)
                if cal is None:
                    message = f"NOT SAVED: {status}. Is something over the plate's edge?"
                else:
                    session.save_empty(frame, cal)
                    view_cal = cal
                    message = f"saved the empty plate and the calibration ({status})"
            elif phase is None:
                message = "all frames captured: press Q to finish"
            else:
                last = session.last_cal
                try:
                    cal, _ = rectify.recalibrate(frame, last)
                    found = True
                except rectify.PlateNotFound:
                    cal, found = last, False
                if not found and key != ord("f"):
                    message = "NOT SAVED: an edge of the plate is hidden. Move hands and the arm away (F saves anyway)"
                else:
                    name = session.save_scene(frame, phase[0], cal, found)
                    view_cal = cal
                    moved = float(np.abs(cal.corners_px - last.corners_px).max())
                    message = f"saved {name} ({phase[0]})"
                    if not found:
                        message += ", plate not found: used the last frame's corners"
                    elif moved > CAMERA_MOVED_PX:
                        message += f". The camera moved {moved:.0f} px since the last frame: is it secure?"
            print(message)

    cap.release()
    cv2.destroyAllWindows()
    done = {kind: session.count(kind) for kind, _, _ in phases}
    print(f"Session {session.name}: " + ", ".join(f"{v} {k}" for k, v in done.items()) + f" frames in {session.dir}")


if __name__ == "__main__":
    main()

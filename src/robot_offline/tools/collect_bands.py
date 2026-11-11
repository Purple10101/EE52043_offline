"""Collects the labelled band dataset from the resistor kit: one value on the plate at a time, so every
body found is labelled by the value it was placed for.

    python -m robot_offline.tools.collect_bands                    # session data/bands/kit/<today>
    python -m robot_offline.tools.collect_bands --session 2026-11-10_a --no-closeups
    python -m robot_offline.tools.collect_bands --session 2026-11-10 --only 56k,100k,1M     # just these, in order

The window shows the plate from above with what segmentation finds, and asks for each value in
configs/resistor_kit.toml in turn: put that value's resistors on the plate (all of them, spread out, leads
not touching, nothing else on it), then SPACE. Each SPACE is one placement: the plate is found, 8 frames
averaged at the config focus, segmented, each body tagged resistor or not (post/resistor_tag.py) and, unless
--no-closeups, a close-up taken of each resistor at the focus that suits it (post/bands/closeup.py). Between
placements, move the parts as the prompt says, so the captures differ: flipped end for end, the other half
of the plate, turned. After `captures` placements it moves to the next value.

Keys:  SPACE capture     r  redo the last capture (deletes it)     n / p  next / previous value
       c  find the plate again     q / Esc  quit (start again later: it carries on where it stopped)

Saved per capture, in <session>/<value>/capNN/: frame.png (the averaged frame), rectify.json, crops/ and
closeups/ (bodyN.png, as crops.cut cuts them for the reader) and capture.json (camera settings, white, and
per body its outline, tag, sharpness, focus and distance). Every crop is also a line in <session>/labels.jsonl,
which training reads; a body tagged "not" is kept but marked, to look at, not to train on.
"""
import argparse
import json
import re
import shutil
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post import resistor_tag
from robot_offline.post.bands import closeup, crops, learned
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.camera import open_camera, show

WINDOW = "collect bands"
ROOT = config.DATA_DIR / "bands" / "kit"
COLOURS = ("black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white")
MULTIPLIER = {-2: "silver", -1: "gold", **{i: c for i, c in enumerate(COLOURS)}}
NOT_RESISTORS = ("diode",)  # in the kit too: captured like a value, labelled not a resistor
PREVIEW_SECONDS = 1.5
# Detection threshold. Lower than live use (0.5): only the one value's resistors are on the plate, so a
# false body is unlikely and is tagged and easy to drop, while at 0.5 about one part a placement was lost,
# mostly on the near half, which the config focus leaves soft (2026-11-10: 150 found there, 218 on the far half).
CONF = 0.3       # segmentation on the live view this often, to check the parts are all found
MOVES = ("spread them out over the plate, leads not touching",
         "turn each one end for end",
         "move them to the other half of the plate (front to back)",
         "swap in fresh ones from the strip, turned any way",
         "put some right by the plate's edges, turned any way")


def parse_ohms(text):
    """Ohms from a strip's marking: 4k7, 4K7, 4.7k, 220R, 0R22, 1M, 4700, 4700R."""
    t = str(text).strip().replace(" ", "").replace("Ω", "").replace("ohm", "")
    m = re.fullmatch(r"(\d*)(?:\.(\d+))?([RrKkMm]?)(\d*)", t)
    if not m or not (m.group(1) or m.group(2)):
        raise ValueError(f"not a resistor value: {text!r}")
    whole, frac, unit, after = m.groups()
    if after and (frac or not unit):
        raise ValueError(f"not a resistor value: {text!r}")
    digits = f"{whole or 0}.{frac or after or 0}"
    return round(float(digits) * {"": 1, "r": 1, "k": 1e3, "m": 1e6}[unit.lower()], 6)


def bands_of(ohms, n=4):
    """Its colour bands with n = 4 (2 digits, multiplier, gold) or 5 (3 digits, multiplier, brown); None if
    the value needs more digits than that. A 0 ohm link has one black band."""
    if ohms is None:
        return None
    if ohms == 0:
        return ("black",)
    digits = n - 2
    for power in range(-2, 10):
        significant = ohms / 10 ** power
        if abs(significant - round(significant)) < 1e-6 and 10 ** (digits - 1) <= round(significant) < 10 ** digits:
            d = [COLOURS[int(c)] for c in str(int(round(significant)))]
            return tuple(d) + (MULTIPLIER[power], "gold" if n == 4 else "brown")
    return None


IMAGE_PX = 860                  # the plate image's side in the window
PANEL_W = 640                   # the text panel beside it
PANEL_BG, INK, MUTED = (246, 246, 246), (25, 25, 25), (110, 110, 110)
STYLES = {"title": (1.2, 2, INK), "big": (0.85, 2, INK), "text": (0.65, 1, INK), "muted": (0.55, 1, MUTED),
          "good": (0.7, 2, (30, 130, 30)), "alert": (0.8, 2, (0, 90, 220))}


def _wrap(text, scale, thick, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if line and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > width:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + ([line] if line else [])


def compose(image, rows):
    """The window: the plate image on the left, and beside it the text on a plain panel, never over the image.
    rows: (text, style) pairs, style one of STYLES; ("", "gap") for space."""
    if image is None:
        image = np.full((IMAGE_PX, IMAGE_PX, 3), 60, np.uint8)
    h, w = image.shape[:2]
    scale = IMAGE_PX / max(h, w)
    image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    left = np.full((IMAGE_PX, IMAGE_PX, 3), 60, np.uint8)
    left[:image.shape[0], :image.shape[1]] = image
    panel = np.full((IMAGE_PX, PANEL_W, 3), PANEL_BG, np.uint8)
    y = 20
    for text, style in rows:
        if style == "gap":
            y += 14
            continue
        size, thick, colour = STYLES[style]
        for line in _wrap(text, size, thick, PANEL_W - 40):
            y += int(34 * size) + 8
            cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (0, IMAGE_PX), (200, 200, 200), 2)
    return np.hstack([left, panel])


def select(markings, values, only):
    """The kit's (markings, values) cut down to those named in `only` (comma separated, in that order),
    each matched by value, so 56k, 56K and 56000 all find the kit's 56k."""
    by_ohms = {v: m for m, v in zip(markings, values)}
    picked = []
    for name in (x.strip() for x in only.split(",") if x.strip()):
        ohms = part_ohms(name)
        match = [v for v in by_ohms if (v is None and ohms is None) or (v is not None and ohms is not None
                                                                         and abs(v - ohms) < 1e-6)]
        if not match:
            raise ValueError(f"{name} isn't in configs/resistor_kit.toml")
        picked.append(match[0])
    return [by_ohms[v] for v in picked], picked


def exact(ohms):
    """Ohms written out in full, no rounding: 4700, 0.22, 1370."""
    return f"{ohms:f}".rstrip("0").rstrip(".")


def folder_name(ohms, marking=None):
    """A value's folder: its exact ohms, so no two values share one (4700R, 0_22R); a part that isn't a
    resistor (ohms None) by its marking (diode)."""
    return marking.lower() if ohms is None else exact(ohms).replace(".", "_") + "R"


def part_ohms(marking):
    """A config entry's ohms, or None for a part that isn't a resistor (NOT_RESISTORS)."""
    return None if str(marking).strip().lower() in NOT_RESISTORS else parse_ohms(marking)


def done_captures(folder):
    return sorted(p for p in folder.glob("cap*") if (p / "capture.json").exists())


def next_capture(folder):
    """The folder for a value's next placement: numbered after every capNN there, complete or not, so a
    placement taken out (moved to <session>/_excluded) never makes a new one reuse a kept one's number.
    A capture left part way (no capture.json) is removed first."""
    for old in folder.glob("cap*"):
        if not (old / "capture.json").exists():
            shutil.rmtree(old)
    numbers = [int(p.name[3:]) for p in folder.glob("cap*") if p.name[3:].isdigit()]
    return folder / f"cap{max(numbers, default=0) + 1:02d}"


def exclude(capture, session, why):
    """Takes a placement out of the dataset (wrong parts, say) without deleting it: moved to
    <session>/_excluded/<value>_<capNN>/ with why.txt. It is then not in labels.jsonl or counted as done."""
    dest = session / "_excluded" / f"{capture.parent.name}_{capture.name}"
    dest.parent.mkdir(exist_ok=True)
    shutil.move(str(capture), str(dest))
    (dest / "why.txt").write_text(why + "\n")
    return dest


def capture(cap, tracker, seg, grey, tagger, camera, closeups, out, models=None):
    """One placement: averaged frame, segmentation, tags, close-ups; written to out. Returns a summary line
    and the top-down view with the bodies drawn."""
    t0 = time.perf_counter()
    for _ in range(closeup.SETTLE):
        cap.read()
    frame = closeup.mean_frame(cap, 8)
    cal = tracker.cal
    ds = config.load("dataset")
    a, tc = ds["work_area"], ds["tiles"]
    plain, gains, white = closeup.white_of(cal, frame)
    view, _ = rectify.model_input(frame, cal, plain)
    t = tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
    polys, scores = tiling.predict_frame(seg, view, t, conf=CONF, grey=grey)
    if not polys:                       # nothing to label: not counted as a placement
        return "no bodies found, nothing saved: are the parts on the plate?", view
    tags = resistor_tag.classify(tagger, view, polys) if tagger is not None else [(None, None)] * len(polys)
    wanted = [i for i, (_, tag) in enumerate(tags) if tag != "not"]
    close = {}
    if closeups and wanted:
        close = dict(zip(wanted, closeup.take(cap, cal, [polys[i] for i in wanted], seg=seg, grey=grey,
                                                   camera=camera)))
    out.mkdir(parents=True)
    (out / "crops").mkdir()
    if close:
        (out / "closeups").mkdir()
    cv2.imwrite(str(out / "frame.png"), frame)
    cal.save(out / "rectify.json")
    bodies, shown = [], view.copy()
    for n, (poly, s, (p, tag)) in enumerate(zip(polys, scores, tags), start=1):
        c = crops.cut(frame, cal, poly, gains)
        cv2.imwrite(str(out / "crops" / f"body{n}.png"), c.image)
        body = dict(body=n, polygon=np.round(poly, 1).tolist(), detection=round(float(s), 3), tag=tag,
                    p_resistor=None if p is None else round(p, 3), sharpness=round(closeup.sharpness(c.image)))
        if n - 1 in close:
            k = close[n - 1]
            cv2.imwrite(str(out / "closeups" / f"body{n}.png"), k.crop.image)
            body["closeup"] = dict(focus=k.focus, distance_mm=round(k.distance_mm, 1), found_again=k.found_again,
                                   sharpness=round(k.sharpness), white=round(k.white, 1))
        bodies.append(body)
        colour = (0, 200, 0) if tag in (None, "resistor") else (0, 0, 255) if tag == "not" else (0, 200, 255)
        cv2.polylines(shown, [np.round(poly).astype(np.int32)], True, colour, 2, cv2.LINE_AA)
        x, y = np.round(poly.min(0)).astype(int)
        cv2.putText(shown, f"{n}" + ("" if tag in (None, "resistor") else f" {tag}"), (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    settings = dict(focus=camera["camera"]["focus"], exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
                    gain=cap.get(cv2.CAP_PROP_GAIN), frames_averaged=8)
    seconds = round(time.perf_counter() - t0, 1)
    (out / "capture.json").write_text(json.dumps(dict(camera=settings, white=round(white, 1), seconds=seconds, conf=CONF, models=models,
                                                      bodies=bodies), indent=1))
    n_not = sum(tag == "not" for _, tag in tags)
    line = (f"saved {len(bodies)} bodies" + (f" ({n_not} not a resistor)" if n_not else "")
            + (f", {len(close)} close-ups" if close else "") + f" in {seconds:.0f} s")
    return line, shown


def write_labels(session):
    """<session>/labels.jsonl afresh from every capture under it: one line per crop."""
    lines = []
    for cj in sorted(session.glob("*/cap*/capture.json")):
        meta = json.loads(cj.read_text())
        info = json.loads((cj.parent.parent / "value.json").read_text())
        for b in meta["bodies"]:
            common = dict(value_ohm=info["ohms"], marking=info["marking"], resistor=info["ohms"] is not None, capture=cj.parent.relative_to(session).as_posix(),
                          body=b["body"], tag=b["tag"], p_resistor=b["p_resistor"])
            lines.append(dict(common, kind="first", image=f"{common['capture']}/crops/body{b['body']}.png",
                              white=meta["white"], focus=meta["camera"]["focus"], sharpness=b["sharpness"]))
            if "closeup" in b:
                k = b["closeup"]
                lines.append(dict(common, kind="closeup", image=f"{common['capture']}/closeups/body{b['body']}.png",
                                  white=k["white"], focus=k["focus"], sharpness=k["sharpness"],
                                  distance_mm=k["distance_mm"], found_again=k["found_again"]))
    (session / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return len(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m robot_offline.tools.collect_bands", description=__doc__.split("\n")[0])
    p.add_argument("--session", default=date.today().isoformat(), help="folder under data/bands/kit (today's date)")
    p.add_argument("--seg-weights", type=Path, default=config.ROOT / "models" / "seg_grey_v2" / "best.pt")
    p.add_argument("--tagger", type=Path, default=config.ROOT / "models" / "resistor_tag_v1" / "resistor_tag.pt")
    p.add_argument("--no-closeups", action="store_true", help="skip the per-resistor close-ups (about 3.5 s each)")
    p.add_argument("--captures", type=int, help="placements per value (configs/resistor_kit.toml)")
    p.add_argument("--only", help="collect just these values, in this order, comma separated (e.g. 56k,100k,1M); "
                                   "each must be in configs/resistor_kit.toml")
    p.add_argument("--search", action="store_true",
                   help="look for the plate afresh instead of using where it was last found (it, and the camera, "
                        "don't move while collecting; c searches at any time)")
    args = p.parse_args(argv)
    kit = config.load("resistor_kit")
    markings = [str(v) for v in kit["values"]]
    if not markings:
        p.error("no values in configs/resistor_kit.toml: list the kit's values there first")
    values = [part_ohms(v) for v in markings]
    if len({folder_name(v, m) for v, m in zip(values, markings)}) != len(values):
        p.error("configs/resistor_kit.toml lists a value twice")
    if args.only:
        try:
            markings, values = select(markings, values, args.only)
        except ValueError as e:
            p.error(str(e))
    per_value = args.captures or kit["captures"]
    per_plate = kit.get("per_plate", 8)
    session = ROOT / args.session
    session.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    seg, grey = YOLO(str(args.seg_weights)), tiling.wants_grey(args.seg_weights)
    tagger = resistor_tag.load(args.tagger) if args.tagger.exists() else None
    models = dict(segmentation=str(args.seg_weights), grey=grey, tagger=str(args.tagger) if tagger is not None else None)
    camera = config.load("camera")
    ds = config.load("dataset")
    a, tc = ds["work_area"], ds["tiles"]
    cap = open_camera(camera)
    # The plate and camera stay put while collecting, so by default the last fit is used as it is (locked):
    # no search, which needs the far edge or the centre cross, and the kit's long leads often cover the
    # cross. If the plate has moved, the top-down image shows it shifted: c searches again.
    last = None if args.search or not plate_search.LAST.exists() else rectify.Calibration.load(plate_search.LAST)
    tracker = plate_search.Tracker(cap, rectify.Settings.from_config(ds), cal=last, camera=camera,
                                   locked=last is not None, level=learned.READING_LEVEL)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def folder(i):
        f = session / folder_name(values[i], markings[i])
        if not (f / "value.json").exists():
            f.mkdir(exist_ok=True)
            (f / "value.json").write_text(json.dumps(dict(marking=markings[i], ohms=values[i],
                                                          bands_4=bands_of(values[i], 4), bands_5=bands_of(values[i], 5))))
        return f

    index = next((i for i in range(len(values)) if len(done_captures(session / folder_name(values[i], markings[i]))) < per_value),
                 len(values))
    message, result, preview, next_preview, shown = "", None, None, 0.0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cal = tracker.update(frame)
            if index >= len(values):
                view = compose(None, [("All done", "title"), ("", "gap"),
                                      (f"All {len(values)} values captured, {per_value} placements each.", "text"),
                                      (f"{write_labels(session)} crops listed in {session / 'labels.jsonl'}", "muted"),
                                      ("", "gap"), ("p  go back to a value      q  quit", "text")])
            else:
                f = folder(index)
                done = len(done_captures(f))
                now = time.perf_counter()
                if cal is not None and not tracker.searching and now >= next_preview and result is None:
                    plain = rectify.rectify(frame, cal)
                    mv, _ = rectify.model_input(frame, cal, plain)
                    t = tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (mv.shape[1], mv.shape[0]),
                                     tc["size"], tc["overlap"])
                    polys, _ = tiling.predict_frame(seg, mv, t, conf=CONF, grey=grey)
                    preview = mv.copy()
                    for poly in polys:
                        cv2.polylines(preview, [np.round(poly).astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
                    preview_count = len(polys)
                    next_preview = time.perf_counter() + PREVIEW_SECONDS
                image = result if result is not None else preview if preview is not None else frame
                b4, b5 = bands_of(values[index], 4), bands_of(values[index], 5)
                if values[index] is None:
                    bands = "not a resistor: for the resistor tag and to check the reader holds back"
                elif b5 == ("black",):
                    bands = "one black band"
                else:
                    bands = "   ".join(f"{len(b)}-band: {' '.join(b)}" for b in (b5, b4) if b)
                todo = (MOVES[done % len(MOVES)] if done else
                        f"put {per_plate} of the {markings[index]}s on the plate (nothing else), " + MOVES[0])
                rows = [(f"Value {index + 1} of {len(values)}", "muted"),
                        (markings[index] + ("" if values[index] is None else f"   ({exact(values[index])} ohm)"), "title"),
                        *[(b, "text") for b in bands.split("   ")], ("", "gap"),
                        (f"Placement {min(done + 1, per_value)} of {per_value}", "big"),
                        (todo + ", then press SPACE." if result is None else "", "text"),
                        ("Using the plate's last position. The top-down image should show the plate filling "
                         "its square; if it looks shifted, press c to find it again." if tracker.locked else
                         "Keep the cross in the middle of the plate clear: in dim or evening light the plate is "
                         "found by it.", "muted"), ("", "gap")]
                if result is not None:
                    rows.append((message, "good" if "saved" in message else "alert"))
                elif preview is not None:
                    rows.append((f"Segmentation sees {preview_count} bodies", "big"))
                elif cal is None or tracker.searching:
                    rows.append(("Finding the plate ... if it takes long, move any lead off the centre cross "
                                 "(or clear the plate until it's found, then put the parts back).", "alert"))
                if message and result is None:
                    rows.append((message, "alert"))
                rows += [("", "gap"), (tracker.status(), "muted"),
                         ("" if tracker.control is None else tracker.control.status(), "muted"), ("", "gap"),
                         ("SPACE  capture", "text"), ("r  redo the last capture", "text"),
                         ("n / p  next / previous value", "text"), ("c  find the plate again", "text"),
                         ("q  quit (carries on where it stopped next time)", "text")]
                view = compose(image, [r for r in rows if r[0] or r[1] == "gap"])
            shown = show(view, shown, WINDOW)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("n") and index < len(values):
                index, message, result, preview = index + 1, "", None, None
            elif key == ord("p") and index > 0:
                index, message, result, preview = index - 1, "", None, None
            elif key == ord("c"):
                tracker.restart()
                message = "looking for the plate again"
            elif key == ord("r") and index < len(values):
                caps = done_captures(folder(index))
                if caps:
                    shutil.rmtree(caps[-1])
                    write_labels(session)
                    message, result = f"deleted {caps[-1].name}: capture it again", None
            elif key == ord(" ") and index < len(values):
                if result is not None:          # the last capture's result is showing: SPACE moves on
                    result, next_preview = None, 0.0
                    if len(done_captures(folder(index))) >= per_value:
                        index, message = index + 1, ""
                    continue
                if cal is None or tracker.searching:
                    message = "the plate isn't found yet: wait for it (c looks again)"
                    continue
                f = folder(index)
                n = len(done_captures(f)) + 1              # the placement's number as the user sees it
                target = next_capture(f)
                busy = compose(image, [(markings[index], "title"), ("", "gap"),
                                       (f"Capturing placement {n} of {per_value}", "big"), ("", "gap"),
                                       ("Keep your hands clear of the plate.", "alert"),
                                       ("Close-ups take about 3.5 s per resistor." if not args.no_closeups else "", "text")])
                shown = show(busy, shown, WINDOW)
                cv2.waitKey(1)
                line, result = capture(cap, tracker, seg, grey, tagger, camera, not args.no_closeups, target, models)
                write_labels(session)
                saved = (target / "capture.json").exists()
                message = f"Placement {n}: {line}." + (" SPACE to go on, r to redo it." if saved else " SPACE to go back.")
                print(message, flush=True)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        if session.exists():
            print(f"{write_labels(session)} crops in {session / 'labels.jsonl'}")


if __name__ == "__main__":
    main()

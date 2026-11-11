"""Live detection: the model run on the camera, with the body outlines drawn on the image.

Opens the camera with configs/camera.toml, finds the plate, and a few times a second rectifies the
newest frame, white-balances it with the plate brought to the brightness it has in training,
masks it to the plate (exactly what the model saw in training), runs the model and maps each
predicted outline back into the camera's view, where it is drawn over the resistor with its spine
and centroid (skeleton.py) in white. The plate is found again every few seconds in the
background (plate_search.Tracker), so a nudged camera doesn't
spoil it and the picture doesn't wait for it. The exposure follows the light, metered on the
plate's white top (exposure.py); E turns that off, back to the fixed exposure in camera.toml.

Detections are held steady from run to run (steady.py): a body shows once it passes the threshold in 2
of the last 5 runs and stays while its mean score holds, so one whose score swings round the threshold
doesn't flicker; its score shown is that mean. H shows single runs instead, to compare.

Each body's value is read from its bands (post/bands/learned.py, --reader), averaged over its last few
runs: "4k7 0.99" is a reading, "? 1k 0.50" a best guess below the reader's limit, which the robot would
measure instead. With values on, the exposure keeps the plate at the reading level (grey 205 to 225),
where the bands read best; B turns values off, and the exposure goes back to the model's band.

Each body is also tagged resistor or not (post/resistor_tag.py, --tagger), its P(resistor) averaged over
the same runs, since an object model outlines screws, diodes and the rest too. Bands are read only on
a body that isn't tagged "not". On one tagged "unsure" the reading is shown as "unsure 0.55: ~4k7, measure":
never a value to use, whatever the reader's confidence. Without the tagger's weights every body counts
as a resistor.

On the right, one tile per detected body shows only its segmented pixels: taken from the top-down
image (the same 10 px/mm for every body, free of the camera's angle), turned so the body lies
horizontally, enlarged, and black outside the predicted outline. Tiles are numbered and coloured
like the outlines. On each, skeleton.py's results: the thinned skeleton (cyan pixels), the spine
from end to end (white) and the centroid (white dot), labelled with the body's length × width and
yaw along its spine and its centroid in scene mm on the plate. K hides and shows all of these.

    python -m robot_offline.inf.live_detect
    python -m robot_offline.inf.live_detect --weights models/seg_grey_v2/best.pt --fps 3 --conf 0.5

Keys:  q / Esc  quit      v  camera view / top-down view      m  top-down: what the model sees (masked)
       + / -  confidence threshold up / down      c  find the plate again      s  save the shown image
       e  exposure following the light, or fixed      k  skeleton, spine and centroid shown or hidden
       h  detections held steady over runs, or each run on its own      b  band values on or off
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.inf.steady import FLOOR, Steady
from robot_offline.post import resistor_tag, skeleton
from robot_offline.post.bands import crops, learned
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.camera import label, open_camera, show

WINDOW = "robot_offline live detection"
CAMERA_SCALE = 0.4          # the camera view is shown at this share of 4608 × 2592
COLOURS = ((0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 200, 80))
TILE = (340, 150)           # px, each body's cut-out tile
ZOOM = 4.0                  # cut-outs are shown this much larger than the 10 px/mm top-down image
PANEL_COLUMNS = 2
SKELETON_COLOUR = (255, 255, 0)


def _text(img, text, org, colour, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw(img, polygons, scores, scale=1.0, axes=(), notes=()):
    """Numbered outlines, a light fill and the confidence on img (BGR), in place. Polygons in
    img px / scale. axes: (spine, centroid) per body, in the same px, drawn as a line and a dot.
    notes: text after each body's confidence (its value)."""
    fill = img.copy()
    for i, poly in enumerate(polygons):
        cv2.fillPoly(fill, [np.round(poly * scale).astype(np.int32)], COLOURS[i % len(COLOURS)], cv2.LINE_AA)
    cv2.addWeighted(fill, 0.25, img, 0.75, 0, dst=img)
    for i, (poly, score) in enumerate(zip(polygons, scores)):
        pts = np.round(poly * scale).astype(np.int32)
        colour = COLOURS[i % len(COLOURS)]
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        x, y = pts.min(0)
        note = f"   {notes[i]}" if i < len(notes) and notes[i] else ""
        _text(img, f"#{i + 1} {score:.2f}{note}", (int(x), int(y) - 4), colour)
    for spine, centroid in axes:
        cv2.polylines(img, [np.round(spine * scale * 16).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA, 4)
        cv2.circle(img, tuple(int(v) for v in np.round(centroid * scale * 16)), 3 * 16, (255, 255, 255), -1,
                   cv2.LINE_AA, 4)


def cutout(image, poly, size=TILE, zoom=ZOOM, body=None):
    """Only the pixels inside poly (image px), turned so the body's long axis is horizontal,
    enlarged by zoom and centred in a size tile; black elsewhere. With body (its
    skeleton.BodySkeleton), its skeleton, spine and centroid are drawn on."""
    m = cv2.moments(poly.astype(np.float32))
    if m["m00"] <= 0:
        return np.zeros((size[1], size[0], 3), np.uint8)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))   # long axis
    rot = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    rot[:, 2] += (size[0] / 2 - cx, size[1] / 2 - cy)
    inside = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    pixels = cv2.warpAffine(image, rot, size, flags=cv2.INTER_CUBIC)
    keep = cv2.warpAffine(inside, rot, size, flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    tile = (pixels * keep).astype(np.uint8)
    if body is not None:
        to_tile = lambda p: np.asarray(p, float).reshape(-1, 2) @ rot[:, :2].T + rot[:, 2]
        for x, y in np.round(to_tile(body.skeleton)).astype(int):
            cv2.circle(tile, (int(x), int(y)), 1, SKELETON_COLOUR, -1)
        cv2.polylines(tile, [np.round(to_tile(body.spine) * 16).astype(np.int32)], False, (255, 255, 255), 1,
                      cv2.LINE_AA, 4)
        centre = tuple(int(v) for v in np.round(to_tile(body.centroid)[0] * 16))
        cv2.circle(tile, centre, 5 * 16, (0, 0, 0), -1, cv2.LINE_AA, 4)
        cv2.circle(tile, centre, 4 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
    return tile


def cutout_panel(image, polygons, scores, height, bodies=(), settings=None):
    """A panel of numbered cut-outs, PANEL_COLUMNS wide and height tall. With bodies
    (skeleton.BodySkeleton or None per polygon) and the rectify settings, each tile shows its body's
    skeleton, spine and centroid, with its length × width, yaw and centroid in scene mm."""
    cell_w, cell_h = TILE[0] + 20, TILE[1] + 34
    panel = np.full((height, PANEL_COLUMNS * cell_w, 3), 24, np.uint8)
    rows = max(1, (height - 10) // cell_h)
    _text(panel, "segmented pixels, top-down, 4x", (10, 20), (220, 220, 220))
    for i, (poly, score) in enumerate(zip(polygons, scores)):
        if i >= rows * PANEL_COLUMNS:
            _text(panel, f"+{len(polygons) - i} more", (10, height - 10), (220, 220, 220))
            break
        col, row = divmod(i, rows)
        x, y = 10 + col * cell_w, 30 + row * cell_h
        colour = COLOURS[i % len(COLOURS)]
        body = bodies[i] if i < len(bodies) and settings is not None else None
        s = settings.px_per_mm if body is not None else 1.0
        size = f"   {body.length / s:.1f} x {body.width / s:.1f} mm  {body.yaw_deg:.0f} deg" if body else ""
        _text(panel, f"#{i + 1}  {score:.2f}{size}", (x, y + 16), colour)
        tile = cutout(image, poly, body=body)
        if body is not None:
            cx, cy = rectify.rect_to_mm(body.centroid, settings)[0]
            _text(tile, f"centroid x {cx:.1f}  y {cy:.1f} mm", (6, TILE[1] - 8), (220, 220, 220), 0.45)
        panel[y + 22:y + 22 + TILE[1], x:x + TILE[0]] = tile
        cv2.rectangle(panel, (x - 1, y + 21), (x + TILE[0], y + 22 + TILE[1]), colour, 1)
    return panel


class TagViews:
    """Each body's resistor logit averaged over its last `window` runs (log-odds average, like
    learned.Views), so one run's odd crop doesn't flip its tag. Bodies are named by steady.Steady's ids."""

    def __init__(self, window=5):
        self.window, self.seen = window, {}

    def add(self, body, logit):
        """One run's logit for body. Returns (P(resistor), tag) over its recent runs."""
        runs = self.seen.setdefault(body, deque(maxlen=self.window))
        runs.append(float(logit))
        p = float(resistor_tag.probability(np.mean(runs)))
        return p, resistor_tag.tag(p)

    def keep_only(self, bodies):
        self.seen = {b: v for b, v in self.seen.items() if b in set(bodies)}


def tag_bodies(tagger, tag_views, model_view, polygons, ids=None):
    """[(P(resistor), tag)] per polygon, from the colour model view; averaged per id when ids are given.
    Without a tagger every body is (None, None): counted as a resistor."""
    if tagger is None:
        return [(None, None)] * len(polygons)
    z = resistor_tag.logits(tagger, [resistor_tag.cutout(model_view, q) for q in polygons])
    if ids is None:
        return [(float(p), resistor_tag.tag(p)) for p in resistor_tag.probability(z)]
    tags = [tag_views.add(i, v) for i, v in zip(ids, z)]
    tag_views.keep_only(ids)
    return tags


def tag_text(p, tag):
    return "" if tag is None else "not a resistor" if tag == "not" else f"{tag} {p:.2f}"


def read_values(net, views, frame, cal, plain, polygons, ids=None, tags=None):
    """Each body's value as text, read as tools.read_plate reads it. ids: the bodies' ids (steady.Steady), to
    average each one's reading over its recent runs; None reads this run alone. tags: (P(resistor), tag) per
    body: a body tagged "not" isn't read, and one tagged "unsure" is shown as a value to measure, not use."""
    tags = tags or [(None, None)] * len(polygons)
    _, gains = rectify.white_balance(plain, cal.settings)          # crops keep the frame's own level
    balanced = np.clip(plain.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    white = float(np.percentile(cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)[150:1050, 150:1050], 70))
    notes = []
    for i, (poly, (p, tag)) in enumerate(zip(polygons, tags)):
        if tag == "not":
            notes.append(tag_text(p, tag))
            continue
        lp = learned.log_probs(net, crops.cut(frame, cal, poly, gains).image, white)
        r = views.add(ids[i], lp) if ids is not None else learned.decode(lp)
        if tag == "unsure":
            notes.append(f"{tag_text(p, tag)}: ~{learned.ohms_text(r.best)}, measure")
        else:
            notes.append(f"{learned.ohms_text(r.value)} {r.confidence:.2f}" if r.value is not None
                         else f"? {learned.ohms_text(r.best)} {r.confidence:.2f}")
    if ids is not None:
        views.keep_only(ids)
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.inf.live_detect",
                                     description="Run the model live on the camera and draw the body outlines.")
    parser.add_argument("--weights", type=Path, default=config.ROOT / "models" / "seg_grey_v2" / "best.pt")
    parser.add_argument("--fps", type=float, default=3.0, help="detections per second (3)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0.5)")
    parser.add_argument("--reader", type=Path, default=config.ROOT / "models" / "band_reader_kit_v1" / "band_reader.pt",
                        help="band reader weights (values aren't shown if missing)")
    parser.add_argument("--tagger", type=Path,
                        default=config.ROOT / "models" / "resistor_tag_v1" / "resistor_tag.pt",
                        help="resistor-or-not weights (every body counts as a resistor if missing)")
    args = parser.parse_args(argv)
    if not args.weights.exists():
        parser.error(f"no weights at {args.weights}; train first, e.g. python -m robot_offline.training.train_synth")
    if args.weights.stat().st_size < 1_000_000:
        parser.error(f"{args.weights} is a Git LFS pointer, not the model; run git lfs pull")

    from ultralytics import YOLO

    ds = config.load("dataset")
    settings = rectify.Settings.from_config(ds)
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    print(f"Loading {args.weights} ...")
    model = YOLO(str(args.weights))
    grey = tiling.wants_grey(args.weights)
    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    camera = config.load("camera")
    cap = open_camera(camera)
    fixed = camera["camera"]["exposure"], camera["camera"]["gain"]
    print(__doc__.split("Keys:", 1)[1].rstrip())
    net = learned.load(args.reader) if args.reader.exists() else None
    if net is None:
        print(f"No band reader at {args.reader}: no values")
    values, views = net is not None, learned.Views()
    tagger = resistor_tag.load(args.tagger) if args.tagger.exists() else None
    if tagger is None:
        print(f"No resistor tag at {args.tagger}: every body counts as a resistor")
    tag_views = TagViews()
    fitter = plate_search.Tracker(cap, settings, camera=camera,      # a hand over an edge: it keeps the last fit
                                  level=learned.READING_LEVEL if values else None)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    top_down, masked, conf, skeletons, hold = False, False, args.conf, True, True
    steady = Steady(conf)
    polygons, scores, notes, view, shown = [], [], [], None, None
    next_run, latency, post, n_runs, t_start = 0.0, 0.0, 0.0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()                             # keep reading so the newest frame is used
        if not ok:
            continue
        cal = fitter.update(frame)
        now = time.perf_counter()

        if now >= next_run:
            next_run = now + 1.0 / args.fps
            t0 = time.perf_counter()
            if cal is not None:
                plain = rectify.rectify(frame, cal)
                model_view, rect = rectify.model_input(frame, cal, plain)
                tiles = tiling.tiles(work_area, (model_view.shape[1], model_view.shape[0]),
                                     tiles_cfg["size"], tiles_cfg["overlap"])
                polygons, scores = tiling.predict_frame(model, model_view, tiles, conf=FLOOR, grey=grey)
                held = steady.update(polygons, scores)         # followed always, so h switches cleanly
                if hold:
                    polygons, scores = held
                else:
                    polygons, scores = [p for p, c in zip(polygons, scores) if c >= conf], [c for c in scores if c >= conf]
                ids = steady.ids if hold else None
                tags = tag_bodies(tagger, tag_views, model_view, polygons, ids)
                notes = (read_values(net, views, frame, cal, plain, polygons, ids, tags) if values
                         else [tag_text(p, t) for p, t in tags])
                t1 = time.perf_counter()
                bodies = [skeleton.body_skeleton(p) for p in polygons] if skeletons else []
                post = time.perf_counter() - t1
                axes = [(b.spine, b.centroid) for b in bodies if b is not None]
                if top_down:
                    view = (model_view if masked else rectify.draw_grid(rect, cal.settings)).copy()
                    draw(view, polygons, scores, axes=axes, notes=notes)
                else:
                    view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                    cv2.polylines(view, [np.round(rectify.plate_outline_px(cal) * CAMERA_SCALE).astype(np.int32)],
                                  True, (255, 255, 255), 1, cv2.LINE_AA)
                    draw(view, [rectify.to_full(p, cal) for p in polygons], scores, CAMERA_SCALE,
                         [(rectify.to_full(s, cal), rectify.to_full(c, cal)[0]) for s, c in axes], notes)
                view = np.hstack([view, cutout_panel(rect, polygons, scores, view.shape[0], bodies, cal.settings)])
            else:
                view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                view = np.hstack([view, cutout_panel(view, [], [], view.shape[0])])
            latency = time.perf_counter() - t0
            n_runs += 1
            rate = n_runs / max(time.perf_counter() - t_start, 1e-6)
            mode = ("top-down, as the model sees it" if masked else "top-down") if top_down else "camera view"
            plate = fitter.status()
            exposure = fitter.control.status() if fitter.follow else f"exposure fixed at {fixed[0]}, gain {fixed[1]}"
            shape = f"skeletons {post * 1000:.0f} ms" if skeletons else "skeletons off"
            if cal is None or tagger is None:
                found = f"{len(polygons)} bodies"
            else:
                found = f"{len(polygons)} bodies, {sum(t == 'resistor' for _, t in tags)} resistors"
            label(view, [f"{found}   conf >= {conf:.2f} "
                         f"({'held over runs' if hold else 'each run alone'})   {mode}",
                         f"{rate:.1f} detections/s, {latency * 1000:.0f} ms each ({shape})   {plate}", exposure,
                         "q quit  v view  m model's input  +/- confidence  c find plate  e exposure  "
                         "k skeletons  h hold  b values  s save"])
            shown = show(view, shown, WINDOW)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("v"):
            top_down, next_run = not top_down, 0.0
        elif key == ord("m"):
            masked, top_down, next_run = not masked, True, 0.0
        elif key in (ord("+"), ord("=")):
            conf, next_run = min(0.95, conf + 0.05), 0.0
            steady.conf = conf
        elif key in (ord("-"), ord("_")):
            conf, next_run = max(0.05, conf - 0.05), 0.0
            steady.conf = conf
        elif key == ord("c"):
            fitter.restart()
            steady.clear()
        elif key == ord("h"):
            hold, next_run = not hold, 0.0
        elif key == ord("b") and net is not None:
            values, next_run = not values, 0.0
            fitter.level = learned.READING_LEVEL if values else None
        elif key == ord("k"):
            skeletons, next_run = not skeletons, 0.0
        elif key == ord("e"):
            fitter.follow = not fitter.follow
            if not fitter.follow:
                cap.set(cv2.CAP_PROP_EXPOSURE, fixed[0])
                cap.set(cv2.CAP_PROP_GAIN, fixed[1])
        elif key == ord("s") and view is not None:
            out = config.DATA_DIR / "live"
            out.mkdir(parents=True, exist_ok=True)
            path = out / datetime.now().strftime("detect_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(path), view)
            print(f"Saved {path}")

    fitter.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

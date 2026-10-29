"""Live detection: the model run on the camera, with the body outlines drawn on the image.

Opens the camera with configs/camera.toml, finds the plate, and a few times a second rectifies the
newest frame, white-balances it with the plate brought to the brightness it has in training,
masks it to the plate (exactly what the model saw in training), runs the model and maps each
predicted outline back into the camera's view, where it is drawn over the resistor with its spine
and centroid (skeleton.py) in white. The plate is found again every REFIND_SECONDS in the
background (live_plate.Fitter), so a nudged camera doesn't
spoil it and the picture doesn't wait for it. The exposure follows the light, metered on the
plate's white top (exposure.py); E turns that off, back to the fixed exposure in camera.toml.

On the right, one tile per detected body shows only its segmented pixels: taken from the top-down
image (the same 10 px/mm for every body, free of the camera's angle), turned so the body lies
horizontally, enlarged, and black outside the predicted outline. Tiles are numbered and coloured
like the outlines. On each, skeleton.py's results: the thinned skeleton (cyan pixels), the spine
from end to end (white) and the centroid (white dot), labelled with the body's length × width and
yaw along its spine and its centroid in scene mm on the plate. K hides and shows all of these.

    python -m robot_offline.inf.live_detect
    python -m robot_offline.inf.live_detect --weights models/synth_only_v1/best.pt --fps 3 --conf 0.5

Keys:  q / Esc  quit      v  camera view / top-down view      m  top-down: what the model sees (masked)
       + / -  confidence threshold up / down      c  find the plate again      s  save the shown image
       e  exposure following the light, or fixed      k  skeleton, spine and centroid shown or hidden
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post import skeleton
from robot_offline.pre import rectify
from robot_offline.pre.camera import label, open_camera, show
from robot_offline.pre.exposure import ExposureControl
from robot_offline.pre.live_plate import Fitter

WINDOW = "robot_offline live detection"
CAMERA_SCALE = 0.4          # the camera view is shown at this share of 4608 × 2592
REFIND_SECONDS = 5.0
COLOURS = ((0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 200, 80))
TILE = (340, 150)           # px, each body's cut-out tile
ZOOM = 4.0                  # cut-outs are shown this much larger than the 10 px/mm top-down image
PANEL_COLUMNS = 2
SKELETON_COLOUR = (255, 255, 0)


def _text(img, text, org, colour, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw(img, polygons, scores, scale=1.0, axes=()):
    """Numbered outlines, a light fill and the confidence on img (BGR), in place. Polygons in
    img px / scale. axes: (spine, centroid) per body, in the same px, drawn as a line and a dot."""
    fill = img.copy()
    for i, poly in enumerate(polygons):
        cv2.fillPoly(fill, [np.round(poly * scale).astype(np.int32)], COLOURS[i % len(COLOURS)], cv2.LINE_AA)
    cv2.addWeighted(fill, 0.25, img, 0.75, 0, dst=img)
    for i, (poly, score) in enumerate(zip(polygons, scores)):
        pts = np.round(poly * scale).astype(np.int32)
        colour = COLOURS[i % len(COLOURS)]
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        x, y = pts.min(0)
        _text(img, f"#{i + 1} {score:.2f}", (int(x), int(y) - 4), colour)
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.inf.live_detect",
                                     description="Run the model live on the camera and draw the body outlines.")
    parser.add_argument("--weights", type=Path, default=config.ROOT / "models" / "synth_only_v1" / "best.pt")
    parser.add_argument("--fps", type=float, default=3.0, help="detections per second (3)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0.5)")
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
    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    camera = config.load("camera")
    cap = open_camera(camera)
    fixed = camera["camera"]["exposure"], camera["camera"]["gain"]
    control = ExposureControl(camera["exposure_control"], *fixed, cap)
    print(__doc__.split("Keys:", 1)[1].rstrip())
    fitter = Fitter(settings, interval=REFIND_SECONDS)     # a hand over an edge: it keeps the last calibration
    fitter.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    top_down, masked, conf, follow, skeletons = False, False, args.conf, True, True
    polygons, scores, view, shown = [], [], None, None
    next_run, latency, post, n_runs, t_start = 0.0, 0.0, 0.0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()                             # keep reading so the newest frame is used
        if not ok:
            continue
        fitter.frame = frame
        cal = fitter.cal
        if follow:
            control.meter(frame, cal)
        now = time.perf_counter()

        if now >= next_run:
            next_run = now + 1.0 / args.fps
            t0 = time.perf_counter()
            if cal is not None:
                model_view, rect = rectify.model_input(frame, cal)
                tiles = tiling.tiles(work_area, (model_view.shape[1], model_view.shape[0]),
                                     tiles_cfg["size"], tiles_cfg["overlap"])
                polygons, scores = tiling.predict_frame(model, model_view, tiles, conf=conf)
                t1 = time.perf_counter()
                bodies = [skeleton.body_skeleton(p) for p in polygons] if skeletons else []
                post = time.perf_counter() - t1
                axes = [(b.spine, b.centroid) for b in bodies if b is not None]
                if top_down:
                    view = (model_view if masked else rectify.draw_grid(rect, cal.settings)).copy()
                    draw(view, polygons, scores, axes=axes)
                else:
                    view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                    cv2.polylines(view, [np.round(rectify.plate_outline_px(cal) * CAMERA_SCALE).astype(np.int32)],
                                  True, (255, 255, 255), 1, cv2.LINE_AA)
                    draw(view, [rectify.to_full(p, cal) for p in polygons], scores, CAMERA_SCALE,
                         [(rectify.to_full(s, cal), rectify.to_full(c, cal)[0]) for s, c in axes])
                view = np.hstack([view, cutout_panel(rect, polygons, scores, view.shape[0], bodies, cal.settings)])
            else:
                view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                view = np.hstack([view, cutout_panel(view, [], [], view.shape[0])])
            latency = time.perf_counter() - t0
            n_runs += 1
            rate = n_runs / max(time.perf_counter() - t_start, 1e-6)
            mode = ("top-down, as the model sees it" if masked else "top-down") if top_down else "camera view"
            if cal is None:
                plate = f"looking for the plate: {fitter.error}" if fitter.error else "looking for the plate"
            else:
                plate = f"plate: near side by the {cal.fit['near_side_by']}"
                if fitter.error:
                    plate += " (not found in the latest frame: kept the last fit)"
            exposure = control.status() if follow else f"exposure fixed at {fixed[0]}, gain {fixed[1]}"
            shape = f"skeletons {post * 1000:.0f} ms" if skeletons else "skeletons off"
            label(view, [f"{len(polygons)} resistor bodies   conf >= {conf:.2f}   {mode}",
                         f"{rate:.1f} detections/s, {latency * 1000:.0f} ms each ({shape})   {plate}", exposure,
                         "q quit  v view  m model's input  +/- confidence  c find plate  e exposure  "
                         "k skeletons  s save"])
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
        elif key in (ord("-"), ord("_")):
            conf, next_run = max(0.05, conf - 0.05), 0.0
        elif key == ord("c"):
            fitter.restart = True
        elif key == ord("k"):
            skeletons, next_run = not skeletons, 0.0
        elif key == ord("e"):
            follow = not follow
            if not follow:
                cap.set(cv2.CAP_PROP_EXPOSURE, fixed[0])
                cap.set(cv2.CAP_PROP_GAIN, fixed[1])
                control = ExposureControl(camera["exposure_control"], *fixed, cap)
        elif key == ord("s") and view is not None:
            out = config.DATA_DIR / "live"
            out.mkdir(parents=True, exist_ok=True)
            path = out / datetime.now().strftime("detect_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(path), view)
            print(f"Saved {path}")

    fitter.stop = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

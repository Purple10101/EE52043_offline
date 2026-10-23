"""Live detection: the model run on the camera, with the body outlines drawn on the image.

Opens the camera with configs/camera.toml, finds the plate, and a few times a second rectifies the
newest frame, masks it to the plate (exactly what the model saw in training), runs the model and
maps each predicted outline back into the camera's view, where it is drawn over the resistor. The
plate is found again every REFIND_SECONDS, so a nudged camera doesn't spoil it.

On the right, one tile per detected body shows only its segmented pixels: taken from the top-down
image (the same 10 px/mm for every body, free of the camera's angle), turned so the body lies
horizontally, enlarged, and black outside the predicted outline. Tiles are numbered and coloured
like the outlines.

    python -m robot_offline.live_detect
    python -m robot_offline.live_detect --weights runs/synth_only/weights/best.pt --fps 3 --conf 0.5

Keys:  q / Esc  quit      v  camera view / top-down view      m  top-down: what the model sees (masked)
       + / -  confidence threshold up / down      c  find the plate again      s  save the shown image
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, rectify, tiling
from robot_offline.capture import find_plate, label, open_camera, show

WINDOW = "robot_offline live detection"
CAMERA_SCALE = 0.4          # the camera view is shown at this share of 4608 × 2592
REFIND_SECONDS = 5.0
COLOURS = ((0, 0, 255), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 200, 80))
TILE = (340, 150)           # px, each body's cut-out tile
ZOOM = 4.0                  # cut-outs are shown this much larger than the 10 px/mm top-down image
PANEL_COLUMNS = 2


def _text(img, text, org, colour, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw(img, polygons, scores, scale=1.0):
    """Numbered outlines, a light fill and the confidence on img (BGR), in place. Polygons in
    img px / scale."""
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


def cutout(image, poly, size=TILE, zoom=ZOOM):
    """Only the pixels inside poly (image px), turned so the body's long axis is horizontal,
    enlarged by zoom and centred in a size tile; black elsewhere."""
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
    return (pixels * keep).astype(np.uint8)


def cutout_panel(image, polygons, scores, height):
    """A panel of numbered cut-outs, PANEL_COLUMNS wide and height tall."""
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
        _text(panel, f"#{i + 1}  {score:.2f}", (x, y + 16), colour)
        tile = cutout(image, poly)
        panel[y + 22:y + 22 + TILE[1], x:x + TILE[0]] = tile
        cv2.rectangle(panel, (x - 1, y + 21), (x + TILE[0], y + 22 + TILE[1]), colour, 1)
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.live_detect",
                                     description="Run the model live on the camera and draw the body outlines.")
    parser.add_argument("--weights", type=Path, default=config.ROOT / "runs" / "synth_only" / "weights" / "best.pt")
    parser.add_argument("--fps", type=float, default=3.0, help="detections per second (3)")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0.5)")
    args = parser.parse_args(argv)
    if not args.weights.exists():
        parser.error(f"no weights at {args.weights}; train first, e.g. python -m robot_offline.train_synth")

    from ultralytics import YOLO

    ds = config.load("dataset")
    settings = rectify.Settings.from_config(ds)
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    print(f"Loading {args.weights} ...")
    model = YOLO(str(args.weights))
    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    cap = open_camera()
    print(__doc__.split("Keys:", 1)[1].rstrip())

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cal, status, last_find = None, "looking for the plate", 0.0
    top_down, masked, conf = False, False, args.conf
    polygons, scores, view, shown = [], [], None, None
    next_run, latency, n_runs, t_start = 0.0, 0.0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()                             # keep reading so the newest frame is used
        if not ok:
            continue
        now = time.perf_counter()
        if cal is None or now - last_find > REFIND_SECONDS:
            last_find = now
            if cal is None:
                cal, status = find_plate(frame, settings)
            else:
                try:
                    cal, _ = rectify.recalibrate(frame, cal)
                except rectify.PlateNotFound:
                    pass                                   # a hand over an edge: keep the last calibration

        if now >= next_run:
            next_run = now + 1.0 / args.fps
            t0 = time.perf_counter()
            if cal is not None:
                rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
                model_view = rectify.mask_outside_plate(rect, cal.settings)
                tiles = tiling.tiles(work_area, (model_view.shape[1], model_view.shape[0]),
                                     tiles_cfg["size"], tiles_cfg["overlap"])
                polygons, scores = tiling.predict_frame(model, model_view, tiles, conf=conf)
                if top_down:
                    view = (model_view if masked else rectify.draw_grid(rect, cal.settings)).copy()
                    draw(view, polygons, scores)
                else:
                    view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                    cv2.polylines(view, [np.round(rectify.plate_outline_px(cal) * CAMERA_SCALE).astype(np.int32)],
                                  True, (255, 255, 255), 1, cv2.LINE_AA)
                    draw(view, [rectify.to_full(p, cal) for p in polygons], scores, CAMERA_SCALE)
                view = np.hstack([view, cutout_panel(rect, polygons, scores, view.shape[0])])
            else:
                view = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
                view = np.hstack([view, cutout_panel(view, [], [], view.shape[0])])
            latency = time.perf_counter() - t0
            n_runs += 1
            rate = n_runs / max(time.perf_counter() - t_start, 1e-6)
            mode = ("top-down, as the model sees it" if masked else "top-down") if top_down else "camera view"
            label(view, [f"{len(polygons)} resistor bodies   conf >= {conf:.2f}   {mode}",
                         f"{rate:.1f} detections/s, {latency * 1000:.0f} ms each   {status}",
                         "q quit  v view  m model's input  +/- confidence  c find plate  s save"])
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
            found, status = find_plate(frame, settings)
            cal = found or cal
            last_find = time.perf_counter()
        elif key == ord("s") and view is not None:
            out = config.DATA_DIR / "live"
            out.mkdir(parents=True, exist_ok=True)
            path = out / datetime.now().strftime("detect_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(path), view)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

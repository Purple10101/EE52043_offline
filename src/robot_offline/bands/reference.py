"""Reference frames: resistors of one known value on the plate, for checking the band reader.

    python -m robot_offline.bands.reference 3k            # capture a frame of 3 kΩ resistors
    python -m robot_offline.bands.reference --previews    # redraw every frame's preview

Each capture is saved in data/reference/<date>/ (--folder to choose another):

    NNNN.png               the raw camera frame
    NNNN_rectify.json      its plate calibration
    previews/NNNN.jpg      the camera view with the bodies found outlined, and their cut-outs
    reference.json         one entry per frame: file, value_ohm, bodies found, scores, polygons
                           (rectified px)

The bodies are found with the segmentation model. A frame whose plate cannot be found on its own is
recalibrated near the folder's previous frame.
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, rectify, tiling

DEFAULT_WEIGHTS = config.ROOT / "models" / "synth_only_v1" / "best.pt"
PREVIEW_SCALE = 0.6         # of the live detection view's size


def load_index(folder):
    path = folder / "reference.json"
    return json.loads(path.read_text()) if path.exists() else []


def find_bodies(frame, cal, model, conf=0.5):
    """(rectified image, white-balance gains, polygons in rectified px, scores) for a raw frame."""
    ds = config.load("dataset")
    area, tc = ds["work_area"], ds["tiles"]
    rect, gains = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
    view = rectify.mask_outside_plate(rect, cal.settings)
    tiles = tiling.tiles((area["x0"], area["y0"], area["width"], area["height"]), (view.shape[1], view.shape[0]),
                         tc["size"], tc["overlap"])
    polygons, scores = tiling.predict_frame(model, view, tiles, conf=conf)
    return rect, gains, polygons, scores


def preview(frame, cal, rect, polygons, scores):
    """The camera view with the bodies outlined, beside the cut-out of each body."""
    from robot_offline.live_detect import CAMERA_SCALE, cutout_panel, draw

    shown = cv2.resize(frame, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
    draw(shown, [rectify.to_full(np.asarray(p), cal) for p in polygons], scores, CAMERA_SCALE)
    shown = np.hstack([shown, cutout_panel(rect, [np.asarray(p) for p in polygons], scores, shown.shape[0])])
    return cv2.resize(shown, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, interpolation=cv2.INTER_AREA)


def save(folder, frame, cal, value_ohm, rect, polygons, scores, sides=()):
    """Saves a frame, its calibration and preview, and appends its entry to reference.json."""
    folder.mkdir(parents=True, exist_ok=True)
    index = load_index(folder)
    stem = f"{len(index) + 1:04d}"
    cv2.imwrite(str(folder / f"{stem}.png"), frame)
    cal.save(folder / f"{stem}_rectify.json")
    (folder / "previews").mkdir(exist_ok=True)
    cv2.imwrite(str(folder / "previews" / f"{stem}.jpg"), preview(frame, cal, rect, polygons, scores))
    entry = {"file": f"{stem}.png", "value_ohm": float(value_ohm),
             "captured": datetime.now().isoformat(timespec="seconds"),
             "bodies": len(polygons), "scores": [round(float(s), 3) for s in scores],
             "polygons_rect": [np.round(p, 2).tolist() for p in polygons],
             "plate_edges_on_line": [round(float(s), 2) for s, _ in sides]}
    index.append(entry)
    (folder / "reference.json").write_text(json.dumps(index, indent=1))
    return entry


def _calibration(frame, folder):
    try:
        return rectify.calibrate(frame, rectify.Settings.from_config())
    except rectify.PlateNotFound:
        previous = sorted(folder.glob("*_rectify.json"))
        if not previous:
            raise
        return rectify.recalibrate(frame, rectify.Calibration.load(previous[-1]))


def capture(value_ohm, folder, weights=DEFAULT_WEIGHTS, conf=0.5):
    from ultralytics import YOLO

    from robot_offline.capture import open_camera

    cap = open_camera()
    for _ in range(5):          # let the exposure settle
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("no frame from the camera")
    cal, sides = _calibration(frame, folder)
    rect, _, polygons, scores = find_bodies(frame, cal, YOLO(str(weights)), conf)
    return save(folder, frame, cal, value_ohm, rect, polygons, scores, sides)


def redraw_previews(folder):
    """Rebuilds previews/ from the saved frames and the bodies recorded in reference.json."""
    (folder / "previews").mkdir(parents=True, exist_ok=True)
    for entry in load_index(folder):
        stem = Path(entry["file"]).stem
        frame = cv2.imread(str(folder / entry["file"]))
        cal = rectify.Calibration.load(folder / f"{stem}_rectify.json")
        rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
        cv2.imwrite(str(folder / "previews" / f"{stem}.jpg"),
                    preview(frame, cal, rect, entry["polygons_rect"], entry["scores"]))
        print(f"  {folder / 'previews' / f'{stem}.jpg'}")


def main(argv=None):
    from robot_offline.capture import VALUE_RANGE_OHM, format_ohms, parse_ohms

    parser = argparse.ArgumentParser(prog="python -m robot_offline.bands.reference",
                                     description="Capture a reference frame of resistors of one known value.")
    parser.add_argument("value", nargs="?", help="the resistors' value, e.g. 1k, 4k7, 470, 47R")
    parser.add_argument("--folder", type=Path, default=config.DATA_DIR / "reference" / date.today().isoformat())
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--previews", action="store_true", help="redraw the folder's previews and exit")
    args = parser.parse_args(argv)

    if args.previews:
        redraw_previews(args.folder)
        return
    if args.value is None:
        parser.error("give the resistors' value, or --previews")
    value = parse_ohms(args.value)
    if not VALUE_RANGE_OHM[0] <= value <= VALUE_RANGE_OHM[1]:
        print(f"warning: {format_ohms(value)} is outside the {format_ohms(VALUE_RANGE_OHM[0])}"
              f"-{format_ohms(VALUE_RANGE_OHM[1])} range the band reader covers")
    entry = capture(value, args.folder, args.weights, args.conf)
    stem = Path(entry["file"]).stem
    print(f"saved {args.folder / entry['file']}: {format_ohms(value)}, {entry['bodies']} bodies found, "
          f"scores {[round(s, 2) for s in entry['scores']]}; preview {args.folder / 'previews' / f'{stem}.jpg'}")


if __name__ == "__main__":
    main()

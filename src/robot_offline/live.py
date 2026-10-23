"""Live straight-down view of the pick plate, for setting up and checking the rig.

Opens the camera with configs/camera.toml, finds the plate in the first frame (or uses a saved
calibration) and shows every frame rectified and white-balanced, with a 10 mm grid. While the plate
can't be found (a hand over an edge, say) it shows the raw view and keeps looking.

    python -m robot_offline.live                          # find the plate in the first frame
    python -m robot_offline.live --calibration <file>     # use a session's saved calibration

Keys:  q / Esc  quit       g  grid on or off       r  raw camera view or rectified
       c  find the plate again (after moving the camera or plate)
       s  save the raw frame (PNG) and the rectified image to --save-dir
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from robot_offline import config, rectify
from robot_offline.capture import RETRY_FRAMES, find_plate, label, open_camera, show, view_of

WINDOW = "robot_offline live"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live straight-down view of the pick plate.")
    parser.add_argument("--calibration", help="saved calibration (.json) instead of finding the plate")
    parser.add_argument("--save-dir", default=str(config.DATA_DIR / "live"),
                        help="where `s` saves frames (default data/live)")
    args = parser.parse_args(argv)

    print("Opening the camera (a few seconds while exposure and white balance settle)...")
    cap = open_camera()
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("the camera gave no frame")

    if args.calibration:
        cal, status = rectify.Calibration.load(args.calibration), f"calibration: {Path(args.calibration).name}"
    else:
        cal, status = find_plate(frame, rectify.Settings.from_config())
    print(status)
    print(__doc__.split("Keys:", 1)[1].rstrip())

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    show_grid, show_raw, shown_size = True, False, None
    times, count = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        count += 1
        if cal is None and count % RETRY_FRAMES == 0:
            cal, status = find_plate(frame, rectify.Settings.from_config())
        times = (times + [time.perf_counter()])[-10:]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

        view = view_of(frame, cal, show_raw, show_grid)
        if show_raw or cal is None:
            mode = "raw"
        else:
            mode = "rectified, 10 px/mm" + (", 10 mm grid" if show_grid else "")
        label(view, [f"{mode}   {fps:.1f} fps", status,
                     "q quit  g grid  r raw/rectified  c find plate  s save"])
        shown_size = show(view, shown_size, WINDOW)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("g"):
            show_grid = not show_grid
        elif key == ord("r"):
            show_raw = not show_raw
        elif key == ord("c"):
            found, status = find_plate(frame, cal.settings if cal else rectify.Settings.from_config())
            cal = found or cal
            print(status)
        elif key == ord("s"):
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = out / datetime.now().strftime("live_%Y%m%d_%H%M%S")
            cv2.imwrite(f"{stem}.png", frame)
            if cal is not None:
                rect, _ = rectify.white_balance(rectify.rectify(frame, cal), cal.settings)
                cv2.imwrite(f"{stem}_rectified.png", rect)
            status = f"saved {stem.name}"
            print(f"Saved {stem}.png" + (" and _rectified.png" if cal is not None else ""))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

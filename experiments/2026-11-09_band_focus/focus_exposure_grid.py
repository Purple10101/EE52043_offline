"""Does a band read better at its own best focus and exposure? Every body on the plate, read over a grid.

    python experiments/2026-11-09_band_focus/focus_exposure_grid.py experiments/2026-11-09_band_focus

Finds the plate (plate_search.find_now), then for each plate level in LEVELS (exposure stepped to it by
ExposureControl) and each focus in FOCUSES: sets the focus, lets it settle, averages AVERAGE frames,
segments that capture (seg_grey_v2, as live), matches the bodies to the first capture's by centre, and
for each body cuts the band crop (crops.cut, as read_plate) and records its sharpness (mean squared Sobel
of the crop, grey) and the learned reader's slot log-probabilities and Reading. Focus goes back to the
config's at the end. Writes grid.json, crops.png (each body: rows = levels, columns = focuses) and one
overview jpg; frames themselves aren't kept (36 MB each).
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.post.bands import crops, learned
from robot_offline.pre import exposure, plate_search, rectify
from robot_offline.pre.camera import open_camera

FOCUSES = (290, 298, 306, 314, 322)
LEVELS = ((170, 185), (205, 225), (230, 242))
AVERAGE = 8
SETTLE = 12                 # frames read after a focus change (the sweep used 10)
MATCH_PX = 30               # rectified px: same body across captures
SEG = config.ROOT / "models/seg_grey_v2/best.pt"
READER = config.ROOT / "runs/band_reader_final/band_reader.pt"


def mean_frame(cap, n=AVERAGE):
    stack = []
    while len(stack) < n:
        ok, f = cap.read()
        if ok:
            stack.append(f.astype(np.float32))
    return np.clip(np.mean(stack, axis=0) + 0.5, 0, 255).astype(np.uint8)


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2))


def settle_level(cap, control, cal, band, timeout_s=20):
    control.aim(*band)
    end = time.time() + timeout_s
    while time.time() < end:
        ok, f = cap.read()
        if ok and not control.meter(f, cal) and control.wait == 0 and control.pending is None:
            break
    return control.exposure, control.gain


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    camera = config.load("camera")
    ds = config.load("dataset")
    a, tc = ds["work_area"], ds["tiles"]
    seg, net = YOLO(str(SEG)), learned.load(READER)
    grey = tiling.wants_grey(SEG)
    config_focus = camera["camera"]["focus"]
    cap = open_camera(camera)
    records, bodies, sheet = [], None, {}
    try:
        cal = plate_search.find_now(cap, camera=camera)
        from robot_offline.pre.exposure import ExposureControl
        control = ExposureControl(camera["exposure_control"], cap.get(cv2.CAP_PROP_EXPOSURE), cap.get(cv2.CAP_PROP_GAIN), cap)
        for band in LEVELS:
            exp, gain = settle_level(cap, control, cal, band)
            for focus in FOCUSES:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                for _ in range(SETTLE):
                    cap.read()
                frame = mean_frame(cap)
                level = exposure.plate_level(frame, cal)[0]
                plain = rectify.rectify(frame, cal)
                view, _ = rectify.model_input(frame, cal, plain)
                t = tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (view.shape[1], view.shape[0]),
                                 tc["size"], tc["overlap"])
                polys, scores = tiling.predict_frame(seg, view, t, conf=0.3, grey=grey)
                if bodies is None:          # the first capture names the bodies
                    bodies = [p.mean(0).tolist() for p in polys]
                    cv2.imwrite(str(out / "overview.jpg"), cv2.resize(view, (900, 900)))
                _, gains = rectify.white_balance(plain, cal.settings)
                bal = np.clip(plain.astype(np.float32) * gains, 0, 255).astype(np.uint8)
                white = float(np.percentile(cv2.cvtColor(bal, cv2.COLOR_BGR2GRAY)[150:1050, 150:1050], 70))
                for p, s in zip(polys, scores):
                    d = [np.linalg.norm(p.mean(0) - b) for b in bodies]
                    if min(d) > MATCH_PX:
                        continue
                    k = int(np.argmin(d))
                    crop = crops.cut(frame, cal, p, gains).image
                    lp = learned.log_probs(net, crop, white)
                    r = learned.decode(lp)
                    records.append(dict(body=k, centre=bodies[k], focus=focus, band=list(band), exposure=exp, gain=gain,
                                        plate=level, detection=float(s), sharpness=sharpness(crop),
                                        best=r.best, value=r.value, confidence=r.confidence, colours=list(r.colours),
                                        log_probs=np.round(lp, 4).tolist()))
                    sheet[(k, band, focus)] = cv2.resize(crop, (240, 80), interpolation=cv2.INTER_AREA)
                    # Full size, for the figures: the crop the reader got, and the raw frame round the body.
                    name = f"body{k}_plate{band[0]}_focus{focus}"
                    (out / "crops").mkdir(exist_ok=True)
                    (out / "raw").mkdir(exist_ok=True)
                    cv2.imwrite(str(out / "crops" / f"{name}.png"), crop)
                    cx, cy = np.round(rectify.to_full(p.mean(0)[None], cal)[0]).astype(int)
                    cv2.imwrite(str(out / "raw" / f"{name}.png"), frame[max(cy - 90, 0):cy + 90, max(cx - 150, 0):cx + 150])
                print(f"level {band} exp {exp:g} gain {gain:g} plate {level:.0f} focus {focus}: "
                      + "  ".join(f"#{r['body']} {learned.ohms_text(r['best'])} {r['confidence']:.2f} s{r['sharpness']:.0f}"
                                  for r in records if r["focus"] == focus and r["band"] == list(band)), flush=True)
    finally:
        cap.set(cv2.CAP_PROP_FOCUS, config_focus)
        cap.release()
    json.dump(dict(focuses=FOCUSES, levels=LEVELS, config_focus=config_focus, bodies=bodies, records=records),
              open(out / "grid.json", "w"), indent=1)
    rows = []
    for k in range(len(bodies)):
        for band in LEVELS:
            cells = [sheet.get((k, band, f), np.zeros((80, 240, 3), np.uint8)) for f in FOCUSES]
            rows.append(np.hstack(cells))
        rows.append(np.full((10, 240 * len(FOCUSES), 3), 255, np.uint8))
    cv2.imwrite(str(out / "crops.png"), np.vstack(rows))
    print("bodies:", [[round(v) for v in b] for b in bodies])


if __name__ == "__main__":
    main()

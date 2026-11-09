"""Classical OpenCV segmentation against YOLO (seg_grey_v2) on the labelled real test frames.

    python experiments/2026-11-09_opencv_vs_yolo/compare.py

Every method gets the same input as the model: the rectified frame, white-balanced with the plate at
grey 200 and masked to the plate (rectify.model_image). The classical methods share one post-process:
- an opening with an elliptical kernel, to cut the leads off the bodies
- a small closing, to fill specks
- connected components between 3 mm² and 400 mm²
- each component's outer contour as its outline

The methods differ in how they make the first mask:

| Method | Mask |
|---|---|
| fixed | grey darker than the plate by more than T |
| otsu | Otsu's threshold over the plate's grey, no parameter |
| adaptive | darker than the local Gaussian mean (block B px) by more than C |
| lab | Lab colour distance from the plate's own colour (median of its brighter half) above T |
| bgsub | Lab distance from the session's empty-plate frame above T: the project's original autolabel |

`bgsub` needs an empty-plate frame, which only 2026-10-23_a and _b have, so it is scored on those
30 frames, with every method rescored on the same subset.

Parameters are tuned on the test frames themselves, by grid search for the best F1 per method.
That is an optimistic best case for OpenCV, which in use would be tuned on other frames. YOLO is
untuned: seg_grey_v2 at conf 0.5, as in use.

Scoring is tools/evaluate.score_frame (one-to-one mask IoU >= 0.5), against the rules-version-2
labels: resistors, plus the screws and bolt. Only frames labelled "objects" are used (every object
outlined, docs/labelling.md). Writes results.json, results.md and the figures to this folder.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, manifest
from robot_offline.inf import tiling
from robot_offline.pre import rectify
from robot_offline.tools import evaluate

HERE = Path(__file__).parent
WEIGHTS = config.ROOT / "models" / "seg_grey_v2" / "best.pt"
MIN_AREA_MM2, MAX_AREA_MM2 = 3.0, 400.0
GRID = {
    "fixed": [{"t": t, "k": k} for t in (10, 15, 20, 30, 40, 55, 70) for k in (7, 9, 11, 13, 15)],
    "otsu": [{"k": k} for k in (7, 9, 11, 13, 15)],
    "adaptive": [{"b": b, "c": c, "k": k} for b in (51, 101, 201) for c in (5, 10, 15, 25) for k in (7, 9, 11, 13, 15)],
    "lab": [{"t": t, "k": k} for t in (10, 15, 20, 25, 30, 40, 55) for k in (7, 9, 11, 13, 15)],
    "bgsub": [{"t": t, "k": k} for t in (10, 15, 20, 25, 30, 40, 55) for k in (7, 9, 11, 13, 15)],
}
# Frames to show, by condition: (title, image)
SHOWN = [
    ("spread out", "captures/2026-10-23_b/rectified/0003.png"),
    ("parallel and close", "captures/2026-10-23_a/rectified/0006.png"),
    ("screws, among resistors", "captures/2026-10-23_a/rectified/0012.png"),
    ("hard light, cast shadows", "captures/2026-11-06_a/rectified/0005.png"),
    ("dark exposure", "captures/2026-11-06_a/rectified/0001.png"),
    ("empty plate, arm and servo, in view", "live/plate_20261105_203513_rectified.png"),
]


def condition(sample):
    s, n = sample["session"], int(Path(sample["image"]).stem.split("_")[-1]) if sample["image"][-8:-4].isdigit() else 0
    if s == "2026-11-06_a":
        return "hard light, 12 exposures"
    if s == "2026-11-05_live":
        return "empty plate, arm in view"
    if s == "2026-10-23_a" and n >= 10:
        return "screws among resistors"
    return "resistors only, room light"


class Frame:
    """One test frame, with what every method needs computed once."""

    def __init__(self, sample, st, cals):
        self.sample = sample
        raw = cv2.imread(str(config.DATA_DIR / sample["image"]))
        self.image = rectify.model_image(sample, raw, st)
        self.plate = rectify.plate_mask(st)
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        p = lab[self.plate].reshape(-1, 3)
        bright = p[p[:, 0] >= np.median(p[:, 0])]
        self.lab_dist = np.linalg.norm(lab - np.median(bright, axis=0), axis=2)
        self.plate_grey = float(np.median(self.grey[self.plate]))
        self.bg_dist = None
        cal = cals.get(sample["session"])
        if cal is not None:
            empty_raw = cv2.imread(str(config.DATA_DIR / "backgrounds" / sample["session"] / "0001.png"))
            empty, _ = rectify.white_balance(rectify.rectify(empty_raw, cal), st, rectify.MODEL_PLATE_LEVEL)
            self.bg_dist = np.linalg.norm(lab - cv2.cvtColor(empty, cv2.COLOR_BGR2LAB).astype(np.float32), axis=2)
        self.labels = manifest.polygons(sample)
        self.kinds = [i.get("kind", "resistor") for i in sample["instances"]]


def first_mask(f, method, prm):
    if method == "fixed":
        m = f.grey < f.plate_grey - prm["t"]
    elif method == "otsu":
        vals = f.grey[f.plate].astype(np.uint8)
        t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = f.grey < t
    elif method == "adaptive":
        mean = cv2.GaussianBlur(f.grey, (0, 0), prm["b"] / 6)
        m = f.grey < mean - prm["c"]
    elif method == "lab":
        m = f.lab_dist > prm["t"]
    else:
        m = f.bg_dist > prm["t"]
    return (m & f.plate).astype(np.uint8)


def segment(f, method, prm, px_per_mm):
    """Outlines (N×2 px) from one classical method."""
    m = first_mask(f, method, prm)
    k = prm["k"]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    lo, hi = MIN_AREA_MM2 * px_per_mm ** 2, MAX_AREA_MM2 * px_per_mm ** 2
    out = []
    for i in range(1, n):
        if not lo <= stats[i, cv2.CC_STAT_AREA] <= hi:
            continue
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(c) >= 3:
            out.append(c)
    return out


def score(frames, preds, px_per_mm):
    """Totals over frames: resistor and screw recall, false detections, merges, splits, geometry errors."""
    res_hit = res_n = scr_hit = scr_n = false = merges = splits = 0
    ious, centre, yaw = [], [], []
    per_cond = {}
    for f, p in zip(frames, preds):
        r = evaluate.score_frame(f.labels, p, px_per_mm)
        hit = {m["label_index"] for m in r["matched"]}
        rh = sum(1 for i, k in enumerate(f.kinds) if k == "resistor" and i in hit)
        rn = sum(1 for k in f.kinds if k == "resistor")
        res_hit += rh
        res_n += rn
        scr_hit += sum(1 for i, k in enumerate(f.kinds) if k != "resistor" and i in hit)
        scr_n += sum(1 for k in f.kinds if k != "resistor")
        false += r["false_detections"]
        merges += r["merges"]
        splits += r["splits"]
        for m in r["matched"]:
            if f.kinds[m["label_index"]] == "resistor":
                ious.append(m["iou"])
                centre.append(m["centre_mm"])
                yaw.append(m["yaw_deg"])
        c = per_cond.setdefault(condition(f.sample), [0, 0, 0, 0])
        c[0] += rh
        c[1] += rn
        c[2] += r["false_detections"]
        c[3] += 1
    tp = res_hit + scr_hit
    n_pred = tp + false
    n_gt = res_n + scr_n
    pct = lambda a, q: round(float(np.percentile(a, q)), 3) if a else None
    return {
        "resistors_found": res_hit, "resistors": res_n, "screws_found": scr_hit, "screws": scr_n,
        "false_detections": false, "merges": merges, "splits": splits,
        "precision": round(tp / n_pred, 3) if n_pred else None, "recall": round(tp / n_gt, 3) if n_gt else None,
        "f1": round(2 * tp / (n_gt + n_pred), 3) if n_gt + n_pred else None,
        "resistor_iou_median": pct(ious, 50), "centre_mm_p95": pct(centre, 95), "yaw_deg_p95": pct(yaw, 95),
        "by_condition": per_cond,
    }


def draw(f, preds, title):
    img = f.image.copy()
    r = evaluate.score_frame(f.labels, preds, 10.0)
    hit_pred = {m["prediction_index"] for m in r["matched"]}
    hit_label = {m["label_index"] for m in r["matched"]}
    for i, q in enumerate(f.labels):
        if i not in hit_label:                                 # missed: blue
            cv2.polylines(img, [np.round(q).astype(np.int32)], True, (255, 120, 0), 3, cv2.LINE_AA)
    for j, q in enumerate(preds):                              # found: green; false: red
        cv2.polylines(img, [np.round(q).astype(np.int32)], True, (0, 200, 0) if j in hit_pred else (0, 0, 255), 2,
                      cv2.LINE_AA)
    img = cv2.resize(img[90:1110, 90:1110], (340, 340), interpolation=cv2.INTER_AREA)
    for col, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
        cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, w, cv2.LINE_AA)
    return img


def main():
    import sys
    from ultralytics import YOLO

    figure_only = "--figure-only" in sys.argv     # redraw the figure with the best settings in results.json

    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    s = st.px_per_mm
    samples = [x for x in manifest.load(ds["labels"]["rules_version"])["samples"]
               if x["session"] in ds["splits"]["test_sessions"] and x.get("labelled") == "objects"]
    cals = {name: rectify.Calibration.load(config.DATA_DIR / "captures" / name / "rectify.json")
            for name in ("2026-10-23_a", "2026-10-23_b")}
    print(f"{len(samples)} frames", flush=True)
    frames = [Frame(x, st, cals) for x in samples]
    with_bg = [i for i, f in enumerate(frames) if f.bg_dist is not None]

    # YOLO, as in use.
    model = YOLO(str(WEIGHTS))
    a = ds["work_area"]
    tiles = tiling.tiles((a["x0"], a["y0"], a["width"], a["height"]), (st.size[0], st.size[1]), 1024, 128)
    yolo, t0 = [], time.perf_counter()
    for f in frames:
        yolo.append(tiling.predict_frame(model, f.image, tiles, conf=0.5, grey=True)[0])
    yolo_s = (time.perf_counter() - t0) / len(frames)

    if figure_only:
        saved = json.loads((HERE / "results.json").read_text())
        best_preds = {}
        for method in GRID:
            subset = with_bg if method == "bgsub" else range(len(frames))
            best_preds[method] = {i: segment(frames[i], method, saved[method]["params"], s) for i in subset}
        figure(frames, best_preds, yolo)
        return

    results = {"frames": len(frames), "frames_with_empty_plate": len(with_bg),
               "yolo": {**score(frames, yolo, s), "params": "conf 0.5", "seconds_per_frame": round(yolo_s, 3)},
               "yolo_on_bg_subset": score([frames[i] for i in with_bg], [yolo[i] for i in with_bg], s)}
    best_preds = {}
    for method, grid in GRID.items():
        subset = with_bg if method == "bgsub" else range(len(frames))
        fs = [frames[i] for i in subset]
        best = None
        for prm in grid:
            preds = [segment(f, method, prm, s) for f in fs]
            sc = score(fs, preds, s)
            if best is None or (sc["f1"] or 0) > (best[0]["f1"] or 0):
                best = (sc, prm, preds)
        sc, prm, preds = best
        t0 = time.perf_counter()
        for f in fs:
            segment(f, method, prm, s)
        sc.update(params=prm, seconds_per_frame=round((time.perf_counter() - t0) / len(fs), 3), configs_tried=len(grid))
        results[method] = sc
        if method != "bgsub":
            results[method + "_on_bg_subset"] = score([frames[i] for i in with_bg], [preds[i] for i in with_bg], s)
        best_preds[method] = dict(zip(subset, preds))
        print(f"{method:9s} best {prm}: resistors {sc['resistors_found']}/{sc['resistors']}, screws "
              f"{sc['screws_found']}/{sc['screws']}, false {sc['false_detections']}, F1 {sc['f1']}", flush=True)
    print(f"yolo     : resistors {results['yolo']['resistors_found']}/{results['yolo']['resistors']}, screws "
          f"{results['yolo']['screws_found']}/{results['yolo']['screws']}, false {results['yolo']['false_detections']}, "
          f"F1 {results['yolo']['f1']}")
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    figure(frames, best_preds, yolo)


def figure(frames, best_preds, yolo):
    """figure_methods.jpg: conditions down, methods across."""
    index = {f.sample["image"]: i for i, f in enumerate(frames)}
    rows = []
    for title, key in SHOWN:
        key = key if key in index else next((k for k in index if k.endswith(Path(key).name)), None)
        if key is None:
            continue
        i = index[key]
        cells = [draw(frames[i], best_preds[m].get(i, []), m) if (m != "bgsub" or i in best_preds[m])
                 else np.full((340, 340, 3), 235, np.uint8) for m in ("fixed", "otsu", "adaptive", "lab", "bgsub")]
        cells.append(draw(frames[i], yolo[i], "YOLO seg_grey_v2"))
        head = np.full((340, 150, 3), 255, np.uint8)
        for k, word in enumerate(title.split(", ")):
            cv2.putText(head, word, (6, 160 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([head] + cells))
    cv2.imwrite(str(HERE / "figure_methods.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()

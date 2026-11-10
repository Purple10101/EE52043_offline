"""Live test: seg_grey_v2 and seg_objects_v1 on the same camera frames, one placement per call.

    python experiments/2026-11-09_seg_objects_v1_test/live_compare.py <tag> [--runs 15]
    python experiments/2026-11-09_seg_objects_v1_test/live_compare.py <tag> --replay raw.png rectify.json

Opens the camera as live_detect does, finds the plate (plate_search.find_now, band-reading level as in
experiments/2026-11-09_front_back), then takes N frames a few apart and runs both models on each, at
steady.FLOOR (conf 0.05, what live_detect keeps). --replay runs on one saved frame instead, to check
the script without the rig.

Per model, the objects found are matched across runs by centroid (within 2 mm). Per object: its score
in every run (0 when absent), runs at 0.5 or more (live_detect's threshold), the outline's median
length and width, centroid jitter (RMS, mm), yaw SD (deg), and its gap to the nearest plate edge (mm,
the plate's straight sides; negative when it overhangs). The two models' objects are then paired by
centroid so each placement reads as one table.

Writes into live/: <tag>.json, <tag>_raw.png (the last run's camera frame, LFS; also the first and
middle runs' as <tag>_runNN_raw.png, so a fix can be rerun offline on real frames), <tag>_rectify.json,
<tag>_model.png (the model input) and <tag>_overlay.png (both models side by side; green = found in
at least 80% of runs at 0.5, red = less, labelled with median score and runs at 0.5).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.inf.steady import FLOOR
from robot_offline.post.bands import learned
from robot_offline.pre import plate_search, rectify
from robot_offline.tools import evaluate

HERE = Path(__file__).parent
MODELS = ["seg_grey_v2", "seg_objects_v1"]
FOUND = 0.5            # live_detect's threshold
MATCH_MM = 2.0


def capture(args, st):
    """[(raw frame)] * runs and the Calibration, from the camera or a replayed frame."""
    if args.replay:
        frame = cv2.imread(str(args.replay[0]))
        return [frame] * args.runs, rectify.Calibration.load(args.replay[1]), {"replay": [str(p) for p in args.replay]}
    from robot_offline.pre.camera import open_camera
    camera = config.load("camera")
    cap = open_camera(camera)
    if args.focus is not None:          # this capture only; camera.toml is unchanged
        cap.set(cv2.CAP_PROP_FOCUS, args.focus)
        for _ in range(10):
            cap.read()
    t0 = time.perf_counter()
    level = learned.READING_LEVEL if args.level == "reading" else None
    cal = plate_search.find_now(cap, st, camera=camera, level=level)
    print(f"plate found in {time.perf_counter() - t0:.0f} s", flush=True)
    frames = []
    while len(frames) < args.runs:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            for _ in range(3):          # a few frames apart, as live_detect at 3 fps
                cap.read()
    info = {"focus": cap.get(cv2.CAP_PROP_FOCUS), "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "gain": cap.get(cv2.CAP_PROP_GAIN), "level": args.level}
    if args.focus is not None:          # put the configured focus back before letting go of the camera
        cap.set(cv2.CAP_PROP_FOCUS, camera["camera"]["focus"])
        for _ in range(7):
            cap.read()
    cap.release()
    return frames, cal, info


def track(runs, st):
    """Objects across runs by centroid: [{"runs": {run: (score, poly)}}]."""
    objects = []
    for r, dets in enumerate(runs):
        for poly, score in dets:
            c = poly.mean(0)
            hit = next((o for o in objects if np.linalg.norm(o["c"] - c) < MATCH_MM * st.px_per_mm), None)
            if hit is None:
                hit = {"c": c, "runs": {}}
                objects.append(hit)
            if r not in hit["runs"] or hit["runs"][r][0] < score:
                hit["runs"][r] = (score, poly)
    return objects


def describe(obj, n_runs, st, plate_mm):
    s = st.px_per_mm
    scores = [obj["runs"].get(r, (0.0, None))[0] for r in range(n_runs)]
    polys = [v[1] for v in obj["runs"].values()]
    geo = np.array([evaluate.body_geometry(p) for p in polys])
    c = geo[:, :2]
    yaw = np.radians(geo[:, 4]) * 2
    mean = np.angle(np.exp(1j * yaw).mean())
    best = max(obj["runs"].values(), key=lambda v: v[0])[1]
    body_mm = Polygon(rectify.rect_to_mm(best, st)).buffer(0)
    gap = plate_mm.exterior.distance(body_mm) * (1 if plate_mm.contains(body_mm) else -1)
    centre_mm = rectify.rect_to_mm(c.mean(0), st)[0]
    return {
        "x_mm": round(float(centre_mm[0]), 1), "y_mm": round(float(centre_mm[1]), 1),
        "edge_gap_mm": round(float(gap), 2),
        "scores": [round(v, 3) for v in scores],
        "median_score": float(np.median(scores)), "runs_found": int(sum(v >= FOUND for v in scores)),
        "length_mm": float(np.median(geo[:, 2]) / s), "width_mm": float(np.median(geo[:, 3]) / s),
        "yaw_deg": float(np.degrees(mean) / 2 % 180),
        "centre_jitter_mm": float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()) / s),
        "yaw_sd_deg": float(np.degrees(np.sqrt((np.angle(np.exp(1j * (yaw - mean))) ** 2).mean())) / 2),
        "poly": np.round(best, 1).tolist(),
    }


def draw(image, objects, title):
    over = image.copy()
    for i, o in enumerate(objects):
        q = np.round(np.asarray(o["poly"])).astype(np.int32)
        ok = o["runs_found"] >= 0.8 * len(o["scores"])
        cv2.polylines(over, [q], True, (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
        x, y = q[:, 0].max() + 6, int(q[:, 1].mean())
        for c, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
            cv2.putText(over, f"{o['median_score']:.2f} {o['runs_found']}/{len(o['scores'])}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, w, cv2.LINE_AA)
    for c, w in (((255, 255, 255), 5), ((0, 0, 0), 2)):
        cv2.putText(over, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, w, cv2.LINE_AA)
    return over


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--level", choices=["model", "reading"], default="reading")
    p.add_argument("--replay", nargs=2, type=Path, metavar=("RAW_PNG", "RECTIFY_JSON"))
    p.add_argument("--out", type=Path, default=HERE / "live")
    p.add_argument("--focus", type=int, help="override camera.toml's focus for this capture")
    p.add_argument("--note", default="", help="what the placement is, for the record")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    plate_mm = Polygon(st.plate_corners_mm)
    models = {m: (YOLO(str(config.ROOT / "models" / m / "best.pt")),
                  tiling.wants_grey(config.ROOT / "models" / m / "best.pt")) for m in MODELS}

    frames, cal, info = capture(args, st)
    runs = {m: [] for m in MODELS}
    for k, frame in enumerate(frames):
        plain = rectify.rectify(frame, cal)
        view, _ = rectify.model_input(frame, cal, plain)
        tiles = tiling.tiles(work, (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
        line = []
        for m, (model, grey) in models.items():
            polys, scores = tiling.predict_frame(model, view, tiles, conf=FLOOR, grey=grey)
            runs[m].append([(np.asarray(q, float), float(s)) for q, s in zip(polys, scores)])
            line.append(f"{m} {sum(s >= FOUND for s in scores)}")
        print(f"run {k + 1}: " + ", ".join(line), flush=True)

    result = {"tag": args.tag, "note": args.note, "runs": len(frames), **info, "models": {}}
    for m in MODELS:
        objs = [describe(o, len(frames), st, plate_mm) for o in track(runs[m], st)]
        objs = [o for o in objs if o["runs_found"] >= 1 or max(o["scores"]) >= 0.25]   # drop one-off specks
        result["models"][m] = sorted(objs, key=lambda o: (-o["y_mm"], o["x_mm"]))

    # Pair the two models' objects by centroid.
    pairs, used = [], set()
    a_objs, b_objs = result["models"][MODELS[0]], result["models"][MODELS[1]]
    for j, b in enumerate(b_objs):
        d = [np.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"]) for a in a_objs]
        i = int(np.argmin(d)) if d else None
        pairs.append((i if i is not None and d[i] < MATCH_MM and i not in used else None, j))
        if pairs[-1][0] is not None:
            used.add(i)
    pairs += [(i, None) for i in range(len(a_objs)) if i not in used]
    result["pairs"] = pairs
    (args.out / f"{args.tag}.json").write_text(json.dumps(result, indent=1))

    frame = frames[-1]
    plain = rectify.rectify(frame, cal)
    view, _ = rectify.model_input(frame, cal, plain)
    cv2.imwrite(str(args.out / f"{args.tag}_raw.png"), frame)
    if not args.replay:         # two more raw frames (first, middle), to rerun fixes offline later
        for k in (0, len(frames) // 2):
            cv2.imwrite(str(args.out / f"{args.tag}_run{k + 1:02d}_raw.png"), frames[k])
    cal.save(args.out / f"{args.tag}_rectify.json")
    cv2.imwrite(str(args.out / f"{args.tag}_model.png"), view)
    sides = [draw(plain, result["models"][m], f"{args.tag}: {m}, {len(frames)} runs") for m in MODELS]
    cv2.imwrite(str(args.out / f"{args.tag}_overlay.jpg"), np.hstack(sides), [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"\n{'x mm':>6} {'y mm':>7} {'edge':>5} | {'v2 med':>6} {'found':>6} {'jit mm':>6} | "
          f"{'v1 med':>6} {'found':>6} {'jit mm':>6} {'L x W mm':>11}")
    for i, j in sorted(pairs, key=lambda ij: -(b_objs[ij[1]] if ij[1] is not None else a_objs[ij[0]])["y_mm"]):
        ref = b_objs[j] if j is not None else a_objs[i]
        cols = []
        for o in (a_objs[i] if i is not None else None, b_objs[j] if j is not None else None):
            cols.append(f"{o['median_score']:6.2f} {o['runs_found']:3d}/{len(o['scores']):<2d} {o['centre_jitter_mm']:6.3f}"
                        if o else f"{'-':>6} {'-':>6} {'-':>6}")
        print(f"{ref['x_mm']:6.1f} {ref['y_mm']:7.1f} {ref['edge_gap_mm']:5.1f} | {cols[0]} | {cols[1]} "
              f"{ref['length_mm']:5.1f} x {ref['width_mm']:4.1f}")


if __name__ == "__main__":
    main()

"""Does a parallax-aware plate mask (rectify.plate_mask(..., lift_mm)) fix the far-edge losses, and
what does it cost elsewhere? Both models, at lift 0 (as live) and LIFT mm, conf 0.5.

    python experiments/2026-11-09_seg_objects_v1_test/mask_lift.py

A. Tonight's live captures (live/*_raw.png, 1 to 3 raw frames each, 4 resistors each). The bodies are
   the ones live_compare recorded for the capture (either model, any run); a body counts as found in
   a frame if a prediction at 0.5 or more has its centre within 3 mm. Anything else at 0.5 or more
   is an extra (false) detection.
B. The real test frames (test_sessions), scored as tools/evaluate does.

Writes mask_lift.json and prints both tables.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config, manifest
from robot_offline.inf import tiling
from robot_offline.pre import rectify
from robot_offline.tools import evaluate as ev

HERE = Path(__file__).parent
LIVE = HERE / "live"
MODELS = ["seg_grey_v2", "seg_objects_v1"]
LIFTS = [0.0, 3.0]
CONF = 0.5
NEAR_MM = 3.0


def live_part(models, work, tc):
    rows = []
    for meta in sorted(LIVE.glob("*.json")):
        if meta.name.endswith("_rectify.json"):
            continue
        tag = meta.stem
        rec = json.loads(meta.read_text())
        cal = rectify.Calibration.load(LIVE / f"{tag}_rectify.json")
        st = cal.settings
        bodies = []
        for m in rec["models"]:            # the bodies the live run recorded
            for o in rec["models"][m]:
                if max(o["scores"]) >= CONF and all(np.hypot(o["x_mm"] - b[0], o["y_mm"] - b[1]) > NEAR_MM for b in bodies):
                    bodies.append((o["x_mm"], o["y_mm"], o["edge_gap_mm"]))
        raws = sorted(LIVE.glob(f"{tag}_run*_raw.png")) + [LIVE / f"{tag}_raw.png"]
        for lift in LIFTS:
            for m, (model, grey) in models.items():
                found = np.zeros(len(bodies), int)
                extra = 0
                for raw in raws:
                    view, _ = rectify.model_input(cv2.imread(str(raw)), cal, lift_mm=lift)
                    tiles = tiling.tiles(work, (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
                    polys, scores = tiling.predict_frame(model, view, tiles, conf=CONF, grey=grey)
                    centres = [rectify.rect_to_mm(np.asarray(q).mean(0), st)[0] for q in polys]
                    hit = set()
                    for i, b in enumerate(bodies):
                        near = [j for j, c in enumerate(centres) if np.hypot(c[0] - b[0], c[1] - b[1]) < NEAR_MM]
                        if near:
                            found[i] += 1
                            hit.update(near)
                    extra += len(polys) - len(hit)
                rows.append({"tag": tag, "lift_mm": lift, "model": m, "frames": len(raws),
                             "bodies": [{"x_mm": b[0], "y_mm": b[1], "edge_gap_mm": b[2], "frames_found": int(f)}
                                        for b, f in zip(bodies, found)],
                             "extra_detections": extra})
    return rows


def test_set_part(models, ds):
    rs = rectify.Settings.from_config(ds)
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    samples = [s for s in manifest.load(ds["labels"]["rules_version"])["samples"]
               if s["session"] in ds["splits"]["test_sessions"]]
    out = []
    for lift in LIFTS:
        for m, (model, grey) in models.items():
            results = []
            for s in samples:
                image = rectify.model_image(s, cv2.imread(str(config.DATA_DIR / s["image"])), rs, lift_mm=lift)
                tiles = tiling.tiles(work, (s["width"], s["height"]), tc["size"], tc["overlap"])
                t0 = time.perf_counter()
                polys, scores = tiling.predict_frame(model, image, tiles, conf=CONF,
                                                     epsilon_px=ds["labels"]["simplify_epsilon_px"], grey=grey)
                r = ev.score_frame(manifest.polygons(s), polys, ds["scale"]["px_per_mm"])
                r.update(seconds=time.perf_counter() - t0)
                results.append(r)
            kinds = ev.by_kind(samples, results)
            false, frames = ev.false_detections_where_complete(samples, results)
            rm = [mt for s, r in zip(samples, results) for mt in r["matched"]
                  if s["instances"][mt["label_index"]].get("kind", "resistor") == "resistor"]
            out.append({"lift_mm": lift, "model": m, "resistors": kinds.get("resistor"), "screws": kinds.get("screw"),
                        "false_complete": false, "complete_frames": frames,
                        "resistor_centre_mm_p95": float(np.percentile([mt["centre_mm"] for mt in rm], 95))})
    return out


def main():
    import argparse
    global MODELS, LIFTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--lifts", nargs="+", type=float, default=LIFTS)
    parser.add_argument("--out", default="mask_lift.json")
    a = parser.parse_args()
    MODELS, LIFTS = a.models, a.lifts
    ds = config.load("dataset")
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    models = {}
    for m in MODELS:
        w = config.ROOT / "models" / m / "best.pt"
        models[m] = (YOLO(str(w)), tiling.wants_grey(w))
    live = live_part(models, work, tc)
    test = test_set_part(models, ds)
    (HERE / a.out).write_text(json.dumps({"live": live, "test_set": test}, indent=1))

    print("A. Live captures: frames found per body (bodies sorted by edge gap), extra detections")
    for tag in sorted({r["tag"] for r in live}):
        for r in [r for r in live if r["tag"] == tag]:
            b = sorted(r["bodies"], key=lambda b: b["edge_gap_mm"])
            print(f"  {tag:18} lift {r['lift_mm']:.0f} {r['model']:15} "
                  + "  ".join(f"{x['edge_gap_mm']:4.1f}mm:{x['frames_found']}/{r['frames']}" for x in b)
                  + f"  extra {r['extra_detections']}")
    print("\nB. Real test frames")
    for r in test:
        print(f"  lift {r['lift_mm']:.0f} {r['model']:15} resistors {r['resistors']} screws {r['screws']} "
              f"false {r['false_complete']} on {r['complete_frames']} frames, centre p95 {r['resistor_centre_mm_p95']:.2f} mm")


if __name__ == "__main__":
    main()

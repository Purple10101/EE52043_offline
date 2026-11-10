"""seg_grey_v2 against seg_objects_v1 on the real test frames, at conf 0.5 (the live default).

    python experiments/2026-11-09_seg_objects_v1_test/compare_models.py

Writes results.json: per model, resistors and screws found, outline errors over resistors only, and
every false detection on the frames with every object outlined (labelled "objects"), with its size,
so the effect of a size cap or a crosshair mask can be read off without rerunning the models.
"""
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from robot_offline import config, manifest
from robot_offline.inf import tiling
from robot_offline.tools import evaluate as ev

HERE = Path(__file__).parent
MODELS = ["seg_grey_v2", "seg_objects_v1"]
CONF = 0.5


def main():
    ds = config.load("dataset")
    ppm = ds["scale"]["px_per_mm"]
    samples = [s for s in manifest.load(ds["labels"]["rules_version"])["samples"]
               if s["session"] in ds["splits"]["test_sessions"]]
    out = {}
    for name in MODELS:
        weights = config.ROOT / "models" / name / "best.pt"
        res = ev.evaluate_samples(YOLO(str(weights)), samples, conf=CONF, grey=tiling.wants_grey(weights))
        for s, r in zip(samples, res):
            for m in r["matched"]:
                m["kind"] = s["instances"][m["label_index"]].get("kind", "resistor")
        rm = [m for r in res for m in r["matched"] if m["kind"] == "resistor"]
        sm = [m for r in res for m in r["matched"] if m["kind"] == "screw"]
        falses = []
        for s, r in zip(samples, res):
            if s.get("labelled") != "objects":
                continue
            matched = {m["prediction_index"] for m in r["matched"]}
            for j, p in enumerate(r["polygons"]):
                if j not in matched:
                    _, _, length, width, _ = ev.body_geometry(p)
                    falses.append(dict(image=s["image"], length_mm=round(length / ppm, 1),
                                       width_mm=round(width / ppm, 1), score=round(float(r["scores"][j]), 2)))
        pct = lambda ms, key, q: float(np.percentile([m[key] for m in ms], q))
        out[name] = dict(
            by_group={g: list(v) for g, v in ev.by_group(samples, res).items()},
            by_kind={k: list(v) for k, v in ev.by_kind(samples, res).items()},
            resistor_iou_median=float(np.median([m["iou"] for m in rm])),
            resistor_centre_mm_p50=pct(rm, "centre_mm", 50), resistor_centre_mm_p95=pct(rm, "centre_mm", 95),
            resistor_yaw_deg_p50=pct(rm, "yaw_deg", 50), resistor_yaw_deg_p95=pct(rm, "yaw_deg", 95),
            resistor_length_bias_mm=float(np.mean([m["length_mm"] for m in rm])),
            resistor_width_bias_mm=float(np.mean([m["width_mm"] for m in rm])),
            screw_iou_median=float(np.median([m["iou"] for m in sm])) if sm else None,
            longest_found_mm=float(max(ev.body_geometry(r["polygons"][m["prediction_index"]])[2]
                                       for r in res for m in r["matched"]) / ppm),
            false_detections_complete=falses,
        )
        print(name, {k: v for k, v in out[name].items() if k != "false_detections_complete"},
              "false (complete):", len(falses))
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

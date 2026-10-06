"""Scores a trained model on the real test sessions, full frames with tiling.

Matches predictions to labels, then reports in robot units: centre, yaw, length and width error,
merges, splits, misses and false detections, plus mask IoU. Targets are in docs/plan.md.

    python -m robot_offline.evaluate --weights runs/resistor_seg/weights/best.pt
    python -m robot_offline.evaluate --weights best.pt --sessions synth_test

Body geometry comes from the minimum-area rectangle around each polygon: its centre, its long
side (length), short side (width) and the long side's direction (yaw, 0-180 degrees). Matching is
one-to-one, greedy by mask IoU, at IoU 0.5 or more.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from robot_offline import config, manifest, tiling

MATCH_IOU = 0.5
PART_OF = 0.2               # a prediction covering this much of a body counts towards a merge
CENTRE_TARGET_MM = 1.0      # docs/plan.md: centre and yaw within these for 95% of bodies
YAW_TARGET_DEG = 3.0
PASS_FRACTION = 0.95


def body_geometry(points):
    """(centre x, centre y, length, width, yaw deg in [0, 180)) of a polygon, in px."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.asarray(points, np.float32))
    if w < h:
        w, h, angle = h, w, angle + 90
    return cx, cy, w, h, angle % 180


def _shape(points):
    shape = Polygon(points)
    return shape if shape.is_valid else shape.buffer(0)


def score_frame(labels, predictions, px_per_mm):
    """Compares one frame's predicted polygons with its labelled ones.

    Returns a dict: per-body records for matched bodies ("matched"), counts of labels, predictions,
    misses, false detections, merges (a prediction over several bodies) and splits (a body under
    several predictions), and "ious": the best IoU of every label (0 if nothing overlaps).
    """
    gt = [_shape(p) for p in labels]
    pr = [_shape(p) for p in predictions]
    iou = np.zeros((len(gt), len(pr)))
    inter = np.zeros((len(gt), len(pr)))
    if gt and pr:
        tree = STRtree(pr)
        for i, g in enumerate(gt):
            for j in tree.query(g):
                a = g.intersection(pr[j]).area
                inter[i, j] = a
                iou[i, j] = a / (g.area + pr[j].area - a)

    pairs = []
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < MATCH_IOU:
            break
        if all(i != a and j != b for a, b in pairs):
            pairs.append((int(i), int(j)))

    matched = []
    for i, j in pairs:
        gx, gy, gl, gw, gyaw = body_geometry(labels[i])
        px, py, pl, pw, pyaw = body_geometry(predictions[j])
        dyaw = abs(gyaw - pyaw) % 180
        matched.append({
            "label_index": i,
            "prediction_index": j,
            "iou": iou[i, j],
            "coverage": inter[i, j] / gt[i].area,
            "centre_mm": float(np.hypot(px - gx, py - gy)) / px_per_mm,
            "dx_mm": (px - gx) / px_per_mm,
            "dy_mm": (py - gy) / px_per_mm,
            "yaw_deg": min(dyaw, 180 - dyaw),
            "length_mm": (pl - gl) / px_per_mm,
            "width_mm": (pw - gw) / px_per_mm,
            "true_length_mm": gl / px_per_mm,
            "true_width_mm": gw / px_per_mm,
            "true_yaw_deg": gyaw,
        })

    big = inter > PART_OF * np.array([g.area for g in gt]).reshape(-1, 1)
    return {
        "labels": len(gt),
        "predictions": len(pr),
        "matched": matched,
        "misses": len(gt) - len(pairs),
        "false_detections": len(pr) - len(pairs),
        "merges": int((big.sum(axis=0) > 1).sum()),
        "splits": int((big.sum(axis=1) > 1).sum()),
        "ious": iou.max(axis=1).tolist() if pr else [0.0] * len(gt),
    }


def summarise(frames):
    """One summary over many score_frame results, with the docs/plan.md targets checked."""
    matched = [m for f in frames for m in f["matched"]]
    n_gt = sum(f["labels"] for f in frames)
    n_pr = sum(f["predictions"] for f in frames)
    tp = len(matched)
    col = lambda k: np.array([m[k] for m in matched]) if matched else np.zeros(0)
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    centre, yaw = col("centre_mm"), col("yaw_deg")
    ious = np.array([v for f in frames for v in f["ious"]])
    empty = [f for f in frames if f["labels"] == 0]
    on_target = (centre < CENTRE_TARGET_MM) & (yaw < YAW_TARGET_DEG)
    out = {
        "frames": len(frames),
        "bodies": n_gt,
        "precision": tp / n_pr if n_pr else float("nan"),
        "recall": tp / n_gt if n_gt else float("nan"),
        "f1": 2 * tp / (n_gt + n_pr) if n_gt + n_pr else float("nan"),
        "misses": n_gt - tp,
        "false_detections": n_pr - tp,
        "false_det_on_empty": sum(f["predictions"] for f in empty),
        "merges": sum(f["merges"] for f in frames),
        "splits": sum(f["splits"] for f in frames),
        "best_iou_mean": float(ious.mean()) if len(ious) else float("nan"),
        "matched_iou_median": pct(col("iou"), 50),
        "centre_mm_p50": pct(centre, 50),
        "centre_mm_p95": pct(centre, 95),
        "yaw_deg_p50": pct(yaw, 50),
        "yaw_deg_p95": pct(yaw, 95),
        "length_mm_bias": float(col("length_mm").mean()) if tp else float("nan"),
        "length_mm_p95_abs": pct(np.abs(col("length_mm")), 95),
        "width_mm_bias": float(col("width_mm").mean()) if tp else float("nan"),
        "width_mm_p95_abs": pct(np.abs(col("width_mm")), 95),
        # Misses count as failures: the target is over all labelled bodies.
        "on_target_fraction": float(on_target.sum() / n_gt) if n_gt else float("nan"),
    }
    out["meets_centre_yaw_target"] = out["on_target_fraction"] >= PASS_FRACTION
    return out


def evaluate_samples(model, samples, conf=0.25, **predict_kwargs):
    """Runs the model over full frames with tiling and scores each against its labels.

    Returns (per-frame score_frame results, each with "image" and "seconds" added, and the
    predicted polygons and scores under "polygons" and "scores").
    """
    ds = config.load("dataset")
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    px_per_mm = ds["scale"]["px_per_mm"]
    eps = ds["labels"]["simplify_epsilon_px"]
    results = []
    for sample in samples:
        image = cv2.imread(str(config.DATA_DIR / sample["image"]))
        frame_tiles = tiling.tiles(work_area, (sample["width"], sample["height"]),
                                   tiles_cfg["size"], tiles_cfg["overlap"])
        t0 = time.perf_counter()
        polygons, scores = tiling.predict_frame(model, image, frame_tiles, conf=conf,
                                                epsilon_px=eps, **predict_kwargs)
        seconds = time.perf_counter() - t0
        result = score_frame(manifest.polygons(sample), polygons, px_per_mm)
        result.update(image=sample["image"], seconds=seconds, polygons=polygons, scores=scores)
        results.append(result)
    return results


def main(argv=None):
    from ultralytics import YOLO

    ds = config.load("dataset")
    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.evaluate",
        description="Score a model on full frames with tiling, in robot units.",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=list(ds["splits"]["test_sessions"]),
                        help="sessions to score; default test_sessions in configs/dataset.toml")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not args.sessions:
        parser.error("no --sessions given and test_sessions in configs/dataset.toml is empty")

    samples = [s for s in manifest.load(ds["labels"]["rules_version"])["samples"]
               if s["session"] in args.sessions]
    if not samples:
        parser.error(f"no frames in the manifest for sessions {args.sessions}")
    results = evaluate_samples(YOLO(str(args.weights)), samples, conf=args.conf)
    for key, value in summarise(results).items():
        print(f"{key:>24}  {value:.3f}" if isinstance(value, float) else f"{key:>24}  {value}")
    print(f"{'seconds_per_frame_p50':>24}  {np.median([r['seconds'] for r in results]):.3f}")


if __name__ == "__main__":
    main()

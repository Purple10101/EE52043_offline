"""Samples and statistics of the objects-profile synthetic set, for the design record.

    python experiments/2026-11-09_object_seg_design/samples.py [--frames 200] [--keep 12] [--seed 100]

Draws frames with `synth --profile objects` settings (configs/dataset.toml [synthetic_objects]), without
touching the manifest, and writes into samples/:
  - frame_NN.jpg, frame_NN_overlay.jpg: the first --keep frames, and with each label outlined in its
    kind's colour and named
  - frame_NN_model.png for the first two: what the model sees (plate-masked, greyscale)
  - sheet.jpg: the overlays side by side; gallery.jpg: examples of every kind, cropped round the label
  - stats.json: counts per kind, sizes, closeness to the plate edge, leads off the plate, compared with
    synth_v2 (from the manifest), the set seg_grey_v2 was trained on
"""
import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString

from robot_offline import config, manifest
from robot_offline.pre import rectify
from robot_offline.training import synth

HERE = Path(__file__).parent
OUT = HERE / "samples"
COLOURS = {  # BGR, by kind group
    "resistor": (60, 200, 60), "diode": (40, 40, 230), "capacitor_axial": (0, 150, 255),
    "capacitor_lying": (0, 150, 255), "capacitor_radial": (0, 150, 255), "capacitor_ceramic": (0, 150, 255),
    "dip8": (230, 120, 30), "to92": (230, 120, 30), "to220": (230, 120, 30), "screw": (200, 60, 200),
    "nut": (200, 60, 200), "block": (220, 220, 0), "blob": (220, 220, 0), "cylinder": (220, 220, 0),
}


def edge_facts(poly, plate, s):
    """(gap to the nearest plate edge mm, which edge, body axis angle to that edge deg: 0 parallel, 90 pointing at it)."""
    x0, y0, x1, y1 = plate
    gaps = [poly[:, 0].min() - x0, x1 - 1 - poly[:, 0].max(), poly[:, 1].min() - y0, y1 - 1 - poly[:, 1].max()]
    side = int(np.argmin(gaps))
    (_, _), (w, h), a = cv2.minAreaRect(poly.astype(np.float32))
    yaw = a if w >= h else a + 90                       # long axis, degrees from image x
    edge_dir = 90.0 if side < 2 else 0.0                # left/right edges run along y, top/bottom along x
    d = abs((yaw - edge_dir) % 180)
    return gaps[side] / s, ("near", "back", "robot base", "far side")[side], min(d, 180 - d)


def one(job):
    i, seed, keep = job
    ds = config.load("dataset")
    st = synth.Settings.from_config(ds, "objects")
    frame = synth.render_frame(st, np.random.default_rng([seed, i]))
    s = st.px_per_mm
    plate = synth._plate_shape(st)
    rows = []
    bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    crops = []
    for poly, kind, leads in zip(frame.polygons, frame.kinds, frame.leads):
        (_, _), (w, h), _ = cv2.minAreaRect(poly.astype(np.float32))
        gap, side, angle = edge_facts(poly, st.plate, s)
        off = any(not plate.contains(LineString(q)) for q in leads) if leads else None
        rows.append({"kind": kind, "length_mm": max(w, h) / s, "width_mm": min(w, h) / s,
                     "edge_gap_mm": gap, "edge": side, "angle_to_edge_deg": angle, "lead_off_plate": off})
        cx, cy = poly.mean(0).astype(int)
        half = int(max(w, h) / 2 + 3 * s)
        c = bgr[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half].copy()
        q = poly - [max(cx - half, 0), max(cy - half, 0)]
        cv2.polylines(c, [np.round(q).astype(np.int32)], True, COLOURS[kind], 1, cv2.LINE_AA)
        crops.append((kind, cv2.resize(c, (120, 120), interpolation=cv2.INTER_AREA)))
    kept = None
    if i < keep:
        over = bgr.copy()
        for poly, kind in zip(frame.polygons, frame.kinds):
            q = np.round(poly).astype(np.int32)
            cv2.polylines(over, [q], True, COLOURS[kind], 2, cv2.LINE_AA)
            x, y = int(poly[:, 0].max()) + 4, int(poly[:, 1].min()) + 10
            for col, wt in (((255, 255, 255), 3), ((0, 0, 0), 1)):
                cv2.putText(over, kind, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, wt, cv2.LINE_AA)
        model = None
        if i < 2:
            sample = {"space": "rectified", "source": "synthetic"}
            m = rectify.model_image(sample, bgr, rectify.Settings.from_config(ds))
            model = cv2.cvtColor(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        kept = (bgr, over, model)
    return i, rows, crops, kept


def synth_v2_edges():
    """seg_grey_v2's training set, the same edge measures, from the manifest."""
    ds = config.load("dataset")
    st = synth.Settings.from_config(ds)
    rows = []
    for smp in manifest.load(ds["labels"]["rules_version"])["samples"]:
        if smp["session"] == "synth_v2":
            for p in manifest.polygons(smp):
                rows.append(edge_facts(p, st.plate, st.px_per_mm))
    return rows


def summary(rows, v2):
    n = len(rows)
    out = {"instances": n, "per_kind": {}, "edge": {}, "synth_v2_edge": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        L = np.array([r["length_mm"] for r in rs]); W = np.array([r["width_mm"] for r in rs])
        out["per_kind"][k] = {"count": len(rs), "share": round(len(rs) / n, 3),
                              "length_mm": [round(float(np.percentile(L, q)), 1) for q in (5, 50, 95)],
                              "width_mm": [round(float(np.percentile(W, q)), 1) for q in (5, 50, 95)]}
    for lim in (3, 5, 8):
        near = [r for r in rows if r["edge_gap_mm"] < lim]
        out["edge"][f"within_{lim}mm"] = {
            "share": round(len(near) / n, 3),
            "pointing_at_edge_ge60deg": sum(r["angle_to_edge_deg"] >= 60 for r in near),
            "diagonal_30_60deg": sum(30 <= r["angle_to_edge_deg"] < 60 for r in near),
            "parallel_lt30deg": sum(r["angle_to_edge_deg"] < 30 for r in near),
            "at_back_edge": sum(r["edge"] == "back" for r in near)}
        v2near = [r for r in v2 if r[0] < lim]
        out["synth_v2_edge"][f"within_{lim}mm"] = {
            "share": round(len(v2near) / max(len(v2), 1), 3),
            "pointing_at_edge_ge60deg": sum(r[2] >= 60 for r in v2near),
            "diagonal_30_60deg": sum(30 <= r[2] < 60 for r in v2near),
            "parallel_lt30deg": sum(r[2] < 30 for r in v2near),
            "at_back_edge": sum(r[1] == "back" for r in v2near)}
    leaded = [r for r in rows if r["lead_off_plate"] is not None]
    out["bodies_with_leads"] = len(leaded)
    out["lead_off_plate_share"] = round(sum(r["lead_off_plate"] for r in leaded) / max(len(leaded), 1), 3)
    out["synth_v2_instances"] = len(v2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--keep", type=int, default=12)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    rows, gallery, kept = [], defaultdict(list), {}
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r, crops, k in pool.map(one, [(i, a.seed, a.keep) for i in range(a.frames)]):
            rows += r
            for kind, c in crops:
                if len(gallery[kind]) < 6:
                    gallery[kind].append(c)
            if k is not None:
                kept[i] = k
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{a.frames}", flush=True)
    for i, (bgr, over, model) in sorted(kept.items()):
        cv2.imwrite(str(OUT / f"frame_{i:02d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"frame_{i:02d}_overlay.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if model is not None:
            cv2.imwrite(str(OUT / f"frame_{i:02d}_model.png"), model)
    small = [cv2.resize(kept[i][1], (400, 400), interpolation=cv2.INTER_AREA) for i in sorted(kept)]
    while len(small) % 4:
        small.append(np.full_like(small[0], 255))
    cv2.imwrite(str(OUT / "sheet.jpg"), np.vstack([np.hstack(small[r:r + 4]) for r in range(0, len(small), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    rows_img = []
    for kind in COLOURS:
        cells = gallery.get(kind, [])
        if not cells:
            continue
        cells = cells + [np.full((120, 120, 3), 255, np.uint8)] * (6 - len(cells))
        name = np.full((120, 170, 3), 255, np.uint8)
        cv2.putText(name, kind, (6, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows_img.append(np.hstack([name] + cells))
    cv2.imwrite(str(OUT / "gallery.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 92])
    stats = summary(rows, synth_v2_edges())
    stats.update(frames=a.frames, seed=a.seed, settings="configs/dataset.toml [synthetic] + [synthetic_objects]")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

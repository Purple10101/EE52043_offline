"""Figure: why bodies at the plate's far (+x) edge are lost, from the live captures of 2026-11-09.

    python experiments/2026-11-09_seg_objects_v1_test/edge_evidence.py

One row per body; columns:
  1. the camera's view, top-down (rectified, not masked), the plate's edge in yellow;
  2. what the model is given (rectify.model_input): everything past the yellow line painted plate grey;
  3. the same with the mask widened 3 mm, and each model's outline on that frame at conf 0.05
     (green seg_grey_v2, magenta seg_objects_v1; score beside).
Rows: the four far-edge bodies of edge_a_light2 (2.4 mm parallax outward), then for comparison a
body at the close edge (edge_near_light2, shift inward) and the blue body in the far corner at the
same distance from the lens on the -y edge (far_focus306, shift along the edge).
Writes edge_evidence.jpg.
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.inf.steady import FLOOR
from robot_offline.pre import rectify

HERE = Path(__file__).parent
LIVE = HERE / "live"
ROWS = [  # tag, body centre (mm), label
    ("edge_a_light2", (48.1, -100.3), "far (+x) edge, gap 0.1 mm, 282 mm from lens"),
    ("edge_a_light2", (46.2, -121.5), "far (+x) edge, gap 0.6 mm"),
    ("edge_a_light2", (46.2, -140.9), "far (+x) edge, gap 0.8 mm"),
    ("edge_a_light2", (47.5, -162.2), "far (+x) edge, gap 0.0 mm"),
    ("edge_near_light2", (-45.8, -155.0), "close (-x) edge, gap 1.4 mm, 229 mm"),
    ("far_focus306", (40.0, -174.6), "-y edge far corner, gap 2.5 mm, 280 mm"),
]
HALF_MM, SCALE, GROW_MM = 7.0, 3, 3.0
MODELS = {"seg_grey_v2": (0, 200, 0), "seg_objects_v1": (255, 0, 255)}


def main():
    ds = config.load("dataset")
    area, tc = ds["work_area"], ds["tiles"]
    work = (area["x0"], area["y0"], area["width"], area["height"])
    models = {m: YOLO(str(config.ROOT / "models" / m / "best.pt")) for m in MODELS}
    cache, rows = {}, []
    for tag, centre, label in ROWS:
        if tag not in cache:
            frame = cv2.imread(str(LIVE / f"{tag}_raw.png"))
            cal = rectify.Calibration.load(LIVE / f"{tag}_rectify.json")
            st = cal.settings
            plain = rectify.rectify(frame, cal)
            view, balanced = rectify.model_input(frame, cal, plain)
            inside = rectify.plate_mask(st)
            r = int(round(GROW_MM * st.px_per_mm))
            keep = cv2.dilate(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            wide = balanced.copy()
            wide[~keep] = view[~inside][0]
            tiles = tiling.tiles(work, (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
            preds = {m: tiling.predict_frame(models[m], wide, tiles, conf=FLOOR, grey=True) for m in MODELS}
            edge, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cache[tag] = (st, balanced, view, wide, preds, edge)
        st, balanced, view, wide, preds, edge = cache[tag]
        cx, cy = rectify.mm_to_rect([centre], st)[0]
        h = int(HALF_MM * st.px_per_mm)
        x0, y0 = int(cx) - h, int(cy) - h
        tiles_out = []
        for k, img in enumerate((balanced, view, wide)):
            c = img.copy()
            if k == 0:
                cv2.drawContours(c, edge, -1, (0, 230, 255), 1, cv2.LINE_AA)
            if k == 2:
                for m, colour in MODELS.items():
                    for q, s in zip(*preds[m]):
                        q = np.asarray(q)
                        if np.linalg.norm(q.mean(0) - (cx, cy)) < 40:
                            cv2.polylines(c, [np.round(q).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
            crop = cv2.resize(c[y0:y0 + 2 * h, x0:x0 + 2 * h], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
            if k == 2:
                ty = 18
                for m, colour in MODELS.items():
                    best = max([s for q, s in zip(*preds[m]) if np.linalg.norm(np.asarray(q).mean(0) - (cx, cy)) < 40], default=0)
                    cv2.rectangle(crop, (2, ty - 15), (190, ty + 5), (255, 255, 255), -1)
                    cv2.putText(crop, f"{m} {best:.2f}", (6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                tuple(int(v * 0.7) for v in colour), 1, cv2.LINE_AA)
                    ty += 22
            tiles_out.append(crop)
        row = np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0)), constant_values=255) for t in tiles_out])
        bar = np.full((26, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{tag}: {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row, np.full((8, row.shape[1], 3), 255, np.uint8)]))
    head = np.full((30, rows[0].shape[1], 3), 255, np.uint8)
    w = rows[0].shape[1] // 3
    for k, t in enumerate(["camera view, plate edge in yellow", "model input (as live)", "mask widened 3 mm + outlines"]):
        cv2.putText(head, t, (k * w + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(HERE / "edge_evidence.jpg"), np.vstack([head, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()

"""Focus sweep: how sharp each body on the plate is at each focus setting, in the raw camera frame.

    python experiments/2026-11-09_front_back/focus_sweep.py experiments/2026-11-09_front_back [--start 270 --stop 350 --step 4]

Finds the plate and the bodies once (seg_grey_v2), holds the exposure, then for each focus value: sets it,
lets the lens settle, and measures each body's sharpness on 3 frames. Sharpness: the mean squared Sobel
gradient of grey inside the body outline (dilated 3 px to include its edges), relative to the same body's
best over the sweep, so bodies of different colour compare. Writes focus_sweep.json, focus_sweep.png
(curves) and focus_sweep_crops.png (each body at the config focus and at its own best).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.camera import open_camera

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--start", type=int, default=270)
p.add_argument("--stop", type=int, default=350)
p.add_argument("--step", type=int, default=4)
p.add_argument("--settle", type=int, default=10, help="frames read after a focus change before measuring")
a = p.parse_args()

ds = config.load("dataset")
st = rectify.Settings.from_config(ds)
camera = config.load("camera")
cam_mm = np.array(st.camera_mm)
config_focus = camera["camera"]["focus"]
cap = open_camera(camera)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cal = plate_search.find_now(cap, st, camera=camera)
for _ in range(8):
    cap.read()
ok, frame = cap.read()
view, _ = rectify.model_input(frame, cal)
w = ds["work_area"]
polys, scores = tiling.predict_frame(YOLO(str(config.ROOT / "models/seg_grey_v2/best.pt")), view,
                                     tiling.tiles((w["x0"], w["y0"], w["width"], w["height"]), view.shape[1::-1], 1024, 128),
                                     conf=0.25, grey=True)
bodies = []
for q in polys:
    full = rectify.to_full(np.asarray(q), cal)
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(full).astype(np.int32)], 1)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    mm = rectify.rect_to_mm(np.asarray(q).mean(0), st)[0]
    bodies.append({"x_mm": float(mm[0]), "y_mm": float(mm[1]),
                   "dist_mm": float(np.linalg.norm([mm[0] - cam_mm[0], mm[1] - cam_mm[1], st.top_z_mm - cam_mm[2]])),
                   "mask": m > 0, "centre_px": full.mean(0)})
bodies.sort(key=lambda b: b["dist_mm"])
print(f"{len(bodies)} bodies, distances " + ", ".join(f"{b['dist_mm']:.0f}" for b in bodies) + " mm", flush=True)


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2
    return [float(e[b["mask"]].mean()) for b in bodies]


focuses = list(range(a.start, a.stop + 1, a.step))
if config_focus not in focuses:
    focuses = sorted(focuses + [config_focus])
table, frames = [], {}
for f in focuses:
    cap.set(cv2.CAP_PROP_FOCUS, f)
    for _ in range(a.settle):
        cap.read()
    vals = []
    for _ in range(3):
        ok, img = cap.read()
        vals.append(sharpness(img))
    frames[f] = img
    table.append(np.mean(vals, 0))
    print(f"focus {f}: reported {cap.get(cv2.CAP_PROP_FOCUS):.0f}  " + "  ".join(f"{v:7.0f}" for v in table[-1]), flush=True)
cap.set(cv2.CAP_PROP_FOCUS, config_focus)
cap.release()

t = np.array(table)                       # focus × body
rel = t / t.max(0)
best = [focuses[i] for i in rel.argmax(0)]
at_config = rel[focuses.index(config_focus)]
worst = rel.min(1)                        # per focus: the least sharp body, relative to its best
print(f"\n{'body':>5} {'dist mm':>8} {'best focus':>10} {'at config ' + str(config_focus):>14}")
for i, b in enumerate(bodies):
    print(f"{i + 1:5d} {b['dist_mm']:8.0f} {best[i]:10d} {at_config[i]:14.2f}")
k = int(worst.argmax())
print(f"\nbest worst case: focus {focuses[k]} (every body at >= {worst[k]:.2f} of its best); "
      f"config {config_focus}: worst body at {worst[focuses.index(config_focus)]:.2f}")

json.dump({"focuses": focuses, "config_focus": config_focus, "sharpness": t.tolist(),
           "bodies": [{k_: v for k_, v in b.items() if k_ not in ("mask", "centre_px")} for b in bodies],
           "best_focus": best, "best_worst_case_focus": focuses[k]},
          open(a.out / "focus_sweep.json", "w"), indent=1)

# Curves.
H, W, pad = 420, 760, 50
plot = np.full((H, W, 3), 255, np.uint8)
xs = lambda f: int(pad + (f - focuses[0]) / (focuses[-1] - focuses[0]) * (W - 2 * pad))
ys = lambda v: int(H - pad - v * (H - 2 * pad))
cv2.line(plot, (xs(config_focus), pad), (xs(config_focus), H - pad), (180, 180, 180), 1)
cv2.putText(plot, f"config {config_focus}", (xs(config_focus) + 4, pad + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
colours = [(200, 120, 0), (0, 150, 0), (0, 120, 220), (0, 0, 200), (150, 0, 150), (100, 100, 0)]
for i, b in enumerate(bodies):
    pts = np.array([[xs(f), ys(v)] for f, v in zip(focuses, rel[:, i])], np.int32)
    cv2.polylines(plot, [pts], False, colours[i % len(colours)], 2, cv2.LINE_AA)
    cv2.putText(plot, f"{b['dist_mm']:.0f} mm from lens, best {best[i]}", (pad + 10, pad + 18 * i + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[i % len(colours)], 1, cv2.LINE_AA)
cv2.rectangle(plot, (pad, pad), (W - pad, H - pad), (0, 0, 0), 1)
cv2.putText(plot, "focus setting", (W // 2 - 50, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(plot, "sharpness / body's best", (5, pad - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
for f in focuses[::2]:
    cv2.putText(plot, str(f), (xs(f) - 10, H - pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.imwrite(str(a.out / "focus_sweep.png"), plot)

# Crops: each body at the config focus, at the best-worst-case focus and at its own best.
rows = []
for i, b in enumerate(bodies):
    cx, cy = b["centre_px"].astype(int)
    cells = []
    for f, name in ((config_focus, "config"), (focuses[k], "best worst case"), (best[i], "own best")):
        c = cv2.resize(frames[f][cy - 45:cy + 45, cx - 60:cx + 60], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f"{b['dist_mm']:.0f} mm, focus {f} ({name})", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cells.append(c)
    rows.append(np.hstack(cells))
cv2.imwrite(str(a.out / "focus_sweep_crops.png"), np.vstack(rows))

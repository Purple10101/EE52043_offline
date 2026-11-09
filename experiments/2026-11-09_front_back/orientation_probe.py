"""Orientation probe on one captured frame: each body turned in place, everything else left as it is.

    python experiments/2026-11-09_front_back/orientation_probe.py experiments/2026-11-09_front_back <tag> [x,y ...]

Takes <tag>_model.png (exactly what the model saw) and, for each body (those in <tag>.json, plus any
extra centres given in rectified px), cuts a disc round it (radius 6 mm), turns the disc by 0 to 345
degrees in 15 degree steps, blends it back in place and runs seg_grey_v2 (conf 0.001). The body keeps its
position, light, focus and pixels; only its direction changes. Writes <tag>_orientation.json,
<tag>_orientation.png (score against the body's resulting direction) and <tag>_orientation_crops.png
(every turned version with its score). Beyond the plate edge the model's input is filled with the plate's
colour, so a disc reaching past the edge turns some of that fill in: it looks like plate.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.pre import rectify
from robot_offline.tools.evaluate import body_geometry

out, tag = Path(sys.argv[1]), sys.argv[2]
extra = [tuple(map(float, s.split(","))) for s in sys.argv[3:]]
ds = config.load("dataset")
st = rectify.Settings.from_config(ds)
w = ds["work_area"]
view = cv2.imread(str(out / f"{tag}_model.png"))
tiles = tiling.tiles((w["x0"], w["y0"], w["width"], w["height"]), view.shape[1::-1], 1024, 128)
model = YOLO(str(config.ROOT / "models/seg_grey_v2/best.pt"))
cam = rectify.mm_to_rect(st.camera_mm[:2], st)[0]

bodies = [np.asarray(b["poly"]).mean(0) for b in json.loads((out / f"{tag}.json").read_text())["bodies"]]
bodies += [np.array(e) for e in extra]
R = int(6 * st.px_per_mm)
angles = list(range(0, 360, 15))
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
alpha = np.clip((R - np.hypot(xx, yy)) / 8, 0, 1)[..., None].astype(np.float32)   # feathered disc edge


def score_at(img, c):
    polys, sc = tiling.predict_frame(model, img, tiles, conf=0.001, grey=True)
    return max([s for p, s in zip(polys, sc) if np.linalg.norm(np.asarray(p).mean(0) - c) < 25], default=0.0)


results, sheets = [], []
for c in bodies:
    cx, cy = np.round(c).astype(int)
    patch = view[cy - R:cy + R + 1, cx - R:cx + R + 1].astype(np.float32)
    # The body's own direction (0 = image x axis), and the direction to the camera, for reference.
    polys, sc = tiling.predict_frame(model, view, tiles, conf=0.001, grey=True)
    near = [p for p in polys if np.linalg.norm(np.asarray(p).mean(0) - c) < 25]
    yaw = float(body_geometry(near[0])[4]) if near else float("nan")
    to_cam = float(np.degrees(np.arctan2(cam[1] - cy, cam[0] - cx)) % 180)
    scores, cells = [], []
    for a in angles:
        m = cv2.getRotationMatrix2D((R, R), a, 1.0)
        turned = cv2.warpAffine(patch, m, patch.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = view.copy()
        img[cy - R:cy + R + 1, cx - R:cx + R + 1] = (turned * alpha + patch * (1 - alpha)).round().astype(np.uint8)
        scores.append(score_at(img, c))
        cell = cv2.resize(img[cy - 2 * R:cy + 2 * R + 1, cx - 2 * R:cx + 2 * R + 1], (160, 160), interpolation=cv2.INTER_AREA)
        ok = scores[-1] >= 0.5
        cv2.rectangle(cell, (0, 0), (159, 159), (0, 170, 0) if ok else (0, 0, 220), 3)
        for col, wt in (((255, 255, 255), 3), ((0, 0, 0), 1)):
            cv2.putText(cell, f"{a} deg: {scores[-1]:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, wt, cv2.LINE_AA)
        cells.append(cell)
    sheets.append(np.vstack([np.hstack(cells[i:i + 12]) for i in (0, 12)]))
    mm = rectify.rect_to_mm(c, st)[0]
    results.append({"centre_px": c.tolist(), "x_mm": float(mm[0]), "y_mm": float(mm[1]), "yaw_deg": yaw,
                    "to_camera_deg": to_cam, "angles": angles, "scores": scores})
    s = np.array(scores)
    print(f"body at ({mm[0]:+.0f}, {mm[1]:.0f}) mm, yaw {yaw:.0f}: as is {s[0]:.2f}; min {s.min():.2f} max {s.max():.2f}; "
          f"<0.5 at {(s < 0.5).sum()}/{len(s)} angles: " + " ".join(f"{a}:{v:.2f}" for a, v in zip(angles, s)), flush=True)
json.dump(results, open(out / f"{tag}_orientation.json", "w"), indent=1)
# Every turned version of every body, with its score (green frame: 0.5 or more), one block of 2 rows per body.
gap = np.full((12, sheets[0].shape[1], 3), 255, np.uint8)
cv2.imwrite(str(out / f"{tag}_orientation_crops.png"), np.vstack([x for s in sheets for x in (s, gap)][:-1]))

# Plot: score against the body's resulting direction in the image (yaw - angle, as cv2 turns anticlockwise).
H, W, pad = 380, 760, 50
plot = np.full((H, W, 3), 255, np.uint8)
xs = lambda d: int(pad + (d % 180) / 180 * (W - 2 * pad))
ys = lambda v: int(H - pad - v * (H - 2 * pad))
cv2.line(plot, (pad, ys(0.5)), (W - pad, ys(0.5)), (200, 200, 200), 1)
colours = [(200, 120, 0), (0, 150, 0), (0, 120, 220), (0, 0, 200), (150, 0, 150), (100, 100, 0)]
for i, r in enumerate(results):
    col = colours[i % len(colours)]
    d = (r["yaw_deg"] + np.array(r["angles"])) % 180          # image y is down: cv2's anticlockwise turn adds to yaw
    for dd, v in zip(d, r["scores"]):
        cv2.circle(plot, (xs(dd), ys(v)), 4, col, -1, cv2.LINE_AA)
    cv2.putText(plot, f"body at ({r['x_mm']:+.0f}, {r['y_mm']:.0f}) mm", (pad + 10, pad + 16 * i + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
cv2.rectangle(plot, (pad, pad), (W - pad, H - pad), (0, 0, 0), 1)
for d in range(0, 181, 30):
    cv2.putText(plot, str(d), (xs(d) - 8 if d < 180 else W - pad - 10, H - pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.putText(plot, "body direction in the top-down image (deg, 0 = across, 90 = towards/away from robot base)",
            (pad, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
cv2.putText(plot, "score", (5, pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite(str(out / f"{tag}_orientation.png"), plot)

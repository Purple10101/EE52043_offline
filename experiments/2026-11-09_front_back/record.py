"""Front-vs-back segmentation test: records what live_detect's model sees, N runs, one placement per call.

    python experiments/2026-11-09_front_back/record.py experiments/2026-11-09_front_back <tag> [--runs 15]

Writes, per tag: <tag>.json (per body: scores, outline size and stability, local plate grey, edge
sharpness, colour), <tag>_raw.png (the last run's camera frame), <tag>_rectify.json (its plate fit),
<tag>_rect.png (top-down), <tag>_model.png (the model's input) and <tag>_overlay.png (outlines drawn).

Opens the camera as live_detect does, finds the plate (plate_search.find_now) at the band-reading
level (live_detect's default with the reader present) or the model's, then runs seg_grey_v2 at
steady.FLOOR on N frames. Per body (matched across runs within 2 mm): its scores (0 when absent), and
from the model's input (grey, plate at 200 overall): the local plate grey round it, its own grey, and
the edge sharpness of its outline. Writes <out>/<tag>.json and the last run's images.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

from robot_offline.tools import evaluate

from robot_offline import config
from robot_offline.inf import tiling
from robot_offline.inf.steady import FLOOR
from robot_offline.post.bands import learned
from robot_offline.pre import plate_search, rectify
from robot_offline.pre.camera import open_camera

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("tag")
p.add_argument("--runs", type=int, default=15)
p.add_argument("--level", choices=["model", "reading"], default="reading")
p.add_argument("--weights", default=str(config.ROOT / "models" / "seg_grey_v2" / "best.pt"))
p.add_argument("--focus", type=int, help="override camera.toml's focus for this capture")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

ds = config.load("dataset")
st = rectify.Settings.from_config(ds)
area, tc = ds["work_area"], ds["tiles"]
work = (area["x0"], area["y0"], area["width"], area["height"])
cam_mm = np.array(st.camera_mm)
model = YOLO(a.weights)
grey_in = tiling.wants_grey(a.weights)
camera = config.load("camera")
cap = open_camera(camera)
if a.focus is not None:
    cap.set(cv2.CAP_PROP_FOCUS, a.focus)
    for _ in range(10):
        cap.read()
focus = cap.get(cv2.CAP_PROP_FOCUS)
level =learned.READING_LEVEL if a.level == "reading" else None
t0 = time.perf_counter()
cal = plate_search.find_now(cap, st, camera=camera, level=level)
print(f"plate found in {time.perf_counter() - t0:.0f} s, exposure {cap.get(cv2.CAP_PROP_EXPOSURE)}, "
      f"gain {cap.get(cv2.CAP_PROP_GAIN)}", flush=True)


def measure(grey, poly):
    """Local plate grey (ring 2 to 6 mm out, other bodies and dark/bright outliers aside), body grey,
    and edge sharpness: 90th pct gradient at the outline / |contrast| (1/px; higher is sharper)."""
    h, w = grey.shape
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = cv2.dilate(m, k(60)) & ~cv2.dilate(m, k(20)) & 1
    inner = cv2.erode(m, k(3))
    plate = float(np.median(grey[ring > 0])) if ring.any() else float("nan")
    body = float(np.median(grey[inner > 0] if inner.any() else grey[m > 0]))
    gx = cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8
    edge = cv2.dilate(m, k(2)) & ~cv2.erode(m, k(2)) & 1
    g = np.hypot(gx, gy)[edge > 0]
    contrast = plate - body
    return plate, body, float(np.percentile(g, 90) / max(abs(contrast), 1.0))


runs, last = [], None
while len(runs) < a.runs:
    ok, frame = cap.read()
    if not ok:
        continue
    plain = rectify.rectify(frame, cal)
    view, _ = rectify.model_input(frame, cal, plain)
    tiles = tiling.tiles(work, (view.shape[1], view.shape[0]), tc["size"], tc["overlap"])
    polys, scores = tiling.predict_frame(model, view, tiles, conf=FLOOR, grey=grey_in)
    g = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    runs.append([(np.asarray(q, float), float(s), *measure(g, np.asarray(q, float))) for q, s in zip(polys, scores)])
    last = (frame, plain, view, polys, scores)
    print(f"run {len(runs)}: " + ", ".join(f"{s:.2f}" for s in sorted(scores, reverse=True)), flush=True)
    for _ in range(3):          # a few frames apart, as live_detect at 3 fps
        cap.read()
exposure, gain = cap.get(cv2.CAP_PROP_EXPOSURE), cap.get(cv2.CAP_PROP_GAIN)
cap.release()

def colour(rect, poly):
    """Median BGR of the body (outline eroded 3 px) in the top-down colour frame, to tell the parts apart."""
    m = np.zeros(rect.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    m = cv2.erode(m, np.ones((7, 7), np.uint8))
    return [int(v) for v in np.median(rect[m > 0], axis=0)] if m.any() else None


def outline(by_run):
    """Outline quality over the runs a body was found in: length, width (median, mm), centroid jitter
    (RMS about the mean, mm), yaw jitter (SD, deg), IoU of consecutive outlines (median)."""
    s = st.px_per_mm
    geo = np.array([evaluate.body_geometry(v[4]) for v in by_run.values()])
    out = {"length_mm": float(np.median(geo[:, 2]) / s), "width_mm": float(np.median(geo[:, 3]) / s),
           "length_sd_mm": float(geo[:, 2].std() / s), "width_sd_mm": float(geo[:, 3].std() / s)}
    c = geo[:, :2]
    out["centre_jitter_mm"] = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()) / s)
    yaw = np.radians(geo[:, 4]) * 2                       # axial: double the angle to average
    mean = np.angle(np.exp(1j * yaw).mean())
    out["yaw_sd_deg"] = float(np.degrees(np.sqrt((np.angle(np.exp(1j * (yaw - mean))) ** 2).mean())) / 2)
    keys = sorted(by_run)
    ious = []
    for r0, r1 in zip(keys, keys[1:]):
        a, b = Polygon(by_run[r0][4]).buffer(0), Polygon(by_run[r1][4]).buffer(0)
        ious.append(a.intersection(b).area / max(a.union(b).area, 1e-9))
    out["run_iou"] = float(np.median(ious)) if ious else float("nan")
    return out


# Bodies across runs, by centroid within 2 mm.
bodies = []
for r, dets in enumerate(runs):
    for poly, score, plate, body, sharp in dets:
        c = poly.mean(0)
        hit = next((b for b in bodies if np.linalg.norm(b["c"] - c) < 2 * st.px_per_mm), None)
        if hit is None:
            hit = {"c": c, "runs": {}, "poly": poly}
            bodies.append(hit)
        if r not in hit["runs"] or hit["runs"][r][0] < score:
            hit["runs"][r] = (score, plate, body, sharp, poly)
out = []
for b in bodies:
    mm = rectify.rect_to_mm(b["c"], st)[0]
    vals = list(b["runs"].values())
    sc = [b["runs"].get(r, (0.0,))[0] for r in range(len(runs))]
    out.append({"x_mm": float(mm[0]), "y_mm": float(mm[1]),
                "dist_mm": float(np.linalg.norm([mm[0] - cam_mm[0], mm[1] - cam_mm[1], st.top_z_mm - cam_mm[2]])),
                "scores": sc, "found": len(vals),
                "plate": float(np.median([v[1] for v in vals])), "body": float(np.median([v[2] for v in vals])),
                "sharp": float(np.median([v[3] for v in vals])), "poly": b["poly"].tolist(),
                "colour_bgr": colour(last[1], b["poly"]), **outline(b["runs"])})
out = [b for b in out if b["found"] >= 3 or max(b["scores"]) >= 0.25]     # drop one-off specks
out.sort(key=lambda b: b["dist_mm"])

# Fragility: the last run's model input shifted by up to 16 px (3 x 3) and turned 180 degrees.
view = last[2]
pad = cv2.copyMakeBorder(view, 32, 32, 32, 32, cv2.BORDER_REPLICATE)
# Bodies the live runs never found but the turned image does (polygon mapped back).
size = np.array(view.shape[1::-1]) - 1
for q, s in zip(*tiling.predict_frame(model, cv2.rotate(view, cv2.ROTATE_180), tiles, conf=0.25, grey=grey_in)):
    q = size - np.asarray(q, float)
    c = q.mean(0)
    if all(np.linalg.norm(np.asarray(b["poly"]).mean(0) - c) >= 25 for b in out):
        mm = rectify.rect_to_mm(c, st)[0]
        out.append({"x_mm": float(mm[0]), "y_mm": float(mm[1]), "dist_mm": float(np.linalg.norm(
            [mm[0] - cam_mm[0], mm[1] - cam_mm[1], st.top_z_mm - cam_mm[2]])), "scores": [0.0] * len(runs),
            "found": 0, "plate": float("nan"), "body": float("nan"), "sharp": float("nan"), "poly": q.tolist()})
out.sort(key=lambda b: b["dist_mm"])
centres = np.array([np.asarray(b["poly"]).mean(0) for b in out]).reshape(-1, 2)


def best_near(img, pts):
    polys, sc = tiling.predict_frame(model, img, tiles, conf=0.001, grey=grey_in)
    cs = [np.asarray(q).mean(0) for q in polys]
    return [max([s for c, s in zip(cs, sc) if np.linalg.norm(c - p) < 25], default=0.0) for p in pts]


shifted = [best_near(pad[32 - dy:32 - dy + view.shape[0], 32 - dx:32 - dx + view.shape[1]], centres + (dx, dy))
           for dx in (-16, 0, 16) for dy in (-16, 0, 16)]
turned = best_near(cv2.rotate(view, cv2.ROTATE_180), np.array(view.shape[1::-1]) - 1 - centres)
for i, b in enumerate(out):
    b["shift_scores"] = [s[i] for s in shifted]
    b["turned_score"] = turned[i]
json.dump({"tag": a.tag, "level": a.level, "focus": focus, "exposure": exposure, "gain": gain, "runs": len(runs), "bodies": out},
          open(a.out / f"{a.tag}.json", "w"), indent=1)
frame, plain, view, polys, scores = last
cv2.imwrite(str(a.out / f"{a.tag}_raw.png"), frame)
cal.save(a.out / f"{a.tag}_rectify.json")
cv2.imwrite(str(a.out / f"{a.tag}_model.png"), view)
cv2.imwrite(str(a.out / f"{a.tag}_rect.png"), plain)
over = plain.copy()
for i, b in enumerate(out):
    q = np.round(np.asarray(b["poly"])).astype(np.int32)
    s = np.array(b["scores"])
    colour_ = (0, 200, 0) if (s >= 0.5).mean() >= 0.8 else (0, 0, 255)
    cv2.polylines(over, [q], True, colour_, 2, cv2.LINE_AA)
    x, y = q[:, 0].max() + 8, int(q[:, 1].mean())
    lines = [f"#{i + 1} score {np.median(s):.2f} ({(s >= 0.5).sum()}/{len(s)})"]
    if b["found"]:
        lines.append(f"{b['length_mm']:.1f} x {b['width_mm']:.1f} mm")
    for k, t in enumerate(lines):
        for c, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
            cv2.putText(over, t, (x, y + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, w, cv2.LINE_AA)
cv2.putText(over, f"{a.tag}: seg_grey_v2, {len(runs)} live runs, focus {focus:.0f}, exposure {exposure:.0f} gain {gain:.0f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(str(a.out / f"{a.tag}_overlay.png"), over)
print(f"\nexposure {exposure}, gain {gain}")
print(f"{'x mm':>6} {'y mm':>7} {'dist':>5}  {'median':>6} {'min':>5} {'>=0.5':>6}  {'plate':>5} {'body':>5} {'sharp':>6}"
      f"  {'shift med':>9} {'shift min':>9} {'180':>5}")
for b in out:
    s = np.array(b["scores"])
    print(f"{b['x_mm']:6.1f} {b['y_mm']:7.1f} {b['dist_mm']:5.0f}  {np.median(s):6.2f} {s.min():5.2f} "
          f"{(s >= 0.5).sum():3d}/{len(s):<2d}  {b['plate']:5.0f} {b['body']:5.0f} {b['sharp']:6.3f}"
          f"  {np.median(b['shift_scores']):9.2f} {min(b['shift_scores']):9.2f} {b['turned_score']:5.2f}")
print(f"\n{'x mm':>6} {'y mm':>7}  {'length':>6} {'width':>5} {'len sd':>6} {'wid sd':>6}  {'centre jit':>10} {'yaw sd':>6} {'run IoU':>7}")
for b in out:
    if b["found"]:
        print(f"{b['x_mm']:6.1f} {b['y_mm']:7.1f}  {b['length_mm']:6.2f} {b['width_mm']:5.2f} {b['length_sd_mm']:6.2f} "
              f"{b['width_sd_mm']:6.2f}  {b['centre_jitter_mm']:10.3f} {b['yaw_sd_deg']:6.2f} {b['run_iou']:7.3f}")

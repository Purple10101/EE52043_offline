"""Front / middle / back comparison from the record.py captures: table (results.md) and figures.

    python experiments/2026-11-09_front_back/summarise.py

Parts are matched across captures by their order across the plate (scene y), which the test kept fixed:
turquoise, blue, beige, dark green from y = -130 to y = -160 mm. Writes results.md, figure_parts.png
(each part's outline at each position) and redraws each <tag>_overlay.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
TAGS = [("front_a", "front"), ("mid_a", "middle"), ("back_a", "back")]
PARTS = ["turquoise", "blue", "beige", "dark green"]
CAMERA_XY = np.array([-145.0, -145.0])     # lens over the rig, mm (configs/dataset.toml [rectify] camera_mm)
CAMERA_Z = 200.0                           # lens above the plate top, mm
PX_PER_MM = 10.0


def load(tag):
    d = json.loads((HERE / f"{tag}.json").read_text())
    bodies = sorted(d["bodies"], key=lambda b: -b["y_mm"])
    assert len(bodies) == len(PARTS), f"{tag}: {len(bodies)} bodies, expected {len(PARTS)}"
    return d, dict(zip(PARTS, bodies))


def overlay(tag, name, d, bodies):
    img = cv2.imread(str(HERE / f"{tag}_rect.png"))
    for part, b in bodies.items():
        q = np.round(np.asarray(b["poly"])).astype(np.int32)
        s = np.array(b["scores"])
        cv2.polylines(img, [q], True, (0, 200, 0) if (s >= 0.5).all() else (0, 0, 255), 2, cv2.LINE_AA)
        x, y = int(q[:, 0].max()) + 8, int(q[:, 1].mean())
        for k, t in enumerate([f"{part}: score {np.median(s):.2f} ({(s >= 0.5).sum()}/{len(s)})",
                               f"{b['length_mm']:.2f} x {b['width_mm']:.2f} mm"]):
            for c, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
                cv2.putText(img, t, (x, y + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, w, cv2.LINE_AA)
    title = f"{name} ({tag}): seg_grey_v2, {d['runs']} live runs, exposure {d['exposure']:.0f}, gain {d['gain']:.0f}"
    for c, w in (((255, 255, 255), 5), ((0, 0, 0), 2)):
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, w, cv2.LINE_AA)
    cv2.imwrite(str(HERE / f"{tag}_overlay.png"), img)
    return img


def crop(tag, b, half=60, zoom=3):
    img = cv2.imread(str(HERE / f"{tag}_rect.png"))
    q = np.asarray(b["poly"])
    cx, cy = np.round(q.mean(0)).astype(int)
    c = img[cy - half:cy + half, cx - half:cx + half].copy()
    c = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    corner = np.array([cx - half, cy - half])
    cv2.polylines(c, [np.round((q - corner + 0.5) * zoom - 0.5).astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
    s = np.array(b["scores"])
    for k, t in enumerate([f"score {np.median(s):.2f} (min {s.min():.2f})", f"{b['length_mm']:.2f} x {b['width_mm']:.2f} mm"]):
        for col, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
            cv2.putText(c, t, (6, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, w, cv2.LINE_AA)
    return c


def main():
    data = {tag: load(tag) for tag, _ in TAGS}
    for tag, name in TAGS:
        overlay(tag, name, *data[tag])

    # Figure: rows are parts, columns front / middle / back.
    head = lambda t, w: cv2.putText(np.full((36, w, 3), 255, np.uint8), t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 0), 2, cv2.LINE_AA)
    rows = []
    for part in PARTS:
        cells = [crop(tag, data[tag][1][part]) for tag, _ in TAGS]
        rows.append(np.hstack([np.full((cells[0].shape[0], 160, 3), 255, np.uint8)] + cells))
        cv2.putText(rows[-1], part, (8, cells[0].shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    w = rows[0].shape[1] - 160
    top = np.hstack([np.full((36, 160, 3), 255, np.uint8)] + [head(f"{n} ({np.mean([b['dist_mm'] for b in data[t][1].values()]):.0f} mm from lens)", w // 3)
                                                             for t, n in TAGS])
    cv2.imwrite(str(HERE / "figure_parts.png"), np.vstack([top] + rows))

    # Table.
    out = ["| Part | Position | Lens dist (mm) | Score median | Score min | Found ≥ 0.5 | Length (mm) | Width (mm) "
           "| Centre jitter (mm) | Yaw SD (°) | Run-to-run IoU | Local plate grey | Edge sharpness | Expected extra length (mm) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for part in PARTS:
        for tag, name in TAGS:
            b = data[tag][1][part]
            s = np.array(b["scores"])
            ground = np.hypot(b["x_mm"] - CAMERA_XY[0], b["y_mm"] - CAMERA_XY[1])
            extra = 2.4 * ground / CAMERA_Z     # a 2.4 mm body's top projected along the line of sight
            out.append(f"| {part} | {name} | {b['dist_mm']:.0f} | {np.median(s):.2f} | {s.min():.2f} | "
                       f"{(s >= 0.5).sum()}/{len(s)} | {b['length_mm']:.2f} | {b['width_mm']:.2f} | "
                       f"{b['centre_jitter_mm']:.3f} | {b['yaw_sd_deg']:.2f} | {b['run_iou']:.3f} | {b['plate']:.0f} | "
                       f"{b['sharp']:.3f} | {extra:.2f} |")
    out.append("")
    out.append("| Position | Score median (mean of parts) | Lowest single-run score | Centre jitter (mean, mm) "
               "| Run-to-run IoU (mean) | Length (mean, mm) | Width (mean, mm) | Local plate grey |")
    out.append("|---|---|---|---|---|---|---|---|")
    for tag, name in TAGS:
        bs = list(data[tag][1].values())
        m = lambda k: np.mean([b[k] for b in bs])
        out.append(f"| {name} | {np.mean([np.median(b['scores']) for b in bs]):.2f} | "
                   f"{min(min(b['scores']) for b in bs):.2f} | {m('centre_jitter_mm'):.3f} | {m('run_iou'):.3f} | "
                   f"{m('length_mm'):.2f} | {m('width_mm'):.2f} | {m('plate'):.0f} |")
    (HERE / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

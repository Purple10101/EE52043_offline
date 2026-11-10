"""Sheet of the cut-off bodies in the seg_objects_v2 training and validation sets, as the model saw them.

    python experiments/2026-11-09_seg_objects_v1_test/cut_samples.py

A label counts as cut when it ends on the plate edge (within 1.5 px) where its frame's body would
overhang: the labels of synth_cut_v1 / _val are what the plate mask leaves. Each tile is the model's
input around the body (rectify.model_image, made grey as the model gets it), 30 × 20 mm, enlarged 3×;
the label in magenta, the plate edge dotted in yellow. The first row of each set is the far (+x) edge.
Writes cut_samples.jpg.
"""
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, manifest
from robot_offline.pre import rectify

HERE = Path(__file__).parent
PER_SET = 12


def cut_labels(sample, inside):
    out = []
    for inst in sample["instances"]:
        p = np.asarray(inst["polygon"], float)
        m = np.zeros(inside.shape, np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        ring = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool) & ~inside
        if ring.sum() > 15 and inst.get("kind", "resistor") == "resistor":
            out.append(p)
    return out


def tile(view, p, edge):
    c = p.mean(0)
    w, h = 300, 200
    x0, y0 = int(c[0] - w / 2), int(c[1] - h / 2)
    g = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    pad = cv2.copyMakeBorder(g, h, h, w, w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    crop = pad[y0 + h:y0 + 2 * h, x0 + w:x0 + 2 * w].copy()
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for q in edge:
        q = (q - (x0, y0)) * 3
        for a, b in zip(q[::4], q[2::4]):
            cv2.line(big, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 220, 255), 1)
    cv2.polylines(big, [np.round((p - (x0, y0)) * 3).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    return big


def main():
    ds = config.load("dataset")
    rs = rectify.Settings.from_config(ds)
    inside = rectify.plate_mask(rs)
    edge, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = [e.reshape(-1, 2).astype(float) for e in edge]
    samples = manifest.load(ds["labels"]["rules_version"])["samples"]
    rows = []
    for name in ("synth_cut_v1", "synth_cut_v1_val"):
        far, other, counts = [], [], [0, 0, 0]
        for s in (s for s in samples if s["session"] == name):
            cuts = cut_labels(s, inside)
            counts[0] += sum(i.get("kind", "resistor") == "resistor" for i in s["instances"])
            counts[1] += len(cuts)
            if not cuts or (len(far) >= PER_SET // 2 and len(other) >= PER_SET // 2):
                continue
            view = rectify.model_image(s, cv2.imread(str(config.DATA_DIR / s["image"])), rs)
            for p in cuts:
                is_far = p[:, 0].max() >= inside.shape[1] - 102
                bucket = far if is_far else other
                if len(bucket) < PER_SET // 2:
                    bucket.append(tile(view, p, edge))
        print(f"{name}: {counts[1]} of {counts[0]} resistor labels end on the plate edge")
        tiles = far + other
        while len(tiles) % 6:
            tiles.append(np.full_like(tiles[0], 255))
        title = np.full((34, 6 * tiles[0].shape[1] + 5 * 6, 3), 255, np.uint8)
        cv2.putText(title, f"{name}: cut resistors as the model saw them (grey, masked); first row far (+x) edge, "
                           f"second other edges. {counts[1]} of {counts[0]} resistor labels are cut",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(title)
        for i in range(0, len(tiles), 6):
            rows.append(np.hstack([np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255) for t in tiles[i:i + 6]])[:, :title.shape[1]])
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(HERE / "cut_samples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

"""Crops of every miss and false detection of a model on the real test frames, at conf 0.5, and one
contact sheet per condition.

    python experiments/2026-11-09_seg_objects_v1_test/errors.py [model name, default seg_objects_v1]

Blue outline: a labelled object the model missed (no prediction at IoU 0.5). Red: a false detection
(a prediction matching no label). Each crop is the model's own input (rectified, plate-masked) in colour.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from robot_offline import config, manifest
from robot_offline.inf import tiling
from robot_offline.pre import rectify
from robot_offline.tools import evaluate as ev

HERE = Path(__file__).parent
CONDITIONS = {  # sheet name: sessions
    "room_light": ["2026-10-23_a", "2026-10-23_b", "2026-10-29_ref", "2026-11-06_b", "2026-11-06_c"],
    "hard_light": ["2026-11-06_a"],
    "empty_plate_arm": ["2026-11-05_live"],
}


def crop(image, poly, colour, pad=60):
    p = np.round(np.asarray(poly)).astype(np.int32)
    x, y, w, h = cv2.boundingRect(p)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    c = image[y0:y + h + pad, x0:x + w + pad].copy()
    cv2.polylines(c, [p - [x0, y0]], True, colour, 2)
    return c


def sheet(paths, dst, size=240, cols=6):
    tiles = []
    for f in paths:
        im = cv2.imread(str(f))
        s = size / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        t = np.full((size + 18, size, 3), 255, np.uint8)
        t[:im.shape[0], :im.shape[1]] = im
        cv2.putText(t, f.stem[-40:], (2, size + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
        tiles.append(t)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(dst), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="seg_objects_v1")
    args = parser.parse_args()
    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    samples = [s for s in manifest.load(ds["labels"]["rules_version"])["samples"]
               if s["session"] in ds["splits"]["test_sessions"]]
    weights = config.ROOT / "models" / args.model / "best.pt"
    res = ev.evaluate_samples(YOLO(str(weights)), samples, conf=0.5, grey=tiling.wants_grey(weights))
    out = HERE / f"errors_{args.model}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    made = {}
    for s, r in zip(samples, res):
        found = {m["label_index"] for m in r["matched"]}
        used = {m["prediction_index"] for m in r["matched"]}
        image = None
        stem = f"{s['session']}_{Path(s['image']).stem}"
        for i, poly in enumerate(manifest.polygons(s)):
            if i not in found:
                image = image if image is not None else rectify.model_image(s, cv2.imread(str(config.DATA_DIR / s["image"])), st)
                f = out / f"{stem}_miss_{s['instances'][i].get('kind', 'resistor')}{i}.jpg"
                cv2.imwrite(str(f), crop(image, poly, (255, 0, 0)))
                made.setdefault(s["session"], []).append(f)
        for j, (poly, score) in enumerate(zip(r["polygons"], r["scores"])):
            if j not in used:
                image = image if image is not None else rectify.model_image(s, cv2.imread(str(config.DATA_DIR / s["image"])), st)
                tag = "false" if s.get("labelled") == "objects" else "unlabelled"
                f = out / f"{stem}_{tag}{j}_{score:.2f}.jpg"
                cv2.imwrite(str(f), crop(image, poly, (0, 0, 255)))
                made.setdefault(s["session"], []).append(f)
    for name, sessions in CONDITIONS.items():
        sheet([f for sess in sessions for f in made.get(sess, [])], HERE / f"sheet_{args.model}_{name}.jpg")
    print({k: len(v) for k, v in made.items()})


if __name__ == "__main__":
    main()

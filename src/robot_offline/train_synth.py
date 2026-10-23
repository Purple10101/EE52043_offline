"""Trains on synthetic sets only, then shows the model on real frames.

1. Exports the synthetic sets as plate-masked tiles to data/yolo_synth/: --train sets as train,
   --val sets as val (`export_yolo` puts every synthetic set in train, which leaves no validation).
2. Trains with configs/train.toml (batch and workers there suit the 6 GB GPU and 16 GB of RAM).
3. Runs the best weights on the real rectified frames (masked the same way) and draws the
   predictions, to runs/<name>/real_preview/. The real frames have no labels yet, so this is for
   looking at, not a score.

    python -m robot_offline.train_synth
    python -m robot_offline.train_synth --epochs 60 --name synth_only_v1
    python -m robot_offline.train_synth --train synth --val synth_val --no-preview

Generate the sets first with `python -m robot_offline.synth --name synth` and
`python -m robot_offline.synth --name synth_val --seed 1 --frames 100`.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config, export_yolo, manifest, rectify, tiling, train


def preview(weights, out_dir, folders, conf=0.25):
    """Draws the model's predictions on rectified real frames; writes each frame and a sheet."""
    from ultralytics import YOLO

    ds = config.load("dataset")
    st = rectify.Settings.from_config(ds)
    area, tiles_cfg = ds["work_area"], ds["tiles"]
    work_area = (area["x0"], area["y0"], area["width"], area["height"])
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for folder in folders:
        for p in sorted(folder.glob("*.png")):
            img = rectify.mask_outside_plate(cv2.imread(str(p)), st)
            frame_tiles = tiling.tiles(work_area, (img.shape[1], img.shape[0]), tiles_cfg["size"], tiles_cfg["overlap"])
            polygons, scores = tiling.predict_frame(model, img, frame_tiles, conf=conf)
            for poly, score in zip(polygons, scores):
                pts = np.round(poly).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"{score:.2f}", tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
            name = f"{folder.parent.name}_{p.stem}"
            cv2.imwrite(str(out_dir / f"{name}.jpg"), img)
            thumb = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"{name}: {len(polygons)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            thumbs.append(thumb)
            print(f"  {name}: {len(polygons)} bodies")
    if thumbs:
        while len(thumbs) % 5:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + 5]) for i in range(0, len(thumbs), 5)]
        cv2.imwrite(str(out_dir / "sheet.jpg"), np.vstack(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.train_synth",
                                     description="Train on synthetic sets only, then preview on real frames.")
    parser.add_argument("--train", nargs="+", default=["synth"], help="synthetic sets to train on (synth)")
    parser.add_argument("--val", nargs="+", default=["synth_val"], help="synthetic sets to validate on (synth_val)")
    parser.add_argument("--name", default="synth_only", help="run name: runs/<name>/ (synth_only)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15, help="stop after this many epochs without improvement")
    parser.add_argument("--no-preview", action="store_true", help="skip drawing predictions on real frames")
    args = parser.parse_args(argv)

    ds = config.load("dataset")
    samples = manifest.load(ds["labels"]["rules_version"])["samples"]
    have = {s["session"] for s in samples}
    missing = [n for n in args.train + args.val if n not in have]
    if missing:
        parser.error(f"not in the manifest: {missing}; generate them with `python -m robot_offline.synth --name <set>`")

    print("Exporting plate-masked tiles...", flush=True)
    splits = {**{n: "train" for n in args.train}, **{n: "val" for n in args.val}}
    yaml_path = export_yolo.export_from_config(samples, splits, config.DATA_DIR / "yolo_synth")

    print("Training...", flush=True)
    run_dir = train.train(yaml_path, name=args.name, epochs=args.epochs, patience=args.patience)
    best = run_dir / "weights" / "best.pt"
    print(f"Run folder: {run_dir}\nBest weights: {best}", flush=True)

    if not args.no_preview and best.exists():
        folders = sorted(p for p in (config.DATA_DIR / "captures").glob("*/rectified") if p.is_dir())
        if folders:
            print("Predictions on the real frames:", flush=True)
            preview(best, run_dir / "real_preview", folders)
            print(f"See {run_dir / 'real_preview' / 'sheet.jpg'}")


if __name__ == "__main__":     # required: on Windows the data-loading workers re-import this module
    main()

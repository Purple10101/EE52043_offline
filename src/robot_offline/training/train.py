"""Fine-tunes a COCO-pretrained YOLO-seg model on data/yolo/ with configs/train.toml.

Output goes to runs/<name>/. Records the tile size, overlap and label rules_version next to the
weights, for the hand-off to the robot repo.

    python -m robot_offline.training.train
    python -m robot_offline.training.train --data data/yolo/dataset.yaml --name trial --epochs 5

Pretrained weights are downloaded once to weights/ in the repo root (not in git).
"""

import argparse
import json
from pathlib import Path

from robot_offline import config

WEIGHTS_DIR = config.ROOT / "weights"


def pretrained(name):
    """Path for a named ultralytics checkpoint in weights/; ultralytics downloads it there."""
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return WEIGHTS_DIR / name


def settings(augment=True, **overrides):
    """Training arguments: [train] from configs/train.toml, then [augment] if augment, then overrides.

    With augment=False the ultralytics default augmentation is used.
    """
    cfg = config.load("train")
    tile_size = config.load("dataset")["tiles"]["size"]
    kw = dict(cfg["train"])
    if augment:
        kw.update(cfg["augment"])
    kw.update(overrides)
    if kw["imgsz"] != tile_size:
        raise config.ConfigError(f"train imgsz {kw['imgsz']} must equal the tile size {tile_size}")
    kw["project"] = str(config.ROOT / kw["project"])
    return kw


def train(data_yaml, weights=None, augment=True, grey=False, **overrides):
    """Trains from weights (default [model] weights in configs/train.toml). Returns the run folder.

    grey: the data is greyscale (export with grey=True): hue and saturation changes are switched off,
    and handoff.json records "input": "grey" so every tool feeds the model grey (tiling.wants_grey).
    """
    from ultralytics import YOLO

    weights = weights or config.load("train")["model"]["weights"]
    if not Path(weights).exists():
        weights = pretrained(weights)
    if grey:
        overrides = {"hsv_h": 0.0, "hsv_s": 0.0, **overrides}
    kw = settings(augment, **overrides)
    model = YOLO(str(weights))
    model.train(data=str(data_yaml), exist_ok=True, **kw)
    run_dir = Path(model.trainer.save_dir)

    ds = config.load("dataset")
    (run_dir / "handoff.json").write_text(json.dumps({
        "weights": "weights/best.pt",
        "start_weights": Path(weights).name,
        "tile_size": ds["tiles"]["size"],
        "tile_overlap": ds["tiles"]["overlap"],
        "rules_version": ds["labels"]["rules_version"],
        "classes": list(ds["labels"]["classes"]),
        "train_args": {k: v for k, v in kw.items() if k != "project"},
        **({"input": "grey"} if grey else {}),
    }, indent=2))
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.training.train",
        description="Fine-tune a YOLO-seg model on the exported tile dataset.",
    )
    parser.add_argument("--data", type=Path, default=config.DATA_DIR / "yolo" / "dataset.yaml")
    parser.add_argument("--name", help="run name; default [train] name in configs/train.toml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--grey", action="store_true", help="the data was exported grey (export_yolo --grey)")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in (("name", args.name), ("epochs", args.epochs)) if v is not None}
    print(f"Run folder: {train(args.data, grey=args.grey, **overrides)}")


if __name__ == "__main__":
    main()

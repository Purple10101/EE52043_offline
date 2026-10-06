"""Exports the manifest as a YOLO-seg dataset of tiles in data/yolo/.

Writes images/{train,val,test}, labels/{train,val,test} (class 0, normalised polygon per line)
and dataset.yaml. Splits by session from configs/dataset.toml; synthetic frames are train only.

    python -m robot_offline.export_yolo

Tiles keep their source format: PNG captures give PNG tiles, synthetic JPEG frames give JPEG tiles
at quality 100, so they are not visibly compressed twice. Tiles with no bodies are kept, with an
empty label file, so the model sees empty table. The output folder is replaced on every export.
"""

import argparse
import shutil
from pathlib import Path

import cv2

from robot_offline import config, manifest, tiling

SPLITS = ("train", "val", "test")


def splits_from_config(samples, dataset_cfg):
    """{session: split} from configs/dataset.toml; synthetic sessions and the rest are train."""
    val, test = set(dataset_cfg["splits"]["val_sessions"]), set(dataset_cfg["splits"]["test_sessions"])
    out = {}
    for s in samples:
        session = s["session"]
        if s["source"] == "synthetic":
            out[session] = "train"
        else:
            out[session] = "test" if session in test else "val" if session in val else "train"
    return out


def yolo_line(points, size):
    """One label line: class 0 and the polygon normalised to the tile.

    Ultralytics scales the coordinates back by the image size, truncates them to integers and fills
    them with cv2.fillPoly, which puts pixel (c, r) at (c, r) as the manifest does. Adding half a
    pixel first turns that truncation into rounding, so training masks are not shifted up-left.
    """
    coords = ((points + 0.5) / size).clip(0, 1)
    return "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)


def export(samples, splits, out_dir, work_area, tile_size, overlap, min_visible_fraction,
           class_names=("resistor_body",)):
    """Writes a YOLO-seg tile dataset of samples to out_dir. Returns the path of dataset.yaml.

    splits: {session: "train" | "val" | "test"}; samples of other sessions are skipped.
    work_area: x0, y0, width, height in full-frame px.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    counts = {split: [0, 0] for split in SPLITS}   # tiles, bodies
    for sample in samples:
        split = splits.get(sample["session"])
        if split is None:
            continue
        src = config.DATA_DIR / sample["image"]
        image = cv2.imread(str(src))
        if image is None:
            raise FileNotFoundError(src)
        polygons = manifest.polygons(sample)
        stem = f"{sample['session']}_{Path(sample['image']).stem}"
        frame_tiles = tiling.tiles(work_area, (sample["width"], sample["height"]), tile_size, overlap)
        for i, tile in enumerate(frame_tiles):
            name = f"{stem}_t{i:02d}"
            crop = image[tile.window]
            if src.suffix.lower() in (".jpg", ".jpeg"):
                cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), crop)
            clipped = tiling.clip_polygons(polygons, tile, min_visible_fraction)
            lines = [yolo_line(p, tile_size) for p in clipped]
            (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            counts[split][0] += 1
            counts[split][1] += len(clipped)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n"
    )
    for split, (n_tiles, n_bodies) in counts.items():
        print(f"  {split}: {n_tiles} tiles, {n_bodies} bodies")
    return yaml_path


def export_from_config(samples, splits, out_dir):
    """export() with the work area, tiling and classes from configs/dataset.toml."""
    ds = config.load("dataset")
    area, tiles = ds["work_area"], ds["tiles"]
    return export(
        samples, splits, out_dir,
        work_area=(area["x0"], area["y0"], area["width"], area["height"]),
        tile_size=tiles["size"], overlap=tiles["overlap"],
        min_visible_fraction=tiles["min_visible_fraction"],
        class_names=tuple(ds["labels"]["classes"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.export_yolo",
        description="Export the manifest as a YOLO-seg dataset of tiles.",
    )
    parser.add_argument("--out", type=Path, default=config.DATA_DIR / "yolo")
    args = parser.parse_args(argv)
    ds = config.load("dataset")
    samples = manifest.load(ds["labels"]["rules_version"])["samples"]
    yaml_path = export_from_config(samples, splits_from_config(samples, ds), args.out)
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()

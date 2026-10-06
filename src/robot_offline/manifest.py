"""The label store, data/manifest.json: every frame, its session, source and body polygons.

Polygons are N×2 float arrays in full-frame pixel coordinates, the same form as the robot repo's
outline_px. Splits come from configs/dataset.toml by session, not from the manifest.
Format: data/README.md. Labelling rules: docs/labelling.md.

The file is written one sample per line, so it stays readable and diffs cleanly.
"""

import json
import os
from pathlib import Path

import numpy as np

from robot_offline.config import DATA_DIR

VERSION = 1
PATH = DATA_DIR / "manifest.json"


def new(rules_version):
    return {"version": VERSION, "rules_version": rules_version, "samples": []}


def load(rules_version, path=PATH):
    """Reads the manifest, or returns an empty one if the file does not exist yet.

    Raises ValueError if the file's format version or label rules version differs from ours.
    """
    path = Path(path)
    if not path.exists():
        return new(rules_version)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise ValueError(f"{path}: manifest version {data.get('version')}, expected {VERSION}")
    if data.get("rules_version") != rules_version:
        raise ValueError(
            f"{path}: labelled under rules_version {data.get('rules_version')}, config says "
            f"{rules_version}; relabel or drop the old frames first (docs/labelling.md)"
        )
    return data


def save(manifest, path=PATH):
    """Writes atomically, so a crash mid-write cannot lose the hand-fixed labels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in manifest.items() if k != "samples"}
    lines = [json.dumps(s, separators=(",", ":")) for s in manifest["samples"]]
    text = json.dumps(head)[:-1] + ', "samples": [\n' + ",\n".join(lines) + "\n]}\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def replace_samples(manifest, prefix, samples):
    """Drops every sample whose image path starts with prefix, then appends samples."""
    kept = [s for s in manifest["samples"] if not s["image"].startswith(prefix)]
    manifest["samples"] = kept + list(samples)


def polygon_to_json(points, decimals=2):
    return [[round(float(x), decimals), round(float(y), decimals)] for x, y in points]


def polygons(sample):
    """The sample's instance polygons as N×2 float arrays."""
    return [np.asarray(inst["polygon"], dtype=float) for inst in sample["instances"]]

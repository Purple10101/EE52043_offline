"""Deletes synthetic sets: their folders in data/synthetic/ and their samples in data/manifest.json.

    python -m robot_offline.clean_synth                    # every set
    python -m robot_offline.clean_synth --name synth_try   # one set (repeat --name for more)

Only data/synthetic/ and manifest samples whose source is "synthetic" are touched; real captures
and their labels never are.
"""

import argparse
import shutil

from robot_offline import config, manifest


def clean(names=None):
    """Removes the named sets, or every set if names is None. Returns the set names removed."""
    root = config.DATA_DIR / "synthetic"
    folders = {p.name: p for p in root.iterdir() if p.is_dir()} if root.exists() else {}

    labels = None
    if manifest.PATH.exists():
        labels = manifest.load(config.load("dataset")["labels"]["rules_version"], manifest.PATH)
    listed = {s["session"] for s in labels["samples"] if s["source"] == "synthetic"} if labels else set()

    found = set(folders) | listed
    targets = found if names is None else set(names)
    missing = targets - found
    if missing:
        raise SystemExit(f"no synthetic set named {', '.join(sorted(missing))}")

    for name in sorted(targets & set(folders)):
        shutil.rmtree(folders[name])
    if labels is not None:
        labels["samples"] = [
            s for s in labels["samples"]
            if not (s["source"] == "synthetic" and s["session"] in targets)
        ]
        manifest.save(labels, manifest.PATH)
    return sorted(targets)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m robot_offline.clean_synth",
        description="Delete synthetic sets and their manifest samples.",
    )
    parser.add_argument("--name", action="append",
                        help="set to delete; repeat for more; default every set")
    args = parser.parse_args(argv)
    removed = clean(args.name)
    print(f"Removed {len(removed)} synthetic set(s): {', '.join(removed) or 'none'}")


if __name__ == "__main__":
    main()

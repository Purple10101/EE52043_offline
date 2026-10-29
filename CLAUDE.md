# CLAUDE.md

## Commit message format (mandatory)

Every commit subject line must be:

```
YYYYMMDD - description
```

- `YYYYMMDD` is the date of the commit, e.g. `20261006 - initial scaffold`.
- A space, hyphen, space separates the date from a short description.
- This is a standing rule: never commit with any other subject format. If a commit is made in
  the wrong format, reword it (amend/rebase and force-push if already pushed) straight away.

## Project

Offline data collection and training for the vision of the EE52043 resistor sorter (sibling repo
`EE52043`). Trains a single-class instance segmentation model for resistor bodies. See `README.md`
and `docs/plan.md`. Also researches reading resistor values from the colour bands, in
`robot_offline.post.bands`, kept separate from the segmentation pipeline; see `docs/classification.md`.

The label format must stay compatible with the robot repo's Detection: one N×2 polygon per resistor
body, in full-frame pixel coordinates, leads excluded.

## Layout

`src/robot_offline/` is split by what a module does: `pre/` (camera, plate calibration,
rectification, white balance, exposure), `inf/` (the model on tiles, live detection), `post/`
(skeleton, centroid, band reading), `tools/` (capture, evaluation) and `training/` (synthetic data,
labels, export, training). Training code goes only in `training/`, and nothing outside it imports
from it (`tests/test_imports.py` checks). `config` and `manifest` are shared, at the top level.
Tests mirror the same parts under `tests/`.

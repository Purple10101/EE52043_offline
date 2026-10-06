# robot_offline

Offline data collection and training for the EE52043 resistor sorter's vision. It trains a
single-class instance segmentation model that outlines each resistor body (leads excluded) in the
overhead camera's frames. The robot repo (`EE52043`, `robot_control/vision/`) loads the trained
weights and turns each mask into a Detection's `outline_px`.

Design and data plan: [docs/plan.md](docs/plan.md). Labelling rules: [docs/labelling.md](docs/labelling.md).

## Quick start: synthetic data

Run from the repo root in PowerShell. One-time setup (synthetic data needs no torch or camera; see
[Setup](#setup) for the full install):

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install numpy opencv-python shapely pytest
pip install -e . --no-deps
```

In each new terminal, run `.venv\Scripts\activate` first.

**Generate a set.**

```
python -m robot_offline.synth --frames 20 --name synth_try   # quick look, about 10 s
python -m robot_offline.synth                                # full set: 100 frames, about 2 min, about 160 MB
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | `synth` | Set name. Frames go to `data/synthetic/<name>/`. |
| `--frames` | `frames` in `configs/dataset.toml` | Number of frames. |
| `--seed` | `seed` in `configs/dataset.toml` | Same seed and settings give the same frames. |
| `--workers` | `4` | Parallel processes; `1` runs in the current process (easier to debug). |
| `--preview` | `5` | Write outlined copies of the first N frames to `preview/`. |
| `--backgrounds` | none | Folder of empty-table frames to draw on instead of the white backdrops. |

**Check it.** Open `data/synthetic/<name>/preview/`: each body should have a tight green outline
with the leads outside it. Frames are `data/synthetic/<name>/NNNNN.jpg`; labels are the
`synthetic/<name>/` samples in `data/manifest.json`.

**Re-run or delete.** Re-running with the same `--name` deletes that set and writes it again.
Sets with other names stay until you delete them:

```
python -m robot_offline.clean_synth                   # delete every synthetic set
python -m robot_offline.clean_synth --name synth_try  # delete one set (repeat --name for more)
```

This removes the set folders and their manifest samples only; real captures and their labels are
never touched.

**Change what is generated** in `configs/dataset.toml`:

- `[work_area]`: the box (`x0`, `y0`, `width`, `height`, in px) that every resistor, leads
  included, stays inside.
- `[synthetic]`: frame count, bodies per frame, the minimum gap between bodies (`min_gap_mm`),
  lead length and bends, touching/overlapping pairs (off by default), empty frames, clutter and
  JPEG quality.
- Lighting, backdrop colours, resistor sizes and band colours are constants at the top of
  `src/robot_offline/synth.py`.
- `[scale] px_per_mm`: sets the resistor size in pixels.

**Run the tests** with `python -m pytest`.

## Layout

- `src/robot_offline/` - the package, one module per pipeline step:
  - `config` - loads the TOML files in `configs/`.
  - `manifest` - the label store: every frame, its session and its body polygons.
  - `capture` - grabs full-resolution frames from the camera into a session folder.
  - `autolabel` - classical pipeline (empty-table subtraction, opening, contours) for draft labels.
  - `cutouts` - builds the resistor cut-out library from spread-out scenes.
  - `synth` - synthetic frames: resistors drawn onto procedural backdrops or empty-table frames,
    touching and overlapping cases included, with exact labels.
  - `clean_synth` - deletes synthetic sets and their manifest samples (all, or `--name <set>`).
  - `tiling` - frames and polygons to tiles, and tile predictions back to full-frame outlines.
  - `export_yolo` - manifest to a YOLO-seg dataset of tiles plus `dataset.yaml`.
  - `train` - fine-tunes a COCO-pretrained YOLO-seg model.
  - `evaluate` - scores the model on the real test sessions in mm and degrees.
  - `visualise` - draws polygons and predictions over frames.
- `configs/` - camera, dataset and training settings.
- `data/` - images and labels. Not in git; see [data/README.md](data/README.md).
- `runs/` - training output. Not in git.
- `tests/` - tests for the package.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

Install the CUDA build of torch first so ultralytics does not pull in the CPU-only one. Python 3.11+
(`tomllib`).

## Workflow

1. `synth` a first synthetic set; it needs no camera. Check its `preview/` folder.
2. `capture` the empty table and spread-out resistors for each session.
3. `autolabel` the spread-out frames, fix them, and build `cutouts` from them.
4. `capture` real scenes, `autolabel` them, and fix merged or broken outlines by hand.
5. `synth` again with `--backgrounds data/backgrounds/<session>`, so the table is the real one.
6. `export_yolo`, then `train`.
7. `evaluate` on the held-out real sessions.
8. Hand off `runs/<name>/weights/best.pt` with the tile size and label rules version it was trained
   with.

Each module runs as `python -m robot_offline.<module>`.

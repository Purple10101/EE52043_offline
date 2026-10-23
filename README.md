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
python -m robot_offline.synth                                # full set: 100 frames, about 30 s
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | `synth` | Set name. Frames go to `data/synthetic/<name>/`. |
| `--frames` | `frames` in `configs/dataset.toml` | Number of frames. |
| `--seed` | `seed` in `configs/dataset.toml` | Same seed and settings give the same frames. |
| `--workers` | `4` | Parallel processes; `1` runs in the current process (easier to debug). |
| `--preview` | `5` | Write outlined copies of the first N frames to `preview/`. |
| `--backgrounds` | none | Folder of rectified empty-plate images to draw on instead of the drawn plate. |

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

- `[rectify]`: the frame itself. Synthetic frames are drawn as rectified frames: the plate from
  straight above at `px_per_mm`, with a margin.
- `[work_area]`: the box (`x0`, `y0`, `width`, `height`, in rectified px) that every resistor,
  leads included, stays inside: the plate.
- `[synthetic]`: frame count, bodies per frame, the minimum gap between bodies (`min_gap_mm`),
  clusters, lead length and curl, touching/overlapping pairs, empty frames, clutter, screws and
  shavings, the camera position (for how bodies lean in view), the camera simulation (where the
  plate sits in the raw frame, focus, sharpening, MJPG quality) and JPEG quality. The defaults were
  matched to the measured look of real rectified frames.
- Lighting, plate and surround colours, resistor packages and band colours are constants at the
  top of `src/robot_offline/synth.py`.

**Run the tests** with `python -m pytest`.

## Layout

- `src/robot_offline/` - the package, one module per pipeline step:
  - `config` - loads the TOML files in `configs/`.
  - `manifest` - the label store: every frame, its session and its body polygons.
  - `capture` - captures a session with a live top-down view: SPACE saves the empty plate, then
    resistor scenes, then scenes with non-resistor objects mixed in, checking the plate each time.
    `--plan cutouts` captures spread-out objects for the cut-out library instead, outlining each
    object live so you can see they are apart.
  - `rectify` - finds the pick plate in a frame and warps frames to a straight-down view of it,
    with polygons converting both ways; also the lens calibration and plate white balance.
  - `live` - live straight-down view of the plate from the camera, with a 10 mm grid.
  - `live_detect` - runs the model live on the camera (3 per second by default) and draws the
    body outlines on the camera image or the top-down view.
  - `autolabel` - classical pipeline (empty-table subtraction, opening, contours) for draft labels.
  - `cutouts` - builds the resistor cut-out library from spread-out scenes.
  - `synth` - synthetic rectified frames of the plate: resistors, leads and clutter drawn as the
    tower camera sees them, touching and overlapping cases included, with exact labels.
  - `clean_synth` - deletes synthetic sets and their manifest samples (all, or `--name <set>`).
  - `tiling` - frames and polygons to tiles, and tile predictions back to full-frame outlines.
  - `export_yolo` - manifest to a YOLO-seg dataset of tiles plus `dataset.yaml`.
  - `train` - fine-tunes a COCO-pretrained YOLO-seg model.
  - `train_synth` - trains on synthetic sets only (one set for training, another for validation),
    then draws the model's predictions on the real rectified frames for a look.
  - `evaluate` - scores the model on the real test sessions in mm and degrees.
  - `visualise` - draws polygons and predictions over frames.
  - `notebooks/` - experiments; `synth_performance_evaluation.ipynb` trains on the synthetic set
    and scores a COCO-pretrained YOLOv8m-seg before and after tuning.
- `configs/` - camera, dataset and training settings.
- `data/` - images and labels. Not in git; see [data/README.md](data/README.md).
- `runs/` - training output. Not in git.
- `weights/` - downloaded pretrained checkpoints. Not in git.
- `tests/` - tests for the package.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev,notebook]
```

Install the CUDA build of torch first so ultralytics does not pull in the CPU-only one. Python 3.11+
(`tomllib`).

## Workflow

1. `synth` a first synthetic set; it needs no camera. Check its `preview/` folder.
2. `capture` the empty table and spread-out resistors for each session.
3. `autolabel` the spread-out frames, fix them, and build `cutouts` from them.
4. `capture` real scenes, `autolabel` them, and fix merged or broken outlines by hand.
5. Optionally `synth` again with `--backgrounds` set to rectified empty-plate images from a session
   that is not used for validation or testing.
6. `export_yolo`, then `train`.
7. `evaluate` on the held-out real sessions.
8. Hand off `runs/<name>/weights/best.pt` with the tile size and label rules version it was trained
   with.

Each module runs as `python -m robot_offline.<module>`.

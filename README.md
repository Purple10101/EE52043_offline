# robot_offline

Offline data collection and training for the EE52043 resistor sorter's vision. It trains a
single-class instance segmentation model that outlines each resistor body (leads excluded) in the
overhead camera's frames. The robot repo (`EE52043`, `robot_control/vision/`) loads the trained
weights and turns each mask into a Detection's `outline_px`.

Design and data plan: [docs/plan.md](docs/plan.md). Labelling rules: [docs/labelling.md](docs/labelling.md).

## Layout

- `src/robot_offline/` - the package, one module per pipeline step:
  - `config` - loads the TOML files in `configs/`.
  - `manifest` - the label store: every frame, its session and its body polygons.
  - `capture` - grabs full-resolution frames from the camera into a session folder.
  - `autolabel` - classical pipeline (empty-table subtraction, opening, contours) for draft labels.
  - `cutouts` - builds the resistor cut-out library from spread-out scenes.
  - `synth` - copy-paste generator: cut-outs onto empty-table frames, touching cases included.
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

1. `capture` the empty table and spread-out resistors for each session.
2. `autolabel` the spread-out frames, fix them, and build `cutouts` from them.
3. `capture` real scenes, `autolabel` them, and fix merged or broken outlines by hand.
4. `synth` synthetic frames.
5. `export_yolo`, then `train`.
6. `evaluate` on the held-out real sessions.
7. Hand off `runs/<name>/weights/best.pt` with the tile size and label rules version it was trained
   with.

Each module runs as `python -m robot_offline.<module>`.

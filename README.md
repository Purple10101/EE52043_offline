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
python -m robot_offline.training.synth --frames 20 --name synth_try   # quick look, about 10 s
python -m robot_offline.training.synth                                # full set: 100 frames, about 30 s
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
python -m robot_offline.training.clean_synth                   # delete every synthetic set
python -m robot_offline.training.clean_synth --name synth_try  # delete one set (repeat --name for more)
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
  top of `src/robot_offline/training/synth.py`.

**Run the tests** with `python -m pytest`.

## Layout

- `src/robot_offline/` - the package, in five parts. Run a module with
  `python -m robot_offline.<part>.<module>`, e.g. `python -m robot_offline.inf.live_detect`.
  Nothing outside `training/` imports from it (`tests/test_imports.py` checks).
  - `config` - loads the TOML files in `configs/`. Shared.
  - `manifest` - the label store: every frame, its session and its body polygons. Shared.
  - `pre/` - pre-processing, from the camera to the image the model sees:
    - `camera` - opens the camera with `configs/camera.toml`; on-screen helpers for the live windows.
    - `rectify` - finds the pick plate in a frame and warps frames to a straight-down view of it,
      with polygons converting both ways; also the lens calibration, plate white balance and the
      model's masked input.
    - `exposure` - exposure that follows the light, metered on the plate's white top.
    - `live` - live straight-down view of the plate from the camera, with a 10 mm grid.
    - `live_plate` - live check of the plate calibration: the camera's view with the fitted plate
      drawn on it, next to the rectified plate.
  - `inf/` - inference:
    - `tiling` - runs the model over overlapping tiles of a frame and merges the predictions back
      into full-frame outlines.
    - `live_detect` - runs the model live on the camera (3 per second by default) and draws the
      body outlines, with each body's skeleton, spine and centroid, on the camera image or the
      top-down view.
  - `post/` - post-processing of the detected bodies:
    - `skeleton` - per predicted outline: its centroid, its thinned skeleton, and its spine (the
      centre line from end to end, for sampling the bands along), with length, width and yaw.
    - `bands/` - research on reading resistor values from the colour bands (see
      [docs/classification.md](docs/classification.md)):
      - `bands.crops` - straightened, white-balanced body crops cut from the raw camera frame at
        20 px/mm, and the crop dataset in `data/bands/`.
      - `bands.reference` - captures frames of resistors of one known value
        (`python -m robot_offline.post.bands.reference 4k7`) into `data/reference/<date>/`, with
        the bodies found and a preview of each.
  - `tools/` - capturing frames and evaluating performance:
    - `capture` - captures a session with a live top-down view: SPACE saves the empty plate, then
      resistor scenes, then scenes with non-resistor objects mixed in, checking the plate each
      time. `--plan cutouts` captures spread-out objects for the cut-out library instead,
      outlining each object live so you can see they are apart.
    - `evaluate` - scores the model on the real test sessions in mm and degrees.
    - `visualise` - draws polygons and predictions over frames.
  - `training/` - everything that makes the model:
    - `synth` - synthetic rectified frames of the plate: resistors, leads and clutter drawn as the
      tower camera sees them, touching and overlapping cases included, with exact labels.
    - `clean_synth` - deletes synthetic sets and their manifest samples (all, or `--name <set>`).
    - `autolabel` - classical pipeline (empty-table subtraction, opening, contours) for draft
      labels.
    - `cutouts` - builds the resistor cut-out library from spread-out scenes.
    - `export_yolo` - manifest to a YOLO-seg dataset of tiles plus `dataset.yaml`.
    - `train` - fine-tunes a COCO-pretrained YOLO-seg model.
    - `train_synth` - trains on synthetic sets only (one set for training, another for
      validation), then draws the model's predictions on the real rectified frames for a look.
    - `notebooks/` - experiments; `synth_performance_evaluation.ipynb` trains on the synthetic set
      and scores a COCO-pretrained YOLOv8m-seg before and after tuning.
- `configs/` - camera, dataset and training settings.
- `data/` - images and labels. Not in git; see [data/README.md](data/README.md).
- `runs/` - training output. Not in git.
- `weights/` - downloaded pretrained checkpoints. Not in git.
- `tests/` - tests for the package, in the same five parts.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev,notebook]
```

Install the CUDA build of torch first so ultralytics does not pull in the CPU-only one. Python 3.11+
(`tomllib`).

## Live rectification on the rig

`live_plate` shows, live, how the plate is found and what the top-down view comes out as. Use it to
check the rig before capturing or detecting: after moving the tower or the plate, or changing the
lights. It needs the camera and the rig, but not torch or the model (the
[quick start](#quick-start-synthetic-data) install is enough).

**1. Set up the rig** as for [live detection](#live-detection-on-the-rig): the camera plugged in
and nothing else using it, the whole plate in view, its front face included, and steady light.

**2. Run it** from the repo root with the venv active:

```
python -m robot_offline.pre.live_plate
```

The camera takes a few seconds to settle. Then the plate is found from scratch (about 1.5 s) and,
from then on, fitted again to the newest frame over and over in the background (about 0.5 s each),
so the picture keeps running while it works.

The exposure follows the light: it is metered on the plate's white top (once found; before that on
the brightest part of the frame) and stepped to keep the plate between grey 165 and 235, about
where the training frames have it, whatever the room does. Gain is only added once the exposure is
at its longest (1/16 s: the camera can't expose longer at its frame rate), since gain adds noise.
From a dark room it takes about 5 s to settle. Settings are in
`configs/camera.toml [exposure_control]`; `e` switches back to the fixed exposure to compare.

**3. Check it.** On the left, the camera's view of the plate with its model drawn where the fit
puts it. Each line should sit on the real edge:

- green: the outline of the plate's top;
- cyan: the bottom edge of its front face;
- magenta: the crosshair, in its groove.

On the right, the newest frame rectified and white-balanced, with a 10 mm grid: the plate's edges
should sit on the red outline and the crosshair inside the magenta ring.

The text at the top says:

| Line | Meaning |
|---|---|
| `fitted ... ago, in ...` | How fresh the fit is. `NOT FOUND` means the latest frame failed (the reason follows) and the last good fit is shown. |
| `near side by the ...` | What fixed the plate's near side: the `crosshair`, or the `face bottom` when something covers the crosshair. `checks` are how far the features not used are from the fit, in mm; a few tenths at most. |
| `found, rms` | For each edge, the share of it found and how far its points are from the fit. Under about 80% (a shadow, a hand, a lead over it) or over about 1 px is worth a look. |
| `camera ..., plate moved ...` | Where the fit puts the lens (scene mm), its tilt and focal length, and how far the plate's corners have moved in the frame since the first fit. With the tower steady this stays under about a pixel; more means the camera is moving. |

| Key | Action |
|---|---|
| `q` / `Esc` | Quit. |
| `g` | Grid on or off. |
| `z` | The whole camera frame, or zoomed on the plate. |
| `l` | Lock the fit (stop refitting), to watch the plate drift against it: a test of the tower's stiffness. |
| `c` | Start again: find the plate from scratch, focal length included (after moving the tower). |
| `e` | Exposure following the light, or fixed at `configs/camera.toml`'s (to compare). |
| `s` | Save the raw frame, the rectified image and the calibration (`_rectify.json`) to `data/live/`. |

**If something is wrong:**

- *"plate not found"* or `NOT FOUND`: keep hands and leads off the plate's edges, and the crosshair or
  the bottom of its front face in view. A lamp low on one side can hide an edge in shadow.
- The lines sit off the real edges: press `c` to start again. If they still do, save a frame with
  `s` and look at the plate settings in `configs/dataset.toml [rectify]` (the plate's model).

## Live detection on the rig

`live_detect` runs the segmentation model on the tower camera and draws each resistor body's outline
over the live image, with a panel of each body's segmented pixels on the right. It needs the whole
setup: the full install, the trained model, the camera and the rig.

**1. Install.** Follow [Setup](#setup) (the CUDA torch build matters: about 0.2 s per detection on
the laptop's GPU against about 1.1 s on the CPU). Then fetch the trained model, which is stored with
Git LFS:

```
git lfs install
git lfs pull
```

`models/synth_only_v1/best.pt` should now be about 52 MB. If it is a few hundred bytes, it is still
the LFS pointer: run `git lfs pull` again.

**2. Set up the rig.**

- The Arducam B030401 (IMX708, 12 MP) is in the camera tower and plugged in over USB. Close
  anything else using it (the Camera app, Teams, OBS). The camera is found by its 4608 × 2592 mode,
  so its device index does not matter.
- The 100 × 100 mm pick plate is in place with all of it in view, its front face included. The
  top-down warp is calibrated from the plate's far and side edges, its centre crosshair and the
  bottom of its front face (the crosshair can be covered, the face's bottom then stands in), so
  keep hands, cables and resistor leads off the edges.
- Lighting is steady (room lights). Focus, exposure and gain are fixed in `configs/camera.toml`
  and tuned for the tower's 200 mm height: change them only if the tower or camera moves.

**3. Run it** from the repo root with the venv active:

```
python -m robot_offline.inf.live_detect
```

The camera takes a few seconds to settle, then the window opens. The plate is outlined in white
once found, and found again every 5 s in the background, so a nudged camera recovers on its own
without the picture pausing. The exposure follows the light, as in
[live rectification](#live-rectification-on-the-rig), and the model is given the plate at the
brightness it has in training (grey 200) whatever the exposure, so a dim room doesn't weaken its
detections.

| Option | Default | Meaning |
|---|---|---|
| `--weights` | `models/synth_only_v1/best.pt` | The model to run, e.g. `runs/<name>/weights/best.pt` after training your own. |
| `--fps` | `3` | Detections per second. |
| `--conf` | `0.5` | Confidence threshold. Below about 0.3 the plate's rim starts to show up as false bodies. |

| Key | Action |
|---|---|
| `q` / `Esc` | Quit. |
| `v` | Switch between the camera view and the top-down view. |
| `m` | Top-down view as the model sees it (outside the plate filled grey). |
| `+` / `-` | Raise or lower the confidence threshold by 0.05. |
| `c` | Find the plate again now, from scratch. |
| `e` | Exposure following the light, or fixed at `configs/camera.toml`'s (to compare). |
| `s` | Save the shown image to `data/live/detect_<time>.jpg`. |

**If something is wrong:**

- *"plate not found"* in the status line: check that the plate's edges are visible and clear, and
  that the crosshair or the bottom of the plate's front face is, then press `c`. Uneven light (a
  lamp low on one side) can hide an edge in shadow.
- *"no camera accepts 4608 × 2592"*: unplug and replug the camera, and check nothing else has it
  open.
- Outlines lag or the rate is well under `--fps`: torch is probably the CPU build. Check with
  `python -c "import torch; print(torch.cuda.is_available())"`, which should print `True`.

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

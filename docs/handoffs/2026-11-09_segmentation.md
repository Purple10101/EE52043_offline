# Handoff: segmentation, 2026-11-09

For the new session that owns **segmentation** in robot_offline. The session that wrote this
(`robot-offline-01`) now owns **band reading only**. Read CLAUDE.md first; this adds what isn't in the
code or git history. Delete this file once its contents are settled elsewhere.

## Who owns what

| Area | Owner | Files |
|---|---|---|
| Segmentation | **you** | `inf/` (tiling, live_detect, steady), `training/synth.py` *geometry and look of bodies*, `training/train*.py`, `training/export_yolo.py`, `tools/evaluate.py`, `models/seg_*`, `docs/labelling.md`, `configs/train.toml`, `[splits]` `[labels]` `[synthetic]` in `configs/dataset.toml` |
| Band reading | robot-offline-01 | `post/bands/value.py`, `post/bands/learned.py`, `training/band_reader.py`, `tools/read_plate.py`, `docs/classification.md`, `data/bands_synth`, `data/bands_real` |
| Band grouping | another session (robot-offline-cc, may be gone) | `post/bands/grouping.py`, `tests/post/test_grouping.py`. Don't regenerate `data/bands/crops` or `crops.json`: its THRESHOLD 17 was tuned on them. `post.bands.crops` now defaults to seg_grey_v2, so pass `--out` elsewhere if you run it. |
| Plate finding and exposure | shared, change with care | `pre/plate_search.py`, `pre/exposure.py`, `configs/camera.toml`. The only plate search in either repo (see below). |
| Robot repo | ee52043-1e (another session) | `C:\Users\joshp\Y4S1\EE52043`. Message it before editing anything there (ListAgents / SendMessage). |

**Shared seam in `synth.py`:** it also draws the colour bands and writes `bands` / `first_end` per
instance into the manifest; the band reader trains on that (`synth_bands` set). Keep `BAND_RGB`,
`BAND_POSITIONS`, `BAND_WIDTH`, `BAND_BODY_SHOWS`, `Resistor.band_shows`, `Frame.bands` and the
`bands` / `first_end` manifest fields working, and tell robot-offline-01 before changing them. If
bodies stop always being resistors (below), non-resistor bodies need `bands` empty or absent.

## Where segmentation stands

- **Model in use:** `models/seg_grey_v2/best.pt` (Git LFS), YOLOv8m-seg, one class `resistor_body`,
  **greyscale input** (`handoff.json` `"input": "grey"`; every tool reads that via
  `tiling.wants_grey`). Trained on synthetic data only: `synth_v2` (800) + `synth_v2_val` (100), 40
  epochs, 61 min on the RTX 3060 laptop GPU (6 GB). Scores and history: `models/seg_grey_v2/README.md`.
- **Real benchmark** (`test_sessions`, 68 real samples, conf 0.5): new parts 93 of 98 (synth_only_v1
  55), turquoise 138 of 143, 1 false detection. Per group: beige 26/30, blue 51/51, green 2/2, green
  small 11/12, small 3/3. Weak on very dark frames (9 of 12 at plate level 59) and slightly on
  turquoise. Run: `python -m robot_offline.tools.evaluate --weights models/seg_grey_v2/best.pt`
  (prints per-group counts).
- **Tried and dropped:** colour input (`seg_colour_v2`, worse: 87/98, 4 false); continuing grey on
  2,000 more frames (`seg_grey_v3`, worse: 86/98, stopped at epoch 13, not conclusive).
- **Every tool defaults to seg_grey_v2** here and in EE52043 (`robot_control/config/vision.toml`).

### The weakness found today

A dark green, fat-ended part on the white lab bench, 15 live frames in a row at live_detect's
exposure:

| Body | synth_only_v1 | seg_grey_v2 |
|---|---|---|
| Blue (two parts) | 0.91 to 0.93, 15/15 | 0.86 to 0.93, 15/15 |
| Beige | 0.64 to 0.79, 15/15 | 0.87 to 0.89, 15/15 |
| **Dark green** | **never found**, even at 0.05 | **0.18 to 0.78, over 0.5 in 7/15** |

A later recording of 40 runs had the same green at median 0.81 (38/40): the light had changed. At
conf 0.05 it was found in 40/40 runs; only the score swings. Recording:
`C:\Users\joshp\AppData\Local\Temp\claude\a32d06ec59ec8b69\ea0f3afa-3368-4a5f-a5b1-c2bad7080e51\scratchpad\runs40.pkl`
(pickle of 40 `(polygons, scores)`; the scratchpad may be cleared).

### Done today for it: smoothing (`inf/steady.py`, uncommitted)

`Steady`: the model runs at `FLOOR` 0.05; a body shows once it passes conf in 2 of the last 5 runs,
stays while its mean over them is at least 0.25, and goes once it's missing entirely for 2 runs in a
row. It's wired into `live_detect` (`h` toggles it, to compare with single runs). Tests:
`tests/inf/test_steady.py`. **Not yet in EE52043's live tools** (`scripts/resistor_nav.py`,
`robot_control/task/live_plan.py`): bring it there, coordinating with ee52043-1e. It assumes a still
scene (bodies matched within 2 mm): the robot should `clear()` it after the arm moves.

## The open design question: what the class means

The user wants segmentation kept separate from "is it a resistor" (which belongs to band reading).
Today the synthetic data teaches resistor-ness: every positive is resistor-shaped (waisted body,
bands), and TO-92, TO-220, DIP-8, screws, shavings and lead offcuts are drawn as unlabelled
background (`synth._distractors`).

Proposal discussed with the user, **not yet decided**: make the class **an axial-lead body** (a body
with a lead out of each end), defined by shape, not look. Resistors, diodes, axial capacitors and
inductors of any colour count; screws, offcuts, transistors, chips don't. Not "any object": the
pick point and the label rule (body only, leads excluded) rely on that shape, and bolts or shavings
would become pickable. The label format stays (one N×2 polygon per body, full-frame px, leads
excluded), so EE52043's `Detection` still works; the class name changes (e.g. `axial_body`) and
`rules_version` is bumped with `docs/labelling.md`.

**Still to ask the user:** if a diode is on the plate, may the robot pick it and reject it after
measuring it electrically (then this design is enough), or must it leave it alone (then a small
"resistor or not" vision check is needed before the pick, still separate from segmentation)?

## Suggested next steps

1. Get the user's answer above; settle the class definition.
2. One training run that tests it and fixes the green weakness together: synthetic positives with
   varied axial bodies (plain cylinders, no waist, glass diodes, bands optional, any colour, dark
   low-contrast ones), plus a fine-tune on real labelled frames that include the green part.
3. Keep a held-out real set honest. `test_sessions` are never trained on. The green part
   (`green_a`, 10 kΩ) is in `2026-11-06_c`, a test session, so training on it means moving that
   session out of test and finding new held-out frames. Today's sessions `2026-11-09_a`, `_b`, `_c`
   (the lab bench, 3 to 4 parts) are captured but **not labelled** and not in the manifest. The band
   session's remove-and-measure round uses `_c`; you can still label its outlines.
4. Move greyscale training into the repo. It currently lives only in a scratch script. What trained
   seg_grey_v2 was:

   ```python
   samples = manifest.load(ds["labels"]["rules_version"])["samples"]
   out = config.DATA_DIR / "yolo_synth_grey"
   yaml_path = export_yolo.export_from_config(samples, {"synth_v2": "train", "synth_v2_val": "val"}, out)
   for path in (out / "images").rglob("*.jpg"):    # to grey after export, 3 channels, quality 100
       grey = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
       cv2.imwrite(str(path), cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
   run = train.train(yaml_path, name="seg_grey_v2", epochs=40, patience=12, hsv_h=0.0, hsv_s=0.0)
   ```

   A `--grey` option in `train_synth` / `export_yolo` would make it reproducible; write `"input":
   "grey"` into the run's `handoff.json` too (it was added by hand for models/seg_grey_v2).

## Things that bit today

- **Score frames through `rectify.model_input`**, which brings the plate to grey 200 as in
  training. Scores taken any other way were wrong by a lot (6 Nov).
- **Plate finding** is now one API, `pre/plate_search.py`: `Tracker` for a live camera, `find_now` for
  a script, `find` for saved frames. It looks for the plate dim (grey 60 to 140) where it's expected,
  then raises the exposure for the model. Don't add plate search or exposure loops anywhere else;
  extend that module. An exposure change shows on the 6th frame read (`settle_frames` 6).
- **The camera:** check the Arducam is plugged in and that nothing else holds it (live_detect, the
  user's own runs) before a capture; a capture with a hand in view averages into a blur.
- **GPU:** the laptop has one 6 GB GPU, so one training run at a time. The band reader trains on the CPU,
  so the two can run together.

## State of the repo

Nothing has been committed since `928c641` (29 Oct). The working tree holds both sessions' work plus
the grouping session's and the user's own. Commit only when the user asks, subject
`YYYYMMDD - description`, and split commits by area; robot-offline-01 can say which files are band
reading. 236 tests passed at the time of writing. EE52043 is uncommitted too (137 tests passed); its
`resistor_nav`, `tip_view`, `camera_calibrate plate`, `live_plan` and `twin_live` were changed to the
new plate search but are untested on the camera.

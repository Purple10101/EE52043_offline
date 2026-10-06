# Plan

**Status:** Draft

## Goal

A model that returns one polygon per resistor body in an overhead frame, leads excluded, accurate
enough that the robot repo's Detection gets the right centre, yaw, length and width, and that
touching resistors come out as separate bodies.

## Contract with the robot repo

- Input: one full frame from the IMX708 at 4608 × 2592, same camera settings as `configs/camera.toml`.
- Output: a list of N×2 polygons in full-frame pixel coordinates. This is `outline_px`; the robot
  repo maps it to metres and builds the Detection.
- Inference runs on 1024 px tiles over the work area and merges results with `tiling`. The robot
  repo uses the same tiling code (copy the module or depend on this package).
- Hand-off: `best.pt` plus the tile size, overlap and label `rules_version` it was trained with.

## Why tiles

At about 9.8 px/mm a 1/4 W body is about 24 × 62 px. Downscaling the whole frame to 640 would shrink
it to about 3 × 9 px. Tiles at full resolution keep the outline sharp and give about 10 training
samples per frame. YOLO-seg masks are at 1/4 of the input resolution (about 0.4 mm here). If that
proves too coarse, Mask R-CNN (28 × 28 mask per body) is the fallback.

## Data plan

| | Amount |
|---|---|
| Proof of concept | ~50 real frames (~750 bodies) |
| Usable | 150–250 real frames, or 50–100 real plus ~100 synthetic |
| Test | ~25 real frames from separate sessions, never synthetic |

Coverage matters more than count. The real frames must include:

- every band colour, including black, brown, silver and gold, against the backdrop;
- beige carbon film and blue metal film, 1/4 W and 1/2 W;
- touching side by side and end to end, crossed leads, leads bent over the body;
- bodies at the edge of the work area and across tile boundaries;
- some lighting variation, unless the lighting is fully controlled;
- empty and near-empty tables, to measure false detections.

## Labelling, cheapest first

1. **Synthetic.** `synth` draws resistors (shaded bodies, bands, leads, shadows) at random poses,
   including touching and overlapping, onto procedural backdrops now and onto real empty-table
   frames once we have them. Labels are exact. Later, paste real cut-outs from spread-out scenes.
2. **Auto-label and fix.** `autolabel` real frames, then fix only the merged or broken outlines.
3. **SAM-assisted** in CVAT or Label Studio, one click per body, for frames autolabel handles badly.

## Evaluation

Run on the real test sessions only. Match predictions to labels, then report in robot units:

- centre error (mm), yaw error (deg), length and width error (mm);
- merges (one prediction over several bodies) and splits (one body in several predictions);
- missed bodies and false detections;
- mask IoU, for comparison with the fibre project.

## Order of work

1. Scaffold, `config`, `manifest`, `visualise`, a first `synth` set (no camera needed).
2. Camera arrives (week of 12 Oct): `capture`, then the backdrop and lighting session.
3. `autolabel` and `cutouts` on spread-out scenes.
4. Real scene captures, label fixing.
5. `tiling` and `export_yolo`, then a first `train` on real data only.
6. `synth`, retrain with synthetic added, compare on the test set.
7. `evaluate` and hand off.

## Done when

- [ ] On the real test set, centre error under 1 mm and yaw error under 3° for 95% of bodies.
- [ ] No merges on the touching-pairs frames in the test set.
- [ ] No false detections on the empty-table test frames.
- [ ] Full-frame inference with tiling runs on the laptop in under 1 s.

## Open points

- Backdrop colour and lighting: decided in the first camera session.
- Will resistors be placed touching or overlapping? Decides how much synthetic touching data we need.
- Camera fixed to the robot base or the table (staff question). A camera move means new sessions.

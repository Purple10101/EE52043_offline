# Plan

**Status:** Draft

## Goal

A model that returns one polygon per resistor body, leads excluded, accurate enough that the robot
repo's Detection gets the right centre, yaw, length and width, and that touching resistors come out
as separate bodies.

## Viewpoint and rectification

The camera can't be straight above the pick plate, because the arm needs that space. It sits on a
tower on the rig's corner block, 200 mm above the plate and aimed at its centre at 36° from vertical.
`../camera_tower/README.md` is the as-designed record (`docs/camera_tower_brief.md` was the brief).

Every real frame is converted to a straight-down view of the plate before anything else uses it, by
`rectify`:

1. **Undistort.** Remove lens distortion using the camera's calibration (intrinsics), measured once
   with a checkerboard at the locked focus. Not measured yet; until then frames are not undistorted.
2. **Warp.** A homography maps the plate surface to a top-down image at a fixed scale. It is
   measured per session from the plate's four top edges (fitted lines, so its rounded corners don't
   matter), against the plate's corners in the rig scene frame (`configs/dataset.toml [rectify]`).
3. **White balance.** The camera's manual white balance isn't repeatable, so it runs on auto and
   each rectified image is scaled so the plate is neutral.

The result is the **rectified image**: the plate plus a 10 mm margin, at 10 px/mm, so 1200 × 1200 px.
It is what the model sees in training and in use, and what `synth` imitates. A 1/4 W body in it is
about 62 × 24 px.

The warp is exact only for the plate surface. What it does not fix:

- **Body height.** A body's centre sits 0.9 to 2.5 mm above the plate, so in the rectified image it
  appears shifted away from the camera by up to about 1.2 mm, and its side is slightly visible. Labels
  outline what is visible. The shift is corrected when an outline becomes a robot position, from the
  camera position and the body's measured width.
- **Detail.** The rectified scale is uniform, but the camera captured the far side of the plate at
  about 8.4 px/mm and the near side at about 13. The far side is softer.
- **Focus.** One focus setting can't make the near and far halves equally sharp. At the tuned
  setting both are at 90% or more of their best, and colour bands are readable across the plate.

## Contract with the robot repo

- Input: one full frame from the IMX708 at 4608 × 2592, same camera settings as `configs/camera.toml`,
  plus the calibration (intrinsics and plate homography) of the current setup.
- The frame is rectified with the `rectify` code from this repo (copy the module or depend on this
  package). The model runs on the rectified image.
- Output: a list of N×2 polygons in full-frame pixel coordinates (`outline_px`), mapped back from the
  rectified image through the inverse homography and the lens distortion. The robot repo maps them to
  metres and builds the Detection.
- Hand-off: `best.pt`, the rectified scale and size, the tile settings (if still tiled), and the label
  `rules_version` it was trained with.

Outlines in rectified millimetres would be simpler for the robot repo than full-frame pixels. That
is an open point to agree with the robot repo.

## Data rules that follow from rectification

These matter from the first capture session, because the collected images are the baseline.

- **Raw frames are the record.** Save every capture as the full, unwarped 4608 × 2592 PNG. Rectified
  images are derived and can be regenerated; never keep only the warped version.
- **Calibrate before collecting.**
  - Lock the focus first, because changing focus changes the intrinsics.
  - Take the checkerboard frames for the intrinsics once per camera and focus setting. About 15 to 20
    frames, tilted at varied angles and covering the plate area.
- **Each session records its geometry.**
  - One frame of the empty plate with its corners or reference marks clearly visible, for the
    homography. This can be one of the empty-table frames already taken for `backgrounds/`.
  - `session.toml` names the intrinsics file and the homography, and records the tower height, tilt
    and focus value.
- **The camera does not move within a session.** If it is knocked, start a new session (or, once
  reference marks exist, re-measure the homography per frame).
- **The whole plate is in frame and the arm is parked out of view** for every capture.
- **Labels stay in full-frame pixels** in `manifest.json`, as now. Labelling can happen on the
  rectified image; polygons convert both ways through the session calibration. Synthetic frames are
  drawn directly in rectified space, and the manifest records which space each sample is in.

## Tiles

At 10 px/mm the rectified plate is 1200 × 1200 px, close to one 1024 px tile. Tiling at 1024 with 128
overlap gives 2 × 2 tiles per frame. Running the whole rectified image at 1216 px is the alternative.
Decide once the first real data exists. YOLO-seg masks are at 1/4 of the input resolution (about
0.4 mm here). If that proves too coarse, Mask R-CNN (28 × 28 mask per body) is the fallback.

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
- bodies at the edge of the plate (near and far side, since the far side is softer) and across tile
  boundaries;
- some lighting variation, unless the lighting is fully controlled;
- empty and near-empty plates, to measure false detections.

## Labelling, cheapest first

1. **Synthetic.** `synth` draws resistors (shaded bodies, bands, leads, shadows) at random poses,
   including touching and overlapping, onto procedural backdrops now and onto real empty-plate
   rectified frames once we have them. Labels are exact. Later, paste real cut-outs from spread-out
   scenes.
2. **Auto-label and fix.** `autolabel` real frames, then fix only the merged or broken outlines.
3. **SAM-assisted** in CVAT or Label Studio, one click per body, for frames autolabel handles badly.

## Evaluation

Run on the real test sessions only. Match predictions to labels, then report in robot units:

- centre error (mm), yaw error (deg), length and width error (mm), with the body-height correction
  applied;
- merges (one prediction over several bodies) and splits (one body in several predictions);
- missed bodies and false detections;
- mask IoU, for comparison with the fibre project.

## Order of work

1. Scaffold, `config`, `manifest`, `visualise`, a first `synth` set (no camera needed). Done.
2. Camera arrives (week of 12 Oct). Done.
3. Tower built and the camera tuned on the rig (focus, exposure, white balance in
   `configs/camera.toml`). Done. Still to do: calibrate the intrinsics with a checkerboard.
4. `rectify`: plate homography from a session frame, warp to the rectified image, polygons both
   ways, plate white balance. Done, checked on a real frame. Still to do: update the work area and
   scale in `configs/dataset.toml` to rectified px, and `synth` to match.
5. `capture`, then the backdrop and lighting session, then the baseline captures.
6. `autolabel` and `cutouts` on spread-out scenes.
7. Real scene captures, label fixing.
8. `tiling` and `export_yolo` on rectified images, then a first `train` on real data only.
9. `synth`, retrain with synthetic added, compare on the test set.
10. `evaluate` and hand off.

## Done when

- [ ] On the real test set, centre error under 1 mm and yaw error under 3° for 95% of bodies.
- [ ] No merges on the touching-pairs frames in the test set.
- [ ] No false detections on the empty-plate test frames.
- [ ] Inference from a raw frame (rectify, model, map back) runs on the laptop in under 1 s.

## Open points

- Backdrop colour and lighting: decided in the first camera session.
- Will resistors be placed touching or overlapping? Decides how much synthetic touching data we need.
- Reference marks on the plate (corners or rim), so the homography can be re-measured on every frame.
- Outlines to the robot repo in full-frame pixels (current contract) or rectified millimetres.

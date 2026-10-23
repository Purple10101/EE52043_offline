# Camera tower: design brief

**Status:** Draft, 2026-10-14 (revised: angled view from the corner block)

## What it is for

The vision model needs one sharp view of the whole 100 × 100 mm pick plate. A camera straight above
the plate would sit where the arm must go to pick, so the camera looks at the plate at an angle from a
tower on the front corner block (`Body1:5`), outside the plate's footprint.

Arm clearance is out of scope: the arm can reach the tower, and that is handled separately.

## Coordinates

All values are in mm, in the frame of the rig scene (see Files). The origin is at the centre of the
robot base, and z is up. The arm STEP uses the same frame.

| Item | Where |
|---|---|
| Baseplate top surface | z = −50 |
| Pick plate (`Body1:1`), top surface | x −50 to 50, y −180 to −80, z = −22. Centre (0, −130, −22) |
| Tower footprint (replaces `Body1:5`) | x −180 to −135, y −180 to −135, on the baseplate (z = −50) |
| Neighbours | Cradle `Body1:2` starts at x = −130 and tray `Body1:4` starts at y = −130 (5 mm gaps). The baseplate edges are at x = −180 and y = −180, so the footprint can't grow |

## Required camera position

| | Value | Tolerance |
|---|---|---|
| Lens centre | **(−145, −145, 228)**: over the plate-side corner of the block, 250 mm above the plate top, 278 mm above the baseplate top | ±5 mm in x and y; height adjustable 225 to 275 mm above the plate |
| Optical axis | aimed at the plate centre (0, −130, −22): **tilted 30° from straight down**, pointing 6° from +x toward +y | ±2° (the plate fits with a wide margin; see below) |
| Rotation about the axis | camera upright: the long side of the image is horizontal in the world | ±2° |
| Focus | locked at 287 mm (the lens to plate-centre distance) | set on the rig |

With this position the plate covers roughly pixels x 1600 to 2900 and y 800 to 1900 of the
4608 × 2592 frame. Its far edge (x = 50) is near the top of the image and its near edge (x = −50) near
the bottom. That leaves about 13° of spare view on every side.

### Why this position

The figures below come from a pinhole model of the IMX708 standard lens (f = 4.74 mm, f/1.8,
1.4 µm pixels). They have not been measured on this camera yet; see Verification.

| Lens height above plate | Detail across the plate (px/mm, worst to best) | Blur at the plate's nearest and furthest points | Max view angle from vertical | Body-height shift (see below) |
|---|---|---|---|---|
| 150 mm | 8.2 to 20.8 | 0.59 mm | 54° | 1.6 mm |
| 200 mm | 8.6 to 16.3 | 0.40 mm | 46° | 1.2 mm |
| **250 mm** | **8.4 to 13.3** | **0.28 mm** | **39°** | **1.0 mm** |
| 300 mm | 7.9 to 11.2 | 0.20 mm | 34° | 0.8 mm |

- **Detail.** At an angle, the plate's far corner is both further away and foreshortened. The worst
  detail is best at a height of about 200 mm and stays almost the same up to 250 mm. 8.4 px/mm at the
  far corner is close to the 9.8 px/mm the vision plan assumed, and about 10 px/mm at the plate centre.
- **Blur.** At this distance the f/1.8 lens has a shallow depth of field, and an angled view puts the
  near and far edges of the plate at different distances (262 and 317 mm). The blur at those extremes
  falls as the camera rises because the distance ratio shrinks. At 250 mm it is about 0.28 mm, or
  2 to 4 px.
- **Body-height shift.** A body's centre sits about 1.2 mm above the plate, so seen at an angle it
  appears shifted away from the camera by up to 1 mm. The vision calibration corrects for this, and
  less shift leaves less to correct.
- **Lens xy.** The plate-side corner of the block is best: at the block's far corner the worst detail
  drops to 7.1 px/mm and the tilt rises to 36°.
- **Higher than 250 mm** gives less blur and shift, but less detail and a taller, wobblier tower on a
  45 mm square foot.

## Design requirements

1. **Base.** Match `Body1:5`: a 45 × 45 mm plate, 5 mm thick, with 4 × Ø3.1 mm through holes (M3
   clearance) inset 5 mm from each corner, a 35 × 35 mm hole pattern. Holes in scene coordinates:
   (−175, −175), (−140, −175), (−140, −140), (−175, −140).
2. **Column.** Its foot is limited to 45 × 45 mm and the lens is about 278 mm up, so stiffness is the
   main challenge. A 20 × 20 mm aluminium extrusion, or a ribbed printed column, is the expected
   route.
3. **Camera bracket.** From the column top, hold the lens at (−145, −145) at the set height. This is a
   short overhang of about 12 mm toward the plate from the column centre. Include a tilt pivot about a
   horizontal axis, adjustable from 25 to 35° from vertical and locked firmly. Include a pan
   adjustment of ±5° about the vertical axis.
4. **Height adjustment.** Allow the lens height to be set from 225 to 275 mm above the plate top (z =
   203 to 253), with a slot or spacers. The tilt changes with the height (33° at 225 mm, 28° at
   275 mm).
5. **Camera holder.** Hold the Arducam board with the sensor perpendicular to the aimed axis. The
   board size and mounting holes are not in this brief: measure the actual board, or use Arducam's
   drawing for the B0304 / B030401. Keep the lens and its autofocus actuator free; don't clamp on the
   lens barrel.
6. **Clear view.** No part of the tower, bracket or cable may be inside the line of sight from the
   lens to any point of the plate. The cradle (`Body1:2`, top at z = −17) is clear of that line by
   about 8 mm at the plate's near edge. Don't make anything taller between the tower and the plate.
7. **Stiffness and stability.** The camera must return to the same place after the robot stops: less
   than 1 px between frames taken at rest. At this distance that is less than about 0.02° of tilt, so
   frames are taken with the robot still. Creep and slow drift are covered by recalibration (see
   item 9), but sharp knocks must not move it.
8. **Cable.** Route the USB cable down the column with clips, and add strain relief at the base so a
   tug on the cable cannot move the camera.
9. **Reference marks (recommended).** If there's room, provide 3 or 4 fixed marks in view that don't
   move relative to the plate, such as small printed targets on the plate's corners or rim. The vision
   code can then check and correct the calibration on every frame, which makes the tower's small
   movements harmless.
10. **Lighting (optional).** The shiny leads and blue bodies reflect light toward the camera from the
    mirror direction: from the plate centre, toward +x at about 60° elevation. Don't put a light there.
    Light from the ±y sides of the plate or from beside the camera. Lighting may be solved separately.

## Verification

1. **Before printing.** Measure the field of view. Hold the camera at a known distance above a ruler
   on a flat surface and capture a frame at 4608 × 2592. If the result differs from the pinhole model
   by more than a few percent, recompute the position.
2. **After fitting.**
   - Capture a frame. The whole plate is in frame with margin on every side.
   - Set the focus manually so that the nearest and furthest edges of the plate are about equally
     sharp. Record the locked value in `configs/camera.toml`.
3. **Stability.** Capture frames before and after an arm move and a light tap on the column. The plate
   corners must not shift by more than 1 px once the camera has settled.

## Files

| What | Where |
|---|---|
| Rig scene, used for all coordinates here (exported 2026-10-14 18:57; units cm, z up; bodies `Body1` baseplate, `Body1:1` pick plate, `Body1:2` cradle, `Body1:3` and `Body1:4` trays, `Body1:5` corner block to replace) | `C:\Users\joshp\Downloads\EduSCARA_resistorrig.obj` |
| Fusion source of that scene | the Fusion design it was exported from (not on disk) |
| Arm with gripper, same frame, for arm clearance | `C:\Users\joshp\Y4S1\Robot Resources\gripper\v1_final\EduSCARA + Gripper v3.step` (and `.f3d`); notes in `v1 design notes.md` |
| Camera settings (focus, exposure, resolution) | `configs/camera.toml` in this repo |
| Vision plan (scale and accuracy targets) | `docs/plan.md` in this repo |

# Handoff to segmentation: seg_grey_v2 on the resistor kit (2026-11-10)

From the band-reading session. On 2026-11-10 we collected a band dataset from a 52-value resistor kit. The
same frames are a large real test of `models/seg_grey_v2`: 1,272 resistor bodies on the plate, of a kind
synth hasn't covered. This note gives what we found, where the data is, and what we think segmentation
should do about it. The band-reading session won't change segmentation code, models or `configs/camera.toml`.

## The data

- `data/bands/kit/2026-11-10/<value>/capNN/`, one folder per placement.
  - Each holds `frame.png` (full-resolution raw frame, the mean of 8 at focus 306) and `rectify.json`.
  - `capture.json` holds the camera settings, plus every body segmentation found: its outline in
    rectified pixels, its score and its resistor tag.
  - Gitignored. The full set is 1.5 GB.
- 35 values (0R to 10k) × 5 placements = **175 placements**, about 7 resistors each. 17 more values and
  the kit's 50 rectifier diodes are still to come (`<session>/diode/`).
- **Every object on a plate is the same resistor value, and nothing else is on it.** So any body-sized
  detection is a resistor, and its label is known. That makes these frames easy to label for segmentation.
- The parts are MIUZEI 1/4 W 1% metal film: 5 bands, blue bodies about 6 × 2 mm, and long leads (about
  60 mm across) that often reach off the plate. Placements were random, with leads sometimes crossing.
- Segmentation was seg_grey_v2 with greyscale input. The threshold was 0.5 for the first 55 placements
  and 0.3 after. `capture.json` records `conf` from then on, and `models` from now on.
- The first 7k5 placement had the wrong strip on the plate. It's in `<session>/_excluded/7500R_cap01/`.
  It is still fine for segmentation (it's all resistors), but it's out of the band labels. The analysis
  below was run before it was moved, over 175 placements.
- **How many parts were actually on each plate was not recorded.** The figures below are therefore
  shares of the bodies the model detected at all (≥ 0.05). Bodies it never detected aren't counted.

## Findings

Re-run with `experiments/2026-11-10_kit_segmentation_check/analyse.py`. It writes `bodies.csv` with one
row per body.

1. **Per frame, 91.2% of detected bodies score ≥ 0.5.** The rest (about 9%) score 0.05–0.5. The 24
   lowest-scoring are all real, clearly visible resistors (`figures/missed_bodies.png`).
2. **Position on the plate matters most.**

   | Rectified x (px) | Scored ≥ 0.5 | Bodies |
   |---|---|---|
   | < 400 (near side, closest to the lens) | **78%** | 292 |
   | 400–600 | 90% | 277 |
   | 600–800 | 97% | 357 |
   | > 800 (far side) | 97% | 346 |

3. **Pale bodies are missed.** By the body's mean grey in the model's input, the palest quarter scores
   ≥ 0.5 only 81% of the time, against 98% for the darkest quarter. In the images, the missed bodies are
   washed-out light cyan with glare. In greyscale they look much closer to the plate than the found ones
   (`figures/missed_bodies.png` against `figures/found_bodies.png`; each tile shows colour then grey).
4. **Angle matters a little.** Bodies lying along y score ≥ 0.5 85% of the time; the other angles score
   89–98%.
5. **Ruled out:**
   - Camera gain: 0, 8, 24 and 32 all give 90–93%.
   - Exposure: -4 and -5 give 91–92%.
   - Time of day: the 18:00 and 21:00–23:00 sessions gave the same results.
   - Focus: on one placement of 8, all 8 scored ≥ 0.5 at every focus from 298 to 326. **Caveat: that
     sweep fed the model colour, not grey, by mistake.** It should be repeated in grey. Separately, at
     the far edge the 2026-11-09 sweep found no focus effect either.
   - The near side's first-capture crops are about half as sharp as the far side's at focus 306. But
     focus alone doesn't explain the misses.
6. **Live detection on the same model looks perfect, which can mislead.** In a live_detect run on 16
   kit resistors (`figures/live_detect_16_bodies.jpg`), Steady showed all 16 from run 4–5. Per frame,
   though, it found 14–16 of 16 at ≥ 0.5, about 95%. Steady hides single-frame misses, while one-frame
   uses (the robot cycle, collection) don't get that help. **The user confirms the lighting was the same,
   and ideal, throughout collection and in that run.** So the lighting didn't change. The near-side,
   pale-body misses come from the fixed geometry of light, camera and plate: bodies on the near side catch
   glare under this rig's light in every capture. The live run's neater layout, with many bodies in rows
   and few near the near edge, is the likely reason it looked better.
7. **No false detections.** Nothing outlined at ≥ 0.5 was anything but a resistor, in any of the 175
   placements or the live run.

## Suggested next steps for segmentation

1. **Use these frames as a real test set for small blue resistors.** Count recall per placement against
   the true number on the plate. From now on the collector could ask the user for that count; the
   band-reading session can add it if you want it.
2. **Train on pale, glary bodies.**
   - Brightness and contrast augmentation per body, and glare highlights in synth.
   - Fine-tune on these frames. The labels nearly write themselves: take the outlines at ≥ 0.5, add the
     bodies at 0.05–0.5 after checking them (they are the hard cases), and add any body missed even at
     0.05 by hand.
3. **Repeat the focus sweep in grey** on a placement with near-side bodies, to settle focus for good.
4. **Lighting is fixed and ideal by the user's account, so treat near-side glare as part of the task.**
   The model has to handle it (step 2). Changing the light is not the plan.

## Also seen

- **Plate finding** fails on restart when the far edge sits against a pale bench in dim or evening light
  and a lead lies over the centre cross. The search needs the far edge or the cross. Collection now
  locks to the last fit (`collect_bands`, `--search` to search instead).
  - A durable fix belongs in `pre/plate_search.py`: when the search fails, check whether the last fit
    still matches the visible edges.
  - Plate search lives in this repo, and band reading has worked on it before. Coordinate before either
    session changes it.
- **The resistor tag v1 on these blue 1% parts:** of 1,187 kit resistors, 1,155 were tagged "resistor",
  25 "unsure" and 7 "not". The tag per body is in `capture.json`.

Earlier handoff: `docs/handoffs/2026-11-09_segmentation.md`.

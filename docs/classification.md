# Classification: reading resistor values

**Status:** Draft, research

## Goal

For each resistor body the segmentation model finds, read its value from the colour bands, so the
robot can sort by value. Parts that can't be read confidently go to an "unknown" bin instead of the
wrong one.

## Scope

Values from 10 Ω to 10 kΩ only. That limits the multiplier band to gold (×0.1), black (×1), brown
(×10) or red (×100); a decoded multiplier of orange or above means a misread (most often gold read
as yellow, or brown read as orange). The decoder uses this, together with the standard series, to
reject and retry readings.

## Where it lives

`robot_offline.post.bands`, separate from the segmentation pipeline. Its input is a body found by the
segmentation model (its polygon in the rectified image) and the raw camera frame. The final reader is
handed to the robot repo as code, like `rectify` and `tiling`.

## Approach

1. **Crops.** Each body is cut from the raw camera frame on a grid along its long axis, at a fixed
   20 px/mm (`bands.crops`). The raw frame is sharper than the rectified image on the near side of
   the plate (up to 13 px/mm against 10). Crops are white-balanced like the rectified image and come
   with the body's mask.
2. **Profiles.** A colour profile along the body's centre line, from the middle of its width and
   away from highlights, shows the bands as runs that differ from the body paint.
3. **Rule-based reader (first).** Find the bands, classify each colour against calibrated reference
   colours, try both reading directions, decode digits and multiplier, and keep only values in the
   standard series (E12 / E24 / E96). A value outside the series is a misread: retry with the
   next-best colour for the least certain band, or report unknown.
4. **Learned band reader (if the rules fall short).** A small model reading the band sequence (a
   colour per position, plus direction), pretrained on synthetic band sequences from `synth` and
   fine-tuned on real crops, with the same decoding and series check after it.

## What decides accuracy

- **Colour calibration.** Plate white balance fixes the overall tint; brown, red and orange need
  more. A printed strip of reference colours on the plate rim would allow per-frame correction.
- **Highlights.** Glossy bands catch the light; use the median of each band, excluding the
  brightest pixels.
- **Body colour showing through bands**, which shifts their colour (more on turquoise bodies).
- **Reading direction**: the tolerance band (brown on 5-band metal film, gold or silver on
  4-band carbon) sits apart near one end; decoding both ways and keeping the valid value settles
  most cases.
- **Masks including lead stubs**: the profile is trimmed to where the paint starts and ends.

## Data

- **Reference crops**: real crops with known values. Each resistor is measured once with a
  multimeter; every crop of it gets that value. The cut-out session (2026-10-23_b) shows the same
  resistors in 12 positions each.
- **Synthetic band sequences** for the learned reader: `synth` already draws bands with exact colours.
- More distinct values are needed for a fair test: 6 resistors is too few.

## Evaluation

Value accuracy per body on the reference crops, the share reported unknown, and a confusion matrix
of band colours. The target is set once the open points below are answered.

## Open points

- Which parts are sorted: 4-band carbon, 5-band metal film, or both? Which series? (Range settled:
  10 Ω to 10 kΩ.)
- How many bins: exact values, or ranges?
- Can the rig check resistance electrically as a backstop?
- The values of the 6 resistors we have, and more resistors with known values. The 6 are three
  pairs; read by eye their bands are A red-violet-(black)-yellow?-brown, B brown-black-green-
  orange?-brown, C brown-black-green-yellow?-brown, which within 10 Ω to 10 kΩ suggests A 27 Ω or
  2.74 kΩ, B 1.05 kΩ, C 10.5 Ω. To be confirmed by measurement.

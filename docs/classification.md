# Classification: reading resistor values

**Status:** Draft, research

## Goal

The first step is grouping: two parts that look the same go in the same bin, whatever their value.
Nothing is decoded, so no colour needs naming and no list of values is needed (see
[Grouping](#grouping-first-step)).

The later goal: for each resistor body the segmentation model finds, read its value from the colour
bands, so the robot can sort by value. Parts that can't be read confidently go to an "unknown" bin
instead of the wrong one.

## Scope

Values from 10 Ω to 10 kΩ only. That limits the multiplier band to gold (×0.1), black (×1), brown
(×10) or red (×100); a decoded multiplier of orange or above means a misread (most often gold read
as yellow, or brown read as orange). The decoder uses this, together with the standard series, to
reject and retry readings.

## Grouping (first step)

`bands.grouping` turns each crop into a colour profile along the body, compares profiles directly,
and clusters them by average linkage. It allows for a part lying either way round, for shift and
stretch (dynamic time warping), and for the ends, which differ between views because the camera sees
one end cap and not the other. Differences are combined so that the worst mismatch dominates, or a
one-band difference would be lost.

"Same" means same appearance, not same value: the same value can land in different groups (blue
5-band and beige 4-band 1 kΩ parts do), and only the bands count, not the body colour.

Results on the 143 crops of sessions 2026-10-23_a and _b (three kinds, the 6 resistors, 2026-11-06):

| Check | Result |
| --- | --- |
| All crops at once | 48 / 48 / 47, no mistake found by eye; three groups for thresholds 15 to 19 |
| Settings | Same split for `END_SKIP` 16 to 24, `POWER` 3 to 5, `WARP_RADIUS` 24 and 32; fails at radius 16 |
| Groups from one session, the other's crops assigned | 140 of 143 right, 0 wrong, 3 unmatched |
| One plate at a time (21 frames of 6 bodies) | 15 frames exactly right, 96% of pairs right |
| Arriving one by one (30 random orders) | Usually 4 groups, not 3: a kind gets split; the worst order mixed 8% of bodies |

These flatter the method. The settings and threshold were chosen on these crops, both sessions show
the same 6 resistors, and the labels are the clustering's own output checked by eye. Single pairs of
views are not reliable (same-kind pairs reach 38.9, different kinds come down to 10.9), so a body
should be compared with a whole group, and the more views a group has the better.

**It fails on the blue metal-film parts.** In the reference frames (2026-10-29), the 1 kΩ, 3 kΩ and
10 kΩ blue parts all land in one group: their bands differ only among brown, black and dark red, and
same-value and different-value pairs have the same distance. Lifting shadows or normalising
intensity afterwards did not separate them. The beige 1 kΩ and the 4.7 kΩ parts, with strong band
colours, did group correctly.

Whether more exposure fixes this is not known. No 3 kΩ or 10 kΩ blue part has been seen at a
brighter exposure than those frames, which are about as bright as exposure -5 (body brightness 102
to 122 against 107 to 124).

Session 2026-11-06_b has measured values: one blue 4.7 kΩ with a green band, two blue 1 kΩ, two beige
1 kΩ. Session 2026-11-06_a shows the plate before the parts were moved, its values inferred from
their look. Comparing 9 views of these at exposure -4:

- `THRESHOLD` is right here. Thresholds 16 to 32 give exactly the three kinds, and the turquoise
  parts need 15 to 19, so 17 sits in both.
- But these kinds are easy: the same kind is up to 16.3 apart, different kinds at least 31.4. They
  say nothing about parts that differ only in dark bands.
- The two blue 1 kΩ parts are 11 to 15 apart, while different blue values in the reference frames are
  about 7 to 11 apart. The exposures differ, so this is not a fair comparison, but it is the wrong way
  round, and the margin for telling dark-banded values apart may be small or absent.

**Looking different is not being different.** At exposure -4 those two 1 kΩ parts looked clearly
different by eye, with redder bands on one, and both measure 1 kΩ. Angle and shading change how
bands look, and these bands read nearly the same from both ends, so the comparison can't tell which
way round they lie either (forward and reversed distances are within 1 of each other for most pairs).
Eye checks need measured values behind them, including the turquoise kinds, which are unmeasured.

Over-exposing keeps bands readable at gains the plate clips at, if white-balance gains come from an
unclipped frame, but it did not change these distances, and bodies clip (40 to 75% of body pixels at
gain 48). Distances grow with exposure (the part with the green band against a blue 1 kΩ went from
about 5 to about 39 between -7 and -4), so `THRESHOLD` holds only near the exposure it was set for.

**Measured, dark bands alone do not separate.** Session 2026-11-06_c adds a blue 100 Ω (one band
black where a blue 1 kΩ has brown), at about the same plate level as the -4 frames. It is 12.3 to 17.5
from the blue 1 kΩ views, which are 6.5 to 14.2 from each other: the ranges overlap, and at
`THRESHOLD` the 100 Ω joins the 1 kΩ group, the wrong bin. Lowering the threshold is no way out: at
14 or below other kinds start splitting (the blue 4.7 kΩ views are up to 15 apart, the beige 1 kΩ
views up to 16.3). It is one 100 Ω part, but with the reference frames that makes two sets of blue
parts where a dark-band difference is no bigger than the spread within one value.

The other kinds in that session behave. At `THRESHOLD` the blue 4.7 kΩ views stay together, and the
beige 10 kΩ stays apart from the beige 1 kΩ views. The beige 10 kΩ and green 10 kΩ group together:
removing the body colour leaves the same bands, so same-value 4-band parts can share a bin across
body colours. The blue and beige 1 kΩ do not (5-band against 4-band), nor do the blue and teal 4.7 kΩ.

So grouping is usable where bands differ in strong colours, and cannot yet be trusted for blue parts
that differ only among brown, black and dark red. Options: a comparison aimed at those bands, tested
on several parts of each value, or a meter settling just those groups.

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
- How many bins: exact values, or ranges? Grouping needs one per kind, plus one for the unmatched.
- Can a comparison aimed at dark bands separate blue parts, or does a meter settle those?
- Are all parts on the plate seen before the first is moved? Arriving one by one is the weakest case.
- Can the rig check resistance electrically as a backstop?
- The values of the 6 resistors we have, and more resistors with known values. The 6 are three
  pairs; read by eye their bands are A red-violet-(black)-yellow?-brown, B brown-black-green-
  orange?-brown, C brown-black-green-yellow?-brown, which within 10 Ω to 10 kΩ suggests A 27 Ω or
  2.74 kΩ, B 1.05 kΩ, C 10.5 Ω. To be confirmed by measurement.

## Reading a value (first attempt)

`bands.value` finds the bands along a crop's centre line and scores every value in scope against
them, both ways round: 4-band E24 with gold, 5-band E24 with brown, 10 Ω to 10 kΩ. A value is
reported only if it clearly beats the next one; otherwise the part is unknown.

**Exposure decides it.** With the plate at about 215 (exposure -4, no gain), brown and black on a blue
body separate: L 17, a* +20 against L 4, a* 0. At about 160 they overlap (L 14 ± 7 against L 5.5 ± 2).
Adding gain weakens the colour again (a* +25 to +18), so a brighter frame than that does not help.

**Band colour depends on the body.** Band paint is thin: yellow on blue looks green, red looks maroon.
Beige bodies show bands nearly pure. Non-blue bodies use a model of paint over the measured body
colour; blue bodies use measured colours, because the model can't reproduce them.

Results, best guess against values measured with a multimeter (parts removed one at a time, each
removal confirmed by camera):

| Bodies | Exposure | Right |
| --- | --- | --- |
| Beige, 1 kΩ, session 2026-11-06_b (paints set on these) | -4 | 2 of 2 |
| Beige, 1 kΩ, session 2026-11-06_a (unseen) | -4 | 2 of 2 |
| All 16 bodies, reference frames (1 kΩ to 10 kΩ) | about -5 | 0 of 16, all reported unknown |
| Blue, 1 kΩ and 4.7 kΩ, session 2026-11-06_b (colours set on these) | -4 | 1 of 3 |
| Blue, 1 kΩ and 4.7 kΩ, session 2026-11-06_a (unseen) | -4 | 1 of 3 |

| Five new bodies, session 2026-11-06_c (never seen; 100 Ω, 4.7 kΩ, 10 kΩ; blue, beige, green, small) | about -4 | 1 of 5, all reported unknown |

These are few parts, and session a's values are inferred from the parts' kinds. The same parts appear
in sessions a and b, so the fair figure is session c's: the reader does not yet work, though it did
not report a wrong value. Beige is shown only for 1 kΩ; the beige 10 kΩ in session c read as 20 Ω.
In session c's crops every band is readable by eye, so what limits it is this rule-based reader, not
the camera. Three more rules (paint from the brightest third, dropping bright runs, judging dark
bands against the body's black) made it worse and were taken out. The next step is the learned band
reader above, trained on synthetic bands and these measured parts.
The band colours are set for exposure -4, so the reader only works there: on dimmer frames it reads
nothing, though it does say unknown rather than a wrong value.

**Blue bodies report unknown for now.** The same brown measured L 17, a* +25 on one part and L 9, a* +1
on another, so a 4.7 kΩ read as 470 Ω with a good margin. Red and orange on blue have not been seen at
-4 at all, so blue 3 kΩ and 10 kΩ are untested. No green or small body has a measured value at -4.

Next: more parts of measured value at -4, above all blue 3 kΩ and 10 kΩ, a green body and a 1/8 W one.

## The learned reader (first version)

`bands.learned` is a network of 151 k parameters, about 2 ms a body on the CPU, trained by
`training.band_reader`. It gives a colour for each of five slots; every value in scope is then scored
from them, both ways round, so only a real value can come out, and that value's share is the
confidence. Below 0.95 the part is unknown, for the robot to measure.

Trained on 12,387 synthetic crops, then adjusted on real ones. Scored on parts it never saw:

| Trained on | Tested on | Right (best guess) | Answers at 0.9 | Wrong among them |
| --- | --- | --- | --- | --- |
| Synthetic only | Session c, 15 views | 0 | 0 | 0 |
| Synthetic, sessions a and b | Session c | 0 | 0 | 0 |
| Synthetic, HiRes photos, sessions a and b | Session c | 3 | 3 | 0 |
| Synthetic, HiRes photos, session c | Sessions a and b, 45 views | 16 | 2 | 1 |

So 19 of 60 views, and one wrong answer in five at 0.9, which is why the limit is 0.95. At 0.95 these
models answered nothing, so **for now it reports nearly everything unknown**. Session c was looked at
more than once, so its row flatters slightly. A wider network (518 k) was no better and was confidently
wrong at 0.93, so it was dropped.

Real photos made the difference: 687 from the HiRes dataset (github.com/HiRes491/HiRes), each checked
against the value in its file name. **That repo has no licence**, so the weights in
`runs/band_reader_final/` are for research and must not ship until that is settled. HiRes's own weights
scored 0 of 15 on our parts, though 23 of 27 on its own images.

`synth` now draws bands from measured colours and widths, but they are still 15 to 27 Lab units from
real ones on blue bodies, and grey, white, silver, and green and blue as digits are estimates.

Next: more measured parts, which is the one thing shown to help, above all blue 3 kΩ and 10 kΩ.
Unanswered: are any parts E96? The reader only considers E24, so a 4.99 kΩ part would be misread.

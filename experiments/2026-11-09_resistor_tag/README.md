# Resistor or not: the resistor tag (2026-11-09)

**Question.** The object model (`seg_objects_v1`) outlines every pickable object as one class,
"object", from greyscale. Which of those outlines are resistors? The robot must not read band values
off a diode or a capacitor.

**Status.** `resistor_tag_v1` is trained, and band reading has wired it into `live_detect` and
`read_plate`. It has been checked on real resistors and screws only; the real look-alikes (diodes,
capacitors, inductors) are untested. That is the next test: `capture_parts.py`, below.

## Design

A small classifier on each object's **colour** crop, separate from segmentation, as planned in
`experiments/2026-11-09_object_seg_design` ("Resistor or not"). What tells a resistor from a diode is
colour and detail, which the segmentation model's greyscale input lacks. The code is in
`src/robot_offline/post/resistor_tag.py`.

**The crop:**
- **Source:** the model view in colour (`rectify.model_input(...)[0]`): rectified at 10 px/mm,
  white-balanced with the plate at the training level, everything off the plate masked. So it doesn't
  drift with the room's light, and it never sees the unknown background.
- **Shape:** turned so the object's long side lies along x, at true scale (size is a cue), in a
  24 × 12 mm window.
- **Edges:** 1 mm is kept round the outline so lead stubs show (band reading's suggestion); black
  beyond.

**The network:** a small CNN with one output, P(resistor). It is scored both ways round (which end
comes first is arbitrary). It runs on the CPU: about 40 to 55 ms per frame for 10 objects.

**Its answers**, and what the robot does with each (agreed with band reading, robot-offline-01):

| P(resistor) | Tag | The robot |
|---|---|---|
| 0.8 or more | "resistor" | reads the bands |
| between 0.2 and 0.8 | "unsure" | reads them, but never trusts the value; it measures the part instead |
| 0.2 or less | "not" | doesn't read |

**The failure to avoid:** a confident band value on a non-resistor. So the headline number is the
share of non-resistors tagged "resistor".

## Training

```
python -m robot_offline.training.train_resistor_tag --name resistor_tag_v1 --device cpu
```

- **Data:** the 10,993 labelled objects of `synth_obj_v1` (7,385 resistors and 13 other kinds), each
  labelled with its kind by the synthetic generator.
- **Run:** 12 epochs on the CPU, about 20 minutes.
- **Augmentation:** either end first, mirrored, light and white balance a little off, soft focus,
  noise, the outline a little off-centre.
- **Weights:** `models/resistor_tag_v1/resistor_tag.pt`, with its scores in `scores.json` beside them.

## Results so far

| Set | Non-resistors tagged "resistor" | Resistors tagged "resistor" |
|---|---|---|
| Synthetic validation (`synth_obj_v1_val`, 1,377 objects) | 0 of 463 | 914 of 914 |
| Real test frames (`test_sessions`, never trained on) | **0 of 24** (23 "not", 1 "unsure") | 239 of 241 (2 "not") |

The synthetic set is evidently easy. The only real non-resistors so far are screws, the easiest kind.

## Next: real mixed parts

```
python experiments/2026-11-09_resistor_tag/capture_parts.py parts_a --device cpu
python experiments/2026-11-09_resistor_tag/capture_parts.py parts_a --kinds "1=resistor 2=diode 3=screw ..."
```

1. **Capture:** the first command captures a placement, segments it with `seg_objects_v1`, tags each
   object, and writes a numbered picture to `parts/`.
2. **Label:** the user names each number with one word.
3. **Score:** the tag is then scored per kind.

Band reading has asked for the diode and inductor numbers in particular: those are the look-alikes
where a band reading would be confidently wrong.

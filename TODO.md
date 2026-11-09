# TODO

Most important first.

1. **Value reading is not usable yet.** On parts it never saw the learned reader got 19 of 60 views
   right and says unknown for nearly everything (docs/classification.md). More measured parts is what
   helps: blue 3 kΩ and 10 kΩ first. Decide whether any parts are E96.

2. **HiRes licence.** The reader's weights were trained on HiRes photos and that repo has no licence.
   Settle this before they ship, or retrain without them.

3. **Plate test.** Run segmentation and reading on the resistors left on the plate, once the camera is
   plugged back in, and save predictions before measuring them.

4. **Reference set labels.**
   - Confirm whether the "3K" set (0002) is 3.0 kΩ or 3.3 kΩ, and fix `value_ohm` in
     `reference.json` if needed.
   - Add the two bodies the model missed in 0003 (the beige one from its 0.50 detection, and the
     green one outlined by hand) so the band reader has all four.
   - Outline the two small bodies the model missed in 0004 by hand.

5. **Plate-rim false detections** at about 0.27 confidence along the plate's edge, hidden by the 0.5
   threshold. A 0.3 mm inset on the plate mask would remove the 1 to 2 px rim of outside colour that
   still leaks in.

6. **Speed:** white balance (about 49 ms) and the plate mask (about 82 ms) take longer than the
   model (about 77 ms) in each live frame.

7. **Auto-labeller** for real captures, deferred until after the synthetic-only model.

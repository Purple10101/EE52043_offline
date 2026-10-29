# TODO

Most important first.

1. **Green resistor bodies are not detected.** The synthetic packages in `src/robot_offline/training/synth.py`
   (`PACKAGES`) are only teal and blue metal film and beige carbon, so `synth_only_v1` has never seen
   a green body. In reference frame `data/reference/2026-10-29/0003.png` (four 10 kΩ) the green-bodied
   resistor was not found at any confidence, even 0.05. To do:
   - Measure the green body's colour, size and band look from real frames, without using them for
     training (real frames stay for validation).
   - Add a green package to `PACKAGES` and generate new synthetic sets.
   - Retrain, and check that the green body in 0003 is found.
   - Check that green-bodied parts on the plate are resistors, not colour-coded axial inductors,
     which look almost the same.

   **Possible solution: segment in greyscale.** Train on greyscale frames and convert every frame to
   greyscale just before inference. The model then finds bodies by shape and shading alone, so a body
   colour it has never seen (green, or any other) is no longer a problem. Colour is still used where
   it matters: the band reader cuts its crops from the raw colour frame (`bands.crops`), not from the
   model's input. Points to check:
   - YOLO expects 3 channels: repeat the grey channel three times, both in `export_yolo` (training)
     and in `rectify.model_input` / `live_detect` / `tiling.predict_frame` callers (inference), so
     the two can never differ.
   - Use the same conversion in both places (e.g. `cv2.cvtColor(..., COLOR_BGR2GRAY)`), applied
     after white balance and plate masking.
   - Contrast: light beige bodies on the light grey plate differ mostly in colour, not brightness,
     so they may get harder to find in grey. Measure each package's grey-level contrast against the
     plate on real frames before committing to this.
   - Non-resistor objects (TO-220, TO-92, DIP-8, screws) are told apart by shape already, but check
     that the false-positive rate on them doesn't rise.
   - Compare against the alternative of keeping colour and adding strong hue augmentation (random
     hue shifts on bodies in `synth`, `hsv_h` in `configs/train.toml`): train both and score each on
     the real reference frames, green body included.
   - A side benefit: the synthetic generator's colours would matter less for segmentation, only its
     shapes and shading.

2. **Small (1/8 W) bodies are not detected.** In reference frame 0004 (four 4.7 kΩ), the two
   miniature teal bodies, about 3.5 mm long and 1.8 mm across, scored 0.16 and nothing. The two
   full-size ones beside them, about 6.7 mm, scored 0.90. The synthetic packages are all 6.3 mm or
   longer, apart from the rare half-watt ones, so the model has never seen a body this small.
   Greyscale (item 1) won't fix this: add a 1/8 W package (about 3.2 to 3.6 mm long, 1.7 to 1.9 mm
   across, measured from real frames) to `PACKAGES`, and check whether the bands on bodies this
   small are still readable at the camera's resolution.

3. **Beige carbon bodies score close to the threshold.** The beige 10 kΩ in 0003 scored 0.50, just
   under `--conf 0.5`. The beige ones in 0001 scored 0.80 to 0.83. Look at how the synthetic beige
   bodies compare with the real ones.

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

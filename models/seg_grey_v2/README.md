# seg_grey_v2

Resistor-body segmentation on greyscale frames, trained on synthetic data only. Chosen on 2026-11-06
over `synth_only_v1` and a colour model trained the same way, on real labelled frames.

| | |
|---|---|
| Weights | `best.pt` (Git LFS), YOLOv8m-seg fine-tuned from COCO `yolov8m-seg.pt` |
| Class | `resistor_body` (leads excluded), label rules version 1 |
| Input | the rectified, plate-masked image in greyscale, grey in all three channels (`"input": "grey"` in `handoff.json`; `tiling.predict_frame(..., grey=True)`) |
| Training data | `synth_v2` (800 frames) and `synth_v2_val` (100): any body paint, 3.2 to 11 mm, hard shadows, shade, tint, exposure |
| Training | 40 epochs, 61 min on an RTX 3060 laptop GPU |

## Scores

Bodies found at conf 0.5 on real frames (`test_sessions` in `configs/dataset.toml`), never trained on:

| | `synth_only_v1` | colour | `seg_grey_v2` |
|---|---|---|---|
| Beige | 7 of 30 | 23 | 26 |
| Blue | 46 of 51 | 49 | 51 |
| Green | 1 of 2 | 2 | 2 |
| Green small | 0 of 12 | 10 | 11 |
| Small | 1 of 3 | 3 | 3 |
| Turquoise, 23 Oct sessions | 143 of 143 | 141 | 138 |
| False detections | 0 | 4 | 1 |

The turquoise labels are `synth_only_v1`'s own outlines, which favours it. 12 of the frames are one
scene at six exposures, so the new-parts figures are not independent, and the darkest of those still
loses bodies (9 of 12 at plate level 59).

Continuing this model on 2,000 more frames, with the corrected bands, made it worse (86 of 98 new
parts, 128 of 143 turquoise); that run was stopped at epoch 13 of 15, so it isn't conclusive.

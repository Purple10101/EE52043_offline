# synth_only_v1

Resistor-body segmentation, trained on synthetic data only. The first model of the rectified
pipeline (commit 8b370b4).

| | |
|---|---|
| Weights | `best.pt` (Git LFS), YOLOv8m-seg fine-tuned from COCO `yolov8m-seg.pt` |
| Class | `resistor_body` (leads excluded), label rules version 1 |
| Input | one 1024 px tile of the rectified, plate-masked image (see below) |
| Training data | `synth` (600 frames) for training, `synth_val` (100 frames, another seed) for validation |
| Training | 31 epochs (early stop), best at epoch 16, batch 4, 67 min on an RTX 3060 laptop GPU |
| Synthetic validation | mask mAP50 0.995, mask mAP50-95 0.933 (saturated: too easy to tell models apart) |

Other files: `handoff.json` (tile size, overlap, rules version, training arguments), `results.csv`
(per-epoch scores), `real_preview.jpg` (predictions on the real frames).

## Feeding it

The model must see exactly what it was trained on: a raw 4608 x 2592 frame rectified to the
1200 x 1200 top-down view at 10 px/mm, white-balanced on the plate, and with everything off the plate
replaced by the plate's grey. In this repo that is one call:

```python
from robot_offline import rectify, tiling
image = rectify.model_input(raw_frame, calibration)            # rectify, white balance, mask
tiles = tiling.tiles((100, 100, 1000, 1000), (1200, 1200), 1024, 128)
polygons, scores = tiling.predict_frame(YOLO("best.pt"), image, tiles, conf=0.5)
points_raw = rectify.to_full(polygons[0], calibration)         # back to raw camera px if needed
```

Polygons come out in rectified px; `rectify.rect_to_mm` gives plate mm. About 0.2 s per frame on the
laptop GPU, 1.1 s on CPU.

## How it does on real frames

Run on the 30 real frames of sessions 2026-10-23_a and 2026-10-23_b (no labels yet, so checked by
eye): every resistor body on the plate found, crowded parallel bodies kept apart, and nothing
detected on screws, foil, wood shavings or empty plates.

Known limits:

- Low-confidence detections (about 0.27) on the thin rim at the plate's edge, where the mask, set
  exactly at the edge, lets a trace of the surroundings through. Use `conf=0.5`; masking 0.3 mm
  inside the edge is the planned fix.
- Masks are slightly generous: a thin rim of plate around bodies, and sometimes a short lead stub.
- Outline accuracy in mm is not measured yet; that needs labelled real frames.
- Session 2026-10-23_a was used to tune the synthetic look, so it is not a fully clean test.

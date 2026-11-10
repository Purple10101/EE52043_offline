# seg_objects_v1

Segmentation of every pickable object (resistor bodies, screws, capacitors, diodes, ICs and so on), one
class `object`, on greyscale frames, trained on synthetic data only (2026-11-09). This is the first model
from the design in `experiments/2026-11-09_object_seg_design`. It is **not yet the live model**:
`seg_grey_v2` stays live until the tests in `experiments/2026-11-09_seg_objects_v1_test` pass.

| | |
|---|---|
| Weights | `best.pt` (Git LFS, epoch 40 of 40, best by box and mask fitness), YOLOv8m-seg fine-tuned from COCO `yolov8m-seg.pt` |
| Class | `object`: an object's body, with leads and pins excluded and screws whole. Label rules version 2 (`docs/labelling.md`) |
| Input | the rectified, plate-masked image in greyscale (`"input": "grey"` in `handoff.json`) |
| Training data | `synth_obj_v1` (1,200 tiles, 10,993 objects) and `synth_obj_v1_val` (150), drawn with `--profile objects` |
| Training | 40 epochs, 3 h 05 min on an RTX 3060 laptop GPU; synthetic validation mask mAP50-95 0.944 |

## Scores

On the real test frames at conf 0.5 (`experiments/2026-11-09_seg_objects_v1_test`):

| | `seg_grey_v2` | `seg_objects_v1` |
|---|---|---|
| Resistors found, of 241 | 231 | 229 |
| Screws and bolt found, of 24 | 0 | 10 |
| False detections, 65 complete frames | 1 | 23 (14 are the robot arm over an empty plate, 4 the plate's crosshair groove under hard light) |
| Resistor centre error, 95th percentile | 0.33 mm | 0.47 mm |
| Resistor yaw error, 95th percentile | 7.6° | 6.2° |

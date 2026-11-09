"""Reading resistor values from their colour bands (research, see docs/classification.md).

Kept apart from the segmentation pipeline: its input is a body the segmentation model found (its
polygon in the rectified image) and the raw camera frame it came from.

- `crops` - straightened, white-balanced body crops from the raw frame, and the crop dataset.
- `grouping` - groups bodies by how their bands look, without reading a value.
- `value` - the rule-based reader, on any body colour; unknown when unsure.
- `learned` - the learned band reader, a small network for the CPU; see docs/classification.md.
"""

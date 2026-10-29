"""Reading resistor values from their colour bands (research, see docs/classification.md).

Kept apart from the segmentation pipeline: its input is a body the segmentation model found (its
polygon in the rectified image) and the raw camera frame it came from.

- `crops` - straightened, white-balanced body crops from the raw frame, and the crop dataset.
"""

"""Vision for the EE52043 resistor sorter: the resistor body segmentation model and what runs
around it. See docs/plan.md.

- pre: camera, plate calibration, rectification, white balance and exposure.
- inf: the model run on rectified frames, and live detection.
- post: skeleton, spine and centroid of each detected body; colour-band reading.
- tools: frame capture and performance evaluation.
- training: synthetic data, labels, export and training. Nothing outside it imports from it.

config and manifest (the label store) are shared by all of them.
"""

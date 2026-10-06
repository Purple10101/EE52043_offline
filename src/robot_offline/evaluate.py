"""Scores a trained model on the real test sessions, full frames with tiling.

Matches predictions to labels, then reports in robot units: centre, yaw, length and width error,
merges, splits, misses and false detections, plus mask IoU. Targets are in docs/plan.md.
"""

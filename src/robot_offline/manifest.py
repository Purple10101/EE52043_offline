"""The label store, data/manifest.json: every frame, its session, source and body polygons.

Polygons are N×2 float arrays in full-frame pixel coordinates, the same form as the robot repo's
outline_px. Splits come from configs/dataset.toml by session, not from the manifest.
Format: data/README.md. Labelling rules: docs/labelling.md.
"""

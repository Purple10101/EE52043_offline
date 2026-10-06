"""Copy-paste synthetic frames: cut-outs placed onto empty-table frames.

Random position and yaw, with a configured share touching or overlapping a neighbour. Polygons are
exact, with overlapped bodies clipped to their visible part (docs/labelling.md rule 4).
Writes frames to data/synthetic/ and adds them to the manifest as source "synthetic".
"""

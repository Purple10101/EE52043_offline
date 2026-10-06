"""Tiles for training and inference.

Training: cuts the work area of a frame into overlapping tiles and clips each polygon to its tile,
dropping bodies with too little of their area inside.
Inference: runs a model over the tiles and merges predictions back into full-frame polygons,
joining duplicates from overlapping tiles. The robot repo uses the same merge.
"""

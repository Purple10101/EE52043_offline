"""Draws labelled polygons and model predictions over frames, for checking labels and results."""

import cv2
import numpy as np


def draw_polygons(image, polygons, colour=(0, 255, 0), thickness=3):
    """Returns a copy of image with each N×2 polygon (full-frame px) outlined."""
    out = image.copy()
    if polygons:
        pts = [np.round(np.asarray(p) * 16).astype(np.int32) for p in polygons]
        cv2.polylines(out, pts, True, colour, thickness, cv2.LINE_AA, shift=4)
    return out

"""Tiles for training and inference.

Training: cuts the work area of a frame into overlapping tiles and clips each polygon to its tile,
dropping bodies with too little of their area inside.
Inference: runs a model over the tiles and merges predictions back into full-frame polygons,
joining duplicates from overlapping tiles. The robot repo uses the same merge.

Each tile owns a core: the part of it nearer its own centre than any neighbour's, so the cores
cover the work area without overlapping. A prediction is kept by the tile whose core holds its
centre. With the overlap larger than the longest body, a body whose centre is in a core lies
wholly inside that tile, so clipped predictions at tile edges are never the ones kept. Cores are
widened by a small margin and the few resulting duplicates are removed by overlap, so a body
whose centre sits on a core boundary is neither lost nor kept twice.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box

CORE_MARGIN_PX = 8.0        # widening of each core, so a body on a core boundary is not lost
DUPLICATE_OVERLAP = 0.5     # intersection over the smaller area above which two are one body


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    size: int
    core: tuple[float, float, float, float]   # x0, y0, x1, y1 in full-frame px

    @property
    def window(self):
        """Slices that cut this tile out of a full frame."""
        return slice(self.y0, self.y0 + self.size), slice(self.x0, self.x0 + self.size)


def _starts(lo, length, size, overlap, frame):
    """Tile starts along one axis: overlapping by at least `overlap`, the last flush with the end."""
    if length <= size:
        return [min(lo, frame - size)]
    count = int(np.ceil((length - size) / (size - overlap))) + 1
    return [int(round(v)) for v in np.linspace(lo, lo + length - size, count)]


def _cores(starts, size):
    """Core interval of each tile: halfway into its overlap with each neighbour."""
    lows = [-np.inf] + [(a + size + b) / 2 for a, b in zip(starts, starts[1:])]
    return list(zip(lows, lows[1:] + [np.inf]))


def tiles(work_area, frame_size, size, overlap):
    """Tiles covering work_area (x0, y0, width, height) of a frame (width, height)."""
    x0, y0, w, h = work_area
    fw, fh = frame_size
    if size > fw or size > fh:
        raise ValueError(f"tile size {size} is larger than the {fw} x {fh} frame")
    xs, ys = _starts(x0, w, size, overlap, fw), _starts(y0, h, size, overlap, fh)
    return [
        Tile(x, y, size, (cx[0], cy[0], cx[1], cy[1]))
        for y, cy in zip(ys, _cores(ys, size))
        for x, cx in zip(xs, _cores(xs, size))
    ]


def clip_polygons(polygons, tile, min_visible_fraction):
    """Polygons (full-frame px) clipped to the tile, in tile px.

    A body keeps its label only if at least min_visible_fraction of its area is inside the tile;
    where clipping leaves several pieces, the largest is kept.
    """
    window = box(tile.x0 - 0.5, tile.y0 - 0.5, tile.x0 + tile.size - 0.5, tile.y0 + tile.size - 0.5)
    out = []
    for points in polygons:
        shape = Polygon(points)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.area <= 0 or not shape.intersects(window):
            continue
        inside = shape.intersection(window)
        if inside.area < min_visible_fraction * shape.area:
            continue
        piece = max(getattr(inside, "geoms", [inside]), key=lambda g: g.area)
        if piece.geom_type != "Polygon":
            continue
        xy = np.asarray(piece.exterior.coords)[:-1] - (tile.x0, tile.y0)
        out.append(xy)
    return out


def mask_to_polygon(mask, epsilon_px=1.0):
    """Outline of the largest blob in a binary mask, simplified by epsilon_px, or None."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, epsilon_px, True) if epsilon_px > 0 else contour
    if len(contour) < 3:
        return None
    return contour[:, 0, :].astype(float)


def merge(tile_predictions):
    """Full-frame predictions from per-tile ones.

    tile_predictions: (tile, polygons in tile px, scores) per tile. Returns (polygons, scores),
    polygons in full-frame px, highest score first.
    """
    kept = []
    for tile, polygons, scores in tile_predictions:
        cx0, cy0, cx1, cy1 = tile.core
        for points, score in zip(polygons, scores):
            points = np.asarray(points, float) + (tile.x0, tile.y0)
            shape = Polygon(points)
            if not shape.is_valid:
                shape = shape.buffer(0)
            if shape.area <= 0:
                continue
            c = shape.centroid
            if (cx0 - CORE_MARGIN_PX <= c.x < cx1 + CORE_MARGIN_PX
                    and cy0 - CORE_MARGIN_PX <= c.y < cy1 + CORE_MARGIN_PX):
                kept.append((float(score), points, shape))

    kept.sort(key=lambda k: -k[0])
    out = []
    for score, points, shape in kept:
        if any(
            shape.intersection(other).area > DUPLICATE_OVERLAP * min(shape.area, other.area)
            for _, _, other in out
        ):
            continue
        out.append((score, points, shape))
    return [p for _, p, _ in out], [s for s, _, _ in out]


def predict_frame(model, image, frame_tiles, conf=0.25, epsilon_px=1.0, batch=4, **kwargs):
    """Runs an ultralytics segmentation model over the tiles of a full frame and merges the results.

    image is a full BGR frame. Extra keyword arguments go to model.predict.
    Returns (polygons in full-frame px, scores).
    """
    per_tile = []
    for i in range(0, len(frame_tiles), batch):
        group = frame_tiles[i:i + batch]
        crops = [np.ascontiguousarray(image[t.window]) for t in group]
        results = model.predict(crops, imgsz=group[0].size, conf=conf, retina_masks=True,
                                verbose=False, **kwargs)
        for tile, result in zip(group, results):
            polygons, scores = [], []
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy() > 0.5
                for mask, score in zip(masks, result.boxes.conf.cpu().numpy()):
                    points = mask_to_polygon(mask, epsilon_px)
                    if points is not None:
                        polygons.append(points)
                        scores.append(float(score))
            per_tile.append((tile, polygons, scores))
    return merge(per_tile)

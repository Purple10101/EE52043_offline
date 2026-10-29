"""Skeleton and centroid of each resistor body the model finds.

Works on the outlines tiling.predict_frame returns (rectified px, leads excluded), one at a time:

- centroid: the outline's area centroid.
- skeleton: the filled outline, drawn SCALE times larger for sub-pixel detail, thinned to a
  one-pixel line (Zhang-Suen). Unordered; it stops about a half-width short of each end, and may
  fork there towards the corners.
- spine: the skeleton cleaned into one ordered line from end to end of the body, the line to sample
  the colour bands along. The longest path through the skeleton is kept; its ends are trimmed back
  to where the body is at least TRIM of its usual half-width there, which cuts off any forks; it is
  smoothed, and each point moved to the middle of the outline's chord across it there (thinning
  alone drifts off the medial axis by about a pixel on a turned body); then it is extended straight
  on from each end to the outline. On a straight body it is the axis to within a fraction of a
  pixel. It follows a bent outline in the middle, but its last half-width or so at each end is
  straight, so on a bent outline it cuts the ends' curve short.
- length (along the spine), width (the median of those chords) and yaw (the spine's end-to-end
  direction, 0 to 180 degrees in image axes, as evaluate.body_geometry).

Everything is in the outline's own px. Map with rectify.to_full (camera frame) or rectify.rect_to_mm
(plate mm) as for the outlines.

    from robot_offline import skeleton
    polygons, scores = tiling.predict_frame(model, image, tiles)
    bodies = [skeleton.body_skeleton(p) for p in polygons]
"""

import heapq
from dataclasses import dataclass

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

SCALE = 2           # the outline is drawn this many times larger before thinning
TRIM = 0.9          # spine ends cut back to where the half-width is this share of its median
STEP_PX = 1.0       # spacing of the spine's points, outline px

# Neighbours in Zhang-Suen order, P2 (north) clockwise to P9 (north-west), as (dy, dx).
_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _thinning_tables():
    """Per 8-neighbour code (bit i set: neighbour P(i+2) is on), whether each sub-iteration of
    Zhang-Suen removes the centre pixel (1) or keeps it (0)."""
    first, second = np.zeros(256, np.uint8), np.zeros(256, np.uint8)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]          # p[0] is P2 ... p[7] is P9
        b = sum(p)
        a = sum(p[i] == 0 and p[(i + 1) % 8] == 1 for i in range(8))
        if 2 <= b <= 6 and a == 1:
            p2, p4, p6, p8 = p[0], p[2], p[4], p[6]
            first[code] = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
            second[code] = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
    return first, second


_REMOVE = _thinning_tables()
_CODE_KERNEL = np.zeros((3, 3), np.float32)      # correlated with a 0/1 image, gives each pixel's code
for _i, (_dy, _dx) in enumerate(_RING):
    _CODE_KERNEL[1 + _dy, 1 + _dx] = 1 << _i


def thin(mask):
    """One-pixel-wide skeleton of a binary mask (Zhang-Suen thinning), as a bool array."""
    img = (np.asarray(mask) != 0).astype(np.uint8)
    while True:
        changed = False
        for table in _REMOVE:
            code = cv2.filter2D(img, -1, _CODE_KERNEL, borderType=cv2.BORDER_CONSTANT)
            remove = cv2.LUT(code, table) & img
            if cv2.countNonZero(remove):
                img -= remove
                changed = True
        if not changed:
            return img.astype(bool)


def _longest_path(skel):
    """Pixels (row, col) of the longest shortest path through a skeleton's largest piece, end to end.

    Two sweeps: the farthest pixel from any start is one end, the farthest from that is the other.
    Exact on a tree, which a thinned body is apart from rare small loops.
    """
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return np.zeros((0, 2), int)
    index = {(y, x): i for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()))}

    def sweep(start):
        dist, prev = {start: 0.0}, {start: None}
        heap = [(0.0, start)]
        while heap:
            d, (y, x) = heapq.heappop(heap)
            if d > dist[(y, x)]:
                continue
            for dy, dx in _RING:
                n = (y + dy, x + dx)
                if n in index:
                    nd = d + (1.4142135623730951 if dy and dx else 1.0)
                    if nd < dist.get(n, np.inf):
                        dist[n], prev[n] = nd, (y, x)
                        heapq.heappush(heap, (nd, n))
        far = max(dist, key=dist.get)
        return far, prev

    # Start in the largest piece: sweep from every unvisited pixel and keep the piece with most pixels.
    best, seen = None, set()
    for p in index:
        if p in seen:
            continue
        _, prev = sweep(p)
        seen.update(prev)
        if best is None or len(prev) > len(best[1]):
            best = (p, prev)
    a, _ = sweep(best[0])
    b, prev = sweep(a)
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return np.array(path[::-1])


def _ray_to_outline(start, direction, shape, reach):
    """Nearest point where a ray from start (inside shape) crosses its outline, or None."""
    ray = LineString([start, start + direction * reach])
    hit = ray.intersection(shape.exterior)
    points = [np.array(g.coords[0]) for g in getattr(hit, "geoms", [hit]) if not g.is_empty]
    if not points:
        return None
    return min(points, key=lambda q: np.hypot(*(q - start)))


def _centre_on_chords(line, shape, reach, iterations=2):
    """line with each point moved to the middle of the outline's chord across the line there, and
    the chords' lengths (nan where the chord did not close on both sides)."""
    boundary = shape.exterior
    lengths = np.full(len(line), np.nan)
    k = max(1, int(round(reach / 8)))                    # about a half-width of points each way
    for _ in range(iterations):
        i = np.arange(len(line))
        tangent = line[np.minimum(i + k, len(line) - 1)] - line[np.maximum(i - k, 0)]
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        chords = shapely.linestrings(np.stack([line - normal * reach, line + normal * reach], 1))
        points, which = shapely.get_coordinates(shapely.intersection(chords, boundary), return_index=True)
        s = np.einsum("ij,ij->i", points - line[which], normal[which])
        left, right = np.full(len(line), -np.inf), np.full(len(line), np.inf)
        np.maximum.at(left, which[s < 0], s[s < 0])          # nearest crossing on each side
        np.minimum.at(right, which[s >= 0], s[s >= 0])
        closed = np.isfinite(left) & np.isfinite(right)
        line = line.copy()
        line[closed] += normal[closed] * ((left + right)[closed] / 2)[:, None]
        lengths = np.where(closed, right - left, np.nan)
    return line, lengths


def _resample(line, step):
    """Polyline resampled at equal steps along its length, both ends kept."""
    seg = np.hypot(*np.diff(line, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return line[:1]
    at = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / step)) + 1))
    return np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


def _smooth(line, window):
    """Moving average along a polyline, ends held."""
    if window < 3 or len(line) < window:
        return line
    k = np.ones(window) / window
    half = window // 2
    padded = np.vstack([np.repeat(line[:1], half, 0), line, np.repeat(line[-1:], half, 0)])
    out = np.column_stack([np.convolve(padded[:, i], k, "valid") for i in range(2)])
    out[0], out[-1] = line[0], line[-1]
    return out


def _orient(line):
    """line, reversed if needed so its end-to-end direction is 0 to 180 degrees in image axes."""
    d = line[-1] - line[0]
    return line[::-1] if d[1] < 0 or (d[1] == 0 and d[0] < 0) else line


@dataclass(eq=False)
class BodySkeleton:
    centroid: np.ndarray    # (x, y), the outline's area centroid
    skeleton: np.ndarray    # M × 2, every thinned pixel centre, unordered
    spine: np.ndarray       # N × 2, end to end along the body, STEP_PX apart, oriented as yaw_deg
    length: float           # along the spine
    width: float            # median length of the outline's chords across the spine
    yaw_deg: float          # spine end-to-end direction in [0, 180), image axes (y down)

    @property
    def midpoint(self):
        """The point half way along the spine. Equal to the centroid on a whole, symmetric body."""
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(self.spine, axis=0).T))])
        half = s[-1] / 2
        return np.array([np.interp(half, s, self.spine[:, 0]), np.interp(half, s, self.spine[:, 1])])


def body_skeleton(polygon, scale=SCALE):
    """Centroid, skeleton and spine of one body outline (N × 2, px), or None if it has no area."""
    shape = Polygon(np.asarray(polygon, float).reshape(-1, 2))
    if not shape.is_valid:
        shape = shape.buffer(0)
        if shape.geom_type != "Polygon":                   # a self-crossing outline: its largest piece
            shape = max(getattr(shape, "geoms", []), key=lambda g: g.area, default=None)
    if shape is None or shape.is_empty or shape.area <= 0:
        return None
    poly = np.asarray(shape.exterior.coords)[:-1]
    centroid = np.array(shape.centroid.coords[0])

    # Fine grid: outline px p sits at fine px scale * (p + 0.5) - 0.5, so pixel centres line up.
    origin = np.floor(poly.min(0)) - 2
    fine = scale * (poly - origin + 0.5) - 0.5
    size = np.ceil(fine.max(0)).astype(int) + 2 * scale + 1
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, [np.round(fine * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    to_px = lambda rc: (rc[:, ::-1] + 0.5) / scale - 0.5 + origin
    skel = thin(mask)
    if not skel.any():
        return None
    skeleton_px = to_px(np.argwhere(skel).astype(float))
    radius = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    path = _longest_path(skel)
    r = radius[path[:, 0], path[:, 1]]
    keep = np.nonzero(r >= TRIM * np.median(r))[0]
    core = path[keep[0]:keep[-1] + 1] if len(keep) else path[:0]
    # The distance transform runs pixel centre to the nearest outside pixel centre: half a pixel long.
    half_width = max((np.median(r[keep[0]:keep[-1] + 1] if len(keep) else r) - 0.5) / scale, 0.5)
    width = 2 * half_width
    reach = 2 * (shape.length + 1)

    if len(core) >= 2 and np.hypot(*(core[-1] - core[0])) >= scale:
        line = _resample(_smooth(to_px(core.astype(float)), 2 * scale + 1), STEP_PX)
        # Its last half half-width at each end still bends towards the corner forks: cut off.
        cut = max(1, int(round(0.5 * half_width / STEP_PX)))
        if len(line) > 2 * cut + 2:
            line = line[cut:-cut]
        # Thinning is only near the medial axis, and drifts where the body widens at the end caps:
        # centred on the outline's chords across it, the line is exact to the outline.
        line, chords = _centre_on_chords(line, shape, 4 * half_width + 2)
        if np.isfinite(chords).any():
            width = float(np.nanmedian(chords))
        line = _smooth(line, 2 * int(half_width / STEP_PX / 2) + 1)    # the outline's pixel steps, evened out
        # Each end goes on straight to the outline, on the line fitted to its last width of spine.
        run = max(2, int(round(2 * half_width / STEP_PX)))
        ends = []
        for seg in (line[:run][::-1], line[-run:]):        # each from inside towards its tip
            centre = seg.mean(0)
            direction = np.linalg.eigh(np.cov((seg - centre).T))[1][:, -1]
            if (seg[-1] - seg[0]) @ direction < 0:
                direction = -direction
            tip = centre + direction * ((seg[-1] - centre) @ direction)
            hit = _ray_to_outline(tip, direction, shape, reach)
            ends.append([tip] if hit is None else [tip, hit])
        line = np.vstack([np.array(ends[0][::-1]), line, np.array(ends[1])])
    else:
        # Too short to have a medial line (a body about as wide as it is long): the long axis
        # through the centroid, outline to outline.
        m = cv2.moments(poly.astype(np.float32))
        angle = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
        d = np.array([np.cos(angle), np.sin(angle)])
        a = _ray_to_outline(centroid, -d, shape, reach)
        b = _ray_to_outline(centroid, d, shape, reach)
        line = np.array([a if a is not None else centroid, b if b is not None else centroid])

    spine = _orient(_resample(line, STEP_PX))
    length = float(np.hypot(*np.diff(spine, axis=0).T).sum())
    d = spine[-1] - spine[0]
    yaw = float(np.degrees(np.arctan2(d[1], d[0])) % 180) if length > 0 else 0.0
    return BodySkeleton(centroid, skeleton_px, spine, length, float(width), yaw)

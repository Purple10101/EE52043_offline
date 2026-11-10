"""Grouping resistors by how their bands look, without reading a value.

Two bodies belong together if their bands look the same. Nothing here names a colour or decodes a
value: a body becomes a colour profile along its axis, two profiles are compared directly, and
similar bodies are clustered. See docs/classification.md.

The comparison has to allow for what changes between two views of the same part:

- Which way round it lies: each profile is also tried reversed.
- Where the bands sit along the body and how stretched they look (tilt, the outline's error): the
  profiles are aligned by dynamic time warping within WARP_RADIUS.
- The ends: the camera looks at bodies at an angle, so an end facing it shows its cap and an end
  facing away hides it. The ends of a profile really do differ between views of one part, so up to
  END_SKIP samples at each end are left out of the alignment at no cost.
- The body colour, which is removed so that only the bands count.

Parts that differ by one band have identical profiles elsewhere, so a mean difference would hide
them: differences are combined with a POWER norm, which the worst mismatch dominates.

A single pair of views is not reliable: distances between views of the same part overlap those
between different parts. Only the average over a group separates them, hence average linkage and
`assign` comparing against a whole group.

    python -m robot_offline.post.bands.grouping --sessions 2026-10-23_a 2026-10-23_b

groups the crops in data/bands/crops.json into data/bands/groups.json and a sheet per group, each
crop turned to face the same way as the group's most typical member, the least typical last.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_offline import config

SAMPLES = 96                # along a body's profile
CENTRE_WIDTH = 0.4          # of the body's width, sampled around its centre line
BRIGHTEST_DROPPED = 25      # percent of each column's pixels, against highlights
SMOOTHING = 5               # samples, along the profile before resampling
LIGHTNESS_WEIGHT = 0.3      # against a* and b*: shading and highlights mostly change lightness
WARP_RADIUS = 24            # samples
END_SKIP = 20               # samples
POWER = 4
THRESHOLD = 17.0            # average distance above which two groups stay apart


def _largest_part(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if count > 1 else mask


def profile(image, mask):
    """SAMPLES × 3 colour profile of a straightened body, its median (the body colour) removed.

    image: H × W × 3 uint8 BGR crop, the body's axis left to right; mask: H × W, nonzero on the body.
    Columns narrower than half the body's width (lead stubs, the tips of the outline) are left out.
    """
    body = _largest_part(np.asarray(mask) > 127)
    heights = body.sum(0)
    columns = np.where(heights > 0.5 * heights.max())[0]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    samples = []
    for x in range(columns[0], columns[-1] + 1):
        rows = np.where(body[:, x])[0]
        if len(rows) < 3:
            samples.append(samples[-1] if samples else lab[rows[0], x] if len(rows) else lab[0, x])
            continue
        centre, half = (rows[0] + rows[-1]) / 2, max(2, (rows[-1] - rows[0]) * CENTRE_WIDTH / 2)
        pixels = lab[int(centre - half):int(centre + half) + 1, x]
        pixels = pixels[pixels[:, 0] <= np.percentile(pixels[:, 0], 100 - BRIGHTEST_DROPPED)]
        samples.append(np.median(pixels, 0))
    line = cv2.GaussianBlur(np.array(samples, np.float32)[None], (SMOOTHING, 1), 0)[0]

    t = np.linspace(0, len(line) - 1, SAMPLES)
    resampled = np.stack([np.interp(t, np.arange(len(line)), line[:, c]) for c in range(3)], 1)
    resampled -= np.median(resampled, 0)
    return (resampled * (LIGHTNESS_WEIGHT, 1, 1)).astype(np.float32)


def _warped(a, others):
    """Distance from profile a to each of others (M × SAMPLES × 3), both free to skip their ends."""
    n, m = len(a), len(others)
    cost = (np.sqrt(((a[None, :, None] - others[:, None]) ** 2).sum(3)) + 1e-6) ** POWER
    far = np.float32(1e30)
    previous = np.full((m, n), far, np.float32)
    for i in range(n):
        current = np.full((m, n), far, np.float32)
        for j in range(max(0, i - WARP_RADIUS), min(n, i + WARP_RADIUS + 1)):
            if (i <= END_SKIP and j == 0) or (j <= END_SKIP and i == 0):
                reached = 0.0
            else:
                reached = previous[:, j]
                if j > 0:
                    reached = np.minimum(reached, np.minimum(previous[:, j - 1], current[:, j - 1]))
            current[:, j] = reached + cost[:, i, j]
        previous = current
        if i == n - 1 - END_SKIP:
            ended = current[:, -1].copy()
        elif i > n - 1 - END_SKIP:
            ended = np.minimum(ended, current[:, -1])
    ended = np.minimum(ended, previous[:, n - 1 - END_SKIP:].min(1))
    return (ended / (n - END_SKIP)) ** (1 / POWER)


def compare(a, others):
    """(distances, reversed) from profile a to each of others, trying each both ways round."""
    others = np.asarray(others, np.float32)
    both = _warped(a, np.concatenate([others, others[:, ::-1]]))
    forwards, backwards = both[:len(others)], both[len(others):]
    return np.minimum(forwards, backwards), backwards < forwards


def distances(profiles):
    """(n × n distances, n × n flags: j reversed to match i) between profiles."""
    pairs = [compare(p, profiles) for p in profiles]
    d = np.array([pair[0] for pair in pairs])
    return np.minimum(d, d.T), np.array([pair[1] for pair in pairs])


def cluster(d, threshold=THRESHOLD):
    """Group labels (0, 1, ... by first member) from an n × n distance matrix, by average linkage."""
    n = len(d)
    between = np.array(d, np.float64)
    np.fill_diagonal(between, np.inf)
    members = {i: [i] for i in range(n)}
    while len(members) > 1:
        a, b = np.unravel_index(np.argmin(between), between.shape)
        if between[a, b] > threshold:
            break
        size_a, size_b = len(members[a]), len(members[b])
        merged = (size_a * between[a] + size_b * between[b]) / (size_a + size_b)
        between[a], between[:, a] = merged, merged
        between[a, a] = np.inf
        between[b], between[:, b] = np.inf, np.inf
        members[a] += members.pop(b)
    labels = np.empty(n, int)
    for label, group in enumerate(sorted(members.values(), key=min)):
        labels[group] = label
    return labels


def assign(p, groups, threshold=THRESHOLD):
    """Index of the group (each an array of profiles) profile p belongs to, or None if none is close.

    A new body is compared with whole groups: its average distance to a group's members.
    """
    averages = [compare(p, group)[0].mean() for group in groups]
    if not averages or min(averages) > threshold:
        return None
    return int(np.argmin(averages))


def _sheet(images, per_row=8, cell=(172, 84)):
    tiles = []
    for image in images:
        tile = np.full((cell[1], cell[0], 3), 30, np.uint8)
        scale = min((cell[0] - 4) / image.shape[1], (cell[1] - 4) / image.shape[0])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile[2:2 + small.shape[0], 2:2 + small.shape[1]] = small
        tiles.append(tile)
    while len(tiles) % per_row:
        tiles.append(np.full((cell[1], cell[0], 3), 30, np.uint8))
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m robot_offline.post.bands.grouping",
                                     description="Group body crops by how their bands look.")
    parser.add_argument("--sessions", nargs="+", help="only these sessions (default: every crop)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--folder", type=Path, default=config.DATA_DIR / "bands")
    args = parser.parse_args(argv)

    entries = json.loads((args.folder / "crops.json").read_text())
    if args.sessions:
        entries = [e for e in entries if e["session"] in args.sessions]
    images = [cv2.imread(str(args.folder / e["file"])) for e in entries]
    masks = [cv2.imread(str(args.folder / e["file"].replace(".png", "_mask.png")), cv2.IMREAD_GRAYSCALE)
             for e in entries]
    d, flipped = distances(np.array([profile(i, m) for i, m in zip(images, masks)]))
    labels = cluster(d, args.threshold)

    sheets = args.folder / "groups"
    sheets.mkdir(exist_ok=True)
    for old in sheets.glob("group_*.jpg"):
        old.unlink()
    for label in range(labels.max() + 1):
        group = np.where(labels == label)[0]
        typical = group[np.argmin(d[np.ix_(group, group)].sum(1))]
        order = group[np.argsort(d[typical, group])]
        turned = [images[i][:, ::-1] if flipped[typical, i] else images[i] for i in order]
        cv2.imwrite(str(sheets / f"group_{label:02d}.jpg"), _sheet(turned))
        print(f"  group {label}: {len(group)} bodies, most typical {entries[typical]['file']}")
    (args.folder / "groups.json").write_text(json.dumps(
        [{"file": e["file"], "group": int(g)} for e, g in zip(entries, labels)], indent=1))
    print(f"{len(entries)} bodies in {labels.max() + 1} groups: {args.folder / 'groups.json'}, {sheets}")


if __name__ == "__main__":
    main()

"""Detections held steady from run to run, for live tools: a body's score can swing between frames.

On 2026-11-09 a dark green body scored 0.18 to 0.78 over 15 frames in a row (passing 0.5 in 7), while
blue ones held at 0.91 to 0.93. On every one of 40 runs the model still found it at a low score; it was
only the score that swung. So the model runs with a low floor (FLOOR), and each body is followed from
run to run by where it is:

- it appears once it reaches conf in `need` of the last `window` runs,
- and stays while its mean score over them is at least `keep`, and it is found at all (at FLOOR) in at
  least one of the last `gone` runs.

A one-off false detection doesn't reach conf twice. A body taken away isn't found at all from then on,
which a body still there almost never is (the swinging green one was found in 40 runs of 40), so it goes
after `gone` runs. A body whose score swings round conf stays. The score given is that mean, so a
weak body still reads as weak.

Bodies are matched by centre within MATCH_PX of the outline's px (rectified: 10 px/mm), so this is for a
still scene: the arm parked. A body moved further than that is followed afresh.

    steady = Steady(conf=0.5)
    polygons, scores = tiling.predict_frame(model, image, tiles, conf=FLOOR, ...)
    polygons, scores = steady.update(polygons, scores)
"""

from collections import deque

import numpy as np

FLOOR = 0.05                # predict at this, so a body is seen on the runs it scores low
MATCH_PX = 20.0             # one body if their centres are this close (2 mm at 10 px/mm)


class _Body:
    count = 0

    def __init__(self, polygon, score, window):
        _Body.count += 1
        self.id = _Body.count       # for following things about a body (its band reading) from run to run
        self.polygon, self.centre = polygon, polygon.mean(0)
        self.scores = deque([score], maxlen=window)
        self.shown = False
        self.missed = 0             # runs in a row it wasn't found at all


class Steady:
    """Follows bodies from run to run. conf: the threshold, as for one run; window, need, keep: see above."""

    def __init__(self, conf=0.5, window=5, need=2, keep=0.25, gone=2):
        self.conf, self.window, self.need, self.keep, self.gone = conf, window, need, keep, gone
        self.bodies = []
        self.ids = []               # the id of each body the last update returned, in order

    def update(self, polygons, scores):
        """One run's outlines and scores (taken at FLOOR or so). Returns (polygons, scores) of the bodies to
        show: each one's newest outline and its mean score over the window."""
        found = sorted(zip(polygons, scores), key=lambda d: -d[1])
        seen = set()
        for polygon, score in found:
            polygon = np.asarray(polygon, float)
            centre = polygon.mean(0)
            near = [(np.linalg.norm(b.centre - centre), i) for i, b in enumerate(self.bodies) if i not in seen]
            near = [(d, i) for d, i in near if d <= MATCH_PX]
            if near:
                i = min(near)[1]
                body = self.bodies[i]
                body.polygon, body.centre, body.missed = polygon, centre, 0
                body.scores.append(float(score))
            else:
                self.bodies.append(_Body(polygon, float(score), self.window))
                i = len(self.bodies) - 1
            seen.add(i)
        for i, body in enumerate(self.bodies):
            if i not in seen:
                body.scores.append(0.0)
                body.missed += 1
        kept = []
        for body in self.bodies:
            mean = float(np.mean(body.scores))
            if not body.shown:
                body.shown = sum(s >= self.conf for s in body.scores) >= self.need
            elif mean < self.keep or body.missed >= self.gone:
                body.shown = False
            if body.missed >= self.gone:
                continue                # gone: forgotten, and followed afresh if it comes back
            if body.shown or mean >= self.keep / 2 or len(body.scores) < self.window:
                kept.append(body)
        self.bodies = kept
        shown = [b for b in self.bodies if b.shown]
        self.ids = [b.id for b in shown]
        return [b.polygon for b in shown], [float(np.mean(b.scores)) for b in shown]

    def clear(self):
        """Forget every body (the scene changed: a new frame, the plate found again)."""
        self.bodies = []

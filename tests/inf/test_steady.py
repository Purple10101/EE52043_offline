import numpy as np

from robot_offline.inf.steady import Steady


def body(x, y, half=25):
    return np.array([[x - half, y - 10], [x + half, y - 10], [x + half, y + 10], [x - half, y + 10]], float)


def run(steady, *found, jitter=1.0, rng=np.random.default_rng(0)):
    """One run: found is (x, y, score) per body, outlines jittered by about a pixel."""
    polys = [body(x, y) + rng.normal(0, jitter, (4, 2)) for x, y, _ in found]
    return steady.update(polys, [s for _, _, s in found])


def shown_at(polygons, x, y):
    return sum(np.linalg.norm(p.mean(0) - (x, y)) < 10 for p in polygons)


def test_a_steady_body_shows_from_the_second_run():
    steady = Steady(conf=0.5)
    assert run(steady, (100, 100, 0.93))[0] == []
    for _ in range(20):
        polys, scores = run(steady, (100, 100, 0.93))
        assert shown_at(polys, 100, 100) == 1 and abs(scores[0] - 0.93) < 0.01


def test_a_body_whose_score_swings_round_conf_stays_shown():
    # As the dark green body on 2026-11-09: 0.18 to 0.78 from frame to frame, over 0.5 about half the time.
    rng = np.random.default_rng(1)
    steady, shown = Steady(conf=0.5), []
    for _ in range(200):
        polys, _ = run(steady, (300, 300, rng.uniform(0.18, 0.78)))
        shown.append(shown_at(polys, 300, 300))
    assert sum(shown[10:]) == len(shown[10:])           # every run once it has appeared
    assert max(shown) == 1


def test_a_one_off_false_detection_never_shows():
    steady = Steady(conf=0.5)
    for i in range(30):
        found = [(100, 100, 0.9)] + ([(500, 500, 0.7)] if i == 10 else []) + [(700, 200, 0.1)]
        polys, _ = run(steady, *found)
        assert shown_at(polys, 500, 500) == 0 and shown_at(polys, 700, 200) == 0


def test_a_body_taken_away_goes_within_a_few_runs():
    steady = Steady(conf=0.5)
    for _ in range(10):
        run(steady, (100, 100, 0.9), (400, 100, 0.9))
    for k in range(1, 10):
        polys, _ = run(steady, (100, 100, 0.9))
        if shown_at(polys, 400, 100) == 0:
            break
    assert k == 2 and shown_at(polys, 100, 100) == 1


def test_bodies_side_by_side_are_kept_apart():
    steady = Steady(conf=0.5)
    for _ in range(10):
        polys, _ = run(steady, (100, 100, 0.9), (140, 100, 0.8))
    assert shown_at(polys, 100, 100) == 1 and shown_at(polys, 140, 100) == 1 and len(polys) == 2


def test_clear_forgets_everything():
    steady = Steady(conf=0.5)
    for _ in range(5):
        run(steady, (100, 100, 0.9))
    steady.clear()
    assert run(steady, (100, 100, 0.9))[0] == []


def test_each_body_shown_keeps_its_id():
    steady = Steady(conf=0.5)
    for _ in range(3):
        run(steady, (100, 100, 0.9), (400, 100, 0.9))
    first = dict(zip([round(p.mean(0)[0], -2) for p in run(steady, (100, 100, 0.9), (400, 100, 0.9))[0]], steady.ids))
    for _ in range(3):
        polys, _ = run(steady, (400, 100, 0.9), (100, 100, 0.9))
    assert dict(zip([round(p.mean(0)[0], -2) for p in polys], steady.ids)) == first

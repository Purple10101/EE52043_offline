import numpy as np

from robot_offline.post.bands import grouping

BODY = (170, 190, 40)                                   # BGR
ORANGE, YELLOW, GREEN, BROWN, RED = (40, 120, 230), (30, 210, 230), (60, 150, 30), (30, 60, 110), (50, 30, 200)
KIND_A = (BROWN, GREEN, ORANGE, GREEN, BROWN)
KIND_B = (BROWN, GREEN, YELLOW, GREEN, BROWN)           # differs from KIND_A by one band
KIND_C = (RED, GREEN, BROWN, YELLOW, RED)


def _crop(bands, shift=0, stretch=1.0, cap=False, reverse=False):
    """A straightened body: bands off-centre, optionally moved, stretched or with a dark end cap."""
    image = np.full((70, 200, 3), 200, np.uint8)
    mask = np.zeros((70, 200), np.uint8)
    image[10:60, 10:190], mask[10:60, 10:190] = BODY, 255
    for k, colour in enumerate(bands):
        x = int(50 + shift + 22 * k * stretch)
        image[10:60, x:x + int(12 * stretch)] = colour
    if cap:
        image[10:60, 10:22] = (30, 30, 30)
    return (image[:, ::-1].copy(), mask[:, ::-1].copy()) if reverse else (image, mask)


def _profiles(crops):
    return np.array([grouping.profile(*c) for c in crops])


def test_a_profile_has_its_body_colour_removed():
    p = grouping.profile(*_crop(KIND_A))
    assert p.shape == (grouping.SAMPLES, 3)
    assert np.allclose(np.median(p, 0), 0, atol=1)
    assert np.abs(p).max() > 20                         # the bands are still there


def test_the_same_part_stays_close_however_it_lies():
    a = grouping.profile(*_crop(KIND_A))
    views = _profiles([_crop(KIND_A, reverse=True), _crop(KIND_A, shift=8), _crop(KIND_A, stretch=1.1),
                       _crop(KIND_A, cap=True)])
    d, _ = grouping.compare(a, views)
    assert (d < grouping.THRESHOLD).all()


def test_compare_says_which_views_it_turned_round():
    a = grouping.profile(*_crop(KIND_C))                # KIND_C reads differently from each end
    _, flipped = grouping.compare(a, _profiles([_crop(KIND_C, reverse=True), _crop(KIND_C, shift=8)]))
    assert flipped.tolist() == [True, False]


def test_one_different_band_is_enough_to_keep_parts_apart():
    d, _ = grouping.compare(grouping.profile(*_crop(KIND_A)), _profiles([_crop(KIND_B)]))
    assert d[0] > grouping.THRESHOLD


def test_cluster_finds_the_kinds_without_being_told_how_many():
    views = [dict(), dict(reverse=True), dict(shift=6), dict(cap=True, reverse=True)]
    crops = [_crop(kind, **v) for kind in (KIND_A, KIND_B, KIND_C) for v in views]
    d, _ = grouping.distances(_profiles(crops))
    labels = grouping.cluster(d)
    assert labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_cluster_uses_the_average_so_one_close_pair_does_not_join_two_groups():
    d = np.full((4, 4), 30.0)
    d[0, 1] = d[1, 0] = d[2, 3] = d[3, 2] = 2.0
    d[1, 2] = d[2, 1] = 2.5                             # one close pair across the two groups
    np.fill_diagonal(d, 0)
    assert grouping.cluster(d, threshold=10).tolist() == [0, 0, 1, 1]


def test_assign_finds_a_known_kind_and_refuses_an_unknown_one():
    groups = [_profiles([_crop(KIND_A), _crop(KIND_A, shift=4)]), _profiles([_crop(KIND_B), _crop(KIND_B, shift=4)])]
    assert grouping.assign(grouping.profile(*_crop(KIND_B, reverse=True)), groups) == 1
    assert grouping.assign(grouping.profile(*_crop(KIND_C)), groups) is None
    assert grouping.assign(grouping.profile(*_crop(KIND_A)), []) is None

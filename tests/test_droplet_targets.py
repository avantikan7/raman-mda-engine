import numpy as np

from raman_mda_engine.aiming.droplets import droplet_targets


def _disc(shape, center, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= r**2


def test_aim_inside_crescent_where_centroid_is_outside():
    masks = np.zeros((100, 100), int)
    masks[_disc(masks.shape, (50, 50), 30) & ~_disc(masks.shape, (50, 62), 26)] = 1
    (t,) = droplet_targets(masks, density=3)
    assert masks[tuple(np.rint(t["centroid"]).astype(int))] == 0  # centroid misses
    assert masks[tuple(t["aim"])] == 1
    assert all(masks[r, c] == 1 for r, c in t["grid"])


def test_grid_is_inside_disc_with_edge_margin_and_spacing():
    masks = np.zeros((80, 80), int)
    masks[_disc(masks.shape, (40, 40), 20)] = 3
    (t,) = droplet_targets(masks, density=5, margin=2)
    assert t["label"] == 3 and abs(t["radius"] - 20) < 1.5
    assert len(t["grid"]) > 10
    assert np.all(np.hypot(*(t["grid"] - 40).T) <= 20 - 2 + 1)


def test_small_objects_are_skipped():
    masks = np.zeros((50, 50), int)
    masks[5:7, 5:7] = 1
    assert droplet_targets(masks) == []


def test_dim_and_oversized_objects_are_filtered():
    masks = np.zeros((120, 120), int)
    image = np.zeros((120, 120))
    masks[_disc(masks.shape, (30, 30), 10)] = 1
    image[masks == 1] = 100.0
    masks[_disc(masks.shape, (80, 80), 10)] = 2
    image[masks == 2] = 10.0  # dim
    assert [t["label"] for t in droplet_targets(masks, image=image, min_rel_intensity=0.3)] == [1]
    assert [t["label"] for t in droplet_targets(masks, max_area=100)] == []


def test_link_labels_across_z_keeps_ids_and_starts_new_ones():
    from raman_mda_engine.aiming.droplets import link_labels_across_z

    stack = np.zeros((3, 60, 60), int)
    stack[0][_disc((60, 60), (20, 20), 8)] = 5
    stack[1][_disc((60, 60), (21, 20), 8)] = 2          # same droplet, new raw id
    stack[1][_disc((60, 60), (45, 45), 6)] = 9          # new droplet
    stack[2][_disc((60, 60), (45, 46), 6)] = 1          # follows the second one
    out = link_labels_across_z(stack)
    a = out[0][20, 20]
    b = out[1][45, 45]
    assert out[1][21, 20] == a and a != b and out[2][45, 46] == b


def test_droplet_point_rows_best_plane_and_grid_options():
    from raman_mda_engine.aiming.droplets import droplet_point_rows

    small = {"label": 1, "track": 7, "radius": 5.0, "aim": np.array([10, 10]),
             "grid": np.array([[10, 12], [12, 10]])}
    wide = {"label": 2, "track": 7, "radius": 9.0, "aim": np.array([11, 11]),
            "grid": np.array([[11, 14]])}
    by_z = {0: [small], 1: [wide]}
    rows = droplet_point_rows(by_z)
    assert {r[0] for r in rows} == {1}                      # only the widest plane
    assert [r[2] for r in rows] == ["aim", "grid"]
    assert len(droplet_point_rows(by_z, include_grid=False)) == 1
    assert len(droplet_point_rows(by_z, best_plane_only=False)) == 5

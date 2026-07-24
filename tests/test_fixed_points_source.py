from types import SimpleNamespace

import numpy as np

from raman_mda_engine.aiming import FixedPointsSource
from raman_mda_engine.aiming.transformers import Identity


def test_fixed_points_source_reuses_points_for_every_position():
    source = FixedPointsSource(
        [[20, 50], [20, 50]],
        name="cells",
        img_shape=(100, 200),
        transformer=Identity(),
    )

    np.testing.assert_allclose(
        source.get_mda_points(SimpleNamespace(index={"p": 19_999})),
        [[0.2, 0.25], [0.2, 0.25]],
    )
    assert source._base_points.shape == (2, 2)
    assert source._position_points == {}


def test_fixed_points_source_stores_only_tracked_position_overrides():
    source = FixedPointsSource(
        [[20, 50]],
        name="cells",
        img_shape=(100, 200),
        transformer=Identity(),
    )
    source.set_mda_points(7, [[40, 100]])

    np.testing.assert_allclose(
        source.get_mda_points(
            SimpleNamespace(index={"p": 7}), transform=False
        ),
        [[0.4, 0.5]],
    )
    np.testing.assert_allclose(
        source.get_mda_points(
            SimpleNamespace(index={"p": 8}), transform=False
        ),
        [[0.2, 0.25]],
    )
    assert list(source._position_points) == [7]

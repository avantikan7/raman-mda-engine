from itertools import islice

import pytest
from useq import MDASequence, AcquireImage

from raman_mda_engine.grid_plan import GRID_KEY, grid_count, grid_position, iter_grid_events


def plan(nx=2, ny=3, snake=None):
    return {"version": 1, "regions": [{"x": [10, 20, nx], "y": [-3, 3, ny],
                                        "z": 100, "tilt": None, "snake_axis": snake}]}


@pytest.mark.parametrize("order", ["tpcz", "cpzt", "ztpc"])
@pytest.mark.parametrize("snake", [None, "x", "y"])
def test_lazy_grid_events_match_native_explicit_events(order, snake):
    sequence = MDASequence(axis_order=order, metadata={GRID_KEY: plan(snake=snake)},
                           channels=[{"config": "BF", "exposure": 3, "z_offset": 2},
                                     {"config": "RM", "exposure": 100, "do_stack": False, "acquire_every": 2}],
                           time_plan={"loops": 3, "interval": 5}, z_plan={"relative": [-1, 0, 1]})
    positions = [grid_position(sequence, i) for i in range(grid_count(sequence))]
    native = sequence.replace(metadata={}, stage_positions=positions)
    actual = list(iter_grid_events(sequence))
    expected = list(native)
    def comparable(event):
        return (dict(event.index), event.x_pos, event.y_pos, event.z_pos,
                event.channel.config, event.exposure, event.min_start_time,
                event.reset_event_timer)
    assert list(map(comparable, actual)) == list(map(comparable, expected))
    assert all(isinstance(event.action, AcquireImage) for event in actual)
    assert not sequence.stage_positions


def test_million_position_plan_and_first_events_stay_compact():
    sequence = MDASequence(metadata={GRID_KEY: plan(1000, 1000)}, channels=["RM"])
    assert grid_count(sequence) == 1_000_000
    assert len(sequence.model_dump_json()) < 2000
    assert [event.index["p"] for event in islice(iter_grid_events(sequence), 3)] == [0, 1, 2]
    assert sequence.stage_positions == ()


def test_multiple_regions_and_tilt_keep_global_indices():
    definition = plan(2, 1)
    definition["regions"].append({"x": [30, 40, 2], "y": [0, 0, 1], "z": None,
                                  "snake_axis": None,
                                  "tilt": {"degree": 1, "origin": [0, 0], "scale": [1, 1],
                                           "coefficients": [2, 3, 4]}})
    sequence = MDASequence(metadata={GRID_KEY: definition}, channels=["RM"])
    events = list(iter_grid_events(sequence))
    assert [event.index["p"] for event in events] == [0, 1, 2, 3]
    assert [event.z_pos for event in events] == [100, 100, 92, 122]

from unittest.mock import MagicMock

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent, MDASequence

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming import SimpleGridSource, SnappableRamanAimingSource


def test_snappable():
    grid = SimpleGridSource(5, 5)
    assert isinstance(grid, SnappableRamanAimingSource)


def test_channel_less_event_collects_raman_at_requested_z():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._mmc.getChannelGroup.return_value = "Channel"
    engine._rm_meta = {"z": [0]}
    engine._rm_channel = "BF"
    engine._rm_z = np.array([0])
    engine._skip_imaging_for_same_pos = False
    engine._z_rel = [0]
    engine.record_raman = MagicMock()
    engine.try_set_config = MagicMock()
    engine.try_snap_image = MagicMock()
    engine.get_frame_metadata = MagicMock(return_value={})
    engine.try_get_image = MagicMock(return_value=np.zeros((2, 2)))
    event = MDAEvent(index={"p": 0, "z": 0})

    payloads = list(engine.exec_event(event))

    engine.record_raman.assert_called_once_with(event)
    engine.try_set_config.assert_called_once_with("Channel", "BF")
    assert len(payloads) == 1


def test_mda_no_autofocus(core: CMMCorePlus, engine: RamanEngine):
    seq = MDASequence(
        metadata={"raman": {"z": "center"}},
        channels=["BF"],
        time_plan={"interval": 1, "loops": 3},
        # z_plan={"range": 30, "step": 10},
        z_plan={"relative": [-15, 0, 15]},
        axis_order="tpcz",
        stage_positions=[(0, 1, 1), (512, 128, 0)],
    )

    rm_mock = MagicMock()
    engine.raman_events.ramanSpectraReady.connect(rm_mock)
    core.mda.run(seq)
    assert rm_mock.call_count == 6

    rm_mock.reset_mock()
    seq = seq.replace(metadata={"raman": {"z": "all"}})
    core.mda.run(seq)
    assert rm_mock.call_count == 18

    # remove aiming source and check that we raise an error
    engine.aiming_sources = []
    with pytest.raises(RuntimeError, match="No aiming sources - cannot collect Raman."):
        core.mda.run(seq)


# TODO: test with autofocus!!

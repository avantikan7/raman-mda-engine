from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent, MDASequence

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming import SimpleGridSource, SnappableRamanAimingSource


def test_snappable():
    grid = SimpleGridSource(5, 5)
    assert isinstance(grid, SnappableRamanAimingSource)


def test_engine_stores_segmentation_circle_radius(core: CMMCorePlus):
    engine = RamanEngine(
        mmc=core,
        spectra_collector=MagicMock(daq=MagicMock()),
        circle_radius=123,
    )

    assert engine._circle_radius == 123


def test_laser_autofocus_stage_positions_use_pixel_to_stage_model():
    engine = object.__new__(RamanEngine)
    engine._image_x = 200
    engine._image_y = 100
    # Vandermonde term order for degree 1 is [1, y, x].
    engine._stage_centering_model = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=float),
        1,
    )
    engine.try_get_XYPosition = MagicMock(return_value=(1000, 2000))

    # Historical source normalization recovers raw point (y=40, x=150).
    original, target = engine._laser_autofocus_stage_positions(
        np.array([40 / 200, 150 / 100])
    )

    assert original == (1000, 2000)
    # Pixel offset from center is (x=50, y=-10), and stage target follows
    # the existing click-to-center convention: original - fitted offset.
    assert target == (950, 2010)


def test_laser_autofocus_centers_galvo_and_restores_stage():
    engine = object.__new__(RamanEngine)
    engine._autofocus_object = "laser"
    engine._autofocus_search_range = 2
    engine._search_pts = 2
    engine._fine_search_range = 1
    engine._fine_search_pts = 2
    engine._shutter_device = None
    engine._max_volt = 1.6
    engine._image_x = 200
    engine._image_y = 100
    engine._stage_centering_model = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=float),
        1,
    )
    engine._last_operation_reloaded = False
    engine._laser_autofocus_active = False
    engine._transformer = MagicMock()
    engine._transformer.BF_to_volts.return_value = np.array([[0.2, -0.3]])
    engine._spectra_collector = MagicMock()
    engine._spectra_collector.collect_spectra_pts.return_value = np.zeros(
        (2, 8)
    )
    engine._mmc = MagicMock()
    engine.try_get_ZPosition = MagicMock(return_value=10)
    engine.try_get_XYPosition = MagicMock(return_value=(1000, 2000))
    stage_settling = MagicMock()
    engine.try_set_XYPosition = stage_settling.set_xy
    engine.try_set_ZPosition = MagicMock()
    engine.try_set_config = MagicMock()
    engine.try_setExp = MagicMock()
    engine.try_snap = MagicMock(return_value=np.ones((3, 4)))
    engine._set_focus_filter = MagicMock()

    with patch(
        "raman_mda_engine._engine.time.sleep",
        new=stage_settling.sleep,
    ):
        engine.autofocus_w_raman(
            last_z=10,
            pt=np.array([40 / 200, 150 / 100]),
            t=0,
            p=0,
        )

    engine._transformer.BF_to_volts.assert_called_once()
    np.testing.assert_array_equal(
        engine._transformer.BF_to_volts.call_args.args[0],
        np.array([[0.5, 0.5]]),
    )
    written_volts = (
        engine._spectra_collector.collect_spectra_pts.call_args.args[0]
    )
    np.testing.assert_array_equal(
        written_volts,
        np.array([[0.2, -0.3], [0.2, -0.3]]),
    )
    assert stage_settling.set_xy.call_args_list == [
        call(950.0, 2010.0),
        call(1000.0, 2000.0),
    ]
    assert stage_settling.mock_calls[:2] == [
        call.set_xy(950.0, 2010.0),
        call.sleep(0.1),
    ]


def test_setup_event_adds_offset_for_each_explicit_channel():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._mmc.getXPosition.return_value = 1
    engine._mmc.getYPosition.return_value = 2
    engine._autofocus = False
    engine._segment_and_track = False
    engine._rm_meta = {"channel": "RM", "z": [0]}
    engine._rm_channel = "RM"
    engine._channel_z_offsets = {"GFP": 1.25, "RM": -5.0}
    engine.try_set_XYPosition = MagicMock()
    engine.try_set_ZPosition = MagicMock()
    engine.try_set_config = MagicMock()

    for channel, expected_z in (("GFP", 101.25), ("RM", 95.0)):
        event = MDAEvent(
            index={"p": 0, "c": 0, "z": 0},
            channel={"group": "Channel", "config": channel},
            z_pos=100,
        )
        engine.setup_event(event)
        assert engine.try_set_ZPosition.call_args == call(expected_z)


def test_channel_less_raman_event_uses_raman_channel_offset():
    engine = object.__new__(RamanEngine)
    engine._rm_meta = {"channel": "RM", "z": [0]}
    engine._rm_channel = "RM"
    engine._channel_z_offsets = {"RM": -4.5}

    event = MDAEvent(index={"p": 0, "z": 0})

    assert engine._channel_offset_for_event(event) == -4.5


def test_channel_less_raman_event_has_no_camera_payload():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._mmc.getChannelGroup.return_value = "Channel"
    engine._rm_meta = {"z": [0]}
    engine._rm_channel = "RM"
    engine._rm_z = np.array([0])
    engine._pre_acq = False
    engine._skip_imaging_for_same_pos = False
    engine._z_rel = [0]
    engine.record_raman = MagicMock()
    engine.try_snap_image = MagicMock()
    engine.get_frame_metadata = MagicMock(return_value={})
    engine.try_get_image = MagicMock(return_value=np.zeros((2, 2)))
    event = MDAEvent(index={"p": 0, "z": 0})

    payloads = list(engine.exec_event(event))

    engine.record_raman.assert_called_once_with(event, pre_acq=False)
    engine.try_snap_image.assert_not_called()
    assert payloads == []


def test_explicit_raman_channel_has_no_camera_payload():
    engine = object.__new__(RamanEngine)
    engine._rm_meta = {"channel": "RM", "z": [0]}
    engine._rm_channel = "RM"
    engine._rm_z = np.array([0])
    engine._pre_acq = False
    engine._skip_imaging_for_same_pos = False
    engine.record_raman = MagicMock()
    engine.try_snap_image = MagicMock()
    event = MDAEvent(
        index={"p": 0, "c": 0, "z": 0},
        channel={"group": "Channel", "config": "RM"},
    )

    payloads = list(engine.exec_event(event))

    engine.record_raman.assert_called_once_with(event, pre_acq=False)
    engine.try_snap_image.assert_not_called()
    assert payloads == []


def test_camera_channel_does_not_force_brightfield():
    engine = object.__new__(RamanEngine)
    engine._rm_meta = {"channel": "RM", "z": [0]}
    engine._rm_channel = "RM"
    engine._rm_z = np.array([0])
    engine._skip_imaging_for_same_pos = False
    engine.try_snap_image = MagicMock()
    engine.try_set_config = MagicMock()
    engine.get_frame_metadata = MagicMock(return_value={})
    engine.try_get_image = MagicMock(return_value=np.zeros((2, 2)))
    event = MDAEvent(
        index={"p": 0, "c": 1, "z": 0},
        channel={"group": "Channel", "config": "GFP"},
    )

    payloads = list(engine.exec_event(event))

    engine.try_snap_image.assert_called_once_with()
    engine.try_set_config.assert_not_called()
    assert len(payloads) == 1


def test_mda_no_autofocus(core: CMMCorePlus, engine: RamanEngine):
    seq = MDASequence(
        metadata={"raman": {"channel": "BF", "z": "center"}},
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
    seq = seq.replace(metadata={"raman": {"channel": "BF", "z": "all"}})
    core.mda.run(seq)
    assert rm_mock.call_count == 18

    # remove aiming source and check that we raise an error
    engine.aiming_sources = []
    with pytest.raises(RuntimeError, match="No aiming sources - cannot collect Raman."):
        core.mda.run(seq)


# TODO: test with autofocus!!

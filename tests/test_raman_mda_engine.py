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


def test_reload_waits_before_unloading_devices():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._config_file = "hardware.cfg"
    engine.try_set_config = MagicMock()

    with patch("raman_mda_engine._engine.time.sleep") as sleep:
        engine.reload(N=1)

    sleep.assert_called_once_with(5.0)
    engine._mmc.unloadAllDevices.assert_called_once_with()


def test_record_raman_combines_named_cell_layers_and_skips_empty_layers():
    class _Source:
        def __init__(self, name, points):
            self.name = name
            self.role = "cell"
            self._points = np.asarray(points, dtype=float).reshape(-1, 2)

        def get_mda_points(self, _event):
            return self._points

    engine = object.__new__(RamanEngine)
    engine._batch = False
    engine._segment_and_track = False
    engine._image_x = 100
    engine._image_y = 100
    engine._scale = 1
    engine._max_volt = 1.6
    engine._default_rm_exp = 100
    engine.aiming_sources = [
        _Source("cell:Type A", [[0.1, 0.2], [0.3, 0.4]]),
        _Source("cell:Empty", []),
        _Source("cell:Type B", [[0.5, 0.6]]),
    ]
    engine.try_set_config = MagicMock()
    engine._transformer = MagicMock()
    engine._transformer.BF_to_volts.side_effect = lambda points, **_kwargs: points
    engine._spectra_collector = MagicMock()
    engine._spectra_collector.collect_spectra_pts.return_value = np.ones(
        (6, 4)
    )
    engine._point_repeats = 2
    engine.raman_events = MagicMock()

    event = MDAEvent(
        index={"p": 0, "t": 0},
        channel={"config": "RM", "group": "Channel"},
    )
    engine.record_raman(event, pre_acq=False)

    emitted = engine.raman_events.ramanSpectraReady.emit.call_args.args
    np.testing.assert_allclose(
        emitted[2],
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.5, 0.6],
        ],
    )
    assert emitted[3] == [
        "cell:Type A",
        "cell:Type A",
        "cell:Type A",
        "cell:Type A",
        "cell:Type B",
        "cell:Type B",
    ]
    collected_points = (
        engine._spectra_collector.collect_spectra_pts.call_args.args[0]
    )
    assert collected_points.shape == (6, 2)


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
    engine._mmc.getCurrentConfig.return_value = "BF"
    engine._mmc.getExposure.return_value = 10.0
    engine._mmc.getAutoShutter.return_value = True
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
    assert engine.try_set_config.call_args_list == [
        call("Channel", "RM"),
        call("Channel", "BF"),
    ]
    assert engine._mmc.setShutterOpen.call_args_list == [
        call(True),
        call(False),
    ]
    assert engine._mmc.setAutoShutter.call_args_list == [
        call(False),
        call(True),
    ]
    assert engine.try_setExp.call_args_list == [
        call(0.1),
        call(0.1),
        call(10.0),
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
    engine._autoshutter_was_set = False
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


@pytest.mark.parametrize("cadence, segment_every, expected_times, expected_segments", [
    (0, None, [0], [0]), (1, None, [0, 1, 2, 3], [0, 1, 2, 3]),
    (2, None, [0, 2], [0, 2]), (2, 3, [0, 2], [0, 3]),
])
def test_refocus_cadence_preserves_initial_focus_and_segmentation(cadence, segment_every, expected_times, expected_segments):
    engine = MagicMock(spec=RamanEngine)
    engine._mmc = MagicMock()
    engine._autofocus = True
    engine._segment_and_track = True
    engine._refocus_every = cadence
    if segment_every is not None:
        engine._segment_every = segment_every
    engine._last_pos = None
    engine._last_best_z = {}
    engine._last_images = {}
    engine._autofocus_p = [0]
    engine._segment_channel = "BF"
    engine._last_operation_reloaded = False
    engine._channel_offset_for_event.return_value = 0
    source = MagicMock()
    source.role = "autofocus"
    source.get_mda_points.return_value = [[512, 512]]
    engine.aiming_sources = [source]
    engine.autofocus_w_raman.return_value = (None, 100, None)

    with patch("raman_mda_engine._engine.time.sleep"):
        for t in range(4):
            event = MDAEvent(index={"p": 0, "t": t}, z_pos=100,
                             channel={"group": "Channel", "config": "GFP"})
            RamanEngine.setup_event(engine, event)
            RamanEngine.setup_event(engine, event)

    assert [c.args[2] for c in engine.autofocus_w_raman.call_args_list] == expected_times
    assert [c.args[1].index["t"] for c in engine.update_aim.call_args_list] == expected_segments
    assert engine.try_set_config.call_args == call("Channel", "GFP")


def test_setup_event_holds_shutter_open_when_requested():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._mmcore_ref = lambda: engine._mmc
    engine._mmc.getAutoShutter.return_value = True
    engine._autoshutter_was_set = True
    engine._autofocus = False
    engine._segment_and_track = False
    engine._channel_z_offsets = {}

    engine.setup_event(MDAEvent(keep_shutter_open=True))

    engine._mmc.setAutoShutter.assert_called_once_with(False)
    engine._mmc.setShutterOpen.assert_called_once_with(True)
    engine._mmc.waitForSystem.assert_called_once_with()


@pytest.mark.parametrize("config_driven", [False, True])
@pytest.mark.parametrize("keep_open", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_raman_keep_open_controls_laser_and_never_opens_camera(config_driven, keep_open, fails):
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._autofocus = engine._segment_and_track = False
    engine._channel_z_offsets = {}
    engine._rm_meta = {"channel": "RM"}
    engine._rm_channel = "RM"
    engine._rm_z = [0]
    engine._pre_acq = False
    states = {"camera": True, "laser": False}
    engine._mmc.getConfigData.side_effect = lambda group, config: (
        [("Core", "Shutter", "camera")] if config == "BF" else
        ([] if config_driven else [("Core", "Shutter", "laser")])
    )
    engine._mmc.getShutterDevice.return_value = "camera"
    engine._mmc.setShutterOpen.side_effect = lambda device, opened: states.update({device: opened})
    engine._mmc.setConfig.side_effect = lambda group, config: states.update(laser=config == "RM")
    engine.try_set_config = engine._mmc.setConfig
    event = MDAEvent(channel={"group": "Channel", "config": "RM"}, keep_shutter_open=keep_open)

    def base_setup(prepared):
        assert not prepared.keep_shutter_open
        engine._mmc.setConfig("Channel", "RM")

    with patch("pymmcore_plus.mda.MDAEngine.setup_event", side_effect=base_setup):
        engine.setup_event(event)
    assert states == {"camera": False, "laser": True}

    def collect(*args, **kwargs):
        assert states == {"camera": False, "laser": True}
        if fails:
            raise RuntimeError("detector failed")

    engine.record_raman = MagicMock(side_effect=collect)
    if fails:
        with pytest.raises(RuntimeError, match="detector failed"):
            list(engine.exec_event(event))
    else:
        assert list(engine.exec_event(event)) == []
    assert states == {"camera": False, "laser": keep_open and not fails}
    assert call("camera", True) not in engine._mmc.setShutterOpen.call_args_list

    engine._autoshutter_was_set = False
    with patch("pymmcore_plus.mda.MDAEngine.teardown_sequence"):
        engine.teardown_sequence(MDASequence())
    assert states == {"camera": False, "laser": False}


def test_teardown_sequence_closes_shutter_and_restores_autoshutter():
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._autoshutter_was_set = True

    with patch("pymmcore_plus.mda.MDAEngine.teardown_sequence") as teardown:
        engine.teardown_sequence(MDASequence())

    teardown.assert_called_once()
    engine._mmc.setShutterOpen.assert_called_once_with(False)
    engine._mmc.setAutoShutter.assert_called_once_with(True)
    engine._mmc.setConfig.assert_called_once_with("Channel", "BF")
    assert engine._mmc.waitForSystem.call_count == 2


@pytest.mark.parametrize("axis_order", ["tpcz", "cpzt"])
def test_brightfield_restored_only_after_every_mda_event(axis_order):
    from pymmcore_plus.mda import MDARunner

    sequence = MDASequence(
        axis_order=axis_order,
        channels=("GFP", "RM"),
        stage_positions=((0, 0), (1, 1)),
        z_plan={"relative": (0, 1)},
        time_plan={"interval": 0, "loops": 2},
    )
    engine = object.__new__(RamanEngine)
    engine._mmc = MagicMock()
    engine._autoshutter_was_set = False
    engine.setup_sequence = MagicMock(return_value={})
    engine.event_iterator = lambda events: iter(events)
    engine.setup_event = MagicMock()
    engine.teardown_event = MagicMock()
    acquired = []

    def acquire(event):
        engine._mmc.setConfig.assert_not_called()
        acquired.append(dict(event.index))
        return ()

    engine.exec_event = acquire
    with patch("pymmcore_plus.mda.MDAEngine.teardown_sequence") as teardown:
        # Base cleanup may restore the original channel; BF must be set later.
        teardown.side_effect = lambda _seq: engine._mmc.setConfig.assert_not_called()
        runner = MDARunner()
        runner.set_engine(engine)
        runner.run(sequence)

    assert acquired == [dict(event.index) for event in sequence.iter_events()]
    assert len(acquired) == 16
    teardown.assert_called_once_with(sequence)
    engine._mmc.setConfig.assert_called_once_with("Channel", "BF")
    engine._mmc.waitForSystem.assert_called_once_with()


def test_cpzt_can_hold_shutter_across_positions_z_and_time():
    sequence = MDASequence(
        axis_order="cpzt",
        channels=("BF", "GFP"),
        stage_positions=((0, 0), (1, 1)),
        z_plan={"relative": (0, 1)},
        time_plan={"interval": 0, "loops": 2},
        keep_shutter_open_across=("p", "z", "t"),
    )
    events = tuple(sequence.iter_events())

    position_transition = next(
        event
        for event in events
        if event.index == {"c": 0, "p": 0, "z": 1, "t": 1}
    )
    channel_transition = next(
        event
        for event in events
        if event.index == {"c": 0, "p": 1, "z": 1, "t": 1}
    )

    assert position_transition.keep_shutter_open
    assert not channel_transition.keep_shutter_open


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
    engine._autoshutter_was_set = False
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
    engine._mmc = MagicMock()
    engine._rm_meta = {"channel": "RM", "z": [0]}
    engine._rm_channel = "RM"
    engine._rm_z = np.array([0])
    engine._pre_acq = False
    engine._skip_imaging_for_same_pos = False
    engine._autoshutter_was_set = False
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
    engine._autoshutter_was_set = True
    engine._mmc = MagicMock()
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
    engine._mmc.setShutterOpen.assert_called_once_with(False)
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

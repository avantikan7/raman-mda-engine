from types import SimpleNamespace

import h5py
import numpy as np
import tifffile
from pymmcore_plus.mda.events import MDASignaler
from useq import MDAEvent, MDASequence

from raman_mda_engine._writers import (
    RamanAcquisitionWriter,
    _image_source_positions,
    _RamanH5Store,
)


class _FakeMDA:
    def __init__(self):
        self.events = MDASignaler()
        self.engine = object()

    @staticmethod
    def status():
        return SimpleNamespace(finish_reason="completed")


class _FakeCore:
    def __init__(self):
        self.mda = _FakeMDA()

    @staticmethod
    def getPixelSizeUm():
        return 0.25

    @staticmethod
    def getXYPosition():
        return (10.5, 20.5)

    @staticmethod
    def getZPosition():
        return 30.5


def test_image_source_positions_reuses_previous_physical_fov():
    np.testing.assert_array_equal(
        _image_source_positions(7, [0, 3, 6]),
        [0, 0, 0, 3, 3, 3, 6],
    )


def test_raman_h5_store_appends_variable_event_payloads(tmp_path):
    path = tmp_path / "raman.h5"
    store = _RamanH5Store(
        path,
        sequence_json="{}",
        wavenumbers=np.array([100.0, 101.0, 102.0]),
        image_source_position=np.array([0, 0, 2]),
        batch=True,
    )
    store.append(
        MDAEvent(index={"t": 1, "p": 2, "z": 3}),
        np.array([1, 2, 3], dtype=np.uint16),
        np.array([[0.25, 0.75], [0.5, 0.5]]),
        ["cell", "background"],
        125.0,
        (1.0, 2.0, 3.0),
    )
    store.append(
        MDAEvent(index={"t": 2, "p": 2, "z": 0}),
        np.array([[4, 5, 6], [7, 8, 9]], dtype=np.uint16),
        np.array([[0.1, 0.2]]),
        ["cell"],
        250.0,
        (4.0, 5.0, 6.0),
    )
    store.close("completed")

    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(
            handle["raman/spectra"][:],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        np.testing.assert_allclose(
            handle["raman/points_yx"][:],
            [[0.25, 0.75], [0.5, 0.5], [0.1, 0.2]],
        )
        assert handle["raman/designations"].asstr()[:].tolist() == [
            "cell",
            "background",
            "cell",
        ]
        events = handle["raman/events"][:]
        assert events[0]["spectra_count"] == 1
        assert events[0]["points_count"] == 2
        assert events[1]["spectra_start"] == 1
        assert events[1]["points_start"] == 2
        np.testing.assert_array_equal(
            handle["positions/image_source_position"][:], [0, 0, 2]
        )
        assert handle.attrs["completion_status"] == "completed"


def test_writer_streams_one_ome_tiff_per_imaged_position(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "run",
        core=core,
        wavenumbers=np.array([100.0, 101.0, 102.0]),
        image_positions=[0, 2],
    )
    sequence = MDASequence(
        axis_order="tpcz",
        time_plan={"interval": 0, "loops": 2},
        stage_positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        channels=["BF", "RM", "GFP"],
        z_plan={"relative": [-1, 1]},
        metadata={"raman": {"channel": "RM", "z": [0]}},
    )

    core.mda.events.sequenceStarted.emit(sequence, {})
    written_values = []
    for event in sequence.iter_events():
        if (
            event.channel is not None
            and event.channel.config != "RM"
            and event.index["p"] in {0, 2}
        ):
            value = len(written_values) + 1
            written_values.append(value)
            core.mda.events.frameReady.emit(
                np.full((4, 5), value, dtype=np.uint16),
                event,
                {
                    "runner_time_ms": value * 10,
                    "exposure_ms": 5,
                    "position": {"x": value, "y": 2, "z": 3},
                },
            )
    writer._save_raman(
        MDAEvent(index={"t": 0, "p": 1, "z": 0}),
        np.array([[1, 2, 3]], dtype=np.uint16),
        np.array([[0.5, 0.5]]),
        ["cell"],
        100,
    )
    core.mda.events.sequenceFinished.emit(sequence)

    assert writer.closed
    assert writer.status == "completed"
    assert writer.path is not None
    image_files = sorted((writer.path / "images").glob("*.ome.tiff"))
    assert len(image_files) == 2
    assert not list(writer.path.glob("t*_p*_c*_z*.tiff"))
    assert not (writer.path / "raman").exists()

    for path in image_files:
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 8
            assert tif.is_ome
            assert tif.is_bigtiff

    with h5py.File(writer.path / "raman.h5", "r") as handle:
        assert len(handle["raman/events"]) == 1
        np.testing.assert_array_equal(
            handle["positions/image_source_position"][:],
            [0, 0, 2],
        )


def test_canceled_partial_run_is_closed_and_marked_canceled(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "run",
        core=core,
        wavenumbers=np.array([100.0, 101.0, 102.0]),
        image_positions=[0],
    )
    sequence = MDASequence(
        time_plan={"interval": 0, "loops": 2},
        channels=["BF"],
    )
    core.mda.events.sequenceStarted.emit(sequence, {})
    first_event = next(sequence.iter_events())
    core.mda.events.frameReady.emit(
        np.ones((4, 5), dtype=np.uint16), first_event, {}
    )
    core.mda.events.sequenceCanceled.emit(sequence)
    core.mda.events.sequenceFinished.emit(sequence)

    assert writer.closed
    assert writer.status == "canceled"
    with h5py.File(writer.path / "raman.h5", "r") as handle:
        assert handle.attrs["completion_status"] == "canceled"
    image_file = next((writer.path / "images").glob("*.ome.tiff"))
    with tifffile.TiffFile(image_file) as tif:
        assert len(tif.pages) == 1


def test_failed_raman_append_keeps_prior_committed_event(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "run",
        core=core,
        wavenumbers=np.array([100.0, 101.0, 102.0]),
        image_positions=[],
    )
    sequence = MDASequence(
        channels=["RM"],
        metadata={"raman": {"channel": "RM", "z": [0]}},
    )
    core.mda.events.sequenceStarted.emit(sequence, {})
    writer._save_raman(
        MDAEvent(index={"t": 0, "p": 0, "z": 0}),
        np.array([1, 2, 3], dtype=np.uint16),
        np.array([[0.5, 0.5]]),
        ["cell"],
        100,
    )
    with np.testing.assert_raises(ValueError):
        writer._save_raman(
            MDAEvent(index={"t": 0, "p": 0, "z": 0}),
            np.array([1, 2], dtype=np.uint16),
            np.array([[0.5, 0.5]]),
            ["cell"],
            100,
        )
    writer.close(status="failed")

    with h5py.File(writer.path / "raman.h5", "r") as handle:
        assert handle.attrs["completion_status"] == "failed"
        assert len(handle["raman/events"]) == 1
        np.testing.assert_array_equal(handle["raman/spectra"][:], [[1, 2, 3]])


def test_out_of_order_image_frame_marks_run_failed(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "run", core=core, image_positions=[0]
    )
    sequence = MDASequence(channels=["BF", "GFP"])
    core.mda.events.sequenceStarted.emit(sequence, {})
    events = tuple(sequence.iter_events())

    with np.testing.assert_raises(RuntimeError):
        writer._on_mda_frame(
            np.ones((2, 3), dtype=np.uint16), events[1], {}
        )
    writer.close(status="failed")

    assert writer.status == "failed"
    assert not list((writer.path / "images").glob("*.ome.tiff"))
    with h5py.File(writer.path / "raman.h5", "r") as handle:
        assert handle.attrs["completion_status"] == "failed"


def test_closed_writer_is_disconnected_before_next_run(tmp_path):
    core = _FakeCore()
    first = RamanAcquisitionWriter(
        tmp_path / "run", core=core, image_positions=[0]
    )
    first.disconnect()
    second = RamanAcquisitionWriter(
        tmp_path / "run", core=core, image_positions=[0]
    )
    sequence = MDASequence(channels=["BF"])
    core.mda.events.sequenceStarted.emit(sequence, {})
    event = next(sequence.iter_events())
    core.mda.events.frameReady.emit(
        np.ones((2, 3), dtype=np.uint16), event, {}
    )
    core.mda.events.sequenceFinished.emit(sequence)

    assert first.path is None
    assert first.status == "incomplete"
    assert second.path is not None
    assert second.status == "completed"

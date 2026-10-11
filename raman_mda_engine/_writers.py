from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from ome_writers import (
    AcquisitionSettings,
    OmeTiffFormat,
    create_stream,
    useq_to_acquisition_settings,
)
from useq import MDAEvent

from ._engine import RamanEngine

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ome_writers import OMEStream
    from pymmcore_plus import CMMCorePlus
    from useq import MDASequence

__all__ = [
    "RamanAcquisitionWriter",
    "RamanTiffAndNumpyWriter",
]

_RAMAN_SCHEMA_VERSION = 1
_EVENT_DTYPE = np.dtype(
    [
        ("t", "<i8"),
        ("p", "<i8"),
        ("z", "<i8"),
        ("timestamp_ns", "<i8"),
        ("exposure_ms", "<f8"),
        ("stage_xyz_um", "<f8", (3,)),
        ("spectra_start", "<i8"),
        ("spectra_count", "<i8"),
        ("points_start", "<i8"),
        ("points_count", "<i8"),
    ]
)


def _json_sequence(sequence: MDASequence) -> str:
    if hasattr(sequence, "model_dump_json"):
        return sequence.model_dump_json(exclude_unset=True)
    return sequence.json()


def _unique_folder(folder_base_name: str | Path) -> Path:
    folder = Path(folder_base_name).expanduser().resolve()
    stem = folder.stem
    index = 1
    while True:
        path = folder.parent / f"{stem}_{index}"
        if not path.exists():
            path.mkdir(parents=True)
            return path
        index += 1


def _overwrite_folder(folder_name: str | Path) -> Path:
    """Replace an existing acquisition folder and return its resolved path."""
    folder = Path(folder_name).expanduser().resolve()
    if folder.parent == folder:
        raise ValueError("Refusing to overwrite a filesystem root")
    if folder.is_symlink():
        raise ValueError("Refusing to overwrite a symbolic link")
    if folder.exists():
        if not folder.is_dir():
            raise NotADirectoryError(
                f"Writer output path is not a directory: {folder}"
            )
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    return folder


def _as_spectra(data: np.ndarray) -> np.ndarray:
    spectra = np.asarray(data)
    spectra = np.squeeze(spectra)
    if spectra.ndim == 1:
        return spectra[np.newaxis, :]
    if spectra.ndim < 1:
        raise ValueError("Raman spectra must contain a spectral axis")
    return spectra.reshape(-1, spectra.shape[-1])


def _as_points(data: np.ndarray) -> np.ndarray:
    points = np.asarray(data, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return points.reshape(-1, 2)


def _image_source_positions(
    logical_position_count: int, image_positions: Iterable[int]
) -> np.ndarray:
    image_positions = np.unique(np.asarray(tuple(image_positions), dtype=np.int64))
    if logical_position_count < 1:
        return np.empty(0, dtype=np.int64)
    if len(image_positions) == 0:
        return np.full(logical_position_count, -1, dtype=np.int64)

    logical = np.arange(logical_position_count, dtype=np.int64)
    source_indices = np.searchsorted(image_positions, logical, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(image_positions) - 1)
    return image_positions[source_indices]


class _RamanH5Store:
    """Append-only Raman event store.

    Payload arrays are extended before the corresponding event record. Readers
    treat the event table as the commit log, so an interrupted write cannot
    expose an incomplete payload as a valid Raman event.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sequence_json: str,
        wavenumbers: np.ndarray | None,
        image_source_position: np.ndarray,
        batch: bool,
    ) -> None:
        self.path = Path(path)
        self._file = h5py.File(self.path, "w", libver="latest")
        self._spectra_dtype: np.dtype | None = None
        self._spectral_length: int | None = None

        self._file.attrs.update(
            {
                "schema_version": _RAMAN_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "completion_status": "acquiring",
                "batch": bool(batch),
                "useq_sequence_json": sequence_json,
            }
        )
        axes = self._file.create_group("axes")
        if wavenumbers is not None:
            axis = np.asarray(wavenumbers, dtype=np.float64)
            if axis.ndim != 1:
                raise ValueError("wavenumbers must be a one-dimensional array")
            axes.create_dataset("wavenumber_cm_inverse", data=axis)
            self._spectral_length = len(axis)

        positions = self._file.create_group("positions")
        positions.create_dataset(
            "image_source_position",
            data=np.asarray(image_source_position, dtype=np.int64),
        )

        raman = self._file.create_group("raman")
        raman.create_dataset(
            "points_yx",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(256, 2),
            dtype=np.float64,
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        raman.create_dataset(
            "designations",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=string_dtype,
        )
        raman.create_dataset(
            "events",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=_EVENT_DTYPE,
        )
        self._file.flush()

    @property
    def closed(self) -> bool:
        return not bool(self._file)

    def _ensure_spectra_dataset(self, spectra: np.ndarray) -> h5py.Dataset:
        raman = self._file["raman"]
        spectral_length = int(spectra.shape[-1])
        if (
            self._spectral_length is not None
            and spectral_length != self._spectral_length
        ):
            raise ValueError(
                "Wavenumber calibration length does not match Raman spectra "
                f"({self._spectral_length} != {spectral_length})"
            )

        if "spectra" not in raman:
            self._spectral_length = spectral_length
            self._spectra_dtype = spectra.dtype
            target_bytes = 1024**2
            rows_per_chunk = max(
                1,
                min(
                    256,
                    target_bytes
                    // max(1, spectral_length * spectra.dtype.itemsize),
                ),
            )
            if "wavenumber_cm_inverse" not in self._file["axes"]:
                self._file["axes"].create_dataset(
                    "wavenumber_cm_inverse",
                    data=np.arange(spectral_length, dtype=np.float64),
                )
                self._file["axes"]["wavenumber_cm_inverse"].attrs["calibrated"] = False
            return raman.create_dataset(
                "spectra",
                shape=(0, spectral_length),
                maxshape=(None, spectral_length),
                chunks=(rows_per_chunk, spectral_length),
                dtype=spectra.dtype,
            )

        dataset = raman["spectra"]
        if spectral_length != dataset.shape[1]:
            raise ValueError(
                "Raman spectrum length changed during acquisition "
                f"({dataset.shape[1]} != {spectral_length})"
            )
        if spectra.dtype != dataset.dtype:
            raise ValueError(
                "Raman spectrum dtype changed during acquisition "
                f"({dataset.dtype} != {spectra.dtype})"
            )
        return dataset

    def append(
        self,
        event: MDAEvent,
        spectra: np.ndarray,
        points: np.ndarray,
        designations: Iterable[str],
        exposure_ms: float,
        stage_xyz_um: tuple[float, float, float],
    ) -> None:
        if self.closed:
            raise RuntimeError("Cannot append to a closed Raman store")

        spectra = _as_spectra(spectra)
        points = _as_points(points)
        labels = np.asarray(tuple(str(value) for value in designations), dtype=object)
        if len(labels) != len(points):
            raise ValueError(
                "Raman designation count does not match point count "
                f"({len(labels)} != {len(points)})"
            )

        spectra_ds = self._ensure_spectra_dataset(spectra)
        points_ds = self._file["raman/points_yx"]
        labels_ds = self._file["raman/designations"]
        events_ds = self._file["raman/events"]

        spectra_start = len(spectra_ds)
        points_start = len(points_ds)

        spectra_ds.resize(spectra_start + len(spectra), axis=0)
        spectra_ds[spectra_start:] = spectra
        points_ds.resize(points_start + len(points), axis=0)
        points_ds[points_start:] = points
        labels_ds.resize(points_start + len(labels), axis=0)
        labels_ds[points_start:] = labels

        record = np.zeros(1, dtype=_EVENT_DTYPE)
        record["t"] = event.index.get("t", 0)
        record["p"] = event.index.get("p", 0)
        record["z"] = event.index.get("z", 0)
        record["timestamp_ns"] = int(datetime.now(timezone.utc).timestamp() * 1e9)
        record["exposure_ms"] = float(exposure_ms)
        record["stage_xyz_um"] = stage_xyz_um
        record["spectra_start"] = spectra_start
        record["spectra_count"] = len(spectra)
        record["points_start"] = points_start
        record["points_count"] = len(points)

        event_index = len(events_ds)
        events_ds.resize(event_index + 1, axis=0)
        events_ds[event_index] = record[0]
        self._file.flush()

    def close(self, status: str) -> None:
        if self.closed:
            return
        self._file.attrs["completion_status"] = str(status)
        self._file.attrs["closed_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.flush()
        self._file.close()


class RamanAcquisitionWriter:
    """Stream camera frames to OME-TIFF and Raman events to one HDF5 file."""

    def __init__(
        self,
        save_dir: str | Path,
        core: CMMCorePlus | None = None,
        wavenumbers: np.ndarray | None = None,
        *,
        image_positions: Iterable[int] | None = None,
        batch: bool = False,
        overwrite: bool = False,
        save_tracking_images: bool = True,
        exact_path: bool = False,
    ) -> None:
        from pymmcore_plus import CMMCorePlus

        self._core = core or CMMCorePlus.instance()
        self._save_dir = Path(save_dir)
        self._wavenumbers = (
            None
            if wavenumbers is None
            else np.asarray(wavenumbers, dtype=np.float64)
        )
        self._requested_image_positions = (
            None
            if image_positions is None
            else tuple(int(value) for value in image_positions)
        )
        self._batch = bool(batch)
        self._overwrite = bool(overwrite)
        self._save_tracking_images = bool(save_tracking_images)
        self._exact_path = bool(exact_path)
        self._path: Path | None = None
        self._sequence: MDASequence | None = None
        self._image_sequence: MDASequence | None = None
        self._image_stream: OMEStream | None = None
        self._raman_store: _RamanH5Store | None = None
        self._expected_image_events: Any = None
        self._image_positions: tuple[int, ...] = ()
        self._rm_channel = "RM"
        self._status = "idle"
        self._canceled = False
        self._closed = False
        self._connected_engine: RamanEngine | None = None

        events = self._core.mda.events
        events.sequenceStarted.connect(self._on_mda_started)
        events.frameReady.connect(self._on_mda_frame)
        events.sequenceCanceled.connect(self._on_mda_canceled)
        events.sequenceFinished.connect(self._on_mda_finished)
        self._connect_raman_engine(self._core.mda.engine)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def status(self) -> str:
        return self._status

    @property
    def completion_status(self) -> str:
        return self._status

    @property
    def closed(self) -> bool:
        return self._closed

    def _connect_raman_engine(self, engine: object) -> None:
        if not isinstance(engine, RamanEngine):
            return
        engine.raman_events.ramanSpectraReady.connect(self._save_raman)
        engine.raman_events.aimUpdated.connect(self._aim_updated)
        self._connected_engine = engine

    def _disconnect_signals(self) -> None:
        events = self._core.mda.events
        for signal, callback in (
            (events.sequenceStarted, self._on_mda_started),
            (events.frameReady, self._on_mda_frame),
            (events.sequenceCanceled, self._on_mda_canceled),
            (events.sequenceFinished, self._on_mda_finished),
        ):
            try:
                signal.disconnect(callback)
            except Exception:
                pass
        if self._connected_engine is not None:
            for signal, callback in (
                (
                    self._connected_engine.raman_events.ramanSpectraReady,
                    self._save_raman,
                ),
                (self._connected_engine.raman_events.aimUpdated, self._aim_updated),
            ):
                try:
                    signal.disconnect(callback)
                except Exception:
                    pass
        self._connected_engine = None

    def _on_mda_started(
        self, sequence: MDASequence, _summary_metadata: dict | None = None
    ) -> None:
        if self._overwrite:
            self._path = _overwrite_folder(self._save_dir)
        elif self._exact_path:
            # The unified editor has already chosen and reviewed this destination.
            self._path = self._save_dir.expanduser().resolve()
            self._path.mkdir(parents=True, exist_ok=False)
        else:
            self._path = _unique_folder(self._save_dir)
        self._sequence = sequence
        self._status = "acquiring"
        self._closed = False
        self._canceled = False

        sequence_json = _json_sequence(sequence)
        (self._path / "useq-sequence.json").write_text(
            sequence_json, encoding="utf-8"
        )
        (self._path / "images").mkdir()
        (self._path / "aiming").mkdir()

        raman_meta = sequence.metadata.get("raman") or {}
        self._rm_channel = str(raman_meta.get("channel", "RM"))
        from .grid_plan import grid_count
        position_count = max(1, grid_count(sequence))
        requested = self._requested_image_positions
        if requested is None:
            requested = tuple(range(position_count))
        self._image_positions = tuple(
            sorted({value for value in requested if 0 <= value < position_count})
        )
        source_positions = _image_source_positions(
            position_count, self._image_positions
        )
        self._raman_store = _RamanH5Store(
            self._path / "raman.h5",
            sequence_json=sequence_json,
            wavenumbers=self._wavenumbers,
            image_source_position=source_positions,
            batch=self._batch,
        )

        self._image_sequence = self._make_image_sequence(sequence)
        self._expected_image_events = iter(self._iter_image_events(sequence))

    def _make_image_sequence(self, sequence: MDASequence) -> MDASequence | None:
        camera_channels = tuple(
            channel
            for channel in sequence.channels
            if channel.config != self._rm_channel
        )
        if not camera_channels or not self._image_positions:
            return None

        updates: dict[str, Any] = {"channels": camera_channels}
        from .grid_plan import GRID_KEY, grid_position
        if GRID_KEY in sequence.metadata:
            updates["stage_positions"] = [
                grid_position(sequence, index).replace(name=f"p{index:03d}")
                for index in self._image_positions
            ]
            order = list(sequence.axis_order)
            if "p" not in order:
                order.insert(order.index("t") + 1 if "t" in order else 0, "p")
            updates["axis_order"] = order
        if sequence.stage_positions:
            positions = []
            for original_index in self._image_positions:
                position = sequence.stage_positions[original_index]
                positions.append(position.replace(name=f"p{original_index:03d}"))
            updates["stage_positions"] = positions

        metadata = dict(sequence.metadata)
        metadata.pop("raman", None)
        metadata.pop(GRID_KEY, None)
        metadata["raman_storage"] = {
            "original_image_positions": list(self._image_positions)
        }
        updates["metadata"] = metadata
        return sequence.replace(**updates)

    def _iter_image_events(self, sequence: MDASequence):
        selected = set(self._image_positions)
        if not selected or not any(channel.config != self._rm_channel for channel in sequence.channels):
            return
        from .grid_plan import GRID_KEY, iter_grid_events
        has_positions = bool(sequence.stage_positions) or GRID_KEY in sequence.metadata
        # Validation must not advance the executed plan's autofocus state.
        sequence = type(sequence).model_validate_json(sequence.model_dump_json())
        for event in iter_grid_events(sequence):
            from useq import AcquireImage
            if not isinstance(getattr(event, "action", None), (AcquireImage, type(None))):
                continue
            if event.channel is None or event.channel.config == self._rm_channel:
                continue
            if has_positions and event.index.get("p", 0) not in selected:
                continue
            yield event

    @staticmethod
    def _event_key(event: MDAEvent) -> tuple[int, int, int, int]:
        return (
            event.index.get("t", 0),
            event.index.get("p", 0),
            event.index.get("c", 0),
            event.index.get("z", 0),
        )

    def _ensure_image_stream(self, image: np.ndarray) -> OMEStream:
        if self._image_stream is not None:
            return self._image_stream
        if self._image_sequence is None or self._path is None:
            raise RuntimeError("Received a camera frame for an RM-only acquisition")

        frame = np.asarray(image)
        if frame.ndim != 2:
            raise ValueError(f"Camera frames must be 2D, got shape {frame.shape}")
        image_height, image_width = frame.shape
        useq_settings = useq_to_acquisition_settings(
            self._image_sequence,
            image_width=image_width,
            image_height=image_height,
            pixel_size_um=self._pixel_size_um(),
        )
        settings = AcquisitionSettings(
            root_path=str(self._path / "images.ome.tiff"),
            dtype=frame.dtype,
            format=OmeTiffFormat(
                prefer_single_file="never",
                multi_file_metadata="redundant",
            ),
            compression="none",
            storage_order="acquisition",
            **useq_settings,
        )
        self._image_stream = create_stream(settings)
        self._image_stream.set_global_metadata(
            "org.kkobayashi-lab.raman",
            {
                "useq_sequence": json.loads(_json_sequence(self._sequence)),
                "original_image_positions": list(self._image_positions),
            },
        )
        return self._image_stream

    def _pixel_size_um(self) -> float | None:
        try:
            value = float(self._core.getPixelSizeUm())
        except Exception:
            return None
        return value if value > 0 else None

    @staticmethod
    def _frame_metadata(event: MDAEvent, metadata: dict | None) -> dict:
        metadata = metadata or {}
        result: dict[str, float] = {}
        if "runner_time_ms" in metadata:
            result["delta_t"] = float(metadata["runner_time_ms"]) / 1000
        exposure = metadata.get("exposure_ms", event.exposure)
        if exposure is not None:
            result["exposure_time"] = float(exposure) / 1000
        position = metadata.get("position") or {}
        for axis in "xyz":
            if axis in position:
                result[f"position_{axis}"] = float(position[axis])
        return result

    def _on_mda_frame(
        self,
        image: np.ndarray,
        event: MDAEvent,
        metadata: dict | None = None,
    ) -> None:
        if self._closed:
            return
        try:
            expected = next(self._expected_image_events)
        except StopIteration as exc:
            self._status = "failed"
            raise RuntimeError("Received more camera frames than expected") from exc
        if self._event_key(expected) != self._event_key(event):
            self._status = "failed"
            raise RuntimeError(
                "Camera frame order does not match the filtered OME sequence: "
                f"expected {self._event_key(expected)}, "
                f"received {self._event_key(event)}"
            )
        self._ensure_image_stream(image).append(
            np.asarray(image),
            frame_metadata=self._frame_metadata(event, metadata),
        )

    def _aim_updated(
        self,
        event: MDAEvent,
        image: np.ndarray,
        mask: np.ndarray,
        new_points: np.ndarray,
        previous_points: np.ndarray,
    ) -> None:
        if self._closed or self._path is None:
            return
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        z = event.index.get("z", 0)
        save_name = (
            self._path
            / "aiming"
            / f"segment_p{p:03d}_t{t:03d}_z{z:03d}.npz"
        )
        payload = dict(mask=mask, new_pts=new_points, prev_aim=previous_points)
        if self._save_tracking_images:
            payload["img"] = image
        np.savez_compressed(save_name, **payload)

    def _stage_xyz_um(self) -> tuple[float, float, float]:
        try:
            x, y = self._core.getXYPosition()
            z = self._core.getZPosition()
            return float(x), float(y), float(z)
        except Exception:
            return (float("nan"), float("nan"), float("nan"))

    def _save_raman(
        self,
        event: MDAEvent,
        spectra: np.ndarray,
        points: np.ndarray,
        designations: list[str],
        exposure_ms: float,
    ) -> None:
        if self._closed or self._raman_store is None:
            return
        try:
            self._raman_store.append(
                event,
                spectra,
                points,
                designations,
                exposure_ms,
                self._stage_xyz_um(),
            )
        except Exception:
            self._status = "failed"
            raise

    def _on_mda_canceled(self, _sequence: MDASequence) -> None:
        self._canceled = True

    def _on_mda_finished(self, _sequence: MDASequence) -> None:
        if self._status == "failed":
            status = "failed"
        else:
            status = "canceled" if self._canceled else self._finish_reason()
        if status == "completed" and self._expected_image_events is not None:
            try:
                next(self._expected_image_events)
            except StopIteration:
                pass
            else:
                status = "failed"
        self.close(status=status)

    def _finish_reason(self) -> str:
        try:
            status = self._core.mda.status()
            reason = str(status.finish_reason).lower()
        except Exception:
            return "completed"
        if "error" in reason:
            return "failed"
        if "cancel" in reason:
            return "canceled"
        return "completed"

    def close(self, status: str | None = None) -> None:
        final_status = status or (
            "canceled" if self._canceled else self._status
        )
        if final_status in {"idle", "acquiring"}:
            final_status = "incomplete"

        if not self._closed:
            self._closed = True
            try:
                if self._image_stream is not None:
                    try:
                        self._image_stream.close()
                    except Exception:
                        final_status = "failed"
                if self._raman_store is not None:
                    self._raman_store.close(final_status)
            finally:
                self._status = final_status
                self._disconnect_signals()

    def disconnect(self) -> None:
        self.close(status="incomplete")


# Compatibility name retained for downstream imports.
RamanTiffAndNumpyWriter = RamanAcquisitionWriter

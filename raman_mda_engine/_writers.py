from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json

import numpy as np
import pandas as pd
from pymmcore_mda_writers import SimpleMultiFileTiffWriter
from useq import MDAEvent

from ._engine import RamanEngine

if TYPE_CHECKING:
    pass

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import PMDAEngine
    from useq import MDASequence

__all__ = [
    "RamanTiffAndNumpyWriter",
]


class RamanTiffAndNumpyWriter(SimpleMultiFileTiffWriter):
    """Writer to save both images and Raman Spectra."""

    def __init__(
        self,
        save_dir: str | Path,
        core: CMMCorePlus = None,
        wavenumbers: np.ndarray | None = None,
    ):
        super().__init__(save_dir, core)
        self._wavenumbers = (
            None if wavenumbers is None
            else np.asarray(wavenumbers, dtype=float)
        )
        if isinstance(self._core.mda.engine, RamanEngine):
            self._core.mda.engine.raman_events.ramanSpectraReady.connect(
                self._save_raman
            )
            self._core.mda.engine.raman_events.aimUpdated.connect(
                self._aim_updated
            )

    def _on_mda_engine_registered(self, newEngine: PMDAEngine, oldEngine: PMDAEngine):
        # super()._on_mda_engine_registered(newEngine, oldEngine)
        if isinstance(oldEngine, RamanEngine):
            oldEngine.raman_events.ramanSpectraReady.disconnect(self._save_raman)
            oldEngine.raman_events.aimUpdated.connect(self._aim_updated)
        if isinstance(newEngine, RamanEngine):
            newEngine.raman_events.ramanSpectraReady.connect(self._save_raman)
            newEngine.raman_events.aimUpdated.connect(self._aim_updated)

    def _aim_updated(
        self, event: MDAEvent, img: np.ndarray, mask: np.ndarray, new_pts: np.ndarray, prev_pts: np.ndarray
    ):
        P, T, Z = event.index["p"], event.index.get("t", 0), event.index["z"]
        savename = self._aiming_path/ f"segment_p{str(P).zfill(3)}_t{str(T).zfill(3)}_z{str(Z).zfill(3)}.npz"
        np.savez_compressed(savename, img=img, mask=mask, new_pts=new_pts, prev_aim = prev_pts)

    def _save_raman(
        self, event: MDAEvent, spectra: np.ndarray, points: np.ndarray, which: list[str], rm_exposure: float
    ):
        from datetime import datetime
        # TODO zarrify this
        pos, t, Z = event.index["p"], event.index.get("t", 0), event.index["z"]
        save_name_base = (
            self._raman_path / f"raman_p{str(pos).zfill(3)}_t{str(t).zfill(3)}_z{str(Z).zfill(3)}"
        )
        np.save(str(save_name_base) + "_data.npy", spectra)
        np.save(str(save_name_base) + "_locations.npy", points)
        np.save(str(save_name_base) + "_designation.npy", which)

        try:
            xy=self._core.getXYPosition()
            z = self._core.getZPosition()
        except RuntimeError:
            xy = [-1, -1]
            z = 0
        meta = {
            'x':xy[0],
            'y':xy[1],
            'z':z,
            # "z":self._core.getPosition("FocusDrive"),
            "rm_exp":rm_exposure,
            "time":datetime.now()
        }

        with open(str(save_name_base) + "_meta.json", 'w') as meta_file:
            json.dump(meta, meta_file,indent=4, sort_keys=True, default=str)


    def _onMDAStarted(self, sequence: MDASequence):
        super()._onMDAStarted(sequence)
        self._raman_path = self._path / "raman"
        self._aiming_path = self._path / "aiming"
        self._raman_path.mkdir()
        self._aiming_path.mkdir()
        if self._wavenumbers is not None:
            np.save(self._raman_path / "wavenumbers.npy", self._wavenumbers)

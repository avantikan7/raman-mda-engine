"""Lazy, version-compatible Cellpose model loading."""

from __future__ import annotations

import os
from pathlib import Path


def _model_path(model) -> Path | None:
    """Return a validated custom-model path, or ``None`` for a model name."""
    text = os.fspath(model).strip()
    candidate = Path(text).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if (
        isinstance(model, os.PathLike)
        or candidate.is_absolute()
        or "/" in text
        or "\\" in text
    ):
        raise FileNotFoundError(f"Cellpose model file not found: {candidate}")
    return None


def new_cellpose_model(model="cyto2"):
    """Load a built-in, registered, or custom Cellpose model.

    Cellpose 3 uses ``model_type`` for named models, while Cellpose 4 uses
    ``pretrained_model``. Both versions use ``pretrained_model`` for a direct
    path to user-trained weights.
    """
    custom_path = _model_path(model)

    try:
        from cellpose import models
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cellpose/PyTorch could not be loaded. Install the "
            "'raman-mda-engine[tracking]' extra in a binary-consistent "
            "environment. On Windows, do not mix Conda MKL/OpenMP runtimes "
            "with a pip PyTorch wheel."
        ) from exc

    try:
        cellpose_model = models.CellposeModel
    except AttributeError as exc:
        raise RuntimeError(
            "This Cellpose installation does not provide CellposeModel."
        ) from exc

    try:
        if custom_path is not None:
            return cellpose_model(
                pretrained_model=str(custom_path),
                gpu=False,
            )
        if hasattr(models, "Cellpose"):
            # Cellpose <=3: named models are selected with model_type.
            return cellpose_model(model_type=str(model), gpu=False)
        # Cellpose >=4: built-in and registered names use pretrained_model.
        return cellpose_model(pretrained_model=str(model), gpu=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Cellpose model {os.fspath(model)!r}: {exc}"
        ) from exc


def eval_cellpose_model(model, image, **kwargs):
    """Evaluate either Cellpose 3 or 4 and return its label mask."""
    result = model.eval(image, **kwargs)
    return result[0] if isinstance(result, tuple) else result

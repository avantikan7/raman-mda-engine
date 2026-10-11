"""Lazy, version-compatible Cellpose model loading."""

from __future__ import annotations

import inspect
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
        legacy = hasattr(models, "Cellpose")
        builtin_names = getattr(models, "MODEL_NAMES", ())
        user_names = getattr(models, "get_user_models", lambda: ())()
        if str(model) not in (*builtin_names, *user_names):
            raise ValueError(
                f"Model {model!r} is not available in this Cellpose installation. "
                "Select an available model or browse to its weights file. "
                "Cellpose 3 models such as cyto2 require Cellpose 3; "
                "Cellpose 4 uses SAM/DINO models."
            )
        if legacy and str(model) in {"cyto", "cyto2", "cyto3", "nuclei"}:
            # Preserve the original Cellpose <=3 path for built-in models.
            # The wrapper owns the ``cp`` inference model used by this project
            # before custom-model support was added.
            return models.Cellpose(model_type=str(model), gpu=False)
        if legacy:
            # Custom/model-zoo models need no size file. The Cellpose wrapper
            # requires one, even though we only evaluate its inference model.
            return cellpose_model(model_type=str(model), gpu=False)
        # Cellpose >=4: built-in and registered names use pretrained_model.
        return cellpose_model(pretrained_model=str(model), gpu=False)
    except Exception as exc:
        compatibility_hint = (
            " Use the Cellpose version/architecture used to train these weights; "
            "Cellpose 3 weights cannot be loaded by Cellpose 4 SAM/DINO models."
            if custom_path is not None else ""
        )
        raise RuntimeError(
            f"Could not load Cellpose model {os.fspath(model)!r}: {exc}"
            f"{compatibility_hint}"
        ) from exc


def eval_cellpose_model(model, image, **kwargs):
    """Evaluate either Cellpose 3 or 4 and return its label mask."""
    inference_model = getattr(model, "cp", model)
    # Older tracking callers supplied tile, which modern eval APIs removed.
    # Preserve it only when the implementation actually accepts it.
    parameters = inspect.signature(inference_model.eval).parameters
    if "tile" not in parameters and not any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    ):
        kwargs.pop("tile", None)
    result = inference_model.eval(image, **kwargs)
    return result[0] if isinstance(result, tuple) else result

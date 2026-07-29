from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = (
    Path(__file__).parents[1]
    / "raman_mda_engine"
    / "aiming"
    / "_cellpose.py"
)
_SPEC = importlib.util.spec_from_file_location("_cellpose_loader_test", _MODULE_PATH)
_LOADER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LOADER)


def _fake_cellpose(monkeypatch, *, legacy):
    calls = []

    class CellposeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    models = ModuleType("cellpose.models")
    models.CellposeModel = CellposeModel
    if legacy:
        models.Cellpose = object
    cellpose = ModuleType("cellpose")
    cellpose.models = models
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return calls


def test_cellpose_4_named_model_uses_pretrained_model(monkeypatch):
    calls = _fake_cellpose(monkeypatch, legacy=False)

    _LOADER.new_cellpose_model("cpsam_v2")

    assert calls == [{"pretrained_model": "cpsam_v2", "gpu": False}]


def test_cellpose_3_named_model_uses_model_type(monkeypatch):
    calls = _fake_cellpose(monkeypatch, legacy=True)

    _LOADER.new_cellpose_model("cyto2")

    assert calls == [{"model_type": "cyto2", "gpu": False}]


def test_custom_weights_use_resolved_path(monkeypatch, tmp_path):
    calls = _fake_cellpose(monkeypatch, legacy=False)
    weights = tmp_path / "my-cell-model"
    weights.write_bytes(b"weights")

    _LOADER.new_cellpose_model(weights)

    assert calls == [
        {"pretrained_model": str(weights.resolve()), "gpu": False}
    ]


def test_missing_custom_weights_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cellpose model file not found"):
        _LOADER.new_cellpose_model(tmp_path / "missing-model")


def test_eval_uses_cellpose_model_directly():
    class Model:
        def eval(self, image, **kwargs):
            assert image == "image"
            assert kwargs == {"diameter": 50}
            return "labels", "flows", "styles"

    assert _LOADER.eval_cellpose_model(
        Model(), "image", diameter=50
    ) == "labels"

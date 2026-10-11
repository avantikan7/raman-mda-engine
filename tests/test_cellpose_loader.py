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

    class Cellpose:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.cp = CellposeModel()

    models = ModuleType("cellpose.models")
    models.CellposeModel = CellposeModel
    models.MODEL_NAMES = ["cyto2"] if legacy else ["cpsam_v2"]
    models.get_user_models = lambda: ["my-registered-model"]
    if legacy:
        models.Cellpose = Cellpose
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

    model = _LOADER.new_cellpose_model("cyto2")

    assert hasattr(model, "cp")
    assert calls == [
        {"model_type": "cyto2", "gpu": False},
        {},
    ]


@pytest.mark.parametrize("legacy", [True, False])
def test_custom_weights_use_resolved_path(monkeypatch, tmp_path, legacy):
    calls = _fake_cellpose(monkeypatch, legacy=legacy)
    weights = tmp_path / "my-cell-model"
    weights.write_bytes(b"weights")

    _LOADER.new_cellpose_model(weights)

    assert calls == [
        {"pretrained_model": str(weights.resolve()), "gpu": False}
    ]


@pytest.mark.parametrize("legacy", [True, False])
def test_registered_model_does_not_require_size_weights(monkeypatch, legacy):
    calls = _fake_cellpose(monkeypatch, legacy=legacy)

    _LOADER.new_cellpose_model("my-registered-model")

    key = "model_type" if legacy else "pretrained_model"
    assert calls == [{key: "my-registered-model", "gpu": False}]


def test_legacy_name_on_cellpose_4_does_not_silently_use_sam(monkeypatch):
    calls = _fake_cellpose(monkeypatch, legacy=False)

    with pytest.raises(RuntimeError, match="cyto2 require Cellpose 3"):
        _LOADER.new_cellpose_model("cyto2")

    assert calls == []


def test_custom_model_failure_explains_training_version(monkeypatch, tmp_path):
    _fake_cellpose(monkeypatch, legacy=False)
    weights = tmp_path / "legacy-weights"
    weights.write_bytes(b"weights")

    def incompatible_model(**kwargs):
        raise ValueError("incompatible state dict")

    monkeypatch.setattr(sys.modules["cellpose.models"], "CellposeModel", incompatible_model)
    with pytest.raises(RuntimeError, match="Cellpose 3 weights cannot be loaded"):
        _LOADER.new_cellpose_model(weights)


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


def test_eval_preserves_legacy_cellpose_cp_path():
    class InferenceModel:
        def eval(self, image, **kwargs):
            assert image == "image"
            assert kwargs == {"diameter": 50}
            return "legacy-labels", "flows", "styles"

    class LegacyCellpose:
        cp = InferenceModel()

        def eval(self, *_args, **_kwargs):
            raise AssertionError("the wrapper eval path must not be used")

    assert _LOADER.eval_cellpose_model(
        LegacyCellpose(), "image", diameter=50
    ) == "legacy-labels"


def test_eval_drops_obsolete_tile_option_for_modern_api():
    class Model:
        def eval(self, image, diameter=None):
            assert diameter == 50
            return "labels", "flows", "styles"

    assert _LOADER.eval_cellpose_model(
        Model(), "image", diameter=50, tile=False
    ) == "labels"

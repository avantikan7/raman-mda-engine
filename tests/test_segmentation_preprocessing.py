from types import SimpleNamespace

import numpy as np

from raman_mda_engine import RamanEngine
from raman_mda_engine.aiming import autotracking
from raman_mda_engine.aiming.autotracking import _normalize_for_cellpose


def test_percentile_normalization_preserves_contrast_with_hot_pixel():
    image = np.linspace(100, 200, 10_000, dtype=np.float32).reshape(100, 100)
    image[0, 0] = 10_000

    normalized = _normalize_for_cellpose(image)

    assert normalized.dtype == np.float32
    assert normalized[0, 0] == 1
    assert normalized[-1, -1] == 1
    assert normalized[50, 50] > 0.45


def test_circle_normalization_uses_roi_and_zeros_the_exterior():
    image = np.zeros((21, 21), dtype=np.float32)
    image[8:13, 8:13] = np.linspace(100, 200, 25).reshape(5, 5)
    image[0, 0] = 10_000

    normalized = _normalize_for_cellpose(
        image,
        circle_center=(10, 10),
        circle_radius=4,
    )

    assert normalized[0, 0] == 0
    assert normalized[10, 10] > 0
    assert normalized[8:13, 8:13].max() == 1


def test_segment_single_img_uses_cell_sized_default_diameter(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        autotracking,
        "_new_cellpose_model",
        lambda _model: object(),
    )

    def fake_eval(_model, image, **kwargs):
        captured["image"] = image
        captured["diameter"] = kwargs["diameter"]
        return np.zeros(image.shape, dtype=np.uint16)

    monkeypatch.setattr(autotracking, "eval_cellpose_model", fake_eval)

    autotracking.segment_single_img(
        np.arange(100, dtype=np.uint16).reshape(10, 10),
        scale=1,
        crop=False,
    )

    assert captured["diameter"] == 15
    assert captured["image"].dtype == np.float32


def test_segment_single_img_can_use_model_training_diameter(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        autotracking,
        "_new_cellpose_model",
        lambda _model: object(),
    )

    def fake_eval(_model, image, **kwargs):
        captured["diameter"] = kwargs["diameter"]
        return np.zeros(image.shape, dtype=np.uint16)

    monkeypatch.setattr(autotracking, "eval_cellpose_model", fake_eval)

    autotracking.segment_single_img(
        np.arange(100, dtype=np.uint16).reshape(10, 10),
        scale=2,
        crop=False,
        diameter=None,
    )

    assert captured["diameter"] is None


def test_segment_single_img_scales_explicit_diameter(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        autotracking,
        "_new_cellpose_model",
        lambda _model: object(),
    )

    def fake_eval(_model, image, **kwargs):
        captured["diameter"] = kwargs["diameter"]
        return np.zeros(image.shape, dtype=np.uint16)

    monkeypatch.setattr(autotracking, "eval_cellpose_model", fake_eval)

    autotracking.segment_single_img(
        np.arange(100, dtype=np.uint16).reshape(10, 10),
        scale=2,
        crop=False,
        diameter=16,
    )

    assert captured["diameter"] == 8


def test_raman_engine_passes_configured_diameter_to_segmentation(monkeypatch):
    captured = {}

    def fake_segment(_image, _scale, **kwargs):
        captured["diameter"] = kwargs["diameter"]
        return np.zeros((4, 4), dtype=np.uint16)

    monkeypatch.setattr(autotracking, "segment_single_img", fake_segment)

    source = SimpleNamespace(
        name="cells",
        get_mda_points=lambda _event, transform=False: np.array([[0.5, 0.5]]),
    )
    emitted = []
    engine = object.__new__(RamanEngine)
    engine._sources = [source]
    engine._image_x = 4
    engine._image_y = 4
    engine._scale = 1
    engine._segment_crop = False
    engine._cellpose_model = "cyto2"
    engine._cell_diameter = None
    engine._circle_center = (2, 2)
    engine._circle_radius = 2
    engine._last_segments = {}
    engine.raman_events = SimpleNamespace(
        aimUpdated=SimpleNamespace(emit=lambda *args: emitted.append(args))
    )

    engine.update_aim(
        pos=0,
        event=SimpleNamespace(index={"p": 0, "t": 0}),
        img=np.zeros((4, 4), dtype=np.uint16),
    )

    assert captured["diameter"] is None
    assert len(emitted) == 1

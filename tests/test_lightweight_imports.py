from __future__ import annotations

import subprocess
import sys


def test_engine_import_does_not_load_ml_stack() -> None:
    forbidden = ("cellpose", "torch", "btrack")
    code = (
        "import sys; "
        "import raman_mda_engine; "
        f"forbidden={forbidden!r}; "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == item or name.startswith(item + '.') "
        "for item in forbidden)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_aiming_import_does_not_load_ml_stack() -> None:
    forbidden = ("cellpose", "torch", "btrack")
    code = "import sys; from raman_mda_engine.aiming import PointsLayerSource; "
    code += f"assert not any(name.startswith({forbidden!r}) for name in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mda_widget_is_created_lazily() -> None:
    from raman_mda_engine.utils import (
        get_mda_widget_from_napari,
    )

    class Settings:
        def value(self):
            return object()

    class Dock:
        def __init__(self, widget):
            self._widget = widget

        def widget(self):
            return self._widget

    class MainWindow:
        def __init__(self):
            self._dock_widgets = {}
            self.created_key = None

        def _show_dock_widget(self, key):
            self.created_key = key
            self._dock_widgets[key] = Dock(Settings())

    main_window = MainWindow()
    mda_widget = get_mda_widget_from_napari(main_window)

    assert main_window.created_key == "MDA"
    assert mda_widget is main_window._dock_widgets["MDA"].widget()
    assert callable(mda_widget.value)

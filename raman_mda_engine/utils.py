from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_micromanager import MainWindow
    from useq import MDASequence

__all__ = ["get_mda_widget_from_napari", "get_seq_from_napari"]


def get_mda_widget_from_napari(main_window: MainWindow):
    """Return napari-micromanager's MDA editor, creating it if needed.

    napari-micromanager creates its dock widgets lazily, so ``"MDA"`` is not
    present in ``_dock_widgets`` until the user opens it once. Prefer the
    dock's public ``widget()`` accessor and retain a child-search fallback for
    older plugin versions.
    """
    if main_window is None:
        raise RuntimeError("napari-micromanager main window is unavailable")

    dock_widgets = getattr(main_window, "_dock_widgets", None)
    if dock_widgets is None:
        raise RuntimeError("napari-micromanager has no dock widget registry")

    if "MDA" not in dock_widgets:
        show_dock_widget = getattr(main_window, "_show_dock_widget", None)
        if not callable(show_dock_widget):
            raise RuntimeError("napari-micromanager cannot create the MDA dock")
        show_dock_widget("MDA")

    try:
        mda_dock = dock_widgets["MDA"]
    except KeyError as e:
        raise RuntimeError("napari-micromanager did not create the MDA dock") from e

    dock_widget = getattr(mda_dock, "widget", None)
    if callable(dock_widget):
        mda_widget = dock_widget()
        if callable(getattr(mda_widget, "value", None)) or callable(
            getattr(mda_widget, "setValue", None)
        ):
            return mda_widget

    if callable(getattr(mda_dock, "value", None)) or callable(
        getattr(mda_dock, "setValue", None)
    ):
        return mda_dock

    children = getattr(mda_dock, "children", None)
    if callable(children):
        for child in children():
            if callable(getattr(child, "value", None)) or callable(
                getattr(child, "setValue", None)
            ):
                return child

    raise RuntimeError("could not find the MDA settings widget inside its dock")


def get_seq_from_napari(
    main_window: MainWindow,
    # autofocus_device: str = "PFS-Offset",
    # rel_focus_device: str = "Focus",
) -> MDASequence:
    """Get the MDA sequence from the MDA widget.

    Also adds the approriate autofocus metadata

    Parameters
    ----------
    main_window : napari_micromanager.MainWindow
        The plugin main widget.
    autofocus_device : str
        The device name for autofocusing. e.g. `PFS-Offset` for TE-2000
    rel_focus_device : str
        The Z-stage to use for relative steps.

    Returns
    -------
    MDASequence
    """
    seq = get_mda_widget_from_napari(main_window).value()
    new_metadata = dict(seq.metadata)
    # new_metadata["autofocus"] = {
    #     "autofocus_device": autofocus_device,
    #     "rel_focus_device": rel_focus_device,
    # }
    return seq.replace(metadata=new_metadata)

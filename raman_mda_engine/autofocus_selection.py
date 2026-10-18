"""Select low-structure autofocus candidates; this is not a cell-absence classifier."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class FocusCandidate:
    x: int
    y: int
    radius_px: float


def find_empty_patch(bf, fluorescence=(), *, margin_px=10, min_radius_px=20):
    """Return the largest candidate patch in registered, full-resolution frames.

    Distances are camera pixels. BF is optional when fluorescence is acquired;
    a supplied dark or flat BF frame cannot establish a usable field of view.
    """
    if not np.isfinite([margin_px, min_radius_px]).all() or margin_px < 0 or min_radius_px <= 0:
        raise ValueError("Autofocus clearance must be finite and positive")
    fluorescence = tuple(np.asarray(frame, dtype=float) for frame in fluorescence)
    if bf is None:
        if not fluorescence or not any(np.isfinite(f).all() and np.ptp(f) > 0 for f in fluorescence):
            raise ValueError("No informative acquisition images for automatic autofocus")
        reference = fluorescence[0]
    else:
        reference = np.asarray(bf, dtype=float)
    bf = None if bf is None else reference
    if reference.ndim != 2 or min(reference.shape) < 64 or not np.isfinite(reference).all():
        raise ValueError("Automatic autofocus needs finite 2D images of at least 64 pixels per side")
    if bf is not None and (np.median(bf) <= 0 or np.ptp(bf) == 0):
        raise ValueError("Automatic autofocus cannot use a dark or flat BF image")
    valid = np.ones(reference.shape, dtype=bool) if bf is None else ndi.gaussian_filter(bf, 8) > .70 * np.median(bf)
    valid = ndi.binary_erosion(valid, iterations=24, border_value=0)
    if not valid.any():
        raise ValueError("No illuminated BF field available for automatic autofocus")

    def threshold(values, k):
        samples = values[valid]
        median = np.median(samples)
        return median + k * 1.4826 * np.median(np.abs(samples - median))

    occupied = np.zeros(reference.shape, dtype=bool)
    if bf is not None:
        detail = ndi.gaussian_filter(bf, 1.2) - ndi.gaussian_filter(bf, 10)
        energy = np.sqrt(ndi.gaussian_filter(detail**2, 3))
        occupied = energy > threshold(energy, 1.5)
    for frame in fluorescence:
        frame = np.asarray(frame, dtype=float)
        if frame.shape != reference.shape or not np.isfinite(frame).all():
            raise ValueError("Acquisition channels must have matching shapes and finite pixels")
        smooth = ndi.gaussian_filter(frame, 1.5)
        for scale in (8, 20, 40):
            detail = smooth - ndi.gaussian_filter(frame, scale)
            occupied |= detail > threshold(detail, 2.5)
        occupied |= smooth > threshold(smooth, 3)
    occupied = ndi.binary_closing(occupied & valid, iterations=2)
    labels, _ = ndi.label(occupied)
    sizes = np.bincount(labels.ravel())
    occupied &= sizes[labels] >= 12
    clearance = ndi.distance_transform_edt(valid & ~occupied)
    distance = ndi.distance_transform_edt(clearance > margin_px)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    radius = float(distance[y, x])
    if radius <= min_radius_px:
        raise ValueError(f"No empty autofocus patch with radius > {min_radius_px:g} px (largest {radius:.1f} px)")
    return FocusCandidate(int(x), int(y), radius)



def field_key(event):
    """Never combine different positions, timepoints, or grid tiles."""
    return (event.index.get("t", 0), event.index.get("p", 0), event.index.get("g", 0),
            event.x_pos, event.y_pos)


class AcquisitionFocusSelection:
    """Buffer scheduled images only, retaining one near-baseline image per channel.

    Wait for the complete camera stack at a field before changing focus so an
    acquisition Z stack is not shifted halfway through. No camera methods here.
    """
    def __init__(self):
        self.pending = set()
        self.finished = set()
        self.frames = {}

    def request(self, event):
        key = field_key(event)
        if key not in self.finished:
            self.pending.add(key)

    def consume(self, event, image, min_radius_px):
        key = field_key(event)
        if key not in self.pending:
            return None
        sequence = event.sequence
        if sequence is None or event.channel is None:
            return None
        t = event.index.get("t", 0)
        rm = sequence.metadata.get("raman", {}).get("channel", "RM")
        channels = {i: c for i, c in enumerate(sequence.channels)
                    if c.config != rm and t % c.acquire_every == 0}
        c = event.index.get("c", 0)
        if c not in channels:
            return None
        z = event.index.get("z", 0)
        planes = list(sequence.z_plan.positions()) if sequence.z_plan else [0.0]
        score = abs(planes[z]) if z < len(planes) else float("inf")
        field = self.frames.setdefault(key, {})
        record = field.setdefault(c, {"seen": set(), "score": float("inf")})
        record["seen"].add(z)
        if score < record["score"]:
            record.update(score=score, image=np.array(image, copy=True))
        if not all(i in field and len(field[i]["seen"]) >= (len(planes) if ch.do_stack else 1)
                   for i, ch in channels.items()):
            return None
        self.pending.remove(key)
        self.finished.add(key)
        self.frames.pop(key)
        bf = next((field[i]["image"] for i, ch in channels.items() if ch.config == "BF"), None)
        fp = [field[i]["image"] for i, ch in channels.items() if ch.config != "BF"]
        try:
            return find_empty_patch(bf, fp, min_radius_px=min_radius_px)
        except ValueError as error:
            # An unusable image or insufficient clearance is a skip, not a run failure.
            return str(error)

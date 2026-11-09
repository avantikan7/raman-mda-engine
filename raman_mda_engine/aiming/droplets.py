"""Aim points and Raman grids for segmented lipid droplets."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def droplet_targets(
    masks,
    density: int = 4,
    margin: float = 1.5,
    min_area: int = 20,
    image=None,
    min_rel_intensity: float = 0.0,
    max_area: int | None = None,
):
    """Return an aim point and an interior lattice for each labelled droplet.

    The aim point is the pixel farthest from the droplet edge (distance-transform
    maximum), so it always lies inside the droplet, unlike the centroid of a
    concave or touching shape. The lattice has ``density``-pixel spacing, is
    centred on the aim point, and keeps only points at least ``margin`` pixels
    from the edge.

    With ``image`` given, droplets whose mean intensity is below
    ``min_rel_intensity`` times the brightest droplet's mean are dropped (dim
    cell body or out-of-focus signal). ``max_area`` drops oversized objects.

    Returns a list of dicts with ``label``, ``centroid`` and ``aim`` (row, col),
    ``radius`` (distance from the aim point to the edge) and ``grid`` (N x 2).
    """
    masks = np.asarray(masks)
    labels = np.unique(masks[masks > 0])
    means = {}
    if image is not None and min_rel_intensity > 0 and len(labels):
        means = {int(l): float(np.mean(np.asarray(image)[masks == l])) for l in labels}
        brightest = max(means.values())
    out = []
    for label in labels:
        region = masks == label
        area = int(region.sum())
        if area < min_area or (max_area is not None and area > max_area):
            continue
        if means and means[int(label)] < min_rel_intensity * brightest:
            continue
        # Pad so droplets touching the image border count the border as an edge.
        dist = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        aim = np.array(np.unravel_index(np.argmax(dist), dist.shape))
        offsets = np.arange(-(max(masks.shape) // density), max(masks.shape) // density + 1) * density
        rows, cols = np.meshgrid(aim[0] + offsets, aim[1] + offsets, indexing="ij")
        pts = np.column_stack([rows.ravel(), cols.ravel()])
        inside = (
            (pts[:, 0] >= 0) & (pts[:, 0] < masks.shape[0])
            & (pts[:, 1] >= 0) & (pts[:, 1] < masks.shape[1])
        )
        pts = pts[inside]
        pts = pts[dist[pts[:, 0], pts[:, 1]] >= margin]
        out.append(
            {
                "label": int(label),
                "centroid": np.array(ndi.center_of_mass(region)),
                "aim": aim,
                "radius": float(dist[tuple(aim)]),
                "grid": pts,
            }
        )
    return out


def link_labels_across_z(stack_masks, min_iou: float = 0.3):
    """Give the same droplet the same label on neighbouring z-planes.

    ``stack_masks`` is a (Z, Y, X) array of independent 2D label images. Each
    plane's objects are matched to the previous plane's objects by overlap
    (intersection over union, greedy by best overlap); unmatched objects get a
    new label. Returns a relabelled uint32 array of the same shape.
    """
    stack_masks = np.asarray(stack_masks)
    out = np.zeros(stack_masks.shape, np.uint32)
    next_label = 1
    for z in range(stack_masks.shape[0]):
        plane = stack_masks[z]
        prev = out[z - 1] if z else None
        pairs = []
        if prev is not None:
            for label in np.unique(plane[plane > 0]):
                region = plane == label
                hits = prev[region]
                for p in np.unique(hits[hits > 0]):
                    inter = np.count_nonzero(hits == p)
                    union = region.sum() + np.count_nonzero(prev == p) - inter
                    iou = inter / union
                    if iou >= min_iou:
                        pairs.append((iou, int(label), int(p)))
        assigned, used = {}, set()
        for iou, label, p in sorted(pairs, reverse=True):
            if label not in assigned and p not in used:
                assigned[label] = p
                used.add(p)
        for label in np.unique(plane[plane > 0]):
            new = assigned.get(int(label))
            if new is None:
                new = next_label
                next_label += 1
            out[z][plane == label] = new
    return out


def droplet_point_rows(targets_by_z, include_grid=True, best_plane_only=True):
    """Flatten per-plane droplet targets into laser aim points.

    ``targets_by_z`` maps a z index to the list returned by ``droplet_targets``
    (each item optionally carrying a ``track`` id from z-linking). Returns a list
    of ``(z, droplet_id, kind, y, x)`` with ``kind`` in ``{"aim", "grid"}``. With
    ``best_plane_only`` each droplet contributes only its widest plane (largest
    inscribed radius), so it is sampled once instead of once per plane. The aim
    point comes first, then the grid points inside the droplet.
    """
    best = {}
    for z, targets in targets_by_z.items():
        for t in targets:
            d = t.get("track", t["label"])
            if d not in best or t["radius"] > best[d][1]:
                best[d] = (z, t["radius"])
    rows = []
    for z in sorted(targets_by_z):
        for t in targets_by_z[z]:
            d = t.get("track", t["label"])
            if best_plane_only and best[d][0] != z:
                continue
            rows.append((z, d, "aim", int(t["aim"][0]), int(t["aim"][1])))
            if include_grid:
                rows.extend((z, d, "grid", int(y), int(x)) for y, x in t["grid"])
    return rows

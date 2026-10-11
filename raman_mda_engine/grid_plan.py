"""Compact Raman grids and lazy acquisition events, without position tables."""
from __future__ import annotations

from copy import deepcopy
import math

from useq import MDAEvent, MDASequence, Position

GRID_KEY = "raman_grid"


def grid_count(sequence):
    plan = sequence.metadata.get(GRID_KEY)
    if not plan:
        return len(sequence.stage_positions)
    validate_grid(plan)
    return sum(region["x"][2] * region["y"][2] for region in plan["regions"])


def validate_grid(plan):
    if plan.get("version") != 1 or not plan.get("regions"):
        raise ValueError("Unsupported or empty Raman grid plan")
    for region in plan["regions"]:
        for axis in ("x", "y"):
            start, stop, count = region[axis]
            if not math.isfinite(start) or not math.isfinite(stop) or not isinstance(count, int) or count < 1:
                raise ValueError("Grid axes require finite bounds and positive integer counts")
        if region.get("snake_axis") not in (None, "x", "y"):
            raise ValueError("Invalid grid traversal order")
        if region.get("z") is not None and not math.isfinite(region["z"]):
            raise ValueError("Grid Z must be finite")
        surface = region.get("tilt")
        if surface:
            degree = surface["degree"]
            if degree not in (1, 2, 3) or len(surface["coefficients"]) != (degree + 1) * (degree + 2) // 2:
                raise ValueError("Invalid grid Z surface")
            if any(not math.isfinite(v) for k in ("origin", "scale", "coefficients") for v in surface[k]) or any(v <= 0 for v in surface["scale"]):
                raise ValueError("Invalid grid Z surface values")


def _axis_value(axis, index):
    start, stop, count = axis
    return (start + stop) / 2 if count == 1 else start + (stop - start) * index / (count - 1)


def grid_position(sequence, index):
    """Calculate one position by index, including raster/snake order and tilt."""
    plan = sequence.metadata[GRID_KEY]
    original_index = index
    for region_index, region in enumerate(plan["regions"]):
        nx, ny = region["x"][2], region["y"][2]
        if index >= nx * ny:
            index -= nx * ny
            continue
        if region.get("snake_axis") == "x":
            iy, ix = divmod(index, nx)
            if iy % 2:
                ix = nx - 1 - ix
        else:
            ix, iy = divmod(index, ny)
            if region.get("snake_axis") == "y" and ix % 2:
                iy = ny - 1 - iy
        x, y = _axis_value(region["x"], ix), _axis_value(region["y"], iy)
        z = region.get("z")
        if surface := region.get("tilt"):
            sx = (x - surface["origin"][0]) / surface["scale"][0]
            sy = (y - surface["origin"][1]) / surface["scale"][1]
            terms = [sx ** xp * sy ** (degree - xp)
                     for degree in range(surface["degree"] + 1)
                     for xp in range(degree, -1, -1)]
            z = sum(c * term for c, term in zip(surface["coefficients"], terms))
        return Position(x=x, y=y, z=z, name=f"Grid {region_index + 1}, p{original_index}")
    raise IndexError(original_index)


def iter_grid_events(sequence):
    """Traverse virtual positions in MDA axis order, allocating one event at a time."""
    if GRID_KEY not in sequence.metadata:
        yield from sequence
        return
    count = grid_count(sequence)
    if sequence.grid_plan or sequence.stage_positions:
        raise ValueError("A Raman grid cannot also use native grid/stage positions")
    axes = {"p": range(count)}
    if sequence.time_plan:
        axes["t"] = tuple(sequence.time_plan)
    if sequence.channels:
        axes["c"] = sequence.channels
    if sequence.z_plan:
        axes["z"] = tuple(sequence.z_plan.positions())
    order = [axis for axis in sequence.axis_order if axis in axes]
    if "p" not in order:
        order.insert(order.index("t") + 1 if "t" in order else 0, "p")
    af = deepcopy(sequence.autofocus_plan)
    if af is not None:
        af._previous = {}
    last_t = None

    def combinations(depth, indices, values):
        if depth == len(order):
            yield indices, values
            return
        axis = order[depth]
        for index, value in enumerate(axes[axis]):
            yield from combinations(depth + 1, {**indices, axis: index}, {**values, axis: value})

    def raw_events():
        nonlocal last_t
        for index, values in combinations(0, {}, {}):
            channel = values.get("c")
            if channel and index.get("t", 0) % channel.acquire_every:
                continue
            if channel and not channel.do_stack and "z" in axes and index["z"] != len(axes["z"]) // 2:
                continue
            position = grid_position(sequence, values["p"])
            z = values.get("z", position.z)
            if "z" in values:
                if sequence.z_plan.is_relative:
                    z += position.z or 0
                z += channel.z_offset if channel else 0
            event = MDAEvent(index=index, sequence=sequence, x_pos=position.x,
                             y_pos=position.y, z_pos=z, pos_name=position.name,
                             channel={"config": channel.config, "group": channel.group} if channel else None,
                             exposure=channel.exposure if channel else None,
                             min_start_time=values.get("t"),
                             reset_event_timer=index.get("t") == 0 and last_t != 0)
            if af is not None:
                if focus := af.event(event):
                    yield focus
            yield event
            last_t = index.get("t")

    previous = None
    for event in raw_events():
        if previous is not None:
            keep_open = bool(sequence.keep_shutter_open_across) and all(
                axis in sequence.keep_shutter_open_across
                for axis, value in previous.index.items() if event.index.get(axis) != value
            )
            yield previous.model_copy(update={"keep_shutter_open": keep_open})
        previous = event
    if previous is not None:
        yield previous

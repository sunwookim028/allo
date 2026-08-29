#!/usr/bin/env python3
"""Small dependency-free stream-graph macro placer.

The optimization lattice is deliberately coarse.  It is not the Innovus site
grid and it is not expected to produce final manufacturing coordinates.  Its
job is to find a compact, connectivity-aware ordering; exact rectangle
legalization and compaction finish the placement in physical units.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from statistics import median


def _suffix_side(kernel: str) -> str | None:
    leaf = kernel.rsplit("/", 1)[-1].lower()
    for suffix, side in (("_w", "W"), ("_e", "E"), ("_n", "N"), ("_s", "S")):
        if leaf.endswith(suffix):
            return side
    return None


def _cells(position: tuple[int, int], size: tuple[int, int]):
    x, y = position
    width, height = size
    return ((xx, yy) for xx in range(x, x + width) for yy in range(y, y + height))


def _overlap_cost(count: int) -> int:
    return max(0, count - 1) ** 2


def _center(position: tuple[int, int], size: tuple[int, int]) -> tuple[float, float]:
    return position[0] + size[0] / 2, position[1] + size[1] / 2


def _direction_cost(
    left_position: tuple[int, int],
    left_size: tuple[int, int],
    right_position: tuple[int, int],
    right_size: tuple[int, int],
    direction: str | None,
) -> float:
    if direction is None:
        return 0.0
    left_x, left_y = _center(left_position, left_size)
    right_x, right_y = _center(right_position, right_size)
    delta = {
        "E": right_x - left_x,
        "W": left_x - right_x,
        "N": right_y - left_y,
        "S": left_y - right_y,
    }[direction]
    return max(0.0, 1.0 - delta) ** 2


def _boundary_cost(
    positions: dict[str, tuple[float, float]],
    sizes: dict[str, tuple[float, float]],
    kernels: dict[str, str],
) -> float:
    """Return a normalized, deliberately modest preferred-side penalty."""
    if not positions:
        return 0.0
    x0, y0, x1, y1 = _bbox(positions, sizes)
    scale = max(1.0, x1 - x0, y1 - y0)
    distances = []
    for name, position in positions.items():
        side = _suffix_side(kernels.get(name, ""))
        if side == "W":
            distances.append(position[0] - x0)
        elif side == "E":
            distances.append(x1 - position[0] - sizes[name][0])
        elif side == "S":
            distances.append(position[1] - y0)
        elif side == "N":
            distances.append(y1 - position[1] - sizes[name][1])
    return sum(max(0.0, value) for value in distances) / max(1.0, len(distances) * scale)


def _logical_seed(items: list[dict], pitch: float) -> dict[str, tuple[int, int]]:
    """Seed all macros together from PID coordinates and boundary semantics."""
    scale = max(2, math.ceil(max(max(item["width"], item["height"]) for item in items) / pitch))
    two_dimensional = [item for item in items if len(item.get("pid", ())) >= 2]
    if two_dimensional:
        rows = [item["pid"][-2] for item in two_dimensional]
        columns = [item["pid"][-1] for item in two_dimensional]
        min_row, max_row = min(rows), max(rows)
        min_column, max_column = min(columns), max(columns)
    else:
        min_row = max_row = min_column = max_column = 0

    positions = {}
    unknown = []
    for item in sorted(items, key=lambda value: value["name"]):
        pid = tuple(item.get("pid", (0,)))
        side = _suffix_side(item.get("kernel", ""))
        lane = pid[-1] if pid else 0
        if len(pid) >= 2:
            # Allo PID rows increase south; physical Y increases north.
            x = (pid[-1] - min_column) * scale
            y = (max_row - pid[-2]) * scale
        elif side == "W":
            x, y = -scale, (max_row - lane) * scale
        elif side == "E":
            x, y = (max_column - min_column + 1) * scale, (max_row - lane) * scale
        elif side == "N":
            x, y = (lane - min_column) * scale, (max_row - min_row + 1) * scale
        elif side == "S":
            x, y = (lane - min_column) * scale, -scale
        else:
            unknown.append(item)
            continue
        positions[item["name"]] = (int(x), int(y))

    side = max(1, math.ceil(math.sqrt(len(unknown))))
    for index, item in enumerate(unknown):
        positions[item["name"]] = ((index % side) * scale, (index // side) * scale)
    return positions


def _bbox(positions: dict[str, tuple[int, int]], sizes: dict[str, tuple[int, int]]):
    return (
        min(position[0] for position in positions.values()),
        min(position[1] for position in positions.values()),
        max(positions[name][0] + sizes[name][0] for name in positions),
        max(positions[name][1] + sizes[name][1] for name in positions),
    )


def _grid_level(
    items: list[dict],
    edges: list[dict],
    pitch: float,
    start: dict[str, tuple[float, float]] | None,
    separation_x: float,
    separation_y: float,
    max_passes: int,
    minimum_improvement: float,
) -> tuple[dict[str, tuple[float, float]], dict]:
    names = [item["name"] for item in items]
    item_by_name = {item["name"]: item for item in items}
    kernels = {item["name"]: item.get("kernel", "") for item in items}
    sizes = {
        item["name"]: (
            max(1, math.ceil((item["width"] + separation_x) / pitch)),
            max(1, math.ceil((item["height"] + separation_y) / pitch)),
        )
        for item in items
    }
    if start is None:
        positions = _logical_seed(items, pitch)
    else:
        positions = {
            name: (round(point[0] / pitch), round(point[1] / pitch))
            for name, point in start.items()
        }

    occupancy: Counter = Counter()
    occupants = defaultdict(set)
    for name in names:
        for cell in _cells(positions[name], sizes[name]):
            occupancy[cell] += 1
            occupants[cell].add(name)
    overlap = sum(_overlap_cost(value) for value in occupancy.values())

    adjacency = defaultdict(list)
    wire = direction = 0.0
    for edge_index, edge in enumerate(edges):
        left, right = edge["left"], edge["right"]
        if left not in positions or right not in positions:
            continue
        adjacency[left].append(edge_index)
        adjacency[right].append(edge_index)
        left_center = _center(positions[left], sizes[left])
        right_center = _center(positions[right], sizes[right])
        wire += edge["weight"] * (
            abs(left_center[0] - right_center[0]) + abs(left_center[1] - right_center[1])
        )
        direction += _direction_cost(
            positions[left], sizes[left], positions[right], sizes[right], edge.get("direction")
        )

    bounds = _bbox(positions, sizes)
    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    footprint = max(1, sum(width * height for width, height in sizes.values()))
    wire_reference = max(1.0, wire)
    direction_reference = max(1.0, float(len(edges)))

    boundary = _boundary_cost(positions, sizes, kernels)

    def total(candidate_wire, candidate_overlap, candidate_area, candidate_direction, candidate_boundary):
        return (
            candidate_wire / wire_reference
            + candidate_area / footprint
            + 2.0 * candidate_overlap / footprint
            + candidate_direction / direction_reference
            + 0.5 * candidate_boundary
        )

    cost = total(wire, overlap, area, direction, boundary)
    active = deque(sorted(names))
    queued = set(names)
    evaluations = accepted = 0
    maximum_evaluations = max(1, max_passes) * max(1, len(names))
    move_steps = tuple(sorted({1, 2, 4, max(1, round(math.sqrt(len(names)) / 2))}, reverse=True))

    while active and evaluations < maximum_evaluations:
        name = active.popleft()
        queued.discard(name)
        evaluations += 1
        old_position = positions[name]
        old_cells = set(_cells(old_position, sizes[name]))
        best = None
        for step in move_steps:
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step), (step, -step), (-step, step), (-step, -step)):
                new_position = (old_position[0] + dx, old_position[1] + dy)
                new_cells = set(_cells(new_position, sizes[name]))
                changed_cells = old_cells | new_cells
                old_overlap = sum(_overlap_cost(occupancy[cell]) for cell in changed_cells)
                new_overlap_local = 0
                for cell in changed_cells:
                    count = occupancy[cell] - (cell in old_cells) + (cell in new_cells)
                    new_overlap_local += _overlap_cost(count)
                candidate_overlap = overlap - old_overlap + new_overlap_local

                candidate_wire, candidate_direction = wire, direction
                for edge_index in adjacency[name]:
                    edge = edges[edge_index]
                    left, right = edge["left"], edge["right"]
                    other = right if left == name else left
                    before_left = positions[left]
                    before_right = positions[right]
                    before_left_center = _center(before_left, sizes[left])
                    before_right_center = _center(before_right, sizes[right])
                    after_left = new_position if left == name else before_left
                    after_right = new_position if right == name else before_right
                    after_left_center = _center(after_left, sizes[left])
                    after_right_center = _center(after_right, sizes[right])
                    candidate_wire += edge["weight"] * (
                        abs(after_left_center[0] - after_right_center[0])
                        + abs(after_left_center[1] - after_right_center[1])
                        - abs(before_left_center[0] - before_right_center[0])
                        - abs(before_left_center[1] - before_right_center[1])
                    )
                    candidate_direction += _direction_cost(
                        after_left, sizes[left], after_right, sizes[right], edge.get("direction")
                    ) - _direction_cost(
                        before_left, sizes[left], before_right, sizes[right], edge.get("direction")
                    )

                trial_positions = positions
                # Bounding-box recomputation is linear only for a boundary macro;
                # ordinary local moves use constant-time extent updates.
                touches_boundary = (
                    old_position[0] == bounds[0]
                    or old_position[1] == bounds[1]
                    or old_position[0] + sizes[name][0] == bounds[2]
                    or old_position[1] + sizes[name][1] == bounds[3]
                )
                if touches_boundary:
                    trial_positions = dict(positions)
                    trial_positions[name] = new_position
                    candidate_bounds = _bbox(trial_positions, sizes)
                else:
                    candidate_bounds = (
                        min(bounds[0], new_position[0]),
                        min(bounds[1], new_position[1]),
                        max(bounds[2], new_position[0] + sizes[name][0]),
                        max(bounds[3], new_position[1] + sizes[name][1]),
                    )
                candidate_area = (
                    (candidate_bounds[2] - candidate_bounds[0])
                    * (candidate_bounds[3] - candidate_bounds[1])
                )
                if trial_positions is positions:
                    trial_positions = dict(positions)
                    trial_positions[name] = new_position
                candidate_boundary = _boundary_cost(trial_positions, sizes, kernels)
                candidate_cost = total(
                    candidate_wire, candidate_overlap, candidate_area, candidate_direction,
                    candidate_boundary,
                )
                improvement = cost - candidate_cost
                if improvement <= minimum_improvement:
                    continue
                candidate = (
                    candidate_cost,
                    abs(dx) + abs(dy),
                    new_position[0],
                    new_position[1],
                    new_position,
                    new_cells,
                    candidate_wire,
                    candidate_overlap,
                    candidate_area,
                    candidate_direction,
                    candidate_boundary,
                    candidate_bounds,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
        if best is None:
            continue

        (
            cost, _distance, _x, _y, new_position, new_cells, wire, overlap,
            area, direction, boundary, bounds,
        ) = best
        for cell in old_cells - new_cells:
            occupancy[cell] -= 1
            occupants[cell].discard(name)
            if occupancy[cell] == 0:
                del occupancy[cell]
                del occupants[cell]
        for cell in new_cells - old_cells:
            occupancy[cell] += 1
            occupants[cell].add(name)
        positions[name] = new_position
        accepted += 1

        affected = {name}
        affected.update(
            edges[index]["right"] if edges[index]["left"] == name else edges[index]["left"]
            for index in adjacency[name]
        )
        for cell in old_cells | new_cells:
            affected.update(occupants.get(cell, ()))
        for affected_name in sorted(affected):
            if affected_name not in queued:
                active.append(affected_name)
                queued.add(affected_name)

    physical = {
        name: (positions[name][0] * pitch, positions[name][1] * pitch)
        for name in names
    }
    return physical, {
        "pitch": pitch,
        "evaluations": evaluations,
        "accepted_moves": accepted,
        "active_queue_empty": not active,
        "normalized_cost": cost,
        "wire_cost": wire / wire_reference,
        "area_cost": area / footprint,
        "overlap_cells": sum(1 for value in occupancy.values() if value > 1),
        "overlap_cost": overlap / footprint,
        "direction_cost": direction / direction_reference,
        "boundary_side_cost": boundary,
    }


def _rectangles_overlap(left: dict, right: dict, separation_x: float, separation_y: float) -> bool:
    return not (
        left["x"] + left["width"] + separation_x <= right["x"]
        or right["x"] + right["width"] + separation_x <= left["x"]
        or left["y"] + left["height"] + separation_y <= right["y"]
        or right["y"] + right["height"] + separation_y <= left["y"]
    )


def _physical_score(placements: list[dict], edges: list[dict]) -> float:
    by_name = {item["name"]: item for item in placements}
    x0 = min(item["x"] for item in placements)
    y0 = min(item["y"] for item in placements)
    x1 = max(item["x"] + item["width"] for item in placements)
    y1 = max(item["y"] + item["height"] for item in placements)
    macro_area = max(1.0, sum(item["width"] * item["height"] for item in placements))
    wire = direction = 0.0
    for edge in edges:
        if edge["left"] not in by_name or edge["right"] not in by_name:
            continue
        left, right = by_name[edge["left"]], by_name[edge["right"]]
        left_center = (left["x"] + left["width"] / 2, left["y"] + left["height"] / 2)
        right_center = (right["x"] + right["width"] / 2, right["y"] + right["height"] / 2)
        wire += edge["weight"] * (
            abs(left_center[0] - right_center[0]) + abs(left_center[1] - right_center[1])
        )
        direction += _direction_cost(
            (left["x"], left["y"]), (left["width"], left["height"]),
            (right["x"], right["y"]), (right["width"], right["height"]),
            edge.get("direction"),
        )
    wire_scale = max(1.0, sum(edge["weight"] for edge in edges) * math.sqrt(macro_area))
    positions = {item["name"]: (item["x"], item["y"]) for item in placements}
    sizes = {item["name"]: (item["width"], item["height"]) for item in placements}
    kernels = {item["name"]: item.get("kernel", "") for item in placements}
    boundary = _boundary_cost(positions, sizes, kernels)
    return (
        (x1 - x0) * (y1 - y0) / macro_area
        + wire / wire_scale
        + direction / max(1, len(edges))
        + 0.5 * boundary
    )


def _legalize(
    placements: list[dict], edges: list[dict], separation_x: float, separation_y: float
) -> int:
    """Resolve residual overlaps once, largest rectangles first."""
    moves = 0
    fixed = []
    for item in sorted(
        placements, key=lambda value: (-value["width"] * value["height"], value["name"])
    ):
        if not any(
            _rectangles_overlap(item, obstacle, separation_x, separation_y)
            for obstacle in fixed
        ):
            fixed.append(item)
            continue
        original = (item["x"], item["y"])
        candidates = set()
        for obstacle in fixed:
            candidates.update({
                (obstacle["x"] + obstacle["width"] + separation_x, item["y"]),
                (obstacle["x"] - item["width"] - separation_x, item["y"]),
                (item["x"], obstacle["y"] + obstacle["height"] + separation_y),
                (item["x"], obstacle["y"] - item["height"] - separation_y),
            })
        legal = []
        for x, y in candidates:
            item["x"], item["y"] = x, y
            if not any(
                _rectangles_overlap(item, obstacle, separation_x, separation_y)
                for obstacle in fixed
            ):
                legal.append((
                    _physical_score(placements, edges),
                    abs(x - original[0]) + abs(y - original[1]),
                    x, y,
                ))
        if not legal:
            raise ValueError(f"unable to legalize macro {item['name']}")
        _score, _distance, item["x"], item["y"] = min(legal)
        moves += 1
        fixed.append(item)
    return moves


def _compact(
    placements: list[dict], edges: list[dict], separation_x: float, separation_y: float,
    passes: int = 4,
) -> int:
    """Slide rectangles to deterministic legal stops in either direction."""
    moves = 0
    for _ in range(max(1, passes)):
        changed = False
        for axis in ("x", "y"):
            dimension = "width" if axis == "x" else "height"
            cross_axis = "y" if axis == "x" else "x"
            cross_dimension = "height" if axis == "x" else "width"
            separation = separation_x if axis == "x" else separation_y
            for item in sorted(placements, key=lambda value: (value[axis], value["name"])):
                original = item[axis]
                bounds_low = min(value[axis] for value in placements)
                bounds_high = max(value[axis] + value[dimension] for value in placements)
                candidates = {bounds_low, bounds_high - item[dimension]}
                for obstacle in placements:
                    if obstacle is item:
                        continue
                    cross_overlap = not (
                        item[cross_axis] + item[cross_dimension] <= obstacle[cross_axis]
                        or obstacle[cross_axis] + obstacle[cross_dimension] <= item[cross_axis]
                    )
                    if cross_overlap:
                        candidates.add(obstacle[axis] + obstacle[dimension] + separation)
                        candidates.add(obstacle[axis] - item[dimension] - separation)
                old_score = _physical_score(placements, edges)
                legal = []
                for target in sorted(candidates):
                    if abs(target - original) <= 1e-9:
                        continue
                    item[axis] = target
                    if any(
                        _rectangles_overlap(item, obstacle, separation_x, separation_y)
                        for obstacle in placements if obstacle is not item
                    ):
                        continue
                    score = _physical_score(placements, edges)
                    if score + 1e-9 < old_score:
                        legal.append((score, abs(target - original), target))
                item[axis] = original
                if legal:
                    _score, _distance, item[axis] = min(legal)
                    moves += 1
                    changed = True
        if not changed:
            break
    return moves


def optimize_macro_placement(
    items: list[dict],
    edges: list[dict],
    separation_x: float,
    separation_y: float,
    max_passes: int = 16,
    minimum_improvement: float = 1e-5,
) -> tuple[list[dict], dict]:
    """Place all macros from the stream graph and return legal local coordinates."""
    if not items:
        return [], {"algorithm": "multiresolution_stream_grid", "levels": []}
    representative = median(min(item["width"], item["height"]) for item in items)
    coarse_pitch = max(1.0, representative / 2.0)
    pitches = [coarse_pitch, max(0.5, coarse_pitch / 2.0)]
    start = None
    levels = []
    for pitch in pitches:
        start, metrics = _grid_level(
            items, edges, pitch, start, separation_x, separation_y,
            max_passes, minimum_improvement,
        )
        levels.append(metrics)

    placements = [
        {**item, "x": start[item["name"]][0], "y": start[item["name"]][1]}
        for item in items
    ]
    legalization_moves = _legalize(placements, edges, separation_x, separation_y)
    compaction_moves = _compact(placements, edges, separation_x, separation_y)
    minimum_x = min(item["x"] for item in placements)
    minimum_y = min(item["y"] for item in placements)
    for item in placements:
        item["x"] -= minimum_x
        item["y"] -= minimum_y
    width = max(item["x"] + item["width"] for item in placements)
    height = max(item["y"] + item["height"] for item in placements)
    return placements, {
        "algorithm": "multiresolution_stream_grid",
        "cost_function": "normalized_wire + normalized_bbox_area + 2*normalized_overlap + normalized_direction + 0.5*normalized_boundary_side",
        "levels": levels,
        "legalization_moves": legalization_moves,
        "compaction_moves": compaction_moves,
        "region_width": width,
        "region_height": height,
        "macro_area": sum(item["width"] * item["height"] for item in placements),
        "coverage": sum(item["width"] * item["height"] for item in placements) / max(1.0, width * height),
    }

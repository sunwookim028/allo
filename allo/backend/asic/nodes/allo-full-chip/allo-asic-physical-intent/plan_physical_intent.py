#!/usr/bin/env python3
"""Plan optional SRAM perimeter packing and connected Allo kernel tiling."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_macro_placement import optimize_macro_placement


def parameter(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def boolean_parameter(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"true", "1", "yes", "on"}


def choice_parameter(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
    return value


def peripheral_sides_parameter(name: str = "peripheral_placement_sides") -> set[str]:
    raw = os.environ.get(name, "all").strip().lower().replace(",", " ")
    tokens = raw.split()
    if tokens == ["all"]:
        return {"left", "right", "top", "bottom"}
    sides = set(tokens)
    valid = {"left", "right", "top", "bottom"}
    if not sides or len(tokens) != len(sides) or not sides <= valid:
        raise ValueError(
            f"{name} must be 'all' or a non-repeated space-separated subset of "
            "top, bottom, left, right"
        )
    return sides


def snap(value: float, grid: float) -> float:
    return round(math.ceil((value - 1e-12) / grid) * grid, 6)


def oriented_dimensions(width: float, height: float, orientation: str) -> tuple[float, float]:
    if orientation in {"R90", "R270", "MXR90", "MYR90"}:
        return height, width
    return width, height


def validate_predicted_density(
    standard_area: float,
    placeable_area: float,
    requested_density: float,
    tolerance: float = 0.07,
) -> tuple[float, float]:
    predicted = standard_area / placeable_area if placeable_area > 0 else math.inf
    limit = min(0.99, requested_density + tolerance)
    if predicted > limit:
        raise ValueError(
            "planned floorplan is under-provisioned: predicted standard-cell "
            f"density {predicted:.3f} exceeds sanity limit {limit:.3f} "
            f"for requested density {requested_density:.3f}"
        )
    return predicted, limit


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the union of sorted one-dimensional closed intervals."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(item[0], item[1]) for item in merged]


def rectangle_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Return the exact union area of axis-aligned (x0, y0, x1, y1) boxes."""
    x_edges = sorted({value for box in rectangles for value in (box[0], box[2])})
    area = 0.0
    for x0, x1 in zip(x_edges, x_edges[1:]):
        intervals = [
            (box[1], box[3])
            for box in rectangles
            if box[0] < x1 and box[2] > x0
        ]
        area += (x1 - x0) * sum(end - start for start, end in merge_intervals(intervals))
    return area


def distribute_peripheral_area(
    width: float,
    height: float,
    added_area: float,
    sides: set[str],
    preferred_aspect: float,
) -> dict[str, float]:
    """Expand selected edges by exactly added_area, preferring the target aspect."""
    x_sides = [side for side in ("left", "right") if side in sides]
    y_sides = [side for side in ("bottom", "top") if side in sides]
    if added_area <= 0:
        return {side: 0.0 for side in ("left", "right", "bottom", "top")}
    target_area = width * height + added_area
    if x_sides and y_sides:
        final_width = math.sqrt(target_area * preferred_aspect)
        final_height = target_area / final_width
        if final_width < width:
            final_width, final_height = width, target_area / width
        if final_height < height:
            final_height, final_width = height, target_area / height
    elif x_sides:
        final_width, final_height = target_area / height, height
    else:
        final_width, final_height = width, target_area / width
    expansion = {side: 0.0 for side in ("left", "right", "bottom", "top")}
    for side in x_sides:
        expansion[side] = (final_width - width) / len(x_sides)
    for side in y_sides:
        expansion[side] = (final_height - height) / len(y_sides)
    return expansion


def short_row_fragment_cuts(
    placements: list[dict],
    core_width: float,
    core_height: float,
    halo: float,
    minimum_width: float,
) -> list[dict]:
    """Find only horizontal row fragments too narrow to retain safely.

    Hard-macro-plus-halo rectangles are treated as existing row cuts. For each
    y-band with constant obstruction geometry, free x-intervals narrower than
    ``minimum_width`` are emitted as additional cuts. Larger gaps remain
    available to standard cells and well taps.
    """
    if halo < 0 or minimum_width < 0:
        raise ValueError("row-fragment halo and minimum width must be nonnegative")
    obstacles = []
    for item in placements:
        x0 = max(0.0, item["x"] - halo)
        y0 = max(0.0, item["y"] - halo)
        x1 = min(core_width, item["x"] + item["width"] + halo)
        y1 = min(core_height, item["y"] + item["height"] + halo)
        if x1 > x0 and y1 > y0:
            obstacles.append((x0, y0, x1, y1))
    y_edges = sorted({0.0, core_height, *(value for box in obstacles for value in (box[1], box[3]))})
    raw = []
    for y0, y1 in zip(y_edges, y_edges[1:]):
        if y1 <= y0 + 1e-9:
            continue
        midpoint = (y0 + y1) / 2
        blocked = merge_intervals([
            (box[0], box[2]) for box in obstacles
            if box[1] < midpoint < box[3]
        ])
        cursor = 0.0
        for start, end in blocked + [(core_width, core_width)]:
            gap = start - cursor
            if 1e-9 < gap < minimum_width - 1e-9:
                raw.append([cursor, y0, start, y1])
            cursor = max(cursor, end)

    # Coalesce vertically adjacent bands with the same x interval so Innovus
    # receives a compact and deterministic set of cutRow rectangles.
    cuts: list[list[float]] = []
    for rectangle in sorted(raw, key=lambda box: (box[0], box[2], box[1], box[3])):
        if (
            cuts
            and abs(cuts[-1][0] - rectangle[0]) < 1e-9
            and abs(cuts[-1][2] - rectangle[2]) < 1e-9
            and abs(cuts[-1][3] - rectangle[1]) < 1e-9
        ):
            cuts[-1][3] = rectangle[3]
        else:
            cuts.append(rectangle)
    return [
        {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]}
        for box in cuts
    ]


def macro_channel_soft_blockages(
    placements: list[dict],
    core_width: float,
    core_height: float,
    separation_x: float,
    separation_y: float,
    spacing_fraction: float,
) -> list[dict]:
    """Cover each hardened PE plus a fraction of its separation."""
    halo_x = separation_x * spacing_fraction
    halo_y = separation_y * spacing_fraction
    blockages = []
    for index, item in enumerate(placements):
        x0 = max(0.0, item["x"] - halo_x)
        y0 = max(0.0, item["y"] - halo_y)
        x1 = min(core_width, item["x"] + item["width"] + halo_x)
        y1 = min(core_height, item["y"] + item["height"] + halo_y)
        if x1 > x0 and y1 > y0:
            blockages.append({
                "name": f"allo_macro_channel_soft_{index}",
                "macro": item["name"],
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
            })
    return blockages


def lef_macros(path: Path) -> dict[str, tuple[float, float, list[str]]]:
    text = path.read_text(errors="replace")
    result = {}
    for match in re.finditer(r"(?ms)^\s*MACRO\s+(\S+)\s*$(?P<body>.*?)^\s*END\s+\1\s*$", text):
        size = re.search(r"\bSIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)\s*;", match.group("body"))
        if size:
            symmetry = re.search(r"\bSYMMETRY\s+([^;]+);", match.group("body"))
            result[match.group(1)] = (
                float(size.group(1)),
                float(size.group(2)),
                symmetry.group(1).split() if symmetry else [],
            )
    return result


def semantic_kernel(semantic_id: str) -> str:
    parts = semantic_id.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else semantic_id


def semantic_pid(semantic_id: str) -> tuple[int, ...]:
    match = re.search(r"pid=([0-9,-]+)", semantic_id)
    return tuple(int(item) for item in match.group(1).split(",")) if match else (0,)


def synthesis_area(reports: Path, require_macro_area: bool = False) -> dict[str, float]:
    """Read DC total and abstract macro areas from one mapped area report.

    The macro value here is the Liberty/DB area included in DC's total.  It is
    deliberately distinct from the physical LEF footprint used for floorplan
    geometry.
    """
    if not reports.exists():
        raise ValueError(f"synthesis report directory does not exist: {reports}")
    for path in reports.rglob("*.area.rpt"):
        text = path.read_text(errors="replace")
        total = None
        for pattern in [r"Total cell area:\s*([0-9.eE+-]+)", r"^\s*Total\s+cell\s+area\s+([0-9.eE+-]+)"]:
            match = re.search(pattern, text, re.I | re.M)
            if match:
                total = float(match.group(1))
                break
        if total is None:
            continue
        macro_match = re.search(
            r"Macro(?:/Black Box|/black box|\s+and\s+black\s+box)?\s+area:\s*([0-9.eE+-]+)",
            text,
            re.I,
        )
        if require_macro_area and macro_match is None:
            raise ValueError(
                f"hierarchical floorplanning requires Macro/Black Box area in {path}"
            )
        macro = float(macro_match.group(1)) if macro_match else 0.0
        if macro < 0 or macro > total:
            raise ValueError(
                f"invalid synthesis areas in {path}: total={total}, macro={macro}"
            )
        return {
            "dc_total_cell_area_um2": total,
            "dc_macro_abstract_area_um2": macro,
            "estimated_standard_cell_area_um2": total - macro,
        }
    raise ValueError(f"no Total cell area found under {reports}")


def discover_srams(root: Path, netlist: str) -> list[dict]:
    if not root.exists():
        return []
    cells = {}
    for lef in sorted(root.rglob("*.lef")):
        cells.update(lef_macros(lef))
    instances = []
    ident = r"[A-Za-z_$][A-Za-z0-9_$]*"
    for cell, (width, height, symmetry) in sorted(cells.items()):
        pattern = re.compile(rf"(?<![A-Za-z0-9_$]){re.escape(cell)}\s+(?P<name>{ident})\s*\(")
        for match in pattern.finditer(netlist):
            instances.append({
                "name": match.group("name"), "cell": cell, "width": width,
                "height": height, "symmetry": symmetry,
            })
    return instances


def edge_pack(items: list[dict], width: float, height: float, keepout: float, spacing: float) -> tuple[list[dict], dict]:
    cursors = {"bottom": keepout, "right": keepout, "top": keepout, "left": keepout}
    bands = {edge: 0.0 for edge in cursors}
    placed = []
    edge_order = ["bottom", "right", "top", "left"]
    edge_index = 0
    for item in sorted(items, key=lambda value: (-max(value["width"], value["height"]), value["name"])):
        for _attempt in range(4):
            edge = edge_order[edge_index % 4]
            w, h = item["width"], item["height"]
            if edge in {"bottom", "top"}:
                fits = cursors[edge] + w <= width - keepout
            else:
                fits = cursors[edge] + h <= height - keepout
            if fits:
                break
            edge_index += 1
        else:
            raise ValueError("SRAM perimeter does not fit estimated core; increase area/aspect ratio")
        if edge == "bottom":
            x, y = cursors[edge], keepout
            cursors[edge] += w + spacing
        elif edge == "right":
            x, y = width - keepout - w, cursors[edge]
            cursors[edge] += h + spacing
        elif edge == "top":
            x, y = width - cursors[edge] - w, height - keepout - h
            cursors[edge] += w + spacing
        else:
            x, y = keepout, height - cursors[edge] - h
            cursors[edge] += h + spacing
        bands[edge] = max(bands[edge], h if edge in {"bottom", "top"} else w)
        placed.append({**item, "x": x, "y": y, "orientation": "R0", "edge": edge, "kind": "sram"})
    return placed, bands


def connection_weights(plan: dict) -> dict[tuple[str, str], float]:
    weights = defaultdict(float)
    for channel in plan.get("whole_region_connections", []):
        kernels = sorted({semantic_kernel(item["pe"]) for item in channel.get("endpoints", [])})
        width_match = re.search(r"[iu](\d+)", str(channel.get("type", "")))
        weight = float(width_match.group(1)) if width_match else 1.0
        for index, left in enumerate(kernels):
            for right in kernels[index + 1 :]:
                weights[(left, right)] += weight
    return dict(weights)


def _stream_width(channel: dict) -> float:
    match = re.search(r"[iu](\d+)", str(channel.get("type", "")))
    return float(match.group(1)) if match else 1.0


def _boundary_side(kernel: str) -> str | None:
    leaf = kernel.rsplit("/", 1)[-1].lower()
    for suffix, side in (("_w", "W"), ("_e", "E"), ("_n", "N"), ("_s", "S")):
        if leaf.endswith(suffix):
            return side
    return None


def _stream_direction(left_semantic: str, right_semantic: str) -> str | None:
    """Return the preferred position of right relative to left.

    Concrete PID deltas are authoritative within one kernel. Directional
    boundary-kernel suffixes are the documented fallback because the current
    whole-region manifest has producer/consumer roles but no cardinal side.
    """
    left_kernel, right_kernel = semantic_kernel(left_semantic), semantic_kernel(right_semantic)
    left_pid, right_pid = semantic_pid(left_semantic), semantic_pid(right_semantic)
    if left_kernel == right_kernel and len(left_pid) >= 2 and len(right_pid) >= 2:
        row_delta = right_pid[-2] - left_pid[-2]
        column_delta = right_pid[-1] - left_pid[-1]
        if column_delta and not row_delta:
            return "E" if column_delta > 0 else "W"
        if row_delta and not column_delta:
            return "S" if row_delta > 0 else "N"
    left_side, right_side = _boundary_side(left_kernel), _boundary_side(right_kernel)
    if left_side and not right_side:
        return {"W": "E", "E": "W", "N": "S", "S": "N"}[left_side]
    if right_side and not left_side:
        return right_side
    return None


def stream_instance_edges(plan: dict, semantic_to_name: dict[str, str]) -> list[dict]:
    """Build exact instance edges from the pre-HLS whole-region stream graph."""
    accumulated = defaultdict(float)
    for channel in plan.get("whole_region_connections", []):
        endpoints = [
            endpoint for endpoint in channel.get("endpoints", [])
            if endpoint.get("pe") in semantic_to_name
        ]
        producers = [item for item in endpoints if item.get("role") == "producer"]
        consumers = [item for item in endpoints if item.get("role") == "consumer"]
        pairs = (
            [(left, right) for left in producers for right in consumers]
            if producers and consumers
            else [(endpoints[index], right) for index, left in enumerate(endpoints) for right in endpoints[index + 1 :]]
        )
        for left, right in pairs:
            left_semantic, right_semantic = left["pe"], right["pe"]
            left_name, right_name = semantic_to_name[left_semantic], semantic_to_name[right_semantic]
            if left_name == right_name:
                continue
            direction = _stream_direction(left_semantic, right_semantic)
            accumulated[(left_name, right_name, direction)] += _stream_width(channel)
    return [
        {"left": left, "right": right, "direction": direction, "weight": weight}
        for (left, right, direction), weight in sorted(accumulated.items())
    ]


D4_PID_TRANSFORMS = {
    "R0": lambda row, col: (row, col),
    "R90": lambda row, col: (-col, row),
    "R180": lambda row, col: (-row, -col),
    "R270": lambda row, col: (col, -row),
    "MX": lambda row, col: (-row, col),
    "MY": lambda row, col: (row, -col),
    "MXR90": lambda row, col: (col, row),
    "MYR90": lambda row, col: (-col, -row),
}


def pid_2d(pid: tuple[int, ...]) -> tuple[int, int]:
    """Use the two spatial PID axes, treating a 1-D mapping as one row."""
    return (pid[-2], pid[-1]) if len(pid) > 1 else (0, pid[-1])


def cross_kernel_pid_relations(plan: dict) -> dict[tuple[str, str], set[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Return concrete cross-kernel PID pairs from the whole-region graph."""
    relations: dict[tuple[str, str], set] = defaultdict(set)
    for channel in plan.get("whole_region_connections", []):
        endpoints = []
        for endpoint in channel.get("endpoints", []):
            semantic_id = endpoint.get("pe")
            if not semantic_id:
                continue
            endpoints.append((semantic_kernel(semantic_id), semantic_pid(semantic_id)))
        for index, (left_kernel, left_pid) in enumerate(endpoints):
            for right_kernel, right_pid in endpoints[index + 1 :]:
                if left_kernel == right_kernel:
                    continue
                if left_kernel < right_kernel:
                    key, pair = (left_kernel, right_kernel), (left_pid, right_pid)
                else:
                    key, pair = (right_kernel, left_kernel), (right_pid, left_pid)
                relations[key].add(pair)
    return dict(relations)


def infer_directed_interleave(
    anchor_kernel: str,
    target_kernel: str,
    anchor_members: list[dict],
    target_members: list[dict],
    pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict | None:
    """Recognize one exact repeated, nonoverlapping D4 PID stencil.

    This deliberately requires complete coverage. It is an opt-in physical
    grouping proof, not a best-effort graph partitioner.
    """
    anchor_slots: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    target_slots: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for member in anchor_members:
        anchor_slots[tuple(member["pid"])].append(member)
    for member in target_members:
        target_slots[tuple(member["pid"])].append(member)
    if (
        len(anchor_slots) < 2
        or any(len(values) != 1 for values in anchor_slots.values())
        or any(len(values) != 1 for values in target_slots.values())
    ):
        return None

    adjacency: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for left_pid, right_pid in pairs:
        if left_pid in anchor_slots and right_pid in target_slots:
            adjacency[left_pid].add(right_pid)
    if set(adjacency) != set(anchor_slots):
        return None
    assigned_targets = [pid for values in adjacency.values() for pid in values]
    if set(assigned_targets) != set(target_slots) or len(assigned_targets) != len(set(assigned_targets)):
        return None

    candidates = []
    for transform_index, (transform_name, transform) in enumerate(D4_PID_TRANSFORMS.items()):
        stencil = None
        valid = True
        for anchor_pid in sorted(anchor_slots):
            anchor_row, anchor_col = transform(*pid_2d(anchor_pid))
            offsets = tuple(sorted(
                (target_row - anchor_row, target_col - anchor_col)
                for target_row, target_col in (
                    pid_2d(target_pid) for target_pid in adjacency[anchor_pid]
                )
            ))
            if stencil is None:
                stencil = offsets
            elif offsets != stencil:
                valid = False
                break
        if valid and stencil:
            span = (
                max(offset[0] for offset in stencil) - min(offset[0] for offset in stencil)
                + max(offset[1] for offset in stencil) - min(offset[1] for offset in stencil)
            )
            candidates.append((span, transform_index, transform_name, stencil))
    if not candidates:
        return None
    _span, _transform_index, transform_name, stencil = min(candidates)
    return {
        "anchor_kernel": anchor_kernel,
        "target_kernel": target_kernel,
        "transform": transform_name,
        "stencil": [list(offset) for offset in stencil],
        "anchor_count": len(anchor_slots),
        "target_count": len(target_slots),
        "groups": [
            {
                "anchor_pid": list(anchor_pid),
                "target_pids": [list(pid) for pid in sorted(adjacency[anchor_pid])],
            }
            for anchor_pid in sorted(anchor_slots)
        ],
        "coverage": 1.0,
        "method": "exact_complete_d4_pid_stencil",
    }


def infer_interleave_pairs(grouped: dict[str, list[dict]], plan: dict) -> tuple[list[dict], list[dict]]:
    """Select nonoverlapping, exact interleaving pairs; report every decision."""
    relations = cross_kernel_pid_relations(plan)
    accepted_candidates = []
    decisions = []
    for (left, right), pairs in sorted(relations.items()):
        if left not in grouped or right not in grouped:
            continue
        directed_pairs = [
            (left, right, pairs),
            (right, left, {(right_pid, left_pid) for left_pid, right_pid in pairs}),
        ]
        candidates = [
            candidate
            for anchor, target, directed in directed_pairs
            if (candidate := infer_directed_interleave(
                anchor, target, grouped[anchor], grouped[target], directed
            )) is not None
        ]
        if not candidates:
            decisions.append({
                "kernels": [left, right], "accepted": False,
                "reason": "no exact complete nonoverlapping repeated PID stencil",
            })
            continue
        # Prefer fewer composite tiles and then a deterministic kernel name.
        candidate = min(candidates, key=lambda item: (item["anchor_count"], item["anchor_kernel"]))
        accepted_candidates.append(candidate)

    # Stronger/repeated candidates claim kernels first. Kernel-disjoint pairing
    # avoids silently constructing transitive many-kernel superclusters.
    claimed = set()
    accepted = []
    for candidate in sorted(
        accepted_candidates,
        key=lambda item: (-item["target_count"], item["anchor_count"], item["anchor_kernel"], item["target_kernel"]),
    ):
        pair = {candidate["anchor_kernel"], candidate["target_kernel"]}
        if pair & claimed:
            decisions.append({
                "kernels": sorted(pair), "accepted": False,
                "reason": "kernel already claimed by a stronger disjoint interleave pair",
            })
            continue
        claimed.update(pair)
        candidate = {**candidate, "accepted": True}
        accepted.append(candidate)
        decisions.append(candidate)
    return accepted, decisions


def build_kernel_cluster(members: list[dict], macro_x: float, macro_y: float) -> dict:
    """Build the original rigid PID-grid cluster for one semantic kernel."""
    slots = defaultdict(list)
    for member in members:
        slots[tuple(member["pid"])].append(member)
    pids = sorted(slots)
    rows = sorted({pid_2d(pid)[0] for pid in pids})
    cols = sorted({pid_2d(pid)[1] for pid in pids})
    col_width = {col: 0.0 for col in cols}
    row_height = {row: 0.0 for row in rows}
    for pid, values in slots.items():
        row, col = pid_2d(pid)
        width = sum(item["width"] for item in values) + macro_x * max(0, len(values) - 1)
        col_width[col] = max(col_width[col], width)
        row_height[row] = max(row_height[row], max(item["height"] for item in values))
    x_offsets, cursor = {}, 0.0
    for col in cols:
        x_offsets[col], cursor = cursor, cursor + col_width[col] + macro_x
    cluster_width = max(0.0, cursor - macro_x)
    y_offsets, cursor = {}, 0.0
    for row in reversed(rows):
        y_offsets[row], cursor = cursor, cursor + row_height[row] + macro_y
    cluster_height = max(0.0, cursor - macro_y)
    local = []
    for pid in pids:
        row, col = pid_2d(pid)
        local_x = x_offsets[col]
        for member in sorted(slots[pid], key=lambda value: value["name"]):
            local.append({**member, "local_x": local_x, "local_y": y_offsets[row]})
            local_x += member["width"] + macro_x
    return {"width": cluster_width, "height": cluster_height, "members": local}


def build_interleaved_cluster(
    inference: dict, grouped: dict[str, list[dict]], macro_x: float, macro_y: float
) -> dict:
    """Tile repeated cross-kernel composite groups on the anchor PID grid."""
    anchor_by_pid = {tuple(item["pid"]): item for item in grouped[inference["anchor_kernel"]]}
    target_by_pid = {tuple(item["pid"]): item for item in grouped[inference["target_kernel"]]}
    tiles = {}
    for group in inference["groups"]:
        anchor_pid = tuple(group["anchor_pid"])
        members = [anchor_by_pid[anchor_pid]] + [
            target_by_pid[tuple(pid)] for pid in group["target_pids"]
        ]
        members = sorted(members, key=lambda item: (item["kernel"] != inference["anchor_kernel"], pid_2d(tuple(item["pid"])), item["name"]))
        width = sum(item["width"] for item in members) + macro_x * (len(members) - 1)
        height = max(item["height"] for item in members)
        tiles[anchor_pid] = {"members": members, "width": width, "height": height}

    rows = sorted({pid_2d(pid)[0] for pid in tiles})
    cols = sorted({pid_2d(pid)[1] for pid in tiles})
    col_width = {col: max(tile["width"] for pid, tile in tiles.items() if pid_2d(pid)[1] == col) for col in cols}
    row_height = {row: max(tile["height"] for pid, tile in tiles.items() if pid_2d(pid)[0] == row) for row in rows}
    x_offsets, cursor = {}, 0.0
    for col in cols:
        x_offsets[col], cursor = cursor, cursor + col_width[col] + macro_x
    width = cursor - macro_x
    y_offsets, cursor = {}, 0.0
    for row in reversed(rows):
        y_offsets[row], cursor = cursor, cursor + row_height[row] + macro_y
    height = cursor - macro_y
    local = []
    for pid, tile in sorted(tiles.items()):
        row, col = pid_2d(pid)
        x = x_offsets[col]
        for member in tile["members"]:
            local.append({**member, "local_x": x, "local_y": y_offsets[row]})
            x += member["width"] + macro_x
    return {"width": width, "height": height, "members": local}


def optimize_slots(kernels: list[str], weights: dict, passes: int) -> dict[str, tuple[int, int]]:
    cols = max(1, math.ceil(math.sqrt(len(kernels))))
    degrees = {kernel: 0.0 for kernel in kernels}
    for (left, right), weight in weights.items():
        degrees[left] = degrees.get(left, 0.0) + weight
        degrees[right] = degrees.get(right, 0.0) + weight
    order = sorted(kernels, key=lambda item: (-degrees.get(item, 0.0), item))
    slots = [(index % cols, index // cols) for index in range(len(order))]

    def cost(candidate: list[str]) -> float:
        positions = {name: slots[index] for index, name in enumerate(candidate)}
        return sum(
            weight * (abs(positions[left][0] - positions[right][0]) + abs(positions[left][1] - positions[right][1]))
            for (left, right), weight in weights.items()
            if left in positions and right in positions
        )

    for _ in range(max(0, passes)):
        baseline = cost(order)
        best = None
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                trial = list(order)
                trial[left], trial[right] = trial[right], trial[left]
                value = cost(trial)
                if value + 1e-9 < baseline and (best is None or value < best[0]):
                    best = (value, trial)
        if best is None:
            break
        order = best[1]
    return {name: slots[index] for index, name in enumerate(order)}


def remap_connection_weights(weights: dict, cluster_by_kernel: dict[str, str]) -> dict:
    """Aggregate semantic-kernel edges onto their selected spatial clusters."""
    result = defaultdict(float)
    for (left, right), weight in weights.items():
        mapped_left = cluster_by_kernel.get(left, left)
        mapped_right = cluster_by_kernel.get(right, right)
        if mapped_left == mapped_right:
            continue
        result[tuple(sorted((mapped_left, mapped_right)))] += weight
    return dict(result)


def variable_slot_extents(
    clusters: dict[str, dict],
    slot_map: dict[str, tuple[int, int]],
    separation_x: float,
    separation_y: float,
) -> tuple[dict[int, float], dict[int, float], float, float]:
    """Size grid columns/rows from only the kernels assigned to each one."""
    if not slot_map:
        return {}, {}, 0.0, 0.0
    column_widths: dict[int, float] = defaultdict(float)
    row_heights: dict[int, float] = defaultdict(float)
    for kernel, (column, row) in slot_map.items():
        column_widths[column] = max(column_widths[column], clusters[kernel]["width"])
        row_heights[row] = max(row_heights[row], clusters[kernel]["height"])

    column_offsets = {}
    cursor = 0.0
    for column in sorted(column_widths):
        column_offsets[column] = cursor
        cursor += column_widths[column] + separation_x
    region_width = cursor - separation_x

    row_offsets = {}
    cursor = 0.0
    for row in sorted(row_heights):
        row_offsets[row] = cursor
        cursor += row_heights[row] + separation_y
    region_height = cursor - separation_y
    return column_offsets, row_offsets, region_width, region_height


ROTATION_MATRICES = {
    "R0": ((1, 0), (0, 1)),
    "R90": ((0, -1), (1, 0)),
    "R180": ((-1, 0), (0, -1)),
    "R270": ((0, 1), (-1, 0)),
    "MX": ((1, 0), (0, -1)),
    "MY": ((-1, 0), (0, 1)),
    "MXR90": ((0, 1), (1, 0)),
    "MYR90": ((0, -1), (-1, 0)),
}
MATRIX_ORIENTATIONS = {value: key for key, value in ROTATION_MATRICES.items()}


def compose_orientation(rotation: str, orientation: str) -> str:
    """Apply a whole-cluster rotation outside a member's existing transform."""
    left, right = ROTATION_MATRICES[rotation], ROTATION_MATRICES[orientation]
    product = tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(2)) for column in range(2))
        for row in range(2)
    )
    return MATRIX_ORIENTATIONS[product]


def legal_cluster_rotations(cluster: dict) -> list[str]:
    # R180 requires X/Y reflection support; quarter turns additionally require
    # R90. Published macro LEFs normally provide the full X Y R90 set.
    rotations = ["R0"]
    if all({"X", "Y"}.issubset(set(item.get("lef_symmetry", []))) for item in cluster["members"]):
        rotations.append("R180")
    if all("R90" in item.get("lef_symmetry", []) for item in cluster["members"]):
        rotations.extend(["R90", "R270"])
    return rotations


def rotate_cluster(cluster: dict, rotation: str) -> dict:
    width, height = cluster["width"], cluster["height"]
    transformed = []
    for item in cluster["members"]:
        x, y, item_w, item_h = item["local_x"], item["local_y"], item["width"], item["height"]
        if rotation == "R0":
            new_x, new_y = x, y
        elif rotation == "R90":
            new_x, new_y = height - y - item_h, x
        elif rotation == "R180":
            new_x, new_y = width - x - item_w, height - y - item_h
        elif rotation == "R270":
            new_x, new_y = y, width - x - item_w
        else:
            raise ValueError(f"unsupported cluster rotation {rotation}")
        transformed.append({
            **item,
            "local_x": new_x,
            "local_y": new_y,
            "width": item_h if rotation in {"R90", "R270"} else item_w,
            "height": item_w if rotation in {"R90", "R270"} else item_h,
            "orientation": compose_orientation(rotation, item["orientation"]),
        })
    return {
        "width": height if rotation in {"R90", "R270"} else width,
        "height": width if rotation in {"R90", "R270"} else height,
        "members": transformed,
        "orientation": rotation,
    }


def choose_cluster_rotations(
    clusters: dict[str, dict],
    slot_map: dict[str, tuple[int, int]],
    weights: dict[tuple[str, str], float],
    separation_x: float,
    separation_y: float,
    passes: int,
) -> dict[str, dict]:
    """Choose legal whole-kernel rotations using area and channel distance."""
    choices = {name: "R0" for name in clusters}

    def materialize() -> dict[str, dict]:
        return {name: rotate_cluster(cluster, choices[name]) for name, cluster in clusters.items()}

    def score(oriented: dict[str, dict]) -> float:
        xs, ys, region_w, region_h = variable_slot_extents(
            oriented, slot_map, separation_x, separation_y
        )
        centers = {
            name: (
                xs[slot_map[name][0]] + oriented[name]["width"] / 2,
                ys[slot_map[name][1]] + oriented[name]["height"] / 2,
            )
            for name in oriented
        }
        wire = sum(
            weight * (abs(centers[left][0] - centers[right][0]) + abs(centers[left][1] - centers[right][1]))
            for (left, right), weight in weights.items()
            if left in centers and right in centers
        )
        return region_w * region_h + wire

    for _ in range(max(1, passes)):
        changed = False
        for name in sorted(clusters):
            original = choices[name]
            best = (score(materialize()), original)
            for rotation in legal_cluster_rotations(clusters[name]):
                choices[name] = rotation
                trial = score(materialize())
                if trial + 1e-9 < best[0]:
                    best = (trial, rotation)
            choices[name] = best[1]
            changed |= best[1] != original
        if not changed:
            break
    return materialize()


def main() -> None:
    inputs, outputs = Path("inputs"), Path("outputs")
    outputs.mkdir(exist_ok=True)
    plan = json.loads((inputs / "assembly-plan.json").read_text())
    registry = json.loads((inputs / "macro-registry.json").read_text())
    netlist = (inputs / "design.v").read_text()
    macro_link = (inputs / "macro-link.rpt").read_text(errors="replace")
    linked_total = re.search(r"^TOTAL\s+(\d+)\s*$", macro_link, re.M)
    if linked_total is None or int(linked_total.group(1)) != plan.get(
        "elaborated_macro_instance_count"
    ):
        raise ValueError(
            "post-synthesis macro-link count does not match assembly plan: "
            f"report={linked_total.group(1) if linked_total else 'missing'}, "
            f"plan={plan.get('elaborated_macro_instance_count')}"
        )
    grid = parameter("placement_grid", 0.005)
    macro_x = parameter("macro_separation_x", 8.0)
    macro_y = parameter("macro_separation_y", 8.0)
    enable_macro_channel_soft_blockages = boolean_parameter(
        "enable_macro_channel_soft_blockages", True
    )
    soft_blockage_fraction = parameter(
        "macro_channel_soft_blockage_fraction", 0.5
    )
    peripheral_sides = peripheral_sides_parameter()
    kernel_x = parameter("kernel_separation_x", 30.0)
    kernel_y = parameter("kernel_separation_y", 30.0)
    keepout = parameter("macro_edge_keepout", 30.0)
    sram_spacing = parameter("sram_separation", 20.0)
    density = parameter("core_density_target", 0.70)
    aspect = parameter("floorplan_aspect_ratio", 1.0)
    passes = int(parameter("kernel_optimization_passes", 12))
    placement_algorithm = choice_parameter(
        "macro_placement_algorithm", "stream_grid", {"stream_grid", "legacy_cluster_grid"}
    )
    grid_passes = int(parameter("stream_grid_max_passes", 16))
    grid_minimum_improvement = parameter("stream_grid_minimum_improvement", 1e-5)
    enable_kernel_rotation = boolean_parameter("enable_kernel_rotation", True)
    interleave_macros = boolean_parameter("interleave_macros", False)
    row_cut_halo = parameter("macro_halo", 2.0)
    min_row_width = parameter("min_placeable_row_segment_width", 12.0)
    if density <= 0 or density >= 1 or aspect <= 0 or grid <= 0:
        raise ValueError("density must be in (0,1), aspect and grid must be positive")
    if soft_blockage_fraction < 0:
        raise ValueError("macro_channel_soft_blockage_fraction must be nonnegative")

    dimensions = {}
    symmetries = {}
    for macro in registry.get("macros", []):
        lef = inputs / "macro-registry" / macro["views"]["lef"]["path"]
        parsed = lef_macros(lef)
        if macro["top_module"] not in parsed:
            raise ValueError(f"LEF lacks canonical macro {macro['top_module']}")
        geometry = parsed[macro["top_module"]]
        dimensions[macro["macro_class_id"]] = geometry[:2]
        symmetries[macro["macro_class_id"]] = geometry[2]

    grouped = defaultdict(list)
    for item in plan.get("replacements", []):
        stable = item["stable_instance_name"]
        if stable not in netlist:
            raise ValueError(f"synthesized netlist lost stable macro identity {stable}")
        width, height = dimensions[item["macro_class_id"]]
        orient = item.get("desired_orientation", "R0")
        if orient not in {"R0", "R90", "R180", "R270", "MX", "MY", "MXR90", "MYR90"}:
            orient = "R0"
        width, height = oriented_dimensions(width, height, orient)
        kernel = semantic_kernel(item["semantic_id"])
        grouped[kernel].append({
            "name": stable, "cell": item["canonical_module"], "semantic_id": item["semantic_id"],
            "pid": semantic_pid(item["semantic_id"]), "width": width, "height": height,
            "orientation": orient, "kind": "pe", "macro_class_id": item["macro_class_id"],
            "lef_symmetry": symmetries[item["macro_class_id"]],
            "candidate_kind": item.get("candidate_kind", "semantic_pe"),
            "kernel": kernel,
        })
    semantic_to_name = {
        item["semantic_id"]: item["name"]
        for members in grouped.values()
        for item in members
    }
    instance_edges = stream_instance_edges(plan, semantic_to_name)

    # Preserve the original one-rigid-cluster-per-kernel behavior unless the
    # explicit experimental flag is enabled and a complete repeated stencil is
    # proven. Rejected/unclaimed kernels still use precisely the same builder.
    interleave_inferences, interleave_decisions = (
        infer_interleave_pairs(grouped, plan) if interleave_macros else ([], [])
    )
    clusters = {}
    interleaved_kernels = set()
    spatial_cluster_by_kernel = {kernel: kernel for kernel in grouped}
    for inference in interleave_inferences:
        anchor = inference["anchor_kernel"]
        target = inference["target_kernel"]
        cluster_name = f"interleave:{anchor}+{target}"
        clusters[cluster_name] = build_interleaved_cluster(
            inference, grouped, macro_x, macro_y
        )
        interleaved_kernels.update({anchor, target})
        spatial_cluster_by_kernel[anchor] = cluster_name
        spatial_cluster_by_kernel[target] = cluster_name
    for kernel, members in grouped.items():
        if kernel not in interleaved_kernels:
            clusters[kernel] = build_kernel_cluster(members, macro_x, macro_y)

    weights = connection_weights(plan)
    placement_weights = remap_connection_weights(weights, spatial_cluster_by_kernel)
    kernel_names = sorted(clusters)
    slot_map = optimize_slots(kernel_names, placement_weights, passes)
    if enable_kernel_rotation:
        clusters = choose_cluster_rotations(
            clusters, slot_map, placement_weights, kernel_x, kernel_y, passes
        )
    else:
        clusters = {name: rotate_cluster(cluster, "R0") for name, cluster in clusters.items()}
    column_offsets, row_offsets, kernel_region_w, kernel_region_h = variable_slot_extents(
        clusters, slot_map, kernel_x, kernel_y
    )

    srams = discover_srams(inputs / "srams", netlist)
    macro_area = sum(item["width"] * item["height"] for values in grouped.values() for item in values)
    sram_area = sum(item["width"] * item["height"] for item in srams)
    area_budget = synthesis_area(
        inputs / "synthesis-reports",
        require_macro_area=bool(grouped or srams),
    )
    # DC's total contains the abstract Liberty/DB area, which need not equal
    # the physical LEF footprint. Remove the former, apply utilization only to
    # standard cells, and add the physical macro footprints exactly once.
    standard_area = area_budget["estimated_standard_cell_area_um2"]
    # Compute the invariant blocked area before choosing which core edges grow.
    # Translation does not change this union, and overlapping half-spacing
    # halos must be counted only once.
    local_pe_placements = []
    for kernel in kernel_names:
        col, row = slot_map[kernel]
        cluster = clusters[kernel]
        for member in cluster["members"]:
            local_pe_placements.append({
                **member,
                "x": column_offsets[col] + member["local_x"],
                "y": row_offsets[row] + member["local_y"],
            })
    placement_metrics = {
        "algorithm": "legacy_cluster_grid",
        "region_width": kernel_region_w,
        "region_height": kernel_region_h,
    }
    if placement_algorithm == "stream_grid" and local_pe_placements:
        source_members = [
            {**item, "orientation": item.get("orientation", "R0")}
            for members in grouped.values()
            for item in members
        ]
        local_pe_placements, placement_metrics = optimize_macro_placement(
            source_members,
            instance_edges,
            separation_x=macro_x,
            separation_y=macro_y,
            max_passes=grid_passes,
            minimum_improvement=grid_minimum_improvement,
        )
        kernel_region_w = placement_metrics["region_width"]
        kernel_region_h = placement_metrics["region_height"]
    halo_x = macro_x * soft_blockage_fraction
    halo_y = macro_y * soft_blockage_fraction
    soft_blockage_union_area = rectangle_union_area([
        (
            item["x"] - halo_x,
            item["y"] - halo_y,
            item["x"] + item["width"] + halo_x,
            item["y"] + item["height"] + halo_y,
        )
        for item in local_pe_placements
    ])
    reserved_pe_area = (
        soft_blockage_union_area
        if enable_macro_channel_soft_blockages
        else macro_area
    )
    target_area = reserved_pe_area + sram_area + standard_area / density

    # SRAM bands, when present, contribute to the minimum macro envelope. The
    # loop stabilizes those edge bands before finalizing the selected-edge
    # expansion; the normal no-SRAM path converges on its first pass.
    bands = {side: 0.0 for side in ("left", "right", "bottom", "top")}
    for _ in range(12):
        margin_left = max(keepout, halo_x) + bands["left"] + (sram_spacing if bands["left"] else 0)
        margin_right = max(keepout, halo_x) + bands["right"] + (sram_spacing if bands["right"] else 0)
        margin_bottom = max(keepout, halo_y) + bands["bottom"] + (sram_spacing if bands["bottom"] else 0)
        margin_top = max(keepout, halo_y) + bands["top"] + (sram_spacing if bands["top"] else 0)
        base_core_w = kernel_region_w + margin_left + margin_right
        base_core_h = kernel_region_h + margin_bottom + margin_top
        requested_added_area = max(0.0, target_area - base_core_w * base_core_h)
        peripheral_expansion = distribute_peripheral_area(
            base_core_w, base_core_h, requested_added_area, peripheral_sides, aspect
        )
        core_w = base_core_w + peripheral_expansion["left"] + peripheral_expansion["right"]
        core_h = base_core_h + peripheral_expansion["bottom"] + peripheral_expansion["top"]
        try:
            sram_placements, new_bands = edge_pack(
                srams, core_w, core_h, keepout, sram_spacing
            )
        except ValueError:
            target_area *= 1.15
            continue
        if new_bands == bands:
            break
        bands = new_bands
    else:
        raise ValueError("unable to create legal SRAM perimeter")
    free = {
        "x": margin_left + peripheral_expansion["left"],
        "y": margin_bottom + peripheral_expansion["bottom"],
        "width": kernel_region_w,
        "height": kernel_region_h,
    }
    origin_x = free["x"]
    origin_y = free["y"]
    pe_placements, cluster_records = [], []
    if placement_algorithm == "stream_grid":
        for member in local_pe_placements:
            pe_placements.append({
                **member,
                "x": snap(origin_x + member["x"], grid),
                "y": snap(origin_y + member["y"], grid),
                "spatial_cluster": "stream_grid",
            })
        for kernel, members in sorted(grouped.items()):
            placed = [item for item in pe_placements if item["kernel"] == kernel]
            if not placed:
                continue
            x0, y0 = min(item["x"] for item in placed), min(item["y"] for item in placed)
            x1 = max(item["x"] + item["width"] for item in placed)
            y1 = max(item["y"] + item["height"] for item in placed)
            cluster_records.append({
                "kernel": kernel, "x": x0, "y": y0,
                "width": x1 - x0, "height": y1 - y0,
                "orientation": "mixed", "member_count": len(placed),
            })
    else:
        for kernel in kernel_names:
            col, row = slot_map[kernel]
            cluster_x = origin_x + column_offsets[col]
            cluster_y = origin_y + row_offsets[row]
            cluster = clusters[kernel]
            cluster_records.append({"kernel": kernel, "x": snap(cluster_x, grid), "y": snap(cluster_y, grid), **{key: cluster[key] for key in ["width", "height", "orientation"]}})
            for member in cluster["members"]:
                pe_placements.append({
                    **{key: value for key, value in member.items() if not key.startswith("local_")},
                    "x": snap(cluster_x + member["local_x"], grid),
                    "y": snap(cluster_y + member["local_y"], grid),
                    "kernel": member.get("kernel", kernel),
                    "spatial_cluster": kernel,
                })
    planned_core_w = round(core_w, 6)
    planned_core_h = round(core_h, 6)
    planned_core_area = planned_core_w * planned_core_h
    minimum_envelope_area = base_core_w * base_core_h
    keepout_allowance = max(
        0.0, minimum_envelope_area - reserved_pe_area - sram_area
    )
    estimated_placeable_area = planned_core_area - reserved_pe_area - sram_area
    predicted_density, density_limit = validate_predicted_density(
        standard_area, estimated_placeable_area, density
    )
    placements = [{**item, "x": snap(item["x"], grid), "y": snap(item["y"], grid)} for item in sram_placements] + pe_placements
    for item in placements:
        if item["x"] < 0 or item["y"] < 0 or item["x"] + item["width"] > planned_core_w + grid or item["y"] + item["height"] > planned_core_h + grid:
            raise ValueError(f"placement outside core: {item['name']}")
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            overlap = not (
                left["x"] + left["width"] <= right["x"]
                or right["x"] + right["width"] <= left["x"]
                or left["y"] + left["height"] <= right["y"]
                or right["y"] + right["height"] <= left["y"]
            )
            if overlap:
                raise ValueError(f"physical-intent overlap: {left['name']} and {right['name']}")
    row_fragment_cuts = short_row_fragment_cuts(
        placements, planned_core_w, planned_core_h, row_cut_halo, min_row_width
    )
    channel_soft_blockages = (
        macro_channel_soft_blockages(
            pe_placements,
            planned_core_w,
            planned_core_h,
            macro_x,
            macro_y,
            soft_blockage_fraction,
        )
        if enable_macro_channel_soft_blockages
        else []
    )
    intent = {
        "schema_version": 1, "stage": "physical_intent", "top_module": plan["top_module"],
        "core": {"width": planned_core_w, "height": planned_core_h, "usable_rectangle": free},
        "constraints": {"macro_separation_x": macro_x, "macro_separation_y": macro_y, "kernel_separation_x": kernel_x, "kernel_separation_y": kernel_y, "edge_keepout": keepout, "peripheral_placement_sides": sorted(peripheral_sides)},
        "standard_cell_area_estimate": standard_area,
        "area_budget": {
            **area_budget,
            "physical_pe_macro_area_um2": macro_area,
            "physical_sram_area_um2": sram_area,
            "soft_blockage_union_area_um2": soft_blockage_union_area,
            "reserved_pe_placement_area_um2": reserved_pe_area,
            "target_standard_cell_density": density,
            "keepout_allowance_um2": keepout_allowance,
            "target_core_area_um2": target_area,
            "planned_core_area_um2": planned_core_area,
            "estimated_placeable_standard_cell_area_um2": estimated_placeable_area,
            "predicted_standard_cell_density": predicted_density,
            "density_sanity_limit": density_limit,
        },
        "peripheral_placement_policy": {
            "sides": sorted(peripheral_sides),
            "base_width": base_core_w,
            "base_height": base_core_h,
            "requested_added_area_um2": requested_added_area,
            "actual_added_area_um2": planned_core_area - minimum_envelope_area,
            "edge_expansion": peripheral_expansion,
            "preferred_aspect_ratio": aspect,
        },
        "kernel_connections": [{"kernels": list(key), "weight": value} for key, value in sorted(weights.items())],
        "instance_stream_connections": instance_edges,
        "kernel_clusters": cluster_records, "placements": placements,
        "kernel_packing_policy": {
            "type": placement_algorithm,
            "whole_kernel_rotation_enabled": enable_kernel_rotation,
            "region_width": kernel_region_w,
            "region_height": kernel_region_h,
            "optimization": placement_metrics,
            "stream_grid_max_passes": grid_passes,
            "stream_grid_minimum_improvement": grid_minimum_improvement,
            "fallback": "legacy_cluster_grid",
        },
        "cross_kernel_interleaving": {
            "enabled": interleave_macros,
            "mode": "exact_complete_d4_pid_stencil" if interleave_macros else "disabled",
            "accepted_pair_count": len(interleave_inferences),
            "accepted_pairs": interleave_inferences,
            "decisions": interleave_decisions,
            "fallback": "rigid_semantic_kernel_cluster",
        },
        "macro_channel_placement_policy": {
            "type": "soft" if enable_macro_channel_soft_blockages else "none",
            "enabled": enable_macro_channel_soft_blockages,
            "spacing_fraction": soft_blockage_fraction,
            "expansion_x": macro_x * soft_blockage_fraction,
            "expansion_y": macro_y * soft_blockage_fraction,
            "region_count": len(channel_soft_blockages),
            "regions": channel_soft_blockages,
        },
        "pe_stream_facing_policy": {
            "pid_column_direction": "east",
            "pid_row_direction": "south",
            "placement_y_direction": "north",
            "method": "reverse_pid_row_order",
        },
        "row_fragment_policy": {
            "macro_halo": row_cut_halo,
            "minimum_retained_width": min_row_width,
            "cut_count": len(row_fragment_cuts),
            "cuts": row_fragment_cuts,
        },
        "sram_support": {"enabled": bool(srams), "instance_count": len(srams), "policy": "sequential_perimeter"},
    }
    (outputs / "physical-intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    tcl = [
        "# Generated physical intent; coordinates are offsets from the core lower-left.",
        f"set allo_core_width {intent['core']['width']}", f"set allo_core_height {intent['core']['height']}",
        "proc place_allo_physical_intent {} {", "  set core [dbGet top.fPlan.coreBox]", "  if {[llength $core] == 1} { set core [lindex $core 0] }", "  set llx [lindex $core 0]", "  set lly [lindex $core 1]", "  set placed 0", "  set all_instance_names [dbGet top.insts.name]",
    ]
    for item in placements:
        tcl.extend([
            f"  set matches [lsearch -all -inline -exact $all_instance_names {{{item['name']}}}]",
            f"  if {{[llength $matches] != 1}} {{ error \"Expected exactly one macro instance {item['name']}\" }}",
            "  set actual_name [lindex $matches 0]",
            f"  placeInstance $actual_name [expr {{$llx + {item['x']}}}] [expr {{$lly + {item['y']}}}] {item['orientation']}",
            "  incr placed",
        ])
    tcl.extend([
        "  if {$placed > 0} {",
        "    setInstancePlacementStatus -allHardMacros -status fixed",
        "  }",
        "  return $placed",
        "}",
    ])
    tcl.extend([
        "",
        "# Remove only row fragments too narrow to support useful placement.",
        "# Large spaces inside and between kernel clusters remain placeable.",
        "proc cut_allo_short_row_fragments {} {",
        "  set core [dbGet top.fPlan.coreBox]",
        "  if {[llength $core] == 1} { set core [lindex $core 0] }",
        "  set llx [lindex $core 0]",
        "  set lly [lindex $core 1]",
        "  set cut_count 0",
    ])
    for cut in row_fragment_cuts:
        tcl.extend([
            f"  cutRow -area [list [expr {{$llx + {cut['x']}}}] [expr {{$lly + {cut['y']}}}] [expr {{$llx + {cut['x']} + {cut['width']}}}] [expr {{$lly + {cut['y']} + {cut['height']}}}]]",
            "  incr cut_count",
        ])
    tcl.extend(["  return $cut_count", "}"])
    tcl.extend([
        "",
        "# Soft blockages exclude ordinary logic but permit optimization cells.",
        "proc create_allo_macro_channel_soft_blockages {} {",
        "  set core [dbGet top.fPlan.coreBox]",
        "  if {[llength $core] == 1} { set core [lindex $core 0] }",
        "  set llx [lindex $core 0]",
        "  set lly [lindex $core 1]",
        "  set blockage_count 0",
    ])
    for blockage in channel_soft_blockages:
        tcl.extend([
            f"  createPlaceBlockage -type soft -snapToSite -name {blockage['name']} -box [list [expr {{$llx + {blockage['x']}}}] [expr {{$lly + {blockage['y']}}}] [expr {{$llx + {blockage['x']} + {blockage['width']}}}] [expr {{$lly + {blockage['y']} + {blockage['height']}}}]]",
            "  incr blockage_count",
        ])
    tcl.extend(["  return $blockage_count", "}"])
    (outputs / "physical-intent.tcl").write_text("\n".join(tcl) + "\n")
    scale = min(1000 / max(core_w, 1), 800 / max(core_h, 1))
    colors = {"pe": "#4c78a8", "sram": "#f58518"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{core_w*scale:.1f}" height="{core_h*scale:.1f}" viewBox="0 0 {core_w} {core_h}">', f'<rect width="{core_w}" height="{core_h}" fill="white" stroke="black"/>']
    for item in placements:
        y = core_h - item["y"] - item["height"]
        svg.append(f'<rect x="{item["x"]}" y="{y}" width="{item["width"]}" height="{item["height"]}" fill="{colors[item["kind"]]}" stroke="black"><title>{item["name"]}</title></rect>')
    svg.append("</svg>")
    (outputs / "physical-intent.svg").write_text("\n".join(svg) + "\n")
    (outputs / "physical-intent-report.txt").write_text(
        f"Core: {intent['core']['width']} x {intent['core']['height']} um\n"
        f"PE macros: {len(pe_placements)}\nSRAMs: {len(srams)} (optional sequential perimeter policy)\n"
        f"Kernel clusters: {len(cluster_records)}\nWeighted inter-kernel cost optimization passes: {passes}\n"
        f"Macro placement algorithm: {placement_algorithm}\n"
        f"Optimized macro region: {kernel_region_w} x {kernel_region_h} um "
        f"(coverage={placement_metrics.get('coverage', 'n/a')})\n"
        f"Stream-grid levels: {len(placement_metrics.get('levels', []))}; "
        f"legalization moves: {placement_metrics.get('legalization_moves', 0)}; "
        f"compaction moves: {placement_metrics.get('compaction_moves', 0)}\n"
        f"Cross-kernel interleaving enabled: {interleave_macros}\n"
        f"Accepted interleave pairs: {len(interleave_inferences)}\n"
        f"Selective short-row cuts: {len(row_fragment_cuts)} (minimum retained width {min_row_width} um)\n"
        f"Macro-channel soft blockages: {len(channel_soft_blockages)} "
        f"(enabled={enable_macro_channel_soft_blockages}, "
        f"fraction={soft_blockage_fraction}, "
        f"expansion={macro_x * soft_blockage_fraction} x "
        f"{macro_y * soft_blockage_fraction} um)\n"
        f"Soft-blockage union area: {soft_blockage_union_area} um^2\n"
        f"Peripheral placement sides: {' '.join(sorted(peripheral_sides))}\n"
        f"Peripheral area added: {planned_core_area - minimum_envelope_area} um^2\n"
        f"Predicted peripheral standard-cell density: {predicted_density}\n"
    )


if __name__ == "__main__":
    main()

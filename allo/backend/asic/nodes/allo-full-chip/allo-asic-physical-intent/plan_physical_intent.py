#!/usr/bin/env python3
"""Plan optional SRAM perimeter packing and connected Allo kernel tiling."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


def parameter(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def snap(value: float, grid: float) -> float:
    return round(math.ceil((value - 1e-12) / grid) * grid, 6)


def oriented_dimensions(width: float, height: float, orientation: str) -> tuple[float, float]:
    if orientation in {"R90", "R270", "MXR90", "MYR90"}:
        return height, width
    return width, height


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


def timing_area(reports: Path) -> float:
    if not reports.exists():
        return 0.0
    for path in reports.rglob("*.area.rpt"):
        text = path.read_text(errors="replace")
        for pattern in [r"Total cell area:\s*([0-9.eE+-]+)", r"^\s*Total\s+cell\s+area\s+([0-9.eE+-]+)"]:
            match = re.search(pattern, text, re.I | re.M)
            if match:
                return float(match.group(1))
    return 0.0


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
    kernel_x = parameter("kernel_separation_x", 30.0)
    kernel_y = parameter("kernel_separation_y", 30.0)
    keepout = parameter("macro_edge_keepout", 30.0)
    sram_spacing = parameter("sram_separation", 20.0)
    density = parameter("core_density_target", 0.70)
    aspect = parameter("floorplan_aspect_ratio", 1.0)
    passes = int(parameter("kernel_optimization_passes", 12))
    row_cut_halo = parameter("macro_halo", 2.0)
    min_row_width = parameter("min_placeable_row_segment_width", 12.0)
    cluster_density = int(parameter("kernel_cluster_max_density_percent", 55))
    if density <= 0 or density >= 1 or aspect <= 0 or grid <= 0:
        raise ValueError("density must be in (0,1), aspect and grid must be positive")
    if cluster_density < 1 or cluster_density > 100:
        raise ValueError("kernel_cluster_max_density_percent must be in [1,100]")

    dimensions = {}
    for macro in registry.get("macros", []):
        lef = inputs / "macro-registry" / macro["views"]["lef"]["path"]
        parsed = lef_macros(lef)
        if macro["top_module"] not in parsed:
            raise ValueError(f"LEF lacks canonical macro {macro['top_module']}")
        dimensions[macro["macro_class_id"]] = parsed[macro["top_module"]][:2]

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
        grouped[semantic_kernel(item["semantic_id"])].append({
            "name": stable, "cell": item["canonical_module"], "semantic_id": item["semantic_id"],
            "pid": semantic_pid(item["semantic_id"]), "width": width, "height": height,
            "orientation": orient, "kind": "pe", "macro_class_id": item["macro_class_id"],
        })

    clusters = {}
    for kernel, members in grouped.items():
        slots = defaultdict(list)
        for member in members:
            slots[member["pid"]].append(member)
        pids = sorted(slots)
        rows = sorted({pid[-2] if len(pid) > 1 else 0 for pid in pids})
        cols = sorted({pid[-1] for pid in pids})
        col_width = {col: 0.0 for col in cols}
        row_height = {row: 0.0 for row in rows}
        slot_width = {}
        for pid, values in slots.items():
            row, col = (pid[-2], pid[-1]) if len(pid) > 1 else (0, pid[-1])
            width = sum(item["width"] for item in values) + macro_x * max(0, len(values) - 1)
            height = max(item["height"] for item in values)
            slot_width[pid] = width
            col_width[col] = max(col_width[col], width)
            row_height[row] = max(row_height[row], height)
        x_offsets, cursor = {}, 0.0
        for col in cols:
            x_offsets[col], cursor = cursor, cursor + col_width[col] + macro_x
        cluster_width = max(0.0, cursor - macro_x)
        y_offsets, cursor = {}, 0.0
        for row in rows:
            y_offsets[row], cursor = cursor, cursor + row_height[row] + macro_y
        cluster_height = max(0.0, cursor - macro_y)
        local = []
        for pid in pids:
            row, col = (pid[-2], pid[-1]) if len(pid) > 1 else (0, pid[-1])
            local_x = x_offsets[col]
            for member in sorted(slots[pid], key=lambda value: value["name"]):
                local.append({**member, "local_x": local_x, "local_y": y_offsets[row]})
                local_x += member["width"] + macro_x
        clusters[kernel] = {"width": cluster_width, "height": cluster_height, "members": local}

    weights = connection_weights(plan)
    kernel_names = sorted(clusters)
    slot_map = optimize_slots(kernel_names, weights, passes)
    max_cluster_w = max((item["width"] for item in clusters.values()), default=0.0)
    max_cluster_h = max((item["height"] for item in clusters.values()), default=0.0)
    slot_cols = max((slot[0] for slot in slot_map.values()), default=0) + 1
    slot_rows = max((slot[1] for slot in slot_map.values()), default=0) + 1
    kernel_region_w = slot_cols * max_cluster_w + max(0, slot_cols - 1) * kernel_x
    kernel_region_h = slot_rows * max_cluster_h + max(0, slot_rows - 1) * kernel_y

    srams = discover_srams(inputs / "srams", netlist)
    macro_area = sum(item["width"] * item["height"] for values in grouped.values() for item in values)
    sram_area = sum(item["width"] * item["height"] for item in srams)
    reported_cell_area = timing_area(inputs / "synthesis-reports")
    # DC hierarchical area includes linked hard macros when their Liberty/DB
    # models carry area. Subtract physical hard-macro area before applying the
    # standard-cell utilization target, then add macros back exactly once.
    standard_area = max(0.0, reported_cell_area - macro_area - sram_area)
    target_area = standard_area / density + macro_area + sram_area + 4 * keepout * keepout
    core_w = max(kernel_region_w + 2 * keepout, math.sqrt(target_area * aspect))
    core_h = max(kernel_region_h + 2 * keepout, math.sqrt(target_area / aspect))
    for _ in range(12):
        try:
            sram_placements, bands = edge_pack(srams, core_w, core_h, keepout, sram_spacing)
            break
        except ValueError:
            core_w *= 1.15
            core_h *= 1.15
    else:
        raise ValueError("unable to create legal SRAM perimeter")
    free = {
        "x": keepout + bands["left"] + (sram_spacing if bands["left"] else 0),
        "y": keepout + bands["bottom"] + (sram_spacing if bands["bottom"] else 0),
    }
    free["width"] = core_w - free["x"] - keepout - bands["right"] - (sram_spacing if bands["right"] else 0)
    free["height"] = core_h - free["y"] - keepout - bands["top"] - (sram_spacing if bands["top"] else 0)
    if kernel_region_w > free["width"] or kernel_region_h > free["height"]:
        grow_x = max(0.0, kernel_region_w - free["width"])
        grow_y = max(0.0, kernel_region_h - free["height"])
        core_w += grow_x
        core_h += grow_y
        free["width"] += grow_x
        free["height"] += grow_y
    origin_x = free["x"] + (free["width"] - kernel_region_w) / 2
    origin_y = free["y"] + (free["height"] - kernel_region_h) / 2
    pe_placements, cluster_records = [], []
    for kernel in kernel_names:
        col, row = slot_map[kernel]
        cluster_x = origin_x + col * (max_cluster_w + kernel_x)
        cluster_y = origin_y + row * (max_cluster_h + kernel_y)
        cluster = clusters[kernel]
        cluster_records.append({"kernel": kernel, "x": snap(cluster_x, grid), "y": snap(cluster_y, grid), **{key: cluster[key] for key in ["width", "height"]}})
        for member in cluster["members"]:
            pe_placements.append({
                **{key: value for key, value in member.items() if not key.startswith("local_")},
                "x": snap(cluster_x + member["local_x"], grid),
                "y": snap(cluster_y + member["local_y"], grid),
                "kernel": kernel,
            })
    planned_core_w = snap(core_w, grid)
    planned_core_h = snap(core_h, grid)
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
    intent = {
        "schema_version": 1, "stage": "physical_intent", "top_module": plan["top_module"],
        "core": {"width": planned_core_w, "height": planned_core_h, "usable_rectangle": free},
        "constraints": {"macro_separation_x": macro_x, "macro_separation_y": macro_y, "kernel_separation_x": kernel_x, "kernel_separation_y": kernel_y, "edge_keepout": keepout},
        "standard_cell_area_estimate": standard_area, "kernel_connections": [{"kernels": list(key), "weight": value} for key, value in sorted(weights.items())],
        "kernel_clusters": cluster_records, "placements": placements,
        "cluster_placement_policy": {
            "type": "partial_blockage",
            "maximum_density_percent": cluster_density,
            "region_count": len(cluster_records),
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
        "proc place_allo_physical_intent {} {", "  set core [dbGet top.fPlan.coreBox]", "  if {[llength $core] == 1} { set core [lindex $core 0] }", "  set llx [lindex $core 0]", "  set lly [lindex $core 1]", "  set placed 0",
    ]
    for item in placements:
        tcl.extend([
            f"  set matches [dbGet top.insts.name {{*{item['name']}}} -p]",
            f"  if {{[llength $matches] != 1 || [lindex $matches 0] eq \"0x0\"}} {{ error \"Expected exactly one macro instance {item['name']}\" }}",
            "  set actual_name [dbGet [lindex $matches 0].name]",
            f"  placeInstance $actual_name [expr {{$llx + {item['x']}}}] [expr {{$lly + {item['y']}}}] {item['orientation']}",
            "  incr placed",
        ])
    tcl.extend(["  setInstancePlacementStatus -allHardMacros -status fixed", "  return $placed", "}"])
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
        "# Limit local standard-cell density without discarding the useful",
        "# placement area between macros in each kernel cluster.",
        "proc create_allo_cluster_density_limits {} {",
        "  set core [dbGet top.fPlan.coreBox]",
        "  if {[llength $core] == 1} { set core [lindex $core 0] }",
        "  set llx [lindex $core 0]",
        "  set lly [lindex $core 1]",
        "  set blockage_count 0",
    ])
    for index, cluster in enumerate(cluster_records):
        tcl.extend([
            "  createPlaceBlockage -type partial "
            f"-density {cluster_density} -name allo_kernel_density_{index} "
            f"-box [list [expr {{$llx + {cluster['x']}}}] "
            f"[expr {{$lly + {cluster['y']}}}] "
            f"[expr {{$llx + {cluster['x']} + {cluster['width']}}}] "
            f"[expr {{$lly + {cluster['y']} + {cluster['height']}}}]]",
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
        f"Selective short-row cuts: {len(row_fragment_cuts)} (minimum retained width {min_row_width} um)\n"
        f"Kernel-cluster partial placement limits: {len(cluster_records)} at {cluster_density}% maximum density\n"
    )


if __name__ == "__main__":
    main()

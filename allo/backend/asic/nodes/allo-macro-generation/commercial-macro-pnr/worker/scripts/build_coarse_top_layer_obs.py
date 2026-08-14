#!/usr/bin/env python3
"""Replace a macro's full top-layer OBS with coarse post-route occupancy."""

from __future__ import annotations

import argparse
from collections import deque
import math
import re
import time
from pathlib import Path


RECT_RE = re.compile(
    r"\bRECT\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*;"
)


def normalize_net_name(name: str) -> str:
    """Remove Tcl list quoting from one Innovus database net name."""
    name = name.strip()
    if len(name) >= 2 and name[0] == "{" and name[-1] == "}":
        name = name[1:-1]
    return name.replace(r"\[", "[").replace(r"\]", "]")


def read_geometry(path: Path):
    die = None
    layer = None
    pitch = None
    rectangles = []
    for raw in path.read_text().splitlines():
        fields = raw.split("\t")
        if fields[0] == "DIE":
            die = tuple(map(float, fields[1:5]))
        elif fields[0] == "LAYER":
            layer = fields[1]
        elif fields[0] == "PITCH":
            pitch = float(fields[1])
        elif fields[0] == "RECT":
            if len(fields) == 7:
                rectangles.append((
                    fields[1], normalize_net_name(fields[2]),
                    tuple(map(float, fields[3:7])),
                ))
            elif len(fields) == 6:  # Backward-compatible diagnostic input.
                rectangles.append((fields[1], "", tuple(map(float, fields[2:6]))))
            else:
                raise ValueError(f"malformed geometry row: {raw}")
    if die is None or layer is None or pitch is None or pitch <= 0:
        raise ValueError(f"incomplete top-layer geometry header: {path}")
    return die, layer, pitch, rectangles


def top_layer_pin_rectangles(lef: str, layer: str):
    rectangles = []
    pin_name = None
    pin_use = "SIGNAL"
    current_layer = None
    for line in lef.splitlines():
        stripped = line.strip()
        if stripped.startswith("PIN "):
            pin_name = stripped.split()[1]
            pin_use = "SIGNAL"
            current_layer = None
        elif pin_name is not None and stripped.startswith("USE "):
            pin_use = stripped.split()[1].rstrip(";").upper()
        elif pin_name is not None and stripped.startswith("LAYER "):
            current_layer = stripped.split()[1]
        elif pin_name is not None and stripped == f"END {pin_name}":
            pin_name = None
            pin_use = "SIGNAL"
            current_layer = None
        elif (
            pin_name is not None
            and current_layer is not None
            and current_layer.lower() == layer.lower()
        ):
            match = RECT_RE.search(stripped)
            if match:
                rectangles.append((
                    pin_name,
                    pin_use,
                    tuple(map(float, match.groups())),
                ))
    return rectangles


def cells_for_rect(rect, die, cell, expansion, nx, ny):
    llx, lly, urx, ury = rect
    dllx, dlly, _, _ = die
    x0 = max(0, math.floor((llx - expansion - dllx) / cell))
    y0 = max(0, math.floor((lly - expansion - dlly) / cell))
    # Treat upper rectangle edges as half-open. A floor operation here marks
    # an extra cell whenever an edge lies exactly on a grid boundary.
    epsilon = 1.0e-9
    x1 = min(nx - 1, math.ceil((urx + expansion - dllx) / cell - epsilon) - 1)
    y1 = min(ny - 1, math.ceil((ury + expansion - dlly) / cell - epsilon) - 1)
    if x1 < x0 or y1 < y0:
        return ()
    return ((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))


def merge_cells(occupied):
    row_runs = {}
    for y in sorted({cell[1] for cell in occupied}):
        xs = sorted(x for x, row in occupied if row == y)
        runs = []
        if xs:
            start = previous = xs[0]
            for x in xs[1:]:
                if x != previous + 1:
                    runs.append((start, previous + 1))
                    start = x
                previous = x
            runs.append((start, previous + 1))
        row_runs[y] = runs

    active = {}
    merged = []
    for y in sorted(row_runs):
        present = set(row_runs[y])
        for run in list(active):
            if run not in present:
                y0, y1 = active.pop(run)
                merged.append((run[0], y0, run[1], y1))
        for run in present:
            if run in active and active[run][1] == y:
                active[run] = (active[run][0], y + 1)
            elif run not in active:
                active[run] = (y, y + 1)
    for run, (y0, y1) in active.items():
        merged.append((run[0], y0, run[1], y1))
    return merged


def rects_intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def boundary_path(starts, forbidden, nx, ny):
    """Find a shortest four-neighbor path from starts to the macro boundary."""
    queue = deque()
    parent = {}
    for cell in sorted(starts):
        if cell not in forbidden:
            queue.append(cell)
            parent[cell] = None
    while queue:
        cell = queue.popleft()
        x, y = cell
        if x == 0 or y == 0 or x == nx - 1 or y == ny - 1:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return tuple(reversed(path))
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            xx, yy = neighbor
            if 0 <= xx < nx and 0 <= yy < ny and neighbor not in forbidden and neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    return ()


def insert_obs(lef: str, layer: str, rectangles):
    obs_match = re.search(r"(?ms)^\s*OBS\s*$.*?^\s*END\s*$", lef)
    lines = [f"    LAYER {layer} ;"]
    lines.extend(
        f"      RECT {llx:.6f} {lly:.6f} {urx:.6f} {ury:.6f} ;"
        for llx, lly, urx, ury in rectangles
    )
    addition = "\n".join(lines) + "\n"
    if obs_match:
        end = lef.rfind("END", obs_match.start(), obs_match.end())
        return lef[:end] + addition + lef[end:]
    macro_end = list(re.finditer(r"(?m)^END\s+\S+\s*$", lef))
    if not macro_end:
        raise ValueError("cannot locate macro END in LEF")
    position = macro_end[-1].start()
    return lef[:position] + "  OBS\n" + addition + "  END\n" + lef[position:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lef", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-lef", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--grid-tracks", type=int, required=True)
    parser.add_argument("--pin-grid-tracks", type=int, required=True)
    parser.add_argument("--spacing-tracks", type=int, required=True)
    args = parser.parse_args()
    if (args.grid_tracks < 1 or args.pin_grid_tracks < 1 or
            args.spacing_tracks < 0 or
            args.grid_tracks % args.pin_grid_tracks != 0):
        raise ValueError("grid tracks must be positive and spacing tracks non-negative")

    started = time.monotonic()
    die, layer, pitch, geometry = read_geometry(args.geometry)
    lef = args.input_lef.read_text()
    cell = pitch * args.grid_tracks
    fine_cell = pitch * args.pin_grid_tracks
    expansion = pitch * args.spacing_tracks
    llx, lly, urx, ury = die
    nx = max(1, math.ceil((urx - llx) / cell))
    ny = max(1, math.ceil((ury - lly) / cell))
    occupied = set()
    for _kind, _net, rectangle in geometry:
        occupied.update(cells_for_rect(rectangle, die, cell, expansion, nx, ny))

    fine_nx = max(1, math.ceil((urx - llx) / fine_cell))
    fine_ny = max(1, math.ceil((ury - lly) / fine_cell))
    all_fine_occupied = set()
    for _kind, _net, rectangle in geometry:
        all_fine_occupied.update(
            cells_for_rect(rectangle, die, fine_cell, expansion, fine_nx, fine_ny)
        )

    # Refine only coarse cells touched by pins/access paths. A published pin may
    # open fine cells occupied by its own physical pin shape, but never cells
    # protected on behalf of another net. Its path to a boundary must otherwise
    # traverse empty, clearance-expanded fine cells.
    safe_open = set()
    access_paths = {}
    pin_records = top_layer_pin_rectangles(lef, layer)
    pg_pin_rectangles = 0
    signal_pin_rectangles = 0
    for pin_name, pin_use, pin_rect in pin_records:
        pin_fine = set(cells_for_rect(
            pin_rect, die, fine_cell, 0.0, fine_nx, fine_ny
        ))
        own_pin_geometry = set()
        unrelated = set()
        for _kind, net_name, rectangle in geometry:
            cells = set(cells_for_rect(
                rectangle, die, fine_cell, expansion, fine_nx, fine_ny
            ))
            if net_name == pin_name:
                own_pin_geometry.update(cells)
            else:
                unrelated.update(cells)
        if not (pin_fine & own_pin_geometry):
            raise ValueError(
                f"pin {pin_name} has no matching owning-net geometry on {layer}"
            )

        # POWER/GROUND pins are distributed PG shapes, not point-to-point
        # signal terminals. Expose their actual pin cells for special routing,
        # but do not fabricate a separate top-layer corridor for every stripe.
        if pin_use in {"POWER", "GROUND"}:
            pg_pin_rectangles += 1
            safe_open.update(pin_fine)
            continue

        signal_pin_rectangles += 1
        unsafe = pin_fine & unrelated
        if unsafe:
            raise ValueError(
                f"pin {pin_name} opening overlaps clearance for unrelated "
                f"top-layer geometry in {len(unsafe)} fine-grid cell(s)"
            )
        # Only the actual published pin cells may overlap physical metal. The
        # remainder of the access path must be empty after spacing expansion.
        forbidden = all_fine_occupied - (pin_fine & own_pin_geometry)
        path = boundary_path(pin_fine, forbidden, fine_nx, fine_ny)
        if not path:
            raise ValueError(
                f"pin {pin_name} has no clear top-layer path to a macro boundary"
            )
        safe_open.update(pin_fine)
        access_paths.setdefault(pin_name, set()).update(path)

    path_cells = set().union(*access_paths.values()) if access_paths else set()
    fine_per_coarse = args.grid_tracks // args.pin_grid_tracks
    refined_coarse = {
        (min(nx - 1, x // fine_per_coarse),
         min(ny - 1, y // fine_per_coarse))
        for x, y in safe_open | path_cells
    }

    # Emit coarse rectangles away from pins and fine rectangles in every
    # refined coarse cell. This keeps the LEF compact without coarse pin holes.
    coarse_kept = occupied - refined_coarse
    fine_kept = set()
    for x, y in all_fine_occupied - safe_open:
        coarse_parent = (
            min(nx - 1, x // fine_per_coarse),
            min(ny - 1, y // fine_per_coarse),
        )
        if coarse_parent in refined_coarse:
            fine_kept.add((x, y))

    grid_rectangles = []
    for x0, y0, x1, y1 in merge_cells(coarse_kept):
        grid_rectangles.append((
            llx + x0 * cell,
            lly + y0 * cell,
            min(urx, llx + x1 * cell),
            min(ury, lly + y1 * cell),
        ))
    for x0, y0, x1, y1 in merge_cells(fine_kept):
        grid_rectangles.append((
            llx + x0 * fine_cell,
            lly + y0 * fine_cell,
            min(urx, llx + x1 * fine_cell),
            min(ury, lly + y1 * fine_cell),
        ))
    args.output_lef.write_text(insert_obs(lef, layer, grid_rectangles))

    total_cells = nx * ny
    args.report.write_text(
        f"layer {layer}\n"
        f"pitch {pitch:.6f}\n"
        f"grid_tracks {args.grid_tracks}\n"
        f"grid_size {cell:.6f}\n"
        f"pin_grid_tracks {args.pin_grid_tracks}\n"
        f"pin_grid_size {fine_cell:.6f}\n"
        f"spacing_tracks {args.spacing_tracks}\n"
        f"input_geometry_rectangles {len(geometry)}\n"
        f"pin_rectangles {len(pin_records)}\n"
        f"signal_pin_rectangles {signal_pin_rectangles}\n"
        f"power_ground_pin_rectangles {pg_pin_rectangles}\n"
        f"pin_access_fine_cells {len(safe_open)}\n"
        f"access_path_fine_cells {len(path_cells)}\n"
        f"refined_coarse_cells {len(refined_coarse)}\n"
        f"occupied_cells {len(occupied)}\n"
        f"total_cells {total_cells}\n"
        f"occupied_fraction {len(occupied) / total_cells:.6f}\n"
        f"merged_obs_rectangles {len(grid_rectangles)}\n"
        f"runtime_seconds {time.monotonic() - started:.6f}\n"
    )


if __name__ == "__main__":
    main()

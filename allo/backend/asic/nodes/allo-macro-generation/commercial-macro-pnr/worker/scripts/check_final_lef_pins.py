#!/usr/bin/env python3
"""Diagnose final macro-LEF pin side/layer drift without failing PNR."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PLAN_RE = re.compile(r"^PLAN\s+(\S+)\s+side=([NSEW])\s+layer=(\S+)\s*$")
SIZE_RE = re.compile(r"^\s*SIZE\s+([0-9.eE+-]+)\s+BY\s+([0-9.eE+-]+)\s*;")
PIN_RE = re.compile(r"^\s*PIN\s+(\S+)\s*$")
LAYER_RE = re.compile(r"^\s*LAYER\s+(\S+)\s*;")
RECT_RE = re.compile(
    r"^\s*RECT\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*;"
)


def read_plan(path: Path) -> dict[str, tuple[str, str]]:
    plan: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        match = PLAN_RE.match(line)
        if match:
            pin, side, layer = match.groups()
            plan[pin] = (side, layer)
    return plan


def read_lef(path: Path) -> tuple[float, float, dict[str, list[tuple[str, tuple[float, ...]]]]]:
    width = height = None
    current_pin = None
    current_layer = None
    shapes: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    for line in path.read_text().splitlines():
        if width is None and (match := SIZE_RE.match(line)):
            width, height = map(float, match.groups())
            continue
        if match := PIN_RE.match(line):
            current_pin = match.group(1)
            current_layer = None
            shapes.setdefault(current_pin, [])
            continue
        if current_pin is not None and (match := LAYER_RE.match(line)):
            current_layer = match.group(1)
            continue
        if current_pin is not None and current_layer and (match := RECT_RE.match(line)):
            shapes[current_pin].append((current_layer, tuple(map(float, match.groups()))))
            continue
        if current_pin is not None and line.strip() == f"END {current_pin}":
            current_pin = None
            current_layer = None
    if width is None or height is None:
        raise ValueError(f"No MACRO SIZE found in {path}")
    return width, height, shapes


def rectangle_sides(rect: tuple[float, ...], width: float, height: float) -> set[str]:
    x1, y1, x2, y2 = rect
    tolerance = 1.0e-6
    result = set()
    if abs(x1) <= tolerance:
        result.add("W")
    if abs(x2 - width) <= tolerance:
        result.add("E")
    if abs(y1) <= tolerance:
        result.add("S")
    if abs(y2 - height) <= tolerance:
        result.add("N")
    return result


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_final_lef_pins.py PLAN_REPORT LEF OUTPUT_REPORT")
    plan_path, lef_path, output_path = map(Path, sys.argv[1:])
    plan = read_plan(plan_path)
    width, height, shapes = read_lef(lef_path)
    lines = []
    matched = missing = mismatched = 0
    for pin, (expected_side, expected_layer) in sorted(plan.items()):
        pin_shapes = shapes.get(pin, [])
        if not pin_shapes:
            missing += 1
            lines.append(f"MISSING pin={pin} expected_side={expected_side} expected_layer={expected_layer}")
            continue
        actual = [
            (layer, sorted(rectangle_sides(rect, width, height)), rect)
            for layer, rect in pin_shapes
        ]
        if any(layer == expected_layer and expected_side in sides for layer, sides, _ in actual):
            matched += 1
        else:
            mismatched += 1
            lines.append(
                f"MISMATCH pin={pin} expected_side={expected_side} "
                f"expected_layer={expected_layer} actual={actual}"
            )
    summary = (
        f"SUMMARY planned={len(plan)} matched={matched} missing={missing} "
        f"mismatched={mismatched} macro_size={width}x{height}"
    )
    output_path.write_text(summary + "\n" + "\n".join(lines) + ("\n" if lines else ""))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

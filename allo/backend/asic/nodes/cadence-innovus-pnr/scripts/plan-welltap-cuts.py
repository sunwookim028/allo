#!/usr/bin/env python3
"""Coalesce untapped row markers into a bounded set of Innovus row cuts."""

import re
import sys
from pathlib import Path

BOUNDS = re.compile(
    r"Bounds:\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)\s*"
    r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)"
)


def coalesce(boxes, tolerance=0.002):
    ordered = sorted(boxes, key=lambda box: (box[0], box[2], box[1], box[3]))
    merged = []
    for x1, y1, x2, y2 in ordered:
        if (merged and abs(merged[-1][0] - x1) <= tolerance
                and abs(merged[-1][2] - x2) <= tolerance
                and y1 <= merged[-1][3] + tolerance):
            merged[-1][3] = max(merged[-1][3], y2)
        else:
            merged.append([x1, y1, x2, y2])
    return [tuple(box) for box in merged]


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: plan-welltap-cuts.py INPUT_REPORT OUTPUT_TCL OUTPUT_REPORT")
    source, tcl_path, report_path = map(Path, sys.argv[1:])
    boxes = [tuple(map(float, match)) for match in BOUNDS.findall(source.read_text())]
    cuts = coalesce(boxes)
    tcl = ["set adaptive_welltap_cut_count 0"]
    report = [f"input_violation_boxes {len(boxes)}", f"coalesced_row_cuts {len(cuts)}"]
    for index, (x1, y1, x2, y2) in enumerate(cuts):
        tcl += [f"cutRow -area [list {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}]",
                "incr adaptive_welltap_cut_count"]
        report.append(f"cut {index} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}")
    tcl_path.write_text("\n".join(tcl) + "\n")
    report_path.write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

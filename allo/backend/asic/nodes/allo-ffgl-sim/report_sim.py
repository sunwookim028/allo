#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--vcd", type=Path, required=True)
    parser.add_argument("--macro-model-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_text = args.compile_log.read_text(errors="replace")
    simulation_text = args.simulation_log.read_text(errors="replace")
    pass_marker = "ALLO_TEST_PASS" in simulation_text
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "status": "passed" if pass_marker and args.vcd.is_file() and args.vcd.stat().st_size else "failed",
        "pass_marker_found": pass_marker,
        "vcd_bytes": args.vcd.stat().st_size if args.vcd.is_file() else 0,
        "macro_model_count": args.macro_model_count,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

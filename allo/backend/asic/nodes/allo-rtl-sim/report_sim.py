#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--vcd", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    simulation_text = args.simulation_log.read_text(errors="replace")
    pass_marker = "ALLO_TEST_PASS" in simulation_text
    vcd_bytes = args.vcd.stat().st_size if args.vcd.is_file() else 0
    contract = json.loads(args.contract.read_text())
    report = {
        "schema_version": 1,
        "mode": "rtl",
        "status": "passed" if pass_marker and vcd_bytes else "failed",
        "pass_marker_found": pass_marker,
        "vcd_bytes": vcd_bytes,
        "backend": contract.get("backend"),
        "protocol": contract.get("interface", {}).get("protocol"),
        "clock_period_ns": contract.get("clock_period_ns"),
        "top_module": contract.get("top_module"),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

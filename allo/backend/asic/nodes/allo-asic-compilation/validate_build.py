#!/usr/bin/env python3
"""Validate that Allo and the selected HLS backend produced an ASIC capture."""

import argparse
import json
import shutil
from pathlib import Path

from backend import find_rtl_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--rtl-output", required=True)
    args = parser.parse_args()

    project = Path(args.project)
    required = [
        "asic-manifest.json",
        "asic-manifest.tcl",
        "asic-manifest-final.json",
        "asic-manifest-final.tcl",
    ]
    missing = [name for name in required if not (project / name).is_file()]
    if missing:
        raise RuntimeError(f"missing Allo ASIC outputs: {missing}")
    if not (project / "asic-debug").is_dir():
        raise RuntimeError("missing Allo ASIC debug directory")

    pre = json.loads((project / "asic-manifest.json").read_text())
    final = json.loads((project / "asic-manifest-final.json").read_text())
    if pre.get("stage") != "pre_hls":
        raise RuntimeError(f"unexpected pre-HLS stage: {pre.get('stage')}")
    if final.get("stage") != "post_hls_enriched":
        raise RuntimeError(f"unexpected final stage: {final.get('stage')}")
    summary = final.get("summary", {})
    if summary.get("unmatched_or_ambiguous", 0) != 0:
        raise RuntimeError(
            "final manifest contains unmatched or ambiguous PE records: "
            f"{summary.get('unmatched_or_ambiguous')}"
        )
    if summary.get("unjoined_post_hls_records", 0) != 0:
        raise RuntimeError(
            "final manifest contains unjoined post-HLS records: "
            f"{summary.get('unjoined_post_hls_records')}"
        )
    if not final.get("pe_instances"):
        raise RuntimeError("final manifest contains no PE instances")

    rtl_dir = find_rtl_directory(args.backend, project)
    rtl_files = list(rtl_dir.glob("*.v")) + list(rtl_dir.glob("*.sv"))
    if not rtl_files:
        raise RuntimeError(f"no synthesized Verilog found in {rtl_dir}")
    rtl_output = Path(args.rtl_output)
    if rtl_output.exists():
        shutil.rmtree(rtl_output)
    shutil.copytree(rtl_dir, rtl_output)


if __name__ == "__main__":
    main()

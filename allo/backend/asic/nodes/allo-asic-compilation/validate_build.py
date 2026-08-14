#!/usr/bin/env python3
"""Validate that Allo and the selected HLS backend produced an ASIC capture."""

import argparse
import json
from pathlib import Path

from backend import publish_rtl_artifacts


def validate_workload_interface(final, workload, vectors):
    """Cross-check frozen calls against the compiler's realized top interface."""
    if not workload.get("enabled", False):
        return
    top_arguments = final.get("top_arguments", [])
    by_name = {argument["name"]: argument for argument in top_arguments}
    signature = workload.get("call_signature", [])
    if signature != [argument["name"] for argument in top_arguments]:
        raise RuntimeError("workload call_signature does not match top argument order")
    for call in workload.get("calls", []):
        expected = set(call.get("expected", {}))
        for name, vector in call.get("arguments", {}).items():
            argument = by_name[name]
            if vector.get("shape") != argument.get("shape"):
                raise RuntimeError(f"workload shape mismatch for {name}")
            protocol = argument.get("interface_protocol")
            packing = argument.get("packing") or {}
            interface = argument.get("interface") or {}
            if protocol in {None, "catapult_direct_array"}:
                if packing and vector.get("element_bits") != packing.get("element_bits"):
                    raise RuntimeError(f"workload element width mismatch for {name}")
                if packing and not packing.get("width_matches_shape", False):
                    raise RuntimeError(f"RTL packed width mismatch for {name}")
            elif protocol in {
                "catapult_sync_memory_read",
                "catapult_sync_memory_write",
                "catapult_sync_memory_readwrite",
            }:
                if vector.get("element_bits") != interface.get("element_bits"):
                    raise RuntimeError(f"workload element width mismatch for {name}")
                if interface.get("data_width") != interface.get("element_bits"):
                    raise RuntimeError(f"Catapult memory data width mismatch for {name}")
                if vector.get("element_count", 0) > interface.get("address_capacity", 0):
                    raise RuntimeError(f"Catapult memory address capacity mismatch for {name}")
                if interface.get("element_count") != vector.get("element_count"):
                    raise RuntimeError(f"Catapult memory depth mismatch for {name}")
            else:
                raise RuntimeError(f"unsupported Catapult argument protocol {protocol!r} for {name}")
            vector_path = vectors.parent / vector["file"]
            if not vector_path.is_file():
                raise RuntimeError(f"missing workload vector: {vector_path}")
            line_count = sum(1 for line in vector_path.read_text().splitlines() if line)
            if line_count != vector.get("element_count"):
                raise RuntimeError(f"workload vector length mismatch for {name}")
        for name in expected:
            direction = by_name[name].get("semantic_direction", by_name[name].get(
                "rtl_direction", by_name[name].get("direction")
            ))
            if direction not in {"output", "inout"}:
                raise RuntimeError(f"expected value targets non-output RTL argument {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--rtl-output", required=True)
    parser.add_argument("--workload-manifest")
    parser.add_argument("--workload-vectors")
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

    rtl_dir = publish_rtl_artifacts(args.backend, project, args.rtl_output)
    rtl_files = list(rtl_dir.glob("*.v")) + list(rtl_dir.glob("*.sv"))
    if not rtl_files:
        raise RuntimeError(f"no synthesized Verilog found in {rtl_dir}")
    if args.workload_manifest:
        workload = json.loads(Path(args.workload_manifest).read_text())
        validate_workload_interface(final, workload, Path(args.workload_vectors))


if __name__ == "__main__":
    main()

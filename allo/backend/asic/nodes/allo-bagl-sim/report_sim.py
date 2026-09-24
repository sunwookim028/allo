#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


CATEGORY_CODES = {
    "unmatched_timingcheck": {"SDFCOM_TANE"},
    "unmatched_iopath": {"SDFCOM_IANE"},
    "ignored_uphier_interconnect": {"SDFCOM_UHICD"},
}


def warning_codes(text):
    # Count the VCS warning header once, not both its header and explanatory line.
    return re.findall(r"(?im)^Warning-\[([A-Z0-9_-]*SDF[A-Z0-9_-]*|SDF[A-Z0-9_-]+)\]", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--simulation-log", type=Path, required=True)
    parser.add_argument("--vcd", type=Path, required=True)
    parser.add_argument("--warning-policy", choices=("report", "error"), required=True)
    parser.add_argument("--unmatched-timingcheck-policy", choices=("report", "error"), required=True)
    parser.add_argument("--unmatched-iopath-policy", choices=("report", "error"), required=True)
    parser.add_argument("--uphier-interconnect-policy", choices=("report", "error"), required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--sdf-requested", action="store_true")
    parser.add_argument("--macro-model-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_text = args.compile_log.read_text(errors="replace")
    simulation_text = args.simulation_log.read_text(errors="replace")
    combined = compile_text + "\n" + simulation_text

    codes = warning_codes(combined)
    counts = Counter(codes)
    category_counts = {}
    categorized = set()
    for category, category_codes in CATEGORY_CODES.items():
        category_counts[category] = sum(counts[code] for code in category_codes)
        categorized.update(category_codes)
    category_counts["other"] = sum(
        count for code, count in counts.items() if code not in categorized
    )
    sdf_errors = re.findall(
        r"(?im)^(?:Error-\[[^]]*SDF[^]]*\]|.*(?:SDF annotation error|SDF error).*)$",
        combined,
    )
    timing_violations = len(re.findall(r"(?im)^.*Timing violation in .*$", simulation_text))
    dut_unknown = bool(re.search(
        r"(?im)(?:became unknown|DUT.*unknown|AXI[^\n]*(?:address|data)[^\n]*[xz])",
        simulation_text,
    ))
    annotation_run_completed = bool(re.search(
        r"(?im)(?:SDF annotation completed|Doing SDF annotation.*Done)", combined
    ))
    manifest = json.loads(args.annotation_manifest.read_text())
    annotation_scopes = set(re.findall(
        r"(?im)^\s*\*+\s*Annotation scope:\s*(\S+)\s*$", compile_text
    ))
    top_annotation_completed = (
        annotation_run_completed and manifest["top_sdf"]["scope"] in annotation_scopes
    )
    macro_expected = int(manifest["macro_sdf_expected_count"])
    missing_macro = sum(
        not Path(item["sdf"]).is_file() for item in manifest["macro_annotations"]
    )
    macro_annotated = sum(
        item["scope"] in annotation_scopes for item in manifest["macro_annotations"]
    ) if annotation_run_completed and not sdf_errors else 0

    policies = {
        "unmatched_timingcheck": args.unmatched_timingcheck_policy,
        "unmatched_iopath": args.unmatched_iopath_policy,
        "ignored_uphier_interconnect": args.uphier_interconnect_policy,
        "other": args.warning_policy,
    }
    policy_failure = (args.warning_policy == "error" and bool(codes)) or any(
        category_counts[name] and policy == "error"
        for name, policy in policies.items()
    )
    pass_marker = "ALLO_TEST_PASS" in simulation_text
    vcd_bytes = args.vcd.stat().st_size if args.vcd.is_file() else 0
    passed = all((
        pass_marker,
        vcd_bytes > 0,
        not sdf_errors,
        top_annotation_completed,
        macro_annotated == macro_expected,
        missing_macro == 0,
        timing_violations == 0,
        not dut_unknown,
        not policy_failure,
    ))
    report = {
        "schema_version": 2,
        "mode": "bagl",
        "status": "passed" if passed else "failed",
        "pass_marker_found": pass_marker,
        "vcd_bytes": vcd_bytes,
        "sdf_requested": args.sdf_requested,
        "sdf_error_count": len(sdf_errors),
        "sdf_warning_count": len(codes),
        "sdf_warning_codes": dict(sorted(counts.items())),
        "sdf_warning_categories": category_counts,
        "sdf_warning_policy": args.warning_policy,
        "sdf_category_policies": policies,
        "top_sdf_annotation_completed": top_annotation_completed,
        "sdf_annotation_run_completed": annotation_run_completed,
        "macro_sdf_expected_count": macro_expected,
        "macro_sdf_annotated_count": macro_annotated,
        "missing_macro_sdf_count": missing_macro,
        "timing_violation_count": timing_violations,
        "dut_unknown_detected": dut_unknown,
        "macro_model_count": args.macro_model_count,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

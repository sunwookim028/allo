#!/usr/bin/env python3
"""Shared VCS runner and report generator for RTL, FFGL, and BAGL."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
CATEGORY_CODES = {
    "unmatched_timingcheck": {"SDFCOM_TANE"},
    "unmatched_iopath": {"SDFCOM_IANE"},
    "ignored_uphier_interconnect": {"SDFCOM_UHICD"},
}


def boolean_parameter(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be True or False")
    return value == "true"


def contract() -> dict:
    result = json.loads((INPUTS / "testbench-contract.json").read_text())
    for key in ("testbench_top", "dut_instance", "pass_marker", "failure_marker"):
        if key not in result:
            raise ValueError(f"testbench contract is missing {key}")
    return result


def model_sources(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [
        str(path) for path in sorted(root.rglob("*"), key=str)
        if path.is_file() and path.suffix.lower() in {".v", ".sv"}
    ]


def run_logged(command: list[str], log: Path, timeout: float | None = None) -> int:
    with log.open("w") as stream:
        invocation = "Running: " + shlex.join(command) + "\n"
        stream.write(invocation)
        stream.flush()
        sys.stdout.write(invocation)
        sys.stdout.flush()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        timed_out = threading.Event()

        def terminate_for_timeout() -> None:
            timed_out.set()
            process.kill()

        timer = threading.Timer(timeout, terminate_for_timeout) if timeout is not None else None
        if timer is not None:
            timer.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            process.wait()
        finally:
            if timer is not None:
                timer.cancel()
        if timed_out.is_set():
            message = f"\nSIMULATION_TIMEOUT after {timeout} seconds\n"
            stream.write(message)
            stream.flush()
            sys.stdout.write(message)
            sys.stdout.flush()
            return 124
        return process.returncode


def warning_codes(text: str) -> list[str]:
    return re.findall(r"(?im)^Warning-\[([A-Z0-9_-]*SDF[A-Z0-9_-]*|SDF[A-Z0-9_-]+)\]", text)


def is_text_vcd(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        header = stream.read(4096)
    if b"\0" in header:
        return False
    return any(marker in header for marker in (b"$date", b"$version", b"$timescale", b"$scope"))


def report(mode: str, tb: dict, compile_rc: int, simulation_rc: int) -> dict:
    compile_text = (OUTPUTS / "compile.log").read_text(errors="replace")
    simulation_text = (OUTPUTS / "simulation.log").read_text(errors="replace")
    pass_found = tb["pass_marker"] in simulation_text
    failure_found = bool(tb["failure_marker"] and tb["failure_marker"] in simulation_text)
    vcd = OUTPUTS / "run.vcd"
    vcd_bytes = vcd.stat().st_size if vcd.is_file() else 0
    vcd_valid = is_text_vcd(vcd)
    result = {
        "schema_version": 1,
        "mode": mode,
        "status": "failed",
        "compile_returncode": compile_rc,
        "simulation_returncode": simulation_rc,
        "simulation_timed_out": simulation_rc == 124,
        "pass_marker": tb["pass_marker"],
        "pass_marker_found": pass_found,
        "failure_marker": tb["failure_marker"],
        "failure_marker_found": failure_found,
        "vcd_bytes": vcd_bytes,
        "vcd_format_valid": vcd_valid,
        "testbench_top": tb["testbench_top"],
        "dut_instance": tb["dut_instance"],
    }
    passed = compile_rc == 0 and simulation_rc == 0 and pass_found and not failure_found and vcd_valid
    if mode == "bagl":
        combined = compile_text + "\n" + simulation_text
        codes = warning_codes(combined)
        counts = Counter(codes)
        categorized = set().union(*CATEGORY_CODES.values())
        categories = {
            name: sum(counts[code] for code in values)
            for name, values in CATEGORY_CODES.items()
        }
        categories["other"] = sum(count for code, count in counts.items() if code not in categorized)
        sdf_errors = re.findall(
            r"(?im)^(?:Error-\[[^]]*SDF[^]]*\]|.*(?:SDF annotation error|SDF error).*)$",
            combined,
        )
        violations = len(re.findall(r"(?im)^.*Timing violation in .*$", simulation_text))
        annotation_complete = bool(re.search(
            r"(?im)(?:SDF annotation completed|Doing SDF annotation.*Done)", combined
        ))
        scope = f'{tb["testbench_top"]}.{tb["dut_instance"]}'
        annotated_scopes = set(re.findall(
            r"(?im)^\s*\*+\s*Annotation scope:\s*(\S+)\s*$", compile_text
        ))
        top_annotated = annotation_complete and scope in annotated_scopes
        unknown = bool(re.search(r"(?im)(?:became unknown|DUT.*unknown)", simulation_text))
        policies = {
            "unmatched_timingcheck": os.environ.get("sdf_unmatched_timingcheck_policy", "report"),
            "unmatched_iopath": os.environ.get("sdf_unmatched_iopath_policy", "report"),
            "ignored_uphier_interconnect": os.environ.get("sdf_uphier_interconnect_policy", "report"),
            "other": os.environ.get("sdf_warning_policy", "report"),
        }
        policy_failure = any(categories[name] and policy == "error" for name, policy in policies.items())
        passed = passed and not sdf_errors and top_annotated and violations == 0 and not unknown and not policy_failure
        result.update({
            "sdf_requested": True,
            "sdf_corner": os.environ.get("sdf_corner", "typ"),
            "sdf_error_count": len(sdf_errors),
            "sdf_warning_count": len(codes),
            "sdf_warning_codes": dict(sorted(counts.items())),
            "sdf_warning_categories": categories,
            "sdf_category_policies": policies,
            "top_sdf_annotation_completed": top_annotated,
            "timing_violation_count": violations,
            "dut_unknown_detected": unknown,
        })
    result["status"] = "passed" if passed else "failed"
    return result


def main() -> None:
    mode = os.environ.get("simulation_mode", "rtl").lower()
    if mode not in {"rtl", "ffgl", "bagl"}:
        raise ValueError("simulation_mode must be rtl, ffgl, or bagl")
    tb = contract()
    timeout = int(tb.get("simulation_timeout_seconds", 3600))
    OUTPUTS.mkdir(exist_ok=True)
    # Preserve legacy/user testbenches that open runtime collateral as
    # inputs/<basename>, while the package remains the source of truth.
    package_files = INPUTS / "testbench" / "files"
    if package_files.is_dir():
        for packaged in package_files.rglob("*"):
            if packaged.is_file():
                alias = INPUTS / packaged.name
                if not alias.exists() and not alias.is_symlink():
                    alias.symlink_to(packaged.resolve())
    for name in ("run.vcd", "compile.log", "simulation.log", "simulation-report.json"):
        (OUTPUTS / name).unlink(missing_ok=True)

    # Preserve debug visibility for user testbenches and gate-level dumping.
    command = ["vcs", "-full64", "-sverilog", "-debug_access+all",
               "-override_timescale=1ns/1ps",
               "-top", tb["testbench_top"], "-o", "simv"]
    if boolean_parameter("xprop_enabled", True):
        command.append("-xprop=tmerge")
    if mode in {"ffgl", "bagl"}:
        command.extend(model_sources(INPUTS / "adk"))
        command.extend(model_sources(INPUTS / "srams"))
    design = INPUTS / ("design.vcs.v" if mode == "bagl" else "design.v")
    rtl_filelist = INPUTS / "rtl-sources.f"
    if mode == "rtl" and rtl_filelist.is_file():
        command.extend(["-f", str(rtl_filelist)])
    else:
        command.append(str(design))
    command.extend(["-f", "inputs/testbench/testbench.f"])
    compile_args = INPUTS / "testbench" / "testbench-compile.args"
    if compile_args.is_file():
        command.extend(shlex.split(compile_args.read_text(), comments=True))
    if mode == "ffgl":
        command.extend(["+delay_mode_zero", "+define+TETRAMAX"])
    if mode == "bagl":
        corner = os.environ.get("sdf_corner", "typ")
        if corner not in {"typ", "min", "max"}:
            raise ValueError("sdf_corner must be typ, min, or max")
        scope = f'{tb["testbench_top"]}.{tb["dut_instance"]}'
        command.extend(["+neg_tchk", "+no_notifier",
                        "+sdfverbose", "+define+NTC", "+define+TETRAMAX",
                        "-sdf", f"{corner}:{scope}:inputs/design.sdf"])

    compile_rc = run_logged(command, OUTPUTS / "compile.log")
    simulation_rc = 1
    if compile_rc == 0:
        runtime = ["./simv"]
        args_file = INPUTS / "testbench" / "testbench-runtime.args"
        if args_file.is_file():
            runtime.extend(shlex.split(args_file.read_text(), comments=True))
        if boolean_parameter("waveform", True):
            runtime.append("+ASIC_DUMP_VCD")
        simulation_rc = run_logged(runtime, OUTPUTS / "simulation.log", timeout)
    else:
        (OUTPUTS / "simulation.log").write_text("Simulation skipped because VCS compilation failed\n")

    result = report(mode, tb, compile_rc, simulation_rc)
    (OUTPUTS / "simulation-report.json").write_text(json.dumps(result, indent=2) + "\n")
    report_only = mode == "bagl" and os.environ.get("bagl_failure_policy", "error") == "report"
    if result["status"] != "passed" and not report_only:
        raise SystemExit("simulation report status is failed")


if __name__ == "__main__":
    main()

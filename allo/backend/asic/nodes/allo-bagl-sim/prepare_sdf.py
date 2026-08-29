#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"


def _load(path):
    return json.loads(path.read_text())


def _source_latencies(sdc_text):
    values = []
    for line in sdc_text.splitlines():
        if "set_clock_latency" not in line or "-source" not in line:
            continue
        match = re.search(r"(?:^|\s)([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*(?:\[|$)", line)
        if match:
            values.append(float(match.group(1)))
    return values


def _find_macro_instances(netlist, expected_by_instance):
    """Find all expected macro instances with one pass over the netlist."""
    if not expected_by_instance:
        return set()
    instance_names = "|".join(
        re.escape(name)
        for name in sorted(expected_by_instance, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_$])(?P<module>{IDENT})\s+"
        rf"(?P<instance>(?:{instance_names}))\s*\("
    )
    return {
        (match.group("module"), match.group("instance"))
        for match in pattern.finditer(netlist)
    }


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    registry = _load(INPUTS / "macro-registry" / "index.json")
    collateral = _load(INPUTS / "macro-collateral.json")
    netlist = (INPUTS / "design.vcs.v").read_text(errors="replace")

    by_module = {item["top_module"]: item for item in registry.get("macros", [])}
    dut_scope = (
        f'{os.environ.get("testbench_name", "allo_generated_testbench")}.'
        f'{os.environ.get("dut_name", "dut")}'
    )
    annotations = []
    seen_instances = set()
    expected_by_instance = {}
    for item in collateral.get("rewritten_instances", []):
        module = item["canonical_module"]
        instance = item["stable_instance_name"]
        if instance in seen_instances:
            raise ValueError(f"duplicate stable macro instance name: {instance}")
        seen_instances.add(instance)
        if module not in by_module:
            raise ValueError(f"macro module {module!r} is absent from registry index")
        sdf_view = by_module[module].get("views", {}).get("sdf")
        if not sdf_view or not sdf_view.get("path"):
            raise ValueError(f"macro module {module!r} has no published SDF view")
        relative_sdf = Path(sdf_view["path"])
        sdf_path = INPUTS / "macro-registry" / relative_sdf
        if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
            raise ValueError(f"missing or empty macro SDF: {sdf_path}")
        expected_by_instance[instance] = module
        annotations.append({
            "module": module,
            "instance": instance,
            "scope_suffix": instance,
            "scope": f"{dut_scope}.{instance}",
            "sdf": str(Path("inputs/macro-registry") / relative_sdf),
        })

    found_instances = _find_macro_instances(netlist, expected_by_instance)
    for instance, module in expected_by_instance.items():
        if (module, instance) not in found_instances:
            raise ValueError(
                f"cannot find macro instance {module} {instance} in design.vcs.v"
            )

    expected = int(collateral.get("macro_instance_count", len(annotations)))
    if expected != len(annotations):
        raise ValueError(
            f"macro collateral expects {expected} instances but describes {len(annotations)}"
        )
    manifest = {
        "schema_version": 1,
        "top_sdf": {"scope_suffix": "", "scope": dut_scope, "sdf": "inputs/design.sdf"},
        "macro_sdf_expected_count": expected,
        "macro_annotations": annotations,
    }
    (OUTPUTS / "sdf-annotation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (OUTPUTS / "sdf-annotations.tsv").write_text(
        "".join(f'{item["scope"]}\t{item["sdf"]}\n' for item in annotations)
    )

    latencies = _source_latencies((INPUTS / "design.pt.sdc").read_text(errors="replace"))
    compensation = max((abs(value) for value in latencies), default=0.0)
    contract = _load(INPUTS / "testbench-contract.json")
    contract_clock_period = float(contract["clock_period_ns"])
    clock_period = float(os.environ.get("clock_period", contract_clock_period))
    if abs(clock_period - contract_clock_period) > 1e-12:
        raise ValueError(
            "BAGL clock_period does not match testbench contract: "
            f"{clock_period} ns != {contract_clock_period} ns"
        )
    if clock_period <= 0.0:
        raise ValueError(f"clock_period must be positive, got {clock_period}")
    clock_half_period = clock_period / 2.0
    input_delay = float(os.environ.get("bagl_input_delay_ns", "0.025"))
    output_delay = float(os.environ.get("bagl_output_delay_ns", "0.025"))
    reset_cycles = int(os.environ.get("bagl_num_reset_cycles", "8"))
    bfm_drive_delay = clock_half_period + compensation + input_delay
    timing = {
        "schema_version": 1,
        "clock_source_latencies_ns": latencies,
        "clock_period_ns": clock_period,
        "clock_half_period_ns": clock_half_period,
        "clock_compensation_ns": compensation,
        "input_delay_ns": input_delay,
        "output_delay_ns": output_delay,
        "num_reset_cycles": reset_cycles,
        "bfm_drive_delay_ns": bfm_drive_delay,
    }
    (OUTPUTS / "timing-config.json").write_text(json.dumps(timing, indent=2) + "\n")
    # A single whitespace-delimited record keeps the shell launcher independent
    # of JSON parsing and nested command quoting. All values remain derived from
    # the post-route SDC and configurable node parameters.
    (OUTPUTS / "timing-config.values").write_text(
        f"{compensation:.12g} {input_delay:.12g} {output_delay:.12g} "
        f"{reset_cycles} {bfm_drive_delay:.12g}\n"
    )


if __name__ == "__main__":
    main()

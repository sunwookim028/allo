"""Backend-specific configuration and artifact discovery for the Allo node."""

import ast
import json
import shutil
from pathlib import Path


def parse_backend_options(options_text):
    """Accept shell-safe key=value options, JSON, or a Python dictionary."""
    options_text = (options_text or "").strip()
    if not options_text:
        return {}
    if "=" in options_text and not options_text.startswith("{"):
        options = {}
        for item in options_text.split(","):
            if "=" not in item:
                raise ValueError(
                    f"invalid backend option {item!r}; expected key=value"
                )
            key, value = (part.strip() for part in item.split("=", 1))
            if not key or not value:
                raise ValueError(
                    f"invalid backend option {item!r}; expected key=value"
                )
            options[key] = value
        return options
    try:
        options = json.loads(options_text)
    except json.JSONDecodeError:
        try:
            options = ast.literal_eval(options_text)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                "backend_options must be a JSON or Python dictionary literal: "
                f"{error}"
            ) from error
    if not isinstance(options, dict):
        raise TypeError("backend_options must decode to a dictionary")
    return options


def configure_backend(backend, clock_period, options_text):
    """Return target, Allo configuration, and descriptive backend metadata."""
    if clock_period <= 0:
        raise ValueError("clock period must be greater than zero")
    options = parse_backend_options(options_text)

    if backend == "vitis":
        device = options.get("device", "u280")
        configs = {
            "frequency": 1000.0 / clock_period,
            "device": device,
            "asic_manifest": {
                "enabled": True,
                "path": "asic-manifest.json",
                "debug_artifacts": True,
                "debug_dir": "asic-debug",
            },
        }
        return "vitis_hls", configs, {"device": device, "rtl_stage": "syn"}
    if backend == "catapult":
        device = options.get("device", "nangate-45nm_beh")
        configs = {
            "frequency": 1000.0 / clock_period,
            "device": device,
            "preserve_hierarchy": options.get("preserve_hierarchy", True),
            "asic_manifest": {
                "enabled": True,
                "path": "asic-manifest.json",
                "debug_artifacts": True,
                "debug_dir": "asic-debug",
            },
        }
        if "sub_funcs" in options:
            configs["sub_funcs"] = options["sub_funcs"]
        return "catapult", configs, {"device": device, "rtl_stage": "rtl"}
    else:
        raise ValueError(
            f"unsupported Allo backend {backend!r}; supported: vitis, catapult"
        )


def find_rtl_directory(backend, project):
    """Locate synthesized RTL without exposing backend layout as parameters."""
    project = Path(project)
    if backend == "vitis":
        candidates = [
            path for path in sorted(project.glob("*.prj/*/syn/verilog"))
            if path.is_dir()
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "expected exactly one Vitis *.prj/*/syn/verilog directory "
                f"under {project}, found {len(candidates)}: {candidates}"
            )
        return candidates[0]
    if backend == "catapult":
        candidates = sorted(
            rtl.parent
            for rtl in project.glob("Catapult*/*.v1/rtl.v")
            if rtl.is_file()
        )
        if len(candidates) != 1:
            raise RuntimeError(
                "expected exactly one Catapult Catapult*/*.v1/rtl.v directory "
                f"under {project}, found {len(candidates)}: {candidates}"
            )
        return candidates[0]
    else:
        raise ValueError(
            f"unsupported Allo backend {backend!r}; supported: vitis, catapult"
        )


def publish_rtl_artifacts(backend, project, output):
    """Publish the stable RTL contract, excluding backend work databases."""
    source = find_rtl_directory(backend, project)
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    if backend == "vitis":
        shutil.copytree(source, output)
        return source

    output.mkdir(parents=True)
    required = ("concat_rtl.v",)
    optional = (
        "cycle.rpt",
        "rtl.rpt",
        "concat_rtl.v.dc.sdc",
    )
    for name in required:
        path = source / name
        if not path.is_file():
            raise RuntimeError(f"missing required Catapult RTL artifact: {path}")
        shutil.copy2(path, output / name)
    for name in optional:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)

    scverify_names = (
        "dut_v_ports.map",
        "mc_dut_wrapper.h",
        "mc_testbench.cpp",
        "mc_testbench.h",
        "scverify_top.cpp",
    )
    scverify_source = source / "scverify"
    selected = [name for name in scverify_names if (scverify_source / name).is_file()]
    if selected:
        scverify_output = output / "scverify"
        scverify_output.mkdir()
        for name in selected:
            shutil.copy2(scverify_source / name, scverify_output / name)
    return source

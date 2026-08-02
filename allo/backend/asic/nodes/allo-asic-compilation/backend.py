"""Backend-specific configuration and artifact discovery for the Allo node."""

import json
from pathlib import Path


def configure_backend(backend, clock_period, options_text):
    """Return target, Allo configuration, and descriptive backend metadata."""
    if clock_period <= 0:
        raise ValueError("clock period must be greater than zero")
    try:
        options = json.loads(options_text or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"backend_options must be valid JSON: {error}") from error
    if not isinstance(options, dict):
        raise TypeError("backend_options must decode to a JSON object")

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
    else:
        raise ValueError(
            f"unsupported Allo backend {backend!r}; currently supported: vitis"
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
    else:
        raise ValueError(
            f"unsupported Allo backend {backend!r}; currently supported: vitis"
        )

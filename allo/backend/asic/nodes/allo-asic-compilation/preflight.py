#!/usr/bin/env python3
"""Runtime preconditions whose values come from mflowgen parameters."""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def validate_clock_periods(clock_period: float, macro_clock_period: float) -> None:
    """Require macros to meet a clock at least as fast as the full chip."""
    if macro_clock_period <= 0 or clock_period <= 0:
        raise RuntimeError("clock periods must be positive")
    if clock_period < macro_clock_period:
        raise RuntimeError(
            "chip clock_period must be greater than or equal to "
            f"macro_clock_period; got {clock_period} < {macro_clock_period} ns"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--construct-path", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--setup-script", required=True)
    parser.add_argument("--clock-period", required=True, type=float)
    parser.add_argument("--macro-clock-period", required=True, type=float)
    args = parser.parse_args()

    validate_clock_periods(args.clock_period, args.macro_clock_period)

    setup_script = Path(args.setup_script).expanduser()
    if not setup_script.is_file():
        raise RuntimeError(f"Allo LLVM setup script does not exist: {setup_script}")

    selected_python = shutil.which(str(Path(args.python_bin).expanduser()))
    if selected_python is None:
        raise RuntimeError(f"Python executable not found: {args.python_bin}")
    requested_python = Path(selected_python).resolve()
    running_python = Path(sys.executable).resolve()
    if requested_python != running_python:
        raise RuntimeError(
            f"requested Python {requested_python}, but preflight is running "
            f"with {running_python}"
        )

    try:
        import allo  # pylint: disable=import-outside-toplevel,unused-import
    except Exception as error:
        raise RuntimeError(
            f"Allo is not importable with {running_python}: {error}"
        ) from error

    design = Path(args.design).expanduser()
    if not design.is_absolute():
        design = Path(args.construct_path).resolve().parent / design
    if not design.is_file():
        raise RuntimeError(f"Allo design does not exist: {design}")
    spec = importlib.util.spec_from_file_location("allo_preflight_design", design)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Allo design module: {design}")

    if args.mode != "csyn":
        raise RuntimeError(
            f"unsupported Allo build mode {args.mode!r}; currently supported: csyn"
        )

    if args.backend == "vitis":
        if shutil.which("vitis_hls") is None:
            raise RuntimeError(
                "Vitis backend selected, but vitis_hls is not available on PATH"
            )
    elif args.backend in {"catapult", "systemc"}:
        if shutil.which("catapult") is None:
            raise RuntimeError(
                f"{args.backend} backend selected, but catapult is not available on PATH"
            )
        if args.backend == "systemc":
            import os  # pylint: disable=import-outside-toplevel
            systemc_home_value = os.environ.get("SYSTEMC_HOME")
            if not systemc_home_value or not Path(systemc_home_value).is_dir():
                raise RuntimeError(
                    "SystemC backend selected, but SYSTEMC_HOME is unset or invalid"
                )
    else:
        raise RuntimeError(
            f"unsupported Allo backend {args.backend!r}; supported: vitis, catapult, systemc"
        )

    print(f"Python: {running_python}")
    print(f"Allo LLVM setup: {setup_script.resolve()}")
    print(f"Allo design: {design.resolve()}")
    print(f"Backend: {args.backend}")
    print(f"Mode: {args.mode}")
    print(f"Chip clock period: {args.clock_period} ns")
    print(f"Macro/HLS clock period: {args.macro_clock_period} ns")


if __name__ == "__main__":
    main()

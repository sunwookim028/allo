#!/usr/bin/env python3
"""Runtime preconditions whose values come from mflowgen parameters."""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--construct-path", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

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
    else:
        raise RuntimeError(
            f"unsupported Allo backend {args.backend!r}; currently supported: vitis"
        )

    print(f"Python: {running_python}")
    print(f"Allo design: {design.resolve()}")
    print(f"Backend: {args.backend}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()

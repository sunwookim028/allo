#!/usr/bin/env python3
"""Load a parameterized Allo design module and invoke its build entrypoint."""

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

from backend import configure_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--entrypoint", default="build")
    parser.add_argument("--project", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--clock-period", type=float, required=True)
    parser.add_argument("--backend-options", default="{}")
    args = parser.parse_args()

    design_path = Path(args.design).resolve()
    spec = importlib.util.spec_from_file_location("mflowgen_allo_design", design_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Allo design: {design_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entrypoint = getattr(module, args.entrypoint, None)
    if entrypoint is None or not callable(entrypoint):
        raise RuntimeError(
            f"{design_path} must define callable {args.entrypoint}(...)"
        )

    target, configs, backend_metadata = configure_backend(
        args.backend, args.clock_period, args.backend_options
    )
    kwargs = {
        "project": Path(args.project).resolve(),
        "target": target,
        "mode": args.mode,
        "configs": configs,
    }
    signature = inspect.signature(entrypoint)
    missing = [name for name in kwargs if name not in signature.parameters]
    if missing:
        raise TypeError(
            f"{args.entrypoint} must accept project, target, mode, and configs; "
            f"missing {missing}"
        )

    entrypoint(**kwargs)
    print(
        json.dumps(
            {
                "design": str(design_path),
                "backend": args.backend,
                "backend_metadata": backend_metadata,
                **kwargs,
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

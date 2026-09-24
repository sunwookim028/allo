#!/usr/bin/env python3
"""Freeze a design's Python workload while its compilation environment is active."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np


def _bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_design(path):
    spec = importlib.util.spec_from_file_location("mflowgen_allo_workload", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Allo design: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_hex(array, path):
    array = np.ascontiguousarray(np.asarray(array))
    if array.dtype.kind not in "biuf":
        raise TypeError(f"unsupported workload dtype {array.dtype}")
    item_bytes = array.dtype.itemsize
    byte_rows = array.reshape(-1).view(np.uint8).reshape(-1, item_bytes)
    with path.open("w", encoding="utf-8") as stream:
        for row in byte_rows:
            # SystemVerilog hexadecimal literals are written most-significant
            # byte first; NumPy values on this host are little-endian.
            stream.write("".join(f"{int(byte):02x}" for byte in row[::-1]) + "\n")


def _serialize_array(value, path, relative_path):
    array = np.asarray(value)
    _write_hex(array, path)
    return {
        "dtype": array.dtype.name,
        "element_bits": array.dtype.itemsize * 8,
        "shape": list(array.shape),
        "element_count": int(array.size),
        "file": relative_path.as_posix(),
    }


def export_workload(design, enabled, factory, top_function, manifest_path, vectors):
    if vectors.exists():
        shutil.rmtree(vectors)
    vectors.mkdir(parents=True)

    if not enabled:
        manifest = {
            "schema_version": 1,
            "enabled": False,
            "calls": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    module = _load_design(design)
    factory_fn = getattr(module, factory, None)
    if not callable(factory_fn):
        raise RuntimeError(f"{design} must define callable {factory}()")
    workload = factory_fn()
    if not isinstance(workload, dict):
        raise TypeError(f"{factory}() must return a dictionary")

    signature = workload.get("call_signature")
    calls = workload.get("calls")
    if not isinstance(signature, list) or not all(isinstance(x, str) for x in signature):
        raise TypeError("workload call_signature must be a list of argument names")
    if not isinstance(calls, list) or not calls:
        raise TypeError("workload calls must be a nonempty list")
    if len(signature) != len(set(signature)):
        raise ValueError("workload call_signature contains duplicate names")

    serialized_calls = []
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TypeError(f"call {call_index} must be a dictionary")
        arguments = call.get("arguments")
        expected = call.get("expected", {})
        if not isinstance(arguments, dict) or set(arguments) != set(signature):
            raise ValueError(
                f"call {call_index} arguments must exactly match call_signature {signature}"
            )
        if not isinstance(expected, dict) or not set(expected).issubset(signature):
            raise ValueError(f"call {call_index} has invalid expected outputs")

        call_dir = vectors / f"call_{call_index:03d}"
        call_dir.mkdir()
        result = {
            "name": call.get("name", f"call_{call_index:03d}"),
            "reset_before": bool(call.get("reset_before", call_index == 0)),
            "arguments": {},
            "expected": {},
            "comparison": call.get("comparison", {}),
        }
        for name in signature:
            rel = Path("workload-vectors") / call_dir.name / f"{name}.initial.hex"
            result["arguments"][name] = _serialize_array(
                arguments[name], call_dir / f"{name}.initial.hex", rel
            )
        for name, value in expected.items():
            rel = Path("workload-vectors") / call_dir.name / f"{name}.expected.hex"
            result["expected"][name] = _serialize_array(
                value, call_dir / f"{name}.expected.hex", rel
            )
        serialized_calls.append(result)

    manifest = {
        "schema_version": 1,
        "enabled": True,
        "design": str(design.resolve()),
        "factory": factory,
        "top_function": workload.get("top_function", top_function),
        "call_signature": signature,
        "default_timeout_cycles": int(workload.get("default_timeout_cycles", 100000)),
        "calls": serialized_calls,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--enabled", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--top-function", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-vectors", type=Path, required=True)
    args = parser.parse_args()
    export_workload(
        args.design.resolve(),
        _bool(args.enabled),
        args.factory,
        args.top_function,
        args.output_manifest,
        args.output_vectors,
    )


if __name__ == "__main__":
    main()

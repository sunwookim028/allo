"""Contract tests for compiler-manifest/workload validation."""

import json
from pathlib import Path

import pytest

from validate_build import validate_workload_interface


def _contract(tmp_path: Path):
    vectors = tmp_path / "workload-vectors"
    call = vectors / "call_000"
    call.mkdir(parents=True)
    (call / "a.initial.hex").write_text("0001\n0002\n")
    (call / "b.initial.hex").write_text("0000\n0000\n")
    final = {
        "top_arguments": [
            {
                "name": "a",
                "shape": [2],
                "rtl_direction": "input",
                "packing": {
                    "element_bits": 16,
                    "width_matches_shape": True,
                },
            },
            {
                "name": "b",
                "shape": [2],
                "rtl_direction": "output",
                "packing": {
                    "element_bits": 16,
                    "width_matches_shape": True,
                },
            },
        ]
    }
    vector = lambda name: {
        "shape": [2],
        "element_bits": 16,
        "element_count": 2,
        "file": f"workload-vectors/call_000/{name}.initial.hex",
    }
    workload = {
        "enabled": True,
        "call_signature": ["a", "b"],
        "calls": [
            {
                "arguments": {"a": vector("a"), "b": vector("b")},
                "expected": {"b": {}},
            }
        ],
    }
    return final, workload, vectors


def test_validate_workload_interface_accepts_matching_contract(tmp_path: Path):
    final, workload, vectors = _contract(tmp_path)
    validate_workload_interface(final, workload, vectors)


def test_validate_workload_interface_rejects_order_mismatch(tmp_path: Path):
    final, workload, vectors = _contract(tmp_path)
    workload["call_signature"] = ["b", "a"]
    with pytest.raises(RuntimeError, match="argument order"):
        validate_workload_interface(final, workload, vectors)


def test_validate_workload_interface_rejects_input_as_expected(tmp_path: Path):
    final, workload, vectors = _contract(tmp_path)
    workload["calls"][0]["expected"] = {"a": {}}
    with pytest.raises(RuntimeError, match="non-output"):
        validate_workload_interface(final, workload, vectors)


def test_validate_workload_interface_accepts_catapult_memory(tmp_path: Path):
    final, workload, vectors = _contract(tmp_path)
    for argument in final["top_arguments"]:
        argument["interface_protocol"] = (
            "catapult_sync_memory_read" if argument["name"] == "a"
            else "catapult_sync_memory_write"
        )
        argument["semantic_direction"] = argument["rtl_direction"]
        argument["packing"] = None
        argument["interface"] = {
            "element_bits": 16, "data_width": 16,
            "element_count": 2, "address_capacity": 4,
        }
    validate_workload_interface(final, workload, vectors)


def test_validate_workload_interface_rejects_insufficient_memory_address_width(tmp_path: Path):
    final, workload, vectors = _contract(tmp_path)
    argument = final["top_arguments"][0]
    argument.update({
        "interface_protocol": "catapult_sync_memory_read",
        "packing": None,
        "interface": {
            "element_bits": 16, "data_width": 16,
            "element_count": 2, "address_capacity": 1,
        },
    })
    with pytest.raises(RuntimeError, match="address capacity"):
        validate_workload_interface(final, workload, vectors)

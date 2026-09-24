"""ASIC-flow tests for Allo's Catapult manifest extractor."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from allo.dataflow import _build_manifest_top_arguments


ALLO_HOME = Path(os.environ.get("ALLO_HOME", Path.home() / "allo"))
SCRIPT = ALLO_HOME / "scripts" / "extract_catapult_pe_manifest.py"
SPEC = importlib.util.spec_from_file_location("catapult_manifest_for_flow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pre_hls_manifest_uses_realized_dataflow_argument_order():
    source_arguments = [
        SimpleNamespace(name="A", top_name="A", shape=(4,), dtype="i16"),
        SimpleNamespace(name="B", top_name="B", shape=(8,), dtype="i32"),
        SimpleNamespace(name="C", top_name="C", shape=(2,), dtype="f16"),
    ]

    arguments = _build_manifest_top_arguments(
        source_arguments,
        ["A", "C", "B"],
        ["in", "out", "both"],
        {"in": "input", "out": "output", "both": "inout"},
    )

    assert [argument["name"] for argument in arguments] == ["A", "C", "B"]
    assert [argument["shape"] for argument in arguments] == [[4], [2], [8]]
    assert [argument["type"] for argument in arguments] == ["i16", "f16", "i32"]
    assert [argument["direction"] for argument in arguments] == [
        "input",
        "output",
        "inout",
    ]


def test_catapult_direct_array_contract_is_positional(tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text(
        "void top(uint16_t v7[2], uint16_t v8[2]) {}\n", encoding="utf-8"
    )
    rtl = tmp_path / "concat_rtl.v"
    rtl.write_text(
        "module top(v7_rsc_dat, v7_triosy_lz, v8_rsc_dat, v8_triosy_lz);\n"
        "input [31:0] v7_rsc_dat; output v7_triosy_lz;\n"
        "output [31:0] v8_rsc_dat; output v8_triosy_lz;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    pre = {
        "schema_version": 2,
        "stage": "pre_hls",
        "top": "top",
        "top_arguments": [
            {
                "ordinal": 0,
                "name": "A",
                "shape": [2],
                "type": "ui16",
                "direction": "input",
            },
            {
                "ordinal": 1,
                "name": "C",
                "shape": [2],
                "type": "ui16",
                "direction": "output",
            },
        ],
        "pe_instances": [],
    }

    manifest = MODULE.build_manifest(kernel, pre, rtl, "top")
    arguments = manifest["top_arguments"]
    assert [(arg["name"], arg["catapult_argument"]) for arg in arguments] == [
        ("A", "v7"),
        ("C", "v8"),
    ]
    assert [arg["rtl_direction"] for arg in arguments] == ["input", "output"]
    assert all(arg["packing"]["width_matches_shape"] for arg in arguments)
    assert manifest["top_interface"]["reset"]["polarity"] == "active_high"
    assert manifest["rtl_artifact"]["published_path"] == "backend-rtl/concat_rtl.v"

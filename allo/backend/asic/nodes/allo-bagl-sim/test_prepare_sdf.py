import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_sdf.py")
SPEC = importlib.util.spec_from_file_location("allo_bagl_prepare_sdf", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_prepare_sdf_builds_instance_manifest_and_timing_config(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    macro_dir = inputs / "macro-registry" / "class0"
    macro_dir.mkdir(parents=True)
    (macro_dir / "macro.sdf").write_text("(DELAYFILE)\n")
    (inputs / "macro-registry" / "index.json").write_text(json.dumps({
        "macros": [{
            "top_module": "kernel_0_0",
            "views": {"sdf": {"path": "class0/macro.sdf"}},
        }]
    }))
    (inputs / "macro-collateral.json").write_text(json.dumps({
        "rewritten_instances": [{
            "canonical_module": "kernel_0_0",
            "stable_instance_name": "compute_0_0",
        }]
    }))
    (inputs / "design.vcs.v").write_text("kernel_0_0 compute_0_0 ( );\n")
    (inputs / "design.pt.sdc").write_text(
        "set_clock_latency -source -0.501 [get_clocks ap_clk]\n"
        "set_clock_latency -source -0.533 [get_clocks ap_clk]\n"
    )
    (inputs / "testbench-contract.json").write_text(json.dumps({
        "clock_period_ns": 10.0,
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("bagl_input_delay_ns", "0.03")
    PREPARE.main()

    manifest = json.loads((tmp_path / "outputs/sdf-annotation-manifest.json").read_text())
    timing = json.loads((tmp_path / "outputs/timing-config.json").read_text())
    assert manifest["macro_sdf_expected_count"] == 1
    assert manifest["macro_annotations"][0]["scope_suffix"] == "compute_0_0"
    assert manifest["macro_annotations"][0]["scope"] == (
        "allo_generated_testbench.dut.compute_0_0"
    )
    assert timing["clock_compensation_ns"] == 0.533
    assert timing["clock_period_ns"] == 10.0
    assert timing["clock_half_period_ns"] == 5.0
    assert timing["input_delay_ns"] == 0.03
    assert abs(timing["bfm_drive_delay_ns"] - 5.563) < 1e-12
    assert (tmp_path / "outputs/timing-config.values").read_text().split() == [
        "0.533", "0.03", "0.025", "8", "5.563"
    ]


def test_prepare_sdf_rejects_missing_macro_instance(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    macro_dir = inputs / "macro-registry" / "class0"
    macro_dir.mkdir(parents=True)
    (macro_dir / "macro.sdf").write_text("sdf\n")
    (inputs / "macro-registry" / "index.json").write_text(json.dumps({
        "macros": [{"top_module": "m", "views": {"sdf": {"path": "class0/macro.sdf"}}}]
    }))
    (inputs / "macro-collateral.json").write_text(json.dumps({
        "rewritten_instances": [{"canonical_module": "m", "stable_instance_name": "u0"}]
    }))
    (inputs / "design.vcs.v").write_text("module top; endmodule\n")
    (inputs / "design.pt.sdc").write_text("")
    (inputs / "testbench-contract.json").write_text(json.dumps({
        "clock_period_ns": 10.0,
    }))
    monkeypatch.chdir(tmp_path)
    try:
        PREPARE.main()
    except ValueError as error:
        assert "cannot find macro instance" in str(error)
    else:
        raise AssertionError("missing netlist instance was accepted")

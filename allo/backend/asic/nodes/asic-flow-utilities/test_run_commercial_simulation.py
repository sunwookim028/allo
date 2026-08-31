import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("run_commercial_simulation.py")
    spec = importlib.util.spec_from_file_location("run_commercial_simulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rtl_report_uses_contract_markers(tmp_path, monkeypatch):
    module = load_module()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "compile.log").write_text("compiled\n")
    (outputs / "simulation.log").write_text("MY_PASS\n")
    (outputs / "run.vcd").write_text("$date\n")
    monkeypatch.setattr(module, "OUTPUTS", outputs)
    result = module.report("rtl", {
        "testbench_top": "ChipTb", "dut_instance": "dut",
        "pass_marker": "MY_PASS", "failure_marker": "MY_FAIL",
    }, 0, 0)
    assert result["status"] == "passed"
    assert result["pass_marker_found"]


def test_bagl_report_categorizes_sdf_and_requires_annotation(tmp_path, monkeypatch):
    module = load_module()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "compile.log").write_text(
        "Doing SDF annotation Done\n*** Annotation scope: ChipTb.dut\n"
        "Warning-[SDFCOM_TANE] ignored check\n"
    )
    (outputs / "simulation.log").write_text("MY_PASS\n")
    (outputs / "run.vcd").write_text("$date\n")
    monkeypatch.setattr(module, "OUTPUTS", outputs)
    result = module.report("bagl", {
        "testbench_top": "ChipTb", "dut_instance": "dut",
        "pass_marker": "MY_PASS", "failure_marker": "MY_FAIL",
    }, 0, 0)
    assert result["status"] == "passed"
    assert result["top_sdf_annotation_completed"]
    assert result["sdf_warning_categories"]["unmatched_timingcheck"] == 1


def test_all_vcs_modes_enable_debug_access_for_testbench_waveform_tasks():
    source = Path(__file__).with_name("run_commercial_simulation.py").read_text()
    base_command = source.split('if boolean_parameter("xprop_enabled"', 1)[0]
    assert '"-debug_access+all"' in base_command
    assert source.count('"-debug_access+all"') == 1


def test_report_rejects_binary_vpd_mislabeled_as_vcd(tmp_path, monkeypatch):
    module = load_module()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "compile.log").write_text("compiled\n")
    (outputs / "simulation.log").write_text("MY_PASS\n")
    (outputs / "run.vcd").write_bytes(b"VCD+ Writer\0binary")
    monkeypatch.setattr(module, "OUTPUTS", outputs)

    result = module.report("rtl", {
        "testbench_top": "ChipTb", "dut_instance": "dut",
        "pass_marker": "MY_PASS", "failure_marker": "MY_FAIL",
    }, 0, 0)

    assert result["status"] == "failed"
    assert not result["vcd_format_valid"]


def test_run_logged_tees_output_to_terminal_and_log(tmp_path, capsys):
    module = load_module()
    log = tmp_path / "command.log"

    result = module.run_logged(
        [sys.executable, "-c", "print('live output')"], log
    )

    assert result == 0
    assert "live output" in capsys.readouterr().out
    assert "live output" in log.read_text()


def test_run_logged_preserves_timeout_status(tmp_path, capsys):
    module = load_module()
    log = tmp_path / "command.log"

    result = module.run_logged(
        [sys.executable, "-c", "import time; print('before timeout', flush=True); time.sleep(5)"],
        log,
        timeout=0.2,
    )

    assert result == 124
    assert "before timeout" in capsys.readouterr().out
    assert "SIMULATION_TIMEOUT after 0.2 seconds" in log.read_text()

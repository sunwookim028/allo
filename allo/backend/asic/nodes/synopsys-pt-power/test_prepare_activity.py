import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_activity.py")


def run_prepare(tmp_path, source, *, explicit=False, mode="averaged"):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    activity = inputs / ("run.saif" if source == "saif" else "run.vcd")
    activity.write_text("activity\n" if source == "saif" else "$date\n$end\n$scope module ChipTb $end\n")
    (inputs / "testbench-contract.json").write_text(json.dumps({
        "testbench_top": "ChipTb", "dut_instance": "dut",
    }))
    if source == "rtl_vcd":
        (inputs / "design.namemap").write_text("# map\n")
    env = os.environ | {
        "activity_source": source,
        "analysis_mode": mode,
        "saif_instance": "custom/dut" if explicit else "auto",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=tmp_path, env=env, check=True)
    return json.loads((tmp_path / "activity-source.json").read_text())


def test_bagl_vcd_derives_strip_path_from_contract(tmp_path):
    result = run_prepare(tmp_path, "bagl_vcd")
    assert result["strip_path"] == "ChipTb/dut"
    assert not result["zero_delay"]
    assert not result["rtl_name_mapping"]


def test_ffgl_and_rtl_vcd_modes_are_explicit(tmp_path):
    ffgl = run_prepare(tmp_path / "ffgl", "ffgl_vcd", explicit=True)
    rtl = run_prepare(tmp_path / "rtl", "rtl_vcd")
    assert ffgl["zero_delay"] and ffgl["strip_path"] == "custom/dut"
    assert rtl["rtl_name_mapping"]


def test_saif_supported_for_averaged_but_not_time_based(tmp_path):
    result = run_prepare(tmp_path / "average", "saif")
    assert result["activity_format"] == "saif"
    inputs = tmp_path / "time" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "run.saif").write_text("activity\n")
    env = os.environ | {
        "activity_source": "saif", "analysis_mode": "time_based",
        "saif_instance": "ChipTb/dut",
    }
    completed = subprocess.run(["python3", str(SCRIPT)], cwd=tmp_path / "time", env=env)
    assert completed.returncode != 0


def test_binary_vpd_mislabeled_as_vcd_is_rejected(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "run.vcd").write_bytes(b"VCD+ Writer\0binary")
    env = os.environ | {
        "activity_source": "bagl_vcd", "analysis_mode": "averaged",
        "saif_instance": "ChipTb/dut",
    }
    completed = subprocess.run(["python3", str(SCRIPT)], cwd=tmp_path, env=env)
    assert completed.returncode != 0

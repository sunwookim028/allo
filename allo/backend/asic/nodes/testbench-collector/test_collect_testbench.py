import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("collect_testbench.py")


def base_environment(construct, testbench):
    return os.environ | {
        "construct_path": str(construct), "testbench_path": str(testbench),
        "testbench_top": "ChipTb", "dut_instance": "dut",
        "testbench_include_dirs": ".", "testbench_defines": "ASIC_SIM=1",
        "pass_marker": "TEST_PASS", "failure_marker": "TEST_FAIL",
        "simulation_timeout_seconds": "120",
    }


def test_simple_mode_packages_neighbors_but_compiles_only_main(tmp_path):
    testbench = tmp_path / "tb"
    testbench.mkdir()
    (testbench / "ChipTb.sv").write_text("module ChipTb; endmodule\n")
    (testbench / "alternative.sv").write_text("module alternative; endmodule\n")
    (testbench / "vectors.hex").write_text("01\n")
    construct = tmp_path / "construct.py"
    construct.write_text("# test\n")
    work = tmp_path / "work"
    work.mkdir()
    env = base_environment(construct, testbench) | {
        "testbench_file": "ChipTb.sv", "testbench_manifest": "",
        "simulation_args_file": "",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=work, env=env, check=True)
    filelist = (work / "outputs/testbench/testbench.f").read_text()
    assert "ChipTb.sv" in filelist
    assert "alternative.sv" not in filelist
    assert (work / "outputs/testbench/files/alternative.sv").is_file()
    assert (work / "outputs/testbench/files/vectors.hex").is_file()
    assert (work / "outputs/testbench/testbench-compile.args").read_text() == ""
    assert (work / "outputs/testbench/testbench-runtime.args").read_text() == ""
    contract = json.loads((work / "outputs/testbench-contract.json").read_text())
    assert contract["testbench_top"] == "ChipTb"
    assert contract["simulation_timeout_seconds"] == 120


def test_manifest_controls_compile_order_and_data(tmp_path):
    testbench = tmp_path / "tb"
    (testbench / "pkg").mkdir(parents=True)
    (testbench / "pkg/test_pkg.sv").write_text("package test_pkg; endpackage\n")
    (testbench / "ChipTb.sv").write_text("module ChipTb; endmodule\n")
    (testbench / "vectors.hex").write_text("01\n")
    manifest = testbench / "testbench_manifest.f"
    manifest.write_text("pkg/test_pkg.sv\nChipTb.sv\n@data vectors.hex\n")
    construct = tmp_path / "construct.py"
    construct.write_text("# test\n")
    work = tmp_path / "work"
    work.mkdir()
    env = base_environment(construct, testbench) | {
        "testbench_file": "undefined", "testbench_manifest": str(manifest),
        "simulation_args_file": "",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=work, env=env, check=True)
    lines = (work / "outputs/testbench/testbench.f").read_text().splitlines()
    source_lines = [line for line in lines if not line.startswith("+")]
    assert source_lines[0].endswith("pkg/test_pkg.sv")
    assert source_lines[1].endswith("ChipTb.sv")
    assert (work / "outputs/testbench/files/vectors.hex").is_file()


def test_upstream_design_args_remain_compile_arguments(tmp_path):
    testbench = tmp_path / "tb"
    testbench.mkdir()
    (testbench / "ChipTb.sv").write_text("module ChipTb; endmodule\n")
    construct = tmp_path / "construct.py"
    construct.write_text("# test\n")
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    (work / "inputs/testbench.sv").write_text("module ChipTb; endmodule\n")
    (work / "inputs/design.args").write_text("-debug_access+pp -sverilog\n")
    env = base_environment(construct, testbench) | {
        "consume_upstream_testbench": "True", "testbench_file": "undefined",
        "testbench_manifest": "", "simulation_args_file": "",
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=work, env=env, check=True)
    package = work / "outputs/testbench"
    assert package.joinpath("testbench-compile.args").read_text() == "-debug_access+pp -sverilog\n"
    assert package.joinpath("testbench-runtime.args").read_text() == ""

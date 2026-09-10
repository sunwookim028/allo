import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_constraints.py")


def run_generator(tmp_path, **parameters):
    env = os.environ | {
        "design_name": "chip",
        "clock_period": "4.0",
        "clock_port": "clock",
        "clock_name": "chip_clock",
        **parameters,
    }
    subprocess.run(["python3", str(SCRIPT)], cwd=tmp_path, env=env, check=True)
    return (tmp_path / "outputs/constraints.tcl").read_text()


def test_generates_single_clock_constraints_and_excludes_clock_from_inputs(tmp_path):
    text = run_generator(tmp_path)

    assert "create_clock -name {chip_clock} -period 4" in text
    assert "get_ports {clock}" in text
    assert "remove_from_collection [all_inputs] $asic_clock_port" in text
    assert "[expr 4 * 0.5]" in text
    assert "[expr 4 * 0.25]" in text
    assert "get_designs {chip}" in text


def test_supports_advanced_timing_parameters(tmp_path):
    text = run_generator(
        tmp_path,
        input_delay_fraction="0.25",
        output_delay_fraction="0.1",
        max_transition_fraction="0.15",
        max_fanout="12",
        clock_uncertainty="0.05",
    )

    assert "[expr 4 * 0.25]" in text
    assert "[expr 4 * 0.1]" in text
    assert "[expr 4 * 0.15]" in text
    assert "set_max_fanout 12" in text
    assert "set_clock_uncertainty 0.05" in text


def test_packages_constraint_override_relative_to_constructor(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    construct = design / "construct-commercial.py"
    construct.write_text("# test\n")
    override = design / "custom.tcl"
    override.write_text("create_clock -period 7 [get_ports clk]\n")
    work = tmp_path / "work"
    work.mkdir()

    text = run_generator(
        work,
        construct_path=str(construct),
        constraints_file="custom.tcl",
    )

    assert text == override.read_text()

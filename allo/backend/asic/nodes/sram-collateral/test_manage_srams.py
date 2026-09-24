import json
import os
import subprocess
import sys
from pathlib import Path


NODE = Path(__file__).parent
SCRIPT = NODE / "manage_srams.py"


def make_macro(root: Path, name: str = "test_sram") -> Path:
    macro = root / name
    macro.mkdir(parents=True)
    for suffix in ("v", "lib", "db", "lef", "gds", "sp"):
        (macro / f"{name}.{suffix}").write_text(f"{name} {suffix}\n")
    return macro


def run_manager(work: Path, *, expect_success: bool = True, preflight: bool = False, **parameters):
    env = os.environ | {key: str(value) for key, value in parameters.items()}
    command = [sys.executable, str(SCRIPT)]
    if preflight:
        command.append("--preflight")
    result = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True)
    assert (result.returncode == 0) is expect_success, result.stdout + result.stderr
    return result


def test_none_publishes_empty_versioned_contract(tmp_path):
    run_manager(tmp_path, sram_mode="none")

    contract = json.loads((tmp_path / "outputs/sram-contract.json").read_text())
    metadata = json.loads((tmp_path / "outputs/sram-metadata.json").read_text())
    assert contract == {
        "schema": "sram-collateral-contract",
        "schema_version": 1,
        "num_srams": 0,
        "srams": [],
    }
    assert metadata["mode"] == "none"
    assert metadata["num_srams"] == 0
    assert (tmp_path / "outputs/srams").is_dir()
    assert "num_srams: 0" in (tmp_path / "outputs/sram-info.txt").read_text()


def test_provided_validates_and_packages_views(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    construct = design / "construct-commercial.py"
    construct.write_text("# test\n")
    make_macro(design / "provided")
    work = tmp_path / "work"
    work.mkdir()

    run_manager(
        work,
        sram_mode="provided",
        construct_path=construct,
        provided_sram_path="provided",
    )

    contract = json.loads((work / "outputs/sram-contract.json").read_text())
    assert contract["num_srams"] == 1
    assert contract["srams"][0]["name"] == "test_sram"
    assert contract["srams"][0]["views"]["gds"] == ["test_sram/test_sram.gds"]
    assert (work / "outputs/srams/test_sram/test_sram.lef").is_file()


def test_provided_preflight_rejects_missing_required_view(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    construct = design / "construct-commercial.py"
    construct.write_text("# test\n")
    macro = make_macro(design / "provided")
    (macro / "test_sram.gds").unlink()
    work = tmp_path / "work"
    work.mkdir()

    result = run_manager(
        work,
        expect_success=False,
        preflight=True,
        sram_mode="provided",
        construct_path=construct,
        provided_sram_path="provided",
    )
    assert "missing required views: gds" in result.stderr


def test_bypass_republishes_upstream_package(tmp_path):
    make_macro(tmp_path / "inputs/srams", "upstream_sram")

    run_manager(tmp_path, sram_mode="bypass")

    metadata = json.loads((tmp_path / "outputs/sram-metadata.json").read_text())
    assert metadata["mode"] == "bypass"
    assert metadata["num_srams"] == 1
    assert (tmp_path / "outputs/srams/upstream_sram/upstream_sram.db").is_file()


def test_generate_rejects_unknown_method_before_tool_checks(tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    construct = design / "construct-commercial.py"
    construct.write_text("# test\n")
    rtl = design / "rtl"
    rtl.mkdir()
    (rtl / "sram_manifest.yml").write_text(
        "srams:\n  - name: test_sram\n    word_size: 8\n    num_words: 16\n"
    )

    result = run_manager(
        tmp_path,
        expect_success=False,
        preflight=True,
        sram_mode="generate",
        generate_method="not-yet-supported",
        construct_path=construct,
    )
    assert "Unsupported generate_method" in result.stderr


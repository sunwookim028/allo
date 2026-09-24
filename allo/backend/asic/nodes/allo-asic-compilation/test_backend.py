"""Tests for HLS backend configuration and RTL artifact discovery."""

from pathlib import Path

import pytest

from backend import configure_backend, find_rtl_directory, publish_rtl_artifacts


def test_configure_vitis_defaults():
    target, configs, metadata = configure_backend("vitis", 8.0, "")
    assert target == "vitis_hls"
    assert configs["frequency"] == 125.0
    assert configs["device"] == "u280"
    assert metadata["rtl_stage"] == "syn"


def test_configure_catapult_defaults():
    target, configs, metadata = configure_backend("catapult", 8.0, "")
    assert target == "catapult"
    assert configs["frequency"] == 125.0
    assert configs["device"] == "nangate-45nm_beh"
    assert configs["preserve_hierarchy"] is True
    assert metadata == {"device": "nangate-45nm_beh", "rtl_stage": "rtl"}


def test_configure_systemc_defaults():
    target, configs, metadata = configure_backend("systemc", 3.333, "")
    assert target == "systemc"
    assert configs["clock_period"] == 3.333
    assert configs["device"] == "nangate-45nm_beh"
    assert configs["preserve_hierarchy"] is True
    assert metadata["frontend"] == "systemc_connections"


def test_configure_catapult_shell_safe_pid_generalization_options():
    target, configs, _ = configure_backend(
        "catapult",
        8.0,
        "device=nangate-45nm_beh,generalize_pid_specializations=true,"
        "pid_generalization_policy=identity_only,preserve_hierarchy=false",
    )
    assert target == "catapult"
    assert configs["generalize_pid_specializations"] is True
    assert configs["pid_generalization_policy"] == "identity_only"
    assert configs["preserve_hierarchy"] is False


def test_configure_catapult_json_options():
    target, configs, _ = configure_backend(
        "catapult",
        4.0,
        '{"device":"sky130","preserve_hierarchy":false,'
        '"sub_funcs":["pe_0","pe_1"],'
        '"generalize_pid_specializations":true,'
        '"pid_generalization_policy":"identity_only"}',
    )
    assert target == "catapult"
    assert configs["device"] == "sky130"
    assert configs["preserve_hierarchy"] is False
    assert configs["sub_funcs"] == ["pe_0", "pe_1"]
    assert configs["generalize_pid_specializations"] is True
    assert configs["pid_generalization_policy"] == "identity_only"


def test_find_catapult_rtl_directory(tmp_path: Path):
    rtl_dir = tmp_path / "Catapult_3" / "top.v1"
    rtl_dir.mkdir(parents=True)
    (rtl_dir / "rtl.v").write_text("module top; endmodule\n")
    assert find_rtl_directory("catapult", tmp_path) == rtl_dir


def test_find_catapult_rtl_directory_rejects_ambiguity(tmp_path: Path):
    for run in ("Catapult_1", "Catapult_2"):
        rtl_dir = tmp_path / run / "top.v1"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "rtl.v").write_text("module top; endmodule\n")
    with pytest.raises(RuntimeError, match="exactly one Catapult"):
        find_rtl_directory("catapult", tmp_path)


def test_find_systemc_rtl_directory_under_build(tmp_path: Path):
    rtl_dir = tmp_path / "build" / "Catapult_1" / "top.v1"
    rtl_dir.mkdir(parents=True)
    (rtl_dir / "rtl.v").write_text("module top; endmodule\n")
    assert find_rtl_directory("systemc", tmp_path) == rtl_dir


def test_publish_systemc_normalizes_rtl_filename(tmp_path: Path):
    rtl_dir = tmp_path / "project" / "build" / "Catapult_1" / "top.v1"
    rtl_dir.mkdir(parents=True)
    (rtl_dir / "rtl.v").write_text("module top; endmodule\n")

    output = tmp_path / "published"
    publish_rtl_artifacts("systemc", tmp_path / "project", output)

    assert (output / "concat_rtl.v").read_text() == "module top; endmodule\n"
    assert not (output / "rtl.v").exists()


def test_publish_catapult_rtl_is_compact(tmp_path: Path):
    rtl_dir = tmp_path / "project" / "Catapult_1" / "top.v1"
    rtl_dir.mkdir(parents=True)
    for name in ("rtl.v", "concat_rtl.v", "cycle.rpt"):
        (rtl_dir / name).write_text(name + "\n")
    (rtl_dir / "schematic").mkdir()
    (rtl_dir / "schematic" / "large.bin").write_bytes(b"unused")
    scverify = rtl_dir / "scverify"
    scverify.mkdir()
    (scverify / "dut_v_ports.map").write_text("ports\n")

    output = tmp_path / "published"
    publish_rtl_artifacts("catapult", tmp_path / "project", output)

    assert (output / "concat_rtl.v").is_file()
    assert not (output / "rtl.v").exists()
    assert (output / "cycle.rpt").is_file()
    assert (output / "scverify" / "dut_v_ports.map").is_file()
    assert not (output / "schematic").exists()

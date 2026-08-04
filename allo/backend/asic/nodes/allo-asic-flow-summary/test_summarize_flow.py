"""Unit tests for report parsing without commercial tools."""

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_flow.py")
SPEC = importlib.util.spec_from_file_location("summarize_flow", MODULE_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_report_parsers():
    assert SUMMARY.parse_lef_size("SIZE 12.5 BY 9.25 ;") == (12.5, 9.25)
    assert SUMMARY.parse_path_slack(
        "slack (MET) 0.120\nslack (VIOLATED) -0.030\n"
    ) == -0.03
    timing = SUMMARY.parse_global_timing(
        "No setup violations found.\nHold violations\nWNS -0.030 0\nTNS -0.052 0\n"
    )
    assert timing == {
        "setup_wns_ns": 0.0,
        "setup_tns_ns": 0.0,
        "hold_wns_ns": -0.03,
        "hold_tns_ns": -0.052,
    }
    assert SUMMARY.parse_drc_count("TOTAL DRC Results Generated: 7 (7)") == 7
    assert SUMMARY.parse_lvs("# CORRECT #") == "passed"
    assert SUMMARY.parse_lvs("# INCORRECT #") == "failed"
    assert SUMMARY.parse_density(
        "default core: bins with density > 0.750 = 44.61 % ( 1 / 2 )"
    ) == 44.61


def test_summary_outputs_are_json_tcl_and_text(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    registry_root = inputs / "macro-registry"
    macro_dir = registry_root / "macro_alpha_test"
    reports = macro_dir / "reports"
    reports.mkdir(parents=True)
    (macro_dir / "macro.lef").write_text("MACRO m\n SIZE 10 BY 20 ;\nEND m\n")
    (reports / "timing-report_global_timing.report").write_text(
        "No setup violations found.\nNo hold violations found.\n"
    )
    (reports / "macro.timing.setup.rpt").write_text("slack (MET) 0.125\n")
    (reports / "macro.timing.hold.rpt").write_text("slack (MET) 0.075\n")
    (reports / "drc.summary").write_text("TOTAL DRC Results Generated: 0 (0)\n")
    (reports / "lvs.report").write_text("# CORRECT #\n")
    registry = {
        "macros": [{
            "macro_class_id": "macro_alpha_test",
            "top_module": "m",
            "reuse_count": 4,
            "views": {"lef": {"path": "macro_alpha_test/macro.lef"}},
        }]
    }
    inputs.mkdir(exist_ok=True)
    (inputs / "macro-registry.json").write_text(json.dumps(registry))
    for name in SUMMARY.METRIC_FILES:
        (inputs / name).write_text(json.dumps({
            "node": name.removesuffix("-metrics.json"),
            "status": "passed",
            "wall_seconds": 1.5,
        }))
    (inputs / "drc.summary").write_text(
        "TOTAL DRC Results Generated: 0 (0)\n"
    )
    (inputs / "innovus-drc.rpt").write_text(
        "Verification Complete : 0 Viols.\n"
    )
    (inputs / "innovus-antenna.rpt").write_text(
        "Verification Complete : 12 Viols.\n"
    )
    (inputs / "drc-policy.json").write_text(json.dumps({
        "antenna_check_policy": "report",
        "antenna_results": 12,
        "non_antenna_results": 0,
    }))
    (inputs / "lvs.report").write_text("# CORRECT #\n")
    monkeypatch.setattr(SUMMARY, "INPUTS", inputs)
    monkeypatch.setattr(SUMMARY, "REGISTRY_ROOT", registry_root)
    monkeypatch.setattr(SUMMARY, "OUTPUTS", outputs)
    SUMMARY.main()
    result = json.loads((outputs / "flow-summary.json").read_text())
    assert result["full_chip_verification"]["innovus_route_drc_results"] == 0
    assert result["full_chip_verification"]["innovus_antenna_results"] == 12
    assert result["macros"][0]["physical_area_um2"] == 200.0
    assert result["macros"][0]["setup_wns_ns"] == 0.125
    assert result["macros"][0]["hold_wns_ns"] == 0.075
    assert result["full_chip_verification"] == {
        "drc_results": 0,
        "calibre_non_antenna_results": 0,
        "calibre_antenna_results": 12,
        "calibre_antenna_policy": "report",
        "innovus_route_drc_results": 0,
        "innovus_antenna_results": 12,
        "lvs_status": "passed",
    }
    assert "allo_asic_macro_count 1" in (outputs / "flow-summary.tcl").read_text()
    assert "Power: unavailable" in (outputs / "flow-summary.txt").read_text()

import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("collect_pnr_metrics.py")
SPEC = importlib.util.spec_from_file_location("collect_pnr_metrics", PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def test_collects_stage_density_congestion_and_antenna_metrics(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    reports.mkdir()
    logs.mkdir()
    (reports / "pnr-stage-times.rpt").write_text("placement 12.5\ncts 3.25\n")
    (reports / "preplace.summary").write_text("Density: 71.25%\n")
    (reports / "signoff.summary").write_text("Density: 72.50%\n")
    (reports / "physical-intent-placement.rpt").write_text("placed_hard_macros 15\n")
    (reports / "innovus-antenna.rpt").write_text("Antenna violations: 7\n")
    (reports / "innovus-drc.rpt").write_text("Verification Complete : 2 Viols.\n")
    (reports / "postcts_hold_hold.summary").write_text(
        "|           WNS (ns):|  0.120  |\n"
        "|           TNS (ns):|  0.000  |\n"
        "|    Violating Paths:|    0    |\n"
    )
    (reports / "postroute_setup.summary").write_text(
        "|           WNS (ns):|  1.250  |\n"
        "|           TNS (ns):|  0.000  |\n"
        "|    Violating Paths:|    0    |\n"
    )
    (reports / "postroute_hold_hold.summary").write_text(
        "|           WNS (ns):|  0.175  |\n"
        "|           TNS (ns):|  0.000  |\n"
        "|    Violating Paths:|    0    |\n"
    )
    (reports / "signoff_hold.summary").write_text(
        "|           WNS (ns):|  0.160  |\n"
        "|           TNS (ns):|  0.000  |\n"
        "|    Violating Paths:|    0    |\n"
    )
    (logs / "run.log").write_text(
        "Total instances in design: 67000\n"
        "Routing Overflow: 0.10% H and 0.75% V\n"
        "EstWL: 12345um\npeak = 4096.0 (MB)\n"
    )
    result = METRICS.collect(tmp_path, 100.0, drc_check_policy="report")
    assert result["stage_wall_seconds"] == {"placement": 12.5, "cts": 3.25}
    assert result["run_mode"] == "full"
    assert result["completed_step"] == "signoff"
    assert result["placement"] == {
        "initial_density_percent": 71.25,
        "final_density_percent": 72.5,
        "instance_count": 67000,
        "hard_macro_count": 15,
    }
    assert result["routing"]["overflow_vertical_percent"] == 0.75
    assert result["routing"]["estimated_wirelength_um"] == 12345
    assert result["resources"]["peak_memory_mb"] == 4096
    assert result["timing"]["postcts_hold"]["wns_ns"] == 0.12
    assert result["timing"]["postroute_setup"]["wns_ns"] == 1.25
    assert result["timing"]["postroute_hold"]["wns_ns"] == 0.175
    assert result["timing"]["signoff_hold"] == {
        "wns_ns": 0.16,
        "tns_ns": 0.0,
        "violating_paths": 0,
        "target_slack_ns": 0.15,
        "target_met": True,
    }
    assert result["drc"] == {
        "policy": "report",
        "status": "violations",
        "violation_count": 2,
    }
    assert result["antenna"] == {"status": "passed", "violation_count": 7.0}


def test_collects_successful_placement_only_experiment(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    reports.mkdir()
    logs.mkdir()
    (reports / "pnr-stage-times.rpt").write_text("placement 408.234\n")
    (reports / "preplace.summary").write_text("Density: 71.25%\n")
    (reports / "place.summary").write_text("Density: 72.357%\n")
    (reports / "physical-intent-placement.rpt").write_text("placed_hard_macros 15\n")
    (logs / "run.log").write_text(
        "Total instances in design: 67000\n"
        "Routing Overflow: 0.16% H and 0.18% V\n"
        "EstWL: 1413352um\npeak = 4096.0 (MB)\n"
    )
    result = METRICS.collect(tmp_path, 470.0, "place")
    assert result["status"] == "passed"
    assert result["run_mode"] == "early_stop"
    assert result["completed_step"] == "place"
    assert result["placement"]["final_density_percent"] == 72.357
    assert result["stage_wall_seconds"] == {"placement": 408.234}
    assert result["timing"]["signoff_hold"]["target_met"] is False
    assert result["antenna"] == {"status": "not_run", "violation_count": None}
    assert result["drc"] == {
        "policy": "error",
        "status": "not_run",
        "violation_count": None,
    }

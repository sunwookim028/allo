import importlib.util
from pathlib import Path


ROOT = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_welltap_cuts_coalesce_rows():
    planner = load("plan_welltap_cuts", ROOT / "scripts/plan-welltap-cuts.py")
    boxes = [(1, 2, 3, 4), (1, 4.001, 3, 5), (8, 2, 9, 4)]
    assert planner.coalesce(boxes) == [(1, 2, 3, 5), (8, 2, 9, 4)]


def test_metrics_are_flat_and_enforce_hold_target(tmp_path):
    metrics = load("collect_pnr_metrics", ROOT / "collect_pnr_metrics.py")
    reports, logs = tmp_path / "reports", tmp_path / "logs"
    reports.mkdir(); logs.mkdir()
    (reports / "pnr-stage-times.rpt").write_text("placement 1.5\n")
    (reports / "preplace.summary").write_text("Density: 60%\n")
    (reports / "signoff.summary").write_text("Density: 65%\n")
    (reports / "signoff_hold.summary").write_text("| WNS (ns):| 0.010 |\n")
    (reports / "innovus-drc.rpt").write_text("Verification Complete : 0 Viols\n")
    (reports / "innovus-antenna.rpt").write_text("Verification Complete : 0 Viols\n")
    (logs / "run.log").write_text("Total instances in design: 100\n")
    result = metrics.collect(tmp_path, 2.0, hold_target_slack=0.005)
    assert result["node"] == "cadence-innovus-pnr"
    assert result["placement"]["instance_count"] == 100
    assert result["timing"]["signoff_hold"]["target_met"]
    assert result["drc"]["violation_count"] == 0

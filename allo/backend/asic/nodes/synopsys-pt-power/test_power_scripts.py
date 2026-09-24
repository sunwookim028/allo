from pathlib import Path


ROOT = Path(__file__).parent


def test_read_design_uses_generated_activity_variable():
    script = (ROOT / "scripts/read-design.tcl").read_text()
    assert "ptpx_activity_is_rtl" in script
    assert "ptpx_rtl_mapping" not in script


def test_start_propagates_sourced_script_errors():
    script = (ROOT / "START.tcl").read_text()
    assert script.count("if {[catch {source -echo -verbose") == 2
    assert script.count("exit 1") >= 3

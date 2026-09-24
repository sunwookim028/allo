import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("report_sim.py")
SPEC = importlib.util.spec_from_file_location("allo_bagl_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_warning_headers_are_counted_once_and_classified():
    text = """Warning-[SDFCOM_TANE] Cannot match timing check
This warning came from an SDF TIMINGCHECK record.
Warning-[SDFCOM_IANE] Cannot match IOPATH
Warning-[SDFCOM_UHICD] Up-hierarchy interconnect ignored
Warning-[SDFCOM_TANE] Cannot match timing check
"""
    codes = REPORT.warning_codes(text)
    assert codes == ["SDFCOM_TANE", "SDFCOM_IANE", "SDFCOM_UHICD", "SDFCOM_TANE"]

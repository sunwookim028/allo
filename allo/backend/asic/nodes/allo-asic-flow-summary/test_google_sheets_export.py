"""Dependency-free tests for optional Google Sheets export plumbing."""

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("export_summary_to_google_sheets.py")
SPEC = importlib.util.spec_from_file_location("export_summary", MODULE_PATH)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_dotted_lookup():
    assert EXPORT.lookup({"a": {"b-c": {"value": 7}}}, "a.b-c.value") == 7


def test_disabled_export_always_emits_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(EXPORT, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setenv("google_sheets_enabled", "false")
    EXPORT.main()
    status = json.loads(
        (tmp_path / "outputs/google-sheets-export.json").read_text()
    )
    assert status == {
        "schema_version": 1,
        "status": "disabled",
        "uploaded": False,
    }

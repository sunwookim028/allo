#!/usr/bin/env python3
"""Optionally append one ordered ASIC-flow summary row to Google Sheets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path.cwd()
OUTPUTS = ROOT / "outputs"


def boolean_parameter(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def lookup(document: object, path: str) -> object:
    value = document
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"summary column path does not exist: {path}")
        value = value[component]
    return value


def write_status(status: dict) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "google-sheets-export.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


def main() -> None:
    enabled = boolean_parameter("google_sheets_enabled", False)
    required = boolean_parameter("google_sheets_required", True)
    if not enabled:
        write_status({
            "schema_version": 1,
            "status": "disabled",
            "uploaded": False,
        })
        print("Google Sheets export is disabled")
        return

    spreadsheet_id = os.environ.get("google_spreadsheet_id", "").strip()
    worksheet_name = os.environ.get("google_worksheet_name", "Results").strip()
    credentials = Path(os.environ.get("google_sheets_credentials", "")).expanduser()
    columns_file = Path(
        os.environ.get("google_sheet_columns_file", "google-sheet-columns.json")
    )
    if not spreadsheet_id or spreadsheet_id == "REPLACE_WITH_SPREADSHEET_ID":
        raise ValueError("replace google_spreadsheet_id before enabling export")
    if not worksheet_name:
        raise ValueError("google_worksheet_name must not be empty")
    if not credentials.is_file():
        raise FileNotFoundError(f"Google service-account key not found: {credentials}")
    if not columns_file.is_file():
        raise FileNotFoundError(f"Google Sheets column configuration not found: {columns_file}")

    summary = json.loads((OUTPUTS / "flow-summary.json").read_text())
    configuration = json.loads(columns_file.read_text())
    columns = configuration.get("columns", [])
    if not columns:
        raise ValueError("Google Sheets column configuration has no columns")
    headers = [item["header"] for item in columns]
    values = [lookup(summary, item["path"]) for item in columns]
    values = ["" if value is None else value for value in values]

    try:
        import gspread
    except ImportError as error:
        raise RuntimeError(
            "gspread is not installed in the environment running this node; "
            "run: python -m pip install gspread"
        ) from error

    client = gspread.service_account(filename=str(credentials))
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)
    existing_headers = worksheet.row_values(1)
    if not existing_headers:
        worksheet.update([headers], "A1", value_input_option="RAW")
    elif existing_headers != headers:
        raise ValueError(
            "Google worksheet header does not match google-sheet-columns.json; "
            f"existing={existing_headers!r}, requested={headers!r}"
        )
    worksheet.append_row(values, value_input_option="RAW")
    status = {
        "schema_version": 1,
        "status": "passed",
        "uploaded": True,
        "spreadsheet_id": spreadsheet_id,
        "worksheet": worksheet_name,
        "row": values,
        "headers": headers,
    }
    write_status(status)
    print(f"Appended ASIC-flow summary row to {spreadsheet.title}/{worksheet.title}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # Ensure a useful local artifact on API failure.
        write_status({
            "schema_version": 1,
            "status": "failed",
            "uploaded": False,
            "error": str(error),
        })
        if boolean_parameter("google_sheets_required", True):
            raise
        print(f"WARNING: Google Sheets export failed: {error}", file=sys.stderr)

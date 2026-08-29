# Allo ASIC flow summary

This dependency-free reporting node reads the published macro registry, preserved
tool reports, and explicit node timing metrics. It emits JSON for automation, Tcl
for commercial-flow scripts, and a short human-readable text report. It does not
scrape `mflowgen-run.log`; unavailable measurements remain explicitly unavailable.
It is connected after full-chip GDS merge, DRC, and LVS and records their node
runtimes plus the explicit full-chip DRC result count and LVS status.
The summary also records full-chip PNR peak memory and the build directory's
apparent size. Directory sizing counts files stored under the build root and
does not follow symbolic links to external data.
When the optional collateral-trim node is connected, the spreadsheet-facing
build size remains the pre-trim size for fair storage scaling estimates. The
JSON summary additionally records post-trim size and bytes removed.

## Optional Google Sheets export

The second command appends one row to a configured Google worksheet whenever
this node actually executes. Export is disabled by default. Edit
`google-sheet-columns.json` to set column names and ordered dotted paths into
`flow-summary.json`, then set these graph parameters:

```python
"google_sheets_enabled": True,
"google_sheets_required": True,
"google_sheets_credentials":
    "/home/jb2698/.config/gspread/service_account.json",
"google_spreadsheet_id": "REPLACE_WITH_SPREADSHEET_ID",
"google_worksheet_name": "Results",
```

Install `gspread` in the Python environment that runs mflowgen:

```bash
python -m pip install gspread
```

Create a Google service account, enable the Sheets API, download its JSON key
outside the repository, protect it with mode 600, and share the target sheet
with the service-account email as an Editor. The exporter opens the spreadsheet
by ID, creates row 1 from the configured headers if the worksheet is empty, and
then appends the result row. A mismatched existing header is an error to prevent
silent column corruption. `google_sheets_required: false` converts API failures
to warnings; either way, `google-sheets-export.json` records the attempt. Never
commit the service-account JSON key.

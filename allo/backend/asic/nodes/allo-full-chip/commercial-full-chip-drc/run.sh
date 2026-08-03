#!/usr/bin/env bash
set -euo pipefail

flow_start_epoch=$(date +%s)
mkdir -p outputs

finish_metrics() {
  rc=$?
  flow_end_epoch=$(date +%s)
  status=failed
  if [[ $rc -eq 0 ]]; then status=passed; fi
  python - "$status" "$((flow_end_epoch - flow_start_epoch))" <<'PY'
import json
import sys
from pathlib import Path

Path("outputs/full-chip-drc-metrics.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "node": "commercial-full-chip-drc",
            "status": sys.argv[1],
            "wall_seconds": float(sys.argv[2]),
        },
        indent=2,
    )
    + "\n"
)
PY
  exit "$rc"
}
trap finish_metrics EXIT

if [[ -f "inputs/adk/${drc_env_setup}" ]]; then
  source "inputs/adk/${drc_env_setup}"
fi
envsubst < drc.runset.template > drc.runset
calibre -gui -drc -batch -runset drc.runset
test -s drc.results
test -s drc.summary

python - <<'PY'
import re
from pathlib import Path

text = Path("drc.summary").read_text(errors="replace")
match = re.search(r"TOTAL DRC Results Generated:\s*(\d+)", text, re.I)
if match is None:
    raise SystemExit("ERROR: Calibre DRC summary has no total result count")
if int(match.group(1)) != 0:
    raise SystemExit(f"ERROR: Calibre DRC reported {match.group(1)} result(s)")
PY

ln -sf ../drc.results outputs/drc.results
ln -sf ../drc.summary outputs/drc.summary

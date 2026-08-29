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
ln -sf ../drc.results outputs/drc.results
ln -sf ../drc.summary outputs/drc.summary

python - "${antenna_check_policy:-report}" "${drc_check_policy:-error}" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path("drc.summary").read_text(errors="replace")
match = re.search(r"TOTAL DRC Results Generated:\s*(\d+)", text, re.I)
if match is None:
    raise SystemExit("ERROR: Calibre DRC summary has no total result count")
antenna_policy = sys.argv[1].lower()
drc_policy = sys.argv[2].lower()
if antenna_policy not in {"error", "report", "off"}:
    raise SystemExit(
        f"ERROR: unsupported antenna_check_policy: {antenna_policy}"
    )
if drc_policy not in {"error", "report"}:
    raise SystemExit(f"ERROR: unsupported drc_check_policy: {drc_policy}")
global_section = re.split(
    r"RULECHECK RESULTS STATISTICS \(BY CELL\)", text, maxsplit=1, flags=re.I
)[0]
rule_counts = [
    (name, int(count))
    for name, count in re.findall(
        r"RULECHECK\s+(\S+)\s+.*?TOTAL Result Count\s*=\s*(\d+)",
        global_section,
        re.I,
    )
]
if not rule_counts:
    raise SystemExit("ERROR: Calibre DRC summary has no per-rule result counts")
antenna = sum(count for name, count in rule_counts if name.lower().startswith("antenna."))
non_antenna = sum(
    count for name, count in rule_counts if not name.lower().startswith("antenna.")
)
total = int(match.group(1))
result = {
    "schema_version": 1,
    "antenna_check_policy": antenna_policy,
    "drc_check_policy": drc_policy,
    "total_results": total,
    "antenna_results": antenna,
    "non_antenna_results": non_antenna,
    "antenna_enforced": antenna_policy == "error",
    "non_antenna_enforced": drc_policy == "error",
    "status": (
        "failed"
        if (drc_policy == "error" and non_antenna)
        or (antenna_policy == "error" and antenna)
        else "reported"
        if non_antenna or antenna
        else "passed"
    ),
}
Path("outputs/drc-policy.json").write_text(json.dumps(result, indent=2) + "\n")
if drc_policy == "error" and non_antenna:
    raise SystemExit(f"ERROR: Calibre DRC reported {non_antenna} non-antenna result(s)")
if drc_policy == "report" and non_antenna:
    print(
        f"WARNING: Calibre DRC reported {non_antenna} non-antenna result(s); "
        "continuing under drc_check_policy=report",
        file=sys.stderr,
    )
if antenna_policy == "error" and antenna:
    raise SystemExit(f"ERROR: Calibre DRC reported {antenna} antenna result(s)")
PY

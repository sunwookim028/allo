#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s); mkdir -p outputs
finish() {
  rc=$?; status=failed; [[ $rc -eq 0 ]] && status=passed
  python3 - "$status" "$(( $(date +%s) - start ))" <<'PY'
import json, sys
from pathlib import Path
Path('outputs/drc-metrics.json').write_text(json.dumps({
  'schema_version': 1, 'node': 'mentor-calibre-drc',
  'status': sys.argv[1], 'wall_seconds': float(sys.argv[2])}, indent=2) + '\n')
PY
  exit "$rc"
}
trap finish EXIT
[[ -f "inputs/adk/${drc_env_setup}" ]] && source "inputs/adk/${drc_env_setup}"
envsubst < drc.runset.template > drc.runset
calibre -gui -drc -batch -runset drc.runset
test -s drc.results; test -s drc.summary
ln -sf ../drc.results outputs/drc.results; ln -sf ../drc.summary outputs/drc.summary
python3 parse_drc.py drc.summary outputs/drc-policy.json \
  "$antenna_check_policy" "$drc_check_policy"

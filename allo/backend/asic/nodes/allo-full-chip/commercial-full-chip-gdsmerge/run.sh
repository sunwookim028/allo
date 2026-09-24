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

Path("outputs/full-chip-gdsmerge-metrics.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "node": "commercial-full-chip-gdsmerge",
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

# Calibre WorkBench expects this directory even for batch-mode calibredrv.
mkdir -p "$HOME/.calibrewb_workspace/tmp"

merge_inputs=""
for pattern in \
  inputs/adk/*.gds* \
  inputs/macro-registry/*/*.gds* \
  inputs/srams/*/*.gds*; do
  for file in $pattern; do
    [[ -e "$file" ]] || continue
    merge_inputs="$merge_inputs -in $file"
  done
done

# -indir inputs supplies the routed top GDS. Explicit -in arguments add every
# standard-cell, hardened-macro, and optional SRAM library definition.
echo | calibredrv -a layout filemerge \
  -indir inputs $merge_inputs \
  -topcell "$design_name" \
  -out design_merged.gds 2>&1 | tee merge.log

test -s design_merged.gds
if grep -q 'WARNING: Ignoring duplicate structure' merge.log; then
  echo 'ERROR: Calibre ignored a duplicate GDS structure during merge' >&2
  exit 1
fi
ln -sf ../design_merged.gds outputs/design_merged.gds
ln -sf ../merge.log outputs/merge.log

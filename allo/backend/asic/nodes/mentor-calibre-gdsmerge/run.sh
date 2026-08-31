#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s); mkdir -p outputs
finish() {
  rc=$?; status=failed; [[ $rc -eq 0 ]] && status=passed
  python3 - "$status" "$(( $(date +%s) - start ))" <<'PY'
import json, sys
from pathlib import Path
Path('outputs/gdsmerge-metrics.json').write_text(json.dumps({
  'schema_version': 1, 'node': 'mentor-calibre-gdsmerge',
  'status': sys.argv[1], 'wall_seconds': float(sys.argv[2])}, indent=2) + '\n')
PY
  exit "$rc"
}
trap finish EXIT
mkdir -p "$HOME/.calibrewb_workspace/tmp"
merge_inputs=()
for pattern in inputs/adk/*.gds* inputs/srams/*/*.gds*; do
  for file in $pattern; do [[ -e "$file" ]] && merge_inputs+=( -in "$file" ); done
done
echo | calibredrv -a layout filemerge -indir inputs "${merge_inputs[@]}" \
  -topcell "$design_name" -out design_merged.gds 2>&1 | tee merge.log
test -s design_merged.gds
! grep -q 'WARNING: Ignoring duplicate structure' merge.log
ln -sf ../design_merged.gds outputs/design_merged.gds
ln -sf ../merge.log outputs/merge.log

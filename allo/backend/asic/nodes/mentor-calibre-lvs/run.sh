#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s); mkdir -p outputs; lvs_result=unavailable
finish() {
  rc=$?; status=failed; [[ $rc -eq 0 ]] && status=passed
  [[ $rc -eq 0 && "$lvs_result" == incorrect ]] && status=reported
  python3 - "$status" "$(( $(date +%s) - start ))" "$lvs_check_policy" "$lvs_result" <<'PY'
import json, sys
from pathlib import Path
Path('outputs/lvs-metrics.json').write_text(json.dumps({
  'schema_version': 1, 'node': 'mentor-calibre-lvs', 'status': sys.argv[1],
  'wall_seconds': float(sys.argv[2]), 'policy': sys.argv[3],
  'lvs_result': sys.argv[4]}, indent=2) + '\n')
PY
  exit "$rc"
}
trap finish EXIT
[[ "$lvs_check_policy" == error || "$lvs_check_policy" == report ]]
for name in "$lvs_power_name" "$lvs_ground_name"; do
  [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_\$]*$ ]] || exit 1
done
[[ "$lvs_power_name" != "$lvs_ground_name" ]]
[[ -z "$lvs_hcells_file" ]] && export lvs_use_hcells=0 || export lvs_use_hcells=1
[[ -z "$lvs_connect_names" ]] && export lvs_connect_names_state=NONE || export lvs_connect_names_state=SOME
envsubst < lvs.runset.template > lvs.runset
touch inputs/rules.svrf
cp inputs/design.lvs.v merged.lvs.v
args=(-v merged.lvs.v -o source.lvs.sp -s0 "$lvs_ground_name" -s1 "$lvs_power_name" -w 2)
for pattern in inputs/adk/*.cdl inputs/adk/*.spi inputs/adk/*.sp inputs/adk/*source.added inputs/srams/*/*.cdl inputs/srams/*/*.spi inputs/srams/*/*.sp; do
  for file in $pattern; do [[ -e "$file" ]] && args+=( -s "$file" ); done
done
if [[ -n "$lvs_extra_spice_include" ]]; then
  read -r -a extras <<< "$lvs_extra_spice_include"
  for file in "${extras[@]}"; do [[ -f "$file" ]] || exit 1; args+=( -s "$file" ); done
fi
v2lvs "${args[@]}" -log v2lvs.log
test -s source.lvs.sp
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_power_name}([[:space:]]|$)" source.lvs.sp
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_ground_name}([[:space:]]|$)" source.lvs.sp
calibre -gui -lvs -batch -runset lvs.runset
test -s lvs.report
ln -sf ../lvs.report outputs/lvs.report
ln -sf ../source.lvs.sp outputs/design.schematic.spi
ln -sf ../merged.lvs.v outputs/design_merged.lvs.v
if grep -Eq '#[[:space:]]+INCORRECT[[:space:]]+#' lvs.report; then
  lvs_result=incorrect; [[ "$lvs_check_policy" == report ]] || exit 1
elif grep -Eq '#[[:space:]]+CORRECT[[:space:]]+#' lvs.report; then lvs_result=correct
else exit 1; fi

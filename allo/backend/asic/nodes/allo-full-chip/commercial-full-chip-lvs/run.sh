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

Path("outputs/full-chip-lvs-metrics.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "node": "commercial-full-chip-lvs",
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

for supply_name in "$lvs_power_name" "$lvs_ground_name"; do
  if [[ ! "$supply_name" =~ ^[A-Za-z_][A-Za-z0-9_$]*$ ]]; then
    echo "ERROR: invalid LVS supply net name: $supply_name" >&2
    exit 1
  fi
done
if [[ "$lvs_power_name" == "$lvs_ground_name" ]]; then
  echo 'ERROR: LVS power and ground net names must differ' >&2
  exit 1
fi

if [[ -z "$lvs_hcells_file" ]]; then export lvs_use_hcells=0; else export lvs_use_hcells=1; fi
if [[ -z "$lvs_connect_names" ]]; then export lvs_connect_names_state=NONE; else export lvs_connect_names_state=SOME; fi

spice_files=()
for pattern in \
  inputs/adk/*.cdl \
  inputs/adk/*.spi \
  inputs/adk/*.sp \
  inputs/adk/*source.added \
  inputs/srams/*/*.cdl \
  inputs/srams/*/*.spi \
  inputs/srams/*/*.sp; do
  for file in $pattern; do
    [[ -e "$file" ]] || continue
    spice_files+=("$file")
  done
done

envsubst < lvs.runset.template > lvs.runset
touch inputs/rules.svrf

# The top physical netlist excludes leaf-cell definitions. Add each published
# canonical macro physical netlist so V2LVS sees the same hierarchy as GDS.
cp inputs/design.lvs.v merged.lvs.v
macro_lvs_count=0
for file in inputs/macro-registry/*/*.lvs.v; do
  [[ -e "$file" ]] || continue
  cat "$file" >> merged.lvs.v
  macro_lvs_count=$((macro_lvs_count + 1))
done
# A zero-count registry is the intentional contract for a flat bypass run.
# The publisher validates nonempty registries before they reach this node.

v2lvs_args=(
  -v merged.lvs.v
  -o source.lvs.sp
  -s0 "$lvs_ground_name"
  -s1 "$lvs_power_name"
  -w 2
)
for file in "${spice_files[@]}"; do
  v2lvs_args+=( -s "$file" )
done
if [[ -n "$lvs_extra_spice_include" ]]; then
  read -r -a extra_spice_files <<< "$lvs_extra_spice_include"
  for file in "${extra_spice_files[@]}"; do
    [[ -f "$file" ]] || {
      echo "ERROR: missing extra LVS SPICE include: $file" >&2
      exit 1
    }
    v2lvs_args+=( -s "$file" )
  done
fi

v2lvs "${v2lvs_args[@]}" -log v2lvs.log
test -s source.lvs.sp
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_power_name}([[:space:]]|$)" source.lvs.sp
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_ground_name}([[:space:]]|$)" source.lvs.sp

calibre -gui -lvs -batch -runset lvs.runset
test -s lvs.report
if grep -Eq '#[[:space:]]+INCORRECT[[:space:]]+#' lvs.report; then
  echo 'ERROR: Calibre LVS reported INCORRECT' >&2
  exit 1
fi
if ! grep -Eq '#[[:space:]]+CORRECT[[:space:]]+#' lvs.report; then
  echo 'ERROR: Calibre LVS report does not contain a CORRECT result' >&2
  exit 1
fi

ln -sf ../lvs.report outputs/lvs.report
ln -sf ../source.lvs.sp outputs/design.schematic.spi
ln -sf ../merged.lvs.v outputs/design_merged.lvs.v

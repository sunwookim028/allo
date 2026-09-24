#!/usr/bin/env bash
set -euo pipefail
for supply_name in "$lvs_power_name" "$lvs_ground_name"; do
  if [[ ! "$supply_name" =~ ^[A-Za-z_][A-Za-z0-9_$]*$ ]]; then
    echo "ERROR: invalid LVS supply net name: $supply_name" >&2
    exit 1
  fi
done
if [[ "$lvs_power_name" == "$lvs_ground_name" ]]; then
  echo "ERROR: LVS power and ground net names must differ" >&2
  exit 1
fi
if [[ -z "${lvs_hcells_file}" ]]; then export lvs_use_hcells=0; else export lvs_use_hcells=1; fi
if [[ -z "${lvs_connect_names}" ]]; then export lvs_connect_names_state=NONE; else export lvs_connect_names_state=SOME; fi
export lvs_spice_include=""
spice_files=()
for pattern in inputs/adk/*.cdl inputs/adk/*.spi inputs/adk/*.sp inputs/adk/*source.added inputs/srams/*/*.cdl inputs/srams/*/*.spi inputs/srams/*/*.sp; do
  for file in $pattern; do
    if [[ -e "$file" ]]; then
      lvs_spice_include="$lvs_spice_include $file"
      spice_files+=("$file")
    fi
  done
done
export lvs_spice_include
envsubst < lvs.runset.template > lvs.runset
[[ -f inputs/rules.svrf ]] || touch inputs/rules.svrf
cp inputs/design.lvs.v merged.lvs.v

# Convert the physical Verilog source explicitly so supply globalization is
# deterministic and independent of Calibre Interactive runset defaults.
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
  # This parameter is an intentional shell-style list of paths, matching the
  # inherited mflowgen LVS interface.
  read -r -a extra_spice_files <<< "$lvs_extra_spice_include"
  for file in "${extra_spice_files[@]}"; do
    [[ -f "$file" ]] || { echo "ERROR: missing extra LVS SPICE include: $file" >&2; exit 1; }
    v2lvs_args+=( -s "$file" )
  done
fi
v2lvs "${v2lvs_args[@]}" -log v2lvs.log
test -s source.lvs.sp
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_power_name}([[:space:]]|$)" source.lvs.sp || {
  echo "ERROR: V2LVS source does not declare global power net $lvs_power_name" >&2
  exit 1
}
grep -Eiq "^\\.GLOBAL.*(^|[[:space:]])${lvs_ground_name}([[:space:]]|$)" source.lvs.sp || {
  echo "ERROR: V2LVS source does not declare global ground net $lvs_ground_name" >&2
  exit 1
}
calibre -gui -lvs -batch -runset lvs.runset
test -s lvs.report
if grep -q '#     INCORRECT     #' lvs.report; then
  echo 'ERROR: Calibre LVS reported INCORRECT' >&2
  exit 1
fi

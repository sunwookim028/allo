#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${lvs_hcells_file}" ]]; then export lvs_use_hcells=0; else export lvs_use_hcells=1; fi
if [[ -z "${lvs_connect_names}" ]]; then export lvs_connect_names_state=NONE; else export lvs_connect_names_state=SOME; fi
export lvs_spice_include=""
for pattern in inputs/adk/*.cdl inputs/adk/*.spi inputs/adk/*.sp inputs/adk/*source.added inputs/srams/*/*.cdl inputs/srams/*/*.spi inputs/srams/*/*.sp; do
  for file in $pattern; do
    [[ ! -e "$file" ]] || lvs_spice_include="$lvs_spice_include $file"
  done
done
export lvs_spice_include
envsubst < lvs.runset.template > lvs.runset
[[ -f inputs/rules.svrf ]] || touch inputs/rules.svrf
cp inputs/design.lvs.v merged.lvs.v
calibre -gui -lvs -batch -runset lvs.runset
test -s lvs.report
if grep -q '#     INCORRECT     #' lvs.report; then
  echo 'ERROR: Calibre LVS reported INCORRECT' >&2
  exit 1
fi

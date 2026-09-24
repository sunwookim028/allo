#!/usr/bin/env bash
set -euo pipefail
if [[ -f "inputs/adk/${drc_env_setup}" ]]; then source "inputs/adk/${drc_env_setup}"; fi
envsubst < drc.runset.template > drc.runset
calibre -gui -drc -batch -runset drc.runset
test -s drc.results
test -s drc.summary

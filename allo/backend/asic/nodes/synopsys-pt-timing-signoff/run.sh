#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s); mkdir -p reports outputs
if [[ -f inputs/sram-contract.json ]]; then
  python3 inputs/resolve-sram-contract.py --contract inputs/sram-contract.json \
    --root inputs/srams --tcl-output sram-views.tcl
fi
pt_shell -file pt.tcl
ln -sf ../design.sdf outputs/design.sdf
ln -sfn ../reports outputs/timing-reports
python3 collect_timing_metrics.py --wall-seconds "$(( $(date +%s) - start ))" \
  --setup-target "$setup_target_slack" --hold-target "$hold_target_slack" \
  --policy "$timing_check_policy"

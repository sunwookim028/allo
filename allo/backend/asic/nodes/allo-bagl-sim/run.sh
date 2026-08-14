#!/usr/bin/env bash
set -euo pipefail

: "${testbench_name:=allo_generated_testbench}"
: "${dut_name:=dut}"
: "${sdf_corner:=typ}"
: "${waveform:=True}"
: "${sdf_warning_policy:=report}"
: "${sdf_unmatched_timingcheck_policy:=report}"
: "${sdf_unmatched_iopath_policy:=report}"
: "${sdf_uphier_interconnect_policy:=report}"

case "$sdf_corner" in typ|min|max) ;; *) echo "invalid sdf_corner: $sdf_corner" >&2; exit 2;; esac
case "$sdf_warning_policy" in report|error) ;; *) echo "invalid sdf_warning_policy: $sdf_warning_policy" >&2; exit 2;; esac
for policy in "$sdf_unmatched_timingcheck_policy" "$sdf_unmatched_iopath_policy" \
              "$sdf_uphier_interconnect_policy"; do
  case "$policy" in report|error) ;; *) echo "invalid SDF category policy: $policy" >&2; exit 2;; esac
done

mkdir -p outputs
rm -f outputs/run.vcd outputs/compile.log outputs/simulation.log \
      outputs/simulation-report.json

read -r clock_compensation input_delay output_delay reset_cycles \
  bfm_drive_delay < outputs/timing-config.values

sources=()
macro_model_count=0
for file in inputs/adk/*.v inputs/adk/*.sv; do
  [[ -e "$file" ]] && sources+=("$file")
done
if [[ -d inputs/srams ]]; then
  while IFS= read -r -d '' file; do sources+=("$file"); done \
    < <(find -L inputs/srams -type f \( -name '*.v' -o -name '*.sv' \) -print0 | sort -z)
fi
if [[ -d inputs/macro-registry ]]; then
  while IFS= read -r -d '' file; do
    sources+=("$file")
    macro_model_count=$((macro_model_count + 1))
  done \
    < <(find -L inputs/macro-registry -type f \
      \( -name '*.v' -o -name '*.sv' \) \
      ! -name '*.lvs.v' ! -name '*.pg.v' -print0 | sort -z)
fi
sources+=(inputs/design.vcs.v)
backend=$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' \
  inputs/testbench-contract.json backend vitis)
case "$backend" in
  vitis)
    sources+=(
      inputs/vitis-axi-memory-bfm.sv
      inputs/vitis-axilite-master-bfm.sv
    )
    ;;
  catapult) ;;
  *) echo "unsupported testbench backend: $backend" >&2; exit 2 ;;
esac
sources+=(inputs/testbench.sv)

sdf_args=(-sdf "$sdf_corner:${testbench_name}.${dut_name}:inputs/design.sdf")
while IFS=$'\t' read -r scope sdf_file; do
  [[ -n "$scope" ]] || continue
  sdf_args+=(-sdf "$sdf_corner:$scope:$sdf_file")
done < outputs/sdf-annotations.tsv

vcs -full64 -sverilog -xprop=tmerge -override_timescale=1ns/1ps \
  -debug_access+all \
  -top "$testbench_name" +neg_tchk +no_notifier +sdfverbose +define+NTC \
  "${sdf_args[@]}" \
  -o simv "${sources[@]}" 2>&1 | tee outputs/compile.log

run_args=()
if [[ "$waveform" == "True" ]]; then
  run_args+=("+ALLO_DUMP_VCD")
fi
run_args+=(
  "+ALLO_BAGL_CLK_INS_SRC_LAT_NS=$clock_compensation"
  "+ALLO_BAGL_BFM_DRIVE_DELAY_NS=$bfm_drive_delay"
  "+ALLO_BAGL_INPUT_DELAY_NS=$input_delay"
  "+ALLO_BAGL_OUTPUT_DELAY_NS=$output_delay"
  "+ALLO_BAGL_NUM_RESET_CYCLES=$reset_cycles"
)
./simv "${run_args[@]}" 2>&1 | tee outputs/simulation.log

python3 report_sim.py --compile-log outputs/compile.log \
  --simulation-log outputs/simulation.log --vcd outputs/run.vcd \
  --warning-policy "$sdf_warning_policy" \
  --unmatched-timingcheck-policy "$sdf_unmatched_timingcheck_policy" \
  --unmatched-iopath-policy "$sdf_unmatched_iopath_policy" \
  --uphier-interconnect-policy "$sdf_uphier_interconnect_policy" \
  --annotation-manifest outputs/sdf-annotation-manifest.json \
  --sdf-requested --macro-model-count "$macro_model_count" \
  --output outputs/simulation-report.json

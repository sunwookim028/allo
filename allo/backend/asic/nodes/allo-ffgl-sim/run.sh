#!/usr/bin/env bash
set -euo pipefail

: "${testbench_name:=allo_generated_testbench}"
: "${waveform:=True}"

mkdir -p outputs
rm -f outputs/run.vcd outputs/compile.log outputs/simulation.log \
      outputs/simulation-report.json

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
sources+=(inputs/design.v)
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
  catapult|systemc) ;;
  *) echo "unsupported testbench backend: $backend" >&2; exit 2 ;;
esac
sources+=(inputs/testbench.sv)

vcs -full64 -sverilog -xprop=tmerge -override_timescale=1ns/1ps \
  -top "$testbench_name" +delay_mode_zero +define+TETRAMAX \
  -o simv "${sources[@]}" \
  2>&1 | tee outputs/compile.log

run_args=()
if [[ "$waveform" == "True" ]]; then
  run_args+=("+ALLO_DUMP_VCD")
fi
./simv "${run_args[@]}" 2>&1 | tee outputs/simulation.log

python3 report_sim.py --mode ffgl --compile-log outputs/compile.log \
  --simulation-log outputs/simulation.log --vcd outputs/run.vcd \
  --macro-model-count "$macro_model_count" \
  --output outputs/simulation-report.json

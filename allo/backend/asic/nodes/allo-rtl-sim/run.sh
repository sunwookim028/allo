#!/usr/bin/env bash
set -euo pipefail

: "${testbench_name:=allo_generated_testbench}"
: "${waveform:=True}"

mkdir -p outputs
rm -f outputs/design.v outputs/run.vcd outputs/compile.log \
      outputs/simulation.log outputs/simulation-report.json

sources=()
if [[ -d inputs/srams ]]; then
  while IFS= read -r -d '' file; do sources+=("$file"); done \
    < <(find -L inputs/srams -type f \( -name '*.v' -o -name '*.sv' \) -print0 | sort -z)
fi
sources+=(
  inputs/design.v
  inputs/vitis-axi-memory-bfm.sv
  inputs/vitis-axilite-master-bfm.sv
  inputs/testbench.sv
)

vcs -full64 -sverilog -xprop=tmerge -override_timescale=1ns/1ps \
  -top "$testbench_name" -o simv "${sources[@]}" \
  2>&1 | tee outputs/compile.log

run_args=()
if [[ "$waveform" == "True" ]]; then
  run_args+=("+ALLO_DUMP_VCD")
fi
./simv "${run_args[@]}" 2>&1 | tee outputs/simulation.log

cp inputs/design.v outputs/design.v
python3 report_sim.py --compile-log outputs/compile.log \
  --simulation-log outputs/simulation.log --vcd outputs/run.vcd \
  --output outputs/simulation-report.json

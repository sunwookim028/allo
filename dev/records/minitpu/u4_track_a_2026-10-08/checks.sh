#!/bin/bash
# U4 track A: every verdict of the record, one log per unit/instance.
# Run from the worktree root after `source examples/minitpu/harness/env-zhang21.sh`
# and `export PYTHONPATH=$PWD`. $1: a writable project directory.
set -u
R=dev/records/minitpu/u4_track_a_2026-10-08
P=${1:-/tmp/u4_track_a_prj}
run() { local log=$1; shift; ( time "$ALLO_PYTHON" "$@" ) > "$R/$log" 2>&1; grep -h "UNIT-\|CONTRACT-\|DERIVED\|REFUSED\|ACCEPTED\|^RTL\|^real" "$R/$log"; }
B="--backend simulator --backend systemc --project $P"
run check_dma_addr_gen.log  -m examples.minitpu.harness.check dma_addr_gen $B
run check_agu_resolve.log   -m examples.minitpu.harness.check agu_resolve $B
run check_seq_decoder.log   -m examples.minitpu.harness.check seq_decoder $B
run check_vpu_adapter.log   -m examples.minitpu.harness.check vpu_adapter $B
run gated_contract_sim.log  -m examples.minitpu.units.vpu_adapter --gated-contract simulator
run gated_contract_sc.log   -m examples.minitpu.units.vpu_adapter --gated-contract systemc $P/gated
for i in lat2 lat1 lat3; do run check_scalar_agu_$i.log -m examples.minitpu.harness.check scalar_agu --inst $i $B; done
for i in iram fq fq_a8; do run check_fetch_$i.log -m examples.minitpu.harness.check fetch --inst $i --variant f1 $B; done
run check_fetch_iram_d12.log -m examples.minitpu.harness.check fetch --inst iram --variant f1_d12 $B
run check_loop_ctrl.log     -m examples.minitpu.harness.check loop_ctrl $B
run control_geometry.log    -m examples.minitpu.template.control_geometry

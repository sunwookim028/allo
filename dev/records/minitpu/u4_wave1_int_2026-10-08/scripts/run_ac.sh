#!/bin/bash
# U4 wave-1 integration: tracks A and C's checks once on the merged tree.
source /work/shared/users/phd/sk3463/scratch/u4int_out/env-sc.sh
set -u
R=/work/shared/users/phd/sk3463/scratch/u4int_out/logs/ac
P=/work/shared/users/phd/sk3463/scratch/u4int_out/prj_ac
mkdir -p $R $P
run() { local log=$1; shift; ( time "$ALLO_PYTHON" "$@" ) > "$R/$log" 2>&1; echo "== $log"; grep -h "UNIT-\|CONTRACT-\|DERIVED\|REFUSED\|ACCEPTED\|REPRODUCES\|FIXED\|^RTL\|MATCH\|DIFF\|^real" "$R/$log"; }
B="--backend simulator --backend systemc --project $P"
# ---- track A (its checks.sh, logs redirected)
run A_check_dma_addr_gen.log  -m examples.minitpu.harness.check dma_addr_gen $B
run A_check_agu_resolve.log   -m examples.minitpu.harness.check agu_resolve $B
run A_check_seq_decoder.log   -m examples.minitpu.harness.check seq_decoder $B
run A_check_vpu_adapter.log   -m examples.minitpu.harness.check vpu_adapter $B
run A_gated_contract_sim.log  -m examples.minitpu.units.vpu_adapter --gated-contract simulator
run A_gated_contract_sc.log   -m examples.minitpu.units.vpu_adapter --gated-contract systemc $P/gated
for i in lat2 lat1 lat3; do run A_check_scalar_agu_$i.log -m examples.minitpu.harness.check scalar_agu --inst $i $B; done
for i in iram fq fq_a8; do run A_check_fetch_$i.log -m examples.minitpu.harness.check fetch --inst $i --variant f1 $B; done
run A_check_fetch_iram_d12.log -m examples.minitpu.harness.check fetch --inst iram --variant f1_d12 $B
run A_check_loop_ctrl.log     -m examples.minitpu.harness.check loop_ctrl $B
run A_control_geometry.log    -m examples.minitpu.template.control_geometry
# ---- track C (its record's reproduce block)
run C_check_dma_desc_adapter.log -m examples.minitpu.harness.check dma_desc_adapter $B
for i in core o1 o4 bw; do run C_check_dma_$i.log -m examples.minitpu.harness.check dma --inst $i $B; done
for i in core o1 o4; do run C_check_dma_vmem_$i.log -m examples.minitpu.harness.check dma_vmem --inst $i $B; done
run C_dma_params.log -m examples.minitpu.units.dma_params
run C_selftimed.log -m examples.minitpu.units.dma_selftimed --inst core --inst o1 --inst o4 --backend simulator --backend systemc
run C_lim_reserved_names.log tests/limits/new_systemc_reserved_local_names.py
run C_lim_local_array_stack.log tests/limits/new_systemc_local_array_stack.py
echo ALLDONE

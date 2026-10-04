#!/bin/bash
# D-12 regression on the worktree: TinyTPU gates, systemc csim, systemc tests, harness checks.
T=/work/shared/users/phd/sk3463/scratch/wt-d12
SP=/tmp/claude-1772902/-work-shared-users-phd-sk3463-allo/3b1c24a0-f6d1-4c37-88e0-31db609cddf4/scratchpad
cd $T; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1; cd $T
export PYTHONPATH=$T
export TPU_MAXDIM=16   # the TinyTPU gates only; unset before pytest tests/act
run() { name=$1; shift; echo "== $name start $(date +%T)"; "$@" > "$SP/d12_reg_$name.log" 2>&1; echo "== $name exit $? $(date +%T): $(grep -v '^\[.*Allo\]' $SP/d12_reg_$name.log | tail -1 | cut -c1-300)"; }
cd $T/examples/tinytpu
run gen_isa $ALLO_PYTHON gen_isa.py --check
run lift $ALLO_PYTHON lift_units.py --check
run bench $ALLO_PYTHON bench_isa.py
run stress $ALLO_PYTHON stress_isa.py
run act $ALLO_PYTHON act_compile.py --gate
cd $T
run sccsim $ALLO_PYTHON examples/tinytpu/systemc_csim.py 3 --project $SP/d12_reg_sc.prj
grep -c 'wrong=0/16' $SP/d12_reg_sccsim.log; grep 'SYSTEMC CSIM' $SP/d12_reg_sccsim.log
run pytest_sc $ALLO_PYTHON -m pytest tests/dataflow/test_systemc*.py tests/test_memory.py tests/dataflow/test_compose_memory_ports.py -q
grep -E "passed|failed" $SP/d12_reg_pytest_sc.log | tail -2
unset TPU_MAXDIM
run pytest_act $ALLO_PYTHON -m pytest tests/act -q
grep -E "passed|failed" $SP/d12_reg_pytest_act.log | tail -1
run bf16_add $ALLO_PYTHON -m examples.minitpu.harness.check bf16_add --backend systemc --project $SP/d12_reg_prj
grep -E "^UNIT" $SP/d12_reg_bf16_add.log
echo "== all done $(date +%T)"

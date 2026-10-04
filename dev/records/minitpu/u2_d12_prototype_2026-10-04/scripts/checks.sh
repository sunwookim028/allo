#!/bin/bash
# The D-12 variants through the harness: simulator and SystemC csim, full joined traces.
T=/work/shared/users/phd/sk3463/scratch/wt-d12
SP=/tmp/claude-1772902/-work-shared-users-phd-sk3463-allo/3b1c24a0-f6d1-4c37-88e0-31db609cddf4/scratchpad
cd $T; source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1; cd $T
export PYTHONPATH=$T MINITPU_HARNESS_CACHE=/work/shared/users/phd/sk3463/scratch/d12/hcache
L=/work/shared/users/phd/sk3463/scratch/d12/logs; mkdir -p $L
go() { u=$1; i=$2; be=$3; v=$4; timeout 3000 $ALLO_PYTHON -X faulthandler -m examples.minitpu.harness.check $u --inst $i --backend $be --variant $v --project /work/shared/users/phd/sk3463/scratch/d12/prj > $L/${u}_${i}_${be}_${v}.log 2>&1; echo "== $u $i $be $v exit $?"; grep -E "^(UNIT|    )|Segmentation|Aborted|Error" $L/${u}_${i}_${be}_${v}.log | head -6 | cut -c1-500; }
go vpu_regfile w16 simulator d12_server &
go vpu_regfile w16 simulator d12_replica &
go vpu_word_array narrow simulator d12_server &
go vpu_word_array narrow systemc d12_server &
go vpu_word_array narrow systemc d12_server_wire &
go vpu_regfile w16 systemc d12_server &
wait
# the whole regfile variant set on systemc (the record's regression ask)
timeout 5400 $ALLO_PYTHON -m examples.minitpu.harness.check vpu_regfile --inst w16 --backend systemc --project /work/shared/users/phd/sk3463/scratch/d12/prj > $L/vpu_regfile_w16_systemc_all.log 2>&1; echo "== regfile systemc all exit $?"; grep -E "^UNIT" $L/vpu_regfile_w16_systemc_all.log | cut -c1-300
echo "== checks done"

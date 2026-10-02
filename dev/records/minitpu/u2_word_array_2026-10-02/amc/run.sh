#!/bin/bash
# AMC column (AMC's own env; see u1_bf16_add_amc/env.sh). From this dir.
source ../../u1_bf16_add_amc/env.sh
T=/work/shared/users/phd/sk3463/scratch/u2wa/trace_narrow.npz
mkdir -p /work/shared/users/phd/sk3463/scratch/u2wa/amc; cd /work/shared/users/phd/sk3463/scratch/u2wa/amc
H=/work/shared/users/phd/sk3463/scratch/wt-u2wa/dev/records/minitpu/u2_word_array_2026-10-02/amc
timeout 1200 python $H/wa_amc.py $T 256 llvm,amc none 2>&1 | grep -v "^\s*$" | tail -6
timeout 1200 python $H/wa_amc.py $T 256 amc pipeline 2>&1 | grep -v "^\s*$" | tail -6
timeout 1200 python $H/wa_amc.py $T 256 amc part 2>&1 | grep -v "^\s*$" | tail -8
timeout 1200 python $H/wa_amc.py $T 256 amc pipeline trace_rw 2>&1 | grep -v "^\s*$" | tail -6
timeout 2400 python $H/wa_amc.py $T 80609 amc pipeline 2>&1 | grep -v "^\s*$" | tail -4

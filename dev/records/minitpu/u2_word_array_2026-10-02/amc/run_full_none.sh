#!/bin/bash
source ../../u1_bf16_add_amc/env.sh
cd /work/shared/users/phd/sk3463/scratch/u2wa/amc
H=/work/shared/users/phd/sk3463/scratch/wt-u2wa/dev/records/minitpu/u2_word_array_2026-10-02/amc
timeout 3000 python $H/wa_amc.py /work/shared/users/phd/sk3463/scratch/u2wa/trace_narrow.npz 80609 amc none 2>&1 | grep -v "^\s*$" | tail -4

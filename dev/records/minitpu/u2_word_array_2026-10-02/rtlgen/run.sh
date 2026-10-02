#!/bin/bash
# RTLGen column (RTLGen's own env; see u1_bf16_add_rtlgen/env.sh). From this dir.
source ../../u1_bf16_add_rtlgen/env.sh
T=/work/shared/users/phd/sk3463/scratch/u2wa/trace_narrow.npz
cd /work/shared/users/phd/sk3463/scratch/u2wa/rtlgen 2>/dev/null || { mkdir -p /work/shared/users/phd/sk3463/scratch/u2wa/rtlgen; cd /work/shared/users/phd/sk3463/scratch/u2wa/rtlgen; }
H=/work/shared/users/phd/sk3463/scratch/wt-u2wa/dev/records/minitpu/u2_word_array_2026-10-02/rtlgen
for v in trace trace_part trace_rw; do
  timeout 2400 python $H/wa_rtlgen.py $T $v ${N:-80609} 2>&1 | grep -v "^\s*$" | tail -8
done

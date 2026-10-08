#!/bin/bash
# usage: run_lit.sh <worktree> ; one process per case (a case may abort the process)
cd $1; source $(conda info --base)/etc/profile.d/conda.sh; conda activate allo; export OMP_NUM_THREADS=8 PYTHONPATH=$1
for c in or_shift and_shift add_big add_lone_big cmp_shift int64_or ann_shift var_shift mul_big global_big aug_or n_u8 n_u8_wrap n_i32_shift31 n_i8_neg; do
  out=$(timeout 300 $CONDA_PREFIX/bin/python /work/shared/users/phd/sk3463/scratch/fix5/f1/probe_lit.py $c 2>&1)
  rc=$?
  line=$(echo "$out" | grep -E "^.{28} (OK |BAD|ERR)" | head -1)
  [ -z "$line" ] && line="$c: CRASH rc=$rc $(echo "$out" | grep -iE "error|assert" | grep -v "^#\|PLEASE" | head -1 | cut -c1-200)"
  echo "$line"
done

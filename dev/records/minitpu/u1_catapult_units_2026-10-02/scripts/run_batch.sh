#!/bin/bash
# usage: run_batch.sh <list> [parallel]   -- one Catapult project per line: <name> <unit> <variant> <emit_csyn.py args>
# Projects land in $S/<name>.prj; one line of outcome per run in $S/<list>.out.
W=/work/shared/users/phd/sk3463/scratch/wt-u1-cat2
S=/work/shared/users/phd/sk3463/scratch/u1_cat2
R=$W/dev/records/minitpu/u1_catapult_units_2026-10-02/scripts
cd $W && source examples/minitpu/harness/env-zhang21.sh >/dev/null 2>&1
export W S R
grep -v '^#' $1 | xargs -P ${2:-4} -L 1 bash -c 'name=$0; unit=$1; var=$2; shift 2; $ALLO_PYTHON $R/emit_csyn.py $unit $var $S/$name.prj "$@" > $S/$name.emit.log 2>&1; tail -1 $S/$name.emit.log | sed "s/^/$name: /"' >> $S/$(basename $1 .txt).out
echo BATCH_DONE >> $S/$(basename $1 .txt).out

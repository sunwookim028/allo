#!/bin/bash
# usage: condense.sh <name> <cmd...>   -- runs cmd, keeps the full log in $FULL (scratch) and the
# condensed log (verdicts, schedule, warnings, errors; tool chatter stripped) in $LOGS/<name>.log
name=$1; shift
FULL=${FULL:-/work/shared/users/phd/sk3463/scratch/u4e_out/full_logs}; mkdir -p "$FULL"
LOGS=${LOGS:-$(cd "$(dirname "$0")" && pwd)/logs}
start=$(date +%s)
"$@" > "$FULL/$name.log" 2>&1; rc=$?
grep -E "^(SCHED|COSIM|UNIT-|CONTRACT|WARN|ERROR|EXPORT|FAIL|BUILT|RAN|NOTE|ACCEPT|REFUSE|LOWER|    )|[Ee]rror|Assertion|Traceback|cycle time|infeasible|WARNING" "$FULL/$name.log" \
  | grep -v "ns INFO\|^ *\*\*\|Wno-\|^ccache\|verilator_includer" | sed 's#/work/shared/users/phd/sk3463#~#g' | head -300 > "$LOGS/$name.log"
echo "WALL $(( $(date +%s) - start ))s rc=$rc" >> "$LOGS/$name.log"
cat "$LOGS/$name.log"

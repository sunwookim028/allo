#!/bin/bash
# re-filter an existing full log: recondense.sh <name>  (keeps the WALL line)
FULL=${FULL:-/work/shared/users/phd/sk3463/scratch/u4e_out/full_logs}; LOGS=$(cd "$(dirname "$0")" && pwd)/logs
w=$(grep '^WALL' "$LOGS/$1.log")
grep -E "^(SCHED|COSIM|UNIT-|CONTRACT|RESULT|CHECK|TB-MEM|WARN|ERROR|EXPORT|FAIL|BUILT|RAN|NOTE|ACCEPT|REFUSE|LOWER|    )|[Ee]rror|Assertion|Traceback|cycle time|infeasible|WARNING" "$FULL/$1.log" \
  | grep -v "ns INFO\|^ *\*\*\|Wno-\|^ccache\|verilator_includer" | sed 's#/work/shared/users/phd/sk3463#~#g' | head -300 > "$LOGS/$1.log"
echo "$w" >> "$LOGS/$1.log"; cat "$LOGS/$1.log"

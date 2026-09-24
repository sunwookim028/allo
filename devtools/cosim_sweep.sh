#!/bin/bash
# =====================================================================================
# Run EVERY tests/dataflow design through systemc csim -> csynth -> Xcelium RTL cosim
# and record one verdict line per design.
#
#   ./devtools/cosim_sweep.sh                 # all tests
#   ./devtools/cosim_sweep.sh test_tiled_gemm.py test_systolic.py   # a subset
#
# Results stream to $OUT/cosim_results.txt as they finish, so it can be tailed live.
#
# HOW IT WORKS. The pytest plugin devtools/scpatch_cosim2.py intercepts df.build() and
# does all the work AT INTERCEPTION -- it does not depend on the test calling the returned
# module. It self-generates inputs from the region signature, runs systemc csim as the
# software golden, csynths, runs the RTL under Xcelium, diffs, then aborts the test. So a
# test that never runs its design still gets cosimmed.
#
# ONE TEST PER PROCESS is mandatory: several JIT-simulator builds in one interpreter
# CORE-DUMP (the OMP-teardown-vs-GC race). The loop below forks pytest per file; do not
# "optimise" it into a single pytest invocation.
#
# Expect 2-8 minutes per design. The full sweep is hours -- run it detached:
#   nohup ./devtools/cosim_sweep.sh > /dev/null 2>&1 &
# =====================================================================================
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$HERE")
source "$HERE/sysc_env.sh"

OUT=${OUT:-/scratch/cosim_work/sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export TMPDIR=/scratch/cosim_work            # Catapult writes a LOT; keep it off /home
mkdir -p "$TMPDIR"
# the plugin is imported by name, so its directory must be importable
export PYTHONPATH=$HERE:$PYTHONPATH
PER_TEST_TIMEOUT=${PER_TEST_TIMEOUT:-1500}

cd "$ROOT"
FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
  mapfile -t FILES < <(cd "$ROOT" && ls tests/dataflow/test_*.py)
fi

R="$OUT/cosim_results.txt"
{
  echo "# cosim sweep  started $(date)"
  echo "# allo        $(python -c 'import allo,os;print(os.path.dirname(allo.__file__))')"
  echo "# commit      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "# designs     ${#FILES[@]}   timeout ${PER_TEST_TIMEOUT}s each"
} > "$R"

echo "sweep -> $R  (${#FILES[@]} files)"
for f in "${FILES[@]}"; do
  [ -f "$ROOT/$f" ] || f="tests/dataflow/$f"
  [ -f "$ROOT/$f" ] || { echo "SKIP (no such file): $f" | tee -a "$R"; continue; }
  echo "== $(date +%H:%M:%S)  $f"
  timeout "$PER_TEST_TIMEOUT" python -u -m pytest "$f" \
      -s -q --tb=no -p no:cacheprovider -p scpatch_cosim2 \
      > "$OUT/$(basename "$f" .py).log" 2>&1
  rc=$?
  got=$(grep -h "COSIM_RESULT" "$OUT/$(basename "$f" .py).log")
  if [ -n "$got" ]; then
    printf '%s\n' "$got" | tee -a "$R"
  else
    # No verdict at all is NOT a pass -- say so explicitly rather than leaving a blank row.
    echo "COSIM_RESULT | NO_VERDICT       | $(basename "$f" .py)  (rc=$rc, see $OUT/$(basename "$f" .py).log)" | tee -a "$R"
  fi
done
echo "# DONE $(date)" >> "$R"

echo
echo "===== SUMMARY ====="
grep -h COSIM_RESULT "$R" | awk -F'|' '{gsub(/ /,"",$2); print $2}' | sort | uniq -c | sort -rn
echo
echo "full results: $R"

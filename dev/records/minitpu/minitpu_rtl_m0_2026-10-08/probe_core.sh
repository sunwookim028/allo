#!/usr/bin/env bash
# M-R0 probe: build minitpu_core (seam K, no AXI) standalone with PR #48's build-step flags
# (rtl.py:732-743 _rtl_command + :802-813 --cc --build ...), then lint variants and the AXI top / TB for contrast.
# Usage: probe_core.sh <exported MiniTPU tree at b3ba0a4d> <out dir>
set -uo pipefail
M=${1:?MiniTPU tree}; OUT=${2:?out dir}
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:/usr/bin:/bin
export PERL5LIB=/work/shared/users/phd/sk3463/tools/verilator/lib/perl5/core_perl   # what rtl.py:_tool sets
mkdir -p "$OUT"; cd "$M"
CORE=$(grep -v '^+incdir' src/core/core.f | sed "s#^#$M/#")          # core.f expanded, as rtl=[...] would be
TOP=$( (grep -v '^+incdir' src/core/core.f; grep -v '^-f' src/minitpu.f) | sed "s#^#$M/#")
run() {  # name, command...
  local name=$1; shift; local t0=$SECONDS
  "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  echo "$name rc=$rc wall=$((SECONDS - t0))s warnings=$(grep -c '^%Warning' "$OUT/$name.log") errors=$(grep -c '^%Error' "$OUT/$name.log")"
}
run core_pr48_exact verilator --top-module minitpu_core $CORE -I$M/src/pkg \
    --cc --build --Mdir "$OUT/vgen" -CFLAGS "-fPIC -fvisibility=hidden"
run core_lint_default   verilator --top-module minitpu_core $CORE -I$M/src/pkg --lint-only
run core_lint_wall      verilator --top-module minitpu_core $CORE -I$M/src/pkg --lint-only -Wall
run core_lint_timing    verilator --top-module minitpu_core $CORE -I$M/src/pkg --lint-only --timing
run core_lint_notiming  verilator --top-module minitpu_core $CORE -I$M/src/pkg --lint-only --no-timing
run core_json_validate  verilator --top-module minitpu_core $CORE -I$M/src/pkg --json-only -Wno-fatal \
    --json-only-output "$OUT/ports.tree.json" --Mdir "$OUT/json"
run top_lint_default    verilator --top-module minitpu $TOP -I$M/src/pkg --lint-only
run tb_lint_timing      verilator --top-module tb_kernel_image -f src/minitpu.f tb/tb_kernel_image.sv --lint-only --timing
# Smoke: construct the model, reset, clock 200 cycles, from a foreign cwd and from the tree root.
R=$(verilator --getenv VERILATOR_ROOT)
g++ -std=c++17 -O1 -I"$OUT/vgen" -I"$R/include" -I"$R/include/vltstd" "$(dirname "$0")/core_smoke.cpp" \
    "$OUT/vgen/libVminitpu_core.a" "$OUT/vgen/libverilated.a" -pthread -latomic -o "$OUT/core_smoke"
(cd "$OUT" && ./core_smoke) 2>&1 | sort | uniq -c | sed 's/^/foreign cwd: /'
(cd "$M" && "$OUT/core_smoke") 2>&1 | sed 's/^/tree root:   /'

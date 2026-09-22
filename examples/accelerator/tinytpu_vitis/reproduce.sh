#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# From a clean checkout to TinyTPU-isa's published cosim cycle counts, in one
# command. See --help for the options and for what the default does NOT run.
set -eo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
# At the SHIPPED default, MAXDIM=64, with no knob pinned -- which is new, and
# is the point of the DMA change that produced these numbers. `dma_ld` used to
# mirror whole DRAM rows on chip, so a small shape paid for the build it ran
# on (4x4x4 was 171 at MAXDIM=16 and 218 at MAXDIM=64); it now reads the T
# bytes an instruction names straight from `m_axi` at a stride the header
# carries, and the five shapes cost 178 / 262 / 416 / 478 / 696 at MAXDIM=64 --
# within eleven cycles of the MAXDIM=16 row and better than it at two shapes.
# So there is nothing left to pin. docs/source/designs/tinytpu_isa.rst,
# "Shapes that do not fit, and tiling them in".
#
# The MAXDIM=16 history below is kept because it is how the previous shift was
# attributed; those numbers are that configuration's, not this script's.
#
# WHY THEY MOVED BY ONE CYCLE, isolated to one variable rather than inferred.
# They were 172 / 262 / 418 / 484 / 686 while the scratchpad and vreg files
# were the literals 512 and 256; they are 171 / 261 / 417 / 483 / 685 now that
# both are DERIVED as MAXDIM^2/T, which is 64 rows each at MAXDIM=16.
# Rebuilding THIS configuration with `TPU_SPAD=512 TPU_NVR=256 TPU_NAR=128`
# and nothing else reverted -- the derived-size expression, both ceiling
# assertions, the test-window floor and the parametric burst loop all still
# present -- returns ALL FIVE of the old numbers exactly. So the memory sizing
# accounts for the entire shift and nothing else in that work touched cycles;
# in particular the parametric burst loop is cycle-neutral at DMA_WORDS=1,
# shown at MAXDIM=16 here and at MAXDIM=64 separately.
#
# The mechanism: a 64-row file is not implemented the way a 512-row one is --
# BRAM 42 -> 40 says two memories left block RAM -- and the shorter operand
# read path takes one cycle out of the FIXED term, which is why the delta is
# the same at every shape regardless of work. It is a small improvement.
EXPECTED="4x4x4=178 8x8x8=262 12x12x12=416 16x16x8=478 16x16x16=696"

usage() {
    cat <<'EOF'
reproduce.sh -- TinyTPU-isa's published cosim cycle counts from a clean checkout.

    reproduce.sh                              # everything
    reproduce.sh --no-cosim                   # functional only, no Vitis
    reproduce.sh --with-mutants               # everything + the mutation suite
    reproduce.sh --no-cosim --with-mutants    # functional + functional mutants

Stages, in order:

  1. builds THIS checkout's MLIR bindings in-tree (`allo/_mlir` is a symlink
     into `mlir/build`; a fresh checkout has none, and the editable install in
     the `allo` env points at whichever tree ran `pip install -e`), and makes
     `import allo` resolve to this checkout;
  2. bench_isa.py -- the published functional sweep, must print ALL EXACT;
  3. stress_isa.py -- the correctness gate, must print STRESS OK;
  3b. mutate.py -- ONLY with --with-mutants; must print MUTATE OK;
  4. cosim.py with the DEFAULT testbench and every TPU_* knob unset -- one
     csynth, one cosim per shape -- and compares the cycle counts with the
     published 178 / 262 / 416 / 478 / 696. Skipped by --no-cosim.

WHAT THE DEFAULT SKIPS. Stage 3b is off unless --with-mutants is given.
bench_isa.py and stress_isa.py both cite mutate.py as the evidence that they
catch a broken design, and that evidence is not produced by a default run: the
gates are run against the correct design only. mutate.py adds ~15 min, because
one mutant (ar_claim_false, a false `#pragma HLS dependence` claim) is caught
in RTL alone and needs a cosim of its own. With --no-cosim, stage 3b runs
`mutate.py --no-rtl`: the functional levels only, ~5 min, and the RTL-only
mutant is reported as not run rather than as caught.

Also not run by any option: `cosim.py TPU_TB=stress`, and
`chia_agent/param_check.py`.

Needs: the `allo` conda env, LLVM at $LLVM_BUILD_DIR (default in this script),
and Vitis HLS 2023.2 at the path `cosim.py` names in VITIS -- the last only for
stages 4 and 3b-with-RTL. Exits nonzero if any step fails or any number differs.

TIMING. A full run is ~6 min once mlir/build is warm. The incremental build
uses `getconf _NPROCESSORS_ONLN`, not `nproc`: this script exports
OMP_NUM_THREADS=8 and nproc honours it, so `-j $(nproc)` would build at -j8
however many cores the host has. The published ~6 min was measured that way.
EOF
}

NO_COSIM=0
WITH_MUTANTS=0
for arg in "$@"; do
    case "$arg" in
        --no-cosim)     NO_COSIM=1 ;;
        --with-mutants) WITH_MUTANTS=1 ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "reproduce.sh: unknown option '$arg'" >&2; usage >&2; exit 2 ;;
    esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allo
set -u                       # after activate: its scripts read unset vars
# The env sets neither of these (CLAUDE.md).
export LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-/home/sk3463/llvm-allo-6b09f739/build}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH=$ROOT
# A knob left in the caller's shell (TPU_SHAPES, TPU_AXI_LATENCY, TPU_TB, ...)
# would change what is measured. The published numbers are the defaults.
for v in $(env | grep -o '^TPU_[A-Z_]*' || true); do unset "$v"; done
# ...and nothing is pinned any more. MAXDIM used to be the DRAM row stride of
# every operand, so the same shape cost more on a bigger build and the
# published numbers had to name the build they were taken on; the stride is
# runtime data now, so the default configuration IS the published one.

PY=$(command -v python)
LOGS=$HERE/.scratch          # gitignored
mkdir -p "$LOGS"
cd "$ROOT"
if [ ! -f mlir/build/build.ninja ]; then
    echo "== configuring mlir/build (this checkout's bindings)"
    # Both Python3_ and Python_: CMake resolves them independently, and
    # nanobind takes its module suffix from the second (CLAUDE.md).
    cmake -G Ninja -S mlir -B mlir/build \
        -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
        -DPython3_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY" \
        -Dnanobind_DIR="$("$PY" -c 'import nanobind; print(nanobind.cmake_dir())')" \
        -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo > "$LOGS/mlir-cmake.log" 2>&1 \
        || { tail -20 "$LOGS/mlir-cmake.log"; exit 1; }
fi
echo "== building mlir/build (incremental)"
# NOT `nproc`: it honours OMP_NUM_THREADS, which this script has already
# exported as 8, so `-j $(nproc)` would build at -j8 on a 72-core host.
JOBS=$(getconf _NPROCESSORS_ONLN)
ninja -C mlir/build -j "$JOBS" > "$LOGS/mlir-ninja.log" 2>&1 || { tail -30 "$LOGS/mlir-ninja.log"; exit 1; }
where=$("$PY" -c 'import allo, os; print(os.path.realpath(allo.__file__))')
case "$where" in
    "$ROOT"/*) echo "   allo -> $where" ;;
    *) echo "allo resolves to $where, not this checkout ($ROOT)"; exit 1 ;;
esac

cd "$HERE"
echo "== bench_isa.py (published functional setup)"
out=$("$PY" bench_isa.py | tail -1); echo "$out"; grep -q "ALL EXACT" <<<"$out"
echo "== stress_isa.py (correctness gate)"
out=$("$PY" stress_isa.py | tail -1); echo "$out"; grep -q "STRESS OK" <<<"$out"
echo "== act_compile.py --gate (every mapping the search accepts, verified)"
out=$("$PY" act_compile.py --gate | tail -1); echo "$out"; grep -q "ACT GATE OK" <<<"$out"

if [ "$WITH_MUTANTS" = 1 ]; then
    # The evidence bench_isa.py and stress_isa.py cite for their own power.
    # --no-rtl drops the one mutant only cosim can catch (see --help).
    if [ "$NO_COSIM" = 1 ]; then
        echo "== mutate.py --no-rtl (does the gate catch a broken design?)"
        "$PY" mutate.py --no-rtl | tee "$LOGS/mutate-reproduce.log"
    else
        echo "== mutate.py (does the gate catch a broken design?)"
        "$PY" mutate.py | tee "$LOGS/mutate-reproduce.log"
    fi
    grep -q "MUTATE OK" "$LOGS/mutate-reproduce.log"
fi

[ "$NO_COSIM" = 1 ] && { echo "REPRODUCED (functional only)"; exit 0; }

echo "== cosim.py, default testbench (csynth once, then one cosim per shape)"
"$PY" cosim.py | tee "$LOGS/cosim-reproduce.log"
# The summary table: "  16x16x16   685", shape fields space-padded.
got=$(awk '/^  shape +cycles/{t=1; next} t && /^ +[0-9]/ {c=$NF; $NF=""; s=$0;
      gsub(/ /,"",s); printf "%s=%s ", s, c}' "$LOGS/cosim-reproduce.log" | sed 's/ $//')
echo "   expected: $EXPECTED"
echo "   got:      $got"
if [ "$got" = "$EXPECTED" ] && grep -q "COSIM OK" "$LOGS/cosim-reproduce.log"; then
    echo "REPRODUCED"
else
    echo "DIFFERS"; exit 1
fi

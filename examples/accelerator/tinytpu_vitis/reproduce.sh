#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# From a clean checkout to TinyTPU-isa's published cosim cycle counts, in one
# command:
#
#     examples/accelerator/tinytpu_vitis/reproduce.sh              # everything
#     examples/accelerator/tinytpu_vitis/reproduce.sh --no-cosim   # functional only
#
# 1. builds THIS checkout's MLIR bindings in-tree (`allo/_mlir` is a symlink
#    into `mlir/build`; a fresh checkout has none, and the editable install in
#    the `allo` env points at whichever tree ran `pip install -e`), and makes
#    `import allo` resolve to this checkout;
# 2. bench_isa.py -- the published functional sweep, must print ALL EXACT;
# 3. stress_isa.py -- the correctness gate, must print STRESS OK;
# 4. cosim.py with the DEFAULT testbench and every TPU_* knob unset -- one
#    csynth, one cosim per shape -- and compares the cycle counts with the
#    published 172 / 262 / 418 / 484 / 686.
#
# Needs: the `allo` conda env, LLVM at $LLVM_BUILD_DIR (default below), and
# Vitis HLS 2023.2 at the path `cosim.py` names in VITIS. ~6 min with
# cosim. Exits nonzero if any step fails or any number differs.
set -eo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
EXPECTED="4x4x4=172 8x8x8=262 12x12x12=418 16x16x8=484 16x16x16=686"

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
ninja -C mlir/build -j "$(nproc)" > "$LOGS/mlir-ninja.log" 2>&1 || { tail -30 "$LOGS/mlir-ninja.log"; exit 1; }
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

[ "${1:-}" = "--no-cosim" ] && { echo "REPRODUCED (functional only)"; exit 0; }

echo "== cosim.py, default testbench (csynth once, then one cosim per shape)"
"$PY" cosim.py | tee "$LOGS/cosim-reproduce.log"
# The summary table: "  16x16x16   686", shape fields space-padded.
got=$(awk '/^  shape +cycles/{t=1; next} t && /^ +[0-9]/ {c=$NF; $NF=""; s=$0;
      gsub(/ /,"",s); printf "%s=%s ", s, c}' "$LOGS/cosim-reproduce.log" | sed 's/ $//')
echo "   expected: $EXPECTED"
echo "   got:      $got"
if [ "$got" = "$EXPECTED" ] && grep -q "COSIM OK" "$LOGS/cosim-reproduce.log"; then
    echo "REPRODUCED"
else
    echo "DIFFERS"; exit 1
fi

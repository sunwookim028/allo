#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Re-derive every number the co-design loop reports, from a clean checkout of
# this branch. No model call, no cost. Three stages, each independently useful:
#
#   mapspace  the exhaustive enumeration and the refusal histogram, on the
#             shipped hardware. Pure python, ~2 s, no Vitis.
#   control   the CO-DESIGN baseline: gate + mapspace gate + RTL cosim of the
#             nest the frozen mapper chose. ~3 min, needs Vitis HLS 2023.2.
#   suite     the whole LLM-free test suite (test_codesign.py), ~7 min.
#
# A candidate's own numbers are re-derived with, from the repository root:
#
#   python examples/accelerator/tinytpu_vitis/chia_agent/evaluate.py \
#       --spec-dir <run>/<worker>/spec --work "$PWD/.chia_scratch/redo" --codesign
#
# and a claim is re-verified at all five shapes on a clean checkout with
#
#   python examples/accelerator/tinytpu_vitis/chia_agent/accept.py --codesign \
#       --diff <run>/<worker>/best.diff --out <run>/accept-<worker>
#
# The Vitis project is deleted at the end of each stage: disk on this host is
# tight, and `evaluate.py` wipes its work directory before every run anyway, so
# a stale report can never be read as a result.
set -euo pipefail

STAGE="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
AGENT="$HERE/chia_agent"
WORK="$REPO/.chia_scratch/reproduce-codesign"

: "${LLVM_BUILD_DIR:?set LLVM_BUILD_DIR (the conda env does not); see CLAUDE.md}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

cd "$REPO"
python -c 'import allo, os, sys; p = os.path.realpath(allo.__file__)
assert p.startswith(os.path.realpath(sys.argv[1])), p' "$REPO" \
  || { echo "allo does not import from this checkout; build mlir/ first:"; \
       echo "  cmake -G Ninja -S mlir -B mlir/build \\"; \
       echo "    -DMLIR_DIR=\$LLVM_BUILD_DIR/lib/cmake/mlir \\"; \
       echo "    -DPython3_EXECUTABLE=\$(which python) \\"; \
       echo "    -DPython_EXECUTABLE=\$(which python) \\"; \
       echo "    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo && ninja -C mlir/build"; \
       exit 1; }

df -BG /home | tail -1

mkdir -p "$WORK/spec"
for f in microarch_isa.py isa_dsl.py; do
  git show "HEAD:examples/accelerator/tinytpu_vitis/$f" > "$WORK/spec/$f"
done

if [[ "$STAGE" == all || "$STAGE" == mapspace ]]; then
  echo "=== mapspace: the exhaustive enumeration, shipped hardware ==========="
  # The refusal histogram. Nothing here is a cycle count.
  PYTHONPATH="$REPO" python "$AGENT/codesign_gate.py" \
      4x4x4,8x8x8,12x12x12,16x16x8,16x16x16
fi

if [[ "$STAGE" == all || "$STAGE" == control ]]; then
  echo "=== control: the co-design baseline, measured in this run ============"
  # Expect cycles {"4x4x4": 169, "16x16x16": 686}. 686 is main @ 476a70d8's
  # published number, because at 16x16x16 the mapper's pick IS the shipped
  # nest; 169 against the published 172 is the trip-count-1 loop the mapper
  # drops at 4x4x4 (two fewer static instructions, same dynamic issues).
  python "$AGENT/evaluate.py" --spec-dir "$WORK/spec" \
      --work "$WORK/eval" --codesign
  rm -rf "$WORK/eval"
fi

if [[ "$STAGE" == all || "$STAGE" == suite ]]; then
  echo "=== suite: the LLM-free test suite, \$0 =============================="
  python "$AGENT/test_codesign.py" --run-dir "$WORK/suite"
  rm -rf "$WORK/suite/work"
fi

rm -rf "$WORK/spec"
df -BG /home | tail -1

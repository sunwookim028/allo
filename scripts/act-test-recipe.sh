#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Recipe for re-measuring Kai Shao's ACT test suite (`tests/dsa/`, 28 files).
# ACT is not on `main`; it lives at tag `chia-codesign-final` (629c2767).
# See ATTRIBUTION.md at that tag, and docs/source/extensions/act.rst.
#
# This script does two things and nothing else:
#   * `probe`  (default) -- check every precondition and print a verdict. Read
#                           only; touches no build, installs nothing.
#   * `run`    -- run `pytest tests/dsa/` if, and only if, `probe` is green.
#
# Written this way on purpose: as of 2026-09-22 the preconditions are NOT met on
# this host, and the point of the recipe is to say exactly which ones and why,
# rather than to leave an uncheckable number standing. Re-run `probe` after a
# migration or a rebuild; if it goes green, `run` produces the citable figure.

set -u
MODE="${1:-probe}"
TAG=chia-codesign-final
WT="${ACT_WT:-$HOME/allo-act-chia}"
PY="${ACT_PY:-$HOME/miniconda3/envs/allo/bin/python}"
MISSING=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  ok      %s\n' "$*"; }
bad()  { printf '  MISSING %s\n' "$*"; MISSING=$((MISSING + 1)); }

say "== ACT test-suite preconditions =="
say

# 1. A worktree at the tag. Read-only on anyone else's tree; make your own.
#      git -C <repo> worktree add --detach "$WT" $TAG      (92 MB)
if [ -d "$WT/allo/exp/dsa" ]; then
  ok "worktree with allo/exp/dsa at $WT"
else
  bad "worktree at $TAG: git worktree add --detach $WT $TAG"
fi

# 2. Pure-python deps. The `allo` conda env (python 3.12) has all four.
#    NOTE: `chia_env` does NOT -- it is the CHIA agent's Ray/GenAI env
#    (python 3.10.19, ray + google-genai + mcp, no numpy and no pytest).
for m in numpy sympy ml_dtypes pytest; do
  if "$PY" -c "import $m" 2>/dev/null; then ok "python dep $m ($PY)"
  else bad "python dep $m in $PY"; fi
done

# 3. torch_mlir. Absent from every env on this host, so the torch-backed tests
#    skip. They assert nothing when skipped, including every numeric cost-model
#    assertion, so a green run with them skipped is not evidence about them.
if "$PY" -c "import torch_mlir" 2>/dev/null; then ok "torch_mlir (torch tests will RUN)"
else say "  note    torch_mlir absent -- the torch-backed tests will SKIP"; fi

# 4. The bindings. `allo/_mlir` is a gitignored symlink to
#    mlir/build/tools/allo/_mlir, produced by building THIS tree's mlir/.
#    It must be built from the chia lineage: ACT reaches `allo._mlir.schedule`
#    (mlir/python/allo/schedule.py) and the `allo` dialect's ISA ops/types
#    (mlir/{include,lib}/allo/IR/AlloISA{Ops,Types}.*), neither of which exists
#    in `main`'s mlir/ tree. Pointing the symlink at main's build fails at
#    `ModuleNotFoundError: No module named 'allo._mlir.schedule'`.
if [ -e "$WT/allo/_mlir/schedule.py" ] || [ -e "$WT/allo/_mlir/schedule/__init__.py" ]; then
  ok "allo/_mlir built from this tree (has .schedule)"
else
  bad "allo/_mlir built from $TAG's mlir/ -- see steps 5-7 below"
fi

# 5. LLVM at the commit THIS tree pins. `mlir/CMakeLists.txt` reads
#    $LLVM_BASE_DIR (note: LLVM_BASE_DIR, *not* the LLVM_BUILD_DIR that main's
#    dataflow simulator wants) or -DLLVM_DIR/-DMLIR_DIR.
#      chia pin: externals/llvm-project 040a641988f6ed6f4fab250706ca2b620c1de2d8
#      main pin: externals/llvm-project 6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb
#    The 11 GB Release build on this host ($HOME/llvm-allo-6b09f739) is main's
#    pin, so it is the wrong commit for this tree.
LLVM_PIN=040a641988f6ed6f4fab250706ca2b620c1de2d8
if [ -d "${LLVM_BASE_DIR:-/nonexistent}/lib/cmake/mlir" ]; then
  ok "MLIR cmake package at \$LLVM_BASE_DIR ($LLVM_BASE_DIR) -- verify it is $LLVM_PIN"
else
  bad "\$LLVM_BASE_DIR with lib/cmake/mlir, built from llvm-project @ $LLVM_PIN"
fi

# 6. CIRCT. `mlir/CMakeLists.txt:17-23,35` is a hard FATAL_ERROR: there is no
#    option to build without it, and `MLIRAlloRegisterEverything` -- which the
#    python bindings link -- lists CIRCTHW/CIRCTComb/CIRCTSeq plus
#    MLIRAlloMicroarch and MLIRAlloScheduling, so it is not separable.
#      externals/circt pin: af5369d7ea19dafe8a48d58fa6577e80cde0e883
#      bootstrap: scripts/build-circt.sh (also builds OR-Tools, ~2.3 GB total)
if [ -n "${CIRCT_DIR:-}" ] && [ -f "$CIRCT_DIR/CIRCTConfig.cmake" ]; then
  ok "CIRCT cmake package at \$CIRCT_DIR"
elif [ -f "$WT/externals/circt/build/lib/cmake/circt/CIRCTConfig.cmake" ]; then
  ok "CIRCT cmake package in-tree"
else
  bad "CIRCT build (externals/circt @ af5369d, scripts/build-circt.sh)"
fi

# 7. OR-Tools *cmake package*. `mlir/CMakeLists.txt:27-33` FATAL_ERRORs without
#    it. $HOME/chia-ortools is NOT this: it ships runtime .so files only
#    (libortools.so.9 + a one-symbol shim) to keep an ALREADY-BUILT tree
#    importable. It cannot configure a build.
#    (`cmake --find-package` writes CMakeFiles/ into its cwd, so probe in /tmp.)
_probe=$(mktemp -d) && if (cd "$_probe" && "$(command -v cmake || echo /bin/false)" \
     --find-package -DNAME=ortools -DCOMPILER_ID=GNU -DLANGUAGE=CXX \
     -DMODE=EXIST) >/dev/null 2>&1; then
  ok "ortools cmake package findable"
else
  bad "ortools cmake package (lib/cmake/ortools on \$CMAKE_PREFIX_PATH)"
fi
rm -rf "$_probe"

say
if [ "$MISSING" -gt 0 ]; then
  say "VERDICT: $MISSING precondition(s) missing -- the suite CANNOT be run here."
  say "Do not cite a pass/skip figure. The last figure that was produced with a"
  say "committed provenance is 238 passed / 40 skipped / 0 failed at commit"
  say "cee57bc0 (2026-09-07). A later 243/0/35 circulates but exists nowhere in"
  say "git -- only in \$HOME/chia-ortools/README.txt, untracked -- so it is not"
  say "citable either."
  say
  say "Static census of the suite, which IS measurable without a build:"
  if [ -d "$WT/tests/dsa" ]; then
    printf '  %s test functions, %s files (parametrisation makes the collected count higher)\n' \
      "$(grep -rhc '^def test_' "$WT"/tests/dsa/*.py | paste -sd+ | bc)" \
      "$(ls -1 "$WT"/tests/dsa/*.py | wc -l)"
    printf '  %s pytest.importorskip("torch*") gates\n' \
      "$(grep -rho 'importorskip("torch[^"]*")' "$WT"/tests/dsa/*.py | wc -l)"
  fi
  exit 1
fi

say "VERDICT: preconditions met."
[ "$MODE" = run ] || { say "Re-run with 'run' to measure."; exit 0; }

cd "$WT" || exit 1
say
say "== pytest tests/dsa/ =="
PYTHONPATH="$WT" "$PY" -m pytest tests/dsa/ -q

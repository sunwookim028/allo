#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every output docs/source/designs/tinytpu_tutorial.rst shows, from
# the runs the page tells the reader to make. The fused-instruction patches are
# applied for the last run and reverted however the script exits.
set -euo pipefail
cd "$(dirname "$0")/.."                      # examples/tinytpu
OUT=../../docs/source/designs/tinytpu_tutorial
PATCHES=(tutorial/mvoutrelu/*.patch)
PYTHON=${PYTHON:-python}
mkdir -p "$OUT"

git apply --check "${PATCHES[@]}" || {
    echo "the mvoutrelu patches do not apply: start from an unpatched tree" >&2
    exit 1; }

run() {  # run <output file> <make arguments...>
    local file=$1; shift
    echo "== $file"
    make -s "$@" > "$OUT/$file" || true      # a refusal exits 1 by design
}

run baseline.txt mlp
run refusal.txt mlp MODEL=mlp_bias
TPU_T=8 run t8.txt mlp

git apply "${PATCHES[@]}"
trap 'git apply -R "${PATCHES[@]}"; $PYTHON gen_isa.py --write > /dev/null' EXIT
$PYTHON gen_isa.py --write > "$OUT/gen_isa.txt"
run mvoutrelu.txt mlp
echo "outputs in $OUT"

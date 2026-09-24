#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch model -> mapping -> cycles -> area, as far as this machine can go.
#
# This is the gate for the one claim no single-flow gate can make. Every error
# in this work has lived in the seams BETWEEN flows -- a configuration claimed
# but not run, a row measured on a different design, an area figure placed
# beside cycles from other RTL -- and each flow on its own is internally
# consistent, so each flow's own gate passes while the join is wrong.
#
# IT IS NOT ONE COMMAND, AND IT MUST NOT BE WRITTEN AS THOUGH IT WERE.
# Area needs a Synopsys Design Compiler licence, the pinned mflowgen and sv2v,
# and about 70 minutes of a machine outside this repository's reach. So the
# gate has two tiers and says which one it ran.
#
#   LOCAL TIER   (this script, ~1 min, no licence, anyone)
#     1. PyTorch nn.Module -> torch.fx + ShapeProp -> per-layer workload specs
#     2. specs -> the ACT mapper -> a TinyTPU program per layer
#     3. programs -> isa_ref and PyTorch -> bit-exact, or it fails
#     4. models -> cycles, and WHICH models carry a measured number
#     5. the join: which cycle count may stand beside which area figure
#     6. every area figure quoted in the docs traces to a committed report
#
#   REMOTE TIER  (printed, not run: it needs the licence)
#     7. RTL export -> sv2v -> DC -> area, then extract_results.py
#
# The join in step 5 is the part that matters, and it is enforced whether or
# not the remote tier ever runs: it refuses to pair a cycle count with an area
# figure from a different configuration, and it treats a configuration key that
# is absent as "cannot pair" rather than as "matches".
#
#   e2e_gate.sh              local tier, then print what the remote tier needs
#   e2e_gate.sh --simulator  also run every program on the built design
#   e2e_gate.sh --preflight  also ask preflight.py what this machine is missing
#
# Exit 0 only if every local-tier gate passes. A missing or unreadable input is
# a failure, not a skip -- four instruments here were caught reporting success
# without having run.

set -uo pipefail

# The checkout root, found by searching upward for a marker. NOT by counting
# levels up from $0: that has broken four times here, and the tree has been
# reorganised under these files.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while [ ! -f "$ROOT/pyproject.toml" ] || [ ! -d "$ROOT/allo" ]; do
    parent=$(dirname "$ROOT")
    if [ "$parent" = "$ROOT" ]; then
        echo "not inside an allo checkout (no pyproject.toml above $(dirname "${BASH_SOURCE[0]}"))" >&2
        exit 1
    fi
    ROOT=$parent
done

DESIGN=$ROOT/examples/tinytpu
REPORTS=$DESIGN/asic_synthesis/reports
EXPORTS=$ROOT/dev/records/tinytpu/rtl_handoff
PAIRINGS=$DESIGN/asic_synthesis/pairings.json

SIMULATOR=""
PREFLIGHT=""
for arg in "$@"; do
    case "$arg" in
        --simulator) SIMULATOR="--simulator" ;;
        --preflight) PREFLIGHT=1 ;;
        -h|--help) sed -n '4,42p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allo
set -u                       # after activate: its scripts read unset vars
export LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-/home/sk3463/llvm-allo-6b09f739/build}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH=$ROOT

# A TPU_* knob left in the caller's shell would change the configuration
# without changing the command, which is this project's most-repeated error.
# The workload gate reports the configuration it ran at, and it must be the one
# the claims were measured on.
for v in $(env | grep -o '^TPU_[A-Z_]*' || true); do unset "$v"; done

PY=$(command -v python)
fails=0
step() { printf '\n== %s\n' "$1"; }

step "1-4  LOCAL TIER: PyTorch -> specs -> mapping -> verified cycles"
echo "     (torch.fx + ShapeProp, the ACT mapper, isa_ref, and the claims ladder)"
"$PY" "$DESIGN/workloads/gate.py" $SIMULATOR || fails=$((fails + 1))

step "5    THE JOIN: which cycle count may stand beside which area figure"
echo "     This is the seam. It runs with or without a licence."
"$PY" "$ROOT/allo/backend/asic/tools/check_pairing.py" \
    --reports "$REPORTS" --exports "$EXPORTS" --pairings "$PAIRINGS" \
    || fails=$((fails + 1))

step "6    every area figure quoted in the docs traces to a committed report"
"$PY" "$ROOT/allo/backend/asic/tools/check_numbers.py" \
    --reports "$REPORTS" --quiet || fails=$((fails + 1))

step "7    REMOTE TIER: not run here, and not claimed"
cat <<'EOF'
     Area needs Synopsys Design Compiler. This repository has the RTL exports,
     the flow, the vendored nodes, the ADK definition and the stdcells.db
     checksum; it does not have, and cannot have, the licence.

     What the remote tier is, on a machine that has one:

       python allo/backend/asic/tools/preflight.py \
           --design examples/tinytpu/asic_synthesis \
           --exports dev/records/tinytpu/rtl_handoff
       # ... then the sequence it prints, ~70 min per variant ...
       python allo/backend/asic/tools/extract_results.py --reports examples/tinytpu/asic_synthesis/reports

     WHAT THIS GATE CANNOT COVER WITHOUT THAT LICENCE, stated rather than
     omitted:

     * that any area figure is reproducible. Every committed area here is
       trusted as recorded; nothing local re-derives it. extract_results.py
       --check proves only that results.json still matches the reports beside
       it, not that a rerun would produce those reports.
     * that a NEW configuration has an area at all. The join can refuse a bad
       pair without a licence and it cannot manufacture a missing one, so the
       end-to-end claim for a configuration nobody has synthesised ends at
       cycles -- which is what the refusals in pairings.json record.
     * that timing closed. A DC node reports status "passed" and returncode 0
       on a design that misses timing, because its postcondition checks that
       synthesis COMPLETED. check_numbers.py reads violating_paths out of the
       committed results.json for that reason, and one run here does miss.
EOF

if [ "$PREFLIGHT" = "1" ]; then
    step "     preflight: what this machine is missing"
    "$PY" "$ROOT/allo/backend/asic/tools/preflight.py" \
        --design "$DESIGN/asic_synthesis" --exports "$EXPORTS" || true
    echo "     (a non-zero preflight is expected without a licence and does"
    echo "      not fail this gate; it is information, not a verdict)"
fi

printf '\n'
if [ "$fails" -eq 0 ]; then
    echo "E2E GATE OK -- LOCAL TIER ONLY."
    echo "  PyTorch model -> mapping -> verified cycles is enforced end to end."
    echo "  The cycles/area join is enforced: every committed area figure is"
    echo "  either paired with a cycle count from the same configuration or"
    echo "  explicitly refused with a reason."
    echo "  Area itself was NOT reproduced here and is not claimed to have been."
    exit 0
fi
echo "E2E GATE FAILED: $fails local-tier gate(s) did not pass. Nothing about"
echo "the remote tier is claimed either way."
exit 1

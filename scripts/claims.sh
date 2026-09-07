#!/usr/bin/env bash
# Run the project's claims and report which hold. Three tiers:
#
#   --fast   no synthesis, no API. Checks the claims that are pure software.
#   (none)   adds Vitis HLS: the QoR claims and the replayed agent results.
#   --full   adds the backend scaffolds and one billed CHIA agent round-trip.
#
# Every tier prints its own wall time so the README's numbers stay honest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f chia.env ] || { echo "missing chia.env (copy scripts/chia.env.example)"; exit 1; }
# shellcheck disable=SC1091
source chia.env

TIER="${1:-default}"
RUN_DIR="${CLAIMS_RUN_DIR:-chia_runs/swarm-20260905-063857}"
PASS=0; FAIL=0
step () {  # step <id> <description> <command...>
  local id="$1" desc="$2"; shift 2
  local t0 rc; t0=$(date +%s)
  printf '  %-6s %-52s ' "$id" "$desc"
  if "$@" >"/tmp/claims_$id.log" 2>&1; then
    printf 'PASS  %3ss\n' "$(( $(date +%s) - t0 ))"; PASS=$((PASS+1))
  else
    printf 'FAIL  %3ss  (/tmp/claims_%s.log)\n' "$(( $(date +%s) - t0 ))" "$id"; FAIL=$((FAIL+1))
  fi
}
TINYTPU_ENV="${TINYTPU_ENV:-allo}"
allo () { conda run -n "$TINYTPU_ENV" "$@"; }

# An editable install resolves `allo` through a meta-path finder that outranks
# PYTHONPATH, so an environment built from another checkout imports THAT
# checkout -- and every claim below would pass while testing someone else's
# code. Refuse to run rather than report a green result for the wrong tree.
RESOLVED="$(allo python -c 'import allo,os;print(os.path.realpath(allo.__file__))' 2>/dev/null | tail -1)"
case "$RESOLVED" in
  "$(cd "$ROOT" && pwd -P)"/*) : ;;
  *) cat >&2 <<MSG
FATAL: conda env '$TINYTPU_ENV' imports allo from
    $RESOLVED
which is not inside this checkout
    $(cd "$ROOT" && pwd -P)
Every claim would pass while exercising a different tree. Build this checkout
into that environment, or set TINYTPU_ENV to the one you built it into.
MSG
     exit 2 ;;
esac
echo "  allo resolves to $RESOLVED"

START=$(date +%s)
echo "== Idea 2: the verifier must be the real tool"
step C2.3 "derived (ii,depth) reproduces measured latency" \
  allo python -m pytest tests/dsa/test_tinytpu_synth.py -q
step C2.1 "frozen cost model scores 22,160 cycles" \
  allo python -m examples.accelerator.tinytpu.ppa --frozen

echo "== Idea 3: agent-editable code is an execution surface"
step C3.1 "self-modifying spec refused; real spec accepted" \
  allo python -m pytest tests/dsa/test_tinytpu_agent_policy.py -q

if [ "$TIER" != "--fast" ]; then
  # shellcheck disable=SC1090
  source "$TINYTPU_VITIS_SETTINGS" >/dev/null
  echo "== Idea 2 (with synthesis)"
  step C2.2 "synthesis measures mxu=72 II=2, dma_load depth 75" \
    allo python -m examples.accelerator.tinytpu.synth --project /tmp/claims_synth
  echo "== Idea 4: breadth beats depth"
  step C4.1 "replay the 4.07x variant and re-derive its score" \
    allo python -m examples.accelerator.tinytpu.verify_variant --run "$RUN_DIR" --worker dram
  step C4.2 "replay the 1.98x variant from another hypothesis" \
    allo python -m examples.accelerator.tinytpu.verify_variant --run "$RUN_DIR" --worker granularity-retry
fi

if [ "$TIER" = "--full" ]; then
  echo "== Idea 1: one specification, three artifacts"
  step C1.1 "same schedule lowers to CPU, Vitis HLS, and RTL" \
    make -C examples/accelerator/tinytpu ENV="$TINYTPU_ENV" oracle compiler cpu hls rtl
  echo "== Idea 5: evaluation is cheap relative to proposal  (billed)"
  step C5.0 "CHIA round-trip: ADC -> Vertex -> opencode -> MCP -> allo" \
    conda run -n "${TINYTPU_CHIA_ENV:-chia_env}" python examples/accelerator/tinytpu/chia_agent/smoke.py
fi

echo
echo "  $PASS passed, $FAIL failed, $(( $(date +%s) - START ))s total (tier: $TIER)"
exit $(( FAIL > 0 ))

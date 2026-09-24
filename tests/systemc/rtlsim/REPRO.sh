#!/bin/bash
# ============================================================================
# Wire verification via RTL simulation of Catapult's own netlist (xsim).
#
# Sources (read-only, from the choonsik1/allo history, branch SystemC-emitter --
# there is no choonsik1/SystemC-emitter repository.  The designs were deleted from
# the branch tip but the netlists survive):
#   git show 0eff4888:agents/noc/rtl/{pe_wire,pe_stream,pe_channel}/rtl.v
#   git show 779e4350^:agents/noc/pe_split.py      (same pe_wire/rtl.v hash)
#
# Reference model: C[i] = sum_{j<=i} A[j]*B[j], A=1..8, B=2..16 step 2
#                  -> 2 10 28 60 110 182 280 408   (numpy cumsum, hardcoded in the tb)
#
# Prereqs: Vivado 2023.2 xsim on PATH (source settings64.sh), or Xcelium with
#          RUNNER=run_mulacc_xrun.sh. No SystemC needed.
# Runtime: ~6 s per case.
# ============================================================================
set -u
S="$(cd "$(dirname "$0")" && pwd)"
r() { printf '%-58s ' "$1"; shift
      v=$("$S/${RUNNER:-run_mulacc.sh}" "$@" 2>&1 | grep -oE 'PASS|FAIL .*')
      echo "${v:-NO VERDICT (the runner produced neither PASS nor FAIL)}" ; }

echo "--- 1. the three boundaries, no pacing games -------------------------------"
r "Stream[int32,2] boundary (control)"          pe_stream
r "Channel[valid_ready] boundary (control)"     pe_channel
r "Wire[int32] boundary (DUT)"                  pe_wire

echo "--- 2. Wire under every producer/consumer pacing ---------------------------"
for si in 1 2 3 4 5 6; do for so in 1 2 3; do
  r "Wire  stall_in=$si stall_out=$so"          pe_wire -d STALL_IN=$si -d STALL_OUT=$so
done; done

# NOTE: LOCKSTEP (here and in the BREAK_WIRE fault below) pokes
# tb.u_mul.mul_0_run_inst.v8_and_cse, which exists only in the 0eff4888 netlists.
# With RTLDIR pointed at guard_experiment/rtl_{base,guard} these rows do not
# elaborate and come out as NO VERDICT -- see README.md.
echo "--- 3. positive control: same Wire RTL, lockstep imposed by the tb ---------"
for d in 2 3 4 5; do
  r "Wire + LOCKSTEP, acc released $d cycles late" pe_wire -d LOCKSTEP -d ACC_RST_DELAY=$d
done

echo "--- 4. deliberate breakage: each of these MUST go red ----------------------"
r "Wire+LOCKSTEP, 1 cycle of latency in the wire" pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3 -d BREAK_WIRE
r "Wire+LOCKSTEP, boundary datum xor 1"           pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3 -d BREAK_DATA
r "Channel,        boundary datum xor 1"          pe_channel -d BREAK_DATA
r "Stream,         boundary datum xor 1"          pe_stream  -d BREAK_DATA

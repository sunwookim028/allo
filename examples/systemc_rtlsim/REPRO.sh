#!/bin/bash
# ============================================================================
# Wire verification via RTL simulation of Catapult's own netlist (xsim).
#
# Sources (read-only, from the SystemC-emitter fork's history -- the designs were
# deleted from the branch tip but the netlists survive):
#   git show 779e4350^:agents/noc/pe_split.py
#   git show 779e4350^:agents/noc/rtl/{pe_wire,pe_stream,pe_channel}/rtl.v
#
# Reference model: C[i] = sum_{j<=i} A[j]*B[j], A=1..8, B=2..16 step 2
#                  -> 2 10 28 60 110 182 280 408   (numpy cumsum, hardcoded in the tb)
#
# Prereqs: Vivado 2023.2 xsim on PATH (source settings64.sh). No SystemC needed.
# Runtime: ~6 s per case.
# ============================================================================
set -u
S="$(cd "$(dirname "$0")" && pwd)"
r() { printf '%-58s ' "$1"; shift; "$S/run_mulacc.sh" "$@" 2>&1 | grep -oE 'PASS|FAIL .*' ; }

echo "--- 1. the three boundaries, no pacing games -------------------------------"
r "Stream[int32,2] boundary (control)"          pe_stream
r "Channel[valid_ready] boundary (control)"     pe_channel
r "Wire[int32] boundary (DUT)"                  pe_wire

echo "--- 2. Wire under every producer/consumer pacing ---------------------------"
for si in 1 2 3 4 5 6; do for so in 1 2 3; do
  r "Wire  stall_in=$si stall_out=$so"          pe_wire -d STALL_IN=$si -d STALL_OUT=$so
done; done

echo "--- 3. positive control: same Wire RTL, lockstep imposed by the tb ---------"
for d in 2 3 4 5; do
  r "Wire + LOCKSTEP, acc released $d cycles late" pe_wire -d LOCKSTEP -d ACC_RST_DELAY=$d
done

echo "--- 4. deliberate breakage: each of these MUST go red ----------------------"
r "Wire+LOCKSTEP, 1 cycle of latency in the wire" pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3 -d BREAK_WIRE
r "Wire+LOCKSTEP, boundary datum xor 1"           pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3 -d BREAK_DATA
r "Channel,        boundary datum xor 1"          pe_channel -d BREAK_DATA
r "Stream,         boundary datum xor 1"          pe_stream  -d BREAK_DATA

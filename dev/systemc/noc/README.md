# NoC — XY/DOR mesh routers in Allo

4×4 mesh NoC, simulator-verified and csynth-clean. Packet = one packed
UInt (data + dest_col + dest_row + valid, valid=MSB so bubble==0). Always-fire/bubble.

## router_XY.py
- Router-only 4×4 mesh; cores are stubbed as inject/deliver streams.
- One router kernel/cycle: read 5 → XY-route → fixed-priority arbitration → crossbar → write 5.
- **Bufferless**: arbiter losers are dropped (no retry).

## router_XY_PEs.py
- All nodes are real PEs (compute + inject), only the 4 mesh edges stubbed.
- Packet carries a `done` bit: operand → PE computes & re-injects; result → PE sinks it.
- Per-PE `fwd` table says where each PE sends its computed result.

## router_XY_PEs_bp.py
- Adds **credit/backpressure** → lossless (no drops, arbiter loser stays buffered & retries).
- Reverse credit streams per link; each router input has a depth-2 FIFO (EVA router_register style).
- PE is credit-aware: holds inject until it has a credit, returns deliver-credit only when free.

## router_XY_rr.py
- `router_XY.py` with **round-robin** arbitration instead of fixed-priority (no input can starve).
- Per-output rotating pointer `rr[o]` (persists across cycles): scan starts at `rr[o]`, first valid requester wins.
- After a grant, `rr[o] = winner+1` → the just-served input drops to lowest priority next cycle.

## router_XY_PEs_bp_rr.py
- `router_XY_PEs_bp.py` (credit/backpressure, lossless) **plus round-robin** arbitration → lossless *and* fair.
- Same `rr[o]` rotation, but scans buffered input heads and keeps the `cred[o]>0` gate.
- Rotates only on an actual grant, so a credit-starved cycle doesn't waste the rotation.

## Synthesis note
Host arrays are owned by single load/store kernels that fan out to per-node streams
(a replicated kernel can't take a whole array — HLS 200-779/979). Stream indices use
`allo.meta_for` (compile-time). Region arg order = `(inj, fwd, dlv)` to match discovery order.

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``matrix.result_latency.vmatpush`` measured from the issue edge, on the pinned RTL.

    $ALLO_PYTHON dev/records/minitpu/u4_track_b_2026-10-08/scripts/mxu_offset.py

The Phase 0 whole-sequencer setup (``units/sequencer.Run``: the program
loaded through the IRAM port, ``start``, the DMA side ideal) runs programs
assembled by MiniTPU's own ``asm.py`` (``harness/minitpu_asm``); its
``vpu_ctrl_o`` trace, row for row, drives ``vpu.sv`` in ``rtl/u4_vpu_mxu.sv``
(``u4_vpu_wb`` plus the MXU port by hierarchical reference). That is
``minitpu_core.sv``'s wiring: ``.vpu_ctrl_o(vpu_ctrl)`` -> ``.ctrl_i(vpu_ctrl)``,
no register between, so row ``t`` of the sequencer is row ``t`` of the VPU and
every distance below counts clock edges from the vmatpush's issue row -- the
edge the calendar's ``W`` counts from.

Measured per vmatpush: the rows its four rows reach the MXU (``input_push_i``),
the first row ``output_valid_o`` is high (U3's push->valid is from the LAST
push), and for a vmatpop issued ``d`` after the push: the row the pop engine
takes the result (``output_pop_i``) and the row the matrix source drives the
writeback mux. And from ``asm.py`` itself: the distance its scheduler puts
between a vmatpush and the vmatpop that drains it.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from examples.minitpu.harness import minitpu_asm, rtl  # noqa: E402
from examples.minitpu.harness import ref_ctrl_wb as WB  # noqa: E402
from examples.minitpu.harness.traces import rng_for  # noqa: E402
from examples.minitpu.units import sequencer as S  # noqa: E402
from examples.minitpu.units import vpu_writeback as P0  # noqa: E402

WRAP = os.path.join(ROOT, "examples", "minitpu", "units", "rtl", "u4_vpu_mxu.sv")
VPU = rtl.RtlUnit(top="u4_vpu_mxu", sources=P0.VPU_F + [WRAP], inputs=P0.RTL.inputs,
                  outputs=list(P0.RTL.outputs) + [("mxu_push_o", 1), ("mxu_accept_o", 1), ("mxu_valid_o", 1),
                                                   ("mxu_pop_o", 1), ("mpop_wb_o", 1)],
                  shape="trace", clk="clk_i", assertions=True)


def run(words, cycles=400):
    cmd = S.Run(words, rng_for("u4b-mxu-offset"), cycles=cycles).cmd()
    seq = rtl.run_trace(S.RTL, {p: rtl.pack(cmd[p], w) for p, w in S.INPUTS})
    ctrl = [int(x) for x in rtl.unpack(seq["vpu_ctrl_o"])]
    n = len(ctrl)
    vin = {"rst_ni": cmd["rst_n"], "ctrl_i": ctrl, "dma_en_i": [0] * n, "dma_we_i": [0] * n,
           "dma_addr_i": [0] * n, "dma_wdata_i": [0] * n}
    v = rtl.run_trace(VPU, {p: rtl.pack(vin[p], w) for p, w in VPU.inputs})
    f = [WB.unpack_ctrl(c) for c in ctrl]
    rows = lambda key: [t for t in range(n) if f[t][key]]  # noqa: E731
    sig = {p: [t for t, x in enumerate(rtl.unpack(v[p])) if x] for p in
           ("mxu_push_o", "mxu_valid_o", "mxu_pop_o", "mpop_wb_o")}
    return rows("vmatpush_valid"), rows("vmatpop_valid"), sig, list(rtl.last_asserts)


def rising(rows):
    return [t for t in rows if t - 1 not in rows]


def main():
    a = minitpu_asm.load()
    print(f"asm.py: _M_RESULT_LATENCY = {a._M_RESULT_LATENCY}, _M_ISSUE_INTERVAL = {a._M_ISSUE_INTERVAL}, "
          f"W_MPOP = {WB.W['mpop']}")
    # 1. asm.py's own schedule: vmatload, vmatpush, vmatpop, scheduled by asm.schedule()
    b = a.AsmBuilder()
    b.bundle(m=b.vmatload(0))
    b.bundle(m=b.vmatpush(4))
    b.bundle(m=b.vmatpop(8))
    b.bundle(x=b.vst(8, 64))
    b.bundle(b.halt())
    words = a.schedule(b.bundles)
    push, pop, sig, asserts = run(words)
    print(f"[asm-scheduled] vmatpush issued row {push}, vmatpop issued row {pop}: asm's distance {pop[0] - push[0]}")
    p = push[0]
    rows_in = [t for t in sig["mxu_push_o"] if t > p][:4]
    v0 = [t for t in rising(sig["mxu_valid_o"]) if t > p][0]
    print(f"   MXU input_push rows {[t - p for t in rows_in]} (from issue); first output_valid +{v0 - p}; "
          f"push->valid from the last push {v0 - rows_in[-1]}")
    print(f"   output_pop +{[t - p for t in sig['mxu_pop_o'] if t > p][:2]}; matrix source at the mux "
          f"+{[t - p for t in sig['mpop_wb_o'] if t > p][:2]}; asserts {len(asserts)}")
    # 2. a vmatpop forced at d after the vmatpush (unscheduled bundles; delay = d - 1)
    print("[forced distance d] pop issue -> output_pop -> mux, and against the calendar (pop + W_mpop - 2 = pop + 1)")
    for d in (82, 83, 84, 85, 86, 90):
        b = a.AsmBuilder()
        b.bundle(m=b.vmatload(0))
        for _ in range(20):
            b.bundle()
        b.bundle(m=b.vmatpush(4))
        b.bundles[-1] = a._set_delay(b.bundles[-1], d - 1)
        b.bundle(m=b.vmatpop(8))
        for _ in range(30):
            b.bundle()
        b.bundle(b.halt())
        push, pop, sig, asserts = run(b.bundles)
        p, q = push[0], pop[0]
        v0 = [t for t in rising(sig["mxu_valid_o"]) if t > p][0]
        op = [t for t in sig["mxu_pop_o"] if t > q][0] if [t for t in sig["mxu_pop_o"] if t > q] else None
        mx = [t for t in sig["mpop_wb_o"] if t > q][0]
        late = mx - (q + WB.L["mpop"])
        print(f"   d={d}: pop at +{q - p}; output_valid +{v0 - p}; output_pop +{op - p if op else None}; "
              f"mux +{mx - p} (pop + {mx - q}); {'as booked' if late == 0 else f'{late} late vs the calendar'}; "
              f"sim asserts {len(asserts)}")


if __name__ == "__main__":
    main()

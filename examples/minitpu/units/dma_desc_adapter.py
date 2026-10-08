# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``dma_desc_adapter``, a D-slot bundle to one ``dma.sv`` descriptor.

Two states. On ``start_i`` (the sequencer's ``run_is_dma``) it snapshots the
decoded D slot and the two SREG values read that cycle (base, plus the
sign-extended 24-bit displacement when ``has_disp``), and from the next cycle
drives ``desc_valid_o`` with the descriptor in dma.sv's units: VMEM row
``word << 2`` (beats) and rows ``(rows_m1 << 2) | 3`` (beats - 1), channel
truncated to ``DMA_CHANNEL_SEL_W`` = 1 bit, cols tied 0. It holds until
``desc_accept_i``; ``done_o`` is that accept, combinationally. A start while
issuing is ignored (the sequencer cannot send one: it waits in S_D_WAIT).

Built with ``+define+SYNTHESIS``: the adapter's sim-only S_LAT check reads
its sibling ``u_scalar_agu`` by hierarchical reference, which only
``sequencer.sv`` provides (``units/rtl/u4_dma_desc_adapter.sv``). The check
itself is the sequencer's (parent track). Reference: ``ref_ctrl_decode.DescAdapter``.
Seed: ``tb_bundle_scalar_agu`` part B (``tb.dutB.u_dma_adapter``).
"""

import os
import sys

from examples.minitpu.harness import rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import Trace, rng_for

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/dma_desc_adapter.sv", os.path.join(HERE, "rtl", "u4_dma_desc_adapter.sv")]
D_W = sum(w for _, w in R.D_SLOT)
INPUTS = [("rst_n", 1), ("start_i", 1), ("d_i", D_W), ("sreg_rd_base_i", 32), ("sreg_rd_stride_i", 32),
          ("desc_accept_i", 1)]
OUTPUTS = [(p, w, "pre") for p, w in R.DESC_OUTS]
RTL = rtl.RtlUnit(top="u4_dma_desc_adapter", sources=SOURCES, inputs=INPUTS, outputs=OUTPUTS, shape="trace",
                  clk="clk", rst_n="rst_n", defines=["SYNTHESIS"])
INSTANCES = {"base": RTL}
DEFAULT = "base"
LATENCY_SOURCE = ("start -> desc_valid 1 (dma_desc_adapter.sv S_IDLE -> S_ISSUE); accept -> done 0 "
                  "(combinational); tb_bundle_scalar_agu: descriptor round trip 2 cycles with an always-accepting DMA")


def REF(inst, cmd):
    want, reason, ev = R.desc_adapter_trace(cmd)
    for p in reason:
        reason[p][0] = "before reset"
    return want, reason, ev


def d_word(rng, **kw):
    d = {n: rng.getrandbits(w) for n, w in R.D_SLOT}
    d["valid"] = 1
    d.update(kw)
    return R.pack(R.D_SLOT, d)


def _trace():
    t = Trace({p: 0 for p, _ in INPUTS})
    t.idle(2, rst_n=0)
    t.defaults["rst_n"] = 1
    return t


def directed(rng):
    out = []
    t = _trace()
    # accepted at once (the sequencer's common case), then held 3 cycles, then a start while issuing
    for disp in (0, 1, (1 << 23), (1 << 24) - 1):
        t.cycle(start_i=1, d_i=d_word(rng, has_disp=1, disp=disp), sreg_rd_base_i=0xFFFFFFF0,
                sreg_rd_stride_i=7)
        t.cycle(desc_accept_i=1)
        t.idle(1)
    t.cycle(start_i=1, d_i=d_word(rng, has_disp=0, rows=0xFFF, vmem_address=0xFFF, channel_sel=3),
            sreg_rd_base_i=123, sreg_rd_stride_i=1)
    t.idle(3, sreg_rd_base_i=999)                     # the snapshot must not follow the live sreg
    t.cycle(start_i=1, d_i=d_word(rng), desc_accept_i=0)  # ignored: still issuing
    t.cycle(desc_accept_i=1)
    t.cycle(start_i=1, d_i=d_word(rng), desc_accept_i=1)  # back to back: accept while idle is ignored
    t.cycle(desc_accept_i=1)
    t.cycle(start_i=1, d_i=d_word(rng))
    t.cycle(rst_n=0)                                  # reset while issuing
    t.idle(3)
    out.append(("directed", t.cmd(), True))
    return out


def random_trace(rng, n):
    t = _trace()
    for _ in range(n):
        t.cycle(start_i=int(rng.random() < 0.3), d_i=d_word(rng), sreg_rd_base_i=rng.getrandbits(32),
                sreg_rd_stride_i=rng.getrandbits(32), desc_accept_i=int(rng.random() < 0.5),
                rst_n=int(rng.random() > 0.002))
    return t.cmd()


def traces(inst):
    rng = rng_for("dma_desc_adapter", inst)
    return directed(rng) + [("random", random_trace(rng, 20000), True)]


def seeds():
    srcs = PKGS + [f"src/core/sequencer/{f}" for f in (
        "sequencer_decoder.sv", "sequencer_iram.sv", "sequencer_fetch_queue.sv", "sequencer_loop_buffer.sv",
        "sequencer_loop_ctrl.sv", "sequencer_agu_resolve.sv", "sequencer_scalar_agu.sv", "dma_desc_adapter.sv",
        "sequencer_vpu_adapter.sv", "sequencer.sv")]
    names = ["rst_n", "start_i", "d_i", "sreg_rd_base_i", "sreg_rd_stride_i", "desc_accept_i"] + \
            [p for p, _ in R.DESC_OUTS]
    rows = R.seed_rows("tb_bundle_scalar_agu", srcs, "dutB.u_dma_adapter", names, clk="clk")
    cmd = {p: rows[p] for p, _ in INPUTS}
    return [("tb_bundle_scalar_agu:B", "base", cmd, {p: rows[p] for p, _ in R.DESC_OUTS}, True)]


def probes(inst):
    rng = rng_for("dma_desc_adapter-probe", inst)
    res = []
    t = _trace()
    t.idle(3)
    ev = len(t)
    t.cycle(start_i=1, d_i=d_word(rng))
    t.idle(4)
    c = t.cmd()
    res.append(("start_i -> desc_valid_o", 1, rtl.probe_trace(RTL, {p: rtl.pack(c[p], w) for p, w in INPUTS},
                                                               "desc_valid_o", ev)))
    t = _trace()
    t.cycle(start_i=1, d_i=d_word(rng))
    t.idle(3)
    ev = len(t)
    t.cycle(desc_accept_i=1)
    t.idle(3)
    c = t.cmd()
    res.append(("desc_accept_i -> done_o (comb)", 0,
                rtl.probe_trace(RTL, {p: rtl.pack(c[p], w) for p, w in INPUTS}, "done_o", ev)))
    return res


# ---------------------------------------------------------------------------
# Allo variants (U4 track C, ``dev/records/minitpu/u4_track_c_2026-10-08.rst``).
# Every variant is driven by the same per-cycle command trace as the RTL. The
# region ports are two lane arrays (U3 P-8): ``IN: UInt(64)[n, 6]`` (the
# ``INPUTS`` columns in order; ``d_i`` is 57 bits, below S8's 2**63) and
# ``OUT: UInt(64)[n, 9]`` (the ``DESC_OUTS`` columns). "Cycle ``t``" is
# iteration ``t``; the outputs of ``t`` are the state it starts in plus the
# cycle's comb inputs (``rtl.py``'s ``pre`` sampling), then the edge.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, int32  # noqa: E402

from examples.minitpu.units.dma_params import DmaGeometry  # noqa: E402

GEOMETRY = DmaGeometry()
OUT_NAMES = [p for p, _ in R.DESC_OUTS]


def bits(n, w=0, inst="base"):
    """Plan A1, cycle-locked: the two-state FSM transcribed. ``d_i`` is
    snapshotted as the four fields the outputs read (``d_q`` whole is
    equivalent: nothing else of it is observable); words -> beats is a
    shift (``{vmem_address, 2'b00}``, ``{rows, 2'b11}``), the D-20
    relation ``ROW_BITS == DESC_BEAT_ROWS_W`` is checked at make time."""
    GEOMETRY.legality()
    SL = GEOMETRY.SUBLANE_SEL_W
    VA = GEOMETRY.VMEM_ADDR_W
    RW = GEOMETRY.DESC_WORD_ROWS_W
    BW = GEOMETRY.ROW_BITS

    @df.region()
    def top(IN: UInt(64)[n, 6], OUT: UInt(64)[n, 9]):
        @df.kernel(mapping=[1], args=[IN, OUT])
        def adapter(cin: UInt(64)[n, 6], cout: UInt(64)[n, 9]):
            issue: int32 = 0
            st_q: UInt(1) = 0
            ch_q: UInt(1) = 0
            va_q: UInt(VA) = 0
            rows_q: UInt(RW) = 0
            base_q: UInt(32) = 0
            stride_q: UInt(32) = 0
            for t in range(n):
                rst: UInt(1) = cin[t, 0]  # S6: every port read unconditionally
                start: UInt(1) = cin[t, 1]
                d: UInt(64) = cin[t, 2]
                sb: UInt(32) = cin[t, 3]
                ss: UInt(32) = cin[t, 4]
                acc: UInt(1) = cin[t, 5]
                if rst == 0:  # asynchronous: the row is sampled in reset
                    issue = 0
                    st_q = 0
                    ch_q = 0
                    va_q = 0
                    rows_q = 0
                    base_q = 0
                    stride_q = 0
                # words -> beats by shift, not a slice store: a slice bounded
                # by closure names widens to UInt(32) with a warning (D-17,
                # finding C3); the shift is the same circuit
                vrow: UInt(BW) = va_q
                vrow = vrow << SL
                brow: UInt(BW) = rows_q
                brow = (brow << SL) | ((1 << SL) - 1)
                done: UInt(1) = 0
                if issue == 1:
                    done = acc
                cout[t, 0] = issue
                cout[t, 1] = st_q
                cout[t, 2] = ch_q
                cout[t, 3] = vrow
                cout[t, 4] = brow
                cout[t, 5] = 0
                cout[t, 6] = base_q
                cout[t, 7] = stride_q
                cout[t, 8] = done
                if rst == 1:
                    if issue == 0:
                        if start == 1:
                            # D_SLOT, LSB first: stride_sreg[0:2] base_sreg[2:4]
                            # disp[4:28] has_disp[28] rows[29:41]
                            # vmem_address[41:53] channel_sel[53:55] is_store[55]
                            st_q = d[55]
                            ch_q = d[53]  # channel_sel[0]: the narrowing drops bit 54
                            va_q = d[41:53]
                            rows_q = d[29:41]
                            disp: UInt(32) = d[4:28]
                            if d[27] == 1:
                                disp[24:32] = 0xFF  # sign extension of the 24-bit disp
                            if d[28] == 1:
                                base_q = sb + disp
                            else:
                                base_q = sb
                            stride_q = ss
                            issue = 1
                    elif acc == 1:
                        issue = 0

    return top


def _run_bits(mod, cmd, n, w=0):
    ins = np.zeros((n, 6), dtype=np.uint64)
    for j, (p, _) in enumerate(INPUTS):
        ins[:, j] = np.asarray([int(x) for x in cmd[p][:n]], dtype=np.uint64)
    outs = np.zeros((n, 9), dtype=np.uint64)
    mod(ins, outs)
    return {p: [int(x) for x in outs[:, j]] for j, p in enumerate(OUT_NAMES)}


WIDTH = {"base": 0}
VARIANTS = {"bits": (bits, _run_bits)}

if __name__ == "__main__":
    sys.exit(0)

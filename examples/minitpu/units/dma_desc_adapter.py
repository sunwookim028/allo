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


VARIANTS = {}

if __name__ == "__main__":
    sys.exit(0)

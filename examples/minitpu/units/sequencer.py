# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer.sv`` whole -- bundle issue on assembled programs.

The sequencer unchanged as the top (no wrapper): the IRAM is loaded through
its write port, ``start`` launches, and the trace is a *program run*. The
reference (``harness/ref_ctrl_issue.py``) composes the Phase 0 models of the
parts (fetch, loop control, decode, address resolve, scalar AGU, descriptor
adapter, VPU adapter) with the top FSM, and predicts every output on every
cycle: the ``vpu_ctrl_t`` command the VPU receives, the DMA descriptor,
``done`` and the channel clears; and, as events, which of the sequencer's
sim-only schedule assertions fire.

Programs come from MiniTPU's own assembler (``asm.py`` via
``harness/minitpu_asm``), scheduled by its ``schedule()``, and from the
shipped images in ``board_package/images``. The DMA side is driven open-loop
(``dma_desc_accept``, ``dma_channel_done``, ``dma_idle`` per cycle) -- the
reference takes any input sequence.

Instance ``shipped`` (``DMA_CHANNEL_COUNT_P`` 2, ``INSTR_ADDR_W`` 12).
"""

import glob
import os

import numpy as np

from examples.minitpu.harness import minitpu_asm, rtl
from examples.minitpu.harness import ref_ctrl_issue as R
from examples.minitpu.harness.traces import rng_for

SOURCES = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv",
           "src/core/sequencer/sequencer_pkg.sv"] + [f"src/core/sequencer/{f}" for f in (
    "sequencer_decoder.sv", "sequencer_iram.sv", "sequencer_fetch_queue.sv", "sequencer_loop_buffer.sv",
    "sequencer_loop_ctrl.sv", "sequencer_agu_resolve.sv", "sequencer_scalar_agu.sv",
    "dma_desc_adapter.sv", "sequencer_vpu_adapter.sv", "sequencer.sv")]
INPUTS = [("rst_n", 1), ("start", 1), ("instr_write_en", 1), ("iram_addr", 12), ("dma_iram_din", 128),
          ("program_id_csr", 32), ("kernel_arg_csr", 128), ("dma_channel_done", 2), ("dma_idle", 1),
          ("dma_desc_accept", 1), ("matrix_busy_i", 1)]
RTL = rtl.RtlUnit(top="sequencer", sources=SOURCES, inputs=INPUTS, outputs=list(R.OUTS),
                  shape="trace", clk="clk", rst_n="rst_n", assertions=True)
INSTANCES = {"shipped": RTL}
DEFAULT = "shipped"
LATENCY_SOURCE = ("issue: one bundle a cycle warm; delay=N holds N cycles (sequencer.sv:271); "
                  "S_LAT 2; descriptor round trip; loop.begin.r skip 3 (isa_latency.json)")
VARIANTS = {}


def REF(inst, cmd):
    return R.issue_trace(cmd)


class Run:
    """A program run: load ``words`` at ``base``, idle, pulse ``start``; then
    the DMA side per ``dma`` policy until ``cycles``."""

    def __init__(self, words, rng, *, base=0, program_id=0, kargs=0, dma="ideal", cycles=None,
                 matrix_busy=0.0):
        self.rows = []
        idle = dict(rst_n=1, start=0, instr_write_en=0, iram_addr=0, dma_iram_din=0,
                    program_id_csr=program_id, kernel_arg_csr=kargs, dma_channel_done=3, dma_idle=1,
                    dma_desc_accept=1, matrix_busy_i=0)
        for _ in range(3):
            self.rows.append(dict(idle, rst_n=0))
        # pad: the fetch queue prefetches past the halt, so no unwritten word reaches the head
        words = list(words) + [0] * 8
        for k, w in enumerate(words):
            self.rows.append(dict(idle, instr_write_en=1, iram_addr=(base + k) & 0xFFF, dma_iram_din=w))
        self.rows += [dict(idle) for _ in range(4)]
        self.rows.append(dict(idle, start=1))
        n = cycles or (40 * len(words) + 200)
        for _ in range(n):
            r = dict(idle)
            if dma == "random":
                r.update(dma_desc_accept=int(rng.random() < 0.4), dma_channel_done=rng.randrange(4),
                         dma_idle=int(rng.random() < 0.3))
            r["matrix_busy_i"] = int(rng.random() < matrix_busy)
            self.rows.append(r)

    def cmd(self):
        return {p: [r[p] for r in self.rows] for p, _ in INPUTS}


# ---- programs ----------------------------------------------------------------
def _asm():
    return minitpu_asm.load()


def prog_straight(a):
    """Every V op, vld/vst with and without AGU, the M commands, every S op, halt."""
    b = a.AsmBuilder()
    b.bundle(x=b.vld(1, 8))
    b.bundle(x=b.vld(2, 12))
    for op in (b.vadd, b.vsub, b.vmul, b.vmax, b.vmin):
        b.bundle(v=op(3, 1, 2))
    for op in (b.vgelu, b.vexp, b.vrecip, b.vrsqrt, b.vredsum, b.vredmax, b.vlanesum, b.vlanemax):
        b.bundle(v=op(4, 3))
    b.bundle(v=b.vmov(5, 4), x=b.vst(4, 40))
    for k in range(4):
        b.bundle(v=b.vtxin(5, k))
    for k in range(4):
        b.bundle(v=b.vtxout(6 + k, k))
    b.bundle(m=b.vmatload(6))
    b.bundle(m=b.vmatpush(1))
    b.bundle(m=b.vmatpop(10))
    b.bundle(s=b.smovi(0, 1234))
    b.bundle(s=b.saddi(1, 0, -7))
    b.bundle(s=b.smov_arg(2, 3))
    b.bundle(x=b.vst(10, 64))
    b.bundle(b.halt())
    return a.schedule(b.bundles)


def prog_loops(a, back_edge_waw=False):
    """Nested loops, X addresses on several levels and shifts, a body over
    LB_CAP (refetch), loop.begin.r from an sreg and from an argument, zero trip.
    ``back_edge_waw``: the two loop.begin.r bodies load a VREG every pass, a
    write-after-write across the back edge that ``schedule()`` does not model
    (``isa_latency.json`` scheduler.not_modelled) and MiniTPU's
    ``tools/check_back_edges.py`` flags (bundle 38: 2 cases)."""
    ld = (lambda v, *a_, **k: b.vld(v, *a_, **k)) if back_edge_waw else (lambda v, *a_, **k: b.vst(v, *a_, **k))
    b = a.AsmBuilder()
    b.bundle(s=b.smovi(0, 3))
    b.bundle()
    b.bundle()
    b.bundle(f=b.lbegin(3))
    b.bundle(f=b.lbegin(4, lo=1, step=2))
    b.bundle(x=b.vld(1, 4, agu=True, shift=4, level=0))
    b.bundle(x=b.vld(2, 8, agu=True, shift=8, level=1))
    b.bundle(v=b.vadd(3, 1, 2), f=b.loop_end())
    b.bundle(f=b.lbegin(2))
    for k in range(26):  # > LB_CAP
        b.bundle(x=b.vst(3, 128 + 4 * k, agu=True, shift=12, level=2))
    b.bundle(f=b.loop_end())
    b.bundle(x=b.vst(3, 512, agu=True, shift=4, level=0), f=b.loop_end())
    b.bundle(f=b.lbegin_r(0, from_arg=False))
    b.bundle(x=ld(4, 0, agu=True, shift=4, level=0))
    b.bundle(f=b.loop_end())
    b.bundle(f=b.lbegin_r(1, from_arg=True))  # kernel argument 1: zero trip in one run
    b.bundle(x=ld(5, 0, agu=True, shift=4, level=0))
    b.bundle(f=b.loop_end())
    b.bundle(b.halt())
    return a.schedule(b.bundles)


def prog_dma(a):
    """Scalar setup, descriptors with and without displacement, waits."""
    b = a.AsmBuilder()
    b.bundle(s=b.smovi(0, 0x1000))
    b.bundle(s=b.smovi(1, 4))
    b.bundle(s=b.smov_arg(2, 0))
    b.bundle()
    b.bundle(d=b.dma_load(0, 0, 16, 0, 1))
    b.bundle(d=b.dma_load(1, 64, 32, 2, 1, disp=-16))
    b.bundle(f=b.wait_channel(1))
    b.bundle(f=b.lbegin(3))
    b.bundle(s=b.saddi(0, 0, 64))
    b.bundle()
    b.bundle(d=b.dma_store(0, 128, 8, 0, 1, disp=3))
    b.bundle(f=b.wait_channel(1))
    b.bundle(f=b.loop_end())
    b.bundle(f=b.wait_channel(3))
    b.bundle(b.halt())
    return a.schedule(b.bundles)


def prog_random(a, rng, n=120):
    """A random legal program: random slot contents, scheduled by asm."""
    b = a.AsmBuilder()
    lo = lambda: rng.randrange(16)  # noqa: E731  V results in v0..v15, loads in v16..v31
    vops = [lambda: b.vadd(lo(), rng.randrange(32), rng.randrange(32)),
            lambda: b.vmul(lo(), rng.randrange(32), rng.randrange(32)),
            lambda: b.vexp(lo(), rng.randrange(32)),
            lambda: b.vredsum(lo(), rng.randrange(32)),
            lambda: b.vlanemax(lo(), rng.randrange(32)),
            lambda: b.vtxout(lo(), rng.randrange(4))]
    b.bundle(f=b.lbegin(rng.randrange(1, 4)))
    for _ in range(n):
        slots = {}
        if rng.random() < 0.7:
            slots["v"] = rng.choice(vops)()
        if rng.random() < 0.4:
            k = rng.randrange(2)
            slots["x"] = (b.vld if k == 0 else b.vst)(16 + rng.randrange(16), 4 * rng.randrange(1024),
                                                      agu=rng.random() < 0.5, shift=4, level=0)
        if rng.random() < 0.3:
            slots["s"] = b.saddi(rng.randrange(4), rng.randrange(4), rng.randrange(-100, 100))
        try:
            b.bundle(**slots)
        except ValueError:
            b.bundle()
    b.bundle(f=b.loop_end())
    b.bundle(b.halt())
    return a.schedule(b.bundles)


def images():
    out = []
    for path in sorted(glob.glob(os.path.join(rtl.minitpu_home(), "board_package", "images", "*.bin"))):
        data = open(path, "rb").read()
        out.append((os.path.basename(path), [int.from_bytes(data[i:i + 16], "little")
                                             for i in range(0, len(data), 16)]))
    return out


def traces(inst):
    a = _asm()
    out = []
    rng = rng_for("u4-seq", "straight")
    out.append(("straight", Run(prog_straight(a), rng).cmd(), True))
    out.append(("straight-at-slot-3", Run(prog_straight(a), rng, base=0x300, program_id=3).cmd(), True))
    out.append(("loops", Run(prog_loops(a), rng, kargs=(5 << 32)).cmd(), True))
    out.append(("loops-zero-trip", Run(prog_loops(a), rng, kargs=0).cmd(), True))
    out.append(("loops-back-edge-waw", Run(prog_loops(a, True), rng, kargs=(5 << 32)).cmd(), False))
    out.append(("dma-ideal", Run(prog_dma(a), rng, kargs=0x8000).cmd(), True))
    out.append(("dma-random", Run(prog_dma(a), rng_for("u4-seq", "dmarand"), kargs=0x8000,
                                  dma="random", cycles=1500).cmd(), True))
    for k in range(4):
        r = rng_for("u4-seq", "random", k)
        out.append((f"random-{k}", Run(prog_random(a, r), r, dma="random" if k % 2 else "ideal").cmd(), True))
    for name, words in images():
        r = rng_for("u4-seq", name)
        out.append((f"image {name}", Run(words, r, kargs=rng_for(name).getrandbits(128) & ~(0xFFFF << 96),
                                         dma="random", cycles=6000).cmd(), True))
    # an unscheduled program: the sim-only checks fire, and the model predicts which
    b = a.AsmBuilder()
    b.bundle(x=b.vld(1, 0))
    b.bundle(v=b.vadd(2, 1, 1))          # RAW on v1 in flight
    b.bundle(v=b.vexp(3, 2))             # W 7 at +2 meets ... and RAW
    b.bundle(v=b.vadd(4, 3, 3), x=b.vld(5, 4))
    b.bundle(v=b.vredsum(6, 5))
    b.bundle(v=b.vlanesum(7, 5))
    b.bundle()
    b.bundle()
    b.bundle(v=b.vlanesum(8, 1))         # silent at vpu.sv: W 11 at +4 after vredsum's 15
    b.bundle(b.halt())                   # halt with writes in flight
    out.append(("unscheduled", Run(b.bundles, rng, cycles=200).cmd(), False))
    out.append(("matrix-busy-random", Run(prog_straight(a), rng, matrix_busy=0.5).cmd(), False))
    return out


def seeds():
    """MiniTPU's sequencer-level tbs, replayed at the sequencer's ports."""
    from examples.minitpu.harness import ref_ctrl_decode as D

    srcs = SOURCES
    out = []
    for tb, scope, extra in (("tb_bundle_loop", "dut", []), ("tb_bundle_interlocks", "dut", []),
                             ("tb_loop_begin_r", "dut", [])):
        names = [p for p, _ in INPUTS] + [p for p, _ in R.OUTS]
        try:
            rows = D.seed_rows(tb, srcs + ["tb/isa_latency_pkg.sv"] + extra, scope, names,
                               clk_scope=scope, clk="clk")
        except Exception as e:  # noqa: BLE001
            print(f"   seed {tb}: not replayed ({str(e).splitlines()[0][:120]})")
            continue
        cmd = {p: rows[p] for p, _ in INPUTS}
        seen = {p: rows[p] for p, _ in R.OUTS}
        out.append((tb, "shipped", cmd, seen, True))
    return out


# ---- probes ------------------------------------------------------------------
def _issue_cycles(cmd, res):
    iss = [int(x) for x in rtl.unpack(res["bundle_issued_o"])]
    return [t for t, v in enumerate(iss) if v]


def probes(inst):
    a = _asm()
    res = []
    rng = rng_for("u4-seq-probe")
    # start -> first issue (cold): entry flush, IRAM read, queue push
    b = a.AsmBuilder()
    for _ in range(6):
        b.bundle(v=b.vadd(1, 2, 3))
    b.bundle(b.halt())
    run = Run(b.bundles, rng, cycles=60)
    cmd = run.cmd()
    st = cmd["start"].index(1)
    r = rtl.run_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS})
    iss = _issue_cycles(cmd, r)
    res.append(("start -> first bundle issued (entry flush + IRAM + queue; fetch probe 3)", 3, iss[0] - st))
    res.append(("straight-line issue interval (warm)", 1, iss[1] - iss[0]))
    # delay = N holds issue N cycles
    for n in (1, 5, 85, 127):
        words = list(b.bundles)
        words[0] = a._set_delay(words[0], n)
        run = Run(words, rng, cycles=200)
        cmd = run.cmd()
        r = rtl.run_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS})
        iss = _issue_cycles(cmd, r)
        res.append((f"delay={n}: issue interval", n + 1, iss[1] - iss[0]))
    # halt -> done (dma idle)
    run = Run(b.bundles, rng, cycles=60)
    cmd = run.cmd()
    st = cmd["start"].index(1)
    r = rtl.run_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS})
    iss = _issue_cycles(cmd, r)
    done = [t for t, v in enumerate(rtl.unpack(r["done"])) if v and t > st]
    res.append(("halt issue -> done (DMA idle)", 2, done[0] - iss[-1]))
    # descriptor: issue -> next bundle, DMA accepting at once (D_WAIT round trip)
    b2 = a.AsmBuilder()
    b2.bundle(s=b2.smovi(0, 8))
    b2.bundle()
    b2.bundle()
    b2.bundle(d=b2.dma_load(0, 0, 4, 0, 0))
    b2.bundle(v=b2.vadd(1, 2, 3))
    b2.bundle(b2.halt())
    run = Run(b2.bundles, rng, cycles=60)
    cmd = run.cmd()
    r = rtl.run_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS})
    iss = _issue_cycles(cmd, r)
    res.append(("descriptor issue -> next issue (accepting DMA; tb_bundle_scalar_agu 2)", 2, iss[4] - iss[3]))
    # the sim-only schedule checks: predicted (the calendar model) vs fired, per check
    cats = {"RAW on VREG": "assert: RAW on a VREG in flight", "write-port collision at": "assert: write-port collision",
            "while its matrix engine is busy": "assert: M command while its engine is busy",
            "halt with VREG": "assert: halt with VREG(s) in flight", "WAW remains": "assert: WAW on a VREG in flight",
            "same-bundle VREG write-port": "assert: same-bundle write-port collision"}
    for label, cmd, legal in traces(inst):
        if legal:
            continue
        packed = {p: rtl.pack(cmd[p], w) for p, w in INPUTS}
        rtl.run_trace(RTL, packed)
        fired = list(rtl.last_asserts)
        _, _, ev = R.issue_trace(packed)
        for key, name in cats.items():
            n_fired = sum(1 for _, m in fired if key in m)
            if n_fired or ev.get(name):
                res.append((f"{label}: '{name}' predicted vs fired", ev.get(name, 0), n_fired))
    return res

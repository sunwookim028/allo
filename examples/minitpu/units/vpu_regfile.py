# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2: ``vpu_regfile``, one sublane's VREG stripe: 32 entries, 3R1W.

Three asynchronous reads (``assign rdata = mem[raddr]``) and one synchronous
write; never reset (``rst_ni`` is unused), so a VREG must be written before it
is read. Driven as a ``trace`` unit (``harness/rtl.py``): every read port is
sampled before the edge, so a read and a write of one VREG in one cycle show
the old value. Reference: ``harness/ref.py`` ``vpu_regfile_trace``.

Instances: ``w16`` is the module's default ``WIDTH = DATA_WIDTH``, what
``tb_vpu_alu_regfile`` drives; ``w256`` is the width ``vpu_vreg_stripe.sv:26``
instantiates (``VMEM_STRIPE_W``, 16 lanes x 16 b), four per VPU.

Declared latencies (``vpu_regfile.sv`` comment "Three async reads, one sync
write"; ``REGISTER_FILE.md``): read 0 edges on every port; write visible to a
read 1 edge later.

Allo variants (U2 pilot, ``dev/records/minitpu/u2_regfile_2026-10-02.rst``).
Every variant is driven by the SAME per-cycle command trace as the RTL: one
array per input port, one element per cycle, and returns one array per read
port, one element per cycle. Addresses are ``UInt(5)``, data ``UInt(W)``,
``we`` ``uint1`` -- the RTL's port widths. "Cycle ``t``" is iteration ``t`` of
the unit's loop: on the simulator it is only an order (reads of iteration
``t`` before its write, before iteration ``t + 1``); in SystemC csim and
Catapult RTL a cycle is whatever the schedule makes of an iteration, which
``check.py`` does not measure and the Catapult track measures on the RTL.

``trace`` (plan R1)
    One kernel, a kernel-local ``mem: UInt(W)[32]``. Per iteration: the three
    reads, *then* the write. A same-cycle read and write of one VREG reads the
    old value by program order.
``stateful`` (R2)
    ``trace`` with ``mem`` declared ``@ Stateful``, and the trace fed in
    chunks of ``CHUNK`` cycles over successive calls of one built module.
``ported`` (R3)
    A ``@df.unit`` whose six command ports and three response ports are
    ``Stream``\ s, one token per port per cycle (lockstep), between a driver
    kernel that replays the trace and a sink that records it.
``shared`` (R4, as the plan words it)
    A region-scope ``@ Stateful`` regfile, one kernel per physical port: three
    reader kernels and one writer kernel, with no other link between them.
    Nothing orders a read of cycle ``t`` against the write of cycle ``t - 1``.
``shared_sync`` (R4 + an ordering the RTL gets from the clock)
    ``shared`` with token streams that make the per-cycle order explicit:
    each reader signals "read ``t`` done" to the writer; the writer, after
    all three, writes and signals "write ``t`` done" to each reader.
``annotated`` (R5)
    ``trace`` with ``mem @ Memory(resource="LUTRAM", storage_type="RAM_1WNR",
    latency=0, depth=32)``: the D-1 sweep (which backend honours, refuses or
    drops each field).
``wire`` (Catapult port shape)
    ``trace``'s body with every port a ``Wire`` (``sc_in``/``sc_out``): the
    port shape of MiniTPU's module. SystemC only. Synthesized with
    ``s.partition("rf_0:mem")`` (complete) -- without it Catapult maps ``mem``
    to a 1R1W RAM and fails SCHD-30 at II=1.
``wire_stateful`` (comb-read form b)
    ``wire`` with ``mem @ Stateful`` (a module member in the emitted SystemC).
    Probe only: ``dev/records/minitpu/u2_comb_read_2026-10-02.rst``.
``comb`` (README D-13)
    ``wire`` with the three read ports declared ``Wire[UInt(w), comb]``: the
    SystemC emitter builds each read cone as an ``SC_METHOD`` over
    ``sc_signal`` storage (the record's form e, no hand patch), so Catapult's
    RTL reads at latency 0 and shows a write 1 edge later, as MiniTPU does.
    SystemC only. ``dev/records/minitpu/u2_comb_wire_impl_2026-10-02.rst``.
``comb_unreset`` (README D-14)
    ``comb`` with ``mem @ Stateful(reset=False)``: the storage is not reset,
    as MiniTPU's is not. The emitter writes it from a clock-edge ``SC_METHOD``
    with no reset action, and ``run.tcl`` sets ``-RESET_CLEARS_ALL_REGS no``.
    SystemC only. ``dev/records/minitpu/u2_unreset_impl_2026-10-02.rst``.
``trace_raw``
    ``trace`` as first written (no workarounds); evidence only.

Workarounds every variant carries (``u2_regfile_2026-10-02.rst``):

* **B4**: an unsigned index is sign-extended on the simulator/LLVM path, so
  every address is widened to ``int32`` before it indexes ``mem``.
* **B5**: ``x: int32 = s.get()`` on a ``UInt(5)`` stream does not cast, so
  stream addresses are read into a ``UInt(5)`` first, then widened.
* **S6**: the SystemC emitter turns an argument array read under ``if`` into
  a conditional stream ``Pop()``, so ``waddr``/``wdata`` are read
  unconditionally and only the store is under ``if we``.
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stateful, Stream, UInt, Wire, comb, int32, uint1
from allo.memory import Memory

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, concat, hot_addr, rng_for, word

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_regfile.sv"]
DEPTH = 32


def _unit(width):
    return rtl.RtlUnit(
        top="vpu_regfile",
        sources=SOURCES,
        inputs=[("rst_ni", 1), ("raddr_a_i", 5), ("raddr_b_i", 5), ("raddr_c_i", 5),
                ("waddr_i", 5), ("wdata_i", width), ("we_i", 1)],
        outputs=[("rdata_a_o", width), ("rdata_b_o", width), ("rdata_c_o", width)],
        shape="trace",
        params={} if width == 16 else {"WIDTH": width},
        assertions=True,
    )


INSTANCES = {"w16": _unit(16), "w256": _unit(256)}
WIDTH = {"w16": 16, "w256": 256}
DEFAULT = "w16"
RTL = INSTANCES[DEFAULT]
CMD = ("raddr_a_i", "raddr_b_i", "raddr_c_i", "waddr_i", "wdata_i", "we_i")
RESP = ("rdata_a_o", "rdata_b_o", "rdata_c_o")
CHUNK = 1024  # ``stateful``: cycles per call
LATENCY_SOURCE = "vpu_regfile.sv:29 comment (\"Three async reads, one sync write\")"


def REF(inst, cmd):
    return ref.vpu_regfile_trace(cmd, WIDTH[inst], DEPTH)


def _defaults():
    return {"rst_ni": 1, "raddr_a_i": 0, "raddr_b_i": 0, "raddr_c_i": 0,
            "waddr_i": 0, "wdata_i": 0, "we_i": 0}


def random_trace(inst, n, seed):
    rng = rng_for("regfile", inst, seed)
    w = WIDTH[inst]
    hot = rng.sample(range(DEPTH), 4)
    t = Trace(_defaults())
    for _ in range(n):
        t.cycle(rst_ni=int(rng.random() > 0.01),
                raddr_a_i=hot_addr(rng, DEPTH, hot), raddr_b_i=hot_addr(rng, DEPTH, hot),
                raddr_c_i=hot_addr(rng, DEPTH, hot), waddr_i=hot_addr(rng, DEPTH, hot),
                wdata_i=word(rng, w), we_i=int(rng.random() < 0.5))
    return t.cmd()


def directed(inst):
    """Write-then-read at every offset -1..+2 on every read port; all three
    ports on one VREG; every VREG written then read; reset mid-trace."""
    w = WIDTH[inst]
    rng = rng_for("regfile-directed", inst)
    out = []
    t = Trace(_defaults())
    for v in range(DEPTH):  # fill, then read back all three ports on one entry
        t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
    for v in range(DEPTH):
        t.cycle(raddr_a_i=v, raddr_b_i=v, raddr_c_i=v)
    out.append(("fill-readback", t.cmd()))
    for port in "abc":
        t = Trace(_defaults())
        for v in range(DEPTH):
            t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
        for off in (-1, 0, 1, 2):  # read of VREG 7 at cycle (write + off)
            seq = [dict() for _ in range(5)]
            seq[1].update(waddr_i=7, wdata_i=word(rng, w), we_i=1)
            seq[1 + off][f"raddr_{port}_i"] = 7
            for row in seq:
                t.cycle(**row)
        # back-to-back writes of one VREG, read every cycle
        for k in range(4):
            t.cycle(**{f"raddr_{port}_i": 9}, waddr_i=9, wdata_i=word(rng, w), we_i=1)
        t.cycle(**{f"raddr_{port}_i": 9})
        out.append((f"wr-offsets-port-{port}", t.cmd()))
    t = Trace(_defaults())  # reset does not clear: write, reset, read
    for v in range(4):
        t.cycle(waddr_i=v, wdata_i=word(rng, w), we_i=1)
    t.idle(4, rst_ni=0)
    for v in range(4):
        t.cycle(raddr_a_i=v, raddr_b_i=v, raddr_c_i=3 - v)
    t.cycle(rst_ni=0, waddr_i=1, wdata_i=word(rng, w), we_i=1)  # a write under reset lands
    t.cycle(raddr_a_i=1)
    out.append(("reset-keeps", t.cmd()))
    return out


def traces(inst):
    """``[(label, cmd, legal)]``. Every regfile trace is legal: it has no
    illegal command; reads of unwritten VREGs are masked, not refused."""
    tr = [(lab, c, True) for lab, c in directed(inst)]
    n = 20000 if inst == "w16" else 5000
    tr += [(f"random-{s}", random_trace(inst, n, s), True) for s in range(3)]
    return tr


def probes(inst):
    """``[(label, declared, measured)]`` by step probes on the RTL."""
    u = INSTANCES[inst]
    w = WIDTH[inst]
    res = []
    for port in "abc":
        t = Trace(_defaults())
        t.cycle(waddr_i=3, wdata_i=1, we_i=1).cycle(waddr_i=4, wdata_i=(1 << w) - 2, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 3})
        ev = len(t)
        t.idle(8, **{f"raddr_{port}_i": 4})
        res.append((f"read {port}", 0, rtl.probe_trace(u, t.cmd(), f"rdata_{port}_o", ev)))
        t = Trace(_defaults())
        t.cycle(waddr_i=5, wdata_i=1, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 5})
        ev = len(t)
        t.cycle(**{f"raddr_{port}_i": 5}, waddr_i=5, wdata_i=2, we_i=1)
        t.idle(8, **{f"raddr_{port}_i": 5})
        res.append((f"write -> read {port}", 1, rtl.probe_trace(u, t.cmd(), f"rdata_{port}_o", ev)))
    return res


def seeds():
    """MiniTPU tb scenarios replayed as traces (``harness/vcd_seed.py``)."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract(
        "tb_vpu_alu_regfile",
        ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_regfile.sv",
         "src/core/vpu/vpu_bf16_add_pipe.sv", "src/core/vpu/vpu_bf16_mul.sv",
         "src/core/vpu/vpu_alu.sv"],
        dut="i_rf", clk="clk_i", unit=INSTANCES["w16"],
    )
    return [("tb_vpu_alu_regfile", "w16", cmd, seen, True)]


# ---------------------------------------------------------------------------
# Allo variants. Each ``make(n, w)`` returns a region over per-port arrays of
# n cycles; each runner takes the built module and a command trace (``CMD``
# columns as integer arrays) and returns ``{resp port: object array of ints}``.
# ---------------------------------------------------------------------------

A5 = UInt(5)


def _np(w):
    return np.uint16 if w <= 16 else np.uint32 if w <= 32 else np.uint64


def _args(cmd, n, w):
    """The trace as the region's input arrays (narrow dtypes: the simulator
    checks the element width), plus zeroed outputs."""
    ins = [np.asarray(cmd[p][:n], dtype=np.uint8) for p in CMD[:4]]
    ins.append(np.asarray([int(x) for x in cmd["wdata_i"][:n]], dtype=_np(w)))
    ins.append(np.asarray(cmd["we_i"][:n], dtype=np.uint8))
    outs = [np.zeros(n, dtype=_np(w)) for _ in RESP]
    return ins, outs


def _run_flat(mod, cmd, n, w):
    ins, outs = _args(cmd, n, w)
    mod(*ins, *outs)
    return {p: o for p, o in zip(RESP, outs)}


def trace(n, w=16):
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE, QA, QB, QC])
        def rf(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n],
               qa: W[n], qb: W[n], qc: W[n]):
            mem: W[32]
            for t in range(n):
                ia: int32 = ra[t]  # B4 workaround: widen before indexing
                qa[t] = mem[ia]
                ib: int32 = rb[t]  # B4 workaround: widen before indexing
                qb[t] = mem[ib]
                ic: int32 = rc[t]  # B4 workaround: widen before indexing
                qc[t] = mem[ic]
                iw: int32 = wa[t]  # S6 workaround: every port read unconditional
                dw: W = wd[t]
                if we[t]:
                    mem[iw] = dw

    return top


def trace_raw(n, w=16):
    """``trace`` as first written: indexes ``mem`` by the ``UInt(5)`` port
    value directly. Evidence for B4 (an unsigned index is sign-extended on
    the simulator/LLVM path): kept, not used for verdicts."""
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE, QA, QB, QC])
        def rf(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n],
               qa: W[n], qb: W[n], qc: W[n]):
            mem: W[32]
            for t in range(n):
                qa[t] = mem[ra[t]]
                qb[t] = mem[rb[t]]
                qc[t] = mem[rc[t]]
                if we[t]:
                    mem[wa[t]] = wd[t]

    return top


def annotated(n, w=16):
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE, QA, QB, QC])
        def rf(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n],
               qa: W[n], qb: W[n], qc: W[n]):
            mem: W[32] @ Memory(resource="LUTRAM", storage_type="RAM_1WNR", latency=0, depth=32)
            for t in range(n):
                ia: int32 = ra[t]  # B4 workaround: widen before indexing
                qa[t] = mem[ia]
                ib: int32 = rb[t]  # B4 workaround: widen before indexing
                qb[t] = mem[ib]
                ic: int32 = rc[t]  # B4 workaround: widen before indexing
                qc[t] = mem[ic]
                iw: int32 = wa[t]  # S6 workaround: every port read unconditional
                dw: W = wd[t]
                if we[t]:
                    mem[iw] = dw

    return top


def stateful(n, w=16):
    """Built for ``CHUNK`` cycles whatever ``n``; ``_run_chunked`` calls it
    ``ceil(n / CHUNK)`` times."""
    W = UInt(w)
    c = CHUNK

    @df.region()
    def top(RA: A5[c], RB: A5[c], RC: A5[c], WA: A5[c], WD: W[c], WE: uint1[c],
            QA: W[c], QB: W[c], QC: W[c]):
        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE, QA, QB, QC])
        def rf(ra: A5[c], rb: A5[c], rc: A5[c], wa: A5[c], wd: W[c], we: uint1[c],
               qa: W[c], qb: W[c], qc: W[c]):
            mem: W[32] @ Stateful = 0
            for t in range(c):
                ia: int32 = ra[t]  # B4 workaround: widen before indexing
                qa[t] = mem[ia]
                ib: int32 = rb[t]  # B4 workaround: widen before indexing
                qb[t] = mem[ib]
                ic: int32 = rc[t]  # B4 workaround: widen before indexing
                qc[t] = mem[ic]
                iw: int32 = wa[t]  # S6 workaround: every port read unconditional
                dw: W = wd[t]
                if we[t]:
                    mem[iw] = dw

    return top


def _run_chunked(mod, cmd, n, w):
    """Pads the trace with idle cycles (``we = 0``) to whole chunks; the
    padding writes nothing, so it changes no defined slot."""
    k = -(-n // CHUNK)
    pad = {p: list(cmd[p][:n]) + [0] * (k * CHUNK - n) for p in CMD}
    res = {p: [] for p in RESP}
    for j in range(k):
        part = {p: pad[p][j * CHUNK:(j + 1) * CHUNK] for p in CMD}
        got = _run_flat(mod, part, CHUNK, w)
        for p in RESP:
            res[p].append(got[p])
    return {p: np.concatenate(v)[:n] for p, v in res.items()}


def _port_unit(n, w):
    """C9: a unit's trip count and widths freeze at ``@df.unit``, so the unit
    is decorated inside a factory per (n, w)."""
    W = UInt(w)

    @df.unit()
    def regfile(ra: Stream[UInt(5), 2], rb: Stream[UInt(5), 2], rc: Stream[UInt(5), 2],
                wa: Stream[UInt(5), 2], wd: Stream[UInt(w), 2], we: Stream[uint1, 2],
                qa: Stream[UInt(w), 2], qb: Stream[UInt(w), 2], qc: Stream[UInt(w), 2]):
        mem: W[32]
        for t in range(n):
            a5: UInt(5) = ra.get()
            a: int32 = a5  # B4 workaround; B5: not in one step
            b5: UInt(5) = rb.get()
            b: int32 = b5  # B4 workaround; B5: not in one step
            c5: UInt(5) = rc.get()
            c: int32 = c5  # B4 workaround; B5: not in one step
            x5: UInt(5) = wa.get()
            x: int32 = x5  # B4 workaround; B5: not in one step
            d: UInt(w) = wd.get()
            e: uint1 = we.get()
            qa.put(mem[a])
            qb.put(mem[b])
            qc.put(mem[c])
            if e:
                mem[x] = d

    return regfile


def ported(n, w=16):
    W = UInt(w)
    rf = _port_unit(n, w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        s_ra: Stream[UInt(5), 2]
        s_rb: Stream[UInt(5), 2]
        s_rc: Stream[UInt(5), 2]
        s_wa: Stream[UInt(5), 2]
        s_wd: Stream[UInt(w), 2]
        s_we: Stream[uint1, 2]
        s_qa: Stream[UInt(w), 2]
        s_qb: Stream[UInt(w), 2]
        s_qc: Stream[UInt(w), 2]

        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
        def drive(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                s_ra.put(ra[t])
                s_rb.put(rb[t])
                s_rc.put(rc[t])
                s_wa.put(wa[t])
                s_wd.put(wd[t])
                s_we.put(we[t])

        rf(ra=s_ra, rb=s_rb, rc=s_rc, wa=s_wa, wd=s_wd, we=s_we, qa=s_qa, qb=s_qb, qc=s_qc)

        @df.kernel(mapping=[1], args=[QA, QB, QC])
        def sink(qa: W[n], qb: W[n], qc: W[n]):
            for t in range(n):
                qa[t] = s_qa.get()
                qb[t] = s_qb.get()
                qc[t] = s_qc.get()

    return top


def shared(n, w=16):
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        mem: W[32] @ Stateful = 0  # region scope: one regfile, four port kernels

        @df.kernel(mapping=[1], args=[RA, QA])
        def port_a(ra: A5[n], qa: W[n]):
            for t in range(n):
                ia: int32 = ra[t]  # B4 workaround: widen before indexing
                qa[t] = mem[ia]

        @df.kernel(mapping=[1], args=[RB, QB])
        def port_b(rb: A5[n], qb: W[n]):
            for t in range(n):
                ib: int32 = rb[t]  # B4 workaround: widen before indexing
                qb[t] = mem[ib]

        @df.kernel(mapping=[1], args=[RC, QC])
        def port_c(rc: A5[n], qc: W[n]):
            for t in range(n):
                ic: int32 = rc[t]  # B4 workaround: widen before indexing
                qc[t] = mem[ic]

        @df.kernel(mapping=[1], args=[WA, WD, WE])
        def port_w(wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                iw: int32 = wa[t]  # S6 workaround: every port read unconditional
                dw: W = wd[t]
                if we[t]:
                    mem[iw] = dw

    return top


def shared_sync(n, w=16):
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        mem: W[32] @ Stateful = 0
        rd_a: Stream[uint1, 2]  # reader -> writer: "read of cycle t done"
        rd_b: Stream[uint1, 2]
        rd_c: Stream[uint1, 2]
        wr_a: Stream[uint1, 2]  # writer -> reader: "write of cycle t done"
        wr_b: Stream[uint1, 2]
        wr_c: Stream[uint1, 2]

        @df.kernel(mapping=[1], args=[RA, QA])
        def port_a(ra: A5[n], qa: W[n]):
            for t in range(n):
                if t > 0:
                    wr_a.get()
                ia: int32 = ra[t]  # B4 workaround: widen before indexing
                qa[t] = mem[ia]
                rd_a.put(1)

        @df.kernel(mapping=[1], args=[RB, QB])
        def port_b(rb: A5[n], qb: W[n]):
            for t in range(n):
                if t > 0:
                    wr_b.get()
                ib: int32 = rb[t]  # B4 workaround: widen before indexing
                qb[t] = mem[ib]
                rd_b.put(1)

        @df.kernel(mapping=[1], args=[RC, QC])
        def port_c(rc: A5[n], qc: W[n]):
            for t in range(n):
                if t > 0:
                    wr_c.get()
                ic: int32 = rc[t]  # B4 workaround: widen before indexing
                qc[t] = mem[ic]
                rd_c.put(1)

        @df.kernel(mapping=[1], args=[WA, WD, WE])
        def port_w(wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                rd_a.get()
                rd_b.get()
                rd_c.get()
                iw: int32 = wa[t]  # S6 workaround: every port read unconditional
                dw: W = wd[t]
                if we[t]:
                    mem[iw] = dw
                if t < n - 1:
                    wr_a.put(1)
                    wr_b.put(1)
                    wr_c.put(1)

    return top


def wire(n, w=16):
    """SystemC/Catapult port shape: the unit kernel ``rf`` has only ``Wire``
    ports (``synth_top="rf_0"``); ``src``/``sink`` replay and record."""
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        w_ra: Wire[UInt(5)]
        w_rb: Wire[UInt(5)]
        w_rc: Wire[UInt(5)]
        w_wa: Wire[UInt(5)]
        w_wd: Wire[UInt(w)]
        w_we: Wire[uint1]
        w_qa: Wire[UInt(w)]
        w_qb: Wire[UInt(w)]
        w_qc: Wire[UInt(w)]

        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
        def src(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_rb.put(rb[t])
                w_rc.put(rc[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: W[32]
            for _ in range(n):
                a5: UInt(5) = w_ra.get()
                a: int32 = a5  # B4 workaround; B5: not in one step
                b5: UInt(5) = w_rb.get()
                b: int32 = b5  # B4 workaround; B5: not in one step
                c5: UInt(5) = w_rc.get()
                c: int32 = c5  # B4 workaround; B5: not in one step
                x5: UInt(5) = w_wa.get()
                x: int32 = x5  # B4 workaround; B5: not in one step
                d: UInt(w) = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qb.put(mem[b])
                w_qc.put(mem[c])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA, QB, QC])
        def sink(qa: W[n], qb: W[n], qc: W[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qb[t] = w_qb.get()
                qc[t] = w_qc.get()

    return top


def wire_stateful(n, w=16):
    """Comb-read form (b): ``wire`` with ``mem @ Stateful`` (a module member in
    the emitted SystemC, zeroed in the reset action) instead of a kernel-local
    array. ``dev/records/minitpu/u2_comb_read_2026-10-02.rst``."""
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        w_ra: Wire[UInt(5)]
        w_rb: Wire[UInt(5)]
        w_rc: Wire[UInt(5)]
        w_wa: Wire[UInt(5)]
        w_wd: Wire[UInt(w)]
        w_we: Wire[uint1]
        w_qa: Wire[UInt(w)]
        w_qb: Wire[UInt(w)]
        w_qc: Wire[UInt(w)]

        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
        def src(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_rb.put(rb[t])
                w_rc.put(rc[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: W[32] @ Stateful = 0
            for _ in range(n):
                a5: UInt(5) = w_ra.get()
                a: int32 = a5  # B4 workaround; B5: not in one step
                b5: UInt(5) = w_rb.get()
                b: int32 = b5  # B4 workaround; B5: not in one step
                c5: UInt(5) = w_rc.get()
                c: int32 = c5  # B4 workaround; B5: not in one step
                x5: UInt(5) = w_wa.get()
                x: int32 = x5  # B4 workaround; B5: not in one step
                d: UInt(w) = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qb.put(mem[b])
                w_qc.put(mem[c])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA, QB, QC])
        def sink(qa: W[n], qb: W[n], qc: W[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qb[t] = w_qb.get()
                qc[t] = w_qc.get()

    return top


def comb_read(n, w=16):
    """D-13: ``wire`` with the read ports declared combinational
    (``Wire[UInt(w), comb]``). The body is ``wire``'s: reads, then the write."""
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        w_ra: Wire[UInt(5)]
        w_rb: Wire[UInt(5)]
        w_rc: Wire[UInt(5)]
        w_wa: Wire[UInt(5)]
        w_wd: Wire[UInt(w)]
        w_we: Wire[uint1]
        w_qa: Wire[UInt(w), comb]
        w_qb: Wire[UInt(w), comb]
        w_qc: Wire[UInt(w), comb]

        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
        def src(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_rb.put(rb[t])
                w_rc.put(rc[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: W[32]
            for _ in range(n):
                a5: UInt(5) = w_ra.get()
                a: int32 = a5  # B4 workaround; B5: not in one step
                b5: UInt(5) = w_rb.get()
                b: int32 = b5  # B4 workaround; B5: not in one step
                c5: UInt(5) = w_rc.get()
                c: int32 = c5  # B4 workaround; B5: not in one step
                x5: UInt(5) = w_wa.get()
                x: int32 = x5  # B4 workaround; B5: not in one step
                d: UInt(w) = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qb.put(mem[b])
                w_qc.put(mem[c])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA, QB, QC])
        def sink(qa: W[n], qb: W[n], qc: W[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qb[t] = w_qb.get()
                qc[t] = w_qc.get()

    return top


def comb_unreset(n, w=16):
    """D-14: ``comb`` with ``mem`` declared ``@ Stateful(reset=False)``: the
    storage survives reset, as MiniTPU's does (``rst_ni`` unused). The SystemC
    emitter moves the write to a clock-edge ``SC_METHOD`` with no reset action
    and ``run.tcl`` gets ``-RESET_CLEARS_ALL_REGS no``. The body is ``comb``'s."""
    W = UInt(w)

    @df.region()
    def top(RA: A5[n], RB: A5[n], RC: A5[n], WA: A5[n], WD: W[n], WE: uint1[n],
            QA: W[n], QB: W[n], QC: W[n]):
        w_ra: Wire[UInt(5)]
        w_rb: Wire[UInt(5)]
        w_rc: Wire[UInt(5)]
        w_wa: Wire[UInt(5)]
        w_wd: Wire[UInt(w)]
        w_we: Wire[uint1]
        w_qa: Wire[UInt(w), comb]
        w_qb: Wire[UInt(w), comb]
        w_qc: Wire[UInt(w), comb]

        @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
        def src(ra: A5[n], rb: A5[n], rc: A5[n], wa: A5[n], wd: W[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_rb.put(rb[t])
                w_rc.put(rc[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: W[32] @ Stateful(reset=False)
            for _ in range(n):
                a5: UInt(5) = w_ra.get()
                a: int32 = a5  # B4 workaround; B5: not in one step
                b5: UInt(5) = w_rb.get()
                b: int32 = b5  # B4 workaround; B5: not in one step
                c5: UInt(5) = w_rc.get()
                c: int32 = c5  # B4 workaround; B5: not in one step
                x5: UInt(5) = w_wa.get()
                x: int32 = x5  # B4 workaround; B5: not in one step
                d: UInt(w) = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qb.put(mem[b])
                w_qc.put(mem[c])
                if e:
                    mem[x] = d

        @df.kernel(mapping=[1], args=[QA, QB, QC])
        def sink(qa: W[n], qb: W[n], qc: W[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qb[t] = w_qb.get()
                qc[t] = w_qc.get()

    return top


VARIANTS = {
    "trace": (trace, _run_flat),
    "trace_raw": (trace_raw, _run_flat),
    "stateful": (stateful, _run_chunked),
    "ported": (ported, _run_flat),
    "shared": (shared, _run_flat),
    "shared_sync": (shared_sync, _run_flat),
    "annotated": (annotated, _run_flat),
    "wire": (wire, _run_flat),
    "wire_stateful": (wire_stateful, _run_flat),
    "comb": (comb_read, _run_flat),
    "comb_unreset": (comb_unreset, _run_flat),
}

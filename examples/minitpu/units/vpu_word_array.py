# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2: ``vpu_word_array``, VMEM: one array of whole words, two ports.

Ports ``compute`` and ``dma`` are symmetric: ``en``/``we``/``addr``/``wdata``
in, ``rdata`` out, whole words only. Reads are registered through a pipe of
``READ_LATENCY`` (compute, ``vpu_pkg::VMEM_READ_LATENCY = 3``) or
``DMA_READ_LATENCY`` (``VMEM_DMA_READ_LATENCY = 2``) stages. No reset. Driven
as a ``trace`` unit with both ``rdata`` ports sampled after the edge; this is
the simulation model (the ``ifndef SYNTHESIS`` branch), not the XPM
``xpm_memory_tdpram`` the board builds. Reference: ``harness/ref.py``
``vpu_word_array_trace``.

Instances (``vpu_pkg`` defines and ``-G`` parameters):

``narrow``      ``MINITPU_NUM_LANES=1``, ``MINITPU_VMEM_ENTRIES_PER_LANE=16``:
                8 words of 64 b, latencies 3/2 -- dense collisions, fast
``narrow_rl2``  the same with ``READ_LATENCY=2``
``narrow16``    ``MINITPU_NUM_SUBLANES=1`` too: 32 words of 16 b (added for the SystemC
                csim column, whose testbench moves data through ``long long``)
``mid``         ``MINITPU_NUM_LANES=1`` alone: 4096 words of 64 b (the Catapult RAM-mapping probe)
``small``       ``MINITPU_NUM_LANES=1``, ``MINITPU_VMEM_ENTRIES_PER_LANE=64``: 32 words of
                64 b -- the smallest OpenRAM 2RW macro, the SRAM path's dry run
                (``asic_memories_2026-10-04.rst``)
``full``        the default geometry: 4096 words of 1024 b, latencies 3/2
``full_rl2``    ``READ_LATENCY=2``: what ``tb_vpu_word_array`` instantiates

Legal traces keep the two ports off one word in a cycle when either writes
(``vpu_vmem_simd.sv:121-131`` asserts it in simulation; nothing else checks
it). Illegal traces break that rule on purpose.

Allo variants (U2, ``dev/records/minitpu/u2_word_array_2026-10-02.rst``),
mirroring ``vpu_regfile.py``'s: every variant is driven by the SAME per-cycle
command trace as the RTL (one array per input port, one element per cycle)
and returns one array per ``rdata`` port, one element per cycle. Addresses
are ``UInt(AW)``, data ``UInt(W)``, ``en``/``we`` ``uint1``.

The new element against the regfile is the **registered read**: a read
issued in cycle ``t`` leaves ``rdata`` in row ``t + L - 1`` (``L`` = 3 on
the compute port, 2 on the DMA port). Two expressions:

``trace`` (plan W1, pipe as data)
    One kernel, kernel-local ``mem: W[WORDS]`` and one shift-register pipe
    per port (``pc: W[3]``, ``pd: W[2]``), the RTL's ``*_read_pipe``
    transcribed: per iteration the pipes shift, each enabled port reads
    (*old* word, before this cycle's writes) or writes -- one access per
    port per cycle, as the board's XPM port does (``WRITE_MODE no_change``:
    on a write cycle the pipe head holds, which is a masked slot) -- then
    ``q[t] = pipe[L - 1]``. The latency is an iteration shift inside the
    kernel, so every backend reproduces the row offset by construction.
``issue`` (W1 as the plan words it, latency left to the RTL)
    The same accesses with no pipe: ``q[t]`` is the read issued at ``t``.
    ``RESP_SHIFT`` tells ``check.py`` to compare ``q[t]`` with the RTL's row
    ``t + L - 1``. The delivery delay is then a contract outside the kernel
    (D-10's ``latency=L``), which no backend can state per port today.
``trace_rw``
    The simulation model transcribed literally: every enabled port reads
    *and* (if ``we``) writes in one cycle -- two accesses per port per
    cycle. Same defined slots as ``trace``; the Catapult probe of a 4-access
    iteration.
``ported`` (R3)
    ``trace`` as a ``@df.unit`` with 8 command + 2 response ``Stream`` ports,
    lockstep, between a driver and a sink kernel.
``shared`` / ``shared_sync`` (W2: the true two-client case)
    A region-scope ``mem @ Stateful`` and one kernel per physical port
    (``compute`` and ``dma``), each with its own read pipe. ``shared`` has
    no link between them; ``shared_sync`` adds a per-cycle barrier (a token
    each way), which is the ordering the RTL gets from the clock. Within a
    cycle the order of the two ports is free: same-word collisions are
    undefined and masked.
``annotated``
    ``trace`` with ``mem @ Memory(resource="URAM", storage_type="RAM_T2P",
    latency=3, depth=WORDS)``: the D-1 sweep (what reaches each backend).
``wire``
    ``trace``'s body with every port a ``Wire`` (``sc_in``/``sc_out``), the
    port shape of MiniTPU's module; unit kernel ``wa`` (``synth_top="wa_0"``).
    SystemC/Catapult only. Each port's access is one ``if we: write else:
    read`` (two RAM ports for Catapult, not four); the two shift loops must
    be ``s.unroll``-ed, or Catapult merges them into the pipelined loop.

Workarounds carried from the regfile pilot: B4 (every address widened to
``int32`` before indexing), B5 (a stream/wire address read into ``UInt(AW)``
first), S6 (every port array read unconditionally; only the access is under
``if``).
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stateful, Stream, UInt, Wire, int32, uint1
from allo.memory import Memory

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, hot_addr, rng_for, word

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_word_array.sv"]
NARROW = ["MINITPU_NUM_LANES=1", "MINITPU_VMEM_ENTRIES_PER_LANE=16"]
NARROW16 = NARROW + ["MINITPU_NUM_SUBLANES=1"]
SMALL = ["MINITPU_NUM_LANES=1", "MINITPU_VMEM_ENTRIES_PER_LANE=64"]
GEOM = {  # instance: (word bits, words, address bits, compute latency, dma latency)
    "narrow": (64, 8, 3, 3, 2),
    "narrow16": (16, 32, 5, 3, 2),  # NUM_SUBLANES = 1: words below 64 b (csim's long long path, S8)
    "narrow_rl2": (64, 8, 3, 2, 2),
    "mid": (64, 4096, 12, 3, 2),  # NUM_LANES = 1 at the default depth: a RAM-sized array of 64 b words
    "small": (64, 32, 5, 3, 2),  # NUM_LANES = 1, 64 entries: the smallest OpenRAM 2RW macro (SRAM dry run)
    "full": (1024, 4096, 12, 3, 2),
    "full_rl2": (1024, 4096, 12, 2, 2),
}
PORTS = ("compute", "dma")


def _unit(inst):
    ww, _, aw, rl, drl = GEOM[inst]
    ins = [("rst_ni", 1)]
    for p in PORTS:
        ins += [(f"{p}_en_i", 1), (f"{p}_we_i", 1), (f"{p}_addr_i", aw), (f"{p}_wdata_i", ww)]
    return rtl.RtlUnit(
        top="vpu_word_array",
        sources=SOURCES,
        inputs=ins,
        outputs=[(f"{p}_rdata_o", ww, "post") for p in PORTS],
        shape="trace",
        defines=NARROW16 if inst == "narrow16" else NARROW if inst.startswith("narrow")
        else NARROW[:1] if inst == "mid" else SMALL if inst == "small" else [],
        params={} if rl == 3 else {"READ_LATENCY": rl},
        assertions=True,
    )


INSTANCES = {k: _unit(k) for k in GEOM}
WIDTH = {k: g[0] for k, g in GEOM.items()}
DEFAULT = "narrow"
RTL = INSTANCES[DEFAULT]
CMD = tuple(f"{p}_{f}_i" for p in PORTS for f in ("en", "we", "addr", "wdata"))
RESP = tuple(f"{p}_rdata_o" for p in PORTS)
LATENCY_SOURCE = "vpu_pkg.sv:58-59 (VMEM_READ_LATENCY = 3, VMEM_DMA_READ_LATENCY = 2)"


def REF(inst, cmd):
    ww, _, _, rl, drl = GEOM[inst]
    return ref.vpu_word_array_trace(cmd, (ww + 63) // 64, rl, drl)


def _defaults():
    d = {"rst_ni": 1}
    for p in PORTS:
        d.update({f"{p}_en_i": 0, f"{p}_we_i": 0, f"{p}_addr_i": 0, f"{p}_wdata_i": 0})
    return d


def _acc(p, addr, data=None):
    """One port's access: a read, or a write of ``data``."""
    a = {f"{p}_en_i": 1, f"{p}_addr_i": addr}
    if data is not None:
        a.update({f"{p}_we_i": 1, f"{p}_wdata_i": data})
    return a


def random_trace(inst, n, seed, legal):
    ww, words, _, _, _ = GEOM[inst]
    rng = rng_for("word_array", inst, seed, legal)
    hot = rng.sample(range(words), 3)
    t = Trace(_defaults())
    for _ in range(n):
        row = {}
        for p in PORTS:
            if rng.random() < 0.75:
                row.update(_acc(p, hot_addr(rng, words, hot),
                                word(rng, ww) if rng.random() < 0.4 else None))
        c, d = row.get("compute_addr_i"), row.get("dma_addr_i")
        if legal and "compute_en_i" in row and "dma_en_i" in row and c == d and (
                "compute_we_i" in row or "dma_we_i" in row):
            row["dma_addr_i"] = (d + 1) % words
        row["rst_ni"] = int(rng.random() > 0.01)
        t.cycle(**row)
    return t.cmd()


def directed(inst):
    """``[(label, cmd, legal)]``: every port pair, write then read at every
    offset -1..+L; same-port read-during-write; both ports reading one word;
    both ports writing one word, and one writing while the other reads; the
    bottom, an interior and the top address; idle cycles between reads."""
    ww, words, _, rl, drl = GEOM[inst]
    lat = {"compute": rl, "dma": drl}
    rng = rng_for("word_array-directed", inst)
    targets = [0, words // 2 + 1, words - 1]
    out = []
    for wp in PORTS:
        for rp in PORTS:
            legal = Trace(_defaults())
            coll = Trace(_defaults())
            for a in targets:
                for t in (legal, coll):
                    t.cycle(**_acc(wp, a, word(rng, ww))).idle(lat[rp] + 1)
                for off in range(-1, lat[rp] + 1):
                    t = coll if (off == 0 and wp != rp) else legal
                    seq = [dict() for _ in range(off + 3)] if off >= 0 else [dict() for _ in range(3)]
                    wi = 1 if off >= 0 else 2
                    seq[wi].update(_acc(wp, a, word(rng, ww)))
                    ri = wi + off
                    if wp == rp and off == 0:
                        pass  # the write's own cycle: covered by "rdw-same-port"
                    else:
                        seq[ri].update(_acc(rp, a))
                    for row in seq:
                        t.cycle(**row)
                    t.idle(lat[rp] + 1)
            out.append((f"wr-offsets-{wp}->{rp}", legal.cmd(), True))
            if wp != rp:
                out.append((f"wr-same-cycle-{wp}->{rp}", coll.cmd(), False))
    t = Trace(_defaults())  # same-port read-during-write: the write cycle's read slot
    for p in PORTS:
        for a in targets:
            t.cycle(**_acc(p, a, word(rng, ww)))
            t.cycle(**_acc(p, a, word(rng, ww)))  # read-during-write of a written word
            t.cycle(**_acc(p, a)).idle(lat[p] + 1)
    out.append(("rdw-same-port", t.cmd(), True))
    t = Trace(_defaults())  # both ports read one word in one cycle: legal
    for a in targets:
        t.cycle(**_acc("compute", a, word(rng, ww))).idle(1)
        t.cycle(**_acc("compute", a), **_acc("dma", a)).idle(max(lat.values()) + 1)
    out.append(("both-read-one-word", t.cmd(), True))
    t = Trace(_defaults())  # both ports write one word: undefined until rewritten
    for a in targets:
        t.cycle(**_acc("compute", a, word(rng, ww)), **_acc("dma", a, word(rng, ww)))
        t.cycle(**_acc("compute", a)).cycle(**_acc("dma", a)).idle(4)
        t.cycle(**_acc("dma", a, word(rng, ww))).cycle(**_acc("compute", a)).idle(4)
    out.append(("ww-collision", t.cmd(), False))
    t = Trace(_defaults())  # gaps: the pipe holds its last read (masked "no read")
    for a in targets:
        t.cycle(**_acc("dma", a, word(rng, ww)))
    for a in targets:
        t.cycle(**_acc("compute", a), **_acc("dma", targets[0])).idle(5)
    out.append(("idle-gaps", t.cmd(), True))
    return out


def traces(inst):
    n = 20000 if inst.startswith("narrow") else 3000
    tr = directed(inst)
    tr += [(f"random-legal-{s}", random_trace(inst, n, s, True), True) for s in range(2)]
    tr += [(f"random-any-{s}", random_trace(inst, n, s, False), False) for s in range(2)]
    return tr


def probes(inst):
    """Read latency per port (a step from one word to another), and write
    visibility (the first read that sees a write, minus the read latency),
    same port and across ports."""
    ww, words, _, rl, drl = GEOM[inst]
    lat = {"compute": rl, "dma": drl}
    u = INSTANCES[inst]
    one, two = 1, (1 << ww) - 2
    res = {}
    out = []
    for p in PORTS:
        o = "dma" if p == "compute" else "compute"
        t = Trace(_defaults())
        t.cycle(**_acc(o, 1, one)).cycle(**_acc(o, 2, two)).idle(2)
        t.idle(8, **_acc(p, 1))
        ev = len(t)
        t.idle(8, **_acc(p, 2))
        res[p] = rtl.probe_trace(u, t.cmd(), f"{p}_rdata_o", ev)
        out.append((f"read {p}", lat[p], res[p]))
    for wp in PORTS:
        for rp in PORTS:
            t = Trace(_defaults())
            t.cycle(**_acc(wp, 3, one)).idle(2)
            t.idle(8, **_acc(rp, 3))
            ev = len(t)
            row = _acc(wp, 3, two)
            if wp != rp:
                row.update({f"{rp}_en_i": 0})  # the reader skips the write's cycle
            t.cycle(**row)
            t.idle(8, **_acc(rp, 3))
            out.append((f"write {wp} -> read {rp} (visibility)", 1,
                        rtl.probe_trace(u, t.cmd(), f"{rp}_rdata_o", ev) - res[rp]))
    return out


TB_SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_word_array.sv",
              "src/core/vpu/vpu_dma_group.sv"]


def seeds():
    """``tb_vpu_word_array`` (DMA beats through ``vpu_dma_group``, compute
    words, ``READ_LATENCY = 2``), replayed at 2 against the tb's own DUT and
    at the production 3 against the reference only."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract("tb_vpu_word_array", TB_SOURCES, dut="dut",
                                 clk="clk_i", unit=INSTANCES["full_rl2"])
    return [("tb_vpu_word_array", "full_rl2", cmd, seen, True),
            ("tb_vpu_word_array@RL3", "full", cmd, None, True)]


# ---------------------------------------------------------------------------
# Allo variants. Each ``make(n, w, inst)`` returns a region over per-port
# arrays of n cycles (``w`` is the word width, ``inst`` the geometry); each
# runner takes the built module and a command trace and returns
# ``{resp port: array of ints}``.
# ---------------------------------------------------------------------------


def _np(w):
    return np.uint8 if w <= 8 else np.uint16 if w <= 16 else np.uint32 if w <= 32 else np.uint64


def _args(cmd, n, w):
    """The trace as the region's input arrays (narrow dtypes: the simulator
    checks the element width), plus zeroed outputs."""
    ins = []
    for p in CMD:
        if p.endswith("wdata_i"):
            ins.append(np.asarray([int(x) for x in cmd[p][:n]], dtype=_np(w)))
        elif p.endswith("addr_i"):
            ins.append(np.asarray(cmd[p][:n], dtype=np.uint16 if max(cmd[p][:n]) > 255 else np.uint8))
        else:
            ins.append(np.asarray(cmd[p][:n], dtype=np.uint8))
    outs = [np.zeros(n, dtype=_np(w)) for _ in RESP]
    return ins, outs


def _run_flat(mod, cmd, n, w):
    ins, outs = _args(cmd, n, w)
    mod(*ins, *outs)
    return {p: o for p, o in zip(RESP, outs)}


def _geom(inst):
    ww, words, aw, rl, drl = GEOM[inst]
    return ww, words, aw, rl, drl


def trace(n, w=64, inst="narrow"):
    ww, words, aw, rl, drl = _geom(inst)
    assert ww == w
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD, QC, QD])
        def wa(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
               de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qc: W[n], qd: W[n]):
            mem: W[words]
            pc: W[rl]  # compute read pipe, the RTL's compute_read_pipe
            pd: W[drl]  # DMA read pipe
            for t in range(n):
                # the pipes shift (stage k takes stage k - 1)
                for k in range(1, rl):  # negative-step range is refused (minor finding)
                    pc[rl - k] = pc[rl - k - 1]
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_c: uint1 = ce[t]  # S6: every port array read unconditionally
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4: widen before indexing
                d_c: W = cd[t]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                # reads first: a read sees the word before this cycle's writes
                if e_c == 1:
                    if w_c == 0:
                        pc[0] = mem[a_c]
                if e_d == 1:
                    if w_d == 0:
                        pd[0] = mem[a_d]
                if e_c == 1:
                    if w_c == 1:
                        mem[a_c] = d_c
                if e_d == 1:
                    if w_d == 1:
                        mem[a_d] = d_d
                qc[t] = pc[rl - 1]
                qd[t] = pd[drl - 1]

    return top


def trace_rw(n, w=64, inst="narrow"):
    """The simulation model literally: an enabled port reads the old word into
    its pipe whether or not it writes (two accesses per port per cycle)."""
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD, QC, QD])
        def wa(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
               de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qc: W[n], qd: W[n]):
            mem: W[words]
            pc: W[rl]
            pd: W[drl]
            for t in range(n):
                for k in range(1, rl):  # negative-step range is refused (minor finding)
                    pc[rl - k] = pc[rl - k - 1]
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_c: uint1 = ce[t]
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4
                d_c: W = cd[t]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                if e_c == 1:
                    pc[0] = mem[a_c]
                if e_d == 1:
                    pd[0] = mem[a_d]
                if e_c == 1:
                    if w_c == 1:
                        mem[a_c] = d_c
                if e_d == 1:
                    if w_d == 1:
                        mem[a_d] = d_d
                qc[t] = pc[rl - 1]
                qd[t] = pd[drl - 1]

    return top


def issue(n, w=64, inst="narrow"):
    """No pipe: ``q[t]`` is the word read by the access issued at ``t`` (the
    last read's word when ``t`` holds no read). ``RESP_SHIFT`` places it."""
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD, QC, QD])
        def wa(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
               de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qc: W[n], qd: W[n]):
            mem: W[words]
            hc: W = 0
            hd: W = 0
            for t in range(n):
                e_c: uint1 = ce[t]
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4
                d_c: W = cd[t]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                if e_c == 1:
                    if w_c == 0:
                        hc = mem[a_c]
                if e_d == 1:
                    if w_d == 0:
                        hd = mem[a_d]
                if e_c == 1:
                    if w_c == 1:
                        mem[a_c] = d_c
                if e_d == 1:
                    if w_d == 1:
                        mem[a_d] = d_d
                qc[t] = hc
                qd[t] = hd

    return top


def _issue_shift(inst):
    _, _, _, rl, drl = GEOM[inst]
    return {"compute_rdata_o": rl - 1, "dma_rdata_o": drl - 1}


# ``check.py``: variant -> (inst -> {resp port: k}); ``got[t]`` is compared
# with the RTL's row ``t + k``.
RESP_SHIFT = {"issue": _issue_shift}


def annotated(n, w=64, inst="narrow"):
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD, QC, QD])
        def wa(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
               de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qc: W[n], qd: W[n]):
            mem: W[words] @ Memory(resource="URAM", storage_type="RAM_T2P", latency=3, depth=words)
            pc: W[rl]
            pd: W[drl]
            for t in range(n):
                for k in range(1, rl):  # negative-step range is refused (minor finding)
                    pc[rl - k] = pc[rl - k - 1]
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_c: uint1 = ce[t]
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4
                d_c: W = cd[t]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                if e_c == 1:
                    if w_c == 0:
                        pc[0] = mem[a_c]
                if e_d == 1:
                    if w_d == 0:
                        pd[0] = mem[a_d]
                if e_c == 1:
                    if w_c == 1:
                        mem[a_c] = d_c
                if e_d == 1:
                    if w_d == 1:
                        mem[a_d] = d_d
                qc[t] = pc[rl - 1]
                qd[t] = pd[drl - 1]

    return top


def _port_unit(n, w, inst):
    """C9: a unit's trip count and widths freeze at ``@df.unit``, so the unit
    is decorated inside a factory per (n, w, inst)."""
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.unit()
    def word_array(ce: Stream[uint1, 2], cw: Stream[uint1, 2], ca: Stream[UInt(aw), 2],
                   cd: Stream[UInt(w), 2], de: Stream[uint1, 2], dw: Stream[uint1, 2],
                   da: Stream[UInt(aw), 2], dd: Stream[UInt(w), 2],
                   qc: Stream[UInt(w), 2], qd: Stream[UInt(w), 2]):
        mem: W[words]
        pc: W[rl]
        pd: W[drl]
        for t in range(n):
            for k in range(1, rl):
                pc[rl - k] = pc[rl - k - 1]
            for j in range(1, drl):
                pd[drl - j] = pd[drl - j - 1]
            e_c: uint1 = ce.get()
            w_c: uint1 = cw.get()
            a_ca: UInt(aw) = ca.get()
            a_c: int32 = a_ca  # B4; B5: not in one step
            d_c: UInt(w) = cd.get()
            e_d: uint1 = de.get()
            w_d: uint1 = dw.get()
            a_da: UInt(aw) = da.get()
            a_d: int32 = a_da  # B4; B5
            d_d: UInt(w) = dd.get()
            if e_c == 1:
                if w_c == 0:
                    pc[0] = mem[a_c]
            if e_d == 1:
                if w_d == 0:
                    pd[0] = mem[a_d]
            if e_c == 1:
                if w_c == 1:
                    mem[a_c] = d_c
            if e_d == 1:
                if w_d == 1:
                    mem[a_d] = d_d
            qc.put(pc[rl - 1])
            qd.put(pd[drl - 1])

    return word_array


def ported(n, w=64, inst="narrow"):
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)
    unit = _port_unit(n, w, inst)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        s_ce: Stream[uint1, 2]
        s_cw: Stream[uint1, 2]
        s_ca: Stream[UInt(aw), 2]
        s_cd: Stream[UInt(w), 2]
        s_de: Stream[uint1, 2]
        s_dw: Stream[uint1, 2]
        s_da: Stream[UInt(aw), 2]
        s_dd: Stream[UInt(w), 2]
        s_qc: Stream[UInt(w), 2]
        s_qd: Stream[UInt(w), 2]

        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD])
        def drive(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
                  de: uint1[n], dw: uint1[n], da: A[n], dd: W[n]):
            for t in range(n):
                s_ce.put(ce[t])
                s_cw.put(cw[t])
                s_ca.put(ca[t])
                s_cd.put(cd[t])
                s_de.put(de[t])
                s_dw.put(dw[t])
                s_da.put(da[t])
                s_dd.put(dd[t])

        unit(ce=s_ce, cw=s_cw, ca=s_ca, cd=s_cd, de=s_de, dw=s_dw, da=s_da, dd=s_dd,
             qc=s_qc, qd=s_qd)

        @df.kernel(mapping=[1], args=[QC, QD])
        def sink(qc: W[n], qd: W[n]):
            for t in range(n):
                qc[t] = s_qc.get()
                qd[t] = s_qd.get()

    return top


def _shared(n, w, inst, sync):
    """``sync=False``: no link between the two port kernels; ``sync=True``: a
    per-cycle barrier. (Two kernel bodies: a closure ``bool`` under ``if`` is
    lowered as an ``i32`` constant and the verifier refuses the ``scf.if``.)"""
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        mem: W[words] @ Stateful = 0  # region scope: one memory, two port kernels

        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, QC])
        def compute(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n], qc: W[n]):
            pc: W[rl]
            for t in range(n):
                for k in range(1, rl):
                    pc[rl - k] = pc[rl - k - 1]
                e_c: uint1 = ce[t]
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4
                d_c: W = cd[t]
                if e_c == 1:
                    if w_c == 0:
                        pc[0] = mem[a_c]
                    else:
                        mem[a_c] = d_c
                qc[t] = pc[rl - 1]

        @df.kernel(mapping=[1], args=[DE, DW, DA, DD, QD])
        def dma(de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qd: W[n]):
            pd: W[drl]
            for t in range(n):
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                if e_d == 1:
                    if w_d == 0:
                        pd[0] = mem[a_d]
                    else:
                        mem[a_d] = d_d
                qd[t] = pd[drl - 1]

    @df.region()
    def top_sync(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
                 DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        mem: W[words] @ Stateful = 0
        c2d: Stream[uint1, 2]  # compute -> dma: "cycle t done"
        d2c: Stream[uint1, 2]  # dma -> compute

        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, QC])
        def compute(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n], qc: W[n]):
            pc: W[rl]
            for t in range(n):
                for k in range(1, rl):
                    pc[rl - k] = pc[rl - k - 1]
                e_c: uint1 = ce[t]
                w_c: uint1 = cw[t]
                a_c: int32 = ca[t]  # B4
                d_c: W = cd[t]
                if e_c == 1:
                    if w_c == 0:
                        pc[0] = mem[a_c]
                    else:
                        mem[a_c] = d_c
                qc[t] = pc[rl - 1]
                c2d.put(1)
                d2c.get()

        @df.kernel(mapping=[1], args=[DE, DW, DA, DD, QD])
        def dma(de: uint1[n], dw: uint1[n], da: A[n], dd: W[n], qd: W[n]):
            pd: W[drl]
            for t in range(n):
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_d: uint1 = de[t]
                w_d: uint1 = dw[t]
                a_d: int32 = da[t]  # B4
                d_d: W = dd[t]
                if e_d == 1:
                    if w_d == 0:
                        pd[0] = mem[a_d]
                    else:
                        mem[a_d] = d_d
                qd[t] = pd[drl - 1]
                d2c.put(1)
                c2d.get()

    return top_sync if sync else top


def shared(n, w=64, inst="narrow"):
    return _shared(n, w, inst, False)


def shared_sync(n, w=64, inst="narrow"):
    return _shared(n, w, inst, True)


def wire(n, w=64, inst="narrow"):
    """SystemC/Catapult port shape: the unit kernel ``wa`` has only ``Wire``
    ports (``synth_top="wa_0"``); ``src``/``sink`` replay and record. Wires
    are declared in ``CMD`` then ``RESP`` order (``cmp_trace`` maps by it)."""
    ww, words, aw, rl, drl = _geom(inst)
    W, A = UInt(w), UInt(aw)

    @df.region()
    def top(CE: uint1[n], CW: uint1[n], CA: A[n], CD: W[n],
            DE: uint1[n], DW: uint1[n], DA: A[n], DD: W[n], QC: W[n], QD: W[n]):
        w_ce: Wire[uint1]
        w_cw: Wire[uint1]
        w_ca: Wire[UInt(aw)]
        w_cd: Wire[UInt(w)]
        w_de: Wire[uint1]
        w_dw: Wire[uint1]
        w_da: Wire[UInt(aw)]
        w_dd: Wire[UInt(w)]
        w_qc: Wire[UInt(w)]
        w_qd: Wire[UInt(w)]

        @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD])
        def src(ce: uint1[n], cw: uint1[n], ca: A[n], cd: W[n],
                de: uint1[n], dw: uint1[n], da: A[n], dd: W[n]):
            for t in range(n):
                w_ce.put(ce[t])
                w_cw.put(cw[t])
                w_ca.put(ca[t])
                w_cd.put(cd[t])
                w_de.put(de[t])
                w_dw.put(dw[t])
                w_da.put(da[t])
                w_dd.put(dd[t])

        @df.kernel(mapping=[1], args=[])
        def wa():
            mem: W[words]
            pc: W[rl]
            pd: W[drl]
            for _ in range(n):
                for k in range(1, rl):  # negative-step range is refused (minor finding)
                    pc[rl - k] = pc[rl - k - 1]
                for j in range(1, drl):  # own variable: Catapult unrolls it by name
                    pd[drl - j] = pd[drl - j - 1]
                e_c: uint1 = w_ce.get()
                w_c: uint1 = w_cw.get()
                a_ca: UInt(aw) = w_ca.get()
                a_c: int32 = a_ca  # B4; B5
                d_c: UInt(w) = w_cd.get()
                e_d: uint1 = w_de.get()
                w_d: uint1 = w_dw.get()
                a_da: UInt(aw) = w_da.get()
                a_d: int32 = a_da  # B4; B5
                d_d: UInt(w) = w_dd.get()
                # one access per port per cycle, as ONE if/else: Catapult then
                # sees the read and the write of a port as exclusive and needs
                # two RAM ports, not four. The order of the two ports is free:
                # a same-word cross-port write is an undefined (masked) slot.
                if e_c == 1:
                    if w_c == 1:
                        mem[a_c] = d_c
                    else:
                        pc[0] = mem[a_c]
                if e_d == 1:
                    if w_d == 1:
                        mem[a_d] = d_d
                    else:
                        pd[0] = mem[a_d]
                w_qc.put(pc[rl - 1])
                w_qd.put(pd[drl - 1])

        @df.kernel(mapping=[1], args=[QC, QD])
        def sink(qc: W[n], qd: W[n]):
            for t in range(n):
                qc[t] = w_qc.get()
                qd[t] = w_qd.get()

    return top


VARIANTS = {
    "trace": (trace, _run_flat),
    "trace_rw": (trace_rw, _run_flat),
    "issue": (issue, _run_flat),
    "ported": (ported, _run_flat),
    "shared": (shared, _run_flat),
    "shared_sync": (shared_sync, _run_flat),
    "annotated": (annotated, _run_flat),
    "wire": (wire, _run_flat),
}

# README D-12: the two ports as two units on one ``compose.Memory``
# (``vpu_word_array_d12.py``); ``_wire`` is the Catapult port shape.
from examples.minitpu.units import vpu_word_array_d12 as _d12  # noqa: E402

VARIANTS["d12_server"] = (_d12.make("simulator"), _run_flat)
VARIANTS["d12_server_wire"] = (_d12.make("systemc"), _run_flat)

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U2: ``vpu_fifo``, the small LUTRAM FIFO, and the MXU's instances of it.

``push``/``push_data``/``pop`` in; ``pop_data``/``empty``/``full`` out, all
combinational from the state (``pop_data = mem[rd_ptr]``: first-word-fall-
through), so every output is sampled before the edge. Pointers and count are
reset; ``mem`` is not. ``DEPTH`` must be a power of two >= 2. Reference:
``harness/ref.py`` ``vpu_fifo_trace``.

Instances:

``w32d4``   the module defaults (``WIDTH = 32``, ``DEPTH = 4``)
``input``   ``mxu.sv:97``'s input FIFO: ``1 + DIM*16 = 257`` b, ``MXU_FIFO_DEPTH = 4``
``output``  ``mxu.sv:154``'s output FIFO, one per lane: ``4*16 = 64`` b (one
            gathered group of ``NUM_SUBLANES`` rows), ``MXU_OUTPUT_FIFO_DEPTH/4 =
            16`` entries. The gather in front of it (``mxu.sv:128-152``) is not
            part of the unit (U2 plan, Q3).

Legal traces obey the simulation assertions (no push on full without a pop,
no pop on empty without a push) and also never push and pop together on an
empty FIFO, which no assertion catches but which loses the pushed word.
Illegal traces do all three; overflow on ``output`` is H7's "the RTL drops".

Allo variants (``dev/records/minitpu/u2_fifo_2026-10-02.rst``). Every variant
is driven by the SAME per-cycle command trace as the RTL (one array per input
port, ``rst_ni`` included, one element per cycle) and returns one array per
output port. Data is ``UInt(W)``, the flags ``uint1``. "Cycle ``t``" is
iteration ``t`` of the unit's loop (see ``vpu_regfile.py``). ``make(n, w)``
takes the width because ``check.py`` does; the depth is ``DEPTH_OF[w]``.

``trace`` (plan F2, "bits")
    One kernel, the ``.sv`` transcribed: a kernel-local ring ``mem: W[d]``,
    ``rd``/``wr`` pointers and a count, all ``int32``; the outputs of cycle
    ``t`` are the state it starts in (pre-edge sample); then the RTL's update
    rules, illegal commands included, so it is bit-exact on illegal traces too.
``ported`` (R3 shape)
    A ``@df.unit`` with ``trace``'s body; four command and three response
    ``Stream`` ports in lockstep, between a driver and a sink kernel.
``wire`` (the Catapult port shape)
    ``trace``'s body in a kernel whose ports are all ``Wire``
    (``synth_top="fifo_0"``); ``src``/``sink`` replay and record.
``stream`` (plan F1: the Allo FIFO primitive)
    The FIFO is a ``Stream[W, d - 1]`` owned by one kernel (a self-FIFO:
    both ends in the same kernel) behind a **head register**: the first-word-
    fall-through adapter the port needs, because a ``Stream`` has no peek
    (``pop_data`` must be visible without a pop). ``empty`` is "no head",
    ``full`` is "head and ``fifo.full()``"; capacity ``d``. A pop refills the
    head from the stream, a push lands in the head when it is free, else
    ``put``. A reset drains the stream (a ``Stream`` has no pointer reset).
    One RTL quirk needs a special case: push and pop together on an empty FIFO
    loses the word in the RTL (H12); a Stream cannot lose it, so the variant
    drops it by hand (marked H12 in the body).
``stream_raw`` (F1 as the plan words it)
    ``Stream[W, d]`` alone, no head register: ``pop_data`` is the word a pop
    returns on the cycle of the pop and 0 otherwise, the flags are the
    stream's own. The pass-through on full is the program order "pop, then
    push"; a push on full is dropped by a ``full()`` guard (``try_put``
    would be, but B7 drops an unused ``try_put`` on the simulator and S9
    drifts its counter in csim). A probe of capacity and flag timing; it
    cannot match the head-visible slots (``scripts/flags_only.py``).

Workarounds every variant carries (``u2_regfile_2026-10-02.rst``): B4 (pointers
are ``int32``, never an unsigned index); S6 (every port read unconditional,
only the state update under ``if``); B5 (stream ``get`` into the port type
first, then widen).
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stream, UInt, Wire, int32, uint1

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

GEOM = {"w32d4": (32, 4), "input": (257, 4), "output": (64, 16),
        "d16w48": (48, 16)}  # probe: the output FIFO's depth at a width csim reads back (S8)
WIDTH = {k: v[0] for k, v in GEOM.items()}
DEPTH_OF = {v[0]: v[1] for v in GEOM.values()}  # make(n, w): the depth from the width
CMD = ("rst_ni", "push_i", "push_data_i", "pop_i")
RESP = ("pop_data_o", "empty_o", "full_o")


def _unit(inst):
    w, d = GEOM[inst]
    return rtl.RtlUnit(
        top="vpu_fifo",
        sources=["src/core/vpu/vpu_fifo.sv"],
        inputs=[("rst_ni", 1), ("push_i", 1), ("push_data_i", w), ("pop_i", 1)],
        outputs=[("pop_data_o", w), ("empty_o", 1), ("full_o", 1)],
        shape="trace",
        params={} if inst == "w32d4" else {"WIDTH": w, "DEPTH": d},
        assertions=True,
    )


INSTANCES = {k: _unit(k) for k in GEOM}
DEFAULT = "w32d4"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = "vpu_fifo.sv:31-33 (flags and pop_data combinational from registered state)"


def REF(inst, cmd):
    w, d = GEOM[inst]
    return ref.vpu_fifo_trace(cmd, w, d)


def _defaults():
    return {"rst_ni": 1, "push_i": 0, "push_data_i": 0, "pop_i": 0}


def _reset(t, k=2):
    return t.idle(k, rst_ni=0)


def random_trace(inst, n, seed, legal):
    """Bursty pushes and pops, biased so the count sweeps empty..full."""
    w, d = GEOM[inst]
    rng = rng_for("fifo", inst, seed, legal)
    t = _reset(Trace(_defaults()))
    count, bias = 0, 0.5
    for i in range(n):
        if i % 64 == 0:
            bias = rng.choice([0.2, 0.5, 0.8])
        push = rng.random() < bias
        pop = rng.random() < 1 - bias
        if legal:
            if push and count == d and not pop:
                push = False
            if pop and count == 0:
                pop = False
        rst = int(rng.random() > 0.002)
        t.cycle(rst_ni=rst, push_i=int(push), push_data_i=word(rng, w), pop_i=int(pop))
        if not rst:
            count = 0
        else:
            dp = push and (count < d or pop)
            dq = pop and (count > 0 or push)
            count += int(dp) - int(dq)
    return t.cmd()


def directed(inst):
    w, d = GEOM[inst]
    rng = rng_for("fifo-directed", inst)
    out = []
    t = _reset(Trace(_defaults()))  # fill to full, pass-through on full, drain
    for _ in range(d):
        t.cycle(push_i=1, push_data_i=word(rng, w))
    for _ in range(3):
        t.cycle(push_i=1, push_data_i=word(rng, w), pop_i=1)
    t.idle(2)
    for _ in range(d):
        t.cycle(pop_i=1)
    t.idle(2)
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(pop_i=1)  # one in, one out
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(push_i=1, push_data_i=word(rng, w), pop_i=1)
    t.cycle(pop_i=1).idle(2)
    out.append(("fill-full-passthrough-drain", t.cmd(), True))
    t = _reset(Trace(_defaults()))  # pointer wrap: 3 * depth through a half-full FIFO
    for _ in range(d // 2):
        t.cycle(push_i=1, push_data_i=word(rng, w))
    for _ in range(3 * d):
        t.cycle(push_i=1, push_data_i=word(rng, w), pop_i=1)
    out.append(("wrap-half-full", t.cmd(), True))
    t = _reset(Trace(_defaults()))  # H12: push and pop together on empty
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(pop_i=1)  # leaves a stale entry
    t.cycle(push_i=1, push_data_i=word(rng, w), pop_i=1)  # the word is lost
    t.idle(2)
    t.cycle(push_i=1, push_data_i=word(rng, w)).idle(2)
    out.append(("push+pop-on-empty", t.cmd(), False))
    t = _reset(Trace(_defaults()))  # overflow: push on full, no pop -> dropped
    for _ in range(d + 3):
        t.cycle(push_i=1, push_data_i=word(rng, w))
    for _ in range(d + 1):
        t.cycle(pop_i=1)
    out.append(("overflow", t.cmd(), False))
    t = _reset(Trace(_defaults()))  # underflow: pop on empty -> ignored
    t.cycle(pop_i=1).cycle(pop_i=1)
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(pop_i=1).cycle(pop_i=1)
    out.append(("underflow", t.cmd(), False))
    t = Trace(_defaults())  # before any reset, then reset mid-stream with data inside
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(pop_i=1)
    _reset(t)
    for _ in range(3):
        t.cycle(push_i=1, push_data_i=word(rng, w))
    t.cycle(rst_ni=0, push_i=1, push_data_i=word(rng, w), pop_i=1)  # reset wins
    t.idle(2)
    t.cycle(push_i=1, push_data_i=word(rng, w)).cycle(pop_i=1).idle(1)
    out.append(("reset-mid-stream", t.cmd(), True))
    return out


def traces(inst):
    n = 20000 if inst not in ("input", "d16w48") else 5000
    tr = directed(inst)
    tr += [(f"random-legal-{s}", random_trace(inst, n, s, True), True) for s in range(2)]
    tr += [(f"random-any-{s}", random_trace(inst, n, s, False), False) for s in range(2)]
    return tr


def probes(inst):
    """Edges from a command to the output it moves: push on empty to
    ``empty_o`` and to ``pop_data_o`` (first word falls through), the
    ``depth``-th push to ``full_o``, a pop to the next ``pop_data_o``."""
    w, d = GEOM[inst]
    u = INSTANCES[inst]
    out = []
    t = _reset(Trace(_defaults())).idle(4)
    ev = len(t)
    t.cycle(push_i=1, push_data_i=1).idle(4)
    out.append(("push -> empty_o", 1, rtl.probe_trace(u, t.cmd(), "empty_o", ev)))
    t = _reset(Trace(_defaults()))
    t.cycle(push_i=1, push_data_i=1).cycle(pop_i=1).idle(4)  # the slot after holds 1
    ev = len(t)
    t.cycle(push_i=1, push_data_i=(1 << w) - 2).idle(4)
    out.append(("push -> pop_data_o (fall-through)", 1, rtl.probe_trace(u, t.cmd(), "pop_data_o", ev)))
    t = _reset(Trace(_defaults()))
    for _ in range(d - 1):
        t.cycle(push_i=1, push_data_i=0)
    t.idle(4)
    ev = len(t)
    t.cycle(push_i=1, push_data_i=0).idle(4)
    out.append(("last push -> full_o", 1, rtl.probe_trace(u, t.cmd(), "full_o", ev)))
    t = _reset(Trace(_defaults()))
    t.cycle(push_i=1, push_data_i=1).cycle(push_i=1, push_data_i=2).idle(4)
    ev = len(t)
    t.cycle(pop_i=1).idle(4)
    out.append(("pop -> pop_data_o", 1, rtl.probe_trace(u, t.cmd(), "pop_data_o", ev)))
    return out


def seeds():
    """``tb_mxu_single_port`` (DIM = 2): each lane's output FIFO gets one
    gathered group of 4 rows, the tb waits for ``output_valid`` and pops."""
    from examples.minitpu.harness import vcd_seed

    src = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_fifo.sv",
           "src/core/mxu/mxu_acc24_add_pipe.sv", "src/core/vpu/vpu_bf16_add.sv",
           "src/core/mxu/mxu_bf16_mul_acc24.sv", "src/core/mxu/mxu_pe.sv",
           "src/core/mxu/mxu_systolic_array.sv", "src/core/mxu/mxu.sv"]
    out = []
    for lane in (0, 1):
        cmd, seen = vcd_seed.extract("tb_mxu_single_port", src,
                                     dut=f"dut.gen_lane_output[{lane}].i_output_fifo",
                                     clk="clk_i", unit=INSTANCES["output"])
        out.append((f"tb_mxu_single_port lane {lane}", "output", cmd, seen, True))
    cmd, seen = vcd_seed.extract("tb_mxu_single_port", src, dut="dut.i_input_fifo",
                                 clk="clk_i", unit=_unit_input_dim2())
    out.append(("tb_mxu_single_port input FIFO (DIM=2: 33 b)", "input_dim2", cmd, seen, True))
    return out


def _unit_input_dim2():
    return rtl.RtlUnit(
        top="vpu_fifo", sources=["src/core/vpu/vpu_fifo.sv"],
        inputs=[("rst_ni", 1), ("push_i", 1), ("push_data_i", 33), ("pop_i", 1)],
        outputs=[("pop_data_o", 33), ("empty_o", 1), ("full_o", 1)],
        shape="trace", params={"WIDTH": 33, "DEPTH": 4}, assertions=True,
    )


INSTANCES["input_dim2"] = _unit_input_dim2()  # seed-only: the tb's DIM = 2 input FIFO
GEOM["input_dim2"] = (33, 4)
WIDTH["input_dim2"] = 33
DEPTH_OF[33] = 4


# ---------------------------------------------------------------------------
# Allo variants. ``make(n, w)`` returns a region over per-port arrays of n
# cycles (depth ``DEPTH_OF[w]``); each runner takes the built module and a
# command trace (``CMD`` columns as integer lists) and returns
# ``{resp port: array of ints}``.
# ---------------------------------------------------------------------------


def _np(w):
    return np.uint8 if w <= 8 else np.uint16 if w <= 16 else np.uint32 if w <= 32 else np.uint64


def _args(cmd, n, w):
    """The trace as the region's input arrays, plus zeroed outputs. A width
    above 64 has no numpy dtype (B6/H8): refused here rather than corrupting
    the simulator's heap."""
    if w > 64:
        raise ValueError(f"B6: no numpy dtype carries a {w}-bit element")
    ins = [np.asarray(cmd["rst_ni"][:n], dtype=np.uint8),
           np.asarray(cmd["push_i"][:n], dtype=np.uint8),
           np.asarray([int(x) for x in cmd["push_data_i"][:n]], dtype=_np(w)),
           np.asarray(cmd["pop_i"][:n], dtype=np.uint8)]
    outs = [np.zeros(n, dtype=_np(w)), np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)]
    return ins, outs


def _run_flat(mod, cmd, n, w):
    ins, outs = _args(cmd, n, w)
    mod(*ins, *outs)
    return {p: o for p, o in zip(RESP, outs)}


def trace(n, w=32):
    W = UInt(w)
    d = DEPTH_OF[w]

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP, QD, QE, QF])
        def fifo(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n],
                 qd: W[n], qe: uint1[n], qf: uint1[n]):
            mem: W[d]
            rd: int32 = 0
            wr: int32 = 0
            cnt: int32 = 0
            for t in range(n):
                empty: int32 = 0
                full: int32 = 0
                if cnt == 0:
                    empty = 1
                if cnt == d:
                    full = 1
                qd[t] = mem[rd]  # first word falls through: visible without a pop
                qe[t] = empty
                qf[t] = full
                r: uint1 = rst[t]  # S6 workaround: every port read unconditional
                p: uint1 = push[t]
                x: W = pd[t]
                q: uint1 = pop[t]
                if r == 0:  # reset clears the pointers and the count, not mem
                    rd = 0
                    wr = 0
                    cnt = 0
                else:
                    do_push: int32 = 0
                    do_pop: int32 = 0
                    if p == 1:
                        if full == 0:
                            do_push = 1
                        elif q == 1:  # pass-through on full
                            do_push = 1
                    if q == 1:
                        if empty == 0:
                            do_pop = 1
                        elif p == 1:  # H12: both pointers move, the word is lost
                            do_pop = 1
                    if do_push == 1:
                        mem[wr] = x
                        if wr == d - 1:
                            wr = 0
                        else:
                            wr = wr + 1
                    if do_pop == 1:
                        if rd == d - 1:
                            rd = 0
                        else:
                            rd = rd + 1
                    cnt = cnt + do_push - do_pop

    return top


def _port_unit(n, w):
    """C9: sizes freeze at ``@df.unit``, so a factory per (n, w)."""
    W = UInt(w)
    d = DEPTH_OF[w]

    @df.unit()
    def fifo(rst: Stream[uint1, 2], push: Stream[uint1, 2], pd: Stream[UInt(w), 2],
             pop: Stream[uint1, 2], qd: Stream[UInt(w), 2], qe: Stream[uint1, 2],
             qf: Stream[uint1, 2]):
        mem: W[d]
        rd: int32 = 0
        wr: int32 = 0
        cnt: int32 = 0
        for t in range(n):
            empty: int32 = 0
            full: int32 = 0
            if cnt == 0:
                empty = 1
            if cnt == d:
                full = 1
            r: uint1 = rst.get()
            p: uint1 = push.get()
            x: UInt(w) = pd.get()
            q: uint1 = pop.get()
            qd.put(mem[rd])
            qe.put(empty)
            qf.put(full)
            if r == 0:
                rd = 0
                wr = 0
                cnt = 0
            else:
                do_push: int32 = 0
                do_pop: int32 = 0
                if p == 1:
                    if full == 0:
                        do_push = 1
                    elif q == 1:
                        do_push = 1
                if q == 1:
                    if empty == 0:
                        do_pop = 1
                    elif p == 1:
                        do_pop = 1
                if do_push == 1:
                    mem[wr] = x
                    if wr == d - 1:
                        wr = 0
                    else:
                        wr = wr + 1
                if do_pop == 1:
                    if rd == d - 1:
                        rd = 0
                    else:
                        rd = rd + 1
                cnt = cnt + do_push - do_pop

    return fifo


def ported(n, w=32):
    W = UInt(w)
    unit = _port_unit(n, w)

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        s_rst: Stream[uint1, 2]
        s_push: Stream[uint1, 2]
        s_pd: Stream[UInt(w), 2]
        s_pop: Stream[uint1, 2]
        s_qd: Stream[UInt(w), 2]
        s_qe: Stream[uint1, 2]
        s_qf: Stream[uint1, 2]

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP])
        def drive(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n]):
            for t in range(n):
                s_rst.put(rst[t])
                s_push.put(push[t])
                s_pd.put(pd[t])
                s_pop.put(pop[t])

        unit(rst=s_rst, push=s_push, pd=s_pd, pop=s_pop, qd=s_qd, qe=s_qe, qf=s_qf)

        @df.kernel(mapping=[1], args=[QD, QE, QF])
        def sink(qd: W[n], qe: uint1[n], qf: uint1[n]):
            for t in range(n):
                qd[t] = s_qd.get()
                qe[t] = s_qe.get()
                qf[t] = s_qf.get()

    return top


def wire(n, w=32):
    """SystemC/Catapult port shape: ``fifo`` has only ``Wire`` ports
    (``synth_top="fifo_0"``); ``src``/``sink`` replay and record."""
    W = UInt(w)
    d = DEPTH_OF[w]

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        w_rst: Wire[uint1]
        w_push: Wire[uint1]
        w_pd: Wire[UInt(w)]
        w_pop: Wire[uint1]
        w_qd: Wire[UInt(w)]
        w_qe: Wire[uint1]
        w_qf: Wire[uint1]

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP])
        def src(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n]):
            for t in range(n):
                w_rst.put(rst[t])
                w_push.put(push[t])
                w_pd.put(pd[t])
                w_pop.put(pop[t])

        @df.kernel(mapping=[1], args=[])
        def fifo():
            mem: W[d]
            rd: int32 = 0
            wr: int32 = 0
            cnt: int32 = 0
            for _ in range(n):
                empty: int32 = 0
                full: int32 = 0
                if cnt == 0:
                    empty = 1
                if cnt == d:
                    full = 1
                r: uint1 = w_rst.get()
                p: uint1 = w_push.get()
                x: UInt(w) = w_pd.get()
                q: uint1 = w_pop.get()
                w_qd.put(mem[rd])
                w_qe.put(empty)
                w_qf.put(full)
                if r == 0:
                    rd = 0
                    wr = 0
                    cnt = 0
                else:
                    do_push: int32 = 0
                    do_pop: int32 = 0
                    if p == 1:
                        if full == 0:
                            do_push = 1
                        elif q == 1:
                            do_push = 1
                    if q == 1:
                        if empty == 0:
                            do_pop = 1
                        elif p == 1:
                            do_pop = 1
                    if do_push == 1:
                        mem[wr] = x
                        if wr == d - 1:
                            wr = 0
                        else:
                            wr = wr + 1
                    if do_pop == 1:
                        if rd == d - 1:
                            rd = 0
                        else:
                            rd = rd + 1
                    cnt = cnt + do_push - do_pop

        @df.kernel(mapping=[1], args=[QD, QE, QF])
        def sink(qd: W[n], qe: uint1[n], qf: uint1[n]):
            for t in range(n):
                qd[t] = w_qd.get()
                qe[t] = w_qe.get()
                qf[t] = w_qf.get()

    return top


def stream(n, w=32):
    """The Allo FIFO primitive behind a head register (first-word-fall-through
    adapter): ``Stream[W, d - 1]`` + head = capacity ``d``."""
    W = UInt(w)
    d = DEPTH_OF[w]
    k = d - 1

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        q_fifo: Stream[UInt(w), k]  # a self-FIFO: one kernel owns both ends

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP, QD, QE, QF])
        def fifo(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n],
                 qd: W[n], qe: uint1[n], qf: uint1[n]):
            head: W = 0
            hv: int32 = 0  # head valid
            for t in range(n):
                empty: int32 = 0
                full: int32 = 0
                if hv == 0:
                    empty = 1
                sf: uint1 = q_fifo.full()
                if hv == 1:
                    if sf == 1:
                        full = 1
                qd[t] = head
                qe[t] = empty
                qf[t] = full
                r: uint1 = rst[t]
                p: uint1 = push[t]
                x: W = pd[t]
                q: uint1 = pop[t]
                if r == 0:  # no pointer reset on a Stream: drain it
                    se: uint1 = q_fifo.empty()
                    while se == 0:
                        junk: UInt(w) = q_fifo.get()
                        se = q_fifo.empty()
                    hv = 0
                else:
                    do_push: int32 = 0
                    do_pop: int32 = 0
                    if p == 1:
                        if full == 0:
                            do_push = 1
                        elif q == 1:
                            do_push = 1
                    if q == 1:
                        if empty == 0:
                            do_pop = 1
                        elif p == 1:  # H12: the RTL loses the word; a Stream cannot
                            do_push = 0
                    if do_pop == 1:  # pop first: pass-through on full needs the slot
                        se2: uint1 = q_fifo.empty()
                        if se2 == 0:
                            head = q_fifo.get()
                        else:
                            hv = 0
                    if do_push == 1:
                        if hv == 0:
                            head = x
                            hv = 1
                        else:
                            q_fifo.put(x)

    return top


def stream_raw(n, w=32):
    """``Stream[W, d]`` alone (plan F1 as worded): no head register, so
    ``pop_data`` is defined only on the cycle of a pop (0 otherwise)."""
    W = UInt(w)
    d = DEPTH_OF[w]

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        q_fifo: Stream[UInt(w), d]

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP, QD, QE, QF])
        def fifo(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n],
                 qd: W[n], qe: uint1[n], qf: uint1[n]):
            for t in range(n):
                e: uint1 = q_fifo.empty()
                f: uint1 = q_fifo.full()
                qe[t] = e
                qf[t] = f
                r: uint1 = rst[t]
                p: uint1 = push[t]
                x: W = pd[t]
                q: uint1 = pop[t]
                out: W = 0
                if r == 0:
                    se: uint1 = q_fifo.empty()
                    while se == 0:
                        junk: UInt(w) = q_fifo.get()
                        se = q_fifo.empty()
                else:
                    if q == 1:  # pop, then push: pass-through on full
                        if e == 0:
                            out = q_fifo.get()
                    if p == 1:  # a refused push is dropped, as the RTL's; put under
                        f2: uint1 = q_fifo.full()  # a guard, not try_put (B7, S9)
                        if f2 == 0:
                            q_fifo.put(x)
                qd[t] = out

    return top


def stream_nodrain(n, w=32):
    """Catapult probe only: ``stream`` without the reset drain loop (a reset
    just drops the head), to tell whether Catapult's SCHD-30 on ``stream`` is
    the ``while`` drain or the self-FIFO itself. Wrong after a reset with
    words inside: not a verdict variant."""
    W = UInt(w)
    d = DEPTH_OF[w]
    k = d - 1

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        q_fifo: Stream[UInt(w), k]

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP, QD, QE, QF])
        def fifo(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n],
                 qd: W[n], qe: uint1[n], qf: uint1[n]):
            head: W = 0
            hv: int32 = 0
            for t in range(n):
                empty: int32 = 0
                full: int32 = 0
                if hv == 0:
                    empty = 1
                sf: uint1 = q_fifo.full()
                if hv == 1:
                    if sf == 1:
                        full = 1
                qd[t] = head
                qe[t] = empty
                qf[t] = full
                r: uint1 = rst[t]
                p: uint1 = push[t]
                x: W = pd[t]
                q: uint1 = pop[t]
                if r == 0:
                    hv = 0
                else:
                    do_push: int32 = 0
                    do_pop: int32 = 0
                    if p == 1:
                        if full == 0:
                            do_push = 1
                        elif q == 1:
                            do_push = 1
                    if q == 1:
                        if empty == 0:
                            do_pop = 1
                        elif p == 1:
                            do_push = 0
                    if do_pop == 1:
                        se2: uint1 = q_fifo.empty()
                        if se2 == 0:
                            head = q_fifo.get()
                        else:
                            hv = 0
                    if do_push == 1:
                        if hv == 0:
                            head = x
                            hv = 1
                        else:
                            q_fifo.put(x)

    return top


def wire_np(n, w=32):
    """Catapult probe only: ``wire`` with the RTL's pointer widths
    (``UInt(log2 d)`` pointers, ``UInt(log2(d + 1))`` count) instead of the
    ``int32`` the B4 workaround forces on the simulator path, to price that
    workaround in area. Not run on the simulator (B4)."""
    import math

    W = UInt(w)
    d = DEPTH_OF[w]
    P = UInt(int(math.log2(d)))
    C = UInt(int(math.log2(d)) + 1)

    @df.region()
    def top(RST: uint1[n], PUSH: uint1[n], PD: W[n], POP: uint1[n],
            QD: W[n], QE: uint1[n], QF: uint1[n]):
        w_rst: Wire[uint1]
        w_push: Wire[uint1]
        w_pd: Wire[UInt(w)]
        w_pop: Wire[uint1]
        w_qd: Wire[UInt(w)]
        w_qe: Wire[uint1]
        w_qf: Wire[uint1]

        @df.kernel(mapping=[1], args=[RST, PUSH, PD, POP])
        def src(rst: uint1[n], push: uint1[n], pd: W[n], pop: uint1[n]):
            for t in range(n):
                w_rst.put(rst[t])
                w_push.put(push[t])
                w_pd.put(pd[t])
                w_pop.put(pop[t])

        @df.kernel(mapping=[1], args=[])
        def fifo():
            mem: W[d]
            rd: P = 0
            wr: P = 0
            cnt: C = 0
            for _ in range(n):
                empty: uint1 = 0
                full: uint1 = 0
                if cnt == 0:
                    empty = 1
                if cnt == d:
                    full = 1
                r: uint1 = w_rst.get()
                p: uint1 = w_push.get()
                x: UInt(w) = w_pd.get()
                q: uint1 = w_pop.get()
                w_qd.put(mem[rd])
                w_qe.put(empty)
                w_qf.put(full)
                if r == 0:
                    rd = 0
                    wr = 0
                    cnt = 0
                else:
                    do_push: uint1 = 0
                    do_pop: uint1 = 0
                    if p == 1:
                        if full == 0:
                            do_push = 1
                        elif q == 1:
                            do_push = 1
                    if q == 1:
                        if empty == 0:
                            do_pop = 1
                        elif p == 1:
                            do_pop = 1
                    if do_push == 1:
                        mem[wr] = x
                        wr = wr + 1  # wraps at the pointer width, as the RTL
                    if do_pop == 1:
                        rd = rd + 1
                    cnt = cnt + do_push - do_pop

        @df.kernel(mapping=[1], args=[QD, QE, QF])
        def sink(qd: W[n], qe: uint1[n], qf: uint1[n]):
            for t in range(n):
                qd[t] = w_qd.get()
                qe[t] = w_qe.get()
                qf[t] = w_qf.get()

    return top


VARIANTS = {
    "trace": (trace, _run_flat),
    "ported": (ported, _run_flat),
    "wire": (wire, _run_flat),
    "stream": (stream, _run_flat),
    "stream_raw": (stream_raw, _run_flat),
    "stream_nodrain": (stream_nodrain, _run_flat),
    "wire_np": (wire_np, _run_flat),
}

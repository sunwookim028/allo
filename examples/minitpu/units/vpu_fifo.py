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

No Allo variant yet (U2 plan, checkpoint 1).
"""

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

GEOM = {"w32d4": (32, 4), "input": (257, 4), "output": (64, 16)}


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
VARIANTS = {}
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
    n = 20000 if inst != "input" else 5000
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

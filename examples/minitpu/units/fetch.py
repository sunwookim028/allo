# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``fetch``, the sequencer's instruction fetch: ``sequencer_iram`` (4,096 x
128 b, one write port for the loader, a registered read every cycle) feeding
``sequencer_fetch_queue`` (a 4-deep shift-register skid FIFO whose head is the
bundle offered to issue).

Instances:

``fq_a8``
    the fetch queue alone at ``tb_fetch_queue_shift``'s geometry
    (``ADDR_WIDTH = 8``, ``DEPTH = 4``); ``bram_data_i`` is a port, so the
    trace drives the read data the IRAM would return. Seeded from that tb.
``fq``
    the fetch queue alone at the shipped geometry (``ADDR_WIDTH = 12``).
``iram``
    ``units/rtl/u4_fetch.sv``: IRAM + fetch queue wired as ``sequencer.sv``
    wires them, ``pause`` brought out (``sequencer.sv`` holds it 0).

Reference: ``harness/ref_ctrl_front.fetch_queue_trace``, a register-level
cycle model. What a client (issue) may rely on -- the contract -- is smaller:
straight-line code offers one bundle a cycle once warm; a flush (taken
branch) in cycle ``e`` offers the target's bundle in cycle ``e + 3``
(two empty cycles between the branch bundle and its target); the head never
changes except by a pop or a push into an empty queue; data read from IRAM is
the word as it stood before a same-cycle write. ``fifo_mem`` and IRAM are
unreset: a head shown while the queue has never been written is masked.
"""

import os

from examples.minitpu.harness import ref_ctrl_front, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
FQ_SRC = PKGS + ["src/core/sequencer/sequencer_fetch_queue.sv"]
FETCH_SRC = PKGS + ["src/core/sequencer/sequencer_iram.sv", "src/core/sequencer/sequencer_fetch_queue.sv",
                    os.path.join(HERE, "rtl", "u4_fetch.sv")]
AW = {"fq_a8": 8, "fq": 12, "iram": 12}


def _fq(aw):
    return rtl.RtlUnit(
        top="sequencer_fetch_queue",
        sources=FQ_SRC,
        inputs=[("rst_n", 1), ("bram_data_i", 128), ("pause_i", 1), ("pc_flush_i", 1),
                ("pc_restart_addr_i", aw), ("bundle_pop_i", 1)],
        outputs=[("rd_addr_o", aw), ("rd_addr_valid_o", 1), ("bundle_valid_o", 1),
                 ("bundle_data_o", 128), ("bundle_addr_o", aw), ("empty_o", 1), ("full_o", 1)],
        shape="trace", clk="clk", rst_n="rst_n", params={"ADDR_WIDTH": aw, "DEPTH": 4},
        assertions=True,
    )


def _fetch():
    return rtl.RtlUnit(
        top="u4_fetch",
        sources=FETCH_SRC,
        inputs=[("rst_n", 1), ("instr_write_en", 1), ("iram_addr", 12), ("dma_iram_din", 128),
                ("pause", 1), ("pc_flush", 1), ("restart_addr", 12), ("bundle_pop", 1)],
        outputs=[("rd_addr", 12), ("rd_addr_valid", 1), ("bundle_valid", 1), ("bundle_data", 128),
                 ("bundle_addr", 12), ("empty", 1), ("full", 1)],
        shape="trace", clk="clk", rst_n="rst_n", assertions=True,
    )


INSTANCES = {"fq_a8": _fq(8), "fq": _fq(12), "iram": _fetch()}
DEFAULT = "iram"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = ("sequencer_pkg.sv:29 IRAM_MEM_LATENCY = 1; sequencer_fetch_queue.sv:25 "
                  "'taken branch; costs 2 empty cycles at IRAM_MEM_LATENCY=1'")


def REF(inst, cmd):
    return ref_ctrl_front.fetch_queue_trace(cmd, addr_w=AW[inst], iram=inst == "iram")


def _ports(inst):
    return {p: 0 for p, _ in INSTANCES[inst].inputs} | {"rst_n": 1}


def _fq_random(inst, n, seed, p_pause, p_flush, p_pop):
    rng = rng_for("fetch", inst, seed)
    t = Trace(_ports(inst))
    t.idle(2, rst_n=0)
    aw = AW[inst]
    for _ in range(n):
        t.cycle(bram_data_i=word(rng, 128), pause_i=int(rng.random() < p_pause),
                pc_flush_i=int(rng.random() < p_flush), pc_restart_addr_i=rng.getrandbits(aw),
                bundle_pop_i=int(rng.random() < p_pop))
    return t.cmd()


def _fq_directed(inst):
    out = []
    # straight line: never paused, pop every cycle once valid (issue at II=1)
    t = Trace(_ports(inst))
    t.idle(2, rst_n=0)
    for k in range(64):
        t.cycle(bram_data_i=0x1000 + k, bundle_pop_i=1)
    out.append(("straight-line", t.cmd(), True))
    # fill: no pops (count saturates at DEPTH - 1), then drain
    t = Trace(_ports(inst))
    t.idle(2, rst_n=0)
    for k in range(12):
        t.cycle(bram_data_i=0x2000 + k)
    for k in range(8):
        t.cycle(bram_data_i=0x3000 + k, bundle_pop_i=1, pause_i=1)
    out.append(("fill-drain", t.cmd(), True))
    # flush with the queue full, mid-pop, and back to back
    t = Trace(_ports(inst))
    t.idle(2, rst_n=0)
    for k in range(40):
        t.cycle(bram_data_i=0x4000 + k, bundle_pop_i=int(k % 3 != 0),
                pc_flush_i=int(k in (6, 7, 15, 22)), pc_restart_addr_i=(17 * k) & ((1 << AW[inst]) - 1))
    out.append(("flushes", t.cmd(), True))
    return out


def _iram_program(inst, seed, n_words=48, n=900):
    """Load ``n_words`` random bundles while the queue runs on unwritten IRAM,
    then flush to 0 and run: pops with stalls, branches back into the image,
    a few branches outside it, and late writes over words being fetched."""
    rng = rng_for("fetch-iram", seed)
    t = Trace(_ports(inst))
    t.idle(2, rst_n=0)
    image = [word(rng, 128) for _ in range(n_words)]
    for a, w in enumerate(image):
        t.cycle(instr_write_en=1, iram_addr=a, dma_iram_din=w, bundle_pop=int(rng.random() < 0.5))
    t.cycle(pc_flush=1, restart_addr=0)
    for _ in range(n):
        kw = {"bundle_pop": int(rng.random() < 0.8)}
        r = rng.random()
        if r < 0.04:
            kw.update(pc_flush=1, restart_addr=rng.randrange(n_words))
        elif r < 0.05:
            kw.update(pc_flush=1, restart_addr=rng.randrange(1 << 12))
        if rng.random() < 0.03:
            kw.update(instr_write_en=1, iram_addr=rng.randrange(n_words), dma_iram_din=word(rng, 128))
        if rng.random() < 0.05:
            kw.update(pause=1)
        t.cycle(**kw)
    return t.cmd()


def traces(inst):
    if inst == "iram":
        return [(f"program-{s}", _iram_program(inst, s), True) for s in range(4)]
    tr = _fq_directed(inst)
    tr += [(f"random-tb-{s}", _fq_random(inst, 4000, s, 1 / 6, 1 / 64, 2 / 3), True) for s in range(2)]
    tr += [(f"random-dense-{s}", _fq_random(inst, 4000, s, 0.0, 0.1, 0.9), True) for s in range(2)]
    return tr


def probes(inst):
    u = INSTANCES[inst]
    res = []
    if inst == "iram":
        # fetch request (pause falls) -> bundle_valid: IRAM_MEM_LATENCY + 1 (the queue's push)
        t = Trace(_ports(inst))
        t.idle(2, rst_n=0)
        for a in range(8):
            t.cycle(instr_write_en=1, iram_addr=a, dma_iram_din=0xA0 + a, pause=1)
        t.cycle(pc_flush=1, restart_addr=0, pause=1)
        t.idle(6, pause=1)
        ev = len(t)
        t.idle(8)
        res.append(("fetch request -> bundle_valid (IRAM_MEM_LATENCY + 1)", 1 + 1,
                    rtl.probe_trace(u, t.cmd(), "bundle_valid", ev)))
        # taken branch: flush in cycle e -> the target's bundle at the head
        t = Trace(_ports(inst))
        t.idle(2, rst_n=0)
        for a in range(40):
            t.cycle(instr_write_en=1, iram_addr=a, dma_iram_din=0xB00 + a)
        t.cycle(pc_flush=1, restart_addr=0)
        t.idle(10)
        ev = len(t)
        t.cycle(pc_flush=1, restart_addr=30)
        t.idle(10)
        res.append(("flush -> target bundle at head (2 empty cycles + 1)", 3,
                    rtl.probe_trace(u, t.cmd(), "bundle_addr", ev)))
    else:
        # pop -> next head (bundle_addr moves after one edge)
        t = Trace(_ports(inst))
        t.idle(2, rst_n=0)
        for k in range(10):
            t.cycle(bram_data_i=0x50 + k)
        t.idle(4, pause_i=1)
        ev = len(t)
        t.cycle(pause_i=1, bundle_pop_i=1)
        t.idle(4, pause_i=1)
        res.append(("pop -> next head", 1, rtl.probe_trace(u, t.cmd(), "bundle_addr_o", ev)))
        # read data in its landing cycle -> bundle_data_o (empty queue)
        t = Trace(_ports(inst))
        t.idle(2, rst_n=0)
        t.cycle(bram_data_i=0x77)
        t.idle(6, pause_i=1, bundle_pop_i=1, bram_data_i=0x77)
        t.cycle(bundle_pop_i=1, bram_data_i=0x77)  # request
        ev = len(t)
        t.cycle(pause_i=1, bram_data_i=0x99)  # lands
        t.idle(4, pause_i=1, bram_data_i=0x99)
        res.append(("bram_data_i (landing cycle) -> bundle_data_o", 1,
                    rtl.probe_trace(u, t.cmd(), "bundle_data_o", ev)))
    return res


def seeds():
    """``tb_fetch_queue_shift`` (``ADDR_W = 8``, ``DEPTH = 4``): 2,000 random cycles."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract("tb_fetch_queue_shift", FQ_SRC, dut="dut", clk="clk",
                                 unit=INSTANCES["fq_a8"])
    return [("tb_fetch_queue_shift", "fq_a8", cmd, seen, True)]


VARIANTS = {}

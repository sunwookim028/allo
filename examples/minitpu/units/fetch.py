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

import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, int32, uint1

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


# ---------------------------------------------------------------------------
# Allo (U4 track A, plan F1 ``bits``). One kernel, one iteration per cycle:
# the "pre" outputs from the registers, then the edge. The fetch queue is
# transcribed register for register (``sequencer_fetch_queue.sv``): the
# free-running address, the in-flight flag and address, the 4-entry shift
# FIFO (head in entry 0) and its count. Behind the ``iram`` instance the
# IRAM is a 4,096 x 128 array read every cycle into a read register (the old
# word on a same-cycle write); IRAM and its read register are written on
# every edge, reset or not, as the RTL's ``always_ff @(posedge clk)``. Bundles
# are four 32-bit lanes (P-8). The FIFO entries and IRAM start as whatever
# the array holds: the harness masks what the RTL leaves unreset (D-14).
#
# F1's IRAM is an array in the body; ``f1_d12`` declares it a D-12 memory.
# ---------------------------------------------------------------------------

WIDTH = {k: 128 for k in AW}
U32 = UInt(32)


def f1(n, w, inst):
    if inst == "iram":
        return _f1_iram(n)
    return _f1_fq(n, AW[inst])


def _f1_iram(n):
    A = UInt(12)

    @df.region()
    def top(RST: uint1[n], WE: uint1[n], WA: A[n], WD: U32[n, 4], PAUSE: uint1[n], FLUSH: uint1[n],
            RADDR: A[n], POP: uint1[n],
            DATA: U32[n, 4], RDA: A[n], RDV: uint1[n], VALID: uint1[n], BADDR: A[n], EMPTY: uint1[n],
            FULL: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, WE, WA, WD, PAUSE, FLUSH, RADDR, POP,
                                      DATA, RDA, RDV, VALID, BADDR, EMPTY, FULL])
        def fetch(rst: uint1[n], we: uint1[n], wa: A[n], wd: U32[n, 4], pause: uint1[n], flush: uint1[n],
                  raddr: A[n], pop_i: uint1[n],
                  data_o: U32[n, 4], rda_o: A[n], rdv_o: uint1[n], valid_o: uint1[n], baddr_o: A[n],
                  empty_o: uint1[n], full_o: uint1[n]):
            iram: U32[4096, 4] = 0
            rd_reg: U32[4] = 0  # sequencer_iram's read register
            fq_addr: A[4] = 0
            fq_data: U32[4, 4] = 0
            next_addr: A = 0
            req_pending: uint1 = 0
            req_addr: A = 0
            count: UInt(3) = 0
            for t in range(n):
                if rst[t] == 0:
                    next_addr = 0
                    req_pending = 0
                    req_addr = 0
                    count = 0
                empty: uint1 = count == 0
                rd_valid: uint1 = 0
                if pause[t] == 0 and count < 3:  # room for the read in flight
                    rd_valid = 1
                for k in range(4):
                    data_o[t, k] = fq_data[0, k]  # A5: the 2-D output stored first
                rda_o[t] = next_addr
                rdv_o[t] = rd_valid
                valid_o[t] = 1 - empty
                baddr_o[t] = fq_addr[0]
                empty_o[t] = empty
                full_o[t] = count == 4
                # ---- rising edge: IRAM (never reset) ----
                bram: U32[4]
                for k in range(4):
                    bram[k] = rd_reg[k]
                ir: int32 = next_addr
                for k in range(4):
                    rd_reg[k] = iram[ir, k]
                if we[t]:
                    iw: int32 = wa[t]
                    for k in range(4):
                        iram[iw, k] = wd[t, k]
                # ---- rising edge: the fetch queue ----
                if rst[t]:
                    if flush[t]:
                        next_addr = raddr[t]
                        req_pending = 0
                        count = 0
                    else:
                        push: uint1 = req_pending
                        pop: uint1 = pop_i[t] & (1 - empty)
                        if pop:
                            for j in range(3):
                                fq_addr[j] = fq_addr[j + 1]
                                for k in range(4):
                                    fq_data[j, k] = fq_data[j + 1, k]
                        if push:
                            slot: UInt(2) = count
                            if pop:
                                slot = count - 1
                            si: int32 = slot
                            fq_addr[si] = req_addr
                            for k in range(4):
                                fq_data[si, k] = bram[k]
                        count = count + push - pop
                        req_pending = rd_valid
                        req_addr = next_addr
                        if rd_valid:
                            next_addr = next_addr + 1

    return top


def _f1_fq(n, aw):
    """The fetch queue alone: ``bram_data_i`` is a port (the IRAM's word)."""
    A = UInt(aw)

    @df.region()
    def top(RST: uint1[n], BRAM: U32[n, 4], PAUSE: uint1[n], FLUSH: uint1[n], RADDR: A[n], POP: uint1[n],
            DATA: U32[n, 4], RDA: A[n], RDV: uint1[n], VALID: uint1[n], BADDR: A[n], EMPTY: uint1[n],
            FULL: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, BRAM, PAUSE, FLUSH, RADDR, POP,
                                      DATA, RDA, RDV, VALID, BADDR, EMPTY, FULL])
        def fq(rst: uint1[n], bram_i: U32[n, 4], pause: uint1[n], flush: uint1[n], raddr: A[n],
               pop_i: uint1[n],
               data_o: U32[n, 4], rda_o: A[n], rdv_o: uint1[n], valid_o: uint1[n], baddr_o: A[n],
               empty_o: uint1[n], full_o: uint1[n]):
            fq_addr: A[4] = 0
            fq_data: U32[4, 4] = 0
            next_addr: A = 0
            req_pending: uint1 = 0
            req_addr: A = 0
            count: UInt(3) = 0
            for t in range(n):
                if rst[t] == 0:
                    next_addr = 0
                    req_pending = 0
                    req_addr = 0
                    count = 0
                empty: uint1 = count == 0
                rd_valid: uint1 = 0
                if pause[t] == 0 and count < 3:
                    rd_valid = 1
                for k in range(4):
                    data_o[t, k] = fq_data[0, k]  # A5: the 2-D output stored first
                rda_o[t] = next_addr
                rdv_o[t] = rd_valid
                valid_o[t] = 1 - empty
                baddr_o[t] = fq_addr[0]
                empty_o[t] = empty
                full_o[t] = count == 4
                bram: U32[4]
                for k in range(4):
                    bram[k] = bram_i[t, k]
                if rst[t]:
                    if flush[t]:
                        next_addr = raddr[t]
                        req_pending = 0
                        count = 0
                    else:
                        push: uint1 = req_pending
                        pop: uint1 = pop_i[t] & (1 - empty)
                        if pop:
                            for j in range(3):
                                fq_addr[j] = fq_addr[j + 1]
                                for k in range(4):
                                    fq_data[j, k] = fq_data[j + 1, k]
                        if push:
                            slot: UInt(2) = count
                            if pop:
                                slot = count - 1
                            si: int32 = slot
                            fq_addr[si] = req_addr
                            for k in range(4):
                                fq_data[si, k] = bram[k]
                        count = count + push - pop
                        req_pending = rd_valid
                        req_addr = next_addr
                        if rd_valid:
                            next_addr = next_addr + 1

    return top


def run_f1(mod, cmd, n, w):
    """Ports by instance: ``iram`` has the loader's write port and plain
    names; ``fq``/``fq_a8`` take ``bram_data_i`` (A8: the runner is not told
    the instance, so the command's ports say which)."""
    from examples.minitpu.units.ctrl_lanes import col, join, split

    fq = "bram_data_i" in cmd
    P = ref_ctrl_front.FQ_PORTS if fq else ref_ctrl_front.FETCH_PORTS
    u8 = lambda p: col(cmd, p, n, np.uint8)  # noqa: E731
    u16 = lambda p: col(cmd, p, n, np.uint16)  # noqa: E731
    data = np.zeros((n, 4), dtype=np.uint32)
    o = {k: np.zeros(n, dtype=np.uint16 if k in ("rd_addr", "addr") else np.uint8)
         for k in ("rd_addr", "rd_valid", "valid", "addr", "empty", "full")}
    tail = [data, o["rd_addr"], o["rd_valid"], o["valid"], o["addr"], o["empty"], o["full"]]
    if fq:
        mod(u8("rst_n"), split(cmd["bram_data_i"][:n], 4), u8(P["pause"]), u8(P["flush"]), u16(P["restart"]),
            u8(P["pop"]), *tail)
    else:
        mod(u8("rst_n"), u8("instr_write_en"), u16("iram_addr"), split(cmd["dma_iram_din"][:n], 4),
            u8(P["pause"]), u8(P["flush"]), u16(P["restart"]), u8(P["pop"]), *tail)
    res = {P["data"]: join(data)}
    for k in o:
        res[P[k]] = o[k]
    return res


VARIANTS = {"f1": (f1, run_f1)}

# F1 with the IRAM declared as a D-12 memory (Stream links: simulator, csim)
from examples.minitpu.units import fetch_d12 as _d12  # noqa: E402

VARIANTS["f1_d12"] = (_d12.make("simulator"), _d12.run)

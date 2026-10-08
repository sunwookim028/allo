# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track A, plan F1 with P-8 / Q5: the IRAM as a README D-12 memory.

``iram`` is a ``compose.Memory`` of ``IRAM_ROWS`` x ``UInt(128)``, never reset
(D-14: ``sequencer_iram.sv`` has no reset), with two ports: ``host`` (``w``,
the loader's port A) and ``fetch`` (``r``, latency 1: port B's read
register). Two units bind one port each: ``loader`` writes under its enable;
``fq`` is the fetch queue, which reads the word at its fetch address every
cycle (the RTL's port B has no enable) and owns everything else of
``sequencer_fetch_queue.sv``, transcribed as in ``fetch.f1``. ``src`` and
``sink`` replay the per-cycle trace, one token per port per cycle.

The read's latency, and the finding it exposes (``u4_track_a`` T-3):
the server lowering writes a latency-L read as a pipe of L registers *as
data* whose last entry it puts in the SAME iteration, so on Stream links
(simulator, csim) an L = 1 read returns ``mem[a]`` of this very iteration;
on the Wire links Catapult gets, the registered link supplies the edge. The
RTL's read register is therefore written in ``fq``'s body (``rd_reg``), which
is right for Stream links and one cycle too many for the Wire form: a D-12
port's latency is honoured in time only by the Wire lowering, so a body that
must be cycle-locked cannot be written once for both.
"""

from __future__ import annotations

import numpy as np

from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

from examples.minitpu.template.control_geometry import SHIPPED as G


@unit(memories=("RST", "WE", "WA", "WD", "PAUSE", "FLUSH", "RADDR", "POP"),
      writes=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop", "h_we", "h_wa", "h_wd"),
      parameters=("N", "AW"))
def src(xrst: uint1[N], xwe: uint1[N], xwa: UInt(AW)[N], xwd: UInt(32)[N, 4], xpause: uint1[N],
        xflush: uint1[N], xraddr: UInt(AW)[N], xpop: uint1[N]):
    for t in range(N):
        d: UInt(128) = 0
        d[0:32] = xwd[t, 0]
        d[32:64] = xwd[t, 1]
        d[64:96] = xwd[t, 2]
        d[96:128] = xwd[t, 3]
        h_we.put(xwe[t])
        h_wa.put(xwa[t])
        h_wd.put(d)
        c_rst.put(xrst[t])
        c_pause.put(xpause[t])
        c_flush.put(xflush[t])
        c_raddr.put(xraddr[t])
        c_pop.put(xpop[t])


@unit(memories=("iram.host",), reads=("h_we", "h_wa", "h_wd"), parameters=("N", "AW"))
def loader(mem):
    for _ in range(N):
        e: uint1 = h_we.get()
        a_: UInt(AW) = h_wa.get()
        a: int32 = a_
        d: UInt(128) = h_wd.get()
        if e:
            mem[a] = d


@unit(memories=("iram.fetch",), reads=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop"),
      writes=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW"))
def fq(mem):
    rd_reg: UInt(128) = 0  # sequencer_iram's read register (see the module note)
    fq_addr: UInt(AW)[4] = 0
    fq_data: UInt(128)[4] = 0
    next_addr: UInt(AW) = 0
    req_pending: uint1 = 0
    req_addr: UInt(AW) = 0
    count: UInt(3) = 0
    for _ in range(N):
        rst: uint1 = c_rst.get()
        pause: uint1 = c_pause.get()
        flush: uint1 = c_flush.get()
        raddr: UInt(AW) = c_raddr.get()
        pop_i: uint1 = c_pop.get()
        if rst == 0:
            next_addr = 0
            req_pending = 0
            req_addr = 0
            count = 0
        empty: uint1 = count == 0
        rd_valid: uint1 = 0
        if pause == 0 and count < 3:
            rd_valid = 1
        q_data.put(fq_data[0])
        q_rda.put(next_addr)
        q_rdv.put(rd_valid)
        q_valid.put(1 - empty)
        q_baddr.put(fq_addr[0])
        q_empty.put(empty)
        full: uint1 = count == 4
        q_full.put(full)
        # the edge: port B reads the fetch address every cycle
        bram: UInt(128) = rd_reg
        ir: int32 = next_addr
        rd_reg = mem[ir]
        if rst:
            if flush:
                next_addr = raddr
                req_pending = 0
                count = 0
            else:
                push: uint1 = req_pending
                pop: uint1 = pop_i & (1 - empty)
                if pop:
                    for j in range(3):
                        fq_addr[j] = fq_addr[j + 1]
                        fq_data[j] = fq_data[j + 1]
                if push:
                    slot: UInt(2) = count
                    if pop:
                        slot = count - 1
                    si: int32 = slot
                    fq_addr[si] = req_addr
                    fq_data[si] = bram
                count = count + push - pop
                req_pending = rd_valid
                req_addr = next_addr
                if rd_valid:
                    next_addr = next_addr + 1


@unit(memories=("DATA", "RDA", "RDV", "VALID", "BADDR", "EMPTY", "FULL"),
      reads=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW"))
def sink(ydata: UInt(32)[N, 4], yrda: UInt(AW)[N], yrdv: uint1[N], yvalid: uint1[N], ybaddr: UInt(AW)[N],
         yempty: uint1[N], yfull: uint1[N]):
    for t in range(N):
        d: UInt(128) = q_data.get()
        ydata[t, 0] = d[0:32]  # A5: the 2-D output stored first
        ydata[t, 1] = d[32:64]
        ydata[t, 2] = d[64:96]
        ydata[t, 3] = d[96:128]
        yrda[t] = q_rda.get()
        yrdv[t] = q_rdv.get()
        yvalid[t] = q_valid.get()
        ybaddr[t] = q_baddr.get()
        yempty[t] = q_empty.get()
        yfull[t] = q_full.get()


IRAM = Memory("iram", "UInt(128)", rows=str(G.IRAM_ROWS),
              ports=(Port("host", "w", visible=1), Port("fetch", "r", latency=G.IRAM_MEM_LATENCY)),
              collision="refuse", reset=False)


def architecture(n):
    a = "UInt(AW)"
    ch = [Channel(c, d, "2") for c, d in (
        ("c_rst", "uint1"), ("c_pause", "uint1"), ("c_flush", "uint1"), ("c_raddr", a), ("c_pop", "uint1"),
        ("h_we", "uint1"), ("h_wa", a), ("h_wd", "UInt(128)"),
        ("q_data", "UInt(128)"), ("q_rda", a), ("q_rdv", "uint1"), ("q_valid", "uint1"), ("q_baddr", a),
        ("q_empty", "uint1"), ("q_full", "uint1"))]
    mems = [Memory(name, dt) for name, dt in (
        ("RST", "uint1[N]"), ("WE", "uint1[N]"), ("WA", f"{a}[N]"), ("WD", "UInt(32)[N, 4]"),
        ("PAUSE", "uint1[N]"), ("FLUSH", "uint1[N]"), ("RADDR", f"{a}[N]"), ("POP", "uint1[N]"),
        ("DATA", "UInt(32)[N, 4]"), ("RDA", f"{a}[N]"), ("RDV", "uint1[N]"), ("VALID", "uint1[N]"),
        ("BADDR", f"{a}[N]"), ("EMPTY", "uint1[N]"), ("FULL", "uint1[N]"))]
    return Architecture(name="fetch_d12", parameters={"N": n, "AW": G.INSTR_ADDR_W},
                        memories=tuple(mems) + (IRAM,), channels=tuple(ch),
                        units=(src, loader, fq, sink))


def make(target):
    def f(n, w, inst):
        assert inst == "iram", "the D-12 form is the IRAM instance (the queue alone has no memory)"
        return architecture(n).region(target, {"iram": "registers"})

    f.__name__ = f"f1_d12_{target}"
    return f


def run(mod, cmd, n, w):
    from examples.minitpu.units.ctrl_lanes import col, join, split

    u8 = lambda p: col(cmd, p, n, np.uint8)  # noqa: E731
    u16 = lambda p: col(cmd, p, n, np.uint16)  # noqa: E731
    data = np.zeros((n, 4), dtype=np.uint32)
    o = {k: np.zeros(n, dtype=np.uint16 if k in ("rd_addr", "bundle_addr") else np.uint8)
         for k in ("rd_addr", "rd_addr_valid", "bundle_valid", "bundle_addr", "empty", "full")}
    mod(u8("rst_n"), u8("instr_write_en"), u16("iram_addr"), split(cmd["dma_iram_din"][:n], 4), u8("pause"),
        u8("pc_flush"), u16("restart_addr"), u8("bundle_pop"), data, o["rd_addr"], o["rd_addr_valid"],
        o["bundle_valid"], o["bundle_addr"], o["empty"], o["full"])
    return {"bundle_data": join(data), **o}

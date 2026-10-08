# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D forms of ``fetch_d12`` (U4 track A, ``f1_d12``) for Catapult: T-2.

``fetch_d12`` declares the IRAM a D-12 ``Memory`` (``host`` w, ``fetch`` r
latency 1, ``reset=False``) and keeps the RTL's read register in the queue
unit's body (``rd_reg``) because on Stream links (simulator, csim) the server
delivers an ``L = 1`` read in the same iteration (T-2). On the SystemC target
the read data link is a ``Wire`` (``compose._pin``: "registered at the owner's
edge, then the pipe"), so that body counts the register twice. README D-12's
2026-10-08 amendment: ``L`` is counted in the owner's own iterations on every
link kind -- deliver at ``t + L``. Two queue bodies, both built with
``target="systemc"`` and checked per cycle on the Catapult RTL:

* ``landed``: ``fetch_d12.fq`` as landed (``bram = rd_reg; rd_reg = mem[a]``);
* ``amended``: the body the amendment asks for (``bram = mem[a]``: the value
  of the read issued one iteration earlier, the memory's ``L = 1``).

The boundary units ``src``/``sink`` carry the 128-bit words as one
``UInt(128)`` per row (U3 track C's C1: a ``[N, 4]`` lane array is RAM pins);
everything else is ``fetch_d12``'s text. Runner: ``fetch_d12.run`` (lane arrays
packed by ``u4d_check.CatapultRtl``).
"""
from __future__ import annotations

from allo.compose import Architecture, Channel, Memory, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401

from examples.minitpu.units import fetch_d12 as D


@unit(memories=("RST", "WE", "WA", "WD", "PAUSE", "FLUSH", "RADDR", "POP"),
      writes=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop", "h_we", "h_wa", "h_wd"),
      parameters=("N", "AW"))
def src(xrst: uint1[N], xwe: uint1[N], xwa: UInt(AW)[N], xwd: UInt(128)[N], xpause: uint1[N],
        xflush: uint1[N], xraddr: UInt(AW)[N], xpop: uint1[N]):
    for t in range(N):
        d: UInt(128) = xwd[t]
        h_we.put(xwe[t])
        h_wa.put(xwa[t])
        h_wd.put(d)
        c_rst.put(xrst[t])
        c_pause.put(xpause[t])
        c_flush.put(xflush[t])
        c_raddr.put(xraddr[t])
        c_pop.put(xpop[t])


@unit(memories=("iram.fetch",), reads=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop"),
      writes=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW"))
def fq_amended(mem):
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
        # port B's address, read before any store (D-13: a comb pin sees the
        # state loaded before the iteration's stores); 0 in a reset row
        ir: int32 = next_addr
        if rst == 0:
            ir = 0
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
        # the edge: port B reads the fetch address every cycle; the port's
        # latency 1 is the memory's (README D-12 amended): no register here
        bram: UInt(128) = mem[ir]
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


@unit(memories=("iram.fetch",), reads=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop"),
      writes=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW"))
def fq_landed(mem):
    rd_reg: UInt(128) = 0  # sequencer_iram's read register (fetch_d12.fq as landed)
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
        ir: int32 = next_addr  # the one edit: the address read before any store (D-13)
        if rst == 0:
            ir = 0
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
        bram: UInt(128) = rd_reg
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
def sink(ydata: UInt(128)[N], yrda: UInt(AW)[N], yrdv: uint1[N], yvalid: uint1[N], ybaddr: UInt(AW)[N],
         yempty: uint1[N], yfull: uint1[N]):
    for t in range(N):
        ydata[t] = q_data.get()
        yrda[t] = q_rda.get()
        yrdv[t] = q_rdv.get()
        yvalid[t] = q_valid.get()
        ybaddr[t] = q_baddr.get()
        yempty[t] = q_empty.get()
        yfull[t] = q_full.get()


def architecture(n, body, kind="wire"):
    a = "UInt(AW)"
    ch = [Channel(c, d, "2", kind=kind) for c, d in (
        ("c_rst", "uint1"), ("c_pause", "uint1"), ("c_flush", "uint1"), ("c_raddr", a), ("c_pop", "uint1"),
        ("h_we", "uint1"), ("h_wa", a), ("h_wd", "UInt(128)"),
        ("q_data", "UInt(128)"), ("q_rda", a), ("q_rdv", "uint1"), ("q_valid", "uint1"), ("q_baddr", a),
        ("q_empty", "uint1"), ("q_full", "uint1"))]
    mems = [Memory(name, dt) for name, dt in (
        ("RST", "uint1[N]"), ("WE", "uint1[N]"), ("WA", f"{a}[N]"), ("WD", "UInt(128)[N]"),
        ("PAUSE", "uint1[N]"), ("FLUSH", "uint1[N]"), ("RADDR", f"{a}[N]"), ("POP", "uint1[N]"),
        ("DATA", "UInt(128)[N]"), ("RDA", f"{a}[N]"), ("RDV", "uint1[N]"), ("VALID", "uint1[N]"),
        ("BADDR", f"{a}[N]"), ("EMPTY", "uint1[N]"), ("FULL", "uint1[N]"))]
    fq = {"landed": fq_landed, "amended": fq_amended}[body]
    return Architecture(name=f"fetch_d12_{body}", parameters={"N": n, "AW": D.G.INSTR_ADDR_W},
                        memories=tuple(mems) + (D.IRAM,), channels=tuple(ch),
                        units=(src, D.loader, fq, sink))


BODY = "landed"  # u4d_build/u4d_check: --form-arg body=amended (env U4D_BODY)


def make(n, w=128, inst="iram"):
    import os
    assert inst == "iram"
    return architecture(n, os.environ.get("U4D_BODY", BODY)).region("systemc", {"iram": "registers"})


run = D.run

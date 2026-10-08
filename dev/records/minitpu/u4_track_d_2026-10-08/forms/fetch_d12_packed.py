# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D forms of ``fetch_d12`` (U4 track A, ``f1_d12``) for Catapult: T-2.

``fetch_d12`` declares the IRAM a D-12 ``Memory`` (``host`` w, ``fetch`` r
latency 1, ``reset=False``) and keeps the RTL's read register in the queue
unit's body (``rd_reg``) because on Stream links (simulator, csim) the server
delivers an ``L = 1`` read in the same iteration (T-2). On the SystemC target
the server's pins are ``comb`` and the read data a registered ``Wire``. README
D-12's 2026-10-08 amendment: ``L`` counts the owner's own iterations on every
link kind (deliver at ``t + L``). Two queue bodies, built with
``target="systemc"`` and checked per cycle on the Catapult RTL:

* ``landed`` (``U4D_BODY=landed``): ``bram = rd_reg; rd_reg = mem[a]``, as
  ``fetch_d12.fq``;
* ``amended`` (default ``BODY`` below is ``landed``; ``U4D_BODY=amended``):
  ``bram = mem[a]``, the port's ``L = 1`` counted as the memory's.

What the landed composition needed to build at all (track D, D-3): D-13
refuses a comb pin that reads a Stream (the loader's address came from
``h_wa``) and any state read after a store in the iteration (the queue's
address after the reset store). So here (a) ``src`` -> owners -> ``sink`` are
``Wire`` channels, (b) both queue bodies read every state first and store
last (the address a select of the old state; the shift as next-state
temporaries) -- the same logic as ``fetch_d12.fq``, and (c) the IRAM has
``U4D_IRAM_ROWS`` rows (default 256; at 4,096 the ``registers`` server ran
Catapult's architect to 15.7 GB, D-4), the address masked: exact on these
traces (every write is below word 48; reads above are uninit, masked). The
128-bit words cross the boundary as one ``UInt(128)`` per row (U3 track C
C1). Runner: ``fetch_d12.run`` (lane arrays packed by ``u4d_check``).
A Wire-linked composite's sink token ``t`` is RTL row ``t - 4`` (``--shift 4``).
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


@unit(memories=("iram.host",), reads=("h_we", "h_wa", "h_wd"), parameters=("N", "AW", "IRM"))
def loader(mem):
    for _ in range(N):
        e: uint1 = h_we.get()
        a_: UInt(AW) = h_wa.get()
        a: int32 = a_ & IRM
        d: UInt(128) = h_wd.get()
        if e:
            mem[a] = d


@unit(memories=("iram.fetch",), reads=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop"),
      writes=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW", "IRM"))
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
        # D-13 discipline (a kernel with comb pins): every state read before any
        # store; an asynchronous reset row shows the reset state (as selects)
        ir: int32 = (next_addr if rst else 0) & IRM  # port B's address, every cycle (rows: IRM + 1)
        bram: UInt(128) = mem[ir]  # the memory's latency 1 (README D-12 amended): no register here
        na: UInt(AW) = next_addr if rst else 0
        rp: uint1 = req_pending if rst else 0
        rq: UInt(AW) = req_addr if rst else 0
        cnt: UInt(3) = count if rst else 0
        a0: UInt(AW) = fq_addr[0]
        a1: UInt(AW) = fq_addr[1]
        a2: UInt(AW) = fq_addr[2]
        a3: UInt(AW) = fq_addr[3]
        d0: UInt(128) = fq_data[0]
        d1: UInt(128) = fq_data[1]
        d2: UInt(128) = fq_data[2]
        d3: UInt(128) = fq_data[3]
        empty: uint1 = cnt == 0
        rd_valid: uint1 = 0
        if pause == 0 and cnt < 3:
            rd_valid = 1
        q_data.put(d0)
        q_rda.put(na)
        q_rdv.put(rd_valid)
        q_valid.put(1 - empty)
        q_baddr.put(a0)
        q_empty.put(empty)
        full: uint1 = cnt == 4
        q_full.put(full)
        # next state
        n_na: UInt(AW) = na
        n_rp: uint1 = rp
        n_rq: UInt(AW) = rq
        n_cnt: UInt(3) = cnt
        n_a0: UInt(AW) = a0
        n_a1: UInt(AW) = a1
        n_a2: UInt(AW) = a2
        n_a3: UInt(AW) = a3
        n_d0: UInt(128) = d0
        n_d1: UInt(128) = d1
        n_d2: UInt(128) = d2
        n_d3: UInt(128) = d3
        if rst:
            if flush:
                n_na = raddr
                n_rp = 0
                n_cnt = 0
            else:
                push: uint1 = rp
                pop: uint1 = pop_i & (1 - empty)
                if pop:
                    n_a0 = a1
                    n_a1 = a2
                    n_a2 = a3
                    n_d0 = d1
                    n_d1 = d2
                    n_d2 = d3
                if push:
                    slot: UInt(3) = cnt - pop
                    if slot == 0:
                        n_a0 = rq
                        n_d0 = bram
                    elif slot == 1:
                        n_a1 = rq
                        n_d1 = bram
                    elif slot == 2:
                        n_a2 = rq
                        n_d2 = bram
                    else:
                        n_a3 = rq
                        n_d3 = bram
                n_cnt = cnt + push - pop
                n_rp = rd_valid
                n_rq = na
                if rd_valid:
                    n_na = na + 1
        # stores
        next_addr = n_na
        req_pending = n_rp
        req_addr = n_rq
        count = n_cnt
        fq_addr[0] = n_a0
        fq_addr[1] = n_a1
        fq_addr[2] = n_a2
        fq_addr[3] = n_a3
        fq_data[0] = n_d0
        fq_data[1] = n_d1
        fq_data[2] = n_d2
        fq_data[3] = n_d3


@unit(memories=("iram.fetch",), reads=("c_rst", "c_pause", "c_flush", "c_raddr", "c_pop"),
      writes=("q_data", "q_rda", "q_rdv", "q_valid", "q_baddr", "q_empty", "q_full"),
      parameters=("N", "AW", "IRM"))
def fq_landed(mem):
    rd_reg: UInt(128) = 0  # sequencer_iram's read register, in the body (fetch_d12.fq as landed)
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
        # D-13 discipline (a kernel with comb pins): every state read before any
        # store; an asynchronous reset row shows the reset state (as selects)
        ir: int32 = (next_addr if rst else 0) & IRM  # port B's address, every cycle (rows: IRM + 1)
        bram: UInt(128) = rd_reg
        rd_new: UInt(128) = mem[ir]
        na: UInt(AW) = next_addr if rst else 0
        rp: uint1 = req_pending if rst else 0
        rq: UInt(AW) = req_addr if rst else 0
        cnt: UInt(3) = count if rst else 0
        a0: UInt(AW) = fq_addr[0]
        a1: UInt(AW) = fq_addr[1]
        a2: UInt(AW) = fq_addr[2]
        a3: UInt(AW) = fq_addr[3]
        d0: UInt(128) = fq_data[0]
        d1: UInt(128) = fq_data[1]
        d2: UInt(128) = fq_data[2]
        d3: UInt(128) = fq_data[3]
        empty: uint1 = cnt == 0
        rd_valid: uint1 = 0
        if pause == 0 and cnt < 3:
            rd_valid = 1
        q_data.put(d0)
        q_rda.put(na)
        q_rdv.put(rd_valid)
        q_valid.put(1 - empty)
        q_baddr.put(a0)
        q_empty.put(empty)
        full: uint1 = cnt == 4
        q_full.put(full)
        # next state
        n_na: UInt(AW) = na
        n_rp: uint1 = rp
        n_rq: UInt(AW) = rq
        n_cnt: UInt(3) = cnt
        n_a0: UInt(AW) = a0
        n_a1: UInt(AW) = a1
        n_a2: UInt(AW) = a2
        n_a3: UInt(AW) = a3
        n_d0: UInt(128) = d0
        n_d1: UInt(128) = d1
        n_d2: UInt(128) = d2
        n_d3: UInt(128) = d3
        if rst:
            if flush:
                n_na = raddr
                n_rp = 0
                n_cnt = 0
            else:
                push: uint1 = rp
                pop: uint1 = pop_i & (1 - empty)
                if pop:
                    n_a0 = a1
                    n_a1 = a2
                    n_a2 = a3
                    n_d0 = d1
                    n_d1 = d2
                    n_d2 = d3
                if push:
                    slot: UInt(3) = cnt - pop
                    if slot == 0:
                        n_a0 = rq
                        n_d0 = bram
                    elif slot == 1:
                        n_a1 = rq
                        n_d1 = bram
                    elif slot == 2:
                        n_a2 = rq
                        n_d2 = bram
                    else:
                        n_a3 = rq
                        n_d3 = bram
                n_cnt = cnt + push - pop
                n_rp = rd_valid
                n_rq = na
                if rd_valid:
                    n_na = na + 1
        # stores
        rd_reg = rd_new
        next_addr = n_na
        req_pending = n_rp
        req_addr = n_rq
        count = n_cnt
        fq_addr[0] = n_a0
        fq_addr[1] = n_a1
        fq_addr[2] = n_a2
        fq_addr[3] = n_a3
        fq_data[0] = n_d0
        fq_data[1] = n_d1
        fq_data[2] = n_d2
        fq_data[3] = n_d3


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
    # IRAM_ROWS (default 256, U4D_IRAM_ROWS): the 4,096-row register lowering ran
    # Catapult's architect to 15.7 GB in 10 min (stopped); every trace writes
    # words < 48 and reads above them are uninit (masked), so 256 rows aliasing
    # the address is exact on every defined slot (track C's C7 workaround).
    import os
    rows = int(os.environ.get("U4D_IRAM_ROWS", "256"))
    iram = Memory("iram", "UInt(128)", rows=str(rows), ports=D.IRAM.ports, collision="refuse", reset=False)
    return Architecture(name=f"fetch_d12_{body}", parameters={"N": n, "AW": D.G.INSTR_ADDR_W, "IRM": rows - 1},
                        memories=tuple(mems) + (iram,), channels=tuple(ch),
                        units=(src, loader, fq, sink))


BODY = "landed"  # u4d_build/u4d_check: --form-arg body=amended (env U4D_BODY)


def make(n, w=128, inst="iram"):
    import os
    assert inst == "iram"
    return architecture(n, os.environ.get("U4D_BODY", BODY)).region("systemc", {"iram": "registers"})


run = D.run

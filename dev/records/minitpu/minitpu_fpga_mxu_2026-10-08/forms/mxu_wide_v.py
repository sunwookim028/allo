# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""minitpu_fpga_mxu_2026-10-08: ``mxu_wide.py`` with the back's lane rings typed
for an FPGA mux: ``mem`` is ``UInt(64)[D, ENTRIES]`` (one row per lane) and the
ring pointers are ``UInt(4)`` (``cnt`` ``UInt(5)``), so a lane's head read is a
16:1 mux of its own row. In ``mxu_wide.py`` the index ``lane * ENTRIES +
rd[lane]`` with an ``int32`` pointer is unbounded to Vitis, and with ``mem``
partitioned complete every lane's read became a ``D * ENTRIES``:1 mux (256:1 at
DIM 16). Same function (ENTRIES = 16; the pointers wrap at ENTRIES - 1 as
before); everything else is ``mxu_wide.py`` verbatim.

Original docstring (``mxu_wide.py``):

D-21 inverse angle: the M1 ``lockstep`` MXU (``mxu_unit.mxu_front`` ->
``pe_unit`` grid -> ``mxu_unit.mxu_back``) with its two lane arrays packed into
one word per cycle, so that EVERY region port is a rank-1 array read or written
once per iteration and the SystemC emitter makes it a Connections stream
(track C finding C1: ``UInt(16)[N * D]`` becomes RAM pins, read one word per
cycle, which is why M1 ran at 74 cycles per row). ``input_data_i`` is
``UInt(16 D)[N]`` and ``output_data_o`` is ``UInt(16 D SUB)[N]`` (sublane ``s``
of lane ``c`` at bits ``16 (s D + c)``), ``mxu.sv``'s own port shapes. The
bodies are ``mxu_unit.py``'s line for line; the three shift loops have
distinct names so ``--unroll-inner`` reaches each (track C proposal 6), and the
``ctl`` link is ``CTLD`` deep so the front is not throttled by the back
(the back's iteration ``t`` runs when row ``D - 1``'s psum tokens for ``t``
arrive, ~2 D hops after the front emitted them).

``make(n, w, inst)`` is ``u3c_build.py``'s call shape; ``n`` is the trace
length of the trace kernels and, for the wrapper (``mxu_allo.sv``), the number
of cycles the RTL runs before ``done`` -- build it large.
"""
from __future__ import annotations

import os

from allo.compose import Architecture, Channel, Memory, unit

from examples.minitpu.harness import ref_mxu
from examples.minitpu.units import mxu as U
from examples.minitpu.units.mxu_pe_unit import pe_channels, pe_unit


def wide_channels(base):
    # MRTL_PE_DEPTH: the grid links' depth (pe_channels: 2); a probe of whether a deeper link lifts the token rate
    qd = os.environ.get("MRTL_PE_DEPTH")
    if qd:
        base = tuple(Channel(c.name, c.dtype, qd, c.shape, c.carries) for c in base)
    return base + (Channel("ctl", "UInt(32)", "CTLD", (), "front -> back: rst_n"),)


@unit(memories=("RST", "PUSH", "KIND", "DATA", "CMT", "RDY", "ACC"),
      writes=("lhsx", "wx", "ctl"), parameters=("N", "D", "PE", "SKEW", "SPAN"))
def mxu_front_w(rst: UInt(1)[N], push: UInt(1)[N], kind: UInt(1)[N], data: UInt(16 * D)[N],
                cmt: UInt(1)[N], rdy: UInt(1)[N], acc: UInt(1)[N]):
    hv: UInt(1) = 0
    hkind: UInt(1) = 0
    hdata: UInt(16)[D] = 0
    skd: UInt(16)[D * SKEW] = 0
    skv: UInt(1)[D * SKEW] = 0
    wss: UInt(1)[SKEW] = 0
    cbs: UInt(1)[SPAN + 1] = 0
    load_bank: UInt(1) = 0
    loaded_bank: UInt(1) = 1
    waiting: UInt(1) = 0
    for t in range(N):
        r_n: UInt(1) = rst[t]
        p: UInt(1) = push[t]
        k: UInt(1) = kind[t]
        dw: UInt(16 * D) = data[t]
        cm: UInt(1) = cmt[t]
        rdy[t] = 1
        acc[t] = hv
        consume: UInt(1) = hv
        tile_starts: UInt(1) = 0
        rhs_beat: UInt(1) = 0
        if consume == 1:
            if hkind == 0:
                if waiting == 1:
                    tile_starts = 1
            else:
                rhs_beat = 1
        wss[0] = tile_starts
        cbs[0] = loaded_bank
        c_t: UInt(32) = 0
        c_t[0] = r_n
        ctl.put(c_t)
        with allo.meta_for(D) as row:
            wt: UInt(32) = 0
            wt[0:16] = skd[row * SKEW + row * PE]
            wt[16] = skv[row * SKEW + row * PE]
            wt[17] = wss[row * PE]
            wt[18] = cbs[row * PE]
            wt[19] = r_n
            lhsx[row, 0].put(wt)
        with allo.meta_for(D) as col:
            w16: UInt(16) = hdata[col]
            nw: UInt(64) = 0
            nw[0:16] = w16
            nw[16:32] = w16
            if rhs_beat == 1:
                if load_bank == 0:
                    nw[32] = 1
                else:
                    nw[33] = 1
            wx[0, col].put(nw)
        for sa in range(SKEW - 1):
            idx: int32 = SKEW - 1 - sa
            with allo.meta_for(D) as lane:
                skd[lane * SKEW + idx] = skd[lane * SKEW + idx - 1]
                skv[lane * SKEW + idx] = skv[lane * SKEW + idx - 1]
        with allo.meta_for(D) as lane:
            skd[lane * SKEW] = hdata[lane]
            skv[lane * SKEW] = 0
            if consume == 1:
                if hkind == 0:
                    skv[lane * SKEW] = 1
        for sb in range(SPAN):
            idx2: int32 = SPAN - sb
            cbs[idx2] = cbs[idx2 - 1]
        for sc in range(SKEW - 1):
            idx3: int32 = SKEW - 1 - sc
            wss[idx3] = wss[idx3 - 1]
        if r_n == 0:
            load_bank = 0
            loaded_bank = 1
            waiting = 0
            for sd in range(SKEW):
                wss[sd] = 0
                with allo.meta_for(D) as lane:
                    skv[lane * SKEW + sd] = 0
            hv = 0
        else:
            if cm == 1:
                lb_old: UInt(1) = load_bank
                load_bank = 1 - lb_old
                loaded_bank = lb_old
                waiting = 1
            elif tile_starts == 1:
                waiting = 0
            hv = p
            hkind = k
            with allo.meta_for(D) as lane:
                hdata[lane] = dw[16 * lane:16 * lane + 16]


@unit(memories=("POP", "VLD", "ODATA"), reads=("px", "ctl"),
      parameters=("N", "D", "SUB", "ENTRIES"))
def mxu_back_w(pop: UInt(1)[N], vld: UInt(1)[N], odata: UInt(16 * D * SUB)[N]):
    gidx: UInt(2)[D] = 0
    gq: UInt(64)[D] = 0
    mem: UInt(64)[D, ENTRIES] = 0
    rd: UInt(4)[D] = 0
    wr: UInt(4)[D] = 0
    cnt: UInt(5)[D] = 0
    for t in range(N):
        q: UInt(1) = pop[t]
        c_t: UInt(32) = ctl.get()
        r_n: UInt(1) = c_t[0]
        valid: UInt(1) = 1
        with allo.meta_for(D) as lane:
            if cnt[lane] == 0:
                valid = 0
        vld[t] = valid
        ow: UInt(16 * D * SUB) = 0
        with allo.meta_for(D) as lane:
            head: UInt(64) = mem[lane, rd[lane]]
            with allo.meta_for(SUB) as sub:
                ow[16 * (sub * D + lane):16 * (sub * D + lane) + 16] = head[16 * sub:16 * (sub + 1)]
        odata[t] = ow
        consume: UInt(1) = q & valid
        with allo.meta_for(D) as lane:
            sp: UInt(32) = px[D, lane].get()
            res: UInt(24) = sp[0:24]
            rv: UInt(1) = sp[24]
            rounded: UInt(24) = res + 0x7F + res[8]
            bf: UInt(16) = rounded[8:24]
            gnext: UInt(64) = gq[lane]
            gi: UInt(2) = gidx[lane]
            with allo.meta_for(SUB) as sub:
                if gi == sub:
                    gnext[16 * sub:16 * (sub + 1)] = bf
            do_push: UInt(1) = 0
            if rv == 1:
                gq[lane] = gnext
                if gi == SUB - 1:
                    do_push = 1
            do_pop: UInt(1) = 0
            if consume == 1:
                if cnt[lane] > 0:
                    do_pop = 1
            if do_push == 1:
                if cnt[lane] == ENTRIES and do_pop == 0:
                    do_push = 0
            if r_n == 0:
                gidx[lane] = 0
                rd[lane] = 0
                wr[lane] = 0
                cnt[lane] = 0
            else:
                if rv == 1:
                    gidx[lane] = gi + 1
                if do_push == 1:
                    mem[lane, wr[lane]] = gnext
                    if wr[lane] == ENTRIES - 1:
                        wr[lane] = 0
                    else:
                        wr[lane] = wr[lane] + 1
                if do_pop == 1:
                    if rd[lane] == ENTRIES - 1:
                        rd[lane] = 0
                    else:
                        rd[lane] = rd[lane] + 1
                cnt[lane] = cnt[lane] + do_push - do_pop


def make(n, w=0, inst="dim2"):
    D = U.DIMS[inst]
    PE = ref_mxu.PE_LATENCY
    mems = (Memory("RST", "UInt(1)[N]"), Memory("PUSH", "UInt(1)[N]"), Memory("KIND", "UInt(1)[N]"),
            Memory("DATA", "UInt(16 * D)[N]"), Memory("CMT", "UInt(1)[N]"), Memory("POP", "UInt(1)[N]"),
            Memory("RDY", "UInt(1)[N]"), Memory("ACC", "UInt(1)[N]"), Memory("VLD", "UInt(1)[N]"),
            Memory("ODATA", "UInt(16 * D * SUB)[N]"))
    params = {"N": n, "D": D, "PE": PE, "SUB": U.SUB, "ENTRIES": U.ENTRIES, "SKEW": (D - 1) * PE + 1,
              "SPAN": ref_mxu.switch_span(D), "ROW0_PSUM_ZERO": 1, "EDGE_OUT": 0, "CTLD": 4 * D + 8}
    arch = Architecture(name=f"mxu_widev_{inst}", parameters=params, memories=mems,
                        channels=wide_channels(pe_channels()), units=(mxu_front_w, pe_unit, mxu_back_w))
    return arch.region()


def run(mod, cmd, n, w):
    raise NotImplementedError("numpy-free form: Catapult RTL only (the simulator cannot pack words, B6)")

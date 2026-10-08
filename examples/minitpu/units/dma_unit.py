# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track C: MiniTPU's DMA as ``compose.unit``\\ s over Streams (plan D1
as the track states it: the engine with its request/response credit pipe and
its descriptor interface as Streams; D2: the VMEM side as a README D-12 port).

Every unit runs one iteration per cycle and every link carries one token per
cycle (the token index IS the cycle), so the composite is ``dma.sv``'s cycle
model in token time and is held per cycle to Phase 0's oracle (U3 track B's
idea). Tokens (all ``UInt``; data as ``L`` lanes of 32 bits, U3 P-8/S8):

  descriptor interface (sequencer -> DMA, ``dma_desc_adapter``'s outputs)
    x_desc    UInt(64): valid[0] is_store[1] channel[2] vmem_row[3:17] rows[17:31] cols[31:38]
    x_base    UInt(32), x_stride UInt(32)
  control
    x_ctl     UInt(32): rst_n[0] clear_channel_done[1:3] dm_req_ready[3] vmem_gnt[4]
  credit pipe, response (bridge -> DMA, in order)
    x_rsp     UInt(32): valid[0] last[1] resp[2:4];  x_rspd[L] the beat
  credit pipe, request (DMA -> bridge)
    y_req     UInt(64): valid[0] we[1] addr[2:31] len[31:39] rsp_ready[39];  y_wd[L] wdata
  VMEM DMA port (DMA -> VMEM side, and back)
    y_vreq    UInt(64): rst_n[0] wr_en[1] rd_en[2] wr_ptr[3:19] rd_ptr[19:35];  y_vwd[L]
    x_vrd[L]  vmem_rd_data (2 cycles after rd_en: P-3)
  status
    y_stat    UInt(64): accept[0] done[1:3] idle[3] err[4] err_ch[5] cause[6:8] resp[8:10] rsp_seen[10]
    y_beats   UInt(32), y_ovl UInt(32)

The memory-side contract this boundary keeps (the D-21 shim's too): the
request ``{we, addr (29-bit DM word), len (words - 1, never across a 4 KiB
page), wdata, wstrb all ones}`` is offered with ``valid`` independent of
``ready`` and fires on ``valid & ready``; responses come back IN ORDER, one
per load word with ``last`` on a burst's final word and one per store burst
(after its last W beat), each with ``resp``; ``rsp_ready`` is the landing
FIFO's room (always 1 with nothing outstanding: stale beats are dropped).

Unit order is declaration order: source, engine, (VMEM side,) sink.
"""

from __future__ import annotations

from allo.compose import Architecture, Channel, Instance, Memory, Port, unit
from allo.ir.types import Stateful  # noqa: F401  (PAYLOAD values)

from examples.minitpu.units.dma_addr_gen import addr  # noqa: F401  (the engine calls it)

ENGINE_PARAMS = ("N", "CH", "OUT", "LND", "TO", "L", "CRED_M", "OUTC_M", "LNDC_M", "TO_M",
                 "TO_LAST", "PW_M1", "ROW_M", "RDL_M1", "LND_LAST", "OUT_LAST", "ADDR_M", "PAYLOAD")


def engine_parameters(g, n, payload="unreset"):
    """The engine's bound parameters from a ``DmaGeometry`` (README D-20:
    every mask and bound is derived from the record, none typed in)."""
    g.legality()
    assert g.CHANNELS == 2, "the trace ports are sized for two channels (ref_ctrl_dma)"
    return {
        "N": n, "CH": g.CHANNELS, "OUT": g.OUTSTANDING, "LND": g.LANDING_DEPTH,
        "TO": g.TIMEOUT_CYCLES, "L": g.LANES,
        "CRED_M": (1 << g.CRED_W) - 1, "OUTC_M": (1 << g.CRED_W) - 1,
        "LNDC_M": (1 << g.LND_CNT_W) - 1, "TO_M": (1 << g.TO_W) - 1,
        "TO_LAST": (g.TIMEOUT_CYCLES - 1) & ((1 << g.TO_W) - 1),
        "PW_M1": g.PAGE_WORDS - 1, "ROW_M": (1 << g.ROW_BITS) - 1,
        "RDL_M1": g.VMEM_DMA_READ_LATENCY - 1, "LND_LAST": g.LANDING_DEPTH - 1,
        "OUT_LAST": g.OUTSTANDING - 1, "ADDR_M": (1 << g.DRAM_BEAT_ADDR_W) - 1,
        "PAYLOAD": Stateful(reset=False) if payload == "unreset" else Stateful,
    }


def dma_channels():
    one = (("x_desc", "UInt(64)"), ("x_base", "UInt(32)"), ("x_stride", "UInt(32)"),
           ("x_ctl", "UInt(32)"), ("x_rsp", "UInt(32)"), ("y_req", "UInt(64)"),
           ("y_vreq", "UInt(64)"), ("y_stat", "UInt(64)"), ("y_beats", "UInt(32)"),
           ("y_ovl", "UInt(32)"))
    lanes = ("x_rspd", "y_wd", "y_vwd", "x_vrd")
    return tuple(Channel(c, t, "2") for c, t in one) + tuple(
        Channel(c, "UInt(32)", "2", ("L",)) for c in lanes)


@unit(memories=("CI", "DI"), writes=("x_desc", "x_base", "x_stride", "x_ctl", "x_rsp", "x_rspd", "x_vrd"),
      parameters=("N", "NC", "L"))
def dma_src(ci: UInt(32)[N, NC], di: UInt(32)[N, 2 * L]):
    for t in range(N):
        d: UInt(64) = 0
        v1: UInt(1) = ci[t, 1]
        s1: UInt(1) = ci[t, 2]
        c1: UInt(1) = ci[t, 3]
        vr: UInt(14) = ci[t, 4]
        rw: UInt(14) = ci[t, 5]
        cl: UInt(7) = ci[t, 6]
        d[0] = v1
        d[1] = s1
        d[2] = c1
        d[3:17] = vr
        d[17:31] = rw
        d[31:38] = cl
        x_desc.put(d)
        x_base.put(ci[t, 7])
        x_stride.put(ci[t, 8])
        c: UInt(32) = 0
        r1: UInt(1) = ci[t, 0]
        clr: UInt(2) = ci[t, 9]
        rdy: UInt(1) = ci[t, 10]
        gnt: UInt(1) = ci[t, 14]
        c[0] = r1
        c[1:3] = clr
        c[3] = rdy
        c[4] = gnt
        x_ctl.put(c)
        r: UInt(32) = 0
        rv: UInt(1) = ci[t, 11]
        rl: UInt(1) = ci[t, 12]
        rr: UInt(2) = ci[t, 13]
        r[0] = rv
        r[1] = rl
        r[2:4] = rr
        x_rsp.put(r)
        with allo.meta_for(L) as k:
            x_rspd[k].put(di[t, k])
            x_vrd[k].put(di[t, L + k])


@unit(memories=("CI", "DI"), writes=("x_desc", "x_base", "x_stride", "x_ctl", "x_rsp", "x_rspd"),
      parameters=("N", "NC", "L"))
def dma_src_closed(ci: UInt(32)[N, NC], di: UInt(32)[N, 2 * L]):
    """``dma_src`` without the VMEM read data: D2 closes it on the VMEM port."""
    for t in range(N):
        d: UInt(64) = 0
        v1: UInt(1) = ci[t, 1]
        s1: UInt(1) = ci[t, 2]
        c1: UInt(1) = ci[t, 3]
        vr: UInt(14) = ci[t, 4]
        rw: UInt(14) = ci[t, 5]
        cl: UInt(7) = ci[t, 6]
        d[0] = v1
        d[1] = s1
        d[2] = c1
        d[3:17] = vr
        d[17:31] = rw
        d[31:38] = cl
        x_desc.put(d)
        x_base.put(ci[t, 7])
        x_stride.put(ci[t, 8])
        c: UInt(32) = 0
        r1: UInt(1) = ci[t, 0]
        clr: UInt(2) = ci[t, 9]
        rdy: UInt(1) = ci[t, 10]
        c[0] = r1
        c[1:3] = clr
        c[3] = rdy
        c[4] = 1  # vmem_gnt: minitpu_core.sv ties it to 1
        x_ctl.put(c)
        r: UInt(32) = 0
        rv: UInt(1) = ci[t, 11]
        rl: UInt(1) = ci[t, 12]
        rr: UInt(2) = ci[t, 13]
        r[0] = rv
        r[1] = rl
        r[2:4] = rr
        x_rsp.put(r)
        with allo.meta_for(L) as k:
            x_rspd[k].put(di[t, k])


@unit(reads=("x_desc", "x_base", "x_stride", "x_ctl", "x_rsp", "x_rspd", "x_vrd"),
      writes=("y_req", "y_wd", "y_vreq", "y_vwd", "y_stat", "y_beats", "y_ovl"),
      parameters=ENGINE_PARAMS, calls=("addr",))
def dma_engine():
    # dma.sv, units/dma.py ``bits`` with its port arrays replaced by tokens.
    # ---- per-channel descriptor state (reset) ----
    st: int32[CH] = 0
    cstore: int32[CH] = 0
    cvmem: int32[CH] = 0
    clast: int32[CH] = 0
    cbase: UInt(32)[CH] = 0
    cstride: UInt(32)[CH] = 0
    cnext: int32[CH] = 0
    call: int32[CH] = 0
    cbif: int32[CH] = 0
    n_st: int32[CH] = 0
    n_cstore: int32[CH] = 0
    n_cvmem: int32[CH] = 0
    n_clast: int32[CH] = 0
    n_cbase: UInt(32)[CH] = 0
    n_cstride: UInt(32)[CH] = 0
    n_cnext: int32[CH] = 0
    n_call: int32[CH] = 0
    n_cbif: int32[CH] = 0
    cdone: int32 = 0
    eng: int32 = 0
    iss: int32 = 0
    rr: int32 = 0
    st_beat: int32 = 0
    rd_lat: int32 = 0
    st_data: UInt(32)[L] = 0
    credit: int32 = OUT
    ord_ch: int32[OUT] = 0
    ord_vmem: int32[OUT] = 0
    ord_store: int32[OUT] = 0
    ord_head: int32 = 0
    ord_tail: int32 = 0
    ord_count: int32 = 0
    lnd_ch: int32[LND] @ PAYLOAD
    lnd_store: int32[LND] @ PAYLOAD
    lnd_dest: int32[LND] @ PAYLOAD
    lnd_data: UInt(32)[LND, L] @ PAYLOAD
    lnd_r: int32[LND] @ PAYLOAD
    lnd_last: int32[LND] @ PAYLOAD
    lnd_head: int32 = 0
    lnd_tail: int32 = 0
    lnd_count: int32 = 0
    rsp_beat: int32 = 0
    to_cnt: int32 = 0
    err: int32 = 0
    err_ch: int32 = 0
    err_cause: int32 = 0
    err_resp: int32 = 0
    rsp_seen: int32 = 0
    beats: UInt(32) = 0
    ovl_max: int32 = 0
    ovl_cyc: int32 = 0
    rdata: UInt(32)[L] = 0
    vdata: UInt(32)[L] = 0
    for _t in range(N):
        xd: UInt(64) = x_desc.get()
        dbase: UInt(32) = x_base.get()
        dstride: UInt(32) = x_stride.get()
        xc: UInt(32) = x_ctl.get()
        xr: UInt(32) = x_rsp.get()
        with allo.meta_for(L) as k:
            rdata[k] = x_rspd[k].get()
        dv: int32 = xd[0]
        dis: int32 = xd[1]
        dch: int32 = xd[2]
        dvr: int32 = xd[3:17]
        drows: int32 = xd[17:31]
        rst: int32 = xc[0]
        clr: int32 = xc[1:3]
        rdy: int32 = xc[3]
        gnt: int32 = xc[4]
        rv: int32 = xr[0]
        rlast: int32 = xr[1]
        rresp: int32 = xr[2:4]
        if rst == 0:
            for c in range(CH):
                st[c] = 0
                cstore[c] = 0
                cvmem[c] = 0
                clast[c] = 0
                cbase[c] = 0
                cstride[c] = 0
                cnext[c] = 0
                call[c] = 0
                cbif[c] = 0
            cdone = 0
            eng = 0
            iss = 0
            rr = 0
            st_beat = 0
            rd_lat = 0
            for k2 in range(L):
                st_data[k2] = 0
            credit = OUT
            for j in range(OUT):
                ord_ch[j] = 0
                ord_vmem[j] = 0
                ord_store[j] = 0
            ord_head = 0
            ord_tail = 0
            ord_count = 0
            lnd_head = 0
            lnd_tail = 0
            lnd_count = 0
            rsp_beat = 0
            to_cnt = 0
            err = 0
            err_ch = 0
            err_cause = 0
            err_resp = 0
            rsp_seen = 0
            beats = 0
            ovl_max = 0
            ovl_cyc = 0
        cur: int32 = cnext[iss]
        cur14: UInt(14) = cur
        agu: UInt(32) = addr(cbase[iss], cur14, cstride[iss])
        logical: int32 = agu & ADDR_M
        contiguous: int32 = 0
        if cstride[iss] == 1:
            contiguous = 1
        left_m1: int32 = (clast[iss] - cur) & ROW_M
        to_bound_m1: int32 = PW_M1 - (logical & PW_M1)
        left_sat: int32 = left_m1 & 255
        if (left_m1 >> 8) != 0:
            left_sat = 255
        blen: int32 = 0
        if contiguous == 1:
            blen = left_sat
            if to_bound_m1 < left_sat:
                blen = to_bound_m1
        last: int32 = 0
        if blen == left_m1:
            last = 1
        vbase: int32 = (cvmem[iss] + cur) & 0xFFFF
        hch: int32 = lnd_ch[lnd_head]
        hstore: int32 = lnd_store[lnd_head]
        hdest: int32 = lnd_dest[lnd_head]
        hr: int32 = lnd_r[lnd_head]
        hlast: int32 = lnd_last[lnd_head]
        req_valid: int32 = 0
        if eng == 1:
            if credit != 0:
                req_valid = 1
        elif eng == 4:
            if st_beat != 0 or credit != 0:
                req_valid = 1
        we: int32 = 0
        if eng == 4:
            we = 1
        rsp_ready: int32 = 1
        if ord_count != 0:
            if lnd_count >= LND:
                rsp_ready = 0
        lnd_ne: int32 = 0
        if lnd_count != 0:
            lnd_ne = 1
        wr_en: int32 = 0
        if lnd_ne == 1 and hstore == 0 and hr == 0:
            wr_en = 1
        rd_en: int32 = 0
        if eng == 2 and wr_en == 0:
            rd_en = 1
        accept: int32 = 0
        if dv == 1 and st[dch] == 0:
            accept = 1
        idle: int32 = 0
        if ord_count == 0 and lnd_count == 0:
            idle = 1
        for c2 in range(CH):
            if st[c2] == 1 or st[c2] == 2:
                idle = 0
        # ---- this cycle's outputs, as tokens ----
        q: UInt(64) = 0
        f1: UInt(1) = req_valid
        f2: UInt(1) = we
        f29: UInt(29) = logical
        f8: UInt(8) = blen
        f3: UInt(1) = rsp_ready
        q[0] = f1
        q[1] = f2
        q[2:31] = f29
        q[31:39] = f8
        q[39] = f3
        y_req.put(q)
        with allo.meta_for(L) as k:
            y_wd[k].put(st_data[k])
        vq: UInt(64) = 0
        g1: UInt(1) = rst
        g2: UInt(1) = wr_en
        g3: UInt(1) = rd_en
        g16: UInt(16) = hdest
        rp: UInt(16) = (cvmem[iss] + cur + st_beat) & 0xFFFF
        vq[0] = g1
        vq[1] = g2
        vq[2] = g3
        vq[3:19] = g16
        vq[19:35] = rp
        y_vreq.put(vq)
        with allo.meta_for(L) as k:
            y_vwd[k].put(lnd_data[lnd_head, k])
        sq: UInt(64) = 0
        h1: UInt(1) = accept
        h2: UInt(2) = cdone
        h3: UInt(1) = idle
        h4: UInt(1) = err
        h5: UInt(1) = err_ch
        h6: UInt(2) = err_cause
        h7: UInt(2) = err_resp
        h8: UInt(1) = rsp_seen
        sq[0] = h1
        sq[1:3] = h2
        sq[3] = h3
        sq[4] = h4
        sq[5] = h5
        sq[6:8] = h6
        sq[8:10] = h7
        sq[10] = h8
        y_stat.put(sq)
        y_beats.put(beats)
        ovl: UInt(32) = ovl_cyc
        ovl = (ovl << 8) | ovl_max
        y_ovl.put(ovl)
        # ---- the VMEM read data of this cycle (2 cycles after its rd_en) ----
        with allo.meta_for(L) as k:
            vdata[k] = x_vrd[k].get()
        # ---- the edge ----
        if rst == 1:
            req_fire: int32 = 0
            if req_valid == 1 and rdy == 1:
                req_fire = 1
            rsp_fire: int32 = 0
            if rv == 1 and rsp_ready == 1:
                rsp_fire = 1
            real_rsp: int32 = 0
            if rsp_fire == 1 and ord_count != 0:
                real_rsp = 1
            lnd_pop: int32 = 0
            if lnd_ne == 1:
                if hstore == 1 or hr != 0 or gnt == 1:
                    lnd_pop = 1
            issue_push: int32 = 0
            ord_pop: int32 = 0
            lnd_push: int32 = 0
            drain_last: int32 = lnd_pop & hlast
            timeout: int32 = 0
            oh_ch: int32 = ord_ch[ord_head]
            oh_vmem: int32 = ord_vmem[ord_head]
            oh_store: int32 = ord_store[ord_head]
            for c3 in range(CH):
                n_st[c3] = st[c3]
                n_cstore[c3] = cstore[c3]
                n_cvmem[c3] = cvmem[c3]
                n_clast[c3] = clast[c3]
                n_cbase[c3] = cbase[c3]
                n_cstride[c3] = cstride[c3]
                n_cnext[c3] = cnext[c3]
                n_call[c3] = call[c3]
                n_cbif[c3] = cbif[c3]
            n_done: int32 = cdone
            n_eng: int32 = eng
            n_iss: int32 = iss
            n_rr: int32 = rr
            n_st_beat: int32 = st_beat
            n_rd_lat: int32 = rd_lat
            cap: int32 = 0
            n_ord_head: int32 = ord_head
            n_ord_tail: int32 = ord_tail
            n_lnd_head: int32 = lnd_head
            n_lnd_tail: int32 = lnd_tail
            n_rsp_beat: int32 = rsp_beat
            n_to_cnt: int32 = to_cnt
            n_err: int32 = err
            n_err_ch: int32 = err_ch
            n_err_cause: int32 = err_cause
            n_err_resp: int32 = err_resp
            n_rsp_seen: int32 = rsp_seen
            n_ovl_max: int32 = ovl_max
            n_ovl_cyc: int32 = ovl_cyc
            if accept == 1:
                n_st[dch] = 1
                n_cstore[dch] = dis
                n_cvmem[dch] = dvr
                n_clast[dch] = drows
                n_cbase[dch] = dbase
                n_cstride[dch] = dstride
                n_cnext[dch] = 0
                n_call[dch] = 0
            for c4 in range(CH):
                if ((clr >> c4) & 1) == 1:
                    n_done = n_done & (3 - (1 << c4))
                    if st[c4] == 3:
                        n_st[c4] = 0
            if eng == 0:
                pick: int32 = 0
                any_p: int32 = 0
                for k3 in range(CH):
                    idx: int32 = (rr + (CH - 1 - k3)) % CH
                    if st[idx] == 1:
                        any_p = 1
                        pick = idx
                if any_p == 1:
                    n_iss = pick
                    n_st[pick] = 2
                    n_rr = (pick + 1) % CH
                    n_st_beat = 0
                    if cstore[pick] == 1:
                        n_eng = 2
                    else:
                        n_eng = 1
            elif eng == 1:
                if req_fire == 1:
                    issue_push = 1
                    if last == 1:
                        n_call[iss] = 1
                        n_eng = 0
                    else:
                        n_cnext[iss] = (cur + blen + 1) & ROW_M
            elif eng == 2:
                if rd_en == 1 and gnt == 1:
                    n_rd_lat = 0
                    n_eng = 3
            elif eng == 3:
                if rd_lat == RDL_M1:
                    cap = 1
                    n_eng = 4
                else:
                    n_rd_lat = (rd_lat + 1) & 3
            elif eng == 4:
                if req_fire == 1:
                    if st_beat == 0:
                        issue_push = 1
                    if st_beat == blen:
                        if last == 1:
                            n_call[iss] = 1
                            n_eng = 0
                        else:
                            n_cnext[iss] = (cur + blen + 1) & ROW_M
                            n_st_beat = 0
                            n_eng = 2
                    else:
                        n_st_beat = (st_beat + 1) & 255
                        n_eng = 2
            if rv == 1:
                n_rsp_seen = 1
            if real_rsp == 1:
                if lnd_tail == LND_LAST:
                    n_lnd_tail = 0
                else:
                    n_lnd_tail = lnd_tail + 1
                lnd_push = 1
                if rlast == 1:
                    ord_pop = 1
                    if ord_head == OUT_LAST:
                        n_ord_head = 0
                    else:
                        n_ord_head = ord_head + 1
                    n_rsp_beat = 0
                else:
                    n_rsp_beat = (rsp_beat + 1) & 255
            if issue_push == 1:
                ord_ch[ord_tail] = iss
                ord_vmem[ord_tail] = vbase
                ord_store[ord_tail] = cstore[iss]
                if ord_tail == OUT_LAST:
                    n_ord_tail = 0
                else:
                    n_ord_tail = ord_tail + 1
            n_ord_count: int32 = (ord_count + issue_push - ord_pop) & OUTC_M
            if lnd_pop == 1:
                if lnd_head == LND_LAST:
                    n_lnd_head = 0
                else:
                    n_lnd_head = lnd_head + 1
                if hr != 0:
                    n_err = 1
                    if err == 0:
                        n_err_ch = hch
                        n_err_cause = 1
                        n_err_resp = hr
            n_lnd_count: int32 = (lnd_count + lnd_push - lnd_pop) & LNDC_M
            bc: int32 = 0
            if lnd_pop == 1 and hstore == 0 and hr == 0:
                bc = 1
            if req_fire == 1 and we == 1:
                bc = bc + 1
            n_beats: UInt(32) = beats + bc
            nonidle: int32 = 0
            for c5 in range(CH):
                if st[c5] == 1 or st[c5] == 2:
                    nonidle = nonidle + 1
            if nonidle > ovl_max:
                n_ovl_max = nonidle & 255
            if nonidle >= 2 and ovl_cyc != 0xFFFFFF:
                n_ovl_cyc = ovl_cyc + 1
            n_credit: int32 = (credit - issue_push + ord_pop) & CRED_M
            for c6 in range(CH):
                inc: int32 = 0
                if issue_push == 1 and iss == c6:
                    inc = 1
                dec: int32 = 0
                if drain_last == 1 and hch == c6:
                    dec = 1
                n_cbif[c6] = (cbif[c6] + inc - dec) & OUTC_M
                if dec == 1 and inc == 0 and cbif[c6] == 1 and call[c6] == 1:
                    n_done = n_done | (1 << c6)
                    n_st[c6] = 3
            if ord_count == 0 or rv == 1 or req_fire == 1:
                n_to_cnt = 0
            elif TO != 0 and to_cnt == TO_LAST:
                timeout = 1
            else:
                n_to_cnt = (to_cnt + 1) & TO_M
            if timeout == 1:
                n_err = 1
                if err == 0:
                    n_err_ch = oh_ch
                    n_err_cause = 2
                    n_err_resp = 0
                for c7 in range(CH):
                    n_cbif[c7] = 0
                    if st[c7] == 1 or st[c7] == 2:
                        n_done = n_done | (1 << c7)
                        n_st[c7] = 3
                n_credit = OUT
                n_ord_head = 0
                n_ord_tail = 0
                n_ord_count = 0
                n_lnd_head = 0
                n_lnd_tail = 0
                n_lnd_count = 0
                n_rsp_beat = 0
                n_st_beat = 0
                n_eng = 0
                n_to_cnt = 0
            if real_rsp == 1:
                lnd_ch[lnd_tail] = oh_ch
                lnd_store[lnd_tail] = oh_store
                lnd_dest[lnd_tail] = (oh_vmem + rsp_beat) & 0xFFFF
                lnd_r[lnd_tail] = rresp
                lnd_last[lnd_tail] = rlast
                for k4 in range(L):
                    lnd_data[lnd_tail, k4] = rdata[k4]
            if cap == 1:
                for k5 in range(L):
                    st_data[k5] = vdata[k5]
            for c8 in range(CH):
                st[c8] = n_st[c8]
                cstore[c8] = n_cstore[c8]
                cvmem[c8] = n_cvmem[c8]
                clast[c8] = n_clast[c8]
                cbase[c8] = n_cbase[c8]
                cstride[c8] = n_cstride[c8]
                cnext[c8] = n_cnext[c8]
                call[c8] = n_call[c8]
                cbif[c8] = n_cbif[c8]
            cdone = n_done & 3
            eng = n_eng
            iss = n_iss
            rr = n_rr
            st_beat = n_st_beat
            rd_lat = n_rd_lat
            credit = n_credit
            ord_head = n_ord_head
            ord_tail = n_ord_tail
            ord_count = n_ord_count
            lnd_head = n_lnd_head
            lnd_tail = n_lnd_tail
            lnd_count = n_lnd_count
            rsp_beat = n_rsp_beat
            to_cnt = n_to_cnt
            err = n_err
            err_ch = n_err_ch
            err_cause = n_err_cause
            err_resp = n_err_resp
            rsp_seen = n_rsp_seen
            beats = n_beats
            ovl_max = n_ovl_max
            ovl_cyc = n_ovl_cyc


@unit(memories=("CO", "DO"), reads=("y_req", "y_wd", "y_vreq", "y_vwd", "y_stat", "y_beats", "y_ovl"),
      parameters=("N", "NO", "L"))
def dma_sink(co: UInt(32)[N, NO], do: UInt(32)[N, 2 * L]):
    # COUT_NAMES order (units/dma.py)
    for t in range(N):
        q: UInt(64) = y_req.get()
        vq: UInt(64) = y_vreq.get()
        sq: UInt(64) = y_stat.get()
        co[t, 0] = sq[0]
        co[t, 1] = sq[1:3]
        co[t, 2] = sq[3]
        co[t, 3] = q[0]
        co[t, 4] = q[1]
        co[t, 5] = q[2:31]
        co[t, 6] = q[31:39]
        co[t, 7] = 0xFFFFFFFF  # dm_req_wstrb: full-row stores (dma.sv:236)
        co[t, 8] = q[39]
        co[t, 9] = vq[1]
        co[t, 10] = vq[3:19]
        co[t, 11] = vq[2]
        co[t, 12] = vq[19:35]
        co[t, 13] = sq[4]
        co[t, 14] = sq[5]
        co[t, 15] = sq[6:8]
        co[t, 16] = sq[8:10]
        co[t, 17] = sq[10]
        co[t, 18] = y_beats.get()
        co[t, 19] = y_ovl.get()
        with allo.meta_for(L) as k:
            do[t, k] = y_wd[k].get()
            do[t, L + k] = y_vwd[k].get()


def streams_architecture(n, inst="core", payload="unreset"):
    """D1 over Streams: source -> engine -> sink, the VMEM read data from the
    trace (open loop, as Phase 0's oracle)."""
    from examples.minitpu.units.dma_params import GEOMETRIES  # noqa: PLC0415
    from examples.minitpu.units.dma import CIN_NAMES, COUT_NAMES  # noqa: PLC0415

    params = engine_parameters(GEOMETRIES[inst], n, payload)
    params.update({"NC": len(CIN_NAMES), "NO": len(COUT_NAMES)})
    return Architecture(
        name=f"dma_streams_{inst}",
        parameters=params,
        memories=(Memory("CI", "UInt(32)[N, NC]"), Memory("DI", "UInt(32)[N, 2 * L]"),
                  Memory("CO", "UInt(32)[N, NO]"), Memory("DO", "UInt(32)[N, 2 * L]")),
        channels=dma_channels(),
        units=(dma_src, dma_engine, dma_sink),
    )


# ---------------------------------------------------------------------------
# D2: the VMEM side as a README D-12 port.
#
# ``vmem`` is the U2 word array (``vpu_word_array_d12``'s declaration at the
# full geometry: 4,096 words of SUB x L x 32 = 1,024 bits, unreset), with the
# compute port ``c`` (``rw``, latency 3) and the DMA port ``d`` (``rw``,
# latency ``VMEM_DMA_READ_LATENCY`` = 2, P-3), each write visible one edge
# later, the same-word collision an obligation (MiniTPU issue #21).
# ``vmem_group`` is ``vpu_dma_group`` + ``vpu_vmem_simd``'s ``dma_rvalid_q``
# and owns ``vmem.d``; ``vmem_compute_idle`` owns ``vmem.c`` and never
# writes (the compute side is out of D2's scope: U5).
#
# Token time: the generated server puts, at iteration t, its read pipe's
# last stage -- the read issued at t - (VL - 1), i.e. the word array's
# ``rdata`` AFTER edge t (the U2 record's post-edge row). The group samples
# its inputs before the edge (``rtl.py``'s pre convention, as the DMA does),
# so it holds the token one iteration (``hold``): during cycle t the word
# array's output is the token of iteration t - 1. That register is the
# token-time image of reading the pipe's last register, not hardware.
# ---------------------------------------------------------------------------


def _p3_legality(p):
    """P-3 across three declarations: the port's read latency, the group's
    rvalid pipe (VL) and the engine's capture (RDL_M1 + 1). Run by
    ``vmem_architecture``: a unit's ``legality`` sees only its own parameters
    and an owner cannot read its port's declared latency (finding C6)."""
    assert p["VL"] == p["RDL_M1"] + 1 == p["PORT_D_LATENCY"], (
        f"vmem.d declares read latency {p['PORT_D_LATENCY']}, vmem_group counts "
        f"{p['VL']}, and the DMA engine captures vmem_rd_data {p['RDL_M1'] + 1} "
        f"cycles after vmem_rd_en (I_ST_RDW); the "
        f"store path would latch the wrong cycle. One number, "
        f"DmaGeometry.VMEM_DMA_READ_LATENCY, must set both (P-3)")

def _group_legality(p):
    assert p["WB"] == p["SUB"] * p["L"] * 32, (
        f"vmem word WB={p['WB']} is not SUB x beat = {p['SUB']} x {p['L']} x 32 bits")


@unit(memories=("vmem.d",), reads=("y_vreq", "y_vwd"), writes=("x_vrd", "g_vmo", "g_vmd"),
      parameters=("N", "L", "SUB", "VL", "WB", "ROWS_M"), legality=_group_legality)
def vmem_group(mem):
    gather: UInt(WB) = 0   # gather_q (unreset in the RTL: never cleared here)
    scatter: UInt(WB) = 0  # scatter_q (unreset)
    hold: UInt(WB) = 0     # the word array's rdata this cycle (token of t - 1)
    gword: int32 = 0
    filled: int32 = 0
    sidx: int32 = 0
    sfrom: int32 = 0
    rvp: int32[VL] = 0     # vpu_vmem_simd dma_rvalid_q
    wd: UInt(32)[L] = 0
    ob: UInt(32)[L] = 0
    for _t in range(N):
        vq: UInt(64) = y_vreq.get()
        with allo.meta_for(L) as k:
            wd[k] = y_vwd[k].get()
        g_vmo.put(vq)
        with allo.meta_for(L) as k:
            g_vmd[k].put(wd[k])
        rst: int32 = vq[0]
        wr_en: int32 = vq[1]
        rd_en: int32 = vq[2]
        wptr: int32 = vq[3:19]
        rptr: int32 = vq[19:35]
        # ---- beat_rdata_o: this cycle's vmem_rd_data, from state ----
        with allo.meta_for(SUB) as s:
            if sidx == s:
                with allo.meta_for(L) as k:
                    if sfrom == 1:
                        ob[k] = scatter[32 * (s * L + k):32 * (s * L + k + 1)]
                    else:
                        ob[k] = hold[32 * (s * L + k):32 * (s * L + k + 1)]
        with allo.meta_for(L) as k:
            x_vrd[k].put(ob[k])
        # ---- word side (minitpu_core.sv's select, then vpu_dma_group) ----
        en: int32 = wr_en | rd_en
        we: int32 = wr_en
        ptr: int32 = rptr
        if wr_en == 1:
            ptr = wptr
        a14: int32 = ptr & 0x3FFF
        word: int32 = a14 >> 2
        idx: int32 = a14 & 3
        commit: int32 = 0
        if en == 1 and we == 1 and idx == SUB - 1:
            commit = 1
        fetch: int32 = 0
        if en == 1 and we == 0 and idx == 0:
            fetch = 1
        wwd: UInt(WB) = gather
        if en == 1 and we == 1:
            with allo.meta_for(SUB) as s:
                if idx == s:
                    with allo.meta_for(L) as k:
                        wwd[32 * (s * L + k):32 * (s * L + k + 1)] = wd[k]
        waddr: int32 = word & ROWS_M  # identity at the full 4,096 rows (C7: csim's rows)
        q: UInt(WB) = mem[waddr]
        cm: uint1 = commit
        if cm:
            mem[waddr] = wwd
        # ---- the edge ----
        rv_last: int32 = rvp[VL - 1]
        if rst == 0:
            filled = 0
            gword = 0
            sidx = 0
            sfrom = 0
            for j in range(VL):
                rvp[j] = 0
        else:
            if en == 1 and we == 1:
                gather = wwd
                gword = word
                if commit == 1:
                    filled = 0
                else:
                    filled = filled | (1 << idx)
            if en == 1 and we == 0:
                sidx = idx
                sfrom = 0
                if idx != 0:
                    sfrom = 1
            if rv_last == 1:
                scatter = hold
            for j2 in range(VL - 1):
                rvp[VL - 1 - j2] = rvp[VL - 2 - j2]
            rvp[0] = fetch
        hold = q


@unit(memories=("vmem.c",), parameters=("N", "WB"))
def vmem_compute_idle(mem):
    # the compute port's owner: one read of word 0 per cycle, never a write
    # (a D-12 port must have exactly one owner; U5 composes the real one)
    for _t in range(N):
        z: int32 = 0
        x: UInt(WB) = mem[z]


def vmem_architecture(n, inst="core", payload="unreset", rows=None):
    """D2: source (no VMEM data) -> engine <-> vmem_group (vmem.d) -> sink;
    vmem_compute_idle owns vmem.c. ``rows`` (default: the full 4,096) below
    the geometry is the csim workaround of finding C7 (a kernel-local array
    lives on the 64 KB SC_THREAD stack; 4,096 x 128 B segfaults): the word
    index wraps at ``rows``, exact on traces that stay below it."""
    from examples.minitpu.units.dma_params import GEOMETRIES  # noqa: PLC0415
    from examples.minitpu.units.dma import CIN_NAMES, COUT_NAMES  # noqa: PLC0415

    g = GEOMETRIES[inst]
    params = engine_parameters(g, n, payload)
    sub = g.NUM_SUBLANES
    words = 1 << g.VMEM_ADDR_W
    rows = rows or words
    assert rows & (rows - 1) == 0 and rows <= words, f"rows={rows}: a power of two <= {words}"
    params.update({"NC": len(CIN_NAMES), "NO": len(COUT_NAMES), "SUB": sub,
                   "VL": g.VMEM_DMA_READ_LATENCY, "WB": sub * g.BEAT_BITS, "ROWS_M": rows - 1})
    port_d = Port("d", "rw", latency=g.VMEM_DMA_READ_LATENCY, visible=1)
    _p3_legality(dict(params, PORT_D_LATENCY=port_d.latency))
    vmem = Memory("vmem", "UInt(WB)", rows=str(rows),
                  ports=(Port("c", "rw", latency=3, visible=1),
                         port_d),
                  collision="obligation", reset=False)
    return Architecture(
        name=f"dma_vmem_{inst}" + ("" if rows == words else f"_r{rows}"),
        parameters=params,
        memories=(Memory("CI", "UInt(32)[N, NC]"), Memory("DI", "UInt(32)[N, 2 * L]"),
                  Memory("CO", "UInt(32)[N, NO]"), Memory("DO", "UInt(32)[N, 2 * L]"), vmem),
        channels=dma_channels() + (Channel("g_vmo", "UInt(64)", "2"),
                                   Channel("g_vmd", "UInt(32)", "2", ("L",))),
        units=(dma_src_closed, dma_engine, vmem_group, vmem_compute_idle,
               Instance(dma_sink, "dma_sink_d2", {"y_vreq": "g_vmo", "y_vwd": "g_vmd"})),
        obligations={"vmem": "MiniTPU issue #21: the program keeps compute and DMA off one "
                             "word in one cycle (vpu_vmem_simd.sv:121, simulation only); "
                             "here the compute port never writes"},
    )

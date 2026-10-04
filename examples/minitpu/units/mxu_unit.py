# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``mxu.sv``'s front and back as ``compose.unit``\\ s (plan M1), around
the ``mxu_pe_unit.pe_unit`` grid. Lazy annotations, as ``mxu_pe_unit.py``.

Every unit runs one iteration per cycle and every link carries one token per
cycle (the token index is the cycle), so the composite is ``mxu.sv``'s cycle
model in token time and is judged per cycle on the push / commit / pop
contract. What ``mxu.sv`` leaves to implementation is kept where it is
cheapest here, and recorded:

* **Input FIFO**: a one-word head register. In use the 4-deep ``vpu_fifo``
  never holds more than one word (``u3_fifo_composed`` Part A), so a word
  pushed in ``t`` is the head in ``t + 1`` and ``input_ready_o`` is 1. A
  ``Stream`` consumed by ``try_get`` cannot be held to a cycle on the untimed
  simulator (the same record: ``try_*`` are scheduling-dependent), which is
  why the front reads the push ports itself.
* **Skew lines**: ``lhs_skew_data_q`` (payload, unreset) and
  ``lhs_skew_valid_q`` as arrays shifted every cycle; the commit skew and the
  bank skew as the RTL's shift registers. The per-PE bank bit is
  ``commit_bank_skew[r*PE + c]`` in the RTL, a pure delay line of
  ``loaded_bank_q``; here row ``r``'s west token carries
  ``commit_bank_skew[r*PE]`` and ``pe_unit`` forwards it east one register
  per PE, which is the same value at every cycle.
* **Output FIFOs**: per lane an explicit ring of ``ENTRIES`` 64-bit groups
  (U2's ``trace`` FIFO form) with ``vpu_fifo``'s rules: a push on full
  without a pop is DROPPED (``mxu.sv:44``), a pop while not valid is ignored.
  ``output_data_o`` is the head group of every lane on every cycle (the
  contract defines it whenever ``output_valid_o``), which a ``Stream`` cannot
  give without a peek; the Stream form is a separate probe.
* **Reset** is the ``rst_ni`` port, a bit in the front's west tokens and in
  the ``ctl`` token to the back; control clears, payload flows (D-14).
"""

from __future__ import annotations

from allo.compose import Channel, unit


def mxu_channels(pe_channels):
    return pe_channels + (Channel("ctl", "UInt(32)", "2", (), "front -> back: rst_n"),)


@unit(memories=("RST", "PUSH", "KIND", "DATA", "CMT", "RDY", "ACC"),
      writes=("lhsx", "wx", "ctl"), parameters=("N", "D", "PE", "SKEW", "SPAN"))
def mxu_front(rst: UInt(1)[N], push: UInt(1)[N], kind: UInt(1)[N], data: UInt(16)[N * D],
              cmt: UInt(1)[N], rdy: UInt(1)[N], acc: UInt(1)[N]):
    # the input FIFO as it is used: one word, visible the cycle after its push
    hv: UInt(1) = 0
    hkind: UInt(1) = 0
    hdata: UInt(16)[D] = 0
    # activation skew lines (payload unreset, valids reset); stage s of lane r at r*SKEW + s
    skd: UInt(16)[D * SKEW] = 0
    skv: UInt(1)[D * SKEW] = 0
    # weight_switch_skew_q[1:SKEW-1] (reset) and commit_bank_skew_q[1:SPAN] (payload)
    wss: UInt(1)[SKEW] = 0
    cbs: UInt(1)[SPAN + 1] = 0
    load_bank: UInt(1) = 0
    loaded_bank: UInt(1) = 1
    waiting: UInt(1) = 0
    for t in range(N):
        r_n: UInt(1) = rst[t]  # S6: every port read unconditional
        p: UInt(1) = push[t]
        k: UInt(1) = kind[t]
        cm: UInt(1) = cmt[t]
        # ---- outputs of cycle t (pre: from the state) ----
        rdy[t] = 1
        acc[t] = hv
        # ---- the combinational cycle ----
        consume: UInt(1) = hv
        tile_starts: UInt(1) = 0
        rhs_beat: UInt(1) = 0
        if consume == 1:
            if hkind == 0:
                if waiting == 1:
                    tile_starts = 1
            else:
                rhs_beat = 1
        wss[0] = tile_starts  # weight_switch_skew = {wss_q, tile_starts}
        cbs[0] = loaded_bank  # commit_bank_skew = {cbs_q, loaded_bank_q}
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
        # ---- the edge ----
        for s in range(SKEW - 1):  # shift the skew lines (payload every cycle)
            idx: int32 = SKEW - 1 - s
            with allo.meta_for(D) as lane:
                skd[lane * SKEW + idx] = skd[lane * SKEW + idx - 1]
                skv[lane * SKEW + idx] = skv[lane * SKEW + idx - 1]
        with allo.meta_for(D) as lane:
            skd[lane * SKEW] = hdata[lane]
            skv[lane * SKEW] = 0
            if consume == 1:
                if hkind == 0:
                    skv[lane * SKEW] = 1
        for s in range(SPAN):  # commit_bank_skew_q <= commit_bank_skew[SPAN-1:0]
            idx2: int32 = SPAN - s
            cbs[idx2] = cbs[idx2 - 1]
        for s in range(SKEW - 1):  # weight_switch_skew_q <= weight_switch_skew[SKEW-2:0]
            idx3: int32 = SKEW - 1 - s
            wss[idx3] = wss[idx3 - 1]
        if r_n == 0:
            load_bank = 0
            loaded_bank = 1
            waiting = 0
            for s in range(SKEW):
                wss[s] = 0
                with allo.meta_for(D) as lane:
                    skv[lane * SKEW + s] = 0
            hv = 0  # pointers cleared: a push under reset is lost
        else:
            if cm == 1:  # both toggle from the OLD load bank (non-blocking)
                lb_old: UInt(1) = load_bank
                load_bank = 1 - lb_old
                loaded_bank = lb_old
                waiting = 1
            elif tile_starts == 1:
                waiting = 0
            hv = p
            hkind = k
            with allo.meta_for(D) as lane:
                hdata[lane] = data[t * D + lane]


@unit(memories=("POP", "VLD", "ODATA"), reads=("px", "ctl"),
      parameters=("N", "D", "SUB", "ENTRIES"))
def mxu_back(pop: UInt(1)[N], vld: UInt(1)[N], odata: UInt(16)[N * D * SUB]):
    gidx: UInt(2)[D] = 0  # gather_index_q (reset)
    gq: UInt(64)[D] = 0  # gather_q (payload)
    mem: UInt(64)[D * ENTRIES] = 0  # the lane FIFOs' storage (unreset)
    rd: int32[D] = 0
    wr: int32[D] = 0
    cnt: int32[D] = 0
    for t in range(N):
        q: UInt(1) = pop[t]
        c_t: UInt(32) = ctl.get()
        r_n: UInt(1) = c_t[0]
        # ---- outputs of cycle t: valid = AND of lane non-empty, data = the heads ----
        valid: UInt(1) = 1
        with allo.meta_for(D) as lane:
            if cnt[lane] == 0:
                valid = 0
        vld[t] = valid
        with allo.meta_for(D) as lane:
            head: UInt(64) = mem[lane * ENTRIES + rd[lane]]
            with allo.meta_for(SUB) as sub:
                odata[t * (D * SUB) + sub * D + lane] = head[16 * sub:16 * (sub + 1)]
        consume: UInt(1) = q & valid
        # ---- the array's results of this cycle, gathered and packed ----
        with allo.meta_for(D) as lane:
            sp: UInt(32) = px[D, lane].get()
            res: UInt(24) = sp[0:24]
            rv: UInt(1) = sp[24]
            rounded: UInt(24) = res + 0x7F + res[8]  # pack_bf16, nearest even
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
            # the lane FIFO (vpu_fifo rules): pop, push; push on full without a pop is dropped
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
                    mem[lane * ENTRIES + wr[lane]] = gnext
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

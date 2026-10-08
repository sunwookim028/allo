# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track-D form of ``fetch.f1`` (U4 track A, plan F1) for Catapult.

The landed ``f1`` is unchanged in logic; two things about its ports keep it
from a per-cycle Connections boundary (U3 track C's C1 and C2):

* the 128-bit words are ``UInt(32)[n, 4]`` lane arrays (P-8), and a rank-2
  boundary array is RAM pins, read one word per cycle;
* ``wa[t]``/``wd[t]`` are read under ``if we[t]`` and ``raddr[t]`` under
  ``if flush[t]``: a conditional port access is RAM pins too.

Here every word port is one ``UInt(128)`` per row (the RTL's own port) and
every port is read once, unconditionally, at the top of the iteration. The
harness runner is ``fetch.run_f1`` unchanged: ``u4d_check.CatapultRtl`` packs
a ``[n, k]`` lane array into the one ``k x 32``-bit port and unpacks outputs.
The IRAM and the queue entries are 128-bit words instead of four lanes (the
same bits). Not runnable on the simulator (no numpy dtype above 64 bits, B6).
"""
import allo.dataflow as df
from allo.ir.types import UInt, int32, uint1

from examples.minitpu.units import fetch as F

W128 = UInt(128)


def make(n, w=128, inst="iram"):
    if inst == "iram":
        return _iram(n)
    return _fq(n, F.AW[inst])


def _iram(n):
    # IRAM rows: 4,096 as the RTL; U4D_IRAM_ROWS (a power of two) cuts it, the
    # address masked (exact on these traces: every write is below word 48 and a
    # read above the written words is uninit, masked). As a body array Catapult
    # maps it to ccs_ram_sync_1R1W and refuses II=1 (SCHD-30: read and write of
    # one RAM in one iteration); partitioned into registers, 4,096 x 128 is the
    # D-12 server's architect blow-up, so the register form is built at 256.
    import os
    ROWS = int(os.environ.get("U4D_IRAM_ROWS", "4096"))
    RM = ROWS - 1
    A = UInt(12)

    @df.region()
    def top(RST: uint1[n], WE: uint1[n], WA: A[n], WD: W128[n], PAUSE: uint1[n], FLUSH: uint1[n],
            RADDR: A[n], POP: uint1[n],
            DATA: W128[n], RDA: A[n], RDV: uint1[n], VALID: uint1[n], BADDR: A[n], EMPTY: uint1[n],
            FULL: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, WE, WA, WD, PAUSE, FLUSH, RADDR, POP,
                                      DATA, RDA, RDV, VALID, BADDR, EMPTY, FULL])
        def fetch(rst: uint1[n], we: uint1[n], wa: A[n], wd: W128[n], pause: uint1[n], flush: uint1[n],
                  raddr: A[n], pop_i: uint1[n],
                  data_o: W128[n], rda_o: A[n], rdv_o: uint1[n], valid_o: uint1[n], baddr_o: A[n],
                  empty_o: uint1[n], full_o: uint1[n]):
            iram: W128[ROWS] = 0
            rd_reg: W128 = 0  # sequencer_iram's read register
            fq_addr: A[4] = 0
            fq_data: W128[4] = 0
            next_addr: A = 0
            req_pending: uint1 = 0
            req_addr: A = 0
            count: UInt(3) = 0
            for t in range(n):
                r: uint1 = rst[t]
                e: uint1 = we[t]
                a_w: A = wa[t]
                d_w: W128 = wd[t]
                ps: uint1 = pause[t]
                fl: uint1 = flush[t]
                ra: A = raddr[t]
                pp: uint1 = pop_i[t]
                if r == 0:
                    next_addr = 0
                    req_pending = 0
                    req_addr = 0
                    count = 0
                empty: uint1 = count == 0
                rd_valid: uint1 = 0
                if ps == 0 and count < 3:  # room for the read in flight
                    rd_valid = 1
                data_o[t] = fq_data[0]
                rda_o[t] = next_addr
                rdv_o[t] = rd_valid
                valid_o[t] = 1 - empty
                baddr_o[t] = fq_addr[0]
                empty_o[t] = empty
                full_o[t] = count == 4
                # ---- rising edge: IRAM (never reset) ----
                bram: W128 = rd_reg
                ir: int32 = next_addr & RM
                rd_reg = iram[ir]
                if e:
                    iw: int32 = a_w & RM
                    iram[iw] = d_w
                # ---- rising edge: the fetch queue ----
                if r:
                    if fl:
                        next_addr = ra
                        req_pending = 0
                        count = 0
                    else:
                        push: uint1 = req_pending
                        pop: uint1 = pp & (1 - empty)
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

    return top


def _fq(n, aw):
    A = UInt(aw)

    @df.region()
    def top(RST: uint1[n], BRAM: W128[n], PAUSE: uint1[n], FLUSH: uint1[n], RADDR: A[n], POP: uint1[n],
            DATA: W128[n], RDA: A[n], RDV: uint1[n], VALID: uint1[n], BADDR: A[n], EMPTY: uint1[n],
            FULL: uint1[n]):
        @df.kernel(mapping=[1], args=[RST, BRAM, PAUSE, FLUSH, RADDR, POP,
                                      DATA, RDA, RDV, VALID, BADDR, EMPTY, FULL])
        def fq(rst: uint1[n], bram_i: W128[n], pause: uint1[n], flush: uint1[n], raddr: A[n],
               pop_i: uint1[n],
               data_o: W128[n], rda_o: A[n], rdv_o: uint1[n], valid_o: uint1[n], baddr_o: A[n],
               empty_o: uint1[n], full_o: uint1[n]):
            fq_addr: A[4] = 0
            fq_data: W128[4] = 0
            next_addr: A = 0
            req_pending: uint1 = 0
            req_addr: A = 0
            count: UInt(3) = 0
            for t in range(n):
                r: uint1 = rst[t]
                bram: W128 = bram_i[t]
                ps: uint1 = pause[t]
                fl: uint1 = flush[t]
                ra: A = raddr[t]
                pp: uint1 = pop_i[t]
                if r == 0:
                    next_addr = 0
                    req_pending = 0
                    req_addr = 0
                    count = 0
                empty: uint1 = count == 0
                rd_valid: uint1 = 0
                if ps == 0 and count < 3:
                    rd_valid = 1
                data_o[t] = fq_data[0]
                rda_o[t] = next_addr
                rdv_o[t] = rd_valid
                valid_o[t] = 1 - empty
                baddr_o[t] = fq_addr[0]
                empty_o[t] = empty
                full_o[t] = count == 4
                if r:
                    if fl:
                        next_addr = ra
                        req_pending = 0
                        count = 0
                    else:
                        push: uint1 = req_pending
                        pop: uint1 = pp & (1 - empty)
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

    return top


run = F.run_f1

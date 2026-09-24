# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Allo dataflow model of MiniTPU's compute core.

Component-for-component with the RTL at ``~/core/minitpu/src/core``, not
cycle-for-cycle.  Every kernel below is one named piece of that machine:

===========================  ====================================================
kernel                       RTL
===========================  ====================================================
``sequencer``                ``sequencer/sequencer.sv`` -- in-order issue, no
                             interlocks
``stream_engine``            ``mxu/mxu_stream_engine.sv`` -- ONE engine shared by
                             ``vmatload`` and ``vmatpush``; this is the
                             serialization point
``pop_engine``               ``mxu/mxu_pop_engine.sv`` -- a SECOND, independent
                             engine, so a pop runs while a load or push streams
``port_c``                   VREG read port C (``vpu.sv:429-431``) -- one physical
                             read shared by the matrix stream engine and the
                             store path, matrix wins, no interlock
``write_port``               the single VREG write port (``vpu.sv:289-305``), an
                             OR-mux over every writer with no arbiter
``dispatch``                 the MXU west/north edge (``mxu.sv``) -- demuxes the
                             shared input FIFO by kind and arms the weight switch
``pe``                       ``mxu/mxu_pe.sv`` x DIM x DIM, weight-stationary,
                             two pending weight banks, 24-bit accumulator
``edge``                     ``pack_bf16`` (``mxu.sv:82-94``) and the per-lane
                             output FIFO gather (``mxu.sv:129-157``)
``alu``                      ``vpu/vpu_alu.sv`` -- BF16, where K > 16 is summed
``vmem_compute``             VMEM's compute port (``vpu_word_array.sv``)
``vmem_dma``                 VMEM's DMA port -- the other of the two ports
===========================  ====================================================

Read ``README.md`` first; it lists where this model diverges from the machine.
"""

import numpy as np
import ml_dtypes

import allo
import allo.dataflow as df
from allo.ir.types import bfloat16, float32, int32, int1, uint32, Stateful, Stream

bf16 = ml_dtypes.bfloat16

# ---------------------------------------------------------------------------
# Geometry, all from ~/core/minitpu/src/core/vpu/vpu_pkg.sv
# ---------------------------------------------------------------------------
DIM = 16  # NUM_LANES        (vpu_pkg.sv:25)  -- array edge, lanes per VREG row
SUB = 4  # NUM_SUBLANES      (vpu_pkg.sv:28)  -- rows per VREG
NVREG = 32  # NUM_VREGS      (vpu_pkg.sv:34)
WEIGHT_BANKS = 2  # MXU_WEIGHT_BANKS (vpu_pkg.sv:108)
FIFO_ENTRIES = 16  # OUTPUT_GROUP_ENTRIES (mxu.sv:116) -- per lane, groups of SUB
MXU_FIFO_DEPTH = 4  # MXU_FIFO_DEPTH (vpu_pkg.sv:51) -- the shared input FIFO
# VMEM is 4096 words of 128 B on the machine (vpu_pkg.sv:89); the model keeps
# the structure and shrinks the capacity, because a simulated Stateful of
# 512 KiB buys nothing.
VMEM_WORDS = 64

# Opcodes of the model's command tape.  MiniTPU packs these into a 128-bit
# bundle (sequencer_pkg.sv:384-393); the model carries one command per entry
# and does not model slots, the shared immediate, DELAY, or the loop stack.
OP_NOP = 0
OP_VLD = 1  # MEM-LDST: VMEM word -> VREG            (w = 6 on the machine)
OP_VST = 2  # MEM-LDST: VREG -> VMEM word, reads port C
OP_MATLOAD = 3  # M: 16 weight rows from 4 VREGs, port C, reverse order
OP_MATPUSH = 4  # M: SUB token rows from 1 VREG, port C
OP_MATPOP = 5  # M: one whole VREG from the output FIFOs, in one write
OP_VADD = 6  # V: BF16 add, ports A and B -> write port
OP_VMOV = 7  # V: BF16 move, port A -> write port
OP_FENCE = 8  # not a MiniTPU instruction; see README, divergence 6


def acc24_np(x):
    """Host-side acc24 rounding, used to check the model. See reference.py."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(8)) & np.uint32(1)
    r = (u + np.uint32(127) + lsb) & np.uint32(0xFFFFFF00)
    return r.view(np.float32)


def round24(x: float32) -> float32:
    """acc24: round a float32 to 15 fraction bits, round-to-nearest-even.

    ``mxu_acc24_add_pipe.sv`` rounds every add to ``MXU_ACC_FRAC_W = 15``
    (``vpu_pkg.sv:100``); add-half-plus-lsb is the RTL's own rule, and
    ``pack_bf16`` (``mxu.sv:89``) applies the same rule eight bits lower.
    """
    u: uint32 = x.bitcast()
    lsb: uint32 = (u >> 8) & 1
    r: uint32 = u + 127 + lsb
    t: uint32 = r & 4294967040
    y: float32 = t.bitcast()
    return y


def build(prog, m_rows, dim=DIM, sub=SUB, vmem_words=VMEM_WORDS, target="simulator"):
    """Elaborate the model for one command tape.

    ``prog`` is a list of ``(op, a, b, c)`` tuples from ``program.py``.
    ``m_rows`` is the number of token rows pushed against each weight tile, so
    every structural trip count below is a build-time constant -- which is what
    an HLS design generator gets to assume and a CPU does not.
    """
    n_load = sum(1 for p in prog if p[0] == OP_MATLOAD)
    n_push = sum(1 for p in prog if p[0] == OP_MATPUSH)
    n_pop = sum(1 for p in prog if p[0] == OP_MATPOP)
    n_ld = sum(1 for p in prog if p[0] == OP_VLD)
    n_st = sum(1 for p in prog if p[0] == OP_VST)
    n_alu = sum(1 for p in prog if p[0] in (OP_VADD, OP_VMOV))
    n_cmd = len(prog)

    assert n_load > 0 and n_push % (m_rows // sub) == 0
    n_tile = n_load
    acts_per_tile = n_push // n_tile * sub
    assert acts_per_tile == m_rows
    # Port C serves the matrix stream engine (dim beats per load, sub per push)
    # and the store path (sub beats per vst).  One physical port, so one count.
    n_portc = n_load * dim + n_push * sub + n_st * sub
    # The single VREG write port: every vld, every pop, every V-slot result.
    n_write = n_ld + n_pop + n_alu
    # Groups of `sub` results leaving each lane of the array.
    n_group = n_tile * (m_rows // sub)

    prog_np = np.zeros((max(n_cmd, 1), 4), dtype=np.int32)
    for i, (op, a, b, c) in enumerate(prog):
        prog_np[i] = (op, a, b, c)

    TW = bfloat16  # the architectural element type: BF16 everywhere
    TA = float32  # the 24-bit accumulator's carrier; rounded by round24()

    @df.region()
    def minitpu(
        prog_in: int32[max(n_cmd, 1), 4],
        dram_in: TW[vmem_words, sub, dim],
        dram_out: TW[vmem_words, sub, dim],
    ):
        # --- architectural state -------------------------------------------
        # 32 VREGs of SUB x DIM BF16 (vpu_pkg.sv:114, vreg_word_t).
        vregs: TW[NVREG, sub, dim] @ Stateful = 0.0
        # One flat word array, whole words only (vpu_word_array.sv:5).
        vmem: TW[vmem_words, sub, dim] @ Stateful = 0.0

        # --- issue ----------------------------------------------------------
        c_stream: Stream[int32, 4]  # to the shared vmatload/vmatpush engine
        c_pop: Stream[int32, 4]  # to the independent pop engine
        c_mem: Stream[int32, 4]  # to the VMEM compute port
        c_alu: Stream[int32, 4]  # to the vector ALU
        wb_tok: Stream[int32, n_write + 4]  # write-port retirement (model only)
        st_tok: Stream[int32, n_st + 4]  # vst retirement (model only)
        fill_done: Stream[int32, 1]  # the DMA channel fence, wait.channel

        # --- VREG ports -------------------------------------------------------
        pc_req_m: Stream[int32, 2]  # port C request, matrix stream engine
        pc_rsp_m: Stream[TW[dim], 2]
        pc_req_s: Stream[int32, 2]  # port C request, store path
        pc_rsp_s: Stream[TW[dim], 2]
        wp_req_p: Stream[int32, 2]  # write port, from the pop engine
        wp_dat_p: Stream[TW[sub, dim], 2]
        wp_req_l: Stream[int32, 2]  # write port, from a vld
        wp_dat_l: Stream[TW[sub, dim], 2]
        wp_req_a: Stream[int32, 2]  # write port, from the V slot
        wp_dat_a: Stream[TW[sub, dim], 2]

        # --- MXU --------------------------------------------------------------
        # The one shared input FIFO, 257 bits wide on the machine (mxu.sv:37):
        # a kind bit plus DIM BF16 values, depth MXU_FIFO_DEPTH.
        mxu_kind: Stream[int32, MXU_FIFO_DEPTH]
        mxu_beat: Stream[TW[dim], MXU_FIFO_DEPTH]
        lhs_v: Stream[TW, 2][dim, dim]  # activations west -> east
        lhs_t: Stream[int32, 2][dim, dim]  # the weight-commit tag, riding east
        wt_v: Stream[TW, 2][dim, dim]  # weights north -> south
        wt_b: Stream[int32, 2][dim, dim]  # its bank tag
        psum: Stream[TA, 2][dim, dim]  # 24-bit partial sums north -> south
        ofifo: Stream[TW[sub], FIFO_ENTRIES][dim]  # per-lane output FIFO

        # ==================================================================
        # sequencer -- in-order issue.  Nothing here stalls on a hazard; the
        # only back pressure is a full engine command queue, which is what
        # "issue stalls only on delay" comes to without a cycle model.
        # ==================================================================
        @df.kernel(mapping=[1], args=[prog_in])
        def sequencer(p: int32[max(n_cmd, 1), 4]):
            for i in range(n_cmd):
                op: int32 = p[i, 0]
                a: int32 = p[i, 1]
                b: int32 = p[i, 2]
                c: int32 = p[i, 3]
                if op == OP_MATLOAD:
                    c_stream.put(a)
                elif op == OP_MATPUSH:
                    c_stream.put(a + 256)
                elif op == OP_MATPOP:
                    c_pop.put(a)
                elif op == OP_VLD:
                    c_mem.put(a * 4096 + b)
                elif op == OP_VST:
                    c_mem.put(a * 4096 + b + 2097152)
                elif op == OP_VADD:
                    c_alu.put(a * 4096 + b * 64 + c)
                elif op == OP_VMOV:
                    c_alu.put(a * 4096 + b * 64 + c + 2097152)
                elif op == OP_FENCE:
                    # Model-only.  MiniTPU has no scoreboard: its assembler
                    # proves the schedule instead.  See README, divergence 6.
                    nf: int32 = 0
                    while nf < a:
                        wb_tok.get()
                        nf = nf + 1

        # ==================================================================
        # VREG read port C.  ONE physical read (vpu.sv:429-431) that the
        # matrix stream engine and the store path both want.  The mux is
        # fixed priority and the matrix wins; nothing stalls and nothing
        # interlocks, so a program that lets them collide is simply wrong.
        # ==================================================================
        @df.kernel(mapping=[1])
        def port_c():
            for ci in range(n_portc):
                served: int32 = 0
                while served == 0:
                    rq: int32 = 0
                    ok: int1 = 0
                    rq, ok = pc_req_m.try_get()  # matrix first: it wins
                    if ok:
                        rv: int32 = rq // sub
                        rs: int32 = rq - rv * sub
                        word: TW[dim] = 0.0
                        for j in range(dim):
                            word[j] = vregs[rv, rs, j]
                        pc_rsp_m.put(word)
                        served = 1
                    else:
                        rq, ok = pc_req_s.try_get()
                        if ok:
                            rv2: int32 = rq // sub
                            rs2: int32 = rq - rv2 * sub
                            word2: TW[dim] = 0.0
                            for j in range(dim):
                                word2[j] = vregs[rv2, rs2, j]
                            pc_rsp_s.put(word2)
                            served = 1

        # ==================================================================
        # The single VREG write port (vpu.sv:289-305).  Six sources OR
        # together in the RTL with a one-hot assertion; the model serves them
        # one at a time, which is the same contract without the garbage.
        # ==================================================================
        @df.kernel(mapping=[1])
        def write_port():
            for wri in range(n_write):
                done: int32 = 0
                while done == 0:
                    idx: int32 = 0
                    ok: int1 = 0
                    idx, ok = wp_req_p.try_get()
                    if ok:
                        wd: TW[sub, dim] = wp_dat_p.get()
                        for s in range(sub):
                            for j in range(dim):
                                vregs[idx, s, j] = wd[s, j]
                        done = 1
                    else:
                        idx, ok = wp_req_l.try_get()
                        if ok:
                            wd2: TW[sub, dim] = wp_dat_l.get()
                            for s in range(sub):
                                for j in range(dim):
                                    vregs[idx, s, j] = wd2[s, j]
                            done = 1
                        else:
                            idx, ok = wp_req_a.try_get()
                            if ok:
                                wd3: TW[sub, dim] = wp_dat_a.get()
                                for s in range(sub):
                                    for j in range(dim):
                                        vregs[idx, s, j] = wd3[s, j]
                                done = 1
                wb_tok.put(1)

        # ==================================================================
        # The shared matrix stream engine (mxu_stream_engine.sv).  vmatload
        # and vmatpush are ONE state machine over ONE read port and ONE input
        # FIFO, so they can never overlap.  This is the serialization the
        # design is worth modelling for.
        # ==================================================================
        @df.kernel(mapping=[1])
        def stream_engine():
            for si in range(n_load + n_push):
                cmd: int32 = c_stream.get()
                if cmd < 256:
                    # vmatload base: DIM weight rows out of four VREGs,
                    # one row a cycle FROM THE LAST (mxu_stream_engine.sv:79).
                    for r in range(dim):
                        row: int32 = dim - 1 - r
                        rw: int32 = row // sub
                        pc_req_m.put((cmd + rw) * sub + (row - rw * sub))
                        beat: TW[dim] = pc_rsp_m.get()
                        mxu_kind.put(0)
                        mxu_beat.put(beat)
                else:
                    # vmatpush vs: SUB token rows out of one VREG, ascending.
                    vs: int32 = cmd - 256
                    for s in range(sub):
                        pc_req_m.put(vs * sub + s)
                        beat2: TW[dim] = pc_rsp_m.get()
                        mxu_kind.put(1)
                        mxu_beat.put(beat2)

        # ==================================================================
        # The MXU edge (mxu.sv).  Drains the shared input FIFO, sends a
        # weight beat down the columns and a token row across the rows, and
        # arms the weight switch: the RTL commits not when the load ends but
        # on the NEXT activation (`tile_starts`, mxu.sv:170), so the commit
        # rides the activation beat here too.
        # ==================================================================
        @df.kernel(mapping=[1])
        def dispatch():
            load_bank: int32 = 0
            pending: int32 = 0
            waiting: int32 = 0
            wrows: int32 = 0
            for di in range(n_load * dim + n_push * sub):
                kind: int32 = mxu_kind.get()
                beat: TW[dim] = mxu_beat.get()
                if kind == 0:
                    with allo.meta_for(dim) as c:
                        wt_v[0, c].put(beat[c])
                        wt_b[0, c].put(load_bank)
                    wrows = wrows + 1
                    if wrows == dim:
                        wrows = 0
                        waiting = 1
                        pending = load_bank
                        load_bank = 1 - load_bank
                else:
                    tag: int32 = 0
                    if waiting == 1:
                        tag = pending + 1
                        waiting = 0
                    with allo.meta_for(dim) as r:
                        lhs_v[r, 0].put(beat[r])
                        lhs_t[r, 0].put(tag)

        # ==================================================================
        # The PE array (mxu_pe.sv x 256).  Weight-stationary, one active
        # weight and WEIGHT_BANKS pending banks that double as the vertical
        # shift chain; 24-bit accumulator; row 0's psum_in is tied to zero
        # (mxu_systolic_array.sv:54) and the adder's zero bypass returns the
        # product bit-for-bit, so the first term of every column is exact.
        # ==================================================================
        @df.kernel(mapping=[dim, dim])
        def pe():
            r, c = df.get_pid()
            wpend: TW[WEIGHT_BANKS] = 0.0
            wact: TW = 0.0
            for ti in range(n_tile):
                # --- the weight shift: one row of the tile per beat ---
                for wi in range(dim):
                    w: TW = wt_v[r, c].get()
                    bnk: int32 = wt_b[r, c].get()
                    with allo.meta_if(r < dim - 1):
                        wt_v[r + 1, c].put(wpend[bnk])
                        wt_b[r + 1, c].put(bnk)
                    wpend[bnk] = w
                # --- the token rows against the committed tile ---
                for ai in range(acts_per_tile):
                    a: TW = lhs_v[r, c].get()
                    tg: int32 = lhs_t[r, c].get()
                    if tg > 0:
                        wact = wpend[tg - 1]
                    av: float32 = a
                    wv: float32 = wact
                    prod: float32 = av * wv
                    with allo.meta_if(r == 0):
                        # psum_in is zero and the adder bypasses: exact.
                        psum[r, c].put(prod)
                    with allo.meta_else():
                        pin: float32 = psum[r - 1, c].get()
                        # acc24: every add rounds to 15 fraction bits, RNE.
                        # Inlined rather than called: a shared helper would be
                        # emitted once per PE and redefine its own symbol.
                        acc: float32 = pin + prod
                        ub: uint32 = acc.bitcast()
                        lsb: uint32 = (ub >> 8) & 1
                        rb: uint32 = ub + 127 + lsb
                        tb: uint32 = rb & 4294967040
                        rounded: float32 = tb.bitcast()
                        psum[r, c].put(rounded)
                    with allo.meta_if(c < dim - 1):
                        lhs_v[r, c + 1].put(a)
                        lhs_t[r, c + 1].put(tg)

        # ==================================================================
        # The array's bottom edge (mxu.sv:129-157).  ONE pack_bf16 per lane
        # -- the only place a 24-bit partial sum becomes BF16 -- then a
        # gather of SUB results into one output-FIFO entry.
        # ==================================================================
        @df.kernel(mapping=[dim])
        def edge():
            c = df.get_pid()
            for gi in range(n_group):
                grp: TW[sub] = 0.0
                for s in range(sub):
                    v: float32 = psum[dim - 1, c].get()
                    grp[s] = v  # pack_bf16: the single rounding point
                ofifo[c].put(grp)

        # ==================================================================
        # The pop engine (mxu_pop_engine.sv).  A SECOND engine: it shares no
        # resource with the stream engine, so a pop runs while a load or push
        # streams.  All DIM lane FIFOs move in lockstep (mxu.sv:109) and one
        # pop writes one whole VREG in one beat (MXU_POP_BEATS = 1).
        # ==================================================================
        @df.kernel(mapping=[1])
        def pop_engine():
            for pi in range(n_pop):
                vd: int32 = c_pop.get()
                word: TW[sub, dim] = 0.0
                with allo.meta_for(dim) as c:
                    grp: TW[sub] = ofifo[c].get()
                    for s in range(sub):
                        word[s, c] = grp[s]
                wp_req_p.put(vd)
                wp_dat_p.put(word)

        # ==================================================================
        # The vector ALU (vpu_alu.sv).  BF16, 64 elements wide.  This is
        # where a contraction deeper than 16 is summed: each 16-deep tile
        # comes out of the MXU already rounded to BF16, and `vadd` combines
        # them in BF16 (ARITHMETIC.md sec. 2).
        # ==================================================================
        @df.kernel(mapping=[1])
        def alu():
            for ei in range(n_alu):
                cmd: int32 = c_alu.get()
                is_mov: int32 = cmd // 2097152
                rest: int32 = cmd % 2097152
                vd: int32 = rest // 4096
                va: int32 = (rest % 4096) // 64
                vb: int32 = rest % 64
                out: TW[sub, dim] = 0.0
                for s in range(sub):
                    for j in range(dim):
                        if is_mov == 1:
                            out[s, j] = vregs[va, s, j]  # port A only
                        else:
                            out[s, j] = vregs[va, s, j] + vregs[vb, s, j]
                wp_req_a.put(vd)
                wp_dat_a.put(out)

        # ==================================================================
        # VMEM's compute port (vpu_word_array.sv, port A).  Whole words only,
        # no byte enables, no arbiter.  A vst reads its data through VREG
        # port C, which is why the store path appears there and not here.
        # ==================================================================
        @df.kernel(mapping=[1])
        def vmem_compute():
            fill_done.get()  # wait.channel(mask): the DMA fill has landed
            for i in range(n_ld + n_st):
                cmd: int32 = c_mem.get()
                is_st: int32 = cmd // 2097152
                rest: int32 = cmd % 2097152
                vreg: int32 = rest // 4096
                word: int32 = rest % 4096
                if is_st == 1:
                    for s in range(sub):
                        pc_req_s.put(vreg * sub + s)
                        line: TW[dim] = pc_rsp_s.get()
                        for j in range(dim):
                            vmem[word, s, j] = line[j]
                    st_tok.put(1)
                else:
                    ld: TW[sub, dim] = 0.0
                    for s in range(sub):
                        for j in range(dim):
                            ld[s, j] = vmem[word, s, j]
                    wp_req_l.put(vreg)
                    wp_dat_l.put(ld)

        # ==================================================================
        # VMEM's DMA port (vpu_word_array.sv, port B).  The second of the two
        # ports; on the machine it moves 32 B beats that vpu_dma_group gathers
        # into whole words.  Here it fills VMEM before the program runs and
        # drains it after -- `vmemld` then `wait.channel` then `vmemst`.
        # ==================================================================
        @df.kernel(mapping=[1], args=[dram_in, dram_out])
        def vmem_dma(din: TW[vmem_words, sub, dim], dout: TW[vmem_words, sub, dim]):
            for w in range(vmem_words):
                for s in range(sub):
                    for j in range(dim):
                        vmem[w, s, j] = din[w, s, j]
            fill_done.put(1)
            for i in range(n_st):
                st_tok.get()
            for w in range(vmem_words):
                for s in range(sub):
                    for j in range(dim):
                        dout[w, s, j] = vmem[w, s, j]

    return df.build(minitpu, target=target), prog_np

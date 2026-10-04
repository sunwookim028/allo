# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: references for MiniTPU's matrix unit -- ``mxu_pe``,
``mxu_systolic_array`` and ``mxu`` -- as trace references (``ref.py``, U2).

Two kinds of reference, on purpose:

* **Cycle models** for ``mxu_pe`` and ``mxu_systolic_array``: every register
  of ``mxu_pe.sv`` written out (the array is a grid of them, wired as
  ``mxu_systolic_array.sv`` wires them), stepped once per cycle. Payload
  registers that MiniTPU never resets (``lhs_o``, ``product_q``, ``psum_q``,
  the adder's payload stages) carry a *taint*: ``uninit`` until a known value
  reaches them, and ``reset fill`` for an adder result whose special-class
  stage was forced by reset while its payload kept flowing. A tainted slot is
  masked, not trusted.
* **A contract model** for ``mxu``: no register of ``mxu.sv`` is modelled.
  It states what a client may rely on -- the push / commit / pop contract --
  and the one timing constant that contract needs:

  - **push**: a word pushed in cycle ``p`` is accepted in cycle ``p + 1``
    (``input_accept_o``; the input FIFO drains every cycle). An RHS word
    shifts into the pending weight bank ``load_bank``: row 0 takes it, row
    ``r`` takes row ``r - 1``'s, so the first of ``DIM`` beats ends in the
    bottom row (``vmatload`` streams rows in reverse).
  - **commit**: ``weight_commit_i`` toggles ``load_bank`` and makes the bank
    just loaded the one the *next* tile uses; the next LHS word starts that
    tile, and from it every activation is multiplied by that bank as it was
    when the tile started.
  - **result**: activation row ``a`` (lane ``r`` -> array row ``r``) gives,
    per lane ``c``, ``pack_bf16(psum)`` with ``psum`` the acc24 chain
    ``psum_r = acc24_add(a_r * W[r][c], psum_{r-1})``, ``psum_{-1} = +0``,
    in ascending ``r`` (``ARITHMETIC.md`` §8). Results gather per lane in
    groups of ``NUM_SUBLANES`` consecutive LHS words; a group of lane ``c``
    enters that lane's output FIFO ``PUSH_TO_LANE + c`` cycles after its last
    word was pushed, and ``output_valid_o`` is the AND of every lane's
    non-empty. A pop (``output_pop_i`` while valid) removes one group from
    every lane. A lane FIFO that is full drops the group (``mxu.sv:44``).
  - **switch rule**: an RHS word into the bank a started tile holds, less
    than ``SPAN`` cycles after that tile started, breaks the tile (the RTL
    asserts, ``mxu.sv:259``); the reference masks that tile's results with
    the reason ``bank overwrite``.

  The timing constant is derived, not fitted: a word accepted in cycle
  ``c0`` enters array row ``r`` in ``c0 + 1 + r*PE``, PE (r, c) sees it
  ``c`` cycles later, and the bottom row's partial sum leaves
  ``DIM * PE`` cycles after it entered row 0, so lane ``c``'s result is
  pushed into its FIFO in cycle ``c0 + 1 + DIM*PE + c`` and is visible from
  the next: ``push -> output_valid = 2 + DIM*(PE + 1)`` for the last lane,
  with ``PE = MXU_PE_LATENCY = 4``. At ``DIM = 16`` that is 82 edges, and
  ``isa_latency.json``'s ``vmatpush`` result latency 85 = 4 (the stream
  engine's 4 rows from ``t + 1``) + 82 - 1 (a ``vmatpop`` issued in ``t``
  fires in ``t + 1``). ``characterize`` holds every ``output_valid_o`` slot
  to it, and a step probe measures it separately.
"""

import numpy as np

from examples.minitpu.harness import ref

PE_LATENCY = 4  # vpu_pkg.sv MXU_PE_LATENCY = 1 + MXU_ACC_ADD_LATENCY
BANKS = 2  # vpu_pkg.sv MXU_WEIGHT_BANKS
NUM_SUBLANES = 4  # vpu_pkg.sv default
ACC_W = 24

OK, UNINIT, FILL = 0, 1, 2
_REASON = {OK: "", UNINIT: "uninit", FILL: "reset fill"}


def pack_bf16(v):
    """``mxu.sv`` ``pack_bf16``: acc24 -> bf16, round to nearest even, by
    the integer trick (``+ 0x7F + bit 8``, keep ``[23:8]``)."""
    v = np.asarray(v, dtype=np.int64)
    return (((v + 0x7F + ((v >> 8) & 1)) & 0xFFFFFF) >> 8).astype(np.int64)


def fields(col, width, count):
    """``uint64[n, nw]`` packed port -> ``int64[n, count]`` of ``width``-bit
    fields, element ``i`` at bits ``[width*i +: width]`` (SV packed order)."""
    col = np.asarray(col, dtype=np.uint64).reshape(len(col), -1)
    out = np.zeros((len(col), count), dtype=np.int64)
    m = (1 << width) - 1
    if col.shape[1] == 1 and width * count <= 64:
        x = col[:, 0]
        for i in range(count):
            out[:, i] = ((x >> np.uint64(width * i)) & np.uint64(m)).astype(np.int64)
        return out
    for t, row in enumerate(col):
        v = sum(int(w) << (64 * j) for j, w in enumerate(row))
        for i in range(count):
            out[t, i] = (v >> (width * i)) & m
    return out


def join(vals, width):
    """``int[n, count]`` -> list of Python ints, field ``i`` at ``width*i``."""
    vals = np.asarray(vals, dtype=np.int64)
    return [sum(int(x) << (width * i) for i, x in enumerate(row)) for row in vals]


def _words(ints, width):
    from examples.minitpu.harness import rtl

    return rtl.pack(ints, width)


class PeGrid:
    """``R x C`` ``mxu_pe`` instances, stepped one cycle at a time.

    State is per PE (arrays shaped ``(R, C)``); ``taint`` arrays hold OK /
    UNINIT / FILL for the unreset payload. ``step`` takes every PE's inputs
    for the cycle and applies the rising edge.
    """

    def __init__(self, R, C):
        z = lambda: np.zeros((R, C), dtype=np.int64)
        self.R, self.C = R, C
        # reset registers: unknown before the first reset edge (masked by the
        # trace references, which start every trace in reset)
        self.active, self.pending = z(), np.zeros((BANKS, R, C), dtype=np.int64)
        self.lhs_valid_q, self.product_valid_q, self.commit_q = z(), z(), z()
        self.v1, self.v2, self.vout = z(), z(), z()
        self.result = z()
        self.result_t = np.full((R, C), UNINIT)
        # payload registers, never reset
        self.lhs_q, self.lhs_t = z(), np.full((R, C), UNINIT)
        self.product_q, self.product_t = z(), np.full((R, C), UNINIT)
        self.psum_q, self.psum_t = z(), np.full((R, C), UNINIT)
        self.s1a, self.s1b, self.s1_t = z(), z(), np.full((R, C), UNINIT)
        self.s2a, self.s2b, self.s2_t = z(), z(), np.full((R, C), UNINIT)
        self.active_t = np.full((R, C), UNINIT)

    def step(self, rst, lhs, lhs_t, lhs_valid, commit, commit_bank, weight, weight_valid,
             psum, psum_t, psum_valid):
        """One cycle. ``weight``/``weight_valid`` are ``(BANKS, R, C)``."""
        # the adder (mxu_acc24_add_pipe): result from stage 2, stage 2 from 1,
        # stage 1 from the PE's product_q / psum_q; special class forced by reset
        if rst:
            nres, nres_t = np.zeros_like(self.result), np.full_like(self.result_t, OK)
        else:
            nres = ref.mxu_acc24_add(self.s2a, self.s2b).astype(np.int64)
            nres_t = self.s2_t.copy()
        ns2a, ns2b = self.s1a, self.s1b
        ns2_t = np.maximum(self.s1_t, FILL) if rst else self.s1_t
        ns1a, ns1b = self.product_q, self.psum_q
        ns1_t = np.maximum(np.maximum(self.product_t, self.psum_t), FILL if rst else OK)
        nv = (0 * self.vout, 0 * self.v2, 0 * self.v1) if rst else (self.v2, self.v1, self.product_valid_q)
        # payload registers, every cycle
        nprod = ref.mxu_bf16_mul_acc24(lhs, self.active).astype(np.int64)
        nprod_t = np.maximum(lhs_t, self.active_t)
        # reset registers
        if rst:
            nact, npend = 0 * self.active, 0 * self.pending
            nact_t = np.full_like(self.active_t, OK)
            nlv, npv, ncq = 0 * self.lhs_valid_q, 0 * self.product_valid_q, 0 * self.commit_q
        else:
            sel = np.take_along_axis(self.pending, commit_bank[None].astype(np.int64), 0)[0]
            nact = np.where(commit != 0, sel, self.active)
            nact_t = np.where(commit != 0, OK, self.active_t)
            npend = np.where(weight_valid != 0, weight, self.pending)
            nlv, npv, ncq = lhs_valid, lhs_valid & psum_valid, commit
        self.result, self.result_t = nres, nres_t
        self.s2a, self.s2b, self.s2_t = ns2a, ns2b, ns2_t
        self.s1a, self.s1b, self.s1_t = ns1a, ns1b, ns1_t
        self.vout, self.v2, self.v1 = nv
        self.product_q, self.product_t = nprod, nprod_t
        self.lhs_q, self.lhs_t = np.array(lhs, dtype=np.int64), np.array(lhs_t)
        self.psum_q, self.psum_t = np.array(psum, dtype=np.int64), np.array(psum_t)
        self.active, self.active_t, self.pending = nact, nact_t, npend
        self.lhs_valid_q, self.product_valid_q, self.commit_q = nlv, npv, ncq


def _bit(cmd, p):
    return np.asarray(cmd[p], dtype=np.uint64).reshape(len(cmd[p]), -1)[:, 0].astype(np.int64)


def mxu_pe_trace(cmd):
    """``mxu_pe.sv`` as a cycle model; every output ``"post"`` (registered).

    ``psum_o`` after edge ``t + 1`` is ``acc24_add(product(lhs, active),
    psum_i)`` of cycle ``t - 3`` (``MXU_PE_LATENCY`` = 4 edges); a commit in
    cycle ``t`` makes ``active`` the pending bank it names from cycle
    ``t + 1``; ``weight_o`` is the pending bank registers themselves.
    """
    n = len(cmd["rst_ni"])
    rst = _bit(cmd, "rst_ni") == 0
    g = PeGrid(1, 1)
    lhs = _bit(cmd, "lhs_i")
    w = fields(cmd["weight_i"], 16, BANKS)
    wv = fields(cmd["weight_valid_i"], 1, BANKS)
    psum = _bit(cmd, "psum_i")
    cols = {k: np.zeros(n, dtype=np.int64) for k in
            ("weight_commit_o", "lhs_o", "lhs_valid_o", "weight_o", "psum_o", "psum_valid_o")}
    why = {k: np.full(n, "", dtype=object) for k in cols}
    started = False
    one = lambda x: np.array([[x]], dtype=np.int64)
    for t in range(n):
        started |= bool(rst[t])
        g.step(bool(rst[t]), one(lhs[t]), np.full((1, 1), OK), one(_bit(cmd, "lhs_valid_i")[t]),
               one(_bit(cmd, "weight_commit_i")[t]), one(_bit(cmd, "weight_commit_bank_i")[t]),
               w[t].reshape(BANKS, 1, 1), wv[t].reshape(BANKS, 1, 1),
               one(psum[t]), np.full((1, 1), OK), one(_bit(cmd, "psum_valid_i")[t]))
        cols["weight_commit_o"][t] = g.commit_q[0, 0]
        cols["lhs_o"][t] = g.lhs_q[0, 0]
        cols["lhs_valid_o"][t] = g.lhs_valid_q[0, 0]
        cols["weight_o"][t] = g.pending[0, 0, 0] | (g.pending[1, 0, 0] << 16)
        cols["psum_o"][t] = g.result[0, 0]
        cols["psum_valid_o"][t] = g.vout[0, 0]
        why["lhs_o"][t] = _REASON[int(g.lhs_t[0, 0])]
        why["psum_o"][t] = _REASON[int(g.result_t[0, 0])]
        if not started:
            for k in why:
                why[k][t] = "before reset"
    width = {"weight_o": 32, "psum_o": 24}
    resp = {k: _words(v.tolist(), width.get(k, 16 if k == "lhs_o" else 1)) for k, v in cols.items()}
    return resp, why, {}


def mxu_systolic_array_trace(cmd, dim):
    """``mxu_systolic_array.sv`` as a grid of ``mxu_pe`` cycle models, wired as
    the RTL wires them; outputs ``result_o``/``result_valid_o`` are the bottom
    row's ``psum_o``/``psum_valid_o`` (``"post"``)."""
    n = len(cmd["rst_ni"])
    rst = _bit(cmd, "rst_ni") == 0
    D = dim
    commit = fields(cmd["weight_commit_i"], 1, D)
    bank = fields(cmd["weight_commit_bank_i"], 1, D * D).reshape(n, D, D)
    lhs = fields(cmd["lhs_i"], 16, D)
    lhs_v = fields(cmd["lhs_valid_i"], 1, D)
    rhs = fields(cmd["rhs_i"], 16, D)
    rhs_v = fields(cmd["rhs_valid_i"], 1, D * BANKS).reshape(n, D, BANKS)
    g = PeGrid(D, D)
    res = np.zeros((n, D), dtype=np.int64)
    res_t = np.zeros((n, D), dtype=np.int64)
    res_v = np.zeros((n, D), dtype=np.int64)
    started = np.zeros(n, dtype=bool)
    s = False
    for t in range(n):
        s |= bool(rst[t])
        started[t] = s
        # inputs, from the ports and the neighbours' registers (before the edge)
        lin = np.empty((D, D), dtype=np.int64)
        lin_t = np.empty((D, D), dtype=np.int64)
        lin[:, 0], lin_t[:, 0] = lhs[t], OK
        lin[:, 1:], lin_t[:, 1:] = g.lhs_q[:, :-1], g.lhs_t[:, :-1]
        lvin = np.empty((D, D), dtype=np.int64)
        lvin[:, 0], lvin[:, 1:] = lhs_v[t], g.lhs_valid_q[:, :-1]
        cin = np.empty((D, D), dtype=np.int64)
        cin[:, 0], cin[:, 1:] = commit[t], g.commit_q[:, :-1]
        win = np.empty((BANKS, D, D), dtype=np.int64)
        win[:, 0, :] = rhs[t][None, :]
        win[:, 1:, :] = g.pending[:, :-1, :]
        wvin = np.broadcast_to(rhs_v[t].T[:, None, :], (BANKS, D, D))
        pin = np.empty((D, D), dtype=np.int64)
        pin_t = np.empty((D, D), dtype=np.int64)
        pin[0], pin_t[0] = 0, OK
        pin[1:], pin_t[1:] = g.result[:-1], g.result_t[:-1]
        pvin = np.empty((D, D), dtype=np.int64)
        pvin[0], pvin[1:] = lvin[0], g.vout[:-1]
        g.step(bool(rst[t]), lin, lin_t, lvin, cin, bank[t], win, wvin, pin, pin_t, pvin)
        res[t], res_t[t], res_v[t] = g.result[-1], g.result_t[-1], g.vout[-1]
    why_r = np.full(n, "", dtype=object)
    why_v = np.full(n, "", dtype=object)
    for t in range(n):
        if not started[t]:
            why_r[t] = why_v[t] = "before reset"
        elif res_t[t].max() != OK:
            why_r[t] = _REASON[int(res_t[t].max())]
    resp = {"result_o": _words(join(res, ACC_W), ACC_W * D),
            "result_valid_o": _words(join(res_v, 1), D)}
    return resp, {"result_o": why_r, "result_valid_o": why_v}, {}


def mxu_row(a, W):
    """One activation row through the array: ``a[DIM]`` bf16, ``W[DIM][DIM]``
    bf16 (``W[r][c]`` at PE (r, c)) -> per-lane bf16, the acc24 chain in
    ascending ``r`` then one ``pack_bf16``."""
    D = len(a)
    psum = np.zeros(D, dtype=np.int64)  # row 0's psum_i is '0
    for r in range(D):
        prod = ref.mxu_bf16_mul_acc24(np.full(D, a[r]), W[r]).astype(np.int64)
        psum = ref.mxu_acc24_add(prod, psum).astype(np.int64)
    return pack_bf16(psum)


def push_to_valid(dim):
    """Edges from the cycle the last LHS word of a group is pushed to the
    first cycle ``output_valid_o`` shows the group (module docstring)."""
    return 2 + dim * (PE_LATENCY + 1)


def switch_span(dim):
    """``mxu.sv`` ``WEIGHT_SWITCH_SPAN`` = ``(DIM-1)*PE + (DIM-1)``."""
    return (dim - 1) * PE_LATENCY + (dim - 1)


def mxu_trace(cmd, dim, input_depth=4, output_rows=64, sublanes=NUM_SUBLANES):
    """``mxu.sv``'s push / commit / pop contract (module docstring).

    Outputs, all ``"pre"`` (combinational from registered state):
    ``input_ready_o``, ``input_accept_o``, ``output_valid_o`` defined on every
    cycle after the first reset edge; ``output_data_o`` defined while
    ``output_valid_o``. Events: ``drop`` (a lane FIFO full when its group
    arrived), ``pop idle`` (``output_pop_i`` with nothing valid), ``bank
    overwrite`` (the switch rule broken).
    """
    n = len(cmd["rst_ni"])
    D = dim
    rst = _bit(cmd, "rst_ni") == 0
    push = _bit(cmd, "input_push_i")
    kind = _bit(cmd, "input_kind_i")  # 0 = LHS, 1 = RHS
    data = fields(cmd["input_data_i"], 16, D)
    commit = _bit(cmd, "weight_commit_i")
    popi = _bit(cmd, "output_pop_i")
    entries = output_rows // sublanes
    span = switch_span(D)
    names = ("input_ready_o", "input_accept_o", "output_valid_o")
    out = {k: np.zeros(n, dtype=np.int64) for k in names}
    odata = [0] * n
    why = {k: np.full(n, "", dtype=object) for k in names + ("output_data_o",)}
    ev = {}

    def event(k):
        ev[k] = ev.get(k, 0) + 1

    def fresh():
        return dict(fifo=[], pend=np.zeros((BANKS, D, D), dtype=np.int64), load_bank=0,
                    loaded_bank=1, waiting=False, active=np.zeros((D, D), dtype=np.int64),
                    tile=0, tile_of_bank={}, tile_start={}, rows=[],
                    lanes=[[] for _ in range(D)], sched={})

    S, started, bad_tiles = None, False, set()
    for t in range(n):
        if not started and not rst[t]:
            for k in why:
                why[k][t] = "before reset"
            continue
        if not started:  # the first reset cycle: its pre-edge state is unknown
            for k in why:
                why[k][t] = "before reset"
            S, started = fresh(), True
            continue
        # outputs of cycle t: the state before edge t + 1
        fifo, lanes = S["fifo"], S["lanes"]
        full, empty = len(fifo) >= input_depth, len(fifo) == 0
        out["input_ready_o"][t] = int(not full)
        out["input_accept_o"][t] = int(not empty)
        valid = all(lanes[c] for c in range(D))
        out["output_valid_o"][t] = int(valid)
        if valid:
            odata[t] = sum(lanes[c][0][0][s] << (16 * (s * D + c))
                           for c in range(D) for s in range(sublanes))
            if any(lanes[c][0][1] & bad_tiles for c in range(D)):
                why["output_data_o"][t] = "bank overwrite"
        else:
            why["output_data_o"][t] = "no valid"
        if rst[t]:  # a mid-trace reset: pointers, banks, gather and valids clear
            S = fresh()
            continue
        # the input FIFO pops its head every cycle it holds one
        started_tile = False
        if not empty:
            k_, d_ = fifo.pop(0)
            if k_ == 1:  # RHS: shift into the load bank
                b = S["load_bank"]
                st = S["tile_start"].get(b)
                if st is not None and t - st < span:
                    event("bank overwrite")
                    bad_tiles.add(S["tile_of_bank"][b])
                S["pend"][b, 1:, :] = S["pend"][b, :-1, :].copy()
                S["pend"][b, 0, :] = d_
            else:  # LHS: maybe start a tile; one result row
                if S["waiting"]:
                    started_tile = True
                    S["active"] = S["pend"][S["loaded_bank"]].copy()
                    S["tile"] += 1
                    S["tile_of_bank"][S["loaded_bank"]] = S["tile"]
                    S["tile_start"][S["loaded_bank"]] = t
                S["rows"].append((mxu_row(np.asarray(d_), S["active"]), S["tile"]))
                if len(S["rows"]) == sublanes:
                    tids = {r[1] for r in S["rows"]}
                    for c in range(D):
                        grp = [int(r[0][c]) for r in S["rows"]]
                        S["sched"].setdefault(t + 1 + D * PE_LATENCY + c, []).append((c, grp, tids))
                    S["rows"] = []
        if commit[t]:
            S["load_bank"], S["loaded_bank"] = 1 - S["load_bank"], S["load_bank"]
            S["waiting"] = True
        elif started_tile:
            S["waiting"] = False
        # output FIFOs: this cycle's pop, then this cycle's lane pushes
        pop = bool(popi[t]) and valid
        if popi[t] and not valid:
            event("pop idle")
        arriving = S["sched"].pop(t, [])
        for c in range(D):
            q = lanes[c]
            was_full = len(q) >= entries
            if pop:
                q.pop(0)
            for c_, grp, tids in arriving:
                if c_ != c:
                    continue
                if was_full and not pop:
                    event("drop")
                    continue
                q.append((grp, tids))
        # input FIFO push: always taken (a full FIFO is also popping this cycle)
        if push[t]:
            fifo.append((int(kind[t]), data[t].copy()))
    resp = {k: _words(v.tolist(), 1) for k, v in out.items()}
    resp["output_data_o"] = _words(odata, 16 * D * sublanes)
    return resp, why, ev

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for MachSuite kernels"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, f64, u8, index
from allo.backend.rtl import RegionKind
from tests.rtl._common import (
    Dcp,
    _sched,
    _to_rtl,
    _iis,
    FADD,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def test_runtime_vs_static_bounds():
    """A runtime-bounded loop leaves the whole-kernel latency unknown; a
    statically-bounded one resolves it. Both still pipeline. CRS's inner loop trip
    comes from a pair of row-pointer loads (a runtime scf.for), and its accumulate
    into out[i] is a memory-carried recurrence that must serialize; an empty row
    covers the zero-trip case. ELLPACK has a static bound with a per-element
    validity guard that if-converts to a select-gated accumulate."""
    SN, NNZ = 4, 6

    @kernel
    def crs(
        val: f64[NNZ], cols: i32[NNZ], row: i32[SN + 1], vec: f64[SN], out: f64[SN]
    ):
        for i in range(SN):
            tmp_begin: i32 = row[i]
            tmp_end: i32 = row[i + 1]
            for j in range(tmp_begin, tmp_end):
                out[i] += val[j] * vec[cols[j]]

    rtl = _to_rtl(crs)
    res = rtl.schedule()
    assert res.func("crs").latency is None  # dynamic trip -> latency omitted
    assert len(res.func("crs").cyclic()) >= 1

    # Row 1 is empty, and rows 0/2/3 hold two non-zeros each, so a dropped
    # accumulate cannot hide behind a single-element row.
    dense = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ],
        np.float64,
    )
    r, c = np.nonzero(dense)
    val = dense[r, c].copy()
    cols = c.astype(np.int32)
    row = np.concatenate(([0], np.cumsum((dense != 0).sum(axis=1)))).astype(np.int32)
    assert row[-1] == NNZ and row[1] == row[2]  # the CSR the kernel is given
    vec = np.array([1.5, -2.0, 0.25, 3.0], np.float64)
    out = np.zeros(SN, np.float64)
    rtl.cosim(val, cols, row, vec, out)
    assert np.allclose(out, dense @ vec)

    # ELLPACK is the static-bound counterpart. Its nest coalesces, so the guard's
    # operands come back as an affine.apply delinearizing the surviving IV.
    L = 4

    @kernel
    def ellpack(NZ: f64[SN * L], cols_e: i32[SN * L], vec_e: f64[SN], out_e: f64[SN]):
        for i in range(SN):
            for j in range(L):
                idx: i32 = j + i * L
                if cols_e[idx] != -1:
                    out_e[i] += NZ[idx] * vec_e[cols_e[idx]]

    rtl = _to_rtl(ellpack)
    res = rtl.schedule()
    assert res.func("ellpack").latency is not None  # static trip
    assert any(r.has("select") for r in res.func("ellpack").cyclic())

    rng = np.random.default_rng(0)
    NZ = rng.random(SN * L)
    cols_e = rng.integers(0, SN, SN * L).astype(np.int32)
    cols_e[3] = cols_e[9] = -1  # invalid slots the guard must mask out
    vec_e = rng.random(SN)
    out_e = np.zeros(SN, np.float64)
    g = np.zeros(SN, np.float64)
    for i in range(SN):
        for j in range(L):
            idx = j + i * L
            if cols_e[idx] != -1:
                g[i] += NZ[idx] * vec_e[cols_e[idx]]
    rtl.cosim(NZ, cols_e, vec_e, out_e)
    assert np.allclose(out_e, g, rtol=2e-3, atol=2e-3)


def test_tiled_trip():
    M, N, K, S = 8, 8, 8, 4

    @kernel
    def bbgemm(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        i_max: i32 = 0
        j_max: i32 = 0
        k_max: i32 = 0
        sum_value: i32 = 0
        for i in range(0, M, S):
            i_max = i + S if i + S < M else M
            for j in range(0, N, S):
                j_max = j + S if j + S < N else N
                for k in range(0, K, S):
                    k_max = k + S if k + S < K else K
                    for ii in range(i, i_max):
                        for jj in range(j, j_max):
                            sum_value = 0
                            for kk in range(k, k_max):
                                sum_value += A[ii, kk] * B[kk, jj]
                            C[ii, jj] += sum_value

    rtl = _to_rtl(bbgemm)
    res = rtl.schedule()
    assert res.func("bbgemm").latency is not None  # should be raised to affine
    assert any(r.has("muli") for r in res.cyclic())  # the matmul pipelines

    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, (M, K)).astype(np.int32)
    B = rng.integers(0, 4, (K, N)).astype(np.int32)
    C = np.zeros((M, N), np.int32)
    rtl.cosim(A, B, C)
    assert np.array_equal(C, A @ B)

    # Zero-trip edge of the same container path: `for j in range(i, Z-1-i)` goes
    # empty once i crosses the midpoint, so the container must issue no child and
    # still complete (the runtime `lb >= ub` guard plus the delayed empty-region
    # done). The even tiling in bbgemm never hits an empty tile, so cover it here.
    Z = 6

    @kernel
    def ztri(a: f32[Z * Z], out: f32[Z]):
        for i in range(Z):
            ub_i: index = Z - 1 - i
            acc: f32 = 0.0
            for j in range(i, ub_i):
                for kk in range(2):
                    acc += a[j * Z + kk]
            out[i] = acc

    a = (rng.random(Z * Z, np.float32) - np.float32(0.5)).astype(np.float32)
    out = np.zeros(Z, np.float32)
    g = np.zeros(Z, np.float32)
    for i in range(Z):
        s = np.float32(0.0)
        for j in range(i, Z - 1 - i):
            for kk in range(2):
                s += a[j * Z + kk]
        g[i] = s
    _to_rtl(ztri).cosim(a, out)
    assert np.allclose(out, g, rtol=2e-3, atol=2e-3)

    # A zero-trip scalar-reduction leaf whose bound is its enclosing container's
    # counter: `for k in range(i, j)` under `for j in range(i, W)`. On the diagonal
    # `j == i` the reduction runs no iteration, so its accumulator survivor must be
    # the identity (0), not the previous invocation's value.
    W = 6

    @kernel
    def tri_reduce(a: f32[W], acc_out: f32[W * W]):
        for i in range(W):
            for j in range(i, W):
                acc: f32 = 0.0
                for k in range(i, j):
                    acc += a[k]
                acc_out[i * W + j] = acc

    a2 = (rng.random(W, np.float32) - np.float32(0.5)).astype(np.float32)
    acc_out = np.zeros(W * W, np.float32)
    g2 = np.zeros(W * W, np.float32)
    for i in range(W):
        for j in range(i, W):
            s = np.float32(0.0)
            for k in range(i, j):
                s += a2[k]
            g2[i * W + j] = s
    _to_rtl(tri_reduce).cosim(a2, acc_out)
    assert np.allclose(acc_out, g2, rtol=2e-3, atol=2e-3)


def test_dynamic_programming():
    """DP kernels with statically-bounded nests, so the latency resolves. Viterbi
    reuses an outer loop's index (delayed to a later stage) across several sibling
    nested regions, each of which must build its own delay chain for that counter;
    its min-cost recurrence and integer backtrace both land exactly (argmin is
    order-stable). Needleman-Wunsch fills a scoring matrix, then a traceback
    carries scalar cursors that both index and SCATTER into `result` at
    data-dependent addresses and read `ptr` at a carried-scalar address."""
    N_OBS, N_STATES, N_TOKENS = 6, 4, 4

    @kernel
    def viterbi(
        obs: i32[N_OBS],
        init: f32[N_STATES],
        transition: f32[N_STATES, N_STATES],
        emission: f32[N_STATES, N_TOKENS],
        path: i32[N_OBS],
    ):
        llike: f32[N_OBS, N_STATES]
        for s in range(N_STATES):
            llike[0, s] = init[s] + emission[s, obs[0]]
        for t in range(1, N_OBS):
            for curr in range(N_STATES):
                min_p: f32 = (
                    llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
                )
                for prev in range(1, N_STATES):
                    p: f32 = (
                        llike[t - 1, prev]
                        + transition[prev, curr]
                        + emission[curr, obs[t]]
                    )
                    if p < min_p:
                        min_p = p
                llike[t, curr] = min_p
        min_s: i32 = 0
        min_p: f32 = llike[N_OBS - 1, 0]
        for s in range(1, N_STATES):
            p: f32 = llike[N_OBS - 1, s]
            if p < min_p:
                min_p = p
                min_s = s
        path[N_OBS - 1] = min_s
        for t in range(N_OBS - 1):
            actual_t: i32 = N_OBS - 2 - t
            min_s = 0
            min_p = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
            for s in range(1, N_STATES):
                p: f32 = llike[actual_t, s] + transition[s, path[actual_t + 1]]
                if p < min_p:
                    min_p = p
                    min_s = s
            path[actual_t] = min_s

    rtl = _to_rtl(viterbi)
    res = rtl.schedule()
    assert res.func("viterbi").latency is not None
    assert len([r for r in res.func("viterbi").regions if r.kind == "cyclic"]) >= 1

    rng = np.random.default_rng(1)
    obs = rng.integers(0, N_TOKENS, N_OBS).astype(np.int32)
    init = rng.random(N_STATES).astype(np.float32)
    transition = rng.random((N_STATES, N_STATES)).astype(np.float32)
    emission = rng.random((N_STATES, N_TOKENS)).astype(np.float32)
    path = np.zeros(N_OBS, np.int32)

    # exact scalar golden (matches the kernel's argmin order)
    llike = np.zeros((N_OBS, N_STATES), np.float32)
    for s in range(N_STATES):
        llike[0, s] = init[s] + emission[s, obs[0]]
    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            mp = llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                q = llike[t - 1, prev] + transition[prev, curr] + emission[curr, obs[t]]
                if q < mp:
                    mp = q
            llike[t, curr] = mp
    g = np.zeros(N_OBS, np.int32)
    ms, mp = 0, llike[N_OBS - 1, 0]
    for s in range(1, N_STATES):
        if llike[N_OBS - 1, s] < mp:
            mp, ms = llike[N_OBS - 1, s], s
    g[N_OBS - 1] = ms
    for t in range(N_OBS - 1):
        at, ms = N_OBS - 2 - t, 0
        mp = llike[at, 0] + transition[0, g[at + 1]]
        for s in range(1, N_STATES):
            q = llike[at, s] + transition[s, g[at + 1]]
            if q < mp:
                mp, ms = q, s
        g[at] = ms

    rtl.cosim(obs, init, transition, emission, path)
    assert np.array_equal(path, g)

    # Needleman-Wunsch: many statically-bounded nests plus a long-recurrence
    # traceback that scatters into `result` at data-dependent addresses.
    ALEN = BLEN = 8
    RESULT_LEN = ALEN + BLEN
    MATRIX_SIZE = (ALEN + 1) * (BLEN + 1)
    MATCH_SCORE, MISMATCH_SCORE, GAP_SCORE = 1, -1, -1
    ALIGN_VAL, SKIPA_VAL, SKIPB_VAL = 1, 2, 3

    @kernel
    def needwun(SEQA: i32[ALEN], SEQB: i32[BLEN], result: i32[2, RESULT_LEN]):
        M: i32[MATRIX_SIZE] = 0
        ptr: i32[MATRIX_SIZE] = 0
        score: i32 = 0
        row_up: i32 = 0
        row: i32 = 0
        up_left: i32 = 0
        up: i32 = 0
        left: i32 = 0
        max_val: i32 = 0
        for i in range(ALEN + 1):
            M[i] = i * GAP_SCORE
        for j in range(BLEN + 1):
            M[j * (ALEN + 1)] = j * GAP_SCORE
        for bi in range(1, BLEN + 1):
            for ai in range(1, ALEN + 1):
                if SEQA[ai - 1] == SEQB[bi - 1]:
                    score = MATCH_SCORE
                else:
                    score = MISMATCH_SCORE
                row_up = (bi - 1) * (ALEN + 1)
                row = bi * (ALEN + 1)
                up_left = M[row_up + (ai - 1)] + score
                up = M[row_up + ai] + GAP_SCORE
                left = M[row + (ai - 1)] + GAP_SCORE
                max_val = up_left
                if up > max_val:
                    max_val = up
                if left > max_val:
                    max_val = left
                M[row + ai] = max_val
                if max_val == left:
                    ptr[row + ai] = SKIPB_VAL
                elif max_val == up:
                    ptr[row + ai] = SKIPA_VAL
                else:
                    ptr[row + ai] = ALIGN_VAL
        a_idx: i32 = ALEN
        b_idx: i32 = BLEN
        a_str_idx: i32 = 0
        b_str_idx: i32 = 0
        r: i32 = 0
        for step in range(ALEN + BLEN):
            if a_idx > 0 or b_idx > 0:
                if a_idx == 0:
                    result[0, a_str_idx] = 45
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    b_idx = b_idx - 1
                elif b_idx == 0:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = 45
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                else:
                    r = b_idx * (ALEN + 1)
                    if ptr[r + a_idx] == ALIGN_VAL:
                        result[0, a_str_idx] = SEQA[a_idx - 1]
                        result[1, b_str_idx] = SEQB[b_idx - 1]
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        a_idx = a_idx - 1
                        b_idx = b_idx - 1
                    elif ptr[r + a_idx] == SKIPB_VAL:
                        result[0, a_str_idx] = SEQA[a_idx - 1]
                        result[1, b_str_idx] = 45
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        a_idx = a_idx - 1
                    else:
                        result[0, a_str_idx] = 45
                        result[1, b_str_idx] = SEQB[b_idx - 1]
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        b_idx = b_idx - 1
        for idx in range(RESULT_LEN):
            if result[0, idx] == 0:
                result[0, idx] = 95
            if result[1, idx] == 0:
                result[1, idx] = 95

    rtl = _to_rtl(needwun)
    res = rtl.schedule()
    assert res.func("needwun").latency is not None
    cyclic = [r for r in res.func("needwun").regions if r.kind == "cyclic"]
    assert len(cyclic) >= 4
    # An all-integer kernel: the bound is the memory-carried recurrence through
    # `result`, so the nest cannot close at II=1. No adder latency is involved.
    assert max(r.interval for r in cyclic) > 1

    def nw_golden(SEQA, SEQB):
        M = np.zeros(MATRIX_SIZE, np.int32)
        ptr = np.zeros(MATRIX_SIZE, np.int32)
        r_out = np.zeros((2, RESULT_LEN), np.int32)
        for i in range(ALEN + 1):
            M[i] = i * GAP_SCORE
        for j in range(BLEN + 1):
            M[j * (ALEN + 1)] = j * GAP_SCORE
        for bi in range(1, BLEN + 1):
            for ai in range(1, ALEN + 1):
                sc = MATCH_SCORE if SEQA[ai - 1] == SEQB[bi - 1] else MISMATCH_SCORE
                ru, rw = (bi - 1) * (ALEN + 1), bi * (ALEN + 1)
                ul = M[ru + ai - 1] + sc
                u = M[ru + ai] + GAP_SCORE
                le = M[rw + ai - 1] + GAP_SCORE
                mv = max(ul, u, le)
                M[rw + ai] = mv
                ptr[rw + ai] = (
                    SKIPB_VAL if mv == le else (SKIPA_VAL if mv == u else ALIGN_VAL)
                )
        a_idx, b_idx, ai_s, bi_s = ALEN, BLEN, 0, 0
        for _ in range(ALEN + BLEN):
            if not (a_idx > 0 or b_idx > 0):
                continue
            if a_idx == 0:
                r_out[0, ai_s], r_out[1, bi_s] = 45, SEQB[b_idx - 1]
                ai_s += 1
                bi_s += 1
                b_idx -= 1
            elif b_idx == 0:
                r_out[0, ai_s], r_out[1, bi_s] = SEQA[a_idx - 1], 45
                ai_s += 1
                bi_s += 1
                a_idx -= 1
            else:
                p = ptr[b_idx * (ALEN + 1) + a_idx]
                if p == ALIGN_VAL:
                    r_out[0, ai_s], r_out[1, bi_s] = SEQA[a_idx - 1], SEQB[b_idx - 1]
                    ai_s += 1
                    bi_s += 1
                    a_idx -= 1
                    b_idx -= 1
                elif p == SKIPB_VAL:
                    r_out[0, ai_s], r_out[1, bi_s] = SEQA[a_idx - 1], 45
                    ai_s += 1
                    bi_s += 1
                    a_idx -= 1
                else:
                    r_out[0, ai_s], r_out[1, bi_s] = 45, SEQB[b_idx - 1]
                    ai_s += 1
                    bi_s += 1
                    b_idx -= 1
        r_out[0][r_out[0] == 0] = 95
        r_out[1][r_out[1] == 0] = 95
        return r_out

    SEQA = rng.integers(0, 4, ALEN).astype(np.int32)
    SEQB = rng.integers(0, 4, BLEN).astype(np.int32)
    gnw = nw_golden(SEQA, SEQB)
    result = np.zeros((2, RESULT_LEN), np.int32)
    rtl.cosim(SEQA, SEQB, result)
    assert np.array_equal(result, gnw)


def test_data_dependent_while_kernels():
    """A data-dependent `while` schedules as a conditional (flushing) pipeline and
    leaves the latency unknown; no raw scf.while survives into the DCP IR. kmp
    nests two backtracking whiles whose conditions read memory, and drives end to
    end. bfs_queue's uncounted `while front != rear` over a modulo ring buffer
    scatters into `level`/`level_counts`/`queue` at data-dependent addresses and
    drives correctly end to end."""
    P, S = 4, 8

    # kmp: two data-dependent while loops (failure-function backtracking) nested in
    # counted for loops. The while conditions read `pattern[k]`/`pattern[q]` at a
    # data-dependent index (a memory-dependent continue-test).
    @kernel
    def kmp(pattern: u8[P], input_str: u8[S], kmp_next: u8[P], matches: u8[1]):
        k: index = 0
        x: index = 1
        for i in range(P - 1):
            while k > 0 and pattern[k] != pattern[x]:
                k = kmp_next[k - 1]
            if pattern[k] == pattern[x]:
                k += 1
            kmp_next[x] = k
            x += 1
        q: index = 0
        for i in range(S):
            while q > 0 and pattern[q] != input_str[i]:
                q = kmp_next[q - 1]
            if pattern[q] == input_str[i]:
                q += 1
            if q >= P:
                matches[0] += 1
                q = kmp_next[q - 1]

    res = _sched(kmp)
    assert res.func("kmp").latency is None  # data-dependent while trips
    assert len([r for r in res.cyclic() if r.conditional]) == 2  # both whiles

    # A small-alphabet input with a repeated prefix makes the failure-function
    # backtracking whiles actually iterate (k/q backtrack through kmp_next) rather
    # than exiting immediately.
    def kmp_golden(pattern, input_str):
        kmp_next = np.zeros(P, np.uint8)
        k = 0
        for x in range(1, P):
            while k > 0 and pattern[k] != pattern[x]:
                k = int(kmp_next[k - 1])
            if pattern[k] == pattern[x]:
                k += 1
            kmp_next[x] = k
        q = matches = 0
        for i in range(S):
            while q > 0 and pattern[q] != input_str[i]:
                q = int(kmp_next[q - 1])
            if pattern[q] == input_str[i]:
                q += 1
            if q >= P:
                matches += 1
                q = int(kmp_next[q - 1])
        return kmp_next, matches

    pat = np.array([0, 0, 1, 0], np.uint8)  # failure fn [0,1,0,1]; while backtracks
    inp = np.array([0, 0, 1, 0, 0, 0, 1, 0], np.uint8)
    gnext, gmatch = kmp_golden(pat, inp)
    assert (gnext > 0).any()  # the mem-condition while actually ran
    knext = np.zeros(P, np.uint8)
    matches = np.zeros(1, np.uint8)
    _to_rtl(kmp).cosim(pat.copy(), inp.copy(), knext, matches)
    assert np.array_equal(knext, gnext)
    assert int(matches[0]) == gmatch

    # bfs_queue: an uncounted `while front != rear` whose body holds a nested
    # data-dependent `for e` carrying the queue tail as an iter-arg. The scatter
    # loop schedules as its own pipeline; the while closes into a sequential
    # (data-dependent length) dcp.pipeline, so it carries no static II.
    N_NODES = 8
    N_NODES_2 = N_NODES * 2
    N_EDGES = 24
    N_LEVELS = 6
    MAX_LEVEL = 999999

    @kernel
    def bfs_queue(
        nodes: i32[N_NODES_2],
        edges: i32[N_EDGES],
        starting_node: i32,
        level: i32[N_NODES],
        level_counts: i32[N_LEVELS],
    ):
        queue: i32[N_NODES] = 0
        front: i32 = 0
        rear: i32 = 0
        level[starting_node] = 0
        level_counts[0] = 1
        queue[rear] = starting_node
        rear = (rear + 1) % N_NODES
        while front != rear:
            n: i32 = queue[front]
            front = (front + 1) % N_NODES
            tmp_begin: i32 = nodes[2 * n]
            tmp_end: i32 = nodes[2 * n + 1]
            for e in range(tmp_begin, tmp_end):
                tmp_dst: i32 = edges[e]
                tmp_level: i32 = level[tmp_dst]
                if tmp_level == MAX_LEVEL:
                    tmp_level = level[n] + 1
                    level[tmp_dst] = tmp_level
                    level_counts[tmp_level] += 1
                    queue[rear] = tmp_dst
                    rear = (rear + 1) % N_NODES

    rtl = _to_rtl(bfs_queue)
    res = rtl.schedule()
    assert len(res.cyclic()) >= 1  # the nested scatter loop got its own pipeline
    # A region nested in an scf.while reports an unknown execution count, so the
    # whole-kernel latency stays unknown.
    assert res.func("bfs_queue").latency is None
    d = Dcp(rtl)
    assert not d.has("scf.while") and d.has("allo.dcp.condition")
    guard = next(r for r in res.cyclic(wrappers=True) if r.conditional)
    assert guard.interval is None  # sequential (data-dependent length), no static II

    # CSR of a small DAG (node i -> {i+1, i+2}); each node is enqueued at most
    # once, so the ring buffer never overruns N_NODES.
    adj = [[j for j in (i + 1, i + 2) if j < N_NODES] for i in range(N_NODES)]
    edge_list: list[int] = []
    nodes = np.zeros(N_NODES_2, np.int32)
    for i in range(N_NODES):
        nodes[2 * i] = len(edge_list)
        edge_list.extend(adj[i])
        nodes[2 * i + 1] = len(edge_list)
    edges = np.zeros(N_EDGES, np.int32)
    edges[: len(edge_list)] = edge_list

    def bfs_golden(start):
        level = np.full(N_NODES, MAX_LEVEL, np.int32)
        counts = np.zeros(N_LEVELS, np.int32)
        queue = np.zeros(N_NODES, np.int32)
        front = rear = 0
        level[start] = 0
        counts[0] = 1
        queue[rear] = start
        rear = (rear + 1) % N_NODES
        while front != rear:
            n = int(queue[front])
            front = (front + 1) % N_NODES
            for e in range(int(nodes[2 * n]), int(nodes[2 * n + 1])):
                dst = int(edges[e])
                if level[dst] == MAX_LEVEL:
                    lv = level[n] + 1
                    level[dst] = lv
                    counts[lv] += 1
                    queue[rear] = dst
                    rear = (rear + 1) % N_NODES
        return level, counts

    gl, gc = bfs_golden(0)
    level = np.full(N_NODES, MAX_LEVEL, np.int32)
    level_counts = np.zeros(N_LEVELS, np.int32)
    rtl.cosim(nodes, edges, np.int32(0), level, level_counts)
    assert np.array_equal(level, gl)
    assert np.array_equal(level_counts, gc)


def test_fft_strided():
    """A radix-2 strided FFT over a power-of-two transform: an outer `while` over
    the stage span and an inner `while` over the butterfly index, a
    double-precision butterfly, and a twiddle-factor guard. Both while conditions
    are loop-carried scalar comparisons, so the whole-kernel latency is not
    statically known. Driven against a NumPy reference."""
    N = 8
    H = N // 2

    @kernel
    def fft(real: f64[N], img: f64[N], real_twid: f64[H], img_twid: f64[H]):
        span: i32 = N >> 1
        log: i32 = 0
        even: i32 = 0
        odd: i32 = 0
        rootindex: i32 = 0
        temp: f64 = 0.0
        while span > 0:
            odd = span
            while odd < N:
                odd |= span
                even = odd ^ span
                temp = real[even] + real[odd]
                real[odd] = real[even] - real[odd]
                real[even] = temp
                temp = img[even] + img[odd]
                img[odd] = img[even] - img[odd]
                img[even] = temp
                rootindex = (even << log) & (N - 1)
                if rootindex > 0:
                    temp = (
                        real_twid[rootindex] * real[odd]
                        - img_twid[rootindex] * img[odd]
                    )
                    img[odd] = (
                        real_twid[rootindex] * img[odd]
                        + img_twid[rootindex] * real[odd]
                    )
                    real[odd] = temp
                odd += 1
            span >>= 1
            log += 1

    res = _sched(fft)
    assert res.func("fft").latency is None  # data-dependent while trips
    assert any(r.conditional for r in res.cyclic())  # the while regions

    def fft_golden(real, img, rt, it):
        span = N >> 1
        log = 0
        while span > 0:
            odd = span
            while odd < N:
                odd |= span
                even = odd ^ span
                temp = real[even] + real[odd]
                real[odd] = real[even] - real[odd]
                real[even] = temp
                temp = img[even] + img[odd]
                img[odd] = img[even] - img[odd]
                img[even] = temp
                ri = (even << log) & (N - 1)
                if ri > 0:
                    temp = rt[ri] * real[odd] - it[ri] * img[odd]
                    img[odd] = rt[ri] * img[odd] + it[ri] * real[odd]
                    real[odd] = temp
                odd += 1
            span >>= 1
            log += 1

    rng = np.random.default_rng(1)
    real = rng.standard_normal(N)
    img = rng.standard_normal(N)
    idx = np.arange(H)
    rt = np.cos(2.0 * np.pi * idx / N)
    it = np.sin(2.0 * np.pi * idx / N)
    gr, gi = real.copy(), img.copy()
    fft_golden(gr, gi, rt, it)
    cr, ci = real.copy(), img.copy()
    _to_rtl(fft).cosim(cr, ci, rt.copy(), it.copy())
    assert np.allclose(cr, gr, rtol=1e-9, atol=1e-9)
    assert np.allclose(ci, gi, rtol=1e-9, atol=1e-9)


def test_grid_parallel():
    """`allo.grid` lowers to a nested affine.for band that the whole scheduling
    pipeline handles: constant trips give a static latency, and a real
    reduction recurrence still closes despite the grid's nodep hint. gemm's
    k-reduction into C[i, j] is raised to an iter_arg, rotated to II=1, and
    drives correctly end to end. The grid stencils live in test_loop_control.py, which cosims them at
    non-power-of-two extents that exercise the div/mod delinearisation this
    file's shapes miss."""
    P = 8

    # The canonical grid() matmul: C[i, j] is affine, so the grid's assume.nodep
    # does not touch the real k-reduction recurrence (raised to an iter_arg).
    @kernel
    def gemm(A: f32[P, P], B: f32[P, P], C: f32[P, P]):
        for i, j in allo.grid(P, P):
            for k in range(P):
                C[i, j] += A[i, k] * B[k, j]

    rtl = _to_rtl(gemm)
    res = rtl.schedule()
    assert res.func("gemm").latency is not None
    assert res.func("gemm").cyclic()[-1].interval == 1

    rng = np.random.default_rng(0)
    A = (rng.random((P, P), dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random((P, P), dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    C = np.zeros((P, P), np.float32)  # a pure-output buffer is zero-inited by cosim
    rtl.cosim(A, B, C)
    # f32 accumulation reassociates in hardware, so compare to a tolerance.
    assert np.allclose(C, A @ B, rtol=2e-3, atol=2e-3)


def test_double_precision_divide():
    """A double-precision force computation exercises the f64 datapath and the
    multi-cycle divide: the guard selecting between the reciprocal and its fallback
    and a gather through NL[...] all have to land on the same cycle budget for `fx`
    to accumulate the right value."""
    nAtoms, maxNeighbors = 8, 4
    lj1, lj2, domainEdge = 1.5, 2.0, 20.0

    @kernel
    def md_x(
        position_x: f64[nAtoms],
        position_y: f64[nAtoms],
        position_z: f64[nAtoms],
        NL: i32[nAtoms * maxNeighbors],
        force_x: f64[nAtoms],
    ):
        i_x: f64 = 0.0
        i_y: f64 = 0.0
        i_z: f64 = 0.0
        jidx: i32 = 0
        j_x: f64 = 0.0
        j_y: f64 = 0.0
        j_z: f64 = 0.0
        delx: f64 = 0.0
        dely: f64 = 0.0
        delz: f64 = 0.0
        r2inv: f64 = 0.0
        r6inv: f64 = 0.0
        potential: f64 = 0.0
        force: f64 = 0.0
        fx: f64 = 0.0

        for i in range(nAtoms):
            i_x = position_x[i]
            i_y = position_y[i]
            i_z = position_z[i]
            fx = 0.0

            for j in range(maxNeighbors):
                jidx = NL[i * maxNeighbors + j]
                j_x = position_x[jidx]
                j_y = position_y[jidx]
                j_z = position_z[jidx]
                delx = i_x - j_x
                dely = i_y - j_y
                delz = i_z - j_z
                if (delx * delx + dely * dely + delz * delz) == 0:
                    r2inv = (domainEdge * domainEdge * 3.0) * 1000
                else:
                    r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
                r6inv = r2inv * r2inv * r2inv
                potential = r6inv * (lj1 * r6inv - lj2)
                force = r2inv * potential
                fx = fx + delx * force
            force_x[i] = fx

    rtl = _to_rtl(md_x)
    res = rtl.schedule()
    assert res.cyclic()  # pipelines
    assert any(r.has("divf") for r in res.cyclic())  # f64 reciprocal

    rng = np.random.default_rng(0)
    px, py, pz = (rng.standard_normal(nAtoms) for _ in range(3))
    # A neighbour list that includes i itself for some i, so the ==0 guard (the
    # self-distance) is actually exercised rather than always taking the divide.
    NL = rng.integers(0, nAtoms, size=nAtoms * maxNeighbors).astype(np.int32)
    NL[0] = 0

    exp = np.zeros(nAtoms, np.float64)
    for i in range(nAtoms):
        fx = 0.0
        for j in range(maxNeighbors):
            jidx = int(NL[i * maxNeighbors + j])
            dx, dy, dz = px[i] - px[jidx], py[i] - py[jidx], pz[i] - pz[jidx]
            r2 = dx * dx + dy * dy + dz * dz
            r2inv = (domainEdge * domainEdge * 3.0) * 1000 if r2 == 0 else 1.0 / r2
            r6inv = r2inv * r2inv * r2inv
            fx += dx * (r2inv * (r6inv * (lj1 * r6inv - lj2)))
        exp[i] = fx

    force_x = np.zeros(nAtoms, np.float64)
    rtl.cosim(px, py, pz, NL, force_x)
    assert np.allclose(force_x, exp, rtol=1e-2, atol=1e-2)


def test_port_bound_ii_read_write_same_array():
    """Loads and stores contending for one array's ports bound the II by resource,
    not by operator type: `weights` sees 2 reads + 2 writes per iteration over 2
    ports, so II = ceil(4/2) = 2. The two elements are exchanged, so time-sharing
    the ports must not let either write land before both reads; the norm comes back
    on the scalar result port at done.

    The accesses are to DISTINCT elements on purpose. A read of the element just
    written is not port pressure at all any more: `scalarize-memory` forwards the
    stored value, which is the right answer but leaves nothing to oversubscribe."""
    NPL, LR = 8, 2

    # Integer accumulate, so the norm recurrence is combinational and the port
    # oversubscription -- the actual subject -- is the binding constraint.
    @kernel
    def wnorm(weights: i32[2 * NPL], dweights: i32[2 * NPL]) -> i32:
        norm: i32 = 0
        for i in range(NPL):
            lo: i32 = weights[2 * i]
            hi: i32 = weights[2 * i + 1]
            weights[2 * i] = hi - dweights[2 * i] * LR
            weights[2 * i + 1] = lo - dweights[2 * i + 1] * LR
            norm += lo * lo + hi * hi
        return norm

    rtl = _to_rtl(wnorm)
    assert _iis(rtl.schedule().cyclic()) == [2]

    rng = np.random.default_rng(0)
    weights = rng.integers(0, 8, size=2 * NPL).astype(np.int32)
    dweights = rng.integers(0, 8, size=2 * NPL).astype(np.int32)
    exp_w = weights.copy()
    exp_w[0::2] = weights[1::2] - dweights[0::2] * LR
    exp_w[1::2] = weights[0::2] - dweights[1::2] * LR
    exp_norm = int(np.sum(weights.astype(np.int64) ** 2))

    r = rtl.cosim(weights, dweights)
    assert np.array_equal(weights, exp_w)
    assert r.result == exp_norm


def test_radix_sort():
    """radixsort's LSD ping-pong: each pass histograms one 2-bit radix, prefix-
    scans the buckets, then scatters into the other buffer -- `if valid_buffer==0:
    read a,write b else: read b,write a`, an if/else whose both arms hold store
    loops and whose data-dependent predicate flips a carried flag. It closes into a
    result-mux dcp.select (a dual guard yielding `valid_buffer`). Sized to 16 i8-
    range keys over 4 passes so the result lands back in `a`."""
    EPB, SIZE, RADIX = 4, 16, 4
    NBLK = SIZE // EPB
    BKT = NBLK * RADIX + 1
    SCAN_BLK = 4
    SCAN_R = (BKT - 1) // SCAN_BLK
    PASSES = 4  # 2 bits/pass -> low 8 bits; even, so the sorted result ends in `a`

    @kernel
    def ss_sort(a: i32[SIZE]):
        b: i32[SIZE] = 0
        bucket: i32[BKT] = 0
        sm: i32[SCAN_R] = 0
        bucket_indx: i32 = 0
        a_indx: i32 = 0
        valid_buffer: i32 = 0
        for exp in range(PASSES):
            for i_init in range(BKT):
                bucket[i_init] = 0
            if valid_buffer == 0:
                for blockID in range(NBLK):
                    for i_h in range(4):
                        a_indx = blockID * EPB + i_h
                        bucket_indx = (
                            ((a[a_indx] >> (exp * 2)) & 0x3) * NBLK + blockID + 1
                        )
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
            else:
                for blockID in range(NBLK):
                    for i_h in range(4):
                        a_indx = blockID * EPB + i_h
                        bucket_indx = (
                            ((b[a_indx] >> (exp * 2)) & 0x3) * NBLK + blockID + 1
                        )
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
            for radixID in range(SCAN_R):
                for i_ls in range(1, SCAN_BLK):
                    bucket_indx = radixID * SCAN_BLK + i_ls
                    bucket[bucket_indx] = bucket[bucket_indx] + bucket[bucket_indx - 1]
            sm[0] = 0
            for radixID_s in range(1, SCAN_R):
                bucket_indx = radixID_s * SCAN_BLK - 1
                sm[radixID_s] = sm[radixID_s - 1] + bucket[bucket_indx]
            for radixID_l in range(SCAN_R):
                for i_lss in range(SCAN_BLK):
                    bucket_indx = radixID_l * SCAN_BLK + i_lss
                    bucket[bucket_indx] = bucket[bucket_indx] + sm[radixID_l]
            if valid_buffer == 0:
                for blockID_u in range(NBLK):
                    for i_u in range(4):
                        bucket_indx = (
                            (a[blockID_u * EPB + i_u] >> (exp * 2)) & 0x3
                        ) * NBLK + blockID_u
                        a_indx = blockID_u * EPB + i_u
                        b[bucket[bucket_indx]] = a[a_indx]
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
                valid_buffer = 1
            else:
                for blockID_u in range(NBLK):
                    for i_u in range(4):
                        bucket_indx = (
                            (b[blockID_u * EPB + i_u] >> (exp * 2)) & 0x3
                        ) * NBLK + blockID_u
                        a_indx = blockID_u * EPB + i_u
                        a[bucket[bucket_indx]] = b[a_indx]
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
                valid_buffer = 0

    rtl = _to_rtl(ss_sort)
    # the ping-pong closes into a result-mux guard
    assert rtl.schedule().regions(RegionKind.GUARD, wrappers=True)

    a = np.random.default_rng(0).integers(0, 256, SIZE).astype(np.int32)
    gold = np.sort(a)
    rtl.cosim(a)
    assert np.array_equal(a, gold)


def test_bfs_bulk():
    """bfs_bulk sweeps the frontier level by level: `if level[n]==horizon:` visits a
    node, and for each edge `if level[dst]==MAX:` marks the neighbour and bumps a
    carried `cnt`. The nested data-dependent guards wrap loops and update `cnt`, so
    they close into result-mux dcp.selects; `if cnt!=0:` records the level size.
    A small star graph (node 0 -> the rest) is one frontier hop."""
    N_NODES, N_EDGES, N_LEVELS, MAX = 8, 8, 4, 999999

    @kernel
    def bfs_bulk(
        nodes: i32[N_NODES * 2],
        edges: i32[N_EDGES],
        starting_node: i32,
        level: i32[N_NODES],
        level_counts: i32[N_LEVELS],
    ):
        for n in range(N_NODES):
            level[n] = MAX
        for l in range(N_LEVELS):
            level_counts[l] = 0
        level[starting_node] = 0
        level_counts[0] = 1
        for horizon in range(N_LEVELS):
            cnt: i32 = 0
            horizon_i32: i32 = horizon
            for n in range(N_NODES):
                if level[n] == horizon_i32:
                    tmp_begin: i32 = nodes[2 * n]
                    tmp_end: i32 = nodes[2 * n + 1]
                    for e in range(tmp_begin, tmp_end):
                        tmp_dst: i32 = edges[e]
                        tmp_level: i32 = level[tmp_dst]
                        if tmp_level == MAX:
                            level[tmp_dst] = horizon_i32 + 1
                            cnt += 1
            if cnt != 0:
                level_counts[horizon + 1] = cnt

    rtl = _to_rtl(bfs_bulk)
    assert rtl.schedule().regions(RegionKind.GUARD, wrappers=True)

    nodes = np.zeros(N_NODES * 2, np.int32)
    nodes[0], nodes[1] = 0, 7  # node 0 owns edges[0:7]
    edges = np.array([1, 2, 3, 4, 5, 6, 7, 0], np.int32)  # -> nodes 1..7
    level = np.zeros(N_NODES, np.int32)
    counts = np.zeros(N_LEVELS, np.int32)

    gl = np.full(N_NODES, MAX, np.int32)
    gc = np.zeros(N_LEVELS, np.int32)
    gl[0], gc[0] = 0, 1
    for horizon in range(N_LEVELS):
        cnt = 0
        for n in range(N_NODES):
            if gl[n] == horizon:
                for e in range(nodes[2 * n], nodes[2 * n + 1]):
                    if gl[edges[e]] == MAX:
                        gl[edges[e]] = horizon + 1
                        cnt += 1
        if cnt != 0:
            gc[horizon + 1] = cnt

    rtl.cosim(nodes, edges, np.int32(0), level, counts)
    assert np.array_equal(level, gl)
    assert np.array_equal(counts, gc)

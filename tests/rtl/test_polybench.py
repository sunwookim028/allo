# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for PolyBench kernels"""

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from allo import kernel
from allo.lang import f32, index
from allo.operators import math as amath
from allo.lang.ip import operator_ip
from allo.backend.rtl.devices import default_device
from _common import (
    Dcp,
    _to_rtl,
    _iis,
    FADD,
    FDIV,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

# f32 accumulation reassociates in hardware (the reduction is balanced into a
# tree), so a cosim result matches a sequential NumPy golden only to f32 epsilon
# grown by the reduction depth -- compare with a tolerance, never exactly.
FTOL = {"rtol": 2e-3, "atol": 2e-3}

# Triangular solves and other divide-heavy kernels grow the f32 error past the
# reduction tolerance, so compare them a little looser.
STOL = {"rtol": 5e-3, "atol": 5e-3}


def _f32(seed, *shape):
    """Deterministic f32 test data in [-0.5, 0.5)."""
    rng = np.random.default_rng(seed)
    return (rng.random(shape, dtype=np.float32) - np.float32(0.5)).astype(np.float32)


def test_matmul_reductions():
    """Matmul stages carry a float accumulation, rotated across accumulators to
    II=1 by default; the elementwise and writeback stages carry no recurrence and
    also pipeline at II=1."""
    P, R, Q, S, alpha, beta = 4, 5, 6, 3, 0.1, 0.5

    # gemm = matmul then a scaled elementwise add.
    @kernel
    def gemm_mm(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def gemm_add(out_AB: f32[P, R], C: f32[P, R], output: f32[P, R]):
        for i2 in range(P):
            for j2 in range(R):
                output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]

    @kernel
    def gemm(A: f32[P, Q], B: f32[Q, R], C: f32[P, R], output: f32[P, R]):
        out_AB: f32[P, R] = 0.0
        gemm_mm(A, B, out_AB)
        gemm_add(out_AB, C, output)

    rtl = _to_rtl(gemm)
    res = rtl.schedule()
    assert res.func("gemm_mm").cyclic()[0].interval == 1
    assert res.func("gemm_add").cyclic()[0].interval == 1

    A, B, C = _f32(0, P, Q), _f32(1, Q, R), _f32(2, P, R)
    output = np.zeros((P, R), np.float32)
    rtl.cosim(A, B, C, output)
    assert np.allclose(output, beta * C + A @ B, **FTOL)

    # two_mm = (A*B)*C scaled and added to D: two chained matmul reductions, the
    # second consuming the first through an internal buffer, then an elementwise
    # stage.
    @kernel
    def tmm_ab(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def tmm_abc(out_AB: f32[P, R], C: f32[R, S], out_ABC: f32[P, S]):
        for i1 in range(P):
            for j1 in range(S):
                for k1 in range(R):
                    out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]

    @kernel
    def tmm_add(out_ABC: f32[P, S], D: f32[P, S], output: f32[P, S]):
        for i2 in range(P):
            for j2 in range(S):
                output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha

    @kernel
    def two_mm(
        A: f32[P, Q], B: f32[Q, R], C: f32[R, S], D: f32[P, S], output: f32[P, S]
    ):
        out_AB: f32[P, R] = 0.0
        out_ABC: f32[P, S] = 0.0
        tmm_ab(A, B, out_AB)
        tmm_abc(out_AB, C, out_ABC)
        tmm_add(out_ABC, D, output)

    rtl = _to_rtl(two_mm)
    res = rtl.schedule()
    assert res.func("tmm_ab").cyclic()[0].interval == 1
    assert res.func("tmm_abc").cyclic()[0].interval == 1
    assert res.func("tmm_add").cyclic()[0].interval == 1

    A, B, C, D = _f32(0, P, Q), _f32(1, Q, R), _f32(2, R, S), _f32(3, P, S)
    output = np.zeros((P, S), np.float32)
    rtl.cosim(A, B, C, D, output)
    assert np.allclose(output, (A @ B) @ C * beta + D * alpha, **FTOL)

    # doitgen: a four-deep nest decomposing into an inner reduction into sum_[p]
    # and a copy region that republishes it over A[r, q, :]; the copy must not
    # start until the reduction has drained.
    DQ, DR, DP = 3, 4, 5

    @kernel
    def doitgen(A: f32[DR, DQ, DP], x: f32[DP, DP], sum_: f32[DP]):
        for r in range(DR):
            for q in range(DQ):
                for p in range(DP):
                    sum_[p] = 0.0
                    for s in range(DP):
                        sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
                for p1 in range(DP):
                    A[r, q, p1] = sum_[p1]

    rtl = _to_rtl(doitgen)
    iis = _iis(rtl.schedule().cyclic())
    assert 1 in iis  # the inner accumulation, rotated to II=1
    assert 1 in iis  # the writeback copy

    A, x = _f32(0, DR, DQ, DP), _f32(1, DP, DP)
    exp = A.copy()
    for r in range(DR):
        for q in range(DQ):
            exp[r, q, :] = exp[r, q, :] @ x
    rtl.cosim(A, x, np.zeros(DP, np.float32))
    assert np.allclose(A, exp, **FTOL)

    # three_mm = (A*B)*(C*D): three matmul reductions, the third consuming two
    # buffered products.
    P3, Q3, R3, S3, T3 = 3, 4, 5, 3, 4

    @kernel
    def mm1(A: f32[P3, Q3], B: f32[Q3, R3], o: f32[P3, R3]):
        for i in range(P3):
            for j in range(R3):
                for kk in range(Q3):
                    o[i, j] += A[i, kk] * B[kk, j]

    @kernel
    def mm2(C: f32[R3, S3], D: f32[S3, T3], o: f32[R3, T3]):
        for i in range(R3):
            for j in range(T3):
                for kk in range(S3):
                    o[i, j] += C[i, kk] * D[kk, j]

    @kernel
    def mm3(AB: f32[P3, R3], CD: f32[R3, T3], o: f32[P3, T3]):
        for i in range(P3):
            for j in range(T3):
                for kk in range(R3):
                    o[i, j] += AB[i, kk] * CD[kk, j]

    @kernel
    def three_mm(
        A: f32[P3, Q3], B: f32[Q3, R3], C: f32[R3, S3], D: f32[S3, T3], out: f32[P3, T3]
    ):
        AB: f32[P3, R3] = 0.0
        CD: f32[R3, T3] = 0.0
        mm1(A, B, AB)
        mm2(C, D, CD)
        mm3(AB, CD, out)

    rtl = _to_rtl(three_mm)
    res = rtl.schedule()
    assert res.func("mm1").cyclic()[0].interval == 1
    assert res.func("mm3").cyclic()[0].interval == 1

    A, B, C, D = _f32(0, P3, Q3), _f32(1, Q3, R3), _f32(2, R3, S3), _f32(3, S3, T3)
    out = np.zeros((P3, T3), np.float32)
    rtl.cosim(A, B, C, D, out)
    assert np.allclose(out, (A @ B) @ (C @ D), **STOL)


def test_reduction_ii_follows_accumulator_location():
    """The accumulator's location sets the II: a memory cell indexed by the inner
    IV carries no recurrence (II=1), while one indexed by the outer IV is a
    loop-carried reduction raised to an iter_arg, so its II is the register
    recurrence FADD."""
    M, N = 6, 5

    # bicg: stageS accumulates into s[j0], the INNER index -- every iteration
    # touches a different cell, so there is no carried recurrence. stageQ
    # accumulates into q[i1], the outer index, across the inner loop.
    @kernel
    def stageS(A: f32[N, M], r: f32[N], s: f32[M]):
        for i0 in range(N):
            local_r: f32 = r[i0]
            for j0 in range(M):
                s[j0] += local_r * A[i0, j0]

    @kernel
    def stageQ(A: f32[N, M], p: f32[M], q: f32[N]):
        for i1 in range(N):
            for j1 in range(M):
                q[i1] += A[i1, j1] * p[j1]

    @kernel
    def bicg(
        A: f32[N, M], A_copy: f32[N, M], p: f32[M], r: f32[N], q: f32[N], s: f32[M]
    ):
        stageS(A, r, s)
        stageQ(A_copy, p, q)

    rtl = _to_rtl(bicg).set_scheduler_opt(accumulators=0)
    res = rtl.schedule()
    assert res.func("stageS").cyclic()[0].interval == 1
    assert res.func("stageQ").cyclic()[0].interval == FADD

    A, p, r = _f32(0, N, M), _f32(1, M), _f32(2, N)
    q, s = np.zeros(N, np.float32), np.zeros(M, np.float32)
    rtl.cosim(A, A.copy(), p, r, q, s)
    assert np.allclose(s, r @ A, **FTOL)
    assert np.allclose(q, A @ p, **FTOL)

    # atax = A^T (A x): both stages accumulate into a memory cell, and the second
    # may not read out_Ax before the first has finished writing it.
    AM, AN = 5, 6

    @kernel
    def atax_m(A: f32[AM, AN], x: f32[AN], out_Ax: f32[AM]):
        for m in range(AM):
            for rr in range(AN):
                out_Ax[m] += A[m, rr] * x[rr]

    @kernel
    def atax_n(A: f32[AM, AN], out_Ax: f32[AM], y: f32[AN]):
        for n in range(AN):
            for k in range(AM):
                y[n] += A[k, n] * out_Ax[k]

    @kernel
    def atax(A: f32[AM, AN], x: f32[AN], y: f32[AN]):
        out_Ax: f32[AM] = 0.0
        atax_m(A, x, out_Ax)
        atax_n(A, out_Ax, y)

    rtl = _to_rtl(atax).set_scheduler_opt(accumulators=0)
    res = rtl.schedule()
    assert res.func("atax_m").cyclic()[0].interval == FADD
    assert res.func("atax_n").cyclic()[0].interval == FADD

    A, x = _f32(0, AM, AN), _f32(1, AN)
    y = np.zeros(AN, np.float32)
    rtl.cosim(A, x, y)
    assert np.allclose(y, A.T @ (A @ x), **FTOL)

    # mvt accumulates into a scalar local, so the recurrence stays in a register
    # and the II is just the add latency. The load-init before and store after the
    # inner loop make it an imperfect nest: a sequential outer wrapper around the
    # pipelined inner region plus acyclic prologue/epilogue regions. The init is a
    # prologue survivor (a load), re-injected on every outer iteration.
    V = 4

    @kernel
    def stageA(x1_in: f32[V], x1_out: f32[V], A: f32[V, V], y1: f32[V]):
        for i0 in range(V):
            x: f32 = x1_in[i0]
            for j0 in range(V):
                x += A[i0, j0] * y1[j0]
            x1_out[i0] = x

    @kernel
    def stageB(x2_in: f32[V], x2_out: f32[V], A: f32[V, V], y2: f32[V]):
        for i1 in range(V):
            x: f32 = x2_in[i1]
            for j1 in range(V):
                x += A[j1, i1] * y2[j1]
            x2_out[i1] = x

    @kernel
    def mvt(
        A: f32[V, V],
        A_copy: f32[V, V],
        y1: f32[V],
        y2: f32[V],
        x1: f32[V],
        x2: f32[V],
        x1_out: f32[V],
        x2_out: f32[V],
    ):
        stageA(x1, x1_out, A, y1)
        stageB(x2, x2_out, A_copy, y2)

    rtl = _to_rtl(mvt).set_scheduler_opt(accumulators=0)
    sa = rtl.schedule().func("stageA")
    assert sa.cyclic()[0].interval == FADD  # register-carried reduction recurrence
    assert len([r for r in sa.regions if r.kind == "acyclic"]) >= 2  # prologue+epilogue
    wrapper = next(r for r in sa.regions if r.is_wrapper)
    assert wrapper.depth == 0 and wrapper.trip_count == V

    A, y1, y2 = _f32(0, V, V), _f32(1, V), _f32(2, V)
    x1, x2 = _f32(3, V), _f32(4, V)
    x1_out, x2_out = np.zeros(V, np.float32), np.zeros(V, np.float32)
    rtl.cosim(A, A.copy(), y1, y2, x1, x2, x1_out, x2_out)
    assert np.allclose(x1_out, x1 + A @ y1, **FTOL)
    assert np.allclose(x2_out, x2 + A.T @ y2, **FTOL)


def test_stencil_ii_port_vs_recurrence_bound():
    """A dependence-free stencil is bound by memory-port pressure; an in-place one
    is bound by its carried recurrence. The in-place kernels only reproduce the
    sequential golden if that recurrence actually serializes."""
    TSTEPS, N = 3, 8
    c = np.float32(0.33333)

    # jacobi_1d: three reads over two ports -> II = ceil(3/2) = 2, on two sweeps.
    @kernel
    def jacobi_1d(A: f32[N], B: f32[N]):
        for m in range(TSTEPS):
            for i0 in range(1, N - 1):
                B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])
            for i1 in range(1, N - 1):
                A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])

    rtl = _to_rtl(jacobi_1d)
    cyclic = rtl.schedule().cyclic()
    # The port bound is a floor: an II below it would oversubscribe the ports.
    # Whether the scheduler reaches the floor is not this test's subject.
    assert len(cyclic) == 2 and all(r.interval >= 2 for r in cyclic)

    A, B = _f32(0, N), _f32(1, N)
    Ag, Bg = A.copy(), B.copy()
    for _ in range(TSTEPS):
        for i in range(1, N - 1):
            Bg[i] = c * (Ag[i - 1] + Ag[i] + Ag[i + 1])
        for i in range(1, N - 1):
            Ag[i] = c * (Bg[i - 1] + Bg[i] + Bg[i + 1])
    rtl.cosim(A, B)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(B, Bg, **FTOL)

    # fdtd_2d: four dependence-free update stages per timestep over three shared
    # buffers, each at II=1; each stage reads what the previous one wrote, so they
    # must not overlap.
    Tmax, Nx, Ny = 2, 4, 5
    h, s = np.float32(0.5), np.float32(0.7)

    @kernel
    def fdtd_2d(ex: f32[Nx, Ny], ey: f32[Nx, Ny], hz: f32[Nx, Ny], fict: f32[Tmax]):
        for m in range(Tmax):
            for j in range(Ny):
                ey[0, j] = fict[m]
            for i in range(1, Nx):
                for j in range(Ny):
                    ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
            for i in range(Nx):
                for j in range(1, Ny):
                    ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
            for i in range(Nx - 1):
                for j in range(Ny - 1):
                    hz[i, j] = hz[i, j] - 0.7 * (
                        ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                    )

    rtl = _to_rtl(fdtd_2d)
    cyclic = rtl.schedule().cyclic()
    assert len(cyclic) == 4 and all(r.interval == 1 for r in cyclic)

    ex, ey, hz, fict = _f32(0, Nx, Ny), _f32(1, Nx, Ny), _f32(2, Nx, Ny), _f32(3, Tmax)
    exg, eyg, hzg = ex.copy(), ey.copy(), hz.copy()
    for m in range(Tmax):
        for j in range(Ny):
            eyg[0, j] = fict[m]
        for i in range(1, Nx):
            for j in range(Ny):
                eyg[i, j] = eyg[i, j] - h * (hzg[i, j] - hzg[i - 1, j])
        for i in range(Nx):
            for j in range(1, Ny):
                exg[i, j] = exg[i, j] - h * (hzg[i, j] - hzg[i, j - 1])
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                hzg[i, j] = hzg[i, j] - s * (
                    exg[i, j + 1] - exg[i, j] + eyg[i + 1, j] - eyg[i, j]
                )
    rtl.cosim(ex, ey, hz, fict)
    assert np.allclose(ex, exg, **FTOL)
    assert np.allclose(ey, eyg, **FTOL)
    assert np.allclose(hz, hzg, **FTOL)

    # heat_3d: a 7-point stencil issues many loads per iteration, so with no
    # recurrence the II is dominated by port pressure. B[i,j,k] is written and
    # then re-read by the A update in the same body (an intra-iteration dep).
    HT, H = 2, 5
    c0, c1 = np.float32(0.125), np.float32(2.0)

    @kernel
    def heat_3d(A: f32[H, H, H], B: f32[H, H, H]):
        const0: f32 = 0.125
        const1: f32 = 2.0
        for m in range(HT):
            for i in range(1, H - 1):
                for j in range(1, H - 1):
                    for k in range(1, H - 1):
                        B[i, j, k] = (
                            const0
                            * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                            + const0
                            * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                            + const0
                            * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                            + A[i, j, k]
                        )
                        A[i, j, k] = (
                            const0
                            * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                            + const0
                            * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                            + const0
                            * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                            + B[i, j, k]
                        )

    # Port pressure, not the adder: the body reads A and B at seven points each
    # over a fixed port budget, so the nest cannot close at II=1.
    rtl = _to_rtl(heat_3d)
    assert rtl.schedule().cyclic()[0].interval > 1

    A, B = _f32(0, H, H, H), _f32(1, H, H, H)
    Ag, Bg = A.copy(), B.copy()
    for _ in range(HT):
        for i in range(1, H - 1):
            for j in range(1, H - 1):
                for k in range(1, H - 1):
                    Bg[i, j, k] = (
                        c0 * (Ag[i + 1, j, k] - c1 * Ag[i, j, k] + Ag[i - 1, j, k])
                        + c0 * (Ag[i, j + 1, k] - c1 * Ag[i, j, k] + Ag[i, j - 1, k])
                        + c0 * (Ag[i, j, k + 1] - c1 * Ag[i, j, k] + Ag[i, j, k - 1])
                        + Ag[i, j, k]
                    )
                    Ag[i, j, k] = (
                        c0 * (Bg[i + 1, j, k] - c1 * Bg[i, j, k] + Bg[i - 1, j, k])
                        + c0 * (Bg[i, j + 1, k] - c1 * Bg[i, j, k] + Bg[i, j - 1, k])
                        + c0 * (Bg[i, j, k + 1] - c1 * Bg[i, j, k] + Bg[i, j, k - 1])
                        + Bg[i, j, k]
                    )
    rtl.cosim(A, B)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(B, Bg, **FTOL)

    # seidel_2d: a 9-point Gauss-Seidel sweep updates A in place, so A[i,j-1] and
    # A[i-1,*] read values written earlier in the same sweep -- the II is set by
    # that carried recurrence (the divide is on its critical path), not by ports.
    SN = 6

    @kernel
    def seidel_2d(A: f32[SN, SN]):
        for t in range(TSTEPS):
            for i in range(1, SN - 1):
                for j in range(1, SN - 1):
                    A[i, j] = (
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9.0

    rtl = _to_rtl(seidel_2d)
    cyclic = rtl.schedule().cyclic()
    assert len(cyclic) == 1 and cyclic[0].interval > FDIV

    A = _f32(0, SN, SN)
    Ag = A.copy()
    for _ in range(TSTEPS):
        for i in range(1, SN - 1):
            for j in range(1, SN - 1):
                Ag[i, j] = (
                    Ag[i - 1, j - 1]
                    + Ag[i - 1, j]
                    + Ag[i - 1, j + 1]
                    + Ag[i, j - 1]
                    + Ag[i, j]
                    + Ag[i, j + 1]
                    + Ag[i + 1, j - 1]
                    + Ag[i + 1, j]
                    + Ag[i + 1, j + 1]
                ) / np.float32(9.0)
    rtl.cosim(A)
    assert np.allclose(A, Ag, **FTOL)


def test_multi_region_single_func():
    """Several sweeps in one function schedule to one cyclic region each, mixing
    dependence-free (II=1) and memory-carried reduction (II>1) loops; each region
    consumes the writes of the previous one through a shared array."""
    N, alpha, beta = 5, 0.1, 0.1

    @kernel
    def gemver(
        A: f32[N, N],
        u1: f32[N],
        u2: f32[N],
        v1: f32[N],
        v2: f32[N],
        x: f32[N],
        y: f32[N],
        w: f32[N],
        z: f32[N],
    ):
        for i in range(N):
            for j in range(N):
                A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
        for i in range(N):
            for j in range(N):
                x[i] = x[i] + beta * A[j, i] * y[j]
        for i in range(N):
            x[i] = x[i] + z[i]
        for i in range(N):
            for j in range(N):
                w[i] = w[i] + alpha * A[i, j] * x[j]

    rtl = _to_rtl(gemver).set_scheduler_opt(accumulators=0)
    iis = set(_iis(rtl.schedule().cyclic()))
    assert 1 in iis and any(v > 1 for v in iis)

    A = _f32(0, N, N)
    u1, u2, v1, v2 = _f32(1, N), _f32(2, N), _f32(3, N), _f32(4, N)
    x, y, z = _f32(5, N), _f32(6, N), _f32(7, N)
    w = np.zeros(N, np.float32)
    Ag, xg, wg = A.copy(), x.copy(), w.copy()
    Ag = Ag + np.outer(u1, v1) + np.outer(u2, v2)
    xg = xg + np.float32(beta) * (Ag.T @ y) + z
    wg = wg + np.float32(alpha) * (Ag @ xg)
    rtl.cosim(A, u1, u2, v1, v2, x, y, w, z)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(x, xg, **FTOL)
    assert np.allclose(w, wg, **FTOL)

    # gesummv: tt and yy accumulate in the same inner body (a reduction), then a
    # second kernel combines them through a handoff buffer (elementwise, II=1).
    G = 5

    @kernel
    def compute_tmp(
        y_in: f32[G], y_out: f32[G], A: f32[G, G], B: f32[G, G], x: f32[G], tmp: f32[G]
    ):
        tt: f32[G] = 0.0
        yy: f32[G]
        for i0 in range(G):
            yy[i0] = y_in[i0]
        for i in range(G):
            for j in range(G):
                tt[i] += A[i, j] * x[j]
                yy[i] += B[i, j] * x[j]
        for i1 in range(G):
            tmp[i1] = tt[i1]
            y_out[i1] = yy[i1]

    @kernel
    def compute_y(y_in: f32[G], y_out: f32[G], tmp: f32[G]):
        for i0 in range(G):
            y_out[i0] = alpha * tmp[i0] + beta * y_in[i0]

    @kernel
    def gesummv(A: f32[G, G], B: f32[G, G], x: f32[G], y: f32[G]):
        y_init: f32[G] = 0.0
        y_fifo: f32[G]
        tmp: f32[G]
        compute_tmp(y_init, y_fifo, A, B, x, tmp)
        compute_y(y_fifo, y, tmp)

    rtl = _to_rtl(gesummv).set_scheduler_opt(scalarize_threshold=0, accumulators=0)
    res = rtl.schedule()
    assert FADD in _iis(res.func("compute_tmp").cyclic())
    assert res.func("compute_y").cyclic()[0].interval == 1

    A, B, x = _f32(0, G, G), _f32(1, G, G), _f32(2, G)
    y = np.zeros(G, np.float32)
    rtl.cosim(A, B, x, y)
    assert np.allclose(
        y, np.float32(alpha) * (A @ x) + np.float32(beta) * (B @ x), **FTOL
    )


def test_if_conversion_in_loops():
    """A guard inside a loop body if-converts to a select so the loop still
    pipelines. trmm's nest coalesces and its guard becomes quasi-affine in the one
    surviving IV, so only the if-conversion can honour it; floyd_warshall's
    conditional store becomes a predicated read-modify-write. The relaxation is a
    real carried dependence, so the pipelined sweep must still agree with the
    sequential one.

    The extents are powers of two so that the nest still coalesces: a guard the
    coalescing leaves alone folds into a loop bound instead, which is a better
    schedule but not the path under test here (test_storage covers that one)."""
    M, N = 4, 8

    # A count-only guard: accumulate 1.0 over the strict upper triangle, so any
    # wrong trip count shows up as an exact integer.
    @kernel
    def tri_count(Cout: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        Cout[i1, j1] += 1.0

    Cout = np.zeros((M, N), np.float32)
    _to_rtl(tri_count).cosim(Cout)
    # Row i1 runs the guard for k1 in (i1, M) -> M - 1 - i1 accumulates.
    assert np.array_equal(
        Cout, np.tile((M - 1 - np.arange(M, dtype=np.float32))[:, None], (1, N))
    )

    # trmm: the same triangular guard (k > i) over a real memory-carried
    # accumulate, then a scale. S0's guarded accumulate pipelines at II>1; S1's
    # plain scale at II=1.
    alpha = 1.5

    @kernel
    def trmm_S0(A: f32[M, M], B: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        B[i1, j1] += A[k1, i1] * B[k1, j1]

    @kernel
    def trmm_S1(B: f32[M, N]):
        for i0 in range(M):
            for j0 in range(N):
                B[i0, j0] = B[i0, j0] * alpha

    @kernel
    def trmm(A: f32[M, M], B: f32[M, N]):
        trmm_S0(A, B)
        trmm_S1(B)

    rtl = _to_rtl(trmm)
    res = rtl.schedule()
    assert res.func("trmm_S0").cyclic()[0].interval > 1
    assert res.func("trmm_S1").cyclic()[0].interval == 1

    A, B = _f32(0, M, M), _f32(1, M, N)
    g = B.copy()
    for i1 in range(M):
        for j1 in range(N):
            for k1 in range(M):
                if k1 > i1:
                    g[i1, j1] += A[k1, i1] * g[k1, j1]
    g *= np.float32(alpha)
    rtl.cosim(A, B)
    assert np.allclose(B, g, **FTOL)

    # update_C (syrk's first stage): an if/ELSE over a triangular region, so the
    # guard if-converts to a select between two speculated values.
    @kernel
    def update_C(Cin: f32[M, M], Cout: f32[M, M]):
        for i0 in range(M):
            for j0 in range(M):
                if j0 <= i0:
                    Cout[i0, j0] = alpha * Cin[i0, j0]
                else:
                    Cout[i0, j0] = Cin[i0, j0]

    rtl = _to_rtl(update_C)
    loop = rtl.schedule().func("update_C").cyclic()[0]
    assert loop.has("select") and not loop.has("if")

    Cin = _f32(2, M, M)
    Cout = np.zeros((M, M), np.float32)
    gc = np.where(
        np.arange(M)[None, :] <= np.arange(M)[:, None], np.float32(alpha) * Cin, Cin
    )
    rtl.cosim(Cin, Cout)
    assert np.allclose(Cout, gc, **FTOL)

    F = 6

    @kernel
    def floyd_warshall(path: f32[F, F]):
        for k in range(F):
            for i in range(F):
                for j in range(F):
                    path_: f32 = path[i, k] + path[k, j]
                    if path[i, j] >= path_:
                        path[i, j] = path_

    rtl = _to_rtl(floyd_warshall)
    loop = rtl.schedule().cyclic()[0]
    assert loop.has("select") and not loop.has("if")

    # Positive edge weights, so the relaxation converges the way a distance
    # matrix should rather than running away negative.
    path = (np.abs(_f32(0, F, F)) + np.float32(0.5)).astype(np.float32)
    g = path.copy()
    for k in range(F):
        for i in range(F):
            for j in range(F):
                p = g[i, k] + g[k, j]
                if g[i, j] >= p:
                    g[i, j] = p
    rtl.cosim(path)
    assert np.allclose(path, g, **FTOL)


def test_syrk_triangular_accumulate():
    """syrk composes update_C (an if/else that if-converts to a select) with
    compute_sum, whose inner update is a triangular guard over a memory-carried
    accumulate (`if j1 <= i1: buffer[i1, j1] += ...`) -- the select gates the
    accumulate's input, so the carried recurrence still has to close and the
    guarded and unguarded cells must both land."""
    N, M, alpha, beta = 5, 4, 1.5, 1.2

    @kernel
    def update_C(Cin: f32[N, N], Cout: f32[N, N]):
        for i0 in range(N):
            for j0 in range(N):
                if j0 <= i0:
                    Cout[i0, j0] = beta * Cin[i0, j0]
                else:
                    Cout[i0, j0] = Cin[i0, j0]

    @kernel
    def compute_sum(A: f32[N, M], A_copy: f32[N, M], Cin: f32[N, N], Cout: f32[N, N]):
        buffer: f32[N, N] = 0.0
        for i0 in range(N):
            for j0 in range(N):
                buffer[i0, j0] = Cin[i0, j0]
        for i1 in range(N):
            for k1 in range(M):
                for j1 in range(N):
                    if j1 <= i1:
                        buffer[i1, j1] += alpha * A[i1, k1] * A_copy[j1, k1]
        for i2 in range(N):
            for j2 in range(N):
                Cout[i2, j2] = buffer[i2, j2]

    @kernel
    def syrk(A: f32[N, M], A_copy: f32[N, M], Cin: f32[N, N], Cout: f32[N, N]):
        C: f32[N, N] = 0.0
        update_C(Cin, C)
        compute_sum(A, A_copy, C, Cout)

    rtl = _to_rtl(syrk)
    res = rtl.schedule()
    assert res.func("update_C").cyclic()[0].has("select")  # if/else if-converted
    # compute_sum's guard takes the other route: `fold-if-statements` folds it
    # into the inner loop's UPPER BOUND, so the accumulate is unconditional and
    # what has to land is the bound, an acyclic region of its own.
    inner = [r for r in res.func("compute_sum").regions if r.kind == "acyclic"]
    assert any(r.has("minsi") for r in inner)

    A = _f32(0, N, M)
    A_copy = A.copy()
    Cin = _f32(1, N, N)
    Cout = np.zeros((N, N), np.float32)

    tri = np.arange(N)[None, :] <= np.arange(N)[:, None]  # j <= i
    g = np.where(tri, np.float32(beta) * Cin, Cin).astype(np.float32)
    for i1 in range(N):
        for k1 in range(M):
            for j1 in range(N):
                if j1 <= i1:
                    g[i1, j1] += np.float32(alpha) * A[i1, k1] * A_copy[j1, k1]

    rtl.cosim(A, A_copy, Cin, Cout)
    assert np.allclose(Cout, g, **FTOL)


def test_correlation_folded_bound():
    """correlation's `if j > i` is affine in the IV, so it folds into the `j`
    loop's lower bound -- an affine.for whose lb is symbolic (`i + 1`). The dead
    iterations are skipped, not predicated, so no `affine.if` survives and the
    inner reduction rotates to II=1. The counter runs `[i+1, CM)`, a
    variable-trip container per outer i."""
    CN, CM = 8, 5

    @kernel
    def compute_corr(data: f32[CN, CM], corr: f32[CM, CM]):
        for i in range(CM - 1):
            corr[i, i] = 1.0
            for j in range(CM):
                if j > i:
                    corr_v: f32 = 0.0
                    for k in range(CN):
                        corr_v += data[k, i] * data[k, j]
                    corr[j, i] = corr_v
                    corr[i, j] = corr_v
        corr[CM - 1, CM - 1] = 1.0

    rtl = _to_rtl(compute_corr)
    assert not Dcp(rtl).has("affine.if")  # folded into the bound, not predicated
    assert _iis(rtl.schedule().func("compute_corr").cyclic()) == [1]

    data = _f32(0, CN, CM)
    corr = np.zeros((CM, CM), np.float32)
    g = np.zeros((CM, CM), np.float32)
    for i in range(CM - 1):
        g[i, i] = 1.0
        for j in range(CM):
            if j > i:
                v = np.float32(0.0)
                for k in range(CN):
                    v += data[k, i] * data[k, j]
                g[j, i] = v
                g[i, j] = v
    g[CM - 1, CM - 1] = 1.0
    rtl.cosim(data, corr)
    assert np.allclose(corr, g, **FTOL)


def test_nussinov_triangular_dp():
    """nussinov is a triangular DP whose inner loops are variable-trip
    (`for j in range(i+1, D)`, `for k in range(i+1, j)`), the k-loop's upper bound
    being its enclosing j-loop's own counter. A child reading its container's
    counter as a bound must see the iteration it is starting. The trip counts are
    data-dependent so the whole-function latency stays unknown, and the table
    update is a memory-carried max, so the pipelined sweep only agrees with the
    sequential one if those iterations serialize."""
    D = 8

    @kernel
    def nussinov(seq: f32[D], table: f32[D, D]):
        for i_inv in range(D):
            i: index = D - 1 - i_inv
            for j in range(i + 1, D):
                if j - 1 >= 0:
                    if table[i, j] < table[i, j - 1]:
                        table[i, j] = table[i, j - 1]
                if i + 1 < D:
                    if table[i, j] < table[i + 1, j]:
                        table[i, j] = table[i + 1, j]
                if j - 1 >= 0 and i + 1 < D:
                    if i < j - 1:
                        w: f32 = seq[i] + seq[j]
                        match: f32 = 0.0
                        if w == 3.0:
                            match = 1.0
                        s2: f32 = table[i + 1, j - 1] + match
                        if table[i, j] < s2:
                            table[i, j] = s2
                    else:
                        if table[i, j] < table[i + 1, j - 1]:
                            table[i, j] = table[i + 1, j - 1]
                for k in range(i + 1, j):
                    s3: f32 = table[i, k] + table[k + 1, j]
                    if table[i, j] < s3:
                        table[i, j] = s3

    rtl = _to_rtl(nussinov)
    res = rtl.schedule()
    assert res.func("nussinov").latency is None  # data-dependent trips
    loop = next(r for r in res.cyclic() if r.is_leaf)
    assert loop.interval > 1  # memory-carried max recurrence into table[i, j]
    assert loop.has("select")  # boundary/compare guards if-converted

    rng = np.random.default_rng(2)
    seq = rng.integers(0, 4, D).astype(np.float32)
    table = rng.random((D, D), np.float32).astype(np.float32)
    g = table.copy()
    for i_inv in range(D):
        i = D - 1 - i_inv
        for j in range(i + 1, D):
            if j - 1 >= 0 and g[i, j] < g[i, j - 1]:
                g[i, j] = g[i, j - 1]
            if i + 1 < D and g[i, j] < g[i + 1, j]:
                g[i, j] = g[i + 1, j]
            if j - 1 >= 0 and i + 1 < D:
                if i < j - 1:
                    w = seq[i] + seq[j]
                    m = np.float32(1.0) if w == 3.0 else np.float32(0.0)
                    s2 = g[i + 1, j - 1] + m
                    if g[i, j] < s2:
                        g[i, j] = s2
                elif g[i, j] < g[i + 1, j - 1]:
                    g[i, j] = g[i + 1, j - 1]
            for k in range(i + 1, j):
                s3 = g[i, k] + g[k + 1, j]
                if g[i, j] < s3:
                    g[i, j] = s3
    rtl.cosim(seq, table)
    assert np.allclose(table, g, **FTOL)


def test_cholesky_triangular():
    """cholesky is the densest variable-trip case: a deep triangular nest whose
    inner `j`/`k` loops are all bounded by an enclosing IV, mixing a memory-carried
    subtraction reduction, an fdiv, and a declared `sqrt` operator IP. The
    data-dependent trips leave the whole-function latency unknown."""
    N = 6

    # sqrt is non-combinational with no built-in characterization, so declare it
    # as an operator IP to fully characterize the kernel.
    @operator_ip(optype="sqrt", latency=7, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def cholesky(A: f32[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
                A[i, j] = A[i, j] / A[j, j]
            for k in range(i):
                A[i, i] = A[i, i] - A[i, k] * A[i, k]
            A[i, i] = amath.sqrt(A[i, i] * 1.0)

    dev = default_device.copy()
    dev.add_operator(fsqrt)
    rtl = _to_rtl(cholesky, device=dev)
    res = rtl.schedule()
    assert res.func("cholesky").latency is None  # data-dependent trips
    assert any(r.interval > 1 for r in res.cyclic())

    # Symmetric positive-definite input so every pivot stays positive (`sqrt` of a
    # negative would diverge from the NumPy golden as a NaN).
    Ms = _f32(0, N, N)
    A = (Ms @ Ms.T + N * np.eye(N, dtype=np.float32)).astype(np.float32)
    g = A.copy()
    for i in range(N):
        for j in range(i):
            for k in range(j):
                g[i, j] = g[i, j] - g[i, k] * g[j, k]
            g[i, j] = g[i, j] / g[j, j]
        for k in range(i):
            g[i, i] = g[i, i] - g[i, k] * g[i, k]
        g[i, i] = np.sqrt(g[i, i] * 1.0)

    buf = A.copy()
    rtl.cosim(buf)
    tril = np.tril_indices(N)  # only the lower triangle + diagonal are written
    assert np.allclose(buf[tril], g[tril], **FTOL)


def test_syr2k_symmetric_rank_2_update():
    """syr2k is syrk's two-product sibling: update_C's if/else if-converts to a
    select, and compute_sum's triangular guard (`if j <= i`) gates a memory-carried
    accumulate of two rank-1 terms."""
    N, M, alpha, beta = 5, 4, 1.5, 1.2

    @kernel
    def update_C(Cin: f32[N, N], Cout: f32[N, N]):
        for i in range(N):
            for j in range(N):
                if j <= i:
                    Cout[i, j] = beta * Cin[i, j]
                else:
                    Cout[i, j] = Cin[i, j]

    @kernel
    def compute_sum(
        A: f32[N, M],
        A_c: f32[N, M],
        B: f32[N, M],
        B_c: f32[N, M],
        Cin: f32[N, N],
        Cout: f32[N, N],
    ):
        buf: f32[N, N] = 0.0
        for i in range(N):
            for j in range(N):
                buf[i, j] = Cin[i, j]
        for i in range(N):
            for kk in range(M):
                for j in range(N):
                    if j <= i:
                        buf[i, j] += (
                            A[j, kk] * alpha * B[i, kk]
                            + B_c[j, kk] * alpha * A_c[i, kk]
                        )
        for i in range(N):
            for j in range(N):
                Cout[i, j] = buf[i, j]

    @kernel
    def syr2k(
        A: f32[N, M],
        A_c: f32[N, M],
        B: f32[N, M],
        B_c: f32[N, M],
        Cin: f32[N, N],
        Cout: f32[N, N],
    ):
        C: f32[N, N] = 0.0
        update_C(Cin, C)
        compute_sum(A, A_c, B, B_c, C, Cout)

    # The if-conversion claim is syrk's, asserted there on the same two
    # sub-kernels; what this kernel adds is the two-product accumulate, so it
    # carries the end-to-end check only.
    rtl = _to_rtl(syr2k)

    A, B, Cin = _f32(0, N, M), _f32(1, N, M), _f32(2, N, N)
    Cout = np.zeros((N, N), np.float32)
    a, be = np.float32(alpha), np.float32(beta)
    tri = np.arange(N)[None, :] <= np.arange(N)[:, None]
    g = np.where(tri, be * Cin, Cin).astype(np.float32)
    for i in range(N):
        for kk in range(M):
            for j in range(N):
                if j <= i:
                    g[i, j] += A[j, kk] * a * B[i, kk] + B[j, kk] * a * A[i, kk]
    rtl.cosim(A, A.copy(), B, B.copy(), Cin, Cout)
    assert np.allclose(Cout, g, **STOL)


def test_symm_symmetric_multiply():
    """symm multiplies a symmetric matrix: compute_sum's `if k < i` guard
    if-converts to a select, and update_C's `for k in range(i)` inner loop is an
    enclosing-IV-bounded variable-trip nest, so the whole-function latency is
    unknown. compute_sum's extents are powers of two so its nest still coalesces
    and the guard has no loop bound to fold into."""
    M, N, alpha, beta = 4, 8, 1.5, 1.2

    @kernel
    def compute_sum(A: f32[M, M], B: f32[M, N], summ: f32[M, N]):
        for i in range(M):
            for j in range(N):
                for kk in range(M):
                    if kk < i:
                        summ[i, j] += B[kk, j] * A[i, kk]

    @kernel
    def update_C(A: f32[M, M], B: f32[M, N], summ: f32[M, N], C: f32[M, N]):
        for i in range(M):
            for kk in range(i):
                for j in range(N):
                    C[kk, j] = C[kk, j] + alpha * B[i, j] * A[i, kk]
            for j in range(N):
                C[i, j] = (
                    beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * summ[i, j]
                )

    @kernel
    def symm(A0: f32[M, M], A1: f32[M, M], B0: f32[M, N], B1: f32[M, N], C: f32[M, N]):
        summ: f32[M, N] = 0.0
        compute_sum(A0, B0, summ)
        update_C(A1, B1, summ, C)

    rtl = _to_rtl(symm)
    res = rtl.schedule()
    assert res.func("symm").latency is None  # update_C's for-range(i) inner nest
    assert res.func("compute_sum").cyclic()[0].has("select")

    A, B, C = _f32(0, M, M), _f32(1, M, N), _f32(2, M, N)
    a, be = np.float32(alpha), np.float32(beta)
    summ = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(N):
            for kk in range(M):
                if kk < i:
                    summ[i, j] += B[kk, j] * A[i, kk]
    g = C.copy()
    for i in range(M):
        for kk in range(i):
            for j in range(N):
                g[kk, j] = g[kk, j] + a * B[i, j] * A[i, kk]
        for j in range(N):
            g[i, j] = be * g[i, j] + a * B[i, j] * A[i, i] + a * summ[i, j]
    rtl.cosim(A, A.copy(), B, B.copy(), C)
    assert np.allclose(C, g, **STOL)


def test_triangular_solve():
    """LU-family kernels with triangular (variable-trip) inner loops, so each has
    an unknown whole-function latency. trisolv is a forward substitution carrying a
    scalar into a memory cell; lu factors in place; ludcmp factors then runs the
    forward/back substitutions."""
    N = 6

    @kernel
    def trisolv(L: f32[N, N], b: f32[N], x: f32[N]):
        for i in range(N):
            x[i] = b[i]
            for j in range(i):
                x[i] -= L[i, j] * x[j]
            x[i] /= L[i, i]

    rtl = _to_rtl(trisolv)
    assert rtl.schedule().func("trisolv").latency is None  # for-range(i) inner loop

    L = _f32(0, N, N) + np.float32(3.0) * np.eye(N, dtype=np.float32)  # diag-dominant
    b = _f32(1, N)
    x = np.zeros(N, np.float32)
    g = np.zeros(N, np.float32)
    for i in range(N):
        g[i] = b[i]
        for j in range(i):
            g[i] -= L[i, j] * g[j]
        g[i] /= L[i, i]
    rtl.cosim(L, b, x)
    assert np.allclose(x, g, **STOL)

    @kernel
    def lu(A: f32[N, N]):
        for i in range(N):
            for j in range(i):
                for kk in range(j):
                    A[i, j] -= A[i, kk] * A[kk, j]
                A[i, j] /= A[j, j]
            for j in range(i, N):
                for kk in range(i):
                    A[i, j] -= A[i, kk] * A[kk, j]

    rtl = _to_rtl(lu)
    res = rtl.schedule()
    assert res.func("lu").latency is None
    # memory-carried subtraction recurrence
    assert any(r.interval is not None and r.interval > 1 for r in res.cyclic())

    A = _f32(0, N, N) + np.float32(4.0) * np.eye(N, dtype=np.float32)
    g = A.copy()
    for i in range(N):
        for j in range(i):
            for kk in range(j):
                g[i, j] -= g[i, kk] * g[kk, j]
            g[i, j] /= g[j, j]
        for j in range(i, N):
            for kk in range(i):
                g[i, j] -= g[i, kk] * g[kk, j]
    buf = A.copy()
    rtl.cosim(buf)
    assert np.allclose(buf, g, **STOL)

    @kernel
    def ludcmp(A: f32[N, N], b: f32[N], x: f32[N], y: f32[N]):
        for i in range(N):
            for j in range(i):
                w_l: f32 = A[i, j]
                for kk in range(j):
                    w_l -= A[i, kk] * A[kk, j]
                A[i, j] = w_l / A[j, j]
            for j in range(i, N):
                w_u: f32 = A[i, j]
                for kk in range(i):
                    w_u -= A[i, kk] * A[kk, j]
                A[i, j] = w_u
        for i in range(N):
            a_y: f32 = b[i]
            for j in range(i):
                a_y -= A[i, j] * y[j]
            y[i] = a_y
        for i_inv in range(N):
            i: index = N - 1 - i_inv
            a_x: f32 = y[i]
            for j in range(i + 1, N):
                a_x -= A[i, j] * x[j]
            x[i] = a_x / A[i, i]

    rtl = _to_rtl(ludcmp)
    assert rtl.schedule().func("ludcmp").latency is None

    A = _f32(0, N, N) + np.float32(4.0) * np.eye(N, dtype=np.float32)
    b = _f32(1, N)
    x = np.zeros(N, np.float32)
    y = np.zeros(N, np.float32)
    g = A.copy()
    for i in range(N):
        for j in range(i):
            w = g[i, j]
            for kk in range(j):
                w -= g[i, kk] * g[kk, j]
            g[i, j] = w / g[j, j]
        for j in range(i, N):
            w = g[i, j]
            for kk in range(i):
                w -= g[i, kk] * g[kk, j]
            g[i, j] = w
    gy = np.zeros(N, np.float32)
    for i in range(N):
        w = b[i]
        for j in range(i):
            w -= g[i, j] * gy[j]
        gy[i] = w
    gx = np.zeros(N, np.float32)
    for i in range(N - 1, -1, -1):
        w = gy[i]
        for j in range(i + 1, N):
            w -= g[i, j] * gx[j]
        gx[i] = w / g[i, i]
    rtl.cosim(A.copy(), b, x, y)
    assert np.allclose(x, gx, **STOL)


def test_covariance_reduction():
    """covariance is two nested reductions with a scalar accumulator (rotated to
    II=1): a column-mean pass then the centered outer product; the trips are all
    constant, so the latency resolves."""
    N, M = 6, 5

    @kernel
    def covariance(data: f32[N, M], mean: f32[M], cov: f32[M, M]):
        for x in range(M):
            total: f32 = 0.0
            for kk in range(N):
                total += data[kk, x]
            mean[x] = total / N
        for i in range(M):
            for j in range(M):
                cv: f32 = 0.0
                for p in range(N):
                    cv += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
                cov[i, j] = cv / (N - 1)

    rtl = _to_rtl(covariance)
    res = rtl.schedule()
    assert res.func("covariance").latency is not None  # constant trips
    assert set(_iis(res.func("covariance").cyclic())) == {
        1
    }  # scalar reductions, rotated to II=1

    data = _f32(0, N, M)
    mean = np.zeros(M, np.float32)
    cov = np.zeros((M, M), np.float32)
    gm = data.mean(axis=0)
    gc = np.zeros((M, M), np.float32)
    for i in range(M):
        for j in range(M):
            gc[i, j] = np.sum((data[:, i] - gm[i]) * (data[:, j] - gm[j])) / (N - 1)
    rtl.cosim(data, mean, cov)
    assert np.allclose(mean, gm, **STOL)
    assert np.allclose(cov, gc, **STOL)


def test_gram_schmidt():
    """Modified Gram-Schmidt: a column-norm reduction, a normalize, then a
    projection sweep whose `for j in range(k+1, N)` inner loop is bounded by the
    enclosing IV, so the whole-function latency is unknown."""
    M, N = 5, 4

    @kernel
    def gramschmidt(A: f32[M, N], Q: f32[M, N], R: f32[N, N]):
        for k in range(N):
            nrm: f32 = 0.0
            for i in range(M):
                nrm += A[i, k] * A[i, k]
            R[k, k] = nrm
            for i in range(M):
                Q[i, k] = A[i, k] / R[k, k]
            for j in range(k + 1, N):
                R[k, j] = 0.0
                for i in range(M):
                    R[k, j] += Q[i, k] * A[i, j]
                for i in range(M):
                    A[i, j] -= Q[i, k] * R[k, j]

    rtl = _to_rtl(gramschmidt)
    assert rtl.schedule().func("gramschmidt").latency is None  # range(k+1, N)

    A = (_f32(0, M, N) + np.float32(1.0)).astype(np.float32)  # keep norms nonzero
    Q = np.zeros((M, N), np.float32)
    R = np.zeros((N, N), np.float32)
    Ag = A.copy()
    Qg = np.zeros((M, N), np.float32)
    Rg = np.zeros((N, N), np.float32)
    for k in range(N):
        nrm = np.float32(0.0)
        for i in range(M):
            nrm += Ag[i, k] * Ag[i, k]
        Rg[k, k] = nrm
        for i in range(M):
            Qg[i, k] = Ag[i, k] / Rg[k, k]
        for j in range(k + 1, N):
            Rg[k, j] = np.float32(0.0)
            for i in range(M):
                Rg[k, j] += Qg[i, k] * Ag[i, j]
            for i in range(M):
                Ag[i, j] -= Qg[i, k] * Rg[k, j]
    rtl.cosim(A, Q, R)
    assert np.allclose(Q, Qg, **STOL)
    assert np.allclose(R, Rg, **STOL)


def test_durbin_recurrence():
    """Levinson-Durbin recursion: each step k reduces `sum_ += r * y` over the
    running solution (`for i in range(k)`), so the inner trip is bounded by the
    enclosing IV and the whole-function latency is unknown."""
    N = 6

    @kernel
    def durbin(r: f32[N], y: f32[N]):
        y[0] = -r[0]
        beta: f32 = 1.0
        alpha: f32 = -r[0]
        for k in range(1, N):
            beta = (1.0 - alpha * alpha) * beta
            sum_: f32 = 0.0
            z: f32[N] = 0.0
            for i in range(k):
                sum_ = sum_ + r[k - i - 1] * y[i]
            alpha = -1.0 * (r[k] + sum_)
            for i in range(k):
                z[i] = y[i] + alpha * y[k - i - 1]
            for i in range(k):
                y[i] = z[i]
            y[k] = alpha

    rtl = _to_rtl(durbin)
    assert rtl.schedule().func("durbin").latency is None  # for-range(k) inner loops

    r = (_f32(0, N) * np.float32(0.3)).astype(np.float32)  # small reflection coeffs
    y = np.zeros(N, np.float32)
    gy = np.zeros(N, np.float32)
    gy[0] = -r[0]
    al = -r[0]
    for k in range(1, N):
        s = np.float32(0.0)
        for i in range(k):
            s = s + r[k - i - 1] * gy[i]
        al = np.float32(-1.0) * (r[k] + s)
        z = gy.copy()
        for i in range(k):
            z[i] = gy[i] + al * gy[k - i - 1]
        for i in range(k):
            gy[i] = z[i]
        gy[k] = al
    rtl.cosim(r, y)
    assert np.allclose(y, gy, **STOL)


def test_adi_sweep():
    """ADI (alternating-direction implicit) solver: a Thomas-style sweep whose
    forward pass carries p[i, j-1]/q[i, j-1] and back pass carries v[j+1, i], each
    through a divide on the critical path -- a distance-1 memory recurrence, so the
    II exceeds the divide latency. All trips are constant, so the latency
    resolves."""
    TSTEPS, N = 2, 5
    DX = DY = 1.0 / N
    DT = 1.0 / TSTEPS
    mul1 = 2.0 * DT / (DX * DX)
    mul2 = 1.0 * DT / (DY * DY)
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    @kernel
    def adi(u: f32[N, N], v: f32[N, N], p: f32[N, N], q: f32[N, N]):
        for t in range(1, TSTEPS + 1):
            for i in range(1, N - 1):
                v[0, i] = 1.0
                p[i, 0] = 0.0
                q[i, 0] = v[0, i]
                for j in range(1, N - 1):
                    p[i, j] = -c / (a * p[i, j - 1] + b)
                    q[i, j] = (
                        -d * u[j, i - 1]
                        + (1.0 + 2.0 * d) * u[j, i]
                        - f * u[j, i + 1]
                        - a * q[i, j - 1]
                    ) / (a * p[i, j - 1] + b)
                v[N - 1, i] = 1.0
                for j_rev in range(N - 1):
                    j: index = N - 2 - j_rev
                    v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]
            for i in range(1, N - 1):
                u[i, 0] = 1.0
                p[i, 0] = 0.0
                q[i, 0] = u[i, 0]
                for j in range(1, N - 1):
                    p[i, j] = -f / (d * p[i, j - 1] + e)
                    q[i, j] = (
                        -a * v[i - 1, j]
                        + (1.0 + 2.0 * a) * v[i, j]
                        - c * v[i + 1, j]
                        - d * q[i, j - 1]
                    ) / (d * p[i, j - 1] + e)
                u[i, N - 1] = 1.0
                for j_rev in range(N - 1):
                    j: index = N - 2 - j_rev
                    u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]

    rtl = _to_rtl(adi)
    res = rtl.schedule()
    assert res.func("adi").latency is not None  # constant trips
    assert max(_iis(res.func("adi").cyclic())) > FDIV  # recurrence through the divide

    u = _f32(0, N, N)
    v = _f32(1, N, N)
    p = _f32(2, N, N)
    q = _f32(3, N, N)
    gu, gv, gp, gq = u.copy(), v.copy(), p.copy(), q.copy()
    for t in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            gv[0, i] = 1.0
            gp[i, 0] = 0.0
            gq[i, 0] = gv[0, i]
            for j in range(1, N - 1):
                gp[i, j] = -c / (a * gp[i, j - 1] + b)
                gq[i, j] = (
                    -d * gu[j, i - 1]
                    + (1.0 + 2.0 * d) * gu[j, i]
                    - f * gu[j, i + 1]
                    - a * gq[i, j - 1]
                ) / (a * gp[i, j - 1] + b)
            gv[N - 1, i] = 1.0
            for j in range(N - 2, -1, -1):
                gv[j, i] = gp[i, j] * gv[j + 1, i] + gq[i, j]
        for i in range(1, N - 1):
            gu[i, 0] = 1.0
            gp[i, 0] = 0.0
            gq[i, 0] = gu[i, 0]
            for j in range(1, N - 1):
                gp[i, j] = -f / (d * gp[i, j - 1] + e)
                gq[i, j] = (
                    -a * gv[i - 1, j]
                    + (1.0 + 2.0 * a) * gv[i, j]
                    - c * gv[i + 1, j]
                    - d * gq[i, j - 1]
                ) / (d * gp[i, j - 1] + e)
            gu[i, N - 1] = 1.0
            for j in range(N - 2, -1, -1):
                gu[i, j] = gp[i, j] * gu[i, j + 1] + gq[i, j]
    rtl.cosim(u, v, p, q)
    assert np.allclose(u, gu, rtol=1e-2, atol=1e-2)


def test_deriche_iir_shift_register():
    """Deriche edge filter: five sweeps of a 2nd-order IIR recurrence, each row/
    column carrying a shift register of scalars (`y[j] = a1*x + a2*xm1 + b1*ym1 +
    b2*ym2` with `xm1=x; ym2=ym1; ym1=y`). The forward/backward recurrence passes
    are bound by the carried fadd chain (II > the add latency); the two combine
    passes are dependence-free (II=1). Because a chained carry (`ym2=ym1`) is a
    distance-2 recurrence, its init must be re-injected for the first TWO
    iterations of every row -- re-injecting only the first lets the previous row's
    stale tail leak in, which shows up on rows past the first."""
    W, H = 5, 6
    alpha = 0.25
    k = (1.0 - math.exp(-alpha)) ** 2 / (
        1.0 + 2.0 * alpha * math.exp(-alpha) - math.exp(2.0 * alpha)
    )
    a1 = a5 = k
    a2 = a6 = k * math.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * math.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * math.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = -math.exp(-2.0 * alpha)
    c1 = c2 = 1.0

    @kernel
    def deriche(imgIn: f32[W, H], imgOut: f32[W, H], y1: f32[W, H], y2: f32[W, H]):
        for i in range(W):
            ym1: f32 = 0.0
            ym2: f32 = 0.0
            xm1: f32 = 0.0
            for j in range(H):
                y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
                xm1 = imgIn[i, j]
                ym2 = ym1
                ym1 = y1[i, j]
        for i in range(W):
            yp1: f32 = 0.0
            yp2: f32 = 0.0
            xp1: f32 = 0.0
            xp2: f32 = 0.0
            for j_inv in range(H):
                j: index = H - 1 - j_inv
                y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
                xp2 = xp1
                xp1 = imgIn[i, j]
                yp2 = yp1
                yp1 = y2[i, j]
        for i in range(W):
            for j in range(H):
                imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])
        for j in range(H):
            tm1: f32 = 0.0
            ym1c: f32 = 0.0
            ym2c: f32 = 0.0
            for i in range(W):
                y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1c + b2 * ym2c
                tm1 = imgOut[i, j]
                ym2c = ym1c
                ym1c = y1[i, j]
        for j in range(H):
            tp1: f32 = 0.0
            tp2: f32 = 0.0
            yp1c: f32 = 0.0
            yp2c: f32 = 0.0
            for i_inv in range(W):
                i: index = W - 1 - i_inv
                y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1c + b2 * yp2c
                tp2 = tp1
                tp1 = imgOut[i, j]
                yp2c = yp1c
                yp1c = y2[i, j]
        for i in range(W):
            for j in range(H):
                imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])

    rtl = _to_rtl(deriche)
    res = rtl.schedule()
    iis = set(_iis(res.func("deriche").cyclic()))
    assert res.func("deriche").latency is not None  # constant trips
    assert 1 in iis  # the dependence-free combine passes
    assert max(iis) > FADD  # the 2nd-order IIR recurrence passes

    img = _f32(0, W, H)
    imgOut = np.zeros((W, H), np.float32)
    y1 = np.zeros((W, H), np.float32)
    y2 = np.zeros((W, H), np.float32)

    gy1 = np.zeros((W, H))
    gy2 = np.zeros((W, H))
    go = np.zeros((W, H))
    for i in range(W):
        ym1 = ym2 = xm1 = 0.0
        for j in range(H):
            gy1[i, j] = a1 * img[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1, ym2, ym1 = img[i, j], ym1, gy1[i, j]
    for i in range(W):
        yp1 = yp2 = xp1 = xp2 = 0.0
        for j in range(H - 1, -1, -1):
            gy2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2, xp1, yp2, yp1 = xp1, img[i, j], yp1, gy2[i, j]
    go[:] = c1 * (gy1 + gy2)
    for j in range(H):
        tm1 = ym1c = ym2c = 0.0
        for i in range(W):
            gy1[i, j] = a5 * go[i, j] + a6 * tm1 + b1 * ym1c + b2 * ym2c
            tm1, ym2c, ym1c = go[i, j], ym1c, gy1[i, j]
    for j in range(H):
        tp1 = tp2 = yp1c = yp2c = 0.0
        for i in range(W - 1, -1, -1):
            gy2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1c + b2 * yp2c
            tp2, tp1, yp2c, yp1c = tp1, go[i, j], yp1c, gy2[i, j]
    go[:] = c2 * (gy1 + gy2)

    rtl.cosim(img, imgOut, y1, y2)
    assert np.allclose(imgOut, go, rtol=1e-2, atol=1e-2)

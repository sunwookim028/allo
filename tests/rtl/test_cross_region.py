# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A region's scalar result crossing to its consumer: survivor registers, cross-region value resolution, and the guard/select shapes that gate them."""

import re
import shutil

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, f32
from allo.backend.rtl import RegionKind

from _common import (  # noqa: E402
    Dcp,
    Mod,
    _to_rtl,
    _latency,
    _iis,
    _outer,
    FADD,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

_DEF = re.compile(r"^%([\w.$-]+) = (.*)$")
_COMPREG = re.compile(r'^seq\.compreg(\.ce)? (?:name "([^"]*)" )?%([\w.$-]+),')
_MUX = re.compile(r"^comb\.mux (?:bin )?%([\w.$-]+), %([\w.$-]+), %([\w.$-]+)")


def _survivors(rtl):
    """Every survivor register in the emitted module as sorted (name, shape).

    A survivor is named ``r<region>_sv<k>`` (`uarch::survivorName`). The shape
    is read off the register's spelling: ``enabled`` is a ``seq.compreg.ce``,
    a capture on its enable with no recurrence; ``latch`` is a plain register
    under ``mux(start, init, mux(capture, value, self))``, preloaded with the
    loop-carried identity so a run that never captures yields the identity
    rather than a stale prior value.
    """
    defs = {}
    for line in rtl.mlir.splitlines():
        m = _DEF.match(line.strip())
        if m:
            defs[m.group(1)] = m.group(2)
    out = []
    for ssa, rhs in defs.items():
        reg = _COMPREG.match(rhs)
        if not reg:
            continue
        name = reg.group(2) or ssa
        if "_sv" not in name:
            continue
        if reg.group(1):  # seq.compreg.ce: a plain capture, no recurrence
            out.append((name, "enabled"))
            continue
        outer = _MUX.match(defs.get(reg.group(3), ""))
        inner = _MUX.match(defs.get(outer.group(3), "")) if outer else None
        shape = "latch" if inner and inner.group(3) == ssa else "other"
        out.append((name, shape))
    return sorted(out)


# ---------------------------------------------------------------------------
# Survivor register shapes: leaf -> acyclic -> container -> while ->
# conditional-container -> guard, then the broader survivor / done mechanics
# ---------------------------------------------------------------------------


def test_leaf_reduction_survivor_preloads_its_init():
    # Leaf counted loop: the accumulator is fused into the datapath, and its
    # FINAL value is captured once, on the last iteration's issue pulse.
    # Preloaded with the reduction identity so a zero-trip run yields it.
    @kernel
    def leaf_reduce(A: i32[4, 8], B: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for j in range(8):
                acc += A[i, j]
            B[i] = acc

    rtl = _to_rtl(leaf_reduce)
    assert _survivors(rtl) == [("r1_sv0", "latch")]

    A = np.arange(32, dtype=np.int32).reshape(4, 8)
    B = np.zeros(4, dtype=np.int32)
    rtl.cosim(A, B)
    assert np.array_equal(B, A.sum(axis=1))


def test_acyclic_result_survivor_has_no_init():
    # An acyclic (straight-line) region's yield lands exactly once, so there
    # is no recurrence to preload and the survivor is the plain enabled form.
    @kernel
    def acyclic_result(A: i32[8], B: i32[8]):
        t: i32 = A[0] * 3
        for i in range(8):
            B[i] = A[i] + t

    rtl = _to_rtl(acyclic_result)
    assert _survivors(rtl) == [("r0_sv0", "enabled")]

    A = np.arange(1, 9, dtype=np.int32)
    B = np.zeros(8, dtype=np.int32)
    rtl.cosim(A, B)
    assert np.array_equal(B, A + A[0] * 3)


def test_container_carry_survivor_advances_per_outer_iteration():
    # A counted CONTAINER carrying an accumulator into an inner reduction.
    # Both halves of one recurrence are emitted, but split in two: the
    # register exists before the children emit, and its next-value is
    # back-edged in once the child that produces it has emitted.
    @kernel
    def container_carry(A: i32[4, 4], B: i32[4]):
        for i in range(4):
            outer: i32 = 0
            for j in range(4):
                inner: i32 = outer
                for k in range(4):
                    inner += A[j, k]
                outer = inner
            B[i] = outer

    rtl = _to_rtl(container_carry)
    # r1 = the j-container's carried `outer`; r2 = the k-loop's fused `inner`.
    assert _survivors(rtl) == [("r1_sv0", "latch"), ("r2_sv0", "latch")]

    A = np.arange(16, dtype=np.int32).reshape(4, 4)
    B = np.zeros(4, dtype=np.int32)
    rtl.cosim(A, B)
    # Each j pass seeds `inner` from the carried `outer`, so one whole-array sum.
    assert np.array_equal(B, np.full(4, A.sum(), dtype=np.int32))


def test_leaf_while_survivors_advance_while_continuing():
    # A leaf while: every CONTINUING iteration advances the recurrences (the
    # doomed exit iteration issues but must not commit), so the capture pulse
    # is `issue & cond` rather than the last-iteration pulse -- the same latch
    # shape on a different pulse.
    @kernel
    def leaf_while(A: i32[16]) -> i32:
        s: i32 = 0
        i: i32 = 0
        while s < 100:
            s += A[i]
            i += 1
        return s

    rtl = _to_rtl(leaf_while)
    assert _survivors(rtl) == [("r0_sv0", "latch"), ("r0_sv1", "latch")]

    A = np.full(16, 10, dtype=np.int32)
    assert rtl.cosim(A).result == 100


def test_conditional_container_survivors():
    # A conditional container (a sequential-wrapper while nesting children):
    # its iter-args are frozen survivors advanced by the children's results,
    # and the children's own results are init-less acyclic yields. Both
    # shapes in one kernel, driven by the CHECK/RUN controller.
    @kernel
    def cond_container(A: i32[8], B: i32[8]):
        n: i32 = 0
        total: i32 = 0
        while total < 50:
            s: i32 = 0
            for j in range(8):
                s += A[j]
            total += s
            B[n] = total
            n += 1

    rtl = _to_rtl(cond_container)
    # r0 = the outer while's two iter-args (n, total); r1 = the inner reduction;
    # r2 = the epilogue's two init-less yields, which feed r0's next-values.
    assert _survivors(rtl) == [
        ("r0_sv0", "latch"),
        ("r0_sv1", "latch"),
        ("r1_sv0", "latch"),
        ("r2_sv0", "enabled"),
        ("r2_sv1", "enabled"),
    ]

    A = np.ones(8, dtype=np.int32)
    B = np.zeros(8, dtype=np.int32)
    rtl.cosim(A, B)
    # total advances by 8 each pass and the loop exits once it reaches 50.
    assert np.array_equal(B, np.array([8, 16, 24, 32, 40, 48, 56, 0], np.int32))


def test_guard_result_survivor_captures_both_arms():
    # Both arms of a result guard capture into one enabled survivor: their
    # drain pulses are disjoint, so the then pulse selects the datum and their
    # OR enables the capture.
    @kernel
    def rmux(a: i32[4, 16], out: i32[4]):
        for g in range(4):
            acc: i32 = 0
            if g < 2:
                for i in range(16):
                    acc += a[g, i]
            out[g] = acc

    rtl = _to_rtl(rmux)
    # r1 latches the guard predicate (`g < 2`), r2 is the guard's one survivor
    # and r3 the guarded reduction inside the then arm.
    assert _survivors(rtl) == [
        ("r1_sv0", "enabled"),
        ("r2_sv0", "enabled"),
        ("r3_sv0", "latch"),
    ]

    a = np.arange(64, dtype=np.int32).reshape(4, 16)
    out = np.zeros(4, dtype=np.int32)
    rtl.cosim(a, out)
    assert np.array_equal(out, np.where(np.arange(4) < 2, a.sum(1), 0).astype(np.int32))


def test_scalar_recurrences():
    # Distinguish a true MEMORY recurrence from a register-based iter-arg
    # accumulator, plus a reduction handed to an epilogue store.
    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    # acc[0] &= A[i] recurs through memory: one array, both read and written.
    @kernel
    def racc(A: i32[16], acc: i32[1]):
        for i in range(16):
            acc[0] = acc[0] & A[i]

    acc = np.array([-1], np.int32)  # all bits set
    _to_rtl(racc).cosim(A16, acc)
    ref = -1
    for x in A16:
        ref &= int(x)
    assert int(acc[0]) == ref

    @kernel
    def reduce_then_store(A: i32[16], out: i32[1]):
        acc: i32 = 0
        for i in range(16):
            acc = acc | A[i]
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(reduce_then_store).cosim(A16, out)
    assert out[0] == np.bitwise_or.reduce(A16)

    # A loop-carried scalar iter_arg stored each iteration: a datapath
    # accumulator register, no memory recurrence.
    @kernel
    def orred(A: i32[16], out: i32[16]):
        acc: i32 = 0
        for i in range(16):
            acc = acc | A[i]
            out[i] = acc

    out = np.zeros(16, np.int32)
    _to_rtl(orred).cosim(A16, out)
    assert np.array_equal(out, np.bitwise_or.accumulate(A16))


def test_multi_store_deepest_drains():
    # Two stores in one region at different pipeline stages: `done` must wait
    # for the DEEPEST store, or the deeper store's tail iterations are dropped.
    # A shallow B[i]=A[i]+1 (one fadd) and a deeper C[i]=A[i]*A[i]+2 (fmul then
    # fadd). Counting store write-enables over-counts when the two retire in
    # the same cycle from different in-flight iterations.
    @kernel
    def twostore(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            B[i] = A[i] + 1.0
            C[i] = A[i] * A[i] + 2.0

    A = np.random.default_rng(0).random(8).astype(np.float32)
    B = np.zeros(8, np.float32)
    C = np.zeros(8, np.float32)
    _to_rtl(twostore).cosim(A, B, C)
    assert np.allclose(B, A + 1.0, rtol=1e-4, atol=1e-5)
    assert np.allclose(C, A * A + 2.0, rtol=1e-4, atol=1e-5)


def test_acyclic_scalar_survivors():
    # A straight-line (dcp.sequential) region can yield a value to a sibling,
    # not only retire stores: each result is captured into its own survivor
    # register and the region's done drains on the latest one.

    # A top-level prologue loads a scalar and hands it to a sibling loop.
    @kernel
    def prol(A: i32[4], out: i32[4]):
        x: i32 = A[0]
        for i in range(4):
            out[i] = x + A[i]

    A = np.arange(4, dtype=np.int32) * 7 + 3
    out = np.zeros(4, np.int32)
    _to_rtl(prol).cosim(A, out)
    assert np.array_equal(out, A[0] + A)

    # An imperfect nest whose prologue becomes an acyclic child of the outer
    # container, re-run each outer iteration and read against the freshly
    # advanced outer counter.
    @kernel
    def imperfect(A: i32[4], B: i32[4, 4], out: i32[4, 4]):
        for i in range(4):
            x: i32 = A[i]
            for j in range(4):
                out[i, j] = B[i, j] + x

    B = (np.arange(16, dtype=np.int32) * 3).reshape(4, 4)
    out = np.zeros((4, 4), np.int32)
    _to_rtl(imperfect).cosim(A, B, out)
    assert np.array_equal(out, B + A[:, None])

    # A prologue that both inits an accumulator and loads an invariant fuses
    # into ONE multi-result acyclic region yielding (0, A[0]); each result
    # gets its own survivor, and the constant identity is still re-injected as
    # the reduction init even though it now arrives as a region result.
    @kernel
    def prol_reduce(A: i32[4], out: i32[1]):
        x: i32 = A[0]
        acc: i32 = 0
        for i in range(4):
            acc = acc + A[i] * x
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(prol_reduce).cosim(A, out)
    assert out[0] == np.sum(A * A[0])


def test_scalar_return_cosim():
    # A scalar function return is a first-class output port driven by the
    # returning region's survivor, sampled at `done` -- for an int, a float
    # (decoded from its bit pattern), and a conditional container's frozen
    # iter-arg.
    N = 16

    @kernel
    def ssum(A: i32[N]) -> i32:
        s: i32 = 0
        for i in range(N):
            s = s + A[i]
        return s

    A = (np.arange(N, dtype=np.int32) * 7 + 3) & 0xFF
    r = _to_rtl(ssum).cosim(A)
    assert r.result == int(A.sum())

    @kernel
    def fsum(A: f32[8]) -> f32:
        s: f32 = 0.0
        for i in range(8):
            s = s + A[i]
        return s

    Af = np.arange(8, dtype=np.float32) + 1.0
    r = _to_rtl(fsum).cosim(Af)
    assert np.isclose(float(r.result), float(Af.sum()))

    # The returned survivor is the conditional container's frozen iter-arg.
    @kernel
    def nested(A: i32[8]) -> i32:
        total: i32 = 0
        s: i32 = 8
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        return total

    A8i = (np.arange(8, dtype=np.int32) * 3 + 1) & 0xFF
    expected = sum(int(A8i[:s].sum()) for s in range(1, 9))
    r = _to_rtl(nested).cosim(A8i)
    assert r.result == expected


def test_multiregion_latency_matches_cosim():
    # The whole-kernel latency must equal the cycle the emitter's `done`
    # actually rises. A cross-region SURVIVOR hand-off is a capture register --
    # one cycle of datapath depth does not count -- while a store-terminated
    # region hands off through memory and adds none.
    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    def survivors():  # two survivor hand-offs (s, then t) + a store region
        @kernel
        def survivors(A: i32[16], out: i32[16]):
            s: i32 = 0
            for i in range(16):
                s = s + A[i]
            t: i32 = 0
            for i in range(16):
                t = t + A[i] * s
            for i in range(16):
                out[i] = A[i] + t

        return survivors

    def stores():  # three store-terminated regions, no survivor register
        @kernel
        def stores(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
            for i in range(16):
                B[i] = A[i] + 1
            for i in range(16):
                C[i] = B[i] * 2
            for i in range(16):
                out[i] = C[i] + 3

        return stores

    lat = _latency(survivors())
    assert lat is not None  # every trip is static
    out = np.zeros(16, np.int32)
    r = _to_rtl(survivors()).cosim(A16, out)
    assert r.cycles == lat  # survivor registers counted -> no drift

    lat = _latency(stores())
    assert lat is not None
    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = _to_rtl(stores()).cosim(A16, B, C, out)
    assert r.cycles == lat  # store hand-off adds nothing -> still exact
    assert np.array_equal(out, (A16 + 1) * 2 + 3)


# A sequenced region latches no level: its completion is already a pulse, so it
# hands its successor an `r<N>_drain` register instead of a rebuilt one. Only
# the container keeps a `done`, whose level the module's own port reads.
def test_a_sequenced_region_hands_over_a_registered_pulse():
    @kernel
    def k(A: i32[16, 16], B: i32[16], C: i32[16]):
        for i in range(16):
            s: i32 = 0
            for j in range(16):
                s += A[i, j]
            B[i] = s
            t: i32 = 0
            for j2 in range(16):
                t += A[i, j2] + 1
            C[i] = t

    rtl = _to_rtl(k)
    m = Mod(rtl.mlir, "k")
    assert m.regions_with("done") == [0], m.regions_with("done")
    # The four children of the outer container: two reductions and two stores.
    assert m.regions_with("drain") == [1, 2, 3, 4], m.regions_with("drain")
    # One register per boundary, and no mux: a held level would be spelled
    # `mux(start, false, mux(set, true, done))`.
    for r in m.regions_with("drain"):
        _, inp = m.reg_named(f"r{r}_drain")
        assert not m.mux(inp), f"r{r}_drain holds a level: {m.defs[inp]}"
    # A store region launches on its predecessor's drain and commits in that
    # same cycle, so its own hand-off is that pulse one register later.
    assert m.reg_named("r2_drain")[1] == "r1_drain"

    A = (np.arange(16 * 16, dtype=np.int32) % 71).reshape(16, 16)
    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    rtl.cosim(A, B, C)
    assert np.array_equal(B, A.sum(1))
    assert np.array_equal(C, (A + 1).sum(1))


def test_independent_siblings_run_concurrently_cosim():
    # Two sibling sweeps on DISJOINT arrays (no shared memref, no survivor)
    # have no dependence, so the composer starts them together instead of
    # serializing -- verified in both the reified latency and real cycles.
    @kernel
    def indep(A: i32[64], B: i32[64], C: i32[64], D: i32[64]):
        for i in range(64):
            C[i] = A[i] + 1
        for i in range(64):
            D[i] = B[i] * 2

    rtl = _to_rtl(indep)
    # Concurrency read off the control structure rather than the clock: the
    # second region's run register is not gated on the first's done, which is
    # exactly what a serializing composer would add. Timing would say the same
    # thing, but only as a number that moves whenever the sweeps get faster.
    m = Mod(rtl.mlir, "indep")
    _, run1 = m.reg_named("r1_run")
    assert "r0_done" not in {m.hint.get(v, v) for v in m.cone(run1, limit=256)}

    A = np.arange(64, dtype=np.int32)
    B = np.arange(64, dtype=np.int32) + 100
    C = np.zeros(64, np.int32)
    D = np.zeros(64, np.int32)
    rtl.cosim(A, B, C, D)
    assert np.array_equal(C, A + 1)
    assert np.array_equal(D, B * 2)


def test_read_read_siblings_overlap_cosim():
    # Two sibling sweeps that only read one shared array have no hazard
    # between them, so the composer starts them together rather than gating
    # the second on the first's done.
    @kernel
    def rrshare(A: i32[64], C: i32[64], D: i32[64]):
        for i in range(64):
            C[i] = A[i] + 1
        for i in range(64):
            D[i] = A[i] * 2

    rtl = _to_rtl(rrshare)
    m = Mod(rtl.mlir, "rrshare")
    _, run1 = m.reg_named("r1_run")
    assert "r0_done" not in {m.hint.get(v, v) for v in m.cone(run1, limit=256)}

    lat = _latency(rrshare)
    assert lat is not None
    A = np.arange(64, dtype=np.int32)
    C = np.zeros(64, np.int32)
    D = np.zeros(64, np.int32)
    r = rtl.cosim(A, C, D)
    assert r.cycles == lat  # the overlapped span is an exact contract
    assert np.array_equal(C, A + 1)
    assert np.array_equal(D, A * 2)


def test_war_siblings_stay_ordered_cosim():
    # A reader followed by a writer of the same array is a WAR hazard: the
    # writer waits for the read sweep to drain, so its run register is gated
    # on the reader's done and the reads see the old values.
    @kernel
    def war(A: i32[64], C: i32[64]):
        for i in range(64):
            C[i] = A[i] + 1
        for i in range(64):
            A[i] = 7

    rtl = _to_rtl(war)
    m = Mod(rtl.mlir, "war")
    _, run1 = m.reg_named("r1_run")
    assert "r0_done" in {m.hint.get(v, v) for v in m.cone(run1, limit=256)}

    A = np.arange(64, dtype=np.int32)
    C = np.zeros(64, np.int32)
    rtl.cosim(A, C)
    assert np.array_equal(C, np.arange(64, dtype=np.int32) + 1)
    assert np.array_equal(A, np.full(64, 7, np.int32))


# ---------------------------------------------------------------------------
# Value resolution: one Value -> Source lookup for every slot in the L2 model,
# including the block-argument case a nested read or a loop bound resolves
# through
# ---------------------------------------------------------------------------


@kernel
def _double_plus_one(x: i32) -> i32:
    return x * 2 + 1


def test_call_scalar_reads_enclosing_iter_arg():
    # A sub-kernel call's scalar operand that is an ENCLOSING container's
    # accumulator resolves to that container's survivor register: the i-loop
    # carries `acc` and nests the j-loop, so `acc` is a latched survivor of
    # the i-region, read directly by the call two regions deeper.
    @kernel
    def call_reads_outer_carry(B: i32[4, 4]):
        acc: i32 = 1
        for i in range(4):
            for j in range(4):
                B[i, j] = _double_plus_one(acc)
            acc = acc + 1

    B = np.zeros((4, 4), dtype=np.int32)
    rtl = _to_rtl(call_reads_outer_carry)
    rtl.cosim(B)
    # Row i sees acc == 1 + i (the survivor advances on each outer drain).
    expected = np.tile((2 * (1 + np.arange(4)) + 1).reshape(4, 1), (1, 4))
    assert np.array_equal(B, expected.astype(np.int32))


def test_reduction_init_from_peeled_prologue():
    # An accumulator seeded from the enclosing loop's IV needs a cast, which
    # the reifier peels into a prologue sub-region: the init the resolver
    # sees is that region's RESULT (a survivor), never the raw counter block
    # argument.
    @kernel
    def init_from_iv(A: i32[4, 8], B: i32[4]):
        for i in range(4):
            acc: i32 = i
            for j in range(8):
                acc += A[i, j]
            B[i] = acc

    A = np.arange(32, dtype=np.int32).reshape(4, 8)
    B = np.zeros(4, dtype=np.int32)
    rtl = _to_rtl(init_from_iv)
    rtl.cosim(A, B)
    assert np.array_equal(B, np.arange(4) + A.sum(axis=1))


def test_loop_bound_from_enclosing_counter():
    # A runtime loop bound (a triangular loop) reads the enclosing loop's
    # COUNTER directly: the other block-argument arm, kept next to the
    # iter-arg reads above since the two are one `if` apart in the resolver.
    @kernel
    def triangular(A: i32[8, 8], B: i32[8]):
        for i in range(8):
            s: i32 = 0
            for j in range(i, 8):
                s += A[i, j]
            B[i] = s

    A = np.arange(64, dtype=np.int32).reshape(8, 8)
    B = np.zeros(8, dtype=np.int32)
    rtl = _to_rtl(triangular)
    rtl.cosim(A, B)
    assert np.array_equal(B, np.array([A[i, i:].sum() for i in range(8)]))


# ---------------------------------------------------------------------------
# Boundary cones: what a TOP-LEVEL loop's bound or a guard's predicate becomes
# when it is an expression. `expand-region-bounds` reifies it before the solve,
# so it lands in the enclosing straight-line region and nothing is left loose
# in the kernel body.
# ---------------------------------------------------------------------------


def _loose_ops(func):
    """The ops in ``func``'s own body that belong to no region. Constants are
    excluded (they tie in as literal cells wherever they sit), as are the region
    ops and the terminator. Empty is the invariant: an expression a region's
    boundary reads is scheduled, so it sits inside a region like anything
    else."""
    body = func.root.regions[0].blocks[0]
    return [
        o.operation.name
        for o in body.operations
        if not o.operation.name.startswith("allo.dcp.")
        and o.operation.name != "arith.constant"
    ]


def test_loop_bound_cone_is_a_scheduled_region():
    # A top-level window loop: `hi + 1` is an affine expression of a scalar
    # argument, so it is reified into an `arith.addi` the solve places, and the
    # loop reads the region's survivor. The bound resolves
    # through it, and the loop runs [lo, hi].
    @kernel
    def windowed(src: i32[16], dst: i32[16], lo: i32, hi: i32):
        for i in range(lo, hi + 1):
            dst[i] = src[i]

    rtl = _to_rtl(windowed)
    # Locked, so the test cannot quietly stop covering the shape if a boundary
    # expression ever escapes back out of a region.
    assert _loose_ops(Dcp(rtl).func("windowed")) == []
    assert any(r.has("addi") for r in rtl.schedule().func("windowed").regions)

    src = (np.arange(16, dtype=np.int32) + 3) * 5
    dst = np.zeros(16, np.int32)
    rtl.cosim(src, dst, np.int32(2), np.int32(9))
    expected = np.zeros(16, np.int32)
    expected[2:10] = src[2:10]  # inclusive of `hi`, and nothing outside
    assert np.array_equal(dst, expected)


def test_guard_predicate_cone_is_a_scheduled_region():
    # The same cone in the other slot: a top-level `if` over a scalar argument
    # closes into a `dcp.select` whose predicate is an `arith.cmpi` the solve
    # placed. The guard must gate its arm, both ways.
    @kernel
    def guarded(flag: i32, out: i32[16]):
        for i in range(16):
            out[i] = i
        if flag == 0:
            for j in range(16):
                out[j] = 99

    rtl = _to_rtl(guarded)
    assert _loose_ops(Dcp(rtl).func("guarded")) == []
    assert any(r.has("cmpi") for r in rtl.schedule().func("guarded").regions)

    taken = np.zeros(16, np.int32)
    rtl.cosim(np.int32(0), taken)
    assert np.array_equal(taken, np.full(16, 99, np.int32))

    skipped = np.zeros(16, np.int32)
    _to_rtl(guarded).cosim(np.int32(1), skipped)
    assert np.array_equal(skipped, np.arange(16, dtype=np.int32))


def test_bound_cone_carries_its_composition_edge():
    # The cone's one timing obligation. The consuming loop's ONLY tie to the
    # region computing its bound is that bound: start it with the kernel and it
    # reads the survivor before it settles. The `j` loop sweeps a disjoint array
    # and must stay concurrent, so this is a lock on the edge being exactly the
    # one the bound implies and no other.
    @kernel
    def scoped_bound(a: i32[16], b: i32[16], n: i32):
        for j in range(16):
            a[j] = j
        # `i + 1`, so a written cell is never the 0 an untouched one reads back
        # as: the trip count is then visible in the result.
        for i in range(0, n + 1):
            b[i] = i + 1

    rtl = _to_rtl(scoped_bound)
    assert _loose_ops(Dcp(rtl).func("scoped_bound")) == []

    # Read off the control structure, the same way the concurrent-siblings
    # lock reads it: what the consuming region's run register is gated on. r1 is
    # the `j` loop, r2 the bound, r3 the loop reading it.
    m = Mod(rtl.mlir, "scoped_bound")
    _, run3 = m.reg_named("r3_run")
    cone = {m.hint.get(v, v) for v in m.cone(run3, limit=256)}
    assert "r2_done" in cone  # gated on the region that latches the bound
    assert "r1_done" not in cone  # and on nothing else

    a = np.zeros(16, np.int32)
    b = np.zeros(16, np.int32)
    rtl.cosim(a, b, np.int32(6))
    assert np.array_equal(a, np.arange(16, dtype=np.int32))
    expected = np.zeros(16, np.int32)
    expected[:7] = np.arange(1, 8)  # exactly n+1 iterations, and no more
    assert np.array_equal(b, expected)


# ---------------------------------------------------------------------------
# Cross-region hand-off: regression witnesses for the survivor path. A sibling
# region can only reach a value through the formal-result path (SSA dominance
# forbids reading another region's inner value), while an enclosing-to-nested
# read is the only shape that could reach the defensive reject; both lower and
# cosim correctly, the reifier threading every escaping scalar as a formal dcp
# result.
# ---------------------------------------------------------------------------


def _no_cross_region_reject(rtl) -> bool:
    """True iff compilation does NOT hit the "cross-region value hand-off not
    yet supported" reject, the defensive backstop for a datapath operand whose
    producer lives in a different region than its consumer and is not a
    formal region result."""
    try:
        rtl.compile()
        return True
    except RuntimeError as e:
        return "cross-region value hand-off" not in str(e)


def test_cross_region_sibling_scalar_cosim():
    # A scalar reduction result computed in one region, read by a later
    # SIBLING region: region P (a reduction loop) yields `s` as a formal
    # result, and region Q reads it as a survivor, latched at P's completion
    # and held while Q runs.
    @kernel
    def sibling_scalar(A: i32[16], out: i32[16]):
        s: i32 = 0
        for i in range(16):
            s += A[i]  # region P: reduction -> scalar survivor `s`
        for j in range(16):
            out[j] = A[j] + s  # region Q (sibling): consumes `s`

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    rtl = _to_rtl(sibling_scalar)
    # The scalar crosses as a formal region result (the survivor path), so the
    # reject is not hit and the value is not spilled to memory.
    assert _no_cross_region_reject(rtl)
    rtl.cosim(A, out)
    assert np.array_equal(out, A + A.sum())


def test_cross_region_enclosing_invariant_cosim():
    # A scalar computed once per outer-loop iteration, read inside the nested
    # inner loop: imperfect-nest decomposition peels the computation into a
    # prologue sub-region sibling of the inner loop, so it becomes a formal
    # result read as a survivor rather than a raw enclosing-region reference.
    @kernel
    def enclosing_invariant(A: i32[8, 8], B: i32[8, 8]):
        for i in range(8):
            s: i32 = A[i, 0] + 1  # enclosing-body scalar, per outer iter
            for j in range(8):
                B[i, j] = A[i, j] * s  # inner loop consumes `s`

    A = np.arange(64, dtype=np.int32).reshape(8, 8)
    B = np.zeros((8, 8), dtype=np.int32)
    rtl = _to_rtl(enclosing_invariant)
    assert _no_cross_region_reject(rtl)
    rtl.cosim(A, B)
    assert np.array_equal(B, A * (A[:, 0:1] + 1))


def test_loop_carry_from_a_resident_source():
    # A recurrence whose next value is resident: a literal, a scalar argument or
    # an upstream region's result does not move while the loop runs, so every
    # iteration past the first reads it off a wire and only iteration 0 takes
    # the init. The three arrive through different channels (a constant cell, an
    # input port, a survivor).
    A = np.arange(10, 18, dtype=np.int32)

    @kernel
    def from_literal(A: i32[8], B: i32[8]):
        prev: i32 = 0
        for i in range(8):
            B[i] = prev + A[i]
            prev = 3

    B = np.zeros(8, np.int32)
    _to_rtl(from_literal).cosim(A, B)
    assert np.array_equal(B, A + np.array([0] + [3] * 7, np.int32))

    @kernel
    def from_argument(A: i32[8], B: i32[8], c: i32):
        prev: i32 = 0
        for i in range(8):
            B[i] = prev + A[i]
            prev = c

    B = np.zeros(8, np.int32)
    _to_rtl(from_argument).cosim(A, B, np.int32(5))
    assert np.array_equal(B, A + np.array([0] + [5] * 7, np.int32))

    @kernel
    def from_survivor(A: i32[8], B: i32[8]):
        s: i32 = 0
        for j in range(8):
            s += A[j]
        prev: i32 = 0
        for i in range(8):
            B[i] = prev + A[i]
            prev = s

    B = np.zeros(8, np.int32)
    _to_rtl(from_survivor).cosim(A, B)
    assert np.array_equal(B, A + np.array([0] + [A.sum()] * 7, np.int32))


def test_carried_identity_reaches_a_store():
    # The identity is re-injected at the consumer, since the recurrence register
    # may sit anywhere in the cycle, and a store is a consumer with no operator
    # input port to hold one. Reading the carry straight into a store is the
    # shape that drops it, writing the previous iteration's datum on iteration 0.
    @kernel
    def delay_line(A: i32[8], B: i32[8]):
        p: i32 = 7
        for i in range(8):
            B[i] = p
            p = A[i]

    A = np.arange(10, 18, dtype=np.int32)
    B = np.zeros(8, np.int32)
    _to_rtl(delay_line).cosim(A, B)
    assert np.array_equal(B, np.concatenate(([7], A[:-1])))


def test_chained_carry_reads_one_identity_per_iteration():
    # `p2 = p1; p1 = A[i]` shifts one carry into the next, so p2 reaches back
    # two iterations and its first two read the inits down that chain, one each,
    # not the outermost one twice. Equal inits hide the difference, so each stage
    # starts from its own value; a third stage locks that the chain is walked
    # rather than two being a special case.
    A = np.arange(10, 18, dtype=np.int32)

    @kernel
    def shift2(A: i32[8], B: i32[8]):
        p1: i32 = 1
        p2: i32 = 2
        for i in range(8):
            B[i] = p2
            p2 = p1
            p1 = A[i]

    B = np.zeros(8, np.int32)
    _to_rtl(shift2).cosim(A, B)
    assert np.array_equal(B, np.concatenate(([2, 1], A[:-2])))

    @kernel
    def shift3(A: i32[8], B: i32[8]):
        p1: i32 = 1
        p2: i32 = 2
        p3: i32 = 3
        for i in range(8):
            B[i] = p3
            p3 = p2
            p2 = p1
            p1 = A[i]

    B = np.zeros(8, np.int32)
    _to_rtl(shift3).cosim(A, B)
    assert np.array_equal(B, np.concatenate(([3, 2, 1], A[:-3])))

    # Fibonacci seeds its two carries differently, so reading one identity twice
    # emits 0, 0, ...
    @kernel
    def fib(B: i32[10]):
        a: i32 = 0
        b: i32 = 1
        for i in range(10):
            B[i] = a
            c: i32 = a + b
            a = b
            b = c

    B = np.zeros(10, np.int32)
    _to_rtl(fib).cosim(B)
    assert np.array_equal(B, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])


# ---------------------------------------------------------------------------
# Guard-select control: a guard that can neither be predicated nor folded into
# a loop bound closes into `dcp.select`, gating exactly its own guarded store
# -- one test per shape (data-dependent, affine two-constraint, non-spanning),
# each combining the structural lock with its correctness cosim
# ---------------------------------------------------------------------------


def test_data_dependent_guard_closes_into_select_and_gates_its_store():
    # A guard whose predicate reads memory (not affine in the IV) cannot be
    # predicated or folded into a loop bound, so it survives to the reifier as
    # a dcp.select; the guarded store fires only where the predicate holds.
    N, M = 8, 4

    @kernel
    def cond_reduce(A: f32[N, M], flag: i32[M], out: f32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: f32 = 0.0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    mod = _to_rtl(cond_reduce)
    res = mod.schedule()
    assert _iis(res.func("cond_reduce").cyclic()) == [FADD]  # guarded reduction
    guard = next(r for r in res.funcs[0].regions if r.kind == "guard")
    assert guard.conditional and guard.container

    A = (np.arange(N * M, dtype=np.float32) * 0.1).reshape(N, M)
    flag = np.array([1, 0, 1, 0], dtype=np.int32)
    out = np.zeros(M, np.float32)
    mod.cosim(A, flag, out)
    golden = np.where(flag > 0, A.sum(axis=0), 0.0).astype(np.float32)
    assert np.allclose(out, golden, rtol=1e-3, atol=1e-3)
    # The guarded-false columns are untouched (0), not the reduction of an
    # ungated store (which would leak the previous column's acc into them).
    assert out[1] == 0.0 and out[3] == 0.0


def test_affine_two_constraint_guard_closes_into_select():
    # A two-constraint affine guard (`i>j and i<j+4`) cannot tighten to a
    # single loop bound, so the reifier materializes its IntegerSet predicate
    # into an i1 and closes it into a dcp.select; the memory-carried store
    # inside fires only where the predicate holds.
    N = M = 8

    @kernel
    def agf(x: f32[N], out: f32[N]):
        for i in range(N):
            for j in range(N):
                if i > j and i < j + 4:
                    for k in range(M):
                        out[i] += x[j]

    mod = _to_rtl(agf)
    res = mod.schedule()
    # Phase A lifts the IntegerSet predicate into start-0 dcp.compute units (the
    # conjunction `andi` of two `sge` compares, predicate 5), so the guard
    # condition is a first-class Source -- no raw arith.cmpi/andi survives for
    # the emitter to re-interpret.
    assert any(r.has("andi") for r in res.regions(wrappers=True))
    assert Dcp(mod).func("agf").attrs("allo.dcp.compute", "predicate").count(5) >= 2
    guard = next(r for r in res.funcs[0].regions if r.kind == "guard")
    assert guard.conditional and guard.container
    assert _iis(res.cyclic()) == [FADD]  # `out[i] +=` raised to a register recurrence

    x = np.arange(N, dtype=np.float32) * 0.1 + 1.0
    out = np.zeros(N, np.float32)
    golden = np.zeros(N, np.float32)
    for i in range(N):
        for j in range(N):
            if i > j and i < j + 4:
                golden[i] += M * x[j]
    mod.cosim(x, out)
    assert np.allclose(out, golden, rtol=1e-3, atol=1e-3)


def test_guard_gates_only_its_own_store_not_a_sibling():
    # An affine guard that does not span its enclosing loop body (a trailing
    # store follows it) keeps the `j` loop an imperfect wrapper rather than a
    # flattenable band, and its dcp.select gates exactly its own guarded
    # store -- a sibling store outside the guard fires unconditionally.
    N = M = 8

    @kernel
    def imp(A: f32[N, M], B: f32[M, N], out: f32[N, N], C: f32[N, N]):
        for i in range(N):
            for j in range(N):
                if i > j:
                    acc: f32 = 0.0
                    for k in range(M):
                        acc += A[i, k] * B[k, j]
                    out[i, j] = acc
                C[i, j] = 1.0  # trailing store -> guard does not span the body

    mod = _to_rtl(imp)
    res = mod.schedule()
    # Scalar-carried reduction inside the guard -> register recurrence (II=FADD).
    assert _iis(res.cyclic()) == [FADD]
    assert any(r.kind == "guard" for r in res.funcs[0].regions)

    A = (np.arange(N * M, dtype=np.float32) * 0.05).reshape(N, M)
    B = (np.arange(M * N, dtype=np.float32) * 0.03).reshape(M, N)
    out = np.zeros((N, N), np.float32)
    C = np.zeros((N, N), np.float32)
    mod.cosim(A, B, out, C)
    assert np.allclose(out, np.tril(A @ B, -1), rtol=1e-2, atol=1e-2)
    assert np.allclose(C, np.ones((N, N), np.float32))


def test_result_mux_select():
    # A dcp.select with a non-empty else / yielded results: both arms run
    # mutually-exclusively under the predicate, and a yielded value is muxed
    # `cond ? then : else`. Covers a dual guard (a store loop in each arm) and
    # a result-mux (a guarded reduction yielding an accumulator).
    N = 16

    # Dual guard, affine predicate: the taken arm's store loop fires, the
    # other's never issues.
    @kernel
    def dual_affine(a: i32[4, N], b: i32[4, N], out: i32[4, N]):
        for g in range(4):
            if g < 2:
                for i in range(N):
                    out[g, i] = a[g, i] + 1
            else:
                for i in range(N):
                    out[g, i] = b[g, i] * 2

    mod = _to_rtl(dual_affine)
    assert mod.schedule().regions(RegionKind.GUARD, wrappers=True)
    a = np.arange(4 * N, dtype=np.int32).reshape(4, N)
    b = a + 1000
    out = np.zeros((4, N), np.int32)
    mod.cosim(a.copy(), b.copy(), out)
    assert np.array_equal(out, np.where(np.arange(4).reshape(4, 1) < 2, a + 1, b * 2))

    # Dual guard, data-dependent predicate (a ping-pong `if sel[g]==0: ... else`):
    # the predicate reads memory, lifting to a settled-survivor dcp.compute.
    @kernel
    def dual_ddep(sel: i32[4], a: i32[4, N], b: i32[4, N], out: i32[4, N]):
        for g in range(4):
            if sel[g] == 0:
                for i in range(N):
                    out[g, i] = a[g, i] + 1
            else:
                for i in range(N):
                    out[g, i] = b[g, i] * 2

    sel = np.array([0, 1, 0, 1], dtype=np.int32)
    out = np.zeros((4, N), np.int32)
    _to_rtl(dual_ddep).cosim(sel, a.copy(), b.copy(), out)
    assert np.array_equal(out, np.where(sel.reshape(4, 1) == 0, a + 1, b * 2))

    # Result-mux: the guard wraps a reduction loop and yields the accumulator;
    # the empty else passes the initial value through -> `cond ? sum : 0`.
    @kernel
    def rmux(a: i32[4, N], out: i32[4]):
        for g in range(4):
            acc: i32 = 0
            if g < 2:
                for i in range(N):
                    acc += a[g, i]
            out[g] = acc

    mod = _to_rtl(rmux)
    assert mod.schedule().regions(RegionKind.GUARD, wrappers=True)
    out = np.zeros(4, np.int32)
    mod.cosim(a.copy(), out)
    assert np.array_equal(out, np.where(np.arange(4) < 2, a.sum(1), 0).astype(np.int32))


def test_container_and_guard_regions_never_own_loose_datapath():
    # Container / guard regions must not own loose datapath ops: each
    # container path emits only part of the datapath, so a store bound into
    # one of those regions would be silently dropped rather than diagnosed.
    # What keeps them empty is the scheduler's imperfect-nest and branch
    # decomposition, which wraps every straight-line span in its own child
    # region; each kernel here is cosimmed, since a dropped store is invisible
    # in the schedule and shows up only as a wrong array.

    # An epilogue store after an inner reduction -- loose in the `i` container
    # unless the imperfect-nest decomposition wraps it.
    @kernel
    def epilogue(A: i32[4, 4], B: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(4):
                s += A[i, j]
            B[i] = s

    A2d = (np.arange(16, dtype=np.int32) * 3 + 1).reshape(4, 4) & 0x3F
    B = np.zeros(4, np.int32)
    _to_rtl(epilogue).cosim(A2d, B)
    assert np.array_equal(B, A2d.sum(axis=1))

    # A store loose in a `while` body: the conditional-container path emits
    # only the condition cone, so this store lives or dies by the while-body
    # decomposition.
    @kernel
    def while_store(A: i32[8], B: i32[8]):
        i: i32 = 0
        while i < 8:
            B[i] = A[i] * 2
            i += 1

    A8 = (np.arange(8, dtype=np.int32) * 5 + 2) & 0x3F
    Bw = np.zeros(8, np.int32)
    _to_rtl(while_store).cosim(A8, Bw)
    assert np.array_equal(Bw, A8 * 2)

    # A dual guard whose arms each mix a loop with a loose store -- the guard
    # path emits no datapath of its own, so both arms must decompose into
    # child regions.
    @kernel
    def dual_guard(A: i32[8], B: i32[8], C: i32[8]):
        for i in range(8):
            if A[i] > 0:
                for j in range(2):
                    C[j] = A[i]
                B[i] = 1
            else:
                for j in range(2):
                    C[j] = 0
                B[i] = 2

    Ag = np.array([1, 0, 3, 0, 5, 0, 7, 0], np.int32)
    Bg = np.zeros(8, np.int32)
    Cg = np.zeros(8, np.int32)
    _to_rtl(dual_guard).cosim(Ag, Bg, Cg)
    assert np.array_equal(Bg, np.where(Ag > 0, 1, 2))


# ---------------------------------------------------------------------------
# Scalar hand-off through calls: a call's scalar operands and results are
# ordinary datapath Sources -- a constant, a sibling's live result, or a real
# cross-region survivor -- resolved the same way a loop's own scalars are
# ---------------------------------------------------------------------------


def test_call_scalar_operand_is_a_constant():
    # A call's scalar operand can be a plain constant, resolved straight into
    # the child's scalar port rather than a survivor or IO read.
    @kernel
    def sch_child(A: i32[16], B: i32[16], s: i32):
        for i in range(16):
            B[i] = A[i] + s  # a scalar operand feeds the child's scalar port

    @kernel
    def sch_top(A: i32[16], out: i32[16]):
        B: i32[16]
        sch_child(A, B, 3)
        for i in range(16):
            out[i] = B[i] * 2

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    _to_rtl(sch_top).cosim(A, out)
    assert np.array_equal(out, (A + 3) * 2)


def test_scalar_result_consumed_by_a_sibling_call():
    # The canonical scalar hand-off: one child returns a scalar (a reduction
    # over an internal buffer), a sibling child consumes it in the same
    # region. The result crosses as a survivor: captured at start+latency,
    # then read as a live value.
    @kernel
    def accum(B: i32[16]) -> i32:
        s: i32 = 0
        for i in range(16):
            s += B[i]  # a scalar result over an internal buffer
        return s

    @kernel
    def scale(s: i32, out: i32[16]):
        for i in range(16):
            out[i] = s * 2  # a scalar operand from the sibling's result

    @kernel
    def top(A: i32[16], out: i32[16]):
        B: i32[16]
        for i in range(16):  # loose region -> mixed container
            B[i] = A[i] + 1
        s: i32 = accum(B)  # scalar result (internal buffer B)
        scale(s, out)  # scalar operand handoff -> boundary out

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    rtl = _to_rtl(top)
    assert Dcp(rtl).func(rtl.top).callees()  # the leaf CallUnit path
    rtl.cosim(A, out)
    s = int((A + 1).sum())  # 120 + 16 = 136
    assert np.array_equal(out, np.full(16, s * 2, dtype=np.int32))


def test_scalar_result_crosses_an_intervening_region():
    # A scalar result that ESCAPES past an intervening loose region: the two
    # calls land in separate regions, so the result crosses as a real
    # survivor register rather than a same-region live read.
    @kernel
    def accum(B: i32[16]) -> i32:
        s: i32 = 0
        for i in range(16):
            s += B[i]
        return s

    @kernel
    def bias(s: i32, C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + s  # a scalar operand from an earlier region's result

    @kernel
    def top(A: i32[16], out: i32[16]):
        B: i32[16]
        C: i32[16]
        for i in range(16):  # loose region 0
            B[i] = A[i] + 1
        s: i32 = accum(B)  # result ESCAPES (consumed after the C region)
        for i in range(16):  # intervening loose region -> separate regions
            C[i] = A[i] * 2
        bias(s, C, out)

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    _to_rtl(top).cosim(A, out)
    s = int((A + 1).sum())  # 136
    assert np.array_equal(out, A * 2 + s)


def test_call_multi_scalar_result():
    # A sub-kernel returning TWO scalars: each result is keyed separately, so
    # the call gets its own child result port per result rather than sharing
    # one.
    @kernel
    def mr_child(x: i32) -> (i32, i32):
        return x + 1, x * 2

    @kernel
    def mr_top(x: i32, out: i32[2]):
        a, b = mr_child(x)
        out[0] = a
        out[1] = b

    rtl = _to_rtl(mr_top)
    # One instance with two results, and two result ports on the child module.
    (call,) = Dcp(rtl).func("mr_top").ops("allo.dcp.instance")
    assert len(call.results) == 2
    out = np.zeros(2, dtype=np.int32)
    rtl.cosim(np.int32(7), out)
    assert np.array_equal(out, np.array([8, 14], dtype=np.int32))


def test_call_scalar_result_consumed_in_its_own_region():
    # A single-result call whose result is consumed by loose ops in the
    # calling region: a store at the call's ready cycle, and a compute one
    # cycle later that needs a delay chain off the call's output. Both emit
    # after the call, and the region's `done` waits for both.
    @kernel
    def sr_child(x: i32) -> i32:
        return x + 1

    @kernel
    def sr_top(x: i32, out: i32[2]):
        a: i32 = sr_child(x)
        out[0] = a
        out[1] = a * 3

    # The complement of test_indeterminate_calls.py, which splits the same shape
    # in two: a DETERMINATE callee needs no isolation, so both consumers stay in
    # the caller's one region. The callee is a separate kernel and a nested
    # region is reported deeper, so neither can add to this count.
    caller = _to_rtl(sr_top).schedule().func("sr_top")
    assert len(_outer(caller, RegionKind.ACYCLIC)) == 1

    out = np.zeros(2, dtype=np.int32)
    _to_rtl(sr_top).cosim(np.int32(7), out)
    assert np.array_equal(out, np.array([8, 24], dtype=np.int32))


def test_call_result_yielded_beside_another_value():
    # A survivor is keyed by the region result it is yielded as, which matches
    # the call's own result index only when the call is all its region yields.
    # Here the region also yields the literal the loop scales by, so the call's
    # one result is region result 1.
    @kernel
    def summed(A: i32[16]) -> i32:
        s: i32 = 0
        for i in range(16):
            s += A[i]
        return s

    @kernel
    def scale_by_sum(A: i32[16], out: i32[16]):
        s: i32 = summed(A)
        for i in range(16):
            out[i] = s * 2  # `2` escapes its region alongside `s`

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    _to_rtl(scale_by_sum).cosim(A, out)
    assert np.array_equal(out, np.full(16, int(A.sum()) * 2, dtype=np.int32))

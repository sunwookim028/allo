# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dcp.determinacy classification"""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, f32, index, Stream
from allo.backend.rtl import LatencyModelWarning, RegionKind

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, Mod, _to_rtl, _sched, _latency, _outer  # noqa: E402

# Applied per test rather than to the module: the taxonomy half never
# simulates, and must keep running where verilator is absent.
needs_verilator = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


# The whole-kernel value is behavior-load-bearing: DataflowTop's
# `calleeDeterminate` reads it, so a callee is a static-offset producer iff it
# is `counted_static`. The region value is the declared controller-regime
# discriminant.
def _regs(result) -> set:
    """Every REGION determinacy the schedule declares."""
    return {r.determinacy for r in result.regions(wrappers=True)}


def _kernels(result) -> set:
    """Every WHOLE-KERNEL determinacy the schedule declares. Kept apart from
    the region classes above, which draw on the same four keywords: a region's
    own class must not be able to satisfy a claim about its kernel."""
    return {f.determinacy for f in result.funcs}


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------


# A counted loop's region and its enclosing kernel are both counted_static.
def test_counted_loop_and_kernel_are_counted_static():
    @kernel
    def leaf(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    res = _sched(leaf)
    assert res.cyclic() and _regs(res) == {"counted_static"}
    assert _kernels(res) == {"counted_static"}


# A static sequential composition: the container and both leaves are exact,
# so a caller can release a consumer at a static offset.
def test_sequential_container_is_counted_static():
    @kernel
    def sc1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def sc2(B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = B[i] * 2

    @kernel
    def seq_top(A: i32[16], B: i32[16], out: i32[16]):
        sc1(A, B)
        sc2(B, out)

    res = _sched(seq_top)
    assert _kernels(res) == {"counted_static"}
    # a plain call graph is not a dataflow spawn
    assert "concurrent" not in _regs(res)


# A genuine (data-dependent-exit) while flushing-pipelines -> its region is
# conditional; the kernel's total latency is unknown -> indeterminate. The exit
# test `acc < limit` is combinational over a data-advanced accumulator, so the
# trip is not statically known (a counted test like `x > 1` would raise instead).
def test_data_dependent_while_is_conditional():
    @kernel
    def wr(A: i32[16], limit: i32, out: i32[1]):
        acc: i32 = 0
        c: i32 = 0
        while acc < limit:
            acc = acc + A[c]
            c = c + 1
        out[0] = c

    res = _sched(wr)
    assert "conditional" in _regs(res)
    assert "indeterminate" in _kernels(res)


# A data-dependent guard closes into a dcp.select -> conditional.
def test_guard_select_is_conditional():
    N, M = 8, 4

    @kernel
    def cond_reduce(A: f32[N, M], flag: i32[M], out: f32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: f32 = 0.0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    res = _sched(cond_reduce)
    assert res.regions(RegionKind.GUARD, wrappers=True) and "conditional" in _regs(res)
    assert "indeterminate" in _kernels(res)


# A dynamic outer trip has no exact latency -> the wrapper region and the
# kernel are both indeterminate (a bounded or unknown span, so no consumer
# can be placed at a static offset).
def test_dynamic_trip_wrapper_is_indeterminate():
    N = 4

    @kernel
    def band(A: f32[N, N], y: f32[N], n: index):
        for i in range(n):
            for j in range(N):
                y[i] += A[i, j]

    res = _sched(band)
    assert "indeterminate" in _regs(res)
    assert "indeterminate" in _kernels(res)


# An await-spawned dataflow container is concurrent (self-timed) -> a caller
# waits on its real done, never a static offset. The spawned leaves are
# INDETERMINATE, and for the same reason the container is: each one puts to or
# gets from a channel, so a full queue or a starved input stretches its run by an
# amount no schedule names. The span each composes is a floor, and publishing it
# as a contract is what let a stream kernel claim 19 cycles and measure 28.
def test_async_dataflow_container_is_concurrent():
    N = 16

    @kernel
    async def dp(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def dc(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def dtop(out: i32[N]):
        fifo: Stream[i32]
        await dp(fifo)
        await dc(fifo, out)

    res = _sched(dtop)
    # every leaf is back-pressured, so nothing here is counted_static
    assert _kernels(res) == {"concurrent", "indeterminate"}
    assert _latency(dp) is None and _latency(dc) is None


# ---------------------------------------------------------------------------
# The published contract
# ---------------------------------------------------------------------------


# The manifest republishes the kernel's determinacy class and span, and agrees
# with the schedule report it is stamped from.
def test_the_manifest_publishes_the_kernel_contract():
    N = 4

    @kernel
    def counted(A: i32[N], B: i32[N]):
        for i in range(N):
            B[i] = A[i] + 1

    @kernel
    def dynamic(A: i32[N], B: i32[N], n: index):
        for i in range(n):
            B[i] = A[i] + 1

    for k, determinacy, exact in (
        (counted, "counted_static", True),
        (dynamic, "indeterminate", False),
    ):
        rtl = _to_rtl(k)
        iface = rtl.interfaces.of_symbol(rtl.top)
        fn = rtl.schedule().func(rtl.top)
        assert iface.determinacy == determinacy
        assert iface.latency_is_exact is exact
        assert (iface.latency is not None) is exact
        assert (iface.latency, iface.latency_is_bound) == (
            fn.latency,
            fn.latency_is_bound,
        )


# The cosim latency oracle, driven with a measurement rather than a real run.
# A run that outlasts the published span raises, since a caller time-triggered
# against the figure would sample early; one that beats it only warns.
def test_the_latency_oracle_fails_only_the_unsound_direction():
    @kernel
    def counted(A: i32[8], B: i32[8]):
        for i in range(8):
            B[i] = A[i] + 1

    rtl = _to_rtl(counted)
    span = rtl.interfaces.of_symbol(rtl.top).latency
    # pylint: disable=protected-access
    rtl._check_latency(span)
    with pytest.raises(RuntimeError, match="UNSOUND"):
        rtl._check_latency(span + 1)
    with pytest.warns(LatencyModelWarning):
        rtl._check_latency(span - 1)


# A guard publishes the deeper arm's span as a bound, so the kernel keeps a
# waitable figure. The bound counts every done-latch cycle while the hardware
# advances the enclosing loop on the guard's completion pulse, so a deeper-arm
# run lands one cycle per iteration under the bound.
@needs_verilator
def test_a_guard_publishes_the_deeper_arms_span_as_a_bound():
    N, M = 8, 4

    @kernel
    def gsum(A: i32[N, M], flag: i32[M], out: i32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: i32 = 0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    rtl = _to_rtl(gsum)
    fn = rtl.schedule().func(rtl.top)
    assert fn.latency is not None and fn.latency_is_bound
    assert fn.determinacy == "indeterminate"  # waitable, never time-triggered
    guard = rtl.schedule().regions(RegionKind.GUARD, wrappers=True)
    assert guard and all(r.latency and r.latency_is_bound for r in guard)
    iface = rtl.interfaces.of_symbol(rtl.top)
    assert (iface.latency, iface.latency_is_bound) == (fn.latency, True)

    A = np.arange(N * M, dtype=np.int32).reshape(N, M)
    out = np.zeros(M, np.int32)
    taken = rtl.cosim(A, np.ones(M, np.int32), out)
    # Every guard takes the deeper arm; the recovered latch per turnover is
    # the only slack under the bound.
    assert taken.cycles == fn.latency - M
    assert np.array_equal(out, A.sum(0))
    out = np.zeros(M, np.int32)
    skipped = rtl.cosim(A, np.zeros(M, np.int32), out)
    assert skipped.cycles < taken.cycles  # empty arms complete early


# A result-less guard hands its successor its completion pulse and the
# enclosing container advances on its last child's pulse, so each iteration
# runs two latch cycles under the published bound.
@needs_verilator
def test_a_result_less_guards_pulse_chains_through_the_iteration():
    N, M = 5, 6

    @kernel
    def gseq(A: i32[N, M], out: i32[M]):
        t: i32 = 1  # runtime-carried, so the guard cannot fold; always true
        for j in range(M):
            if t > 0:
                acc: i32 = 0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc
            t = t + 1

    rtl = _to_rtl(gseq)
    fn = rtl.schedule().func(rtl.top)
    assert fn.latency is not None and fn.latency_is_bound

    A = np.arange(N * M, dtype=np.int32).reshape(N, M)
    out = np.zeros(M, np.int32)
    r = rtl.cosim(A, out)
    assert r.cycles == fn.latency - 2 * M
    assert np.array_equal(out, A.sum(0))


# A result-yielding guard advances its container on the completion pulse and
# latches the carried scalar through the capture D wire, so each iteration
# runs one latch cycle under the published bound.
@needs_verilator
def test_a_carried_result_rides_the_guards_pulse():
    N, M = 5, 6

    @kernel
    def gcarry(A: i32[N, M], out: i32[1]):
        t: i32 = 0
        for j in range(M):
            if A[0, j] > 0:  # memory predicate: a prologue leaf, then the guard
                acc: i32 = 0
                for k in range(N):
                    acc += A[k, j]
                t = t + acc
        out[0] = t

    rtl = _to_rtl(gcarry)
    fn = rtl.schedule().func(rtl.top)
    assert fn.latency is not None and fn.latency_is_bound

    A = np.ones((N, M), np.int32)  # every arm taken: recovery is the only slack
    out = np.zeros(1, np.int32)
    r = rtl.cosim(A, out)
    assert r.cycles == fn.latency - M
    assert out[0] == A.sum()


# The guard's predicate reads the scalar the guard itself yields. The
# predicate leaf launches a cycle after the pulse advance, when the iter-arg
# register has already latched the D-wire value.
@needs_verilator
def test_a_guard_predicate_reads_the_scalar_its_own_pulse_latched():
    N, M = 5, 6

    @kernel
    def gclip(A: i32[N, M], out: i32[1]):
        t: i32 = 0
        for j in range(M):
            if t < 40:
                acc: i32 = 0
                for k in range(N):
                    acc += A[k, j]
                t = t + acc
        out[0] = t

    rtl = _to_rtl(gclip)
    fn = rtl.schedule().func(rtl.top)
    assert fn.latency is not None and fn.latency_is_bound

    A = np.ones((N, M), np.int32)  # column sums keep t < 40: every arm taken
    out = np.zeros(1, np.int32)
    r = rtl.cosim(A, out)
    assert r.cycles == fn.latency - M
    assert out[0] == A.sum()


# A bounded callee composes onward as a bound: the instance carries its
# (latency, determinacy), so the caller publishes a ceiling of its own rather
# than an exact span or no figure at all.
@needs_verilator
def test_a_callers_span_composes_a_bounded_callee_as_a_bound():
    N = 8

    @kernel
    def bc_leaf(a: i32[N], flag: i32, out: i32[N]):
        if flag > 0:
            for i in range(N):
                out[i] = a[i] + 1
        else:
            out[0] = 0

    @kernel
    def bc_top(a: i32[N], flag: i32, out: i32[N]):
        bc_leaf(a, flag, out)
        for i in range(N):
            out[i] = out[i] * 2

    rtl = _to_rtl(bc_top)
    for name in ("bc_leaf", "bc_top"):
        fn = rtl.schedule().func(name)
        assert fn.latency is not None and fn.latency_is_bound
        assert fn.determinacy == "indeterminate"

    a = np.arange(N, dtype=np.int32)
    out = np.zeros(N, np.int32)
    span = rtl.interfaces.of_symbol(rtl.top).latency
    assert rtl.cosim(a, np.int32(1), out).cycles == span  # deeper arm, tight
    assert np.array_equal(out, (a + 1) * 2)
    out = np.zeros(N, np.int32)
    assert rtl.cosim(a, np.int32(0), out).cycles < span


# ---------------------------------------------------------------------------
# What `indeterminate` costs a call: the partitioner must isolate it
# ---------------------------------------------------------------------------

# A run whose length depends on the data: the while stops at the first
# non-positive element, so `A` decides how many cycles the child takes.
A_RUN = np.array([3, 4, 5, 0, 9, 9, 9, 9], np.int32)
RUN_SUM = 12  # 3 + 4 + 5


@kernel
def ic_sum(A: i32[8]) -> i32:
    i: i32 = 0
    s: i32 = 0
    while A[i] > 0:
        s += A[i]
        i += 1
    return s


@kernel
def ic_sum_out(A: i32[8], B: i32[1]):
    i: i32 = 0
    s: i32 = 0
    while A[i] > 0:
        s += A[i]
        i += 1
    B[0] = s


def _regions(m):
    """The caller's own top-level acyclic regions. A callee is a separate
    kernel and a nested region is reported at a greater depth, so neither can
    add to this."""
    return _outer(m.schedule().func(m.top), RegionKind.ACYCLIC)


# --- the scalar result --------------------------------------------------------


# A scalar result read by consumers written in the caller's own span. The
# partitioner splits them off into a region started by the child's done.
@needs_verilator
def test_a_scalar_result_consumed_in_the_callers_own_span():
    @kernel
    def ic_scalar(A: i32[8], B: i32[2]):
        r: i32 = ic_sum(A)
        B[0] = r + 1
        B[1] = r * 2

    m = _to_rtl(ic_scalar)
    assert len(_regions(m)) == 2, _regions(m)
    B = np.zeros(2, np.int32)
    m.cosim(A_RUN, B)
    assert np.array_equal(B, [RUN_SUM + 1, RUN_SUM * 2]), list(B)


# A call-to-call hand-off already starts the consumer on the producer's
# `done`. With both isolated it goes through the sibling sequencer instead
# and must still see the settled value.
@needs_verilator
def test_a_scalar_result_handed_to_a_second_call():
    @kernel
    def ic_twice(v: i32) -> i32:
        return v * 3

    @kernel
    def ic_chain(A: i32[8], B: i32[1]):
        r: i32 = ic_sum(A)
        B[0] = ic_twice(r)

    B = np.zeros(1, np.int32)
    _to_rtl(ic_chain).cosim(A_RUN, B)
    assert B[0] == RUN_SUM * 3, list(B)


# --- the memory half ----------------------------------------------------------


# Sharing the span schedules the load at the call's own start cycle, so it
# reads the buffer before the child has written it: the right hardware
# shape, the wrong answer, and no diagnostic.
@needs_verilator
def test_a_buffer_the_child_writes_and_the_caller_reads():
    @kernel
    def ic_buf(A: i32[8], out: i32[1]):
        t: i32[1]
        ic_sum_out(A, t)
        out[0] = t[0] + 1

    m = _to_rtl(ic_buf)
    assert len(_regions(m)) == 2, _regions(m)
    out = np.zeros(1, np.int32)
    m.cosim(A_RUN, out)
    assert out[0] == RUN_SUM + 1, list(out)


# The same hazard through a kernel ARGUMENT the child masters, where a port
# group rather than an internal hlmem carries the writes.
@needs_verilator
def test_a_boundary_buffer_the_child_writes():
    @kernel
    def ic_bnd(A: i32[8], B: i32[1], out: i32[1]):
        ic_sum_out(A, B)
        out[0] = B[0] * 2

    B, out = np.zeros(1, np.int32), np.zeros(1, np.int32)
    _to_rtl(ic_bnd).cosim(A_RUN, B, out)
    assert (B[0], out[0]) == (RUN_SUM, RUN_SUM * 2), (list(B), list(out))


# An indeterminate producer writing a buffer a SIBLING CALL then reads: the
# consumer has no static offset to be placed at, so only the producer's real
# done will do. Keyed on the callee's determinacy, not a container-wide mode.
@needs_verilator
def test_a_buffer_handed_to_a_sibling_call():
    @kernel
    def sp_prod(n0: i32, B: i32[16]):
        c: i32 = 0
        x: i32 = n0
        while x > 1:  # data-dependent trip -> whole-kernel latency unknown
            x = x - 1
            c = c + 1  # c = n0 - 1, escapes to the store loop (not DCE-able)
        for i in range(16):
            B[i] = i + c

    @kernel
    def sp_cons(B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = B[i] + 100

    @kernel
    def sp_top(n0: i32, B: i32[16], out: i32[16]):
        sp_prod(n0, B)  # indeterminate producer writes B
        sp_cons(B, out)  # reads B -> must wait for prod's real done

    assert _latency(sp_prod) is None

    mod = _to_rtl(sp_top)
    assert Dcp(mod).func("sp_top").callees() == ["sp_top.sp_prod", "sp_top.sp_cons"]
    assert _latency(sp_top) is None  # container inherits the while's indeterminacy

    spB = np.zeros(16, np.int32)
    spout = np.zeros(16, np.int32)
    mod.cosim(np.int32(5), spB, spout)  # n0 = 5 -> c = 4
    exp = np.array([i + 4 + 100 for i in range(16)], np.int32)
    assert np.array_equal(spout, exp), list(spout)


# --- the isolation is specific ----------------------------------------------


# The negative control for this half: isolation must not swallow the
# entry-block rule. A call with statically-known latency is a time-triggered
# node the sequencer may legitimately overlap with its neighbours, so it is
# NOT isolated and stays inside its straight-line span.
def test_a_determinate_call_still_shares_its_span():
    @kernel
    def ic_fixed(v: i32) -> i32:
        return v + 7

    @kernel
    def ic_det(A: i32[8], B: i32[2]):
        r: i32 = ic_fixed(A[0])
        B[0] = r + 1

    assert len(_regions(_to_rtl(ic_det))) == 1


# Where the isolated region's own drain lands. A call with no contract is
# priced at latency zero, so the only static statement left is that the call
# occupies the cycle it issues in. Charging it `latency - 1` cycles underflowed
# that zero into 2^32, which made the exact scheduler's drain bound narrower
# than the term it bounds and CP-SAT called the region infeasible.
@pytest.mark.parametrize("scheduler", ["heuristic", "exact"])
def test_an_indeterminate_calls_region_drains_at_its_own_start(scheduler):
    @kernel
    def ic_drain(A: i32[8], B: i32[1]):
        ic_sum_out(A, B)

    (region,) = _regions(_to_rtl(ic_drain).set_scheduler_opt(scheduler=scheduler))
    assert region.cost.drain == 0


# That drain is not a SPAN. A leaf's drain prices a call from its contract, so
# a child without one leaves the region completing on a `done` no number names,
# and a container around it inherits that rather than multiplying a fiction by
# its trip. The sibling is the control: a region that does have a span keeps it.
@needs_verilator
def test_a_region_holding_an_indeterminate_call_declares_no_span():
    @kernel
    def ic_nospan(A: i32[8], B: i32[1], C: i32[4]):
        for i in range(4):
            ic_sum_out(A, B)
            C[i] = B[0] + 1

    rtl = _to_rtl(ic_nospan)
    top = rtl.schedule().func("ic_nospan")
    (container,) = [r for r in top.regions if r.container]
    assert (container.determinacy, container.latency, container.interval) == (
        "indeterminate",
        None,
        None,
    )
    held, sibling = [r for r in top.regions if not r.container]
    assert (held.determinacy, held.latency) == ("indeterminate", None)
    assert sibling.determinacy == "counted_static" and sibling.latency

    # The done-paced container is the point: every pass must wait out a child
    # whose length the data decides, not re-fire on a counted cadence.
    B, C = np.zeros(1, np.int32), np.zeros(4, np.int32)
    rtl.cosim(A_RUN, B, C)
    assert B[0] == RUN_SUM and np.array_equal(C, [RUN_SUM + 1] * 4), (list(B), list(C))


# Isolation adds a region, not an ordering: a sibling that shares nothing
# with the call has no dependence on it and still starts with the kernel.
@needs_verilator
def test_an_independent_sibling_still_runs_concurrently():
    @kernel
    def ic_indep(A: i32[8], B: i32[1], C: i32[4]):
        B[0] = ic_sum(A)
        for i in range(4):
            C[i] = i * 2

    rtl = _to_rtl(ic_indep)
    # The claim, read off the control structure rather than the clock: no
    # region's run register is gated on another region's done. A sibling that
    # DID share state shows the isolated call's done in exactly this cone.
    m = Mod(rtl.mlir, "ic_indep")
    runs = m.regions_with("run")
    assert runs, "no run register to check -- the assertion below would be vacuous"
    for rid in runs:
        _, nxt = m.reg_named(f"r{rid}_run")
        gates = {m.hint.get(v, v) for v in m.cone(nxt, limit=400)}
        assert not [g for g in gates if g.endswith("_done")], (rid, sorted(gates))

    B, C = np.zeros(1, np.int32), np.zeros(4, np.int32)
    rtl.cosim(A_RUN, B, C)
    assert B[0] == RUN_SUM and np.array_equal(C, [0, 2, 4, 6]), (list(B), list(C))


# The caller's answer must track a child whose length is decided by the
# data, not by the schedule.
@needs_verilator
@pytest.mark.parametrize("stop", [1, 4, 7])
def test_a_run_length_that_actually_varies(stop):
    @kernel
    def ic_var(A: i32[8], B: i32[1]):
        r: i32 = ic_sum(A)
        B[0] = r + 1

    m = _to_rtl(ic_var)
    A = np.where(np.arange(8) < stop, np.arange(8) + 1, 0).astype(np.int32)
    B = np.zeros(1, np.int32)
    m.cosim(A, B)
    assert B[0] == A[:stop].sum() + 1, (stop, list(B))

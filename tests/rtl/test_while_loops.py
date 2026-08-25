# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`while`-loop scheduling and correctness: flushing pipelines, CHECK/RUN sequential control, nested whiles, and the various continue-condition shapes."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, f32, index
from allo.lang.ip import OperatorType
from allo.backend.rtl.devices import default_device

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, Mod, _sched, _to_rtl, _one_region, _hold_done  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


# --- schedule shape -----------------------------------------------------------


def test_while_scheduling():
    # A counted while is raised to a for and schedules identically to one; a
    # data-dependent while stays conditional, scheduled as a flushing pipeline
    # with its trip -- and therefore latency -- left unknown.
    @kernel
    def wc(A: i32[128], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while i < 128:
            s = s + A[i]
            i = i + 1
        out[0] = s

    @kernel
    def fc(A: i32[128], out: i32[1]):
        s: i32 = 0
        for i in range(128):
            s = s + A[i]
        out[0] = s

    w = _sched(wc).cyclic()[0]
    f = _sched(fc).cyclic()[0]
    # Raised to a constant-bound for, so the schedule matches `range(128)`
    # exactly -- same II, length, and (non-bound) latency -- and is not
    # conditional (no flushing controller).
    assert (w.interval, w.iteration_latency, w.latency) == (
        f.interval,
        f.iteration_latency,
        f.latency,
    )
    assert not w.conditional and not w.latency_is_bound
    # The data-dependent counterpart is scheduled and driven together in
    # test_while_flushing_pipeline_cosim.


def test_decreasing_index_while_raises_cosim():
    # A decreasing (countdown) counted while with an `index` IV raises to a
    # counted for. The rewrite reconstructs the IV from the loop counter as
    # `init - j`, so the body sees i = N, N-1, ..., 1 in the original order and
    # the loop schedules as counted, not a flushing pipeline. Covers a constant
    # trip, a runtime (dynamic-bound) trip, and the zero-trip entry.
    N = 64
    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF

    @kernel
    def dec_const(A: i32[N], out: i32[1]):
        i: index = N
        s: i32 = 0
        while i > 0:
            s = s + A[i - 1]
            i = i - 1
        out[0] = s

    loop = _sched(dec_const).cyclic()[0]
    # Constant trip: counted, known latency, and not a conditional/flushing
    # region. The span is N issues one II apart from the start cycle, the
    # accumulate landing, and the completion latch.
    assert not loop.conditional and not loop.latency_is_bound
    assert loop.latency == N + 1
    out = np.zeros(1, np.int32)
    _to_rtl(dec_const).cosim(A, out)
    assert out[0] == int(A.sum())  # counts i down from N, so sums all of A

    @kernel
    def dec_var(A: i32[N], n0: index, out: i32[1]):
        i: index = n0
        s: i32 = 0
        while i > 0:
            s = s + A[i - 1]
            i = i - 1
        out[0] = s

    mod = _to_rtl(dec_var)
    loop = mod.schedule().cyclic()[0]
    # Runtime trip: still counted (a dynamic bound), so not conditional, but its
    # length -- hence latency -- is left unknown.
    assert not loop.conditional and loop.latency is None
    for n0 in (0, 1, 5, N):  # n0 == 0 exits on entry
        out = np.zeros(1, np.int32)
        mod.cosim(A, np.int64(n0), out)
        assert out[0] == int(A[:n0].sum())


def test_i32_counted_while_raises_cosim():
    # A counted while whose IV is `i32` (not `index`) raises like an index one:
    # the matcher looks through the extend/truncate the frontend wraps i32
    # arithmetic in for overflow, and the rewrite casts the counter back to i32.
    # Covers both directions.
    N = 32
    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF

    @kernel
    def incr(A: i32[N], out: i32[1]):
        i: i32 = 0
        s: i32 = 0
        while i < N:
            s = s + A[i]
            i = i + 1
        out[0] = s

    @kernel
    def decr(A: i32[N], out: i32[1]):
        i: i32 = N
        s: i32 = 0
        while i > 0:
            s = s + A[i - 1]
            i = i - 1
        out[0] = s

    for k in (incr, decr):
        assert not _sched(k).cyclic()[0].conditional  # raised to a counted for
        out = np.zeros(1, np.int32)
        _to_rtl(k).cosim(A, out)
        assert out[0] == int(A.sum())


def test_used_iv_result_while_raises_cosim():
    # A counted while whose IV is read after the loop still raises: the pass
    # rebuilds the exit IV as `init + trip*delta`, the first value failing the
    # test. Covers an increasing runtime bound (result == n, and result == init
    # for the zero-trip case) and a decreasing constant bound (result == 3).
    N = 16
    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)

    @kernel
    def up(A: i32[N], n: i32, out: i32[2]):
        i: i32 = 0
        s: i32 = 0
        while i < n:
            s = s + A[i]
            i = i + 1
        out[0] = s
        out[1] = i  # first i failing i < n == max(0, n)

    mod = _to_rtl(up)
    assert not mod.schedule().cyclic()[0].conditional  # raised, not flushing
    for n in (0, 5, N):  # n == 0 exits on entry, so i stays init (0)
        out = np.zeros(2, np.int32)
        mod.cosim(A, np.int32(n), out)
        assert out[0] == int(A[:n].sum())
        assert out[1] == n

    @kernel
    def down(A: i32[N], out: i32[2]):
        i: i32 = N
        s: i32 = 0
        while i > 3:
            s = s + A[i - 1]
            i = i - 1
        out[0] = s
        out[1] = i  # first i failing i > 3 == 3

    mod = _to_rtl(down)
    out = np.zeros(2, np.int32)
    mod.cosim(A, out)
    assert out[0] == int(A[3:N].sum())
    assert out[1] == 3


def test_while_with_nested_while():
    # A data-dependent double-while: both continue-tests are combinational over
    # carried accumulators (`oacc < olimit`, `iacc < ilimit`) advanced by the
    # data, so neither is counted and neither raises. The outer holds a nested
    # loop, so it runs sequentially; the inner flushes. Exercises the
    # nested-loop-in-while decomposition recursing through a while child.
    N = 16

    @kernel
    def nested_while(A: i32[N], olimit: i32, ilimit: i32, out: i32[1]):
        total: i32 = 0
        oacc: i32 = 0
        i: i32 = 0
        while oacc < olimit:
            iacc: i32 = 0
            j: i32 = 0
            while iacc < ilimit:
                iacc = iacc + A[j]
                total = total + A[j]
                j = j + 1
            oacc = oacc + iacc
            i = i + 1
        out[0] = total

    mod = _to_rtl(nested_while)
    res = mod.schedule()
    assert len(res.cyclic()) >= 1  # the inner while pipelines
    assert res.func("nested_while").latency is None  # data-dependent trips
    # Both whiles close to dcp: the inner -> flushing pipeline, the outer ->
    # sequential container wrapping it. No raw scf.while; two dcp.condition ends.
    d = Dcp(mod)
    assert not d.has("scf.while")
    assert d.func("nested_while").count("allo.dcp.condition") == 2


# --- flushing-pipeline correctness ---------------------------------------------


def test_while_flushing_pipeline_cosim():
    # The flushing pipeline emitted end-to-end: `running` gated by the exit
    # condition, each loop-carried iter-arg frozen into a survivor register at
    # exit, and the sibling store reading the frozen count. The exit test
    # `acc < limit` is combinational over the carried `acc`, so the loop flushes
    # while its trip -- driven by the data in A -- stays unknown. `c` counts the
    # committed iterations, including the zero-iteration case (limit <= 0).
    N = 32

    @kernel
    def wr(A: i32[N], limit: i32, out: i32[1]):
        acc: i32 = 0
        c: i32 = 0
        while acc < limit:
            acc = acc + A[c]
            c = c + 1
        out[0] = c

    mod = _to_rtl(wr)
    # The schedule this rests on: a data-dependent while stays conditional, and
    # its trip -- so its latency -- is left unknown rather than faked.
    loop = mod.schedule().cyclic()[0]
    assert loop.conditional is True
    assert loop.latency is None
    assert Dcp(mod).has("allo.dcp.condition")  # reified while terminator

    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)

    def gold(limit):  # smallest c with sum(A[:c]) >= limit
        acc = c = 0
        while acc < limit:
            acc += int(A[c])
            c += 1
        return c

    for limit in (0, 1, 10, 50):  # limit == 0 exits on entry
        out = np.zeros(1, np.int32)
        mod.cosim(A, np.int32(limit), out)
        assert out[0] == gold(limit)


def test_while_pipeline_operators_are_allocated():
    # A flushing while is a pipeline like any counted loop, so the exact
    # scheduler decides how many copies of each operator its body builds.
    #
    # The two products are independent so that the decision has somewhere to go:
    # a chain would leave every multiply in one congruence class at this II,
    # where one unit per operation is the only legal count and a live allocation
    # would look exactly like a dead one.
    N = 32

    @kernel
    def wmul(Ai: i32[N], A: f32[N], out: f32[N], limit: i32):
        c: i32 = 0
        acc: i32 = 0
        while acc < limit:
            acc = acc + Ai[c]
            out[c] = (A[c] * A[c + 1]) * (A[c + 2] * A[c + 3])
            c = c + 1

    # At a slower clock, because an allocation has to be legal to be made: the
    # operands arrive from array ports, and a port's read cone plus a select
    # cone plus the multiply's own input cone do not fit the default period.
    mod = _to_rtl(wmul, freq_mhz=150.0).set_scheduler_opt(scheduler="exact")
    assert mod.schedule().cyclic()[0].conditional  # a flushing while, not a for
    mod.compile()
    (region,) = [r for f in mod.microarch.funcs for r in f.regions]
    assert region.shared_units, "the while body never reached the allocation"

    # `Ai` all ones makes the trip exactly `limit`, so `c + 3` stays inside `A`.
    Ai = np.ones(N, np.int32)
    A = np.random.default_rng(9).random(N, dtype=np.float32).astype(np.float32)
    for limit in (0, 1, 8):
        out = np.zeros(N, np.float32)
        mod.cosim(Ai, A, out, np.int32(limit))
        gold = np.zeros(N, np.float32)
        for c in range(limit):
            gold[c] = (A[c] * A[c + 1]) * (A[c + 2] * A[c + 3])
        np.testing.assert_allclose(out, gold, rtol=2e-3, atol=1e-5)


def test_while_two_carried_accumulate_cosim():
    # A while carrying TWO recurrences whose result depends on both: `acc` drives
    # the combinational exit test while `s` folds the running `acc`, so the
    # frozen `s` survivor depends on the whole acc trajectory and the trip.
    N = 32

    @kernel
    def wacc(A: i32[N], limit: i32, out: i32[1]):
        acc: i32 = 0
        s: i32 = 0
        i: i32 = 0
        while acc < limit:
            acc = acc + A[i]
            s = s + acc
            i = i + 1
        out[0] = s

    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)
    mod = _to_rtl(wacc)

    def gold(limit):
        acc = s = i = 0
        while acc < limit:
            acc += int(A[i])
            s += acc
            i += 1
        return s

    for limit in (0, 1, 10, 50):
        out = np.zeros(1, np.int32)
        mod.cosim(A, np.int32(limit), out)
        assert out[0] == gold(limit)


def test_while_multistage_flush_cosim():
    # A store-less while whose *body* spans two stages (the `A[i]` load pushes
    # `next_acc` to stage 1) but whose condition `acc < limit` is combinational
    # over the carried `acc`. The flushing pipeline drains the deeper survivor:
    # `acc` advances one cycle after each issue, and the exit is delayed to
    # match, so the frozen `acc` is the correct sum.
    N = 64

    @kernel
    def wsum(A: i32[N], limit: i32, out: i32[1]):
        acc: i32 = 0
        i: i32 = 0
        while acc < limit:
            acc = acc + A[i]
            i = i + 1
        out[0] = acc

    A = (np.arange(N, dtype=np.int32) % 9 + 1).astype(np.int32)
    mod = _to_rtl(wsum)

    def gold(limit):  # acc once it first reaches limit
        acc = i = 0
        while acc < limit:
            acc += int(A[i])
            i += 1
        return acc

    for limit in (0, 1, 20, 100):
        out = np.zeros(1, np.int32)
        mod.cosim(A, np.int32(limit), out)
        assert out[0] == gold(limit)


def test_while_in_loop_store_cosim():
    # A leaf flushing-while that writes memory in its body. The doomed exit
    # iteration is issued but must commit nothing: emitWrites gates each
    # store's write-enable by the continue-condition (`issue & cond`), the same
    # rule the loop-carried survivors follow. Covers a single-stage store, a
    # multi-stage store fed by an in-loop carried scalar (deeper drain), and the
    # zero-trip case (no write). Unwritten output elements read back as the
    # memory init (0).
    N = 32

    @kernel
    def wstore(A: i32[N], limit: i32, B: i32[N]):  # write-once per iteration
        acc: i32 = 0
        i: i32 = 0
        while acc < limit:
            B[i] = A[i] * 2
            acc = acc + A[i]
            i = i + 1

    @kernel
    def wscan(A: i32[N], limit: i32, B: i32[N]):  # store the running prefix sum
        acc: i32 = 0
        i: i32 = 0
        while acc < limit:
            acc = acc + A[i]
            B[i] = acc
            i = i + 1

    ma, mb = _to_rtl(wstore), _to_rtl(wscan)
    assert ma.schedule().cyclic()[0].conditional and Dcp(ma).has("allo.dcp.condition")
    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)

    def gold_store(limit):
        B = np.zeros(N, np.int32)
        acc = i = 0
        while acc < limit:
            B[i] = int(A[i]) * 2
            acc += int(A[i])
            i += 1
        return B

    def gold_scan(limit):
        B = np.zeros(N, np.int32)
        acc = i = 0
        while acc < limit:
            acc += int(A[i])
            B[i] = acc
            i += 1
        return B

    for limit in (0, 1, 20, 50):  # limit == 0 writes nothing
        B = np.zeros(N, np.int32)
        ma.cosim(A, np.int32(limit), B)
        assert np.array_equal(B, gold_store(limit))

        B = np.zeros(N, np.int32)
        mb.cosim(A, np.int32(limit), B)
        assert np.array_equal(B, gold_scan(limit))


# --- condition shapes: memory, IP, nested --------------------------------------


def test_while_mem_condition_cosim():
    # A while loop whose continue-condition reads memory (`A[i] != key`): the
    # loop index advances until the searched element is found, and the
    # loop-carried value is read after the loop. Covers a single-value carry, a
    # two-value carry (the index and a step counter), and a zero-iteration exit
    # (the condition false on entry).
    A = np.arange(16, dtype=np.int32)  # A[i] == i, so the found index equals key

    @kernel
    def linsearch(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        while A[i] != key:
            i = i + 1
        out[0] = i

    out = np.zeros(1, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(11), out)
    assert out[0] == 11

    @kernel
    def search_steps(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        c: i32 = 0
        while A[i] != key:
            i = i + 1
            c = c + 1
        out[0] = c

    out = np.zeros(1, np.int32)
    _to_rtl(search_steps).cosim(A, np.int32(9), out)
    assert out[0] == 9

    # A[0] == key: the condition is false on entry, so the body never runs and
    # the carried index holds its initial value.
    out = np.full(1, 999, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(0), out)
    assert out[0] == 0


def test_while_mem_condition_shared_array_cosim():
    # A while loop that reads the same array in BOTH its continue-condition
    # (`A[i] > 0`) and its body (`s += A[i]`). Each access is a distinct memory
    # read, so the condition and the body do not contend for a port.
    @kernel
    def wmem(A: i32[16], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while A[i] > 0:
            s = s + A[i]
            i = i + 1
        out[0] = s

    rtl = _to_rtl(wmem)
    # The non-contention claim, stated: A carries a read port group per access,
    # so the condition never waits on the body's port.
    rd = [p.base for acc in rtl.interfaces["wmem"].reads for p in acc]
    assert rd == ["A_rd0", "A_rd1"]

    A = np.array([5, 3, 8, 2, 0] + [9] * 11, dtype=np.int32)  # sentinel 0 at idx 4
    out = np.zeros(1, np.int32)
    rtl.cosim(A, out)
    assert out[0] == 5 + 3 + 8 + 2  # sum until A[4] == 0 stops the loop


def test_while_ip_condition_cosim():
    # A while whose continue-condition is a multi-cycle floating-point
    # operation rather than a memory read. The loop iterates until the float
    # condition settles false; the body advances a float-carried value. Covers
    # a single float comparison (`r > tol`) and a float subtraction feeding a
    # comparison (`x - b > 0`), the latter a multi-stage condition cone. The
    # condition is not settled in the issue cycle, so the loop runs
    # sequentially (a conditional region) rather than as a flushing pipeline.
    @kernel
    def fconverge(x: f32, tol: f32, out: f32[1]):
        r: f32 = x
        while r > tol:
            r = r * 0.5
        out[0] = r

    mod = _to_rtl(fconverge)
    assert mod.schedule().cyclic()[0].conditional
    # The extern module name carries the compare predicate, so it is the
    # operator's symbol plus the predicate the op declares.
    fcmp = next(o for o in default_device.operators if o.optype is OperatorType.CMP)
    assert f"hw.module.extern @{fcmp.symbol}_ogt" in mod.mlir

    def gold_halve(x, tol):
        r = np.float32(x)
        while r > np.float32(tol):
            r = np.float32(r * np.float32(0.5))
        return r

    for x, tol in [(100.0, 1.0), (7.0, 1.0), (0.5, 1.0)]:  # last exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(x), np.float32(tol), out)
        assert out[0] == gold_halve(x, tol)

    @kernel
    def fcountdown(a: f32, b: f32, out: f32[1]):
        x: f32 = a
        while x - b > 0.0:
            x = x - 1.0
        out[0] = x

    mod = _to_rtl(fcountdown)
    assert mod.schedule().cyclic()[0].conditional

    def gold_count(a, b):
        x = np.float32(a)
        while np.float32(x - np.float32(b)) > np.float32(0.0):
            x = np.float32(x - np.float32(1.0))
        return x

    for a, b in [(10.0, 2.5), (5.0, 5.0), (3.0, 0.0)]:  # middle exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(a), np.float32(b), out)
        assert out[0] == gold_count(a, b)


def test_nested_while_cosim():
    # A sequential-wrapper while (outer `oacc`) around a flushing-pipeline while
    # (inner `iacc`), carrying a cross-region accumulator `total`. The outer is a
    # conditional container: its iter-args are survivor registers, the
    # combinational `oacc < olimit` test is evaluated over them, and the inner
    # re-runs each outer iteration (which advances `oacc` by the inner's result).
    N = 16

    @kernel
    def nested(A: i32[N], olimit: i32, ilimit: i32, out: i32[1]):
        total: i32 = 0
        oacc: i32 = 0
        i: i32 = 0
        while oacc < olimit:
            iacc: i32 = 0
            j: i32 = 0
            while iacc < ilimit:
                iacc = iacc + A[j]
                total = total + A[j]
                j = j + 1
            oacc = oacc + iacc
            i = i + 1
        out[0] = total

    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)

    def gold(olimit, ilimit):
        total = oacc = 0
        while oacc < olimit:
            iacc = j = 0
            while iacc < ilimit:
                iacc += int(A[j])
                total += int(A[j])
                j += 1
            oacc += iacc
        return total

    for olimit, ilimit in [(1, 1), (20, 10), (50, 15)]:
        out = np.zeros(1, np.int32)
        _to_rtl(nested).cosim(A, np.int32(olimit), np.int32(ilimit), out)
        assert out[0] == gold(olimit, ilimit)


# --- call-in-while control drop ------------------------------------------------


def test_call_in_a_while_body():
    # A while whose body calls a sub-kernel cannot flushing-pipeline at all:
    # that schedule issues an iteration per cycle, which a child instance fired
    # and awaited per iteration can never follow. It drops to the sequential
    # CHECK/RUN controller, the same route a nested loop or a non-combinational
    # condition takes. The memory-sentinel condition also keeps it uncounted, so
    # the doomed iteration (where A[i] == 0) is never fired and never stores.
    @kernel
    def wc_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def wc_top(A: i32[16], B: i32[16]):
        i: i32 = 0
        while A[i] != 0:
            wc_child(A, B, i)
            i += 1

    rtl = _to_rtl(wc_top)
    # The drop, stated: a CHECK/RUN while emits an `r<N>_check` and its region
    # carries no static II. A flushing pipeline has neither.
    assert Mod(rtl.mlir, "wc_top").regions_with("check")
    assert rtl.schedule().funcs[0].regions[0].interval is None

    A16 = np.array([3, 5, 7, 2, 4, 6, 0] + [9] * 9, dtype=np.int32)  # sentinel at 6
    B = np.zeros(16, np.int32)
    rtl.cosim(A16, B)
    gold = np.zeros(16, np.int32)
    gold[:6] = A16[:6] * 2  # only the pre-sentinel iterations store
    assert np.array_equal(B, gold)


def test_while_condition_reads_a_shape_the_check_cannot_evaluate():
    # The sequential CHECK region evaluates arithmetic and array reads and
    # nothing else, so a sub-kernel call in the continue-condition has to be
    # reported. What the test locks is WHERE and against WHAT: the scheduler
    # names the call, which is the only place that fact is known. Leaving the
    # cone unscheduled instead pushes the report to the emitter, which sees the
    # `index_cast`/`cmpi` beside the call go untimed and blames those.
    #
    # Both shapes below must reach that one report. The second is the one that
    # matters: with no nested loop to force the sequential route, the call in
    # the condition has to be what forces it. A call is timed by its CALLEE's
    # schedule and the operator library has no row to answer for it, so reading
    # a latency off the library called it combinational, the while took the
    # flushing schedule, and the emitter ABORTED on a leaf-loop-over-calls
    # invariant it was never meant to be handed.
    N = 16

    @kernel
    def wcc_child(A: i32[N], i: index) -> i32:
        return A[i] * 2

    @kernel
    def wcc_nested(A: i32[N], B: i32[N]):
        i: i32 = 0
        while wcc_child(A, i) != 0:
            for j in range(4):
                B[j] = B[j] + i
            i = i + 1

    @kernel
    def wcc_flat(A: i32[N], B: i32[N]):
        i: i32 = 0
        while wcc_child(A, i) != 0:
            B[i] = i
            i = i + 1

    for k in (wcc_nested, wcc_flat):
        with pytest.raises(RuntimeError):
            _to_rtl(k).schedule()


# --- checked-iteration skeleton reuse -------------------------------------------


def test_checked_while_reuses_the_counted_skeleton():
    # A CHECK/RUN while (a conditional container wrapping a flushing-pipeline
    # inner while) keeps the same fire / done-latch pair a counted cell uses,
    # replacing only the counter-driven test with a delayed condition pulse:
    # no counter, no separate empty term, since the first CHECK already answers
    # it. The outer test `oacc < olimit` is combinational over a carried
    # accumulator (data-advanced, so uncounted), which is what settles the CHECK
    # in one cycle. Cosims a nested double-while summation.
    N = 16

    @kernel
    def nested(A: i32[N], olimit: i32, ilimit: i32, out: i32[1]):
        total: i32 = 0
        oacc: i32 = 0
        i: i32 = 0
        while oacc < olimit:  # conditional container
            iacc: i32 = 0
            j: i32 = 0
            while iacc < ilimit:  # flushing-pipeline leaf
                iacc = iacc + A[j]
                total = total + A[j]
                j = j + 1
            oacc = oacc + iacc
            i = i + 1
        out[0] = total

    rtl = _to_rtl(nested)
    m = Mod(rtl.mlir, "nested")
    r = _one_region(m)

    check = m.signal(f"r{r}_check")
    fire = m.signal(f"r{r}_fire")
    finish = _hold_done(m, r)
    # Launch and finish are the two arms of ONE pulse: both are `check & (~)cond`
    # over the same settled CHECK, so the container cannot do both.
    assert check in m.cone(fire) and check in m.cone(finish)
    cond = [v for v in m.cone(fire) if m.defs.get(v, "").startswith("comb.icmp")]
    assert cond, "the fire pulse does not test the continue condition"
    assert any(c in m.cone(finish) for c in cond), "finish tests another condition"
    # No counter: termination is by condition alone, so no induction arithmetic
    # reaches the launch decision.
    assert not [v for v in m.cone(fire) if m.defs.get(v, "").startswith("comb.add")]

    A = (np.arange(N, dtype=np.int32) % 7 + 1).astype(np.int32)

    def gold(olimit, ilimit):
        total = oacc = 0
        while oacc < olimit:
            iacc = j = 0
            while iacc < ilimit:
                iacc += int(A[j])
                total += int(A[j])
                j += 1
            oacc += iacc
        return total

    out = np.zeros(1, np.int32)
    rtl.cosim(A, np.int32(20), np.int32(10), out)
    assert out[0] == gold(20, 10)


# --- region-boundary pass-through ---------------------------------------------


def test_a_chained_container_turns_over_in_the_commit_cycle():
    # `for i: [while, epilogue]` is the flat-FSM shape: the while hands the
    # epilogue its finish pulse, the container relaunches on the epilogue's
    # commit pulse, and the carried q crosses through the live result wire. A
    # zero-trip iteration costs CHECK(1) + t_cond + drain, with the container's
    # done latch as the only cycle outside the loop.
    N = 16

    @kernel
    def scan(x: i32[N], out: i32[N]):
        q: i32 = 0
        for i in range(N):
            while q > x[i]:
                q = q - x[i]
            out[i] = q
            q = q + 2

    rtl = _to_rtl(scan)
    regions = rtl.schedule().func("scan").regions
    cond = next(r for r in regions if r.conditional)
    epi = [r for r in regions if r.kind.value == "acyclic"][-1]
    t_cond = max(op.t for op in cond.ops)
    per_iter = 1 + t_cond + epi.cost.drain

    x = np.full(N, 100, np.int32)  # q never exceeds 2N: every while is zero-trip
    out = np.zeros(N, np.int32)
    r = rtl.cosim(x, out, timeout=1000)
    assert r.cycles == N * per_iter + 1
    assert np.array_equal(out, np.arange(0, 2 * N, 2, np.int32))


def test_a_backtracking_while_carries_its_state_across_the_turnover():
    # Data-dependent while trips: q crosses while -> epilogue on the finish
    # pulse, epilogue -> next iteration through the container's live latch, and
    # into the while's own iter-arg through the register's D wire. A stale
    # sample at any of the three hand-offs changes the values, not just the
    # timing.
    N = 16

    @kernel
    def scan(x: i32[N], out: i32[N]):
        q: i32 = 0
        for i in range(N):
            while q > x[i]:
                q = q - x[i]
            out[i] = q
            q = q + 2

    rtl = _to_rtl(scan)
    rng = np.random.default_rng(0)
    x = rng.integers(1, 4, N).astype(np.int32)
    out = np.zeros(N, np.int32)
    rtl.cosim(x, out, timeout=2000)

    q, want = 0, []
    for i in range(N):
        while q > x[i]:
            q -= int(x[i])
        want.append(q)
        q += 2
    assert np.array_equal(out, np.array(want, np.int32))

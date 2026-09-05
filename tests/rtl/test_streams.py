# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream/FIFO channel behavior: occupancy and II accounting, multi-access ordering, fan-out, kernel-local channels, predicated access, and back-pressure token-exactness."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, Stream

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, _sched, _to_rtl, _iis, FADD, Mod  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

N = 8
A8 = np.arange(1, 9, dtype=np.int32)
A16 = np.arange(1, 17, dtype=np.int32)
_STALLS = [0.0, 0.6]


def _df(k):
    """Export `k` with its calls spawned as concurrent processes."""
    s = k.schedule()
    s.dataflow()
    return s.export("rtl")


def _delay_line(a, seed, step):
    """The software model of a 1-slot delay line: the FIFO holds `seed`, and
    each iteration pops it, records it, and pushes `step(popped, a[i])`."""
    out = np.zeros(len(a), a.dtype)
    held = seed
    for i in range(len(a)):
        out[i] = held
        held = step(held, a[i])
    return out


def _fifos(m):
    """The ``(rdEn, wrEn)`` operand pair of every ``seq.fifo`` in ``m``."""
    out = []
    for line in m.text.splitlines():
        fm = re.search(r"seq\.fifo .*rdEn %([\w.$-]+) wrEn %([\w.$-]+)", line)
        if fm:
            out.append(fm.groups())
    return out


# --- occupancy & multi-access -------------------------------------------------


# N accesses to ONE channel cannot go faster than II=N: each needs its own
# cycle on the single handshake. Reported as a resource-min II, so the search
# starts there rather than discovering it by failing.
@pytest.mark.parametrize("n", [1, 2, 3])
def test_channel_occupancy_bounds_ii(n):
    if n == 1:

        @kernel
        def prod(A: i32[8], s: Stream[i32, 8]):
            for i in range(8):
                s.put(A[i])

    elif n == 2:

        @kernel
        def prod(A: i32[8], s: Stream[i32, 8]):
            for i in range(8):
                s.put(A[i])
                s.put(A[i] + 1)

    else:

        @kernel
        def prod(A: i32[8], s: Stream[i32, 8]):
            for i in range(8):
                s.put(A[i])
                s.put(A[i] + 1)
                s.put(A[i] + 2)

    regions = _to_rtl(prod).schedule().func("prod").regions
    assert [r.interval for r in regions if r.interval is not None] == [n]


# The resource is per channel, not per stream op: two puts to DIFFERENT
# channels have independent handshakes and still pipeline at II=1. This is
# what keeps the resource from serializing every multi-stream process.
def test_distinct_channels_do_not_contend():
    @kernel
    def fork(A: i32[8], s: Stream[i32, 8], t: Stream[i32, 8]):
        for i in range(8):
            s.put(A[i])
            t.put(A[i] + 1)

    regions = _to_rtl(fork).schedule().func("fork").regions
    assert [r.interval for r in regions if r.interval is not None] == [1]


# Both tokens reach the consumer, in program order. The two put values are
# combinational (an int add), so nothing but the resource separates them.
@pytest.mark.parametrize("stall", _STALLS)
def test_two_puts_to_one_channel(stall):
    @kernel
    def p2(A: i32[8], s: Stream[i32, 8]):
        for i in range(8):
            s.put(A[i])
            s.put(A[i] + 1)

    @kernel
    def c2(s: Stream[i32, 8], B: i32[16]):
        for i in range(16):
            B[i] = s.get()

    @kernel
    def two_put(A: i32[8], B: i32[16]):
        f: Stream[i32, 8]
        p2(A, f)
        c2(f, B)

    B = np.zeros(16, np.int32)
    _df(two_put).cosim(A8, B, stall_prob=stall)
    exp = np.empty(16, np.int32)
    exp[0::2], exp[1::2] = A8, A8 + 1
    assert np.array_equal(B, exp)


# Three deep: the shift that keeps a bumped put off its neighbour has to
# compose, not just handle a single pair.
@pytest.mark.parametrize("stall", _STALLS)
def test_three_puts_to_one_channel(stall):
    @kernel
    def p3(A: i32[8], s: Stream[i32, 8]):
        for i in range(8):
            s.put(A[i])
            s.put(A[i] + 1)
            s.put(A[i] + 2)

    @kernel
    def c3(s: Stream[i32, 8], B: i32[24]):
        for i in range(24):
            B[i] = s.get()

    @kernel
    def three_put(A: i32[8], B: i32[24]):
        f: Stream[i32, 8]
        p3(A, f)
        c3(f, B)

    B = np.zeros(24, np.int32)
    _df(three_put).cosim(A8, B, stall_prob=stall)
    exp = np.empty(24, np.int32)
    exp[0::3], exp[1::3], exp[2::3] = A8, A8 + 1, A8 + 2
    assert np.array_equal(B, exp)


# The read end is bound the same way. a * 10 + b is order-sensitive, so a
# swapped pair fails the comparison rather than cancelling out.
@pytest.mark.parametrize("stall", _STALLS)
def test_two_gets_from_one_channel(stall):
    @kernel
    def pg(A: i32[16], s: Stream[i32, 8]):
        for i in range(16):
            s.put(A[i])

    @kernel
    def cg(s: Stream[i32, 8], B: i32[8]):
        for i in range(8):
            a: i32 = s.get()
            b: i32 = s.get()
            B[i] = a * 10 + b

    @kernel
    def two_get(A: i32[16], B: i32[8]):
        f: Stream[i32, 8]
        pg(A, f)
        cg(f, B)

    B = np.zeros(8, np.int32)
    _df(two_get).cosim(A16, B, stall_prob=stall)
    assert np.array_equal(B, A16[0::2] * 10 + A16[1::2])


# The control: one access per channel per iteration pipelines at II=1 and
# delivers the same tokens. A resource that over-constrained shows up here.
@pytest.mark.parametrize("stall", _STALLS)
def test_single_access_channel_still_pipelines(stall):
    @kernel
    def p1(A: i32[16], s: Stream[i32, 4]):
        for i in range(16):
            s.put(A[i] * 2)

    @kernel
    def c1(s: Stream[i32, 4], B: i32[16]):
        for i in range(16):
            B[i] = s.get() + 1

    @kernel
    def one_each(A: i32[16], B: i32[16]):
        f: Stream[i32, 4]
        p1(A, f)
        c1(f, B)

    mod = _df(one_each)
    B = np.zeros(16, np.int32)
    mod.cosim(A16, B, stall_prob=stall)
    assert np.array_equal(B, A16 * 2 + 1)


# Pipelining off paces the loop by its schedule depth. A put commits in its
# issue cycle, the FIFO's own register absorbing the write latency, so the
# depth ends one cycle after the put's stage.
@pytest.mark.parametrize("stall", _STALLS)
def test_nonpipelined_put_commits_in_its_issue_cycle(stall):
    @kernel
    def seqp(A: i32[8], s: Stream[i32, 8]):
        for i in range(8, name="i"):
            s.put(A[i])

    sch = seqp.schedule()
    sch.pipeline("i", ii=-1)
    mod = sch.export("rtl")

    loop = mod.schedule().func("seqp").cyclic()[0]
    put = loop.op("stream.put")
    assert loop.interval == loop.iteration_latency == put.t + 1
    assert loop.cost.drain == put.t

    B = np.zeros(8, np.int32)
    mod.cosim(A8, B, stall_prob=stall)
    assert np.array_equal(B, A8)


# A channel is one {data,valid,ready} triple, time-shared by every access to
# it: several gets/puts per iteration interleave inside the II, and dependence
# edges (not a stream-port resource) order accesses within one II across regions.
def test_stream_multi_access_per_channel():
    n = 8

    # Two tokens consumed per iteration. The dependence pair (dist-0 forward +
    # dist-1 back edge) forces distinct stages spanning less than the II, which
    # is exactly "II >= accesses on the channel".
    @kernel
    def mg_two(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(n):
            y_out.put(x_in.get() + x_in.get())

    loop = _sched(mg_two).func("mg_two").cyclic()[0]
    gets = sorted(o.t for o in loop.ops if o.kind == "stream.get")
    # Distinct stages spanning less than the II; WHICH stages is the scheduler's.
    assert (
        loop.interval >= 2
        and len(set(gets)) == 2
        and gets[-1] - gets[0] < loop.interval
    )

    x = np.arange(2 * n, dtype=np.int32) * 7 + 3
    rtl = _to_rtl(mg_two)
    for gap in (0.0, 0.6):
        y = np.zeros(n, np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, x[0::2] + x[1::2]), f"gap={gap}: {list(y)}"

    # Two tokens produced per iteration: `_valid` is the OR of the two activation
    # pulses and `_data` a mux selected by them.
    @kernel
    def mp_two(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(n):
            v: i32 = x_in.get()
            y_out.put(v)
            y_out.put(v + 1)

    x = np.arange(n, dtype=np.int32) * 5 - 11
    rtl = _to_rtl(mp_two)
    for gap in (0.0, 0.6):
        y = np.zeros(2 * n, np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, np.stack([x, x + 1], 1).reshape(-1)), list(y)

    # Three gets: the all-pairs serialization scales, II rising to match.
    @kernel
    def mg_three(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(n):
            y_out.put(x_in.get() + x_in.get() * 2 + x_in.get() * 4)

    loop = _sched(mg_three).func("mg_three").cyclic()[0]
    gets = sorted(o.t for o in loop.ops if o.kind == "stream.get")
    assert (
        loop.interval >= 3
        and len(set(gets)) == 3
        and gets[-1] - gets[0] < loop.interval
    )
    x = np.arange(3 * n, dtype=np.int32) * 3 + 1
    y = np.zeros(n, np.int32)
    _to_rtl(mg_three).cosim(x, y, stall_prob=0.6)
    assert np.array_equal(y, x[0::3] + x[1::3] * 2 + x[2::3] * 4), list(y)

    # A conditional put in both arms of an `if`: if-conversion masks them into
    # one region, so both share the channel's handshake under their predicates.
    @kernel
    def mp_guard(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(n):
            v: i32 = x_in.get()
            if v > 0:
                y_out.put(v)
            else:
                y_out.put(0 - v)

    x = np.arange(n, dtype=np.int32) * 5 - 11
    y = np.zeros(n, np.int32)
    _to_rtl(mp_guard).cosim(x, y, stall_prob=0.6)
    assert np.array_equal(y, np.abs(x)), list(y)

    # Across regions: two sequential loops draining one channel. A stream
    # sibling edge serializes them; run concurrently they would drive the
    # channel's `ready` together and pop the same token twice.
    @kernel
    def mg_regions(x_in: Stream[i32], out: i32[2]):
        a: i32 = 0
        for i in range(4):
            a += x_in.get()
        out[0] = a
        b: i32 = 0
        for j in range(4):
            b += x_in.get()
        out[1] = b

    x = np.arange(8, dtype=np.int32) * 5 - 11
    rtl = _to_rtl(mg_regions)
    for gap in (0.0, 0.6):
        out = np.zeros(2, np.int32)
        rtl.cosim(x, out, stall_prob=gap)
        assert np.array_equal(out, [x[:4].sum(), x[4:].sum()]), f"{gap}: {list(out)}"


# A single latency-insensitive stream process: its schedule shape, then cosim
# determinism at full rate and under stall for a combinational and for a
# multi-cycle IP datapath.
def test_stream_li_shell():
    @kernel
    def prod(srm: Stream[i32]):
        for i in range(10):
            srm.put(i)

    @kernel
    def cons(srm: Stream[i32], out: i32[1]):
        acc: i32 = 0
        for i in range(10):
            acc += srm.get()
        out[0] = acc

    @kernel
    def top(out: i32[1]):
        srm: Stream[i32]
        prod(srm)
        cons(srm, out)

    res = _sched(top)
    loop = res.func("cons").cyclic()[0]
    assert loop.interval == 1
    assert loop.op("stream.get").t <= loop.op("addi").t
    # The epilogue store lands in its own acyclic region.
    assert any(
        o.kind == "store"
        for r in res.func("cons").regions
        if r.kind == "acyclic"
        for o in r.ops
    )

    # One input stream, one output stream, counted loop, combinational datapath;
    # cocotb drives the FIFO {data,valid,ready} ports directly. KPN determinism:
    # the shell bubbles on an empty input and freezes on a full output, so it
    # never loses or duplicates a token and the result is stall-independent.
    @kernel
    def stage(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(16):
            y_out.put(x_in.get() + 7)

    rtl = _to_rtl(stage)
    x = np.arange(16, dtype=np.int32) * 5 - 3
    exp = x + 7
    for gap in (0.0, 0.5, 0.8):
        y = np.zeros(16, dtype=np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, exp), f"gap={gap}: {list(y)} != {list(exp)}"

    # The same shell with a multi-cycle IP datapath (a float multiply into a
    # float add, an 11-deep pipeline between get and put). The `ce` stall
    # contract freezes the IP pipeline in lockstep with the shell's shift chains;
    # a free-running IP would keep clocking under back-pressure and desync.
    @kernel
    def fstage(x_in: Stream[f32], y_out: Stream[f32]):
        for i in range(16):
            y_out.put(x_in.get() * 2.0 + 1.0)

    frtl = _to_rtl(fstage)
    fx = (np.arange(16, dtype=np.float32) * 0.5 - 3.0).astype(np.float32)
    fexp = fx * 2.0 + 1.0
    for gap in (0.0, 0.5, 0.8):
        fy = np.zeros(16, dtype=np.float32)
        frtl.cosim(fx, fy, stall_prob=gap)
        assert np.allclose(fy, fexp), f"gap={gap}: {list(fy)} != {list(fexp)}"


# A slow (II>1) f32-accumulate stream consumer draining a memory-read-fed
# producer.
def test_stream_ii_gt1_with_memory_read_producer():
    def build(K):
        @kernel
        def top(A: f32[K], out: f32[1]):
            fifo: Stream[f32]

            @kernel(mapping=[2])
            def pe(A: f32[K], out: f32[1], fifo: Stream[f32]):
                p = allo.get_wid(0)
                if p == 0:
                    for k in range(K):  # memory-read-fed put (II=1 producer)
                        fifo.put(A[k])
                else:
                    c: f32 = 0.0
                    for k in range(K):  # recurrence -> II == FADD (slow drain)
                        c += fifo.get()
                    out[0] = c

            pe(A, out, fifo)

        return top

    # The consumer's inner loop is recurrence-bound: the shell runs the modulo
    # (II>1) regime, not the II==1 fast path.
    iis = [
        r.interval
        for f in _to_rtl(build(8)).set_scheduler_opt(accumulators=0).schedule().funcs
        for r in f.regions
        if r.interval is not None
    ]
    assert max(iis) > 1  # the modulo regime; the exact II is the scheduler's

    for K in (8, 16):
        A = (2.0 ** np.arange(K)).astype(np.float32)  # 1, 2, 4, ... 2**(K-1)
        exp = float(A.sum())  # == 2**K - 1
        out = np.zeros(1, dtype=np.float32)
        _to_rtl(build(K)).cosim(A, out)
        assert abs(out[0] - exp) < 0.5, f"K={K}: {out[0]} != {exp} (dropped a token)"


# --- fan-out ------------------------------------------------------------------


# Several readers are a fan-out the operator inserts: one queue each, one push.
# The producer writes only when every consumer can accept, so the copies stay
# in step, and there is exactly ONE such join, not one per consumer.
@pytest.mark.parametrize("stall", _STALLS)
def test_fanout_is_one_push_and_n_queues(stall):
    @kernel
    async def fo_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i])

    @kernel
    async def fo_c0(s: Stream[i32], o0: i32[N]):
        for i in range(N):
            o0[i] = s.get() + 1

    @kernel
    async def fo_c1(s: Stream[i32], o1: i32[N]):
        for i in range(N):
            o1[i] = s.get() * 3

    @kernel
    async def fo_top(x: i32[N], o0: i32[N], o1: i32[N]):
        s: Stream[i32]
        await fo_prod(x, s)
        await fo_c0(s, o0)
        await fo_c1(s, o1)

    mod = _to_rtl(fo_top)
    m = Mod(mod.mlir, "fo_top")
    fifos = _fifos(m)
    assert len(fifos) == 2, fifos
    # One push: both queues take the SAME write enable, whose cone waits on
    # every queue's `full`.
    assert fifos[0][1] == fifos[1][1], fifos
    assert fifos[0][0] != fifos[1][0], fifos
    cone = m.cone(fifos[0][1])
    assert {"full", "full_2"} <= cone or len(
        [v for v in cone if "full" in v]
    ) == 2, cone

    x = np.arange(1, N + 1, dtype=np.int32)
    o0 = np.zeros(N, np.int32)
    o1 = np.zeros(N, np.int32)
    mod.cosim(x, o0, o1, stall_prob=stall)
    assert np.array_equal(o0, x + 1), list(o0)
    assert np.array_equal(o1, x * 3), list(o1)


# The fan-out is N-ary, not a special case of two.
def test_local_channel_fans_out_to_three_consumers():
    @kernel
    async def f3_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i])

    @kernel
    async def f3_a(s: Stream[i32], o: i32[N]):
        for i in range(N):
            o[i] = s.get() + 1

    @kernel
    async def f3_b(s: Stream[i32], o: i32[N]):
        for i in range(N):
            o[i] = s.get() * 3

    @kernel
    async def f3_c(s: Stream[i32], o: i32[N]):
        for i in range(N):
            o[i] = s.get() - 2

    @kernel
    async def f3_top(x: i32[N], o1: i32[N], o2: i32[N], o3: i32[N]):
        f: Stream[i32]
        await f3_prod(x, f)
        await f3_a(f, o1)
        await f3_b(f, o2)
        await f3_c(f, o3)

    mod = _to_rtl(f3_top)
    assert mod.mlir.count("seq.fifo") == 3

    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    o3 = np.zeros(N, np.int32)
    mod.cosim(A8, o1, o2, o3)
    assert np.array_equal(o1, A8 + 1), list(o1)
    assert np.array_equal(o2, A8 * 3), list(o2)
    assert np.array_equal(o3, A8 - 2), list(o3)


# A container stream INPUT read by several processes: the top keeps its one
# stream port and feeds a FIFO per consumer from it, so the port's ready is
# the join of theirs.
@pytest.mark.parametrize("stall", _STALLS)
def test_a_boundary_stream_argument_fans_out(stall):
    @kernel
    async def bf_c1(s: Stream[i32], o1: i32[N]):
        for i in range(N):
            o1[i] = s.get() + 1

    @kernel
    async def bf_c2(s: Stream[i32], o2: i32[N]):
        for i in range(N):
            o2[i] = s.get() * 3

    @kernel
    async def bf_top(f: Stream[i32], o1: i32[N], o2: i32[N]):
        await bf_c1(f, o1)
        await bf_c2(f, o2)

    mod = _to_rtl(bf_top)
    assert mod.mlir.count("seq.fifo") == 2

    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(A8, o1, o2, stall_prob=stall)
    assert np.array_equal(o1, A8 + 1), list(o1)
    assert np.array_equal(o2, A8 * 3), list(o2)


# One consumer far slower than the other: the producer stalls at the slowest
# (inherent to any bounded-memory fork; a decoupled fork would need unbounded
# buffering), and neither drops a token. The lag comes from a memory-carried
# accumulate recurrence, not a nested loop -- the latter would hit an
# unrelated acyclic-region-cannot-wait-for-stream-input bug.
def test_a_fanned_out_consumer_that_lags_does_not_lose_tokens():
    @kernel
    async def lag_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i])

    @kernel
    async def lag_fast(s: Stream[i32], o1: i32[N]):
        for i in range(N):
            o1[i] = s.get()

    @kernel
    async def lag_slow(s: Stream[i32], o2: i32[N]):
        for i in range(N):
            o2[0] = o2[0] + s.get()  # II = read + add + write, so it lags

    @kernel
    async def lag_top(x: i32[N], o1: i32[N], o2: i32[N]):
        f: Stream[i32]
        await lag_prod(x, f)
        await lag_fast(f, o1)
        await lag_slow(f, o2)

    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    _to_rtl(lag_top).cosim(A8, o1, o2)
    assert np.array_equal(o1, A8), list(o1)
    # Every token reached the slow side exactly once: none dropped, none
    # duplicated by the lock-step push.
    assert o2[0] == int(A8.sum()), list(o2)


# Fan-out composes with feedback seeding: the init-prepend shim is a
# CONSUMER-side thing, so each of the N consumers gets its own and every one
# sees [init] ++ [produced].
def test_a_seeded_channel_fans_out():
    @kernel
    async def sf_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i])

    @kernel
    async def sf_c1(s: Stream[i32], o1: i32[N]):
        for i in range(N):
            o1[i] = s.get() + 1

    @kernel
    async def sf_c2(s: Stream[i32], o2: i32[N]):
        for i in range(N):
            o2[i] = s.get() * 3

    @kernel
    async def sf_top(x: i32[N], o1: i32[N], o2: i32[N]):
        f: Stream[i32, 2] = [99]
        await sf_prod(x, f)
        await sf_c1(f, o1)
        await sf_c2(f, o2)

    mod = _to_rtl(sf_top)
    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(A8, o1, o2)
    seen = np.concatenate([[99], A8])[:N]  # the seed, then the produced tokens
    assert np.array_equal(o1, seen + 1), list(o1)
    assert np.array_equal(o2, seen * 3), list(o2)


# --- kernel-local channels -----------------------------------------------------


# A self-loop needs no directional port: the module's port list carries only
# its memory interfaces, and the whole loop is a single seq.fifo in the body
# (its prologue put and the loop's accesses name different SSA values, keyed
# by the same storage root, so the storage-root peel must not build the queue
# twice).
def test_local_channel_has_no_port_and_is_one_fifo():
    @kernel
    def lc_ports(A: i32[N], out: i32[N]):
        f: Stream[i32]
        f.put(A[0])
        for i in range(N):
            v: i32 = f.get()
            out[i] = v
            f.put(v + A[i])

    m = _to_rtl(lc_ports).mlir
    header = m.split("hw.module @lc_ports(")[1].split(")")[0]
    assert "seq.fifo" in m
    assert "_st_data" not in header and "_st_valid" not in header, header
    assert m.count("seq.fifo") == 1


# A channel used as a loop-carried delay line: the get -> put -> get
# recurrence forces II=2 for the int case; the float case pushes the put
# further out still, since the recurrence now also carries an operator
# latency.
@pytest.mark.parametrize("dtype", [i32, f32], ids=["i32", "f32"])
def test_self_loop_delay_line(dtype):
    @kernel
    def lc_sl(A: dtype[N], out: dtype[N]):
        f: Stream[dtype]
        f.put(A[0])
        for i in range(N):
            v: dtype = f.get()
            out[i] = v
            f.put(v + A[i])

    rtl = _to_rtl(lc_sl)
    if dtype is i32:
        assert rtl.schedule().func("lc_sl").cyclic()[0].interval >= 2
        A = np.arange(N, dtype=np.int32) * 3 - 5
        out = np.zeros(N, np.int32)
        rtl.cosim(A, out)
        assert np.array_equal(out, _delay_line(A, A[0], lambda h, x: h + x)), list(out)
    else:
        # The claim the i32 arm cannot make: the recurrence now runs through the
        # float adder, so the channel-occupancy floor is no longer what binds.
        assert rtl.schedule().func("lc_sl").cyclic()[0].interval >= FADD
        A = (np.arange(N, dtype=np.float32) + 1) * 0.5
        out = np.zeros(N, np.float32)
        rtl.cosim(A, out)
        assert np.allclose(
            out, _delay_line(A, A[0], lambda h, x: np.float32(h + x))
        ), list(out)


# Two tokens in flight: the prologue seeds both, so each iteration reads the
# value two iterations back. Depth 4 leaves the queue slack.
def test_a_two_slot_delay_line():
    @kernel
    def lc_d2(A: i32[N], out: i32[N]):
        f: Stream[i32, 4]
        f.put(A[0])
        f.put(A[1])
        for i in range(N):
            v: i32 = f.get()
            out[i] = v
            f.put(v + 1)

    A = np.arange(1, N + 1, dtype=np.int32)
    out = np.zeros(N, np.int32)
    _to_rtl(lc_d2).cosim(A, out)
    exp, q = np.zeros(N, np.int32), [A[0], A[1]]
    for i in range(N):
        exp[i] = q.pop(0)
        q.append(exp[i] + 1)
    assert np.array_equal(out, exp), list(out)


# Not a self-loop: one loop fills the channel, a later one drains it. A shared
# channel orders sibling regions, so the two are serialized and the queue must
# hold the whole run, depth N here -- a shallower queue deadlocks, which is
# the completion-gate's own limitation rather than this feature, so it is not
# exercised here.
def test_a_local_channel_between_two_loops():
    @kernel
    def lc_pipe(A: i32[N], out: i32[N]):
        f: Stream[i32, N]
        for i in range(N):
            f.put(A[i] * 2)
        for j in range(N):
            out[j] = f.get() + 1

    A = np.arange(N, dtype=np.int32) - 3
    out = np.zeros(N, np.int32)
    _to_rtl(lc_pipe).cosim(A, out)
    assert np.array_equal(out, A * 2 + 1), list(out)


# Both kinds in one kernel: the argument keeps its port triple, the local one
# has none, and each access resolves its handshake from its own side. Under
# back-pressure the boundary stall and the local starvation freeze hold the
# same shell together.
@pytest.mark.parametrize("stall", [0.0, 0.6])
def test_a_local_channel_beside_a_boundary_one(stall):
    @kernel
    def lc_mix(x_in: Stream[i32], out: i32[N]):
        f: Stream[i32]
        f.put(0)
        for i in range(N):
            v: i32 = f.get()
            t: i32 = x_in.get()
            out[i] = v + t
            f.put(v + 1)

    m = _to_rtl(lc_mix)
    header = m.mlir.split("hw.module @lc_mix(")[1].split(")")[0]
    assert "x_in_st_valid" in header and header.count("_st_valid") == 1, header
    x = np.arange(N, dtype=np.int32) * 7
    out = np.zeros(N, np.int32)
    m.cosim(x, out, stall_prob=stall)
    assert np.array_equal(out, x + np.arange(N, dtype=np.int32)), list(out)


# The same seq.fifo and the same handshake whether the channel's ends are two
# loops of one kernel or two processes of a container. rdEn = ready & ~empty
# and wrEn = valid & ~full is the whole protocol, written once and shared by
# both substrates.
def test_a_channel_is_one_queue_at_either_substrate():
    @kernel
    def lc(x: i32[N], out: i32[N]):
        c: Stream[i32, N]  # both ends inside one leaf; the queue holds the run
        for i in range(N):
            c.put(x[i] * 2)
        for i in range(N):
            out[i] = c.get() + 1

    @kernel
    async def cc_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i] * 2)

    @kernel
    async def cc_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def cc_top(x: i32[N], out: i32[N]):
        s: Stream[i32]
        await cc_prod(x, s)
        await cc_cons(s, out)

    def one_queue(mod, name):
        m = Mod(mod.mlir, name)
        fifos = _fifos(m)
        assert len(fifos) == 1, fifos
        rd, wr = fifos[0]
        # Both enables are a two-input AND; the FIFO's own status feeds each
        # through a single inversion, which is what makes the two spellings
        # comparable at all.
        assert m.defs[rd].startswith("comb.and"), m.defs[rd]
        assert m.defs[wr].startswith("comb.and"), m.defs[wr]
        inverted = {
            v
            for v in m.operands(rd) + m.operands(wr)
            if "comb.xor" in m.defs.get(v, "")
        }
        assert {"empty", "full"} <= {
            o for v in inverted for o in m.operands(v)
        }, inverted
        return m

    leaf = _to_rtl(lc)
    one_queue(leaf, "lc")
    top = _to_rtl(cc_top)
    one_queue(top, "cc_top")

    x = np.arange(1, N + 1, dtype=np.int32)
    for mod in (leaf, top):
        out = np.zeros(N, np.int32)
        mod.cosim(x, out)
        assert np.array_equal(out, x * 2 + 1), list(out)


# --- predicated access & back-pressure token-exactness ------------------------


# FIFO-din stability under back-pressure (regression for the transient-din
# register). A STAGE>=1 transient put (f(load) = B[k]*3) has a delayed valid,
# so back-pressure can hold it into the drain where the counter resets;
# without capturing din into a chain-enable-frozen register (bump the put one
# stage, Vitis's v3_reg), the held valid re-addresses the live read and
# commits a corrupted final token. A STAGE-0 counter-fed put (put(k)) must
# instead freeze atomically without being over-registered (the
# dcpStart(put)>=1 guard). A depth<K systolic column forces this
# deterministically since M=1 makes it a back-pressure chain, not a depth<K
# deadlock, isolating the value bug from a hang.
def test_transient_din_stability_under_backpressure():
    M, Nc, K, DEPTH = 1, 2, 3, 2  # depth < K => the last put is held into the drain
    P0, P1 = M + 2, Nc + 2

    @kernel
    def sa_fload(A: i32[M, K], B: i32[K, Nc], C: i32[M, Nc]):
        fifo_A: Stream[i32, DEPTH][P0, P1]
        fifo_B: Stream[i32, DEPTH][P0, P1]

        @kernel(mapping=[P0, P1])
        def pe(
            A: i32[M, K],
            B: i32[K, Nc],
            C: i32[M, Nc],
            fifo_A: Stream[i32, DEPTH][P0, P1],
            fifo_B: Stream[i32, DEPTH][P0, P1],
        ):
            i = allo.get_wid(0)
            j = allo.get_wid(1)
            if (i == 0 or i == M + 1) and (j == 0 or j == Nc + 1):
                pass
            elif j == 0:
                for k in range(K):
                    fifo_A[i, j + 1].put(A[i - 1, k])
            elif i == 0:
                for k in range(K):
                    fifo_B[i + 1, j].put(B[k, j - 1] * 3)  # f(load): stage>=1
            elif i == M + 1:
                for k in range(K):
                    b: i32 = fifo_B[i, j].get()
            elif j == Nc + 1:
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
            else:
                c: i32 = 0
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
                    b: i32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                    fifo_B[i + 1, j].put(b)
                C[i - 1, j - 1] = c

        pe(A, B, C, fifo_A, fifo_B)

    @kernel
    def sa_counter(A: i32[M, K], B: i32[K, Nc], C: i32[M, Nc]):
        fifo_A: Stream[i32, DEPTH][P0, P1]
        fifo_B: Stream[i32, DEPTH][P0, P1]

        @kernel(mapping=[P0, P1])
        def pe(
            A: i32[M, K],
            B: i32[K, Nc],
            C: i32[M, Nc],
            fifo_A: Stream[i32, DEPTH][P0, P1],
            fifo_B: Stream[i32, DEPTH][P0, P1],
        ):
            i = allo.get_wid(0)
            j = allo.get_wid(1)
            if (i == 0 or i == M + 1) and (j == 0 or j == Nc + 1):
                pass
            elif j == 0:
                for k in range(K):
                    fifo_A[i, j + 1].put(A[i - 1, k])
            elif i == 0:
                for k in range(K):
                    fifo_B[i + 1, j].put(k)  # counter: stage-0, atomically frozen
            elif i == M + 1:
                for k in range(K):
                    b: i32 = fifo_B[i, j].get()
            elif j == Nc + 1:
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
            else:
                c: i32 = 0
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
                    b: i32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                    fifo_B[i + 1, j].put(b)
                C[i - 1, j - 1] = c

        pe(A, B, C, fifo_A, fifo_B)

    A = np.array([[4, 3, 2]], dtype=np.int32)
    B = np.array([[1, 1], [2, 0], [1, 3]], dtype=np.int32)

    # f(load): the forwarded b-token is 3*B[k], so C = A @ (3*B). The final put is
    # held onto the counter reset -- without the din register it commits 3*B[0].
    mod = _to_rtl(sa_fload)
    out = np.zeros((M, Nc), np.int32)
    mod.cosim(A, B, out)
    exp = A @ (3 * B)
    assert np.array_equal(out, exp), (list(out.ravel()), list(exp.ravel()))

    # counter: the forwarded b-token is k (every column), so C[i,j] = sum_k A[i,k]*k.
    # The stage-0 put must be correct WITHOUT the extra register.
    mod = _to_rtl(sa_counter)
    out = np.zeros((M, Nc), np.int32)
    mod.cosim(A, B, out)
    exp = A @ np.repeat(np.arange(K, dtype=np.int32)[:, None], Nc, axis=1)
    assert np.array_equal(out, exp), (list(out.ravel()), list(exp.ravel()))


# Data-dependent conditional put / get: the branch condition becomes the
# access's i1 predicate, so it stays in the pipelined region. End-to-end, a
# filter's output rate and a gated read's input rate are data-dependent.
def test_dataflow_predicated_stream_access():
    # Masked in place rather than serialized into a guard region.
    @kernel
    async def pp_prod(a: i32[16], y: Stream[i32]):
        for i in range(16):
            x = a[i]
            if x > 0:
                y.put(x)

    mod = _to_rtl(pp_prod)
    res = mod.schedule()
    # The put carries its optional predicate operand, the loop pipelines at
    # II=1, and no guard (dcp.select) / raw scf.if is left.
    d = Dcp(mod)
    (put,) = d.func("pp_prod").ops("allo.stream.put")
    # segments are (stream, indices, value, pred): the last is the predicate
    assert put.attributes["operandSegmentSizes"][3] == 1
    assert not d.has("scf.if")
    assert _iis(res.func("pp_prod").cyclic()) == [1]
    assert not any(r.kind == "guard" for r in res.funcs[0].regions)

    # A filter process puts only the tokens that pass a data-dependent test, so
    # its output rate is data-dependent (non-SDF -- what Vitis dataflow rejects).
    # The consumer reads the M tokens that pass.
    Nf, M = 16, 8

    @kernel
    async def pf_prod(a: i32[Nf], y: Stream[i32]):
        for i in range(Nf):
            x = a[i]
            if x > 0:
                y.put(x)

    @kernel
    async def pf_cons(y: Stream[i32], out: i32[M]):
        for i in range(M):
            out[i] = y.get()

    @kernel
    async def pf_top(a: i32[Nf], out: i32[M]):
        y: Stream[i32]
        await pf_prod(a, y)
        await pf_cons(y, out)

    rtl = _to_rtl(pf_top)
    # a[i] positive at even i -> exactly M tokens pass the filter.
    a = np.array([(i + 1) if i % 2 == 0 else -(i + 1) for i in range(Nf)], np.int32)
    exp = np.array([i + 1 for i in range(Nf) if i % 2 == 0], np.int32)

    golden = np.zeros(M, np.int32)
    rtl.csim(a, golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(M, np.int32)
        rtl.cosim(a, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # The consumer side: a gated read consumes a token only where a data-dependent
    # select holds, else emits a default WITHOUT popping the channel. The
    # predicated get pops only when consuming and never stalls the pipeline on the
    # (empty) channel in the skipped iterations.
    NG, MG = 8, 4

    @kernel
    async def pg_prod(y: Stream[i32]):
        for i in range(MG):  # exactly as many tokens as are read
            y.put(i)

    @kernel
    async def pg_cons(sel: i32[NG], y: Stream[i32], out: i32[NG]):
        for i in range(NG):
            v: i32 = -1
            if sel[i] > 0:
                v = y.get()  # pop only where sel>0; single store below -> II=1
            out[i] = v

    @kernel
    async def pg_top(sel: i32[NG], out: i32[NG]):
        y: Stream[i32]
        await pg_prod(y)
        await pg_cons(sel, y, out)

    rtl = _to_rtl(pg_top)
    sel = np.array([1, -1, 1, -1, 1, -1, 1, -1], np.int32)  # 4 reads
    exp = np.array([0, -1, 1, -1, 2, -1, 3, -1], np.int32)
    golden = np.zeros(NG, np.int32)
    rtl.csim(sel, golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(NG, np.int32)
        rtl.cosim(sel, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# Two ways a freeze can corrupt the latency-insensitive shell. Both shapes
# hold a single access per channel, so they are independent of multi-access
# support, though multi-access reaches them constantly: it forces II >= 2 and
# pushes accesses to deeper stages.
def test_stream_shell_freeze_is_token_exact():
    n = 8
    x = np.arange(n, dtype=np.int32) * 5 - 11

    # (1) A stage>=1 put's `valid` is a delayed issue pulse riding the chain, so
    # a freeze holds it high after the handshake has fired and a ready consumer
    # recaptures the token; the `sent` latch retires it. The loop-carried
    # accumulator makes the region `cycleIndexedState`, so an input starvation
    # freezes the chain rather than only deferring the issue.
    @kernel
    def fz_put(x_in: Stream[i32], y_out: Stream[i32]):
        acc: i32 = 0
        for i in range(n, name="i"):
            acc += x_in.get()
            y_out.put(acc + 7)

    rtl = _to_rtl(fz_put)
    loop = rtl.schedule().func("fz_put").cyclic()[0]
    put = next(o for o in loop.ops if o.kind == "stream.put")
    # `deriveStallShell` arms the latch on a put at stage >= 1, and nothing
    # else; the stage's residue modulo the II decides nothing.
    assert put.t >= 1, put.t
    for gap in (0.0, 0.6):
        y = np.zeros(n, np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(
            y, np.cumsum(x, dtype=np.int32) + 7
        ), f"gap={gap}: {list(y)}"

    # (2) A starved input at II==1 must freeze rather than bubble: a bubble
    # still advances the shift chains, so a loop-carried accumulator folds in
    # the stale data port. A store is immune, its write-enable rides `issue`.
    @kernel
    def fz_acc(x_in: Stream[i32], out: i32[1]):
        a: i32 = 0
        for i in range(n):
            a += x_in.get()
        out[0] = a

    rtl = _to_rtl(fz_acc)
    assert _iis(rtl.schedule().func("fz_acc").cyclic()) == [1]
    for gap in (0.0, 0.6):
        out = np.zeros(1, np.int32)
        rtl.cosim(x, out, stall_prob=gap)
        assert out[0] == x.sum(), f"gap={gap}: {out[0]} != {x.sum()}"


# --- an acyclic region's single pass defers, it is not dropped ----------------


# A straight-line region issues ONE pass, and a pass that
# cannot be gated can only be dropped: a stage-0 get would sample `_data` at
# that pulse whatever `_valid` said, and never pop, leaving every later
# iteration one token behind. Reached by any stream consumer whose loop body
# needs an inner loop, since the imperfect-nest decomposition puts the get in an
# acyclic sub-region of its own.
#
# A LOCAL channel is what makes this deterministic at stall_prob=0: both
# processes start together, so the queue is genuinely empty at the consumer's
# first pass, where a boundary port driven by the testbench already holds a
# token. Reading one token early is then a fixed phase error, not a race.
@pytest.mark.parametrize("stall", _STALLS)
def test_acyclic_region_waits_for_stream_input(stall):
    @kernel
    def feed(A: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(A[i])

    @kernel
    def slow(s: Stream[i32], o: i32[N]):
        for i in range(N):
            v: i32 = s.get()  # region 0: acyclic, holds the stage-0 get
            acc: i32 = 0
            for j in range(4):  # region 1: forces the imperfect-nest split
                acc += v * j
            o[i] = acc  # region 2

    @kernel
    def top(A: i32[N], o: i32[N]):
        s: Stream[i32]
        feed(A, s)
        slow(s, o)

    rtl = _df(top)
    get = next(
        g
        for r in rtl.schedule().func("slow").regions
        if r.kind == "acyclic"
        for g in r.ops
        if g.kind == "stream.get"
    )
    assert get.t == 0, f"the get is at stage {get.t}, so not the stage-0 case"

    out = np.zeros(N, np.int32)
    rtl.cosim(A8, out, stall_prob=stall)
    assert np.array_equal(out, A8 * 6), f"stall={stall}: {list(out)}"


# The same pass at FUNC scope, where the region is top-level and issues on
# `start` itself, with no inner loop in sight: the consumer samples the channel
# on its very first cycle, before the producer has run at all.
@pytest.mark.parametrize("stall", _STALLS)
def test_top_level_acyclic_region_waits_for_stream_input(stall):
    @kernel
    def one(A: i32[N], s: Stream[i32]):
        s.put(A[0])

    @kernel
    def head(s: Stream[i32], o: i32[2]):
        v: i32 = s.get()
        o[0] = v
        o[1] = v * 3

    @kernel
    def top(A: i32[N], o: i32[2]):
        s: Stream[i32]
        one(A, s)
        head(s, o)

    rtl = _df(top)
    kinds = [r.kind for r in rtl.schedule().func("head").regions]
    assert kinds == ["acyclic"], kinds

    out = np.zeros(2, np.int32)
    rtl.cosim(A8, out, stall_prob=stall)
    assert np.array_equal(out, [A8[0], A8[0] * 3]), f"stall={stall}: {list(out)}"


# The output half of the same defect. A stage-0 put presents `valid` for the one
# pulse cycle, so a full downstream dropped the token; and since such a region's
# completion reduces to `issue & ready`, the drop also left its `done` low
# forever. The failure is a cosim timeout rather than a wrong answer.
@pytest.mark.parametrize("stall", _STALLS)
def test_acyclic_region_waits_for_stream_output(stall):
    @kernel
    def emit(A: i32[N], o: i32[N], y_out: Stream[i32]):
        for i in range(N):
            y_out.put(i)  # region 0: acyclic, stage 0 (the IV is combinational)
            acc: i32 = 0
            for j in range(4):  # region 1
                acc += A[i] * j
            o[i] = acc  # region 2

    rtl = _to_rtl(emit)
    put = next(
        p
        for r in rtl.schedule().func("emit").regions
        if r.kind == "acyclic"
        for p in r.ops
        if p.kind == "stream.put"
    )
    assert put.t == 0, f"the put is at stage {put.t}, so not the stage-0 case"

    o = np.zeros(N, np.int32)
    y = np.zeros(N, np.int32)
    rtl.cosim(A8, o, y, stall_prob=stall)
    assert np.array_equal(y, np.arange(N)), f"stall={stall}: {list(y)}"
    assert np.array_equal(o, A8 * 6), f"stall={stall}: {list(o)}"

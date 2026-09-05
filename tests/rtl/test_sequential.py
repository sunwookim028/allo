# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequential (non-dataflow) sub-kernel call composition"""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, index
from allo.backend.rtl import RegionKind

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, Mod, _latency, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
B16 = (np.arange(16, dtype=np.int32) * 5 + 3) & 0xFF


# --- Chained calls through shared storage ------------------------------------


# Two plain sub-kernels chained through a shared boundary array: the composed
# latency is the sum of the child latencies, both reported and actual.
def test_sequential_two_kernel_shared_array():
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

    l1, l2 = _latency(sc1), _latency(sc2)
    assert l1 is not None and l2 is not None
    assert _latency(seq_top) == l1 + l2

    rtl = _to_rtl(seq_top)
    # A pure serial call graph with no loose datapath still lowers via the leaf
    # CallUnit path, both children instantiated in the container's own module.
    assert Dcp(rtl).func(rtl.top).callees()
    assert rtl.mlir.count("hw.instance") >= 2

    B = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = rtl.cosim(A16, B, out)
    assert np.array_equal(out, (A16 + 1) * 2)  # out = child2(child1(A))
    assert r.cycles == l1 + l2  # serial: the children do not overlap


# Two sub-kernels chained through a container-LOCAL buffer: it lowers to an
# on-chip seq.hlmem rather than a top port, serialized by the RAW dependence.
def test_sequential_internal_buffer_shared():
    @kernel
    def sib_prod(A: i32[16], tmp: i32[16]):
        for i in range(16):
            tmp[i] = A[i] * 3

    @kernel
    def sib_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] - 7

    @kernel
    def sib_top(A: i32[16], out: i32[16]):
        tmp: i32[16]  # container-local buffer -> on-chip hlmem, not a top port
        sib_prod(A, tmp)
        sib_cons(tmp, out)

    mod = _to_rtl(sib_top)
    assert "seq.hlmem" in mod.mlir  # the internal buffer, on-chip in the top
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)  # tmp is not a top port, so cosim drives only A / out
    assert np.array_equal(out, A * 3 - 7)


# A STATIC zero-trip loop over a sub-kernel call is erased outright by
# `loop-canonicalization`, so the child is never instantiated and no controller
# is asked to complete a loop that fires nothing (regression: this used to hang
# on cosim's watchdog). Nothing is left of the kernel afterwards, which is also
# what pins the zero-region module: it completes a cycle after `start` instead
# of never.
def test_zero_trip_loop_over_calls():
    @kernel
    def zlc_step(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def zlc_top(A: i32[16], B: i32[16]):
        for i in range(0):
            zlc_step(A, B, i)

    rtl = _to_rtl(zlc_top)
    assert Dcp(rtl).func("zlc_top").callees() == []  # nothing left to call
    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.full(16, 9, np.int32)
    rtl.cosim(A, B)
    # Nothing is left to run, so nothing is written
    assert np.array_equal(B, np.full(16, 9, np.int32))


# A callee that computes nothing is not free the way an inlined no-op would be:
# it is emitted as its own module, instantiated, wired to the arrays it names and
# sequenced by the caller's controller. `drop-trivial-func` erases both shapes
# it recognizes, an empty body (here what the zero-trip erasure leaves) and an
# identity return, along with every call, so neither reaches the emitter.
def test_trivial_callees_are_dropped():
    @kernel
    def tc_empty(A: i32[16], B: i32[16]):
        for i in range(0):
            B[i] = A[i]

    @kernel
    def tc_ident(x: i32) -> i32:
        return x

    @kernel
    def tc_real(x: i32) -> i32:
        return x * 3

    @kernel
    def tc_top(A: i32[16], out: i32[1]):
        B: i32[16]
        tc_empty(A, B)
        out[0] = tc_ident(A[0]) + tc_real(A[1])

    rtl = _to_rtl(tc_top)
    # Only the callee that computes something is left, its call included.
    assert sorted(Dcp(rtl).kernels) == ["tc_top", "tc_top.tc_real"]
    assert Dcp(rtl).func("tc_top").callees() == ["tc_top.tc_real"]
    out = np.zeros(1, np.int32)
    rtl.cosim(A16, out)
    # The identity call's result is its operand: the caller reads A[0] itself.
    assert out[0] == A16[0] + A16[1] * 3, int(out[0])


# An UNGATED call (no call predecessor to hand off from) is released at the cycle
# it was SCHEDULED at, not at the region's issue pulse. Nothing makes its
# operands ready at issue: a scalar argument read from memory is only valid once
# the load's latency has passed, so a call fired at issue latches whatever the
# data port held before.
def test_an_ungated_call_waits_for_its_loaded_operand():
    @kernel
    def ugc_add3(x: i32) -> i32:
        y: i32 = x + 3
        return y * 2

    # From a LOAD: the operand is not valid at the region's issue pulse.
    @kernel
    def ugc_from_load(A: i32[4], out: i32[1]):
        out[0] = ugc_add3(A[1])

    # From an ARGUMENT: valid at issue, so this arm passed either way and pins
    # that the offset release did not break it.
    @kernel
    def ugc_from_arg(x: i32, out: i32[1]):
        out[0] = ugc_add3(x)

    A = np.array([10, 20, 30, 40], np.int32)
    out = np.zeros(1, np.int32)
    _to_rtl(ugc_from_load).cosim(A, out)
    assert out[0] == (20 + 3) * 2, int(out[0])

    out = np.zeros(1, np.int32)
    _to_rtl(ugc_from_arg).cosim(np.int32(20), out)
    assert out[0] == (20 + 3) * 2, int(out[0])

    # Several of them in one region, which is what a fully unrolled body is: each
    # is ungated (no call orders another; they share no array), so each waits for
    # its own load rather than all firing together at issue.
    @kernel
    def ugc_many(A: i32[4], out: i32[1]):
        out[0] = ugc_add3(A[0]) + ugc_add3(A[1]) + ugc_add3(A[2]) + ugc_add3(A[3])

    out = np.zeros(1, np.int32)
    _to_rtl(ugc_many).cosim(A, out)
    assert out[0] == sum((int(a) + 3) * 2 for a in A), int(out[0])


# --- What a `done` level means on the pass after the first -------------------


# A kernel `done` is the conjunction of its regions' `done`s, but each of those
# is a level cleared by the start of the region that OWNS it, which for anything
# but the first is later than the kernel's own start. So on a re-invocation the
# later regions still read the previous one's TRUE, and the conjunction rises as
# soon as the FIRST region completes: the caller latches a result the callee has
# not computed yet. Needs no unroll and no interesting memory, only a callee with
# two regions invoked more than once.
def test_a_reinvoked_callee_is_not_done_when_its_first_region_is():
    @kernel
    def dtr_two_regions(a: i32[4]) -> i32:
        acc: i32 = 0
        for i in range(4):
            acc += a[i]
        scaled: i32 = 0
        for j in range(4):
            scaled += acc
        return scaled

    # The same arithmetic in ONE region: no conjunction to read stale, so this
    # arm passed either way and holds the fix to the multi-region case.
    @kernel
    def dtr_one_region(a: i32[4]) -> i32:
        acc: i32 = 0
        for i in range(4):
            acc += a[i] * 4
        return acc

    def caller(callee):
        @kernel
        def dtr_top(src: i32[3, 4], out: i32[3]):
            for t in range(3):
                buf: i32[4] = 0
                for k in range(4):
                    buf[k] = src[t, k]
                out[t] = callee(buf)

        return dtr_top

    src = np.arange(12, dtype=np.int32).reshape(3, 4)
    want = src.sum(axis=1) * 4
    for callee in (dtr_two_regions, dtr_one_region):
        out = np.zeros(3, np.int32)
        _to_rtl(caller(callee)).cosim(src.copy(), out)
        # Every element, not just the first: the first invocation is correct
        # even with a stale conjunction, since nothing has latched a TRUE yet.
        assert np.array_equal(out, want), (callee.__name__, list(out))


# A region that drains in the cycle it is ISSUED sets its `done` on the same
# pulse that clears it. The set wins, so the level latches high and every later
# pass re-sets it from 1, leaving a consumer that watches for a 0->1 edge waiting
# forever. `emitDone` masks the start cycle out of the level for exactly this.
# A comb-only callee is the reachable case: it drains at stage 0 off an acyclic
# region whose issue IS the kernel start, and being tiny it gets re-invoked from
# a loop, where a lost edge is a deadlock rather than a wrong value.
def test_a_zero_drain_callee_re_edges_on_every_invocation():
    @kernel
    def zd_bump(x: i32) -> i32:
        return x + 1

    # zd_bump is invoked once per iteration of the callee's own loop, and the
    # callee once per iteration of the caller's: the edge has to come back at
    # both levels.
    @kernel
    def zd_sum(src: i32[3, 2], n: i32, acc: i32[1]):
        d: i32 = 0
        for i in range(2):
            d += zd_bump(src[n, i])
        acc[0] = d

    @kernel
    def zd_top(src: i32[3, 2], out: i32[3]):
        for n in range(3):
            acc: i32[1] = 0
            zd_sum(src, n, acc)
            out[n] = acc[0]

    src = np.arange(1, 7, dtype=np.int32).reshape(3, 2)
    out = np.zeros(3, np.int32)
    _to_rtl(zd_top).cosim(src.copy(), out)
    assert np.array_equal(out, src.sum(axis=1) + 2), list(out)

    # The invariant itself, at the port: zd_bump's `done` output is its done
    # register ANDed with ~start, so the level reads 0 on the start cycle
    # whatever the region behind it does with that pulse.
    m = Mod(_to_rtl(zd_top).mlir, "zd_top_zd_sum_zd_bump")
    done = m.text.split("hw.output %")[1].split(",")[0].strip()
    assert m.defs[done].startswith("comb.and"), m.defs[done]
    gate = [v for v in m.operands(done) if v != "r0_done"]
    assert len(gate) == 1 and m.defs[gate[0]].startswith("comb.xor %start"), (
        m.defs[done],
        [m.defs.get(v) for v in gate],
    )


# --- Concurrency inference between independent calls -------------------------


# Interprocedural per-argument footprint analysis: disjoint writers and pure
# readers overlap, and a genuine WAW serializes. The port groups those disjoint
# writers get are test_ports.py's subject; here the footprint is only the reason
# no ordering edge is added.
def test_concurrent_shared_array_access():
    # Two sub-kernels WRITING one shared array in disjoint slices: the
    # per-argument callee footprint proves they cannot collide, so no edge is
    # added and the writers overlap.
    @kernel
    def cw1(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i] = A[i] + 1

    @kernel
    def cw2(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i + 8] = A[i + 8] * 2

    @kernel
    def cw_top(A: i32[16], B: i32[16]):
        cw1(A, B)
        cw2(A, B)

    l1 = _latency(cw1)
    l2 = _latency(cw2)
    B = np.zeros(16, np.int32)
    r = _to_rtl(cw_top).cosim(A16, B)
    assert np.array_equal(B, np.concatenate([A16[:8] + 1, A16[8:] * 2]))
    assert r.cycles == max(l1, l2)  # disjoint slices: the writers overlap

    # Two sub-kernels READING one shared input array: neither writes it, so
    # there is no ordering constraint at all.
    @kernel
    def sr1(A: i32[16], o1: i32[16]):
        for i in range(16):
            o1[i] = A[i] + 1

    @kernel
    def sr2(A: i32[16], o2: i32[16]):
        for i in range(16):
            o2[i] = A[i] * 2

    @kernel
    def sr_top(A: i32[16], o1: i32[16], o2: i32[16]):
        sr1(A, o1)
        sr2(A, o2)

    sl1 = _latency(sr1)
    sl2 = _latency(sr2)
    o1 = np.zeros(16, np.int32)
    o2 = np.zeros(16, np.int32)
    r = _to_rtl(sr_top).cosim(A16, o1, o2)
    assert np.array_equal(o1, A16 + 1)
    assert np.array_equal(o2, A16 * 2)
    assert r.cycles == max(sl1, sl2)  # read-only sharing: the readers overlap

    # The dual and the soundness guard: two writers of the SAME elements are a
    # real WAW, so the scheduler orders them and they do not overlap.
    @kernel
    def ow1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def ow2(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] * 2

    @kernel
    def ow_top(A: i32[16], B: i32[16]):
        ow1(A, B)
        ow2(A, B)  # overwrites every element ow1 wrote

    ol1 = _latency(ow1)
    ol2 = _latency(ow2)
    ob = np.zeros(16, np.int32)
    r = _to_rtl(ow_top).cosim(A16, ob)
    assert np.array_equal(ob, A16 * 2)  # the later writer wins: they ran in order
    assert r.cycles == ol1 + ol2  # a real WAW: the writers do NOT overlap

    # A container-local buffer filled by TWO children writing disjoint halves
    # concurrently, then read by a third. The reader conflicts with both
    # writers and so is ordered after both.
    @kernel
    def ibw1(A: i32[16], t: i32[16]):
        for i in range(8):
            t[i] = A[i] + 1

    @kernel
    def ibw2(A: i32[16], t: i32[16]):
        for i in range(8):
            t[i + 8] = A[i + 8] * 2

    @kernel
    def ibrd(t: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = t[i] - 3

    @kernel
    def ibw_top(A: i32[16], out: i32[16]):
        t: i32[16]  # container-local -> on-chip hlmem, two writers + one reader
        ibw1(A, t)
        ibw2(A, t)
        ibrd(t, out)

    mod = _to_rtl(ibw_top)
    # Two writers, neither ordered against the other, so no addressed row has a
    # port for each and the buffer is held in registers.
    assert mod.microarch.mem("t").storage == "register"

    # Every span comes from this compile: `t` is registers here and addressed
    # storage when a child is compiled on its own, and the realization is what
    # times its accesses.
    def span(name):
        return next(
            c.latency for c in mod.microarch.top.calls if c.callee.endswith(name)
        )

    out = np.zeros(16, np.int32)
    r = mod.cosim(A16, out)
    assert np.array_equal(out, np.concatenate([A16[:8] + 1, A16[8:] * 2]) - 3)
    # The writers overlap; the reader waits.
    assert r.cycles == max(span("ibw1"), span("ibw2")) + span("ibrd")


# Two pure-seq calls on disjoint arrays have no shared-memref dependence, so
# the leaf starts them concurrently and finishes in one child's latency.
def test_independent_calls_on_disjoint_arrays_overlap():
    @kernel
    def ov1(A: i32[16], oa: i32[16]):
        for i in range(16):
            oa[i] = A[i] + 1

    @kernel
    def ov2(B: i32[16], ob: i32[16]):
        for i in range(16):
            ob[i] = B[i] * 2

    @kernel
    def ov_top(A: i32[16], B: i32[16], oa: i32[16], ob: i32[16]):
        ov1(A, oa)
        ov2(B, ob)  # disjoint from ov1 -> overlaps it on the leaf

    rtl = _to_rtl(ov_top)
    assert Dcp(rtl).func(rtl.top).callees()  # leaf CallUnit path (structural lock)
    l1, l2 = _latency(ov1), _latency(ov2)
    A = np.arange(16, dtype=np.int32)
    B = np.arange(16, dtype=np.int32) + 100
    oa = np.zeros(16, np.int32)
    ob = np.zeros(16, np.int32)
    r = rtl.cosim(A, B, oa, ob)
    assert np.array_equal(oa, A + 1)
    assert np.array_equal(ob, B * 2)
    assert r.cycles == max(l1, l2)  # concurrent, not l1 + l2


# --- Nested composition -------------------------------------------------------


# Seq-in-seq: a container whose first child is itself a container. The parent
# places the following sibling after the WHOLE inner container's latency.
def test_nested_sequential_composition():
    @kernel
    def nt_leaf1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def nt_leaf2(B: i32[16], C: i32[16]):
        for i in range(16):
            C[i] = B[i] * 2

    @kernel
    def nt_mid(A: i32[16], B: i32[16], C: i32[16]):
        nt_leaf1(A, B)  # B = A + 1
        nt_leaf2(B, C)  # C = (A + 1) * 2

    @kernel
    def nt_leaf3(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + 3

    @kernel
    def nt_top(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        nt_mid(A, B, C)  # a nested CONTAINER child: C = (A + 1) * 2
        nt_leaf3(C, out)  # reads the inner container's output

    lmid = _latency(nt_mid)
    l3 = _latency(nt_leaf3)
    assert lmid is not None and l3 is not None

    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    rtl = _to_rtl(nt_top)
    assert "nt_top.nt_mid" in Dcp(rtl).func("nt_top").callees()
    r = rtl.cosim(A16, B, C, out)
    assert np.array_equal(out, (A16 + 1) * 2 + 3)
    assert r.cycles == lmid + l3  # nt_leaf3 waits for the whole inner container


# Seq-in-seq on the leaf: the inner container instantiates as a plain CallUnit,
# wired exactly like any leaf, since its interface is memory-port based.
def test_nested_container_instantiates_as_a_plain_call():
    @kernel
    def r1b_l1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def r1b_l2(B: i32[16], C: i32[16]):
        for i in range(16):
            C[i] = B[i] * 2

    @kernel
    def r1b_mid(A: i32[16], B: i32[16], C: i32[16]):
        r1b_l1(A, B)  # B = A + 1
        r1b_l2(B, C)  # C = (A + 1) * 2

    @kernel
    def r1b_l3(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + 3

    @kernel
    def r1b_top(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        r1b_mid(A, B, C)  # a nested CONTAINER child (CountedStatic)
        r1b_l3(C, out)  # reads the container's output -> serial on the leaf

    rtl = _to_rtl(r1b_top)
    # The container child instantiates in r1b_top's OWN body, which is what
    # separates the outer invoke from the inner r1b_mid.r1b_l* ones.
    assert Dcp(rtl).func("r1b_top").callees() == [
        "r1b_top.r1b_mid",
        "r1b_top.r1b_l3",
    ]
    A = np.arange(16, dtype=np.int32)
    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    rtl.cosim(A, B, C, out)
    assert np.array_equal(out, (A + 1) * 2 + 3)


# --- Mixed containers (loose datapath beside sub-kernel calls) ---------------


# A container mixing its own datapath with a call mastering only
# container-local buffers: the call instantiates in the container's own module.
def test_mixed_container_internal_buffer_call():
    @kernel
    def ib_child(B: i32[16], C: i32[16]):  # internal -> internal, no boundary
        for i in range(16):
            C[i] = B[i] + 10

    @kernel
    def ib_top(A: i32[16], out: i32[16]):
        B: i32[16]  # region 0 writes B (boundary A -> internal B)
        C: i32[16]  # the child reads B, writes C; the last region reads C
        for i in range(16):
            B[i] = A[i] + 1
        ib_child(B, C)
        for i in range(16):
            out[i] = C[i] * 2

    rtl = _to_rtl(ib_top)
    # a scheduled call node
    assert "ib_top.ib_child" in Dcp(rtl).func("ib_top").callees()
    assert "hw.instance" in rtl.mlir  # instantiated in the container's module
    assert "seq.hlmem" in rtl.mlir  # the shared buffers, on-chip
    A = np.arange(1, 17, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, ((A + 1) + 10) * 2)


# A loose datapath region interleaved between two calls schedules in program
# order against the calls it depends on.
def test_mixed_container_loose_region_between_calls():
    @kernel
    def mr1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1  # boundary read A, internal write B

    @kernel
    def mr2(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] * 2  # internal read C, boundary write out

    @kernel
    def mr_top(A: i32[16], out: i32[16]):
        B: i32[16]
        C: i32[16]
        mr1(A, B)
        for i in range(16):  # loose region between the two calls
            C[i] = B[i] + 5
        mr2(C, out)

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    _to_rtl(mr_top).cosim(A, out)
    assert np.array_equal(out, ((A + 1) + 5) * 2)


# A loose region that writes the boundary output after a call must not be
# silently dropped by the leaf CallUnit path.
def test_loose_region_after_a_call_writes_boundary_output():
    @kernel
    def mcb1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1  # A boundary read, B internal write

    @kernel
    def mcb_top(A: i32[16], out: i32[16]):
        B: i32[16]
        mcb1(A, B)
        for i in range(16):  # loose region writing the top output (parent access)
            out[i] = B[i] + 5

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    _to_rtl(mcb_top).cosim(A, out)
    assert np.array_equal(out, (A + 1) + 5)


# Two ADJACENT calls with no intervening loose op reify into ONE region; they
# still serialize, and the boundary arg they read is wired to one port per
# surviving access.
def test_adjacent_calls_with_no_loose_op_between_them():
    @kernel
    def cc1(x: i32[8], p: i32[8], q: i32[8]):
        for i in range(8):
            p[i] = x[i] + 1  # x read twice (two ports) -> internal p, q
            q[i] = x[i] + 2

    @kernel
    def cc2(p: i32[8], q: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = p[i] * 10 + q[i]  # reads internals cc1 wrote

    @kernel
    def cc_top(x: i32[8], out: i32[8]):
        p: i32[8]
        q: i32[8]
        d: i32[8]
        for i in range(8):
            d[i] = 0  # loose region -> mixed container
        cc1(x, p, q)  # adjacent calls, one region: cc1 must finish before cc2
        cc2(p, q, out)

    rtl = _to_rtl(cc_top)
    # One read group per surviving ACCESS: cc1's two source-level reads of x[i]
    # are the same subscript, so load CSE leaves one. What a group per distinct
    # access buys is tested in test_ports.py.
    rd = [p.base for acc in rtl.interfaces["cc_top"].reads for p in acc]
    assert rd == ["x_rd0"]

    x = np.arange(8, dtype=np.int32) + 1
    out = np.zeros(8, dtype=np.int32)
    rtl.cosim(x, out)
    assert np.array_equal(out, (x + 1) * 10 + (x + 2))


# --- Loop-body call sequencing ------------------------------------------------

A_LOOP16 = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F


# The second child reads what the first wrote in the same iteration: the
# consumer is sequenced on the producer's real done, not a static offset.
def test_two_calls_in_one_loop_body_chained_through_a_buffer():
    @kernel
    def ch_a(A: i32[16], T: i32[16], i: index):
        T[i] = A[i] * 2

    @kernel
    def ch_b(T: i32[16], C: i32[16], i: index):
        C[i] = T[i] + 1

    @kernel
    def ch_top(A: i32[16], C: i32[16]):
        T: i32[16]
        for i in range(16):
            ch_a(A, T, i)
            ch_b(T, C, i)

    C = np.zeros(16, np.int32)
    _to_rtl(ch_top).cosim(A_LOOP16, C)
    assert np.array_equal(C, A_LOOP16 * 2 + 1)


# A call and unrelated arithmetic in one loop body: the loose store is the
# part a lone-call leaf controller would drop.
def test_call_beside_loose_compute_in_a_loop_body():
    @kernel
    def lc_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def lc_top(A: i32[16], B: i32[16], C: i32[16]):
        for i in range(16):
            lc_child(A, B, i)
            C[i] = A[i] + 1

    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    _to_rtl(lc_top).cosim(A_LOOP16, B, C)
    assert np.array_equal(B, A_LOOP16 * 2)
    assert np.array_equal(C, A_LOOP16 + 1)


# Loose work computes a scalar the call consumes, ordered before the call and
# crossing into it as a cross-region survivor.
def test_loose_compute_feeding_the_call_it_shares_a_body_with():
    @kernel
    def sf_child(A: i32[16], B: i32[16], i: index, k: i32):
        B[i] = A[i] * k

    @kernel
    def sf_top(A: i32[16], B: i32[16]):
        for i in range(16):
            k: i32 = A[i] + 1
            sf_child(A, B, i, k)

    B = np.zeros(16, np.int32)
    _to_rtl(sf_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, A_LOOP16 * (A_LOOP16 + 1))


# B[i] = f(A, i): storing a scalar-returning call's result forces the same
# loop-body decomposition as any other loose work sharing the body.
def test_loop_over_a_scalar_returning_call():
    @kernel
    def sr_child(A: i32[16], i: index) -> i32:
        v: i32 = A[i] * 2 + 5
        return v

    @kernel
    def sr_top(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = sr_child(A, i)

    B = np.zeros(16, np.int32)
    _to_rtl(sr_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, A_LOOP16 * 2 + 5)


# Two was the pair; three checks that the sequencing composes rather than
# special-casing a producer/consumer pair.
def test_three_calls_in_one_loop_body():
    @kernel
    def th_a(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def th_b(A: i32[16], C: i32[16], i: index):
        C[i] = A[i] + 1

    @kernel
    def th_c(A: i32[16], D: i32[16], i: index):
        D[i] = A[i] - 1

    @kernel
    def th_top(A: i32[16], B: i32[16], C: i32[16], D: i32[16]):
        for i in range(16):
            th_a(A, B, i)
            th_b(A, C, i)
            th_c(A, D, i)

    B, C, D = (np.zeros(16, np.int32) for _ in range(3))
    _to_rtl(th_top).cosim(A_LOOP16, B, C, D)
    assert np.array_equal(B, A_LOOP16 * 2)
    assert np.array_equal(C, A_LOOP16 + 1)
    assert np.array_equal(D, A_LOOP16 - 1)


# An if guarding a call: the call becomes a predicated child region rather
# than loose work inside a predicated span.
def test_call_guarded_by_an_if_inside_a_loop():
    @kernel
    def gi_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def gi_top(A: i32[16], B: i32[16]):
        for i in range(16):
            if A[i] > 20:
                gi_child(A, B, i)

    B = np.zeros(16, np.int32)
    _to_rtl(gi_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, np.where(A_LOOP16 > 20, A_LOOP16 * 2, 0))


# A body that is exactly one call keeps the cheap leaf loop-over-calls
# controller (one flat dcp.pipeline holding the invoke). Structural only.
def test_a_lone_call_body_stays_on_the_leaf_controller():
    @kernel
    def lone_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def lone_top(A: i32[16], B: i32[16]):
        for i in range(16):
            lone_child(A, B, i)

    top = _to_rtl(lone_top).schedule().func("lone_top")
    assert len(top.cyclic(wrappers=True)) == 1
    # no child region: still a leaf
    assert not [r for r in top.regions if r.kind is RegionKind.ACYCLIC]


# A body with a call plus loose work decomposes into a container with
# sub-regions, so the loose work sequences against the call's real done.
# Structural only.
def test_a_mixed_call_body_becomes_a_container():
    @kernel
    def mx_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def mx_top(A: i32[16], B: i32[16], C: i32[16]):
        for i in range(16):
            mx_child(A, B, i)
            C[i] = A[i] + 1

    top = _to_rtl(mx_top).schedule().func("mx_top")
    # One outer pipeline wrapping a dcp.sequential that holds the invoke, plus
    # a second child region for the loose store.
    assert len(top.cyclic(wrappers=True)) == 1
    assert len([r for r in top.regions if r.kind is RegionKind.ACYCLIC]) == 2
    call_region = next(r for r in top.regions if r.has("dcp.instance"))
    # the call region holds only the call
    assert [o.kind for o in call_region.ops] == ["dcp.instance"]

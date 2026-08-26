# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store->load forwarding on a RAM read-modify-write recurrence."""

import os
import sys

import numpy as np

from allo import kernel
from allo.lang import f32, i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _LAT, PERIOD_NS, _iis, _op_row, default_device  # noqa: E402

N, BINS = 32, 8


def _bumpy(rng):
    # Runs of equal values, so consecutive iterations hit one bin: exactly the
    # collision the shadow serves.
    x = np.repeat(rng.integers(0, BINS, N // 2), 2).astype(np.int32)[:N]
    return x


def _hist():
    @kernel
    def hist(x: i32[N], h: i32[BINS]):
        for i in range(N):
            v: i32 = x[i]
            h[v] = h[v] + 1

    return hist


def test_a_ram_rmw_loop_forwards_the_uncommitted_store():
    # At a 4 ns period the load->add->store chain fits one cycle, so the store
    # issues in the same cycle as the next iteration's load and the relaxed
    # recurrence reaches II=1. Without forwarding the round trip pins II at 3.
    s = _hist().schedule()
    mod = s.export("rtl", freq_mhz=250)
    assert _iis(mod.schedule().func("hist").regions) == [1]

    rng = np.random.default_rng(0)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))


def test_forwarding_survives_the_chain_break_at_the_default_clock():
    # At the default clock the chain is split and the store lands two cycles
    # after the load, so the shadow pairs iteration k's store with iteration
    # k+1's load. One cycle better than the unforwarded round trip.
    s = _hist().schedule()
    mod = s.export("rtl")
    assert _iis(mod.schedule().func("hist").regions) == [2]

    rng = np.random.default_rng(1)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))


def test_a_cone_fed_store_still_earns_the_window():
    # A floyd-style min-relax: the store's datum is a select cone,
    # combinational when the store issues. The youngest window arm taps that
    # cone straight into the load's data mux, so its delay is priced onto the
    # load's output rather than refusing the window, and the recurrence
    # closes one read latency tighter than the plain (windowless) relaxation.
    N = 32

    @kernel
    def relax(A: f32[N], acc: f32[4]):
        for j in range(N):
            t: f32 = acc[0] + A[j]
            if acc[0] >= t:
                acc[0] = t

    fadd, fcmp = _LAT[("add", "float32")], _LAT[("cmp", "float32")]
    row = default_device.storage["lutram"]
    arrival = row.read_delay_ns + _op_row("add", "float32", fadd).timing.in_delay_ns
    w = row.read_latency
    while w and row.read_latency + fadd + fcmp - w < w + 1:
        w = max(1, row.read_latency + fadd + fcmp - w) - 1
    expect = row.read_latency - w + fadd + fcmp + (0 if arrival <= PERIOD_NS else 1)

    mod = relax.schedule().export("rtl")
    assert _iis(mod.schedule().func("relax").regions) == [expect]

    rng = np.random.default_rng(3)
    A = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    acc = np.full(4, 2.0, np.float32)
    exp = acc.copy()
    for j in range(N):
        t = np.float32(exp[0] + A[j])
        if exp[0] >= t:
            exp[0] = t
    mod.cosim(A, acc)
    assert np.allclose(acc, exp, rtol=2e-3, atol=2e-3), (list(acc), list(exp))


def test_an_unrolled_rmw_body_forwards_from_every_paired_store():
    # Unrolled by two, each load pairs with both stores (the same-iteration one
    # at distance 0, the carried ones at distance 1) and its data out muxes over
    # several shadow arms. At most one arm matches in a cycle, and the duplicate
    # indices make every arm fire somewhere in the run.
    s = _hist().schedule()
    s.unroll(s.loop("i"), factor=2)
    mod = s.export("rtl", freq_mhz=250)

    rng = np.random.default_rng(2)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))

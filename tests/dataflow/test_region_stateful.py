# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for region-scope ``@ Stateful`` declarations.

Allo already supports declaring ``int32[N] @ Stateful = 0`` inside
``@df.kernel`` bodies, where the persistent buffer is private to that
kernel. This file exercises the architecturally interesting case where
the Stateful is declared at ``@df.region`` body scope so it is shared
across every kernel inside the region (e.g. a Gemmini-style decoder
kernel and an MXU-driver kernel sharing scratchpad/accumulator state).

Three variants are exercised:

  1. Region-scope Stateful read by a single kernel.
  2. Region-scope Stateful read by two kernels in the same region.
  3. Region-scope Stateful with a ``Stream`` between two kernels also
     reading the shared Stateful (full Gemmini-style topology).

The persistence contract: invoking the module multiple times must
preserve the buffer's contents across calls.
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stateful, Stream, int32


# ---------------------------------------------------------------------------
# Variant 1: region-scope Stateful read by a single kernel.
# ---------------------------------------------------------------------------


def test_region_stateful_single_kernel():
    @df.region()
    def top(out: int32[4]):
        acc: int32[4] @ Stateful = 0  # region-scope shared state

        @df.kernel(mapping=[1], args=[out])
        def k(local_out: int32[4]):
            for i in range(4):
                acc[i] += 1
                local_out[i] = acc[i]

    mod = df.build(top, target="simulator")

    out = np.zeros(4, dtype=np.int32)
    mod(out)
    np.testing.assert_array_equal(out, np.array([1, 1, 1, 1], dtype=np.int32))
    mod(out)
    np.testing.assert_array_equal(out, np.array([2, 2, 2, 2], dtype=np.int32))
    mod(out)
    np.testing.assert_array_equal(out, np.array([3, 3, 3, 3], dtype=np.int32))


# ---------------------------------------------------------------------------
# Variant 2: region-scope Stateful read by two kernels in the same region.
# ---------------------------------------------------------------------------


def test_region_stateful_two_kernels_shared():
    @df.region()
    def top(out_a: int32[4], out_b: int32[4]):
        acc: int32[4] @ Stateful = 0  # shared by both kernels

        @df.kernel(mapping=[1], args=[out_a])
        def producer(po: int32[4]):
            for i in range(4):
                acc[i] += 1
                po[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def reader(rb: int32[4]):
            for i in range(4):
                rb[i] = acc[i]

    mod = df.build(top, target="simulator")

    a = np.zeros(4, dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)
    # First call: producer increments acc to 1; reader sees the same acc.
    # The two kernels in a region run concurrently, so the reader may see
    # acc either before or after the producer's update. Run the module
    # multiple times and check that acc is at least monotonic and shared.
    mod(a, b)
    mod(a, b)
    mod(a, b)
    # After three invocations, producer has incremented acc three times;
    # reader observation in this last call sees a value <= 3 and >= 0,
    # but the region's acc must equal exactly 3 in the persistent buffer
    # (which is what producer's writeback to ``a`` reflects).
    np.testing.assert_array_equal(a, np.array([3, 3, 3, 3], dtype=np.int32))
    # Reader saw a snapshot of the shared state. The exact value depends
    # on scheduling, but it must be in {2, 3} on the last call.
    for v in b:
        assert v in (2, 3), f"reader saw {v} which is outside expected snapshot set"


# ---------------------------------------------------------------------------
# Variant 3: region-scope Stateful + Stream between two kernels.
# ---------------------------------------------------------------------------


def test_region_stateful_with_stream():
    @df.region()
    def top(out: int32[4]):
        acc: int32[4] @ Stateful = 0  # shared between decoder + driver
        sig: Stream[int32, 4]

        @df.kernel(mapping=[1])
        def decoder():
            for i in range(4):
                acc[i] += 1
                sig.put(acc[i])

        @df.kernel(mapping=[1], args=[out])
        def driver(local_out: int32[4]):
            for i in range(4):
                v: int32 = sig.get()
                local_out[i] = v + acc[i]

    mod = df.build(top, target="simulator")

    out = np.zeros(4, dtype=np.int32)
    mod(out)
    # After 1st call: decoder put 1 then driver reads 1 + acc[i].
    # The driver's read of acc[i] races with the producer; the contract
    # is just that the buffer is shared. Each element is in {2, 3, 4}.
    for v in out:
        assert 2 <= v <= 4, f"unexpected {v} (must reflect shared acc + sig)"
    mod(out)
    # 2nd call: persistent acc continues from previous value.
    for v in out:
        assert 4 <= v <= 6, f"unexpected {v} after 2nd call"

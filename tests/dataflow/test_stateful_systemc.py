# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""`x: T @ Stateful` through the SystemC/Catapult backend.

A stateful variable lowers to a private `memref.global` (name prefixed `__stateful_`,
tagged `static`, carrying an initial value). Vitis declares it as a function-scope
`static`, which is right for a function CALLED REPEATEDLY. A SystemC kernel is an
SC_THREAD entered once that loops internally, so the emitter instead declares it in the
RESET ACTION, above the body: that gives per-instance state which the RTL reset
re-initialises. A `static` would be shared across module instances AND skipped by reset.

Cosim is the load-bearing check here, not csim. csim initialises the variable in C++;
only cosim proves the RTL reset produces the same values -- exactly the class of
divergence the `#ifdef __SYNTHESIS__` splits can hide.
"""

import numpy as np

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stateful, Stream

N = 8
W = 4


def test_stateful_scalar_systemc():
    """A running sum: state must survive from one loop iteration to the next."""

    @df.region()
    def top(A: int32[N], B: int32[N]):
        S: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for i in range(N):
                S.put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            run_sum: int32 @ Stateful = 0
            for i in range(N):
                run_sum = run_sum + S.get()
                b[i] = run_sum

    A = np.arange(1, N + 1, dtype=np.int32)
    B = np.zeros(N, dtype=np.int32)
    expected = np.cumsum(A).astype(np.int32)

    mod = df.build(top, target="systemc", mode="cosim",
                   project="test_stateful_scalar_systemc")
    mod(A, B)
    np.testing.assert_array_equal(B, expected)
    print("stateful scalar: SystemC cosim passed!")


def test_stateful_array_systemc():
    """A 4-tap moving sum: a stateful ARRAY plus a stateful index.

    Covers the rank-N path (`T name[W] = {0, 0, 0, 0};`) alongside the rank-0 scalar,
    and would fail if either were emitted `const` -- both are written every iteration.
    """

    @df.region()
    def top(A: int32[N], B: int32[N]):
        S: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for i in range(N):
                S.put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            window: int32[W] @ Stateful = 0
            pos: int32 @ Stateful = 0
            for i in range(N):
                window[pos] = S.get()
                acc: int32 = 0
                for k in range(W):
                    acc = acc + window[k]
                b[i] = acc
                pos = (pos + 1) % W

    A = np.arange(1, N + 1, dtype=np.int32)
    B = np.zeros(N, dtype=np.int32)

    win = [0] * W
    pos, out = 0, []
    for v in A:
        win[pos] = int(v)
        out.append(sum(win))
        pos = (pos + 1) % W
    expected = np.array(out, dtype=np.int32)

    mod = df.build(top, target="systemc", mode="cosim",
                   project="test_stateful_array_systemc")
    mod(A, B)
    np.testing.assert_array_equal(B, expected)
    print("stateful array: SystemC cosim passed!")


if __name__ == "__main__":
    test_stateful_scalar_systemc()
    test_stateful_array_systemc()

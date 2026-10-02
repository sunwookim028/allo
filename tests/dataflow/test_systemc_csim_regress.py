# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SystemC csim regressions: each test COMPILES AND RUNS the emitted testbench.

The emit-only tests check the text the emitter writes; every bug below passed
them. Each test here builds with ``mode="csim"`` and calls the module, so g++
and libsystemc see the code. They skip without Catapult (``MGC_HOME``) and a
SystemC install (``SYSTEMC_HOME``). See docs/source/backends/systemc.rst,
"Known fixed".
"""

import os
import tempfile

import ml_dtypes
import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import bfloat16, float16, float32, int32, Stream

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

_NP = {bfloat16: ml_dtypes.bfloat16, float16: np.float16, float32: np.float32}


def _csim(top, *args):
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        mod(*args)


def _square_stream(T, N):
    @df.region()
    def top(A: T[N], B: T[N]):
        S: Stream[T, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: T[N]):
            for i in range(N):
                S.put(a[i] * a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: T[N]):
            for i in range(N):
                b[i] = S.get()

    return top


@needs_csim
@pytest.mark.parametrize("T", [bfloat16, float16, float32])
def test_float_stream_compiles_and_runs(T):
    """Bug 1: a float Connections port needs an sc_trace the port can find.

    Connections and sc_signal call sc_trace unqualified from inside their own
    namespaces, so only argument-dependent lookup finds an overload: bf16's must
    be in namespace ac. A global one compiled in every emit test and failed g++.
    """
    N = 8
    a = np.array([0.5, 1.5, -2, 3, 0.25, -0.75, 4, 1], dtype=_NP[T])
    b = np.zeros(N, dtype=_NP[T])
    _csim(_square_stream(T, N), a, b)
    np.testing.assert_array_equal(b.astype(np.float32), (a * a).astype(np.float32))


def _two_in_two_out(N):
    @df.region()
    def top(A: int32[N], B: int32[N], C: int32[N], D: int32[N]):
        @df.kernel(mapping=[1], args=[A, B, C, D])
        def k(a: int32[N], b: int32[N], c: int32[N], d: int32[N]):
            for i in range(N):
                x: int32 = a[i]
                y: int32 = b[i]
                c[i] = x + y
                d[i] = x - y

    return top


@needs_csim
def test_testbench_feeds_streams_independently():
    """Bug 2: two input streams popped interleaved, two outputs pushed interleaved.

    The testbench pushed all of input 0 before any of input 1 (and drained all of
    output 0 before output 1) from one thread, over channels that hold no data:
    the csim hung after one element. One thread per stream now.
    """
    N = 64
    a = np.arange(N, dtype=np.int32) * 3
    b = np.arange(N, dtype=np.int32) - 7
    c = np.zeros(N, dtype=np.int32)
    d = np.zeros(N, dtype=np.int32)
    _csim(_two_in_two_out(N), a, b, c, d)
    np.testing.assert_array_equal(c, a + b)
    np.testing.assert_array_equal(d, a - b)

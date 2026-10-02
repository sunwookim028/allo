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
from allo.ir.types import bfloat16, float16, float32, int32, uint16, UInt, Stream

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


def _copy(T, N, reverse):
    """``reverse=False``: in-order scans (stream ports). ``True``: reversed
    indexing, which makes both boundary arrays addressable memory ports (the
    preload / read-out path of the testbench)."""
    if reverse:

        @df.region()
        def top(A: T[N], B: T[N]):
            @df.kernel(mapping=[1], args=[A, B])
            def k(a: T[N], b: T[N]):
                for i in range(N):
                    b[N - 1 - i] = a[N - 1 - i]

    else:

        @df.region()
        def top(A: T[N], B: T[N]):
            @df.kernel(mapping=[1], args=[A, B])
            def k(a: T[N], b: T[N]):
                for i in range(N):
                    b[i] = a[i]

    return top


_SPECIAL_BITS = {
    # +0, +-inf, NaN of both signs and several payloads, smallest/largest
    # subnormal, smallest normal, and values whose shortest decimal reads back
    # BELOW them through a truncating float->bf16 conversion (1 + ulp, 3 ulps).
    # -0 is in test_signed_zero_survives_a_signal: it needs the signal fix too.
    bfloat16: [0x0000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF,
               0x0001, 0x807F, 0x0080, 0x3F81, 0x3F83, 0xC0A3, 0x7F7F, 0x3EAB],
    float16: [0x0000, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0xFFFF,
              0x0001, 0x83FF, 0x0400, 0x3C01, 0x3555, 0xC248, 0x7BFF, 0x2E66],
    float32: [0x00000000, 0x7F800000, 0xFF800000, 0x7FC00000,
              0xFFC00000, 0x7F800001, 0xFFFFFFFF, 0x00000001, 0x807FFFFF,
              0x00800000, 0x3F800001, 0x3EAAAAAB, 0xC0490FDB, 0x7F7FFFFF,
              0x3DCCCCCD],
}


@needs_csim
@pytest.mark.parametrize("reverse", [False, True], ids=["stream", "memport"])
@pytest.mark.parametrize("T", [bfloat16, float16, float32])
def test_float_io_is_bit_exact(T, reverse):
    """Bug 3: the testbench's data files carried floats as decimal text.

    ``>> float`` fails on "nan"/"inf" and every read after it, NaN sign and
    payload were lost, and ac::bfloat16(float) truncates, so a bf16 value's
    shortest decimal came back as the bf16 below it. Files now carry raw bits.
    A copy must return every pattern unchanged, NaNs included.
    """
    bits = np.array(_SPECIAL_BITS[T], dtype=np.uint32 if T is float32 else np.uint16)
    a = bits.view(_NP[T])
    b = np.zeros(len(a), dtype=_NP[T])
    _csim(_copy(T, len(a), reverse), a, b)
    got = b.view(bits.dtype)
    assert [hex(x) for x in got] == [hex(x) for x in bits]


@needs_csim
@pytest.mark.parametrize("reverse", [False, True], ids=["stream", "memport"])
@pytest.mark.parametrize("T", [bfloat16, float16, float32])
def test_signed_zero_survives_a_signal(T, reverse):
    """Bug 4: sc_signal skips a write that compares `==` to its current value.

    IEEE says +0 == -0, so a -0 after a +0 (or the reverse) never reached the
    reader of a Connections channel or a memory pin; csim flipped the sign of
    zeros that the RTL's wires carry. Float signals now compare bits.
    """
    w = 32 if T is float32 else 16
    sign = 1 << (w - 1)
    bits = np.array([0, sign, 0, 0, sign, sign, 0, sign], dtype=np.uint32 if w == 32 else np.uint16)
    a = bits.view(_NP[T])
    b = np.zeros(len(a), dtype=_NP[T])
    _csim(_copy(T, len(a), reverse), a, b)
    got = b.view(bits.dtype)
    assert [hex(x) for x in got] == [hex(x) for x in bits]


def _uint_copy(N):
    @df.region()
    def top(A: uint16[N], C: uint16[N]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: uint16[N], c: uint16[N]):
            for i in range(N):
                c[i] = a[i]

    return top


@needs_csim
def test_uint_ports_compile_and_run():
    """S1: a UInt port's sign comes from the function's ``itypes``.

    A kernel's write port (stores carry no ``unsigned`` attr) and every region
    port (no tagged user at all) emitted as signed ``ac_int``, so binding them
    to the unsigned read port failed g++ for any ``uint16`` region.
    """
    N = 8
    a = np.array([0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xBEEF, 0x8001], dtype=np.uint16)
    c = np.zeros(N, dtype=np.uint16)
    _csim(_uint_copy(N), a, c)
    np.testing.assert_array_equal(c, a)

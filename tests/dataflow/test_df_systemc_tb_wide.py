# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemC emitter: conditional argument reads (S6) and > 64-bit ports (S7).

S6: an argument array read under an ``if`` became a conditional ``Pop()``
on the Connections stream the array is turned into, so the stream fell out
of step with the loop: wrong values, no diagnostic. Such an array is now a
memory port, which is read at the index the body asks for.

S7/S8: the testbench moved every integer port through ``long long``: a
``UInt(256)`` port failed at read-back (``KeyError 'ui256'``) and Catapult
refused the file (CRD-413); a 64-bit value >= 2^63 overflowed the
extraction, which stored LLONG_MAX, set failbit and lost every later value
of that file (2/67,717 slots at the word array). Every ac_int port, at any
width, now goes through decimal text by digit arithmetic
(``_rdwide``/``_wrwide``), the same text the Python side writes and parses;
an unreadable value aborts the testbench instead of sticking.

The emit-only tests run everywhere; the csim ones compile and run the
testbench and skip without Catapult (``MGC_HOME``) and ``SYSTEMC_HOME``.
See ``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import os
import re
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import UInt, Int, int32

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

N = 6


def _conditional_read_region():
    """q[t] = the last d[t] at which c[t] != 0: d is read under an if."""

    @df.region()
    def top(C: int32[N], D: int32[N], Q: int32[N]):
        @df.kernel(mapping=[1], args=[C, D, Q])
        def k(c: int32[N], d: int32[N], q: int32[N]):
            acc: int32 = 0
            for t in range(N):
                if c[t] != 0:
                    acc = d[t]
                q[t] = acc

    return top


S6_C = np.array([1, 0, 0, 1, 0, 1], np.int32)
S6_D = np.arange(10, 10 + N, dtype=np.int32)
S6_WANT = [10, 10, 10, 13, 13, 15]


def _obj(vals):
    out = np.empty(len(vals), dtype=object)
    out[:] = vals
    return out


W = 4  # elements per wide array


def _wide_stream_region(T):
    """Pure sequential in/out: both arrays become stream ports. (``T`` is
    only in annotations, so each caller holds a local ``T``: allo resolves
    annotation names up the call stack.)"""

    @df.region()
    def top(A: T[W], B: T[W]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: T[W], b: T[W]):
            for t in range(W):
                b[t] = a[t] + 1

    return top


def _wide_memory_region(T):
    """Reversed access: both arrays become memory ports (preloaded and read
    out by the testbench through the wide helpers)."""

    @df.region()
    def top(A: T[W], B: T[W]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: T[W], b: T[W]):
            for t in range(W):
                b[t] = a[W - 1 - t] + 1

    return top


WIDE_U = [0, (1 << 64) + 5, (1 << 200) + 3, (1 << 256) - 2]
WIDE_S = [-1, -(1 << 127), (1 << 127) - 2, 7]
U64 = [1, (1 << 63) - 1, (1 << 63) + 5, (1 << 64) - 2]  # S8: >= 2^63 and after
U1024 = [1, (1 << 1023) + 7, (1 << 1024) - 2, 3]


# --- emit-only -----------------------------------------------------------

def test_conditional_read_is_not_a_stream():
    code = df.build(_conditional_read_region(), target="systemc").hls_code
    body = code.split("SC_MODULE(k_0)")[1].split("SC_MODULE(top)")[0]
    # c is read once per iteration: it stays a stream, popped at the top of
    # the loop body. d is read under the if: a memory port (read-address and
    # read-enable pins), never a Pop(). Was: two Pops, d's inside the if.
    assert body.count(".Pop()") == 1, body
    assert "_radr" in body and "_re" in body, body
    if_block = body.split("if (")[1].split("}")[0]
    assert ".Pop()" not in if_block, if_block


def test_wide_port_testbench_avoids_long_long():
    T = UInt(256)
    for region in (_wide_stream_region(T), _wide_memory_region(T)):
        code = df.build(region, target="systemc").hls_code
        tb = code.split("SC_MODULE(tb)")[1]
        assert "_rdwide(" in tb and "_wrwide(" in tb, tb
        assert "(long long)" not in tb and "long long _v" not in tb, tb


def test_every_integer_port_takes_the_digit_reader():
    """Ports are Catapult-native ac_int at every width (int32 included), so
    none of them goes through `long long` any more; a float port keeps its
    raw-bits path."""
    T = int32
    code = df.build(_wide_stream_region(T), target="systemc").hls_code
    tb = code.split("SC_MODULE(tb)")[1]
    assert "_rdwide(" in tb and "_wrwide(" in tb
    assert "long long _v" not in tb


def test_64bit_ac_int_port_avoids_long_long():
    """S8: a UInt(64) port is an ac_int<64, false>; `>> long long` of a
    value >= 2^63 fails and sticks, so it takes the digit reader too."""
    T = UInt(64)
    code = df.build(_wide_stream_region(T), target="systemc").hls_code
    tb = code.split("SC_MODULE(tb)")[1]
    assert "_rdwide(" in tb and "_wrwide(" in tb, tb
    assert "long long _v" not in tb
    assert "_f.fail()" in tb  # an unreadable value aborts, never LLONG_MAX


# --- csim ----------------------------------------------------------------

def _csim(top, *args):
    with tempfile.TemporaryDirectory() as tmp:
        df.build(top, target="systemc", mode="csim", project=tmp)(*args)


@needs_csim
def test_s6_conditional_read_matches_simulator():
    sim = np.zeros(N, np.int32)
    df.build(_conditional_read_region(), target="simulator")(S6_C, S6_D, sim)
    assert sim.tolist() == S6_WANT
    got = np.zeros(N, np.int32)
    _csim(_conditional_read_region(), S6_C, S6_D, got)
    assert got.tolist() == S6_WANT  # was [10, 10, 10, 11, 11, 12]


@needs_csim
@pytest.mark.parametrize(
    "T, vals",
    [
        (UInt(64), U64),  # S8
        (UInt(256), WIDE_U),
        (UInt(1024), U1024),
        (Int(128), WIDE_S),
        (UInt(72), [1 << 70, 1, 2, (1 << 72) - 2]),
    ],
)
def test_s7_s8_wide_stream_ports_round_trip(T, vals):
    b = _obj([0] * 4)
    _csim(_wide_stream_region(T), _obj(vals), b)
    assert b.tolist() == [v + 1 for v in vals]


@needs_csim
def test_s8_uint64_numpy_array_round_trip():
    """The record's repro: a uint64 numpy array with a value >= 2^63; every
    value after it used to read back as LLONG_MAX."""
    T = UInt(64)
    a = np.array([1, 2**63 - 1, 2**63 + 5, 7], np.uint64)
    b = np.zeros(4, np.uint64)
    _csim(_wide_stream_region(T), a, b)
    assert b.tolist() == [2, 2**63, 2**63 + 6, 8]


@needs_csim
def test_s7_wide_memory_ports_round_trip():
    T = UInt(256)
    b = _obj([0] * 4)
    _csim(_wide_memory_region(T), _obj(WIDE_U), b)
    assert b.tolist() == [v + 1 for v in reversed(WIDE_U)]


@needs_csim
def test_s7_wide_result_into_uint64_refused_when_it_does_not_fit():
    T = UInt(256)
    a = np.array([1, 2, 3, (1 << 64) - 1], np.uint64)
    b = np.zeros(4, np.uint64)
    with pytest.raises(OverflowError):
        _csim(_wide_stream_region(T), a, b)


if __name__ == "__main__":
    pytest.main([__file__])

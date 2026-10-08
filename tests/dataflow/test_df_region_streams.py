# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A plain ``@df.kernel`` region refuses a stream with one or no endpoint (E1).

The ``@df.unit`` netlist path refused it; a region of nested kernels built and
ran a stream no kernel touched, and built (and emitted SystemC for) one that
was only written, which blocks once full. A one-ended stream used only
through the non-blocking ops never blocks and stays legal. See
``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.netlist import NetlistError

N = 8


def test_untouched_stream_refused():
    with pytest.raises(NetlistError) as err:

        @df.region()
        def untouched(A: int32[N], B: int32[N]):
            dangling: Stream[int32, 4]

            @df.kernel(mapping=[1], args=[A, B])
            def k(a: int32[N], b: int32[N]):
                for i in range(N):
                    b[i] = a[i] + 1

    msg = str(err.value)
    assert "unconnected-stream" in msg and "untouched.dangling" in msg
    assert "no writer and no reader" in msg


def test_written_only_stream_refused():
    with pytest.raises(NetlistError) as err:

        @df.region()
        def written_only(A: int32[N], B: int32[N]):
            orphan: Stream[int32, 4]

            @df.kernel(mapping=[1], args=[A, B])
            def k(a: int32[N], b: int32[N]):
                for i in range(N):
                    b[i] = a[i] + 1
                    orphan.put(a[i])

    msg = str(err.value)
    assert "written_only.orphan" in msg
    assert "no reader, and k blocks on orphan.put() once it is full" in msg


def test_read_only_stream_refused():
    with pytest.raises(NetlistError) as err:

        @df.region()
        def read_only(A: int32[N], B: int32[N]):
            empty: Stream[int32, 4]

            @df.kernel(mapping=[1], args=[B])
            def k(b: int32[N]):
                for i in range(N):
                    b[i] = empty.get()

    assert "no writer, and k blocks on empty.get()" in str(err.value)


def test_one_sided_non_blocking_stream_accepted():
    """try_put and full() never block: a one-ended stream used only through
    them is a probe of the non-blocking ops, not a dangling channel."""

    @df.region()
    def probe(F: int32[N]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[F])
        def k(f: int32[N]):
            for i in range(N):
                ok: int32 = s.try_put(i)
                f[i] = s.full() + ok - ok

    f = np.zeros(N, dtype=np.int32)
    df.build(probe, target="simulator")(f)
    np.testing.assert_array_equal(f, [0, 0, 0, 1, 1, 1, 1, 1])


def test_connected_streams_accepted_and_run():
    @df.region()
    def chain(A: int32[N], B: int32[N]):
        s: Stream[int32, 4]
        pipe: Stream[int32, 4][2]

        @df.kernel(mapping=[1], args=[A])
        def src(a: int32[N]):
            for i in range(N):
                s.put(a[i])

        @df.kernel(mapping=[2])
        def mid():
            j = df.get_pid()
            with allo.meta_if(j == 0):
                for i in range(N):
                    pipe[0].put(s.get() + 1)
            with allo.meta_else():
                for i in range(N):
                    pipe[1].put(pipe[0].get() * 2)

        @df.kernel(mapping=[1], args=[B])
        def snk(b: int32[N]):
            for i in range(N):
                b[i] = pipe[1].get()

    mod = df.build(chain, target="simulator")
    a = np.arange(N, dtype=np.int32)
    b = np.zeros(N, dtype=np.int32)
    mod(a, b)
    np.testing.assert_array_equal(b, (a + 1) * 2)


def test_self_stream_accepted():
    """One kernel that both puts and gets is both endpoints."""

    @df.region()
    def loop(A: int32[N], B: int32[N]):
        q: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[N], b: int32[N]):
            for i in range(N):
                q.put(a[i])
                b[i] = q.get()

    mod = df.build(loop, target="simulator")
    a = np.arange(N, dtype=np.int32)
    b = np.zeros(N, dtype=np.int32)
    mod(a, b)
    np.testing.assert_array_equal(b, a)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

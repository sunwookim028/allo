# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-25: flushable streams, ``Stream[T, D, flush]`` and ``s.flush()``.

A producer puts one token every step and runs ahead of its consumer, which
takes one token every other step and flushes on chosen steps. Each step is
ordered by two control streams (the composition's obligation,
``docs/source/developer/stream_ports.rst``): the producer's ``ack`` follows
its data put, and the consumer takes it before it flushes or gets, so the put
of the flush step is already buffered and is dropped with the rest; the
consumer's ``k`` ends its step, and the producer takes it before its next put,
so every put after the flush survives. On the simulator and in SystemC csim
the consumer must see exactly the model's tokens: the next token after a
flush is the first put after it.

Refusals, each naming the channel: ``flush()`` from the producer side, on a
stream not declared flushable, and on every HLS backend but SystemC.
"""

import os
import re
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import Stream, flush, int32, uint1

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

N = 40
FLUSH_EVERY = 6  # flushes on steps 5, 11, 17, ...


def _region(n, depth=4):
    @df.region()
    def top(F: uint1[n], O: int32[n]):
        s: Stream[int32, depth, flush]
        ack: Stream[uint1, 2]
        k: Stream[uint1, 2]

        @df.kernel(mapping=[1], args=[])
        def prod():
            for t in range(n):
                v: int32 = t + 1
                s.put(v)
                ack.put(1)
                go: uint1 = k.get()

        @df.kernel(mapping=[1], args=[F, O])
        def cons(f: uint1[n], o: int32[n]):
            for t in range(n):
                a: uint1 = ack.get()
                x: int32 = 0
                if f[t]:
                    s.flush()
                elif t % 2 == 1:
                    x = s.get()
                o[t] = x
                k.put(1)

    return top


def _flags(n):
    return np.array([1 if t % FLUSH_EVERY == FLUSH_EVERY - 1 else 0 for t in range(n)], np.uint8)


def _model(flags):
    """The consumer's view: a FIFO of every put; a flush empties it, the put
    of its own step included (that put precedes the flush in the step)."""
    q, out = [], []
    for t, f in enumerate(flags):
        q.append(t + 1)
        # the obligation the trace keeps: never more than the depth buffered,
        # or the producer blocks in put while the consumer waits for its ack
        assert len(q) <= 4, f"step {t}: {len(q)} tokens buffered in a depth-4 stream"
        if f:
            q.clear()
            out.append(0)
        elif t % 2 == 1:
            out.append(q.pop(0))
        else:
            out.append(0)
    return np.array(out, np.int32)


def _check(o, flags):
    want = _model(flags)
    assert np.array_equal(o, want), (list(o), list(want))
    # spelled out: after each flush the next token seen is the first put after it
    last = None
    for u in range(len(o)):
        if flags[u]:
            last = u
        elif o[u] and last is not None:
            assert int(o[u]) == last + 2, (last, int(o[u]))  # step last+1's put
            last = None


def test_flush_simulator():
    flags = _flags(N)
    o = np.zeros(N, np.int32)
    df.build(_region(N), target="simulator")(flags, o)
    _check(o, flags)
    assert any(o) and _model(flags)[FLUSH_EVERY + 1] == FLUSH_EVERY + 1  # step 6 put: dropped


def test_flush_emit_systemc():
    """The channel is the fork's cleared FIFO; the consumer drives its clear as
    a toggle and resets it; the producer has no clear port."""
    code = df.build(_region(8), target="systemc").hls_code
    assert re.search(r"AlloFifoClr< ac_int<32, true>, 4 > \w+_fifo;", code), code
    assert re.search(r"\w+_fifo\.clr\(\w+_clr\);", code)
    cons = re.search(r"SC_MODULE\(cons_0\) \{(.*?)\n\};", code, re.S).group(1)
    prod = re.search(r"SC_MODULE\(prod_0\) \{(.*?)\n\};", code, re.S).group(1)
    port = re.search(r"sc_out<bool> (\w+)_clr;  // flush toggle", cons).group(1)
    assert f"{port}_tog = false;" in cons and f"{port}_clr.write(false);" in cons
    assert f"{port}_tog = !{port}_tog; {port}_clr.write({port}_tog);" in cons
    assert "_clr" not in prod


@needs_csim
def test_flush_csim():
    flags = _flags(N)
    o = np.zeros(N, np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_region(N), target="systemc", mode="csim", project=tmp)
        mod(flags, o)
    _check(o, flags)


@needs_csim
def test_flush_consecutive_csim():
    """Flushes on consecutive steps (T-5's case for the 1-bit epoch): two
    clears, no stale token."""
    flags = _flags(N)
    flags[[6, 7, 18, 30]] = 1
    o = np.zeros(N, np.int32)
    df.build(_region(N), target="simulator")(flags, o)
    _check(o, flags)
    o2 = np.zeros(N, np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_region(N), target="systemc", mode="csim", project=tmp)
        mod(flags, o2)
    _check(o2, flags)


def test_flush_from_producer_refused():
    @df.region()
    def top(O: int32[4]):
        s: Stream[int32, 4, flush]

        @df.kernel(mapping=[1], args=[])
        def prod():
            for t in range(4):
                s.put(t)
                s.flush()

        @df.kernel(mapping=[1], args=[O])
        def cons(o: int32[4]):
            for t in range(4):
                o[t] = s.get()

    with pytest.raises(Exception) as ex:
        df.build(top, target="simulator")
    msg = str(ex.value)
    assert "s.flush() in kernel prod" in msg and "puts to it" in msg, msg


def test_flush_undeclared_refused(capfd):
    @df.region()
    def top(O: int32[4]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[])
        def prod():
            for t in range(4):
                s.put(t)

        @df.kernel(mapping=[1], args=[O])
        def cons(o: int32[4]):
            for t in range(4):
                s.flush()
                o[t] = s.get()

    with pytest.raises((RuntimeError, SystemExit)):
        df.build(top, target="simulator")
    out, err = capfd.readouterr()
    assert "stream `s` is not declared flushable" in out + err


@pytest.mark.parametrize("target", ["vhls", "catapult"])
def test_flush_other_backends_refused(target):
    with pytest.raises(NotImplementedError) as ex:
        df.build(_region(8), target=target)
    assert "flushable stream `s`" in str(ex.value), str(ex.value)


def test_flush_marker_value():
    t = Stream[int32, 4, flush]
    assert t.flush_ok and t.depth == 4
    assert not Stream[int32, 4].flush_ok
    with pytest.raises(AssertionError):
        Stream[int32, 4, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

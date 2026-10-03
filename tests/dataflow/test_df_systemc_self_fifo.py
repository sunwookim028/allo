# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A self-FIFO (one kernel both puts and gets a Stream) in the SystemC
emitter: ``try_put`` is refused exactly when ``full()`` (S9), and synthesis
modes refuse the construct that Catapult cannot schedule (C1).

S9: the self-FIFO's ``full()`` reads a synchronous occupancy counter that
``try_put`` advanced by its ``PushNB`` result -- but the enq Combinational
holds one value in flight beyond the AlloFifo's depth, so the fifth push
into a ``Stream[int32, 4]`` still succeeded, the counter ran to depth + 1
and ``full()`` was never true again: ``[0,0,0,1,0,0]`` for the simulator's
``[0,0,0,1,1,1]``. The push is now gated by the counter.

C1: Catapult cannot schedule the self-FIFO lowering in any loop that puts
and gets conditionally: ``could not schedule partition '/top/fifo_0/run'
even with unlimited resources`` (SCHD-30), with and without pipelining,
with and without the reset drain. The build failed late in csynth; now
every SystemC mode but ``csim`` refuses at emission, naming the stream.
See ``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import os
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import int32, uint1, Stream

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

N = 6
WANT_FULL = [0, 0, 0, 1, 1, 1]


def _try_put_region():
    @df.region()
    def top(F: uint1[N]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[F])
        def k(f: uint1[N]):
            for i in range(N):
                ok: uint1 = s.try_put(i)
                f[i] = s.full() + ok - ok

    return top


def _refused_then_drained_region():
    """Four puts fill it, two more are refused, two gets drain two, two more
    puts succeed: the counter must follow the refusals."""

    @df.region()
    def top(F: uint1[10], G: int32[2]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[F, G])
        def k(f: uint1[10], g: int32[2]):
            for i in range(6):
                ok: uint1 = s.try_put(i)
                f[i] = s.full() + ok - ok
            for j in range(2):
                g[j] = s.get()
            for i in range(6, 8):
                ok2: uint1 = s.try_put(i)
                f[i] = ok2
            f[8] = s.full()
            f[9] = s.empty()

    return top


def test_try_put_is_gated_by_the_occupancy_counter():
    code = df.build(_try_put_region(), target="systemc").hls_code
    body = code.split("SC_MODULE(k_0)")[1].split("SC_MODULE(top)")[0]
    pushes = [l.strip() for l in body.splitlines() if "PushNB(" in l]
    assert len(pushes) == 1 and "_cnt < 4) &&" in pushes[0], pushes


def test_simulator_full_after_each_try_put():
    f = np.zeros(N, np.uint8)
    df.build(_try_put_region(), target="simulator")(f)
    assert f.tolist() == WANT_FULL


@needs_csim
def test_s9_csim_full_matches_simulator():
    f = np.zeros(N, np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_try_put_region(), target="systemc", mode="csim", project=tmp)(f)
    assert f.tolist() == WANT_FULL  # was [0, 0, 0, 1, 0, 0]


@needs_csim
def test_s9_csim_refused_puts_then_gets_then_puts():
    want_f = np.zeros(10, np.uint8)
    want_g = np.zeros(2, np.int32)
    df.build(_refused_then_drained_region(), target="simulator")(want_f, want_g)
    f = np.zeros(10, np.uint8)
    g = np.zeros(2, np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_refused_then_drained_region(), target="systemc", mode="csim", project=tmp)(f, g)
    assert want_f.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 0] and want_g.tolist() == [0, 1]
    assert g.tolist() == [0, 1]
    # the counter's guarantees: four accepted, two refused at full, and after
    # two gets the next try_put succeeds and the FIFO is not empty. The second
    # back-to-back try_put (f[7]) and full() right after it (f[8]) depend on
    # the enq handshake register, which only a clock edge drains, and two
    # non-blocking puts in one thread iteration have no clock between them:
    # that is the port's timing, not the counter's (csim gives 0 there).
    assert f.tolist()[:7] == [0, 0, 0, 1, 1, 1, 1] and f[9] == 0


@pytest.mark.parametrize("mode", ["csyn", "ppa"])
def test_c1_synthesis_modes_refuse_a_self_fifo(mode):
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="self-FIFO.*SCHD-30") as e:
            df.build(_try_put_region(), target="systemc", mode=mode, project=tmp)
    assert "kernel 'k_0'" in str(e.value)


def test_c1_csim_still_builds_a_self_fifo():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            df.build(_try_put_region(), target="systemc", mode="csim", project=tmp)
        except RuntimeError as e:  # no SystemC toolchain: the g++ step, not the refusal
            assert "self-FIFO" not in str(e)


if __name__ == "__main__":
    pytest.main([__file__])

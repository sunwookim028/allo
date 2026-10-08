# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SystemC csim testbench waits for the DUT before reading memories (A5).

With stream outputs, the last stream sink ``sc_stop()``\\ ped the simulation as
soon as it had drained its words. A kernel whose last iteration pushes its
stream words and *then* stores to a memory-port output lost those stores --
the first landed, the rest did not (MiniTPU's reduction tree read back one
wrong slot of 204,120). With memory outputs too, the last sink now waits for
the DUT's ``done`` and lets writes settle. See
``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import os
import tempfile

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import uint8, uint16

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

n = 12


def _region(store_last):
    @df.region()
    def top(X: uint8[n], A: uint8[n], B: uint16[n], D: uint16[n, 4]):
        @df.kernel(mapping=[1], args=[X, A, B, D])
        def k(x: uint8[n], a: uint8[n], b: uint16[n], d: uint16[n, 4]):
            for t in range(n):
                v: uint8 = x[t]
                with allo.meta_if(store_last):
                    a[t] = v
                    b[t] = t
                    for s in range(4):
                        d[t, s] = t * 4 + s + v
                with allo.meta_else():
                    for s in range(4):
                        d[t, s] = t * 4 + s + v
                    a[t] = v
                    b[t] = t

    return top


def _stream_only():
    @df.region()
    def top(X: uint8[n], A: uint8[n]):
        @df.kernel(mapping=[1], args=[X, A])
        def k(x: uint8[n], a: uint8[n]):
            for t in range(n):
                a[t] = x[t] + 1

    return top


def _want(x):
    t = np.arange(n)[:, None]
    return (t * 4 + np.arange(4)[None, :] + x[:, None]).astype(np.uint16)


def test_sink_waits_for_done_with_memory_outputs():
    code = df.customize(_region(True)).build(target="systemc").hls_code
    assert "while (!done_sig.read()) wait();" in code


def test_stream_only_tb_unchanged():
    code = df.customize(_stream_only()).build(target="systemc").hls_code
    assert "done_sig.read()" not in code
    assert "if (++_snk_done == 1) sc_stop();" in code


@needs_csim
@pytest.mark.parametrize(
    "store_last", [True, False], ids=["stores_last", "stores_first"]
)
def test_last_iteration_stores_kept(store_last):
    x = np.arange(n, dtype=np.uint8) * 3
    a, b = np.zeros(n, np.uint8), np.zeros(n, np.uint16)
    d = np.zeros((n, 4), np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_region(store_last), target="systemc", mode="csim", project=tmp)(
            x, a, b, d
        )
    np.testing.assert_array_equal(d, _want(x))
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, np.arange(n))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

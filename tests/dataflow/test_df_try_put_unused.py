# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``try_put`` whose result is not read still pushes (B7).

The simulator lowers ``try_put`` to an ``scf.if`` whose result is the
success flag and whose then-block pushes the word; ``junk: uint1 =
s.try_put(x)`` left the flag unused, and ``cleanUpUnusedOps`` in the
memref DCE (run by the composite-type lowering) erased every op with an
unused result by ``use_empty()`` alone -- the push went with it, silently:
``empty()`` stayed 1 after four of them. The sweep now asks MLIR whether the
op is trivially dead, nested effects included. See
``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import int32, uint1, Stream

N = 4


def _region(use_result):
    @df.region()
    def top(X: int32[N], E: uint1[N]):
        s: Stream[int32, N]

        @df.kernel(mapping=[1], args=[X, E])
        def k(x: int32[N], e: uint1[N]):
            for i in range(N):
                if use_result == 1:
                    ok: uint1 = s.try_put(x[i])
                    e[i] = s.empty() + ok - ok
                else:
                    junk: uint1 = s.try_put(x[i])
                    e[i] = s.empty()

    return top


@pytest.mark.parametrize("use_result", [1, 0])
def test_try_put_pushes_whether_or_not_its_flag_is_read(use_result):
    e = np.zeros(N, np.uint8)
    df.build(_region(use_result), target="simulator")(np.arange(N, dtype=np.int32), e)
    assert e.tolist() == [0, 0, 0, 0]  # unused flag gave [1, 1, 1, 1]


def test_unused_try_put_fills_the_fifo():
    """Five unread try_puts into a depth-4 stream: the fifth is refused, so
    the FIFO holds exactly four words."""

    @df.region()
    def top(Q: int32[6]):
        s: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[Q])
        def k(q: int32[6]):
            for i in range(5):
                junk: uint1 = s.try_put(i + 10)
            q[0] = s.full()
            for j in range(4):
                q[1 + j] = s.get()
            q[5] = s.empty()

    q = np.zeros(6, np.int32)
    df.build(top, target="simulator")(q)
    assert q.tolist() == [1, 10, 11, 12, 13, 1]


if __name__ == "__main__":
    pytest.main([__file__])

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the simulator's stream-lowering pass when a sub-region
is invoked from inside ``affine.for`` / ``scf.if`` (rather than directly
from the parent kernel body).

Before the deep-scan in ``_process_function_streams`` was added, the
top-level scan only walked ``func.body.blocks[0].operations`` for nested
``func.call`` ops. A sub-region call placed inside a Python ``for`` loop
in the parent kernel never triggered processing of the callee, so the
callee's own ``allo.stream_put`` / ``allo.stream_get`` ops survived into
LLVM lowering. ``convert-func-to-llvm`` then failed with::

    cannot be converted to LLVM IR: missing
    LLVMTranslationDialectInterface registration for dialect for op:
    func.func

This test exercises that exact shape and is expected to fail to even
construct a simulator module without the deep-scan in place.
"""

import numpy as np

import allo.dataflow as df
from allo.ir.types import Stream, int32


@df.region()
def _inner(slot: int32[1]):
    s: Stream[int32, 4]

    @df.kernel(mapping=[1])
    def producer():
        s.put(7)

    @df.kernel(mapping=[1], args=[slot])
    def consumer(o: int32[1]):
        v: int32 = s.get()
        o[0] = v


@df.region()
def _top(out: int32[1]):
    @df.kernel(mapping=[1], args=[out])
    def driver(o: int32[1]):
        for _ in range(4):
            _inner(o)


def test_subregion_with_stream_inside_for():
    mod = df.build(_top, target="simulator")
    out = np.zeros(1, dtype=np.int32)
    mod(out)
    assert out[0] == 7


if __name__ == "__main__":
    test_subregion_with_stream_inside_for()
    print("OK")

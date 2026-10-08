# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track E finding E-F1 (dev/records/minitpu/u4_track_e_2026-10-08.rst): an
untyped integer literal is ``int32``, so ``1 << 35`` inside an expression
assigned to a ``UInt(64)`` local is computed at 32 bits and contributes
nothing: ``x | (1 << 35)`` leaves bit 35 clear. No error and no warning, on
``customize(...).build("llvm")`` and on the dataflow simulator alike; the
same text compiles the same way in AMC's vendored Allo (shared lineage).
Found while hand-writing ``seq_decoder``'s field inserts as shift-and-or
for AMC (``1 << 56`` for ``d.valid``): every field above bit 31 vanished.
Workaround: a typed one (``ONE: UInt(64) = 1``; ``ONE << 35``). Fix: type
a literal shift by its context (the annotated target), or refuse a shift
amount >= the literal's width."""
import _worktree
from _worktree import verdict

ITEM = "new-wide-literal-shift"
import numpy as np  # noqa: E402

import allo  # noqa: E402
import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, uint32  # noqa: E402


def lit(A: uint32[2], O: uint32[2]):
    for i in range(2):
        x: UInt(64) = A[i]
        y: UInt(64) = x | (1 << 35)
        O[i] = y[32:64]


def typed(A: uint32[2], O: uint32[2]):
    for i in range(2):
        ONE: UInt(64) = 1
        x: UInt(64) = A[i]
        y: UInt(64) = x | (ONE << 35)
        O[i] = y[32:64]


@df.region()
def top(A: uint32[2], O: uint32[2]):
    @df.kernel(mapping=[1], args=[A, O])
    def k(a: uint32[2], o: uint32[2]):
        for i in range(2):
            x: UInt(64) = a[i]
            y: UInt(64) = x | (1 << 35)
            o[i] = y[32:64]


def main():
    a = np.array([0, 5], dtype=np.uint32)
    res = {}
    for name, fn in (("llvm literal", lit), ("llvm typed", typed)):
        o = np.zeros(2, dtype=np.uint32)
        allo.customize(fn).build(target="llvm")(a, o)
        res[name] = o.tolist()
    o = np.zeros(2, dtype=np.uint32)
    df.build(top, target="simulator")(a, o)
    res["simulator literal"] = o.tolist()
    print(res, "want [8, 8]")
    verdict(ITEM, res["llvm typed"] == [8, 8] and (res["llvm literal"] != [8, 8] or res["simulator literal"] != [8, 8]),
            str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)

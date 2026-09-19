# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NEW (not in ALLO_SHORTCOMINGS.md): a scalar `Stream.get()` result is never
implicitly cast to the destination type. `x: int8 = s.get()` or
`b[i] = s.get()` with an int32 stream and an int8 destination fails IR
verification (`'affine.store' op value to store must have the same type as
memref element type`), while the same narrowing from an int32 *array* element
works. Workaround: land it in a temporary of the stream's own type first."""
import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-stream-get-cast"
import allo.dataflow as df
from allo.ir.types import Stream, int8, int32


@df.region()
def via_stream(A: int32[4], B: int8[4]):
    s: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[A])
    def p(a: int32[4]):
        for i in range(4):
            s.put(a[i])

    @df.kernel(mapping=[1], args=[B])
    def c(b: int8[4]):
        for i in range(4):
            x: int8 = s.get()
            b[i] = x


@df.region()
def via_array(A: int32[4], B: int8[4]):
    @df.kernel(mapping=[1], args=[A, B])
    def c(a: int32[4], b: int8[4]):
        for i in range(4):
            x: int8 = a[i]
            b[i] = x


def run(top):
    try:
        mod = df.build(top, target="simulator")
        a = np.array([1, 2, 300, -5], dtype=np.int32)
        b = np.zeros(4, dtype=np.int8)
        mod(a, b)
        return "ok" if (b == a.astype(np.int8)).all() else f"WRONG {b}"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[1][:90] if len(str(e).splitlines()) > 1 else str(e)[:90]}"


def main():
    res = {"int32 array -> int8": run(via_array), "int32 stream -> int8": run(via_stream)}
    print(res)
    verdict(ITEM, res["int32 stream -> int8"] != "ok", str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)

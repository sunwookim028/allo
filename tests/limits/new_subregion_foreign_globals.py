# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item H (fork issue #4): a sub-region defined in another module is
type-checked against the CALLER's globals, not its own module's.

`builder.py` builds a called sub-region with
`ASTContext(global_vars=ctx.global_vars.copy(), ...)`, i.e. the calling
region's globals. A `Stream[...][N_SUB]` whose shape constant lives only in the
sub-region's module then fails with "stream array shape should be a compile
time constant" (or "Unsupported type `Stream`" when the caller does not import
`Stream` either). The sub-region builds fine standalone.

Variant (a): caller imports only the sub-region -> expected to fail.
Variant (b): caller also imports `N_SUB` and `Stream` (the workaround) -> ok.
"""
import os
import sys
import tempfile
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = "new-subregion-globals"

SUB_SRC = '''
import allo.dataflow as df
from allo.ir.types import Stream, int32

N_SUB = 2


@df.region()
def inner(A: int32[N_SUB], B: int32[N_SUB]):
    fifo: Stream[int32, 4][N_SUB]

    @df.kernel(mapping=[N_SUB], args=[A])
    def prod(a: int32[N_SUB]):
        i = df.get_pid()
        fifo[i].put(a[i])

    @df.kernel(mapping=[N_SUB], args=[B])
    def cons(b: int32[N_SUB]):
        i = df.get_pid()
        b[i] = fifo[i].get() + 1
'''

# (b) lives in its own module: a region must be defined at module scope (a
# nested definition trips the separate re-parse IndentationError, upstream #588).
CALLER_B_SRC = '''
import allo.dataflow as df
from allo.ir.types import Stream, int32  # noqa: F401  (the workaround)
from limits_sub_mod import inner, N_SUB  # noqa: F401  (the workaround)


@df.region()
def top_b(A: int32[2], B: int32[2]):
    @df.kernel(mapping=[1], args=[A, B])
    def drv(a: int32[2], b: int32[2]):
        inner(a, b)
'''

_tmp = tempfile.mkdtemp(prefix="allo_limits_sub_")
for _name, _src in (("limits_sub_mod", SUB_SRC), ("limits_caller_b", CALLER_B_SRC)):
    with open(os.path.join(_tmp, _name + ".py"), "w") as f:
        f.write(_src)
sys.path.insert(0, _tmp)

import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32  # noqa: E402
from limits_sub_mod import inner  # noqa: E402


@df.region()
def top_a(A: int32[2], B: int32[2]):
    @df.kernel(mapping=[1], args=[A, B])
    def drv(a: int32[2], b: int32[2]):
        inner(a, b)


def _run(region):
    try:
        mod = df.build(region, target="simulator")
        A = np.array([3, 5], dtype=np.int32)
        B = np.zeros(2, dtype=np.int32)
        mod(A, B)
        return "ok" if list(B) == [4, 6] else f"wrong {list(B)}"
    except (Exception, SystemExit) as e:  # noqa: BLE001  (customize() sys.exit(1)s)
        traceback.print_exc()
        return f"{type(e).__name__}: {str(e)[:120]}"


def main():
    from limits_caller_b import top_b

    res = {
        "(a) caller lacks N_SUB/Stream": _run(top_a),
        "(b) caller imports N_SUB, Stream": _run(top_b),
    }
    verdict(ITEM, res["(a) caller lacks N_SUB/Stream"] != "ok", str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)

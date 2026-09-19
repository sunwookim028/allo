# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 19: try_get/try_put on the TAPA target fail to emit, and the error
names an unrelated cause (wrap_io / multi-dimensional arrays) instead of the
unsupported op. Codegen only."""
import _worktree
from _worktree import verdict

ITEM = 19
import allo
import allo.dataflow as df
from allo.ir.types import Stream, int32


@df.region()
def top():
    S: Stream[int32, 2][1]
    O: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def k():
        d, ok = S[0].try_get()
        if ok:
            O[0].put(d)


def main():
    try:
        code = allo.customize(top).build(target="tapa").hls_code
        emitted = ".try_read(" in code
        verdict(ITEM, not emitted, f"emitted; try_read present={emitted}")
    except (Exception, SystemExit) as e:  # noqa: BLE001
        msg = str(e)
        names_op = "try_get" in msg.lower() or "trygetop" in msg.lower() or "stream_try" in msg.lower()
        print(f"  message: {msg}")
        verdict(ITEM, True, f"fails to emit; message names the op: {names_op}; "
                f"mentions wrap_io: {'wrap_io' in msg}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)

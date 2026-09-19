# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 9: "sim cache invalidation misses imported helpers".

Original claim: `.cache/llvm_sim/` is keyed on the level's `tpu.py` only, so
editing an imported helper leaves a stale compiled simulator.

Test: a kernel that calls a helper from a separate module. Build and run it in
a fresh process, rewrite the helper, build and run again in another fresh
process. If Allo cached anything keyed on the top file, the second result would
be stale."""
import os
import subprocess
import sys
import _worktree
from _worktree import verdict

ITEM = 9
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "_item09_scratch")

CHILD = r'''
import sys
sys.path.insert(0, {here!r}); sys.path.insert(0, {scratch!r})
import _worktree
import numpy as np
import allo.dataflow as df
from allo.ir.types import int32
from helper import bump

@df.region()
def top(out: int32[1]):
    @df.kernel(mapping=[1], args=[out])
    def k(o: int32[1]):
        o[0] = bump(o[0])

mod = df.build(top, target="simulator")
o = np.zeros(1, dtype=np.int32)
mod(o)
print("RESULT", int(o[0]))
'''


def write_helper(n):
    with open(os.path.join(SCRATCH, "helper.py"), "w") as f:
        f.write(
            "from allo.ir.types import int32\n\n"
            f"def bump(x: int32) -> int32:\n    return x + {n}\n"
        )


def run_child():
    child = os.path.join(SCRATCH, "child.py")  # a file: inspect.getsource needs one
    with open(child, "w") as f:
        f.write(CHILD.format(here=HERE, scratch=SCRATCH))
    p = subprocess.run(
        [sys.executable, child],
        capture_output=True, text=True, check=False,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return int(line.split()[1])
    raise RuntimeError(p.stderr[-800:])


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        write_helper(1)
        r1 = run_child()
        write_helper(100)
        r2 = run_child()
    finally:
        import shutil
        shutil.rmtree(SCRATCH, ignore_errors=True)
    stale = r2 != 100
    print(f"helper x+1 -> {r1}; helper edited to x+100 -> {r2}")
    if stale:
        verdict(ITEM, True, "second build used the stale helper")
    else:
        print(f"[item {ITEM}] NOT-A-LIMITATION: Allo keeps no simulator cache; "
              "the stale `.cache/llvm_sim/` was the allo-tpu harness's own")


if __name__ == "__main__":
    main()

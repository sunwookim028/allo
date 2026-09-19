# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 10: errors from lowering point at the lowered MLIR (`loc("-":N:M)`),
not the Python source; and "the MLIR Context cannot be re-instantiated in the
same process (`LLVM ERROR: Option 'fast' already exists!`)".

Probe for (a): an op the dataflow simulator cannot lower (`allo.exp`; see the
new-limitation list -- the simulator pipeline has no math lowering). The
diagnostic should name this file. Probe for (b): build and run three separate
regions in one process, each in a fresh MLIR Context."""
import os
import re
import subprocess
import sys
import _worktree
from _worktree import verdict

ITEM = 10
THIS = os.path.abspath(__file__)


def child_a():
    import allo
    import allo.dataflow as df
    from allo.ir.types import float32

    @df.region()
    def top(A: float32[4], B: float32[4]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: float32[4], b: float32[4]):
            for i in range(4):
                b[i] = allo.exp(a[i])  # PROBE-LINE

    df.build(top, target="simulator")


def child_b():
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import int32

    for n in (1, 2, 3):
        @df.region()
        def top(out: int32[1]):
            @df.kernel(mapping=[1], args=[out])
            def k(o: int32[1]):
                o[0] = o[0] + 1

        mod = df.build(top, target="simulator")
        o = np.zeros(1, dtype=np.int32)
        mod(o)
        assert o[0] == 1
    print("B-OK")


def main():
    if len(sys.argv) > 1:
        {"a": child_a, "b": child_b}[sys.argv[1]]()
        return
    pa = subprocess.run([sys.executable, THIS, "a"], capture_output=True, text=True, check=False)
    locs = re.findall(r'loc\(([^)]*)\): error', pa.stderr + pa.stdout)
    probe_line = next(i for i, l in enumerate(open(THIS), 1) if "# PROBE-LINE" in l)
    names_source = any(os.path.basename(THIS) in l for l in locs)
    print(f"(a) diagnostic locations: {locs}  (source line is {probe_line})")
    pb = subprocess.run([sys.executable, THIS, "b"], capture_output=True, text=True, check=False)
    b_ok = "B-OK" in pb.stdout
    print(f"(b) three builds in one process: {'ok' if b_ok else pb.stderr[-300:]}")
    verdict(ITEM, not names_source,
            f"(a) error names Python source: {names_source}; (b) rebuild in-process ok: {b_ok}")


if __name__ == "__main__":
    main()

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and measure TinyTPU-grid.

Three targets, in increasing cost:
  simulator -- the KPN dataflow simulator; checks the program runs and the
               answer is right, and is the only one that exercises the
               process structure directly
  llvm      -- the same thing through the CPU backend
  vhls      -- Vitis HLS; `csyn` gives per-region II and latency, `cosim` a
               real cycle count. This is where a number comparable to the
               chia branch's `cosim` comes from.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
import allo.backend.hls as hls  # noqa: E402
from examples.accelerator.tinytpu_grid.microarch import (  # noqa: E402
    tinytpu_grid, program, M, K, N, T, P0, P1, NINSTR,
)


def inputs(relu, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = np.zeros((M, N), np.float32)
    imem = np.array(program(relu), np.int32)
    gold = (A @ B).astype(np.float32)
    if relu:
        gold = np.maximum(gold, 0.0)
    return imem, A, B, C, gold


def check(C, gold, tag):
    ok = np.allclose(C, gold, rtol=1e-4, atol=1e-4)
    print(f"  {tag:26s} correct={ok}  maxerr={np.abs(C - gold).max():.2e}")
    return ok


def run(target, relu=False, mode="csim", project="grid.prj"):
    imem, A, B, C, gold = inputs(relu)
    tag = f"{'mm.relu' if relu else 'mm'} {M}x{K}x{N} [{target}]"
    if target == "simulator":
        mod = df.build(tinytpu_grid, target="simulator")
        mod(imem, A, B, C)
        return check(C, gold, tag)
    if target == "llvm":
        mod = df.build(tinytpu_grid)
        mod(imem, A, B, C)
        return check(C, gold, tag)
    # Vitis HLS
    mod = df.build(tinytpu_grid, target="vhls", mode=mode, project=project)
    if mode == "csim":
        mod(imem, A, B, C)
        return check(C, gold, tag)
    mod()   # csyn / cosim write their reports into the project directory
    print(f"  {tag:26s} {mode} done -> {project}")
    return True


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "simulator"
    print(f"TinyTPU-grid: {P0}x{P1} instances ({T}x{T} PEs + feed/drain), "
          f"{NINSTR} instructions")
    if what in ("simulator", "llvm", "csim"):
        t = "vhls" if what == "csim" else what
        ok = run(t, relu=False) and run(t, relu=True)
        sys.exit(0 if ok else 1)
    elif what in ("csyn", "cosim"):
        assert hls.is_available("vitis_hls"), "Vitis HLS not on PATH"
        run("vhls", relu=False, mode=what, project=f"grid_{what}.prj")

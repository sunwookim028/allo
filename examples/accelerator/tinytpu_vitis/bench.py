# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and measure TinyTPU-vitis.

    simulator   the KPN dataflow simulator: does the program run, is it right
    csim        the same through Vitis' C++ (compiles the emitted kernel.cpp)
    csyn        Vitis synthesis: per-process II and latency
    cosim       Vitis RTL co-simulation: a real cycle count, comparable to the
                chia branch's `cosim` numbers and to Gemmini's rdcycle
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch import (  # noqa: E402
    tinytpu_vitis, program, M, K, N, T, NI,
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


def run(target, relu=False, mode="csim", project="vitis.prj"):
    imem, A, B, C, gold = inputs(relu)
    tag = f"{'mm.relu' if relu else 'mm':7s} {M}x{K}x{N}"
    if target == "simulator":
        df.build(tinytpu_vitis, target="simulator")(imem, A, B, C)
    elif mode == "csim":
        df.build(tinytpu_vitis, target="vhls", mode="csim",
                 project=project)(imem, A, B, C)
    else:
        df.build(tinytpu_vitis, target="vhls", mode=mode, project=project)()
        print(f"  {tag} {mode} scaffolded -> {project}")
        return True
    ok = np.allclose(C, gold, rtol=1e-4, atol=1e-4)
    print(f"  {tag} [{target}] correct={ok}  maxerr={np.abs(C - gold).max():.2e}")
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "simulator"
    print(f"TinyTPU-vitis: {T}x{T} PEs + sequencer/loader/drainer, "
          f"{NI} instructions, {M}x{K}x{N}")
    if what == "simulator":
        ok = run("simulator", False) and run("simulator", True)
        sys.exit(0 if ok else 1)
    if what == "csim":
        ok = run("vhls", False, "csim", os.path.abspath("vitis_csim.prj"))
        sys.exit(0 if ok else 1)
    run("vhls", False, what, os.path.abspath(f"vitis_{what}.prj"))

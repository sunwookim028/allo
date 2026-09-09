# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and measure TinyTPU-ws (the weight-stationary array).

    simulator   the KPN dataflow simulator: does the program run, is it right
    csyn        Vitis synthesis: per-process II and latency
    cosim       Vitis RTL co-simulation: a real cycle count
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from allo.dataflow import customize  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_ws import (  # noqa: E402
    tinytpu_ws, program, schedule, M, K, N, T, NI, Kt, Nt, DTYPE,
    CLIP_LO, CLIP_HI,
)

# Same operand distribution as the Gemmini benchmark this is compared against:
# `allo_cmp.c` fills with `(nextrand() % 9) - 4`, i.e. integers in [-4, 4].
NP_IN = np.int8 if DTYPE == "int8" else np.float32
NP_OUT = np.int8 if DTYPE == "int8" else np.float32


def inputs(relu, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (M, K)).astype(NP_IN)
    B = rng.integers(-4, 5, (K, N)).astype(NP_IN)
    C = np.zeros((M, N), NP_OUT)
    imem = np.array(program(relu), np.int32)
    gold = A.astype(np.int64) @ B.astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    if DTYPE == "int8":
        # Gemmini passes ACC_SCALE_IDENTITY with shift 0 and mvouts elem_t,
        # so the accumulator is clipped, not scaled.
        gold = np.clip(gold, CLIP_LO, CLIP_HI)
    return imem, A, B, C, gold.astype(NP_OUT)


def run_sim(relu=False):
    imem, A, B, C, gold = inputs(relu)
    df.build(tinytpu_ws, target="simulator")(imem, A, B, C)
    err = np.abs(C.astype(np.int64) - gold.astype(np.int64)).max()
    ok = err == 0 if DTYPE == "int8" else err < 1e-3
    print(f"  {'mm.relu' if relu else 'mm':7s} {M}x{K}x{N}  NI={NI}  "
          f"{DTYPE}  correct={ok}  maxerr={err}")
    return ok


def run_hls(mode, sched=True, project=None):
    project = project or os.path.abspath(
        f"ws_{mode}_{DTYPE}_{M}x{K}x{N}.prj")
    s = customize(tinytpu_ws)
    if sched:
        schedule(s)
    mod = s.build(target="vitis_hls", mode=mode, project=project)
    print(f"  scaffolded {mode} (partitioned={sched}) -> {project}")
    return mod


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "simulator"
    print(f"TinyTPU-ws: {T}x{T} weight-stationary PEs, {NI} instructions "
          f"(Nt={Nt} x Kt={Kt}), {M}x{K}x{N}, dtype={DTYPE}")
    if what == "simulator":
        ok = run_sim(False) and run_sim(True)
        sys.exit(0 if ok else 1)
    sched = "--nosched" not in sys.argv
    run_hls(what, sched)

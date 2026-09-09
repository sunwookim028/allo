# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify and measure TinyTPU-isa: the instruction-programmable tiled GEMM."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from allo.dataflow import customize  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, gemm_program, schedule, M, K, N, T, NPROG, Kt, Nt,
)

IMEM_WORDS = 1024


def inputs(relu, seed=0):
    """Operands in [-4, 4], the range `allo_cmp.c` fills for Gemmini."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (M, K)).astype(np.int8)
    B = rng.integers(-4, 5, (K, N)).astype(np.int8)
    C = np.zeros((M, N), np.int8)
    prog = gemm_program(relu)
    imem = np.zeros(IMEM_WORDS, np.uint64)          # 0 == OP_NOP
    imem[: len(prog)] = np.array(prog, np.uint64)
    gold = A.astype(np.int64) @ B.astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    gold = np.clip(gold, -128, 127)                  # Gemmini mvout clips
    return imem, A, B, C, gold.astype(np.int8), len(prog)


def run_sim(relu=False):
    imem, A, B, C, gold, n = inputs(relu)
    df.build(tinytpu_isa, target="simulator")(imem, A, B, C)
    bad = int((C.astype(np.int64) != gold.astype(np.int64)).sum())
    print(f"  {'gemm.relu' if relu else 'gemm':9s} {M}x{K}x{N}  "
          f"{n}/{NPROG} instrs  wrong={bad}/{M*N}")
    if bad:
        print("   got\n", C, "\n   want\n", gold)
    return bad == 0


def run_hls(mode):
    project = os.path.abspath(f"isa_{mode}_{M}x{K}x{N}.prj")
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode=mode, project=project)
    print(f"  scaffolded {mode} -> {project}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "simulator"
    print(f"TinyTPU-isa: {T}x{T} WS array + vector unit + SIMD scratchpad, "
          f"{M}x{K}x{N} (Kt={Kt}, Nt={Nt}), {NPROG} instruction slots")
    if what == "simulator":
        ok = run_sim(False) and run_sim(True)
        sys.exit(0 if ok else 1)
    run_hls(what)

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify TinyTPU-isa across workload sizes on ONE fixed hardware build.

The point of this driver is that `tinytpu_isa` is built once. Shapes are swept
as *data*: `gemm_program(M, K, N)` assembles a different instruction stream and
the same RTL runs it. Earlier revisions rebuilt the accelerator per shape, which
made a comparison against Gemmini's single elaboration meaningless.

    python bench_isa.py                # sweep every shape up to MAXDIM
    python bench_isa.py 8 8 8          # one shape
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, gemm_program, vadd_program, assemble, MAXDIM, T, IMEM_SIZE,
)

SHAPES = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]


def buffers(seed=0):
    """Host buffers at the fixed MAXDIM stride, as `allo_cmp.c` uses for
    Gemmini. Operands in [-4, 4], the same distribution it fills."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    B = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    return A, B


def imem_of(prog):
    w = assemble(prog)
    m = np.zeros(IMEM_SIZE, np.uint64)
    m[: len(w)] = np.array(w, np.uint64)
    return m, len(w)


def check(mod, M, K, N, relu):
    A, B = buffers()
    C = np.zeros(MAXDIM * MAXDIM, np.int8)
    prog = gemm_program(M, K, N, relu)
    imem, _ = imem_of(prog)
    mod(imem, A.reshape(-1), B.reshape(-1), C)
    C = C.reshape(MAXDIM, MAXDIM)
    gold = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    gold = np.clip(gold, -128, 127).astype(np.int8)
    bad = int((C[:M, :N] != gold).sum())
    print(f"  {'gemm.relu' if relu else 'gemm':9s} {M:2d}x{K:2d}x{N:2d}  "
          f"{len(prog):3d} instrs  wrong={bad}/{M*N}")
    return bad == 0


def check_vadd(mod, M, K, N):
    A, B = buffers()
    C = np.zeros(MAXDIM * MAXDIM, np.int8)
    prog = vadd_program(M, K, N)
    imem, _ = imem_of(prog)
    mod(imem, A.reshape(-1), B.reshape(-1), C)
    C = C.reshape(MAXDIM, MAXDIM)
    gold = 2 * (A[:M, :T].astype(np.int64) @ B[:T, :T].astype(np.int64))
    gold = np.clip(np.maximum(gold, 0), -128, 127).astype(np.int8)
    bad = int((C[:M, :T] != gold).sum())
    print(f"  {'vadd+relu':9s} {M:2d}x{K:2d}x{N:2d}  {len(prog):3d} instrs  "
          f"wrong={bad}/{M*T}")
    return bad == 0


if __name__ == "__main__":
    shapes = SHAPES
    if len(sys.argv) == 4:
        shapes = [tuple(int(a) for a in sys.argv[1:4])]
    print(f"TinyTPU-isa: ONE build -- {T}x{T} array, MAXDIM={MAXDIM}, "
          f"imem {IMEM_SIZE}; sweeping {len(shapes)} shape(s)")
    mod = df.build(tinytpu_isa, target="simulator")     # built once
    ok = True
    for (M, K, N) in shapes:
        ok &= check(mod, M, K, N, False)
        ok &= check(mod, M, K, N, True)
    ok &= check_vadd(mod, *shapes[-1])
    print("  ALL EXACT" if ok else "  FAILURES")
    sys.exit(0 if ok else 1)

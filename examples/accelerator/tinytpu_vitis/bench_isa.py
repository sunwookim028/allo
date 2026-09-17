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
    tinytpu_isa, gemm_program, gemm_program_flat, vadd_program, assemble,
    expand, MAXDIM, T, IMEM_SIZE, NHDR, IWORDS,
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


def check(mod, M, K, N, relu, looped=True):
    A, B = buffers()
    C = np.zeros(MAXDIM * MAXDIM, np.int8)
    prog = (gemm_program if looped else gemm_program_flat)(M, K, N, relu)
    # imem is sized to the LOOPED program, which is the one the machine ships.
    # The unrolled reference outgrows it at the larger shapes -- that is the
    # point of having control flow, not a problem with the test -- so it runs
    # where it fits and the static `expand` equivalence covers the rest.
    if NHDR + IWORDS * len(prog) > IMEM_SIZE:
        print(f"  {'gemm.relu' if relu else 'gemm':9s} {M:2d}x{K:2d}x{N:2d} flat  "
              f"{len(prog):3d} static  SKIP (needs "
              f"{NHDR + IWORDS * len(prog)} words > imem {IMEM_SIZE})")
        return True
    imem, _ = imem_of(prog)
    mod(imem, A.reshape(-1), B.reshape(-1), C)
    C = C.reshape(MAXDIM, MAXDIM)
    gold = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    gold = np.clip(gold, -128, 127).astype(np.int8)
    bad = int((C[:M, :N] != gold).sum())
    tag = "loop" if looped else "flat"
    print(f"  {'gemm.relu' if relu else 'gemm':9s} {M:2d}x{K:2d}x{N:2d} {tag}  "
          f"{len(prog):3d} static -> {len(expand(prog)):3d} dynamic  "
          f"wrong={bad}/{M*N}")
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
    # The strong equivalence check, and it covers every shape: the looped and
    # unrolled programs must issue the identical dynamic opcode stream.
    for (M, K, N) in shapes:
        for r in (False, True):
            a = expand(gemm_program(M, K, N, r))
            b = expand(gemm_program_flat(M, K, N, r))
            assert a == b, f"loop/flat dynamic streams differ at {M}x{K}x{N}"
    print(f"  loop == flat dynamic opcode stream at all {len(shapes)} shapes")
    mod = df.build(tinytpu_isa, target="simulator")     # built once
    ok = True
    # The flat program is the reference: same shape, same result, no control
    # flow. Running both is the cheap way to test a PC and an AGU -- any
    # disagreement is the loop or the address resolution, nothing else.
    for (M, K, N) in shapes:
        ok &= check(mod, M, K, N, False, looped=False)
        ok &= check(mod, M, K, N, False, looped=True)
        ok &= check(mod, M, K, N, True, looped=False)
        ok &= check(mod, M, K, N, True, looped=True)
    ok &= check_vadd(mod, *shapes[-1])
    print("  ALL EXACT" if ok else "  FAILURES")
    sys.exit(0 if ok else 1)

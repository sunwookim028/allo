# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify TinyTPU-isa across workload sizes on ONE fixed hardware build.

The point of this driver is that `tinytpu_isa` is built once. Shapes are swept
as *data*: `gemm_program(M, K, N)` generates a different instruction stream and
the same RTL runs it. Earlier revisions rebuilt the accelerator per shape, which
made a comparison against Gemmini's single elaboration meaningless.

    python bench_isa.py                # sweep every shape up to MAXDIM
    python bench_isa.py 8 8 8          # one shape

This is the PUBLISHED setup -- [-4, 4] operands, seed 0, `C` zeroed, only the
result region compared -- and it is deliberately weak: an int16 accumulator,
an off-by-one clip, and a unit ignoring a field GEMM never varies all print
ALL EXACT here. `stress_isa.py` is the correctness gate; `mutate.py` shows
which of the two catches what.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import allo.dataflow as df  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, gemm_program_flat, vadd_program, assemble,
    expand, MAXDIM, T, IMEM_SIZE, NHDR, IWORDS,
)
# The shipped GEMM program is GENERATED: the loop levels come from the nesting
# of `with k.loop(...)`, not from integers typed into `enc_agu`. The
# hand-emitted form it replaced stays in `microarch_isa` as the reference the
# assertion below holds it against, word for word.
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    gemm_program, assert_matches_handwritten,
)

# Re-exported: `act_compile`, `kpn_model`, `isa_dsl`, `stress_isa` and
# `tests/act/test_tinytpu.py` all import `bench_isa.SHAPES` and all mean the
# canonical five. The definition itself is in `shapes.py`, which imports
# nothing, so the CHIA harness (a conda env without `allo`) can read it too.
from examples.accelerator.tinytpu_vitis.shapes import SHAPES  # noqa: E402

# ---- THE TWO BENCHMARK SETS ----
# They answer different questions and are never mixed into one verdict; the
# accounting is in `docs/source/designs/benchmarks.rst`.
#
#   LATENCY   the five shapes the published 175/265/421/482/674 come from --
#             `SHAPES` itself, aliased so the two sets read symmetrically.
#             At 16x16x16 this 4x4 array does 16 tile-matmuls and 256
#             wavefront rows while a 16x16 array does ONE weight load and one
#             pass, so the same shape is not the same work and the number is
#             dominated by the fixed pipeline term. It measures
#             time-to-first-result, not throughput.
#   STEADY    shapes big enough to amortise that fixed term, so MACs/cycle
#             approaches the array's T*T peak and the number characterises
#             the MACHINE. Cubic 16..64 plus two non-cubic shapes, which
#             separate the M (wavefront rows) term from the N*K (tile count)
#             term.
LATENCY = SHAPES
STEADY = [(16, 16, 16), (32, 32, 32), (48, 48, 48), (64, 64, 64),
          (64, 32, 64), (32, 64, 32)]


def runnable(shapes):
    """The subset this build can express: a multiple of T, within MAXDIM."""
    return [s for s in shapes
            if all(d % T == 0 and d <= MAXDIM for d in s)]


# The set THIS RUN sweeps, and it is deliberately NOT called `SHAPES`.
# `SHAPES` is re-exported and read POSITIONALLY elsewhere -- `accept.BASELINES`
# is indexed against `shapes.NAMES`, and `act_compile`/`kpn_model`/`isa_dsl`/
# `tests/act/test_tinytpu.py` all want the five -- so a knob that changed its
# value would silently change what those modules measure. An earlier revision
# of this file did exactly that.
#
# `runnable` is applied to BOTH sets: T is a working parameter (T=8 verifies
# bit-exact), and at T=8 three of the five latency shapes are not multiples of
# T. The published five survive it unchanged at T=4.
SWEEP = runnable(LATENCY)
if os.environ.get("TPU_SET") == "steady":
    SWEEP = runnable(STEADY)
elif os.environ.get("TPU_SET") == "all":
    SWEEP = runnable(LATENCY) + [s for s in runnable(STEADY)
                                 if s not in LATENCY]
if os.environ.get("TPU_SHAPES"):        # e.g. TPU_SHAPES=4x4x4,64x64x64
    SWEEP = [tuple(int(x) for x in t.split("x"))
             for t in os.environ["TPU_SHAPES"].split(",")]


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
    shapes = SWEEP
    if len(sys.argv) == 4:
        shapes = [tuple(int(a) for a in sys.argv[1:4])]
    print(f"TinyTPU-isa: ONE build -- {T}x{T} array, MAXDIM={MAXDIM}, "
          f"imem {IMEM_SIZE}; sweeping {len(shapes)} shape(s)")
    # The generator's own check, and it is the strict one: the generated
    # program must be BIT-IDENTICAL to the hand-emitted reference, both words
    # of every instruction, at every shape and both relu settings. Nothing
    # about cycles can change if the words do not.
    assert_matches_handwritten(shapes)
    print(f"  generated == hand-written word-for-word at all {len(shapes)} "
          f"shapes x {{gemm, gemm.relu}}")
    # The strong equivalence check, and it covers every shape: the looped and
    # unrolled programs must issue the identical dynamic opcode stream.
    for (M, K, N) in shapes:
        for r in (False, True):
            # the SHIPPED order is the one the flat form unrolls; another
            # `TPU_PROGRAM` order is checked below against numpy instead
            a = expand(gemm_program(M, K, N, r, order="shipped"))
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

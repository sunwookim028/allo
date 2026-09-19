# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A semantic gate stricter than `bench_isa.py`. FROZEN: the agent cannot edit it.

`bench_isa.py` and `cosim.py` feed one operand distribution: seed 0, values in
[-4, 4]. At K <= 16 every dot product then fits in 9 bits, so a design that
narrowed its int32 accumulator to int16 -- or dropped any other property those
inputs happen not to exercise -- would still print ALL EXACT and pass every
cosim testbench. A search loop optimising cycles will find such a change faster
than a person would. This gate closes the cheap ones:

* full-range int8 operands ([-128, 127]), so a narrowed accumulator wraps and
  the clipped result goes wrong;
* several seeds and a mid-range distribution, so saturation does not hide it;
* shapes outside the five scored ones, so a design special-cased to exactly
  those programs fails;
* `C` prefilled with a sentinel, with the sentinel required to survive outside
  the `M x N` result, so a design may not rely on `C` arriving zeroed and may
  not scribble outside the result.

It runs on Allo's simulator (functional, not RTL), in the evaluation tree, so it
imports the candidate's `microarch_isa` / `isa_dsl`. What it cannot see --
state carried across two invocations of the RTL, e.g. an initialisation removed
because one call per testbench never needs it -- stays a manual review item.
"""

import sys

import numpy as np

import allo.dataflow as df
from examples.accelerator.tinytpu_vitis.microarch_isa import (
    tinytpu_isa, assemble, MAXDIM, IMEM_SIZE, NHDR, IWORDS,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program

SCORED = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]
EXTRA = [(4, 16, 4), (16, 4, 16), (8, 12, 4), (12, 8, 16), (4, 8, 12)]
# (seed, lo, hi): full range twice, then a range whose sums straddle int8.
DISTS = [(11, -128, 127), (12, -128, 127), (13, -16, 16)]
SENTINEL = 0x5A


def run(mod, M, K, N, relu, seed, lo, hi):
    rng = np.random.default_rng(seed)
    A = rng.integers(lo, hi + 1, (MAXDIM, MAXDIM)).astype(np.int8)
    B = rng.integers(lo, hi + 1, (MAXDIM, MAXDIM)).astype(np.int8)
    prog = gemm_program(M, K, N, relu)
    words = assemble(prog)
    if NHDR + IWORDS * len(prog) > IMEM_SIZE or len(words) > IMEM_SIZE:
        return f"program for {M}x{K}x{N} does not fit imem ({len(words)} > {IMEM_SIZE})"
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    C = np.full(MAXDIM * MAXDIM, SENTINEL, np.int8)
    mod(imem, A.reshape(-1), B.reshape(-1), C)
    C = C.reshape(MAXDIM, MAXDIM)
    gold = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    gold = np.clip(gold, -128, 127).astype(np.int8)
    wrong = int((C[:M, :N] != gold).sum())
    outside = C.copy()
    outside[:M, :N] = SENTINEL
    clobbered = int((outside != SENTINEL).sum())
    if wrong or clobbered:
        return (f"{M}x{K}x{N} relu={relu} seed={seed} range=[{lo},{hi}]: "
                f"wrong={wrong}/{M * N} clobbered_outside={clobbered}")
    return None


def main():
    mod = df.build(tinytpu_isa, target="simulator")
    failures, n = [], 0
    # Largest first, so a later small program runs after a large one in the
    # same built module.
    for shape in sorted(SCORED + EXTRA, key=lambda s: -s[0] * s[1] * s[2]):
        for relu in (False, True):
            for seed, lo, hi in DISTS:
                n += 1
                bad = run(mod, *shape, relu, seed, lo, hi)
                if bad:
                    failures.append(bad)
    for f in failures:
        print("  STRESS FAIL", f)
    print(f"  STRESS {'OK' if not failures else 'FAILED'}: {n - len(failures)}/{n} "
          f"runs exact (full-range operands, sentinel-filled C, "
          f"{len(SCORED + EXTRA)} shapes)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

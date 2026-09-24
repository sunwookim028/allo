# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the MiniTPU model, run it, and check it against MiniTPU's arithmetic.

    python examples/minitpu/run.py            # three shapes, about 6 minutes
    python examples/minitpu/run.py --quick    # one shape, about 2 minutes
    python examples/minitpu/run.py --shapes 16x64x16

Each shape is run twice: once for the numbers, once to show that the
assembler refuses the schedules the hardware does not enforce.
"""

import argparse
import sys
import time

import numpy as np
import ml_dtypes

if __package__ in (None, ""):  # allow `python examples/minitpu/run.py`
    sys.path.insert(0, __file__.rsplit("/examples/", 1)[0])
    __package__ = "examples.minitpu"

from . import microarch as U
from . import program as P
from . import reference as R

bf16 = ml_dtypes.bfloat16

# 16x16x16 is the ARRAY's own tile -- one vmatload, one vmatpush per VREG,
# one vmatpop -- not a product MiniTPU's compiler would ever build: its
# blocks round up to multiples of 16 on M and N and its reduction pads to a
# 64-deep quantum, so the smallest product `minitpu-cc` emits is
# [32,64]@[64,16]. Both are run: the ISA-level tile and their floor shape.
SHAPES = ["16x16x16", "16x64x16", "16x32x32", "32x64x16"]
QUICK = ["16x16x16"]


def run_shape(spec, seed=7):
    m, k, n = (int(x) for x in spec.split("x"))
    layout = P.Layout(m, k, n)
    prog = P.assemble(layout)
    P.check(prog)

    t0 = time.time()
    mod, prog_np = U.build(prog, m, vmem_words=layout.n_words)
    t_build = time.time() - t0

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(np.float32).astype(bf16)
    b = rng.standard_normal((k, n)).astype(np.float32).astype(bf16)
    dram_in = layout.pack(a, b)
    dram_out = np.zeros_like(dram_in)

    t0 = time.time()
    mod(prog_np, dram_in, dram_out)
    t_run = time.time() - t0

    got = layout.unpack(dram_out)
    want = R.gemm(a, b)
    exact = np.array_equal(np.asarray(got), np.asarray(want))
    err = R.rel_error(got, R.fp64_gemm(a, b))
    bound = np.sqrt(k / R.DIM) * 2.0**-8

    print(
        f"  {spec:>10}  {len(prog):>4} cmds  build {t_build:5.1f}s  run {t_run:5.1f}s  "
        f"{'EXACT' if exact else 'MISMATCH':>8} vs model  "
        f"rel err {100 * err:6.3f}%  (bf16 tile bound {100 * bound:.3f}%)"
    )
    return exact


def show_refusals():
    """The three rules the hardware does not enforce, each one refused."""
    layout = P.Layout(16, 16, 16)
    w, a, c = layout.v_w, layout.v_a, layout.v_c
    cases = {
        "vst inside a matrix burst (VREG port C)": [
            (U.OP_MATLOAD, w, 0, 0),
            (U.OP_VST, c, 0, 0),
        ],
        "output FIFO overflow (64 result rows/lane)": [(U.OP_MATLOAD, w, 0, 0)]
        + [(U.OP_MATPUSH, a, 0, 0)] * 17,
        "vmatload refills an undrained weight bank": [
            (U.OP_MATLOAD, w, 0, 0),
            (U.OP_MATPUSH, a, 0, 0),
            (U.OP_MATLOAD, w, 0, 0),
            (U.OP_MATPUSH, a, 0, 0),
            (U.OP_MATLOAD, w, 0, 0),
        ],
    }
    ok = True
    for name, prog in cases.items():
        try:
            P.check(prog)
        except P.ProgramError as exc:
            print(f"  refused: {name}\n           {str(exc).split(':')[0]}")
            continue
        print(f"  NOT REFUSED: {name}")
        ok = False
    # ...and the legal schedule is accepted.
    P.check(P.assemble(layout))
    print("  accepted: the assembled 16x16x16 schedule")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shapes", default=None)
    args = ap.parse_args()
    shapes = (
        args.shapes.split(",") if args.shapes else (QUICK if args.quick else SHAPES)
    )

    print("MiniTPU model -- 16x16 BF16 weight-stationary array, 24-bit accumulator")
    print("Functional check: the model against MiniTPU's own arithmetic\n")
    ok = all([run_shape(s) for s in shapes])
    print("\nSchedule check: the assembler carries the correctness argument\n")
    ok = show_refusals() and ok
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

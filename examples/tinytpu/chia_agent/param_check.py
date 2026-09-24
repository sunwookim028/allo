# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parametricity gate: the candidate, rebuilt at a configuration it is NOT
scored at, must still be a correct GEMM machine. FROZEN.

Everything else in the gate runs at the scored configuration, T=4 and
MAXDIM=16, so a design that hard-codes those numbers -- or specialises a unit to
them -- passes every check and scores well. The design is meant to be
parametric (`microarch_isa.py` reads `TPU_T` / `TPU_MAXDIM`), and a win that
only exists at the scored point is overfitting to the evaluator, not a better
machine. The first paid CHIA run produced exactly that shape of candidate: its
diff replaced `T = int(os.environ.get("TPU_T", 4))` with `T = 4`.

This check is run with `TPU_MAXDIM` (and optionally `TPU_T`) set by the
evaluator to a configuration the shipped design supports, and requires:

1. the built module to REPORT that configuration (a literal `MAXDIM = 16`
   builds at 16 and is refused here, not silently checked at 16);
2. every multiple-of-T GEMM shape up to that MAXDIM, both ReLU settings, to be
   bit-exact on the simulator, with full-range, corner and boundary operands
   and a randomly prefilled `C` compared in full (the operand generators and
   golden model are `stress_isa.py`'s, imported unchanged);
3. random valid programs (`stress_isa.random_program`) to agree with
   `isa_ref` (the ISA as numpy), as in `stress_isa.py`.

Main's `stress_isa.py` itself is not run here: its scored-shape list,
`vector_program(8)` and validator controls are written for MAXDIM >= 16, and
`random_program` / `vector_program` emit some programs that address column
block 2 regardless of MAXDIM (measured on the unmodified design at MAXDIM=8:
3 of 24 seeds, and `vector_program(4)`, are refused by `check_program`). A
seed whose generation raises is therefore SKIPPED, not failed, and at least
MIN_FUZZ must generate; vector programs are left to `stress_isa.py`.

    TPU_MAXDIM=8 python param_check.py      # via gate_runner.py in the gate
Returns 0 and prints `PARAM OK: n/n ...`, or lists failures and returns 1.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from examples.tinytpu import microarch_isa as U  # noqa: E402
from examples.tinytpu import isa_ref  # noqa: E402
from examples.tinytpu.isa_dsl import gemm_program  # noqa: E402
from examples.tinytpu import stress_isa as S  # noqa: E402

N_FUZZ = 24
MIN_FUZZ = 16


def main(argv=()):
    want_md = int(os.environ["TPU_MAXDIM"])
    want_t = int(os.environ.get("TPU_T", U.T))
    fails = []
    if U.MAXDIM != want_md or U.T != want_t:
        print(f"  PARAM FAIL built T={U.T} MAXDIM={U.MAXDIM}, asked for "
              f"T={want_t} MAXDIM={want_md}: the design does not honour its "
              f"parameters (hard-coded?)", flush=True)
        return 1
    mod = df.build(U.tinytpu_isa, target="simulator")
    crng = np.random.default_rng(97)
    n, seed = 0, 500
    shapes = [s for s in S.ALL_SHAPES]
    for (M, K, N) in shapes:
        for relu in (False, True):
            for dist in ("full", "corner", "boundary"):
                seed += 1
                n += 1
                A, B = (S.boundary_operands(M, K, N, seed) if dist == "boundary"
                        else S.operands(dist, seed))
                C0 = crng.integers(-128, 128, U.MAXDIM * U.MAXDIM).astype(np.int8)
                tag = f"gemm{'.relu' if relu else ''} {M}x{K}x{N} {dist}"
                try:
                    prog = gemm_program(M, K, N, relu)
                    gold = S.gemm_gold(A, B, M, K, N, relu, C0)
                    bad = S.compare(tag, S.execute(mod, prog, A, B, C0), gold, M, N)
                except Exception as e:  # noqa: BLE001 -- a build/program error fails
                    bad = f"{tag}: {type(e).__name__}: {e}"
                if bad:
                    fails.append(bad)
    progs, skipped = [], 0
    for s in range(N_FUZZ):
        try:
            progs.append((f"fuzz seed={9000 + s}", S.random_program(9000 + s)))
        except Exception:  # noqa: BLE001 -- main's generator, not the design
            skipped += 1
    if len(progs) < MIN_FUZZ:
        fails.append(f"only {len(progs)} of {N_FUZZ} random programs could be "
                     f"generated (need {MIN_FUZZ})")
    for tag, prog in progs:
        n += 1
        A, B = S.operands("full", seed)
        seed += 1
        C0 = crng.integers(-128, 128, U.MAXDIM * U.MAXDIM).astype(np.int8)
        try:
            bad = S.compare(tag, S.execute(mod, prog, A, B, C0),
                            isa_ref.run(prog, A, B, C0), None, None)
        except Exception as e:  # noqa: BLE001
            bad = f"{tag}: {type(e).__name__}: {e}"
        if bad:
            fails.append(bad)
    for f in fails:
        print("  PARAM FAIL", f, flush=True)
    print(f"  PARAM {'OK' if not fails else 'FAILED'}: {n - len(fails)}/{n} runs "
          f"exact at T={U.T} MAXDIM={U.MAXDIM} ({len(shapes)} GEMM shapes x relu x "
          f"full/corner/boundary, {len(progs)} random programs; {skipped} seeds "
          f"skipped as ungeneratable at this MAXDIM)", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

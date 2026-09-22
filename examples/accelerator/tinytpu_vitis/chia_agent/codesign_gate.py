# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mapspace gate: enumerate, count the refusals, and prove the survivors.

FROZEN, and run under `gate_runner.py` so its verdict is vouched for rather
than read from its own stdout. Four things, in order, at every requested shape:

1. **The seam.** `gemm_from_nest(canonical(...))` must re-emit
   `isa_dsl.gemm_program` word for word, at every shape and both relu settings.
   That is what ties the mapper's encoder to the program `bench_isa.py` and
   `stress_isa.py` actually verify: `bench_isa` requires `gemm_program` to equal
   `microarch_isa.gemm_program_handwritten` bit for bit, so this identity makes
   the mapper's emitter the same generator, exercised through a nest. A
   candidate whose encoder has drifted from its own shipped program is refused
   here, before any cycle is measured.

2. **The exhaustive enumeration**, and the refusal histogram. Printed as
   `MAPSPACE <shape>: ...` lines, which `evaluate.py` parses into the verdict.
   This is the co-design signal: the count of encodable nests, and which
   constraint refused the rest.

3. **Every survivor is correct**, against `isa_ref.run` -- the frozen numpy
   model of what each instruction means -- at full-range int8 with `C`
   prefilled, compared over the whole of `C` so nothing outside the M x N
   result may be touched. Up to `CHECK_MAX` of them per shape and relu setting,
   best-ranked first, and always the chosen one. A nest the mapper would offer
   as "the best this hardware can run" that computes the wrong thing is a
   failure of the candidate, not of the mapper.

4. **The chosen program fits.** `microarch_isa.assemble` asserts
   `len(words) <= IMEM_SIZE`, so this is already true of anything counted
   encodable; it is restated as an explicit line because "the program our imem
   cannot hold" is the named failure mode of scoring on a cost model that does
   not know about instruction memory (docs/source/extensions/act.rst).

    python gate_runner.py codesign 4x4x4,16x16x16
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import mapspace  # noqa: E402
from examples.accelerator.tinytpu_vitis import isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    gemm_from_nest, gemm_program,
)
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    IMEM_SIZE, MAXDIM, T, assemble,
)

#: How many of a shape's encodable nests are proved against `isa_ref`. The
#: shipped design has 3; a candidate that unlocks hundreds is checked on the
#: best `CHECK_MAX` of them, which always includes the one that gets cosimmed.
CHECK_MAX = 48
DEFAULT_SHAPES = "4x4x4,16x16x16"


def shapes_from(argv):
    text = argv[1] if len(argv) > 1 and argv[1] else DEFAULT_SHAPES
    out = []
    for tok in text.split(","):
        if tok.strip():
            out.append(tuple(int(x) for x in tok.strip().split("x")))
    return out


def seam(shapes):
    """The nest that describes the shipped tiling re-emits the shipped program."""
    for (M, K, N) in shapes:
        for relu in (False, True):
            nest = mapspace.canonical(M, K, N, T)
            got = gemm_from_nest(nest, M, K, N, relu)
            ref = gemm_program(M, K, N, relu)
            tag = f"{'gemm.relu' if relu else 'gemm'} {M}x{K}x{N}"
            if len(got) != len(ref):
                raise AssertionError(
                    f"seam {tag}: the canonical nest emits {len(got)} "
                    f"instructions, gemm_program emits {len(ref)}")
            for i, (g, r) in enumerate(zip(got, ref)):
                if g != r:
                    raise AssertionError(
                        f"seam {tag}: instruction {i} differs -- nest "
                        f"{g[0]:#018x}/{g[1]:#018x} vs gemm_program "
                        f"{r[0]:#018x}/{r[1]:#018x}. The encoder and the "
                        f"shipped program must stay the same generator.")
    print(f"  SEAM OK: the canonical nest re-emits gemm_program word for word "
          f"at {len(shapes)} shape(s) x {{gemm, gemm.relu}}")


def correct(prog, M, K, N, relu, seed):
    """`prog` computes relu(A@B) into the M x N region and touches nothing else."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, (MAXDIM, MAXDIM)).astype(np.int8)
    B = rng.integers(-128, 128, (MAXDIM, MAXDIM)).astype(np.int8)
    C0 = rng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
    gold = C0.reshape(MAXDIM, MAXDIM).copy()
    acc = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        acc = np.maximum(acc, 0)
    gold[:M, :N] = np.clip(acc, -128, 127).astype(np.int8)
    got = isa_ref.run(prog, A.reshape(-1), B.reshape(-1), C0)
    return np.array_equal(got.reshape(MAXDIM, MAXDIM), gold)


def main(argv):
    shapes = shapes_from(argv)
    print(f"TinyTPU-isa co-design mapspace: T={T}, MAXDIM={MAXDIM}, "
          f"IMEM_SIZE={IMEM_SIZE}; {len(shapes)} shape(s)")
    seam(shapes)
    ok = True
    for (M, K, N) in shapes:
        found = mapspace.search(M, K, N)
        tag = f"{M}x{K}x{N}"
        print(f"MAPSPACE {tag}: {found['encodable']}/{found['total']} encodable")
        for cause, n in found["refused"].items():
            print(f"MAPSPACE {tag}: refused {n:6d}  {cause}")
        if not found["ranked"]:
            print(f"MAPSPACE {tag}: CHOSEN none   <-- FAIL: this hardware can "
                  f"encode no nest at all for {tag}")
            ok = False
            continue
        checked = 0
        for i, (dyn, words, name, nest, prog) in enumerate(found["ranked"]):
            if i >= CHECK_MAX:
                break
            for relu in (False, True):
                p = prog if not relu else gemm_from_nest(nest, M, K, N, True)
                w = assemble(p)
                if len(w) > IMEM_SIZE:            # assemble asserts it too
                    print(f"MAPSPACE {tag}: nest {name} needs {len(w)} words > "
                          f"IMEM_SIZE={IMEM_SIZE}   <-- FAIL")
                    ok = False
                    continue
                if not correct(p, M, K, N, relu, 7000 + 13 * i + relu):
                    print(f"MAPSPACE {tag}: nest {name} relu={relu} is WRONG "
                          f"against isa_ref   <-- FAIL")
                    ok = False
                checked += 1
        dyn, words, name, _nest, _prog = found["ranked"][0]
        print(f"MAPSPACE {tag}: CHOSEN {name} ({dyn} dynamic, {words} words, "
              f"IMEM_SIZE={IMEM_SIZE})")
        print(f"MAPSPACE {tag}: verified {checked} program(s) against isa_ref "
              f"(top {min(found['encodable'], CHECK_MAX)} of "
              f"{found['encodable']} encodable x {{gemm, gemm.relu}})")
    print("  MAPSPACE OK" if ok else "  MAPSPACE FAILED")
    return 0 if ok else 1


sys.exit(main(sys.argv))

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the cheap gate is worth: one csynth, then cosim on everything asked for.

Absolute error against `cycles.estimate` on held-out specs, and -- the property
a gate actually needs -- whether the estimate orders the variants of one spec
the way cosim does. Prose: docs/source/extensions/act_specs.rst.

    calibrate.py specs    [name ...]     one measurement per spec
    calibrate.py variants [name ...]     every variant of each spec"""

import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.act import (  # noqa: E402
    baseline, correctness, cycles, legality, measure, variants,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402

DEFAULT_SPECS = ["gemm_4x4x4", "gemm_8x8x8", "gemm_12x12x12", "gemm_16x16x8",
                 "gemm_16x16x16", "gemm_relu_16x16x16", "gemm_reuse_m_16x16x4",
                 "gemm_reuse_n_4x4x16", "gemm_reuse_k_4x16x4",
                 "batched_matmul_2x4x4x4", "relu_16x16", "row_reduce_16x16"]
DEFAULT_VARIANTS = ["gemm_8x8x8", "gemm_16x16x8", "gemm_16x16x16"]


def jobs(mode, names):
    for name in names:
        sp = spec_mod.by_name(name)
        makers = (variants.VARIANTS if mode == "variants"
                  else {"reference": baseline.program})
        for tag, make in makers.items():
            try:
                prog = make(sp)
            except baseline.Unsupported as e:
                print(f"  {name} / {tag}: UNSUPPORTED: {e}")
                continue
            bad = legality.check(sp, prog)
            if bad is not None:
                print(f"  {name} / {tag}: ILLEGAL\n{bad}")
                continue
            yield sp, tag, prog


def rank_agreement(measured):
    """Pairs of variants of one spec the estimate orders as cosim does."""
    agree = total = 0
    lines = []
    for name, rows in measured.items():
        for (t1, e1, c1), (t2, e2, c2) in itertools.combinations(rows, 2):
            if c1 is None or c2 is None or c1 == c2:
                continue
            total += 1
            ok = (e1 < e2) == (c1 < c2) or (e1 == e2 and False)
            agree += ok
            if not ok:
                lines.append(f"    {name}: estimate puts {t1} ({e1:.0f}) "
                             f"{'below' if e1 < e2 else 'above'} {t2} "
                             f"({e2:.0f}), cosim says {c1} vs {c2}")
    return agree, total, lines


def main(argv):
    mode = argv[0] if argv else "specs"
    names = argv[1:] or (DEFAULT_VARIANTS if mode == "variants"
                         else DEFAULT_SPECS)
    work = list(jobs(mode, names))
    mod = correctness.build_module()
    for sp, tag, prog in work:
        fails = correctness.check(sp, prog, mod, dists=("small",), repeats=1)
        if fails and not spec_mod.known_gap(sp):
            print(f"  {sp['name']} / {tag}: NOT BIT-EXACT, not measured")
            for line in fails:
                print(f"      {line}")
            return 1
    prj = measure.PRJ
    print(f"  synthesizing once into {prj} ...", flush=True)
    measure.synthesize(prj)
    print(f"  {'spec':28s} {'variant':18s} {'critical':>8s} {'work':>6s} "
          f"{'estimate':>9s} {'cosim':>7s} {'error':>8s}   testbench")
    measured = {}
    for sp, tag, prog in work:
        est = cycles.estimate(prog)
        n, line = measure.measure(sp, prog, prj, tag=f"{sp['name']}_{tag}")
        err = "" if not n else f"{(est - n) / n * 100:+7.1f}%"
        print(f"  {sp['name']:28s} {tag:18s} {cycles.critical_unit(prog):>8s} "
              f"{cycles.critical_work(prog):6d} {est:9.0f} {str(n):>7s} "
              f"{err:>8s}   {line}", flush=True)
        measured.setdefault(sp["name"], []).append((tag, est, n))
    if mode == "variants":
        agree, total, lines = rank_agreement(measured)
        print(f"  rank agreement: {agree}/{total} ordered pairs")
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

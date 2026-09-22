# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parity baseline's measurement: program orders x shapes on ONE build,
scored against matched Gemmini by the rule fixed in advance.

    python parity_sweep.py                        # the named `parity` config
    TPU_PARITY_ORDERS=shipped,interleaved python parity_sweep.py
    python parity_sweep.py --list                 # invariants only, no Vitis

The configuration is the environment the design module reads (`TPU_T`,
`TPU_MAXDIM`, `TPU_DMA_WIDEN`); `PARITY_CONFIGS` below names the two the
baseline is kept at. `csynth_design` runs once; every (order, shape) cosim runs
in its own copy of the synthesized project, `TPU_JOBS` at a time, and the copy
is deleted as soon as its number is read. A cosim that yields no number is
retried once (the pipeline has been seen to drop one in nine).

Parity at a shape (docs/source/designs/gemmini_comparison.rst, "The parity
baseline", fixed before any candidate was measured): ours <= Gemmini median +
its published min-max spread. `faster` is ours < median.
"""

import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))

#: THE NAMED PARITY BASELINES. Kept beside the shipped design, not replacing
#: it (docs/source/designs/gemmini_comparison.rst, "The parity baseline").
#:
#: Three changes from the shipped design, each with one mechanism: the banked
#: burst widening, the `interleaved` program order, and QD=32. The depth is
#: load-bearing twice over -- it is what makes the `interleaved` order legal
#: (it deadlocks at `Kt >= QD`, limitations item 24), and at T=4/MAXDIM=64 the
#: largest expressible `Kt` is MAXDIM/T = 16, so QD=32 clears every shape this
#: build can express rather than the ones that happened to be measured.
PARITY_CONFIGS = {
    "parity-t4": dict(TPU_T="4", TPU_MAXDIM="64", TPU_DMA_WIDEN="1",
                      TPU_QD="32", TPU_PROGRAM="interleaved"),
    # T=8 is NOT yet measured in this configuration; its published column is
    # the widening alone at QD=8. See the page.
    "parity-t8": dict(TPU_T="8", TPU_MAXDIM="64", TPU_DMA_WIDEN="1",
                      TPU_PROGRAM="shipped"),
}

#: Matched Gemmini, median of five trials and full min-max spread, from
#: docs/source/designs/benchmarks.rst (T=4 vs DIM=4 and T=8 vs DIM=8, both at
#: MAXDIM=64). The shape sets are those tables' own, unedited.
GEMMINI = {
    4: {(4, 4, 4): (208, 25), (8, 8, 8): (324, 36), (12, 12, 12): (458, 17),
        (16, 16, 8): (527, 44), (16, 16, 16): (691, 44),
        (32, 32, 32): (2977, 34), (48, 48, 48): (9100, 35),
        (64, 64, 64): (20287, 34), (64, 32, 64): (11175, 18),
        (32, 64, 32): (5570, 147)},
    8: {(8, 8, 8): (305, 17), (16, 16, 8): (500, 16), (16, 16, 16): (535, 45),
        (32, 32, 32): (1280, 18), (48, 48, 48): (3028, 35),
        (64, 64, 64): (6033, 35), (64, 32, 64): (3897, 35),
        (32, 64, 32): (1866, 36)},
}

cfg = os.environ.get("TPU_PARITY_CONFIG")
if cfg:
    for k, v in PARITY_CONFIGS[cfg].items():
        os.environ.setdefault(k, v)

import numpy as np  # noqa: E402
from allo.dataflow import customize  # noqa: E402
from examples.accelerator.tinytpu_vitis import cosim as C  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, assemble, expand, schedule, T, MAXDIM, WPR, DMA_WORDS,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program  # noqa: E402

PRJ = os.path.abspath(os.environ.get("TPU_PRJ", "parity.prj"))
JOBS = int(os.environ.get("TPU_JOBS", "6"))
ORDERS = os.environ.get("TPU_PARITY_ORDERS",
                        os.environ.get("TPU_PROGRAM", "interleaved")).split(",")
SHAPES = sorted(GEMMINI[T], key=lambda s: s[0] * s[1] * s[2])
if os.environ.get("TPU_SHAPES"):
    SHAPES = [tuple(int(x) for x in t.split("x"))
              for t in os.environ["TPU_SHAPES"].split(",")]


def invariants(prog):
    """Integer invariants of a program: they change only if the machine's
    work changes, so a moved cycle count with equal invariants is a timing
    effect, not a different amount of work."""
    w = assemble(prog)
    na, nb = w[7] & 0xFFFF, (w[7] >> 16) & 0xFFFF
    return dict(dyn=len(expand(prog)), dld=w[1], spm=w[2], vru=w[3],
                accu=w[5], dst=w[6],
                burst=-(-max(na, nb) * WPR // DMA_WORDS))


def csynth_totals(prj):
    rpt = open(os.path.join(prj, "out.prj/solution1/syn/report/csynth.rpt")).read()
    row = re.search(r"\|\+ tinytpu_isa\*?\s*\|[^\n]*", rpt).group(0)
    cells = [c.strip() for c in row.split("|")]
    nums = [re.match(r"(\d+)", c).group(1) for c in cells[-6:-1] if re.match(r"\d", c)]
    log = open(os.path.join(prj, "csynth.log"), errors="replace").read()
    fmax = re.findall(r"Estimated Fmax: ([\d.]+) MHz", log)
    return dict(zip(("BRAM", "DSP", "FF", "LUT"), nums),
                period_ns=round(1000 / float(fmax[-1]), 3) if fmax else None)


def synth():
    base = os.path.join(PRJ, "base")
    if os.environ.get("TPU_REUSE_SYN") == "1" and os.path.exists(
            os.path.join(base, "out.prj/solution1/syn/report/csynth.rpt")):
        return base
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=base, wrap_io=False,
            configs={"align_value": 64})
    C.patch_axi_depths(base)
    open(os.path.join(base, "tb.cpp"), "w").write(C.testbench(*SHAPES[0]))
    for attempt in (1, 2):
        C.vitis(base, C.TCL_SYN, "csynth.log")
        if os.path.exists(os.path.join(base, "out.prj/solution1/syn/report/csynth.rpt")):
            return base
    raise SystemExit(f"csynth failed twice; see {base}/csynth.log")


def one(base, order, shape):
    M, K, N = shape
    prog = gemm_program(M, K, N, order=order)
    job = os.path.join(PRJ, f"{order}_{M}x{K}x{N}")
    n = None
    for attempt in (1, 2):
        shutil.rmtree(job, ignore_errors=True)
        shutil.copytree(base, job, symlinks=True,
                        ignore=shutil.ignore_patterns("sim"))
        # the solution records its project path; point it at the copy
        subprocess.call(["grep", "-rlZ", base, os.path.join(job, "out.prj")],
                        stdout=open(os.path.join(job, "paths"), "w"))
        for p in filter(None, open(os.path.join(job, "paths")).read().split("\0")):
            t = open(p, errors="surrogateescape").read()
            open(p, "w", errors="surrogateescape").write(t.replace(base, job))
        open(os.path.join(job, "tb.cpp"), "w").write(
            C.testbench(M, K, N, prog=prog))
        # A cosim that hangs (this design can, for a program every functional
        # model accepts -- see the parity baseline's notes) must not hold a
        # worker forever; `timeout` kills it and the job reports no number.
        open(os.path.join(job, "run.tcl"), "w").write(C.TCL_COSIM)
        with open(os.path.join(job, "cosim.log"), "w") as f:
            subprocess.call(
                ["timeout", "-k", "10", os.environ.get("TPU_COSIM_TIMEOUT", "3600"),
                 "bash", "-lc",
                 f"source {C.VITIS} && cd {job} && vitis_hls -f run.tcl"],
                stdout=f, stderr=subprocess.STDOUT)
        text = open(os.path.join(job, "cosim.log"), errors="replace").read()
        mm = [l.strip() for l in text.splitlines() if "mismatches" in l]
        n = C.cycles(job)
        exact = bool(mm) and re.search(r"mismatches = 0\b", mm[-1]) is not None
        if n is not None:
            break
    if n is not None:
        shutil.rmtree(job, ignore_errors=True)
    return order, shape, n, exact, invariants(prog)


def verdict(shape, n):
    if n is None or shape not in GEMMINI[T]:
        return "-", ""
    med, spr = GEMMINI[T][shape]
    tag = "FASTER" if n < med else ("parity" if n <= med + spr else "BEHIND")
    return f"{med} +/- {spr}", f"{n / med:.3f}x {tag}"


def main():
    print(f"config: T={T} MAXDIM={MAXDIM} DMA_WORDS={DMA_WORDS} "
          f"orders={ORDERS} shapes={len(SHAPES)}", flush=True)
    if "--list" in sys.argv:
        for o in ORDERS:
            for s in SHAPES:
                print(f"  {o:12s} {'%dx%dx%d' % s:9s} "
                      f"{invariants(gemm_program(*s, order=o))}")
        return 0
    base = synth()
    tot = csynth_totals(base)
    print(f"csynth: {tot}", flush=True)
    res = {}
    with cf.ThreadPoolExecutor(JOBS) as ex:
        futs = [ex.submit(one, base, o, s) for o in ORDERS for s in SHAPES]
        for f in cf.as_completed(futs):
            o, s, n, ok, inv = f.result()
            res[(o, s)] = (n, ok, inv)
            print(f"  {o:12s} {'%dx%dx%d' % s:9s} cycles={n} "
                  f"{'exact' if ok else 'MISMATCH'} {inv}", flush=True)
    print(f"\n  T={T} MAXDIM={MAXDIM} DMA_WORDS={DMA_WORDS}  csynth {tot}")
    print(f"  {'shape':9s} " + " ".join(f"{o:>12s}" for o in ORDERS)
          + f"  {'Gemmini':>14s}  verdict ({ORDERS[-1]})")
    all_ok = True
    for s in SHAPES:
        row = [res.get((o, s), (None, False, None)) for o in ORDERS]
        all_ok &= all(r[1] for r in row)
        g, v = verdict(s, row[-1][0])
        print(f"  {'%dx%dx%d' % s:9s} " + " ".join(f"{str(r[0]):>12s}" for r in row)
              + f"  {g:>14s}  {v}")
    print("  ALL EXACT" if all_ok else "  MISMATCHES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

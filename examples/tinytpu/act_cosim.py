#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosim the mappings the search ranks, so the cost model can be wrong out loud.

    TPU_PRJ=/tmp/act-cosim.prj python act_cosim.py gemm.relu 16x16x16 --top 2

One synthesis, one cosim per mapping. The cycle column is the only measurement
in this flow; `makespan` is a model over the units `assemble()` promises.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from allo.dataflow import customize  # noqa: E402
from examples.tinytpu import cosim  # noqa: E402
from examples.tinytpu.act_compile import (  # noqa: E402
    compile_one, extents_of,
)
from examples.tinytpu.isa_dsl import (  # noqa: E402
    gemm_program,
)
from examples.tinytpu.microarch_isa import (  # noqa: E402
    expand, schedule, tinytpu_isa,
)
from act import target, workloads  # noqa: E402
from examples.tinytpu import act_target  # noqa: E402,F401


def measure(prj, shape, relu, program, tag):
    report = os.path.join(prj, "out.prj/solution1/sim/report",
                          "tinytpu_isa_cosim.rpt")
    if os.path.exists(report):
        os.remove(report)
    open(os.path.join(prj, "tb.cpp"), "w").write(
        cosim.testbench(*shape, relu=relu, prog=program))
    text = cosim.vitis(prj, cosim.TCL_COSIM, f"cosim_{tag}.log")
    lines = [l.strip() for l in text.splitlines() if "mismatches" in l]
    exact = bool(lines) and re.search(r"mismatches = 0\b", lines[-1])
    return cosim.cycles(prj), bool(exact)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workload", nargs="?", default="gemm.relu")
    ap.add_argument("shape", nargs="?", default="16x16x16")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--only", default=None,
                    help="measure just the mapping with this label, wherever "
                         "the search ranked it")
    ap.add_argument("--baseline", action="store_true",
                    help="also measure isa_dsl.gemm_program, the hand-written "
                         "generator's program for the same shape")
    args = ap.parse_args(argv)

    machine = target.get("tinytpu-isa")
    workload = workloads.get(args.workload)
    extents = extents_of(workload, args.shape)
    _, result = compile_one(args.workload, extents, machine)
    chosen = ([c for c in result.candidates if c.label == args.only]
              if args.only else result.ranked(args.top))
    if args.only and not chosen:
        raise SystemExit(
            f"no encodable mapping labelled {args.only!r}; the search found "
            f"{[c.label for c in result.candidates]}")
    shape = tuple(extents[r] for r in workload.ranks)
    print(f"{args.workload} {args.shape}: cosim the top {len(chosen)} of "
          f"{len(result.candidates)} encodable mappings")

    prj = cosim.PRJ
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False,
            configs={"align_value": 64})
    cosim.patch_axi_depths(prj)
    open(os.path.join(prj, "tb.cpp"), "w").write(
        cosim.testbench(*shape, relu=bool(workload.epilogue)))
    print("  synthesizing once ...", flush=True)
    cosim.vitis(prj, cosim.TCL_SYN, "csynth.log")

    rows = []
    if args.baseline:
        hand = gemm_program(*shape, relu=bool(workload.epilogue))
        cycles, exact = measure(prj, shape, bool(workload.epilogue), hand,
                                "baseline")
        print(f"  {'hand-written':14s} {len(hand):3d} instr "
              f"{len(expand(hand)):4d} dyn             cycles={cycles}  "
              f"{'exact' if exact else 'MISMATCH'}", flush=True)
    for rank, candidate in enumerate(chosen):
        cycles, exact = measure(prj, shape, bool(workload.epilogue),
                                candidate.program, candidate.label)
        rows.append((candidate, cycles, exact))
        print(f"  {candidate.label:14s} {len(candidate.program):3d} instr "
              f"{len(expand(candidate.program)):4d} dyn  model "
              f"{candidate.cost[0]:5d}  cycles={cycles}  "
              f"{'exact' if exact else 'MISMATCH'}", flush=True)
    measured = [(c.label, n) for c, n, ok in rows if n is not None and ok]
    if len(measured) < 2:
        print("  ACT COSIM: not enough measurements to judge the ranking")
        return 0 if all(ok for _, _, ok in rows) else 1
    ordered = [label for label, _ in measured]
    fastest = min(measured, key=lambda kv: kv[1])[0]
    print(f"\n  model ranks {ordered[0]} first; cosim says {fastest} is "
          f"fastest -- the model is "
          f"{'right' if fastest == ordered[0] else 'WRONG'} here")
    return 0


if __name__ == "__main__":
    sys.exit(main())

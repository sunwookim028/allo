#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One workload spec in, a schedule and a verified program out.

    python act_compile.py gemm.relu 16x16x16
    python act_compile.py --list
    python act_compile.py --gate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from act import target, workloads  # noqa: E402
from act.search import Problem, search  # noqa: E402
from examples.accelerator.tinytpu_vitis.act_target import (  # noqa: E402
    CAUSE_KIND,
)
from examples.accelerator.tinytpu_vitis.bench_isa import SHAPES  # noqa: E402

GATE = [("gemm", s) for s in SHAPES] + [("gemm.relu", s) for s in SHAPES] + [
    ("gemm.sum", (8, 8, 8)), ("gemm.sum.relu", (16, 16, 4))]


def extents_of(workload, text):
    sizes = [int(n) for n in text.replace(",", "x").split("x")]
    if len(sizes) != len(workload.ranks):
        raise SystemExit(
            f"{workload.name} has ranks {list(workload.ranks)}; "
            f"{text!r} gives {len(sizes)} extents")
    return dict(zip(workload.ranks, sizes))


def compile_one(name, extents, machine, slots=2):
    problem = Problem(workloads.get(name), extents)
    return problem, search(problem, machine, slots)


def show(result, machine, top, verify):
    print(f"{result.problem.title}: {result.considered} nests, "
          f"{len(result.candidates)} encodable")
    for cause, count in result.census.rows():
        where, detail = result.census.examples[cause]
        every = result.census.any_counts.get(cause, count)
        print(f"  {count:6d} first {every:6d} total  "
              f"{CAUSE_KIND.get(cause, '?'):8s} {cause:12s} "
              f"{where}: {detail[:52]}")
    if not result.best:
        raise SystemExit("every nest was refused")
    print(f"\n  {'mapping':16s} {'rows':>4s} {'static':>6s} {'words':>5s} "
          f"{'emits':>5s} {'makespan':>8s} {'bottleneck':>12s}  check")
    for candidate in result.ranked(top):
        counts = machine.report(candidate.program)
        plan = candidate.priced.schedule
        unit, load = plan.bottleneck
        verdict = ("correct" if machine.verify(
            result.problem.workload, result.problem.extents,
            candidate.program) else "WRONG") if verify else "-"
        print(f"  {candidate.label:16s} "
              f"{intrinsic_rows(candidate.nest):>4d} "
              f"{counts['static']:>6d} {counts['words']:>5d} "
              f"{counts['dynamic']:>5d} {plan.makespan:>8d} "
              f"{unit + ' ' + str(load):>12s}  {verdict}")
    return result.best


def intrinsic_rows(nest):
    return max(l.factor for l in nest if l.level == "intrinsic")


def gate(machine):
    bad = []
    for name, shape in GATE:
        workload = workloads.get(name)
        extents = dict(zip(workload.ranks, shape))
        _, result = compile_one(name, extents, machine)
        if not result.best:
            bad.append(f"{name} {shape}: every nest refused")
            continue
        for candidate in result.candidates:
            if not machine.verify(workload, extents, candidate.program):
                bad.append(f"{name} {shape} {candidate.label}: wrong result")
    total = sum(1 for _ in GATE)
    for line in bad:
        print(f"  FAIL {line}")
    print(f"ACT GATE {'OK' if not bad else 'FAILED'}: {total - len(bad)}/"
          f"{total} problems, every encodable mapping verified")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workload", nargs="?", default="gemm.relu")
    ap.add_argument("shape", nargs="?", default="16x16x16")
    ap.add_argument("--target", default="tinytpu-isa")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args(argv)
    machine = target.get(args.target)
    if args.list:
        print("workloads:", " ".join(sorted(workloads.WORKLOADS)))
        print("targets:  ", " ".join(sorted(target.TARGETS)))
        return 0
    if args.gate:
        return gate(machine)
    workload = workloads.get(args.workload)
    _, result = compile_one(args.workload, extents_of(workload, args.shape),
                            machine, args.slots)
    best = show(result, machine, args.top, not args.no_verify)
    print(f"\n  chosen: {best.label}  cost {best.cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

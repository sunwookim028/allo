# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three verdicts on a generated program, one command each.

    judge.py specs                       what the corpus asks for
    judge.py rules                       the rejection rules, and their self-test
    judge.py legal   [--spec S] [--program REF]
    judge.py correct [--spec S] [--program REF]
    judge.py fast    [--spec S] [--program REF] [--cosim]

Prose: docs/source/extensions/act_specs.rst."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.act import (  # noqa: E402
    baseline, correctness, cycles, legality, submission,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402

GATE_MARGIN = 1.10


def chosen(args):
    """The specs to judge: everything this build can hold, unless named."""
    if args.spec != "all":
        return [spec_mod.by_name(args.spec)]
    keep = []
    for sp in spec_mod.corpus():
        why = spec_mod.fits_build(sp)
        if why is None:
            keep.append(sp)
        else:
            print(f"  {sp['name']:28s} SKIPPED  {why}")
    return keep


def build_program(make, sp):
    """`(program, None)` or `(None, why the generator could not map the spec)`."""
    try:
        return make(sp), None
    except baseline.Unsupported as e:
        return None, f"the generator has no mapping: {e}"
    except Exception as e:   # noqa: BLE001 -- a generator fault is a verdict
        return None, f"the generator raised {type(e).__name__}: {e}"


def cmd_specs(args):
    for sp in chosen(args):
        print("  " + spec_mod.summary(sp))
        print(f"      {sp['stresses']}")
        if spec_mod.known_gap(sp):
            print(f"      KNOWN GAP: {spec_mod.known_gap(sp)}")
    return 0


def cmd_rules(args):
    from examples.accelerator.tinytpu_vitis.act import rules_test
    for layer, rules in (("isa", legality.ISA_RULES),
                         ("encoding", legality.ENCODING_RULES),
                         ("protocol", (legality.PROTOCOL_RULE,)),
                         ("spec", legality.SPEC_RULES)):
        for r in rules:
            print(f"  {layer:8s} {r.name:24s} {r.statement}")
    return rules_test.main()


def cmd_legal(args):
    make = submission.load(args.program)
    bad = 0
    for sp in chosen(args):
        prog, why = build_program(make, sp)
        if prog is None:
            print(f"  {sp['name']:28s} NO PROGRAM  {why}")
            bad += spec_mod.known_gap(sp) is None
            continue
        r = legality.check(sp, prog)
        if r is None:
            print(f"  {sp['name']:28s} LEGAL")
        else:
            bad += 1
            print(f"  {sp['name']:28s} {r}")
    print("  LEGAL OK" if not bad else f"  {bad} ILLEGAL")
    return 0 if not bad else 1


def cmd_correct(args):
    make = submission.load(args.program)
    specs = chosen(args)
    mod = None if args.no_simulator else correctness.build_module()
    bad = 0
    for sp in specs:
        prog, why = build_program(make, sp)
        if prog is None:
            print(f"  {sp['name']:28s} NO PROGRAM  {why}")
            bad += spec_mod.known_gap(sp) is None
            continue
        r = legality.check(sp, prog)
        if r is not None:
            print(f"  {sp['name']:28s} ILLEGAL, not run\n{r}")
            bad += 1
            continue
        fails = correctness.check(sp, prog, mod)
        gap = spec_mod.known_gap(sp)
        if not fails:
            print(f"  {sp['name']:28s} BIT-EXACT"
                  + ("   <-- a known gap was expected here" if gap else ""))
            bad += gap is not None
        elif gap:
            print(f"  {sp['name']:28s} KNOWN GAP: {gap}")
            print(f"      {fails[0]}")
        else:
            bad += 1
            print(f"  {sp['name']:28s} WRONG")
            for line in fails:
                print(f"      {line}")
    print("  CORRECT OK" if not bad else f"  {bad} NOT AS EXPECTED")
    return 0 if not bad else 1


def cmd_fast(args):
    make = submission.load(args.program)
    specs = chosen(args)
    rows = []
    bad = 0
    for sp in specs:
        prog, why = build_program(make, sp)
        ref, _ = build_program(baseline.program, sp)
        if prog is None:
            print(f"  {sp['name']:28s} NO PROGRAM  {why}")
            continue
        r = legality.check(sp, prog)
        if r is not None:
            print(f"  {sp['name']:28s} ILLEGAL, not measured\n{r}")
            bad += 1
            continue
        est = cycles.estimate(prog)
        budget = cycles.estimate(ref) * GATE_MARGIN if ref else None
        passed = budget is None or est <= budget
        bad += not passed
        print(f"  {sp['name']:28s} {cycles.report(prog)}"
              + ("" if budget is None else
                 f"; gate {'PASS' if passed else 'REJECT'} against the "
                 f"reference submission's {cycles.estimate(ref):.0f} "
                 f"x {GATE_MARGIN}"))
        rows.append((sp, prog, est, passed))
    if not args.cosim:
        print("  GATE OK" if not bad else f"  {bad} REJECTED BY THE GATE")
        return 0 if not bad else 1
    from examples.accelerator.tinytpu_vitis.act import measure
    prj = measure.PRJ
    print(f"  synthesizing once into {prj} ...", flush=True)
    measure.synthesize(prj)
    print(f"  {'spec':28s} {'estimate':>9s} {'cosim':>7s} {'error':>7s}   testbench")
    for sp, prog, est, passed in rows:
        if not passed and not args.measure_rejected:
            continue
        n, line = measure.measure(sp, prog, prj)
        err = "" if not n else f"{(est - n) / n * 100:+6.1f}%"
        print(f"  {sp['name']:28s} {est:9.0f} {str(n):>7s} {err:>7s}   {line}",
              flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=("specs", "rules", "legal", "correct", "fast"))
    ap.add_argument("--spec", default="all")
    ap.add_argument("--program", default=submission.BASELINE)
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--measure-rejected", action="store_true")
    ap.add_argument("--no-simulator", action="store_true")
    args = ap.parse_args(argv)
    return {"specs": cmd_specs, "rules": cmd_rules, "legal": cmd_legal,
            "correct": cmd_correct, "fast": cmd_fast}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

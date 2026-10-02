# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential check of one MiniTPU unit: Allo backends against the RTL.

    python -m examples.minitpu.harness.check bf16_add [--backend simulator]
        [--variant native] [--n 251936]

The RTL (Verilator) is the oracle. For each Allo variant and backend the
check prints one verdict line,

    UNIT-MATCH <unit> <variant> <backend> n/n latency=L
    UNIT-DIFF  <unit> <variant> <backend> k/n differ ...

and, on a difference, a classification of the differing vectors against the
unit's reference model (``harness/ref.py``), so a mismatch names its cause.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from examples.minitpu.harness import rtl, stimulus  # noqa: E402

BACKENDS = ("simulator", "systemc")
# What each backend can say about the unit's latency (values are checked here;
# latency on the Allo side only on RTL Allo produced, the Catapult track).
LATENCY_SEEN = {"simulator": "untimed", "systemc": "unchecked"}


def build(unit_mod, variant, backend, n, project):
    import allo.dataflow as df  # noqa: F401 -- imported late: slow

    make, runner = unit_mod.VARIANTS[variant]
    top = make(n)
    if backend == "simulator":
        mod = df.build(top, target="simulator")
    elif backend == "systemc":
        mod = df.build(top, target="systemc", mode="csim", project=project)
    else:
        raise ValueError(backend)
    return mod, runner


def classify(stim, got, want, explain):
    """Group differing vectors by the first rule in ``explain`` they fit."""
    groups = {}
    for i in np.flatnonzero(got != want):
        key = "unexplained"
        for name, rule in explain:
            if rule(stim[i], got[i], want[i]):
                key = name
                break
        groups.setdefault(key, []).append(i)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit")
    ap.add_argument("--backend", action="append", choices=BACKENDS)
    ap.add_argument("--variant", action="append")
    ap.add_argument("--n", type=int, default=0, help="first n vectors only")
    ap.add_argument("--project", default="/tmp/minitpu_harness_prj")
    args = ap.parse_args(argv)

    u = importlib.import_module(f"examples.minitpu.units.{args.unit}")
    # a unit may bring its own stimulus (acc24 operands, ALU op codes)
    stim = u.stimulus() if hasattr(u, "stimulus") else stimulus.binary_bf16()
    if args.n:
        stim = stim[: args.n]
    n = len(stim)

    t = time.time()
    want, cyc = rtl.run(u.RTL, stim.astype(np.uint64))
    want = want[:, 0].astype(np.uint16 if u.RTL.outputs[0][1] <= 16 else np.uint32)
    lat = sorted(set(cyc.tolist()))
    print(f"RTL {u.RTL.top}: {n} vectors, latency {lat} (declared {u.RTL.latency}), {time.time() - t:.1f}s", flush=True)
    if lat != [u.RTL.latency]:
        print(f"LATENCY-DIFF {args.unit} rtl: measured {lat} declared {u.RTL.latency}")

    bad_any = False
    for variant in args.variant or list(u.VARIANTS):
        for backend in args.backend or ["simulator"]:
            t = time.time()
            prj = os.path.join(args.project, f"{args.unit}_{variant}_{backend}")
            try:
                mod, runner = build(u, variant, backend, n, prj)
                tb = time.time() - t
                got = runner(mod, stim)
            except Exception as e:  # a build or run failure is a finding, not a crash
                print(f"UNIT-FAIL  {args.unit} {variant} {backend}: {type(e).__name__}: {str(e)[:400]}")
                bad_any = True
                continue
            k = int((got != want).sum())
            tag = "UNIT-MATCH" if k == 0 else "UNIT-DIFF "
            # The Allo side's latency is NOT measured here: the simulator is
            # untimed, and SystemC csim's cycles are the handshakes', not the
            # unit's (u1_pipe_2026-10-02.rst). Say so rather than echo the RTL's.
            print(f"{tag} {args.unit} {variant} {backend} {n - k}/{n} "
                  f"latency={LATENCY_SEEN[backend]} (rtl {u.RTL.latency}) "
                  f"(build {tb:.1f}s, run {time.time() - t - tb:.1f}s)", flush=True)
            if k:
                bad_any = True
                for name, idx in classify(stim, got, want, getattr(u, "EXPLAIN", [])).items():
                    ex = ", ".join(
                        f"{'+'.join(f'{x:04x}' for x in stim[i])}: allo {got[i]:04x} rtl {want[i]:04x}"
                        for i in idx[:3]
                    )
                    print(f"    {len(idx):7d}  {name}  e.g. {ex}")
    return 1 if bad_any else 0


if __name__ == "__main__":
    sys.exit(main())

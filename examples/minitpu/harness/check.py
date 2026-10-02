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

A ``trace``-shape unit (U2 storage; ``RTL.shape == "trace"``) is checked per
cycle instead: every trace of ``u.traces(inst)`` and ``u.seeds()`` is joined
into one command trace, run through the RTL (Verilator, seeded random
initial state) and through each Allo variant, and compared on every slot the
unit's trace reference calls *defined*. Undefined slots are masked and
counted by reason (owner decision: masked and counted, not refused)::

    UNIT-MATCH <unit> <variant> <backend> k/k defined (m masked: uninit=m; allo==rtl on f) cycle=...
    UNIT-DIFF  <unit> <variant> <backend> k/K defined ...

``cycle=`` says what an iteration is on that backend: on the simulator an
*order* only; in SystemC csim the check compares values per iteration and
does not look at time; per-cycle timing is checked on Catapult RTL only.
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


CYCLE_SEEN = {"simulator": "order-only", "systemc": "per-iteration, time unchecked"}


def _trace_all(u, inst, n_max=0):
    """Every trace of the instance joined end to end, with label spans."""
    from examples.minitpu.harness.traces import concat

    parts = [(lab, c) for lab, c, _legal in u.traces(inst)]
    for lab, sinst, c, _seen, _legal in (u.seeds() if hasattr(u, "seeds") else []):
        if sinst == inst:
            parts.append((lab, c))
    spans, t = [], 0
    for lab, c in parts:
        k = len(next(iter(c.values())))
        spans.append((lab, t, t + k))
        t += k
    cmd = concat(*[c for _, c in parts])
    if n_max:
        cmd = {p: v[:n_max] for p, v in cmd.items()}
        spans = [(lab, a, min(b, n_max)) for lab, a, b in spans if a < n_max]
    return cmd, spans


def check_trace(u, args):
    import allo.dataflow as df  # noqa: F401 -- slow

    inst = args.inst or u.DEFAULT
    unit = u.INSTANCES[inst]
    w = u.WIDTH[inst]
    cmd, spans = _trace_all(u, inst, args.n)
    n = len(next(iter(cmd.values())))
    packed = {p: rtl.pack(cmd[p], wd) for p, wd in unit.inputs}
    t = time.time()
    rtl_out = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = u.REF(inst, packed)
    ndef = sum(int((reason[p] == "").sum()) for p in want)
    bad_ref = sum(int(((reason[p] == "") & (rtl_out[p] != want[p]).any(axis=1)).sum()) for p in want)
    print(f"RTL {unit.top}:{inst}: {n} cycles in {len(spans)} traces, {ndef} defined slots, "
          f"RTL vs reference on defined: {ndef - bad_ref}/{ndef} ({time.time() - t:.1f}s)", flush=True)
    rtl_int = {p: rtl.unpack(rtl_out[p]) for p in want}
    bad_any = bad_ref != 0
    for variant in args.variant or list(u.VARIANTS):
        for backend in args.backend or ["simulator"]:
            t = time.time()
            prj = os.path.join(args.project, f"{args.unit}_{inst}_{variant}_{backend}")
            make, runner = u.VARIANTS[variant]
            try:
                top = make(n, w)
                if backend == "simulator":
                    mod = df.build(top, target="simulator")
                else:
                    mod = df.build(top, target="systemc", mode="csim", project=prj)
                tb = time.time() - t
                got = runner(mod, cmd, n, w)
            except Exception as e:  # a build or run failure is a finding, not a crash
                print(f"UNIT-FAIL  {args.unit} {variant} {backend}: {type(e).__name__}: {str(e)[:600]}")
                bad_any = True
                continue
            k = tot = masked = masked_eq = 0
            per_label = {}
            first = []
            census = {}
            for p in want:
                g = [int(x) for x in got[p]]
                r = rtl_int[p]
                for i in range(n):
                    why = reason[p][i]
                    if why:
                        masked += 1
                        census[why] = census.get(why, 0) + 1
                        masked_eq += int(g[i] == r[i])
                        continue
                    tot += 1
                    if g[i] != r[i]:
                        k += 1
                        lab = next(lb for lb, a, b in spans if a <= i < b)
                        per_label[lab] = per_label.get(lab, 0) + 1
                        if len(first) < 4:
                            first.append(f"{lab} cycle {i} {p}: allo {g[i]:x} rtl {r[i]:x}")
            tag = "UNIT-MATCH" if k == 0 else "UNIT-DIFF "
            cen = ", ".join(f"{a}={b}" for a, b in sorted(census.items()))
            print(f"{tag} {args.unit}:{inst} {variant} {backend} {tot - k}/{tot} defined "
                  f"({masked} masked: {cen}; allo==rtl on {masked_eq}) cycle={CYCLE_SEEN[backend]} "
                  f"(build {tb:.1f}s, run {time.time() - t - tb:.1f}s)", flush=True)
            if k:
                bad_any = True
                print("    differing defined slots by trace: "
                      + ", ".join(f"{lb}={c}" for lb, c in per_label.items()))
                for f in first:
                    print(f"    e.g. {f}")
    return 1 if bad_any else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit")
    ap.add_argument("--backend", action="append", choices=BACKENDS)
    ap.add_argument("--variant", action="append")
    ap.add_argument("--n", type=int, default=0, help="first n vectors only")
    ap.add_argument("--project", default="/tmp/minitpu_harness_prj")
    ap.add_argument("--inst", default=None, help="trace units: the RTL instance (default u.DEFAULT)")
    args = ap.parse_args(argv)

    u = importlib.import_module(f"examples.minitpu.units.{args.unit}")
    if u.RTL.shape == "trace":
        return check_trace(u, args)
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

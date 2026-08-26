# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness over the benchmark bed, swept over the BINDING axis.

    python -m benchmark.verify                    # trivial and exact-share
    python -m benchmark.verify --binding planned --scheduler exact
    python -m benchmark.verify -k gemm

The correctness half of the bed. `spec.py` says what a benchmark is, `report.py`
says what a variant COSTS, and this says whether it computes the right answer:
every variant cosims against the numpy `reference` its benchmark declares, so
the bed's own claim, that a schedule may change the hardware and never the
function, is the thing under test.

The axis is the BINDING, i.e. how many physical units the schedule is realized
on: the binding the scheduler implies, against the trivial control of one unit
per op, so a sharing bug that needs a real workload to expose is caught over
the whole bed rather than by another unit test. `--scheduler` is a scalar and
not a second axis on purpose: the two compose, and a sweep of the product
answers a question nobody asked.

Two things it reports beyond pass and fail, both because a green run can be
empty:

    rtl_sha  the emitted Verilog's hash. Two bindings that produce byte
             identical RTL checked ONE hardware twice, which is exactly what
             `planned` under the heuristic scheduler does, so the probe states
             how many variants it actually distinguished rather than letting a
             row count imply it.
    cycles   the measured cosim cycle count. `cosim` holds the kernel's
             published latency contract to it, failing a run that outlasts the
             figure, so a binding that moved the schedule shows up as a number
             rather than as a silent pass.

Each run is a subprocess for the reason `report.py` uses one: a solver that does
not terminate, an assert that fires and a simulator that dies are all results,
and only a process boundary survives all three.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark._child import run_child

REPO = Path(__file__).resolve().parents[1]
MARK = "@@VERIFY@@"


@dataclass(frozen=True)
class Knobs:
    """What one run is compiled and simulated under, off the command line."""

    scheduler: str
    seed: int
    cycles: int
    objective: str
    budget: float | None
    #: Multipliers on the device's per-resource scarcity prices, as sorted
    #: (name, factor) pairs. A per-run scheduling parameter, not device data.
    weights: tuple[tuple[str, float], ...] = ()


# What one cosim may run for, derived from the model rather than fixed: a design
# that hangs then aborts at a multiple of what it should have taken, instead of
# at whatever constant covers the biggest kernel in the bed.
_CYCLE_FLOOR = 40_000
_CYCLE_SLACK = 4
# A kernel whose span the model does not know (a dynamic trip count, an
# indeterminate body) has nothing to scale, so it gets a flat ceiling.
_CYCLE_UNKNOWN = 2_000_000


def _cycle_budget(latency: int | None) -> int:
    if latency is None:
        return _CYCLE_UNKNOWN
    return max(_CYCLE_FLOOR, _CYCLE_SLACK * latency)


def _mismatch(bench, args, expected) -> dict | None:
    """The first output argument that differs, or None if every one matches.

    Reported as how MANY elements differ and by how much, not as a bare failure:
    a shared datapath usually breaks one operator, so one wrong element in a
    thousand and a thousand in a thousand are different bugs and the probe
    should not need a re-run to tell them apart."""
    assert len(expected) == len(bench.outputs), "reference does not cover outputs"
    for idx, exp in zip(bench.outputs, expected):
        got, exp = np.asarray(args[idx]), np.asarray(exp)
        if bench.tolerance:
            rtol, atol = bench.tolerance
            bad = ~np.isclose(got, exp, rtol=rtol, atol=atol)
        else:
            bad = got != exp
        if not bad.any():
            continue
        # In float64 whatever the kernel's type: an unsigned difference wraps.
        diff = np.abs(got.astype(np.float64) - exp.astype(np.float64))
        at = np.unravel_index(int(diff.argmax()), diff.shape)
        return {
            "arg": idx,
            "bad": int(bad.sum()),
            "of": int(bad.size),
            "max_abs": float(diff.max()),
            "at": [int(i) for i in at],
        }
    return None


# --- one run -----------------------------------------------------------------


def verify_one(item, knobs: Knobs) -> dict:
    """Compile one variant under one binding, cosim it and compare.

    ``binding`` is the operator-sharing policy; ``knobs.cycles`` overrides the
    derived simulation budget; ``knobs.objective`` is the exact solver's
    optimization direction (the ``O`` knob)."""
    from allo.backend.rtl import LatencyModelWarning
    from benchmark.spec import find

    key, variant, binding = item
    bench = find(key)
    out: dict = {
        "key": key,
        "variant": variant,
        "binding": binding,
        "scheduler": knobs.scheduler,
        "objective": knobs.objective,
        "stage": "build",
        "status": "error",
    }
    t0 = time.time()
    if variant in bench.skip:
        out.update(status="skip", stage="skip", note=bench.skip[variant])
        return out

    try:
        parts = bench.build()
        sched = bench.schedules[variant](parts)
        solver = {"scheduler": knobs.scheduler, "O": knobs.objective}
        if knobs.budget is not None:
            solver["budget"] = knobs.budget
        if knobs.weights:
            solver["resource_weights"] = dict(knobs.weights)
        rtl = sched.export("rtl").set_scheduler_opt(**solver)
        if binding == "trivial":
            rtl.use_trivial_binding()

        out["stage"] = "schedule"
        fn = rtl.schedule().func(rtl.top)
        assert rtl.binding == binding, (
            f"binding follows the scheduler: {knobs.scheduler!r} implies "
            f"{rtl.binding!r}, not {binding!r}"
        )
        out["latency"] = fn.latency
        out["latency_exact"] = fn.latency_is_exact
        out["determinacy"] = fn.determinacy

        out["stage"] = "compile"
        rtl.compile()
        out["rtl_sha"] = hashlib.sha256(rtl.verilog.encode()).hexdigest()[:16]

        rng = np.random.default_rng(knobs.seed)
        args = bench.inputs(rng)
        # `reference` takes the kernel's arguments, output buffers included, so
        # it gets copies: one that accumulated into a buffer it was handed would
        # otherwise be scored against the DUT's own answer.
        expected = bench.reference(
            *(a.copy() if isinstance(a, np.ndarray) else a for a in args)
        )

        out["stage"] = "cosim"
        # A run that outlasts the published latency raises and lands in `error`
        # below; the pessimism warning would die with the child's stderr, so it
        # is carried in the row instead.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LatencyModelWarning)
            out["cycles"] = rtl.cosim(
                *args, timeout=knobs.cycles or _cycle_budget(fn.latency)
            ).cycles
        early = [w for w in caught if issubclass(w.category, LatencyModelWarning)]
        if early:
            out["latency_warn"] = " ".join(str(early[0].message).split())[:200]

        out["stage"] = "compare"
        bad = _mismatch(bench, args, expected)
        if bad is not None:
            out.update(status="wrong", mismatch=bad)
        else:
            out["status"] = "pass"
    except BaseException as e:  # a fired assert is a result, not a crash
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        out["seconds"] = round(time.time() - t0, 1)
    return out


def _run_child(item, knobs: Knobs, timeout: int) -> dict:
    key, variant, binding = item
    argv = [
        "--one",
        f"{key}::{variant}::{binding}",
        "--scheduler",
        knobs.scheduler,
        "--seed",
        str(knobs.seed),
        "--cycles",
        str(knobs.cycles),
        "--objective",
        knobs.objective,
    ]
    if knobs.budget is not None:
        argv += ["--budget", str(knobs.budget)]
    for _n, _v in knobs.weights:
        argv += ["--weight", f"{_n}={_v}"]
    t0 = time.time()
    base = {
        "key": key,
        "variant": variant,
        "binding": binding,
        "scheduler": knobs.scheduler,
    }
    row, _ = run_child("benchmark.verify", MARK, argv, timeout, base)
    if row["status"] in ("timeout", "crash"):
        row.update(stage="?", seconds=round(time.time() - t0, 1))
    return row


# --- tables ------------------------------------------------------------------


def _key_of(r) -> str:
    return f"{r['key']}/{r['variant']}"


_COLS = [("status", "status", 7), ("cycles", "cycles", 10), ("bad", "bad", 7)]
_GROUP = sum(w for _, _, w in _COLS) + len(_COLS)


def variant_table(results: list[dict], bindings: list[str]) -> str:
    """One row per variant, one column group per binding.

    `bad` is how many output elements differed, so a wrong row says whether one
    operator or the whole datapath is broken."""
    by: dict = {}
    for r in results:
        by.setdefault(_key_of(r), {})[r["binding"]] = r

    top = f"{'':<34}" + "".join(f"  {('[' + b + ']').center(_GROUP)}" for b in bindings)
    head = f"{'benchmark/variant':<34}" + "".join(
        "  " + "".join(" " + label.rjust(w) for _, label, w in _COLS) for _ in bindings
    )
    lines = [top, head, "-" * len(head)]
    for name in sorted(by):
        row = f"{name:<34}"
        for b in bindings:
            r = by[name][b]  # every (variant, binding) pair is one run
            row += "  "
            for field, _, w in _COLS:
                if field == "bad":
                    v = r.get("mismatch", {}).get("bad", "-")
                else:
                    v = r.get(field, "-")
                row += " " + str(v).rjust(w)
        lines.append(row)
    return "\n".join(lines)


def coverage_note(results: list[dict], bindings: list[str]) -> str:
    """How much of the sweep was a SECOND hardware rather than the same one
    again. A binding that folds nothing emits the RTL the trivial binding
    already emitted, and those rows cost a cosim without guarding anything."""
    if len(bindings) < 2:
        return ""
    shas: dict = {}
    for r in results:
        if r.get("rtl_sha"):
            shas.setdefault(_key_of(r), {})[r["binding"]] = r["rtl_sha"]
    full = [m for m in shas.values() if len(m) == len(bindings)]
    same = [m for m in full if len(set(m.values())) == 1]
    return (
        f"{len(full) - len(same)} of {len(full)} variants emitted DIFFERENT RTL "
        f"across {bindings}; the other {len(same)} checked one hardware "
        f"{len(bindings)} times"
    )


def oracle_note(results: list[dict]) -> str:
    """How many runs were held to a published latency, and every run that beat
    its own contract. A kernel whose span is data-dependent publishes no
    figure, so its row rests on the functional comparison alone."""
    ran = [r for r in results if r.get("cycles") is not None]
    if not ran:
        return ""
    held = [
        r
        for r in ran
        if r.get("latency") is not None and r.get("determinacy") != "concurrent"
    ]
    lines = [
        f"{len(held)} of {len(ran)} cosim runs were held to a published latency; "
        f"the other {len(ran) - len(held)} publish no static span to hold"
    ]
    for r in ran:
        if r.get("latency_warn"):
            lines.append(f"  early: {_key_of(r)} [{r['binding']}] {r['latency_warn']}")
    return "\n".join(lines)


def failure_report(results: list[dict]) -> str:
    """Every run that did not pass, with what it was doing when it stopped."""
    bad = [r for r in results if r["status"] not in ("pass", "skip")]
    if not bad:
        return ""
    lines = [f"{len(bad)} failing run(s):", ""]
    for r in sorted(bad, key=lambda r: (_key_of(r), r["binding"])):
        detail = r.get("error") or json.dumps(r.get("mismatch", {}))
        # Flattened rather than cut at the first newline: an emitter diagnostic
        # arrives wrapped under a generic first line, so a line-1 excerpt says
        # only that something failed.
        lines.append(
            f"{_key_of(r)} [{r['binding']}] {r['status']} at {r['stage']}: "
            + " ".join(detail.split())[:220]
        )
    return "\n".join(lines)


# --- driver ------------------------------------------------------------------


def _weights(spec: list[str]) -> tuple[tuple[str, float], ...]:
    """`["lut=2"]` as sorted (name, factor) pairs. Sorted so two runs asking for
    the same reweighting build the same knobs whatever order they were typed."""
    pairs = []
    for s in spec:
        name, _, value = s.partition("=")
        pairs.append((name.strip(), float(value)))
    return tuple(sorted(pairs))


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--one", help=argparse.SUPPRESS)  # the child entry point
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("-k", "--filter", default="", help="substring of suite/name")
    ap.add_argument(
        "--binding",
        default="trivial,exact-share",
        help="comma-separated axis. The binding follows the scheduler "
        "('exact-share' under the heuristic, 'planned' under an exact one); "
        "'trivial' (one unit per op) is the dev control that joins either",
    )
    ap.add_argument(
        "--scheduler",
        default="heuristic",
        help="the solver each binding is realized on. A scalar, not an axis",
    )
    ap.add_argument(
        "-O",
        "--objective",
        default="cycles",
        help="the exact solver's optimization direction (the O knob); the "
        "heuristic ignores it",
    )
    ap.add_argument("--seed", type=int, default=0, help="the input generator's seed")
    ap.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="simulation budget per run; 0 derives it from the modelled latency",
    )
    ap.add_argument(
        "--budget",
        type=float,
        help="what one exact solve may spend, in the solver's deterministic "
        "time units; unset keeps the compiler default",
    )
    ap.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="multiply one resource's scarcity price, e.g. --weight lut=2. "
        "Repeatable; the axis the area currency is swept over",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="wall seconds per run")
    ap.add_argument("-o", "--out", default="verify.json")
    args = ap.parse_args()
    knobs = Knobs(
        scheduler=args.scheduler,
        seed=args.seed,
        cycles=args.cycles,
        objective=args.objective,
        budget=args.budget,
        weights=_weights(args.weight),
    )

    if args.one:
        print(MARK + json.dumps(verify_one(args.one.split("::"), knobs)))
        return

    sys.path.insert(0, str(REPO))
    from benchmark.spec import discover

    if shutil.which("verilator") is None:
        raise SystemExit("cosim needs verilator on PATH")
    bindings = [b for b in args.binding.split(",") if b]
    if not bindings:
        raise SystemExit("no binding to run")
    derived = "planned" if args.scheduler.startswith("exact") else "exact-share"
    bad = sorted(set(bindings) - {"trivial", derived})
    if bad:
        raise SystemExit(
            f"binding follows the scheduler: under '{args.scheduler}' the axis "
            f"may hold 'trivial' and '{derived}', not {bad}"
        )

    work = [
        (b.key, v, binding)
        for b in discover()
        for v in b.schedules
        for binding in bindings
        if args.filter in b.key
    ]
    print(
        f"{len(work)} runs, {args.jobs} jobs, scheduler={args.scheduler},"
        f" bindings={bindings}",
        flush=True,
    )

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_child, w, knobs, args.timeout) for w in work]
        for f in futs:
            r = f.result()
            results.append(r)
            done += 1
            tag = {"pass": "ok", "skip": "--"}.get(r["status"], r["status"].upper())
            print(
                f"[{done:3d}/{len(work)}] {tag:>8}  {_key_of(r)}"
                f" [{r['binding']}]  {r.get('seconds', 0)}s",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}\n")
    print(variant_table(results, bindings))
    for note in (coverage_note(results, bindings), oracle_note(results)):
        if note:
            print("\n" + note)

    tally: dict = {}
    for r in results:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    report = failure_report(results)
    if report:
        print("\n" + report)
        sys.exit(1)


if __name__ == "__main__":
    main()

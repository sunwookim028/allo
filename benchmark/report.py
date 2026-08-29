# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""QoR over the benchmark bed: one command, one table, two schedulers.

    python -m benchmark.report                    # both schedulers, compile stage
    python -m benchmark.report --per-region       # the region-level dump
    python -m benchmark.report --compare base.json

This is the measurement half of the bed. `spec.py` says what a benchmark is and
`verify.py` answers whether a variant is CORRECT; this answers what it COSTS,
which is the question a scheduling-model change has to be argued from.

What it reports, and why each number rather than a neighbouring one:

    latency      the kernel's published span, and whether it is EXACT. Only an
                 exact one may be compared against hardware, so a variant whose
                 latency is a bound or whose kernel is indeterminate is carried
                 but never summed into a headline.
    ii           per region, what the solver decided.
    length       the schedule DEPTH: the cycle by which every op has completed.
    drain        the TERMINAL cycle, the last issue pulse to the deepest output
                 committing. Reported beside `length` rather than instead of it
                 because a span composes off `drain` and the two differ by
                 whatever slack the solver left above the last commit, which is
                 a scheduling decision worth seeing.
    reg bits     flip-flops the design holds, split into the activation pulse
                 chains, the value delay chains and everything else. A COUNT off
                 the emitter's own ledger, and the split is the role each
                 register was BUILT for rather than a guess from its name. Bits,
                 not registers: a narrower counter is fewer bits on the same
                 declaration, which is the one axis a schedule can move.
    solve ms     wall time of each region's solve, from the scheduler itself.
                 Per REGION, because a whole-compile figure cannot tell a model
                 change that cost 2 ms everywhere from one that cost 4 s once.

Each (benchmark, variant, scheduler) runs in its own subprocess, for the reason
the cosim probe does: a scheduler that does not terminate, an assert that fires
and a solver that runs away all have to be survivable, and only a process
boundary survives all three.

NOT a correctness suite. It stops at `compile`, so nothing here says a variant
computes the right answer; that is `verify.py`'s job and the two should not be
conflated again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from benchmark._child import run_child

REPO = Path(__file__).resolve().parents[1]
MARK = "@@QOR@@"


@dataclass(frozen=True)
class Knobs:
    """What one run is measured under, off the command line. The scheduler is
    not here: it is the axis, so it travels with the case."""

    stage: str
    binding: str
    objective: str
    freq: float | None
    budget: float | None
    workers: int | None
    deterministic: bool
    area_slack: float
    #: Multipliers on the device's per-resource scarcity prices, as sorted
    #: (name, factor) pairs. A per-run scheduling parameter, not device data.
    weights: tuple[tuple[str, float], ...] = ()


# --- what the emitter built --------------------------------------------------


def registers(microarch) -> dict:
    """Flip-flops the design holds, split by the role they were BUILT for.

    A count and not a reading: every register passes one line of the emitter,
    which charges the ledger this reads. Bits, not registers: a narrower counter
    is fewer bits on the same declaration, which is the one axis a schedule can
    move."""
    by_role = Counter()
    runs = 0
    for f in microarch.funcs:
        runs += sum(c.count for c in f.regs)
        for role, bits in f.reg_bits_by_role().items():
            by_role[role.value] += bits
    return {
        "reg_bits": microarch.reg_bits,
        "reg_runs": runs,
        "pulse_bits": by_role["pulse"] + by_role["counted"],
        "delay_bits": by_role["value"],
        "reg_bits_by_role": dict(by_role),
    }


def allocation(microarch) -> list[dict]:
    """Per region, what the binding cost: the operations it bound, the units it
    built for them, and the interconnect sharing grew."""
    return [
        {
            "func": f.func,
            "order": r.order,
            "ops": r.compute_ops,
            "units": len(r.units),
            "ip": len([u for u in r.units if not u.comb]),
            "muxes": sum(m.count for m in r.muxes),
            "mux_inputs": r.cost.mux_inputs,
            "mux_bits": r.cost.mux_bits,
        }
        for f in microarch.funcs
        for r in f.regions
    ]


def memories(microarch) -> list[dict]:
    """Per array, the ports it was given and where they came from. A second
    write port defeats RAM inference unless the schedule proved the two never
    collide, so writers spread over regions are paying for a concurrency that
    region ordering rules out."""
    return [
        {
            "func": f.func,
            "owner": m.owner,
            "depth": m.depth_words,
            "width": m.width,
            "banks": m.banks,
            "writes": m.writes,
            "storage": m.storage,
            # What it was built as, which is not the row: a boundary holds no
            # cells here, a ROM is logic and a complete partition is registers.
            "realization": m.realization,
            "instances": m.cost.instances,
            # Copies the schedule reserved against; `instances` past it is read
            # bandwidth no cycle was cut for.
            "copies_budget": m.cost.copies_budget,
            "from_calls": m.cost.call_writes,
            # Which side of the array's reads the buses were coloured for: this
            # module's own accesses, or the ports its children master.
            "reads": m.reads,
            "call_reads": m.cost.call_reads,
            "write_ports": m.cost.write_ports,
            "read_ports": m.cost.read_ports,
            "ports": m.cost.ports,
            # What one cycle asks of one bank, against the ports built for it.
            # Every port past the first that a read holds is a further copy of
            # the whole array.
            "read_concurrency": m.cost.read_concurrency,
            "write_concurrency": m.cost.write_concurrency,
            "boundary_ports": m.cost.boundary_ports,
            "external": m.external,
        }
        for f in microarch.funcs
        for m in f.mems
    ]


def area_of(q) -> dict:
    """One QoR estimate as a JSON row, split by what spends each resource."""
    lut = {k: u.lut for k, u in q.by_kind.items()}
    return {
        "lut": q.area.lut,
        "srl": q.area.srl,
        "ff": q.area.ff,
        "dsp": q.area.dsp,
        "carry8": q.area.carry8,
        "bram36": q.area.bram36,
        "uram288": q.area.uram288,
        "unit_lut": lut.get("units", 0),
        "mux_lut": lut.get("muxes", 0),
        "reg_lut": lut.get("regs", 0),
        "mem_lut": lut.get("memories", 0),
        "control_lut": lut.get("control", 0),
        "reg_ff": sum(u.ff for k, u in q.by_kind.items() if k == "regs"),
        # What the design DECLARES, which is also what the scheduling objective
        # charges for the same registers.
        "reg_bits": q.reg_bits,
        "mem_bits": q.mem_bits,
        "unmodelled": q.unmodelled,
        "counted": sorted(q.counted),
        # Resources the design asks for more of than the part has, per run and
        # never summed.
        "over_capacity": {k: round(v, 3) for k, v in q.over_capacity.items()},
    }


# --- one run -----------------------------------------------------------------

# The scheduler's own report that a region's II exceeds the bound its LP
# justifies, which is the one place the compiler admits it may have lost. It is
# the HEURISTIC's warning, so it fires under `scheduler="exact"` too, which runs
# the heuristic as its warm start; the count is the same in both columns by
# construction and describes the problem rather than the solver.
_II_GAP = re.compile(r"Scheduled at II=(\d+) against a lower bound of II=(\d+)")
# How many `memref` accesses were raised into the dependence test's reach. Needs
# ALLO_LOG_LEVEL=info and is absent otherwise; it prices what a better raise
# could still recover.
_RAISED = re.compile(r"Raised (\d+) loop\(s\) and (\d+) further memref access")


def measure_one(item, knobs: Knobs) -> dict:
    """Schedule (and by default compile) one variant, returning its metrics.

    ``freq`` overrides the device's default clock (MHz), i.e. the period the
    chaining half of every problem is cut against. ``budget`` overrides what one
    exact solve may spend, in deterministic time units. ``workers`` overrides how
    many search workers one exact solve runs; ``deterministic`` off lets them
    race. ``binding`` is ``"trivial"`` (one
    unit per op) or ``"auto"``, the binding the scheduler implies; the recorded
    row carries the resolved name. ``objective`` is the ``O`` knob (``"cycles"``
    or a period policy); the heuristic ignores it. ``area_slack`` is the
    fraction of the minimal span the area solve may trade for a smaller
    design."""
    from benchmark.spec import find

    key, variant, scheduler = item
    bench = find(key)
    out: dict = {
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
        "freq_mhz": knobs.freq,
        "budget": knobs.budget,
        "workers": knobs.workers,
        "deterministic": knobs.deterministic,
        "binding": knobs.binding,
        "objective": knobs.objective,
        "area_slack": knobs.area_slack,
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

        out["stage"] = "schedule"
        assert knobs.binding in ("trivial", "auto"), knobs.binding
        opts = {}
        if knobs.freq is not None:
            opts["freq_mhz"] = knobs.freq
        solver = {
            "scheduler": scheduler,
            "O": knobs.objective,
            "area_slack": knobs.area_slack,
            "deterministic": knobs.deterministic,
        }
        if knobs.budget is not None:
            solver["budget"] = knobs.budget
        if knobs.workers is not None:
            solver["workers"] = knobs.workers
        if knobs.weights:
            solver["resource_weights"] = dict(knobs.weights)
        rtl = sched.export("rtl", **opts).set_scheduler_opt(**solver)
        if knobs.binding == "trivial":
            rtl.use_trivial_binding()
        t1 = time.time()
        res = rtl.schedule()
        out["binding"] = rtl.binding
        out["schedule_s"] = round(time.time() - t1, 2)

        fn = res.func(rtl.top)
        out["latency"] = fn.latency
        out["latency_exact"] = fn.latency_is_exact
        out["determinacy"] = fn.determinacy
        # Every func, not just the top: a sub-kernel's regions are as much of
        # the hardware, and a schedule change may land entirely in one.
        out["regions"] = [
            {
                "func": f.name,
                "order": r.order,
                "nesting": r.depth,
                "kind": str(r.kind.value),
                "container": r.container,
                "ii": r.interval,
                "length": r.iteration_latency,
                "drain": r.cost.drain,
                "trip": r.trip_count,
                "latency": r.latency,
                "ops": len(r.ops),
            }
            for f in res.funcs
            for r in f.regions
        ]
        out["solves"] = [
            {
                "func": s.func,
                "where": s.where,
                "kind": s.kind,
                "ops": s.ops,
                "limited_ops": s.limited_ops,
                "ii": s.interval,
                "ms": round(s.ms, 2),
                "solver": s.solver,
                "proven": s.proven,
                "span_proven": s.span_proven,
                "exhausted": s.budget_exhausted,
                "fallback": s.fallback,
                "area": s.model_area,
                "area_bound": s.model_area_bound,
            }
            for s in res.compiler.solves
        ]
        out["solve_ms"] = round(sum(s.ms for s in res.compiler.solves), 1)
        out["solve_ms_max"] = round(
            max((s.ms for s in res.compiler.solves), default=0.0), 1
        )
        out["ops_max"] = max((s.ops for s in res.compiler.solves), default=0)
        # The dependence analysis's residue: accesses outside the polyhedral
        # test's reach, and pairs it accepted but could not decide.
        out["dep_residual"] = [
            sum(x.conservative_accesses for x in res.compiler.dependence),
            sum(x.undecided_pairs for x in res.compiler.dependence),
        ]

        if knobs.stage != "schedule":
            out["stage"] = "compile"
            t1 = time.time()
            rtl.compile()
            out["compile_s"] = round(time.time() - t1, 2)
            # What the emitter BUILT, off its own report rather than off the
            # emitted text: the register ledger, the per-region allocation, the
            # storage each array was given, and what all of it prices at.
            uarch = rtl.microarch
            qor = rtl.estimation
            out["area"] = area_of(qor)
            # The modelled clock beside the target it was cut to, plus the
            # worst paths that clock comes from.
            out["fmax"] = round(qor.fmax, 1)
            out["fmax_target"] = round(qor.fmax_target, 1)
            out["critical_paths"] = [
                {
                    "total_ns": round(p.total, 3),
                    "slack_ns": round(p.slack, 3),
                    "endpoint": p.endpoint,
                    "where": p.where,
                    "steps": [
                        {"what": s.what, "ns": round(s.delay, 3)} for s in p.steps
                    ],
                }
                for p in qor.critical_paths[:3]
            ]
            out.update(registers(uarch))
            out["alloc"] = allocation(uarch)
            out["mem_ports"] = memories(uarch)
            for f in ("ops", "units", "muxes", "mux_bits"):
                out[f"alloc_{f}"] = sum(a[f] for a in out["alloc"])
            out["ip_units"] = sum(a["ip"] for a in out["alloc"])
            verilog = rtl.verilog
            out["verilog_lines"] = verilog.count("\n")
            # The RTL itself, so a re-run can be checked for BYTE identity
            # rather than for metrics that agree. Determinism is a property of
            # the emitted hardware, and two schedules can differ while every
            # figure below matches.
            out["verilog_sha"] = hashlib.sha256(verilog.encode()).hexdigest()[:16]

        out["status"] = "pass"
    except BaseException as e:  # a fired assert is a result, not a crash
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        out["seconds"] = round(time.time() - t0, 1)
    return out


def _run_child(item, knobs: Knobs, timeout: int) -> dict:
    key, variant, scheduler = item
    argv = [
        "--one",
        f"{key}::{variant}::{scheduler}",
        "--stage",
        knobs.stage,
        "--binding",
        knobs.binding,
        "--objective",
        knobs.objective,
    ]
    if knobs.freq is not None:
        argv += ["--freq", str(knobs.freq)]
    if knobs.budget is not None:
        argv += ["--budget", str(knobs.budget)]
    if knobs.workers is not None:
        argv += ["--workers", str(knobs.workers)]
    if not knobs.deterministic:
        argv += ["--nondeterministic"]
    if knobs.area_slack:
        argv += ["--area-slack", str(knobs.area_slack)]
    for _n, _v in knobs.weights:
        argv += ["--weight", f"{_n}={_v}"]
    t0 = time.time()
    base = {"key": key, "variant": variant, "scheduler": scheduler}
    d, text = run_child("benchmark.report", MARK, argv, timeout, base)
    if d["status"] in ("timeout", "crash"):
        d.update(stage="?", seconds=round(time.time() - t0, 1))
        return d
    # The II-vs-bound warnings, which no field of the schedule result carries:
    # the bound is settled inside the simplex and reported only as a diagnostic.
    d["ii_gaps"] = [{"ii": int(a), "bound": int(b)} for a, b in _II_GAP.findall(text)]
    d["budget_exhausted"] = sum(1 for s in d.get("solves", []) if s.get("exhausted"))
    d["raised"] = [sum(int(m[i]) for m in _RAISED.findall(text)) for i in (0, 1)]
    d["warnings"] = [l.strip()[:300] for l in text.splitlines() if "WARN" in l][:20]
    return d


# --- tables ------------------------------------------------------------------


def _fmt(v, width, prec=None):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{prec or 1}f}".rjust(width)
    return str(v).rjust(width)


def _key_of(r) -> str:
    return f"{r['key']}/{r['variant']}"


# Per-scheduler columns. `gaps` counts the regions whose achieved II exceeded
# the bound the LP justifies, the one figure the compiler publishes about its
# own possible loss; `bdgt` counts the exact solves that shipped an unproven
# placement, which is the one way an exact run can be WORSE than a heuristic one.
_COLS = [
    ("latency", "latency", 10),
    ("reg_bits", "regFF", 7),
    ("delay_bits", "dlyFF", 7),
    ("gaps", "gaps", 5),
    ("budget_exhausted", "bdgt", 5),
    ("solve_ms", "solve_ms", 9),
]
_GROUP = sum(w for _, _, w in _COLS) + len(_COLS)


def variant_table(results: list[dict], schedulers: list[str]) -> str:
    """One row per variant, one column group per scheduler."""
    by = {}
    for r in results:
        by.setdefault(_key_of(r), {})[r["scheduler"]] = r

    top = f"{'':<34}" + "".join(
        f"  {('[' + s + ']').center(_GROUP)}" for s in schedulers
    )
    head = f"{'benchmark/variant':<34}" + "".join(
        "  " + "".join(" " + label.rjust(w) for _, label, w in _COLS)
        for _ in schedulers
    )
    lines = [top, head, "-" * len(head)]
    for name in sorted(by):
        row = f"{name:<34}"
        for s in schedulers:
            r = by[name].get(s)
            row += "  "
            if r is None or r["status"] != "pass":
                row += ((r or {}).get("status", "-")).center(_GROUP)
                continue
            for field, _, w in _COLS:
                v = len(r.get("ii_gaps", [])) if field == "gaps" else r.get(field)
                # A latency that is not exact is parenthesized: it is an upper
                # bound, so it may not be differenced against another run's.
                if field == "latency" and v is not None and not r.get("latency_exact"):
                    row += " " + f"({v})".rjust(w)
                else:
                    row += " " + _fmt(v, w)
        lines.append(row)
    return "\n".join(lines)


def region_table(results: list[dict]) -> str:
    """One row per region, for the runs that reached a schedule."""
    head = (
        f"{'benchmark/variant':<30} {'sched':<5} {'func':<18} {'#':>3} {'kind':<8}"
        f" {'ii':>5} {'len':>6} {'drain':>6} {'trip':>7} {'lat':>9} {'ops':>5}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        for g in r.get("regions", []):
            lines.append(
                f"{_key_of(r):<30} {r['scheduler'][:5]:<5} {g['func'][:18]:<18}"
                f" {g['order']:>3} {g['kind']:<8}"
                f" {_fmt(g['ii'], 5)} {_fmt(g['length'], 6)} {_fmt(g['drain'], 6)}"
                f" {_fmt(g['trip'], 7)} {_fmt(g['latency'], 9)} {g['ops']:>5}"
            )
    return "\n".join(lines)


def solve_table(results: list[dict], top: int) -> str:
    """The slowest solves, which is what a compile-time regression shows up in."""
    rows = [
        (s["ms"], r["scheduler"], _key_of(r), s)
        for r in results
        for s in r.get("solves", [])
    ]
    rows.sort(reverse=True, key=lambda t: t[0])
    head = (
        f"{'ms':>9} {'sched':<5} {'benchmark/variant':<30} {'kind':<8}"
        f" {'ops':>5} {'lim':>5} {'ii':>5} {'st':<4}  where"
    )

    # The solver's verdict: proven optimal, ran out of budget (so the result
    # may not reproduce), or fell back to the heuristic's schedule.
    def verdict(s):
        if s.get("fallback"):
            return "fell"
        if s.get("exhausted"):
            return "bdgt"
        if s.get("proven"):
            return "opt"
        return "-"

    lines = [head, "-" * len(head)]
    for ms, sched, name, s in rows[:top]:
        lines.append(
            f"{ms:>9.1f} {sched[:5]:<5} {name:<30} {s['kind']:<8}"
            f" {s['ops']:>5} {s['limited_ops']:>5} {_fmt(s['ii'], 5)}"
            f" {verdict(s):<4}  {s['where']}"
        )
    return "\n".join(lines)


def alloc_table(results: list[dict]) -> str:
    """Per variant, how many physical units the schedule was realized on and
    what interconnect that took.

    `ops/unit` is the sharing ratio, 1.00 under the trivial binding. `muxFF` is
    the 2:1-mux bit count that sharing cost."""
    rows = [r for r in results if r.get("alloc")]
    if not rows:
        return "no allocation data (needs --stage compile)"
    head = (
        f"{'benchmark/variant':<34} {'sched':<6} {'regions':>7} {'ops':>7}"
        f" {'units':>7} {'ops/unit':>9} {'muxes':>7} {'muxFF':>9}"
    )
    lines = [head, "-" * len(head)]
    tot = dict.fromkeys(("ops", "units", "muxes", "mux_bits"), 0)
    for r in sorted(rows, key=lambda r: -r.get("alloc_mux_bits", 0)):
        o, u = r["alloc_ops"], max(r["alloc_units"], 1)
        for f in tot:
            tot[f] += r[f"alloc_{f}"]
        lines.append(
            f"{_key_of(r):<34} {r['scheduler'][:6]:<6} {len(r['alloc']):>7}"
            f" {o:>7} {u:>7} {o / u:>9.2f} {r['alloc_muxes']:>7}"
            f" {r['alloc_mux_bits']:>9}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<34} {'':<6} {'':>7} {tot['ops']:>7} {tot['units']:>7}"
        f" {tot['ops'] / max(tot['units'], 1):>9.2f} {tot['muxes']:>7}"
        f" {tot['mux_bits']:>9}"
    )
    return "\n".join(lines)


def area_table(results: list[dict]) -> str:
    """Per variant, what the emitted structures price at against the device.

    `untLUT`/`muxLUT`/`ctlLUT`/`memLUT` split the LUT total by what spends it,
    which is the split an allocation objective trades along: a fold removes a
    unit and grows the muxes feeding the one it folded onto. `SRL` is the delay
    chains, which is where the register term's cost ACTUALLY lands; `regFF`
    beside `decFF`, the flip-flops the design DECLARES, is what those chains cost
    against what the objective charges for them. Memory is carried apart and
    never summed in."""
    rows = [r for r in results if r.get("area")]
    if not rows:
        return "no area data (needs --stage compile)"
    head = (
        f"{'benchmark/variant':<34} {'sched':<6} {'LUT':>8} {'untLUT':>8}"
        f" {'muxLUT':>8} {'ctlLUT':>8} {'memLUT':>8} {'SRL':>6} {'DSP':>5}"
        f" {'BRAM':>5} {'URAM':>5} {'regFF':>8} {'decFF':>8} {'ramKb':>7}"
    )
    lines = [head, "-" * len(head)]
    tot = dict.fromkeys(
        (
            "lut",
            "unit_lut",
            "mux_lut",
            "control_lut",
            "mem_lut",
            "srl",
            "dsp",
            "bram36",
            "uram288",
            "reg_ff",
            "reg_bits",
            "mem_bits",
        ),
        0,
    )
    unmodelled: dict[str, int] = {}
    over_capacity: dict[str, str] = {}
    for r in sorted(rows, key=lambda r: -r["area"]["lut"]):
        a = r["area"]
        for f in tot:
            tot[f] += a[f]
        for k, n in a.get("unmodelled", {}).items():
            unmodelled[k] = unmodelled.get(k, 0) + n
        if a.get("over_capacity"):
            over_capacity[f"{_key_of(r)} {r['scheduler'][:6]}"] = ", ".join(
                f"{k} {v:.2f}x" for k, v in sorted(a["over_capacity"].items())
            )
        lines.append(
            f"{_key_of(r):<34} {r['scheduler'][:6]:<6} {a['lut']:>8}"
            f" {a['unit_lut']:>8} {a['mux_lut']:>8} {a['control_lut']:>8}"
            f" {a['mem_lut']:>8} {a['srl']:>6} {a['dsp']:>5}"
            f" {a.get('bram36', 0):>5} {a.get('uram288', 0):>5}"
            f" {a['reg_ff']:>8} {a['reg_bits']:>8} {a['mem_bits'] / 1024:>7.1f}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<34} {'':<6} {tot['lut']:>8} {tot['unit_lut']:>8}"
        f" {tot['mux_lut']:>8} {tot['control_lut']:>8} {tot['mem_lut']:>8}"
        f" {tot['srl']:>6} {tot['dsp']:>5} {tot['bram36']:>5}"
        f" {tot['uram288']:>5} {tot['reg_ff']:>8}"
        f" {tot['reg_bits']:>8} {tot['mem_bits'] / 1024:>7.1f}"
    )
    # The ratio below is the shift-register discount the device row models; the
    # scheduling objective charges `chainPrice`, not this bit count.
    over = tot["reg_bits"] / max(tot["reg_ff"], 1)
    lines.append("")
    lines.append(
        f"chains declare {tot['reg_bits']} register bits, held in "
        f"{tot['reg_ff']} FF + {tot['srl']} SRL: {over:.1f}x extracted"
    )
    lines.append(
        f"storage costs {tot['mem_lut']} LUTs, {tot['bram36']} BRAM and "
        f"{tot['uram288']} URAM"
    )
    if over_capacity:
        lines.append(
            f"OVER THE PART ({len(over_capacity)} run(s) do not fit): "
            + ", ".join(f"{k} {v}" for k, v in sorted(over_capacity.items()))
        )
    if unmodelled:
        lines.append(f"UNMODELLED (scored as zero): {unmodelled}")
    return "\n".join(lines)


def compare_table(base: list[dict], new: list[dict]) -> str:
    """What moved between two runs.

    Only exact latencies are differenced: a bound may move because the
    assumption behind it moved, which is not a schedule getting better."""
    fields = [("latency", "latency"), ("reg_bits", "regFF"), ("solve_ms", "solve_ms")]
    index = lambda rs: {(_key_of(r), r["scheduler"]): r for r in rs}
    b, n = index(base), index(new)
    head = f"{'benchmark/variant':<30} {'sched':<6}"
    for _, label in fields:
        head += f" {label + ' base':>13} {label + ' new':>13} {'delta':>9}"
    lines = [head, "-" * len(head)]
    moved = 0
    for k in sorted(set(b) & set(n)):
        rb, rn = b[k], n[k]
        if rb["status"] != "pass" or rn["status"] != "pass":
            continue
        cells, changed = "", False
        for field, _ in fields:
            vb, vn = rb.get(field), rn.get(field)
            if field == "latency" and not (
                rb.get("latency_exact") and rn.get("latency_exact")
            ):
                vb = vn = None
            if vb is None or vn is None:
                cells += f" {'-':>13} {'-':>13} {'-':>9}"
                continue
            d = vn - vb
            # Solve time is wall time, so a small move is noise; a schedule
            # figure is exact and any move is real.
            if d and (field != "solve_ms" or abs(d) > 0.2 * max(vb, 1)):
                changed = True
            cells += f" {_fmt(vb, 13)} {_fmt(vn, 13)} {_fmt(d, 9)}"
        if changed:
            moved += 1
            lines.append(f"{k[0]:<30} {k[1][:6]:<6}{cells}")
    only = (set(b) ^ set(n)) or None
    lines.append("")
    lines.append(f"{moved} of {len(set(b) & set(n))} runs moved")
    if only:
        lines.append(f"present in only one run: {sorted(x[0] for x in only)}")
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
        "--stage",
        default="compile",
        choices=("schedule", "compile"),
        help="`schedule` is fast and has no register counts",
    )
    ap.add_argument(
        "--scheduler",
        default="heuristic,exact",
        help="comma-separated axis of solver kinds",
    )
    ap.add_argument(
        "--freq",
        type=float,
        help="target clock (MHz), overriding the device default. The period is "
        "what chaining is cut against, so this is the axis a chaining change "
        "is swept over",
    )
    ap.add_argument(
        "--budget",
        type=float,
        help="what one exact solve may spend, in the solver's "
        "deterministic time units (default 10). The axis a budget policy is "
        "swept over; it does nothing to the heuristic",
    )
    ap.add_argument(
        "--binding",
        default="trivial",
        help="'trivial' (the default, one unit per op) or 'auto', the binding "
        "the scheduler implies: 'exact-share' under the heuristic, 'planned' "
        "under an exact one",
    )
    ap.add_argument(
        "--workers",
        type=int,
        help="search workers one exact solve runs (default 1). Above one the "
        "portfolio is interleaved, so the deterministic budget still bounds a "
        "deterministic search; the axis a parallel-solve change is swept over",
    )
    ap.add_argument(
        "--nondeterministic",
        action="store_true",
        help="let the exact solve's workers race instead of interleaving (the "
        "deterministic knob off), each under budget/workers wall seconds: the "
        "budget's core-seconds in a fraction of the wall, but no exact solve "
        "is reproducible, so verilog_sha stops being an oracle",
    )
    ap.add_argument(
        "-O",
        "--objective",
        default="cycles",
        help="the O knob: 'cycles' (minimize span, then area under it) or the "
        "'freq'/'wall' period policies",
    )
    ap.add_argument(
        "--area-slack",
        type=float,
        default=0.0,
        help="fraction of the minimal span the area solve may trade for a "
        "smaller design (the area_slack knob); 0 keeps the tightest span",
    )
    ap.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="multiply one resource's scarcity price, e.g. --weight lut=2. "
        "Repeatable; the axis the area currency is swept over",
    )
    ap.add_argument("--timeout", type=int, default=900, help="wall seconds per run")
    ap.add_argument("-o", "--out", default="qor.json")
    ap.add_argument("--per-region", action="store_true")
    ap.add_argument("--solves", type=int, default=0, metavar="N", help="slowest N")
    ap.add_argument(
        "--alloc",
        action="store_true",
        help="the per-variant allocation (units, muxes)",
    )
    ap.add_argument(
        "--area",
        action="store_true",
        help="what the emitted structures price at against the measured device "
        "tables, split by what spends them. The scoreboard an allocation change "
        "is argued from",
    )
    ap.add_argument("--compare", metavar="BASE.json", help="diff against a saved run")
    args = ap.parse_args()
    knobs = Knobs(
        stage=args.stage,
        binding=args.binding,
        objective=args.objective,
        freq=args.freq,
        budget=args.budget,
        workers=args.workers,
        deterministic=not args.nondeterministic,
        area_slack=args.area_slack,
        weights=_weights(args.weight),
    )

    if args.one:
        print(MARK + json.dumps(measure_one(args.one.split("::"), knobs)))
        return

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        new = json.loads(Path(args.out).read_text())
        print(compare_table(base, new))
        return

    sys.path.insert(0, str(REPO))
    from benchmark.spec import discover

    schedulers = [s for s in args.scheduler.split(",") if s]
    if not schedulers:
        raise SystemExit("no scheduler to run")

    work = [
        (b.key, v, s)
        for b in discover()
        for v in b.schedules
        for s in schedulers
        if args.filter in b.key
    ]
    clock = f", freq={args.freq}MHz" if args.freq else ""
    pool_size = f", budget={args.budget}" if args.budget else ""
    nproc = f", workers={args.workers}" if args.workers else ""
    loop = ", nondeterministic" if args.nondeterministic else ""
    direction = f", O={args.objective}" if args.objective != "cycles" else ""
    slack = f", area_slack={args.area_slack}" if args.area_slack else ""
    print(
        f"{len(work)} runs, {args.jobs} jobs, stage={args.stage}"
        f", binding={args.binding}{clock}{pool_size}{nproc}{loop}{direction}{slack}",
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
                f" [{r['scheduler']}]  {r.get('seconds', 0)}s",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}\n")

    ok = [r for r in results if r["status"] == "pass"]
    print(variant_table(results, schedulers))
    if args.per_region:
        print("\n" + region_table(ok))
    if args.solves:
        print("\n" + solve_table(ok, args.solves))
    if args.alloc:
        print("\n" + alloc_table(ok))
    if args.area:
        print("\n" + area_table(ok))

    tally = Counter(r["status"] for r in results)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()

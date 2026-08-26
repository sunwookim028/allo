# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual area over the benchmark bed: scaffold, synthesize, compare.

    python -m benchmark.synth -k atax/none            # OOC synthesis, one case
    python -m benchmark.synth --impl                  # place+route, timing too
    python -m benchmark.synth --skip-synth            # scaffolds only

Each case scaffolds through `RTL.scaffold_project`, so the RTL, the
operator-core wrappers and the core-generation script are the shipped flow's.
One Vivado process per design, since Vivado can segfault on emitted RTL and
since two variants of one kernel share a top module name. A design whose extern
operators have no realization synthesizes its black boxes to nothing, so its
actual area under-counts and its row is marked. Stale utilization and timing
reports are deleted before a design runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from benchmark._child import run_child

REPO = Path(__file__).resolve().parents[1]
MARK = "@@SYNTH@@"


@dataclass(frozen=True)
class Knobs:
    """What one design is compiled under, off the command line."""

    binding: str
    objective: str
    freq: float | None
    area_slack: float
    budget: float | None
    #: Multipliers on the device's per-resource scarcity prices, as sorted
    #: (name, factor) pairs. A per-run scheduling parameter, not device data.
    weights: tuple[tuple[str, float], ...] = ()


def _tag(key: str, variant: str, scheduler: str) -> str:
    """One design's name, and the stem of every file its run writes."""
    return f"{key}/{variant}/{scheduler}".replace("/", "_")


# --- the child: emit and scaffold one design --------------------------------


def emit_one(item, knobs: Knobs, work: Path) -> dict:
    """Compile one (benchmark, variant, scheduler) and scaffold it under
    ``work``, returning the row the synthesis phase consumes."""
    from benchmark.report import area_of
    from benchmark.spec import find

    key, variant, scheduler = item
    tag = _tag(key, variant, scheduler)
    out: dict = {
        "tag": tag,
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
        "binding": knobs.binding,
        "status": "error",
    }
    bench = find(key)
    if variant in bench.skip:
        out.update(status="skip", note=bench.skip[variant])
        return out
    try:
        parts = bench.build()
        sched = bench.schedules[variant](parts)
        opts = {"freq_mhz": knobs.freq} if knobs.freq is not None else {}
        solver = {
            "scheduler": scheduler,
            "O": knobs.objective,
            "area_slack": knobs.area_slack,
        }
        if knobs.budget is not None:
            solver["budget"] = knobs.budget
        if knobs.weights:
            solver["resource_weights"] = dict(knobs.weights)
        rtl = sched.export("rtl", **opts).set_scheduler_opt(**solver)
        assert knobs.binding in ("trivial", "auto"), knobs.binding
        if knobs.binding == "trivial":
            rtl.use_trivial_binding()
        res = rtl.schedule()
        rtl.compile()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rtl.scaffold_project(str(work / f"{tag}.prj"))
        q = rtl.estimation
        period = 1000.0 / rtl.freq_mhz
        # Under a period-choosing objective the compile wrote its clock back
        # to the handle; otherwise the constraint is the derated model period.
        cycle_ns = (
            period if knobs.objective in ("freq", "wall") else res.cycle_ns or period
        )
        out.update(
            status="pass",
            top=rtl.top,
            part=rtl.device.part,
            clk=rtl.interfaces.of_symbol(rtl.top).control.clk,
            cycle_ns=cycle_ns,
            predicted={**area_of(q), "mem_bits": q.mem_bits},
            blackboxes=[str(w.message) for w in caught],
        )
    except BaseException as e:  # a fired assert is a row, not a crash
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
    return out


def _run_child(item, knobs: Knobs, work: Path, timeout: int) -> dict:
    key, variant, scheduler = item
    argv = [
        "--one",
        f"{key}::{variant}::{scheduler}",
        "--binding",
        knobs.binding,
        "--objective",
        knobs.objective,
        "--work",
        str(work),
    ]
    if knobs.freq is not None:
        argv += ["--freq", str(knobs.freq)]
    if knobs.area_slack:
        argv += ["--area-slack", str(knobs.area_slack)]
    if knobs.budget is not None:
        argv += ["--budget", str(knobs.budget)]
    for _n, _v in knobs.weights:
        argv += ["--weight", f"{_n}={_v}"]
    base = {
        "tag": _tag(key, variant, scheduler),
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
    }
    row, _ = run_child("benchmark.synth", MARK, argv, timeout, base)
    return row


# --- the synthesis phase -----------------------------------------------------


def vivado_command(explicit: str | None) -> str:
    """The shell prefix that reaches a `vivado` binary: an explicit path, one
    already on PATH, or the newest install under /tools/Xilinx/Vivado."""
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            settings = p / "settings64.sh"
            if not settings.exists():
                raise SystemExit(f"--vivado: no {settings}")
            return f"source {settings} && vivado"
        if not p.exists():
            raise SystemExit(f"--vivado: no such binary {p}")
        return str(p)
    if shutil.which("vivado"):
        return "vivado"
    if xv := os.environ.get("XILINX_VIVADO"):
        settings = Path(xv) / "settings64.sh"
        if not settings.exists():
            raise SystemExit(f"XILINX_VIVADO: no {settings}")
        return f"source {settings} && vivado"
    installs = sorted(Path("/tools/Xilinx/Vivado").glob("*/settings64.sh"))
    if installs:
        return f"source {installs[-1]} && vivado"
    raise SystemExit(
        "no vivado: put one on PATH, set XILINX_VIVADO to its install "
        "directory, or pass --vivado"
    )


def design_tcl(d: dict, work: Path, impl: bool) -> Path:
    """One design's whole run: its own project (via the scaffold's
    `gen_ip.tcl` when it has cores), the RTL off `filelist.f`, OOC synthesis,
    and under ``--impl`` a clock constraint plus place and route."""
    prj = work / f"{d['tag']}.prj"
    gen_ip = prj / "gen_ip.tcl"
    if gen_ip.exists():
        project = [f"source {gen_ip}"]
    else:
        project = [
            f"create_project -in_memory -part {d['part']}",
            "set_property target_language Verilog [current_project]",
        ]
    reads = [
        f"  read_verilog -sv -quiet {prj / name}"
        for name in (prj / "filelist.f").read_text().split()
    ]
    if (prj / "shims.v").exists():
        reads.append(f"  read_verilog -sv -quiet {prj / 'shims.v'}")
    steps = [
        f"  synth_design -top {d['top']} -part {d['part']}"
        " -mode out_of_context -flatten_hierarchy none",
    ]
    if impl:
        steps += [
            f"  create_clock -period {d['cycle_ns']} -name bed_clk"
            f" [get_ports {d['clk']}]",
            "  opt_design",
            "  place_design",
            "  route_design",
            f"  report_timing_summary -file {work / d['tag']}_timing.rpt",
        ]
    steps.append(f"  report_utilization -file {work / d['tag']}_util.rpt")
    lines = [
        *project,
        "if {[catch {",
        *reads,
        *steps,
        f'  puts "### DONE {d["tag"]}"',
        '} err]} { puts "### FAIL ' + d["tag"] + ': $err" }',
    ]
    p = work / f"synth_{d['tag']}.tcl"
    p.write_text("\n".join(lines) + "\n")
    return p


def run_vivado(vivado: str, tcl: Path, log: Path, timeout: int) -> None:
    """One Vivado process, killed at ``timeout``; the report it then fails to
    write is what marks the design's row."""
    with log.open("w") as sink:
        try:
            subprocess.run(
                f"{vivado} -mode batch -nojournal -nolog -source {tcl}",
                shell=True,
                executable="/bin/bash",
                cwd=tcl.parent,
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            sink.write(f"\n### TIMEOUT after {timeout}s\n")


#: `report_utilization` row -> the key it lands under. "LUT as Memory" is
#: excluded: it is the sum of the two rows below it.
_UTIL_ROWS = {
    "LUT as Logic": "lut_logic",
    "LUT as Distributed RAM": "lut_mem",
    "LUT as Shift Register": "srl",
    "CLB Registers": "ff",
    "Block RAM Tile": "bram",
    "URAM": "uram",
    "DSPs": "dsp",
    "CARRY8": "carry8",
}


def read_utilization(work: Path, tag: str) -> dict | None:
    """One design's cell counts off its `report_utilization`. A Block RAM Tile
    is reported in halves (an 18Kb RAMB is 0.5), so counts stay floats. The
    first occurrence of a row wins, since a post-route report repeats the names
    in later per-region tables whose leading column is not a count."""
    p = work / f"{tag}_util.rpt"
    if not p.exists():
        return None
    first: dict = {}
    for line in p.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and (key := _UTIL_ROWS.get(cells[1])) and key not in first:
            first[key] = float(cells[2])
    out = {v: first.get(v, 0.0) for v in _UTIL_ROWS.values()}
    # Total LUT sites, whichever role each is in.
    out["lut"] = out["lut_logic"] + out["lut_mem"] + out["srl"]
    return {k: int(v) if v == int(v) else v for k, v in out.items()}


def read_wns(work: Path, tag: str) -> float | None:
    """The design's worst negative slack off `report_timing_summary`: the
    first value row under the ``WNS(ns)`` header."""
    p = work / f"{tag}_timing.rpt"
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    head = next((i for i, line in enumerate(lines) if "WNS(ns)" in line), None)
    if head is None:
        return None
    for row in lines[head + 1 :]:
        tok = row.split()
        if tok and not set(row) <= set("- |"):
            try:
                return float(tok[0])
            except ValueError:
                return None
    return None


# --- the table and the CSV ---------------------------------------------------

_PRED = ("lut", "ff", "dsp", "srl", "mem_bits")
_ACT = ("lut", "lut_logic", "lut_mem", "srl", "ff", "dsp", "carry8", "bram", "uram")


def write_csv(work: Path, rows: list[dict], impl: bool) -> None:
    with (work / "synth.csv").open("w") as f:
        w = csv.writer(f)
        head = ["tag", "status", "clock_mhz"] + [f"pred_{k}" for k in _PRED]
        head += list(_ACT)
        if impl:
            head += ["wns_ns", "fmax_mhz"]
        w.writerow(head)
        for r in rows:
            clock = round(1000.0 / r["cycle_ns"], 1) if "cycle_ns" in r else ""
            line = [r["tag"], r["status"], clock]
            line += [r.get("predicted", {}).get(k, "") for k in _PRED]
            a = r.get("actual") or {}
            line += [a.get(k, "") for k in _ACT]
            if impl:
                line += [r.get("wns_ns", ""), r.get("fmax_mhz", "")]
            w.writerow(line)


def print_table(rows: list[dict], impl: bool) -> None:
    head = (
        f"{'design':<38} {'LUT p/a':>15} {'ratio':>6} {'FF p/a':>15}"
        f" {'ratio':>6} {'DSP p/a':>9} {'SRL p/a':>11}"
    )
    if impl:
        head += f" {'WNS':>7} {'fmax':>6}"
    print()
    print(head)
    print("-" * len(head))
    for r in rows:
        if r["status"] != "pass":
            note = r.get("note") or r.get("error", "")
            head = f"{r['tag']:<38} [{r['status']}]"
            print(f"{head} {note.splitlines()[0][:80]}" if note else head)
            continue
        p, a = r["predicted"], r.get("actual")
        note = "  [BLACK BOXES]" if r.get("blackboxes") else ""
        if a is None:
            print(f"{r['tag']:<38} {p['lut']:>7}/{'--':<7}{note}")
            continue
        # `srl` in the estimate covers every state-holding LUT site, so its
        # actual is the shift registers plus the distributed RAM.
        astate = a["srl"] + a["lut_mem"]
        line = (
            f"{r['tag']:<38} {p['lut']:>7}/{a['lut']:<7}"
            f" {p['lut'] / max(a['lut'], 1):>6.2f}"
            f" {p['ff']:>7}/{a['ff']:<7} {p['ff'] / max(a['ff'], 1):>6.2f}"
            f" {p['dsp']:>4}/{a['dsp']:<4} {p['srl']:>5}/{astate:<5}"
        )
        if impl:
            wns = r.get("wns_ns")
            fmax = r.get("fmax_mhz")
            line += f" {wns:>7.3f}" if wns is not None else f" {'--':>7}"
            line += f" {fmax:>6.1f}" if fmax is not None else f" {'--':>6}"
        print(line + note)


# --- main --------------------------------------------------------------------


def _weights(spec: list[str]) -> tuple[tuple[str, float], ...]:
    """`["lut=2"]` as sorted (name, factor) pairs. Sorted so two runs asking for
    the same reweighting build the same knobs whatever order they were typed."""
    pairs = []
    for s in spec:
        name, _, value = s.partition("=")
        pairs.append((name.strip(), float(value)))
    return tuple(sorted(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--one", help=argparse.SUPPRESS)  # the child entry point
    ap.add_argument(
        "-k", "--filter", default="", help="substring of suite/name/variant"
    )
    ap.add_argument(
        "--scheduler",
        default="heuristic",
        help="comma-separated; each case is emitted under each",
    )
    ap.add_argument(
        "--binding",
        default="trivial",
        help="'trivial' or 'auto', as the bed scan takes it",
    )
    ap.add_argument(
        "--objective",
        default="cycles",
        help="the O knob each case compiles under; 'freq' constrains each "
        "design at the clock its sweep chose",
    )
    ap.add_argument(
        "--freq", type=float, help="target clock (MHz), overriding the device default"
    )
    ap.add_argument(
        "--area-slack",
        type=float,
        default=0.0,
        help="fraction the area solve's span leash is widened by",
    )
    ap.add_argument(
        "--budget",
        type=float,
        help="what one exact solve may spend, in the solver's deterministic "
        "time units; unset keeps the compiler default",
    )
    ap.add_argument(
        "--impl",
        action="store_true",
        help="place and route after synthesis, and report timing",
    )
    ap.add_argument(
        "--skip-synth", action="store_true", help="scaffold and predict only"
    )
    ap.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="multiply one resource's scarcity price, e.g. --weight lut=2. "
        "Repeatable; the axis the area currency is swept over",
    )
    ap.add_argument("-j", "--jobs", type=int, default=8, help="emit children in flight")
    ap.add_argument(
        "--synth-jobs",
        type=int,
        default=4,
        help="Vivado sessions in flight (about 8 GB peak each)",
    )
    ap.add_argument(
        "--timeout", type=int, default=900, help="wall seconds per emit child"
    )
    ap.add_argument(
        "--synth-timeout",
        type=int,
        default=7200,
        help="wall seconds per Vivado session",
    )
    ap.add_argument("--vivado", help="vivado binary or install directory")
    ap.add_argument("--work", default=str(REPO / "benchmark" / "synth_work"))
    args = ap.parse_args()

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    knobs = Knobs(
        binding=args.binding,
        objective=args.objective,
        freq=args.freq,
        area_slack=args.area_slack,
        budget=args.budget,
        weights=_weights(args.weight),
    )

    if args.one:
        row = emit_one(args.one.split("::"), knobs, work)
        print(MARK + json.dumps(row), flush=True)
        return

    vivado = None if args.skip_synth else vivado_command(args.vivado)

    sys.path.insert(0, str(REPO))
    from benchmark.spec import discover

    items = [
        (b.key, v, s)
        for b in discover()
        for v in b.schedules
        for s in args.scheduler.split(",")
        if args.filter in f"{b.key}/{v}"
    ]
    print(
        f"{len(items)} designs, binding={args.binding}"
        + (f", freq={args.freq}MHz" if args.freq else ""),
        flush=True,
    )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_child, it, knobs, work, args.timeout) for it in items]
        for i, f in enumerate(futs, 1):
            r = f.result()
            rows.append(r)
            print(f"[{i}/{len(items)}] emit {r['tag']}: {r['status']}", flush=True)
            for miss in r.get("blackboxes", []):
                print(
                    f"  !! {r['tag']}: {miss} -- actual area will "
                    "under-count; do not believe this row",
                    flush=True,
                )

    designs = [r for r in rows if r["status"] == "pass"]
    if not args.skip_synth:

        def synth(d):
            (work / f"{d['tag']}_util.rpt").unlink(missing_ok=True)
            (work / f"{d['tag']}_timing.rpt").unlink(missing_ok=True)
            t0 = time.time()
            run_vivado(
                vivado,
                design_tcl(d, work, args.impl),
                work / f"synth_{d['tag']}.out",
                args.synth_timeout,
            )
            return d["tag"], round(time.time() - t0, 1)

        with ThreadPoolExecutor(max_workers=args.synth_jobs) as pool:
            for i, f in enumerate([pool.submit(synth, d) for d in designs], 1):
                tag, seconds = f.result()
                print(f"[{i}/{len(designs)}] synth {tag}: {seconds}s", flush=True)

    for d in designs:
        d["actual"] = read_utilization(work, d["tag"])
        if args.impl and (wns := read_wns(work, d["tag"])) is not None:
            d["wns_ns"] = wns
            d["fmax_mhz"] = round(1000.0 / (d["cycle_ns"] - wns), 1)

    write_csv(work, rows, args.impl)
    print_table(rows, args.impl)
    print(f"\nrows: {work / 'synth.csv'}")


if __name__ == "__main__":
    main()

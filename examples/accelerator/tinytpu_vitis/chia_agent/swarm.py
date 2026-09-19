"""Run several TinyTPU-isa co-design searches at once, then report the best.

Each worker gets its own spec directory (a private copy of the two editable
files, under the run directory), its own evaluation scratch, its own MCP tool
name, and one framing of the problem. No git worktree is created for a worker:
the evaluator composes its tree from git plus the spec directory, so a worker
needs nothing else -- and there is no `git worktree remove --force` at the end to
delete a worker's `variants.jsonl` along with its checkout, which is how an
earlier run on chia-codesign had to rescue its logs by hand. Every artefact a
worker writes is already under `--run-dir`.

Spend: the cap is global. Each loop refuses to start a model call that would
pass it (projected from the largest call seen), and this process polls the
opencode DB and kills every worker outright if the cap is reached anyway.

    python chia_agent/swarm.py --workers 2 --iterations 3 --budget-usd 15
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from spend import spent_since

DEFAULT_CALL_USD = 3.5  # as loop.py

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[3]

#: Framings of the same objective, each grounded in something measured on this
#: design (RESULTS_ISA.md / COMPARISON.md). Each worker gets one angle to start
#: from; none of them is an instruction to make a particular change.
STRATEGIES = (
    (
        "operand-path",
        """Look at the operand delivery path spm -> vru -> array. At 16x16x16,
208 of the 464 words vru handles are overhead rather than MACs, and the 64
B-side vld words cross vru twice. Gemmini has no vector-register tier between
its scratchpad and the array. Is the vru tier paying for itself here?""",
    ),
    (
        "weight-prologue",
        """Look at the per-`mm` weight prologue: every mm instruction first
pushes T weight words into the array before any MAC, and that prologue is
serial with the MACs. Gemmini hides the equivalent with double-buffered weight
registers (preload into one set while computing with the other).""",
    ),
    (
        "accumulator",
        """Look at the accumulator: its read-add-write into a register file is
a real recurrence and synthesizes at II=2. An earlier flat-accumulator attempt
bought 2.3% for 13.7x the flip-flops in that unit and was reverted (audit item
21 in RESULTS_ISA.md), so weigh area as well as cycles.""",
    ),
)

BASE_TASK = """Lower TinyTPU-isa's RTL cosim cycle count on tiled int8 GEMM.
Current cosim cycles (all five shapes, for context): 4x4x4=252, 8x8x8=383,
12x12x12=591, 16x16x8=667, 16x16x16=919. A matched 4x4 int8 Gemmini, measured
over the same window, takes 161(or 144)/220/347/391/593, so this design is
1.55-1.8x slower. The search scores 4x4x4 + 16x16x16; a winner is re-verified
bit-exact at all five shapes.

Known, measured, still open (context -- not a list of instructions): vru word
count (208 of 464 words at 16x16x16 are overhead); the per-mm T-word weight
prologue (Gemmini double-buffers weights); the accumulator's II=2 recurrence.

Your starting angle:
{angle}
"""


def launch(worker, angle, run_dir: Path, iterations, soft_budget, t0_ms):
    log_dir = run_dir / worker
    log_dir.mkdir(parents=True, exist_ok=True)
    work = REPO_ROOT / ".chia_scratch" / run_dir.name / worker
    env = os.environ | {"CHIA_RUN_T0_MS": str(t0_ms),
                        "CHIA_BUDGET_USD": str(soft_budget)}
    command = [sys.executable, "-u", str(AGENT_DIR / "loop.py"),
               "--task", BASE_TASK.format(angle=angle),
               "--iterations", str(iterations),
               "--log-dir", str(log_dir),
               "--spec-dir", str(log_dir / "spec"),
               "--work-dir", str(work),
               "--tool-name", f"tpu{worker.replace('-', '')}"]
    handle = (log_dir / "worker.log").open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=AGENT_DIR, env=env, stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True)


def read_variants(log_dir: Path) -> list[dict]:
    path = log_dir / "variants.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def report(run_dir: Path, workers: list[str], t0_ms: int, started: float) -> dict:
    print("=" * 78)
    print("Swarm results (all cycle counts are RTL cosim)")
    print("=" * 78)
    summary = {"workers": {}, "baseline": None, "best": None}
    for worker in workers:
        entries = read_variants(run_dir / worker)
        base = next((e for e in entries if e["kind"] == "baseline"), None)
        if base and base["verdict"].get("ok"):
            summary["baseline"] = base["verdict"]["cycles"]
        rows = []
        for e in entries:
            if e["kind"] != "candidate":
                continue
            v = e["verdict"]
            rows.append({"iteration": e["iteration"], "accepted": e["accepted"],
                         "ok": v.get("ok"), "stage": v.get("stage"),
                         "cycles": v.get("cycles"), "total": v.get("total_cycles"),
                         "llm_usd": e.get("llm_usd")})
            print(f"  {worker:<16} iter {e['iteration']}: "
                  + (f"cosim {v['cycles']} total {v['total_cycles']}"
                     if v.get("ok") else f"FAILED at {v.get('stage')}")
                  + f"  {'ACCEPTED' if e['accepted'] else 'rejected'}"
                  + f"  ${e.get('llm_usd', 0):.2f}")
            if e["accepted"] and (summary["best"] is None
                                  or v["total_cycles"] < summary["best"]["total"]):
                summary["best"] = {"worker": worker, "iteration": e["iteration"],
                                   "total": v["total_cycles"], "cycles": v["cycles"]}
        summary["workers"][worker] = rows
    spend = spent_since(t0_ms)
    summary["spend"] = spend
    summary["wall_seconds"] = round(time.time() - started, 1)
    print(f"\n  baseline {summary['baseline']}")
    print(f"  best     {summary['best'] or 'no candidate beat the baseline'}")
    print(f"  spend    ${spend['usd']:.2f} ({spend['messages']} model messages); "
          f"wall {summary['wall_seconds'] / 60:.1f} min")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def kill(procs, opencode_too=False):
    for _, p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if opencode_too:
        # opencode runs as a grandchild of a Ray worker, outside the loops'
        # process groups, and would keep spending after its driver died. Match
        # this checkout's own install only.
        subprocess.run(["pkill", "-f", str(AGENT_DIR / "node_modules")],
                       capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--budget-usd", type=float, default=15.0)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "chia_runs"
                        / f"isa-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--stagger", type=float, default=60.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(STRATEGIES)[: args.workers]
    started = time.time()
    t0_ms = int(started * 1000)
    (run_dir / "run.json").write_text(json.dumps({
        "t0_ms": t0_ms, "workers": [w for w, _ in strategies],
        "iterations": args.iterations, "budget_usd": args.budget_usd,
        "model": os.environ.get("TINYTPU_OPENCODE_MODEL"),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "task_template": BASE_TASK, "strategies": dict(strategies)}, indent=1))

    # Each loop checks "spent + its next call <= cap" on its own, so two loops
    # can pass that check at the same moment. Reserve one projected call per
    # other worker; the hard cap below is the backstop.
    soft = args.budget_usd - DEFAULT_CALL_USD * (len(strategies) - 1)
    procs = []
    capped = False
    try:
        for worker, angle in strategies:
            if procs and args.stagger:
                time.sleep(args.stagger)
            print(f"launching worker '{worker}'", flush=True)
            procs.append((worker, launch(worker, angle, run_dir, args.iterations,
                                         soft, t0_ms)))
        while any(p.poll() is None for _, p in procs):
            spent = spent_since(t0_ms)["usd"]
            if spent >= args.budget_usd:
                print(f"HARD CAP: spent ${spent:.2f} >= ${args.budget_usd:.2f}; "
                      f"killing workers", flush=True)
                capped = True
                kill(procs, opencode_too=True)
                break
            time.sleep(15)
        for worker, p in procs:
            print(f"worker '{worker}' exited with {p.wait()}", flush=True)
    finally:
        kill(procs)
    summary = report(run_dir, [w for w, _ in strategies], t0_ms, started)
    summary["hard_cap_hit"] = capped
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    raise SystemExit(0)


if __name__ == "__main__":
    main()

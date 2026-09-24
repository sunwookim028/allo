"""Run several TinyTPU-isa co-design searches at once, then report the best.

Each worker gets its own spec directory (a private copy of the two editable
files, under the run directory), its own evaluation scratch, its own MCP tool
name, and one framing of the problem. No git worktree is created for a worker:
the evaluator composes its tree from git plus the spec directory, so a worker
needs nothing else -- and there is no `git worktree remove --force` at the end to
delete a worker's `variants.jsonl` along with its checkout, which is how an
earlier run on chia-codesign had to rescue its logs by hand. Every artefact a
worker writes is already under `--run-dir`.

Spend: `--budget-usd` is required, and the pre-flight gate (`preflight.py`)
runs before any worker starts: the project must bill CHIA2026, Vertex AI must
be enabled, and CHIA's cumulative spend on CHIA2026 plus this cap must fit
`CHIA_TOTAL_CAP_USD`. The per-run cap is global across THIS RUN's workers and
blind to any other run: every session a worker opens is titled with the run's
tag, and the cap sums those (`spend.run_spend`). Each loop refuses to start a
model call that would pass it (projected from the largest call seen), and this
process polls the opencode DB and kills every worker outright if the cap is
reached anyway.

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

import preflight
from spend import of_run, run_spend, spent_since

DEFAULT_CALL_USD = 3.5  # as loop.py

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[2]

#: Framings of the same objective, each grounded in something MEASURED on the
#: design as shipped at main @ 476a70d8 (172 / 262 / 418 / 484 / 686). The
#: previous angles -- the vru tier, the per-mm weight prologue, accu at II=2 --
#: are exactly what e24e433b landed, so they are gone. The facts come from
#: dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/ (per-process cosim timeline, measured)
#: and gemmini_comparison.rst's attribution (whose residual split is an
#: ESTIMATE, labelled as such). Each worker gets one angle to start from; none
#: is an instruction to make a particular change.
STRATEGIES = (
    (
        "front-end",
        """The front of the 16x16x16 run, measured (per-process cosim timeline
of the shipped design, window 47-688, reference `timeline_16x16x16`): dma_ld
runs 211 cycles back to back (68-279) and no PE computes before cycle 285 --
about 238 of the 641 cycles pass before the first MAC. vru runs 74 cycles,
then is blocked 60 (226-286) until the array starts. The docs' attribution
estimates (not measured) ~84 cycles of the remaining gap to Gemmini as operand
staging plus the serial DMA ahead of the first weight. What in that prologue
is forced by data dependence, and what is ordering?""",
    ),
    (
        "tail",
        """The back of the 16x16x16 run, measured (same timeline): the last PE
finishes at 624, accu runs 312-653 in four ~80-cycle bursts each followed by a
5-cycle block, and dma_st alternates 38 cycles running with 47 starved and
finishes at 688, 35 cycles after accu. The sequencer is blocked 134 of its
cycles. The docs' attribution lists the drain as part of the (estimated,
unbuilt) residual. Where does the tail's time go, and what of it overlaps work
that could already have finished?""",
    ),
    (
        "small-shape",
        """The fixed cost. Least squares over the five shapes gives a fixed
74.5 cycles plus 21.70 per dynamic instruction; at 4x4x4 the design takes 172
against Gemmini's 144-161 (docs: tinytpu_isa.rst, gemmini_comparison.rst).
Region start (s_axilite programming) sits inside the cosim window. Which part
of the fixed cost is the design's?""",
    ),
)

#: CO-DESIGN angles (`--codesign`). Each names one of the four constraints that
#: were MEASURED to collapse the mapspace at 16x16x16, and asks the worker what
#: the design looks like on the other side of it. None of them says what to
#: change: the refusal counts are facts, the response is the agent's.
CODESIGN_STRATEGIES = (
    (
        "acc-follows-k",
        """The binding constraint, measured, and stated as a SYMPTOM rather than
a diagnosis. 1,150 of the 1,226 enumerated nests die because the encoder cannot
make `acc` follow the k loop, so the k=0 tile has to be a peelable prefix --
which pins K innermost and unsplit.

What is measured about it, verified in-tree today:

  * `acc` is field `f2`, and `f2` IS an AGU target. One additive AGU term on it
    gives acc = [0] at Kt=1 and acc = [0, 1] at Kt=2. The Kt=2 form is a
    genuine no-peel GEMM and is numerically EXACT against `isa_ref`.
  * It dies at Kt>=3 with `f2=2, must be 0 (overwrite) or 1 (accumulate)`. The
    term's growth is ADDITIVE and MONOTONE where the k=0 test needs a step.
    Staticness is not the obstacle; monotonicity is.
  * A term on `f2` COSTS ONE of the three AGU terms, and the two constraints
    are IN SERIES with the first masking the second. The accumulating `mm`
    names A (f0, one term), acc (f2, one term) and its weights at
    B_SP + nb*MAXDIM + kb*T (f3, TWO terms): four in all as soon as the program
    has an n loop at all. Measured, for the no-peel `mm` at each (Kt, Nt) --
    reproduce it with `histogram.py --interaction`:

        AGU_TERMS=3, every Kt in {2,3,4} x Nt in {1,2,4}:  refused, AGU budget
        AGU_TERMS=4, Kt=2, any Nt:                         expressible, EXACT
        AGU_TERMS=4, Kt>=3, any Nt:                        refused, f2 range

    So at three terms the monotonicity limit is never reached and cannot be
    observed at all; widen the budget and it becomes the binding one. Neither
    change shows anything alone. Note also that Kt=2 is K <= 8 while the scored
    shape is 16x16x16 with Kt=4, so four terms plus an additive term is
    necessary and NOT sufficient.
  * A rule of the environment, not a hint: `isa_ref.run` iterates
    `expand(prog)`, i.e. the AGU-RESOLVED fields. A change resolved in the AGU
    resolution (the sequencer's kernel, with `expand` kept in lockstep) leaves
    what reaches a unit -- and so the instruction's architectural meaning and
    the frozen reference model -- untouched. The same change resolved in a
    unit's decode alters what a field VALUE means, and `isa_ref` will reject
    it. Both locations are yours to edit; they cost you different things.
  * A first-cause histogram overstates the prize. Remove the position check and
    re-census and 897 of those 1,150 are refused by the OTHER acc-peel branch
    (K split across two emitted loops) and 274 by the encoder's own m/n
    interleave limitation, with only 12 becoming expressible. Run
    `histogram.py --second-cause` yourself. Do not assume a fix here frees
    1,150 nests.

Find a mechanism. Price it: the decoder and the accumulator's write path are
real area and the clock must still close at 3.33 ns.""",
    ),
    (
        "open",
        """You get the whole first-cause histogram at 16x16x16 and no preferred
answer: 1,150 the encoder cannot make `acc` follow the k loop, 54 the encoder's
own m/n interleave limitation, 17 `AGU_TERMS=3`, 2 the accumulator RAW
distance, 0 `LOOP_DEPTH`.

Two things are measured and should stop you wasting iterations:

  * `AGU_TERMS=4` (the 64-bit AGU word repacked to four 16-bit terms) raises
    encodable nests from 3 to 7 and drives the `agu-terms` refusals to zero --
    and the mapper's CHOSEN nest does not change, so the RTL runs the same
    stream and the cycle count does not move while the area does. A 6-frame
    loop stack alone changes not one number. `IMEM_SIZE` 56 -> 104 changes not
    one number. Read on their own, each of those is a cost with no benefit --
    but `histogram.py --interaction` shows the address-term budget MASKS a
    second constraint on the accumulate field, so "alone" may be the problem
    rather than the change.
  * The mapping search alone is worth 3 cycles, at one shape: at 4x4x4 the
    mapper's program is 24 words against the hand-written 28, with the same
    four dynamic issues.

So the expressible set is not obviously the binding constraint on this machine,
and one honest outcome of your iteration is to show that it is not. Decide for
yourself what to attack -- including the intrinsic tile (one `mm` performs rows
x TxT), the encoder's own 54 refusals, or something the histogram does not name
at all -- and justify the choice from the numbers before you edit anything.""",
    ),
)

CODESIGN_TASK = """Find a TinyTPU-isa DESIGN POINT that a better loop nest can
use. You propose the hardware and the ISA; a frozen mapper enumerates the whole
mapspace and cosims the best nest your hardware can encode. Neither the mapper
nor its selection rule nor any test is reachable from your tools.

The design you start from, measured (RTL cosim, bit-exact, all five shapes):
4x4x4=172, 8x8x8=262, 12x12x12=418, 16x16x8=484, 16x16x16=686 cycles; the
search scores 4x4x4 and 16x16x16. A matched 4x4 int8 Gemmini takes
144-161/220/347/391/593 over the same window. Resources at that build, csynth:
see the pair in the prompt. A winner is re-verified bit-exact at all five
shapes, by stress_isa, and by the RTL stress testbench.

At 16x16x16 your hardware can encode 3 of 1,226 nests. Raising that number is
the point of the exercise, and the cycle count is the point of raising it; a
change that unlocks nests without lowering cycles is a real and reportable
outcome, not a failure, as long as you say which happened.

Already landed on this design (do not re-propose): program prefetch 8 words a
cycle, weights by scratchpad address with per-PE double-buffered weight loaders
(wld), A no longer through spad->vld->vr, accu at II=1 via s.dependence (valid
only because check_program enforces AR_RAW_DIST=4 between an accumulator write
and a read of it -- a closer read is an RTL-only failure the simulator does not
show), sequencer-precomputed row counts.

Your starting angle:
{angle}
"""

BASE_TASK = """Lower TinyTPU-isa's RTL cosim cycle count on tiled int8 GEMM.
Current cosim cycles (all five shapes, bit-exact, main @ 476a70d8):
4x4x4=172, 8x8x8=262, 12x12x12=418, 16x16x8=484, 16x16x16=686. A matched 4x4
int8 Gemmini, measured over the same window, takes 144-161/220/347/391/593, so
this design is 1.07-1.24x slower. The search scores 4x4x4 + 16x16x16; a winner
is re-verified bit-exact at all five shapes, by stress_isa, and by the RTL
stress testbench.

Already landed (do not re-propose): program prefetch 8 words a cycle, weights
by scratchpad address with per-PE double-buffered weight loaders (wld), A no
longer through spad->vld->vr, accu at II=1 via s.dependence (valid only
because check_program enforces AR_RAW_DIST=4 between an accumulator write and
a read of it -- a closer read is an RTL-only failure the simulator does not
show), sequencer-precomputed row counts.

Your starting angle:
{angle}
"""


def run_tag(run_dir: Path, t0_ms: int) -> str:
    """The tag every opencode session of this run is titled with."""
    return f"chia-run {run_dir.name}@{t0_ms}"


def launch(worker, angle, run_dir: Path, iterations, soft_budget, t0_ms,
           codesign=False):
    log_dir = run_dir / worker
    log_dir.mkdir(parents=True, exist_ok=True)
    work = REPO_ROOT / ".chia_scratch" / run_dir.name / worker
    env = os.environ | {"CHIA_RUN_T0_MS": str(t0_ms),
                        "CHIA_RUN_TAG": run_tag(run_dir, t0_ms),
                        "CHIA_BUDGET_USD": str(soft_budget)}
    command = [sys.executable, "-u", str(AGENT_DIR / "loop.py"),
               "--task", (CODESIGN_TASK if codesign
                          else BASE_TASK).format(angle=angle),
               "--iterations", str(iterations),
               "--log-dir", str(log_dir),
               "--spec-dir", str(log_dir / "spec"),
               "--work-dir", str(work),
               "--tool-name", f"tpu{worker.replace('-', '')}"]
    if codesign:
        command.append("--codesign")
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
            v = e.get("verdict") or {}   # None: no change was made, not scored
            rows.append({"iteration": e["iteration"], "accepted": e["accepted"],
                         "ok": v.get("ok"), "stage": v.get("stage"),
                         "cycles": v.get("cycles"), "total": v.get("total_cycles"),
                         "llm_usd": e.get("llm_usd")})
            print(f"  {worker:<16} iter {e['iteration']}: "
                  + (f"cosim {v['cycles']} total {v['total_cycles']}"
                     if v.get("ok") else f"FAILED at {v.get('stage')}"
                     if v else f"not scored ({e.get('reason')})")
                  + f"  {'ACCEPTED' if e['accepted'] else 'rejected'}"
                  + f"  ${e.get('llm_usd', 0):.2f}")
            if e["accepted"] and (summary["best"] is None
                                  or v["total_cycles"] < summary["best"]["total"]):
                summary["best"] = {"worker": worker, "iteration": e["iteration"],
                                   "total": v["total_cycles"], "cycles": v["cycles"]}
        summary["workers"][worker] = rows
    spend = run_spend(run_tag(run_dir, t0_ms) + " ", t0_ms)
    window = spent_since(t0_ms)
    summary["spend"] = spend
    #: Context, never a cap: everything opencode billed on this account in the
    #: same window, this run included.
    summary["account_window_usd"] = window["usd"]
    summary["wall_seconds"] = round(time.time() - started, 1)
    print(f"\n  baseline {summary['baseline']}")
    print(f"  best     {summary['best'] or 'no candidate beat the baseline'}")
    print(f"  spend    ${spend['usd']:.2f} over {len(spend['sessions'])} session(s), "
          f"{spend['messages']} model messages (the whole account in the same "
          f"window: ${window['usd']:.2f}); wall {summary['wall_seconds'] / 60:.1f} min")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    #: The same figure again, standalone and self-describing, so that reading
    #: what a run cost needs neither this summary's schema nor the run's tag.
    (run_dir / "spend.json").write_text(json.dumps(of_run(run_dir), indent=1))
    return summary


def status(run_dir: Path) -> None:
    """What this run is, was asked, and has reached -- from its directory
    alone. Safe on a finished, a live, or an abandoned run; reads nothing but
    `run.json`, the workers' logs and opencode's DB."""
    run_dir = run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    # Older run.json files (run 1, the smoke run) predate `run_tag`; it is
    # derivable, so a committed evidence directory reads back too.
    tag = run.get("run_tag") or run_tag(run_dir, run["t0_ms"])
    done = run_dir / "summary.json"
    alive = subprocess.run(["pgrep", "-f", f"--log-dir {run_dir}"],
                           capture_output=True, text=True).stdout.split()
    print(f"run      {run_dir}")
    print(f"  tag    {tag}"
          + ("" if run.get("run_tag") else "   (derived; run.json predates it)"))
    print(f"  began  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['t0_ms'] / 1000))}"
          f"  ({(time.time() - run['t0_ms'] / 1000) / 60:.1f} min ago)")
    print(f"  head   {run['head']}   model {run['model']}")
    print(f"  plan   {len(run['workers'])} worker(s) x {run['iterations']} "
          f"iteration(s), cap ${run['budget_usd']:.2f}"
          + ("  [CO-DESIGN]" if run.get("codesign") else ""))
    print(f"  state  " + ("FINISHED (summary.json written)" if done.exists()
                          else f"RUNNING ({len(alive)} worker process(es))"
                          if alive else
                          "STOPPED with no summary.json -- interrupted or "
                          "killed. There is no resume; see 'Abandoning a run' "
                          "in README.md"))
    for worker in run["workers"]:
        entries = read_variants(run_dir / worker)
        cands = [e for e in entries if e["kind"] == "candidate"]
        graded = [e for e in cands if (e.get("verdict") or {}).get("ok")]
        last = entries[-1]["iteration"] if entries else 0
        print(f"\n  {worker}: iteration {last} of {run['iterations']}, "
              f"{len(cands)} candidate(s), {len(graded)} graded, "
              f"{sum(1 for e in cands if e['accepted'])} accepted")
        print(f"    asked: {run['strategies'][worker].strip().splitlines()[0]}")
        for e in cands:
            v = e.get("verdict") or {}
            print(f"    iter {e['iteration']}: "
                  + (f"cosim {v['cycles']} total {v['total_cycles']}"
                     if v.get("ok") else f"FAILED at {v.get('stage')}" if v
                     else f"not scored ({e.get('reason')})")
                  + ("  ACCEPTED" if e["accepted"] else "  rejected"))
        log = run_dir / worker / "worker.log"
        if log.exists() and not done.exists():
            tail = log.read_text(errors="replace").splitlines()[-1:]
            print(f"    log tail: {tail[0][:100] if tail else '(empty)'}")
    spend = run_spend(tag + " ", run["t0_ms"])
    print(f"\n  spend  ${spend['usd']:.2f} of ${run['budget_usd']:.2f} over "
          f"{len(spend['sessions'])} session(s), {spend['messages']} model "
          f"messages -- from opencode's DB, never the `usage` field")
    if not spend["usd"]:
        recorded = (json.loads(done.read_text()).get("spend", {}).get("usd")
                    if done.exists() else None)
        print("         no session carries that tag"
              + (f"; summary.json recorded ${recorded:.2f}" if recorded else "")
              + ". `python3 spend.py report <repo>` attributes by time window.")
    print(f"  the full question: python3 -c \"import json;r=json.load("
          f"open('{run_dir}/run.json'));print(r['task_template'].format("
          f"angle=r['strategies']['{run['workers'][0]}']))\"")


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
    parser.add_argument("--status", type=Path, metavar="RUN_DIR",
                        help="report what that run is, was asked and has "
                             "reached, then exit. Starts nothing, spends "
                             "nothing.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    # Required: no run starts without an explicit per-run spend cap.
    # (Enforced below, not by argparse, so --status needs no budget.)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "chia_runs"
                        / f"isa-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--stagger", type=float, default=60.0)
    parser.add_argument("--codesign", action="store_true",
                        help="the CO-DESIGN loop: the workers propose hardware, "
                             "a frozen mapper enumerates the mapspace "
                             "exhaustively, and the best nest each candidate can "
                             "encode is what gets cosimmed")
    args = parser.parse_args()
    if args.status:
        status(args.status)
        raise SystemExit(0)
    if args.budget_usd is None:
        parser.error("--budget-usd is required: no run starts without an "
                     "explicit per-run spend cap")
    run_dir = args.run_dir.resolve()
    # A run directory is written once and never resumed. Re-using one
    # overwrites run.json -- and with it the tag that attributes this run's
    # spend -- truncates worker.log, appends a second run's iterations to
    # variants.jsonl, and leaves the previous run's edited spec in place, so
    # the "baseline" measured is that design rather than HEAD's.
    if (run_dir / "run.json").exists():
        raise SystemExit(
            f"refusing to re-use {run_dir}: it already holds a run.json.\n"
            f"  There is no resume. Inspect it with\n"
            f"    python swarm.py --status {run_dir}\n"
            f"  and start a fresh --run-dir, or delete this one.")
    run_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(CODESIGN_STRATEGIES if args.codesign
                      else STRATEGIES)[: args.workers]
    started = time.time()
    t0_ms = int(started * 1000)
    # Before any worker: which account and project this run charges, and
    # whether its cap fits. Cheap (gcloud reads), no model call.
    charge = preflight.require(args.budget_usd, run_t0_ms=t0_ms)
    (run_dir / "run.json").write_text(json.dumps({
        "t0_ms": t0_ms, "run_tag": run_tag(run_dir, t0_ms),
        "workers": [w for w, _ in strategies],
        "iterations": args.iterations, "budget_usd": args.budget_usd,
        "preflight": charge,
        "model": os.environ.get("TINYTPU_OPENCODE_MODEL"),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "codesign": args.codesign,
        "task_template": CODESIGN_TASK if args.codesign else BASE_TASK,
        "strategies": dict(strategies)}, indent=1))

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
                                         soft, t0_ms, args.codesign)))
        # THIS RUN's sessions, found by the title tag every worker passes to
        # opencode -- not the account's window, which also holds any other CHIA
        # run on the host and is how run 2 was killed after 2 of 5 iterations.
        # The scripted test model cannot be billed at all (pre-flight said so),
        # and its sessions cost $0, so the same sum is right for it.
        billable = charge.get("mode") != "test-model"
        tag = run_tag(run_dir, t0_ms) + " "
        while any(p.poll() is None for _, p in procs):
            spent = run_spend(tag, t0_ms)["usd"]
            if billable and spent >= args.budget_usd:
                print(f"HARD CAP: spent ${spent:.2f} >= ${args.budget_usd:.2f}; "
                      f"killing workers", flush=True)
                capped = True
                kill(procs, opencode_too=True)
                break
            time.sleep(15)
        for worker, p in procs:
            print(f"worker '{worker}' exited with {p.wait()}", flush=True)
    finally:
        # opencode_too: on Ctrl-C or an exception as much as on the hard cap.
        # opencode is a grandchild outside the loops' process groups and keeps
        # spending after its driver dies.
        kill(procs, opencode_too=True)
    summary = report(run_dir, [w for w, _ in strategies], t0_ms, started)
    summary["hard_cap_hit"] = capped
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    raise SystemExit(0)


if __name__ == "__main__":
    main()

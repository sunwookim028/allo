# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One search, in either disposition, with per-iteration instrumentation.

Same shape as the design-level `chia_agent/loop.py` -- propose, gate, score,
keep or rewind, never take the agent's word -- with three differences that the
brief asks for:

1. **Two dispositions.** `--disposition using` searches the design with today's
   Allo abstractions; `--disposition maintaining` searches Allo itself. Same
   workload, same gates, same design cases, same PPA feedback, so the two are
   comparable.

2. **The workload is an input.** `--workload` names an entry in
   `workloads.py`, which the prompt is built from. Pointing the loop at a new
   workload is a new entry there plus a design case that can run it.

3. **Per-iteration instrumentation, because the comparison IS the result.**
   `maintaining` is expected to be harder, and how much harder is a number
   worth having. Every iteration records which rung of the ladder it reached:

       proposed    the agent returned and left a non-empty diff
       policy      the diff passed the path/line/primitive policy
       built       the C++ built and Allo imported
       cheap       Allo's own suites still pass against main's baseline
       correct     every design case bit-exact, limits verdicts unchanged
       resourced   inside the resource budget on every case
       improved    strictly fewer cycles on at least one case, none worse

   `summary.json` reports the rate at each rung. A run where nothing improved
   but everything built is a different result from one where nothing built, and
   the brief wants to tell them apart.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import ray
from chia.base.ChiaFunction import get
from chia.models.opencode import AdditionalModelProvider, RateLimitError

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DESIGN_AGENT = REPO / "examples/accelerator/tinytpu_vitis/chia_agent"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DESIGN_AGENT))

import heldout                                               # noqa: E402
import objective                                             # noqa: E402
import patch_policy                                          # noqa: E402
import prompt as brief                                       # noqa: E402
import workloads                                             # noqa: E402
from abs_tool import AlloCompilerTool                        # noqa: E402
#: Reused verbatim from the design-level harness: the billing pre-flight, the
#: spend accounting over opencode's own DB, and the 40-minute-MCP-timeout LLM.
import preflight                                             # noqa: E402
from llm import IsaOpenCodeLLM                               # noqa: E402
import spend                                                 # noqa: E402
from spend import spent_since                                # noqa: E402

MODEL = os.environ.get("CHIA_ABS_MODEL",
                       os.environ.get("TINYTPU_OPENCODE_MODEL",
                                      "google-vertex/gemini-3.1-pro-preview"))
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("TINYTPU_VERTEX_LOCATION", "global")
TEST_BASE_URL = os.environ.get("TINYTPU_OPENCODE_BASE_URL")
ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")
DEFAULT_CALL_USD = 3.5
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_BACKOFF = 45.0

#: The rungs, in order. An iteration's `reached` is the highest one it got to.
#: `none` is the rung below `proposed`: the agent returned and left no diff at
#: all. It is a real and distinct outcome -- the first pilot launch scored it
#: twice, because the tool server had not bound and the agent had no way to
#: edit anything -- and leaving it out of this tuple crashed the summary.
RUNGS = ("none", "proposed", "policy", "built", "cheap", "correct",
         "resourced", "improved")
#: Which harness stage failing means which rung was NOT reached.
_STAGE_RUNG = {
    "policy": "proposed", "assemble": "proposed", "tamper": "proposed",
    "gate:build": "policy", "gate:import": "policy",
    "gate:tests": "built",
    "gate:bench_isa": "cheap", "gate:stress": "cheap", "gate:limits": "cheap",
    "gate:probe": "cheap",
    "gate:resources": "correct",
    "ppa:cosim": "correct", "ppa:csynth": "correct", "ppa:timing": "correct",
    "setup": "proposed", "harness": "proposed", "git": "proposed",
}


class BudgetExhausted(Exception):
    pass


class Budget:
    """This run's spend, attributed to THIS run's sessions.

    `spend.spent_since(t0)` sums every opencode message on the account since
    t0. Its docstring calls over-counting "the safe direction", and it is --
    for one track at a time. With two tracks running concurrently on one
    account it is not conservative, it is WRONG in a way that stops the wrong
    run: measured 2026-09-22, `spent_since` for this run read $14.63 against a
    $15 cap, of which $4.46 was this run and $10.17 was the co-design track's
    two workers. This loop would have been halted before its second iteration
    by another track's spending.

    So the per-run cap sums only the sessions this loop has seen come back.
    A call in flight is invisible until it returns -- its session id is not
    known before then -- which is acceptable because the cap is checked BEFORE
    a call, when every previous session is known. The in-flight blind spot is
    bounded by `largest_call`, which is projected from the GLOBAL delta and so
    stays conservative.

    The CUMULATIVE cap (`CHIA_TOTAL_CAP_USD`, checked by preflight) is a
    different question and is correctly account-wide: the account is shared, so
    the ceiling is shared. It is the per-RUN cap that must be per-run.
    """

    def __init__(self, cap_usd: float, t0_ms: int):
        self.cap, self.t0 = cap_usd, t0_ms
        self.largest_call = DEFAULT_CALL_USD
        self.sessions: set[str] = set()

    def note_session(self, session_id):
        if session_id:
            self.sessions.add(str(session_id))

    def spent(self) -> float:
        """USD over this run's own sessions."""
        if not self.sessions:
            return 0.0
        return spent_by_sessions(self.sessions, self.t0)["usd"]

    def spent_account_wide(self) -> float:
        """Every track's spend since t0. Used only to PROJECT the next call,
        where over-counting really is the safe direction."""
        return spent_since(self.t0)["usd"]

    def check(self, what: str) -> float:
        spent = self.spent()
        if spent + self.largest_call > self.cap:
            raise BudgetExhausted(
                f"not starting {what}: this run has spent ${spent:.2f} over "
                f"{len(self.sessions)} session(s) + next call "
                f"~${self.largest_call:.2f} would pass the ${self.cap:.2f} cap")
        return spent

    def observe(self, usd: float):
        self.largest_call = max(self.largest_call, usd)


def spent_by_sessions(session_ids, t0_ms: int) -> dict:
    """USD and messages for the named opencode sessions, since t0.

    Reads the same database `spend.py` reads, restricted to sessions this run
    created. `spend.py` itself is the design track's frozen file and is not
    edited here.
    """
    import sqlite3
    out = {"usd": 0.0, "messages": 0}
    ids = [str(s) for s in session_ids if s]
    if not ids or not spend.DB.exists():
        return out
    con = sqlite3.connect(f"file:{spend.DB}?mode=ro", uri=True, timeout=30)
    try:
        marks = ",".join("?" * len(ids))
        # `cost` is a field inside the message's `data` JSON blob, not a
        # column -- the same shape `spend.spent_since` reads. A query against
        # a `cost` column raises rather than returning 0, which is what a cap
        # needs: an unexpected schema must be loud, not silently free.
        rows = con.execute(
            f"select data from message where session_id in ({marks}) "
            f"and time_created >= ?", [*ids, t0_ms]).fetchall()
    finally:
        con.close()
    for (data,) in rows:
        d = json.loads(data)
        if d.get("role") != "assistant":
            continue
        out["usd"] += d.get("cost", 0) or 0
        out["messages"] += 1
    out["usd"] = round(out["usd"], 4)
    return out


def vertex_provider(provider: str, model_id: str) -> AdditionalModelProvider:
    if not PROJECT:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before a Vertex run.")
    if provider != "google-vertex" or not model_id:
        raise ValueError("the model must be google-vertex/<model-id>.")
    return AdditionalModelProvider(
        id="google-vertex", npm="@ai-sdk/google-vertex",
        name="Google Vertex AI", models=[model_id],
        options={"project": PROJECT, "location": LOCATION})


def make_llm(tool, disposition, workload, pilot=None):
    provider, _, model_id = MODEL.partition("/")
    if TEST_BASE_URL:
        additional = [AdditionalModelProvider(
            id=provider, models=[model_id], base_url=TEST_BASE_URL,
            api_key="unused")]
    else:
        additional = [vertex_provider(provider, model_id)]
    return IsaOpenCodeLLM(
        model=MODEL,
        system_message=(brief.pilot_brief(pilot, workload, tool.name) if pilot
                        else brief.system_message(disposition).format(
                            workload=workloads.brief(workloads.get(workload)),
                            common=brief.COMMON)),
        timeout_seconds=2400,
        retries=1,
        additional_providers=additional,
        # opencode's own file and shell tools are denied: the MCP tools are the
        # agent's only capability, so it has no path to a frozen file.
        config={"*": "deny", f"{tool.name}_*": "allow"})


def ask(llm, tool, text, budget: Budget, what: str, calls: list):
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        before = budget.check(what)
        before_account = budget.spent_account_wide()
        started = time.time()
        try:
            response = get(
                llm.prompt.options(resources={"opencode_creds": 1}).chia_remote(
                    llm, text, [tool]))
        except RateLimitError:
            if attempt == RATE_LIMIT_RETRIES:
                raise
            delay = RATE_LIMIT_BACKOFF * (2 ** attempt) * (0.5 + random.random())
            print(f"  rate limited; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            continue
        usage = dict(getattr(response, "usage", None) or {})
        budget.note_session(getattr(response, "session_id", None))
        # The projection uses the ACCOUNT-WIDE delta on purpose: it is an upper
        # bound on what one call can cost, and over-projecting is safe.
        delta = budget.spent_account_wide() - before_account
        budget.observe(max(usage.get("cost_usd", 0.0), delta))
        calls.append({"what": what, "seconds": round(time.time() - started, 1),
                      "session_id": getattr(response, "session_id", None),
                      "global_usd_during_call": round(delta, 4),
                      "completed": bool(getattr(response, "success", True)),
                      **usage})
        print(f"  [{what}] ${usage.get('cost_usd', 0):.2f}, "
              f"{usage.get('num_turns', '?')} turns, "
              f"{time.time() - started:.0f}s; run total ${budget.spent():.2f}",
              flush=True)
        return response
    raise RuntimeError("unreachable")


def reached(verdict: dict) -> str:
    """The highest rung this verdict got to."""
    if not verdict:
        return "proposed"
    if verdict.get("ok"):
        obj = verdict.get("objective") or {}
        if verdict.get("gate_only"):
            return "correct"
        if obj.get("verdict") in objective.KEEP:
            return "improved"
        return "resourced"
    stage = verdict.get("stage", "harness")
    return _STAGE_RUNG.get(stage, "proposed")


def summarize(v: dict) -> str:
    if not v:
        return "no verdict"
    if not v.get("ok"):
        return f"FAILED at {v.get('stage')}"
    obj = v.get("objective") or {}
    if not obj:
        return "gates passed (no PPA)"
    return (f"{obj.get('verdict', '?')}: {obj.get('summary', '')} | "
            f"resources: {obj.get('resource_summary', '')}")


def record(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(entry, sort_keys=True) + "\n")


def ensure_tree(tree: Path, ref: str) -> None:
    if not (tree / ".git").exists():
        tree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), ref],
                        cwd=REPO, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-f", ref], cwd=tree,
                   check=True, capture_output=True)
    subprocess.run(["git", "clean", "-xffd", "-e", "/mlir/build"], cwd=tree,
                   capture_output=True)


def run(args, budget: Budget) -> int:
    # CHIA_FROZEN_REF, not HEAD: the held-out experiment evaluates at a
    # generated ref where the answer has been removed, and the agent's tree,
    # the evaluator's slot and the prompt must all agree on it.
    ref = subprocess.run(
        ["git", "rev-parse", "--verify",
         os.environ.get("CHIA_FROZEN_REF", "HEAD") + "^{commit}"],
        cwd=REPO, capture_output=True, text=True).stdout.strip()
    if not ref:
        print("cannot resolve CHIA_FROZEN_REF")
        return 2
    dirty = [l for l in subprocess.run(
        ["git", "status", "--porcelain", "--", "chia_abstraction", "tests",
         "examples/accelerator/tinytpu_vitis"], cwd=REPO, capture_output=True,
        text=True).stdout.splitlines() if l.strip()]
    if dirty:
        print("Refusing to search: frozen paths differ from HEAD:\n"
              + "\n".join(dirty))
        return 2

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    tree = log_dir / "tree"
    ensure_tree(tree, ref)
    # working_dir is this directory; PYTHONPATH additionally puts the design
    # loop's `chia_agent/` on the worker's path, because the modules this loop
    # REUSES from it -- `llm`, `preflight`, `spend` -- are pickled into the
    # remote call and must be importable there. Packaging the repo root instead
    # would ship 725 MB of node_modules past Ray's 512 MB limit.
    # CHIA_TOOL_BASE_PORT is read from the WORKER environment by CHIA's tool
    # server actor, so it has to go in runtime_env, not the driver's env. The
    # default is 8000, which the design track's tool servers already hold on
    # this host: the first pilot launch bound nothing, the agent was handed NO
    # TOOLS, and it spent two iterations making no edit. That is the same
    # failure the design loop's smoke run hit, from a different cause.
    # The LLM binary must resolve on the PATH the WORKERS get, or every model
    # call "returns" with no work done: measured twice. The first time a
    # worker's PATH lacked it; the second time the worktree it was borrowed
    # from had been deleted, and three iterations each scored `none` at $0.
    import shutil as _sh
    oc = _sh.which("opencode", path=os.environ.get("PATH", ""))
    if not oc:
        print("Refusing to search: `opencode` does not resolve on the PATH the "
              "workers will get. Source chia.env, whose OPENCODE_BIN must "
              "point at THIS worktree's node_modules/.bin.")
        return 2
    print(f"  opencode: {oc}", flush=True)
    runtime_env = {"working_dir": str(HERE),
                   "env_vars": {
                       "PYTHONPATH": str(DESIGN_AGENT),
                       "CHIA_TOOL_BASE_PORT": os.environ.get(
                           "CHIA_TOOL_BASE_PORT", "8400"),
                       "CHIA_TOOL_MAX_PORT": os.environ.get(
                           "CHIA_TOOL_MAX_PORT", "8499"),
                       # The worker inherits the RAYLET's environment, not the
                       # driver's, so `opencode` is only on its PATH if the
                       # head happened to be started from a shell that had
                       # sourced chia.env. Ours was not, and CHIA reported
                       # `[Errno 2] No such file or directory: 'opencode'` --
                       # for free, because a failed call costs nothing.
                       "PATH": os.environ.get("PATH", "")}}
    # CHIA_RAY_ADDRESS pins this loop to ITS OWN head. `address="auto"` reads
    # the host's newest GCS address file, and this host runs several tracks at
    # once: "auto" found two clusters, connected to another track's, and then
    # the fallback `resources=` was rejected because that cluster already
    # existed. Starting a head on a distinct port and naming it here keeps the
    # tracks apart without any `ray stop` that could kill another agent's run.
    addr = os.environ.get("CHIA_RAY_ADDRESS")
    if addr:
        ray.init(address=addr, runtime_env=runtime_env)
    else:
        try:
            ray.init(address="auto", runtime_env=runtime_env)
        except (ConnectionError, ValueError):
            ray.init(resources={"opencode_creds": 1}, runtime_env=runtime_env)

    os.environ["CHIA_ABS_SLOT"] = str(args.slot)
    if args.pilot:
        os.environ["CHIA_ABS_PILOT"] = args.pilot
    tool = AlloCompilerTool(
        args.tool_name, str(tree), str(log_dir / "work"), str(HERE), str(REPO),
        args.disposition, args.workload, ref, ALLO_PYTHON, LLVM_BUILD_DIR)
    log = log_dir / "variants.jsonl"
    calls: list[dict] = []
    iters: list[dict] = []
    try:
        print("=" * 72)
        print(f"{args.disposition.upper()} disposition, workload "
              f"{args.workload}, ref {ref[:8]}")
        print("=" * 72, flush=True)
        # The baseline is the harness's own no-patch measurement, which is
        # already recorded in baseline/ppa.json; the loop reads it rather than
        # re-spending ten minutes on a number it has.
        # CHIA_BASELINE_DIR, resolved exactly as the evaluator resolves it.
        # Reading HEAD's baseline while the evaluator scores against the
        # held-out one would tell the agent a target it cannot reach and then
        # grade it against a different one. Caught in the first pilot launch,
        # before any model call completed.
        base_file = Path(os.environ.get(
            "CHIA_BASELINE_DIR", HERE / "baseline")) / "ppa.json"
        if not base_file.exists():
            print(f"No PPA baseline at {base_file}. Run:\n  python "
                  f"chia_abstraction/evaluate_abs.py --out DIR "
                  f"--record-baseline")
            return 2
        baseline = json.loads(base_file.read_text())
        print("Baseline, per design case:")
        for c, v in sorted(baseline["cases"].items()):
            print(f"  {c}: cycles {json.dumps(v.get('cycles'))} "
                  f"area {json.dumps(v.get('area'))} "
                  f"clock {v.get('estimated_ns')} ns")
        record(log, {"iteration": 0, "kind": "baseline", "ref": ref,
                     "disposition": args.disposition,
                     "workload": args.workload, "baseline": baseline})
        # The tool surface must actually be up before any model call. A tool
        # server that failed to bind still yields a working LLM call -- with
        # no tools -- and the agent then spends an iteration doing nothing.
        # Measured: that is exactly what happened on the first launch.
        probe = tool.check_policy()
        print(f"  tool server: {getattr(tool, 'hostname', '?')}:{getattr(tool, 'port', '?')}"
              f" -- check_policy says {probe!r}", flush=True)
        if "No changes" not in probe:
            print(f"Refusing to search: the tool surface is not answering "
                  f"({probe[:300]!r}).")
            return 2

        best_diff, best_obj = "", None
        history: list[str] = []
        llm = make_llm(tool, args.disposition, args.workload, args.pilot)

        for it in range(1, args.iterations + 1):
            print("=" * 72)
            print(f"Iteration {it}/{args.iterations} [{args.disposition}]")
            print("=" * 72, flush=True)
            tool.apply(best_diff)
            # Probe declarations belong to ONE candidate; a stale one from a
            # rewound iteration must not be scored against the next.
            (log_dir / "work" / "probe_calls.json").unlink(missing_ok=True)
            started, n_calls = time.time(), len(calls)
            text = brief.task(args.disposition, args.workload, args.angle,
                              baseline["cases"], history)
            if args.pilot:
                # The per-iteration prompt carries the PILOT brief, not the
                # first night's maintaining brief with its gap inventory.
                tail = text.split("YOUR ANGLE for this search:", 1)[1]
                text = (brief.pilot_brief(args.pilot, args.workload, tool.name)
                        + "\n\nYOUR ANGLE for this search:" + tail
                        + brief.PILOT_CLOSING.replace("{tool}", tool.name))
            if args.heldout:
                # The PROMPT is a channel too. Scan the whole assembled prompt
                # -- system message and task -- and refuse on any hit. The first
                # held-out run's tree was clean and its prompt named the answer.
                sysmsg = (brief.pilot_brief(args.pilot, args.workload, tool.name)
                          if args.pilot else brief.system_message(
                              args.disposition).format(
                              workload=workloads.brief(
                                  workloads.get(args.workload)),
                              common=brief.COMMON))
                leaks = heldout.scan_text(sysmsg + "\n" + text + "\n"
                                          + heldout.symptom())
                if leaks:
                    print("Refusing the held-out run: the PROMPT names the "
                          "answer:\n  " + "\n  ".join(leaks[:20]))
                    return 2
                # The held-out experiment: the agent is given the SYMPTOM and
                # nothing else. `heldout.py` has already removed the answer,
                # its tests and every mention of it from the ref the agent
                # reads, so this is the whole of its starting information.
                text += "\n" + heldout.symptom()
            text += f"""
Your tools are prefixed `{tool.name}_`. Read first
(`{tool.name}_read_source`, `{tool.name}_read_reference`), edit with
`{tool.name}_replace_text` (exact, unique -- far more reliable than a diff),
check the policy for free with `{tool.name}_check_policy`, then
`{tool.name}_build_allo` (~30 s; do this after EVERY C++ edit),
`{tool.name}_run_gates` (~5 min), and `{tool.name}_score` (~10 min) if the
gates pass. The harness re-measures your final tree independently either way.
"""
            try:
                response = ask(llm, tool, text, budget, f"iter{it}", calls)
            except BudgetExhausted as stop:
                print(f"  BUDGET: {stop}")
                record(log, {"iteration": it, "kind": "stopped",
                             "reason": str(stop)})
                break
            print(response.result, flush=True)
            agent_summary = str(response.result)[-4000:]
            if (not getattr(response, "success", True)
                    and "No such file or directory" in agent_summary):
                # Infrastructure, not the agent: stop, do not score `none`.
                print("  the model call could not START; stopping the run "
                      "rather than recording an empty iteration", flush=True)
                record(log, {"iteration": it, "kind": "stopped",
                             "reason": "model call could not start: "
                                       + agent_summary[-300:]})
                break
            diff = tool.diff()
            entry = {"iteration": it, "kind": "candidate",
                     "disposition": args.disposition, "diff": diff,
                     "agent_summary": agent_summary,
                     "llm_usd": round(sum(c.get("cost_usd", 0)
                                          for c in calls[n_calls:]), 4),
                     "llm_calls": calls[n_calls:]}
            if not diff.strip() or diff == best_diff:
                entry.update(accepted=False, reason="no diff",
                             reached="proposed" if diff.strip() else "none",
                             verdict=None,
                             seconds=round(time.time() - started, 1))
                history.append(f"iteration {it}: no change was made")
                print("  no change was made; nothing to score", flush=True)
                record(log, entry)
                iters.append(entry)
                continue
            try:
                touches = sorted(patch_policy.paths_in_diff(diff))
            except patch_policy.PatchError:
                touches = []
            entry["touches"] = touches
            verdict = tool._evaluate("loop", False)  # noqa: SLF001 -- returns a dict
            for attempt in range(1, args.max_debug_attempts + 1):
                if verdict.get("ok"):
                    break
                print(f"  candidate failed at {verdict.get('stage')} "
                      f"(debug attempt {attempt})", flush=True)
                try:
                    response = ask(llm, tool,
                                   f"The candidate was REJECTED at stage "
                                   f"'{verdict.get('stage')}':\n```\n"
                                   f"{verdict.get('detail', '')[-5000:]}\n```\n"
                                   f"Diagnose it and fix it, rebuild with "
                                   f"{tool.name}_build_allo, confirm with "
                                   f"{tool.name}_run_gates, then stop.",
                                   budget, f"iter{it}-debug{attempt}", calls)
                except BudgetExhausted as stop:
                    print(f"  BUDGET: {stop}")
                    break
                print(response.result, flush=True)
                diff = tool.diff()
                verdict = tool._evaluate("loop", False)  # noqa: SLF001 -- returns a dict
            rung = reached(verdict)
            obj = verdict.get("objective") or {}
            # `expressive` counts, and ranks above `win`: an abstraction that
            # makes a second architecture expressible is the project's
            # standard, and a faster current design is not. objective.KEEP.
            improved = obj.get("verdict") in objective.KEEP
            entry.update(verdict=verdict, reached=rung, accepted=improved,
                         diff=diff,
                         objective=obj.get("verdict"),
                         seconds=round(time.time() - started, 1),
                         llm_usd=round(sum(c.get("cost_usd", 0)
                                           for c in calls[n_calls:]), 4),
                         llm_calls=calls[n_calls:])
            print(f"  reached `{rung}`: {summarize(verdict)}")
            print(f"  {'ACCEPTED' if improved else 'REJECTED'} after "
                  f"{entry['seconds']:.0f}s", flush=True)
            record(log, entry)
            iters.append(entry)
            history.append(
                f"iteration {it}: reached `{rung}`, "
                f"{'ACCEPTED' if improved else 'rejected'}"
                + (f" at {verdict.get('stage')}" if not verdict.get("ok") else "")
                + f", touched {touches} -- {agent_summary[-300:]!r}")
            if improved:
                best_diff, best_obj = diff, obj
                if obj.get("newly_expressible") or obj.get("limits_fixed"):
                    print(f"  EXPRESSIVENESS: newly expressible "
                          f"{obj.get('newly_expressible')}, limits fixed "
                          f"{obj.get('limits_fixed')}", flush=True)
            else:
                tool.apply(best_diff)

        tool.apply(best_diff)
        rates = {r: sum(1 for e in iters
                        if RUNGS.index(e.get("reached") or "none")
                        >= RUNGS.index(r))
                 for r in RUNGS}
        n = max(len(iters), 1)
        summary = {
            "disposition": args.disposition, "workload": args.workload,
            "angle": args.angle, "model": MODEL, "ref": ref,
            "heldout": args.heldout, "pilot": args.pilot,
            "seeded": brief.SEEDED.get(args.pilot) if args.pilot else None,
            "iterations": len(iters), "rungs_reached": rates,
            "rates": {r: round(rates[r] / n, 3) for r in RUNGS},
            "llm_usd": round(sum(c.get("cost_usd", 0) for c in calls), 4),
            "llm_calls": len(calls),
            "spend_from_db_usd": round(budget.spent(), 4),
            "best": best_obj, "best_diff_lines": len(best_diff.splitlines()),
            "stages_failed": [e["verdict"].get("stage") for e in iters
                              if e.get("verdict") and not e["verdict"].get("ok")],
            "objectives": [e.get("objective") for e in iters],
            "newly_expressible": sorted({
                x for e in iters
                for x in ((e.get("verdict") or {}).get("objective") or {})
                .get("newly_expressible", [])}),
            "limits_fixed": sorted({
                x for e in iters
                for x in ((e.get("verdict") or {}).get("objective") or {})
                .get("limits_fixed", [])}),
        }
        (log_dir / "best.diff").write_text(best_diff)
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        (log_dir / "calls.json").write_text(json.dumps(calls, indent=1))
        print("=" * 72)
        print(f"Search complete [{args.disposition}]")
        for r in RUNGS:
            print(f"  {r:<10} {rates[r]}/{len(iters)}")
        print(f"  spend (opencode DB, this run): "
              f"${summary['spend_from_db_usd']:.2f}")
        return 0
    finally:
        tool.stop()
        ray.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disposition", required=True,
                    choices=patch_policy.DISPOSITIONS)
    ap.add_argument("--workload", default=workloads.DEFAULT_WORKLOAD)
    ap.add_argument("--angle", default="anything you can defend")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--max-debug-attempts", type=int, default=1)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--pilot", default=None, choices=("A", "B"),
                    help="A: measurements only, find the gap. B: the memory "
                         "write-port property named, with its ground truth. "
                         "Replaces the first night's gap inventory.")
    ap.add_argument("--heldout", default=None, choices=("dependence",),
                    help="the rediscovery experiment: append the measured "
                         "SYMPTOM to the task. Requires CHIA_FROZEN_REF to be "
                         "the held-out ref that heldout.py generated -- the "
                         "loop refuses to start otherwise, because a run at "
                         "HEAD would hand the agent the answer.")
    ap.add_argument("--tool-name", default="allo")
    ap.add_argument("--budget-usd", type=float,
                    default=os.environ.get("CHIA_BUDGET_USD"))
    ap.add_argument("--t0-ms", type=int,
                    default=int(os.environ.get("CHIA_RUN_T0_MS", "0")) or None)
    a = ap.parse_args()
    if a.heldout:
        ref = os.environ.get("CHIA_FROZEN_REF", "")
        leak = subprocess.run(
            ["git", "grep", "-l", "-E", heldout.LEAK_RE, ref or "HEAD"],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        if not ref or leak:
            raise SystemExit(
                f"--heldout {a.heldout} needs CHIA_FROZEN_REF set to a "
                f"held-out ref with no leaks. ref={ref or '(unset)'}; "
                f"leaks:\n{leak or '(none)'}\n"
                f"Build one with: python chia_abstraction/heldout.py make")
    t0 = a.t0_ms or int(time.time() * 1000)
    # The design loop's pre-flight, reused verbatim: the right billing account,
    # the API enabled, an explicit per-run cap, and room under the cumulative
    # cap. No model call happens before it passes.
    preflight.require(a.budget_usd, run_t0_ms=t0)
    a.budget_usd = float(a.budget_usd)
    raise SystemExit(run(a, Budget(a.budget_usd, t0)))


if __name__ == "__main__":
    main()

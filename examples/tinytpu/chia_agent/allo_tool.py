"""Narrow MCP tool surface for co-designing TinyTPU-isa (examples/tinytpu).

The agent may edit exactly the files `design.EDITABLE` names -- the instruction
set (`isa_spec.json`, the `isa_encoding.py` generated from it and the
`isa_ref.py` built on that), the eight units under `ip/units/`, the
architecture that wires them, the assembler, the programs, `microarch_isa.py`
(the parameter set, and the `CHIA_CONFIG` declaration that proposes the
configuration to be scored at) and `isa_dsl.py` (the program generator) -- and
only in a private spec directory, never in the repository. Everything that
decides the score is frozen and outside its reach; see `evaluate.py` for the
list and how it is enforced.

`isa_encoding.py` is GENERATED. Editing `isa_spec.json` by itself leaves it
stale and `gen_isa.py --conform` refuses the candidate, so there is a tool --
`regenerate_isa` -- that runs the FROZEN generator on the spec and writes the
artefact back. It is the only tool that writes a file the agent did not type.

THE FEEDBACK LADDER, and why it has three rungs. Run 3 was measured from
opencode's own database: `score_cycles` was 37 % of the run, the gate 15 %,
model latency 48 %, and the MCP layer itself SIX SECONDS. The bottleneck was
never the plumbing -- it was that the cheapest thing an agent could ask cost
52-76 s and the scoring tool 235 s, while the functional check underneath them
runs in seconds. So there was no tool it could call in under fifty seconds, and
fast iteration was impossible rather than merely slow.

  check_bit_exact        ~16 s   the PyTorch oracle alone
  run_functional_check   ~30 s   + conform, bench_isa, stress_isa
  score_cycles          ~240 s   + parametricity, csynth and RTL cosim

Each rung sees strictly more than the one below it. NONE of them is on the path
to acceptance and none of them can be: the harness re-scores the final spec
itself with the full gate and a fresh nonce-vouched cosim, `evaluate.py`
refuses a cheap tier on any scoring run, and `score_cycles` is capped at one
measured call per iteration (a repeat on byte-identical spec content is
answered from the cache and is free). What the cheap tiers cannot see is
written into their descriptions, in the words that matter: a functional check
says an edit is LEGAL, never that it is FASTER.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import functools
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import ray
from chia.base.tools.ChiaTool import ChiaTool

from design import EDITABLE, UNITS
from spec_policy import doc_violations, policy_violations
from session_trace import Trace, traced

#: CHIA binds each tool server to `ray.util.get_node_ip_address()`, which on a
#: Ray worker is the host's routable address -- an unauthenticated server
#: anyone on the network can call, able to edit the candidate and start Vitis
#: runs. So the DEFAULT is loopback. A multi-host swarm, where opencode and the
#: tool server sit on different nodes, must opt in explicitly with
#: TINYTPU_TOOL_HOST=node (Ray's routable address) or a specific address -- and
#: should then add authentication or a firewall, which this module does not.
#: This module is imported in the tool's actor before the server starts.
_TOOL_HOST = os.environ.get("TINYTPU_TOOL_HOST", "127.0.0.1")
if _TOOL_HOST != "node":
    ray.util.get_node_ip_address = lambda *args, **kwargs: _TOOL_HOST

#: Read-only context the agent may look at, served from git HEAD: the frozen
#: code that judges a candidate, then the design's documentation. The docs
#: pages replaced RESULTS_ISA.md / COMPARISON.md when main moved its notes into
#: the docs site; the reST is served as source.
REFERENCE = {
    "cosim.py": "examples/tinytpu/cosim.py",
    # The five scored shapes, which cosim.py and bench_isa.py now import from
    # here rather than each spelling out.
    "shapes.py": "examples/tinytpu/shapes.py",
    "bench_isa.py": "examples/tinytpu/bench_isa.py",
    "stress_isa.py": "examples/tinytpu/stress_isa.py",
    # `isa_ref.py` is no longer here: it is EDITABLE now, and `read_spec` is
    # how a writable file is read. `gen_isa.py` took its place, because it is
    # what an ISA change is held to.
    "gen_isa.py": "examples/tinytpu/gen_isa.py",
    "evaluate.py": "examples/tinytpu/chia_agent/evaluate.py",
    "param_check.py": "examples/tinytpu/chia_agent/param_check.py",
    # The co-design loop's frozen half. Readable on purpose: the agent should
    # be able to see exactly what the mapper enumerates, what it refuses, and
    # by what rule it picks -- it just cannot change any of it.
    "mapspace.py": "examples/tinytpu/chia_agent/mapspace.py",
    "codesign_gate.py": "examples/tinytpu/chia_agent/codesign_gate.py",
    "codesign_cosim.py": "examples/tinytpu/chia_agent/codesign_cosim.py",
    "act.rst": "docs/source/extensions/act.rst",
    "tinytpu_isa.rst": "docs/source/designs/tinytpu_isa.rst",
    "tinytpu_history.rst": "docs/source/designs/tinytpu_history.rst",
    "gemmini_comparison.rst": "docs/source/designs/gemmini_comparison.rst",
    "limitations.rst": "docs/source/developer/limitations.rst",
    # Measured per-process cosim timeline of the shipped design at 16x16x16.
    "timeline_16x16x16": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/README.md",
    "timeline_16x16x16.txt": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/timeline.txt",
    "timeline_16x16x16_rle.txt": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/rle.txt",
}


#: THE MEASURED COST OF EACH TOOL, on this host, at the shipped design. Every
#: number here was taken with `session_trace.py` on a scripted session; none is
#: an estimate, and the docstrings below quote this table rather than a
#: remembered figure. The one that used to be remembered said
#: `run_functional_check` cost "~15 s" when it cost 52-76 s, so an agent trying
#: to budget its session budgeted four times low.
#:
#: Read it as a ladder. Each rung sees strictly more than the one below and
#: costs roughly an order of magnitude more; the bottom rung exists because
#: before it there was NO tool an agent could call in under fifty seconds,
#: which is why fast iteration was impossible rather than merely slow.
TOOL_SECONDS = {
    "read_spec": 0.01,
    "read_reference": 0.05,
    "replace_text": 0.2,
    "apply_spec_patch": 0.3,
    "insert_after": 0.2,
    "regenerate_isa": 3,
    "check_bit_exact": 16,
    "run_functional_check": 30,
    "mapspace_report": 60,
    "score_cycles": 240,
}

#: The wall clock one opencode session gets (`loop.py`'s `timeout_seconds`).
#: Kept here as well as there because the agent is never told it otherwise:
#: run 3 spent whole 2400 s sessions with nothing in the loop having mentioned
#: that 2400 s existed, and four of them returned nothing at all.
SESSION_BUDGET_S = 2400

#: Measured `score_cycles` calls one iteration may buy. Run 3's sessions called
#: it 2-5 times each and it was 37 % of the whole run -- used as a hill-climbing
#: oracle although the prompt already says the harness re-scores the final spec
#: independently, so most of those minutes re-measured something that was going
#: to be measured anyway. A call that hits the cache (the spec is byte-identical
#: to one already scored) is free and does not count against this.
SCORE_CALLS_PER_ITERATION = 1


class AlloSpecTool(ChiaTool):
    """Let an agent co-edit TinyTPU-isa's hardware and program generator only."""

    def setup(self, spec_dir: str, work_dir: str, agent_dir: str, repo: str,
              allo_python: str, llvm_build_dir: str,
              codesign: bool = False, trace_dir: str | None = None,
              session_budget_s: int = SESSION_BUDGET_S) -> None:
        #: Co-design mode: the evaluator also enumerates the mapspace against
        #: the candidate's hardware and cosims the nest the FROZEN mapper
        #: chose, rather than the program the agent hand-wrote.
        self.codesign = bool(codesign)
        self.spec_dir = Path(spec_dir).resolve()
        self.sources = {rel: self.spec_dir / rel for rel in EDITABLE}
        self.work_dir = Path(work_dir).resolve()
        # Absolute path to the checkout's own evaluator: the actor runs from a
        # Ray copy of chia_agent/, and the score must not come from that copy.
        self.evaluator = str(Path(agent_dir).resolve() / "evaluate.py")
        self.repo = str(Path(repo).resolve())
        # Captured here, in the driver, because a Ray actor inherits the
        # raylet's environment rather than the shell that launched the loop.
        self.eval_env = {
            "TINYTPU_ALLO_PYTHON": allo_python,
            "LLVM_BUILD_DIR": llvm_build_dir,
        }
        self._lock = threading.Lock()
        # This server is the ONE component that sees every round trip from the
        # far side of the MCP boundary, so its timings are the ground truth
        # for what a tool call cost and are independent of opencode's stream
        # format. `traced` puts a line out before the work and another after,
        # which is why a tool call still running when the process dies leaves
        # a `tool.start` with no `tool.end` -- naming what it was doing.
        # Passed in rather than read from the environment: a Ray actor
        # inherits the raylet's environment, not the shell that launched the
        # loop (the same reason `eval_env` is captured in the driver).
        self.trace = Trace(Path(trace_dir) / "trace.jsonl") if trace_dir else Trace()
        #: The session wall clock, and the per-iteration allowance of measured
        #: cosims. Both are reported back with every tool result, because
        #: nothing else tells the agent either number.
        self.session_budget_s = int(session_budget_s)
        #: Per-iteration state, in a FILE rather than on the object: the driver
        #: opens each iteration and the Ray actor serves the tools, two
        #: processes over one spec directory, and an attribute set in one of
        #: them is invisible to the other. It sits beside the work directories
        #: rather than inside one, because the evaluator wipes those.
        self.state_path = self.work_dir / "session_budget.json"
        # The frozen half of the cache key: a score is only reusable while the
        # checkout that judged it has not moved.
        self.frozen_ref = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo,
            capture_output=True, text=True).stdout.strip()
        assert all(path.is_file() for path in self.sources.values()), self.sources
        for method in (self.read_spec, self.read_reference, self.replace_text,
                       self.apply_spec_patch, self.insert_after,
                       self.regenerate_isa, self.check_bit_exact,
                       self.run_functional_check,
                       self.score_cycles) + (
                           (self.mapspace_report,) if self.codesign else ()):
            name = f"{self.name}_{method.__name__}"
            self.mcp.add_tool(traced(self.trace, name, self._metered(method)),
                              name=name)

    def __getstate__(self):
        # The tool is re-pickled on every prompt; a Lock cannot be.
        state = dict(super().__getstate__())
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        # A fresh lock per unpickled copy, made HERE rather than lazily in
        # evaluate(): two MCP calls arrive on two threads at once, and a lazy
        # `if self._lock is None` lets both create a lock, so two evaluations
        # would share one work directory -- each wiping the other's.
        super().__setstate__(state)
        self._lock = threading.Lock()

    # -- The session budget, and what a measured score costs -----------------
    #
    # Everything in this block makes the AGENT's feedback cheaper. None of it
    # is on the path to acceptance: the harness re-scores the final spec with
    # `evaluate(work="harness")`, which runs the full gate and a fresh cosim
    # and never reads the cache below. See `score_cycles` for the one place
    # that matters.
    def _state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        state.setdefault("iteration", 0)
        state.setdefault("started", time.time())
        state.setdefault("scored", 0)
        state.setdefault("cache", {})
        return state

    def _put_state(self, state: dict) -> None:
        # Best effort by design: bookkeeping must never fail a tool call, and a
        # lost write costs at worst one extra cosim.
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    def begin_iteration(self, n: int) -> None:
        """Open an iteration: reset its clock and its score allowance.

        Driver-side, like `snapshot` / `restore` -- deliberately NOT an MCP
        tool. An agent that could open a new iteration could lift its own cap,
        and a budget the budgeted party may reset is not a budget. The score
        cache is carried across, because a spec that is byte-identical to one
        already measured has the same cycle count whichever iteration asks.
        """
        state = self._state()
        self._put_state({"iteration": int(n), "started": time.time(),
                         "scored": 0, "cache": state.get("cache", {})})

    def spec_hash(self) -> str:
        """The candidate's identity: every writable byte, plus the frozen ref.

        This is what `score_cycles` memoises on. Two calls with the same hash
        ask the same question of the same referee, so the second is answered
        from the first rather than paid for again -- which is most of what run
        3 spent its cosim minutes on, the agent re-scoring a spec it had not
        touched since the last score.
        """
        h = hashlib.sha256()
        h.update(self.frozen_ref.encode())
        h.update(b"\0codesign=" + str(self.codesign).encode())
        for name in sorted(self.sources):
            h.update(b"\0" + name.encode() + b"\0")
            h.update(self.sources[name].read_bytes())
        return h.hexdigest()

    def budget(self) -> dict:
        """What is left: seconds of this session, and measured cosims of this
        iteration. Returned with EVERY tool result."""
        state = self._state()
        used = max(0.0, time.time() - float(state["started"]))
        return {
            "iteration": state["iteration"],
            "session_seconds": self.session_budget_s,
            "seconds_used": round(used, 1),
            "seconds_left": round(max(0.0, self.session_budget_s - used), 1),
            "score_cycles_left": max(
                0, SCORE_CALLS_PER_ITERATION - int(state["scored"])),
            "tool_seconds": TOOL_SECONDS,
        }

    def _budget_line(self) -> str:
        b = self.budget()
        return (f"\n[budget] {b['seconds_left']:.0f} s left of this session's "
                f"{b['session_seconds']} s (iteration {b['iteration']}); "
                f"{b['score_cycles_left']} measured score_cycles left this "
                f"iteration. Typical cost: check_bit_exact "
                f"{TOOL_SECONDS['check_bit_exact']} s, run_functional_check "
                f"{TOOL_SECONDS['run_functional_check']} s, score_cycles "
                f"{TOOL_SECONDS['score_cycles']} s; an edit or a read is free.")

    def _metered(self, method):
        """Attach the remaining budget to a tool's result.

        A text tool gets a trailing `[budget]` line; a tool that returns the
        evaluator's JSON gets a `budget` object inside it, because
        `test_harness` and the loop both `json.loads` those and a trailing line
        would make the verdict unparseable. `functools.wraps` is what keeps
        FastMCP deriving the schema from the real signature, so this wrapper is
        invisible to the tool surface.
        """
        def attach(result):
            if not isinstance(result, str):
                return result
            text = result.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    verdict = json.loads(text)
                except json.JSONDecodeError:
                    return result + self._budget_line()
                if isinstance(verdict, dict):
                    verdict["budget"] = self.budget()
                    return json.dumps(verdict, indent=1)
            return result + self._budget_line()

        if inspect.iscoroutinefunction(method):
            @functools.wraps(method)
            async def wrapper(*args, **kwargs):
                return attach(await method(*args, **kwargs))
        else:
            @functools.wraps(method)
            def wrapper(*args, **kwargs):
                return attach(method(*args, **kwargs))
        return wrapper

    # -- Variant bookkeeping. Deliberately *not* MCP tools: the search harness
    # -- accepts or rewinds a candidate, the agent does not get to choose.
    def snapshot(self) -> dict[str, bytes]:
        return {name: path.read_bytes() for name, path in self.sources.items()}

    def restore(self, snapshot: dict[str, bytes]) -> None:
        for name, content in snapshot.items():
            self.sources[name].write_bytes(content)

    def diff_against(self, snapshot: dict[str, bytes]) -> str:
        chunks = []
        for name, path in self.sources.items():
            before = snapshot.get(name, b"").decode("utf-8").splitlines(keepends=True)
            after = path.read_text(encoding="utf-8").splitlines(keepends=True)
            chunks.extend(difflib.unified_diff(before, after, f"a/{name}", f"b/{name}"))
        return "".join(chunks)

    def evaluate(self, gate_only: bool = False, shapes: str | None = None,
                 work: str = "agent", gate_tier: str = "full") -> dict:
        """Run the frozen evaluator on the spec dir; return its JSON verdict.

        `work` names a subdirectory of the work dir. The agent's MCP calls run
        in the tool's actor and the loop's own re-scoring runs in the driver --
        two copies of this object, two locks -- so they get separate
        directories: an agent evaluation still running after its opencode call
        timed out must not be wiped by the harness's next one.

        `gate_tier` DEFAULTS TO FULL, and the default is what matters: every
        caller that decides anything -- `loop.py`'s `evaluate(work="harness")`,
        `accept.py`, `control.py`, `score_cycles` -- takes the default and so
        runs the whole gate. Only the two fast tools pass anything else, and
        the evaluator itself refuses a cheaper tier on a scoring run, so a
        cheap tier cannot reach a cycle count at all."""
        cmd = [sys.executable, self.evaluator, "--spec-dir", str(self.spec_dir),
               "--work", str(self.work_dir / work)]
        if self.codesign:
            cmd.append("--codesign")
        if gate_only:
            cmd.append("--gate-only")
        if gate_tier != "full":
            cmd += ["--gate-tier", gate_tier]
        if shapes:
            cmd += ["--shapes", shapes]
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(self.eval_env)
        with self._lock:  # one Vitis project per tool; never two at once
            p = subprocess.run(cmd, cwd=self.repo, env=env, text=True,
                               capture_output=True, timeout=5400)
        out = (p.stdout + p.stderr).strip()
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    verdict = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            verdict = {"ok": False, "stage": "evaluator"}
        if not verdict.get("ok"):
            verdict["detail"] = out[-6000:]
        return verdict

    # -- MCP tools ---------------------------------------------------------
    def _check(self, name: str, source: str) -> str | None:
        # `isa_spec.json` is data, not a module: the policy parses it as JSON
        # and holds the expressions its actions carry -- the strings `isa_ref`
        # evaluates -- to the same expression language `isa_ref` admits.
        if not name.endswith(".json"):
            try:
                ast.parse(source, filename=name)
            except SyntaxError as error:
                return (f"Rejected: the edit leaves {name} unparseable -- "
                        f"{error.msg} at line {error.lineno}. The file is "
                        f"unchanged.")
        base = subprocess.run(
            ["git", "show", f"HEAD:examples/tinytpu/{name}"],
            cwd=self.repo, capture_output=True, text=True).stdout
        problems = policy_violations(name, source) + (
            doc_violations(name, base, source) if base else [])
        if problems:
            return ("Rejected: the edit is outside what a spec may contain -- "
                    + "; ".join(problems)
                    + ". Describe hardware; do not read, write, or execute. "
                    "The file is unchanged.")
        return None

    def read_spec(self, path: str = "", start_line: int = 0,
                  max_lines: int = 0) -> str:
        """Read one writable file (or a line range of it), or list them all.

        With no argument: every writable path with its line count -- the eight
        units under `ip/units/`, the composition `ip/tinytpu.py`, the ISA
        `ip/isa.py`, the assembler, the programs, `microarch_isa.py` (the
        parameter set), `isa_dsl.py` (the program generator) and the
        instruction set itself -- `isa_spec.json` (the source of truth),
        `isa_encoding.py` (generated from it: change it with `regenerate_isa`,
        never by hand) and `isa_ref.py` (the reference model). With a path:
        that file, whole by default.

        `start_line` (1-based) and `max_lines` read a WINDOW instead. Prefer
        one once you know where you are working: the two big files are 96 KB
        together, run 3's sessions read them 12-19 times each, and every one of
        those re-reads stays in the context of every later turn -- context
        growth was 48 % of a session's wall clock. A window costs the same
        milliseconds and a twentieth of the tokens. The header always names the
        range and the file's full length, so a window never reads as a whole
        file.

        `allo/compose.py` and `ip/params.py` are frozen machinery and are not
        writable; read them with `read_reference`.
        """
        if not path:
            rows = [f"  {rel:34s} "
                    f"{len(p.read_text(encoding='utf-8').splitlines()):4d} lines"
                    + ("   (a unit)" if rel in UNITS else "")
                    for rel, p in self.sources.items()]
            return ("Writable files (read one with read_spec(path=...), or a "
                    "window with read_spec(path=..., start_line=..., "
                    "max_lines=...)):\n" + "\n".join(rows))
        target = self.sources.get(path)
        if target is None:
            return f"Unknown path; writable files are {list(self.sources)}."
        source = target.read_text(encoding="utf-8")
        if not (start_line or max_lines):
            return f"===== {path} =====\n{source}"
        lines = source.splitlines()
        start = max(1, int(start_line or 1))
        chunk = lines[start - 1: start - 1 + max(1, int(max_lines or 400))]
        return (f"===== {path} lines {start}-{start + len(chunk) - 1} of "
                f"{len(lines)} =====\n" + "\n".join(chunk))

    def read_reference(self, name: str, start_line: int = 1,
                       max_lines: int = 400) -> str:
        """Read a FROZEN file for context (you cannot edit these).

        ``name`` is one of: mapspace.py (THE MAPPER -- its exhaustive
        enumerator, the refusal histogram, and the rule by which it picks the
        nest that gets cosimmed; read this first in co-design mode),
        codesign_gate.py / codesign_cosim.py (how the mapper's choice reaches
        the RTL), act.rst (why the mapper looks like this, and the measured
        refusal counts on the shipped design), cosim.py (the RTL cosim scorer and its testbench),
        bench_isa.py (the published-setup functional check), stress_isa.py
        (the correctness gate: 492 runs at 476a70d8, full-range operands, 64 shapes, whole
        C compared, vector and random programs), isa_ref.py (what each
        instruction means -- the reference stress_isa checks against),
        evaluate.py (how the score is computed), param_check.py (the
        parametricity gate: the design rebuilt at MAXDIM 8 and 12 must be exact), tinytpu_isa.rst (the design,
        its ISA, and how to verify a change), tinytpu_history.rst (what was
        tried, measured, and reverted), gemmini_comparison.rst (the Gemmini
        comparison and where the gap comes from), limitations.rst (Allo
        frontend/simulator limitations and workarounds), timeline_16x16x16 /
        timeline_16x16x16.txt / timeline_16x16x16_rle.txt (the shipped
        design's measured per-process cosim timeline at 16x16x16). Returns
        ``max_lines`` lines from ``start_line`` (1-based).
        """
        rel = REFERENCE.get(name)
        if rel is None:
            return f"Unknown reference; choose one of {sorted(REFERENCE)}."
        p = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=self.repo,
                           capture_output=True, text=True)
        lines = p.stdout.splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1: start - 1 + max(1, min(int(max_lines), 1200))]
        return (f"{name}: lines {start}-{start + len(chunk) - 1} of {len(lines)}\n"
                + "\n".join(chunk))

    #: Lines of context diff an edit reports back. Enough to see the edit in
    #: place, short enough that it is never a reason to re-read the file.
    _DIFF_LINES = 40

    def _edit_report(self, path: str, before: str, after: str) -> str:
        """The edit, shown in place, so nothing has to re-read the file.

        A whole-file re-read after every edit was the second-largest source of
        context growth after `read_spec`, and it is entirely avoidable: the
        file changed exactly where the tool changed it, and this says where
        with line numbers.
        """
        diff = list(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            f"a/{path}", f"b/{path}", lineterm="", n=3))
        if not diff:
            return "  (the file is byte-identical; nothing changed)"
        body = diff[:self._DIFF_LINES]
        more = ("" if len(diff) <= self._DIFF_LINES
                else f"\n  ... {len(diff) - self._DIFF_LINES} more diff lines")
        return "\n".join(body) + more

    def replace_text(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of ``old`` with ``new`` in ``path``.

        The preferred edit. ``path`` is one of the writable paths
        ``read_spec()`` lists (e.g. ``ip/units/pe.py``); ``old``
        must occur exactly once (include enough surrounding lines to make it
        unique, with exact indentation). The result must parse and pass the
        spec policy, or nothing is written.

        On success it returns the edit AS A DIFF, with line numbers. That is
        the whole file you need to see; do not re-read the file to check that
        the edit landed.
        """
        target = self.sources.get(path)
        if target is None:
            return f"Rejected: path must be one of {list(self.sources)}."
        source = target.read_text(encoding="utf-8")
        n = source.count(old) if old else 0
        if n != 1:
            return (f"Rejected: `old` occurs {n} times in {path}; it must occur "
                    f"exactly once. The file is unchanged.")
        updated = source.replace(old, new, 1)
        broken = self._check(path, updated)
        if broken:
            return broken
        target.write_text(updated, encoding="utf-8")
        return (f"Replaced 1 occurrence in {path}.\n"
                + self._edit_report(path, source, updated))

    def apply_spec_patch(self, patch: str) -> str:
        """Apply a unified diff touching only the writable files.

        Use git-style headers ``--- a/<path>`` / ``+++ b/<path>`` with the
        paths ``read_spec()`` lists (e.g. ``ip/units/pe.py``). Any other path
        is rejected: the evaluator, testbench, shapes, golden reference, Vitis
        settings and the composition machinery are frozen.
        """
        headers = [l for l in patch.splitlines() if l.startswith(("--- ", "+++ "))]
        if not headers or len(headers) % 2:
            return "Rejected: patch needs paired git-style file headers."
        paths = []
        for old, new in zip(headers[::2], headers[1::2]):
            if not old.startswith("--- a/") or not new.startswith("+++ b/"):
                return "Rejected: use matching a/<file> and b/<file> headers."
            old_path = old.removeprefix("--- a/").split("\t")[0].strip()
            new_path = new.removeprefix("+++ b/").split("\t")[0].strip()
            if old_path != new_path or old_path not in self.sources:
                return (f"Rejected: patches may touch only "
                        f"{list(self.sources)}. Everything else is frozen.")
            paths.append(old_path)
        if len(paths) != len(set(paths)):
            return "Rejected: each writable file may appear only once per patch."
        with tempfile.TemporaryDirectory(prefix="tinytpu-isa-patch-") as tmp:
            sandbox = Path(tmp)
            for name, source in self.sources.items():
                (sandbox / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, sandbox / name)
            applied = subprocess.run(
                ["patch", "--batch", "--forward", "--no-backup-if-mismatch", "-p1"],
                input=patch, text=True, cwd=sandbox, capture_output=True)
            if applied.returncode:
                return f"Rejected: patch does not apply.\n{applied.stdout}{applied.stderr}"
            patched = {name: (sandbox / name).read_bytes() for name in paths}
            for name, content in patched.items():
                broken = self._check(name, content.decode("utf-8"))
                if broken:
                    return broken
            for name, content in patched.items():
                self.sources[name].write_bytes(content)
        return f"Patch applied to {', '.join(paths)}."

    def insert_after(self, path: str, anchor: str, content: str) -> str:
        """Insert text after the line containing one unique anchor.

        ``path`` is one of the writable paths ``read_spec()`` lists. Prefer a
        unified diff when a change must update several files together. On
        success it returns the edit as a diff, so the file need not be re-read.
        """
        target = self.sources.get(path)
        if target is None:
            return f"Rejected: path must be one of {list(self.sources)}."
        source = target.read_text(encoding="utf-8")
        if source.count(anchor) != 1:
            return f"Rejected: anchor must occur exactly once in {path}."
        cut = source.index(anchor) + len(anchor)
        line_end = source.find("\n", cut)
        cut = len(source) if line_end == -1 else line_end + 1
        updated = source[:cut] + content + source[cut:]
        broken = self._check(path, updated)
        if broken:
            return broken
        target.write_text(updated, encoding="utf-8")
        return (f"Content inserted into {path}.\n"
                + self._edit_report(path, source, updated))

    # The two evaluator tools are async and push the work to a thread: a sync
    # tool runs on the MCP server's event loop, and in the smoke run one hung
    # evaluation blocked every other request -- including tool listing for
    # the next session, which then reported that no tools existed.
    def regenerate_isa(self) -> str:
        """Regenerate `isa_encoding.py` from your `isa_spec.json`.

        Run this after every edit to `isa_spec.json`. `isa_encoding.py` is
        generated from the spec, and `run_functional_check` refuses a candidate
        whose generated module is not byte-identical to what its own spec
        produces (`gen_isa.py --conform`), so a spec edit on its own is always
        a refusal. The generator is FROZEN and is read from git, not from your
        spec directory: it writes what your spec says, and you cannot change
        what "says" means.

        Nothing else is touched, and the spec itself is never rewritten.
        """
        gen = subprocess.run(["git", "show", f"HEAD:examples/tinytpu/gen_isa.py"],
                             cwd=self.repo, capture_output=True, text=True)
        if gen.returncode:
            return f"Rejected: cannot read the frozen generator ({gen.stderr[:200]})."
        encoding = self.sources["isa_encoding.py"]
        with tempfile.TemporaryDirectory(prefix="tinytpu-isa-gen-") as tmp:
            # The whole spec directory, because `gen_isa.py` lists
            # `ip/units/` at import time -- a sandbox holding only the spec
            # raises before it reads a line of it.
            sandbox = Path(tmp) / "tinytpu"
            shutil.copytree(self.spec_dir, sandbox)
            (sandbox / "gen_isa.py").write_text(gen.stdout, encoding="utf-8")
            # `--write --no-doc` reads the spec and writes one file. It imports
            # nothing of the design, so it is fast and cannot run the
            # candidate's code.
            done = subprocess.run([sys.executable, "gen_isa.py", "--write",
                                   "--no-doc"], cwd=sandbox, text=True,
                                  capture_output=True, timeout=300)
            if done.returncode:
                return ("Rejected: the generator could not read your spec.\n"
                        + (done.stdout + done.stderr)[-2000:])
            produced = (sandbox / "isa_encoding.py").read_text(encoding="utf-8")
        before = encoding.read_text(encoding="utf-8")
        if produced == before:
            return ("isa_encoding.py already matches isa_spec.json; nothing "
                    "was written.")
        broken = self._check("isa_encoding.py", produced)
        if broken:
            return broken
        encoding.write_text(produced, encoding="utf-8")
        added = sum(1 for line in difflib.unified_diff(
            before.splitlines(), produced.splitlines()) if line.startswith("+"))
        removed = sum(1 for line in difflib.unified_diff(
            before.splitlines(), produced.splitlines()) if line.startswith("-"))
        return (f"Regenerated isa_encoding.py from isa_spec.json "
                f"(+{added} / -{removed} lines).")

    async def check_bit_exact(self) -> str:
        """THE CHEAPEST SIGNAL, ~16 s: the PyTorch oracle alone.

        The workload suite's MLPs run on your design and are compared byte for
        byte with `torch.nn.Linear`'s own forward on the same weights. It is
        the one check with something outside this repository on one side, so it
        is the check a rewritten reference model cannot satisfy -- and it is
        the first thing the full gate runs, which is why failing it here fails
        it there too.

        CALL IT FREELY. It is the tool this surface did not have: everything
        else costs 30 s or 4 minutes, and a debugging loop needs an answer in
        seconds.

        WHAT IT CANNOT SEE, and it is most of what matters:

        * NOTHING ABOUT SPEED. A pass says the edit is LEGAL, not that it is
          faster. There is no cycle count anywhere in this result and no
          proxy for one. An edit that passes here and doubles the cycle count
          passes here exactly the same.
        * not `gen_isa --conform`, so a stale `isa_encoding.py` still passes
          here and is refused by the gate;
        * not `bench_isa`, so the hand-written GEMM need not match;
        * not `stress_isa`, so full-range operands, prefilled C, the 64 shapes
          and the random programs are all unexercised;
        * not parametricity, so a change specialised to the scored
          configuration passes here;
        * not the RTL. This is the Allo simulator.

        Use it to find a mistake fast, `run_functional_check` to believe a
        candidate is correct, and `score_cycles` to learn whether it is
        faster."""
        verdict = await asyncio.to_thread(self.evaluate, True, None, "agent",
                                          "oracle")
        return json.dumps(verdict, indent=1)

    async def run_functional_check(self) -> str:
        """CORRECTNESS AT THE SCORED CONFIGURATION, ~30 s (measured; the
        docstring here used to say "~15 s" and the real number was 52-76 s).

        The PyTorch oracle, `gen_isa.py --conform` on your own spec,
        bench_isa.py (the published [-4, 4] setup) and stress_isa.py (492 runs:
        full-range/corner/boundary int8, all 64 shapes, C prefilled and
        compared in full, vector and random programs, many calls on one build)
        must all pass. Functional (Allo simulator), not RTL. A deadlocked
        dataflow fails after 4 minutes.

        IT COSTS HALF WHAT IT USED TO because the PARAMETRICITY SWEEP is no
        longer here: the design is not rebuilt at the other MAXDIMs or at the
        second T. That property is unchanged and still refuses a candidate --
        it has simply moved to the full gate, which every scored run and the
        harness's own independent verdict use. So a change specialised to
        T=4/MAXDIM=16 passes HERE and is rejected at `gate:param` when it is
        scored. Keep the design parametric; this tier will not tell you that
        you did not.

        It still says nothing about SPEED. Passing means legal, not faster."""
        verdict = await asyncio.to_thread(self.evaluate, True, None, "agent",
                                          "fast")
        return json.dumps(verdict, indent=1)

    async def score_cycles(self) -> str:
        """THE OBJECTIVE, ~4 min: the FULL gate, then Vitis HLS csynth + RTL
        C/RTL cosim at 4x4x4 and 16x16x16, each testbench bit-exact against
        numpy. The only tool that reports cycles.

        ONCE PER ITERATION. This is a cap, and a refusal names what is left.
        You do not need it as a hill-climbing oracle: the harness re-scores
        your final spec independently whether or not you call this, so a second
        measured score buys the search nothing and costs a quarter of your
        session. Measured on run 3: this tool was 37 % of the whole run.

        A repeat on an UNCHANGED spec is free and does not count -- the answer
        is remembered per exact spec content, so re-asking costs milliseconds
        and returns the identical verdict with `cached: true`. Edit, check with
        the cheap tools, and score once you believe the candidate.

        This tier runs the whole gate, including parametricity, so it is also
        where a change specialised to the scored configuration is refused.

        In co-design mode the program under the RTL is the best nest the FROZEN
        mapper can encode on your hardware, chosen by exhaustive enumeration --
        not the program you hand-wrote. The verdict reports a PAIR and never
        collapses it: `cycles` per shape from the cosim, and `resources` (FF,
        LUT, BRAM18K, DSP, URAM and the estimated clock, which must meet
        3.33 ns) from the csynth of the same build. Buying cycles with block RAM
        is a trade, not a win, and it is reported as one. `mapspace` records how
        many nests your hardware made encodable and which constraint refused
        the rest."""
        key = self.spec_hash()
        state = self._state()
        hit = state["cache"].get(key)
        if hit is not None:
            # Free, and identical: the same bytes judged by the same frozen
            # referee. Reported as a cache hit so it is never mistaken for a
            # fresh measurement.
            return json.dumps(dict(hit, cached=True), indent=1)
        if int(state["scored"]) >= SCORE_CALLS_PER_ITERATION:
            return json.dumps({
                "ok": False, "stage": "budget",
                "detail": (
                    f"score_cycles is capped at {SCORE_CALLS_PER_ITERATION} "
                    f"measured call per iteration and this iteration has used "
                    f"it. Re-asking on an unchanged spec is free; this spec has "
                    f"changed since, so a fresh cosim would cost about "
                    f"{TOOL_SECONDS['score_cycles']} s of a "
                    f"{self.session_budget_s} s session. You do not need it: "
                    f"the harness re-scores your final spec independently "
                    f"either way. Use run_functional_check "
                    f"(~{TOOL_SECONDS['run_functional_check']} s) and "
                    f"check_bit_exact (~{TOOL_SECONDS['check_bit_exact']} s) to "
                    f"finish the candidate, then say what you changed."),
            }, indent=1)
        # Charged BEFORE the run, not after: a cosim that times out or is
        # killed with the session still spent the minutes.
        self._put_state(dict(state, scored=int(state["scored"]) + 1))
        verdict = await asyncio.to_thread(self.evaluate)
        if verdict.get("ok"):
            state = self._state()
            # Only a successful verdict is remembered. A rejection is cheap to
            # reproduce and usually means the spec is about to change anyway.
            state["cache"][key] = verdict
            self._put_state(state)
        return json.dumps(verdict, indent=1)

    async def mapspace_report(self) -> str:
        """The co-design signal, ~60 s and no Vitis: the fast gate plus the
        exhaustive mapspace enumeration against your hardware.

        Reports, per scored shape, how many of the enumerated loop nests this
        hardware can ENCODE, which constraint refused each of the rest, and
        which nest the frozen mapper would choose. Use it to see whether a
        hardware change actually widened what the machine can say, before
        paying for a cosim. It is a count of nests, not a cycle count: nothing
        here is a performance number, and a wider mapspace is a hypothesis
        about speed, not a measurement of it."""
        verdict = await asyncio.to_thread(self.evaluate, True, None, "agent",
                                          "fast")
        out = {k: verdict.get(k) for k in ("ok", "stage", "mapspace", "gate")}
        if not verdict.get("ok"):
            out["detail"] = verdict.get("detail", "")[-4000:]
        return json.dumps(out, indent=1)

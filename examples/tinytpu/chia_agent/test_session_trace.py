# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a CHIA session leaves behind while it is running -- and when it is not
allowed to finish. Costs $0: no Vertex call, no Ray, no tool server.

The gap this covers. A 40-minute session reported nothing until it returned,
and a session killed at `timeout_seconds` returned nothing at all: no session
id, no turn count, `$0` on `usage`. Run 3 had four such calls, each a full
2400 s. So the cases worth testing are the ones a passing test would skip --
the call that is killed, and the call that hits the wall.

  t1  a session that finishes     the trace names every turn and every tool
                                  call as it happens, and its turn count
                                  agrees with opencode's own `usage`
  t2  a session KILLED mid-flight the trace written up to the kill is complete
      (SIGKILL to the process      and readable, the raw opencode stream is
      group, no chance to flush)   still on disk, and the report names what
                                  was in flight when the process died
  t3  a session that hits the     `model_call.timeout` records how many turns
      wall (`timeout_seconds`)    and tool calls preceded the wall, and the
                                  raw stream survives -- CHIA's `_capture`
                                  unlinks it in a `finally`, which is why a
                                  timed-out call used to report nothing
  t4  a tool call interrupted     `tool.start` with no `tool.end` is what the
                                  tool server leaves, naming the tool that
                                  never came back
  t5  durability                  every event a SIGKILLed process logged is
                                  present and parses; nothing is lost to a
                                  buffer and no line is torn

How the model is faked: `fake_model.py` is an OpenAI-compatible endpoint that
replays a script, and opencode talks to it through the same
`@ai-sdk/openai-compatible` provider it uses for Vertex AI. The scripted
replies are made SLOW here, because a long session is exactly what is being
tested: a model that does not come back is what a 2400 s timeout looks like.

The MCP tool server is deliberately NOT started. It binds ports from 8000
upward and a paid run holds them; `traced()` is exercised directly instead
(t4), and `test_harness.py --phases loop` covers it over real MCP.

    python test_session_trace.py            # ~2 minutes, $0
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

import session_trace
from session_trace import Trace, load, report, traced

#: The harness installs opencode under the agent directory (`npm ci`); a
#: worktree without `node_modules/` can point at another checkout's copy.
OPENCODE_BIN = os.environ.get(
    "TINYTPU_OPENCODE_BIN", str(AGENT_DIR / "node_modules/.bin/opencode"))

RESULTS: list[tuple[str, bool]] = []


def check(case: str, expected: str, actual, passed: bool) -> None:
    RESULTS.append((case, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {case}")
    if not passed:
        print(f"         expected: {expected}")
        print(f"         actual:   {actual}")


# -- the child: one opencode session, driven by the scripted model ------------

def session_child() -> int:
    """Run ONE opencode session and exit. Invoked as a subprocess.

    `_run_opencode` rather than `prompt`: `prompt` is a `ChiaFunction`, whose
    profiler auto-initialises Ray and would join the live cluster. Everything
    this file tests -- `_capture`, the tailer, the trace -- is below that line.
    """
    url, trace_dir, work = sys.argv[2], sys.argv[3], sys.argv[4]
    from chia.models.opencode import AdditionalModelProvider
    from llm import IsaOpenCodeLLM

    llm = IsaOpenCodeLLM(
        model="scripted/replay", system_message="You are a test.",
        timeout_seconds=int(os.environ.get("TRACE_TEST_TIMEOUT", "600")),
        retries=1,
        additional_providers=[AdditionalModelProvider(
            id="scripted", models=["replay"], base_url=url, api_key="unused")],
        # opencode's own `read`, allowed only here: it gives the session real
        # tool calls without an MCP server on a port a paid run may hold.
        config={"*": "deny", "read": "allow"},
        opencode_bin=OPENCODE_BIN, work_dir=work,
        extra_cli_args=["--title", "trace-test [w0]"])
    llm.trace_dir, llm.trace_label = trace_dir, "trace-test [w0]"
    try:
        result = llm._run_opencode("Read the notes, then say what you found.", [])
    except subprocess.TimeoutExpired:
        print("CHILD timeout", flush=True)
        return 3
    print("CHILD " + json.dumps({"rc": result.returncode,
                                 "session": result.session_id,
                                 "usage": result.usage}, default=str), flush=True)
    return 0


def start_session(fake_url, trace_dir: Path, work: Path, timeout_s: int):
    """Spawn the child in its OWN process group, so it can be killed whole.

    A SIGKILL to the driver alone would orphan opencode; killing the group is
    both the honest model of a killed run and what leaves no orphan behind.
    """
    env = dict(os.environ, TRACE_TEST_TIMEOUT=str(timeout_s))
    env.pop("RAY_ADDRESS", None)     # never join a live cluster from a test
    return subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--session",
         fake_url, str(trace_dir), str(work)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        start_new_session=True)


def wait_for(trace_path: Path, predicate, limit=180):
    """Poll the trace WHILE the session runs -- which is the whole point: a
    reader must be able to answer 'what is it doing now', not only 'what did
    it do'. Returns the events seen, or [] if the deadline passed."""
    deadline = time.time() + limit
    while time.time() < deadline:
        if trace_path.exists():
            events = load(trace_path)
            if predicate(events):
                return events
        time.sleep(0.5)
    return []


def script(work: Path, steps: int = 4):
    notes = work / "notes.txt"
    return [[("read", {"filePath": str(notes)})] * steps
            + [("say", "The notes say forty-two.")]]


def slow_fake(scripts, seconds: float):
    """The scripted model, made slow: `seconds` of thinking per reply."""
    from fake_model import FakeModel
    fake = FakeModel(scripts).start()
    inner = fake.respond
    fake.respond = lambda body: (time.sleep(seconds), inner(body))[1]

    def handle_error(request, client_address):
        # Killing opencode mid-stream breaks the connection under the
        # endpoint. That is the test working; anything else is not.
        error = sys.exc_info()[1]
        if not isinstance(error, (BrokenPipeError, ConnectionResetError)):
            fake.errors.append(repr(error))

    fake.server.handle_error = handle_error
    return fake


# -- t1 / t2: a finished session, then one that is killed ---------------------

def case_finished(root: Path) -> None:
    print("== t1: a session that finishes", flush=True)
    work, trace_dir = _workspace(root, "finished")
    fake = slow_fake(script(work, 2), 2.0)
    proc = start_session(fake.url, trace_dir, work, 600)
    out = proc.communicate(timeout=300)[0]
    fake.stop()
    events = load(trace_dir / "trace.jsonl")
    kinds = [e["kind"] for e in events]
    turns = [e for e in events if e.get("part") == "step-start"]
    tools = [e for e in events
             if e.get("part") == "tool" and e.get("status") == "completed"]
    end = next((e for e in events if e["kind"] == "model_call.end"), {})
    reported = json.loads(out.split("CHILD ", 1)[1].splitlines()[0]) \
        if "CHILD {" in out else {}
    check("t1.turns-match-opencode",
          f"trace turn count == usage.num_turns ({(reported.get('usage') or {}).get('num_turns')})",
          f"trace {len(turns)} turns, end-event says {end.get('turns')}",
          len(turns) == (reported.get("usage") or {}).get("num_turns")
          and end.get("turns") == len(turns))
    check("t1.tool-calls-named-and-timed",
          "every tool call with its name and its duration",
          [(t.get("tool"), t.get("seconds")) for t in tools],
          len(tools) == 2 and all(t.get("tool") == "read" for t in tools)
          and all(isinstance(t.get("seconds"), (int, float)) for t in tools))
    check("t1.wall-time-accounted",
          "model_call.end carries the call's wall time and the session id",
          {k: end.get(k) for k in ("seconds", "session_id", "tools")},
          isinstance(end.get("seconds"), (int, float)) and end.get("seconds") > 0
          and bool(end.get("session_id")) and end.get("tools") == {"read": 2})
    check("t1.raw-stream-kept",
          "opencode's own event stream kept beside the trace",
          sorted(p.name for p in (trace_dir / "opencode").glob("*.ndjson")),
          any((trace_dir / "opencode").glob("*.ndjson")))
    print(f"  ({len(events)} events, {len(turns)} turns, {len(tools)} tool calls)")
    print("   " + "\n   ".join(report(events).splitlines()[-4:]))


def case_killed(root: Path) -> None:
    """THE case. A session killed with no chance to clean up or flush."""
    print("== t2: a session KILLED mid-flight (SIGKILL, no cleanup)", flush=True)
    work, trace_dir = _workspace(root, "killed")
    trace_path = trace_dir / "trace.jsonl"
    fake = slow_fake(script(work, 6), 4.0)
    proc = start_session(fake.url, trace_dir, work, 2400)
    # Wait until the session is demonstrably under way -- two turns in -- and
    # kill it there. Reading the trace to decide WHEN to kill is itself the
    # claim under test: the record is usable while the process is alive.
    seen = wait_for(trace_path,
                    lambda ev: sum(1 for e in ev
                                   if e.get("part") == "step-start") >= 2)
    alive = bool(seen)
    killed_at = time.time()
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=60)
    fake.stop()

    raw = sorted((trace_dir / "opencode").glob("*.ndjson"))
    text = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
    lines = [l for l in text.splitlines() if l.strip()]
    parsed, torn = [], 0
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    turns = [e for e in parsed if e.get("part") == "step-start"]
    tools = [e for e in parsed if e.get("part") == "tool"]

    check("t2.killed-before-finishing",
          "the process died mid-session (no model_call.end)",
          f"returncode {proc.returncode}, "
          f"{[e['kind'] for e in parsed].count('model_call.end')} end events",
          alive and proc.returncode in (-signal.SIGKILL, 137)
          and not any(e["kind"] == "model_call.end" for e in parsed))
    check("t2.trace-survives",
          ">= 2 turns and >= 1 tool call recorded before the kill",
          f"{len(turns)} turns, {len(tools)} tool events, {len(lines)} lines",
          len(turns) >= 2 and len(tools) >= 1)
    check("t2.every-line-complete",
          "every line parses: one os.write per event, nothing buffered",
          f"{torn} torn line(s) of {len(lines)}", torn == 0 and len(lines) > 3)
    check("t2.written-as-it-happened",
          "the last event predates the kill by under 5 s",
          f"last event {killed_at - max(e['t'] for e in parsed):.1f}s "
          f"before the kill" if parsed else "no events",
          bool(parsed) and 0 <= killed_at - max(e["t"] for e in parsed) < 5)
    check("t2.raw-stream-survives",
          "opencode's own stream still on disk (CHIA unlinks it)",
          [p.name for p in raw],
          len(raw) == 1 and raw[0].stat().st_size > 0)
    summary = report(parsed)
    check("t2.report-names-what-was-in-flight",
          "the reader says the model call never returned",
          summary.splitlines()[-1],
          "STILL OPEN" in summary and "model_call" in summary.split(
              "STILL OPEN")[-1])
    print("  --- what the killed session left behind "
          "(python session_trace.py <log-dir>) ---")
    print("  " + "\n  ".join(summary.splitlines()))


# -- t3: the wall ------------------------------------------------------------

def case_timeout(root: Path) -> None:
    print("== t3: a session that hits `timeout_seconds`", flush=True)
    work, trace_dir = _workspace(root, "timeout")
    fake = slow_fake(script(work, 6), 4.0)
    # Short enough to fire after a turn or two, long enough that opencode has
    # started. The production value (2400 s) is NOT changed by this test.
    proc = start_session(fake.url, trace_dir, work, 20)
    out = proc.communicate(timeout=300)[0]
    fake.stop()
    events = load(trace_dir / "trace.jsonl")
    wall = next((e for e in events if e["kind"] == "model_call.timeout"), None)
    raw = sorted((trace_dir / "opencode").glob("*.ndjson"))
    check("t3.timeout-propagates",
          "TimeoutExpired reaches the caller, as before",
          out.strip().splitlines()[-1:] or out, "CHILD timeout" in out)
    check("t3.timeout-is-recorded",
          "model_call.timeout naming the wall, the turns and the tool calls",
          wall, bool(wall) and wall.get("timeout_s") == 20
          and wall.get("turns", 0) >= 1)
    check("t3.raw-stream-survives-the-wall",
          "the transcript CHIA's `_capture` would have unlinked",
          [p.name for p in raw], len(raw) == 1 and raw[0].stat().st_size > 0)
    if wall:
        print(f"  ({wall['turns']} turns, {wall['tool_calls']} tool calls in "
              f"{wall['seconds']}s before the {wall['timeout_s']}s wall)")


# -- t4 / t5: the tool server's side, and durability --------------------------

def case_tool_span(root: Path) -> None:
    print("== t4: a tool call interrupted", flush=True)
    path = root / "tool" / "trace.jsonl"
    trace = Trace(path)

    def score_cycles() -> str:
        """The objective."""
        return json.dumps({"ok": True, "total_cycles": 1802})

    wrapped = traced(trace, "tinytpu_score_cycles", score_cycles)
    check("t4.schema-preserved",
          "name and docstring survive the wrapper (FastMCP builds the tool "
          "schema from them)",
          (wrapped.__name__, wrapped.__doc__),
          wrapped.__name__ == "score_cycles" and wrapped.__doc__ == "The objective.")
    wrapped()

    # A tool call that never returns: the span is opened, and the process dies
    # inside it. `os._exit` skips every finally, atexit and buffer flush --
    # the most hostile ending a log can be asked to survive.
    code = (f"import sys, os; sys.path.insert(0, {str(AGENT_DIR)!r});"
            f"from session_trace import Trace, traced;"
            f"t = Trace({str(path)!r});"
            f"f = traced(t, 'tinytpu_run_functional_check',"
            f"           lambda: os._exit(9)); f()")
    dead = subprocess.run([sys.executable, "-c", code])
    events = load(path)
    starts = [e for e in events if e["kind"] == "tool.start"]
    ends = [e for e in events if e["kind"] == "tool.end"]
    summary = report(events)
    check("t4.completed-call-has-both-ends",
          "tool.start and tool.end for the call that returned",
          [(e["kind"], e.get("tool")) for e in events],
          any(e.get("tool") == "tinytpu_score_cycles" for e in ends))
    check("t4.interrupted-call-names-itself",
          "tool.start with no tool.end for the call that did not",
          f"{len(starts)} starts, {len(ends)} ends, exit {dead.returncode}",
          len(starts) == 2 and len(ends) == 1
          and "tinytpu_run_functional_check" in summary.split("STILL OPEN")[-1])
    check("t4.result-head-recorded",
          "the first line of the result, so a refusal is told from a score",
          [e.get("head") for e in ends],
          any("total_cycles" in (e.get("head") or "") for e in ends))


def case_durability(root: Path) -> None:
    print("== t5: durability under SIGKILL", flush=True)
    path = root / "durable" / "trace.jsonl"
    code = (f"import sys, time; sys.path.insert(0, {str(AGENT_DIR)!r});"
            f"from session_trace import Trace;"
            f"t = Trace({str(path)!r});"
            f"[t.event('beat', n=i, pad='x' * 400) for i in range(500)];"
            f"print('written', flush=True); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-u", "-c", code],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "written"
    proc.kill()
    proc.wait(timeout=30)
    events = load(path)
    raw = path.read_text().splitlines()
    check("t5.nothing-lost-to-a-buffer",
          "all 500 events present after SIGKILL",
          f"{len(events)} parsed of {len(raw)} lines",
          len(events) == 500 and len(raw) == 500)
    check("t5.ordered-and-intact",
          "in order, none torn",
          f"first {events[0].get('n')} last {events[-1].get('n')}" if events else "none",
          bool(events) and [e["n"] for e in events] == list(range(500)))


def _workspace(root: Path, name: str) -> tuple[Path, Path]:
    work = root / name / "work"
    trace_dir = root / name / "log"
    work.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (work / "notes.txt").write_text("the notes say: forty-two\n")
    return work, trace_dir


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--session":
        return session_child()
    root = Path(os.environ.get("TRACE_TEST_DIR")
                or f"/tmp/tinytpu-trace-test-{os.getpid()}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"scratch: {root}\nopencode: {OPENCODE_BIN}\n")
    case_tool_span(root)
    case_durability(root)
    if Path(OPENCODE_BIN).exists():
        case_finished(root)
        case_killed(root)
        case_timeout(root)
    else:
        # Named, not skipped silently: without opencode there is no session to
        # kill, and a run that reports all-pass here would be lying.
        check("opencode-installed",
              f"{OPENCODE_BIN} (npm ci --prefix chia_agent), or "
              "TINYTPU_OPENCODE_BIN", "missing", False)
    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed"
          + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

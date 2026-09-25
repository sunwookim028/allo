"""Per-turn observability for a CHIA session: one line per event, written AS
IT HAPPENS, so a session killed at the timeout still leaves a usable record.

The gap this closes. A session used to report nothing until it returned, and a
session killed at `timeout_seconds` returned nothing at all -- not a turn
count, not a tool call, not a session id, and `$0` on `usage`. Run 3 spent 167
minutes of which cosim was 661 s (7 %), and four model calls each ran the full
2400 s wall with no record of what they did in it. The calls we most need to
understand are exactly the ones that reported nothing.

The record already existed and was being deleted. opencode's `run --format
json` emits newline-delimited events LIVE -- one per message part, per step,
per tool call. CHIA's `OpenCodeLLM._capture` streams them to a
`NamedTemporaryFile` and then `os.unlink`s it in a `finally`, so the stream
survives only when the call returns; on `subprocess.TimeoutExpired` the one
authoritative transcript is destroyed on the way out. `llm.py` keeps it
instead, and tails it into this log while the call runs.

Three writers, one file per worker (`<log-dir>/trace.jsonl`):

  llm.py        `model_call.*`, `turn`, `stream` -- opencode's own event
                stream, normalised, plus the raw stream kept beside it
  allo_tool.py  `tool.start` / `tool.end` -- every MCP round trip the tool
                server serves: name, arguments, duration, outcome. The tool
                server is the only component that sees a round trip from the
                far side of the MCP boundary, so its timings hold even when
                opencode's stream shape changes under us.
  loop.py       `call.*`, `iteration.*` -- the loop's own spans, so a reader
                can divide the wall clock into model, tools and harness.

Durability. Every event is one `os.write` on an `O_APPEND` descriptor: one
syscall, nothing buffered inside the process, and the kernel holds it the
moment the call returns. A `SIGKILL` therefore loses nothing already logged --
a buffered log is no better than no log when the process is killed. O_APPEND
also makes the file safe to share between the driver, the Ray actor that hosts
the tool server, and the worker that runs `prompt`.

Reading one back::

    python session_trace.py <log-dir>            # or <log-dir>/trace.jsonl
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

#: Set by `loop.py` for the processes it starts; the tool actor is told
#: explicitly instead, because a Ray actor inherits the raylet's environment
#: rather than the shell that launched the loop.
ENV_DIR = "TINYTPU_TRACE_DIR"

#: Arguments are digested, never copied: a `replace_text` call carries whole
#: file bodies and the log must stay readable.
_ARG_CHARS = 120


class Trace:
    """An append-only JSONL event log. Never raises; never buffers.

    A `Trace(None)` is a working no-op, so every call site can be written
    without a conditional and a run with no log directory still works.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._fd = None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(
                    self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            except OSError:
                self._fd = None  # logging must never fail a run

    # Picklable: the fd is reopened in whichever process unpickles it, which
    # is how the same file serves the driver, the tool actor and the worker
    # that runs `prompt`.
    def __getstate__(self):
        return {"path": str(self.path) if self.path else None}

    def __setstate__(self, state):
        self.__init__(state.get("path"))

    def __repr__(self):
        return f"Trace({self.path!s})"

    def event(self, kind: str, **fields) -> None:
        if self._fd is None:
            return
        record = {"t": round(time.time(), 3), "kind": kind, "pid": os.getpid()}
        record.update({k: v for k, v in fields.items() if v is not None})
        try:
            line = json.dumps(record, default=str) + "\n"
        except (TypeError, ValueError):
            line = json.dumps({"t": record["t"], "kind": kind,
                               "unserialisable": True}) + "\n"
        try:
            os.write(self._fd, line.encode("utf-8", "replace"))
        except OSError:
            pass

    @contextmanager
    def span(self, kind: str, **fields):
        """`<kind>.start` now, `<kind>.end` when the body leaves.

        Yields a dict the body may fill in; it is merged into the end record.
        A start with no end is the signal that matters: it is what a killed
        process leaves behind, and it names what it was doing when it died.
        """
        started = time.time()
        self.event(f"{kind}.start", **fields)
        extra: dict = {}
        try:
            yield extra
        except BaseException as error:
            self.event(f"{kind}.end", seconds=round(time.time() - started, 2),
                       ok=False, error=f"{type(error).__name__}: {error}"[:300],
                       **fields, **extra)
            raise
        self.event(f"{kind}.end", seconds=round(time.time() - started, 2),
                   ok=True, **fields, **extra)

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def digest(value) -> str:
    """A short, honest rendering of a tool argument: shape, not content."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={digest(v)}" for k, v in value.items()) + "}"
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= _ARG_CHARS:
        return text
    return f"{text[:_ARG_CHARS]}... [{len(text)} chars]"


def traced(trace: Trace, name: str, fn):
    """Wrap one tool callable so its round trip is logged as it happens.

    `tool.start` goes out before the work and `tool.end` after it, each on its
    own line, so a tool call still running when the process dies leaves a
    `tool.start` with no `tool.end` -- which is precisely the evidence a
    killed session used to lose. Sync and async callables are both handled;
    `functools.wraps` keeps `__doc__`, `__name__` and `__wrapped__`, so
    FastMCP still derives the tool's schema from the real signature.
    """
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            with trace.span("tool", tool=name,
                            args=digest(kwargs) if kwargs else None) as extra:
                result = await fn(*args, **kwargs)
                extra["chars"] = len(result) if isinstance(result, str) else None
                extra["head"] = _head(result)
                return result
    else:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with trace.span("tool", tool=name,
                            args=digest(kwargs) if kwargs else None) as extra:
                result = fn(*args, **kwargs)
                extra["chars"] = len(result) if isinstance(result, str) else None
                extra["head"] = _head(result)
                return result
    return wrapper


def _head(result) -> str | None:
    """The first line of a tool result: enough to tell a refusal from a score."""
    if not isinstance(result, str):
        return None
    for line in result.splitlines():
        if line.strip():
            return line.strip()[:160]
    return None


# -- normalising opencode's live event stream --------------------------------

def normalise(event: dict) -> dict | None:
    """One opencode `run --format json` event -> the fields worth keeping.

    Tolerant on purpose. opencode's event shape is not ours and has changed
    between releases, so anything unrecognised is still recorded by its `type`
    -- and the raw stream is kept beside the trace regardless, so a normaliser
    that misses a shape costs detail, never the record.
    """
    if not isinstance(event, dict):
        return None
    etype = event.get("type")
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    out: dict = {"event": etype}
    ptype = part.get("type")
    if ptype:
        out["part"] = ptype
    if part.get("id"):
        out["part_id"] = part["id"]
    if ptype == "tool":
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        out["tool"] = part.get("tool")
        out["status"] = state.get("status")
        stamps = state.get("time") if isinstance(state.get("time"), dict) else {}
        if stamps.get("start") and stamps.get("end"):
            out["seconds"] = round((stamps["end"] - stamps["start"]) / 1000.0, 2)
        if isinstance(state.get("input"), dict):
            out["args"] = digest(state["input"])
        output = state.get("output")
        if output is not None:
            out["output_chars"] = len(
                output if isinstance(output, str) else json.dumps(output, default=str))
    elif ptype in ("text", "reasoning"):
        out["chars"] = len(part.get("text") or "")
    elif ptype == "step-finish":
        # Tokens and the stop reason, and deliberately NOT the step's cost.
        # opencode's `usage` money is not this project's money -- a call that
        # timed out was billed and reports $0 -- so spend comes from
        # opencode's database (`spend.py`) and from nowhere else. A cost here
        # would look authoritative and would not be; `test_harness.py`'s
        # "no money from usage" guard refuses any module that reads it.
        out["tokens"] = part.get("tokens")
        out["reason"] = part.get("reason")
    if etype == "error":
        error = event.get("error")
        if isinstance(error, dict):
            out["error"] = error.get("name")
            out["detail"] = digest(error.get("data") or {})
    return out


def material(record: dict) -> tuple:
    """What makes a normalised event worth a new line.

    opencode re-emits a part on every delta while it streams, so the raw
    stream carries thousands of near-identical events. A line goes out when
    the part is new, when its status moves, when it gains a duration or a
    cost, or when its text has grown by another 2000 characters -- which keeps
    the log both live and bounded.
    """
    return (record.get("event"), record.get("part"), record.get("part_id"),
            record.get("tool"), record.get("status"), record.get("error"),
            record.get("seconds") is not None,
            record.get("tokens") is not None,
            (record.get("chars") or 0) // 2000,
            (record.get("output_chars") or 0) // 2000)


class StreamNormaliser:
    """Turns the raw NDJSON stream into deduplicated trace lines, live."""

    #: A runaway session must not fill the disk; the raw stream is kept whole
    #: either way, so the cap costs nothing that cannot be recovered.
    MAX_EVENTS = 4000

    def __init__(self, trace: Trace, call_id: str):
        self.trace, self.call_id = trace, call_id
        self.seen: set = set()
        self.counted: set = set()   # part ids already counted as a tool call
        self.turns = 0
        self.tool_calls: dict[str, int] = {}
        self.emitted = 0
        self.session_id = None
        self.error = None

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict):
            sid = event.get("sessionID") or (
                (event.get("part") or {}).get("sessionID")
                if isinstance(event.get("part"), dict) else None)
            if sid and self.session_id is None:
                self.session_id = sid
                self.trace.event("session", call_id=self.call_id, session_id=sid)
        record = normalise(event)
        if record is None:
            return
        key = material(record)
        if key in self.seen:
            return
        self.seen.add(key)
        # Counters move on the FIRST sight of a part, so they are right even
        # if the process dies before the part completes.
        if record.get("part") == "step-start":
            self.turns += 1
            record["turn"] = self.turns
        if record.get("part") == "tool" and record.get("tool"):
            # Counted once per part, on FIRST sight -- a tool call that never
            # came back is still a tool call, and is the one we most want
            # counted.
            ident = record.get("part_id") or (record["tool"], len(self.counted))
            if ident not in self.counted:
                self.counted.add(ident)
                self.tool_calls[record["tool"]] = \
                    self.tool_calls.get(record["tool"], 0) + 1
        if record.get("error"):
            self.error = record["error"]
        if self.emitted < self.MAX_EVENTS:
            self.emitted += 1
            self.trace.event("stream", call_id=self.call_id, **record)
        elif self.emitted == self.MAX_EVENTS:
            self.emitted += 1
            self.trace.event("stream.capped", call_id=self.call_id,
                             after=self.MAX_EVENTS)

    def totals(self) -> dict:
        return {"turns": self.turns,
                "tool_calls": sum(self.tool_calls.values()),
                "tools": dict(sorted(self.tool_calls.items())) or None,
                "session_id": self.session_id,
                "stream_error": self.error}


# -- reading one back --------------------------------------------------------

def load(path) -> list[dict]:
    """Every complete line of a trace. A half-written final line is dropped.

    A killed process cannot leave a torn line -- each event is one `os.write`
    -- but a trace copied out of a live run can end mid-line, and a reader
    that dies on that is a reader nobody trusts.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "trace.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def report(events: list[dict]) -> str:
    """A human timeline: the turns, the tool calls, and where the wall went."""
    if not events:
        return "empty trace"
    t0 = min(e["t"] for e in events if "t" in e)
    lines = [f"{len(events)} events over "
             f"{max(e['t'] for e in events if 't' in e) - t0:.0f}s"]
    open_spans: dict[tuple, dict] = {}
    model_s = tool_s = model_tool_s = 0.0
    mcp_tools: dict[str, int] = {}
    model_tools: dict[str, int] = {}
    turns = 0
    for event in events:
        kind = event.get("kind", "")
        at = f"  t+{event.get('t', t0) - t0:7.1f}s"
        if kind.endswith(".start"):
            open_spans[(kind[:-6], event.get("call_id"), event.get("tool"))] = event
            what = event.get("tool") or event.get("what") or ""
            lines.append(f"{at}  {kind[:-6]:<12} {what} started"
                         + (f"  args={event['args']}" if event.get("args") else ""))
        elif kind.endswith(".end"):
            open_spans.pop(
                (kind[:-4], event.get("call_id"), event.get("tool")), None)
            seconds = event.get("seconds") or 0.0
            if kind.startswith("tool"):
                tool_s += seconds
                mcp_tools[event.get("tool", "?")] = \
                    mcp_tools.get(event.get("tool", "?"), 0) + 1
            elif kind.startswith(("model_call", "call")):
                model_s += seconds
            what = event.get("tool") or event.get("what") or ""
            lines.append(f"{at}  {kind[:-4]:<12} {what} -> "
                         f"{'ok' if event.get('ok') else 'FAILED'} in {seconds:.1f}s"
                         + (f"  {event['head']}" if event.get("head") else "")
                         + (f"  {event['error']}" if event.get("error") else ""))
        elif kind == "stream":
            if event.get("part") == "step-start":
                turns = max(turns, event.get("turn", turns))
                lines.append(f"{at}  turn {event.get('turn')}")
            elif event.get("part") == "tool":
                # The model's side of the same round trip. Counted separately
                # from the tool server's `tool.*` spans: where the two
                # disagree -- a call the server served that the model never
                # saw come back -- the disagreement is the finding.
                if event.get("status") == "completed":
                    model_tool_s += event.get("seconds") or 0.0
                    model_tools[event.get("tool", "?")] = \
                        model_tools.get(event.get("tool", "?"), 0) + 1
                    lines.append(f"{at}  tool (model)  {event.get('tool')} "
                                 f"{event.get('seconds', '?')}s "
                                 f"-> {event.get('output_chars', '?')} chars")
                elif event.get("status") in (None, "pending", "running"):
                    lines.append(f"{at}  tool (model)  {event.get('tool')} "
                                 f"{event.get('status') or 'requested'}")
            elif event.get("part") == "step-finish":
                lines.append(f"{at}  step finished: {event.get('reason')} "
                             f"{json.dumps(event.get('tokens') or {})}")
        elif kind == "model_call.timeout":
            lines.append(f"{at}  MODEL CALL KILLED AT THE {event.get('timeout_s')}s "
                         f"WALL after {event.get('seconds')}s: "
                         f"{event.get('turns')} turns, "
                         f"{event.get('tool_calls')} tool calls; raw stream "
                         f"{event.get('stream_file')}")
        elif kind == "session":
            lines.append(f"{at}  opencode session {event.get('session_id')}")
    unclosed = []
    span = max((e["t"] for e in events if "t" in e), default=t0) - t0
    lines.append("")
    lines.append(f"turns {turns}; tool calls {sum(model_tools.values())} "
                 f"seen by the model {model_tools or '{}'}, "
                 f"{sum(mcp_tools.values())} served by the tool server "
                 f"{mcp_tools or '{}'}")
    lines.append(f"wall {span:.0f}s: model calls {model_s:.0f}s, tool calls "
                 f"{tool_s or model_tool_s:.0f}s")
    for (kind, _, _), event in open_spans.items():
        unclosed.append(f"{kind} {event.get('tool') or event.get('what') or ''} "
                        f"(started t+{event['t'] - t0:.1f}s, never returned)")
    if unclosed:
        lines.append("STILL OPEN WHEN THE TRACE ENDS (this is what the process "
                     "was doing when it died):")
        lines += [f"  {u}" for u in unclosed]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <log-dir | trace.jsonl>", file=sys.stderr)
        return 2
    print(report(load(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

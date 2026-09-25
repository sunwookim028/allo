"""CHIA's OpenCodeLLM with the MCP request timeout this design needs, and a
session that says what it is doing while it does it.

**The MCP timeout.** opencode's MCP client applies a per-server request timeout
(`mcp.<name>.timeout`, falling back to `experimental.mcp_timeout`) and CHIA's
`_build_config` sets neither. The 2026-09-19 smoke run measured the
consequence: `MCP error -32001: Request timed out` after 60 s. A cosim score
takes 2-4 minutes, so the agent's `score_cycles` could never return. The
harness's own scoring was unaffected, since it does not go through MCP.

**The observability.** `OpenCodeLLM._capture` streams opencode's `run --format
json` output -- newline-delimited events, one per message part, step and tool
call, written LIVE -- into a `NamedTemporaryFile`, and then `os.unlink`s it in
a `finally`. So the transcript exists only if the call returns: a call killed
at `timeout_seconds` has its one authoritative record deleted on the way out,
which is why a timed-out call reports no session id, no turns and `$0`. Run 3
had four such calls, each a full 2400 s, and 281 model messages across 10
sessions whose content is unrecorded.

`_capture` is overridden here to (a) write the run stream to a DURABLE path
under the worker's log directory and never unlink it, and (b) tail that file
in a thread while the call runs, appending normalised per-turn events to
`trace.jsonl` as they arrive. Nothing new is asked of the model and no extra
process is started: the events were always being produced, and were being
thrown away. `export` calls still go to the base implementation -- an export
is a single blob, not a stream, and it only happens after a call has returned.

Lives in its own module, not in loop.py, so Ray can import it by name on the
worker that runs `prompt`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from chia.models.opencode import OpenCodeLLM

from session_trace import StreamNormaliser, Trace

#: Longer than the evaluator's worst case (gate 2 x 240 s + cosim 1800 s).
MCP_TIMEOUT_MS = 40 * 60 * 1000

#: How often the tailer looks for new bytes. The events are for a human
#: reading a 40-minute session, so a second of latency is free; polling costs
#: one `read` per second against a file the kernel already has.
POLL_SECONDS = 1.0


class _StreamTail(threading.Thread):
    """Follow the run stream file while opencode writes it.

    Deliberately a separate reader on the same path rather than a pipe: a pipe
    is what the base implementation avoided (opencode truncates at the 64 KiB
    pipe buffer and still exits 0), and a reader that owned the pipe would
    have to reproduce that fix. Reading the file the child is appending to
    costs nothing and cannot lose a byte the child has written.
    """

    def __init__(self, path: Path, normaliser: StreamNormaliser):
        super().__init__(daemon=True)
        self.path, self.normaliser = path, normaliser
        self._stopping = threading.Event()   # NOT _stop: Thread._stop is a method
        self._pending = ""
        self._offset = 0

    def run(self):
        while True:
            last = self._stopping.is_set()
            self._drain()
            if last:
                return
            self._stopping.wait(POLL_SECONDS)

    def _drain(self):
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return
        if not chunk:
            return
        self._pending += chunk
        # Only whole lines: a partial trailing line is an event opencode has
        # not finished writing, and is kept for the next pass.
        *complete, self._pending = self._pending.split("\n")
        for line in complete:
            self.normaliser.feed(line)

    def finish(self):
        """Stop, after one last pass over whatever was written meanwhile."""
        self._stopping.set()
        self.join(timeout=30)


class IsaOpenCodeLLM(OpenCodeLLM):
    #: Set by `loop.make_llm`, and carried to the worker with the pickled
    #: object -- a Ray worker inherits the raylet's environment, not the
    #: shell's, so an env var would not arrive.
    trace_dir: str | None = None
    #: What this worker's sessions are titled; repeated on every event so one
    #: trace file can hold several workers.
    trace_label: str | None = None

    def _build_config(self, tools):
        cfg = super()._build_config(tools)
        for entry in (cfg.get("mcp") or {}).values():
            entry["timeout"] = MCP_TIMEOUT_MS
        cfg.setdefault("experimental", {})["mcp_timeout"] = MCP_TIMEOUT_MS
        return cfg

    # -- observability -----------------------------------------------------
    def _trace(self) -> Trace:
        base = self.trace_dir or os.environ.get("TINYTPU_TRACE_DIR")
        return Trace(Path(base) / "trace.jsonl") if base else Trace(None)

    def _stream_path(self, call_id: str) -> Path:
        """Where the raw run stream is kept. Under the log directory when
        there is one; otherwise a temp file that is NOT unlinked, because a
        transcript in /tmp beats no transcript at all."""
        base = self.trace_dir or os.environ.get("TINYTPU_TRACE_DIR")
        if base:
            directory = Path(base) / "opencode"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                return directory / f"run-{call_id}.ndjson"
            except OSError:
                pass
        return Path(tempfile.gettempdir()) / f"opencode_run_{call_id}.ndjson"

    def _capture(self, cmd: list, env: dict) -> SimpleNamespace:
        """As the base, except the run stream is kept and tailed live.

        Same contract: a ``SimpleNamespace(returncode, stdout, stderr)``, and
        ``subprocess.TimeoutExpired`` propagates to the caller -- which is how
        `prompt` learns the call hit the wall. What is different is that when
        it propagates, the transcript is still on disk and the trace already
        says how many turns and tool calls preceded it.
        """
        if len(cmd) < 2 or cmd[1] != "run":
            return super()._capture(cmd, env)   # `export`: one blob, after the fact

        trace = self._trace()
        call_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        out_path = self._stream_path(call_id)
        normaliser = StreamNormaliser(trace, call_id)
        tail = _StreamTail(out_path, normaliser)
        started = time.time()
        trace.event("model_call.start", call_id=call_id, title=self.trace_label,
                    model=self.model, timeout_s=self.timeout_seconds,
                    prompt_chars=len(cmd[-1]), stream_file=str(out_path))
        proc = None
        try:
            with out_path.open("w") as out_fh:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=out_fh,
                    stderr=subprocess.PIPE, text=True, env=env)
                tail.start()
                try:
                    stderr = proc.communicate(timeout=self.timeout_seconds)[1]
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.communicate(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass   # a child still holding stderr must not hang us
                    tail.finish()
                    trace.event("model_call.timeout", call_id=call_id,
                                seconds=round(time.time() - started, 1),
                                timeout_s=self.timeout_seconds,
                                stream_file=str(out_path), **normaliser.totals())
                    raise
                finally:
                    tail.finish()
        except BaseException as error:
            if not isinstance(error, subprocess.TimeoutExpired):
                trace.event("model_call.error", call_id=call_id,
                            seconds=round(time.time() - started, 1),
                            error=f"{type(error).__name__}: {error}"[:300],
                            stream_file=str(out_path), **normaliser.totals())
            raise
        try:
            stdout = out_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stdout = ""
        trace.event("model_call.end", call_id=call_id, ok=proc.returncode == 0,
                    seconds=round(time.time() - started, 1),
                    returncode=proc.returncode, stream_file=str(out_path),
                    **normaliser.totals())
        return SimpleNamespace(returncode=proc.returncode, stdout=stdout,
                               stderr=stderr or "")

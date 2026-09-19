"""A scripted stand-in for the LLM: an OpenAI-compatible chat endpoint that
replays tool calls instead of thinking. Test support for `test_harness.py`.

opencode talks to it through its bundled `@ai-sdk/openai-compatible` provider,
exactly as it talks to Vertex AI, so a test run exercises the real path --
opencode -> MCP client (with its request timeout) -> the Ray-hosted tool server
-> the frozen evaluator -- and costs nothing.

A *script* is a list of steps. Each step is either a tool call
``("read_spec", {})`` (the name is matched as a suffix of the tool names opencode
advertises, so the MCP server prefix does not matter) or a final answer
``("say", "text")``. Every new opencode session pops the next script from the
queue; a session is recognised by a request with no assistant turn yet. Requests
that carry no tools (opencode's title generation) get a one-line answer and do
not consume a script.

Everything the fake saw is kept in ``FakeModel.log``: for each tool call, the
result text opencode fed back to the model. That is how a test checks what the
agent actually received mid-turn -- e.g. that ``score_cycles`` returned a score
rather than ``MCP error -32001: Request timed out``.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModel:
    def __init__(self, scripts=None, host="127.0.0.1", port=0):
        self.scripts: list[list] = list(scripts or [])
        self.lock = threading.Lock()
        self.log: list[dict] = []          # one entry per session
        self.errors: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):
                self._json(200, {"object": "list", "data": [
                    {"id": "scripted", "object": "model", "owned_by": "test"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    reply = fake.respond(body)
                except Exception as error:  # surfaced to the test, not swallowed
                    fake.errors.append(repr(error))
                    reply = {"text": f"fake model error: {error!r}"}
                if body.get("stream"):
                    self._stream(reply)
                else:
                    self._json(200, self._completion(reply))

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            @staticmethod
            def _completion(reply):
                msg = {"role": "assistant", "content": reply.get("text")}
                if "call" in reply:
                    msg["tool_calls"] = [reply["call"]]
                return {"id": "fake", "object": "chat.completion",
                        "created": int(time.time()), "model": "scripted",
                        "choices": [{"index": 0, "message": msg, "finish_reason":
                                     "tool_calls" if "call" in reply else "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2}}

            def _stream(self, reply):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                base = {"id": "fake", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": "scripted"}

                def emit(delta, finish=None, usage=None):
                    chunk = dict(base, choices=[{"index": 0, "delta": delta,
                                                 "finish_reason": finish}])
                    if usage:
                        chunk["usage"] = usage
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                if "call" in reply:
                    call = reply["call"]
                    emit({"role": "assistant", "content": None, "tool_calls": [
                        {"index": 0, "id": call["id"], "type": "function",
                         "function": call["function"]}]})
                    finish = "tool_calls"
                else:
                    emit({"role": "assistant", "content": reply["text"]})
                    finish = "stop"
                emit({}, finish, {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.url = f"http://{host}:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()

    # -- the "model" -------------------------------------------------------
    def respond(self, body: dict) -> dict:
        messages = body.get("messages", [])
        tools = [t["function"]["name"] for t in body.get("tools") or []]
        if not tools:
            return {"text": "scripted session"}
        with self.lock:
            assistant_turns = [m for m in messages if m.get("role") == "assistant"]
            if not assistant_turns:
                if not self.scripts:
                    raise RuntimeError("no script left for a new session")
                self.log.append({"script": self.scripts.pop(0), "tools": tools,
                                 "calls": [], "started": time.time()})
            session = self.log[-1]
            # Record the tool results opencode fed back since the last step.
            results = [m for m in messages if m.get("role") == "tool"]
            for call, res in zip(session["calls"], results):
                call.setdefault("result", _text(res.get("content")))
                call.setdefault("returned_after_s",
                                round(time.time() - call["sent"], 1))
            step = len(assistant_turns)
            script = session["script"]
            if step >= len(script):
                return {"text": "done"}
            name, args = script[step]
            if name == "say":
                return {"text": args}
            matches = [t for t in tools if t.endswith(name)]
            if len(matches) != 1:
                raise RuntimeError(f"tool '{name}' matches {matches} among {tools}")
            session["calls"].append({"tool": matches[0], "args": args,
                                     "sent": time.time()})
            return {"call": {"id": f"call_{len(self.log)}_{step}", "type": "function",
                             "function": {"name": matches[0],
                                          "arguments": json.dumps(args)}}}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return json.dumps(content)

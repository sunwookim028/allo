"""Smallest end-to-end check that the whole agent path is wired up.

ADC -> Vertex AI -> opencode -> CHIA's Ray actor and MCP server -> this tool ->
the frozen evaluator's gate in the `allo` env. One model call that must read
the spec to answer, so it costs cents, not dollars. No cosim. The pre-flight
gate runs first, so the call is charged to the configured CHIA project and
billing account (chia2026-tinytpu / CHIA2026) or not made at all.

    python chia_agent/smoke.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preflight  # noqa: E402
from allo_tool import AlloSpecTool  # noqa: E402
from loop import (AGENT_DIR, ALLO_PYTHON, LLVM_BUILD_DIR, REPO_ROOT,  # noqa: E402
                  Budget, ask, make_llm, seed_spec)

#: Per-run spend cap for the smoke call (one prompt; ~$0.05 measured).
SMOKE_CAP_USD = 5.0
#: The unit names in microarch_isa.py; the model can only know them by reading.
EXPECTED = ("sequencer", "dma_ld", "spm", "vru", "accu")


def main() -> int:
    started = time.time()
    preflight.require(SMOKE_CAP_USD, run_t0_ms=int(started * 1000))
    try:
        ray.init(address="auto", ignore_reinit_error=True,
                 runtime_env={"working_dir": str(AGENT_DIR)})
    except (ConnectionError, ValueError):
        ray.init(resources={"opencode_creds": 1}, ignore_reinit_error=True,
                 runtime_env={"working_dir": str(AGENT_DIR)})
    scratch = REPO_ROOT / ".chia_scratch" / "smoke"
    spec = Path(tempfile.mkdtemp(prefix="spec-", dir=scratch.parent))
    seed_spec(spec)
    tool = AlloSpecTool("tpusmoke", str(spec), str(scratch), str(AGENT_DIR),
                        str(REPO_ROOT), ALLO_PYTHON, LLVM_BUILD_DIR)
    try:
        print("[1/3] tool up; running the frozen gate on the unmodified spec...")
        v = tool.evaluate(gate_only=True)
        if not v.get("ok"):
            print(f"FAIL: gate did not pass\n{v}")
            return 1
        print(f"      gate OK: {v['gate']}")
        print("[2/3] one model call through the MCP tool (Vertex AI)...")
        calls = []
        t0 = int(started * 1000)
        budget = Budget(SMOKE_CAP_USD, t0, f"chia-smoke@{t0}", "tpusmoke")
        response = ask(make_llm(tool, budget.title), tool,
                       "Call tpusmoke_read_spec exactly once. Then reply with a "
                       "single line listing the names of the @df.kernel functions "
                       "defined inside the tinytpu_isa region in microarch_isa.py, "
                       "comma separated, and nothing else.",
                       budget, "smoke", calls)
        text = str(response.result)
        print(f"      model replied: {text.strip()[:300]}")
        print("[3/3] checking the reply came from the spec...")
        missing = [u for u in EXPECTED if u not in text]
        if missing:
            print(f"FAIL: reply did not name {missing}")
            return 1
        print(f"SMOKE OK ({time.time() - started:.0f}s, "
              f"${calls[0]['usd']:.3f})")
        return 0
    finally:
        tool.stop()
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

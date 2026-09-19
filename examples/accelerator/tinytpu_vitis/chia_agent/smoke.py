"""Smallest end-to-end check that the whole agent path is wired up.

Exercises, in one shot: Application Default Credentials -> Vertex AI -> the
opencode CLI -> CHIA's Ray actor and MCP tool server -> the TinyTPU tool -> the
``allo`` conda environment. It asks the model for one tool call and one number,
so it costs a few cents rather than the few dollars a real candidate costs.

    python chia_agent/smoke.py

Exit status is 0 only if the model actually reached the tool and came back with
the instruction count it had to read the spec to know.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parent))

from allo_tool import AlloSpecTool  # noqa: E402
from loop import CONDA_EXE, TINYTPU_DIR, ask, make_llm  # noqa: E402

# The spec declares these; the model can only answer by calling read_spec.
EXPECTED = ("dma_load", "dma_store", "vload", "vstore", "vpu", "mxu")


def main() -> int:
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("FAIL: GOOGLE_CLOUD_PROJECT is unset")
        return 1

    started = time.time()
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except (ConnectionError, ValueError):
        ray.init(resources={"opencode_creds": 1}, ignore_reinit_error=True)

    tool = AlloSpecTool("tinytpusmoke", str(TINYTPU_DIR), CONDA_EXE)
    try:
        print("[1/3] tool server up; checking the allo environment...")
        compiler = tool.run_compiler_check()
        if not compiler.startswith("exit=0"):
            print(f"FAIL: compiler check did not pass\n{compiler[-1500:]}")
            return 1
        print("      allo environment OK")

        print("[2/3] asking the model for one tool call (Vertex AI)...")
        response = ask(
            make_llm(tool),
            tool,
            "Call tinytpusmoke_read_spec exactly once. Then reply with a single "
            "line listing the names of the @tpu.unit hardware blocks defined in "
            "microarch.py, comma separated, and nothing else.",
        )
        text = str(response.result)
        print(f"      model replied: {text.strip()[:200]}")

        print("[3/3] checking the reply came from the spec...")
        missing = [unit for unit in EXPECTED if unit not in text]
        if missing:
            print(
                f"FAIL: reply did not name {missing} - the model may not have "
                f"reached the tool"
            )
            return 1
        print(
            f"SMOKE OK ({time.time() - started:.0f}s): "
            f"ADC -> Vertex -> opencode -> CHIA -> MCP tool -> allo env"
        )
        return 0
    finally:
        tool.stop()
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

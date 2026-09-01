"""A minimal CHIA generate -> validate -> debug loop for TinyTPU's Allo ISA."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import ray

from chia.base.ChiaFunction import get
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

from allo_tool import AlloSpecTool


AGENT_DIR = Path(__file__).resolve().parent
TINYTPU_DIR = AGENT_DIR.parent
CONDA_EXE = os.environ.get("TINYTPU_CONDA", shutil.which("conda") or "conda")
MODEL = os.environ.get(
    "TINYTPU_OPENCODE_MODEL", "google-vertex/gemini-3.1-pro-preview"
)
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("TINYTPU_VERTEX_LOCATION", "global")


def make_llm(tool: AlloSpecTool) -> OpenCodeLLM:
    if not PROJECT:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT before running the Vertex AI agent.")
    provider, _, model_id = MODEL.partition("/")
    if provider != "google-vertex" or not model_id:
        raise ValueError("TINYTPU_OPENCODE_MODEL must be google-vertex/<model-id>.")
    return OpenCodeLLM(
        model=MODEL,
        system_message=(
            "You are a TPU ISA co-design engineer. The hardware and compiler are "
            "complete and must be treated as fixed. Work only at the Allo ISA layer. "
            "Use the provided TinyTPU MCP tools; do not attempt other tools or make "
            "changes outside isa.py. Preserve existing operand order and semantics."
        ),
        timeout_seconds=900,
        additional_providers=[
            AdditionalModelProvider(
                id="google-vertex",
                npm="@ai-sdk/google-vertex",
                name="Google Vertex AI",
                models=[model_id],
                options={"project": PROJECT, "location": LOCATION},
            )
        ],
        # OpenCode's local file and shell tools run in its own container. The
        # MCP server is intentionally its only capability for this flow.
        config={"*": "deny", f"{tool.name}_*": "allow"},
    )


def ask(llm: OpenCodeLLM, tool: AlloSpecTool, prompt: str):
    return get(
        llm.prompt.options(resources={"opencode_creds": 1}).chia_remote(
            llm, prompt, [tool]
        )
    )


def run(task: str, max_debug_attempts: int) -> int:
    ray.init(address="auto", runtime_env={"working_dir": str(AGENT_DIR)})
    tool = AlloSpecTool("tinytpu", str(TINYTPU_DIR), CONDA_EXE)
    try:
        llm = make_llm(tool)
        response = ask(
            llm,
            tool,
            f"""Implement this ISA-level change:

{task}

First inspect isa.py. If a change is needed, use tinytpu_insert_after rather
than a unified diff: anchor on the exact text `return primitive.negate(a)` and
insert the complete vabs definition immediately after it. Then call
tinytpu_run_compiler_check. Report the change and the check result succinctly.
""",
        )
        print(response.result)
        for attempt in range(1, max_debug_attempts + 1):
            check = tool.run_compiler_check()
            if check.startswith("exit=0"):
                print("TinyTPU compiler check: PASS")
                return 0
            response = ask(
                llm,
                tool,
                f"The compiler check failed after your change (attempt {attempt}):\n"
                f"```\n{check}\n```\nDiagnose it, patch only isa.py, and rerun the check.",
            )
            print(response.result)
        print("TinyTPU compiler check: FAILED after debug budget")
        return 1
    finally:
        tool.stop()
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Allo ISA change for the agent")
    parser.add_argument("--max-debug-attempts", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run(args.task, args.max_debug_attempts))


if __name__ == "__main__":
    main()

"""A minimal CHIA generate -> validate -> score loop for TinyTPU co-design."""

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
            "You are a TPU ISA/microarchitecture co-design engineer. You may edit "
            "only isa.py and microarch.py through the TinyTPU MCP tools; the generic "
            "ACT compiler, runtime, tests, and benchmark are fixed. microarch.py is "
            "a composition of named Allo-HLS blocks: dma_load/dma_store (DRAM-VMEM), "
            "vload/vstore (VMEM-VREG), vpu, mxu, and tinytpu's decoder/composition. "
            "Preserve this compositional structure: implement an architectural change "
            "by connecting or refining these blocks (or adding one focused @tpu.unit), "
            "then keep ISA encoding, decoder, operands, schedules, and ISA semantics "
            "consistent. Do not modify anything outside the two writable files."
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
            f"""Implement this TinyTPU co-design task:

{task}

First inspect the full writable specification, then call
tinytpu_score_access_cost to establish a baseline. Implement the smallest
correct candidate, using a unified diff for coupled ISA/microarchitecture
changes. Verify it with tinytpu_run_compiler_check, then score it again. The
optimization objective is only the reported VREG/VMEM storage-access cost,
with frozen costs VREG=1 and VMEM=4; DRAM traffic is informational and there
is no direct-path mnemonic bonus. HLS export and synthesis are deferred until
the dedicated tooling server is available. Report baseline, candidate score,
and compiler feasibility succinctly.
""",
        )
        print(response.result)
        for attempt in range(1, max_debug_attempts + 1):
            compiler_check = tool.run_compiler_check()
            score = tool.score_access_cost()
            if compiler_check.startswith("exit=0") and score.startswith("exit=0"):
                print("TinyTPU compiler check: PASS")
                print(score)
                return 0
            response = ask(
                llm,
                tool,
                f"The candidate failed evaluation (attempt {attempt}):\n"
                f"compiler:\n```\n{compiler_check}\n```\n"
                f"score:\n```\n{score}\n```\n"
                "Diagnose it, patch only isa.py and/or microarch.py, then rerun all gates.",
            )
            print(response.result)
        print("TinyTPU co-design evaluation: FAILED after debug budget")
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

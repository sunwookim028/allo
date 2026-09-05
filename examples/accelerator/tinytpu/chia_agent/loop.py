"""A CHIA generate -> synthesize -> score -> keep-or-rewind loop for TinyTPU.

Each iteration asks the agent for one candidate ISA/microarchitecture edit,
gates it on the direct-TOSA compiler check, then scores it by *synthesizing the
candidate with Vitis HLS* and costing the GEMM benchmarks under the measured
per-unit latency table. A candidate is kept only if it strictly beats the best
score so far; otherwise the writable spec is rewound and the next iteration is
told what failed. The objective is cycles alone -- area and Fmax are recorded
for every candidate but never scored.

The agent can edit only ``isa.py`` and ``microarch.py``, and it cannot reach the
evaluator. It also cannot win by declaring an optimistic ``ISA.latency``: those
declarations are overwritten from the synthesis report before scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import ray

from chia.base.ChiaFunction import get
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM

from allo_tool import AlloSpecTool


AGENT_DIR = Path(__file__).resolve().parent
TINYTPU_DIR = AGENT_DIR.parent
CONDA_EXE = os.environ.get("TINYTPU_CONDA", shutil.which("conda") or "conda")
MODEL = os.environ.get("TINYTPU_OPENCODE_MODEL", "google-vertex/gemini-3.1-pro-preview")
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("TINYTPU_VERTEX_LOCATION", "global")

SYSTEM_MESSAGE = (
    "You are a TPU ISA/microarchitecture co-design engineer. You may edit "
    "only isa.py and microarch.py through the TinyTPU MCP tools; the generic "
    "ACT compiler, runtime, tests, benchmarks, and the evaluator are fixed. "
    "microarch.py is a composition of named Allo-HLS blocks: dma_load/dma_store "
    "(DRAM-VMEM), vload/vstore (VMEM-VREG), vpu, mxu, and tinytpu's "
    "decoder/composition. Preserve this compositional structure: implement an "
    "architectural change by connecting or refining these blocks (or adding one "
    "focused @tpu.unit), then keep ISA encoding, decoder, operands, schedules, "
    "and ISA semantics consistent. Do not modify anything outside the two "
    "writable files. Your score is the synthesized cycle count: the design is "
    "put through Vitis HLS C-synthesis and every unit's (ii, depth) is "
    "re-measured from that report, so editing an ISA.latency declaration "
    "changes nothing. Reduce real cycles instead."
)


def make_llm(tool: AlloSpecTool) -> OpenCodeLLM:
    if not PROJECT:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT before running the Vertex AI agent."
        )
    provider, _, model_id = MODEL.partition("/")
    if provider != "google-vertex" or not model_id:
        raise ValueError("TINYTPU_OPENCODE_MODEL must be google-vertex/<model-id>.")
    return OpenCodeLLM(
        model=MODEL,
        system_message=SYSTEM_MESSAGE,
        timeout_seconds=1800,
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


def parse_score(output: str) -> dict | None:
    """Pull the evaluator's JSON out of an ``exit=<rc>\\n<output>`` tool result."""
    if not output.startswith("exit=0"):
        return None
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def score_candidate(tool: AlloSpecTool) -> tuple[dict | None, str]:
    """Run both gates. Returns ``(score_json_or_None, detail)``."""
    compiler = tool.run_compiler_check()
    if not compiler.startswith("exit=0"):
        return None, f"compiler check failed:\n{compiler[-4000:]}"
    scored = tool.score_cycles()
    parsed = parse_score(scored)
    if parsed is None:
        return None, f"synthesis/score failed:\n{scored[-4000:]}"
    return parsed, "ok"


def summarize(score: dict) -> str:
    synthesis = score.get("synthesis") or {}
    area = synthesis.get("area", {})
    per_bench = ", ".join(
        f"{tuple(b['shape'])}={b['cycles']:.0f}c" for b in score["benchmarks"]
    )
    return (
        f"total_cycles={score['total_cycles']:.0f} "
        f"(roofline {score['total_bottleneck_cycles']:.0f}) [{per_bench}] "
        f"Fmax={synthesis.get('fmax_mhz')}MHz "
        f"LUT={area.get('lut')} FF={area.get('ff')} DSP={area.get('dsp')} "
        f"BRAM={area.get('bram18k')}"
    )


def _record(log_path: Path, entry: dict) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def run(
    task: str,
    iterations: int,
    max_debug_attempts: int,
    log_dir: Path,
) -> int:
    runtime_env = {"working_dir": str(AGENT_DIR)}
    try:
        ray.init(address="auto", runtime_env=runtime_env)
    except ConnectionError:
        # Some shared hosts immediately tear down ``ray start --head``. Keep the
        # single-machine demo self-contained by creating the same credential
        # resource in an in-process local Ray cluster.
        ray.init(resources={"opencode_creds": 1}, runtime_env=runtime_env)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "variants.jsonl"
    tool = AlloSpecTool("tinytpu", str(TINYTPU_DIR), CONDA_EXE)
    try:
        baseline_snapshot = tool.snapshot()

        print("=" * 72)
        print("Baseline: synthesizing the unmodified design")
        print("=" * 72)
        baseline, detail = score_candidate(tool)
        if baseline is None:
            print(f"Baseline evaluation failed:\n{detail}")
            return 1
        print(f"  {summarize(baseline)}")
        _record(
            log_path,
            {"iteration": 0, "kind": "baseline", "accepted": True, "score": baseline},
        )

        best = baseline
        best_snapshot = baseline_snapshot
        history: list[str] = []
        llm = make_llm(tool)

        for iteration in range(1, iterations + 1):
            print("=" * 72)
            print(
                f"Iteration {iteration}/{iterations} — best so far "
                f"{best['total_cycles']:.0f} cycles"
            )
            print("=" * 72)
            # Always propose from the best-known design.
            tool.restore(best_snapshot)

            tried = (
                "\n".join(f"  - {line}" for line in history[-8:])
                if history
                else "  (nothing tried yet)"
            )
            started = time.time()
            response = ask(
                llm,
                tool,
                f"""TinyTPU co-design task:

{task}

The writable spec is currently the best design found so far, scoring
{best['total_cycles']:.0f} total cycles. Per-unit busy cycles on the larger
benchmark: {json.dumps(best['benchmarks'][-1]['unit_cycles'])}. The measured
latency table from its synthesis: {json.dumps(best['latency_table'])}.

Already attempted:
{tried}

Propose ONE new candidate that should reduce the synthesized cycle count.
Inspect the spec first, apply a coupled ISA/microarchitecture edit as a unified
diff, and verify it with tinytpu_run_compiler_check. You may call
tinytpu_score_cycles yourself to see the synthesized result. Explain in one or
two sentences what you changed and why it should cost fewer cycles.
""",
            )
            print(response.result)

            score, detail = score_candidate(tool)
            for attempt in range(1, max_debug_attempts + 1):
                if score is not None:
                    break
                print(f"  candidate broken (debug attempt {attempt}): {detail[:400]}")
                response = ask(
                    llm,
                    tool,
                    f"The candidate failed its gates (attempt {attempt}):\n"
                    f"```\n{detail}\n```\n"
                    "Diagnose it, patch only isa.py and/or microarch.py, then stop.",
                )
                print(response.result)
                score, detail = score_candidate(tool)

            elapsed = time.time() - started
            if score is None:
                history.append(f"iteration {iteration}: rejected — did not build/score")
                print(f"  REJECTED (no valid score) after {elapsed:.0f}s")
                _record(
                    log_path,
                    {
                        "iteration": iteration,
                        "kind": "candidate",
                        "accepted": False,
                        "reason": "gates failed",
                        "detail": detail[-4000:],
                        "seconds": elapsed,
                    },
                )
                tool.restore(best_snapshot)
                continue

            improved = score["total_cycles"] < best["total_cycles"]
            delta = score["total_cycles"] - best["total_cycles"]
            print(f"  {summarize(score)}")
            print(
                f"  {'ACCEPTED' if improved else 'REJECTED'} "
                f"({delta:+.0f} cycles vs best) after {elapsed:.0f}s"
            )
            _record(
                log_path,
                {
                    "iteration": iteration,
                    "kind": "candidate",
                    "accepted": improved,
                    "delta_cycles": delta,
                    "score": score,
                    "diff": tool.diff_against(best_snapshot),
                    "seconds": elapsed,
                },
            )
            if improved:
                history.append(
                    f"iteration {iteration}: ACCEPTED, {score['total_cycles']:.0f} "
                    f"cycles ({delta:+.0f})"
                )
                best, best_snapshot = score, tool.snapshot()
            else:
                history.append(
                    f"iteration {iteration}: rejected, {score['total_cycles']:.0f} "
                    f"cycles ({delta:+.0f}) — no improvement"
                )
                tool.restore(best_snapshot)

        # Leave the repository holding the best design found.
        tool.restore(best_snapshot)
        improvement = baseline["total_cycles"] - best["total_cycles"]
        print("=" * 72)
        print("Search complete")
        print(f"  baseline : {summarize(baseline)}")
        print(f"  best     : {summarize(best)}")
        print(
            f"  improvement: {improvement:.0f} cycles "
            f"({100.0 * improvement / baseline['total_cycles']:.1f}%)"
        )
        print(f"  variant log: {log_path}")
        best_diff = tool.diff_against(baseline_snapshot)
        (log_dir / "best.diff").write_text(best_diff, encoding="utf-8")
        if best_diff:
            print(f"  best diff  : {log_dir / 'best.diff'}")
        return 0
    finally:
        tool.stop()
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Allo ISA change for the agent")
    parser.add_argument(
        "--iterations", type=int, default=5, help="candidates to explore"
    )
    parser.add_argument("--max-debug-attempts", type=int, default=1)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("TINYTPU_CHIA_LOG_DIR", "chia_runs/latest")),
    )
    args = parser.parse_args()
    raise SystemExit(
        run(args.task, args.iterations, args.max_debug_attempts, args.log_dir)
    )


if __name__ == "__main__":
    main()

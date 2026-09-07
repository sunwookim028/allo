"""Run several TinyTPU co-design searches at once, then keep the best.

One search is bounded by the model, not the tools: an agent turn costs minutes
while a full synthesize-and-score costs about 50 seconds on one core, so a
serial loop leaves a 144-core machine essentially idle. Two things serialize it,
and both are lifted here:

* CHIA gates every LLM call on a cluster resource (``opencode_creds``). Start
  the head with as many units as workers -- see ``--workers``.
* ``AlloSpecTool`` edits one checkout. Each worker therefore gets its own git
  worktree, so the writable spec, the HLS project, and the MCP tool name are
  all per-worker.

Running K searches also buys what one search cannot: diversity. A single loop
is greedy hill-climbing from one incumbent, so it re-treads variants of
whatever last worked. K searches started on different framings of the problem
explore different parts of the space, and the best of K is kept.

    python chia_agent/swarm.py --workers 4 --iterations 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
TINYTPU_DIR = AGENT_DIR.parent
REPO_ROOT = TINYTPU_DIR.parents[2]

#: Different framings of the same objective. Each worker gets one, so the
#: population starts from genuinely different hypotheses rather than from K
#: samples of one prompt.
STRATEGIES = (
    (
        "dram",
        """Attack DRAM staging. Synthesis measures dma_load at depth 75 -- the
m_axi read latency Vitis actually builds -- and the compiler issues one
dma_load per tile, so that latency is paid over and over. Reduce the NUMBER of
DRAM transactions: larger blocks, coalesced tiles, or staging a tile once and
reusing it across the K loop.""",
    ),
    (
        "onchip",
        """Attack the on-chip round trip. Values move BRAM->VREG->compute->VREG->BRAM
even when a producer and consumer are adjacent. Let MXU and VPU exchange values
through VREG directly, or otherwise remove a staging hop, without breaking the
compiler's allocation.""",
    ),
    (
        "mxu",
        """Attack the matmul unit itself. Synthesis reports mxu at 72 cycles per
4x4 tile with ii=2, because its body synthesizes as two sequential passes over
the tile. Restructure it so the staging pass and the dot-product pass overlap,
or so one pass is eliminated, to get ii closer to 1.""",
    ),
    (
        "granularity",
        """Attack instruction granularity. Each instruction pays its unit's full
pipeline depth, and the benchmarks issue 216 and 1792 instructions. A coarser
instruction that does more work per issue amortizes that depth. Add or widen
one instruction so the same computation issues fewer times.""",
    ),
)

BASE_TASK = """Reduce the synthesized cycle count of the tiled GEMM benchmarks.
Any change must keep the GEMM results numerically correct.

Your assigned angle for this search:
{angle}
"""


def make_worktree(base: Path, worker: str) -> Path:
    """A private checkout for one worker, at the current commit."""
    path = base / f"worker-{worker}"
    if path.exists():
        return path
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return path


def launch(
    worker: str, angle: str, workspace: Path, log_dir: Path, iterations: int
) -> subprocess.Popen:
    env = os.environ | {
        # Each worker evaluates its own copy of the spec.
        "PYTHONPATH": str(workspace),
        "TINYTPU_SYNTH_PROJECT": f"/tmp/tinytpu_swarm/{worker}",
    }
    command = [
        sys.executable,
        "-u",  # unbuffered, so a long run's progress is visible while it runs
        str(AGENT_DIR / "loop.py"),
        "--task",
        BASE_TASK.format(angle=angle),
        "--iterations",
        str(iterations),
        "--workspace",
        str(workspace / "examples" / "accelerator" / "tinytpu"),
        "--tool-name",
        f"tinytpu{worker}",
        "--log-dir",
        str(log_dir),
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    handle = (log_dir / "worker.log").open("w", encoding="utf-8")
    return subprocess.Popen(
        command, cwd=workspace, env=env, stdout=handle, stderr=subprocess.STDOUT
    )


def read_variants(log_dir: Path) -> list[dict]:
    path = log_dir / "variants.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(run_dir: Path, workers: list[str]) -> int:
    """Print each worker's trajectory and the best design across all of them."""
    baseline = None
    best = (None, None, None)  # (cycles, worker, entry)
    print("=" * 78)
    print("Swarm results")
    print("=" * 78)
    for worker in workers:
        entries = read_variants(run_dir / worker)
        if not entries:
            print(f"  worker {worker}: no results")
            continue
        accepted = [e for e in entries if e.get("accepted") and e.get("score")]
        base = next((e for e in entries if e["kind"] == "baseline"), None)
        if base:
            baseline = base["score"]["total_cycles"]
        tried = len([e for e in entries if e["kind"] == "candidate"])
        kept = [e for e in accepted if e["kind"] == "candidate"]
        top = min(
            (e for e in accepted),
            key=lambda e: e["score"]["total_cycles"],
            default=None,
        )
        cycles = top["score"]["total_cycles"] if top else None
        print(
            f"  worker {worker:<12} candidates={tried:<3} accepted={len(kept):<3} "
            f"best={cycles}"
        )
        if cycles is not None and (best[0] is None or cycles < best[0]):
            best = (cycles, worker, top)

    if best[0] is None or baseline is None:
        print("\n  no scored candidate produced")
        return 1
    gain = baseline - best[0]
    print(
        f"\n  baseline {baseline:.0f} cycles -> best {best[0]:.0f} cycles "
        f"({gain:+.0f}, {100.0 * gain / baseline:.1f}%) from worker '{best[1]}'"
    )
    diff = (best[2] or {}).get("diff") or ""
    if diff:
        winner = run_dir / "swarm_best.diff"
        winner.write_text(diff, encoding="utf-8")
        print(f"  winning diff: {winner}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=len(STRATEGIES))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "chia_runs" / f"swarm-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--worktree-base", type=Path, default=Path("/tmp/tinytpu_swarm_trees")
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=90.0,
        help="seconds between worker launches, to spread the initial burst",
    )
    args = parser.parse_args()

    strategies = list(STRATEGIES)[: args.workers]
    args.worktree_base.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    running = []
    for worker, angle in strategies:
        workspace = make_worktree(args.worktree_base, worker)
        log_dir = args.run_dir / worker
        print(f"launching worker '{worker}' in {workspace}")
        running.append(
            (worker, launch(worker, angle, workspace, log_dir, args.iterations))
        )

    print(f"\n{len(running)} searches running; logs under {args.run_dir}")
    for worker, process in running:
        code = process.wait()
        print(f"worker '{worker}' exited with {code}")

    raise SystemExit(report(args.run_dir, [worker for worker, _ in strategies]))


if __name__ == "__main__":
    main()

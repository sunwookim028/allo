# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded co-design variant and re-derive its score from scratch.

A run's ``variants.jsonl`` stores each accepted candidate as a diff against the
best design at the time it was proposed. That is enough to reconstruct any
variant exactly: check the repository out clean, apply the accepted diffs in
order, and score the result. This turns "the agent reported 4.07x" into a
command anyone can run, which is the difference between a logged number and a
reproduced one.

    python -m examples.accelerator.tinytpu.verify_variant --run <dir> --worker dram

The replay happens in a throwaway git worktree, so it neither reads nor disturbs
the working tree the agent left behind -- and because it re-runs synthesis, it
also exercises the whole toolchain end to end, which makes it usable as a smoke
test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TINYTPU_REL = Path("examples") / "accelerator" / "tinytpu"


def load_entries(run_dir: Path, worker: str) -> list[dict]:
    path = run_dir / worker / "variants.jsonl"
    if not path.exists():
        raise SystemExit(f"no variant log at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accepted_through(entries: list[dict], iteration: int | None) -> list[dict]:
    """The accepted candidates to replay, in the order they were accepted."""
    accepted = [
        entry
        for entry in entries
        if entry["kind"] == "candidate" and entry.get("accepted")
    ]
    if iteration is not None:
        accepted = [e for e in accepted if e["iteration"] <= iteration]
    return accepted


def replay(worktree: Path, diffs: list[str]) -> None:
    target = worktree / TINYTPU_REL
    for index, diff in enumerate(diffs, start=1):
        result = subprocess.run(
            ["patch", "--batch", "--forward", "-p1"],
            input=diff,
            text=True,
            cwd=target,
            capture_output=True,
        )
        if result.returncode:
            raise SystemExit(
                f"diff {index} did not apply:\n{result.stdout}{result.stderr}"
            )


def score(worktree: Path, project: Path, frozen: bool) -> dict:
    command = [
        "conda",
        "run",
        "-n",
        "allo",
        "python",
        "-m",
        "examples.accelerator.tinytpu.ppa",
        "--project",
        str(project),
    ]
    if frozen:
        command.append("--frozen")
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(worktree),
        "SKBUILD_EDITABLE_SKIP": str(REPO_ROOT / "build"),
    }
    result = subprocess.run(
        command, cwd=worktree, env=env, text=True, capture_output=True
    )
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise SystemExit(
        f"scoring produced no result (exit {result.returncode}):\n"
        f"{(result.stdout + result.stderr)[-3000:]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a chia_runs/<run> dir")
    parser.add_argument("--worker", required=True)
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="replay accepted candidates up to this iteration (default: all)",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="score without synthesis (fast; checks the replay, not the QoR)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="permitted relative difference from the recorded cycle count",
    )
    args = parser.parse_args()

    entries = load_entries(args.run, args.worker)
    accepted = accepted_through(entries, args.iteration)
    if not accepted:
        raise SystemExit(f"no accepted candidate in {args.run / args.worker}")
    expected = accepted[-1]["score"]["total_cycles"]
    baseline = next(
        (e["score"]["total_cycles"] for e in entries if e["kind"] == "baseline"), None
    )

    print(
        f"replaying {len(accepted)} accepted candidate(s) from "
        f"{args.run.name}/{args.worker}"
    )
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="tinytpu-replay-") as tmp:
        worktree = Path(tmp) / "tree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        try:
            replay(worktree, [e["diff"] for e in accepted])
            result = score(worktree, Path(tmp) / "prj", args.frozen)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO_ROOT,
                capture_output=True,
            )

    actual = result["total_cycles"]
    elapsed = time.time() - started
    drift = abs(actual - expected) / expected if expected else 0.0
    print(f"  recorded : {expected:,.0f} cycles")
    print(f"  replayed : {actual:,.0f} cycles")
    if baseline:
        print(f"  baseline : {baseline:,.0f} cycles  ->  {baseline / actual:.2f}x")
    if result.get("synthesis"):
        area = result["synthesis"]["area"]
        print(
            f"  area     : LUT {area['lut']} FF {area['ff']} DSP {area['dsp']} "
            f"BRAM18K {area['bram18k']}  Fmax {result['synthesis']['fmax_mhz']} MHz"
        )
    print(f"  numerics : {result['status']}  ({elapsed:.0f}s)")

    if drift > args.tolerance:
        print(f"MISMATCH: replay differs from the record by {drift:.2%}")
        sys.exit(1)
    print("VERIFIED")


if __name__ == "__main__":
    main()

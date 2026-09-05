# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthesis-grounded cycle objective for TinyTPU ISA co-design.

This is the evaluator the CHIA agent optimizes against, and it is deliberately
outside the agent's writable set: a candidate can only improve its score by
building hardware that runs the benchmarks in fewer cycles, not by re-weighting
the objective.

The loop it closes, per candidate:

1. export the composed ``top_s`` schedule to Vitis HLS and run C-synthesis,
2. rebuild every unit's ``(ii, depth)`` from the measured report
   (:mod:`.synth`) -- replacing ``microarch.py``'s frozen pre-synthesis guesses,
3. compile the GEMM benchmarks and check them against NumPy,
4. score ``CompiledProgram.cycles()`` under that measured table.

Area and Fmax are reported for every candidate but **not** scored, so the
objective stays one number: cycles. ``bottleneck_cycles`` (the roofline, where
units overlap perfectly) and per-unit busy time are reported alongside, because
they say *where* a candidate's cycles went.

Run ``python -m examples.accelerator.tinytpu.ppa``; add ``--frozen`` to score
against the declared table without invoking Vitis.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .feedback import access_profile
from .isa import tpu
from .synth import (
    DEFAULT_PART,
    apply_latency_table,
    derive_latency_table,
    report_summary,
    synthesize,
)
from .verify import _gemm_source

#: The same two GEMMs the pre-synthesis objective used, so scores stay
#: comparable across the checkpoint.
BENCHMARKS = ((8, 8, 8), (8, 16, 32))


def evaluate_gemm(M: int, K: int, N: int) -> dict:
    """Compile and run one TOSA GEMM, then cost it with the current table."""
    rng = np.random.default_rng(M * 10_000 + K * 100 + N)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    b = rng.standard_normal((1, K, N)).astype(np.float32)
    prog = tpu.compile_program(_gemm_source(M, K, N))
    np.testing.assert_allclose(prog(a, b), a @ b, rtol=1e-4, atol=1e-4)
    unit_cycles = prog.unit_cycles()
    return {
        "shape": [M, K, N],
        "instructions": len(prog.emits),
        "cycles": prog.cycles(),
        "bottleneck_cycles": prog.bottleneck_cycles(),
        "unit_cycles": {unit: round(c, 1) for unit, c in sorted(unit_cycles.items())},
        # Storage traffic stays visible: it is what the pre-synthesis checkpoint
        # scored, and it explains a cycle count that moved.
        "access_profile": access_profile(prog),
    }


def evaluate(
    *,
    synthesize_design: bool = True,
    part: str = DEFAULT_PART,
    freq_mhz: float = 300.0,
    project: str = "tinytpu_synth_prj",
) -> dict:
    """Score the candidate. With ``synthesize_design``, the latency table is
    measured by Vitis HLS first; otherwise ``microarch.py``'s declared table is
    used as-is (useful for a fast smoke check, not for a reported score)."""
    synthesis = None
    latency_table = None
    if synthesize_design:
        report = synthesize(part=part, freq_mhz=freq_mhz, project=project)
        table = derive_latency_table(report)
        apply_latency_table(table)
        synthesis = report_summary(report)
        latency_table = {name: lat.as_dict() for name, lat in table.items()}

    benchmarks = [evaluate_gemm(*shape) for shape in BENCHMARKS]
    total_cycles = sum(bench["cycles"] for bench in benchmarks)
    return {
        "status": "pass",
        "objective": {
            "metric": "total_cycles",
            "definition": "sum over benchmarks of CompiledProgram.cycles()",
            "latency_source": "vitis-csynth" if synthesize_design else "declared",
            "note": "area and Fmax are reported but not scored",
        },
        "latency_table": latency_table,
        "synthesis": synthesis,
        "benchmarks": benchmarks,
        "total_cycles": total_cycles,
        "total_bottleneck_cycles": sum(
            bench["bottleneck_cycles"] for bench in benchmarks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="score against the declared latency table instead of synthesizing",
    )
    parser.add_argument("--part", default=DEFAULT_PART)
    parser.add_argument("--freq", type=float, default=300.0)
    parser.add_argument("--project", default="tinytpu_synth_prj")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                synthesize_design=not args.frozen,
                part=args.part,
                freq_mhz=args.freq,
                project=args.project,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

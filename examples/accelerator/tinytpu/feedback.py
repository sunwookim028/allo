# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace-derived storage-traffic objective for TinyTPU ISA co-design.

The objective deliberately prices *declared storage accesses*, rather than
instruction names or a preferred data path. A candidate ISA can therefore earn a
lower score by making the MXU/VPU exchange data through VREG, but only if the
compiler can still allocate and execute the resulting trace.

Run with ``python -m examples.accelerator.tinytpu.feedback``. The defaults price
one VREG word at 1 and one BRAM/VMEM word at 4; override them with
``TINYTPU_VREG_WORD_COST`` and ``TINYTPU_VMEM_WORD_COST``.
"""

from __future__ import annotations

import json
from math import prod

import numpy as np

from allo.exp.dsa.core import access_map, param_roles, trace_instruction

from .isa import tpu
from .verify import _gemm_source


# Frozen architectural assumptions for this pre-synthesis search checkpoint.
# The agent cannot alter this evaluator, so it cannot improve a score by changing
# the weights rather than reducing the declared storage traffic.
VREG_WORD_COST = 1.0
VMEM_WORD_COST = 4.0
BENCHMARKS = ((8, 8, 8), (8, 16, 32))


def _words(pattern, shape_params: dict[int, int]) -> int:
    sizes = [size for size, _stride in access_map(pattern, shape_params)]
    assert all(isinstance(size, int) for size in sizes), sizes
    return prod(sizes)


def access_profile(prog) -> dict[str, int]:
    """Count read/write words by storage level from each emitted ISA access."""
    profile = {f"{level}_{direction}": 0 for level in ("vreg", "vmem", "dram") for direction in ("read", "write")}
    for emit in prog.emits:
        spec = tpu._ops[emit.name].spec
        patterns, _, _ = trace_instruction(spec)
        roles, _ = param_roles(spec)
        shape_params = {
            index: emit.addr[index]
            for index, role in roles.items()
            if role == "shape"
        }
        for index, (buffer, pattern) in enumerate(zip(spec.buffers, patterns)):
            level = "vmem" if buffer.name == "bram" else buffer.name
            if level not in ("vreg", "vmem", "dram"):
                continue
            direction = "read" if index < len(spec.sources) else "write"
            profile[f"{level}_{direction}"] += _words(pattern, shape_params)
    return profile


def evaluate_gemm(M: int, K: int, N: int) -> dict:
    """Compile and execute one standard TOSA GEMM, then return its traffic cost."""
    rng = np.random.default_rng(M * 10_000 + K * 100 + N)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    b = rng.standard_normal((1, K, N)).astype(np.float32)
    prog = tpu.compile_program(_gemm_source(M, K, N))
    np.testing.assert_allclose(prog(a, b), a @ b, rtol=1e-4, atol=1e-4)
    profile = access_profile(prog)
    vreg_words = profile["vreg_read"] + profile["vreg_write"]
    vmem_words = profile["vmem_read"] + profile["vmem_write"]
    return {
        "shape": [M, K, N],
        "instructions": len(prog.emits),
        "profile": profile,
        "vreg_words": vreg_words,
        "vmem_words": vmem_words,
        "access_cost": VREG_WORD_COST * vreg_words + VMEM_WORD_COST * vmem_words,
    }


def evaluate() -> dict:
    results = [evaluate_gemm(*shape) for shape in BENCHMARKS]
    return {
        "status": "pass",
        "cost_model": {
            "vreg_word_cost": VREG_WORD_COST,
            "vmem_word_cost": VMEM_WORD_COST,
            "objective": "sum(vreg_words * vreg_word_cost + vmem_words * vmem_word_cost)",
        },
        "benchmarks": results,
        "total_access_cost": sum(result["access_cost"] for result in results),
    }


def main() -> None:
    print(json.dumps(evaluate(), sort_keys=True))


if __name__ == "__main__":
    main()

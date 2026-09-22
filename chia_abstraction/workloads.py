# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload-level specifications. FROZEN: the agent can neither edit nor import it.

The loop's input is a WORKLOAD, not a file. A workload says what the machine has
to compute and at which sizes; it does not say how. Both dispositions of the loop
(`using` Allo's abstractions to change the design, `maintaining` Allo itself when
an abstraction is what blocks the design) start from the same workload spec and
are scored on the same shapes, so the two are comparable.

Keeping the spec an explicit input rather than a constant is the point: pointing
the loop at a new workload is a new entry here plus a design case that can run
it, not a change to the loop, the evaluator or the gates.

What limits the set today: the shipped TinyTPU-isa build caps at MAXDIM=16
(`cosim.py`'s `_ALL` is every (M, K, N) with each dimension in {4, 8, 12, 16}
and M, K, N multiples of T), so a GPT-2 projection -- 768x768 and 768x3072 --
cannot be expressed as one GEMM on this design yet. `gpt2_proj_tile` is
therefore recorded as a TILED workload: the shapes a 16-wide machine actually
executes when a 768-column projection is tiled over it, plus the tile count, so
the per-call cycle cost is measured now and the projection's cost is that
number times the tile count. It is marked `blocked_on` so nothing reports it as
an end-to-end GPT-2 result.
"""

from __future__ import annotations

#: Every shape `cosim.py` can build at the shipped TPU_MAXDIM=16 / TPU_T=4.
#: Anything a workload scores must be in here, or the frozen scorer refuses it.
COSIM_SHAPES = ["4x4x4", "8x8x8", "12x12x12", "16x16x8", "16x16x16"]

WORKLOADS = {
    "gemm_int8_16": {
        "title": "int8 GEMM, any multiple-of-4 shape up to 16x16x16",
        "statement": (
            "Compute C[M,N] += A[M,K] * B[K,N] in int8 with int32 accumulation, "
            "for any M, K, N that are multiples of 4 and at most 16, on one "
            "instruction-programmable machine: one build, one program memory, "
            "the shape carried by the program rather than by the RTL. Operands "
            "span the full int8 range, C is not assumed zeroed, nothing outside "
            "the MxN result may be written, and a single build must stay correct "
            "across repeated calls."
        ),
        #: Scored for PPA. Two shapes: the small one is latency-dominated
        #: (prologue, program fetch, drain), the large one throughput-dominated.
        "scored_shapes": ["4x4x4", "16x16x16"],
        #: Verified bit-exact at acceptance. All five, i.e. the whole workload.
        "verify_shapes": COSIM_SHAPES,
        "dtype": "int8 -> int32",
        #: Design cases that can execute this workload; see design_cases.py.
        "design_cases": ["tinytpu_isa", "systolic_gemm", "gemv_relay"],
        "blocked_on": None,
    },
    "gemm_int8_16_square": {
        "title": "int8 GEMM, square shapes only -- the cheap workload",
        "statement": (
            "As `gemm_int8_16`, restricted to square shapes. Used to make a "
            "harness test cheap; not a research workload."
        ),
        "scored_shapes": ["4x4x4"],
        "verify_shapes": ["4x4x4", "8x8x8"],
        "dtype": "int8 -> int32",
        "design_cases": ["tinytpu_isa"],
        "blocked_on": None,
    },
    "gpt2_proj_tile": {
        "title": "GPT-2 projection, as the tiles a 16-wide machine executes",
        "statement": (
            "GPT-2 small's attention output projection is C[768,768] = "
            "A[768,768] * B[768,768] and its MLP fc is [768,3072]. A machine "
            "that caps at 16x16x16 executes it as 48*48*48 = 110592 tiles of "
            "16x16x16 (and 4x that for fc). The workload therefore scores the "
            "16x16x16 tile, and the projection's cost is that number times the "
            "tile count -- which is only a valid model while the tile loop is "
            "the design's own program loop and no inter-tile reuse is claimed."
        ),
        "scored_shapes": ["16x16x16"],
        "verify_shapes": ["16x16x16", "16x16x8"],
        "dtype": "int8 -> int32",
        "design_cases": ["tinytpu_isa"],
        "tiles": {"attn_proj": 48 * 48 * 48, "mlp_fc": 48 * 48 * 192},
        #: Not an end-to-end GPT-2 measurement. A separate effort is raising
        #: MAXDIM; until it lands this workload is the tile, times a count.
        "blocked_on": (
            "MAXDIM caps at 16, so the 768-wide projection cannot be one GEMM "
            "on this design. Report the tile cycles and the tile count, never a "
            "GPT-2 end-to-end number."
        ),
    },
}

DEFAULT_WORKLOAD = "gemm_int8_16"


def get(name: str) -> dict:
    if name not in WORKLOADS:
        raise KeyError(f"unknown workload {name!r}; one of {sorted(WORKLOADS)}")
    spec = dict(WORKLOADS[name])
    spec["name"] = name
    bad = [s for s in spec["scored_shapes"] + spec["verify_shapes"]
           if s not in COSIM_SHAPES]
    if bad:
        raise ValueError(f"workload {name}: shapes {bad} are not buildable "
                         f"(COSIM_SHAPES = {COSIM_SHAPES})")
    return spec


def brief(spec: dict) -> str:
    """The workload as the agent is told it, in the task statement."""
    lines = [
        f"WORKLOAD `{spec['name']}` -- {spec['title']}",
        "",
        spec["statement"],
        "",
        f"  dtype          {spec['dtype']}",
        f"  scored shapes  {', '.join(spec['scored_shapes'])} (PPA)",
        f"  verified       {', '.join(spec['verify_shapes'])} (bit-exact)",
        f"  design cases   {', '.join(spec['design_cases'])}",
    ]
    if spec.get("tiles"):
        lines.append(f"  tiling         {spec['tiles']}")
    if spec.get("blocked_on"):
        lines += ["", f"  LIMIT: {spec['blocked_on']}"]
    return "\n".join(lines)

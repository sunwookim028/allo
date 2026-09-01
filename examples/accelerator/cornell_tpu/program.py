# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling external TOSA programs onto CornellTPU via ``tpu.compile_program``.

The source program is always an external TOSA-dialect module string -- here built
with torch_mlir's TOSA backend -- which the search backend covers with CornellTPU
instructions, allocates onto the buffers, and inserts data movement for. The
result is a ``CompiledProgram``; ``print(prog)`` shows the chosen instruction
sequence (with the allocated bram offsets / vreg slots), and ``prog(*inputs)``
runs it on the functional simulator.

The vector workloads are length-8, so the elementwise ops land on the 8-lane VPU
and the search must allocate the *8-register vector file* -- the interesting bit on
this architecture. The longer ones deliberately keep more than 8 values live at
once, forcing the allocator to spill registers to bram and reload them (visible as
extra ``vstore`` / ``vload`` mid-stream).

The matmul workloads are *pre-transposed* 2mm / 3mm. The systolic array computes
``Z = X @ W^T`` natively, so every matmul is written in that form and the host is
assumed to supply weights already transposed; the NumPy golden uses the same
convention. The weight transpose is absorbed by the systolic instruction (a TOSA
normalization, ``normalize_source``, sinks torch's 2-D transpose + batch reshape
into the batched form the instruction carries), so a chain of matmuls compiles to
pure ``matmul`` instructions with zero on-chip transposes -- even when a matmul
reads an *intermediate* result transposed (the 3mm case).

Importing ``microarch`` binds every instruction to the hardware unit that runs it and
declares each unit's ``(ii, depth)``, so the search minimizes **cycles** rather than
instruction count and each compiled program reports a cycle estimate. Without that
import the ISA still compiles — it just falls back to counting operations.

Each program prints two bounds and the unit between them. Every one of these turns out
to be ``dma_load``-bound, with the ``mxu`` close behind on the matmul chains (52 vs 56
on 2mm) — i.e. this microarchitecture is memory-bound and its systolic array is nearly
saturated. That is a statement no single cycle number can make.

Run all examples:  ``python -m example.accelerator.cornell_tpu.program``
"""

import numpy as np

from .isa import tpu
from . import microarch  # noqa: F401  -- binds instructions to units + latencies


def _tosa(model, *inputs) -> str:
    """Export a torch model to a TOSA module string (the source-program contract).
    torch is imported lazily, after allo, to avoid a dual-LLVM JIT clash."""
    import torch
    import torch_mlir.fx as fx

    tensors = [torch.from_numpy(np.asarray(x, np.float32)) for x in inputs]
    return str(
        fx.export_and_import(model.eval(), *tensors, output_type=fx.OutputType.TOSA)
    )


def _run(prog, inputs, want) -> None:
    """print() the compiled sequence, run it, and diff against NumPy."""
    print(prog)
    n_store = sum(1 for e in prog.emits if e.name == "vstore")
    spills = n_store - len(prog.outputs)
    note = f"{len(prog.emits)} instructions"
    note += f", {spills} register spill(s) to bram" if spills > 0 else ", no spills"
    # Every instruction is bound to a unit with a declared latency (microarch.py), so
    # the program's cycle count is bracketed: `cycles()` assumes nothing overlaps (an
    # upper bound), `bottleneck_cycles()` assumes every unit runs concurrently and
    # reports the busiest one (a lower bound). Naming that unit is the useful part --
    # it says where the time sits, which no single number can.
    units = prog.unit_cycles()
    busiest = max(units, key=lambda u: units[u])
    note += (
        f", {prog.cycles():.0f} cycles serial / "
        f"{prog.bottleneck_cycles():.0f} bound on '{busiest}'"
    )
    out = np.asarray(prog(*inputs), np.float32)
    np.testing.assert_allclose(out, np.asarray(want, np.float32), rtol=1e-4, atol=1e-4)
    print(f"  -> {note}; matches NumPy reference\n")


# ==========================================================================#
# Pre-transposed matmul chains (systolic-native Z = X @ W^T)
# ==========================================================================#
def two_mm_pretransposed() -> None:
    """Pre-transposed 2mm: out = (A @ B^T) @ C^T over 4x4 tiles. Each matmul is in
    the systolic's native X @ W^T form (host supplies B, C already transposed), so
    each maps to one `matmul` instruction with the weight transpose absorbed -- two
    matmuls, no data movement, all on bram."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c):
            return (a @ b.T) @ c.T

    rng = np.random.default_rng(10)
    a, b, c = (rng.standard_normal((4, 4)).astype(np.float32) for _ in range(3))
    _run(tpu.compile_program(_tosa(M(), a, b, c)), [a, b, c], (a @ b.T) @ c.T)


def three_mm_pretransposed() -> None:
    """Pre-transposed 3mm: out = (A @ B^T) @ (C @ D^T)^T. The third matmul reads the
    *intermediate* result (C @ D^T) transposed -- which the systolic does for free --
    so the whole chain is three `matmul`s with zero on-chip transposes, and the
    allocator reuses freed bram slots between them."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c, d):
            return (a @ b.T) @ (c @ d.T).T

    rng = np.random.default_rng(11)
    a, b, c, d = (rng.standard_normal((4, 4)).astype(np.float32) for _ in range(4))
    _run(
        tpu.compile_program(_tosa(M(), a, b, c, d)),
        [a, b, c, d],
        (a @ b.T) @ (c @ d.T).T,
    )


# ==========================================================================#
# 1. Wide reduction: sum of 10 independent pairwise adds  (spills heavily)
# ==========================================================================#
def wide_sum() -> None:
    """y = (x0+x1) + (x2+x3) + ... + (x18+x19), all length-8 vectors. torch emits
    the ten partial sums first, so all ten are live before the reduction begins --
    far more than the 8 vector registers. The allocator spills the surplus to bram
    and reloads them, which you can see as ``vstore``/``vload`` of intermediates."""
    import torch

    class M(torch.nn.Module):
        def forward(self, *xs):
            ts = [xs[2 * i] + xs[2 * i + 1] for i in range(len(xs) // 2)]
            acc = ts[0]
            for t in ts[1:]:
                acc = acc + t
            return acc

    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(8).astype(np.float32) for _ in range(20)]
    want = sum(xs[2 * i] + xs[2 * i + 1] for i in range(10))
    _run(tpu.compile_program(_tosa(M(), *xs)), xs, want)


# ==========================================================================#
# 2. Sum of 8 elementwise products  (spills: 8 products + operands > 8 regs)
# ==========================================================================#
def sum_of_products() -> None:
    """y = sum_i x_i * y_i over 8 pairs of length-8 vectors. The eight products are
    computed before the reduction, and loading the last operands while seven
    products are still live exceeds the register file -- another spill case, this
    time driven by multiplies."""
    import torch

    class M(torch.nn.Module):
        def forward(self, *xs):
            ps = [xs[2 * i] * xs[2 * i + 1] for i in range(len(xs) // 2)]
            acc = ps[0]
            for p in ps[1:]:
                acc = acc + p
            return acc

    rng = np.random.default_rng(1)
    xs = [rng.standard_normal(8).astype(np.float32) for _ in range(16)]
    want = sum(xs[2 * i] * xs[2 * i + 1] for i in range(8))
    _run(tpu.compile_program(_tosa(M(), *xs)), xs, want)


# ==========================================================================#
# 3. Deep fused chain  (in-place coalescing, register reuse, no spill)
# ==========================================================================#
def fused_chain() -> None:
    """y = relu(((a + b) * c - d) * e). A single deep chain: each result can reuse
    a dying operand's register (in-place coalescing), so the whole thing fits in a
    couple of registers -- the print shows the same low slot numbers recycled."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c, d, e):
            return torch.relu(((a + b) * c - d) * e)

    rng = np.random.default_rng(2)
    a, b, c, d, e = (rng.standard_normal(8).astype(np.float32) for _ in range(5))
    want = np.maximum(((a + b) * c - d) * e, 0.0)
    _run(tpu.compile_program(_tosa(M(), a, b, c, d, e)), [a, b, c, d, e], want)


# ==========================================================================#
# 4. Diamond: a shared subexpression materialized once
# ==========================================================================#
def diamond() -> None:
    """y = relu(s) * s - relu(a * b), where s = a + b is used twice. A multi-use
    value is a forced cut point, so the search covers s a single time and feeds it
    to both uses (one vadd, not two), keeping it live across the intervening ops."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b):
            s = a + b
            return torch.relu(s) * s - torch.relu(a * b)

    rng = np.random.default_rng(3)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    s = a + b
    want = np.maximum(s, 0.0) * s - np.maximum(a * b, 0.0)
    _run(tpu.compile_program(_tosa(M(), a, b)), [a, b], want)


# ==========================================================================#
# 5. Polynomial (Horner) in one variable: a kept live across the chain
# ==========================================================================#
def poly_horner() -> None:
    """y = ((a*a + b) * a + c) * a. The variable a is read at three points spread
    across the sequence, so it must stay resident in one register while the other
    operands churn through the rest of the file."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c):
            return ((a * a + b) * a + c) * a

    rng = np.random.default_rng(4)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    c = rng.standard_normal(8).astype(np.float32)
    want = ((a * a + b) * a + c) * a
    _run(tpu.compile_program(_tosa(M(), a, b, c)), [a, b, c], want)


EXAMPLES = [
    two_mm_pretransposed,
    three_mm_pretransposed,
    wide_sum,
    sum_of_products,
    fused_chain,
    diamond,
    poly_horner,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"All {len(EXAMPLES)} programs compiled, ran, and matched NumPy.")


if __name__ == "__main__":
    main()

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling external TOSA programs onto mininpu via ``npu.compile_program``.

The source program is always an external TOSA-dialect module string -- here built
with torch_mlir's TOSA backend -- which the search backend covers with mininpu
instructions, allocates onto the buffers, and inserts data movement for. The
result is a ``CompiledProgram``; ``print(prog)`` shows the chosen instruction
sequence (with the allocated vmem offsets / vr slots), and ``prog(*inputs)`` runs
it on the functional simulator.

The interesting case here is the matrix path. A source matmul is *one* op, but the
MXU is driven by three instructions, and the compiler never sees them as a group:
``vmatpush`` is the only one that computes, so the matcher selects it, and the
weight tile it reads (``mxu_w``) and the queue slot it writes (``mxu_q``) are then
reached by ordinary data movement -- ``vmatload`` in, ``vmatpop`` out, exactly like
``vld`` / ``vst`` around the VPU. Modeling the MXU's state as buffers is what makes
that fall out of routing instead of needing a special case.

Matmuls are written pre-transposed (``x @ W.T``): the array consumes the weight
tile transposed, so that form is the systolic-native one and the transpose is
absorbed into ``vmatpush`` -- and the host is assumed to supply weights in the
PyTorch ``[out, in]`` order. Shapes are the array's own (one 16-lane activation row
against a 16x16 tile); tiling a larger GEMM down to that is the mid-end's job, not
this backend's.

``rejected_programs`` at the end is part of the picture too: it shows the three
sources this ISA model turns away, and why each is a property of the model rather
than a missing case.

Run all examples:  ``python -m examples.accelerator.mininpu.program``
"""

from collections import Counter

import numpy as np

from allo.exp.dsa.errors import CompileError

from .isa import VEC_LANES, VEC_REGS, npu


def _tosa(model, *inputs) -> str:
    """Export a torch model to a TOSA module string (the source-program contract).
    torch is imported lazily, after allo, to avoid a dual-LLVM JIT clash."""
    import torch
    import torch_mlir.fx as fx

    tensors = [torch.from_numpy(np.asarray(x, np.float32)) for x in inputs]
    return str(
        fx.export_and_import(model.eval(), *tensors, output_type=fx.OutputType.TOSA)
    )


def _run(prog, inputs, want, brief=False) -> None:
    """print() the compiled sequence, run it, and diff against NumPy. ``brief``
    prints an instruction histogram instead of the listing (for long programs)."""
    if brief:
        counts = Counter(e.name for e in prog.emits)
        print(f"CompiledProgram[{prog.isa.name}]  io={prog.io_buffer.name}")
        print(f"  {len(prog.inputs)} inputs, {len(prog.outputs)} output(s)")
        print("  " + ", ".join(f"{n}x{k}" for k, n in counts.most_common()))
    else:
        print(prog)
    out = np.asarray(prog(*inputs), np.float32)
    np.testing.assert_allclose(out.reshape(np.shape(want)), want, rtol=1e-4, atol=1e-4)
    print(f"  -> {len(prog.emits)} instructions; matches NumPy reference\n")


def _rng(seed):
    rng = np.random.default_rng(seed)
    return lambda *shape: rng.standard_normal(shape).astype(np.float32)


# ==========================================================================#
# 1. RMSNorm: the MXU standing in for a reduction  (most interesting)
# ==========================================================================#
def rms_norm() -> None:
    """y = x * rsqrt(mean(x^2)) -- with the mean done *on the matrix unit*. The
    modeled ISA has no matchable reduction (see ``rejected_programs``), but a
    matmul against a constant 1/16 tile computes exactly the splatted mean, so the
    host stages that tile and the sum rides the MXU. The result: one vmul for
    x^2, a full load/push/pop round for the mean, then vrsqrt + vmul on the VPU --
    both datapaths in an eleven-instruction program."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, s):
            return x * torch.rsqrt((x * x) @ s.T)

    r = _rng(0)
    x = r(1, VEC_LANES)
    s = np.full((VEC_LANES, VEC_LANES), 1.0 / VEC_LANES, np.float32)
    _run(npu.compile_program(_tosa(M(), x, s)), [x, s], x / np.sqrt((x * x).mean()))


# ==========================================================================#
# 2. Residual block: two MXU rounds around a VPU scale-and-add
# ==========================================================================#
def residual_block() -> None:
    """y = ((x @ W1^T) * g + x) @ W2^T. The skip connection keeps ``x`` live in a
    register across the whole first MXU round, and the two layers each reload the
    stationary tile -- the MXU holds one, so ``vmatload`` is per-layer, while the
    activations never leave the register file between the two rounds."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, w1, w2, g):
            return ((x @ w1.T) * g + x) @ w2.T

    r = _rng(1)
    x, g = r(1, VEC_LANES), r(1, VEC_LANES)
    w1, w2 = r(VEC_LANES, VEC_LANES), r(VEC_LANES, VEC_LANES)
    want = ((x @ w1.T) * g + x) @ w2.T
    _run(npu.compile_program(_tosa(M(), x, w1, w2, g)), [x, w1, w2, g], want)


# ==========================================================================#
# 3. Two-layer perceptron: the weight tile is reloaded between layers
# ==========================================================================#
def two_layer_mlp() -> None:
    """y = (x @ W1^T) @ W2^T. The MXU holds one stationary tile, so the second
    layer's `vmatload` overwrites the first's -- the sequence is two full
    load/push/pop rounds, and the hidden row never leaves the register file."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, w1, w2):
            return (x @ w1.T) @ w2.T

    r = _rng(2)
    x = r(1, VEC_LANES)
    w1, w2 = r(VEC_LANES, VEC_LANES), r(VEC_LANES, VEC_LANES)
    _run(npu.compile_program(_tosa(M(), x, w1, w2)), [x, w1, w2], (x @ w1.T) @ w2.T)


# ==========================================================================#
# 4. Diamond over an MXU result: a shared value materialized once
# ==========================================================================#
def matmul_diamond() -> None:
    """y = h * h - h where h = x @ W^T. A multi-use value is a forced cut point, so
    the matmul is covered once and its popped register feeds all three reads --
    one load/push/pop round, not three."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, w):
            h = x @ w.T
            return h * h - h

    r = _rng(3)
    x, w = r(1, VEC_LANES), r(VEC_LANES, VEC_LANES)
    h = x @ w.T
    _run(npu.compile_program(_tosa(M(), x, w)), [x, w], h * h - h)


# ==========================================================================#
# 5. Matrix path, minimal: one activation row through the MXU
# ==========================================================================#
def matvec_pretransposed() -> None:
    """y = x @ W^T for a 1x16 x against a 16x16 weight -- the systolic's native
    form. The matcher picks `vmatpush`, and the router supplies its two MXU
    operands on its own: `vmatload` fills the stationary tile and `vmatpop` drains
    the result queue, with the DMA + `vld` staging that feeds them."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, w):
            return x @ w.T

    r = _rng(4)
    x, w = r(1, VEC_LANES), r(VEC_LANES, VEC_LANES)
    _run(npu.compile_program(_tosa(M(), x, w)), [x, w], x @ w.T)


# ==========================================================================#
# 6. Register pressure: 35 live vectors in a 32-entry file  (spills)
# ==========================================================================#
def register_pressure() -> None:
    """y = sum_i x_i * x_{i+1} over 36 length-16 vectors. torch emits all 35
    products before the reduction begins, so more values are live than the 32-entry
    register file holds and the allocator has to spill. The spill path runs to the
    io buffer, so each spilled vector leaves as `vst` + `vmemst` and comes back as
    `vmemld` + `vld` -- visible as the DMA/load counts exceeding the input count."""
    import torch

    class M(torch.nn.Module):
        def forward(self, *xs):
            ts = [xs[i] * xs[i + 1] for i in range(len(xs) - 1)]
            acc = ts[0]
            for t in ts[1:]:
                acc = acc + t
            return acc

    r = _rng(5)
    xs = [r(VEC_LANES) for _ in range(36)]
    want = sum(xs[i] * xs[i + 1] for i in range(len(xs) - 1))
    prog = npu.compile_program(_tosa(M(), *xs))
    counts = Counter(e.name for e in prog.emits)
    spills = counts["vmemst"] - len(prog.outputs)
    _run(prog, xs, want, brief=True)
    print(
        f"     {len(xs) - 1} products live in a {VEC_REGS}-entry file"
        f" -> {spills} spilled to dram and reloaded"
        f" ({counts['vld'] - len(xs)} extra vld)\n"
    )


# ==========================================================================#
# 7. Vector path: a fused elementwise chain
# ==========================================================================#
def fused_chain() -> None:
    """y = ((a + b) * c - a) over 16 lanes. A deep chain whose results reuse dying
    operands' registers, so it stays in a couple of vr slots; `a` is read again at
    the end, so it has to stay resident across the whole sequence."""
    import torch

    class M(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c - a

    r = _rng(6)
    a, b, c = r(VEC_LANES), r(VEC_LANES), r(VEC_LANES)
    _run(npu.compile_program(_tosa(M(), a, b, c)), [a, b, c], (a + b) * c - a)


# ==========================================================================#
# 8. Softmax's exponential: a literal that belongs to the instruction
# ==========================================================================#
def base2_exponential() -> None:
    """y = 2**x. The ``2`` is part of ``vexp``, not of the program -- the instruction
    takes one register and the base is wired into the hardware. Its compute region is
    ``pow(const(2.0), a)``, and torch lowers ``exp2`` to exactly that shape
    (``tosa.pow`` of a ``dense<2.0>`` constant), so the constant is what makes the two
    line up: a source ``pow(3, x)`` would not select this instruction."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.exp2(x)

    x = np.linspace(-3.0, 3.0, VEC_LANES).astype(np.float32)
    _run(npu.compile_program(_tosa(M(), x)), [x], 2.0**x)


# ==========================================================================#
# 9. The model's edges: sources this ISA refuses, and why
# ==========================================================================#
def rejected_programs() -> None:
    """Three programs the backend turns away. None is a missing case in the search
    -- each is a property of the ISA model, and the failure is loud and specific:

    - **relu**: 0.5.0 has no relu opcode. Its activations are ``vexp`` (modeled) and
      ``vgelu``, which is left out because the spec does not say *which* GELU --
      not, any more, because the constants were inexpressible.
    - **sum along a row**: ``vredsum`` *is* modeled, but its splat is written as
      ``reduce + const(0)``, so the pattern root is the splat's ``add``, not the
      reduce -- the matcher never reaches it. (A constant did not fix this; making
      the reduce the root needs a *broadcast* prim.) The reductions are oracle-only
      (hand-written assembly); ``rms_norm`` above is how a compiled program gets a
      row sum instead.
    - **tied weights**: ``(x @ W^T) @ W^T`` shares one transposed weight between
      two matmuls. A multi-use value must be materialized on its own, but the
      transpose only exists *inside* ``vmatpush``, so it cannot be shared -- the
      second matmul then finds no plain-matmul instruction. Passing the weight
      twice (two SSA values) compiles.
    """
    import torch

    class Relu(torch.nn.Module):
        def forward(self, x, w):
            return torch.relu(x @ w.T)

    class RowSum(torch.nn.Module):
        def forward(self, x):
            return x * x.sum(dim=1, keepdim=True)

    class Tied(torch.nn.Module):
        def forward(self, x, w):
            return (x @ w.T) @ w.T

    r = _rng(7)
    x, w = r(1, VEC_LANES), r(VEC_LANES, VEC_LANES)
    for label, model, inputs in [
        ("relu activation", Relu(), (x, w)),
        ("row sum", RowSum(), (x,)),
        ("tied weights", Tied(), (x, w)),
    ]:
        try:
            npu.compile_program(_tosa(model, *inputs))
            reason = None
        except CompileError as e:
            reason = str(e).strip().splitlines()[0]
        assert reason is not None, f"{label}: expected the backend to reject this"
        print(f"  [rejected] {label}: {reason}")
    print()


# ==========================================================================#
# 9. A whole layer: one match, expanded into the MXU's row loop
# ==========================================================================#
def whole_layer() -> None:
    """``Z[8,16] = X[8,16] @ W^T`` -- eight rows, matched as *one* layer.

    Every other example above multiplies a single row, which ``vmatpush`` (a
    fixed 1-row instruction) matches directly. Eight rows do not fit it, so the
    layer-level ``matmul_layer`` macro matches instead and its ``@I.expand``
    lowers it to the run the MXU actually needs: **one** ``vmatload`` -- the
    stationary tile is loaded once and amortized across the whole layer -- then
    eight ``vld`` / ``vmatpush`` / ``vmatpop`` / ``vst`` rounds. That the weight
    load is hoisted out of the row loop is encoded nowhere; the expansion loop
    *is* its definition, and the macro's ``@compute`` states what that loop must
    equal, so the two are diffable on the simulator."""
    import torch

    class M(torch.nn.Module):
        def forward(self, x, w):
            return x @ w.T

    r = _rng(11)
    x, w = r(8, VEC_LANES), r(VEC_LANES, VEC_LANES)
    _run(npu.compile_program(_tosa(M(), x, w)), [x, w], x @ w.T)


EXAMPLES = [
    rms_norm,
    residual_block,
    two_layer_mlp,
    matmul_diamond,
    matvec_pretransposed,
    register_pressure,
    fused_chain,
    whole_layer,
    base2_exponential,
    rejected_programs,
]


def main() -> None:
    for fn in EXAMPLES:
        print(f"=== {fn.__name__} ===")
        print(f"    {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print(f"All {len(EXAMPLES) - 1} programs compiled, ran, and matched NumPy;")
    print("the rejected-program boundary cases all failed as expected.")


if __name__ == "__main__":
    main()

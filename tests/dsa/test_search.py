# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search backend: compile torch_mlir TOSA programs onto CornellTPU and oracle-diff.

The source program is always an external TOSA module string (here from torch_mlir's
TOSA backend) — we own no source generator.
"""

import numpy as np
import pytest

from allo.exp.dsa.errors import (
    AcceleratorDescriptionError,
    AllocationError,
    NoMatchError,
    ShapeError,
)

from allo.exp.dsa import primitive
from allo.exp.dsa.access import collapse, contiguous, expand, strided, view
from allo.exp.dsa.core import ISA
from examples.accelerator.cornell_tpu.isa import tpu
from allo.lang.core import f32


def _fused_tpu(vaddrelu_cost=1.0):
    """A CornellTPU-like ISA that *also* has a fused ``vaddrelu = relu(add)``.

    ``vaddrelu_cost`` weights the fused op for the tree-DP cost model.
    """
    isa = ISA("FusedTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)
    vreg = isa.vector("vreg", slots=8, shape=(8,), dtype=f32)

    @isa.instruction(src=bram, dst=vreg)
    def vload(I):
        @I.access
        def _(s, d):
            return (contiguous(bram, s, 8), contiguous(vreg, d, 1))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=vreg, dst=bram)
    def vstore(I):
        @I.access
        def _(s, d):
            return (contiguous(vreg, s, 1), contiguous(bram, d, 8))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    def _ew3(I):
        @I.access
        def _(a, b, d):
            return (
                contiguous(vreg, a, 1),
                contiguous(vreg, b, 1),
                contiguous(vreg, d, 1),
            )

    @isa.instruction(src=[vreg, vreg], dst=vreg)
    def vadd(I):
        _ew3(I)

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    @isa.instruction(src=vreg, dst=vreg)
    def vrelu(I):
        @I.access
        def _(a, d):
            return (contiguous(vreg, a, 1), contiguous(vreg, d, 1))

        @I.compute
        def _(a, d):
            return primitive.relu(a)

    @isa.instruction(src=[vreg, vreg], dst=vreg, cost=vaddrelu_cost)
    def vaddrelu(I):
        _ew3(I)

        @I.compute
        def _(a, b, d):
            return primitive.relu(primitive.add(a, b))

    return isa


def _param_tpu():
    """An ISA with a single PARAMETRIC elementwise add over any length N: the
    access counts are a shape param the solver infers from the source."""
    isa = ISA("ParamTPU")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def add_n(I):
        @I.access
        def _(a, b, d, n):  # a,b,d = offsets; n = shape param (the count)
            return (contiguous(mem, a, n), contiguous(mem, b, n), contiguous(mem, d, n))

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    return isa


def _torch_tosa(model, *inputs) -> str:
    """Export a torch model to a TOSA module string (the source-program contract)."""
    import torch

    fx = pytest.importorskip("torch_mlir.fx")
    tensors = [torch.from_numpy(np.asarray(x, np.float32)) for x in inputs]
    module = fx.export_and_import(
        model.eval(), *tensors, output_type=fx.OutputType.TOSA
    )
    return str(module)


def _relu_add_src(*inputs) -> str:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class Model(torch.nn.Module):
        def forward(self, a, b):
            return F.relu(a + b)

    return _torch_tosa(Model(), *inputs)


def test_3mm_compiles_and_matches_golden():
    """Pre-transposed 3mm ``(a@b^T)@(c@d^T)^T``: the systolic computes X@W^T, so
    each matmul is written in that native form. Inputs are staged in dram and
    brought on-chip, so the chain compiles to three matmuls bracketed by dma
    moves (the freed tile slots are reused, so only one trailing dma_store)."""
    torch = pytest.importorskip("torch")

    class TMM(torch.nn.Module):
        def forward(self, a, b, c, d):
            return (a @ b.T) @ (c @ d.T).T

    rng = np.random.default_rng(0)
    A, B, C, D = (rng.standard_normal((4, 4)).astype(np.float32) for _ in range(4))
    prog = tpu.compile_program(_torch_tosa(TMM(), A, B, C, D))
    assert [e.name for e in prog.emits] == [
        "dma_load",
        "dma_load",
        "matmul",
        "dma_load",
        "dma_load",
        "matmul",
        "matmul",
        "dma_store",
    ]

    out = prog(A, B, C, D)
    np.testing.assert_allclose(out, (A @ B.T) @ (C @ D.T).T, rtol=1e-5, atol=1e-5)


def test_relu_add_inserts_data_movement():
    """relu(A+B): a vreg kernel with I/O in dram — search must auto-route each
    operand dram->bram->vreg (dma_load + vload) and the result back vreg->bram->dram
    (vstore + dma_store) over the move graph."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = tpu.compile_program(_relu_add_src(A, B))
    assert [e.name for e in prog.emits] == [
        "dma_load",
        "vload",
        "dma_load",
        "vload",
        "vadd",
        "vrelu",
        "vstore",
        "dma_store",
    ]

    out = prog(A, B)
    np.testing.assert_allclose(out, np.maximum(A + B, 0.0), rtol=1e-5, atol=1e-5)


def test_reject_unfittable_shape():
    """8x8 matmul in the systolic-native a@b^T form matches structurally but cannot
    fit the fixed 4x4 instruction (no tiling) -> reject on shape."""
    torch = pytest.importorskip("torch")

    class MM(torch.nn.Module):
        def forward(self, a, b):
            return a @ b.T

    x = np.zeros((8, 8), np.float32)
    # ShapeError, not NoMatchError: the structure *does* match (a @ b^T is the
    # systolic's native form) — it is Stage 2 that refuses, and the two are now
    # distinguishable, which a bare AssertionError could not express.
    with pytest.raises(ShapeError, match="expects 4 but source is 8"):
        tpu.compile_program(_torch_tosa(MM(), x, x))


def test_vaddrelu_fuses_and_runs():
    """relu(A+B) on the fused ISA: tree-DP folds add+relu into one vaddrelu (the
    larger tile), which then executes end-to-end through the oracle."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = _fused_tpu().compile_program(_relu_add_src(A, B))
    assert [e.name for e in prog.emits] == ["vload", "vload", "vaddrelu", "vstore"]

    out = prog(A, B)
    np.testing.assert_allclose(out, np.maximum(A + B, 0.0), rtol=1e-5, atol=1e-5)


def test_cost_model_rejects_expensive_fusion():
    """tree-DP is cost-aware: an expensive fused op is dropped for the cheaper
    vadd+vrelu cover — even though it is the larger tile greedy munch would pick."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = _fused_tpu(vaddrelu_cost=5.0).compile_program(_relu_add_src(A, B))
    assert [e.name for e in prog.emits] == ["vload", "vload", "vadd", "vrelu", "vstore"]


def test_shared_value_materialized_once():
    """A multi-use value is a forced cut point: covered once, fed to both uses."""
    torch = pytest.importorskip("torch")

    class Diamond(torch.nn.Module):
        def forward(self, a, b):
            s = a + b
            return torch.relu(s) * s

    rng = np.random.default_rng(2)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = tpu.compile_program(_torch_tosa(Diamond(), A, B))
    # one vadd (not two): the shared sum is materialized a single time, then fed to
    # both the vrelu and the vmul (I/O routed through dram on either end)
    assert [e.name for e in prog.emits] == [
        "dma_load",
        "vload",
        "dma_load",
        "vload",
        "vadd",
        "vrelu",
        "vmul",
        "vstore",
        "dma_store",
    ]

    out = prog(A, B)
    s = A + B
    np.testing.assert_allclose(out, np.maximum(s, 0.0) * s, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("length", [4, 8])
def test_parametric_add_any_length(length):
    """One parametric add_n covers a shape family: sympy solves N from the source
    add and the same instruction runs end-to-end for both length 4 and 8."""
    torch = pytest.importorskip("torch")

    class Add(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    rng = np.random.default_rng(length)
    A = rng.standard_normal(length).astype(np.float32)
    B = rng.standard_normal(length).astype(np.float32)
    prog = _param_tpu().compile_program(_torch_tosa(Add(), A, B))
    assert [e.name for e in prog.emits] == ["add_n"]
    # the solved shape param N is emitted as the count (last addr param)
    assert prog.emits[0].addr[-1] == length

    out = prog(A, B)
    np.testing.assert_allclose(out, A + B, rtol=1e-5, atol=1e-5)


# ==========================================================================#
# Stage 3 — liveness-driven allocation, slot reuse, in-place coalescing
# ==========================================================================#


def _vec_tpu(slots):
    """A vreg-only elementwise ISA with a tunable register count, to stress the
    allocator: 4 inputs feeding a chain/tree of adds onto ``slots`` registers."""
    isa = ISA("VecTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)
    vreg = isa.vector("vreg", slots=slots, shape=(8,), dtype=f32)

    @isa.instruction(src=bram, dst=vreg)
    def vload(I):
        @I.access
        def _(s, d):
            return (contiguous(bram, s, 8), contiguous(vreg, d, 1))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=vreg, dst=bram)
    def vstore(I):
        @I.access
        def _(s, d):
            return (contiguous(vreg, s, 1), contiguous(bram, d, 8))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=[vreg, vreg], dst=vreg)
    def vadd(I):
        @I.access
        def _(a, b, d):
            return (
                contiguous(vreg, a, 1),
                contiguous(vreg, b, 1),
                contiguous(vreg, d, 1),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    return isa


def _matmul_tpu(bram_slots):
    """A bram-only 4x4 matmul ISA with a tunable scratchpad size."""
    isa = ISA("MMTPU")
    bram = isa.global_("bram", shape=(bram_slots,), dtype=f32)

    @isa.instruction(src=[bram, bram], dst=bram)
    def matmul(I):
        @I.access
        def _(x, w, z):
            return (
                view(bram, x, (1, 4, 4)),
                view(bram, w, (1, 4, 4)),
                view(bram, z, (1, 4, 4)),
            )

        @I.compute
        def _(x, w, z):
            return primitive.matmul(x, w)

    return isa


def _hier_tpu():
    """A 3-level hierarchy with NO direct bram<->vreg move: bram <-> spm <-> vreg.
    Reaching the vreg VPU from bram forces the allocator to route through spm."""
    isa = ISA("HierTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)
    spm = isa.scalar("spm", slots=256, dtype=f32)
    vreg = isa.vector("vreg", slots=8, shape=(8,), dtype=f32)

    def _copy(I, src, dst, sc, dc):
        @I.access
        def _(s, d):
            return (contiguous(src, s, sc), contiguous(dst, d, dc))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=bram, dst=spm)
    def b2s(I):
        _copy(I, bram, spm, 8, 8)

    @isa.instruction(src=spm, dst=bram)
    def s2b(I):
        _copy(I, spm, bram, 8, 8)

    @isa.instruction(src=spm, dst=vreg)
    def s2v(I):
        _copy(I, spm, vreg, 8, 1)

    @isa.instruction(src=vreg, dst=spm)
    def v2s(I):
        _copy(I, vreg, spm, 1, 8)

    @isa.instruction(src=[vreg, vreg], dst=vreg)
    def vadd(I):
        @I.access
        def _(a, b, d):
            return (
                contiguous(vreg, a, 1),
                contiguous(vreg, b, 1),
                contiguous(vreg, d, 1),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    return isa


@pytest.mark.parametrize("slots", [2, 3, 8])
def test_chain_add_fits_via_reuse(slots):
    """``a+b+c+d`` makes 7 vreg values but never more than 2 are live at once.
    Liveness + in-place coalescing collapse them onto 2 registers, so it runs even
    on a 2-slot file (bump allocation would have needed 7)."""
    torch = pytest.importorskip("torch")

    class Chain(torch.nn.Module):
        def forward(self, a, b, c, d):
            return a + b + c + d

    rng = np.random.default_rng(7)
    xs = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]
    prog = _vec_tpu(slots).compile_program(_torch_tosa(Chain(), *xs))
    assert [e.name for e in prog.emits] == [
        "vload",
        "vload",
        "vadd",
        "vload",
        "vadd",
        "vload",
        "vadd",
        "vstore",
    ]

    out = prog(*xs)
    np.testing.assert_allclose(out, sum(xs), rtol=1e-5, atol=1e-5)


def test_chain_add_overflow_below_peak():
    """One register cannot hold a binary add's two live operands; spilling can't fix
    an instruction whose own operands don't fit -> rejected upfront (capacity)."""
    torch = pytest.importorskip("torch")

    class Chain(torch.nn.Module):
        def forward(self, a, b, c, d):
            return a + b + c + d

    rng = np.random.default_rng(7)
    xs = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]
    with pytest.raises(AllocationError, match="capacity too small"):
        _vec_tpu(1).compile_program(_torch_tosa(Chain(), *xs))


def test_tree_add_spills_at_capacity():
    """``(a+b)+(c+d)`` has register pressure 3 (one partial sum live while building
    the other pair). At 3 registers it fits with no spill; at 2 the allocator spills
    a partial sum to the backing store and reloads it (extra vstore/vload) — both run
    correctly, proving P-B turns an over-pressure kernel into runnable code."""
    torch = pytest.importorskip("torch")

    class Tree(torch.nn.Module):
        def forward(self, a, b, c, d):
            return (a + b) + (c + d)

    rng = np.random.default_rng(11)
    xs = [rng.standard_normal(8).astype(np.float32) for _ in range(4)]
    src = _torch_tosa(Tree(), *xs)
    expected = (xs[0] + xs[1]) + (xs[2] + xs[3])

    prog3 = _vec_tpu(3).compile_program(src)
    prog2 = _vec_tpu(2).compile_program(src)
    # the 2-register build spilled: strictly more data movement than the 3-register one
    moves3 = [e.name for e in prog3.emits if e.name in ("vload", "vstore")]
    moves2 = [e.name for e in prog2.emits if e.name in ("vload", "vstore")]
    assert len(moves2) > len(moves3)

    np.testing.assert_allclose(prog3(*xs), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(prog2(*xs), expected, rtol=1e-5, atol=1e-5)


def test_matmul_holds_operands_through_op():
    """A matmul reads every operand element repeatedly, so its result must NOT
    coalesce in place. The allocator frees operands only *after* placing the
    result: 48 slots (A+B+C, all distinct) run correctly, while 32 slots — enough
    only if the result wrongly aliased a still-read operand — overflow."""
    torch = pytest.importorskip("torch")

    class MM(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 4)).astype(np.float32)
    B = rng.standard_normal((4, 4)).astype(np.float32)
    src = _torch_tosa(MM(), A, B)

    out = _matmul_tpu(48).compile_program(src)(A, B)
    np.testing.assert_allclose(out, A @ B, rtol=1e-5, atol=1e-5)

    with pytest.raises(AllocationError, match="overflow"):
        _matmul_tpu(32).compile_program(src)


def test_multihop_routing():
    """The VPU is on vreg but there is no direct bram<->vreg move. The allocator
    routes each operand over the move graph bram->spm->vreg (and the result back
    vreg->spm->bram), so a plain ``a+b`` lowers to a two-hop load/store chain."""
    rng = np.random.default_rng(13)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = _hier_tpu().compile_program(_torch_add(A, B))
    assert [e.name for e in prog.emits] == [
        "b2s",
        "s2v",
        "b2s",
        "s2v",
        "vadd",
        "v2s",
        "s2b",
    ]

    out = prog(A, B)
    np.testing.assert_allclose(out, A + B, rtol=1e-5, atol=1e-5)


def _torch_add(*inputs) -> str:
    torch = pytest.importorskip("torch")

    class Add(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    return _torch_tosa(Add(), *inputs)


def _torch_matmul(A, B) -> str:
    torch = pytest.importorskip("torch")

    class MM(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    return _torch_tosa(MM(), A, B)


# ==========================================================================#
# Stage 2 — shape inference as constraint solving (the rewritten `solve`)
# ==========================================================================#


def _param_matmul_tpu():
    """A fully PARAMETRIC matmul: M, K, N are shape params the solver must infer.
    K appears in *both* operands, so the system is over-determined-but-consistent."""
    isa = ISA("ParamMM")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)

    @isa.instruction(src=[bram, bram], dst=bram)
    def matmul(I):
        @I.access
        def _(x, w, z, M, K, N):
            return (
                view(bram, x, (1, M, K)),
                view(bram, w, (1, K, N)),
                view(bram, z, (1, M, N)),
            )

        @I.compute
        def _(x, w, z):
            return primitive.matmul(x, w)

    return isa


def _one_param_add(build_patterns):
    """A bram-only elementwise add whose access is supplied by ``build_patterns`` —
    a knob to construct each pathological shape system for the solver."""
    isa = ISA("ProbeTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)

    @isa.instruction(src=[bram, bram], dst=bram)
    def padd(I):
        I.access(build_patterns(bram))

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    return isa


def test_param_matmul_infers_mkn():
    """``(4x8) @ (8x16)``: M,K,N are inferred from the operands+result (K twice,
    consistently). One parametric instruction covers the whole matmul shape family."""
    rng = np.random.default_rng(21)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 16)).astype(np.float32)
    prog = _param_matmul_tpu().compile_program(_torch_matmul(A, B))
    assert [e.name for e in prog.emits] == ["matmul"]
    # solved (M, K, N) = (4, 8, 16) emitted as the trailing addr params
    assert prog.emits[0].addr[-3:] == [4, 8, 16]

    out = prog(A, B)
    np.testing.assert_allclose(out, A @ B, rtol=1e-5, atol=1e-5)


def test_inconsistent_shape_rejected():
    """A square-only add (operands forced to N x N) matched against a 4x8 source:
    N == 4 and N == 8 cannot both hold -> the system is inconsistent."""

    def square(bram):
        def access(a, b, d, N):
            mk = lambda base: expand(  # noqa: E731
                strided(bram, basis=base, counts=N * N, strides=1),
                [[0, 1]],
                shape=(N, N),
            )
            return (mk(a), mk(b), mk(d))

        return access

    rng = np.random.default_rng(22)
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((4, 8)).astype(np.float32)
    with pytest.raises(ShapeError, match="inconsistent"):
        _one_param_add(square).compile_program(_torch_add(A, B))


def test_parametric_cost_uses_solved_shape_params():
    """``cost`` may be a callable over the instruction's shape params, so a parametric
    instruction can price itself by the work it actually does at each site."""
    isa = ISA("CostTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)

    @isa.instruction(src=[bram, bram], dst=bram, cost=lambda n: n / 8)
    def padd(I):
        @I.access
        def _(a, b, d, n):
            return (
                contiguous(bram, a, n),
                contiguous(bram, b, n),
                contiguous(bram, d, n),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    rng = np.random.default_rng(30)
    A = rng.standard_normal(32).astype(np.float32)
    prog = isa.compile_program(_torch_add(A, A))
    assert prog.emits[0].addr[-1] == 32  # n solved once, in Stage 1
    assert padd.spec.cost_of({3: 8}) == 1.0 and padd.spec.cost_of({3: 32}) == 4.0


def test_cost_param_must_be_a_shape_param():
    """A cost callable naming something that is not a solved shape param fails loudly
    rather than silently pricing the instruction at zero."""
    isa = ISA("BadCostTPU")
    bram = isa.global_("bram", shape=(8192,), dtype=f32)

    @isa.instruction(src=[bram, bram], dst=bram, cost=lambda rows: rows)
    def padd(I):
        @I.access
        def _(a, b, d, n):
            return (
                contiguous(bram, a, n),
                contiguous(bram, b, n),
                contiguous(bram, d, n),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    rng = np.random.default_rng(31)
    A = rng.standard_normal(8).astype(np.float32)
    with pytest.raises(AcceleratorDescriptionError, match="cost needs param"):
        isa.compile_program(_torch_add(A, A))


def test_underconstrained_param_rejected():
    """``count = p + q`` against a length-8 source gives one equation in two params:
    under-determined, so the solver rejects and names the free param."""

    def split(bram):
        def access(a, b, d, p, q):
            mk = lambda base: strided(
                bram, basis=base, counts=p + q, strides=1
            )  # noqa: E731
            return (mk(a), mk(b), mk(d))

        return access

    rng = np.random.default_rng(23)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    with pytest.raises(ShapeError, match="under-constrained"):
        _one_param_add(split).compile_program(_torch_add(A, B))


def test_collapse_two_symbolic_dims_rejected():
    """Collapsing two symbolic dims gives ``p * q == M`` — a factorization with no
    unique answer. The nonlinear constraint is rejected up front."""

    def folded(bram):
        def access(a, b, d, p, q):
            mk = lambda base: collapse(  # noqa: E731
                strided(bram, basis=base, counts=(p, q), strides=(1, 1)), [[0, 1]]
            )
            return (mk(a), mk(b), mk(d))

        return access

    rng = np.random.default_rng(24)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    with pytest.raises(ShapeError, match="nonlinear"):
        _one_param_add(folded).compile_program(_torch_add(A, B))


def test_stride_param_is_solved_from_residence():
    """A param that appears only as an addressing stride carries no *shape* info, so
    Stage 2 cannot see it — Stage 2b pins it from where the data actually lives. Here
    every operand is program I/O, which the host ABI lays out densely, so ``s`` = 1."""

    def strided_param(bram):
        def access(a, b, d, s):
            mk = lambda base: strided(
                bram, basis=base, counts=8, strides=s
            )  # noqa: E731
            return (mk(a), mk(b), mk(d))

        return access

    rng = np.random.default_rng(25)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = _one_param_add(strided_param).compile_program(_torch_add(A, B))
    assert prog.emits[0].addr[3] == 1
    np.testing.assert_allclose(prog(A, B), A + B, rtol=1e-5, atol=1e-5)


def test_dump_prints_instruction_sequence(capsys):
    """``CompiledProgram.dump()`` prints the I/O map and the emit stream in order."""
    rng = np.random.default_rng(31)
    A = rng.standard_normal(8).astype(np.float32)
    B = rng.standard_normal(8).astype(np.float32)
    prog = tpu.compile_program(_relu_add_src(A, B))

    prog.dump()
    text = capsys.readouterr().out
    assert "CompiledProgram[CornellTPU]  io=dram" in text
    assert "arg0 = dram[" in text and "out0 = dram[" in text
    # the emit stream appears in program order, one mnemonic per line
    block = text.split("  program:")[1].split("  outputs:")[0]
    emitted = [ln.strip().split("(")[0] for ln in block.strip().splitlines()]
    assert emitted == [e.name for e in prog.emits]

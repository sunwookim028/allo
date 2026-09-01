# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Microarchitecture-derived cost: ``ISA.bind`` + ``ISA.latency``, and the two bounds
a placed program is reported between — ``cycles()`` (nothing overlaps, upper) and
``bottleneck_cycles()`` / ``unit_cycles()`` (every unit runs concurrently, lower).

Importing ``cornell_tpu.microarch`` binds every CornellTPU instruction to the unit
that runs it, so from here on the ISA prices instructions in cycles rather than
counting them. That binding is global and permanent for the session — verified
benign: the whole suite selects identically with it on or off.
"""

import numpy as np
import pytest

from allo.exp.dsa.errors import AcceleratorDescriptionError

from examples.accelerator.cornell_tpu import microarch  # noqa: F401  (binds the ISA)
from examples.accelerator.cornell_tpu.isa import tpu, VEC_LANES


def _torch_relu_add(a, b) -> str:
    import torch

    fx = pytest.importorskip("torch_mlir.fx")

    class M(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(x + y)

    tensors = [torch.from_numpy(np.asarray(t, np.float32)) for t in (a, b)]
    return str(
        fx.export_and_import(M().eval(), *tensors, output_type=fx.OutputType.TOSA)
    )


def test_bind_derives_cost_from_unit_latency():
    """A bound instruction costs ``depth + ii * trips`` — a cycle count read off the
    microarchitecture, not a hand-assigned weight."""
    vadd, mxu_op = tpu._ops["vadd"].spec, tpu._ops["matmul"].spec
    assert vadd.unit.func_name == "vpu"
    assert vadd.cost_of({}) == 5 + 1 * VEC_LANES  # vpu: ii=1, depth=5, 8 lanes
    assert mxu_op.cost_of({}) == 20 + 1 * 16  # mxu: ii=1, depth=20, 4x4 tile


def test_bound_unit_is_shared_across_opcodes():
    """The four VPU ops are one opcode-multiplexed datapath: latency is declared once
    on the unit, and every instruction bound to it shares that record."""
    specs = [tpu._ops[n].spec for n in ("vadd", "vsub", "vmul", "vrelu")]
    assert {s.unit.func_name for s in specs} == {"vpu"}
    assert len({id(s.unit_latency) for s in specs}) == 1  # the same shared object
    assert len({s.cost_of({}) for s in specs}) == 1


def test_trips_scale_a_parametric_movers_cost():
    """``dma_load`` is a burst copy, so its cost must scale with the block it moves —
    ``trips=lambda n: n``. A per-instruction constant could not express this."""
    spec = tpu._ops["dma_load"].spec
    n_param = 2  # access is (s, d, n)
    assert spec.cost_of({n_param: 8}) == 8 + 1 * 8
    assert spec.cost_of({n_param: 64}) == 8 + 1 * 64


def test_program_cycles_sums_the_placed_program():
    """``cycles()`` prices the whole placed program, recovering each emit's solved
    shape params from its address list."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(VEC_LANES).astype(np.float32)
    prog = tpu.compile_program(_torch_relu_add(a, a))
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
    # 2*(8+8) dma + 2*(3+8) vload + (5+8) vadd + (5+8) vrelu + (3+8) vstore + (8+8)
    assert prog.cycles() == 107.0


# --- per-unit occupancy: where the time actually sits ---------------------------


def _relu_add_program(n=VEC_LANES):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    return tpu.compile_program(_torch_relu_add(a, a))


def test_unit_cycles_attributes_each_instruction_to_the_unit_that_runs_it():
    """``cycles()`` cannot say *where* a program's time goes — it has no notion of
    which unit runs what. ``unit_cycles()`` does, and that is the whole point: it is
    the quantity that changes when a transformation moves work between units."""
    units = _relu_add_program().unit_cycles()
    assert set(units) == {"dma_load", "vload", "vpu", "vstore", "dma_store"}
    assert "mxu" not in units  # no matmul in this program, so the array is idle


def test_a_shared_unit_pays_its_drain_once_not_per_instruction():
    """``vadd`` and ``vrelu`` are one opcode-multiplexed datapath. Back to back they
    occupy it for ``2 * ii * lanes`` and drain its 5-deep pipeline **once** — which is
    exactly what a per-instruction sum cannot express: ``cycles()`` charges depth twice.
    """
    units = _relu_add_program().unit_cycles()
    assert units["vpu"] == 2 * 1 * VEC_LANES + 5  # 2 ops, ii=1, one drain of depth 5
    # the serial model charges the drain per instruction: 2*(5 + 8) = 26, not 21
    assert units["vpu"] < 2 * (5 + VEC_LANES)


def test_bottleneck_is_the_busiest_unit_and_brackets_the_serial_estimate():
    """The two models bound the same program from opposite sides: nothing overlaps
    (``cycles()``, upper) versus everything does (``bottleneck_cycles()``, lower)."""
    prog = _relu_add_program()
    units = prog.unit_cycles()
    assert prog.bottleneck_cycles() == max(units.values())
    assert prog.bottleneck_cycles() <= prog.cycles()


def test_spilling_moves_the_bottleneck_onto_the_mover():
    """A program that spills adds *data movement*, not compute. The serial estimate
    charges that as if it could never overlap; the per-unit model localizes it — the
    spill traffic piles onto ``dma_load`` and leaves the VPU well below it.

    This is the reason the bound exists: a tiling that trades recompute for traffic is
    scored wrongly by a sum over instructions and correctly by the busiest unit."""
    import torch

    fx = pytest.importorskip("torch_mlir.fx")

    class M(torch.nn.Module):
        def forward(self, *xs):
            ts = [xs[2 * i] + xs[2 * i + 1] for i in range(len(xs) // 2)]
            acc = ts[0]
            for t in ts[1:]:
                acc = acc + t
            return acc

    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(VEC_LANES).astype(np.float32) for _ in range(20)]
    tensors = [torch.from_numpy(x) for x in xs]
    src = str(
        fx.export_and_import(M().eval(), *tensors, output_type=fx.OutputType.TOSA)
    )
    prog = tpu.compile_program(src)

    assert sum(1 for e in prog.emits if e.name == "vstore") > len(
        prog.outputs
    )  # spills
    units = prog.unit_cycles()
    assert max(units, key=lambda u: units[u]) == "dma_load"
    assert units["vpu"] < units["dma_load"]
    # the serial model is several times the roofline here, and the gap *is* the spills
    assert prog.cycles() > 4 * prog.bottleneck_cycles()


def test_cycles_refuses_an_unmodeled_isa():
    """An ISA with no bound units has no cycle model; every reporting entry point says
    so instead of presenting the abstract search weight as if it were cycles."""
    from allo.exp.dsa import primitive
    from allo.exp.dsa.access import contiguous
    from allo.exp.dsa.core import ISA
    from allo.lang.core import f32

    bare = ISA("no-microarch")  # declares instructions, binds no @unit
    dram = bare.global_("dram", shape=(256,), dtype=f32)
    vreg = bare.vector("vreg", slots=4, shape=(8,), dtype=f32)

    @bare.instruction(src=dram, dst=vreg)
    def load(I):
        @I.access
        def _(s, d):
            return (contiguous(dram, s, 8), contiguous(vreg, d, 1))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @bare.instruction(src=vreg, dst=dram)
    def store(I):
        @I.access
        def _(s, d):
            return (contiguous(vreg, s, 1), contiguous(dram, d, 8))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @bare.instruction(src=[vreg, vreg], dst=vreg)
    def vadd(I):
        @I.access
        def _(x, y, d):
            return (
                contiguous(vreg, x, 1),
                contiguous(vreg, y, 1),
                contiguous(vreg, d, 1),
            )

        @I.compute
        def _(x, y, d):
            return primitive.add(x, y)

    prog = bare.compile_program(
        """
func.func @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %r = tosa.add %a, %b : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
"""
    )
    for report in (prog.cycles, prog.unit_cycles, prog.bottleneck_cycles):
        with pytest.raises(AcceleratorDescriptionError, match="has no cycle model"):
            report()

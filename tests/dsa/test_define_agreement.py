# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An instruction's two regions must agree on what it writes.

``@I.access`` says where the destination lives and how much of it the instruction
touches; ``@I.compute`` says what value comes out. They are traced independently and
nothing forced them to describe the same tensor: an instruction could declare a
4-element destination and yield 8 elements, and the mismatch would surface only much
later — as a wrong shape in the lowered IR, or not at all.

Checked in two places on purpose. ``trace_instruction`` catches it in Python, at
``catalog()`` time, with the buffer's name in the message; ``DefineOp::verify``
catches it in the IR, so a catalog written by hand or by another producer is guarded
too. The second half of this file goes through MLIR text to reach the C++ verifier,
since the Python check would otherwise always fire first.
"""

import pytest

from allo._mlir import ir
from allo._mlir.dialects import allo as allo_d
from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import AcceleratorDescriptionError
from allo.lang.core import f32


def _isa(name):
    isa = ISA(name)
    return isa, isa.global_("mem", shape=(256,), dtype=f32)


def _define(isa, mem, access, compute):
    @isa.instruction(src=mem, dst=mem, name="op")
    def _(I):
        I.access(access)
        I.compute(compute)


# ==========================================================================#
# Python: caught while building the catalog
# ==========================================================================#


def test_a_destination_that_is_too_small_is_rejected():
    """The instruction claims 4 slots of ``mem`` but computes 8 values."""
    isa, mem = _isa("narrow-dst")
    _define(
        isa,
        mem,
        lambda s, d: (contiguous(mem, s, 8), contiguous(mem, d, 4)),
        lambda a, o: primitive.relu(a),
    )
    with pytest.raises(AcceleratorDescriptionError, match=r"axis 0: 8 vs 4"):
        isa.catalog()


def test_a_destination_of_the_wrong_rank_is_rejected():
    isa, mem = _isa("wrong-rank")
    _define(
        isa,
        mem,
        lambda s, d: (view(mem, s, (2, 4)), contiguous(mem, d, 8)),
        lambda a, o: primitive.relu(a),
    )
    with pytest.raises(AcceleratorDescriptionError, match="rank-2 .* writes rank-1"):
        isa.catalog()


def test_a_reduce_whose_destination_keeps_the_reduced_axis_is_rejected():
    """The realistic version: the compute reduces (4, 4) -> (4, 1) but the access
    pattern still describes the full tile."""
    isa, mem = _isa("reduce-dst")
    _define(
        isa,
        mem,
        lambda s, d: (view(mem, s, (4, 4)), view(mem, d, (4, 4))),
        lambda a, o: primitive.reduce_sum(a, axis=1),
    )
    with pytest.raises(AcceleratorDescriptionError, match=r"axis 1: 1 vs 4"):
        isa.catalog()


def test_the_wrong_number_of_yielded_values_is_rejected():
    isa, mem = _isa("two-yields")
    _define(
        isa,
        mem,
        lambda s, d: (contiguous(mem, s, 8), contiguous(mem, d, 8)),
        lambda a, o: (primitive.relu(a), primitive.abs(a)),
    )
    with pytest.raises(AcceleratorDescriptionError, match="must yield 1 value"):
        isa.catalog()


def test_a_parametric_destination_is_not_second_guessed():
    """The control. A dim that depends on an address param is solved per call site
    (Stage 2), so it is unknown here — the check must compare only what is static,
    or every parametric instruction becomes undescribable."""
    isa, mem = _isa("parametric")
    _define(
        isa,
        mem,
        lambda s, n, d, m: (contiguous(mem, s, n), contiguous(mem, d, m)),
        lambda a, o: primitive.relu(a),
    )
    assert isa.catalog().operation.verify()


# ==========================================================================#
# C++: the same check on the IR, reached by hand-written MLIR
# ==========================================================================#


def _catalog(dst_slots: int, yields: str) -> str:
    return f"""
module {{
  allo.buffer @mem extents(256) : !allo.scalar<f32>
  allo.define @op {{
    src(@mem) dst(@mem)
    addr(%s: index, %d: index) {{
      %0 = allo.patterns.strided basis(%s) counts(8) strides(1)
      %1 = allo.patterns.strided basis(%d) counts({dst_slots}) strides(1)
      allo.yield %0, %1 : !allo.pattern, !allo.pattern
    }}
    compute(%x: tensor<8xf32>, %o: tensor<{dst_slots}xf32>){{
      {yields}
    }}
  }}
}}
"""


_ADD = """%r = tosa.add %x, %x : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
      allo.yield %r : tensor<8xf32>"""


def _parse(text: str):
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        return str(ir.Module.parse(text))


def test_the_ir_verifier_accepts_a_matching_define():
    """The control: same catalog, destination sized to what the compute yields."""
    assert "allo.define" in _parse(_catalog(8, _ADD))


def test_the_ir_verifier_rejects_a_mismatched_define():
    """A hand-written catalog cannot smuggle the mismatch past the Python check."""
    with pytest.raises(ir.MLIRError, match="but its access pattern writes"):
        _parse(_catalog(4, _ADD))

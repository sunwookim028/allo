# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulation across instructions (an already-tiled K-reduction).

A hardware "mac" instruction accumulates a partial product into a running sum:
``C += A @ B``. Since tiling is the mid-end's job, the backend sees the reduction
*already* unrolled into a chain of accumulate ops and must select them. The faithful
model carries the accumulator as an explicit **read operand** (exactly what the
hardware's block buffer is); the allocator then coalesces the chain back onto **one**
slot. An instruction that instead reads its own *destination* (a true in-place
accumulate) is oracle-only — the matcher cannot bind a dst read, so it is refused at
compile time rather than mis-compiled.

Sources are hand-written value-semantics TOSA (clean tile IR), so no torch."""

import numpy as np
import pytest

from allo.exp.dsa.errors import NoMatchError

from allo.exp.dsa import primitive
from allo.exp.dsa.access import view
from allo.exp.dsa.core import ISA
from allo.lang.core import f32


def _mac_isa():
    """``mm`` (C = A@B) + ``mm_acc`` (C_out = C_in + A@B), 4x4 tiles in one pool."""
    isa = ISA("mac")
    mem = isa.global_("mem", shape=(512,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def mm(I):
        @I.access
        def _(a, b, d):
            return (
                view(mem, a, (1, 4, 4)),
                view(mem, b, (1, 4, 4)),
                view(mem, d, (1, 4, 4)),
            )

        @I.compute
        def _(a, b, d):
            return primitive.matmul(a, b)

    @isa.instruction(src=[mem, mem, mem], dst=mem)
    def mm_acc(I):
        @I.access
        def _(a, b, c, d):
            return (
                view(mem, a, (1, 4, 4)),
                view(mem, b, (1, 4, 4)),
                view(mem, c, (1, 4, 4)),  # accumulator (a read operand)
                view(mem, d, (1, 4, 4)),
            )

        @I.compute
        def _(a, b, c, d):
            return primitive.add(c, primitive.matmul(a, b))

    return isa


def _kreduction_src(n_tiles):
    """C = sum_i A_i @ B_i as hand-written 3-D TOSA: matmuls summed left-to-right."""
    args = ", ".join(
        f"%a{i}: tensor<1x4x4xf32>, %b{i}: tensor<1x4x4xf32>" for i in range(n_tiles)
    )
    lines = [
        '  %zp = "tosa.const"() {values = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>'
    ]
    ty = "(tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x4xf32>"
    for i in range(n_tiles):
        lines.append(f"  %m{i} = tosa.matmul %a{i}, %b{i}, %zp, %zp : {ty}")
    acc = "%m0"
    for i in range(1, n_tiles):
        addty = "(tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>"
        lines.append(f"  %s{i} = tosa.add {acc}, %m{i} : {addty}")
        acc = f"%s{i}"
    body = "\n".join(lines)
    return (
        f"func.func @main({args}) -> tensor<1x4x4xf32> {{\n{body}\n"
        f"  return {acc} : tensor<1x4x4xf32>\n}}\n"
    )


def test_kreduction_selects_mac_chain_and_matches():
    tiles = [
        tuple(
            np.random.default_rng(i).standard_normal((1, 4, 4)).astype(np.float32)
            for _ in range(2)
        )
        for i in range(3)
    ]
    inputs = [t for pair in tiles for t in pair]  # a0,b0,a1,b1,a2,b2
    prog = _mac_isa().compile_program(_kreduction_src(3))
    assert [e.name for e in prog.emits] == ["mm", "mm_acc", "mm_acc"]
    want = sum(a @ b for a, b in tiles)
    got = np.asarray(prog(*inputs)).reshape(1, 4, 4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_accumulator_coalesces_to_one_block_slot():
    """Every accumulate writes its result back onto the accumulator it read, so the
    whole K-reduction chain lives in a single slot (the hardware block buffer)."""
    prog = _mac_isa().compile_program(_kreduction_src(3))
    accs = [e for e in prog.emits if e.name == "mm_acc"]
    # mm_acc addr = [a, b, c_in, d]; c_in == d means in-place onto the accumulator,
    # and all accumulates share that one slot.
    slots = {e.addr[3] for e in accs}
    assert all(e.addr[2] == e.addr[3] for e in accs)
    assert len(slots) == 1


def test_in_place_accumulate_is_refused_not_miscompiled():
    """An instruction whose compute reads its *destination* buffer cannot be bound by
    the matcher; it must be refused (no silent wrong code)."""
    isa = ISA("inplace")
    mem = isa.global_("mem", shape=(64,), dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def macc(I):
        @I.access
        def _(a, b, c):
            return (view(mem, a, 4), view(mem, b, 4), view(mem, c, 4))

        @I.compute
        def _(a, b, c):
            return primitive.add(c, primitive.add(a, b))  # reads dst c

    src = """
    func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> tensor<4xf32> {
      %0 = tosa.add %x, %y : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %1 = tosa.add %0, %z : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      return %1 : tensor<4xf32>
    }
    """
    with pytest.raises(NoMatchError, match="no instruction matches"):
        isa.compile_program(src)


def test_oracle_supports_in_place_accumulate():
    """Hand-written assembly *can* use an in-place accumulate (dst read+written): the
    functional oracle applies a read-modify-write. This is the faithful FeatherX form
    for a hand-authored tile sequence."""
    isa = ISA("ip_oracle")
    mem = isa.scalar("mem", slots=64, dtype=f32)

    @isa.instruction(src=[mem, mem], dst=mem)
    def macc(I):
        @I.access
        def _(a, b, c):
            return (view(mem, a, 4), view(mem, b, 4), view(mem, c, 4))

        @I.compute
        def _(a, b, c):
            return primitive.add(c, primitive.add(a, b))  # c += a + b, in place

    rng = np.random.default_rng(0)
    a, b, c0 = (rng.standard_normal(4).astype(np.float32) for _ in range(3))
    init = np.zeros(64, np.float32)
    init[0:4], init[4:8], init[8:12] = a, b, c0

    @isa.oracle(init={mem: init})
    def prog():
        macc(a=0, b=4, c=8)  # accumulate two partials into c
        macc(a=0, b=4, c=8)
        isa.inspect(mem[8:12], label="c")

    res = prog()
    np.testing.assert_allclose(res["c"], c0 + 2 * (a + b), rtol=1e-5, atol=1e-6)

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Computational attributes (ACT's α): ``@I.compute``'s extra parameters.

An instruction's parameters come in three kinds, distinguished by *who supplies the
value*: an address param is assigned by allocation (Stage 3), a shape param is solved
from the source shapes (Stage 2), and an **α is bound from the source program** — the
one place a source value flows into the instruction word instead of into memory.

The IR reserved the slot from the start (``allo.define``'s trailing int/index block
args, ``allo.emit``'s ``staticComputeParams``) but nothing filled it: ``trace_instruction``
passed only buffer args, so an ISA that declared an α crashed with a ``TypeError``
before it could be looked at. These tests drive the whole path — declare, trace, emit
IR, match, bind, execute.

α is an **integer** immediate, which is the IR's contract and also the hardware's: an
immediate field in an instruction encoding holds an integer. A fixed float literal is
already expressible (``primitive.const(2.0)``, see ``test_constants.py``) and a
variable one is program data (the constant pool, same file); α is the third case, the
one the *instruction word* carries.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import AcceleratorDescriptionError, NoMatchError
from allo.exp.dsa.search import _movement_catalog, expand_emits
from allo.lang.core import f32, i32

N = 8


def _isa(dtype=i32, *, alpha=True, plain=False, vadd_cost=1.0):
    """A vector machine over ``mem``: load / store plus ``vaddi vr[d], vr[a], #k``.

    ``alpha`` drops the immediate form, ``plain`` adds a two-operand ``vadd`` that
    reads the addend from memory — the two ways to spell ``x + c``, so a test can put
    them in the same ISA and watch cost decide."""
    isa = ISA("Alpha")
    mem = isa.global_("mem", shape=(256,), dtype=dtype)
    vr = isa.vector("vr", slots=8, shape=(N,), dtype=dtype)

    @isa.instruction(src=mem, dst=vr)
    def vload(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, N), contiguous(vr, d, 1)

        @I.compute
        def _(a, d):
            return primitive.identity(a)

    @isa.instruction(src=vr, dst=mem)
    def vstore(I):
        @I.access
        def _(s, d):
            return contiguous(vr, s, 1), contiguous(mem, d, N)

        @I.compute
        def _(a, d):
            return primitive.identity(a)

    if alpha:

        @isa.instruction(src=vr, dst=vr)
        def vaddi(I):
            """``vaddi vr[d], vr[a], #k`` -- add the immediate ``k`` to every lane."""

            @I.access
            def _(s, d):
                return contiguous(vr, s, 1), contiguous(vr, d, 1)

            @I.compute
            def _(a, d, k):
                return primitive.add(a, primitive.const(k, dtype=dtype))

    if plain:

        @isa.instruction(src=[vr, vr], dst=vr, cost=vadd_cost)
        def vadd(I):
            @I.access
            def _(x, y, d):
                return (
                    contiguous(vr, x, 1),
                    contiguous(vr, y, 1),
                    contiguous(vr, d, 1),
                )

            @I.compute
            def _(a, b, d):
                return primitive.add(a, b)

    return isa


def _src(literal="7", elt="i32", width=1) -> str:
    """``x + c``. ``width`` is the constant's own extent: 1 is the broadcast scalar
    torch emits for a bias, N is a full-width splat (which a two-operand instruction
    can also take as a memory operand)."""
    v, c = f"tensor<{N}x{elt}>", f"tensor<{width}x{elt}>"
    return f"""
func.func @main(%x: {v}) -> {v} {{
  %c = "tosa.const"() {{values = dense<{literal}> : {c}}} : () -> {c}
  %r = tosa.add %x, %c : ({v}, {c}) -> {v}
  return %r : {v}
}}
"""


def _emit(prog, name):
    (rec,) = [e for e in prog.emits if e.name == name]
    return rec


# ==========================================================================#
# Declaring one: trace + IR
# ==========================================================================#


def test_a_compute_param_is_traced_and_becomes_a_block_arg():
    """The dead path: ``trace_instruction`` used to pass buffer args only, so this
    ISA raised ``TypeError: _() missing 1 required positional argument: 'k'``."""
    text = str(_isa().catalog())
    # trailing index arg after the two tensor operands, splatted into the datapath
    assert "tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: index" in text
    assert "arith.index_cast %arg2 : index to i32" in text
    assert "tensor.splat" in text


def test_a_float_datapath_widens_the_immediate_rather_than_rounding_it():
    """``index -> float`` is not one arith cast, and α is an integer, so it goes
    through i64. The point is that this is a widening: no immediate is ever rounded."""
    text = str(_isa(dtype=f32).catalog())
    assert "arith.index_cast %arg2 : index to i64" in text
    assert "arith.sitofp" in text


def test_the_assembler_signature_carries_the_compute_param():
    isa = _isa()
    assert isa._ops["vaddi"].addr_params == ["s", "d"]
    assert isa._ops["vaddi"].compute_params == ["k"]


def test_a_compute_param_absent_from_the_semantics_is_rejected():
    """Nothing but a ``const`` leaf binds an α, so a declared-but-unused one has no
    value the compiler could ever supply. Caught when the catalog is indexed."""
    isa = ISA("Dangling")
    mem = isa.global_("mem", shape=(64,), dtype=i32)

    @isa.instruction(src=mem, dst=mem)
    def vneg(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, N), contiguous(mem, d, N)

        @I.compute
        def _(a, d, k):
            return primitive.negate(a)  # k never reaches the DAG

    with pytest.raises(AcceleratorDescriptionError, match="never appear"):
        isa.compile_program(_src())


# ==========================================================================#
# Binding one: the match reads it off the source
# ==========================================================================#


@pytest.mark.parametrize("k", [7, 3, 0, 1])
def test_the_immediate_is_bound_from_the_source_not_compared(k):
    """A fixed ``primitive.const`` is *compared* (``pow(2,x)`` != ``pow(3,x)``); a
    parametric one is *bound*, so one instruction covers every immediate."""
    prog = _isa().compile_program(_src(literal=str(k)))
    assert _emit(prog, "vaddi").compute == [k]


def test_the_bound_immediate_executes():
    x = np.arange(N, dtype=np.int32)
    prog = _isa().compile_program(_src(literal="7"))
    np.testing.assert_array_equal(prog(x), x + 7)


def test_an_integer_valued_float_constant_binds_on_a_float_datapath():
    x = np.arange(N, dtype=np.float32)
    prog = _isa(dtype=f32).compile_program(_src(literal="3.0", elt="f32"))
    assert _emit(prog, "vaddi").compute == [3]
    np.testing.assert_allclose(prog(x), x + 3.0)


def test_a_non_integer_constant_does_not_bind():
    """An immediate field holds an integer. Rounding ``0.5`` to ``0`` would compile a
    different function, so the instruction simply does not match."""
    with pytest.raises(NoMatchError):
        _isa(dtype=f32).compile_program(_src(literal="0.5", elt="f32"))


def test_a_non_uniform_constant_does_not_bind():
    """One immediate is one number: a per-element constant is program data (the
    constant pool's job), not something an instruction word can carry."""
    src = """
func.func @main(%x: tensor<4xi32>) -> tensor<4xi32> {
  %c = "tosa.const"() {values = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %r = tosa.add %x, %c : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %r : tensor<4xi32>
}
"""
    with pytest.raises(NoMatchError):
        _isa().compile_program(src)


def test_a_non_constant_operand_does_not_bind():
    """The addend must be a literal the compiler can read — a second program input is
    a buffer operand, and this instruction has only one."""
    src = """
func.func @main(%x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %r = tosa.add %x, %y : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}
"""
    with pytest.raises(NoMatchError):
        _isa().compile_program(src)


def test_one_param_used_twice_must_bind_the_same_value():
    """``(x + k) * k``: the ISA says both immediates are the *same* field, so a
    source using two different constants is not this instruction."""
    isa = ISA("Twice")
    mem = isa.global_("mem", shape=(64,), dtype=i32)

    @isa.instruction(src=mem, dst=mem)
    def scale_bias(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, N), contiguous(mem, d, N)

        @I.compute
        def _(a, d, k):
            kk = primitive.const(k, dtype=i32)
            return primitive.mul(primitive.add(a, kk), kk)

    def src(k1, k2):
        return f"""
func.func @main(%x: tensor<8xi32>) -> tensor<8xi32> {{
  %a = "tosa.const"() {{values = dense<{k1}> : tensor<1xi32>}} : () -> tensor<1xi32>
  %b = "tosa.const"() {{values = dense<{k2}> : tensor<1xi32>}} : () -> tensor<1xi32>
  %s = "tosa.const"() {{values = dense<0> : tensor<1xi8>}} : () -> tensor<1xi8>
  %t = tosa.add %x, %a : (tensor<8xi32>, tensor<1xi32>) -> tensor<8xi32>
  %r = tosa.mul %t, %b, %s : (tensor<8xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}}
"""

    x = np.arange(N, dtype=np.int32)
    prog = isa.compile_program(src(5, 5))
    assert _emit(prog, "scale_bias").compute == [5]
    np.testing.assert_array_equal(prog(x), (x + 5) * 5)
    with pytest.raises(NoMatchError):
        isa.compile_program(src(5, 6))


# ==========================================================================#
# α vs. the constant pool: two ways to spell `x + c`
# ==========================================================================#


def test_an_immediate_and_a_memory_constant_compete_on_cost():
    """*One* source constant, two correct compilations: an immediate in the
    instruction word, or a pool entry loaded into a register for a two-operand add.
    Structure cannot choose between them, so cost does."""
    wide = _src(width=N)
    cheap = _isa(alpha=True, plain=True, vadd_cost=10.0).compile_program(wide)
    assert [e.name for e in cheap.emits] == ["vload", "vaddi", "vstore"]
    assert not cheap.constants  # the 7 lives in the instruction, not in memory

    dear = _isa(alpha=True, plain=True, vadd_cost=0.1).compile_program(wide)
    assert "vadd" in [e.name for e in dear.emits]
    assert len(dear.constants) == 1  # ... and here it lives in memory instead

    x = np.arange(N, dtype=np.int32)
    np.testing.assert_array_equal(cheap(x), x + 7)
    np.testing.assert_array_equal(dear(x), x + 7)


def test_without_the_immediate_form_the_constant_still_goes_to_the_pool():
    """The phase-5 path is untouched: no α instruction, so the bias is program data."""
    prog = _isa(alpha=False, plain=True).compile_program(_src(width=N))
    assert len(prog.constants) == 1
    x = np.arange(N, dtype=np.int32)
    np.testing.assert_array_equal(prog(x), x + 7)


# ==========================================================================#
# The rest of the pipeline
# ==========================================================================#


def test_hand_written_assembly_supplies_the_immediate():
    """``Instruction.__call__`` already accepted an α; now the instruction it belongs
    to can be described at all, so the oracle runs one end to end."""
    isa = _isa()
    mem, x = isa.buffers["mem"], np.arange(N, dtype=np.int32)
    init = np.zeros(256, np.int32)
    init[:N] = x

    @isa.oracle(init={mem: init})
    def prog():
        isa._ops["vload"](s=0, d=0)
        isa._ops["vaddi"](s=0, d=1, k=-4)
        isa._ops["vstore"](s=1, d=64)
        isa.inspect(mem[64 : 64 + N], label="y")

    np.testing.assert_array_equal(prog()["y"], x - 4)


def test_the_dump_shows_the_immediate():
    """An address and an immediate are different things; a dump that printed only
    ``vaddi(0, 0)`` could not tell ``+7`` from ``+3``."""
    assert "vaddi(0, 0, #7)" in str(_isa().compile_program(_src()))


def test_the_pattern_description_names_the_param():
    """The diagnostic for an unmatched op lists candidate patterns; an α renders as
    ``#name`` so the reader can see the operand must be an immediate."""
    src = """
func.func @main(%x: tensor<8xi32>, %y: tensor<8xi32>) -> tensor<8xi32> {
  %r = tosa.add %x, %y : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %r : tensor<8xi32>
}
"""
    with pytest.raises(NoMatchError, match=r"vaddi: add\(%0, #k\)"):
        _isa().compile_program(src)


def test_expand_and_alpha_cannot_be_combined():
    """``@expand`` receives address params only, so it cannot pass a bound immediate
    on to the tiles it issues — note the body below has no ``k`` to give. Refused
    where the expansion is run, rather than silently emitting some other immediate."""
    isa = ISA("Expanding")
    mem = isa.global_("mem", shape=(64,), dtype=i32)

    @isa.instruction(src=mem, dst=mem)
    def unit_addi(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, N), contiguous(mem, d, N)

        @I.compute
        def _(a, d, k):
            return primitive.add(a, primitive.const(k, dtype=i32))

    @isa.instruction(src=mem, dst=mem)
    def layer_addi(I):
        @I.access
        def _(s, d, n):
            return contiguous(mem, s, n), contiguous(mem, d, n)

        @I.compute
        def _(a, d, k):
            return primitive.add(a, primitive.const(k, dtype=i32))

        @I.expand
        def _(s, d, n):
            for i in range(n // N):
                unit_addi(s=s + i * N, d=d + i * N, k=0)  # <- no `k` in scope

    with pytest.raises(AcceleratorDescriptionError, match="cannot be combined"):
        expand_emits(isa, isa._ops["layer_addi"].spec, [0, 8, N])


def test_a_data_movement_instruction_cannot_take_one():
    """The planner inserts moves itself, so no source constant supplies an α."""
    isa = ISA("MovingAlpha")
    mem = isa.global_("mem", shape=(64,), dtype=i32)
    vr = isa.vector("vr", slots=4, shape=(N,), dtype=i32)

    @isa.instruction(src=mem, dst=vr)
    def vload(I):
        @I.access
        def _(s, d):
            return contiguous(mem, s, N), contiguous(vr, d, 1)

        @I.compute
        def _(a, d, k):
            return primitive.identity(a)

    with pytest.raises(AcceleratorDescriptionError, match="data-movement"):
        _movement_catalog(isa)

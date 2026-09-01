# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout as a *solved* access parameter (Stage 2b).

An access is an affine index map from tensor coordinates to buffer addresses. Its
``offset`` is assigned by allocation and its ``sizes`` are solved from the source
shapes — but a dense packing's **dimension ordering** appears in neither. It changes
where the data lives, not which tensor it is, so the visible shape is identical for
every ordering and Stage 2 is blind to it.

What pins it is that a value has *one* residence: every access of it has to describe
the same map. That is a unification of index maps on the SSA edge, and it is the whole
difference between *solving* an ordering and *picking* one — the deleted FEATHER path
carried ``order`` as an enum mode chosen by a config-switch proxy cost, which is a
vote, not a constraint.

This stage **solves; it does not check.** Accesses are grouped per ``(value, buffer)``
and a parametric one adopts the first concrete map in its group (program I/O and the
constant pool seed theirs with the host ABI). Two accesses may still disagree
afterwards — whether that is compilable depends on the machine having a mover that
repacks between them, which only the planner knows. See ``test_relayout_repair.py``.

A **mover** is the one instruction this does not reach, because the planner is what
inserts it and it therefore unifies with nothing. Its ordering params are *chosen*
rather than solved, by the router: one assignment is one edge of the movement graph,
so picking a path picks the ordering. That is what lets a single instruction stand for
a whole family of relayouts — MINISA's ``Set*VNLayout`` — instead of forcing the ISA
author to write one per permutation.
"""

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, layout, strided, view
from allo.exp.dsa.core import ISA, access_map, param_roles, residence
from allo.exp.dsa.errors import (
    AcceleratorDescriptionError,
    AssemblyError,
    LayoutError,
)
from allo.exp.dsa.search import _movement_catalog, expand_emits
from allo.lang.core import f32

A, B, C = 2, 3, 4
N = A * B * C
CL = (2, 0, 1)  # dim 2 outermost -- "channel last"
ROW = (0, 1, 2)
_T = f"tensor<{A}x{B}x{C}xf32>"

# relu, then double it: two matchable ops with one value between them, and the
# result is a permutation of the truth when the ordering is wrong.
_SRC = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({_T}) -> {_T}
  %y = tosa.add %t, %t : ({_T}, {_T}) -> {_T}
  return %y : {_T}
}}
"""

_X = (np.arange(N, dtype=np.float32) + 1).reshape(A, B, C)


def _vnpu(produced=CL, consumed=None, written=None):
    """A one-buffer machine whose two instructions meet on a value in that buffer.

    ``vnout`` computes relu and packs its result in ``produced``; ``vnin`` doubles it,
    reading its operands in ``consumed`` (an ordering param by default) and writing in
    ``written``. Everything lives in the I/O buffer, so the host ABI pins both ends and
    no data movement sits in between."""
    isa = ISA("vnpu")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def vnout(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C)),
                layout(mem, d, (A, B, C), order=produced),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    def patterns(a, b, d, src):
        return (
            layout(mem, a, (A, B, C), order=src),
            layout(mem, b, (A, B, C), order=src),
            layout(mem, d, (A, B, C), order=written),
        )

    @isa.instruction(src=[mem, mem], dst=mem)
    def vnin(I):
        if consumed is None:
            I.access(lambda a, b, d, q: patterns(a, b, d, q))
        else:
            I.access(lambda a, b, d: patterns(a, b, d, consumed))

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    return isa


# --- the model: an ordering is storage, not shape ---------------------------------


def test_an_ordering_permutes_storage_not_the_operand():
    """The compute region sees ``sizes`` in logical order whatever the ordering — so
    the same semantics serve every packing, and the ordering is free to be unknown
    while shapes are being solved. In the IR the packing is a dense run expanded into
    *storage* order and transposed back."""
    text = str(_vnpu().catalog())
    assert text.count(f"compute(%arg0: {_T}, %arg1: {_T})") == 1  # vnout, both ways
    assert f"output_shape [{C}, {A}, {B}]" in text  # stored channel-last ...
    assert "allo.patterns.transpose %1 permutation = [1, 2, 0]" in text  # ... read back


def test_row_major_needs_no_transpose_and_equals_a_plain_view():
    """The default ordering is the host's own packing, and it must cost nothing: with
    it a layout *is* a contiguous view, down to the emitted access ops."""

    def build(access_fn):
        probe = ISA("probe")
        buf = probe.global_("mem", shape=(256,), dtype=f32)

        @probe.instruction(src=[buf], dst=buf)
        def op(I):
            I.access(access_fn(buf))

            @I.compute
            def _(s, d):
                return primitive.relu(s)

        return str(probe.catalog())

    tile = (A, B, C)
    laid = build(lambda b: lambda s, d: (layout(b, s, tile), view(b, d, tile)))
    viewed = build(lambda b: lambda s, d: (view(b, s, tile), view(b, d, tile)))
    assert laid == viewed
    assert "transpose" not in laid


def test_a_fixed_ordering_really_permutes_the_bytes():
    """The ordering has to survive lowering and JIT, not just verification: pack a
    tensor channel-last, read it back with the same ordering, and the round trip is
    the identity while the stored bytes are the transpose."""
    isa = ISA("dma")
    mem = isa.global_("mem", shape=(256,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def pack(I):
        @I.access
        def _(s, d):
            return (layout(mem, s, (A, B, C)), layout(mem, d, (A, B, C), order=CL))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[mem], dst=mem)
    def unpack(I):
        @I.access
        def _(s, d):
            return (layout(mem, s, (A, B, C), order=CL), layout(mem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.oracle(init={mem: _X})
    def run():
        pack(0, 64)
        unpack(64, 128)
        isa.inspect(mem[slice(64, 64 + N)], label="packed")
        isa.inspect(mem[slice(128, 128 + N)], label="back")

    out = run()
    np.testing.assert_allclose(out["packed"], np.transpose(_X, CL).reshape(-1))
    np.testing.assert_allclose(out["back"], _X.reshape(-1))


# --- solving ----------------------------------------------------------------------


def test_an_ordering_param_is_solved_from_the_producers_packing():
    """The acceptance: ``vnin`` encodes its input ordering and nothing in the *shapes*
    says what it is. The producer packs channel-last, so that is what the consumer has
    to decode, and the numbers come out exactly right."""
    prog = _vnpu(produced=CL).compile_program(_SRC)
    assert prog.emits[-1].addr[3] == CL
    np.testing.assert_allclose(prog(_X), 2 * _X)


def test_the_solution_follows_the_producer_rather_than_a_default():
    """Same ISA, same program, a different producer — and the solved ordering moves
    with it. Without this the previous test would pass on a hard-coded row-major."""
    row = _vnpu(produced=ROW).compile_program(_SRC)
    chan = _vnpu(produced=CL).compile_program(_SRC)
    assert row.emits[-1].addr[3] == ROW
    assert chan.emits[-1].addr[3] == CL
    np.testing.assert_allclose(row(_X), 2 * _X)
    np.testing.assert_allclose(chan(_X), 2 * _X)


def test_a_free_ordering_takes_the_host_packing():
    """The intermediate here is written *and* read through ordering params, so nothing
    pins it: the producer's input and the consumer's output are the ends the host ABI
    fixes, and neither reaches across. A cost model with no memory model prices every
    ordering the same, so the group takes the host's packing rather than a tie broken
    at random — and both params fall out of that one choice."""
    isa = ISA("free")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def writer(I):
        @I.access
        def _(s, d, q):
            return (layout(mem, s, (A, B, C)), layout(mem, d, (A, B, C), order=q))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem], dst=mem)
    def reader(I):
        @I.access
        def _(s, d, p):
            return (layout(mem, s, (A, B, C), order=p), layout(mem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    src = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({_T}) -> {_T}
  %y = tosa.abs %t : ({_T}) -> {_T}
  return %y : {_T}
}}
"""
    prog = isa.compile_program(src)
    assert [e.addr[2] for e in prog.emits] == [ROW, ROW]
    np.testing.assert_allclose(prog(_X), _X)


def test_program_io_is_pinned_to_the_host_abi():
    """The one map the compiler does not get to choose. ``__call__`` reads an output
    back densely, so an instruction that writes the program's result channel-last is
    handing the host scrambled data — refused, not silently accepted."""
    with pytest.raises(LayoutError, match="strides"):
        _vnpu(produced=CL, consumed=CL, written=CL).compile_program(_SRC)


def test_the_constant_pool_is_host_data_too():
    """A constant operand is written into the I/O buffer before the run, densely, just
    like an input — ACT Def 3.8 puts both in one ASM. So an instruction that reads it
    repacked is reading data nobody repacked."""

    def build(order):
        isa = ISA("consts")
        mem = isa.global_("mem", shape=(1024,), dtype=f32)

        @isa.instruction(src=[mem, mem], dst=mem)
        def biased(I):
            @I.access
            def _(a, b, d):
                return (
                    layout(mem, a, (A, B, C)),
                    layout(mem, b, (A, B, C), order=order),
                    layout(mem, d, (A, B, C)),
                )

            @I.compute
            def _(a, b, d):
                return primitive.add(a, b)

        return isa

    def literal(a):
        return f"{float(a):.6e}" if a.ndim == 0 else f"[{', '.join(map(literal, a))}]"

    bias = np.arange(N, dtype=np.float32).reshape(A, B, C) / 10
    src = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %c = "tosa.const"() {{values = dense<{literal(bias)}> : {_T}}} : () -> {_T}
  %y = tosa.add %x, %c : ({_T}, {_T}) -> {_T}
  return %y : {_T}
}}
"""
    np.testing.assert_allclose(build(ROW).compile_program(src)(_X), _X + bias)
    with pytest.raises(LayoutError, match="strides"):
        build(CL).compile_program(src)


def test_two_accesses_that_disagree_are_refused():
    """Producer packs channel-last, consumer insists on row-major, and this machine has
    no mover that repacks between them. Whether a disagreement is repairable is the
    move graph's business, so the *planner* is what refuses (see
    ``test_relayout_repair.py``), and it names the consumer and both residences."""
    with pytest.raises(LayoutError) as exc:
        _vnpu(produced=CL, consumed=ROW).compile_program(_SRC)
    text = str(exc.value)
    assert "vnin operand 0" in text
    assert "as sizes [2, 3, 4] strides [12, 4, 1]" in text  # wants it row-major ...
    assert "as sizes [2, 3, 4] strides [3, 1, 6]" in text  # ... but it is channel-last


def test_one_param_cannot_hold_two_residences():
    """An ordering param names one packing, so an instruction that reads and writes
    through the same param is stating that both operands live alike. Here they cannot:
    the operand is channel-last and the result is the program's output, which the host
    reads densely. Solving one end fixes the param, and the other end is then a plain
    residence mismatch — reported, not resolved by whichever operand came last."""
    isa = ISA("clash")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def packer(I):
        @I.access
        def _(s, d):
            return (layout(mem, s, (A, B, C)), layout(mem, d, (A, B, C), order=CL))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem, mem], dst=mem)
    def both(I):
        """Reads one operand and writes its result with the *same* ordering param —
        but the operand is channel-last while the result is program output."""

        @I.access
        def _(a, b, d, q):
            return (
                layout(mem, a, (A, B, C), order=q),
                layout(mem, b, (A, B, C)),
                layout(mem, d, (A, B, C), order=q),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    src = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({_T}) -> {_T}
  %y = tosa.add %t, %x : ({_T}, {_T}) -> {_T}
  return %y : {_T}
}}
"""
    with pytest.raises(LayoutError) as exc:
        isa.compile_program(src)
    text = str(exc.value)
    assert "result #0" in text  # the end that cannot be satisfied is the host's
    assert "as sizes [2, 3, 4] strides [12, 4, 1]" in text
    assert "as sizes [2, 3, 4] strides [3, 1, 6]" in text


def test_a_residence_that_is_not_a_dense_packing_has_no_ordering():
    """A layout is a *permutation* of a dense packing. A producer that leaves gaps
    describes a residence no ordering reproduces, and saying so beats rounding to the
    nearest permutation."""
    isa = ISA("sparse")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def spread(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C)),
                strided(mem, [d], [N], [2]).reshape((A, B, C)),  # every other word
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem, mem], dst=mem)
    def reader(I):
        @I.access
        def _(a, b, d, q):
            return (
                layout(mem, a, (A, B, C), order=q),
                layout(mem, b, (A, B, C), order=q),
                layout(mem, d, (A, B, C)),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    with pytest.raises(LayoutError, match="dense packing"):
        isa.compile_program(_SRC)


# --- the MINISA shape -------------------------------------------------------------


def test_a_parametric_layout_solves_its_sizes_and_its_ordering():
    """``Set*VNLayout``'s shape: the virtual-neuron dims are carried integer payload
    and the ordering is a field of the same instruction. The two are solved by
    different mechanisms on the same access — the dims by shape unification (Stage 2),
    the ordering by map unification (Stage 2b) — which is exactly the split between
    what a tensor *is* and where it *lives*."""
    isa = ISA("minisa")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def set_ovn(I):
        @I.access
        def _(s, d, kl1, nl0, nl1):
            dims = (kl1, nl0, nl1)
            return (layout(mem, s, dims), layout(mem, d, dims, order=CL))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem, mem], dst=mem)
    def set_ivn(I):
        @I.access
        def _(a, b, d, kl1, nl0, nl1, order):
            dims = (kl1, nl0, nl1)
            return (
                layout(mem, a, dims, order=order),
                layout(mem, b, dims, order=order),
                layout(mem, d, dims),
            )

        @I.compute
        def _(a, b, d):
            return primitive.add(a, b)

    roles, _ = param_roles(set_ivn.spec)
    assert [roles[i] for i in range(7)] == ["offset"] * 3 + ["shape"] * 3 + ["layout"]

    prog = isa.compile_program(_SRC)
    ivn = prog.emits[-1]
    assert ivn.addr[3:6] == [A, B, C]  # {K_L1, N_L0, N_L1}, solved from the shapes
    assert ivn.addr[6] == CL  # the ordering, solved from the residence
    np.testing.assert_allclose(prog(_X), 2 * _X)


# --- strides ----------------------------------------------------------------------


def test_a_stride_param_is_solved_against_an_arrays_pitch():
    """A stride is residence too. Here the value is a block of a row-major array whose
    pitch the instruction encodes, and the solver reads it off the array rather than
    the shapes — which cannot see it."""
    R, W, T = 8, 8, 4
    isa = ISA("pitch")
    dram = isa.hbm("dram", shape=(R, W), dtype=f32, is_global=True)

    @isa.instruction(src=[dram], dst=dram)
    def block_relu(I):
        @I.access
        def _(sr, sc, dr, dc, s):
            return (
                strided(dram, [sr, sc], [T, T], [s, 1]),
                strided(dram, [dr, dc], [T, T], [s, 1]),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    src = f"""
func.func @main(%x: tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32> {{
  %y = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : (tensor<{T}x{T}xf32>) -> tensor<{T}x{T}xf32>
  return %y : tensor<{T}x{T}xf32>
}}
"""
    prog = isa.compile_program(src)
    assert prog.emits[0].addr[4] == 1  # one row step per row of the block
    x = np.arange(T * T, dtype=np.float32).reshape(T, T) - 5
    np.testing.assert_allclose(prog(x), np.maximum(x, 0))


def test_a_residence_param_that_nothing_can_observe_is_reported():
    """A stride on an axis whose count is 1 selects a single slot, so it never appears
    in any map and no residence can pin it. That is an ISA the compiler cannot
    complete, and it says so instead of emitting whatever happens to be in the slot."""
    W = 8
    isa = ISA("unobservable")
    lanes = isa.buffer("lanes", (64,), f32, slot=(W,), is_global=True)

    @isa.instruction(src=[lanes], dst=lanes)
    def row_relu(I):
        @I.access
        def _(s_addr, d_addr, s):
            return (
                strided(lanes, [s_addr], [1], [s]),
                strided(lanes, [d_addr], [1], [s]),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    src = f"""
func.func @main(%x: tensor<{W}xf32>) -> tensor<{W}xf32> {{
  %y = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : (tensor<{W}xf32>) -> tensor<{W}xf32>
  return %y : tensor<{W}xf32>
}}
"""
    with pytest.raises(LayoutError, match="under-constrained"):
        isa.compile_program(src)


# --- interaction with the rest of the pipeline ------------------------------------


def test_an_op_that_relayouts_does_not_inherit_its_operands_slot():
    """Position-preserving semantics let a result be written over a dying operand —
    but only when the two sit at the same address. An elementwise op that also
    repacks breaks that, so the result needs its own space.

    Both halves matter: with matching residence the slot *is* handed over, so the
    check below is a filter on coalescing rather than a ban on it."""
    dense = _vnpu(produced=ROW, consumed=ROW).compile_program(_SRC)
    out, add = dense.emits
    assert add.addr[0] == out.addr[1] == add.addr[2]  # result takes the operand's slot
    np.testing.assert_allclose(dense(_X), 2 * _X)

    # Same program, but the consumer reads channel-last and writes row-major: the same
    # tensor positions at different addresses, so no coalescing.
    prog = _vnpu(produced=CL, consumed=CL).compile_program(_SRC)
    out, add = prog.emits
    assert add.addr[0] == out.addr[1]  # reads what vnout wrote ...
    assert add.addr[2] != add.addr[0]  # ... and writes somewhere else
    np.testing.assert_allclose(prog(_X), 2 * _X)


def _minisa(produced):
    """A machine whose only path off `mem` is a gather that carries its source
    packing as a parameter -- MINISA's ``Set*VNLayout`` shape. `produce` leaves its
    result in `mem` packed in ``produced``; `consume` runs on a dense `vmem` tile."""
    isa = ISA("minisa")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=mem)
    def produce(I):
        @I.access
        def _(s, d):
            return (
                layout(mem, s, (A, B, C)),
                layout(mem, d, (A, B, C), order=produced),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem], dst=vmem)
    def load_vn(I):
        @I.access
        def _(s, d, q):
            return (layout(mem, s, (A, B, C), order=q), view(vmem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=mem)
    def store_vn(I):
        @I.access
        def _(s, d, q):
            return (view(vmem, s, (A, B, C)), layout(mem, d, (A, B, C), order=q))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    @isa.instruction(src=[vmem], dst=vmem)
    def consume(I):
        @I.access
        def _(s, d):
            return (view(vmem, s, (A, B, C)), view(vmem, d, (A, B, C)))

        @I.compute
        def _(s, d):
            return primitive.abs(s)

    return isa


_ABS_SRC = f"""
func.func @main(%x: {_T}) -> {_T} {{
  %t = tosa.clamp %x {{min_val = 0.0 : f32, max_val = 3.40282347E+38 : f32}} : ({_T}) -> {_T}
  %y = tosa.abs %t : ({_T}) -> {_T}
  return %y : {_T}
}}
"""


@pytest.mark.parametrize("produced", [ROW, CL, (1, 2, 0)])
def test_a_movers_ordering_is_chosen_by_the_router(produced):
    """A mover's residence params are not *solved* — nothing unifies with them, since
    the planner is what inserts the move. They are **chosen**, and the chooser is the
    router: one ordering assignment is one edge of the movement graph, so picking a
    path picks the ordering. One ``load_vn`` therefore covers every source packing,
    where the ISA would otherwise need one instruction per permutation."""
    prog = _minisa(produced).compile_program(_ABS_SRC)
    (load,) = [e for e in prog.emits if e.name == "load_vn"]
    (store,) = [e for e in prog.emits if e.name == "store_vn"]
    assert load.addr[2] == produced  # matches where the producer left it
    assert store.addr[2] == ROW  # the result must arrive in the host ABI's packing
    np.testing.assert_allclose(prog(_X), np.abs(_X))


def test_a_data_movement_instruction_needs_a_domain_for_a_stride_param():
    """Choosing needs a domain to choose from. An ordering's is intrinsic (rank!), so
    the movement graph enumerates it unasked; a stride's is the integers, so an
    undeclared one is refused — see ``test_schedule_param`` for declaring it."""
    isa = ISA("mover")
    mem = isa.global_("mem", shape=(256,), dtype=f32)
    vmem = isa.scalar("vmem", 256, f32)

    @isa.instruction(src=[mem], dst=vmem)
    def dma(I):
        @I.access
        def _(s, d, n, k):
            return (strided(mem, s, n, k), contiguous(vmem, d, n))

        @I.compute
        def _(s, d):
            return primitive.identity(s)

    with pytest.raises(AcceleratorDescriptionError, match="stride param"):
        _movement_catalog(isa)


def test_the_catalog_enumerates_the_orderings_a_program_solves_one():
    """An ordering is *structure* — a reassociation and a permutation are attributes,
    not operands — so one define is emitted per ordering, with that param specialized
    away. Described on its own, an instruction shows every configuration it has;
    compiled, only the one that was solved."""
    text = str(_vnpu(produced=CL).catalog())
    assert text.count("allo.define @vnin$") == 6  # every ordering of 3 dims
    assert "allo.define @vnin$201" in text
    assert "addr(%arg0: index, %arg1: index, %arg2: index)" in text  # q is gone

    isa = _vnpu(produced=CL)
    prog = isa.compile_program(_SRC)
    prog(_X)  # builds the module for this program only -- one variant, and it runs


def test_a_catalog_stops_enumerating_before_it_becomes_a_search_space():
    """Enumerating orderings is a description, not a search. Two orderings of three
    dims is already 36 configurations, and printing them is not what a reader wants —
    so the catalog says so and points at compiling a program instead."""
    isa = ISA("many")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def both_ends(I):
        @I.access
        def _(s, d, p, q):
            return (
                layout(mem, s, (A, B, C), order=p),
                layout(mem, d, (A, B, C), order=q),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    with pytest.raises(AcceleratorDescriptionError, match="36 combinations"):
        isa.catalog()


def test_an_ordering_param_cannot_ride_along_with_expansion():
    """A layer-level op lowers to tiles addressing sub-blocks, and a sub-block of a
    reordered layer is not at a fixed stride from its base. Translating one map to the
    other is the tiling this frontend leaves out, so the combination is refused rather
    than left to an expansion body nothing here can check."""
    isa = ISA("layered")
    mem = isa.global_("mem", shape=(1024,), dtype=f32)

    @isa.instruction(src=[mem], dst=mem)
    def tile_relu(I):
        @I.access
        def _(s, d):
            return (view(mem, s, (B, C)), view(mem, d, (B, C)))

        @I.compute
        def _(s, d):
            return primitive.relu(s)

    @isa.instruction(src=[mem], dst=mem)
    def layer_relu(I):
        @I.access
        def _(s, d, q):
            return (
                layout(mem, s, (A, B, C), order=q),
                layout(mem, d, (A, B, C), order=q),
            )

        @I.compute
        def _(s, d):
            return primitive.relu(s)

        @I.expand
        def _(s, d, q):
            for i in range(A):
                tile_relu(s + i * B * C, d + i * B * C)

    with pytest.raises(AcceleratorDescriptionError, match="ordering param"):
        expand_emits(isa, layer_relu.spec, [0, 64, ROW])


# --- declaration-time rules -------------------------------------------------------


def test_a_layout_needs_a_flat_word_addressable_buffer():
    """A layout linearizes elements. A multi-element slot or a multi-extent address
    space has already fixed part of the packing, so the ordering would be describing
    something other than where the data is."""
    isa = ISA("shapes")
    tiled = isa.hbm("tiled", shape=(8, 8), dtype=f32)
    lanes = isa.vector("lanes", 8, (4,), f32)
    with pytest.raises(AcceleratorDescriptionError, match="extents"):
        layout(tiled, 0, (A, B, C))
    with pytest.raises(AcceleratorDescriptionError, match="slots"):
        layout(lanes, 0, (A, B, C))


def test_an_ordering_argument_is_checked_as_a_permutation():
    """Hand-written assembly supplies the ordering itself, and an ordering that is not
    a permutation of the dims it orders is caught at the call rather than emitted."""
    isa = _vnpu(produced=CL)
    vnin = isa._ops["vnin"]

    @isa.oracle
    def bad():
        vnin(0, 0, 32, (0, 1))

    with pytest.raises(AssemblyError, match="permutation"):
        bad()


def test_the_map_is_the_thing_that_is_compared():
    """The map is what the whole stage rests on, so check it directly: a dense
    ordering gives suffix products taken in that order, and a rank alias (a leading
    unit dim) is transparent, exactly as it is when shapes are unified."""
    isa = ISA("maps")
    mem = isa.global_("mem", shape=(256,), dtype=f32)
    assert access_map(layout(mem, 0, (A, B, C)), {}) == [(A, 12), (B, 4), (C, 1)]
    # channel-last stores [C, A, B], so dim 1 is innermost and dim 2 is outermost
    assert access_map(layout(mem, 0, (A, B, C), order=CL), {}) == [
        (A, B),
        (B, 1),
        (C, A * B),
    ]
    batched = residence(access_map(view(mem, 0, (1, B, C)), {}))
    flat = residence(access_map(layout(mem, 0, (B, C)), {}))
    assert batched == flat == ((B, 4), (C, 1))

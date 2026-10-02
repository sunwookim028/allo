# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U1: ``vpu_alu``, one lane of the VPU's elementwise ALU.

An input register, then ``vpu_bf16_add_pipe`` and ``vpu_bf16_mul_pipe`` side
by side and a two-deep delay for mov/max/min: latency
``vpu_pkg::VPU_ALU_LATENCY = 3`` for every op. (``docs/isa_latency.json``'s
``w: 5`` for vadd..vmov is the ISA-level writeback offset, which adds the
operand read and ``VPU_WB_STAGES``; it is not this unit's latency.)

``op_i`` is ``vpu_pkg::vpu_alu_op_e`` (``logic [3:0]``); ``harness/ref.py``
``ALU_OP`` holds its values. The stimulus drives all 16 codes, including
the declared AND/OR/XOR and the seven unused ones.

Allo variants (``VARIANTS``), each a composition of engine units rather than
re-inlined arithmetic -- what each mechanism allowed is the U1 finding
(``dev/records/minitpu/u1_alu_2026-10-02.rst``):

``native``          ``bfloat16`` ``+ - * max min`` in one kernel
``bits``            the RTL's structure: both bit-level engines and the
                    selector on every pair, then the result mux
``bits_dispatch``   the same engines, each called only under its own op
``engines_native``  ``bits``'s skeleton with ``bfloat16`` engines swapped in
``add_bits_mul_native``  one engine of each kind
``netlist``         the engines as ``@df.unit`` instances on streams, the way
                    ``vpu_alu.sv`` instantiates its adder and multiplier
``netlist_native``  ``netlist`` with the ``bfloat16`` engine units
"""

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import Stream, UInt, bfloat16, uint1, uint16

from examples.minitpu.harness import ref, rtl
from examples.minitpu.harness import stimulus as stim
from examples.minitpu.units.bf16_add import add_bits
from examples.minitpu.units.bf16_mul_pipe import mul_bits

RTL = rtl.RtlUnit(
    top="vpu_alu",
    sources=[
        "src/core/vpu/vpu_pkg.sv",
        "src/core/vpu/vpu_bf16_add_pipe.sv",
        "src/core/vpu/vpu_bf16_mul.sv",
        "src/core/vpu/vpu_alu.sv",
    ],
    inputs=[("op_i", 4), ("op_a_i", 16), ("op_b_i", 16)],
    outputs=[("result_o", 16)],
    shape="valid",
    latency=3,
)
LATENCY_SOURCE = "vpu_pkg.sv localparam VPU_ALU_LATENCY = 3"


REF = ref.vpu_alu
IEEE = ref.ieee_vpu_alu
PROBE = ((ref.ALU_OP["ADD"], 0x3F80, 0x3F80), (ref.ALU_OP["ADD"], 0x4000, 0x3F80))


def stimulus():
    return stim.alu_ops(range(16), n_random=200_000)


# ---------------------------------------------------------------------------
# The Allo ALU, composed out of Allo units (README D-9: the composition is the
# probe). An engine is a module-level function on bit patterns,
# ``(uint16, uint16) -> uint16``, so that a bit-level engine and a
# ``bfloat16`` one share one interface and either can be dropped in:
#
#   add  ``bf16_add.add_bits``    reused, not copied: the U1 pilot's adder,
#                                 hoisted out of its kernel into a function
#        ``add_native``           ``bfloat16`` ``+`` behind two bitcasts
#   mul  ``bf16_mul_pipe.mul_bits`` reused: line-for-line
#                                 ``vpu_bf16_mul_pipe``, one pair, also checked
#                                 on its own as that unit's ``fn`` variant
#        ``mul_native``           ``bfloat16`` ``*`` behind two bitcasts
#
# Allo binds a call to a callee by the *name at the call site* but emits the
# callee under its *def name* (``allo/ir/builder.py`` ``build_Call``), so an
# engine passed in as a value (``alu(add=add_native)``) fails to link. The
# swap is therefore made by the one convention that works: every engine for
# one slot is wrapped in a function whose def name is the slot's name
# (``engine_add`` / ``engine_mul``), chosen per build by ``_engines``. Two
# engines for one slot can never sit in one region: same def name, one
# ``func.func`` (``redefinition of symbol``).
#
# A reused function also resolves its free names in one flattened namespace,
# caller's module first: if this module ever defined a name that
# ``bf16_add.add_bits`` reads (``leading_zeros17``, say), the adder would
# silently use this module's (record, C3).
# ---------------------------------------------------------------------------

OP_ADD, OP_SUB, OP_MUL, OP_MOV, OP_MAX, OP_MIN = (
    ref.ALU_OP[k] for k in ("ADD", "SUB", "MUL", "MOV", "MAX", "MIN"))


def add_native(a: uint16, b: uint16) -> uint16:
    s: bfloat16 = a.bitcast(bfloat16) + b.bitcast(bfloat16)
    return s.bitcast(uint16)


def mul_native(a: uint16, b: uint16) -> uint16:
    p: bfloat16 = a.bitcast(bfloat16) * b.bitcast(bfloat16)
    return p.bitcast(uint16)


def bf16_gt(a: uint16, b: uint16) -> uint1:
    """``vpu_pkg::bf16_gt``: unsigned compare of sign-flipped keys. The keys
    carry a spare zero bit on top (B1)."""
    ka: UInt(17) = 0
    kb: UInt(17) = 0
    if a[15]:
        ka[0:16] = a ^ 0xFFFF
    else:
        ka[0:16] = a ^ 0x8000
    if b[15]:
        kb[0:16] = b ^ 0xFFFF
    else:
        kb[0:16] = b ^ 0x8000
    gt: uint1 = ka > kb
    return gt


def _engines(add, mul):
    """One engine per slot, under the slot's def name (see the header)."""
    if add == "bits":
        def engine_add(a: uint16, b: uint16) -> uint16:
            return add_bits(a, b)
    else:
        def engine_add(a: uint16, b: uint16) -> uint16:
            return add_native(a, b)
    if mul == "bits":
        def engine_mul(a: uint16, b: uint16) -> uint16:
            return mul_bits(a, b)
    else:
        def engine_mul(a: uint16, b: uint16) -> uint16:
            return mul_native(a, b)
    return engine_add, engine_mul


def composed(n, add="bits", mul="bits"):
    """The RTL's structure: both engines and the selector see every operand
    pair, then the op picks one result (``vpu_alu.sv``'s result mux)."""
    engine_add, engine_mul = _engines(add, mul)

    @df.region()
    def top(OP: uint16[n], A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[OP, A, B, C])
        def alu(opv: uint16[n], av: uint16[n], bv: uint16[n], cv: uint16[n]):
            for i in range(n):
                op: UInt(4) = opv[i][0:4]
                a: uint16 = av[i]
                b: uint16 = bv[i]
                # {(op == SUB) ^ b[15], b[14:0]}. The compare goes through a
                # uint1 first: a compare is typed as its operands (B2), and
                # ``b[15] ^ (op == OP_SUB)`` is then an i1 ^ i4 xori.
                is_sub: uint1 = op == OP_SUB
                add_rhs: uint16 = b
                add_rhs[15] = b[15] ^ is_sub
                sum_r: uint16 = engine_add(a, add_rhs)
                mul_r: uint16 = engine_mul(a, b)
                gt: uint1 = bf16_gt(a, b)
                sel: uint16 = 0
                if op == OP_MAX:
                    sel = a if gt else b
                else:
                    sel = b if gt else a
                result: uint16 = a  # MOV, AND/OR/XOR, 9..15: the default arm
                if op == OP_ADD or op == OP_SUB:
                    result = sum_r
                elif op == OP_MUL:
                    result = mul_r
                elif op == OP_MAX or op == OP_MIN:
                    result = sel
                cv[i] = result

    return top


def dispatched(n, add="bits", mul="bits"):
    """The same engines, each called only under its own op: the shape a
    software ALU takes. Checks whether the backends care which is written."""
    engine_add, engine_mul = _engines(add, mul)

    @df.region()
    def top(OP: uint16[n], A: uint16[n], B: uint16[n], C: uint16[n]):
        @df.kernel(mapping=[1], args=[OP, A, B, C])
        def alu(opv: uint16[n], av: uint16[n], bv: uint16[n], cv: uint16[n]):
            for i in range(n):
                op: UInt(4) = opv[i][0:4]
                a: uint16 = av[i]
                b: uint16 = bv[i]
                result: uint16 = a
                if op == OP_ADD:
                    result = engine_add(a, b)
                elif op == OP_SUB:
                    # Through a typed local: a call argument is not coerced
                    # to the parameter type, and ``b ^ 0x8000`` is an i32.
                    neg_b: uint16 = b ^ 0x8000
                    result = engine_add(a, neg_b)
                elif op == OP_MUL:
                    result = engine_mul(a, b)
                elif op == OP_MAX:
                    result = a if bf16_gt(a, b) else b
                elif op == OP_MIN:
                    result = b if bf16_gt(a, b) else a
                cv[i] = result

    return top


def native(n):
    """Allo's own ``bfloat16``: ``+``, ``-``, ``*``, ``max``, ``min``."""

    @df.region()
    def top(OP: uint16[n], A: bfloat16[n], B: bfloat16[n], C: bfloat16[n]):
        @df.kernel(mapping=[1], args=[OP, A, B, C])
        def alu(opv: uint16[n], av: bfloat16[n], bv: bfloat16[n], cv: bfloat16[n]):
            for i in range(n):
                op: uint16 = opv[i]
                a: bfloat16 = av[i]
                b: bfloat16 = bv[i]
                result: bfloat16 = a
                if op == OP_ADD:
                    result = a + b
                elif op == OP_SUB:
                    result = a - b
                elif op == OP_MUL:
                    result = a * b
                elif op == OP_MAX:
                    result = max(a, b)
                elif op == OP_MIN:
                    result = min(a, b)
                cv[i] = result

    return top


# ---------------------------------------------------------------------------
# The same ALU as a netlist, the way ``vpu_alu.sv`` instantiates
# ``vpu_bf16_add_pipe`` and ``vpu_bf16_mul_pipe``: each engine is a
# ``@df.unit`` with stream ports, an issue kernel fans every operand pair out
# to both engines, and a writeback kernel muxes by the op that rode alongside.
# Operand pairs travel as one word, ``{b, a}``.
#
# A unit's names -- its port shapes *and* its body's trip counts -- are a
# snapshot taken when ``@df.unit`` runs (``allo/netlist.py`` ``UnitSpec``
# keeps ``get_global_vars(func)``), and a unit cannot be parametrized at the
# instantiation (stream_ports.rst, "What it does not do"). A module-level
# engine unit would be fixed at one vector count, and setting a module global
# later changes nothing (a unit that still loops the old count deadlocks its
# region). So each engine unit is decorated inside a factory, per ``n``, and
# its count is a closure variable. The engine inside it is still called by its
# def name (see ``_engines``), so there is one factory per engine.
# ---------------------------------------------------------------------------


def _unit_add_bits(n):
    @df.unit()
    def unit_add_bits(src: Stream[UInt(32), 4], dst: Stream[uint16, 4]):
        for i in range(n):
            w: UInt(32) = src.get()
            x: uint16 = w[0:16]
            y: uint16 = w[16:32]
            dst.put(add_bits(x, y))

    return unit_add_bits


def _unit_add_native(n):
    @df.unit()
    def unit_add_native(src: Stream[UInt(32), 4], dst: Stream[uint16, 4]):
        for i in range(n):
            w: UInt(32) = src.get()
            x: uint16 = w[0:16]
            y: uint16 = w[16:32]
            dst.put(add_native(x, y))

    return unit_add_native


def _unit_mul_bits(n):
    @df.unit()
    def unit_mul_bits(src: Stream[UInt(32), 4], dst: Stream[uint16, 4]):
        for i in range(n):
            w: UInt(32) = src.get()
            x: uint16 = w[0:16]
            y: uint16 = w[16:32]
            dst.put(mul_bits(x, y))

    return unit_mul_bits


def _unit_mul_native(n):
    @df.unit()
    def unit_mul_native(src: Stream[UInt(32), 4], dst: Stream[uint16, 4]):
        for i in range(n):
            w: UInt(32) = src.get()
            x: uint16 = w[0:16]
            y: uint16 = w[16:32]
            dst.put(mul_native(x, y))

    return unit_mul_native


def netlist(n, add_unit=_unit_add_bits, mul_unit=_unit_mul_bits):
    """Engines are swapped by passing a different unit factory: a unit
    instantiation, unlike a function call, resolves the value bound to the
    name it is called by."""
    adder = add_unit(n)
    multiplier = mul_unit(n)

    @df.region()
    def top(OP: uint16[n], A: uint16[n], B: uint16[n], C: uint16[n]):
        to_add: Stream[UInt(32), 4]
        to_mul: Stream[UInt(32), 4]
        to_wb: Stream[UInt(40), 4]
        from_add: Stream[uint16, 4]
        from_mul: Stream[uint16, 4]

        @df.kernel(mapping=[1], args=[OP, A, B])
        def issue(opv: uint16[n], av: uint16[n], bv: uint16[n]):
            for i in range(n):
                op: UInt(4) = opv[i][0:4]
                a: uint16 = av[i]
                b: uint16 = bv[i]
                is_sub: uint1 = op == OP_SUB
                add_word: UInt(32) = 0
                add_word[0:16] = a
                add_word[16:32] = b
                add_word[31] = b[15] ^ is_sub
                mul_word: UInt(32) = 0
                mul_word[0:16] = a
                mul_word[16:32] = b
                gt: uint1 = bf16_gt(a, b)
                sel: uint16 = 0
                if op == OP_MAX:
                    sel = a if gt else b
                else:
                    sel = b if gt else a
                wb_word: UInt(40) = 0
                wb_word[0:16] = sel
                wb_word[16:32] = a
                wb_word[32:36] = op
                to_add.put(add_word)
                to_mul.put(mul_word)
                to_wb.put(wb_word)

        adder(src=to_add, dst=from_add)
        multiplier(src=to_mul, dst=from_mul)

        @df.kernel(mapping=[1], args=[C])
        def writeback(cv: uint16[n]):
            for i in range(n):
                sum_r: uint16 = from_add.get()
                mul_r: uint16 = from_mul.get()
                w: UInt(40) = to_wb.get()
                op: UInt(4) = w[32:36]
                result: uint16 = w[16:32]
                if op == OP_ADD or op == OP_SUB:
                    result = sum_r
                elif op == OP_MUL:
                    result = mul_r
                elif op == OP_MAX or op == OP_MIN:
                    result = w[0:16]
                cv[i] = result

    return top


def run_bits(mod, stim):
    op, a, b = (np.ascontiguousarray(stim[:, k]).astype(np.uint16) for k in range(3))
    c = np.zeros(len(stim), dtype=np.uint16)
    mod(op, a, b, c)
    return c


def run_native(mod, stim):
    op = np.ascontiguousarray(stim[:, 0]).astype(np.uint16)
    a, b = (np.ascontiguousarray(stim[:, k]).astype(np.uint16).view(ml_dtypes.bfloat16)
            for k in (1, 2))
    c = np.zeros(len(stim), dtype=ml_dtypes.bfloat16)
    mod(op, a, b, c)
    return c.view(np.uint16)


VARIANTS = {
    "native": (native, run_native),
    "bits": (composed, run_bits),
    "bits_dispatch": (dispatched, run_bits),
    "engines_native": (lambda n: composed(n, "native", "native"), run_bits),
    "add_bits_mul_native": (lambda n: composed(n, "bits", "native"), run_bits),
    "netlist": (netlist, run_bits),
    "netlist_native": (lambda n: netlist(n, _unit_add_native, _unit_mul_native), run_bits),
}


def _op(s):
    return int(s[0])


def _nan_in(s):
    return ref.is_nan(s[1]) or ref.is_nan(s[2])


DEVIATIONS = [
    ("AND/OR/XOR not implemented: rtl returns a (mov)",
     lambda s, g, w: _op(s) in (6, 7, 8) and int(w) == int(s[1])),
    ("ADD/SUB/MUL: NaN result is always +0x7fc0",
     lambda s, g, w: _op(s) in (0, 1, 2) and ref.is_nan(g) and int(w) == 0x7FC0),
    ("ADD: (+0)+(-0) = -0 (IEEE: +0)",
     lambda s, g, w: _op(s) == 0 and ref.is_zero(s[1]) and ref.is_zero(s[2])),
    ("SUB: (+0)-(+0) = -0 (IEEE: +0)",
     lambda s, g, w: _op(s) == 1 and ref.is_zero(s[1]) and ref.is_zero(s[2])),
    ("MUL: subnormal operand flushed to signed 0",
     lambda s, g, w: _op(s) == 2 and ref.is_zero(w)
     and (ref.exp_field(s[1]) == 0 or ref.exp_field(s[2]) == 0)),
    ("MUL: product < 2^-126 flushed to signed 0",
     lambda s, g, w: _op(s) == 2 and ref.is_zero(w)),
    ("MAX/MIN: NaN ordered by bf16_gt key, payload kept (IEEE maximum: NaN)",
     lambda s, g, w: _op(s) in (4, 5) and _nan_in(s)),
]
# Allo against the RTL: the RTL's departures from IEEE, then the ones that
# are Allo's own (``max``/``min`` are IEEE ``maximum``/``minimum``).
EXPLAIN = [
    ("simulator: a bf16 value through a phi is re-rounded via f32 (x86 "
     "__truncsfbf2), so a NaN that should pass through loses its payload",
     lambda s, g, w: _op(s) not in (0, 1, 2, 4, 5) and ref.is_nan(s[1])
     and int(w) == int(s[1]) and int(g) & 0x7FFF == 0x7FC0),
] + DEVIATIONS + [
    ("MAX/MIN: rtl orders -0 < +0 by bf16_gt, allo returns either zero",
     lambda s, g, w: _op(s) in (4, 5) and ref.is_zero(s[1]) and ref.is_zero(s[2])),
]

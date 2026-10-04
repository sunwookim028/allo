# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MAC plug-in: what an engine declares, and the two instances.

An ``Engine`` is the whole of what a composed unit needs to know about its
multiply-accumulate arithmetic, and the whole of what a swap changes:

``IN`` / ``IN_BITS``      the operand type and its width (a type to annotate
                          with, an integer to slice with -- the two are held
                          to each other, as ``reduction_tree_legality`` does)
``ACC`` / ``ACC_BITS``    the accumulator type every partial sum carries
``OUT`` / ``OUT_BITS``    what leaves the engine after ``pack``
``mul``, ``add``, ``pack`` the bodies, module-level Allo functions
``mul_latency``, ``add_latency``  the engine's declared pipeline depths, in
                          edges. Consumed by nothing in the body: a composite
                          derives its own timing from them (``legality.py``)
                          and a backend reports what it built (D-10)
``order``                 the accumulate order the engine is exact for. The
                          sequential acc24 chain rounds once per term in
                          ascending index; a balanced tree rounds once per
                          level. Different orders are different functions
                          (``ARITHMETIC.md`` §8), so the order is part of the
                          type, not an implementation detail
``ref_mul``, ``ref_add``, ``ref_pack``  the same arithmetic in numpy (U1's
                          references), so a composite's contract reference
                          can be evaluated with the engine's own arithmetic
``directives``            what the engine needs of a schedule to reach its
                          declared latency (C10: the adder's leading-zero
                          count must be unrolled), applied by every unit that
                          binds the engine

Both engines here are exact in the sense the README asks for: the bf16 one is
MiniTPU's (U1 bit-exact), the int8 one is TinyTPU's (an int8 x int8 product
is exact in int16 and 2^16 of them fit an int32 with room).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allo.ir.types import UInt, int8, int16, int32

from examples.minitpu.harness import ref
from examples.minitpu.harness.ref_mxu import pack_bf16
from examples.minitpu.template import bf16_engine


@dataclass(frozen=True)
class Engine:
    name: str
    IN: type
    IN_BITS: int
    ACC: type
    ACC_BITS: int
    OUT: type
    OUT_BITS: int
    mul: callable
    add: callable
    pack: callable
    mul_latency: int
    add_latency: int
    order: str
    ref_mul: callable
    ref_add: callable
    ref_pack: callable
    directives: callable = None
    #: numpy dtypes the rig moves each port in (the region's boundary arrays)
    np_in: type = np.uint16
    np_acc: type = np.uint32
    np_out: type = np.uint16

    def __post_init__(self):
        for width, dtype in (("IN_BITS", "IN"), ("ACC_BITS", "ACC"),
                             ("OUT_BITS", "OUT")):
            assert getattr(self, width) == getattr(self, dtype).bits, (
                f"engine {self.name}: {width}={getattr(self, width)} but "
                f"{dtype} is {getattr(self, dtype)}; a unit that slices a "
                f"different width from the one it computes in reads the "
                f"wrong bits without failing")
        assert self.order in ("sequential",), (
            f"engine {self.name}: a MAC engine accumulates one term at a "
            f"time; order {self.order!r} belongs to a matrix engine")

    def namespace(self, prefix="MAC") -> dict:
        """The names a unit binds: ``MAC_IN``, ``MAC_MUL``, ..."""
        return {f"{prefix}_IN": self.IN, f"{prefix}_IN_BITS": self.IN_BITS,
                f"{prefix}_ACC": self.ACC, f"{prefix}_ACC_BITS": self.ACC_BITS,
                f"{prefix}_OUT": self.OUT, f"{prefix}_OUT_BITS": self.OUT_BITS,
                f"{prefix}_MUL": self.mul, f"{prefix}_ADD": self.add,
                f"{prefix}_PACK": self.pack}

    @staticmethod
    def names(prefix="MAC") -> tuple:
        return tuple(f"{prefix}_{n}" for n in (
            "IN", "IN_BITS", "ACC", "ACC_BITS", "OUT", "OUT_BITS", "MUL",
            "ADD", "PACK"))


# --- bf16 -> acc24 -> bf16: MiniTPU's MXU ------------------------------------

def _bf16_directives(s, ctx):
    # C10: the acc24 adder's normalisation needs `leading_zeros19` unrolled
    # for II=1 (U1: Catapult on the ALU). The engine says so once; every unit
    # binding the engine applies it.
    s.unroll("leading_zeros19:offset")


BF16_ACC24 = Engine(
    name="bf16_acc24",
    IN=UInt(16), IN_BITS=16, ACC=UInt(24), ACC_BITS=24, OUT=UInt(16), OUT_BITS=16,
    mul=bf16_engine.mul_acc24_bits, add=bf16_engine.acc24_add_bits,
    pack=bf16_engine.pack_bf16_bits,
    mul_latency=0,     # combinational in the PE; the PE registers the product
    add_latency=3,     # vpu_pkg MXU_ACC_ADD_LATENCY
    order="sequential",
    ref_mul=lambda a, w: ref.mxu_bf16_mul_acc24(a, w).astype(np.int64),
    ref_add=lambda p, q: ref.mxu_acc24_add(p, q).astype(np.int64),
    ref_pack=pack_bf16,
    directives=_bf16_directives,
    np_in=np.uint16, np_acc=np.uint32, np_out=np.uint16,
)


# --- int8 -> int32 -> int32: TinyTPU's PE -------------------------------------

def mul_int8(a: int8, w: int8) -> int32:
    a16: int16 = a
    w16: int16 = w
    p: int32 = a16 * w16
    return p


def add_int32(p: int32, q: int32) -> int32:
    s: int32 = p + q
    return s


def pack_int32(v: int32) -> int32:
    return v


def _ref_mul_int8(a, w):
    return np.asarray(a, dtype=np.int64) * np.asarray(w, dtype=np.int64)


def _ref_add_int32(p, q):
    return ((np.asarray(p, dtype=np.int64) + np.asarray(q, dtype=np.int64)
             + 2**31) % 2**32 - 2**31)


INT8_INT32 = Engine(
    name="int8_int32",
    IN=int8, IN_BITS=8, ACC=int32, ACC_BITS=32, OUT=int32, OUT_BITS=32,
    mul=mul_int8, add=add_int32, pack=pack_int32,
    mul_latency=1, add_latency=1,   # TinyTPU's Vitis schedule (one DSP, one add)
    order="sequential",
    ref_mul=_ref_mul_int8, ref_add=_ref_add_int32,
    ref_pack=lambda v: np.asarray(v, dtype=np.int64),
    np_in=np.int8, np_acc=np.int32, np_out=np.int32,
)

ENGINES = {e.name: e for e in (BF16_ACC24, INT8_INT32)}

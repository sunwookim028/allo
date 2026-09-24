# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provenance: cherry-picked from Kai Shao's ACT work --
# https://github.com/kkkaishao/allo, branch ``act``, commit ``3c1ad38``,
# file ``allo/exp/dsa/primitive.py``, lines 26-95 (the categories and the
# registry). See ``ATTRIBUTION.md``.

"""The closed compute vocabulary, as a table.

``REGISTRY`` names every prim the recognizer knows, its *category* (which fixes
shape/dtype inference and the TOSA calling convention), and whether it commutes.
Adding a prim that fits an existing category is one row -- no new branch anywhere.
The few ops with irregular shape or codegen (``relu``/``transpose``/``matmul`` and
the conv family) are bespoke and are deliberately **not** in ``REGISTRY``; the
recognizer lists them separately in ``recognize._BESPOKE``.

What was *not* taken from Kai's file: the traced prim functions below line 95
(``add``, ``matmul``, ``conv2d``, ...) and their shape inference. Those return a
``TensorProxy`` and import ``allo.lang.core`` and his ``core.py``, which is the
``allov2`` dependency this fork does not carry. The registry itself is data and
has no imports at all, which is what makes it portable.

This is the *standard vocabulary* half of the reason for the port: ACT compiles a
TOSA program (``docs/source/extensions/act.rst``), so the names a workload's
epilogue and contractions are spelled in should be TOSA's, not a two-word set of
our own (``allo/act/workload.py``'s ``POINTWISE``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Each category fixes (a) shape/dtype inference and (b) the TOSA calling
# convention -- how many of an op's operands are data and what the rest mean.
UNARY = "unary"  # Op(out, x);            shape = in,        dtype = in
UNARY_ZP = "unary_zp"  # Op(out, x, zp, zp);    shape = in,        dtype = in
BINARY = "binary"  # Op(out, a, b);         shape = a (== b),  dtype = a
BINARY_SHIFT = "binary_shift"  # Op(out, a, b, shift);  shape = a,         dtype = a
COMPARE = "compare"  # bool result;           shape = a,         dtype = u1
SELECT = "select"  # Op(out, cond, a, b);   shape = a,         dtype = a
REDUCE = "reduce"  # Op(in, axis, results); shape = in[axis->1],dtype = in
CAST = "cast"  # Op(out, x);            shape = in,        dtype = target

CATEGORIES = (UNARY, UNARY_ZP, BINARY, BINARY_SHIFT, COMPARE, SELECT, REDUCE, CAST)


def _camel(tag: str) -> str:
    return "".join(part.capitalize() for part in tag.split("_")) + "Op"


@dataclass(frozen=True)
class PrimSpec:
    tag: str
    category: str
    commutative: bool = False
    cls: str = ""  # TOSA op class name override (default: Camel(tag) + "Op")

    def tosa_class(self) -> str:
        return self.cls or _camel(self.tag)

    @property
    def tosa_name(self) -> str:
        return f"tosa.{self.tag}"


_REGISTRY = [
    # binary arithmetic
    PrimSpec("add", BINARY, commutative=True),
    PrimSpec("sub", BINARY),
    PrimSpec("mul", BINARY_SHIFT, commutative=True),
    PrimSpec("maximum", BINARY, commutative=True),
    PrimSpec("minimum", BINARY, commutative=True),
    PrimSpec("pow", BINARY),
    PrimSpec("intdiv", BINARY, cls="IntDivOp"),  # op name tosa.intdiv, class IntDivOp
    # unary math / activations
    PrimSpec("abs", UNARY),
    PrimSpec("exp", UNARY),
    PrimSpec("log", UNARY),
    PrimSpec("rsqrt", UNARY),
    PrimSpec("reciprocal", UNARY),
    PrimSpec("floor", UNARY),
    PrimSpec("ceil", UNARY),
    PrimSpec("sin", UNARY),
    PrimSpec("cos", UNARY),
    PrimSpec("tanh", UNARY),
    PrimSpec("sigmoid", UNARY),
    PrimSpec("erf", UNARY),
    PrimSpec("negate", UNARY_ZP),
    # comparison + select
    PrimSpec("equal", COMPARE, commutative=True),
    PrimSpec("greater", COMPARE),
    PrimSpec("greater_equal", COMPARE),
    PrimSpec("select", SELECT),
    # reductions
    PrimSpec("reduce_sum", REDUCE),
    PrimSpec("reduce_max", REDUCE),
    PrimSpec("reduce_min", REDUCE),
    PrimSpec("reduce_product", REDUCE),
    # type conversion
    PrimSpec("cast", CAST),
]

REGISTRY: dict[str, PrimSpec] = {p.tag: p for p in _REGISTRY}

COMMUTATIVE = frozenset(tag for tag, p in REGISTRY.items() if p.commutative)
REDUCE_TAGS = frozenset(tag for tag, p in REGISTRY.items() if p.category == REDUCE)

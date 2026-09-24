# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fork-local. It stands on the recognizer cherry-picked from Kai Shao's ACT work
# (https://github.com/kkkaishao/allo, branch ``act``, commit ``3c1ad38``) -- see
# ``allo/act/recognize.py`` and ``ATTRIBUTION.md`` -- but the target here is this
# fork's ``allo.act.workload.Workload``, which ACT does not have.

"""A TOSA program in, one :class:`~allo.act.workload.Workload` out.

This is the front door ``allo/act/`` has been missing. ``workloads.py`` is a
hand-written registry of four entries; ``examples/tinytpu/workloads/extract.py``
reads a PyTorch module through ``torch.fx`` and ShapeProp and writes specs in a
vocabulary of our own (``epilogue: ["relu", "saturate"]``). Neither speaks the
language ``docs/source/extensions/act.rst`` says ACT compiles, which is **TOSA**.
This module does: it parses a TOSA module, recognizes every op through
:mod:`allo.act.recognize`, and builds the same ``Workload`` object the registry
holds -- so a spec is still its own gold, evaluated by ``np.einsum``.

**Not wired in as the default path.** ``extract.py`` stays exactly as it is until
this is proven on a real model, which needs the ``tosa.rescale`` opcode
(``dma_st``'s ``mvout`` mode, scoped separately).

The grammar it accepts is the one a ``Workload`` can express, and nothing wider:

.. code-block:: text

    result := epilogue* ( matmul ( + matmul )* )
    matmul := tosa.matmul over 3-D operands with a unit batch dim

which covers ``gemm``, ``gemm.relu``, ``gemm.sum`` and ``gemm.sum.relu``. Every op
outside it is a named refusal, never an approximation -- the policy
``extract.py`` already follows.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._mlir import ir
from .._mlir.dialects import allo as allo_d
from .errors import NoMatchError, QuantizationError, ShapeError
from .recognize import (
    _LAYOUT_AND_CONST,
    _canon,
    entry_block,
    normalize_source,
    perms_of,
    recognize,
)
from .workload import POINTWISE, Tensor, Workload

_SKIP = _LAYOUT_AND_CONST | {"func.return"}

# lhs operands are named A, A2, ...; rhs (weights) B, B2, ...; the result C. The
# naming mirrors allo/act/workloads.py so a recognized Workload is comparable with
# a hand-written one.
_LHS_NAMES = "A"
_RHS_NAMES = "B"


@dataclass(frozen=True)
class Recognized:
    """A recognized TOSA program: the workload, plus what the workload cannot hold.

    ``Workload`` is deliberately extent-free and dtype-free -- a shape is supplied
    at mapping time -- so the concrete shapes, element types and host ABI order
    that the source *did* pin live here instead of being thrown away."""

    workload: Workload
    extents: dict
    dtypes: dict
    arg_shapes: tuple  # (operand name, shape) in func.func argument order
    quantization: tuple  # (op name, Quantization) for every recognized op

    @property
    def symmetric(self) -> bool:
        return all(q.is_symmetric for _, q in self.quantization)


def parse(source: str):
    """Parse a TOSA module and normalize it. Returns ``(context, module)``.

    The context is returned because the module's ops are views into it: drop it and
    every ``ir.Value`` below becomes a dangling reference."""
    ctx = ir.Context()
    allo_d.register_dialect(ctx)
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(source)
        normalize_source(module)
    return ctx, module


class _Builder:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.fn, self.block = entry_block(module)
        self.args = list(self.block.arguments)
        self.visited = set()
        self.quant = []
        self.names = {}  # (arg index, transposed) -> operand name
        self.operands = []  # Tensor, in first-seen order
        self.arg_shapes = {}

    # -- refusals ---------------------------------------------------------
    def _refuse(self, op, why, cls=NoMatchError):
        raise cls(f"{self.name}: {op.operation.name} at {op.location}: {why}")

    # -- op recognition ---------------------------------------------------
    def _match(self, op):
        m = recognize(op)
        if not m:
            self._refuse(op, m.refusal)
        self.quant.append((op.operation.name, m.quant))
        if not m.quant.is_symmetric:
            self._refuse(
                op,
                f"{m.quant.why_not_neutral()}. This machine requires per-tensor "
                f"**symmetric** int8 quantization: (A-za)(B-zb) needs row/column "
                f"correction terms the MXU cannot produce. See "
                f"allo/act/recognize.py.",
                QuantizationError,
            )
        if not m.quant.is_neutral:
            self._refuse(
                op,
                f"{m.quant.why_not_neutral()}. A non-zero shift is requantization; "
                f"the opcode for it is an `mvout` mode of `dma_st`, scoped "
                f"separately and not built.",
                QuantizationError,
            )
        self.visited.add(op.operation)
        return m

    # -- tensor tracing ---------------------------------------------------
    def _trace(self, value):
        """A matmul operand back to a function argument, plus whether its last two
        dims are swapped on the way. ``tosa.reshape`` is peeled by ``_canon``;
        ``tosa.transpose`` of the trailing two dims is absorbed into the operand's
        rank order, which is what lets ``a @ b.T`` (how torch lowers ``nn.Linear``)
        become a plain (K, N) weight indexed (N, K)."""
        transposed = False
        while True:
            value = _canon(value)
            owner = value.owner
            if isinstance(owner, ir.Block):
                for i, a in enumerate(self.args):
                    if a == value:
                        return i, transposed
                raise NoMatchError(
                    f"{self.name}: a matmul operand is a block argument of a "
                    f"region other than the entry block"
                )
            if owner.operation.name != "tosa.transpose":
                self._refuse(
                    owner,
                    "a matmul operand must trace back to a function argument "
                    "through reshapes and transposes only",
                )
            self._match(owner)
            perms = perms_of(owner)
            n = len(perms)
            if perms != list(range(n - 2)) + [n - 1, n - 2]:
                self._refuse(
                    owner,
                    f"only a trailing-two-dim transpose can be absorbed into an "
                    f"operand's rank order; perms={perms}",
                )
            transposed = not transposed
            value = owner.operands[0]

    def _tensor(self, value, side, ranks):
        idx, transposed = self._trace(value)
        key = (idx, transposed)
        if key not in self.names:
            prefix = _LHS_NAMES if side == "lhs" else _RHS_NAMES
            n = 1 + sum(1 for nm in self.names.values() if nm.startswith(prefix))
            self.names[key] = prefix if n == 1 else f"{prefix}{n}"
            order = tuple(reversed(ranks)) if transposed else tuple(ranks)
            self.operands.append(Tensor(self.names[key], order))
            self.arg_shapes[self.names[key]] = (
                idx,
                tuple(ir.RankedTensorType(self.args[idx].type).shape),
            )
        return self.names[key]

    # -- the grammar ------------------------------------------------------
    def _matmul_term(self, value, extents, dtypes):
        value = _canon(value)
        owner = value.owner
        if isinstance(owner, ir.Block):
            self._fail(
                "a workload's result is a contraction, but this one is a function "
                "argument passed straight through"
            )
        m = self._match(owner)
        if m.tag != "matmul":
            raise NoMatchError(
                f"{self.name}: a workload is a sum of contractions; "
                f"{owner.operation.name} (recognized as {m.tag!r}) is not one. "
                f"allo.act.workload.Workload has no node for it."
            )
        a_ty = ir.RankedTensorType(m.ins[0].type)
        b_ty = ir.RankedTensorType(m.ins[1].type)
        o_ty = ir.RankedTensorType(owner.results[0].type)
        if a_ty.rank != 3 or b_ty.rank != 3:
            self._refuse(owner, "tosa.matmul operands must be 3-D", ShapeError)
        if a_ty.shape[0] != 1 or b_ty.shape[0] != 1:
            self._refuse(
                owner,
                f"batched matmul (batch {a_ty.shape[0]}x{b_ty.shape[0]}) has no "
                f"rank in this Workload; only a unit batch dim is accepted",
                ShapeError,
            )
        dims = {"M": a_ty.shape[1], "K": a_ty.shape[2], "N": b_ty.shape[2]}
        if b_ty.shape[1] != dims["K"]:
            self._refuse(owner, "matmul operands disagree on K", ShapeError)
        for r, v in dims.items():
            if extents.setdefault(r, v) != v:
                raise ShapeError(
                    f"{self.name}: two contractions disagree on rank {r} "
                    f"({extents[r]} vs {v}); they are not one iteration space"
                )
        dtypes.setdefault("C", str(o_ty.element_type))
        act = self._tensor(m.ins[0], "lhs", ("M", "K"))
        wgt = self._tensor(m.ins[1], "rhs", ("K", "N"))
        dtypes[act] = str(a_ty.element_type)
        dtypes[wgt] = str(b_ty.element_type)
        return (act, wgt)

    def _fail(self, why):
        raise NoMatchError(f"{self.name}: {why}")

    def build(self):
        ret = [op for op in self.block.operations if op.operation.name == "func.return"]
        if len(ret) != 1 or len(ret[0].operands) != 1:
            self._fail("the source func must return exactly one value")
        value = ret[0].operands[0]

        # 1. the epilogue: a chain of pointwise ops on the contraction's result.
        epilogue = []
        while True:
            value = _canon(value)
            owner = value.owner
            if isinstance(owner, ir.Block):
                break
            m = recognize(owner)
            if not m or m.tag not in POINTWISE:
                break
            self._match(owner)
            epilogue.append(m.tag)
            value = m.ins[0]
        epilogue.reverse()  # innermost first, the order Workload applies them

        # 2. the contraction sum: tosa.add flattened into terms.
        terms, stack = [], [value]
        while stack:
            v = _canon(stack.pop())
            owner = v.owner
            if not isinstance(owner, ir.Block) and recognize(owner).tag == "add":
                m = self._match(owner)
                stack.extend(reversed(m.ins))
                continue
            terms.append(v)

        extents, dtypes = {}, {}
        contractions = tuple(self._matmul_term(t, extents, dtypes) for t in terms)

        # 3. nothing may be left over: an unvisited compute op is a part of the
        #    program this Workload does not describe.
        leftover = [
            op
            for op in self.block.operations
            if op.operation.name not in _SKIP and op.operation not in self.visited
        ]
        if leftover:
            self._fail(
                "these ops are not reachable from the result, so the Workload "
                "would not describe the whole program: "
                + ", ".join(sorted({op.operation.name for op in leftover}))
            )

        workload = Workload(
            name=self.name,
            ranks=("M", "K", "N"),
            reduce=("K",),
            operands=tuple(self.operands),
            result=Tensor("C", ("M", "N")),
            contractions=contractions,
            epilogue=tuple(epilogue),
        )
        arg_shapes = tuple(
            (name, shape)
            for name, (_, shape) in sorted(
                self.arg_shapes.items(), key=lambda kv: kv[1][0]
            )
        )
        return Recognized(
            workload=workload,
            extents=dict(extents),
            dtypes=dtypes,
            arg_shapes=arg_shapes,
            quantization=tuple(self.quant),
        )


def workload_from_tosa(source: str, name: str = "tosa") -> Recognized:
    """Recognize a TOSA program as a :class:`~allo.act.workload.Workload`.

    Raises a :class:`~allo.act.errors.CompileError` subclass, naming the op and the
    reason, for anything the workload model cannot hold."""
    _ctx, module = parse(source)
    return _Builder(module, name).build()

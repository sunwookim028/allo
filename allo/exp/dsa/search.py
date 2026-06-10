# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search backend: compile a source program onto an ISA.

The source program is a TOSA-dialect MLIR module supplied as *text* (e.g. from
torch_mlir's TOSA backend) — ``compile_program`` parses it; we own no source
generator. A ``Catalog`` indexes the ISA's compute instructions by the root *prim
tag* of their pattern DAG (and recognizes the tag of a source op); three stages
then run, top to bottom in compilation order:

- ``match_program`` (Stage 1) — cover the source compute DAG with instruction
  patterns via cost-aware tree-DP, folding a multi-node subgraph into one
  instruction; binds each instruction's source buffers to source SSA values.
- ``solve`` (Stage 2) — infer each instruction's shape params by unifying its
  symbolic visible shapes with the bound source shapes (exact-fit; no tiling).
- ``plan`` (Stage 3) — liveness-driven slot allocation + data movement (routing
  and spilling), producing a ``CompiledProgram`` (an emit list + I/O map).

The public entry is ``ISA.compile_program(source)`` (sugar over ``compile_program``
here); the returned ``CompiledProgram`` is callable — ``prog(*inputs)`` runs it on
the functional simulator (the same oracle backbone hand-written assembly uses) — and
``prog.dump()`` prints the emitted instruction sequence.

See ``todos/search.md`` for the full per-stage algorithm analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod

import numpy as np
import sympy

from ..._mlir import ir
from ..._mlir.dialects import tosa
from . import primitive
from .core import ISA, Instruction, _index_params, arity, param_roles, trace_instruction
from .oracle import EmitRecord, OracleConfig, OracleProgram, simulate

# ==========================================================================#
# Catalog: prim-tag index + source-op recognizer
# ==========================================================================#

# Source op-name -> prim tag. The source is value-semantics TOSA throughout, and a
# prim's source op name is always `tosa.<tag>`, so the map is derived from the prim
# registry; matmul/transpose are bespoke (not in the registry). relu is recognized
# separately (it is a tosa.clamp with min == 0, not a tosa.relu). tosa.matmul is
# batched 3-D (its 2-D<->3-D reshapes are handled by `_canon`).
_NAMED_TAG = {f"tosa.{tag}": tag for tag in primitive.REGISTRY}
# bespoke prims (not in the registry): matmul / transpose / reverse and the conv family.
for _bespoke in (
    "matmul",
    "transpose",
    "reverse",
    "conv2d",
    "depthwise_conv2d",
    "max_pool2d",
    "avg_pool2d",
):
    _NAMED_TAG[f"tosa.{_bespoke}"] = _bespoke

# Pure layout / constant ops: transparent to matching (reshape is an alias; consts
# carry no compute). `_canon` peels reshapes; use-counting skips all of these.
_LAYOUT_AND_CONST = {"tosa.reshape", "tosa.const", "tosa.const_shape"}


def _canon(value):
    """Peel ``tosa.reshape`` chains to the underlying value. Reshape is a layout
    alias (e.g. TOSA's batched-matmul 2-D<->3-D wrapping at I/O), transparent for
    both matching and allocation — two reshapes of one value share its slot."""
    while True:
        owner = value.owner
        if isinstance(owner, ir.Block) or owner.operation.name != "tosa.reshape":
            return value
        value = owner.operands[0]  # reshape input1 (the data operand)


def instruction_pattern(instruction: Instruction):
    """The compute pattern of an instruction: the root ``TensorProxy`` of its
    semantics DAG. Internal nodes are prim ops; ``arg`` leaves bind to the
    instruction's source buffers (by ``buffer_index``). A 1:1 instruction is just
    a depth-1 pattern. Returns ``None`` for data-movement (identity) or
    multi-output instructions (not matched yet)."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return None
    root = results[0]
    return None if root.kind in ("identity", "arg") else root


def source_tag(op) -> str | None:
    """Recognize a source op into a prim tag, or None if unsupported."""
    name = op.operation.name
    if name in _NAMED_TAG:
        return _NAMED_TAG[name]
    if name == "tosa.clamp" and _is_relu_clamp(op):
        return "relu"
    return None


def _is_relu_clamp(op) -> bool:
    """``tosa.clamp`` with ``min_val == 0`` is relu (max is +inf). This is the
    form torch_mlir's TOSA backend emits for aten.relu."""
    return ir.FloatAttr(op.operation.attributes["min_val"]).value == 0.0


class Catalog:
    """Indexes an ISA's compute instructions by the *root* prim tag of their
    pattern, so single-op and multi-node instructions are looked up uniformly."""

    def __init__(self, isa: ISA):
        self.isa = isa
        self.patterns: dict[str, list[tuple[Instruction, object]]] = {}
        for spec in isa.instructions:
            instr = isa._ops[spec.name]
            root = instruction_pattern(instr)
            if root is not None:
                self.patterns.setdefault(root.kind, []).append((instr, root))

    def candidates(self, tag: str | None) -> list[tuple[Instruction, object]]:
        return self.patterns.get(tag, []) if tag is not None else []


# ==========================================================================#
# Stage 1 — semantic matching
# ==========================================================================#


@dataclass
class Match:
    instruction: object  # Instruction
    operand_values: list  # leaf bindings, in source-buffer order
    result_value: object  # the tile's output ir.Value
    shape_params: dict = field(default_factory=dict)  # solved access param -> int

    @property
    def bound_values(self) -> list:
        """Source values bound to the instruction buffers, in [src..., dst] order."""
        return self.operand_values + [self.result_value]


@dataclass
class Selection:
    func: object
    matches: list


def _entry_block(module):
    for op in module.body.operations:
        if op.operation.name == "func.func":
            return op, op.regions[0].blocks[0]
    raise ValueError("source module has no func.func")


# ==========================================================================#
# Source normalization (run on the parsed TOSA before matching)
# ==========================================================================#


def _const_shape(shape):
    n = len(shape)
    vals = ir.Attribute.parse(
        f"dense<[{', '.join(map(str, shape))}]> : tensor<{n}xindex>"
    )
    return tosa.ConstShapeOp(ir.Type.parse(f"!tosa.shape<{n}>"), vals).result


def normalize_source(module):
    """Canonicalize torch_mlir's TOSA so instruction patterns can match it.

    Sinks ``reshape(transpose(X, p))`` -> ``transpose(reshape(X), p')`` when the
    reshape only prepends unit (batch) dims. torch lowers ``a @ b.T`` as a 2-D
    ``tosa.transpose`` (perms ``[1, 0]``) *then* a batch reshape to 3-D, whereas an
    instruction's semantics carry the weight transpose in batched 3-D form (perms
    ``[0, 2, 1]``). Without this rewrite the two never line up and the systolic
    matmul cannot absorb the transpose."""
    _, block = _entry_block(module)
    for op in list(block.operations):
        if op.operation.name != "tosa.reshape":
            continue
        t = op.operands[0].owner
        if isinstance(t, ir.Block) or t.operation.name != "tosa.transpose":
            continue
        in_ty = ir.RankedTensorType(t.operands[0].type)
        t_out = ir.RankedTensorType(t.results[0].type)
        r_out = ir.RankedTensorType(op.results[0].type)
        k = r_out.rank - t_out.rank  # number of prepended dims
        if (
            k <= 0
            or list(r_out.shape[:k]) != [1] * k
            or list(r_out.shape[k:]) != list(t_out.shape)
        ):
            continue  # reshape does more than prepend unit dims -> leave it alone
        perms = list(ir.DenseI32ArrayAttr(t.operation.attributes["perms"]))
        new_perms = list(range(k)) + [p + k for p in perms]
        new_shape = [1] * k + list(in_ty.shape)
        with ir.InsertionPoint(op), ir.Location.unknown():
            reshaped = tosa.ReshapeOp(
                t.operands[0],
                _const_shape(new_shape),
                results=[ir.RankedTensorType.get(new_shape, in_ty.element_type)],
            )
            new_t = tosa.TransposeOp(r_out, reshaped.result, new_perms)
        op.results[0].replace_all_uses_with(new_t.result)
        op.operation.erase()
        if not list(
            t.results[0].uses
        ):  # the old transpose precedes op, already visited
            t.operation.erase()


_COMMUTATIVE = {tag for tag, p in primitive.REGISTRY.items() if p.commutative}
_REDUCE_TAGS = {
    tag for tag, p in primitive.REGISTRY.items() if p.category == primitive.REDUCE
}


def _source_ins(op) -> list:
    """The data-input operands of a recognized (value-semantics TOSA) source op —
    all operands, minus tosa.mul's trailing ``shift`` / tosa.matmul's zero-points."""
    operands = list(op.operands)
    # data operands only — drop trailing shift / zero-points / bias-side constants.
    keep = {
        "tosa.mul": 2,
        "tosa.matmul": 2,
        "tosa.negate": 1,
        "tosa.conv2d": 3,
        "tosa.depthwise_conv2d": 3,  # input, weight, bias (+ 2 zps)
        "tosa.avg_pool2d": 1,  # input (+ 2 zps)
    }.get(op.operation.name)
    return operands[:keep] if keep is not None else operands


def _perms(op) -> list:
    """The permutation of a ``tosa.transpose`` source op (an ``array<i32>`` attr)."""
    return list(ir.DenseI32ArrayAttr(op.operation.attributes["perms"]))


def _axis_attr(op) -> int:
    """The ``axis`` i32 attr of a ``tosa.reduce_*`` / ``tosa.reverse`` source op."""
    return ir.IntegerAttr(op.operation.attributes["axis"]).value


def _i64_attr(op, name) -> list:
    return list(ir.DenseI64ArrayAttr(op.operation.attributes[name]))


_CONV_TAGS = {"conv2d", "depthwise_conv2d"}
_POOL_TAGS = {"max_pool2d", "avg_pool2d"}


def _attrs_match(pnode, op) -> bool:
    """Spatial attributes are part of the semantics (like transpose's perms)."""
    if pnode.kind in _CONV_TAGS:
        a = pnode.attrs
        return all(_i64_attr(op, k) == a[k] for k in ("pad", "stride", "dilation"))
    if pnode.kind in _POOL_TAGS:
        a = pnode.attrs
        return all(_i64_attr(op, k) == a[k] for k in ("kernel", "stride", "pad"))
    return True


def _operand_orders(pnode, ins):
    """Operand orderings to try; both orders for a commutative binary prim."""
    if pnode.kind in _COMMUTATIVE and len(ins) == 2:
        return (ins, [ins[1], ins[0]])
    return (ins,)


def _match_pattern(pnode, value, def_op, use, bindings, within, interior) -> bool:
    """Match pattern node ``pnode`` against source ``value``.

    ``arg`` leaves bind a source buffer to ``value``; internal nodes must align
    with a recognized source op of the same prim tag and recurse on its inputs.
    Records each folded source value (whose defining op is absorbed into the tile)
    in ``interior``, and the within-tile use count of each operand in ``within``, so
    the caller can reject a fold in which a folded non-root value *escapes* (is also
    used outside the tile and therefore must be materialized). This deferred
    cut-point test permits internal fan-out — e.g. softmax's ``exp`` feeding both the
    reduce and the divide — which a per-node single-use test would wrongly forbid.
    Mutates ``bindings``/``within``/``interior`` in place, rolling back on a failed
    branch.
    """
    if pnode.kind == "arg":
        prev = bindings.get(pnode.buffer_index)
        if prev is not None and prev != value:
            return False
        bindings[pnode.buffer_index] = value
        return True
    op = def_op.get(value)
    if op is None or source_tag(op) != pnode.kind:
        return False
    if pnode.kind == "transpose" and _perms(op) != list(pnode.permutation):
        return False  # the permutation is part of the semantics, not just the tag
    if pnode.kind in _REDUCE_TAGS and _axis_attr(op) != pnode.axis:
        return False  # the reduced axis is part of the semantics
    if pnode.kind == "reverse" and _axis_attr(op) != pnode.axis:
        return False  # the reversed axis is part of the semantics
    if not _attrs_match(pnode, op):
        return False  # conv/pool spatial attrs (pad/stride/dilation/kernel)
    ins = _source_ins(op)
    if len(ins) != len(pnode.args):
        return False
    for order in _operand_orders(pnode, ins):
        saved_b, saved_w, saved_i = dict(bindings), dict(within), set(interior)
        for sv in order:  # each operand is one within-tile use of that value
            within[_canon(sv)] = within.get(_canon(sv), 0) + 1
        if all(
            _match_pattern(pa, sv, def_op, use, bindings, within, interior)
            for pa, sv in zip(pnode.args, order)
        ):
            interior.add(value)
            return True
        bindings.clear()
        bindings.update(saved_b)
        within.clear()
        within.update(saved_w)
        interior.clear()
        interior.update(saved_i)
    return False


@dataclass
class _Choice:
    cost: float
    instruction: object
    operands: list


def _pattern_has(node, kind) -> bool:
    return node.kind == kind or any(_pattern_has(a, kind) for a in node.args)


def _describe_pattern(node) -> str:
    """A compact source-level rendering of an instruction's compute pattern, e.g.
    ``matmul(%0, transpose(%1))`` — the shape of source DAG it matches."""
    if node.kind == "arg":
        return f"%{node.buffer_index}"
    return f"{node.kind}({', '.join(_describe_pattern(a) for a in node.args)})"


def _no_match_error(op, catalog) -> str:
    """An actionable message for an unmatched source op: show its operand shapes,
    the candidate instructions' patterns, and — the common case — a hint when an
    instruction consumes an operand transposed but the source provides it plain."""
    tag = source_tag(op)
    shapes = [tuple(ir.RankedTensorType(o.type).shape) for o in _source_ins(op)]
    head = f"no instruction matches source op '{op.operation.name}' with operand shapes {shapes}"
    candidates = catalog.candidates(tag)
    if not candidates:
        return f"{head}: the ISA defines no instruction computing '{tag}'."
    lines = [
        f"{head}.",
        f"  '{tag}' instruction(s) exist but none matches structurally:",
    ]
    lines += [
        f"    {instr.name}: {_describe_pattern(root)}" for instr, root in candidates
    ]
    if any(_pattern_has(root, "transpose") for _, root in candidates):
        lines.append(
            "  hint: an instruction consumes an operand transposed (the systolic "
            "computes X @ W^T). Write the source op in that form (e.g. `a @ b.T`) "
            "or pre-transpose the operand on the host."
        )
    return "\n".join(lines)


def match_program(catalog: Catalog, source_module) -> Selection:
    """Cover the source compute DAG with instruction patterns via cost-aware
    tree-DP. A value used more than once is a forced cut point (it cannot be
    folded into a consumer's tile), so the foldable subgraphs are trees and a
    per-value DP is globally optimal.

    ``materialize(v)`` returns the cheapest tile rooted at ``v``: instruction cost
    plus the materialization cost of its operands — but only *single-use* operands
    are charged, because a shared (multi-use) operand is materialized once as its
    own root and must not be billed to every consumer. The optimum is reconstructed
    from the returned values and scheduled in def-before-use order."""
    func, block = _entry_block(source_module)
    ops = list(block.operations)
    terminator = ops[-1]

    def_op: dict = {}  # ir.Value -> the recognized op defining it
    index: dict = {}  # that op's result value -> block position (for scheduling)
    for i, op in enumerate(ops):
        if source_tag(op) is not None:
            def_op[op.results[0]] = op
            index[op.results[0]] = i

    # Use-counts on canonical values, skipping pure layout/const ops (a reshape's
    # use of its input is plumbing, not a real consumer).
    use: dict = {}
    for op in ops:
        if op.operation.name in _LAYOUT_AND_CONST:
            continue
        for v in op.operands:
            cv = _canon(v)
            use[cv] = use.get(cv, 0) + 1

    memo: dict = {}  # canonical value -> _Choice (optimal tile to materialize it)

    def materialize(v) -> _Choice:
        if v in memo:
            return memo[v]
        op = def_op[v]
        chosen = None
        for instr, root in catalog.candidates(source_tag(op)):
            bindings, within, interior = {}, {}, set()
            if not _match_pattern(root, v, def_op, use, bindings, within, interior):
                continue
            # Deferred cut-point test: a folded (non-root) value must be used only
            # within this tile; if its global use count exceeds its within-tile use
            # count it escapes and must be its own root, so this fold is invalid.
            if any(use.get(iv, 0) != within.get(iv, 0) for iv in interior if iv != v):
                continue
            n_src = len(instr.spec.sources)
            if not all(i in bindings for i in range(n_src)):
                continue
            operands = [bindings[i] for i in range(n_src)]
            cost = instr.spec.cost + sum(
                materialize(_canon(ov)).cost
                for ov in operands
                if _canon(ov) in def_op and use.get(_canon(ov), 0) == 1
            )
            if chosen is None or cost < chosen.cost:
                chosen = _Choice(cost, instr, operands)
        assert chosen is not None, _no_match_error(op, catalog)
        memo[v] = chosen
        return chosen

    matches: list[Match] = []
    visited: set = set()

    def schedule(v):
        if v in visited:
            return
        visited.add(v)
        ch = materialize(v)
        matches.append(Match(ch.instruction, ch.operands, v))
        for ov in ch.operands:
            if _canon(ov) in def_op:
                schedule(_canon(ov))

    for v in terminator.operands:
        if _canon(v) in def_op:
            schedule(_canon(v))

    assert matches, "no source compute ops matched any instruction"
    matches.sort(key=lambda m: index[m.result_value])
    return Selection(func, matches)


# ==========================================================================#
# Stage 2 — parameter solving / shape validation
# ==========================================================================#


def _static_shape(value) -> list[int]:
    ty = ir.RankedTensorType(value.type)
    shape = list(ty.shape)
    assert all(d >= 0 for d in shape), f"source value has dynamic shape {shape}"
    return shape


def _to_sympy(e, symtab: dict):
    """An ``IndexExpr`` (over shape params) -> a sympy expression, registering one
    nonnegative-integer ``Symbol`` per access-param index in ``symtab``."""
    if e.kind == "const":
        return sympy.Integer(e.value)
    if e.kind == "param":
        return symtab.setdefault(
            e.param_index,
            sympy.Symbol(f"p{e.param_index}", integer=True, nonnegative=True),
        )
    if e.kind == "add":
        return _to_sympy(e.lhs, symtab) + _to_sympy(e.rhs, symtab)
    if e.kind == "mul":
        return _to_sympy(e.lhs, symtab) * _to_sympy(e.rhs, symtab)
    raise NotImplementedError(f"index expr '{e.kind}'")


def _is_affine(expr, syms) -> bool:
    """True if ``expr`` is degree <= 1 in ``syms`` (a linear shape constraint). A
    higher degree is a product of params (e.g. a collapse of two symbolic dims):
    its factorization is ambiguous, so we reject rather than guess."""
    try:
        return sympy.Poly(expr, *syms).total_degree() <= 1
    except sympy.PolynomialError:
        return False


def solve(selection: Selection) -> Selection:
    """Infer each instruction's shape params by unifying its symbolic visible shape
    with the bound source shapes — shape inference as constraint solving.

    Every operand+result dimension yields one constraint ``visible_dim == source_dim``
    (the access patterns each contribute their own dims; ``trace_instruction`` has
    already composed them). A param-free dim is checked directly (exact fit — no
    tiling); a param-bearing dim becomes a linear equation, and the per-match system
    is solved with ``linsolve``:

    - empty solution    -> the shapes are inconsistent (the instruction does not fit);
    - a free symbol left -> a param is under-constrained (a future explicit constraint
      could pin it — for now reject and name it);
    - a unique solution  -> each param must resolve to a non-negative integer.

    Nonlinear constraints (a collapse of multiple symbolic dims) are rejected up front;
    a param that is pure addressing (stride) cannot be inferred and is unsupported."""
    for m in selection.matches:
        spec = m.instruction.spec
        name = m.instruction.name
        _, arg_shapes, _ = trace_instruction(spec)
        bound = m.bound_values
        assert len(arg_shapes) == len(
            bound
        ), f"{name}: {len(arg_shapes)} access operands but {len(bound)} bound values"
        roles, _ = param_roles(spec)
        stride_params = [i for i, r in roles.items() if r == "stride"]
        assert not stride_params, (
            f"{name}: params {stride_params} appear only as strides — stride params "
            f"are not supported yet"
        )

        symtab: dict = {}
        eqs = []
        for ishape, value in zip(arg_shapes, bound):
            sshape = _static_shape(value)
            assert len(ishape) == len(
                sshape
            ), f"{name}: rank mismatch {ishape} vs {sshape}"
            for idim, sdim in zip(ishape, sshape):
                if _index_params(idim):  # depends on shape params -> an equation
                    eqs.append(sympy.Eq(_to_sympy(idim, symtab), sdim))
                else:  # statically known -> exact-fit check
                    fixed = idim if isinstance(idim, int) else idim.static_int()
                    assert fixed == sdim, (
                        f"{name}: shape mismatch — expects {fixed} but source is "
                        f"{sdim} (no tiling)"
                    )
        if not symtab:
            continue

        syms = [symtab[i] for i in sorted(symtab)]
        for eq in eqs:
            assert _is_affine(eq.lhs - eq.rhs, syms), (
                f"{name}: shape constraint is nonlinear in its params (a collapse of "
                f"multiple symbolic dims is ambiguous) — under-determined"
            )
        solutions = sympy.linsolve(eqs, syms)
        assert (
            solutions
        ), f"{name}: shapes are inconsistent — the source does not fit (no tiling)"
        (values,) = solutions  # a consistent linear system has one solution tuple
        for i, val in zip(sorted(symtab), values):
            assert not val.free_symbols, (
                f"{name}: shape param p{i} is under-constrained ({val}); no source "
                f"dimension pins it"
            )
            assert val.is_integer and val >= 0, (
                f"{name}: shape param p{i} = {val} is not a non-negative integer "
                f"(no tiling)"
            )
            m.shape_params[i] = int(val)
    return selection


# ==========================================================================#
# Stage 3 — allocation, data movement, scheduling, emission
# ==========================================================================#


@dataclass
class CompiledProgram:
    isa: object
    io_buffer: object  # the global buffer holding program I/O
    emits: list  # list[EmitRecord]
    inputs: list  # per func arg: (offset, shape)
    outputs: list  # per func result: (offset, shape, label)

    def _format(self) -> str:
        io = self.io_buffer.name
        lines = [f"CompiledProgram[{self.isa.name}]  io={io}", "  inputs:"]
        for i, (off, shape) in enumerate(self.inputs):
            lines.append(f"    arg{i} = {io}[{off}]  shape={tuple(shape)}")
        lines.append("  program:")
        for e in self.emits:
            lines.append(f"    {e.name}({', '.join(str(a) for a in e.addr)})")
        lines.append("  outputs:")
        for off, shape, label in self.outputs:
            lines.append(f"    {label} = {io}[{off}]  shape={tuple(shape)}")
        return "\n".join(lines)

    def dump(self) -> None:
        """Print the compiled instruction sequence (I/O map + emit stream)."""
        print(self._format())

    def __str__(self) -> str:
        return self._format()

    def __call__(self, *inputs):
        """Run the compiled program on the functional simulator; returns the result
        array (or a list of arrays for a multi-output program)."""
        assert len(inputs) == len(
            self.inputs
        ), f"expected {len(self.inputs)} inputs, got {len(inputs)}"
        buf = self.io_buffer
        init = np.zeros(buf.size, np.float32)
        for (offset, _shp), arr in zip(self.inputs, inputs):
            flat = np.asarray(arr, np.float32).reshape(-1)
            init[offset : offset + flat.size] = flat

        program = OracleProgram()
        program.steps.extend(("emit", e) for e in self.emits)
        for offset, shape, label in self.outputs:
            program.record_inspect(buf, slice(offset, offset + prod(shape)), label)

        results = simulate(self.isa, program, OracleConfig(init={buf: init}))
        outs = [results[label].reshape(shape) for _o, shape, label in self.outputs]
        return outs[0] if len(outs) == 1 else outs


def _shape(value) -> tuple:
    return tuple(ir.RankedTensorType(value.type).shape)


def _solve_move_params(spec, value_size: int) -> dict:
    """Shape params for a planner-inserted movement instruction.

    A move is an identity flat copy, so each of its access patterns transfers the
    moved value's ``value_size`` elements: a shape param therefore satisfies
    ``prod(visible_shape) == value_size``. This is the move analogue of Stage-2
    ``solve`` (which runs on matched compute instructions, not on moves inserted in
    Stage 3) — it replaces the old "shape param == source word count" heuristic,
    which was only correct when the param was an unscaled word count (e.g. a
    scalar↔scalar dma), not when an access scales it (e.g. ``view(d0, a, (n, 64))``,
    where ``n`` is rows and the word count is ``64·n``)."""
    _, arg_shapes, _ = trace_instruction(spec)
    roles, _ = param_roles(spec)
    shape_idxs = {i for i, r in roles.items() if r == "shape"}
    if not shape_idxs:
        return {}
    symtab, eqs = {}, []
    for ishape in arg_shapes:
        used = set()
        for d in ishape:
            used |= _index_params(d)
        if not (used & shape_idxs):
            continue
        prod_expr = sympy.Integer(1)
        for d in ishape:
            prod_expr *= (
                _to_sympy(d, symtab)
                if _index_params(d)
                else sympy.Integer(d if isinstance(d, int) else d.static_int())
            )
        eqs.append(sympy.Eq(prod_expr, value_size))
    syms = [symtab[i] for i in sorted(symtab)]
    (vals,) = sympy.linsolve(eqs, syms)
    out = {}
    for i, val in zip(sorted(symtab), vals):
        assert (
            val.is_integer and val >= 0
        ), f"{spec.name}: move shape param p{i} = {val} is not a non-negative integer"
        out[i] = int(val)
    return out


def _io_buffer(isa):
    globals_ = [b for b in isa.buffers.values() if b.is_global]
    assert (
        len(globals_) == 1
    ), f"expected exactly one global buffer, got {[b.name for b in globals_]}"
    return globals_[0]


def _movement_catalog(isa) -> dict:
    """Identity (single src -> single dst) move mnemonics, keyed by buffer pair —
    the edges of the data-movement graph used for routing and spilling."""
    moves = {}
    for spec in isa.instructions:
        if len(spec.sources) == 1 and len(spec.destinations) == 1:
            _, _, results = trace_instruction(spec)
            if len(results) == 1 and results[0].kind == "identity":
                moves[(spec.sources[0].name, spec.destinations[0].name)] = spec.name
    return moves


@dataclass(eq=False)  # identity-based: distinct residences must never compare equal
class _Loc:
    """A *location*: one value's residence in one buffer, the unit of allocation.

    A value may hold several locations over its life (e.g. a ``bram`` copy and a
    ``vreg`` copy, or — after a spill — two ``vreg`` copies split around the spill
    gap). Each occupies ``size`` contiguous slots at ``offset`` and is read at the
    steps in ``uses`` (``last_use`` = the last), at which point its slot is released;
    spilling is just ending one location and opening another."""

    value: object
    buffer: object
    size: int  # slots
    offset: int = -1
    last_use: int = -1
    uses: list = field(default_factory=list)  # step indices that read this location
    freed: bool = False


@dataclass
class _Move:
    name: str
    read: _Loc
    write: _Loc


@dataclass
class _Compute:
    name: str
    reads: list  # list[_Loc], in source-buffer order
    write: _Loc
    offset_of: dict  # access param -> buffer position
    shape_params: dict  # access param -> solved size
    n_addr: int
    in_place: bool  # the result may reuse a dying operand's slot


def _in_place_safe(instruction) -> bool:
    """Whether the instruction's result may alias one of its dying operands.

    Safe iff the compute is purely element-wise — every output element reads only
    the same-position inputs (add/sub/mul/relu/identity). A matmul, or any op that
    reads across elements, must not overwrite an operand it is still reading."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return False

    def elementwise(n) -> bool:
        if n.kind == "arg":
            return True
        if n.kind == "matmul":
            return False
        return all(elementwise(a) for a in n.args)

    return elementwise(results[0])


def _loc_size(value, buf) -> int:
    return max(prod(_shape(value)) // buf.slot_size, 1)


def _route(moves: dict, src: str, dst: str) -> list | None:
    """Shortest buffer path ``[src, ..., dst]`` over the data-movement graph (edges
    = available identity moves), or ``None`` if unreachable. BFS = fewest hops."""
    if src == dst:
        return [src]
    adj: dict = {}
    for s, d in moves:
        adj.setdefault(s, []).append(d)
    prev = {src: None}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v in prev:
                continue
            prev[v] = u
            if v == dst:
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            queue.append(v)
    return None


def plan(isa, selection: Selection) -> CompiledProgram:
    """Liveness-driven, buffer-aware allocation over *locations* (see ``_Loc``).

    1. *Schedule* — lower each match to a linear stream of moves + computes,
       inserting data movement (``bring_to``) whenever a value is not resident in
       the buffer an instruction needs it in. A move that has no direct edge is
       routed over the move graph as a chain of hops (**P-C**), each hop a short-
       lived intermediate location. Program I/O lives in the global buffer.
    2. *Liveness* — def step + last-use step (and the full use list) per location.
    3. *Allocation* — best-fit free-list per buffer, releasing a location at its
       last use so slots are reused; a result coalesces in place onto a dying
       element-wise operand. On overflow a Belady victim (resident, not used at the
       overflow step, farthest next use) is **spilled** to the backing store and
       reloaded before its next use (**P-B**); inserting the spill grows the
       schedule, so liveness + allocation are re-run to a fixpoint."""
    io = _io_buffer(isa)
    moves = _movement_catalog(isa)
    block = selection.func.regions[0].blocks[0]
    func_args = list(block.arguments)

    # --- pass 1: schedule (moves + computes) over locations ------------------
    loc: dict = {}  # value -> {buffer_name: _Loc} (residences during scheduling)
    steps: list = []

    def make_loc(value, buf) -> _Loc:
        l = _Loc(value, buf, _loc_size(value, buf))
        loc.setdefault(value, {})[buf.name] = l
        return l

    def route_move(cur: _Loc, path: list, sink: list) -> _Loc:
        """Append a move per hop along ``path`` (a list of buffer names starting at
        ``cur``'s buffer); return the final location."""
        for nxt_name in path[1:]:
            nxt = isa.buffers[nxt_name]
            dst = make_loc(cur.value, nxt)
            sink.append(_Move(moves[(cur.buffer.name, nxt_name)], cur, dst))
            cur = dst
        return cur

    def bring_to(value, target) -> _Loc:
        here = loc.get(value, {})
        if target.name in here:
            return here[target.name]
        assert here, "value has no source location to move from"
        path = min(
            (p for p in (_route(moves, s, target.name) for s in here) if p),
            key=len,
            default=None,
        )
        assert (
            path is not None
        ), f"no data-movement route from {list(here)} to '{target.name}'"
        return route_move(here[path[0]], path, steps)

    input_locs = [make_loc(a, io) for a in func_args]

    for m in selection.matches:
        spec = m.instruction.spec
        reads = [
            bring_to(_canon(v), buf) for v, buf in zip(m.operand_values, spec.sources)
        ]
        write = make_loc(_canon(m.result_value), spec.destinations[0])
        _, offset_of = param_roles(spec)
        steps.append(
            _Compute(
                m.instruction.name,
                reads,
                write,
                offset_of,
                m.shape_params,
                arity(spec.access_fn),
                _in_place_safe(m.instruction),
            )
        )

    terminator = list(block.operations)[-1]
    output_vals = list(terminator.operands)
    output_locs = [bring_to(_canon(v), io) for v in output_vals]

    # --- helpers shared by the liveness + allocation fixpoint ----------------
    def reads_of(st) -> list:
        return [st.read] if isinstance(st, _Move) else st.reads

    def all_locs() -> list:
        seen, out = set(map(id, input_locs)), list(input_locs)
        for st in steps:
            for l in reads_of(st) + [st.write]:
                if id(l) not in seen:
                    seen.add(id(l))
                    out.append(l)
        return out

    def liveness():
        final = len(steps)  # virtual step: the terminator reads the outputs
        for l in all_locs():
            l.last_use, l.uses, l.offset, l.freed = -1, [], -1, False
        for i, st in enumerate(steps):
            for r in reads_of(st):
                r.last_use = i
                r.uses.append(i)
        for l in output_locs:
            l.last_use = final
            l.uses.append(final)

    def allocate():
        """Assign offsets in one walk; on overflow return ``(victim, step)`` to
        spill, else ``None`` (offsets are final). Belady victim selection."""
        free = {name: [(0, buf.size)] for name, buf in isa.buffers.items()}
        live = {name: [] for name in isa.buffers}  # placed, not-yet-freed locations

        def release(l):
            runs = sorted(free[l.buffer.name] + [(l.offset, l.size)])
            merged = [runs[0]]
            for off, sz in runs[1:]:
                poff, psz = merged[-1]
                if poff + psz == off:
                    merged[-1] = (poff, psz + sz)
                else:
                    merged.append((off, sz))
            free[l.buffer.name] = merged
            live[l.buffer.name].remove(l)
            l.freed = True

        def best_fit(buf, size) -> int | None:
            runs = free[buf.name]
            pick = min(
                (i for i, (_o, sz) in enumerate(runs) if sz >= size),
                key=lambda i: runs[i][1],
                default=-1,
            )
            if pick < 0:
                return None
            off, sz = runs.pop(pick)
            if sz > size:
                runs.append((off + size, sz - size))
            return off

        def place(l) -> bool:
            l.offset = best_fit(l.buffer, l.size)
            if l.offset is None:
                return False
            live[l.buffer.name].append(l)
            return True

        for l in input_locs:
            assert place(l), f"backing store '{io.name}' overflow on inputs"

        for t, st in enumerate(steps):
            reads, write = reads_of(st), st.write
            reused = None
            if isinstance(st, _Compute) and st.in_place:
                reused = next(
                    (
                        r
                        for r in reads
                        if r.buffer is write.buffer
                        and r.last_use == t
                        and r.size == write.size
                        and not r.freed
                    ),
                    None,
                )
            if reused is not None:
                write.offset = reused.offset  # coalesce: hand the slot to the result
                live[write.buffer.name].remove(reused)
                live[write.buffer.name].append(write)
                reused.freed = True
            elif not place(write):
                return _pick_victim(write.buffer, t, live[write.buffer.name]), t
            for r in reads:
                if r.last_use == t and not r.freed:
                    release(r)
        return None

    def _pick_victim(buf, t, resident):
        # Spill the resident location whose *next* use is farthest away (Belady),
        # excluding anything this very step still reads.
        candidates = [l for l in resident if t not in l.uses]
        assert candidates, (
            f"buffer '{buf.name}' overflow at step {t}: no spillable location "
            f"(an instruction needs more slots than '{buf.name}' has)"
        )
        return max(candidates, key=lambda l: min(s for s in l.uses if s > t))

    def spill(victim, t):
        """Evict ``victim`` from its buffer to the backing store over [t, next-use):
        store it down before step t, reload it back before its next use, and repoint
        the later uses onto the reloaded copy. Grows ``steps`` (re-run liveness)."""
        u = min(s for s in victim.uses if s > t)
        assert victim.buffer is not io, f"backing store '{io.name}' overflow"
        down = _route(moves, victim.buffer.name, io.name)
        up = _route(moves, io.name, victim.buffer.name)
        assert (
            down and up
        ), f"cannot spill from '{victim.buffer.name}': no route to/from '{io.name}'"
        store_steps: list = []
        spilled = route_move(victim, down, store_steps)
        reload_steps: list = []
        reloaded = route_move(spilled, up, reload_steps)
        for i, st in enumerate(steps):
            if i >= u:
                if isinstance(st, _Move):
                    if st.read is victim:
                        st.read = reloaded
                else:
                    st.reads = [reloaded if r is victim else r for r in st.reads]
        steps[u:u] = reload_steps  # reload before the next use ...
        steps[t:t] = store_steps  # ... and store before the overflow step (t < u)

    # An instruction's operands are all live at once and none can be spilled (they
    # are in use), so a buffer that cannot even hold one instruction's operands is
    # infeasible — reject upfront. This also guarantees the spill loop terminates:
    # every remaining overflow then has a non-operand victim to evict, so each spill
    # resolves the earliest overflow and pushes the frontier strictly later.
    for st in steps:
        if isinstance(st, _Compute):
            need: dict = {}
            for r in dict.fromkeys(st.reads):  # distinct locations (handles a*a)
                need[r.buffer.name] = need.get(r.buffer.name, 0) + r.size
            for name, n in need.items():
                assert n <= isa.buffers[name].size, (
                    f"{st.name}: operands need {n} '{name}' slot(s) but '{name}' has "
                    f"only {isa.buffers[name].size} (capacity too small to spill into)"
                )

    # --- passes 2+3: liveness + allocation, iterated to a no-spill fixpoint ---
    while True:
        liveness()
        outcome = allocate()
        if outcome is None:
            break
        spill(*outcome)

    # --- emit with concrete offsets ------------------------------------------
    emits: list[EmitRecord] = []
    for st in steps:
        if isinstance(st, _Move):
            # A move fills its access params like a compute: offset params take the
            # source/destination offsets; shape params are solved from the moved
            # value's element count (prod(visible) == value size), since the move was
            # inserted in Stage 3 and never went through Stage-2 solve.
            spec = isa._ops[st.name].spec
            _, offset_buffer = param_roles(spec)
            shape_params = _solve_move_params(spec, prod(_shape(st.read.value)))
            buf_offsets = [st.read.offset, st.write.offset]
            addr = [
                buf_offsets[offset_buffer[i]] if i in offset_buffer else shape_params[i]
                for i in range(arity(spec.access_fn))
            ]
            emits.append(EmitRecord(st.name, addr, []))
        else:
            buf_offsets = [r.offset for r in st.reads] + [st.write.offset]
            addr = [
                (
                    buf_offsets[st.offset_of[i]]
                    if i in st.offset_of
                    else st.shape_params[i]
                )
                for i in range(st.n_addr)
            ]
            emits.append(EmitRecord(st.name, addr, []))

    inputs = [(l.offset, _shape(l.value)) for l in input_locs]
    outputs = [
        (l.offset, _shape(v), f"out{i}")
        for i, (l, v) in enumerate(zip(output_locs, output_vals))
    ]
    return CompiledProgram(isa, io, emits, inputs, outputs)


# ==========================================================================#
# Driver
# ==========================================================================#


def compile_program(source: str, isa) -> CompiledProgram:
    """Compile a source program onto ``isa``.

    The source program is a TOSA-dialect MLIR module given as *text* — we generate
    none ourselves; the caller hands us a module string (e.g. from torch_mlir's
    TOSA backend) and we ``Module.parse`` it here. The returned ``CompiledProgram``
    holds only plain data (no IR handles), so the parse context can be dropped."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        normalize_source(module)
        catalog = Catalog(isa)
        selection = match_program(catalog, module)
        solve(selection)
        return plan(isa, selection)

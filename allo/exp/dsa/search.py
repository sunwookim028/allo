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
- ``solve_layouts`` (Stage 2b) — infer the access params that describe *residence*
  (strides, a ``layout``'s dimension ordering) by unifying the index maps of every
  access of one value; program I/O is pinned to the host ABI.
- ``plan`` (Stage 3) — liveness-driven slot allocation + data movement (routing
  and spilling), producing a ``CompiledProgram`` (a placed program + I/O map).

Denotationally (``drafts/schedule-isa-summary.md`` v2 §3): each emitted
instruction is one **epoch** — a total configuration for the machine's kernel —
and the emit stream is the program's ∘-composition; ``epoch.py`` materializes
that reading (``CompiledProgram.epochs``). Stage 2b is then the inter-epoch
interface condition, and the movers Stage 3 inserts are the delta encoding of
the time-varying λ.

The public entry is ``ISA.compile_program(source)`` (sugar over ``compile_program``
here); the returned ``CompiledProgram`` is callable — ``prog(*inputs)`` runs it on
the functional simulator (the same oracle backbone hand-written assembly uses) — and
``prog.dump()`` prints the emitted instruction sequence.

See ``todos/search.md`` for the full per-stage algorithm analysis.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import inf, prod

import ml_dtypes
import numpy as np
import sympy

from ..._mlir import ir
from ..._mlir.dialects import tosa
from . import primitive
from .core import (
    ISA,
    Instruction,
    Ref,
    ScalarProxy,
    Tile,
    _index_params,
    access_map,
    access_names,
    arity,
    layout_params,
    buffer_weights,
    compute_params,
    dense_strides,
    is_mover,
    mover_domains,
    param_roles,
    pin_access,
    residence,
    show_map,
    trace_instruction,
)
from .errors import (
    AcceleratorDescriptionError,
    AllocationError,
    AssemblyError,
    CompileError,
    DTypeError,
    LayoutError,
    NoMatchError,
    ShapeError,
)
from .oracle import EmitRecord, OracleConfig, OracleProgram, _np_dtype, simulate

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


def _reads_index_ge(node, n_src: int) -> bool:
    """True if the compute DAG reads a buffer arg at index >= ``n_src`` (a
    destination). Such an instruction reads its own output (an in-place accumulate)."""
    if node.kind == "arg":
        return node.buffer_index >= n_src
    return any(_reads_index_ge(a, n_src) for a in node.args)


def instruction_pattern(instruction: Instruction):
    """The compute pattern of an instruction: the root ``TensorProxy`` of its
    semantics DAG. Internal nodes are prim ops; ``arg`` leaves bind to the
    instruction's source buffers (by ``buffer_index``). A 1:1 instruction is just
    a depth-1 pattern. Returns ``None`` for data-movement (identity), multi-output
    instructions (not matched yet), or an instruction whose compute reads its own
    destination buffer (an in-place accumulate): the matcher cannot bind a dst read,
    so it is oracle-only — to compile an accumulate, model the accumulator as a
    source operand (``dst = add(c_in, matmul(a, b))``), which the allocator then
    coalesces back onto one slot."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return None
    root = results[0]
    if root.kind in ("identity", "arg"):
        return None
    if _reads_index_ge(root, len(instruction.spec.sources)):
        return None  # reads its own destination -> not selectable (would mis-bind)
    return root


# Recognized source ops carrying trailing *non-data* operands: the count of leading
# data operands. Everything after them — tosa.mul's shift, the matmul / conv /
# negate / avg_pool zero-points — is quantization, which this frontend does not
# model. They are therefore not dropped but *required to be neutral*, else a `>>3`
# fixed-point multiply would select a plain float `mul` and run.
_DATA_OPERANDS = {
    "tosa.mul": 2,  # + shift
    "tosa.matmul": 2,  # + a_zp, b_zp
    "tosa.negate": 1,  # + input_zp, output_zp
    "tosa.conv2d": 3,  # input, weight, bias (+ input_zp, weight_zp)
    "tosa.depthwise_conv2d": 3,
    "tosa.avg_pool2d": 1,  # + input_zp, output_zp
}

# The largest finite value of each float type. torch_mlir's TOSA backend spells
# relu's *open* upper bound as this value rather than +inf, so both count as "does
# not clip from above".
_FLOAT_MAX = {
    "f16": float(np.finfo(np.float16).max),
    "bf16": float(ml_dtypes.finfo(ml_dtypes.bfloat16).max),
    "f32": float(np.finfo(np.float32).max),
    "f64": float(np.finfo(np.float64).max),
}


def _source_ins(op) -> list:
    """The data-input operands of a recognized (value-semantics TOSA) source op."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return list(op.operands)[:n] if n is not None else list(op.operands)


def const_elements(value) -> list | None:
    """The elements of a source ``tosa.const``, or ``None`` if ``value`` is not a
    constant we can read.

    ``None`` covers both "not a constant" and "a constant whose data lives in a
    ``dialect_resource`` blob" — torch's TOSA backend stores model weights that way
    and the Python bindings expose no reader. Callers must treat unknown as
    *unusable*, never as a default value."""
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name != "tosa.const":
        return None
    attr = owner.operation.attributes["values"]
    if not isinstance(attr, (ir.DenseFPElementsAttr, ir.DenseIntElementsAttr)):
        return None
    return list(attr)


def _is_zero_const(value) -> bool:
    """True only if ``value`` is *provably* an all-zero constant."""
    elems = const_elements(value)
    return elems is not None and all(v == 0 for v in elems)


def _quantization_is_neutral(op) -> bool:
    """Whether ``op``'s trailing shift / zero-point operands are all zero."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return n is None or all(_is_zero_const(v) for v in list(op.operands)[n:])


def source_tag(op) -> str | None:
    """Recognize a source op into a prim tag, or None if unsupported.

    Recognition is **fail-safe**: an op earns a tag only when every part of its
    definition is accounted for by a prim. Anything unmodeled — a non-zero shift or
    zero-point, a clamp that is not relu — yields ``None``, which costs a clean
    "no instruction matches" rather than a program that compiles and computes a
    different function."""
    name = op.operation.name
    tag = _NAMED_TAG.get(name)
    if tag is None:
        if name != "tosa.clamp" or not _is_relu_clamp(op):
            return None
        tag = "relu"
    return tag if _quantization_is_neutral(op) else None


def _is_relu_clamp(op) -> bool:
    """``tosa.clamp`` is relu only when it clamps to ``[0, +inf)`` over a float
    type — the form torch_mlir's TOSA backend emits for aten.relu.

    *Both* bounds are semantics. Checking only the lower one recognized relu6
    (``[0, 6]``) as relu, which compiled and ran and returned wrong numbers. An
    integer clamp carries ``IntegerAttr`` bounds — a different attribute type, and
    no unbounded value — so it is never this pattern."""
    elt = ir.ShapedType(op.operands[0].type).element_type
    if str(elt) not in _FLOAT_MAX:
        return False
    attrs = op.operation.attributes
    return (
        ir.FloatAttr(attrs["min_val"]).value == 0.0
        and ir.FloatAttr(attrs["max_val"]).value >= _FLOAT_MAX[str(elt)]
    )


def pattern_alpha(node) -> set:
    """The computational-attribute (α) indices a pattern's ``const`` leaves read."""
    if node.kind == "const":
        return (
            {node.value.param_index} if isinstance(node.value, ScalarProxy) else set()
        )
    out: set = set()
    for a in node.args:
        out |= pattern_alpha(a)
    return out


class Catalog:
    """Indexes an ISA's compute instructions by the *root* prim tag of their
    pattern, so single-op and multi-node instructions are looked up uniformly."""

    def __init__(self, isa: ISA):
        self.isa = isa
        self.patterns: dict[str, list[tuple[Instruction, object]]] = {}
        for spec in isa.instructions:
            instr = isa._ops[spec.name]
            root = instruction_pattern(instr)
            if root is None:
                continue  # oracle-only: its α (if any) is supplied by hand
            # Every α a *selectable* instruction declares has to be bound from the
            # source, and the only thing that binds one is a const leaf. One that
            # appears nowhere in the pattern has no value the compiler could give it.
            missing = set(range(len(compute_params(spec)))) - pattern_alpha(root)
            if missing:
                names = [compute_params(spec)[i] for i in sorted(missing)]
                raise AcceleratorDescriptionError(
                    f"{spec.name}: compute param(s) {names} never appear in the "
                    f"semantics (a compute param reaches the compute DAG through "
                    f"primitive.const), so nothing binds them at a match"
                )
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
    # Solved access param -> int. ``None`` until Stage 1 solves it, and left ``None``
    # for a candidate whose shapes do NOT fit, which is what Stage 2 re-solves to get
    # the diagnostic. Stage 2b then adds the residence params to the same dict.
    shape_params: dict | None = None
    alpha: dict = field(default_factory=dict)  # compute param -> bound immediate
    # Schedule param -> chosen value: the configuration Stage 1 picked for this site
    # (``InstructionSpec.configure``). Empty unless the ISA declared @I.schedule.
    schedule: dict = field(default_factory=dict)
    # ``(mapping.Mapping, mapping.Binding)`` when this site is lowered by an imported
    # mapping rather than performed by one instruction. Such a site is a **tile run**:
    # its bound values are the source op's own tensors (not the instruction's
    # operands), its instruction is the one the binding names, and its shapes come
    # from the mapping's innermost factors — so neither Stage 2 nor Stage 2b applies
    # to it and ``plan`` lowers it through ``mapping.lower_site``.
    mapping: tuple | None = None

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
    raise CompileError("source module has no func.func")


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


def _match_const(pnode, value, alpha: dict) -> bool:
    """A ``const`` pattern leaf against a source constant — compared, or bound.

    A **fixed** literal is part of the semantics — ``pow(2, x)`` and ``pow(3, x)`` are
    different functions — so this is the constant's counterpart of a transpose's
    permutation check. Compared after rounding through the pattern's dtype, since
    the source holds an f32 while the ISA was written with a Python float.

    A **parametric** one (a ``ScalarProxy``: ACT's α) is instead *bound* into ``alpha``:
    the ISA states that this operand is an immediate the instruction word carries, and
    the match reads its value off the source. This is the only place a source value
    flows into an instruction's encoding rather than into memory."""
    elems = const_elements(_canon(value))
    if elems is None:
        return False
    if isinstance(pnode.value, ScalarProxy):
        return _bind_alpha(pnode.value, elems, alpha)
    cast = _np_dtype(pnode.dtype).type
    want = cast(pnode.value)
    return all(cast(v) == want for v in elems)


def _bind_alpha(param: ScalarProxy, elems: list, alpha: dict) -> bool:
    """Bind one α from a source constant's elements, or reject.

    An immediate field holds **one integer**, so the constant has to be a splat whose
    value is an exact integer: ``x + 3.0`` binds ``#k = 3``, ``x + 0.5`` does not match
    at all (rounding it would silently compile a different function), and a
    non-uniform constant is program data, not an immediate. A param appearing twice in
    one pattern must bind the same value both times."""
    first = elems[0]
    if any(v != first for v in elems) or int(first) != first:
        return False
    return alpha.setdefault(param.param_index, int(first)) == int(first)


def _operand_orders(pnode, ins):
    """Operand orderings to try; both orders for a commutative binary prim."""
    if pnode.kind in _COMMUTATIVE and len(ins) == 2:
        return (ins, [ins[1], ins[0]])
    return (ins,)


def _match_pattern(pnode, value, def_op, alpha, bindings, within, interior) -> bool:
    """Match pattern node ``pnode`` against source ``value``.

    ``arg`` leaves bind a source buffer to ``value`` and ``const`` leaves either check
    or bind a literal (``alpha``); internal nodes must align with a recognized source
    op of the same prim tag and recurse on its inputs.
    Records each folded source value (whose defining op is absorbed into the tile)
    in ``interior``, and the within-tile use count of each operand in ``within``, so
    the caller can reject a fold in which a folded non-root value *escapes* (is also
    used outside the tile and therefore must be materialized). This deferred
    cut-point test permits internal fan-out — e.g. softmax's ``exp`` feeding both the
    reduce and the divide — which a per-node single-use test would wrongly forbid.
    Mutates ``alpha``/``bindings``/``within``/``interior`` in place, rolling back on a
    failed branch.
    """
    if pnode.kind == "arg":
        prev = bindings.get(pnode.buffer_index)
        if prev is not None and prev != value:
            return False
        bindings[pnode.buffer_index] = value
        return True
    if pnode.kind == "const":
        return _match_const(pnode, value, alpha)
    # Recognize an internal (folded) op through any reshape wrappers (torch emits
    # 2-D<->3-D reshapes around matmul); arg leaves above stay raw so their shapes
    # still drive the shape solver.
    op = def_op.get(_canon(value))
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
        saved = dict(bindings), dict(alpha), dict(within), set(interior)
        for sv in order:  # each operand is one within-tile use of that value
            within[_canon(sv)] = within.get(_canon(sv), 0) + 1
        if all(
            _match_pattern(pa, sv, def_op, alpha, bindings, within, interior)
            for pa, sv in zip(pnode.args, order)
        ):
            interior.add(_canon(value))  # canonical key (matches use/within counting)
            return True
        for live, restore in zip((bindings, alpha, within, interior), saved):
            live.clear()
            live.update(restore)
    return False


@dataclass
class _Choice:
    cost: float
    instruction: object
    operands: list
    shape_params: dict | None  # solved sizes, or None if the shapes do not fit
    alpha: dict  # compute params bound from the source's constants
    schedule: dict = field(default_factory=dict)  # chosen schedule params
    mapping: tuple | None = None  # an imported lowering for this site


def _pattern_has(node, kind) -> bool:
    return node.kind == kind or any(_pattern_has(a, kind) for a in node.args)


def _describe_pattern(node) -> str:
    """A compact source-level rendering of an instruction's compute pattern, e.g.
    ``matmul(%0, transpose(%1))`` — the shape of source DAG it matches. A ``const``
    leaf shows its literal, or ``#name`` when it is a compute param (α)."""
    if node.kind == "arg":
        return f"%{node.buffer_index}"
    if node.kind == "const":
        return f"{node.value}"  # ScalarProxy renders as `#name`
    return f"{node.kind}({', '.join(_describe_pattern(a) for a in node.args)})"


def _no_match_error(op, catalog) -> str:
    """An actionable message for an unmatched source op: show its operand shapes,
    the candidate instructions' patterns, and — the common case — a hint when an
    instruction consumes an operand transposed but the source provides it plain."""
    tag = source_tag(op)
    shapes = [tuple(ir.RankedTensorType(o.type).shape) for o in _source_ins(op)]
    head = f"no instruction matches source op '{op.operation.name}' with operand shapes {shapes}"
    if not _quantization_is_neutral(op):
        return (
            f"{head}: it carries a non-zero shift / zero-point operand, and this "
            f"frontend models no quantization — so the op is not recognized at all "
            f"rather than matched as its unquantized namesake."
        )
    if tag is None:
        return f"{head}: no prim in the compute vocabulary models this op."
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


def match_program(catalog: Catalog, source_module, mapping_for=None) -> Selection:
    """Cover the source compute DAG with instruction patterns via cost-aware
    tree-DP. A value used more than once is a forced cut point (it cannot be
    folded into a consumer's tile), so the foldable subgraphs are trees and a
    per-value DP is globally optimal.

    ``materialize(v)`` returns the cheapest tile rooted at ``v``: instruction cost
    plus the materialization cost of its operands — but only *single-use* operands
    are charged, because a shared (multi-use) operand is materialized once as its
    own root and must not be billed to every consumer. The optimum is reconstructed
    from the returned values and scheduled in def-before-use order.

    ``mapping_for(op)`` — when given — returns ``(Mapping, Binding)`` for a source op
    that an external mapper has already tiled, or ``None``. A mapped op is not
    matched: the binding names the instruction and the mapping says what run of it
    the site becomes, so the site is decided before the DP sees it. It is consulted
    once per recognized op, because a driver of it may be a whole mapper run."""
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

    # Asked once per recognized op, before any matching: a driver may run a whole
    # external mapper, and `materialize` is not the place to do that repeatedly.
    mapped: dict = {}  # the op's result value -> (Mapping, Binding)
    for value, op in def_op.items():
        got = mapping_for(op) if mapping_for is not None else None
        if got is not None:
            mapped[value] = got

    memo: dict = {}  # canonical value -> _Choice (optimal tile to materialize it)

    def materialize(v) -> _Choice:
        if v in memo:
            return memo[v]
        op = def_op[v]
        if v in mapped:
            # The site is already decided, so there is nothing to choose and nothing
            # to fit: the binding names the instruction and the mapping says which
            # run of it this op becomes. The cost is the operands' — the run's own
            # price is not a DP quantity here, since a mapped site has no alternative
            # to be weighed against and a consumer pays the same for every candidate.
            _m, binding = mapped[v]
            operands = _source_ins(op)
            memo[v] = _Choice(
                sum(
                    materialize(_canon(ov)).cost
                    for ov in operands
                    if _canon(ov) in def_op and use.get(_canon(ov), 0) == 1
                ),
                catalog.isa._ops[binding.compute],
                operands,
                {},
                {},
                {},
                mapped[v],
            )
            return memo[v]
        fitting = None  # cheapest candidate that also *fits* the source shapes
        fallback = None  # first structural match that does not fit (error reporting)
        unconfigurable: list = []  # fitting matches with no legal @schedule assignment
        for instr, root in catalog.candidates(source_tag(op)):
            bindings, alpha, within, interior = {}, {}, {}, set()
            if not _match_pattern(root, v, def_op, alpha, bindings, within, interior):
                continue
            # Deferred cut-point test: a folded (non-root) value must be used only
            # within this tile; if its global use count exceeds its within-tile use
            # count it escapes and must be its own root, so this fold is invalid. A
            # *mapped* value is a forced cut point for the same reason a shared one
            # is — folding it away would discard the lowering that was imported for
            # it — so it never becomes another tile's interior.
            if any(
                use.get(iv, 0) != within.get(iv, 0) or iv in mapped
                for iv in interior
                if iv != v
            ):
                continue
            n_src = len(instr.spec.sources)
            if not all(i in bindings for i in range(n_src)):
                continue
            operands = [bindings[i] for i in range(n_src)]
            # An ISA may offer both a fixed-size tile instruction and a layer-level
            # (parametric, @expand-ing) one for the same prim. They match the same
            # structure, so structure alone cannot choose: solve here and prefer a
            # candidate that actually *fits* the source shapes. The solved params are
            # carried on (Match.shape_params) so no later stage re-solves them; keeping
            # one unfitting candidate lets Stage 2 report *why* nothing fits when that
            # is the real error.
            fit = _fit(instr, operands, v)
            if fit is None:
                if fallback is None:
                    fallback = _Choice(0.0, instr, operands, None, alpha)
                continue
            # Configure before costing: an instruction with schedule params is only a
            # candidate here if some legal assignment of them exists, and its price is
            # the price of the cheapest one (`InstructionSpec.configure`).
            config = instr.spec.configure(fit)
            if config is None:
                unconfigurable.append(instr.name)
                continue
            chosen, own_cost = config
            cost = own_cost + sum(
                materialize(_canon(ov)).cost
                for ov in operands
                if _canon(ov) in def_op and use.get(_canon(ov), 0) == 1
            )
            # Strictly cheaper wins, so equal-cost candidates resolve to the
            # earlier-declared one — deterministic, and the right default when a
            # parametric op degenerates to exactly the fixed one it would expand into.
            if fitting is None or cost < fitting.cost:
                fitting = _Choice(cost, instr, operands, fit, alpha, chosen)
        best = fitting or fallback
        if best is None:
            if unconfigurable:
                raise NoMatchError(
                    f"{source_tag(op)}: {sorted(set(unconfigurable))} match and fit, "
                    f"but no legal configuration of their @schedule params exists for "
                    f"this site — the hardware cannot be configured to run it"
                )
            raise NoMatchError(_no_match_error(op, catalog))
        memo[v] = best
        return best

    matches: list[Match] = []
    visited: set = set()

    def schedule(v):
        if v in visited:
            return
        visited.add(v)
        ch = materialize(v)
        matches.append(
            Match(
                ch.instruction,
                ch.operands,
                v,
                ch.shape_params,
                ch.alpha,
                ch.schedule,
                ch.mapping,
            )
        )
        for ov in ch.operands:
            if _canon(ov) in def_op:
                schedule(_canon(ov))

    for v in terminator.operands:
        cv = _canon(v)
        if cv in def_op:
            schedule(cv)
        elif not isinstance(cv.owner, ir.Block):
            # A returned value the recognizer refused outright: name that op and
            # why, rather than reporting the whole program as unmatched.
            raise NoMatchError(_no_match_error(cv.owner, catalog))

    if not matches:
        raise NoMatchError("no source compute ops matched any instruction")
    matches.sort(key=lambda m: index[m.result_value])
    return Selection(func, matches)


# ==========================================================================#
# Stage 2 — parameter solving / shape validation
# ==========================================================================#


def _shape(value) -> tuple:
    # A `Tile` is a value the lowering invented (an expansion's staging), so it
    # carries its shape directly rather than through an IR type.
    if isinstance(value, Tile):
        return value.shape
    return tuple(ir.RankedTensorType(value.type).shape)


def _static_shape(value) -> list[int]:
    shape = list(_shape(value))
    if any(d < 0 for d in shape):
        raise ShapeError(f"source value has dynamic shape {shape}")
    return shape


def _strip_leading_units(shape) -> list:
    """Drop leading statically-1 dims. An element is an ``int`` (a source shape) or
    an ``IndexExpr`` (an instruction's visible shape); a symbolic dim is never
    dropped, since its value is unknown."""

    def unit(dim) -> bool:
        if isinstance(dim, int):
            return dim == 1
        return not _index_params(dim) and dim.static_int() == 1

    i = 0
    while i < len(shape) and unit(shape[i]):
        i += 1
    return list(shape[i:])


def _check_dtype(name: str, buf, value) -> None:
    """Unify one instruction buffer's element type with the bound source value's.

    Structural matching is type-blind, and running an ``i32`` program on an ``f32``
    datapath is a *different function* (``add`` wraps, ``intdiv`` truncates), so the
    element types have to be reconciled here.

    The one deliberate relaxation is **float-to-float**. Every float op is already
    approximate, and running it narrower changes the rounding error, not the
    operation — which is exactly what reduced-precision hardware is for (QKV's bf16
    datapath runs an f32-typed source graph, and its examples diff against a bf16
    tolerance). Integer width is not like that: ``i32`` on an ``i8`` datapath
    wraps around, so there the types must be equal."""
    elt = ir.RankedTensorType(value.type).element_type
    dtype = buf.kind.dtype
    if dtype.is_float() and str(elt) in _FLOAT_MAX:
        return
    if dtype.materialize(elt.context) != elt:
        raise DTypeError(
            f"{name}: buffer '{buf.name}' is {dtype} but the source value is {elt}"
        )


def _align_ranks(ishape, sshape) -> tuple[list, list]:
    """Align an instruction's visible shape with a bound source shape *modulo
    leading unit (batch) dims* — the shape-solver counterpart of ``_canon``.

    A leading ``1`` does not change the linear value sequence, so it is a rank alias
    carrying no shape information: ``[1, 4, 4]`` and ``[4, 4]`` describe the same 16
    values. torch_mlir makes this unavoidable — it brackets every 2-D ``a @ b`` in
    reshapes to batched 3-D and back, so within one chain the matmuls are 3-D while
    the elementwise ops around them are 2-D, and an instruction written at either
    rank meets the other (FeatherX's 3-D ``mac`` accumulates a 2-D partial sum;
    QKV's 2-D ``softmax`` consumes a 3-D matmul). Stripping only when the ranks
    actually differ leaves every same-rank comparison an exact-fit check."""
    if len(ishape) == len(sshape):
        return list(ishape), list(sshape)
    return _strip_leading_units(ishape), _strip_leading_units(sshape)


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


def _solve_match(m: Match) -> None:
    """Solve one match's shape params in place (see ``solve`` for the method); raises
    ``ShapeError`` if the instruction does not fit the bound source shapes."""
    spec = m.instruction.spec
    name = m.instruction.name
    _, arg_shapes, _ = trace_instruction(spec)
    bound = m.bound_values
    # By construction: `trace_instruction` yields one pattern per src+dst buffer and
    # the matcher binds exactly those. A genuine invariant, so it stays an assert.
    assert len(arg_shapes) == len(
        bound
    ), f"{name}: {len(arg_shapes)} access operands but {len(bound)} bound values"
    m.shape_params = {}
    symtab: dict = {}
    eqs = []
    for buf, ishape, value in zip(spec.buffers, arg_shapes, bound):
        _check_dtype(name, buf, value)
        ishape, sshape = _align_ranks(ishape, _static_shape(value))
        if len(ishape) != len(sshape):
            raise ShapeError(f"{name}: rank mismatch {ishape} vs {sshape}")
        for idim, sdim in zip(ishape, sshape):
            if _index_params(idim):  # depends on shape params -> an equation
                eqs.append(sympy.Eq(_to_sympy(idim, symtab), sdim))
            else:  # statically known -> exact-fit check
                fixed = idim if isinstance(idim, int) else idim.static_int()
                if fixed != sdim:
                    raise ShapeError(
                        f"{name}: shape mismatch — expects {fixed} but source is "
                        f"{sdim} (no tiling)"
                    )
    if not symtab:
        return

    syms = [symtab[i] for i in sorted(symtab)]
    for eq in eqs:
        if not _is_affine(eq.lhs - eq.rhs, syms):
            raise ShapeError(
                f"{name}: shape constraint is nonlinear in its params (a collapse of "
                f"multiple symbolic dims is ambiguous) — under-determined"
            )
    solutions = sympy.linsolve(eqs, syms)
    if not solutions:
        raise ShapeError(
            f"{name}: shapes are inconsistent — the source does not fit (no tiling)"
        )
    (values,) = solutions  # a consistent linear system has one solution tuple
    for i, val in zip(sorted(symtab), values):
        if val.free_symbols:
            raise ShapeError(
                f"{name}: shape param p{i} is under-constrained ({val}); no source "
                f"dimension pins it"
            )
        if not (val.is_integer and val >= 0):
            raise ShapeError(
                f"{name}: shape param p{i} = {val} is not a non-negative integer "
                f"(no tiling)"
            )
        m.shape_params[i] = int(val)


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

    Nonlinear constraints (a collapse of multiple symbolic dims) are rejected up front.
    Params that describe *residence* rather than shape — a stride, a dimension
    ordering — leave no trace in a visible shape at all, so they are not solvable here;
    ``solve_layouts`` (Stage 2b) pins them from the maps instead.

    Stage 1 already solved every *fitting* candidate (it had to, to choose among them
    and to cost a parametric instruction), so this stage only has work to do for a
    match that is known NOT to fit — precisely the case where ``_solve_match``'s
    message is the diagnostic the user needs."""
    for m in selection.matches:
        if m.shape_params is None:
            _solve_match(m)
    return selection


# ==========================================================================#
# Stage 2b — layout solving (the access params that describe residence)
# ==========================================================================#


def _dense_map(shape, buf) -> tuple:
    """The dense (row-major) residence of a value of ``shape`` in ``buf``: a flat pool
    packs it with suffix-product strides, a multi-dimensional array gives it the
    array's own pitch.

    In the I/O buffer this is the **host ABI**, the one map the compiler does not get
    to choose: ``CompiledProgram.__call__`` writes an input into the region the
    allocator gave it, densely, and reads an output back the same way. Elsewhere it is
    the default a free group falls back to."""
    if buf.address_rank == 1:
        return residence(list(zip(shape, dense_strides(shape))))
    weight = buffer_weights(buf)
    dims = _placement_dims(shape, buf)
    return residence([(d, weight[k]) for k, d in enumerate(dims)])


def _site_map(m: Match, pattern) -> tuple | None:
    """One access's residence, or ``None`` while a param of it is unsolved."""
    mapping = access_map(pattern, m.shape_params)
    if any(stride is None for _size, stride in mapping):
        return None
    return residence(mapping)


def _group_residence(isa, moves, key, group) -> tuple | None:
    """The residence one ``(value, buffer)`` group adopts, or ``None`` while every
    access of it is still parametric.

    A value has one residence in one buffer, so this is a decision *about the group*
    rather than about any one access — the first thing in this frontend that is. Two
    cases, and only the second is a choice:

    - **The access that writes the value is concrete.** Then the value is packed where
      its producer packs it, and that is the residence. Adopting anything else could
      only add repacks, never remove one: a reader wanting some other map needs the
      relayout either way, and readers wanting the same one share it.
    - **Otherwise the compiler is choosing the packing**, and it takes the concrete map
      its readers describe that leaves the cheapest repacks between them — priced on the
      movement graph, the same one ``plan`` will route over. Ties keep source order, so
      a machine that prices no movement is left with the first-concrete-wins rule this
      generalizes.

    Costing distinct maps (rather than accesses) is what makes it agree with what
    ``plan`` will actually do: two readers wanting the same packing are served by one
    relayout, because routing reuses the state it repacked into."""
    maps = [(_site_map(m, p), write) for m, p, write in group]
    writer = next((r for r, write in maps if write and r is not None), None)
    if writer is not None:
        return writer
    wanted = list(dict.fromkeys(r for r, _write in maps if r is not None))
    if len(wanted) < 2:
        return wanted[0] if wanted else None
    buf = key[1]
    edges = _move_edges(isa, moves, prod(_static_shape(key[0])))

    def repacking(target) -> float:
        _order, _prev, dist = _explore(edges, [(buf, target)])
        return sum(dist.get((buf, m), inf) for m in wanted if m != target)

    return min(wanted, key=repacking)


def solve_layouts(isa, selection: Selection) -> Selection:
    """Stage 2b — solve the access params that describe **residence**: strides, and
    the dimension ordering of a ``layout``.

    Neither shows up in a visible shape, so Stage 2 cannot see them. What pins them is
    the residence its neighbours describe: accesses are grouped per ``(value, buffer)``
    and a parametric one adopts the map its group settles on, which is a unification of
    index maps on the SSA edge rather than a vote among enum labels — the whole
    difference between solving an ordering and picking one. Program I/O and the constant
    pool seed their groups with the host ABI.

    What the unification *is*, denotationally: the **inter-epoch interface
    condition** (``epoch.py``). Epochs compose with ∘ only if a value one epoch
    leaves behind is found by the next under the same map — "a value has one
    residence, and every access of it must describe the same map" is that condition
    stated per value. Where two epochs cannot agree on a map, the condition is met
    by ``plan`` inserting a mover between them: an explicit λ-update, not an
    exception to the rule.

    The group, not the access, is the unit of decision (``_group_residence``): where the
    producer's own access is concrete it dictates the packing, and where it is not the
    compiler is genuinely choosing one, so it takes the reader map that leaves the
    cheapest repacks. This is the one place a decision here ranges over more than one
    instruction, and the only place this stage consults ``plan``'s movement graph.

    This stage **solves; it does not check.** Two accesses of one value may still
    disagree afterwards, and whether that is compilable depends on the machine having a
    mover that repacks between them — which only ``plan`` knows, so ``plan`` decides
    (and inserts the relayout).

    A group with no concrete map at all is free, and takes the dense row-major packing
    — the host's — because with nothing to repack towards there is nothing to price.

    None of this reaches a **mover**: the planner is what inserts one, so it takes part
    in no unification. A mover's own residence params are chosen instead, by the router,
    one assignment per movement-graph edge — see ``_order_assignments``.

    Nor a **mapped site**: its accesses are tile accesses inside a run, described by
    the mapping rather than by a bound source value, and the residence its level-0
    tiles have is whatever this stage settled for the surrounding program — which is
    the sense in which a mapping enters at an SSA edge like any other lowering."""
    io = _io_buffer(isa)
    block = selection.func.regions[0].blocks[0]

    moves = _movement_catalog(isa)
    sites: dict = {}  # (value, buffer name) -> [(match, pattern, writes?)]
    for m in selection.matches:
        if m.mapping is not None:
            continue  # a mapped site's accesses are the mapping's, not this stage's
        spec = m.instruction.spec
        patterns, _, _ = trace_instruction(spec)
        for i, (buf, pattern, value) in enumerate(
            zip(spec.buffers, patterns, m.bound_values)
        ):
            site = (m, pattern, i >= len(spec.sources))
            sites.setdefault((_canon(value), buf.name), []).append(site)

    # Host-supplied data: the arguments, the results, and the constant pool — all of
    # them written into (or read out of) the I/O buffer densely before/after the run,
    # so their residence there is the ABI's rather than the compiler's. ACT Def 3.8
    # puts inputs and constants in one ASM for exactly this reason.
    host = list(block.arguments) + list(list(block.operations)[-1].operands)
    host += [
        v
        for m in selection.matches
        for v in m.operand_values
        if _const_array(v) is not None
    ]
    pinned: dict = {}  # (value, buffer name) -> the residence map it must have
    for value in host:
        key = (_canon(value), io.name)
        if key in sites:
            pinned[key] = _dense_map(_static_shape(_canon(value)), io)

    def propagate() -> bool:
        moved = False
        for key, group in sites.items():
            target = pinned.get(key)
            if target is None:
                # The group picks its own residence (`_group_residence`); the host ABI
                # already pinned it above when the value is program data. Accesses that
                # then *disagree* are not an error here: whether the machine can repack
                # between them is the move graph's business, so `plan` decides (and
                # inserts the relayout).
                target = _group_residence(isa, moves, key, group)
                if target is None:
                    continue
                pinned[key] = target
                moved = True
            _value, name = key
            for m, pattern, _write in group:
                if _site_map(m, pattern) is not None:
                    continue
                who = f"{m.instruction.name} on '{name}'"
                # Pinning resolves a whole pattern at once, so an access sharing a
                # residence param with one already pinned is no longer unsolved and
                # is skipped above — nothing is pinned twice.
                for i, val in pin_access(pattern, m.shape_params, target, who).items():
                    assert i not in m.shape_params, f"{who}: p{i} pinned twice"
                    m.shape_params[i] = val
                    moved = True
        return moved

    while True:
        while propagate():
            pass
        free = next((k for k in sites if k not in pinned), None)
        if free is None:
            break
        pinned[free] = _dense_map(_static_shape(free[0]), isa.buffers[free[1]])

    for m in selection.matches:
        if m.mapping is not None:
            continue  # the binding supplies them (`Binding.params`)
        roles, _ = param_roles(m.instruction.spec)
        loose = [
            i
            for i, role in roles.items()
            if role in ("stride", "layout") and i not in m.shape_params
        ]
        if loose:
            raise LayoutError(
                f"{m.instruction.name}: residence param(s) {loose} are "
                f"under-constrained — they address a dimension no operand of this "
                f"instruction spans, so nothing pins them"
            )
    return selection


def _fit(instr, operands, result_value) -> dict | None:
    """The shape params ``instr`` solves to at this site, or ``None`` if it does not
    fit. Both a candidate filter and the source of the params a parametric
    instruction's cost is a function of, so Stage 1 solves once and carries the
    result (``Match.shape_params``) rather than any stage re-deriving it."""
    probe = Match(instr, operands, result_value)
    try:
        _solve_match(probe)
    except CompileError:
        return None
    return probe.shape_params


# ==========================================================================#
# Stage 3 — allocation, data movement, scheduling, emission
# ==========================================================================#


@dataclass
class CompiledProgram:
    isa: object
    io_buffer: object  # the global buffer holding program I/O
    emits: list  # list[EmitRecord] (the compute + data-movement stream)
    inputs: list  # per func arg: (offset, shape)
    outputs: list  # per func result: (offset, shape, label)
    constants: list = field(default_factory=list)  # (offset, ndarray), preloaded
    # Per emit, the spatial instance a lowering imported for it (``None`` where the
    # compiler placed the instruction itself). σ's one part this frontend cannot
    # derive: ``ISA.bind`` gives one unit per *mnemonic*, so a derived σ serializes
    # every invocation of an instruction, and a mapping says which instance runs what.
    instances: list = field(default_factory=list)

    def _issue(self, rec) -> tuple[str, float, float]:
        """``(unit name, issue cycles, pipeline depth)`` for one emitted instruction.

        *Issue* is ``ii * trips`` — the slots the instruction occupies on its unit;
        *depth* is drain, paid once per unit rather than per instruction. Stage 2's
        solved shape params are recovered from the emitted address list (a shape
        param's slot holds its solved size).

        Requires the instruction to be bound (``ISA.bind``) to a unit with a declared
        ``ISA.latency``: an abstract search weight is not a cycle count, so an ISA with
        no modeled microarchitecture is refused rather than reported in made-up units.
        Always the latency-derived value, even where the ISA overrode ``cost`` with an
        abstract weight — these methods report cycles, not the search objective."""
        spec = self.isa._ops[rec.name].spec
        if spec.unit_latency is None or not spec.unit_latency.declared:
            raise AcceleratorDescriptionError(
                f"{self.isa.name}: '{rec.name}' has no cycle model — bind it to a "
                f"@unit (ISA.bind) and declare that unit's ISA.latency(ii=, depth=)"
            )
        roles, _ = param_roles(spec)
        shape_params = {i: rec.addr[i] for i, r in roles.items() if r == "shape"}
        lat = spec.unit_latency
        return (
            spec.unit.func_name,
            lat.ii * spec.trips_at(shape_params),
            float(lat.depth),
        )

    def unit_cycles(self) -> dict[str, float]:
        """How long each hardware unit is engaged: ``sum(ii * trips) + depth``.

        A unit issuing instructions back to back pays its pipeline drain *once*, not
        per instruction — which is exactly what ``cycles()`` cannot express, since it
        has no notion of which unit runs what. This is the quantity that says where a
        program's time actually sits, and the one to watch when a transformation moves
        work between units (tiling, or trading recompute for data movement)."""
        busy: dict[str, float] = {}
        depth: dict[str, float] = {}
        for rec in self.emits:
            unit, issue, d = self._issue(rec)
            busy[unit] = busy.get(unit, 0.0) + issue
            depth[unit] = d
        return {u: busy[u] + depth[u] for u in busy}

    def bottleneck_cycles(self) -> float:
        """The busiest unit's engaged time — a **lower** bound on the placed program
        (a roofline): every unit runs concurrently and the slowest one sets the pace.

        Pair it with ``cycles()``, which bounds the same program from **above** by
        assuming nothing overlaps. The frontend models issue cost per unit, not a
        schedule, so the true count is not derivable — but it is bracketed, and which
        end it sits near is precisely what a schedule decides. Use this one to compare
        the *shapes* of two compilations: a variant that moves work off the bottleneck
        unit is genuinely faster, whereas ``cycles()`` would charge it for the extra
        instructions as if they could never overlap with anything."""
        return max(self.unit_cycles().values(), default=0.0)

    def cycles(self) -> float:
        """Serial cycle estimate of the placed program: ``sum(depth + ii * trips)``
        over every emit.

        This assumes **no** overlap — one instruction at a time, its pipeline fully
        drained before the next issues — so it is an **upper** bound, and it charges a
        program for parallelism it may well have. ``bottleneck_cycles()`` bounds the
        same program from below; see there for which to use when."""
        total = 0.0
        for rec in self.emits:
            _unit, issue, depth = self._issue(rec)
            total += depth + issue
        return total

    def _format(self) -> str:
        io = self.io_buffer.name
        lines = [f"CompiledProgram[{self.isa.name}]  io={io}", "  inputs:"]
        for i, (off, shape) in enumerate(self.inputs):
            lines.append(f"    arg{i} = {io}{list(off)}  shape={tuple(shape)}")
        if self.constants:
            lines.append("  constants:")
            for i, (off, data) in enumerate(self.constants):
                lines.append(f"    c{i} = {io}{list(off)}  shape={tuple(data.shape)}")
        lines.append("  program:")
        for rec in self.emits:
            # `#v` marks a computational attribute (α) — an immediate in the
            # instruction word, not an address; `@v` a schedule param, a field the
            # compiler *chose* rather than solved or bound.
            args = (
                [str(a) for a in rec.addr]
                + [f"#{v}" for v in rec.compute]
                + [f"@{v}" for v in rec.schedule]
            )
            lines.append(f"    {rec.name}({', '.join(args)})")
        lines.append("  outputs:")
        for off, shape, label in self.outputs:
            lines.append(f"    {label} = {io}{list(off)}  shape={tuple(shape)}")
        return "\n".join(lines)

    def epochs(self) -> list:
        """This program read as its sequence of epochs — the denotational view
        (``epoch.py``): one total configuration per emitted instruction, with the
        per-operand regions its λ-fragment names."""
        from .epoch import epochs

        return epochs(self.isa, self.emits)

    def schedule(self):
        """This program's minimal σ (``epoch.schedule``): its epochs ASAP-placed
        onto the microarchitecture's units, respecting the dependences derived
        from their regions. In cycles when the whole ISA has a cycle model —
        ``Schedule.makespan`` is then the point this program actually achieves
        inside the ``bottleneck_cycles()`` / ``cycles()`` bracket — and in unit
        steps otherwise.

        Where a lowering imported a spatial assignment (``instances``), σ carries
        it: the derivation would have serialized those epochs onto one unit."""
        from .epoch import pe_names, schedule

        eps = self.epochs()
        pes = pe_names(self.isa, eps, self.instances) if self.instances else None
        return schedule(self.isa, eps, pes)

    def check(self, **kw) -> list:
        """Check this program against the constraint system (``check.py``); an
        empty list means every executable obligation holds. ``sigma=`` verifies
        an externally supplied ``Schedule`` instead of the derived one;
        ``reachable=`` supplies the machine's R over RAW event pairs."""
        from .check import check

        return check(self, **kw)

    def dump(self) -> None:
        """Print the compiled instruction sequence (I/O map + emit stream)."""
        print(self._format())

    def __str__(self) -> str:
        return self._format()

    def _region(self, offset, shape) -> tuple:
        """The slice of the I/O buffer a value of ``shape`` placed at ``offset``
        occupies — one component per address axis, so it indexes a flat pool and a
        multi-dimensional array alike."""
        dims = _placement_dims(shape, self.io_buffer)
        return tuple(slice(o, o + d) for o, d in zip(offset, dims))

    def __call__(self, *inputs):
        """Run the compiled program on the functional simulator; returns the result
        array (or a list of arrays for a multi-output program)."""
        if len(inputs) != len(self.inputs):
            raise AssemblyError(
                f"expected {len(self.inputs)} inputs, got {len(inputs)}"
            )
        buf = self.io_buffer
        # The I/O pool's own dtype, not f32: staging an i32 program's inputs through
        # float silently rounds anything past 2**24.
        np_dt = _np_dtype(buf.kind.dtype)
        init = np.zeros(buf.memref_shape, np_dt)

        def load(offset, shape, arr):
            region = self._region(offset, shape)
            init[region] = np.asarray(arr, np_dt).reshape(init[region].shape)

        for offset, data in self.constants:
            load(offset, data.shape, data)
        for (offset, shape), arr in zip(self.inputs, inputs):
            load(offset, shape, arr)

        program = OracleProgram()
        program.steps.extend(("emit", e) for e in self.emits)
        for offset, shape, label in self.outputs:
            program.record_inspect(buf, self._region(offset, shape), label)

        results = simulate(self.isa, program, OracleConfig(init={buf: init}))
        outs = [results[label].reshape(shape) for _o, shape, label in self.outputs]
        return outs[0] if len(outs) == 1 else outs


def _solve_move_params(spec, value_size: int) -> dict:
    """Shape params for a planner-inserted movement instruction.

    A move is an identity copy, so each of its access patterns transfers the moved
    value's ``value_size`` elements: a shape param therefore satisfies
    ``prod(visible_shape) == value_size``. This is the move analogue of Stage-2
    ``solve``, which runs on matched compute instructions and so never sees a move
    the planner inserted in Stage 3. Solving the product (rather than equating the
    param with a word count) is what handles an access that *scales* its param, e.g.
    ``view(d0, a, (n, 64))``, where ``n`` is rows and the word count is ``64·n``."""
    _, arg_shapes, _ = trace_instruction(spec)
    roles, _ = param_roles(spec)
    shape_idxs = {i for i, r in roles.items() if r == "shape"}
    symtab, eqs = {}, []
    exprs = []
    for ishape in arg_shapes:
        prod_expr = sympy.Integer(1)
        for d in ishape:
            prod_expr *= (
                _to_sympy(d, symtab)
                if _index_params(d)
                else sympy.Integer(d if isinstance(d, int) else d.static_int())
            )
        exprs.append(prod_expr)
        if {i for d in ishape for i in _index_params(d)} & shape_idxs:
            eqs.append(sympy.Eq(prod_expr, value_size))
    out = {}
    if symtab:
        syms = [symtab[i] for i in sorted(symtab)]
        (vals,) = sympy.linsolve(eqs, syms)
        for i, val in zip(sorted(symtab), vals):
            if not (val.is_integer and val >= 0):
                raise ShapeError(
                    f"{spec.name}: move shape param p{i} = {val} is not a "
                    f"non-negative integer"
                )
            out[i] = int(val)
    # A move copies a whole value, so each of its patterns must transfer exactly that
    # many elements. Checked for *every* pattern, including the param-free ones a
    # fixed-size relayout is made of: routing picks moves by buffer pair, so without
    # this a value of the wrong size would be silently truncated to the tile the
    # instruction happens to describe.
    for ishape, expr in zip(arg_shapes, exprs):
        moved = expr.subs({symtab[i]: v for i, v in out.items()})
        if moved != value_size:
            raise AllocationError(
                f"{spec.name}: moves {moved} element(s) per access but the value has "
                f"{value_size} — no data-movement instruction fits this value"
            )
    return out


def _io_buffer(isa):
    globals_ = [b for b in isa.buffers.values() if b.is_global]
    if len(globals_) != 1:
        raise AcceleratorDescriptionError(
            f"expected exactly one global buffer, got {[b.name for b in globals_]}"
        )
    return globals_[0]


def _movement_catalog(isa) -> list[str]:
    """The identity (single src -> single dst) move mnemonics, in declaration order —
    the instructions the data-movement graph is built from (``_move_edges``).

    A **relayout** — an identity move whose two access patterns *differ*, e.g. a rank-2
    block of a row-major array gathered into a contiguous tile — is an ordinary move.
    Its multi-dimensional basis is filled from the operand's placement coordinate, so
    routing and spilling can use it like any other; ``_solve_move_params`` then rejects
    a value it does not fit. A machine may declare several movers between the same two
    buffers differing only in what they do to the layout, so which one applies to a
    value is decided by residence, never by the buffer pair."""
    moves = []
    for spec in isa.instructions:
        # A configuring instruction may look like a copy (it writes a state
        # register), but it *assigns configuration*: inserting one to move a value
        # would reconfigure the machine behind the program's back.
        if not is_mover(spec) or spec.configures:
            continue
        if compute_params(spec):
            # The planner inserts moves itself, so there is no source constant to
            # bind an immediate from.
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction cannot take compute "
                f"params (α) — nothing supplies them"
            )
        roles, offset_of = param_roles(spec)
        names = access_names(spec)
        loose = [
            names[i]
            for i, r in roles.items()
            if r == "stride" and names[i] not in spec.schedule_residence
        ]
        if loose:
            # A residence param on a mover is not *solved* — nothing unifies with it,
            # because the planner is what inserts the move — so it has to be chosen,
            # and choosing needs a domain. An ordering's is intrinsic (its rank!); a
            # stride's is the integers, so the ISA has to say which of them the
            # hardware can actually encode.
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction's stride param(s) {loose} "
                f"have no domain — a stride is otherwise pinned by unifying the maps "
                f"of a value's matched accesses, and a move takes part in no "
                f"unification. Give it one with @I.schedule({loose[0]}=[...])"
            )
        if _alias_groups(offset_of):
            # A move is inserted between two *independently placed* locations, so it
            # has no way to honour "read and write at one address".
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction cannot share an address "
                f"param between its source and destination"
            )
        moves.append(spec.name)
    return moves


@dataclass(eq=False)  # identity-based: distinct residences must never compare equal
class _Loc:
    """A *location*: one value's residence in one buffer, the unit of allocation.

    A value may hold several locations over its life (e.g. a ``bram`` copy and a
    ``vreg`` copy, or — after a spill — two ``vreg`` copies split around the spill
    gap). Each occupies ``size`` contiguous units at ``base`` and is read at the steps
    in ``uses`` (``last_use`` = the last), at which point its space is released;
    spilling is just ending one location and opening another.

    **``map`` is the residence itself** (see ``core.residence``): which address
    inside the location holds which logical element. A value may hold two
    locations in the *same* buffer differing only in that — one row-major, one
    channel-last — which is what makes a relayout a routing step rather than a special
    case. It is also what an instruction's access demands of an operand: a location is
    usable only if its map is the one that access describes.

    **Placement is a coordinate, not a number.** A buffer declared with more than one
    extent needs one component per axis, which is what an access pattern's
    multi-dimensional ``basis`` consumes. Allocation still packs a *single*
    axis — the outermost — so ``base`` stays one integer and the free list stays 1-D;
    the remaining components are 0. A value therefore occupies a whole band
    ``[base : base+size, 0 :, ...]`` rather than a sub-rectangle: packing rectangles is
    2-D bin packing, and the price of not doing it is unused columns, not wrong code."""

    value: object
    buffer: object
    size: int  # units along the allocated (outermost) axis
    map: tuple = ()  # the residence: (size, stride) per spanning dim
    base: int = -1  # that axis's coordinate; -1 until allocated
    last_use: int = -1
    uses: list = field(default_factory=list)  # step indices that read this location
    defs: list = field(default_factory=list)  # step indices that write it
    freed: bool = False

    @property
    def offset(self) -> tuple:
        """The placement coordinate, one component per address axis of the buffer."""
        return (self.base,) + (0,) * (self.buffer.address_rank - 1)


@dataclass
class _Move:
    name: str
    read: _Loc
    write: _Loc
    chosen: dict = field(default_factory=dict)  # access params the router chose
    schedule: list = field(default_factory=list)  # fresh schedule params, in order
    pe: object = None  # σ's spatial instance, when a lowering imported one


@dataclass
class _Compute:
    name: str
    reads: list  # list[_Loc], in source-buffer order
    write: _Loc
    offset_of: dict  # access param -> [(buffer position, coordinate axis)]
    shape_params: dict  # access param -> solved size
    reusable: set  # source-operand indices whose slot the result may reuse in place
    alpha: list  # computational attributes (α), bound from the source's constants
    schedule: list  # schedule params, in declaration order (the chosen configuration)
    # Per-axis element offsets *into* each operand's location (reads then write,
    # one ``{axis: shift}`` dict apiece), non-empty only for a lowering's tile
    # steps: the lowering says which sub-block of a value this instruction
    # touches, allocation says where the value starts.
    offsets: list = field(default_factory=list)
    # σ's spatial instance, when a lowering imported one (`mapping.assemble`); `None`
    # leaves the axis to `epoch.schedule`'s own per-unit derivation.
    pe: object = None


def _alias_groups(offset_of: dict) -> list:
    """The must-alias constraints an instruction's access states: one entry per
    address param that is the basis of more than one buffer.

    Only axis 0 is constrained in practice — allocation packs that axis and leaves the
    rest at 0, so a shared param on any other axis is satisfied by construction."""
    return [
        (param, [pos for pos, _axis in refs])
        for param, refs in offset_of.items()
        if len(refs) > 1 and refs[0][1] == 0
    ]


# Prim tags whose output position i depends only on input position i. Derived from
# the registry categories, plus the two bespoke elementwise prims. An **allowlist**,
# so anything else — a reduction, a contraction, a permutation, a windowed op, or a
# prim added later — is conservatively excluded: guessing wrong here silently
# overwrites an operand that the result does not line up with.
_POSITION_PRESERVING = {"identity", "relu"} | {
    tag
    for tag, p in primitive.REGISTRY.items()
    if p.category
    in (
        primitive.UNARY,
        primitive.UNARY_ZP,
        primitive.BINARY,
        primitive.BINARY_SHIFT,
        primitive.COMPARE,
        primitive.SELECT,
        primitive.CAST,
    )
}


def _reusable_operands(instruction) -> set:
    """Source-operand indices whose slot the result may safely reuse in place.

    Reuse is safe for an operand iff its value reaches the result through only
    position-preserving ops: output position i then depends only on that operand's
    position i, so writing the result over the operand is fine. A matmul mixes
    positions, so any operand feeding a matmul is not reusable — but an *accumulator*
    read only by an element-wise add (``c + a @ b``) is, which lets a K-reduction
    chain collapse onto a single accumulator slot (the hardware's block buffer). For
    a purely element-wise op every operand is reusable; for a plain matmul none is.

    The functional oracle cannot check this — it reads all operands before writing any
    destination, so an overwrite it should have observed simply does not happen there.
    Hence the conservative allowlist rather than a denylist of known mixers."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return set()
    n_src = len(instruction.spec.sources)
    safe: set = set()

    def walk(node, position_preserving: bool):
        if node.kind == "arg":
            if position_preserving and node.buffer_index < n_src:
                safe.add(node.buffer_index)
            return
        child_ok = position_preserving and node.kind in _POSITION_PRESERVING
        for a in node.args:
            walk(a, child_ok)

    walk(results[0], True)
    return safe


def _colocatable(m: Match) -> set:
    """Source-operand indices laid out exactly like the destination.

    Position-preserving semantics say output tensor position ``i`` depends only on
    operand position ``i``; writing the result over that operand is safe only if the
    two positions are also the same *address*. An instruction that relayouts while it
    computes (elementwise semantics, differing maps) breaks that, so residence has to
    agree before a slot is handed over."""
    spec = m.instruction.spec
    patterns, _, _ = trace_instruction(spec)
    n_src = len(spec.sources)
    dst = residence(access_map(patterns[n_src], m.shape_params))
    return {
        i
        for i in range(n_src)
        if residence(access_map(patterns[i], m.shape_params)) == dst
    }


# ==========================================================================#
# Tile expansion — lower one layer-level match to a run of tile instructions
# ==========================================================================#


class _ExpandRecorder:
    """Collects the instruction calls an ``@expand`` body issues, as *values*.

    Same protocol as ``OracleProgram``: ``Instruction.__call__`` records into whatever
    ``isa._active_oracle`` holds. The difference from an ``@oracle`` body is who is
    writing the assembly. An oracle is hand-written, so its emits are taken as given;
    an expansion is the *compiler's own lowering*, and it runs **inside** the planner
    (``plan`` pass 1), before anything is allocated — so what it records is not an
    address list but a list of ``Ref``s (a value plus an offset into it) alongside the
    solved shape params. Turning those into locations, and the locations into
    addresses, is the planner's job, which is exactly what an expansion could not
    reach when it ran last."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple] = []

    def record_emit(self, name, addr, compute):
        assert not compute, f"{self.name}: @expand issued '{name}' with α"
        self.calls.append((name, list(addr)))


def expand_calls(isa, spec, args: list) -> list:
    """Run an instruction's ``@expand`` body and return the calls it issues, as
    ``(mnemonic, [Ref | int])``.

    An instruction that carries ``@expand`` is *layer-level*: it matches as one
    operation and lowers to many. The body's own operands arrive as ``Ref``s to the
    planner's locations for them and its shape params as Stage-2-solved ints, so the
    body computes offsets **within** a value and never an absolute address. Its
    ``@compute`` region stays the layer's semantics: the catalog states what the
    expansion must equal, and the oracle executes the expansion, so the two can be
    diffed."""
    if compute_params(spec):
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand and compute params (α) cannot be combined — the "
            f"expansion body receives address params only, so it has no way to pass "
            f"the bound immediate on to the tile instructions it issues"
        )
    if layout_params(spec):
        # The body *is* handed the ordering, but a sub-block of a layer laid out in
        # some ordering does not sit at a fixed stride from the layer's base -- the
        # translation is the tiling this frontend leaves to the mid-end. Refused
        # rather than trusted to a body nothing here can check.
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand and an ordering param cannot be combined — the "
            f"tiles the expansion issues address sub-blocks whose residence is not "
            f"the layer's map"
        )
    recorder = _ExpandRecorder(spec.name)
    prev = isa._active_oracle
    isa._active_oracle = recorder
    try:
        spec.expand_fn(*args)
    finally:
        isa._active_oracle = prev
    if not recorder.calls:
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand issued no instructions"
        )
    return recorder.calls


def _placement_dims(shape, buf) -> tuple:
    """The extent a value of ``shape`` occupies in ``buf``'s coordinates.

    A slot-addressed buffer takes one number (how many slots the value fills); a
    multi-dimensional buffer takes the value's own dims, aligned to the buffer's rank
    modulo leading unit dims (the same rank alias ``_align_ranks`` looks through, so a
    torch-batched ``1x4x4`` places as a ``4x4`` block)."""
    if buf.address_rank == 1:
        return (max(prod(shape) // buf.slot_size, 1),)
    dims = list(shape)
    if len(dims) != buf.address_rank:
        dims = _strip_leading_units(dims)
    if len(dims) != buf.address_rank:
        raise AllocationError(
            f"'{buf.name}' is addressed by {buf.address_rank} indices but the value's "
            f"shape is {tuple(shape)} — a multi-dimensional buffer holds values of its "
            f"own rank"
        )
    if any(d > e for d, e in zip(dims[1:], buf.extents[1:])):
        raise AllocationError(
            f"'{buf.name}': a value of shape {tuple(dims)} does not fit the array's "
            f"{buf.extents} extent"
        )
    return tuple(dims)


def _loc_size(value, buf) -> int:
    """Units of the allocated (outermost) axis one location of ``value`` occupies."""
    return _placement_dims(_shape(value), buf)[0]


def _const_array(value) -> np.ndarray | None:
    """A source constant's data, shaped like its tensor type, or ``None`` if it is
    not a constant this frontend can read (see ``const_elements``)."""
    elems = const_elements(_canon(value))
    if elems is None:
        return None
    shape = _shape(_canon(value))
    arr = np.array(elems)
    if arr.size == 1:  # a splat prints as one element whatever its extent
        arr = np.full(prod(shape), elems[0])
    return arr.reshape(shape)


@dataclass(frozen=True)
class _Edge:
    """One usable data-movement step for one value size.

    ``relayout`` is the ``(read, write)`` residence pair when the move's two access
    patterns describe *different* maps, and ``None`` when they describe the same one.
    That distinction is the whole of what a move does to a layout:

    - **equal maps** — the address correspondence is the identity, so the move copies
      the region verbatim and carries **whatever** residence the value had. This is why
      a plain dma can spill a channel-last value and reload it unharmed.
    - **different maps** — the correspondence is a genuine permutation of addresses, so
      the move is usable only on a value laid out exactly as it *reads*, and it then
      lays it out exactly as it *writes*. Applied to any other residence it would
      produce a scrambling no instruction asked for, so it simply is not an edge there.
    """

    src: str
    dst: str
    name: str
    relayout: tuple | None
    # Ordering params -> the permutation this edge was built with. A mover with an
    # ordering param is one instruction offering a *family* of relayouts; each member
    # is its own edge, so choosing a route is what chooses the ordering.
    chosen: dict = field(default_factory=dict)
    # Fresh schedule params, in declaration order — a mover configures like any other
    # instruction, so a burst width or a mode field is chosen here too.
    schedule: list = field(default_factory=list)
    # What this step costs: the mover's own `InstructionSpec.cost_of`, so routing is
    # minimized against the same number selection is. 1.0 when the ISA declares
    # nothing, which is the hop count this used to be.
    cost: float = 1.0

    def follow(self, res: tuple) -> tuple | None:
        """The residence a value laid out as ``res`` has after this move, or ``None``
        if the move does not apply to it."""
        if self.relayout is None:
            return res
        read, write = self.relayout
        return write if read == res else None


def _move_edges(isa, moves: list, size: int) -> list:
    """The data-movement edges available to a value of ``size`` elements.

    A move that does not *fit* the value is not an edge: ``_solve_move_params`` sizes
    each mover against the value, so a fixed-size relayout only ever appears for the
    values it can actually carry.

    A mover **configures** exactly like a matched instruction: ``configurations`` is
    handed the residence params it chooses rather than solves (``mover_domains``) and
    returns every assignment its ``@schedule`` predicate admits, priced. One assignment
    is one edge, so choosing a route is what chooses the configuration — that is how a
    layout-reconfiguring move is expressible (MINISA's ``Set*VNLayout``) without the ISA
    author writing one instruction per permutation, and how a machine states that its
    permutation network cannot reach every packing: the predicate simply admits fewer.

    Each edge is priced by its mover's own ``cost_of``, at this value's size and this
    edge's configuration. That is the whole of "pricing a relayout": an ISA that charges
    a repacking dma more than a plain copy — or a wide burst less per element than a
    narrow one — states it the same way it states any other instruction's cost, and
    ``_explore`` routes against it."""
    edges = []
    for name in moves:
        spec = isa._ops[name].spec
        try:
            params = _solve_move_params(spec, size)
        except CompileError:
            continue
        patterns, _, _ = trace_instruction(spec)
        slot = {n: i for i, n in enumerate(access_names(spec))}
        for chosen, cost in spec.configurations(params, mover_domains(spec)):
            # An access param's choice fills its own address slot, exactly as a solved
            # shape param does; a fresh schedule param goes in the instruction word.
            residence_params = {slot[n]: v for n, v in chosen.items() if n in slot}
            bound = params | residence_params
            read = residence(access_map(patterns[0], bound))
            write = residence(access_map(patterns[1], bound))
            edges.append(
                _Edge(
                    spec.sources[0].name,
                    spec.destinations[0].name,
                    name,
                    None if read == write else (read, write),
                    residence_params,
                    [chosen[n] for n in spec.schedule_domains],
                    cost,
                )
            )
    return edges


def _explore(edges: list, starts) -> tuple[list, dict, dict]:
    """Dijkstra over ``(buffer, residence)`` states: every state a value can be moved
    into **cheapest first**, the tree of cheapest predecessors (``state -> (previous
    state, edge) | None``), and the cost of reaching each.

    Routing over *states* rather than buffers is what makes a relayout something the
    planner can find on its own: a value that is in the right buffer but the wrong
    layout is simply a state one repacking edge away — including a repack from a buffer
    to itself, which as a plain buffer path would have been a zero-hop no-op. The
    predecessor is the *edge* rather than its name, because an edge also carries the
    ordering assignment it was built with — the move's own residence params.

    Distance is the sum of ``_Edge.cost``, so a machine that prices its movers gets
    routed by price rather than by hop count. When it prices none of them every edge
    costs 1.0 and this is exactly the breadth-first search it replaces: the insertion
    counter keeps equal-cost states settling in discovery order, so the paths (and the
    tie-breaks between them) are unchanged."""
    dist = {s: 0.0 for s in starts}
    prev: dict = {s: None for s in starts}
    heap = [(0.0, i, s) for i, s in enumerate(starts)]
    tick, settled, seen = len(heap), [], set()
    while heap:
        d, _, state = heapq.heappop(heap)
        if state in seen:
            continue
        seen.add(state)
        settled.append(state)
        buf, res = state
        for edge in edges:
            if edge.src != buf:
                continue
            carried = edge.follow(res)
            if carried is None:
                continue
            nxt = (edge.dst, carried)
            if d + edge.cost < dist.get(nxt, inf):
                dist[nxt] = d + edge.cost
                prev[nxt] = (state, edge)
                heapq.heappush(heap, (dist[nxt], tick, nxt))
                tick += 1
    return settled, prev, dist


def _reachable(edges: list, starts) -> list:
    """Every ``(buffer, residence)`` state a value can be moved into, cheapest first."""
    return _explore(edges, starts)[0]


def _route(edges: list, starts, goal: tuple) -> list | None:
    """The cheapest path from any of ``starts`` to ``goal``, as
    ``[(state, edge | None), ...]`` beginning at the reached start, or ``None`` if
    ``goal`` is unreachable."""
    _, prev, _dist = _explore(edges, starts)
    if goal not in prev:
        return None
    path, state = [], goal
    while state is not None:
        path.append((state, prev[state][1] if prev[state] else None))
        state = prev[state][0] if prev[state] else None
    return list(reversed(path))


class _Planner:
    """The locations and steps of one compilation, and what turns them into
    addresses.

    Pass 1 appends steps over locations; passes 2 and 3 (``finish``) are liveness,
    best-fit allocation with Belady spilling to a fixpoint, and emission. Only pass
    1 depends on where the program came from — ``plan`` drives it from a
    ``Selection``, ``lower_expansion`` from the calls an ``@expand`` body issues,
    and ``mapping.assemble`` from an imported nest — which is what makes *a mapping
    is an imported expansion* a fact about the code rather than only about the
    account."""

    def __init__(self, isa):
        self.isa = isa
        self.io = _io_buffer(isa)
        self.moves = _movement_catalog(isa)
        self.loc: dict = {}  # value -> {(buffer name, residence): _Loc}
        self.steps: list = []
        self.constants: list = []  # (_Loc in io, ndarray) per constant used as data
        self._edges_for: dict = {}  # element count -> the moves usable at that size
        self._run_edges: dict = {}  # (src, dst, run length) -> the mover chosen

    # ---- locations, and the data movement between them --------------------- #

    def edges(self, size: int) -> list:
        if size not in self._edges_for:
            self._edges_for[size] = _move_edges(self.isa, self.moves, size)
        return self._edges_for[size]

    def make_loc(self, value, buf, res) -> _Loc:
        l = _Loc(value, buf, _loc_size(value, buf), res)
        self.loc.setdefault(value, {})[(buf.name, res)] = l
        return l

    def at(self, value, buf, res) -> _Loc:
        """The location of ``value`` in ``buf`` laid out as ``res``, made if new.

        Unlike ``make_loc`` this is idempotent: a tile the lowering invented is
        written many times (once per round of the run it lowers to) and every write
        is the *same* location — one slot, reused — which is what makes its live
        range the whole run and its address the allocator's to pick."""
        here = self.loc.setdefault(value, {})
        return here.get((buf.name, res)) or self.make_loc(value, buf, res)

    def abi(self, value) -> tuple:
        """The host ABI's residence for a value in the I/O buffer."""
        return _dense_map(_static_shape(value), self.io)

    def route_move(self, cur: _Loc, path: list, sink: list) -> _Loc:
        """Append a move per hop along ``path`` (states from ``_route``, starting at
        ``cur``'s own state); return the final location."""
        for (name, res), edge in path[1:]:
            dst = self.make_loc(cur.value, self.isa.buffers[name], res)
            sink.append(_Move(edge.name, cur, dst, edge.chosen, edge.schedule))
            cur = dst
        return cur

    def bring_to(self, value, target, want, who) -> _Loc:
        """A location of ``value`` in ``target`` laid out as ``want``.

        A value that is in the right buffer but the wrong layout is not resident: it
        is one repacking edge away, and finding that edge is the same search as
        finding a route between buffers. Which is the point — a relayout is data
        movement, so the planner inserts it exactly the way it inserts any other
        move, and prices it the same way too."""
        here = self.loc.get(value, {})
        if (target.name, want) in here:
            return here[(target.name, want)]
        if not here:
            # A constant used as a *data* operand is program data, not part of any
            # instruction: ACT Def 3.8 puts it in the ASM alongside the inputs
            # (`concat(bflat(X), bflat(const))`), which is exactly a location in the
            # I/O buffer that `CompiledProgram.__call__` fills before the run.
            data = _const_array(value)
            if data is None:
                raise CompileError(
                    f"a value of shape {_shape(value)} has no location to move from: "
                    f"it is neither a program input, nor the result of a matched "
                    f"instruction, nor a readable constant (a constant whose data is "
                    f"a `dialect_resource` blob cannot be read — pass it as a function "
                    f"argument instead)"
                )
            self.constants.append(
                (self.make_loc(value, self.io, self.abi(value)), data)
            )
            here = self.loc[value]
            if (target.name, want) in here:
                return here[(target.name, want)]
        avail = self.edges(prod(_shape(value)))
        path = _route(avail, list(here), (target.name, want))
        if path is None:
            raise self._unroutable(value, here, target, want, who)
        return self.route_move(here[path[0][0]], path, self.steps)

    def _unroutable(self, value, here, target, want, who) -> CompileError:
        """Say *why* a value cannot get where it is needed: an unreachable buffer, or a
        reachable one in the wrong layout with nothing that repacks it."""
        avail = self.edges(prod(_shape(value)))
        anywhere = {
            res for buf, res in _reachable(avail, list(here)) if buf == target.name
        }
        where = ", ".join(f"'{b}' as {show_map(r)}" for b, r in here)
        if not anywhere:
            starts = {b for b, _r in here}
            live = {e.name for e in avail}
            # A mover that leaves one of these buffers but contributed no edge was
            # refused *for this value* — by its own @schedule predicate, or for not
            # fitting the size. That is a configuration failure rather than a missing
            # instruction, and the two want different fixes.
            silent = sorted(
                name
                for name in self.moves
                if name not in live
                and self.isa._ops[name].spec.sources[0].name in starts
            )
            note = (
                f" — {silent} leave(s) those buffers but no legal configuration of "
                f"them exists for a value of {prod(_shape(value))} element(s)"
                if silent
                else ""
            )
            return AllocationError(
                f"{who}: no data-movement route from {sorted(starts)} "
                f"to '{target.name}'{note}"
            )
        return LayoutError(
            f"{who}: needs a value of shape {_shape(value)} in '{target.name}' laid "
            f"out as {show_map(want)}, but it lives in {where} and no data movement "
            f"relayouts between them — declare the relayout as a move, or have the "
            f"two ends agree on a layout"
        )

    def copy_run(self, src, src_off, dst, dst_off, n: int, who: str, pe=None) -> None:
        """One contiguous ``n``-element transfer, ``(location, offset)`` to
        ``(location, offset)``, as a step.

        ``bring_to`` moves a whole *value* and opens a fresh location at each hop;
        this moves a run **inside** two locations that already exist, which is what
        a tile transfer is made of. The instruction is not named by the caller but
        found in the same edge graph routing uses — every mover the ISA declares, at
        every configuration its ``@schedule`` admits, priced by its own
        ``cost_of``."""
        edge = self._run_edge(src.buffer, dst.buffer, n, who)
        spec = self.isa._ops[edge.name].spec
        _, offset_of = param_roles(spec)
        self.steps.append(
            _Compute(
                edge.name,
                [src],
                dst,
                offset_of,
                _solve_move_params(spec, n) | edge.chosen,
                set(),  # a transfer never coalesces onto its source
                [],
                edge.schedule,
                [{0: src_off}, {0: dst_off}],
                pe,
            )
        )

    def reduce_run(self, src, src_off, dst, dst_off, n: int, who: str, pe=None):
        """One contiguous ``n``-element transfer that **combines** instead of
        overwriting, with the instruction the machine declares for it
        (``ISA.network(reduces=...)``).

        This is what makes a spatial reduction expressible rather than asserted:
        the combination is an ordinary instruction with an ordinary compute region,
        so it is an epoch like any other — the oracle executes it, definedness sees
        that it reads the accumulator, and the dependence edges through the shared
        destination serialize the instances' contributions."""
        instr = self.isa.reduces
        assert instr is not None, f"{who}: no reducing instruction is declared"
        spec = instr.spec
        want = (src.buffer, dst.buffer, dst.buffer)
        got = tuple(spec.buffers)
        if got != want:
            raise AllocationError(
                f"{who}: '{spec.name}' combines "
                f"{[b.name for b in got]}, but this reduction has to combine a "
                f"partial in '{src.buffer.name}' into '{dst.buffer.name}'"
            )
        if compute_params(spec):
            raise AcceleratorDescriptionError(
                f"{spec.name}: a reducing transfer takes no computational attribute "
                f"— nothing supplies one where the planner issues it"
            )
        roles, offset_of = param_roles(spec)
        params = _solve_move_params(spec, n)
        config = spec.configure(
            {i: params[i] for i, r in roles.items() if r == "shape"}
        )
        if config is None:
            raise AllocationError(
                f"{who}: '{spec.name}' has no legal @schedule configuration for a run "
                f"of {n} element(s)"
            )
        chosen, _cost = config
        self.steps.append(
            _Compute(
                spec.name,
                [src, dst],
                dst,
                offset_of,
                params,
                set(),
                [],
                [chosen[k] for k in spec.schedule_domains],
                [{0: src_off}, {0: dst_off}, {0: dst_off}],
                pe,
            )
        )

    def _run_edge(self, src, dst, n: int, who: str) -> _Edge:
        """The cheapest mover carrying an ``n``-element run from ``src`` to ``dst``.

        A run is dense at both ends, so the move has to carry the residence
        verbatim: an edge that relayouts states a different address correspondence
        and would scramble it. Choosing here rather than being told is what lets a
        mover with an ordering param be *used* — at the assignments that copy
        verbatim — where naming one instruction has to refuse it for having any."""
        key = (src.name, dst.name, n)
        if key not in self._run_edges:
            usable = [
                e
                for e in self.edges(n)
                if e.src == src.name and e.dst == dst.name and e.relayout is None
            ]
            if not usable:
                raise AllocationError(
                    f"{who}: no data-movement instruction carries a run of {n} "
                    f"element(s) from '{src.name}' to '{dst.name}'"
                )
            self._run_edges[key] = min(usable, key=lambda e: e.cost)
        return self._run_edges[key]

    # ---- pass 1, for a match that lowers to a run of tile instructions ------ #

    def _tile_maps(self, calls, who) -> dict:
        """Each staging tile's residence, from the accesses that *fix* one.

        A mover whose read and write maps agree carries a residence rather than
        fixing it — the same fact ``_move_edges`` records as a residence-preserving
        edge — so what pins a tile is its non-mover accesses, unified across them.
        That is Stage 2b's rule applied to a value the lowering invented, and it is
        why a body may stage a run of words with one instruction and read it back as
        a multi-dimensional tile with the next."""
        fixed: dict = {}
        carried: dict = {}
        for name, addr in calls:
            spec = self.isa._ops[name].spec
            patterns, _, _ = trace_instruction(spec)
            params = {i: v for i, v in enumerate(addr) if not isinstance(v, Ref)}
            maps = [_wants(p, params) for p in patterns]
            _, offset_of = param_roles(spec)
            preserving = is_mover(spec) and maps[0] == maps[1]
            for i, v in enumerate(addr):
                if not isinstance(v, Ref) or not isinstance(v.value, Tile):
                    continue
                pos = offset_of[i][0][0]
                into = carried if preserving else fixed
                if into.setdefault(v.value, maps[pos]) != maps[pos]:
                    raise LayoutError(
                        f"{who}: its staging tile of shape {v.value.shape} is read as "
                        f"{show_map(into[v.value])} by one instruction and "
                        f"{show_map(maps[pos])} by '{name}' — a value has one "
                        f"residence, so the expansion has to stage them separately"
                    )
        out = {}
        for tile, res in (carried | fixed).items():
            if prod(_shape(tile)) != prod(s for s, _st in res):
                raise ShapeError(
                    f"{who}: its staging tile of shape {tile.shape} is accessed as "
                    f"{show_map(res)} — the expansion moves a different number of "
                    f"elements than the tile holds"
                )
            out[tile] = res
        return out

    def lower_expansion(self, m, spec, reads, write) -> None:
        """Lower one layer-level match to the run of tile instructions its
        ``@expand`` body issues — as *steps over locations*, so the staging it needs
        is allocated, made live and spillable like anything else.

        The body is handed a ``Ref`` to the planner's location for each of its own
        operands and the Stage-2-solved size for each shape param. An operand the
        issued instruction addresses by a *constant* (a state buffer: the MXU's
        stationary tile) has no parameter to name a value with, so it resolves to
        whatever the expansion last put in that buffer — which is what the hardware
        means by it."""
        isa = self.isa
        _, layer_offsets = param_roles(spec)
        operands = reads + [write]
        args = [
            (
                Ref(operands[layer_offsets[i][0][0]])
                if i in layer_offsets
                else m.shape_params[i]
            )
            for i in range(arity(spec.access_fn))
        ]
        calls = expand_calls(isa, spec, args)
        tiles = self._tile_maps(calls, spec.name)
        resident: dict = {}  # buffer name -> what a constant-addressed access means
        for name, addr in calls:
            issued = isa._ops[name].spec
            issued_patterns, _, _ = trace_instruction(issued)
            params = {i: v for i, v in enumerate(addr) if not isinstance(v, Ref)}
            _, offset_of = param_roles(issued)
            n_src = len(issued.sources)
            locs: list = [None] * len(issued.buffers)
            rel = [{} for _ in issued.buffers]
            for i, v in enumerate(addr):
                if not isinstance(v, Ref):
                    continue
                for pos, axis in offset_of[i]:
                    buf = issued.buffers[pos]
                    if isinstance(v.value, Tile):
                        if pos < n_src and v.value not in self.loc:
                            raise CompileError(
                                f"{spec.name}: '{name}' reads a staging tile of shape "
                                f"{v.value.shape} that nothing has written yet"
                            )
                        locs[pos] = self.at(v.value, buf, tiles[v.value])
                    else:
                        locs[pos] = v.value
                    rel[pos][axis] = v.offset
            maps = [_wants(p, params) for p in issued_patterns]
            carries = is_mover(issued) and maps[0] == maps[1]
            for pos in range(len(locs)):
                buf = issued.buffers[pos]
                if locs[pos] is None and pos >= n_src:
                    # A constant-addressed *destination* is a mover's: a fresh
                    # location of the value it carries, keeping that value's
                    # residence when the mover preserves one.
                    src = locs[0]
                    if src is None:
                        raise CompileError(
                            f"{spec.name}: '{name}' writes '{buf.name}' at a fixed "
                            f"address with no operand to carry there"
                        )
                    locs[pos] = self.at(
                        src.value, buf, src.map if carries else maps[pos]
                    )
                elif locs[pos] is None:
                    # A constant-addressed *read* names no value; within an expansion
                    # it is whatever was last written to that buffer, which is what a
                    # state buffer (the MXU's stationary tile) means by it.
                    locs[pos] = resident.get(buf.name)
                    if locs[pos] is None:
                        raise CompileError(
                            f"{spec.name}: '{name}' reads '{buf.name}' at a fixed "
                            f"address, but nothing in the expansion has written there"
                        )
                    if locs[pos].map != maps[pos]:
                        raise LayoutError(
                            f"{spec.name}: '{name}' reads '{buf.name}' as "
                            f"{show_map(maps[pos])}, but the expansion left it holding "
                            f"{show_map(locs[pos].map)}"
                        )
                if pos >= n_src:
                    resident[buf.name] = locs[pos]
            config = issued.configure(
                {i: addr[i] for i, r in param_roles(issued)[0].items() if r == "shape"}
            )
            if config is None:
                raise CompileError(
                    f"{spec.name}: its @expand issues '{name}' with no legal "
                    f"@schedule configuration — the expansion asks for a "
                    f"configuration the hardware cannot be put into, so the "
                    f"layer-level instruction's own @schedule predicate is admitting "
                    f"more than it can actually lower"
                )
            chosen, _cost = config
            self.steps.append(
                _Compute(
                    name,
                    locs[:n_src],
                    locs[n_src],
                    offset_of,
                    params,
                    set(),  # an issued instruction never coalesces onto an operand
                    [],
                    [chosen[n] for n in issued.schedule_domains],
                    rel,
                )
            )

    # ---- passes 2 + 3 ------------------------------------------------------ #

    def finish(self, inputs: list, outputs: list) -> CompiledProgram:
        """Liveness, allocation to a no-spill fixpoint, and emission.

        ``inputs`` are the locations the host fills before the run, in argument
        order; ``outputs`` are the ``(location, shape, label)`` it reads back."""
        isa, io, steps = self.isa, self.io, self.steps
        output_locs = [l for l, _shp, _label in outputs]

        def reads_of(st) -> list:
            return [st.read] if isinstance(st, _Move) else st.reads

        def all_locs() -> list:
            seeds = inputs + [l for l, _data in self.constants]
            seen, out = set(map(id, seeds)), list(seeds)
            for st in steps:
                for l in reads_of(st) + [st.write]:
                    if id(l) not in seen:
                        seen.add(id(l))
                        out.append(l)
            return out

        def liveness():
            final = len(steps)  # virtual step: the terminator reads the outputs
            for l in all_locs():
                l.last_use, l.uses, l.defs, l.base, l.freed = -1, [], [], -1, False
            for i, st in enumerate(steps):
                for r in reads_of(st):
                    r.last_use = i
                    r.uses.append(i)
                # A location is live until its last *write* too, not only its last
                # read. A source value is SSA and written once, so this says nothing
                # about it; a tile the lowering invented is refilled every round, and
                # freeing it at the last read of round 1 would hand its slot away
                # while the rest of the run still writes there.
                st.write.defs.append(i)
                st.write.last_use = max(st.write.last_use, i)
            for l in output_locs:
                l.last_use = final
                l.uses.append(final)

        def allocate():
            """Assign offsets in one walk; on overflow return ``(victim, step)`` to
            spill, else ``None`` (offsets are final). Belady victim selection."""
            free = {name: [(0, buf.capacity)] for name, buf in isa.buffers.items()}
            live = {name: [] for name in isa.buffers}  # placed, not-yet-freed

            def release(l):
                runs = sorted(free[l.buffer.name] + [(l.base, l.size)])
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
                l.base = best_fit(l.buffer, l.size)
                if l.base is None:
                    return False
                live[l.buffer.name].append(l)
                return True

            def forced_alias(st, reads, write, t):
                """The operand location this step's write **must** be placed on top
                of, or ``None`` if its access forces nothing.

                An address param used as the basis of two buffers is not a hint: the
                ISA is saying those operands are at one address (QKV's ``softmax``
                reads and writes one ``addr``). Allocation therefore has to guarantee
                it, rather than leave it to the opportunistic reuse below — which
                does nothing when the operand outlives the instruction."""
                if not isinstance(st, _Compute):
                    return None
                n = len(reads)
                target = None
                for param, positions in _alias_groups(st.offset_of):
                    operands = [p for p in positions if p < n]
                    for p in operands[1:]:
                        if reads[p] is not reads[operands[0]]:
                            raise AllocationError(
                                f"{st.name}: address param p{param} puts operands "
                                f"{operands[0]} and {p} at one address, but they are "
                                f"bound to different values"
                            )
                    if n in positions and operands:
                        target = reads[operands[0]]
                if target is None or target is write:
                    # `target is write` is an accumulator a lowering already placed
                    # in one slot: the aliasing the access states holds by
                    # construction, so there is nothing to coalesce.
                    return None
                if target.buffer is not write.buffer or target.size != write.size:
                    raise AllocationError(
                        f"{st.name}: writes its result over the operand it reads, but "
                        f"the two do not occupy the same space "
                        f"({target.buffer.name}[{target.size}] vs "
                        f"{write.buffer.name}[{write.size}])"
                    )
                if target.last_use != t or target.freed:
                    raise AllocationError(
                        f"{st.name}: writes its result over the operand it reads, but "
                        f"that operand is read again at step "
                        f"{min(u for u in target.uses if u > t)}"
                        f" — the in-place write would destroy it. Copy it first, or "
                        f"use an out-of-place instruction."
                    )
                return target

            # Constants are resident from the start, exactly like inputs: nothing in
            # the program writes them, so the allocator must reserve their space up
            # front.
            for l in inputs + [l for l, _data in self.constants]:
                if not place(l):
                    raise AllocationError(
                        f"backing store '{io.name}' overflow on inputs"
                    )

            for t, st in enumerate(steps):
                reads, write = reads_of(st), st.write
                reused = forced_alias(st, reads, write, t)
                if reused is None and isinstance(st, _Compute) and st.reusable:
                    reused = next(
                        (
                            reads[i]
                            for i in st.reusable
                            if reads[i].buffer is write.buffer
                            and reads[i].last_use == t
                            and reads[i].size == write.size
                            and not reads[i].freed
                        ),
                        None,
                    )
                if reused is not None:
                    write.base = reused.base  # coalesce: hand the slot to the result
                    live[write.buffer.name].remove(reused)
                    live[write.buffer.name].append(write)
                    reused.freed = True
                elif write.base >= 0 and not write.freed:
                    # Already placed and still live: a tile the lowering invented,
                    # written once per round of the run it lowers to. One location,
                    # one slot — a source value is SSA and lands here only once, but
                    # such a tile is not, and re-placing it would consume a fresh
                    # block every round.
                    pass
                elif not place(write):
                    return _pick_victim(write.buffer, t, live[write.buffer.name]), t
                for r in reads:
                    if r.last_use == t and not r.freed:
                        release(r)
            return None

        def _pick_victim(buf, t, resident):
            # Spill the resident location whose *next* use is farthest away
            # (Belady), excluding anything this very step still reads. A location the
            # program writes again is excluded too: the reload lands after that write
            # and would undo it. Tiles a lowering invented are exactly that — one
            # slot refilled every round — so a machine too small for a lowering's
            # working set is an allocation error, not a spill that loses a refill.
            candidates = [
                l
                for l in resident
                if t not in l.uses
                and any(s > t for s in l.uses)
                and not any(s > t for s in l.defs)
            ]
            if not candidates:
                raise AllocationError(
                    f"buffer '{buf.name}' overflow at step {t}: no spillable location "
                    f"(an instruction needs more slots than '{buf.name}' has)"
                )
            return max(candidates, key=lambda l: min(s for s in l.uses if s > t))

        def spill(victim, t):
            """Evict ``victim`` from its buffer to the backing store over
            [t, next-use): store it down before step t, reload it back before its
            next use, and repoint the later uses onto the reloaded copy. Grows
            ``steps`` (re-run liveness)."""
            u = min(s for s in victim.uses if s > t)
            if victim.buffer is io:
                raise AllocationError(f"backing store '{io.name}' overflow")
            # A spill must come back **as it left** — the later uses were routed to
            # this residence — but only the *round trip* has to preserve it, not each
            # leg: a machine whose only path to the backing store is a repacking dma
            # spills fine as long as the reload repacks back. So the store residence
            # is searched for, cheapest first, rather than assumed to be the victim's.
            home = (victim.buffer.name, victim.map)
            avail = self.edges(prod(_shape(victim.value)))
            down = up = None
            for state in _reachable(avail, [home]):
                if state[0] != io.name:
                    continue
                down, up = _route(avail, [home], state), _route(avail, [state], home)
                if down and up:
                    break
                down = up = None
            if not (down and up):
                raise AllocationError(
                    f"cannot spill from '{victim.buffer.name}': no round trip to "
                    f"'{io.name}' returns the value to {show_map(victim.map)}"
                )
            store_steps: list = []
            spilled = self.route_move(victim, down, store_steps)
            reload_steps: list = []
            reloaded = self.route_move(spilled, up, reload_steps)
            for i, st in enumerate(steps):
                if i >= u:
                    if isinstance(st, _Move):
                        if st.read is victim:
                            st.read = reloaded
                    else:
                        st.reads = [reloaded if r is victim else r for r in st.reads]
            steps[u:u] = reload_steps  # reload before the next use ...
            steps[t:t] = store_steps  # ... and store before the overflow step (t < u)

        # An instruction's operands are all live at once and none can be spilled
        # (they are in use), so a buffer that cannot even hold one instruction's
        # operands is infeasible — reject upfront. This also guarantees the spill
        # loop terminates: every remaining overflow then has a non-operand victim to
        # evict, so each spill resolves the earliest overflow and pushes the frontier
        # strictly later.
        for st in steps:
            if isinstance(st, _Compute):
                need: dict = {}
                for r in dict.fromkeys(st.reads):  # distinct locations (handles a*a)
                    need[r.buffer.name] = need.get(r.buffer.name, 0) + r.size
                for name, n in need.items():
                    if n > isa.buffers[name].capacity:
                        raise AllocationError(
                            f"{st.name}: operands need {n} unit(s) of '{name}' but it "
                            f"holds only {isa.buffers[name].capacity} "
                            f"(capacity too small to spill into)"
                        )

        while True:
            liveness()
            outcome = allocate()
            if outcome is None:
                break
            spill(*outcome)

        # --- emit with concrete offsets --------------------------------------
        emits: list[EmitRecord] = []
        for st in steps:
            if isinstance(st, _Move):
                # A move fills its access params like a compute: offset params take
                # the source/destination placements; shape params are solved from the
                # moved value's element count (prod(visible) == value size), since the
                # move was inserted in Stage 3 and never went through Stage-2 solve.
                spec = isa._ops[st.name].spec
                _, offset_buffer = param_roles(spec)
                # `st.chosen` are the residence params the router chose by choosing
                # this edge; they fill their own slots in the address list exactly as
                # a solved shape param does.
                shape_params = _solve_move_params(spec, prod(_shape(st.read.value)))
                shape_params |= st.chosen
                addr = _addr(
                    offset_buffer,
                    [st.read, st.write],
                    shape_params,
                    arity(spec.access_fn),
                )
                emits.append(EmitRecord(st.name, addr, [], st.schedule))
            else:
                spec = isa._ops[st.name].spec
                addr = _addr(
                    st.offset_of,
                    st.reads + [st.write],
                    st.shape_params,
                    arity(spec.access_fn),
                    st.offsets,
                )
                emits.append(EmitRecord(st.name, addr, st.alpha, st.schedule))

        return CompiledProgram(
            isa,
            io,
            emits,
            [(l.offset, _shape(l.value)) for l in inputs],
            [(l.offset, tuple(shape), label) for l, shape, label in outputs],
            [(l.offset, d) for l, d in self.constants],
            [st.pe for st in steps] if any(st.pe for st in steps) else [],
        )


def _wants(pattern, params) -> tuple:
    """The residence one access describes, at these params."""
    return residence(access_map(pattern, params))


def _addr(offset_of, locs, shape_params, n_addr, rel=()) -> list:
    """Fill an instruction's address params. An offset param names one coordinate
    *component* of one operand — ``(buffer position, axis)`` — so a multi-index
    access takes several, all read off that operand's placement. A param naming that
    component in several operands has been forced to one address by ``allocate``, so
    any of its references gives the same number.

    ``rel`` shifts an operand within its location, for a lowering's tile steps: the
    allocator supplies the value's base and the lowering supplies, **per coordinate
    axis**, the offset of the sub-block this instruction touches — a rank-2 operand
    has one shift per axis, and folding them into one number was exactly the bug
    that mis-addressed every off-origin block."""

    def component(i):
        pos, axis = offset_of[i][0]
        shift = rel[pos].get(axis, 0) if rel else 0
        return locs[pos].offset[axis] + shift

    return [component(i) if i in offset_of else shape_params[i] for i in range(n_addr)]


def plan(isa, selection: Selection) -> CompiledProgram:
    """Liveness-driven, buffer-aware allocation over *locations* (see ``_Loc``).

    1. *Schedule* — lower each match to a linear stream of moves + computes,
       inserting data movement (``bring_to``) whenever a value is not resident in
       the buffer an instruction needs it in **and laid out the way that instruction
       reads it**. Routing runs over ``(buffer, residence)`` states, so a repacking
       move is found the same way a copy is; each hop is a short-lived intermediate
       location (**P-C**). Program I/O lives in the global buffer, in the host ABI's
       layout at both ends. Each inserted move is one update of the time-varying λ
       — the delta encoding v2 §3.3 names — which is why the emitted stream, read
       as epochs (``epoch.py``), meets the interface condition by construction.
    2. *Liveness* — def step + last-use step (and the full use list) per location.
    3. *Allocation* — best-fit free-list per buffer, releasing a location at its
       last use so slots are reused; a result coalesces in place onto a dying
       element-wise operand. On overflow a Belady victim (resident, not used at the
       overflow step, farthest next use) is **spilled** to the backing store and
       reloaded before its next use (**P-B**); inserting the spill grows the
       schedule, so liveness + allocation are re-run to a fixpoint.

    Passes 2 and 3 are ``_Planner.finish`` — they are the same wherever the steps
    came from, which is what an imported mapping enters through."""
    p = _Planner(isa)
    block = selection.func.regions[0].blocks[0]
    func_args = list(block.arguments)

    # --- pass 1: schedule (moves + computes) over locations ------------------
    inputs = [p.make_loc(a, p.io, p.abi(a)) for a in func_args]

    for m in selection.matches:
        if m.mapping is not None:
            # An imported lowering: the site is a tile run, so its operands are the
            # source op's tensors rather than the instruction's, and everything about
            # it — the nest, the transfers, the staging — is the mapping's.
            from .mapping import lower_site

            lower_site(m, p)
            continue
        spec = m.instruction.spec
        patterns, _, _ = trace_instruction(spec)
        reads = [
            p.bring_to(
                _canon(v),
                buf,
                _wants(pat, m.shape_params),
                f"{m.instruction.name} operand {i}",
            )
            for i, (v, buf, pat) in enumerate(
                zip(m.operand_values, spec.sources, patterns)
            )
        ]
        write = p.make_loc(
            _canon(m.result_value),
            spec.destinations[0],
            _wants(patterns[len(spec.sources)], m.shape_params),
        )
        if spec.expand_fn is not None:
            p.lower_expansion(m, spec, reads, write)
            continue
        _, offset_of = param_roles(spec)
        p.steps.append(
            _Compute(
                m.instruction.name,
                reads,
                write,
                offset_of,
                m.shape_params,
                _reusable_operands(m.instruction) & _colocatable(m),
                [m.alpha[i] for i in range(len(compute_params(spec)))],
                [m.schedule[n] for n in spec.schedule_domains],
            )
        )

    terminator = list(block.operations)[-1]
    # The host reads a result back densely, so an output must *arrive* in the ABI's
    # layout: a program that computed it repacked needs the relayout inserted here.
    outputs = [
        (
            p.bring_to(_canon(v), p.io, p.abi(_canon(v)), f"result #{i}"),
            # The *terminator's* shape, not the canonical value's: a result the host
            # reads back through a reshape is still that many elements, laid out the
            # way the source's own type says.
            _shape(v),
            f"out{i}",
        )
        for i, v in enumerate(terminator.operands)
    ]
    return p.finish(inputs, outputs)


# ==========================================================================#
# Driver
# ==========================================================================#


def compile_program(source: str, isa, mapping_for=None) -> CompiledProgram:
    """Compile a source program onto ``isa``.

    The source program is a TOSA-dialect MLIR module given as *text* — we generate
    none ourselves; the caller hands us a module string (e.g. from torch_mlir's
    TOSA backend) and we ``Module.parse`` it here. The returned ``CompiledProgram``
    holds only plain data (no IR handles), so the parse context can be dropped.

    ``mapping_for(op) -> (Mapping, Binding) | None`` supplies an externally chosen
    tiling for a source op — the read-back half of an external mapper (Timeloop).
    A callable rather than a table keyed by op, because a TOSA op has no stable
    name to key on and a driver is written once for a whole class of them. Where it
    answers, the op is **not** required to be one instruction's worth of work: its
    shapes derive the iteration domain and the mapping's innermost factors give the
    instruction its own, which is the exact point at which tiling enters the
    backend.
    """
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        normalize_source(module)
        catalog = Catalog(isa)
        selection = match_program(catalog, module, mapping_for)
        solve(selection)
        solve_layouts(isa, selection)
        return plan(isa, selection)

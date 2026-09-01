# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ISA model + trace engine for the DSA frontend.

The user describes an accelerator by declaring buffers and instructions on an
``ISA`` object. Each instruction has two traced regions:

- ``@I.access``: address params -> tuple of access-pattern expressions
  (``PatternExpr`` DAG over ``IndexExpr`` leaves). Pure Python, no MLIR.
- ``@I.compute``: one tensor arg per buffer -> canonical-linalg value(s)
  (``TensorProxy`` DAG). Argument tensor shapes are *inferred* from the access
  patterns' visible shapes, so no annotations are needed.

``codegen.build_catalog`` walks these DAGs to construct ``allo.buffer`` +
``allo.define`` ops.
"""

from __future__ import annotations

import inspect
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Generic, ParamSpec, TypeVar, overload

from ..._mlir import ir
from ...lang.core import DType, Template
from ...lang.kernel import Kernel, KernelOptions
from .errors import AcceleratorDescriptionError, AssemblyError, LayoutError

P = ParamSpec("P")
R = TypeVar("R")

# ==========================================================================#
# ISA buffer element type
# ==========================================================================#


@dataclass
class BufferKind:
    """A buffer's **slot**: what one addressable position holds.

    ``shape`` is the per-slot element shape (``()`` for a word), and the mnemonic
    follows from its rank — ``scalar`` / ``vector`` / ``tile`` — which is how it
    materializes to ``!allo.{mnemonic}<...>`` via ``ir.Type.parse`` (no CAPI),
    reusing the dtype materialization from ``lang.core``.

    A slot says nothing about *how many* positions there are or along how many axes;
    that is the buffer's ``extents``. Off-chip memory is therefore not a kind of its
    own: it is an ordinary buffer with multi-dimensional extents and a scalar slot.
    """

    dtype: DType
    shape: tuple[int, ...] = field(default_factory=tuple)

    @property
    def mnemonic(self) -> str:
        return ("scalar", "vector")[len(self.shape)] if len(self.shape) < 2 else "tile"

    def mlir_text(self, context: ir.Context) -> str:
        elt = self.dtype.materialize(context)  # e.g. prints as "f32"
        if not self.shape:
            return f"!allo.scalar<{elt}>"
        dims = "x".join(str(d) for d in self.shape)
        return f"!allo.{self.mnemonic}<{dims}x{elt}>"

    def materialize(self, context: ir.Context) -> ir.Type:
        return ir.Type.parse(self.mlir_text(context), context)


# ==========================================================================#
# Access-region trace proxies
# ==========================================================================#


class IndexExpr:
    """An index-valued expression over instruction address parameters.

    Leaves are ``param`` (an addr block-arg) or ``const``; ``+`` and ``*`` build
    ``add`` / ``mul`` nodes that lower to ``arith.addi`` / ``arith.muli``.
    """

    __slots__ = ("kind", "value", "param_index", "lhs", "rhs")

    def __init__(self, kind, *, value=None, param_index=None, lhs=None, rhs=None):
        self.kind = kind  # "param" | "const" | "add" | "mul"
        self.value = value
        self.param_index = param_index
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def param(index: int) -> IndexExpr:
        return IndexExpr("param", param_index=index)

    @staticmethod
    def const(value: int) -> IndexExpr:
        return IndexExpr("const", value=int(value))

    def __add__(self, other) -> IndexExpr:
        return IndexExpr("add", lhs=self, rhs=_to_index(other))

    __radd__ = __add__

    def __mul__(self, other) -> IndexExpr:
        return IndexExpr("mul", lhs=self, rhs=_to_index(other))

    __rmul__ = __mul__

    def static_int(self):
        """Return the constant value if this expression is statically known."""
        if self.kind == "const":
            return self.value
        if self.kind == "add":
            l, r = self.lhs.static_int(), self.rhs.static_int()
            return None if l is None or r is None else l + r
        if self.kind == "mul":
            l, r = self.lhs.static_int(), self.rhs.static_int()
            return None if l is None or r is None else l * r
        return None


def _to_index(o) -> IndexExpr:
    return o if isinstance(o, IndexExpr) else IndexExpr.const(int(o))


def _dim(x):
    """Resolve an int | IndexExpr count/shape entry to a concrete int when it is
    statically known, else keep the ``IndexExpr`` so the shape solver can use it."""
    if isinstance(x, int):
        return x
    s = x.static_int()
    return s if s is not None else x


def _infer_reassociation(src: list, dst: list) -> list:
    """Reassociation groups for a reshape between visible shapes ``src`` and ``dst``.

    The group list always indexes the *longer* shape, one group per dim of the
    shorter (matching ``expand``/``collapse`` semantics). Two cases are unambiguous
    even with symbolic dims — either side being 1-D (the flat-run<->tile case), and
    the longer side merely *prepending unit dims* (a batch wrapper: a parametric
    ``[M,K]`` region seen as the ``[1,M,K]`` TOSA batched matmul wants). Otherwise
    dims are product-matched greedily, which requires static sizes."""
    long, short = (dst, src) if len(dst) >= len(src) else (src, dst)
    if len(short) == 1:
        return [list(range(len(long)))]
    extra = len(long) - len(short)
    if extra and all(_dim(d) == 1 for d in long[:extra]):
        # leading unit dims carry no values, so they fold into the first group and the
        # rest pair off positionally — no products, hence no need for static dims.
        return [list(range(extra + 1))] + [[extra + i] for i in range(1, len(short))]
    groups, i = [], 0
    for s in short:
        size = _dim(s)
        if not isinstance(size, int):
            raise AcceleratorDescriptionError(
                "reshape: product-matching needs static dims"
            )
        grp, acc = [i], _dim(long[i])
        i += 1
        while acc != size and i < len(long):
            grp.append(i)
            acc *= _dim(long[i])
            i += 1
        if acc != size:
            raise AcceleratorDescriptionError(
                f"reshape: {src} -> {dst} is not a pure reshape"
            )
        groups.append(grp)
    while i < len(long):  # trailing unit dims fold into the last group
        groups[-1].append(i)
        i += 1
    return groups


def prod_dims(dims):
    """The product of shape entries, staying symbolic when one is an ``IndexExpr``
    (so a parametric tile keeps its shape params)."""
    count = dims[0]
    for d in dims[1:]:
        count = count * d
    return count


def _inverse_perm(order) -> list[int]:
    inv = [0] * len(order)
    for k, d in enumerate(order):
        inv[d] = k
    return inv


def as_permutation(order, rank: int, who: str) -> tuple[int, ...]:
    """Validate an explicit dimension ordering: a permutation of ``range(rank)``."""
    try:
        perm = tuple(int(k) for k in order)
    except TypeError as exc:
        raise AssemblyError(f"{who}: {order!r} is not a dimension ordering") from exc
    if sorted(perm) != list(range(rank)):
        raise AssemblyError(
            f"{who}: {perm} is not a permutation of the {rank} dimension(s) it orders"
        )
    return perm


class PatternExpr:
    """A node in the access-pattern DAG.

    ``strided`` and ``layout`` are rooted at a buffer; ``expand`` / ``collapse`` /
    ``transpose`` wrap a source ``PatternExpr``. An access changes only where data
    *lives* (its address map), never the logical tensor — which is why a ``layout``'s
    dimension ordering belongs here while a ``prim.transpose`` (a different logical
    tensor) belongs in the compute region.

    ``transpose`` is the one exception to "no reordering here" and is not exposed as a
    builder: ``layout`` expands into ``strided -> expand -> transpose`` for codegen,
    where the transpose undoes the storage ordering and hands the compute region its
    operand in logical order.
    """

    def __init__(
        self,
        kind,
        *,
        buffer=None,
        basis=None,
        counts=None,
        strides=None,
        source=None,
        reassociation=None,
        output_shape=None,
        order=None,
        permutation=None,
    ):
        self.kind = kind
        self.buffer = buffer
        self.basis = basis
        self.counts = counts
        self.strides = strides
        self.source = source
        self.reassociation = reassociation
        self.output_shape = output_shape
        self.order = order  # for "layout": the dimension ordering (perm | IndexExpr)
        self.permutation = permutation  # for "transpose": the dim permutation

    def expand_layout(self, order) -> PatternExpr:
        """This ``layout`` node as the equivalent ``strided -> expand -> transpose``
        chain under the concrete dimension ordering ``order``.

        ``order`` lists the logical dims outermost-first, so the data sits in the
        buffer as a dense tensor of shape ``sizes[order]`` and the transpose that
        brings it back to logical order is ``order``'s inverse."""
        sizes = list(self.counts)
        storage = [sizes[k] for k in order]
        run = PatternExpr(
            "strided",
            buffer=self.buffer,
            basis=list(self.basis),
            counts=[prod_dims(storage)],
            strides=[1],
        )
        packed = run if len(storage) == 1 else run.reshape(storage)
        if list(order) == sorted(order):
            return packed  # row-major: storage order *is* logical order
        return PatternExpr("transpose", source=packed, permutation=_inverse_perm(order))

    def reshape(self, *shape) -> PatternExpr:
        """Reinterpret this pattern's visible tensor as ``shape`` — the unified
        access-layer reshape. Adding dims routes to ``expand``, removing them to
        ``collapse``; the reassociation is inferred from the current visible shape
        (one group when either side is 1-D — the common flat-run<->tile case;
        product-matched otherwise). Reshape is value-preserving (a layout view), so
        it lives here, not in ``prim`` (value-reordering relayouts like transpose)."""
        dims = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )
        src = self.visible_shape()
        if len(dims) == len(src):
            return self  # same rank -> identity (a permutation would be a transpose)
        reassoc = _infer_reassociation(src, list(dims))
        if len(dims) > len(src):
            return PatternExpr(
                "expand", source=self, reassociation=reassoc, output_shape=list(dims)
            )
        return PatternExpr("collapse", source=self, reassociation=reassoc)

    def visible_shape(self) -> list:
        """The tensor shape the compute region sees for this operand.

        Mirrors the C++ ``materialize`` semantics: the counts that span a range,
        followed by the slot's own dims. A count of exactly 1 **selects** one slot
        along that address axis rather than spanning a range, so — like numpy's
        ``a[3]`` versus ``a[3:4]`` — it contributes no tensor dimension: ``vld``
        reads one slot of a vector register file and the compute region sees the
        lanes, not a ``1 x lanes`` tensor. ``StridedOp::materialize``'s
        ``rankReduction`` is the same rule on the IR side; the two must agree or
        the inlined semantics get an operand of the wrong rank.

        Entries are ints, an ``IndexExpr`` for a symbolic (parametric) dim, or
        ``None`` for a dim that is dynamic but not solvable here.
        """
        if self.kind == "strided":
            spanning = [c for c in (_dim(c) for c in self.counts) if c != 1]
            return spanning + list(self.buffer.kind.shape)
        if self.kind == "layout":
            # The ordering permutes *storage*, never the operand the compute region
            # sees — which is why an unsolved `order` still has a known shape.
            return [_dim(d) for d in self.counts]
        if self.kind == "transpose":
            src = self.source.visible_shape()
            return [src[p] for p in self.permutation]
        if self.kind == "expand":
            return [_dim(d) for d in self.output_shape]
        if self.kind == "collapse":
            src = self.source.visible_shape()
            out = []
            for group in self.reassociation:
                dims = [src[i] for i in group]
                out.append(None if any(d is None for d in dims) else math.prod(dims))
            return out
        raise NotImplementedError(f"visible_shape for pattern '{self.kind}'")


# ==========================================================================#
# Compute-region trace proxies
# ==========================================================================#


class ScalarProxy:
    """A **computational attribute** (ACT's α): one extra ``@I.compute`` parameter.

    Not a ``TensorProxy`` — it has no shape and never is a DAG node. It stands for a
    scalar *immediate encoded in the instruction word*, so the only place it may appear
    is inside a ``primitive.const``, which broadcasts it to a tensor operand. That
    restriction is the IR's, not a simplification: ``allo.emit`` carries its compute
    params as a ``DenseI64ArrayAttr`` and ``allo.define`` requires the matching block
    args to be int/index — an instruction encoding holds integers.

    An α differs from every other parameter in *who* supplies it. An address param is
    assigned by allocation, a shape param is solved from the source shapes; an α is
    **bound from the source program** — it is the one place the value of a source
    constant flows into the instruction word rather than into memory."""

    def __init__(self, index: int, name: str):
        self.param_index = index  # position among the extra @I.compute params
        self.name = name

    def __repr__(self):
        return f"#{self.name}"


class TensorProxy:
    """A node in the compute DAG. Leaves are ``arg`` (a buffer block arg) or
    ``const`` (a literal baked into the instruction)."""

    def __init__(
        self,
        kind,
        dtype: DType,
        shape,
        *,
        args=(),
        buffer_index=None,
        permutation=None,
        axis=None,
        attrs=None,
        value=None,
    ):
        self.kind = kind  # "arg" | "const" | "identity" | a prim tag (see REGISTRY)
        self.dtype = dtype
        self.shape = tuple(shape)
        self.args = tuple(args)
        self.buffer_index = buffer_index
        self.permutation = permutation  # for "transpose": the dim permutation
        self.axis = axis  # for "reduce_*" / "reverse": the axis
        self.attrs = attrs or {}  # for conv/pool: pad/stride/dilation/kernel
        self.value = value  # for "const": the scalar every element holds


# ==========================================================================#
# ISA model
# ==========================================================================#


@dataclass(eq=False)  # identity-based: a buffer is unique, usable as a dict key
class BufferSpec:
    """A buffer is an **address space times a slot**: ``extents`` addressable
    positions (along one axis or several), each holding one ``kind``."""

    name: str
    extents: tuple[int, ...]  # addressable positions, one entry per address axis
    kind: BufferKind
    is_global: bool = False  # off-chip / main memory: where program I/O lives

    def __getitem__(self, key) -> BufferSlice:
        return BufferSlice(self, key)

    @property
    def slot_size(self) -> int:
        """Elements per slot (1 for scalar buffers)."""
        return math.prod(self.kind.shape) or 1

    @property
    def memref_shape(self) -> list[int]:
        """The shape of this buffer's lowered ``memref.global``. Mirrors
        ``ConvertDeclareBufferOpPattern``: a buffer is its address space times its
        slot, so the memref is exactly ``extents ++ slot shape``."""
        return list(self.extents) + list(self.kind.shape)

    @property
    def address_rank(self) -> int:
        """How many coordinate components address this buffer — one per extent. A
        flat register file takes a single index; a row-major array takes a full
        coordinate, which is what makes a rank-2 access (and so a relayout)
        expressible on it."""
        return len(self.extents)

    @property
    def capacity(self) -> int:
        """Allocatable units along the axis the planner packs: the outermost extent
        (slots for a flat buffer, rows for a 2-D array)."""
        return self.extents[0]


@dataclass
class BufferSlice:
    """A ``buffer[slice]`` selection, used as an ``inspect`` target."""

    buffer: BufferSpec
    key: object


@dataclass
class UnitLatency:
    """One ``@unit``'s issue interval and pipeline depth, in cycles.

    Shared (by reference) with every instruction bound to that unit, so
    ``ISA.bind`` and ``ISA.latency`` may be written in either order."""

    ii: int | None = None
    depth: int | None = None

    @property
    def declared(self) -> bool:
        return self.ii is not None and self.depth is not None


class InstructionSpec:
    def __init__(self, name, sources, destinations, cost=None):
        self.name = name
        self.sources = list(sources)
        self.destinations = list(destinations)
        # Search cost (tree-DP objective). A number, or a callable over the
        # instruction's *shape* params; ``None`` = derive it — see `cost_of`.
        self.cost = cost
        self.unit = None  # the @unit this instruction issues to (ISA.bind)
        self.unit_latency: UnitLatency | None = None  # that unit's (ii, depth)
        self.trips = None  # how many times it occupies the unit (shape callable)
        self.access_fn = None
        self.compute_fn = None
        self.expand_fn = None  # optional @I.expand: lower one match to a tile run
        # optional @I.schedule: the free schedule params and when a choice is legal
        self.schedule_fn = None
        self.schedule_domains: dict = {}  # fresh param name -> the values it may take
        # Domains declared for *access* params, which turns them from solved into
        # chosen. Only a mover's residence params qualify — see `_check_schedule`.
        self.schedule_residence: dict = {}
        # Declared by `ISA.configures`: this instruction *assigns configuration* —
        # it sets machine state a later instruction runs under, rather than running
        # a kernel of its own. Its emitted fields join into the epoch it precedes.
        self.configures = False
        self.doc = None  # the defining function's docstring (carried onto Instruction)

    @property
    def buffers(self) -> list[BufferSpec]:
        return self.sources + self.destinations

    def _over_params(self, fn, shape_params: dict, what: str, chosen: dict = {}):
        """Call ``fn`` with the params it declares *by name*: the solved shape params
        (under their ``@access`` parameter names) plus any ``chosen`` schedule params.
        """
        names = access_names(self)
        bound = {names[i]: v for i, v in shape_params.items()} | chosen
        wanted = list(inspect.signature(fn).parameters)
        missing = [n for n in wanted if n not in bound]
        if missing:
            raise AcceleratorDescriptionError(
                f"{self.name}: {what} needs param(s) {missing}, but only "
                f"{sorted(bound)} are bound — a {what} param must be an @access shape "
                f"param or a declared @schedule param"
            )
        return fn(**{n: bound[n] for n in wanted})

    def trips_at(self, shape_params: dict, chosen: dict = {}) -> float:
        """How many times this instruction occupies its unit at a site whose shape
        params solved to ``shape_params`` (1 unless ``ISA.bind`` declared otherwise)."""
        if self.trips is None:
            return 1.0
        return float(self._over_params(self.trips, shape_params, "trips", chosen))

    def configurations(self, shape_params: dict, free: dict = {}) -> list[tuple]:
        """Every configuration of this instruction's **chosen** params that its
        ``@schedule`` predicate admits at a site whose shape params solved to
        ``shape_params``, each paired with its cost, in declaration order.

        A chosen param is the third kind of instruction parameter. An ``@access`` param
        is *solved* — from the source shapes (Stage 2) or from a neighbour's residence
        (Stage 2b); a compute param (α) is *bound* from a constant in the source, and
        the computed value depends on it. A chosen param is neither: it is **freely
        picked, and the value does not depend on it**. That is what the configuration of
        a schedule-ISA machine carries — a fold factor, a spatial/temporal split, a
        packing — and it reaches the compiler through exactly two channels, legality and
        cost, which are this method's predicate and its pricing.

        ``free`` names *access* params the caller has decided are chosen rather than
        solved, with their domains: a mover's residence params (``mover_domains``),
        which nothing unifies with because the planner is what inserts the move. They
        bind by name like everything else, so one predicate and one cost function see
        fresh schedule params and free residence params alike.

        This is the single place a finite parameter domain is enumerated. The empty
        product yields one empty assignment, so a predicate with no domains at all is
        still evaluated — it then simply restricts which *shapes* are acceptable."""
        domains = free | self.schedule_domains
        return [
            (chosen, self.cost_of(shape_params, chosen))
            for combo in itertools.product(*domains.values())
            if self.admits(shape_params, chosen := dict(zip(domains, combo)), free)
        ]

    def admits(self, shape_params: dict, chosen: dict, free: dict = {}) -> bool:
        """**Check mode**: does this instruction admit the *given* configuration?

        The same domains and the same ``@I.schedule`` predicate that
        ``configurations`` enumerates, applied to one assignment instead of all of
        them: ``chosen`` must bind exactly the declared params, each value must lie
        in its domain, and the predicate must hold. ``configure`` chooses (argmin
        over the enumeration, which filters through this very method); a checker
        checks (this method on a configuration read off the wire). One constraint
        object, two entries — which is what makes verifying an externally supplied
        program the same machinery as compiling one."""
        domains = free | self.schedule_domains
        if set(chosen) != set(domains):
            return False
        if any(chosen[n] not in domain for n, domain in domains.items()):
            return False
        return self.schedule_fn is None or bool(
            self._over_params(self.schedule_fn, shape_params, "schedule", chosen)
        )

    def configure(
        self, shape_params: dict, free: dict = {}
    ) -> tuple[dict, float] | None:
        """The cheapest configuration of this instruction, or ``None`` when the
        ``@schedule`` predicate admits none — the instruction cannot be configured for
        this site, so it is not a candidate there. That is how a machine states a limit
        it has no other way to state: an array too small for the requested fold, a field
        too narrow for the requested count. Ties go to the earlier-declared assignment.
        """
        return min(
            self.configurations(shape_params, free), key=lambda c: c[1], default=None
        )

    def cost_of(self, shape_params: dict, chosen: dict = {}) -> float:
        """This instruction's search cost at a site whose shape params solved to
        ``shape_params`` (``{access-param index -> size}``, from Stage 2). Resolved
        in three steps, most specific first:

        1. **A declared ``cost``.** A constant is shape-independent — right for a
           fixed-size instruction, wrong for a parametric one: a layer-level op that
           ``@expand``s into ``M/TILE`` tiles is that much more expensive than one
           tile, and a constant would leave the DP unable to compare the two. So
           ``cost`` may instead be a callable declaring the shape params it needs by
           name, e.g. ``cost=lambda M: M // TILE``.
        2. **The bound unit's latency** (``ISA.bind`` + ``ISA.latency``):
           ``depth + ii * trips(shape_params)`` — a *cycle count*, derived from the
           microarchitecture rather than hand-assigned. This is the preferred form
           when the ISA has a modeled microarch.
        3. **1.0**, i.e. minimize instruction count."""
        if self.cost is not None:
            if not callable(self.cost):
                return float(self.cost)
            return float(self._over_params(self.cost, shape_params, "cost", chosen))
        if self.unit_latency is not None and self.unit_latency.declared:
            lat = self.unit_latency
            return lat.depth + lat.ii * self.trips_at(shape_params, chosen)
        return 1.0


class InstructionBuilder:
    """The ``I`` handed to an ``@instruction`` body; registers the two traced
    regions onto the spec."""

    def __init__(self, spec: InstructionSpec):
        self.spec = spec

    def access(self, fn):
        self.spec.access_fn = fn
        return fn

    def compute(self, fn):
        self.spec.compute_fn = fn
        return fn

    def schedule(self, fn=None, **domains):
        """Declare this instruction's free **schedule params** and when a choice of
        them is legal.

        The keyword arguments name the params and give each a finite domain
        (``@I.schedule(fold=[1, 2, 4])``); the body is a *predicate* over the
        ``@access`` shape params and the chosen values, and returns whether that
        configuration is one the hardware can actually run. ``core.InstructionSpec
        .configure`` picks the cheapest admitted assignment, and an instruction that
        admits none is not a candidate at that site.

        This is ACT's ``e_theta`` — the validity constraint a field width or an array
        dimension imposes — and it is the only place an ISA can state a limit that is
        neither a shape nor a value. Declaring no domains is allowed and useful: the
        predicate then simply restricts which *shapes* the instruction accepts.

        A domain may also name an existing ``@access`` param, which does not introduce
        a param but **turns that one from solved into chosen**. Only a mover's residence
        params qualify (a shape param is pinned by the source program, an offset by the
        allocator, and a matched instruction's residence by unification), and that is
        what makes a data-movement instruction with a free stride expressible — the
        author supplies the domain the frontend has no way to invent. An ordering param
        needs no declaration at all: its domain is intrinsic, and declaring one narrows
        it."""

        def decorator(fn):
            self.spec.schedule_fn = fn
            self.spec.schedule_domains = {n: tuple(v) for n, v in domains.items()}
            return fn

        return decorator(fn) if fn is not None else decorator

    def expand(self, fn):
        self.spec.expand_fn = fn
        return fn


def _bind_call(names, args, kwargs, who):
    if len(args) > len(names):
        raise AssemblyError(f"{who}: too many positional args")
    bound = dict(zip(names, args))
    for k, v in kwargs.items():
        if k not in names:
            raise AssemblyError(f"{who}: unknown parameter '{k}'")
        if k in bound:
            raise AssemblyError(f"{who}: parameter '{k}' given twice")
        bound[k] = v
    missing = [n for n in names if n not in bound]
    if missing:
        raise AssemblyError(f"{who}: missing parameters {missing}")
    return [bound[n] for n in names]


class Instruction(Generic[P, R]):
    """A callable assembler mnemonic. Calling it inside an ``@oracle`` body
    records one ``allo.emit``. Parameter names come from the ``@I.access``
    signature (address params) plus any extra ``@I.compute`` params.

    Generic in ``[P, R]`` (like ``Kernel``) so a call site type-checks against the
    instruction's signature; the defining function's ``__doc__`` / ``__name__`` are
    copied on so hovering a call still surfaces the instruction's documentation."""

    def __init__(self, isa: ISA, spec: InstructionSpec):
        self.isa = isa
        self.spec = spec
        self.name = spec.name
        self.addr_params = access_names(spec)
        self.compute_params = compute_params(spec)
        self.layout_params = dict(layout_params(spec))  # addr index -> ordered rank
        self.__name__ = spec.name
        self.__doc__ = spec.doc

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        program = self.isa._active_oracle
        if program is None:
            raise AssemblyError(
                f"instruction '{self.name}' can only be called inside @oracle"
            )
        names = self.addr_params + self.compute_params
        bound = _bind_call(names, args, kwargs, self.name)
        n = len(self.addr_params)
        # An ordering param's argument is a permutation, not an address: it selects
        # how the operand is packed, so it is checked here rather than emitted blind.
        addr = [
            (
                as_permutation(v, rank, f"{self.name}: {names[i]}")
                if (rank := self.layout_params.get(i)) is not None
                else v
            )
            for i, v in enumerate(bound[:n])
        ]
        return program.record_emit(self.name, addr, bound[n:])

    def __repr__(self):
        return f"Instruction<{self.name}>"


# ==========================================================================#
# What an ``@I.expand`` body names: values and offsets into them, not addresses
# ==========================================================================#


@dataclass(eq=False)
class Tile:
    """A value an expansion *invents*: staging the lowering needs and the source
    program never mentions.

    It carries a shape and nothing else — where it lives is the allocator's
    answer. That is the whole point of naming one: an expansion used to run after
    allocation, so it had to pick its own addresses, which meant reserving a
    corner of a buffer by hand and trusting that nothing else reached it.

    Identity, not value, distinguishes two tiles (``eq=False``): two tiles of the
    same shape are two different values, and merging them would silently share a
    slot."""

    shape: tuple


@dataclass(frozen=True)
class Ref:
    """An address an expansion body names: a value, plus an element offset into it.

    The body computes offsets **within** a value (``ifm + n * image``) and the
    allocator supplies that value's base, which is what lets one body text work
    against locations rather than hand-picked addresses. ``value`` is either a
    ``Tile`` the body made or the planner's own location for one of the expanding
    instruction's operands."""

    value: object
    offset: int = 0

    def __add__(self, n: int) -> "Ref":
        return Ref(self.value, self.offset + n)

    __radd__ = __add__


def scratch(shape) -> Ref:
    """A fresh staging tile for an ``@I.expand`` body — ``shape`` in elements.

    Its buffer and address are decided by the instructions that use it: the buffer
    from where they read and write it, the residence from what their access
    patterns describe, and the address by ordinary allocation over its live
    range."""
    return Ref(Tile(tuple(_as_list(shape))))


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


class ISA:
    """Registry of buffers + instructions for one accelerator."""

    def __init__(self, name: str):
        self.name = name
        self.buffers: dict[str, BufferSpec] = {}
        self.instructions: list[InstructionSpec] = []
        self._ops: dict[str, object] = {}  # Instruction by name
        self._active_oracle = None  # the OracleProgram currently being traced
        self.kernels: list[Kernel] = []  # every @unit / @entry kernel
        self.top: Kernel | None = None  # the unique @entry kernel
        self.latencies: dict[str, UnitLatency] = {}  # unit name -> (ii, depth)
        # The interconnect (ISA.network): R as a predicate over σ's spatial axis,
        # and the instruction that combines partial sums across instances. Both
        # ``None`` = undeclared, which is what every operation ISA is.
        self.reaches = None
        self.reduces = None

    # --- memory hierarchy declarations ---
    def buffer(self, name, extents, dtype: DType, *, slot=(), is_global=False):
        """Declare a buffer: ``extents`` addressable positions holding one ``slot``
        each. Every other declaration below is sugar over this one.

        ``extents`` is a tuple with one entry per address axis — ``(8192,)`` is a flat
        scratchpad, ``(1024, 64)`` a row-major array whose access takes a full
        coordinate (which is what makes a relayout expressible). ``slot`` is the
        element shape of one position: ``()`` a word, ``(16,)`` a vector register,
        ``(4, 4)`` a tile. The two are independent — a 2-D array *of vector registers*
        is an ordinary declaration, not a special case."""
        if name in self.buffers:
            raise AcceleratorDescriptionError(f"duplicate buffer '{name}'")
        spec = BufferSpec(
            name, tuple(extents), BufferKind(dtype, tuple(slot)), is_global
        )
        self.buffers[name] = spec
        return spec

    def global_(self, name, shape, dtype: DType) -> BufferSpec:
        """Off-chip / main memory as a flat, word-addressable pool. Program I/O is
        marshalled into the global buffer; ``shape`` is flattened, so the host hands
        over one linear array."""
        return self.buffer(name, (math.prod(shape),), dtype, is_global=True)

    def scalar(self, name, slots, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` word-addressable (scalar) entries."""
        return self.buffer(name, (slots,), dtype)

    def vector(self, name, slots, shape, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` entries, each a 1-D vector of ``shape``."""
        return self.buffer(name, (slots,), dtype, slot=shape)

    def tile(self, name, slots, shape, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` entries, each an N-D tile of ``shape``."""
        return self.buffer(name, (slots,), dtype, slot=shape)

    def hbm(self, name, shape, dtype: DType, *, is_global=False) -> BufferSpec:
        """Off-chip memory addressed as one N-D array of ``shape`` — a buffer whose
        extents are ``shape`` and whose slot is a word.

        Named after the memory, not a distinct kind: an access indexes it with a full
        coordinate exactly as it would any multi-dimensional buffer. That is what a
        rearranging DMA needs — a 2-D block of a row-major matrix is a rank-2
        ``strided``, and no 1-D pattern describes it. ``is_global`` makes it the buffer
        program I/O is marshalled into, so the host hands over N-D arrays."""
        return self.buffer(name, shape, dtype, is_global=is_global)

    # --- instruction declaration ---
    # --- microarchitecture binding (instruction -> @unit, and that unit's latency) ---
    def _unit_latency(self, unit: Kernel) -> UnitLatency:
        if unit not in self.kernels:
            raise AcceleratorDescriptionError(
                f"ISA '{self.name}': '{getattr(unit, 'func_name', unit)}' is not a "
                f"@unit / @entry of this ISA"
            )
        return self.latencies.setdefault(unit.func_name, UnitLatency())

    def bind(self, instruction: Instruction, unit: Kernel, *, trips=None) -> None:
        """Declare that ``instruction`` issues to hardware ``unit``.

        This is the link between the ISA (what an instruction means) and the
        microarchitecture (what runs it) — data, rather than the opcode-naming
        convention a hand-written decoder relies on. ``trips`` is an optional callable
        over the instruction's shape params giving how many times it occupies the unit
        (default 1): a burst mover is ``trips=lambda n: n``, a layer-level op that
        expands into ``M/TILE`` tiles is ``trips=lambda M: M // TILE``.

        Several instructions may share one unit (an opcode-multiplexed ALU); the
        unit's ``latency`` is then declared once for all of them."""
        spec = instruction.spec
        if spec not in self.instructions:
            raise AcceleratorDescriptionError(
                f"ISA '{self.name}': '{spec.name}' is not an instruction of this ISA"
            )
        if spec.unit is not None:
            raise AcceleratorDescriptionError(
                f"{spec.name}: already bound to '{spec.unit.func_name}' — an "
                f"instruction issues to exactly one unit"
            )
        spec.unit = unit
        spec.unit_latency = self._unit_latency(unit)
        spec.trips = trips

    def network(self, *, reaches=None, reduces: Instruction | None = None) -> None:
        """Declare the interconnect between spatial instances — the machine half of
        constraints 4 and 7, which have had no declaration to check against.

        ``reaches(producer_pe, consumer_pe) -> bool`` is **R**: whether a value one
        instance writes can be read by another. The arguments are σ's own spatial
        names (``"mac#0"``), so the relation is stated where σ states position and
        nowhere else. Undeclared, R is total — precisely the operation-ISA case,
        where every producer and consumer meet in a shared memory and R drops out
        of the model. ``CompiledProgram.check`` then quantifies every RAW pair over
        it without the caller having to remember.

        ``reduces`` names the instruction that **combines partial sums across
        instances** — a reducing network (BIRRD's), which in this vocabulary is not
        a flag but an instruction: it reads a partial and the accumulator and writes
        their sum, so it is an ordinary compute instruction with an ordinary compute
        region, and the functional oracle executes it like any other. That is the
        difference between declaring a network and asserting one. An imported
        mapping may fan a reduction rank across instances exactly when this exists
        (``mapping.Mapping.check`` refuses it otherwise), and the drain of each
        instance's partial is issued with it instead of a copy."""
        if reduces is not None:
            spec = reduces.spec
            if spec not in self.instructions:
                raise AcceleratorDescriptionError(
                    f"ISA '{self.name}': '{spec.name}' is not an instruction of this "
                    f"ISA"
                )
            if len(spec.sources) != 2 or len(spec.destinations) != 1:
                raise AcceleratorDescriptionError(
                    f"{spec.name}: a reducing transfer reads the partial and the "
                    f"accumulator and writes their combination — two sources and one "
                    f"destination, not {len(spec.sources)} and "
                    f"{len(spec.destinations)}"
                )
            self.reduces = reduces
        if reaches is not None:
            self.reaches = reaches

    def configures(self, instruction: Instruction) -> None:
        """Declare that ``instruction`` **assigns configuration rather than running a
        kernel** — MINISA's ``Set*VNLayout`` before an ``ExecuteMapping``.

        This is the machine-side declaration that gives a class its *epoch
        granularity* (v2 §3.2). An epoch is a segment ending at the instruction that
        runs the kernel, under the configuration folded from every update before it;
        every machine compiled here so far sits at the operation end, where each
        instruction's own assignment is already total, nothing is installed, and an
        epoch is one instruction.

        Declaring an instruction as configuring says it writes configuration
        **registers**: what it assigns is *installed*, and every later epoch runs
        under it until some instruction assigns the same field again. Two
        consequences worth stating, because both are what a machine actually does
        and neither was expressible while an epoch's configuration was a ⊔ of its
        own instructions: **one setter configures many runs** (MINISA installs a
        layer's layouts once and then issues a run of ``ExecuteMapping``s), and
        **reconfiguring is legal** (writing the same field again is a layer
        boundary, where a join would have called it a contradiction).

        The setter still joins the segment it precedes, so σ places the pair
        together and its own write is one event at one point in the stream — the
        effect persists, the event does not repeat.

        The declaration is about *meaning*, not cost: a configuring instruction's
        own issue time is not modelled separately, because it is part of the epoch
        it configures. A machine whose configuration write costs cycles states that
        in the executing unit's ``ii``."""
        spec = instruction.spec
        if spec not in self.instructions:
            raise AcceleratorDescriptionError(
                f"ISA '{self.name}': '{spec.name}' is not an instruction of this ISA"
            )
        spec.configures = True

    def latency(self, unit: Kernel, *, ii: int, depth: int) -> None:
        """Declare ``unit``'s issue interval and pipeline depth, in cycles. An
        instruction bound to it then costs ``depth + ii * trips`` — a cycle estimate
        derived from the microarchitecture rather than a hand-assigned weight.

        ``(ii, depth)`` rather than one number because a unit's latency is usually
        *not* a constant: a mover's cycle count scales with the block it copies. It is
        also the pair a synthesis report yields (``pipeline_ii`` / ``pipeline_depth``
        in ``allo.backend.vitis.report``), so a measured table can replace an authored
        one without an API change."""
        lat = self._unit_latency(unit)
        lat.ii, lat.depth = int(ii), int(depth)

    def instruction(self, src, dst, *, name=None, cost: float | Callable | None = None):
        """Decorate ``def <mnemonic>(I): ...`` -> a callable ``Instruction``.

        The decorated function's name is the mnemonic, so the returned object
        binds to that name and can be called bare inside an ``@oracle``.
        ``cost`` is the search objective weight (default 1 = minimize op count); for a
        parametric instruction it may be a callable over its shape params — see
        ``InstructionSpec.cost_of``.
        """

        def decorator(fn) -> Instruction:
            spec = InstructionSpec(
                name or fn.__name__, _as_list(src), _as_list(dst), cost
            )
            spec.doc = fn.__doc__
            fn(InstructionBuilder(spec))
            if spec.access_fn is None:
                raise AcceleratorDescriptionError(f"{spec.name}: missing @I.access")
            if spec.compute_fn is None:
                raise AcceleratorDescriptionError(f"{spec.name}: missing @I.compute")
            if spec.schedule_fn is not None:
                _check_schedule(spec)
            if spec.expand_fn is not None:
                # @expand is called with the instruction's solved address params, so
                # it must take exactly the @access signature.
                access_params = access_names(spec)
                expand_params = list(inspect.signature(spec.expand_fn).parameters)
                if expand_params != access_params:
                    raise AcceleratorDescriptionError(
                        f"{spec.name}: @expand takes {expand_params} but must take "
                        f"the @access signature {access_params}"
                    )
            self.instructions.append(spec)
            op = Instruction(self, spec)
            self._ops[op.name] = op
            return op

        return decorator

    # --- kernel declaration (@unit worker, @entry top) ---
    def _define(self, args, mapping, options, definition_scope, *, top):
        """Shared body for ``unit`` / ``entry``: build a ``Kernel`` (the exact same
        machinery as ``@allo.kernel``) and register it on this ISA."""
        if len(args) == 1 and callable(args[0]) and not isinstance(args[0], Template):
            fn, template = args[0], ()
        else:
            fn, template = None, args
            if not all(isinstance(a, Template) for a in template):
                raise AcceleratorDescriptionError(
                    f"@unit/@entry: expected Template arguments, got {template!r}"
                )

        def decorator(fn) -> Kernel:
            k = Kernel(
                fn,
                mapping=mapping,
                options=options,
                template=template,
                definition_scope=definition_scope,
            )
            self.kernels.append(k)
            if top:
                if self.top is not None:
                    raise AcceleratorDescriptionError(
                        f"ISA '{self.name}': @entry already defined as "
                        f"'{self.top.func_name}'"
                    )
                self.top = k
            return k

        return decorator(fn) if fn is not None else decorator

    @overload
    def unit(self, fn: Callable[P, R]) -> Kernel[P, R]: ...
    @overload
    def unit(
        self,
        *template: Template,
        mapping: Sequence = (),
        options: KernelOptions = KernelOptions(),
    ) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...
    def unit(self, *args, mapping=(), options=KernelOptions()):
        """A worker kernel. Identical surface to ``@allo.kernel`` (templates /
        ``mapping`` / ``options``); the resulting ``Kernel`` is appended to
        ``self.kernels``."""
        definition_scope = inspect.currentframe().f_back.f_locals.copy()
        return self._define(args, mapping, options, definition_scope, top=False)

    @overload
    def entry(self, fn: Callable[P, R]) -> Kernel[P, R]: ...
    @overload
    def entry(
        self,
        *template: Template,
        options: KernelOptions = KernelOptions(),
    ) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...
    def entry(self, *args, options=KernelOptions()):
        """The unique top-level kernel: like ``@unit`` but with no dataflow
        ``mapping``. Registered as ``self.top`` (and in ``self.kernels``)."""
        definition_scope = inspect.currentframe().f_back.f_locals.copy()
        return self._define(args, (), options, definition_scope, top=True)

    def schedule(self):
        """Schedule the ``@entry`` top kernel (sugar for ``self.top.schedule()``)."""
        if self.top is None:
            raise AcceleratorDescriptionError(
                f"ISA '{self.name}' defines no @entry top to schedule"
            )
        return self.top.schedule()

    # --- oracle (functional simulation of hand-written assembly) ---
    def inspect(self, target, *, label=None):
        """Record a snapshot of ``target`` (a buffer or ``buffer[slice]``) at the
        current program point."""
        if self._active_oracle is None:
            raise AssemblyError("inspect can only be called inside @oracle")
        if isinstance(target, BufferSpec):
            buf, sl = target, None
        elif isinstance(target, BufferSlice):
            buf, sl = target.buffer, target.key
        else:
            raise AssemblyError(
                f"cannot inspect {target!r}; expected a buffer or slice"
            )
        self._active_oracle.record_inspect(buf, sl, label)

    def oracle(self, fn=None, **config):
        """Decorate a hand-written assembly function into a runnable simulation.

        Usable bare (``@oracle``) or configured (``@oracle(init=...)``).
        Calling the result traces the body, builds ``func @main``, and runs the
        functional simulator, returning a dict of inspected arrays.
        """
        from .oracle import Oracle, OracleConfig

        def make(f):
            return Oracle(self, f, OracleConfig(**config))

        return make(fn) if fn is not None else make

    def catalog(self):
        """Build the ``allo.buffer`` + ``allo.define`` catalog module."""
        from .codegen import build_catalog

        return build_catalog(self)

    def compile_program(self, source: str, mapping_for=None):
        """Compile a source program (a TOSA-dialect MLIR module given as *text*)
        onto this ISA, returning a runnable ``CompiledProgram``.

        ``mapping_for`` reads back an externally chosen tiling for a source op; see
        ``search.compile_program``."""
        from .search import compile_program

        return compile_program(source, self, mapping_for)


def arity(fn) -> int:
    return len(inspect.signature(fn).parameters)


def access_names(spec: InstructionSpec) -> list[str]:
    """The ``@I.access`` parameter names, in address-slot order — the one mapping
    between an address param's *index* (how the emit and the solvers key it) and its
    *name* (how a predicate, a cost function or an assembler call refers to it)."""
    return list(inspect.signature(spec.access_fn).parameters)


def access_patterns(spec: InstructionSpec) -> list[PatternExpr]:
    """The instruction's access patterns, traced over symbolic address params."""
    params = [IndexExpr.param(i) for i in range(arity(spec.access_fn))]
    patterns = spec.access_fn(*params)
    return list(patterns) if isinstance(patterns, (tuple, list)) else [patterns]


def _access_chain(spec: InstructionSpec):
    """Per access: ``(buffer position, the value-preserving relayouts stripped to reach
    the root, the buffer-rooted node)``.

    The one traversal the param classifiers share, so they cannot drift apart about
    what counts as a root."""
    for pos, p in enumerate(access_patterns(spec)):
        node, stripped = p, []
        while node.kind in ("expand", "collapse", "transpose"):
            stripped.append(node)
            node = node.source
        yield pos, stripped, node


def layout_params(spec: InstructionSpec) -> list[tuple[int, int]]:
    """``(access-param index, rank)`` for each dimension-ordering param, in parameter
    order. The rank is how many dims that ordering permutes, i.e. the size of its
    finite domain's alphabet — so the domain is that many factorial."""
    found: dict[int, int] = {}
    for _pos, _stripped, node in _access_chain(spec):
        if node.kind == "layout" and isinstance(node.order, IndexExpr):
            rank = len(node.counts)
            if found.setdefault(node.order.param_index, rank) != rank:
                raise AcceleratorDescriptionError(
                    f"{spec.name}: ordering param p{node.order.param_index} orders "
                    f"{found[node.order.param_index]} dims in one access and {rank} "
                    f"in another — one ordering permutes one set of dims"
                )
    return sorted(found.items())


# A catalog lists the configurations the hardware *has*; past a full S_4 it would be
# a search space instead, and so would the movement graph built from the same domains.
_VARIANT_LIMIT = 24


def _order_domains(spec: InstructionSpec) -> dict[int, tuple]:
    """Each ordering param's own domain — the ``rank!`` permutations it may take —
    keyed by access-param index, bounded so the product stays enumerable."""
    domains = {
        i: tuple(itertools.permutations(range(rank))) for i, rank in layout_params(spec)
    }
    total = math.prod(len(d) for d in domains.values())
    if total > _VARIANT_LIMIT:
        raise AcceleratorDescriptionError(
            f"{spec.name}: its {len(domains)} ordering param(s) have {total} "
            f"combinations, over the {_VARIANT_LIMIT} this frontend enumerates"
        )
    return domains


def order_assignments(spec: InstructionSpec) -> list[dict]:
    """Every assignment of this instruction's ordering params, as ``{access-param
    index -> permutation}``, in parameter order. What a catalog specializes into one
    ``allo.define`` apiece."""
    combos: list[dict] = [{}]
    for i, domain in _order_domains(spec).items():
        combos = [c | {i: p} for c in combos for p in domain]
    return combos


def mover_domains(spec: InstructionSpec) -> dict[str, tuple]:
    """The residence params a **mover** chooses, as ``name -> domain``.

    On a matched instruction a stride or a dimension ordering is *solved* — unified
    against the maps the value's other accesses describe (Stage 2b). A mover unifies
    with nothing, because the planner is what inserts it, so the same params are
    *chosen* instead, and choosing needs something to choose from. An ordering's domain
    is intrinsic (its ``rank!`` permutations); a stride's is not, so ``@I.schedule``
    must declare it. A declared domain wins over the intrinsic one, which is how a
    machine says its permutation network cannot reach every packing."""
    names = access_names(spec)
    intrinsic = {names[i]: d for i, d in _order_domains(spec).items()}
    return intrinsic | spec.schedule_residence


def is_mover(spec: InstructionSpec) -> bool:
    """Whether this instruction is *data movement*: one source, one destination, an
    identity compute. The planner inserts these itself (routing, spilling), which is
    exactly what makes their residence params the compiler's to choose."""
    if len(spec.sources) != 1 or len(spec.destinations) != 1:
        return False
    _, _, results = trace_instruction(spec)
    return len(results) == 1 and results[0].kind == "identity"


def compute_params(spec: InstructionSpec) -> list[str]:
    """The names of ``@I.compute``'s extra (non-buffer) parameters — the
    instruction's computational attributes (ACT's α). One truth for the tracer, the
    codegen and the assembler call signature.

    A ``*args`` body has none: it absorbs the buffers and there is no name left to
    hand a ``ScalarProxy`` to."""
    params = list(inspect.signature(spec.compute_fn).parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return []
    if len(params) < len(spec.buffers):
        raise AcceleratorDescriptionError(
            f"{spec.name}: compute takes {[p.name for p in params]} but its first "
            f"{len(spec.buffers)} parameter(s) must be the src+dst buffers"
        )
    return [p.name for p in params[len(spec.buffers) :]]


def _check_schedule(spec: InstructionSpec) -> None:
    """Check an ``@I.schedule`` declaration, and split it.

    A domain naming an ``@access`` param does not introduce a param — it **turns that
    one from solved into chosen**, which is only meaningful for a mover's residence
    params (``mover_domains``). Every other name is a fresh schedule param. The
    predicate may read either, plus the access shape params."""
    names = access_names(spec)
    roles, _ = param_roles(spec)
    for name, domain in list(spec.schedule_domains.items()):
        if not domain:
            raise AcceleratorDescriptionError(
                f"{spec.name}: schedule param '{name}' has an empty domain, so no "
                f"configuration of this instruction could ever be legal"
            )
        if name not in names:
            continue
        role = roles[names.index(name)]
        if role not in ("stride", "layout"):
            raise AcceleratorDescriptionError(
                f"{spec.name}: '{name}' is a '{role}' access param, which the compiler "
                f"does not choose — a shape param is pinned by the source program and "
                f"an offset param by the allocator. Only a residence param (a stride "
                f"or a dimension ordering) can be given a domain"
            )
        if not is_mover(spec):
            raise AcceleratorDescriptionError(
                f"{spec.name}: '{name}' is a residence param of an instruction that is "
                f"*matched*, so it is solved by unifying the maps of the value's "
                f"accesses (Stage 2b) and a domain would go unused. Only a "
                f"data-movement instruction, which the planner inserts and which "
                f"therefore unifies with nothing, chooses its own residence"
            )
        if role == "layout":
            rank = dict(layout_params(spec))[names.index(name)]
            domain = tuple(
                as_permutation(v, rank, f"{spec.name}: {name}") for v in domain
            )
        spec.schedule_residence[name] = domain
        del spec.schedule_domains[name]
    unknown = [
        n
        for n in inspect.signature(spec.schedule_fn).parameters
        if n not in names and n not in spec.schedule_domains
    ]
    if unknown:
        raise AcceleratorDescriptionError(
            f"{spec.name}: @schedule reads {unknown}, which is neither an @access "
            f"param nor one of its declared schedule params "
            f"{sorted(spec.schedule_domains)}"
        )


def trace_instruction(spec: InstructionSpec):
    """Trace an instruction's access + compute regions into Python DAGs.

    Returns ``(patterns, arg_shapes, results)``: one ``PatternExpr`` per buffer,
    the inferred visible shape per buffer (ints, ``None`` for dynamic), and the
    yielded ``TensorProxy`` list. Shared by codegen (IR construction) and the
    search backend (semantic tag + symbolic shape), so both see one truth.

    The two regions are traced independently, so this is also where they are
    reconciled: the compute must yield one value per destination, each shaped like
    what that destination's access pattern writes. Neither region can catch a
    mismatch alone.
    """
    patterns = access_patterns(spec)
    if len(patterns) != len(spec.buffers):
        raise AcceleratorDescriptionError(
            f"{spec.name}: access yields {len(patterns)} patterns, "
            f"expected {len(spec.buffers)} (one per src+dst buffer)"
        )
    arg_shapes = [p.visible_shape() for p in patterns]
    comp_args = [
        TensorProxy("arg", buf.kind.dtype, shape, buffer_index=i)
        for i, (buf, shape) in enumerate(zip(spec.buffers, arg_shapes))
    ]
    comp_args += [ScalarProxy(i, n) for i, n in enumerate(compute_params(spec))]
    results = spec.compute_fn(*comp_args)
    results = list(results) if isinstance(results, (tuple, list)) else [results]
    _check_destinations(spec, arg_shapes, results)
    return patterns, arg_shapes, results


def _check_destinations(spec, arg_shapes, results) -> None:
    """The compute's yielded shapes must match what the destination accesses write.

    Compared dim by dim, and only where *both* sides are statically known — a
    parametric dim is solved per call site (Stage 2), so it can legitimately be
    anything here."""
    if len(results) != len(spec.destinations):
        raise AcceleratorDescriptionError(
            f"{spec.name}: compute must yield {len(spec.destinations)} value(s) "
            f"(one per destination buffer), got {len(results)}"
        )
    for k, result in enumerate(results):
        want = arg_shapes[len(spec.sources) + k]
        got = list(result.shape)
        dst = spec.destinations[k].name
        if len(got) != len(want):
            raise AcceleratorDescriptionError(
                f"{spec.name}: compute yields rank-{len(got)} {got} but the access "
                f"pattern for destination '{dst}' writes rank-{len(want)} {want}"
            )
        for axis, (g, w) in enumerate(zip(got, want)):
            gi = g if isinstance(g, int) else g.static_int()
            wi = w if isinstance(w, int) else w.static_int()
            if gi is not None and wi is not None and gi != wi:
                raise AcceleratorDescriptionError(
                    f"{spec.name}: compute yields {got} but the access pattern for "
                    f"destination '{dst}' writes {want} (axis {axis}: {gi} vs {wi})"
                )


# ==========================================================================#
# Access as an affine index map
# ==========================================================================#


def dense_strides(sizes, order=None) -> list[int]:
    """Suffix-product strides for ``sizes``, its dims packed outermost-first in
    ``order`` (row-major when omitted — the host ABI's own packing, and the one every
    layout in this frontend is a permutation of)."""
    strides, acc = [0] * len(sizes), 1
    for k in reversed(range(len(sizes)) if order is None else list(order)):
        strides[k] = acc
        acc *= sizes[k]
    return strides


def buffer_weights(buf: BufferSpec) -> list[int]:
    """The element stride of each axis of a buffer's flat storage: its memref packed
    row-major."""
    return dense_strides(buf.memref_shape)


def _resolve(x, params: dict):
    """An int | ``IndexExpr`` access entry -> its value under ``params``, or ``None``
    while it still depends on an unsolved param."""
    if isinstance(x, int):
        return x
    if x.kind == "const":
        return x.value
    if x.kind == "param":
        v = params.get(x.param_index)
        return v if isinstance(v, int) else None
    lhs, rhs = _resolve(x.lhs, params), _resolve(x.rhs, params)
    if lhs is None or rhs is None:
        return None
    return lhs + rhs if x.kind == "add" else lhs * rhs


def access_map(p: PatternExpr, params: dict) -> list[tuple]:
    """The access as an affine index map: ``(size, stride)`` per visible dimension,
    strides in **elements** relative to the access's own base.

    This is what makes two accesses comparable. A value's residence is its map, so a
    producer and a consumer describe the same data iff their maps agree — which is
    also the only thing that can pin a dimension ordering, since an ordering never
    shows up in a visible *shape*. A stride is ``None`` while it depends on an
    unsolved param; ``params`` maps access-param index -> solved value (an int, or a
    permutation tuple for an ordering param)."""
    if p.kind == "strided":
        weight = buffer_weights(p.buffer)
        out = []
        for axis, count in enumerate(p.counts):
            n = _resolve(count, params)
            if n == 1:
                continue  # selects one slot along this axis -> no tensor dim
            stride = _resolve(p.strides[axis], params)
            out.append((n, None if stride is None else stride * weight[axis]))
        n_addr = len(p.counts)
        for k, d in enumerate(p.buffer.kind.shape):
            out.append((d, weight[n_addr + k]))
        return out
    if p.kind == "layout":
        sizes = [_resolve(d, params) for d in p.counts]
        order = p.order
        if isinstance(order, IndexExpr):
            order = params.get(order.param_index)
        if order is None or any(s is None for s in sizes):
            return [(s, None) for s in sizes]
        return list(zip(sizes, dense_strides(sizes, order)))
    if p.kind == "transpose":
        src = access_map(p.source, params)
        return [src[k] for k in p.permutation]
    if p.kind == "expand":
        src = access_map(p.source, params)
        shape = [_resolve(d, params) for d in p.output_shape]
        out: list = [None] * len(shape)
        for d, group in enumerate(p.reassociation):
            stride, acc = src[d][1], 1
            for i in reversed(group):  # dims within a group are packed row-major
                out[i] = (shape[i], None if stride is None else stride * acc)
                acc = None if acc is None or shape[i] is None else acc * shape[i]
        return out
    if p.kind == "collapse":
        src = access_map(p.source, params)
        out = []
        for group in p.reassociation:
            dims = [src[i] for i in group]
            size = 1
            for s, _stride in dims:
                size = None if size is None or s is None else size * s
            out.append((size, dims[-1][1]))  # merged dims share the innermost stride
        return out
    raise NotImplementedError(f"access_map for pattern '{p.kind}'")


def residence(m: list[tuple]) -> tuple:
    """A map as a *residence*: which addresses hold the value, hashable and comparable.

    Just the dims that span a range — a size-1 dim holds one element whatever its
    stride, so it says nothing about where the data is, and dropping it is what makes a
    torch-batched ``1xMxN`` access comparable with a plain ``MxN`` one (the rank alias
    ``_align_ranks`` looks through when solving shapes). Nothing more is needed: an
    access's visible shape has to match the value's (Stage 2), so every access of one
    value already reaches it at one rank, and the host ABI's map is built at that rank.
    """
    return tuple((s, st) for s, st in m if s != 1)


def show_map(m) -> str:
    """A residence for an error message."""
    return f"sizes {[s for s, _ in m]} strides {[t for _, t in m]}"


def _order_from(sizes, target, who: str) -> tuple[int, ...]:
    """The dimension ordering that packs ``sizes`` densely with ``target``'s strides.

    Outermost is the largest stride. Unit dims carry no data, so they may sit
    anywhere; they go outermost, which is the row-major convention. The result is
    verified rather than trusted: a target that is not a dense permutation of these
    dims (a padded or overlapping residence) has no ordering at all."""
    span = [d for d, s in enumerate(sizes) if s != 1]
    if len(span) != len(target) or [sizes[d] for d in span] != [s for s, _ in target]:
        raise LayoutError(
            f"{who}: laid out as {[s for s, _ in target]} but the access describes "
            f"{sizes}"
        )
    stride_of = dict(zip(span, (st for _s, st in target)))
    order = [d for d in range(len(sizes)) if d not in stride_of]
    order += sorted(span, key=lambda d: -stride_of[d])
    check = dense_strides([sizes[k] for k in order])
    for k, d in enumerate(order):
        if d in stride_of and stride_of[d] != check[k]:
            raise LayoutError(
                f"{who}: strides {[st for _s, st in target]} are not a dense "
                f"packing of {sizes} — no dimension ordering produces them"
            )
    return tuple(order)


def pin_access(p: PatternExpr, params: dict, target, who: str) -> dict:
    """The assignments for ``p``'s unsolved access params that make its map equal the
    residence ``target``.

    A solvable param has to sit in the access's root pattern: a reshape above it
    mixes several dims into one stride, so inverting through it would be a guess."""
    if p.kind == "layout" and isinstance(p.order, IndexExpr):
        sizes = [_resolve(d, params) for d in p.counts]
        return {p.order.param_index: _order_from(sizes, target, who)}
    if p.kind == "strided":
        weight, out, k = buffer_weights(p.buffer), {}, 0
        for axis, count in enumerate(p.counts):
            if _resolve(count, params) == 1:
                continue
            entry = p.strides[axis]
            if not isinstance(entry, int) and _resolve(entry, params) is None:
                if entry.kind != "param":
                    raise AcceleratorDescriptionError(
                        f"{who}: a solvable stride must be a bare access param, not "
                        f"an expression over one"
                    )
                if k >= len(target):
                    raise LayoutError(
                        f"{who}: the value has no dimension {k} to lay out"
                    )
                want = target[k][1]
                if want % weight[axis]:
                    raise LayoutError(
                        f"{who}: dimension {k} sits at element stride {want}, which is "
                        f"not a whole number of '{p.buffer.name}' axis-{axis} steps "
                        f"({weight[axis]})"
                    )
                out[entry.param_index] = want // weight[axis]
            k += 1
        return out
    raise AcceleratorDescriptionError(
        f"{who}: a solvable stride / ordering param must sit in the access's root "
        f"pattern, not underneath a reshape"
    )


def _index_params(item) -> set:
    """The set of access-param indices an int | IndexExpr count/basis entry uses."""
    if isinstance(item, int):
        return set()
    if item.kind == "param":
        return {item.param_index}
    if item.kind in ("add", "mul"):
        return _index_params(item.lhs) | _index_params(item.rhs)
    return set()  # const


def param_roles(spec: InstructionSpec):
    """Classify every access param by the access-pattern semantics it appears under.

    Each access kind models its params differently, and only some participate in
    shape solving:

    - ``offset`` — a strided ``basis`` entry: a buffer address, assigned by
      allocation (Stage 3), not solved from shapes.
    - ``shape``  — a ``counts`` / ``expand`` output entry: a tensor dimension,
      solved from the source shape (Stage 2).
    - ``stride`` — a strided ``stride`` entry: pure addressing. Invisible in the
      operand's *shape*, so it is solved from the residence the value's other
      accesses describe (Stage 2b), not from the source shapes.
    - ``layout`` — a ``layout`` node's dimension ordering: likewise residence, and
      likewise Stage 2b. Its value is a permutation, not a number.

    Returns ``(roles{param -> role}, offset_refs{param -> [(buffer position, axis)]})``
    and checks that the roles partition every param exactly once. An offset param names
    one *coordinate component* of one buffer, not a whole address: a multi-dimensional
    access has one per axis, and the allocator fills them in from the value's placement.

    A param may name that component in **more than one** buffer, and then the list has
    more than one entry: the ISA is stating that those operands sit at the same
    address — an in-place instruction (QKV's ``softmax`` reads and writes one ``addr``).
    That is a *constraint on allocation*, so every reference has to be kept."""
    roles: dict = {}
    offset_refs: dict = {}

    def mark(idx, role):
        if roles.setdefault(idx, role) != role:
            raise AcceleratorDescriptionError(
                f"{spec.name}: param p{idx} used as both '{roles[idx]}' and '{role}'"
            )

    for buf_pos, stripped, node in _access_chain(spec):
        for relayout in stripped:  # relayouts carry no params besides expand's shape
            if relayout.kind == "expand":
                for d in relayout.output_shape:
                    for i in _index_params(d):
                        mark(i, "shape")
        for axis, item in enumerate(node.basis):
            for i in _index_params(item):
                mark(i, "offset")
                offset_refs.setdefault(i, []).append((buf_pos, axis))
        for item in node.counts:
            for i in _index_params(item):
                mark(i, "shape")
        for item in node.strides or ():  # a layout node derives its strides
            for i in _index_params(item):
                mark(i, "stride")
        if node.kind == "layout" and isinstance(node.order, IndexExpr):
            mark(node.order.param_index, "layout")
    for i in range(arity(spec.access_fn)):
        if i not in roles:
            raise AcceleratorDescriptionError(
                f"{spec.name}: param p{i} is never used by the access pattern"
            )
    for i, refs in offset_refs.items():
        axes = {axis for _pos, axis in refs}
        if len(axes) != 1:
            raise AcceleratorDescriptionError(
                f"{spec.name}: address param p{i} is the basis of axes {sorted(axes)} "
                f"— one param names one coordinate component, so it cannot stand for "
                f"different axes of an address"
            )
    return roles, offset_refs

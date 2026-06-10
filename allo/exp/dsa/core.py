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
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Generic, ParamSpec, TypeVar, overload

from ..._mlir import ir
from ...lang.core import DType, Template
from ...lang.kernel import Kernel, KernelOptions

P = ParamSpec("P")
R = TypeVar("R")

# ==========================================================================#
# ISA buffer element type
# ==========================================================================#


@dataclass
class BufferKind:
    """The element type of an ISA buffer: the shape + dtype of a single slot.

    ``mnemonic`` is one of ``scalar`` / ``vector`` / ``tile`` / ``hbm``; ``shape``
    is the per-slot element shape (``()`` for scalar). It materializes to the
    corresponding ``!allo.{mnemonic}<...>`` MLIR type via ``ir.Type.parse`` (no
    CAPI), reusing the dtype materialization from ``lang.core``.
    """

    mnemonic: str
    dtype: DType
    shape: tuple[int, ...] = field(default_factory=tuple)

    @property
    def element_shape(self) -> tuple[int, ...]:
        return self.shape

    def mlir_text(self, context: ir.Context) -> str:
        elt = self.dtype.materialize(context)  # e.g. prints as "f32"
        if self.mnemonic == "scalar":
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
    shorter (matching ``expand``/``collapse`` semantics). When either side is 1-D the
    grouping is unambiguous (one group spanning all dims) and works for symbolic
    dims; otherwise dims are product-matched greedily (requires static sizes)."""
    long, short = (dst, src) if len(dst) >= len(src) else (src, dst)
    if len(short) == 1:
        return [list(range(len(long)))]
    groups, i = [], 0
    for s in short:
        size = _dim(s)
        assert isinstance(size, int), "reshape: product-matching needs static dims"
        grp, acc = [i], _dim(long[i])
        i += 1
        while acc != size and i < len(long):
            grp.append(i)
            acc *= _dim(long[i])
            i += 1
        assert acc == size, f"reshape: {src} -> {dst} is not a pure reshape"
        groups.append(grp)
    while i < len(long):  # trailing unit dims fold into the last group
        groups[-1].append(i)
        i += 1
    return groups


class PatternExpr:
    """A node in the access-pattern DAG.

    ``strided`` / ``tiled`` are rooted at a buffer; ``expand`` / ``collapse`` wrap
    a source ``PatternExpr``. Access is value-transparent (a reshape/affine view):
    any value-reordering relayout (e.g. transpose) belongs in the compute region
    as a ``prim``, not here.
    """

    def __init__(
        self,
        kind,
        *,
        buffer=None,
        basis=None,
        counts=None,
        strides=None,
        tile_sizes=None,
        source=None,
        reassociation=None,
        output_shape=None,
    ):
        self.kind = kind
        self.buffer = buffer
        self.basis = basis
        self.counts = counts
        self.strides = strides
        self.tile_sizes = tile_sizes
        self.source = source
        self.reassociation = reassociation
        self.output_shape = output_shape

    def root_buffer(self) -> BufferSpec:
        if self.kind in ("strided", "tiled"):
            return self.buffer
        return self.source.root_buffer()

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

        Mirrors the C++ ``materialize`` semantics: a strided access aligns the
        trailing buffer-element dims and rank-reduces a leading unit slot dim.
        Entries are ints, an ``IndexExpr`` for a symbolic (parametric) dim, or
        ``None`` for a dim that is dynamic but not solvable here.
        """
        if self.kind == "strided":
            counts = [_dim(c) for c in self.counts]
            dims = counts + list(self.buffer.kind.element_shape)
            # rank-reduce a single, statically-unit leading slot dim
            if counts and counts[0] == 1:
                dims = dims[1:]
            return dims
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


class TensorProxy:
    """A node in the compute DAG. ``arg`` leaves bind to buffer block args."""

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
    ):
        self.kind = kind  # "arg" | "identity" | a prim tag (see primitive.REGISTRY)
        self.dtype = dtype
        self.shape = tuple(shape)
        self.args = tuple(args)
        self.buffer_index = buffer_index
        self.permutation = permutation  # for "transpose": the dim permutation
        self.axis = axis  # for "reduce_*" / "reverse": the axis
        self.attrs = attrs or {}  # for conv/pool: pad/stride/dilation/kernel


# ==========================================================================#
# ISA model
# ==========================================================================#


@dataclass(eq=False)  # identity-based: a buffer is unique, usable as a dict key
class BufferSpec:
    name: str
    size: int  # number of slots
    kind: BufferKind
    is_global: bool = False  # off-chip / main memory: where program I/O lives

    def __getitem__(self, key) -> BufferSlice:
        return BufferSlice(self, key)

    @property
    def slot_size(self) -> int:
        """Elements per slot (1 for scalar buffers)."""
        return math.prod(self.kind.element_shape) or 1


@dataclass
class BufferSlice:
    """A ``buffer[slice]`` selection, used as an ``inspect`` target."""

    buffer: BufferSpec
    key: object


class InstructionSpec:
    def __init__(self, name, sources, destinations, cost=1.0):
        self.name = name
        self.sources = list(sources)
        self.destinations = list(destinations)
        self.cost = cost  # search cost (tree-DP objective); default 1 = op count
        self.access_fn = None
        self.compute_fn = None
        self.doc = None  # the defining function's docstring (carried onto Instruction)

    @property
    def buffers(self) -> list[BufferSpec]:
        return self.sources + self.destinations


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


def _bind_call(names, args, kwargs, who):
    assert len(args) <= len(names), f"{who}: too many positional args"
    bound = dict(zip(names, args))
    for k, v in kwargs.items():
        assert k in names, f"{who}: unknown parameter '{k}'"
        assert k not in bound, f"{who}: parameter '{k}' given twice"
        bound[k] = v
    missing = [n for n in names if n not in bound]
    assert not missing, f"{who}: missing parameters {missing}"
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
        self.addr_params = list(inspect.signature(spec.access_fn).parameters)
        compute = list(inspect.signature(spec.compute_fn).parameters)
        self.compute_params = compute[len(spec.buffers) :]
        self.__name__ = spec.name
        self.__doc__ = spec.doc

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        program = self.isa._active_oracle
        assert (
            program is not None
        ), f"instruction '{self.name}' can only be called inside @oracle"
        names = self.addr_params + self.compute_params
        bound = _bind_call(names, args, kwargs, self.name)
        n = len(self.addr_params)
        return program.record_emit(self.name, bound[:n], bound[n:])

    def __repr__(self):
        return f"Instruction<{self.name}>"


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


class ISA:
    """Registry of buffers + instructions for one accelerator."""

    def __init__(self, name: str):
        self.name = name
        self.buffers: dict[str, BufferSpec] = {}
        self.instructions: list[InstructionSpec] = []
        self._ops: dict[str, Instruction] = {}
        self._active_oracle = None  # the OracleProgram currently being traced
        self.kernels: list[Kernel] = []  # every @unit / @entry kernel
        self.top: Kernel | None = None  # the unique @entry kernel

    def _add_buffer(self, spec: BufferSpec) -> BufferSpec:
        assert spec.name not in self.buffers, f"duplicate buffer '{spec.name}'"
        self.buffers[spec.name] = spec
        return spec

    # --- memory hierarchy declarations ---
    def global_(self, name, shape, dtype: DType) -> BufferSpec:
        """Off-chip / main memory, word-addressable (scalar slots). Program I/O
        is marshalled into the global buffer."""
        return self._add_buffer(
            BufferSpec(
                name, math.prod(shape), BufferKind("scalar", dtype), is_global=True
            )
        )

    def scalar(self, name, slots, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` word-addressable (scalar) entries."""
        return self._add_buffer(BufferSpec(name, slots, BufferKind("scalar", dtype)))

    def vector(self, name, slots, shape, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` entries, each a 1-D vector of ``shape``."""
        return self._add_buffer(
            BufferSpec(name, slots, BufferKind("vector", dtype, tuple(shape)))
        )

    def tile(self, name, slots, shape, dtype: DType) -> BufferSpec:
        """An on-chip buffer of ``slots`` entries, each an N-D tile of ``shape``."""
        return self._add_buffer(
            BufferSpec(name, slots, BufferKind("tile", dtype, tuple(shape)))
        )

    # --- instruction declaration ---
    def instruction(self, src, dst, *, name=None, cost=1.0):
        """Decorate ``def <mnemonic>(I): ...`` -> a callable ``Instruction``.

        The decorated function's name is the mnemonic, so the returned object
        binds to that name and can be called bare inside an ``@oracle``.
        ``cost`` is the search objective weight (default 1 = minimize op count).
        """

        def decorator(fn) -> Instruction:
            spec = InstructionSpec(
                name or fn.__name__, _as_list(src), _as_list(dst), cost
            )
            spec.doc = fn.__doc__
            fn(InstructionBuilder(spec))
            assert spec.access_fn is not None, f"{spec.name}: missing @I.access"
            assert spec.compute_fn is not None, f"{spec.name}: missing @I.compute"
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
            assert all(isinstance(a, Template) for a in template)

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
                assert (
                    self.top is None
                ), f"ISA '{self.name}': @arch already defined as '{self.top.func_name}'"
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
            raise ValueError(f"ISA '{self.name}' defines no @entry top to schedule")
        return self.top.schedule()

    # --- oracle (functional simulation of hand-written assembly) ---
    def inspect(self, target, *, label=None):
        """Record a snapshot of ``target`` (a buffer or ``buffer[slice]``) at the
        current program point."""
        assert (
            self._active_oracle is not None
        ), "inspect can only be called inside @oracle"
        if isinstance(target, BufferSpec):
            buf, sl = target, None
        elif isinstance(target, BufferSlice):
            buf, sl = target.buffer, target.key
        else:
            raise TypeError(f"cannot inspect {target!r}; expected a buffer or slice")
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

    def compile_program(self, source: str):
        """Compile a source program (a TOSA-dialect MLIR module given as *text*)
        onto this ISA, returning a runnable ``CompiledProgram``."""
        from .search import compile_program

        return compile_program(source, self)


def arity(fn) -> int:
    return len(inspect.signature(fn).parameters)


def trace_instruction(spec: InstructionSpec):
    """Trace an instruction's access + compute regions into Python DAGs.

    Returns ``(patterns, arg_shapes, results)``: one ``PatternExpr`` per buffer,
    the inferred visible shape per buffer (ints, ``None`` for dynamic), and the
    yielded ``TensorProxy`` list. Shared by codegen (IR construction) and the
    search backend (semantic tag + symbolic shape), so both see one truth.
    """
    params = [IndexExpr.param(i) for i in range(arity(spec.access_fn))]
    patterns = spec.access_fn(*params)
    patterns = list(patterns) if isinstance(patterns, (tuple, list)) else [patterns]
    assert len(patterns) == len(spec.buffers), (
        f"{spec.name}: access yields {len(patterns)} patterns, "
        f"expected {len(spec.buffers)} (one per src+dst buffer)"
    )
    arg_shapes = [p.visible_shape() for p in patterns]
    comp_args = [
        TensorProxy("arg", buf.kind.dtype, shape, buffer_index=i)
        for i, (buf, shape) in enumerate(zip(spec.buffers, arg_shapes))
    ]
    results = spec.compute_fn(*comp_args)
    results = list(results) if isinstance(results, (tuple, list)) else [results]
    return patterns, arg_shapes, results


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

    - ``offset`` — a strided/tiled ``basis`` entry: a buffer address, assigned by
      allocation (Stage 3), not solved from shapes.
    - ``shape``  — a ``counts`` / ``expand`` output / ``tiled`` tile-size entry: a
      tensor dimension, solved from the source shape (Stage 2).
    - ``stride`` — a strided ``stride`` entry: pure addressing, shape-irrelevant.

    Returns ``(roles{param -> role}, offset_buffer{param -> buffer position})`` and
    asserts the roles partition every param exactly once (disjoint + complete)."""
    params = [IndexExpr.param(i) for i in range(arity(spec.access_fn))]
    patterns = spec.access_fn(*params)
    patterns = list(patterns) if isinstance(patterns, (tuple, list)) else [patterns]
    roles: dict = {}
    offset_buffer: dict = {}

    def mark(idx, role):
        assert (
            roles.setdefault(idx, role) == role
        ), f"{spec.name}: param p{idx} used as both '{roles[idx]}' and '{role}'"

    for buf_pos, p in enumerate(patterns):
        node = p
        while node.kind in ("expand", "collapse"):
            if node.kind == "expand":
                for d in node.output_shape:
                    for i in _index_params(d):
                        mark(i, "shape")
            node = node.source  # relayouts carry no params besides expand's shape
        for item in node.basis:
            for i in _index_params(item):
                mark(i, "offset")
                offset_buffer[i] = buf_pos
        for item in node.counts:
            for i in _index_params(item):
                mark(i, "shape")
        for item in node.strides:
            for i in _index_params(item):
                mark(i, "stride")
        for item in node.tile_sizes or []:
            for i in _index_params(item):
                mark(i, "shape")
    for i in range(len(params)):
        assert (
            i in roles
        ), f"{spec.name}: param p{i} is never used by the access pattern"
    return roles, offset_buffer

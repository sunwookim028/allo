# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose an Allo dataflow region out of separately written units.

Allo reaches a ``@df.kernel`` only as a nested ``ast.FunctionDef`` inside the
``@df.region()`` that owns it, and resolves the names in its body against the
region's scope, so a unit cannot be imported into a region the way a Python
function is imported into a module. This module composes the region's *source*
instead: a unit is an ordinary module-level function plus an interface
declaration, and an ``Architecture`` binds that interface -- channels,
memories, parameters, ISA names -- when it emits the region.

What the front end refuses, and why this is the shape that works today, is on
``docs/source/designs/tinytpu_library.rst``, which uses it.
"""

from __future__ import annotations

import ast
import inspect
import json
import linecache
import math
import os
import textwrap
from dataclasses import dataclass, field

import allo
import allo.dataflow as df
from allo.ir.types import (Stateful, Stream, UInt, Wire, comb, int8, int16,
                           int32, uint1)

# Names every unit body may use without declaring them: the front end itself.
FRONTEND_NAMES = {"allo": allo, "df": df, "Stream": Stream, "UInt": UInt,
                  "int8": int8, "int16": int16, "int32": int32,
                  "uint1": uint1, "Wire": Wire, "comb": comb,
                  "Stateful": Stateful}
_BUILTINS = {"range", "len", "min", "max", "abs", "int", "bool"}


@dataclass(frozen=True)
class Channel:
    """One point-to-point stream, or an array of them, between two units.

    ``dtype`` and ``depth`` are Allo type *expressions*, not values: they are
    emitted into the region and evaluated there, so ``UInt(VW)`` means whatever
    the architecture that instantiates the channel says ``VW`` is.
    """

    name: str
    dtype: str = ""
    depth: str = "QD"
    shape: tuple[str, ...] = ()
    carries: str = ""
    lanes: str = ""
    lane_bits: str = ""
    # "stream" (default), "wire" or "comb" (README D-13). A Wire link is
    # SystemC-only; ``Architecture.region(target="simulator")`` emits every
    # link as a Stream, since the simulator is untimed and refuses Wire.
    kind: str = "stream"

    def __post_init__(self):
        """A packed word declares how many lanes it carries and how wide one
        is; its bit width is DERIVED from the pair, so there is one
        declaration instead of two that can disagree. The lane count is what a
        reduction's leaf order is checked against
        (``docs/source/designs/ip_gaps.rst``).
        """
        if self.lanes and self.lane_bits:
            derived = f"UInt({self.lanes} * {self.lane_bits})"
            if not self.dtype:
                object.__setattr__(self, "dtype", derived)
            else:
                assert self.dtype == derived, (
                    f"channel {self.name}: dtype {self.dtype!r} is not the "
                    f"width its {self.lanes} lanes of {self.lane_bits} bits "
                    f"come to ({derived!r}); declare the lanes and let the "
                    f"width follow")
        assert self.dtype, (
            f"channel {self.name}: declare a dtype, or the lanes and the "
            f"lane width to derive one from")
        assert self.kind in {"stream", "wire", "comb"}, (
            f"channel {self.name}: kind {self.kind!r} is not stream, wire or comb")
        assert self.kind == "stream" or not self.shape, (
            f"channel {self.name}: an array of {self.kind} links is not supported")

    @property
    def declaration(self) -> str:
        return self.declaration_as(self.kind)

    def declaration_as(self, kind: str) -> str:
        dims = f"[{', '.join(self.shape)}]" if self.shape else ""
        if kind == "stream":
            line = f"{self.name}: Stream[{self.dtype}, {self.depth}]{dims}"
        elif kind == "wire":
            line = f"{self.name}: Wire[{self.dtype}]"
        else:
            line = f"{self.name}: Wire[{self.dtype}, comb]"
        return f"{line}    # {self.carries}" if self.carries else line


PORT_KINDS = ("r", "w", "rw")
COLLISIONS = ("refuse", "obligation", "undefined")


@dataclass(frozen=True)
class Port:
    """One port of an on-chip ``Memory`` (README D-12).

    ``kind`` is ``r``, ``w`` or ``rw``. ``latency`` is the read latency in
    edges from address to data, ``0`` meaning asynchronous (a combinational
    read, D-13); a ``w`` port has none. ``visible`` is the edges from a write
    on this port until a read on ANY port sees it. ``count`` gives that many
    interchangeable ports (AMC's port ``count``): a unit binding the port may
    make up to ``count`` accesses of each direction per iteration.
    """

    name: str
    kind: str
    latency: int = None
    visible: int = 1
    count: int = 1

    def __post_init__(self):
        assert self.name.isidentifier(), f"port {self.name!r}: not an identifier"
        assert self.kind in PORT_KINDS, (
            f"port {self.name}: kind {self.kind!r} is not one of {PORT_KINDS}")
        if self.kind == "w":
            assert self.latency is None, (
                f"port {self.name}: a write port has no read latency "
                f"(got latency={self.latency})")
        else:
            assert isinstance(self.latency, int) and self.latency >= 0, (
                f"port {self.name}: a {self.kind!r} port declares its read "
                f"latency in edges (latency=0 is asynchronous); got "
                f"{self.latency!r}")
        assert isinstance(self.visible, int) and self.visible >= 0, (
            f"port {self.name}: visible={self.visible!r} is not a count of edges")
        assert isinstance(self.count, int) and self.count >= 1, (
            f"port {self.name}: count={self.count!r} must be >= 1")

    @property
    def reads(self) -> bool:
        return self.kind in {"r", "rw"}

    @property
    def writes(self) -> bool:
        return self.kind in {"w", "rw"}

    def manifest(self) -> dict:
        out = {"kind": self.kind, "count": self.count}
        if self.reads:
            out["latency"] = self.latency
        if self.writes:
            out["visible"] = self.visible
        return out


@dataclass(frozen=True)
class Memory:
    """An array the units address.

    Without ``rows`` it is one array at the region boundary -- off-chip, one
    ``m_axi`` port -- and a unit binds it by name, as before.

    With ``rows`` and ``ports`` it is an on-chip memory that declares its
    ports (README D-12): a unit binds a PORT (``memories=("vreg.ra",)``),
    each port has exactly one owner, and the composition checks every body
    against its ports' direction and accesses per iteration. ``collision``
    states what a same-word access on two write-capable ports in one cycle
    means: ``refuse`` (only legal with at most one write-capable port),
    ``obligation`` (the composition states why it cannot happen:
    ``Architecture(obligations=...)``) or ``undefined`` (masked in verdicts).
    ``reset=False`` declares storage that survives reset (README D-14).
    ``dtype`` is the element type; ``rows`` an expression over the
    architecture's parameters.
    """

    name: str
    dtype: str
    rows: str = None
    ports: tuple = ()
    collision: str = "refuse"
    reset: bool = True

    def __post_init__(self):
        if self.rows is None:
            assert not self.ports, (
                f"memory {self.name}: ports need rows= (a memory without rows "
                f"is a boundary array with its one m_axi port)")
            return
        assert self.ports, f"memory {self.name}: declare its ports"
        names = [p.name for p in self.ports]
        assert len(set(names)) == len(names), (
            f"memory {self.name}: port names repeat: {names}")
        assert self.collision in COLLISIONS, (
            f"memory {self.name}: collision={self.collision!r} is not one of "
            f"{COLLISIONS}")
        writers = [p.name for p in self.ports if p.writes]
        nw = sum(p.count for p in self.ports if p.writes)
        assert not (nw > 1 and self.collision == "refuse"), (
            f"memory {self.name}: {nw} write-capable ports ({', '.join(writers)}) "
            f"with collision='refuse': two ports may write one word in one cycle "
            f"and nothing can refuse it statically. Declare collision="
            f"'obligation' (and state why it cannot happen) or 'undefined' "
            f"(README D-12)")

    @property
    def ported(self) -> bool:
        return self.rows is not None

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise AssertionError(
            f"memory {self.name} has no port {name!r} (its ports: "
            f"{', '.join(p.name for p in self.ports)})")

    @property
    def declaration(self) -> str:
        assert not self.ported, f"memory {self.name} is on chip; it has no argument"
        return f"{self.name}: {self.dtype},"


# ---------------------------------------------------------------------------
# Engines (README D-15): a swappable arithmetic is a declared record, and the
# accumulate order it is exact for is part of the function.
# ---------------------------------------------------------------------------

#: The names an engine binds into a unit, ``<SLOT>_<FIELD>`` (``MAC_IN``,
#: ``MAC_ADD``, ...). Types first, then their widths, then the bodies.
ENGINE_FIELDS = ("IN", "IN_BITS", "ACC", "ACC_BITS", "OUT", "OUT_BITS",
                 "MUL", "ADD", "PACK")
_ENGINE_BODIES = {"MUL": "mul", "ADD": "add", "PACK": "pack"}
ORDERS = ("sequential", "tree")


def engine_slot(name: str):
    """``"MAC_IN_BITS" -> ("MAC", "IN_BITS")``; ``None`` if ``name`` is not an
    engine slot name. The longest field wins, so ``_IN_BITS`` is not ``_IN``."""
    for f in sorted(ENGINE_FIELDS, key=len, reverse=True):
        if name.endswith("_" + f) and len(name) > len(f) + 1:
            return name[: -len(f) - 1], f
    return None


def _type_sig(t):
    """What two Allo types must share to be one type: kind, width, fraction."""
    return (type(t).__name__, getattr(t, "bits", None), getattr(t, "fracs", None))


@dataclass(frozen=True, kw_only=True)
class Engine:  # pylint: disable=too-many-instance-attributes
    """A swappable arithmetic engine, declared (README D-15).

    ``IN``/``ACC``/``OUT`` are Allo types, ``IN_BITS``/``ACC_BITS``/``OUT_BITS``
    their widths (one to annotate with, one to slice with; held to each
    other). ``mul``/``add``/``pack`` are module-level Allo functions.
    ``latency`` is the D-10 ``latency=`` of each body, ``{"mul": L, "add": L,
    "pack": L}`` in edges, ``0`` meaning no register inside the body and a
    missing or ``None`` entry meaning unconstrained (the backend reports what
    it built in ``latency.json``); it is what a composition BOOKS, and no
    body consumes it. ``order`` is the accumulate order the engine is exact
    for (``sequential``: one rounding per term in ascending index; ``tree``:
    one per level of a balanced tree) -- different orders are different
    functions. ``ref_mul``/``ref_add``/``ref_pack`` are the same arithmetic in
    numpy, so a composite's contract reference is evaluated in the engine's
    own arithmetic (``dot``). ``directives(s, ctx)`` is what the engine needs
    of a schedule; ``Architecture.directives`` applies it once for every unit
    that binds the engine, with ``ctx.unit`` and ``ctx.engine`` set.

    A unit binds an engine's names as ``engines=`` slots (``MAC_IN``,
    ``MAC_ADD``, ...), never as a bare function in ``parameters``.
    """

    name: str
    IN: object
    IN_BITS: int
    ACC: object
    ACC_BITS: int
    OUT: object
    OUT_BITS: int
    mul: callable
    add: callable
    pack: callable
    order: str
    ref_mul: callable
    ref_add: callable
    ref_pack: callable
    latency: dict = field(default_factory=dict)
    directives: callable = None

    def __post_init__(self):
        for width, dtype in (("IN_BITS", "IN"), ("ACC_BITS", "ACC"),
                             ("OUT_BITS", "OUT")):
            got = getattr(getattr(self, dtype), "bits", None)
            assert getattr(self, width) == got, (
                f"engine {self.name}: {width}={getattr(self, width)} but {dtype} "
                f"is {getattr(self, dtype)} ({got} bits); a unit that slices a "
                f"different width from the one it computes in reads the wrong "
                f"bits without failing (README D-15)")
        assert self.order in ORDERS, (
            f"engine {self.name}: order {self.order!r} is not one of {ORDERS}")
        for body in ("mul", "add", "pack"):
            assert callable(getattr(self, body)), (
                f"engine {self.name}: {body} is not a function")
        for key, lat in self.latency.items():
            assert key in {"mul", "add", "pack"}, (
                f"engine {self.name}: latency names {key!r}; an engine's bodies "
                f"are mul, add and pack")
            assert lat is None or (isinstance(lat, int) and lat >= 0), (
                f"engine {self.name}: latency[{key!r}]={lat!r} is not a count "
                f"of edges (D-10: None leaves it to the backend)")

    @property
    def mul_latency(self):
        return self.latency.get("mul")

    @property
    def add_latency(self):
        return self.latency.get("add")

    @property
    def pack_latency(self):
        return self.latency.get("pack")

    def namespace(self, slot: str = "MAC") -> dict:
        """The names a unit binds through slot ``slot``: ``MAC_IN``, ..."""
        out = {}
        for f in ENGINE_FIELDS:
            out[f"{slot}_{f}"] = getattr(self, _ENGINE_BODIES.get(f, f))
        return out

    @staticmethod
    def names(slot: str = "MAC") -> tuple:
        return tuple(f"{slot}_{f}" for f in ENGINE_FIELDS)

    @staticmethod
    def rebind(slot: str, new_slot: str) -> dict:
        """The ``Instance`` binding that points a unit's ``<slot>_*`` names
        at ``<new_slot>_*`` (README D-17): ``{"MAC_IN": "MAC__a_IN", ...}``."""
        return dict(zip(Engine.names(slot), Engine.names(new_slot)))

    def accumulate(self, terms, order: str = None):
        """Fold ``terms`` (ACC values, numpy) with ``ref_add`` in ``order``
        (default: the engine's): ``sequential`` is ``add(t[r], acc)`` from a
        zero accumulator in ascending ``r``; ``tree`` is a balanced binary
        tree, left child first, over a power-of-two count."""
        import numpy as np  # noqa: PLC0415

        order = order or self.order
        assert order in ORDERS, f"order {order!r} is not one of {ORDERS}"
        terms = list(terms)
        if order == "sequential":
            acc = np.zeros_like(np.asarray(terms[0], dtype=np.int64))
            for t in terms:
                acc = self.ref_add(t, acc)
            return acc
        n = len(terms)
        assert n >= 1 and n & (n - 1) == 0, (
            f"engine {self.name}: a balanced tree over {n} terms needs a power of two")
        while len(terms) > 1:
            terms = [self.ref_add(terms[2 * k], terms[2 * k + 1])
                     for k in range(len(terms) // 2)]
        return terms[0]

    def dot(self, A, W, order: str = None):
        """The contract reference of a dot-product composite with the order
        as an argument (README D-15): every row of ``A`` (``[n, K]``) against
        ``W`` (``[K, M]``) in this engine's numpy arithmetic, accumulated in
        ``order``, packed once."""
        import numpy as np  # noqa: PLC0415

        A = np.asarray(A, dtype=np.int64)
        W = np.asarray(W, dtype=np.int64)
        n, K = A.shape
        out = np.zeros((n, W.shape[1]), dtype=np.int64)
        for i in range(n):
            prods = [self.ref_mul(np.full(W.shape[1], A[i, r]), W[r])
                     for r in range(K)]
            out[i] = self.ref_pack(self.accumulate(prods, order))
        return out


@dataclass(frozen=True)
class Unit:
    """One kernel: a body, the instances it is replicated into, and every name
    the body needs from outside itself.

    The declaration is checked against the body rather than trusted, so a
    channel a unit touches, a parameter it reads or an ISA name it decodes
    cannot go unstated -- ``isa=()`` on the array units is the claim that the
    PEs decode nothing, and it is enforced.

    ``legality`` is the unit's own condition on the parameter set it is being
    instantiated at, run at composition time. ``Unit.check`` answers *which
    names a unit may use*, which is a different question from *which values it
    works at*: without ``legality`` a unit sized past what its arithmetic is
    exact for composes, builds and gives wrong answers
    (``docs/source/designs/ip_gaps.rst``).

    ``instances`` is the emitted ``mapping=``, as expressions over the
    architecture's parameters (``("T", "T")`` for a T x T array). ``memories``
    names the region arguments the body's own parameters bind to, positionally.

    ``calls`` names the module-level functions the body calls that are part
    of the unit itself (``add_bits``, ``sfu_bits``): resolved from the
    body's own module, never bound by the architecture and never swapped --
    a swappable function is an engine slot (README D-15, D-19).

    ``engines`` names the engine slots the body binds (``MAC_IN``,
    ``MAC_ADD``, ...: ``<slot>_<field>``, README D-15); the architecture binds
    an ``Engine`` to each slot. ``order`` is the accumulate order the body
    itself implements, for a unit that IS an engine (a matrix engine:
    ``sequential`` for a systolic chain, ``tree`` for an adder tree); the
    composite's ``Architecture(order=, accepts=)`` holds it, like an engine's.
    """

    body: callable
    instances: tuple[str, ...] = ("1",)
    memories: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    isa: tuple[str, ...] = ()
    directives: callable = None
    legality: callable = None
    engines: tuple[str, ...] = ()
    order: str = None
    calls: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.body.__name__

    @property
    def declared_names(self) -> set[str]:
        return set(self.reads) | set(self.writes) | set(self.parameters) \
            | set(self.isa) | set(self.engines) | set(self.calls)

    @property
    def engine_slots(self) -> dict:
        """``{slot: {field, ...}}`` of the engine names the unit binds."""
        out = {}
        for name in self.engines:
            slot = engine_slot(name)
            if slot is not None:
                out.setdefault(slot[0], set()).add(slot[1])
        return out

    def source(self) -> str:
        """The body's text, with its own decorators stripped, ready to nest."""
        text = textwrap.dedent(inspect.getsource(self.body))
        lines = text.splitlines(True)
        return "".join(lines[ast.parse(text).body[0].lineno - 1:]).rstrip("\n")

    def kernel_source(self) -> str:
        args = f", args=[{', '.join(self.memories)}]" if self.memories else ""
        decorator = f"@df.kernel(mapping=[{', '.join(self.instances)}]{args})"
        return textwrap.indent(decorator + "\n" + self.source(), "    ")

    def free_names(self) -> set[str]:
        """Every name the body reads and does not itself bind."""
        tree = ast.parse(self.source()).body[0]
        bound = {a.arg for a in tree.args.args}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                bound.update(n.id for n in ast.walk(node.optional_vars)
                             if isinstance(n, ast.Name))
        read = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        return read - bound - _BUILTINS - set(FRONTEND_NAMES)

    def arrays(self) -> dict:
        """Every array the body addresses, and how it addresses it.

        ``{name: {"rows": expression or None, "local": bool,
        "read": bool, "write": bool}}``. Local arrays are the ones the body
        declares itself (``ar: UInt(AW)[NAR]``); the rest are the region
        arguments the unit was given, named positionally by ``memories``.

        The same AST ``free_names`` checks the declaration against, asked a
        second question. It lives here rather than in a consumer because a
        unit's memory is a structural fact of the unit, and this module exists
        so that a structural fact is stated once."""
        tree = ast.parse(self.source()).body[0]
        parameters = [a.arg for a in tree.args.args]
        bound = dict(zip(parameters, self.memories))
        out = {}
        for node in ast.walk(tree):
            # An INITIALISED array is still an array. The `node.value is None`
            # this used to also require made `spad: UInt(VW)[SPAD_ROWS] = 0`
            # invisible here while the body still read and wrote it, so
            # `gen_isa.py --conform`'s port arm reported that the composed
            # region implied no `spad` port -- refusing a one-line, bit-exact
            # hardware edit (the zero-fill b4be2b10 removed) for a change in
            # what this function could see rather than in what the design does.
            # The annotation being a Subscript is what makes it an array; a
            # scalar `n: int32 = 0` annotates a Name and is still excluded.
            # `chia_agent/area_proxy.py`'s own census never had the guard,
            # which is the second reason to believe this one was a defect: the
            # two walks of the same AST disagreed about the same array.
            if isinstance(node, ast.AnnAssign) \
                    and isinstance(node.annotation, ast.Subscript):
                out[node.target.id] = {
                    "rows": ast.unparse(node.annotation.slice),
                    "local": True, "read": False, "write": False}
        for name in parameters:
            out[name] = {"rows": None, "local": False,
                         "read": False, "write": False}
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) \
                    and isinstance(node.value, ast.Name) \
                    and node.value.id in out:
                where = out[node.value.id]
                where["write" if isinstance(node.ctx, ast.Store)
                      else "read"] = True
        return {bound.get(name, name): use for name, use in out.items()}

    def check(self):
        assert len(self.memories) == len(inspect.signature(self.body).parameters), (
            f"unit {self.name}: {len(self.memories)} memories declared for "
            f"{len(inspect.signature(self.body).parameters)} body parameters")
        for name in self.engines:
            assert engine_slot(name) is not None, (
                f"unit {self.name}: engine slot {name!r} is not "
                f"<slot>_<field> with a field in {ENGINE_FIELDS} (README D-15)")
        free, declared = self.free_names(), self.declared_names
        assert free == declared, (
            f"unit {self.name}: the body uses {sorted(free - declared)} "
            f"without declaring them and declares "
            f"{sorted(declared - free)} without using them")
        self._check_engines()
        self._check_calls()
        self._check_slice_bounds(free)

    def call_targets(self) -> dict:
        """``{name: function}`` of ``calls``, from the body's own module."""
        return {n: self.body.__globals__.get(n) for n in self.calls}

    def _check_calls(self):
        for n, fn in self.call_targets().items():
            assert inspect.isfunction(fn), (
                f"unit {self.name}: calls={n!r}, which is not a function of "
                f"the body's module ({self.body.__module__})")

    def _check_slice_bounds(self, free):
        """README D-17: a bit slice whose bounds are only names from outside
        the body (``word[0:MAC_IN_BITS]``) is refused. The front end cannot
        infer its width ("Cannot infer the bitwidth of the slice, use
        UInt(32) as default") and widens the extract to 32 bits -- a
        different circuit, or ``trunci i32 -> i32`` and no build. Inside
        ``meta_for`` (a bound index in the bound) the expression folds."""
        tree = ast.parse(self.source()).body[0]
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)):
                continue
            names = {x.id for b in (n.slice.lower, n.slice.upper) if b is not None
                     for x in ast.walk(b) if isinstance(x, ast.Name)}
            if names and names <= free:
                raise AssertionError(
                    f"unit {self.name} line {n.lineno}: slice "
                    f"`{ast.unparse(n)}` has bounds that are only parameters "
                    f"({', '.join(sorted(names))}); Allo cannot infer the "
                    f"bitwidth of the slice (UInt(32) by default, "
                    f"tinytpu_library.rst 'symbolic slice'). Convert by typed "
                    f"assignment (`x: {sorted(names)[0].replace('_BITS', '')} = "
                    f"word`) until the front end folds constants into slice "
                    f"bounds (README D-17)")

    def _check_engines(self):
        """README D-15: the engine names the body binds are declared as slots,
        each one ``<slot>_<field>``, and a body is only CALLED."""
        assert self.order is None or self.order in ORDERS, (
            f"unit {self.name}: order {self.order!r} is not one of {ORDERS}")
        slots = self.engine_slots
        for name in self.parameters:
            hit = engine_slot(name)
            assert hit is None or hit[0] not in slots, (
                f"unit {self.name}: {name!r} names engine slot {hit[0]!r} but "
                f"is declared a parameter; a unit binds an engine's names as "
                f"engines=, never as a bare parameter (README D-15)")
        tree = ast.parse(self.source()).body[0]
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id in self.engines \
                    and engine_slot(n.id)[1] in _ENGINE_BODIES:
                assert id(n) in called, (
                    f"unit {self.name}: engine body {n.id!r} is used other "
                    f"than as a call (line {n.lineno}); an engine body is "
                    f"called, never passed on (README D-15)")


# ---------------------------------------------------------------------------
# Instantiation (README D-17): one Unit, any number of times in one region,
# each instance binding the body's free names to its own.
# ---------------------------------------------------------------------------


class _Rename(ast.NodeTransformer):
    """Rename every ``Name`` in ``bind`` (loads and stores alike)."""

    def __init__(self, bind):
        self.bind = bind

    def visit_Name(self, node):  # pylint: disable=invalid-name
        if node.id in self.bind:
            return ast.copy_location(ast.Name(id=self.bind[node.id], ctx=node.ctx), node)
        return node


def _rename_expr(text: str, bind: dict) -> str:
    if not bind:
        return text
    tree = _Rename(bind).visit(ast.parse(text, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(tree))


@dataclass(frozen=True, init=False)
class Instance(Unit):
    """``Instance(unit, name, bind)``: ``unit`` instantiated under ``name``
    with a binding of its free names (README D-17).

    ``bind`` maps a name of the unit to the name this instance uses in the
    architecture: a parameter (``DIM -> DIM_a``), a channel (``lhs ->
    lhs_a``), an engine slot name (``MAC_IN -> MAC__a_IN``; see
    ``Engine.rebind``) or a memory the unit binds (``A -> A_a``,
    ``vreg.ra -> vreg_a.ra``). Only free names of the body and the unit's
    memories may be bound; ``Architecture`` refuses anything else. The body
    is emitted under ``name`` with every bound name renamed, and every check
    ``compose`` makes runs on the instance: the declaration against the
    renamed body, one owner per channel and port, ``legality`` on the
    parameter set as the unit sees it through the binding.
    """

    unit: Unit = None
    instance_name: str = ""
    bind: dict = None

    def __init__(self, unit: Unit, name: str, bind: dict = None):  # pylint: disable=super-init-not-called,redefined-outer-name
        assert isinstance(unit, Unit), (
            f"Instance({name!r}): instantiate a compose.Unit, got {unit!r}")
        assert isinstance(name, str) and name.isidentifier(), (
            f"Instance of {unit.name}: name {name!r} is not an identifier")
        bind = dict(bind or {})
        if isinstance(unit, Instance):
            # An instance re-bound (an option rebinding a neighbour, D-17):
            # one binding of the original unit, the later one applied last.
            extra = sorted(set(bind) - unit.free_names() - set(unit.memories))
            assert not extra, (
                f"Instance {name} of {unit.name}: bind names {extra}, which "
                f"are not free in the instance (README D-17)")
            merged = {k: bind.get(v, v) for k, v in unit.bind.items()}
            for k, v in bind.items():
                if k not in unit.bind.values():
                    merged[k] = v
            unit, bind = unit.unit, merged
        allowed = unit.free_names() | set(unit.memories)
        extra = sorted(set(bind) - allowed)
        assert not extra, (
            f"Instance {name} of unit {unit.name}: bind names {extra}, which "
            f"{'is' if len(extra) == 1 else 'are'} not free in the body "
            f"(free: {sorted(unit.free_names())}; memories: "
            f"{list(unit.memories)}). A binding renames only what the body "
            f"takes from outside (README D-17)")
        called = sorted(set(bind) & set(unit.calls))
        assert not called, (
            f"Instance {name} of unit {unit.name}: bind renames {called}, "
            f"which the unit calls as its own functions; a function an "
            f"instance chooses is an engine slot (README D-15)")

        def sub(names):
            return tuple(bind.get(n, n) for n in names)

        fields = {
            "body": unit.body,
            "instances": tuple(_rename_expr(e, bind) for e in unit.instances),
            "memories": sub(unit.memories), "reads": sub(unit.reads),
            "writes": sub(unit.writes), "parameters": sub(unit.parameters),
            "isa": sub(unit.isa), "directives": unit.directives,
            "legality": unit.legality, "engines": sub(unit.engines),
            "order": unit.order, "calls": unit.calls, "unit": unit,
            "instance_name": name, "bind": bind}
        for k, v in fields.items():
            object.__setattr__(self, k, v)

    @property
    def name(self) -> str:
        return self.instance_name

    def source(self) -> str:
        fn = ast.parse(Unit.source(self)).body[0]
        fn.name = self.instance_name
        fn.decorator_list = []
        fn = _Rename(self.bind).visit(fn)
        return ast.unparse(ast.fix_missing_locations(fn))

    def bound_parameters(self, parameters: dict) -> dict:
        """The parameter set as the unit sees it: each name it was written
        with, at the value of the name the binding gave it."""
        out = dict(parameters)
        for k, v in self.bind.items():
            if v in parameters:
                out[k] = parameters[v]
        return out


# ---------------------------------------------------------------------------
# Optional modules (README D-19): a declared delta over an architecture, with
# the ISA slots that exist only with it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One optional module (README D-19): the units, channels and memories it
    adds, the ``rebind`` of its neighbours' names when it is present
    (``{unit name: {name: new name}}``, applied as an ``Instance``, D-17),
    the parameters and engines it needs, and the ISA ``isa`` slots that exist
    only with it. ``Architecture.with_options(base, *options)`` composes it."""

    name: str
    units: tuple = ()
    channels: tuple = ()
    memories: tuple = ()
    rebind: dict = field(default_factory=dict)
    isa: tuple = ()
    parameters: dict = field(default_factory=dict)
    engines: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.name.isidentifier(), f"option {self.name!r}: not an identifier"


def isa_slots(arch) -> dict:
    """``{slot: module}`` of a composed instance, in order (README D-19): the
    base machine's slots (module ``"base"``), then each option's. What an
    assembler may emit and what ``gen_isa --check`` holds an instance's spec
    to: the ISA table is derived from the composition."""
    out = {s: "base" for s in arch.slots}
    for o in arch.options:
        out.update({s: o.name for s in o.isa})
    return out


def check_program(arch, program, known=()) -> list:
    """Refuse a program naming an instruction whose module is not composed
    into ``arch`` (README D-19), naming the module when one of ``known``
    options brings it. Returns the program."""
    slots = isa_slots(arch)
    owner = {s: o.name for o in known for s in o.isa}
    for op in program:
        if op not in slots:
            why = (f"its module {owner[op]!r} is not composed in" if op in owner
                   else "no module of this instance brings it")
            raise AssertionError(
                f"{arch.name}: {op!r} is not an instruction of this instance: "
                f"{why} (slots: {sorted(slots)}; README D-19)")
    return list(program)


# ---------------------------------------------------------------------------
# Memory ports (README D-12): what a body does through a port, and how a
# ported memory is lowered into the region's source.
# ---------------------------------------------------------------------------


def _is_site(node, pname):
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
        and node.value.id == pname


def port_uses(u, pname, mem, port, tree=None):
    """The accesses unit ``u`` makes through body parameter ``pname``, bound
    to port ``mem`` (``"vreg.ra"``), checked against the port.

    Returns ``{"loads": [(Subscript, stmt)], "stores": [(address, value,
    enable or None, stmt)], "loop": For, "tree": FunctionDef}`` on the given
    (or a fresh) parse of the body. Refused, naming the port:

    * a use of the parameter other than ``p[address]``;
    * an access outside the unit's one top-level ``for`` loop (one iteration
      is one cycle of the port), or under control flow -- an access is made
      exactly once per iteration, except a write, which may sit alone under
      ``if <enable>:`` (the enable is a pin of the port; the S6 shape);
    * a read through a ``w`` port or a write through an ``r`` port;
    * more accesses of one direction per iteration than ``count``;
    * on an ``rw`` port, a read and a write at different addresses (one
      access = one address).
    """
    tree = tree if tree is not None else ast.parse(u.source()).body[0]
    where = f"unit {u.name} (port {mem}, kind {port.kind!r})"
    names = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id == pname)
    sites = [n for n in ast.walk(tree) if _is_site(n, pname)]
    assert names == len(sites), (
        f"{where}: `{pname}` is used other than as `{pname}[address]`; a port "
        f"is reached only by subscript (README D-12)")
    assert sites, f"{where}: binds the port and never accesses it"
    loops = [st for st in tree.body if isinstance(st, ast.For)]
    in_loop = {id(n) for lp in loops for n in ast.walk(lp)}
    assert len(loops) == 1 and all(id(n) in in_loop for n in sites), (
        f"{where}: every access to a memory port sits in the unit's one "
        f"top-level `for` loop, whose iteration is one cycle of the port "
        f"(README D-12)")
    loop = loops[0]
    loads, stores = [], []
    for st in loop.body:
        inner = [n for n in ast.walk(st) if _is_site(n, pname)]
        if not inner:
            continue
        if isinstance(st, ast.If) and not st.orelse and len(st.body) == 1 \
                and isinstance(st.body[0], ast.Assign) \
                and len(st.body[0].targets) == 1 \
                and _is_site(st.body[0].targets[0], pname) \
                and len(inner) == 1:
            tgt = st.body[0].targets[0]
            stores.append((tgt.slice, st.body[0].value, st.test, st))
            continue
        assert not isinstance(st, (ast.If, ast.For, ast.While, ast.With)), (
            f"{where}: line {st.lineno} accesses the port under control flow. "
            f"An access is made exactly once per iteration; only a write may "
            f"sit, alone, under `if <enable>:` (README D-12)")
        assert not isinstance(st, ast.AugAssign), (
            f"{where}: line {st.lineno} reads and writes through the port in "
            f"one statement; write the read and the write separately")
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and _is_site(st.targets[0], pname):
            assert len(inner) == 1, (
                f"{where}: line {st.lineno} reads and writes through the port "
                f"in one statement; write the read and the write separately")
            stores.append((st.targets[0].slice, st.value, None, st))
            continue
        assert all(isinstance(n.ctx, ast.Load) for n in inner), (
            f"{where}: line {st.lineno}: an unsupported write through the port")
        loads += [(n, st) for n in inner]
    if loads:
        assert port.reads, (
            f"{where}: reads through a write-only port (line "
            f"{loads[0][1].lineno}); direction is part of the port (README D-12)")
    if stores:
        assert port.writes, (
            f"{where}: writes through a read-only port (line "
            f"{stores[0][3].lineno}); direction is part of the port (README D-12)")
    for what, got in (("reads", loads), ("writes", stores)):
        assert len(got) <= port.count, (
            f"{where}: {len(got)} {what} per iteration through a port that "
            f"serves count={port.count} (README D-12: accesses per iteration)")
    if port.kind == "rw":
        for k in range(min(len(loads), len(stores))):
            ra, wa = ast.unparse(loads[k][0].slice), ast.unparse(stores[k][0])
            assert ra == wa, (
                f"{where}: access {k} reads `{pname}[{ra}]` and writes "
                f"`{pname}[{wa}]`; one access of an rw port has one address")
    return {"loads": loads, "stores": stores, "loop": loop, "tree": tree}


LOWERINGS = ("local", "replica", "server", "shared")
# The lowering a multi-owner ported memory gets when the caller names none.
# `server` is the general one (rw ports, any read latency) and holds ONE
# storage: on the regfile both it and `replica` matched MiniTPU per cycle on
# Catapult, but the replica's three copies cost a Catapult area score of 8,576
# against the server's 3,877 (DC then merges the equal registers, OPT-1215, so
# its area alone hides the cost; record u2_d12_prototype_2026-10-04.rst).
DEFAULT_LOWERING = "server"


def _stmts(text):
    return ast.parse(textwrap.dedent(text)).body


def _addr_type(rows):
    return f"UInt({max(1, math.ceil(math.log2(max(2, rows))))})"


def _rename(tree, old, new):
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == old:
            n.id = new


def unit(**kwargs):
    """Declare a unit: ``@unit(instances=..., reads=..., writes=..., ...)``."""

    def wrap(body):
        return Unit(body=body, **kwargs)

    return wrap


@dataclass
class Directives:
    """What a unit may ask of the Vitis schedule, without having to know the
    names Allo gives its instances or the region it was composed into."""

    top: str
    parameters: dict
    # Set while an engine's directives run (README D-15): the unit binding it
    # and the slot it is bound through.
    unit: str = None
    engine: str = None

    def instance(self, unit_name: str, *pid: int) -> str:
        return "_".join([unit_name] + [str(p) for p in (pid or (0,))])

    def memory(self, name: str) -> str:
        return f"{self.top}:{name}"


@dataclass
class Architecture:  # pylint: disable=too-many-instance-attributes
    """A set of units, the channels that wire them, and the parameters they are
    all instantiated at. Emits the region, and applies the units' directives.

    Unit order is declaration order and it is load-bearing: Vitis ``csim`` runs
    a dataflow region's processes in it, so a consumer declared before its
    producer reads an empty stream (limitations register item 15).
    """

    name: str
    parameters: dict
    memories: tuple[Memory, ...]
    channels: tuple[Channel, ...]
    units: tuple[Unit, ...]
    # {memory: why no two write-capable ports write one word in one cycle}:
    # the premise a `collision="obligation"` memory needs (README D-12). Like
    # `deadlock_free_because`, nothing checks it; a stress cosim discharges it.
    obligations: dict = field(default_factory=dict)
    # README D-15. {slot: Engine}: the engine bound to every unit slot
    # ``<slot>_<field>``. ``order`` is the accumulate order the composite's
    # contract reference takes; ``accepts`` the other orders it admits, each
    # a DIFFERENT function verified against the reference with that order
    # (``reference_order``). With no ``order``, nothing is declared and the
    # bound orders must agree among themselves.
    engines: dict = field(default_factory=dict)
    order: str = None
    accepts: tuple = ()
    # README D-19. ``slots``: the ISA slots the base machine always has;
    # ``options``: the Option records composed in (``with_options``), whose
    # slots exist only with them. ``isa_slots(arch)`` reads both.
    slots: tuple = ()
    options: tuple = ()
    #: The order the composite's verdict uses: ``order``, or the accepted
    #: order a bound part changed it to. Set by ``_check``.
    reference_order: str = field(default=None, init=False)
    #: The geometry record ``parameters`` was given as, if any (README D-20).
    geometry: object = field(default=None, init=False)
    _region: object = field(default=None, init=False, repr=False)
    _regions: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.parameters, dict):
            self._bind_geometry()
        self._check()

    def _bind_geometry(self):
        """README D-20: ``parameters`` may be a frozen geometry record whose
        derived numbers are properties; its ``namespace()`` is what the units
        bind. A key of the namespace that names a property of the record must
        be that property's value -- a derived number typed in beside the one
        it must equal is refused, naming it. The units' ``legality`` then
        runs on the record's namespace, as on any parameter set."""
        record = self.parameters
        assert hasattr(record, "namespace"), (
            f"{self.name}: parameters={record!r} is neither a dict nor a "
            f"geometry record with namespace() (README D-20)")
        ns = dict(record.namespace())
        for key, value in ns.items():
            prop = getattr(type(record), key, None)
            if isinstance(prop, property):
                derived = getattr(record, key)
                assert value == derived, (
                    f"{self.name}: geometry {type(record).__name__} declares "
                    f"{key}={value!r}, but its derived {key} is {derived!r}; a "
                    f"derived parameter is a property, never a second "
                    f"declaration (README D-20)")
        self.geometry = record
        self.parameters = ns

    def _check(self):
        declared = {c.name for c in self.channels}
        shape = {c.name: c.shape for c in self.channels}
        memories = {m.name for m in self.memories if not m.ported}
        ported = {m.name: m for m in self.memories if m.ported}
        writer, reader, owner = {}, {}, {}
        self._check_engines()
        for u in self.units:
            u.check()
            for name in u.parameters + u.isa:
                assert name in self.parameters, (
                    f"{self.name}: unit {u.name} needs {name!r}, which this "
                    f"architecture does not define")
                assert not inspect.isfunction(self.parameters[name]), (
                    f"{self.name}: unit {u.name} binds function "
                    f"{self.parameters[name].__name__!r} as bare parameter "
                    f"{name!r}; declare a compose.Engine and bind its slots "
                    f"with engines= (README D-15)")
            if u.legality is not None:
                u.legality(u.bound_parameters(self.parameters)
                           if isinstance(u, Instance) else self.parameters)
            for role, names, seen in (("writes", u.writes, writer),
                                      ("reads", u.reads, reader)):
                for ch in names:
                    assert ch in declared, (
                        f"{self.name}: unit {u.name} {role} undeclared "
                        f"channel {ch!r}")
                    # Allo allows one reader and one writer per stream. On a
                    # chain (a stream ARRAY) the owner is per element -- stage
                    # i reads `ch[i]` and writes `ch[i + 1]` -- and which
                    # element is a runtime index, so only the scalar channels
                    # can be held to it here.
                    assert ch not in seen or shape[ch], (
                        f"{self.name}: channel {ch!r} {role} in both "
                        f"{seen[ch]} and {u.name}")
                    seen[ch] = u.name
            params = list(inspect.signature(u.body).parameters)
            for pname, mem in zip(params, u.memories):
                if "." in mem:  # a port of an on-chip memory (README D-12)
                    mname, pport = mem.split(".", 1)
                    assert mname in ported, (
                        f"{self.name}: unit {u.name} binds port {mem!r}, but "
                        f"{mname!r} is not a memory with ports"
                        + (" (it is a boundary array: bind it by name)"
                           if mname in memories else ""))
                    port = ported[mname].port(pport)
                    assert mem not in owner, (
                        f"{self.name}: port {mem!r} is bound by both "
                        f"{owner[mem]} and {u.name}; each port has exactly "
                        f"one owner (README D-12). A port several units "
                        f"share in hardware is owned by one unit fed over "
                        f"channels")
                    owner[mem] = u.name
                    port_uses(u, pname, mem, port)
                    continue
                assert mem not in ported, (
                    f"{self.name}: unit {u.name} binds memory {mem!r} itself; "
                    f"a memory with ports is reached through a port: bind one "
                    f"of {', '.join(f'{mem}.{p.name}' for p in ported[mem].ports)} "
                    f"(README D-12)")
                assert mem in memories, (
                    f"{self.name}: unit {u.name} names {mem!r}, which is not a "
                    f"region argument")
                assert mem not in owner, (
                    f"{self.name}: memory {mem!r} is addressed by both "
                    f"{owner[mem]} and {u.name}")
                owner[mem] = u.name
        for ch in sorted(declared):
            assert ch in writer, f"{self.name}: nothing writes {ch!r}"
            assert ch in reader, f"{self.name}: nothing reads {ch!r}"
        for m in ported.values():
            for p in m.ports:
                assert f"{m.name}.{p.name}" in owner, (
                    f"{self.name}: port {m.name}.{p.name} has no owner; a "
                    f"port nobody binds is refused, like a channel nobody "
                    f"reads (README D-12)")
            nw = sum(p.count for p in m.ports if p.writes)
            if nw > 1 and m.collision == "obligation":
                assert self.obligations.get(m.name), (
                    f"{self.name}: memory {m.name} has {nw} write-capable "
                    f"ports and collision='obligation': state why no two of "
                    f"them write one word in one cycle, "
                    f"Architecture(obligations={{{m.name!r}: '<why>'}}) "
                    f"(README D-12; a stress cosim discharges it)")
        for name in self.obligations:
            assert name in ported, (
                f"{self.name}: obligations name {name!r}, which is not a "
                f"memory with ports")

    # -- calls (README D-19: a unit's own functions) ---------------------------

    def _call_namespace(self) -> dict:
        out = {}
        for u in self.units:
            for n, fn in u.call_targets().items():
                assert out.get(n, fn) is fn, (
                    f"{self.name}: two units call different functions named "
                    f"{n!r} ({out[n].__module__} and {fn.__module__}); one "
                    f"region has one namespace")
                assert n not in self.parameters, (
                    f"{self.name}: {n!r} is both a parameter and a function "
                    f"unit {u.name} calls")
                out[n] = fn
        return out

    # -- optional modules (README D-19) ---------------------------------------

    @classmethod
    def with_options(cls, base: "Architecture", *options: "Option", name=None):
        """``base`` with ``options`` composed in: each option's units,
        channels, memories, parameters and engines added, its ``rebind``
        applied to the neighbours it names (an ``Instance`` of each, README
        D-17), and its ISA slots added. Legal iff the result passes the
        netlist rules: an option added without its rebind, or a channel left
        with one endpoint, is refused at composition, naming the channel."""
        names = [o.name for o in base.options]
        units = list(base.units)
        channels, memories = list(base.channels), list(base.memories)
        params, engines = dict(base.parameters), dict(base.engines)
        slots = {s: "base" for s in base.slots}
        for o in base.options:
            slots.update({s: o.name for s in o.isa})
        for o in options:
            assert isinstance(o, Option), f"{base.name}: {o!r} is not a compose.Option"
            assert o.name not in names, f"{base.name}: option {o.name!r} composed twice"
            names.append(o.name)
            for k, v in o.parameters.items():
                assert params.get(k, v) is v or params.get(k, v) == v, (
                    f"{base.name}: option {o.name} sets {k}={v!r}, which the "
                    f"architecture already sets to {params[k]!r}")
                params[k] = v
            for k, v in o.engines.items():
                assert engines.get(k, v) is v, (
                    f"{base.name}: option {o.name} binds engine slot {k!r}, "
                    f"which is already bound")
                engines[k] = v
            for s in o.isa:
                assert s not in slots, (
                    f"{base.name}: option {o.name} brings ISA slot {s!r}, which "
                    f"{slots[s]} already brings")
                slots[s] = o.name
            have = {u.name: i for i, u in enumerate(units)}
            for uname, bind in o.rebind.items():
                assert uname in have, (
                    f"{base.name}: option {o.name} rebinds unit {uname!r}, "
                    f"which the architecture does not have "
                    f"(units: {sorted(have)})")
                units[have[uname]] = Instance(units[have[uname]], uname, bind)
            channels += list(o.channels)
            memories += list(o.memories)
            units += list(o.units)
        return cls(name=name or "_".join([base.name] + [o.name for o in options]),
                   parameters=params, memories=tuple(memories),
                   channels=tuple(channels), units=tuple(units),
                   obligations=dict(base.obligations), engines=engines,
                   order=base.order, accepts=base.accepts, slots=base.slots,
                   options=tuple(base.options) + tuple(options))

    # -- engines (README D-15) -------------------------------------------------

    def _engine_namespace(self) -> dict:
        out = {}
        for slot, eng in self.engines.items():
            out.update(eng.namespace(slot))
        return out

    def _check_engines(self):
        """Every slot a unit binds has an engine, every engine is bound, the
        values the bodies move in an engine's types travel on channels of
        those types, and every order agrees with the composite's."""
        for slot, eng in self.engines.items():
            assert isinstance(eng, Engine), (
                f"{self.name}: engines[{slot!r}] is {eng!r}, not a compose.Engine")
            for n in Engine.names(slot):
                assert n not in self.parameters, (
                    f"{self.name}: {n!r} is both a parameter and a name of the "
                    f"engine bound to slot {slot!r}")
        used = set()
        for u in self.units:
            for slot in u.engine_slots:
                assert slot in self.engines, (
                    f"{self.name}: unit {u.name} binds engine slot {slot!r} "
                    f"({', '.join(sorted(n for n in u.engines if n.startswith(slot + '_')))}), "
                    f"but this architecture binds no engine to it "
                    f"(Architecture(engines={{{slot!r}: <Engine>}}), README D-15)")
                used.add(slot)
        for slot, eng in self.engines.items():
            assert slot in used, (
                f"{self.name}: engine {eng.name} is bound to slot "
                f"{slot!r}, which no unit binds")
        if self.engines:
            self._check_engine_channels()
        self.reference_order = self._check_order()

    def _check_engine_channels(self):
        """README D-15: a value a body annotates with an engine's type travels
        on a channel of that type; a packed channel's ``lane_bits`` is the
        engine width the body slices it by (``lane_bits == IN_BITS``)."""
        ns = dict(FRONTEND_NAMES)
        ns.update(self.parameters)
        ns.update(self._engine_namespace())
        chans = {c.name: c for c in self.channels}

        def ev(text, what):
            try:
                return eval(text, {}, ns)  # pylint: disable=eval-used
            except Exception as e:
                raise AssertionError(f"{self.name}: {what} {text!r} does not "
                                     f"evaluate: {e}") from e

        def method_on_channel(n, method):
            return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == method and chan_of(n) is not None

        def chan_of(call):
            v = call.func.value if isinstance(call.func, ast.Attribute) else None
            while isinstance(v, ast.Subscript):
                v = v.value
            return v.id if isinstance(v, ast.Name) and v.id in chans else None

        for u in self.units:
            slots = set(u.engine_slots)
            if not slots:
                continue
            tree = ast.parse(u.source()).body[0]
            ann = {n.target.id: n.annotation for n in ast.walk(tree)
                   if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
            sites = []
            for n in ast.walk(tree):
                if isinstance(n, ast.AnnAssign) and method_on_channel(n.value, "get"):
                    sites.append((chan_of(n.value), n.annotation, n.lineno))
                elif method_on_channel(n, "put") and n.args \
                        and isinstance(n.args[0], ast.Name) and n.args[0].id in ann:
                    sites.append((chan_of(n), ann[n.args[0].id], n.lineno))
            for ch, a, line in sites:
                hits = [engine_slot(x.id) for x in ast.walk(a)
                        if isinstance(x, ast.Name) and x.id in u.engines]
                hits = [h for h in hits if h and h[0] in slots]
                if not hits:
                    continue
                c = chans[ch]
                if c.lane_bits:  # the specific rule first: lane width vs engine width
                    lb = ev(c.lane_bits, f"channel {ch} lane_bits")
                    for slot, f in hits:
                        if f.endswith("_BITS"):
                            eb = getattr(self.engines[slot], f)
                            assert lb == eb, (
                                f"{self.name}: channel {ch!r} declares lane_bits="
                                f"{c.lane_bits!r} ({lb}), but unit {u.name} slices "
                                f"it by {slot}_{f} = {eb} (engine "
                                f"{self.engines[slot].name}; README D-15)")
                want, got = ev(ast.unparse(a), f"unit {u.name} annotation"), \
                    ev(c.dtype, f"channel {ch} dtype")
                assert _type_sig(want) == _type_sig(got), (
                    f"{self.name}: unit {u.name} line {line} moves "
                    f"`{ast.unparse(a)}` ({want}, engine "
                    f"{', '.join(sorted({self.engines[h[0]].name for h in hits}))}) on "
                    f"channel {ch!r}, which carries {c.dtype!r} ({got}) (README D-15)")

    def _check_order(self):
        """README D-15: the composite's contract reference takes ``order``.
        An engine (or a unit that is one) of another order is refused unless
        the composite ``accepts`` it; then the verdict uses the reference
        with THAT order. Returns the order the verdict uses."""
        parts = [(f"engine slot {s!r} ({e.name})", e.order)
                 for s, e in self.engines.items()]
        parts += [(f"unit {u.name}", u.order) for u in self.units if u.order]
        if self.order is None:
            assert not self.accepts, (
                f"{self.name}: accepts={self.accepts} without order=: declare "
                f"the order the contract reference takes")
            orders = {o for _, o in parts}
            assert len(orders) <= 1, (
                f"{self.name}: {'; '.join(f'{p} accumulates in {o!r}' for p, o in parts)}: "
                f"a composite of two orders computes neither reference; declare "
                f"Architecture(order=, accepts=) (README D-15)")
            return orders.pop() if orders else None
        assert self.order in ORDERS, (
            f"{self.name}: order {self.order!r} is not one of {ORDERS}")
        for o in self.accepts:
            assert o in ORDERS and o != self.order, (
                f"{self.name}: accepts {o!r}, not another of {ORDERS}")
        allowed = {self.order} | set(self.accepts)
        for part, o in parts:
            assert o in allowed, (
                f"{self.name}: {part} accumulates in order {o!r}, but the "
                f"composite's contract reference takes {self.order!r}"
                + (f" and accepts only {tuple(self.accepts)}" if self.accepts
                   else " and accepts no other")
                + f". A swap that changes the order is a different function: "
                f"declare accepts=({o!r},) to verify it against the reference "
                f"with order {o!r} (README D-15)")
        changed = sorted({o for _, o in parts} - {self.order})
        assert len(changed) <= 1, (
            f"{self.name}: parts in orders {changed}; one composite has one "
            f"reference order")
        return changed[0] if changed else self.order

    # -- memories with ports (README D-12) -----------------------------------

    def _ported(self):
        return [m for m in self.memories if m.ported]

    def _owners(self, m):
        """``{port name: (unit, body parameter)}`` of one ported memory."""
        out = {}
        for u in self.units:
            params = list(inspect.signature(u.body).parameters)
            for pname, mem in zip(params, u.memories):
                if mem.startswith(m.name + "."):
                    out[mem.split(".", 1)[1]] = (u, pname)
        return out

    def _rows(self, m) -> int:
        return int(eval(str(m.rows), {}, dict(self.parameters)))  # pylint: disable=eval-used

    def plan(self, target=None, lowering=None) -> dict:
        """``{memory: lowering}`` for every memory with ports, checked.

        ``local``: one unit owns every port -- today's kernel-local array, the
        degenerate case. ``replica``: every read port's owner holds a copy and
        the one write port's owner sends each write to every copy (the
        write-broadcast replica; ``r`` ports at latency 0 and one ``w`` port
        only). ``server``: one storage in a generated kernel; each port is a
        bundle of links (address, data, enable) to its owner. ``shared``: one
        region-scope ``Stateful`` the owners address directly -- refused by
        D-11 unless premised, kept to record the refusal.

        Vitis refuses a memory with more than one owner (README D-12).
        """
        lowering = dict(lowering or {})
        names = {m.name for m in self._ported()}
        for k in lowering:
            assert k in names, f"{self.name}: lowering names {k!r}, not a memory with ports"
        out = {}
        for m in self._ported():
            owners = sorted({u.name for u, _ in self._owners(m).values()})
            if target == "vhls" and len(owners) > 1:
                raise NotImplementedError(
                    f"{self.name}: memory {m.name} has {len(owners)} owners "
                    f"({', '.join(owners)}); Vitis refuses a memory with more "
                    f"than one owner (README D-12). Build it with "
                    f'target="systemc" or "simulator"')
            choice = lowering.get(m.name) or (
                "local" if len(owners) == 1 else DEFAULT_LOWERING)
            assert choice in LOWERINGS, (
                f"memory {m.name}: lowering {choice!r} is not one of {LOWERINGS}")
            if choice == "local":
                assert len(owners) == 1, (
                    f"memory {m.name}: the local lowering needs one owner of "
                    f"every port; it has {owners}")
            if choice in {"replica", "server"}:
                for p in m.ports:
                    assert not p.writes or p.visible == 1, (
                        f"memory {m.name}: port {p.name} declares visible="
                        f"{p.visible}; only visible=1 has a lowering (a write "
                        f"at a clock edge, seen from the next cycle)")
            if choice == "replica":
                ws = [p for p in m.ports if p.writes]
                assert len(ws) == 1 and ws[0].kind == "w" and ws[0].count == 1, (
                    f"memory {m.name}: the replica lowering takes one `w` port "
                    f"(count 1); it has {[p.name for p in ws]}. Two write ports "
                    f"need N_R x N_W copies and a live-value table: refused")
                for p in m.ports:
                    assert p.kind != "r" or p.latency == 0, (
                        f"memory {m.name}: the replica lowering reads a copy "
                        f"combinationally; port {p.name} declares latency "
                        f"{p.latency}")
            out[m.name] = choice
        return out

    def port_kernels(self, target=None, lowering=None) -> list:
        """The kernels a ported memory's lowering consists of -- the owners of
        its ports and any generated server -- as emitted module names
        (``<kernel>_0``): the synthesis group of the memory."""
        plan = self.plan(target, lowering)
        out = []
        for u in self.units:
            if any("." in mem for mem in u.memories):
                out.append(f"{u.name}_0")
        out += [f"{m}_mem_0" for m, c in plan.items() if c == "server"]
        return out

    def _pin(self, m, pk, role, port, links):
        """One link of a port's bundle: name and declaration."""
        name = f"{m.name}_{pk}_{role}"
        dtype = {"a": _addr_type(self._rows(m)), "q": m.dtype, "d": m.dtype,
                 "e": "uint1"}[role[0]]
        if links == "stream":
            kind = "stream"
        elif role == "q" and port.latency:
            kind = "wire"  # registered at the owner's edge, then the pipe
        else:
            kind = "comb"  # same-cycle pin: address, data, enable; latency-0 data
        return name, Channel(name, dtype, depth="2", kind=kind).declaration_as(kind)

    def _lower(self, plan, links):
        """Rewrite every port-owning unit for its memories' lowering.

        Returns ``(unit sources by name, extra channel declarations,
        region-scope declarations, generated kernel sources)``."""
        storage_attr = "" if links == "stream" else " @ Stateful(reset=False)"
        bindings = {}  # unit name -> [(pname, memory, port)]
        for m in self._ported():
            for pport, (u, pname) in self._owners(m).items():
                bindings.setdefault(u.name, []).append((pname, m, m.port(pport)))
        chans, region_decls, servers, srcs = [], [], [], {}
        iters = {}  # memory -> {loop iter text: [units]}
        trees = {}
        for u in self.units:
            if u.name not in bindings:
                continue
            assert tuple(u.instances) == ("1",), (
                f"unit {u.name} owns a memory port and is replicated "
                f"(instances={u.instances}); each port has one owner")
            tree = ast.parse(u.source()).body[0]
            trees[u.name] = tree
            for pname, m, port in bindings[u.name]:
                uses = port_uses(u, pname, f"{m.name}.{port.name}", port, tree)
                iters.setdefault(m.name, {}).setdefault(
                    ast.unparse(uses["loop"].iter), []).append(u.name)
                self._rewrite(plan[m.name], tree, pname, m, port, uses,
                              links, storage_attr, chans, region_decls)
        for m in self._ported():
            if plan[m.name] in {"server", "replica"}:
                its = iters.get(m.name, {})
                assert len(its) == 1, (
                    f"memory {m.name}: its owners iterate differently "
                    f"({its}); one iteration is one cycle of every port, so "
                    f"the {plan[m.name]} lowering needs one loop range")
            if plan[m.name] == "server":
                servers.append(self._server(m, next(iter(iters[m.name])),
                                            storage_attr))
        for u in self.units:
            if u.name not in trees:
                continue
            tree = trees[u.name]
            ported_params = {p for p, _, _ in bindings[u.name]}
            keep = [(a, mem) for a, mem in zip(tree.args.args, u.memories)
                    if a.arg not in ported_params]
            tree.args.args = [a for a, _ in keep]
            tree.decorator_list = []
            args = [mem for _, mem in keep]
            deco = f"@df.kernel(mapping=[{', '.join(u.instances)}]" + (
                f", args=[{', '.join(args)}])" if args else ")")
            srcs[u.name] = textwrap.indent(deco + "\n" + ast.unparse(tree), "    ")
        return srcs, chans, region_decls, servers

    def _rewrite(self, choice, tree, pname, m, port, uses, links,
                 storage_attr, chans, region_decls):
        loop = uses["loop"]
        rows, dt = m.rows, m.dtype
        at = _addr_type(self._rows(m))
        attr = storage_attr if not m.reset else ""
        if choice in {"local", "shared"}:
            name = m.name
            _rename(tree, pname, name)
            decl = f"{name}: {dt}[{rows}]" + (
                attr if choice == "local" else
                (" @ Stateful(reset=False)" if not m.reset else " @ Stateful"))
            if choice == "local":
                if not any(isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == name
                           for s in tree.body):
                    tree.body[:0] = _stmts(decl)
            elif decl not in region_decls:
                region_decls.append(decl)
            return
        if choice == "replica" and port.kind == "r":
            copy_name = f"_{m.name}_{port.name}"
            _rename(tree, pname, copy_name)
            tree.body[:0] = _stmts(f"{copy_name}: {dt}[{rows}]{attr}")
            w = next(p for p in m.ports if p.writes)
            pre = f"_{m.name}_{port.name}_w"
            names = {}
            for role in ("a", "d", "e"):  # declared by the writer's side
                names[role] = self._pin(m, w.name, f"{role}_{port.name}", w, links)[0]
            loop.body += _stmts(f"""
                {pre}a: {at} = {names['a']}.get()
                {pre}i: int32 = {pre}a
                {pre}d: {dt} = {names['d']}.get()
                {pre}e: uint1 = {names['e']}.get()
                if {pre}e:
                    {copy_name}[{pre}i] = {pre}d
                """)
            return
        if choice == "replica":  # the one write port: broadcast to every copy
            readers = [p for p in m.ports if p.kind == "r"]
            sets = []
            for r in readers:
                pins = {}
                for role in ("a", "d", "e"):
                    nm, decl = self._pin(m, port.name, f"{role}_{r.name}", port, links)
                    pins[role] = nm
                    chans.append(decl)
                sets.append(pins)
            addr, value, enable, st = uses["stores"][0]
            pre = f"_{m.name}_{port.name}_"
            new = [f"{pre}a: {at} = {ast.unparse(addr)}",
                   f"{pre}d: {dt} = {ast.unparse(value)}",
                   f"{pre}e: uint1 = {ast.unparse(enable) if enable is not None else 1}"]
            for pins in sets:
                new += [f"{pins['a']}.put({pre}a)", f"{pins['d']}.put({pre}d)",
                        f"{pins['e']}.put({pre}e)"]
            i = loop.body.index(st)
            loop.body[i:i + 1] = _stmts("\n".join(new))
            return
        # server: each access becomes a pin bundle to the memory's server kernel
        for k in range(port.count):
            pk = port.name + (str(k) if port.count > 1 else "")
            pre = f"_{m.name}_{pk}_"
            pins = {}
            roles = ["a"] + (["q"] if port.reads else []) + (["d", "e"] if port.writes else [])
            for role in roles:
                nm, decl = self._pin(m, pk, role, port, links)
                pins[role] = nm
                chans.append(decl)
            load = uses["loads"][k] if k < len(uses["loads"]) else None
            store = uses["stores"][k] if k < len(uses["stores"]) else None
            if load is not None:
                node, st = load
                new = [f"{pre}a: {at} = {ast.unparse(node.slice)}",
                       f"{pins['a']}.put({pre}a)",
                       f"{pre}q: {dt} = {pins['q']}.get()"]
                if port.writes and store is None:
                    new += [f"{pins['d']}.put(0)", f"{pins['e']}.put(0)"]
                i = loop.body.index(st)
                loop.body[i:i] = _stmts("\n".join(new))
                # the load itself becomes the data the server returned
                for parent in ast.walk(st):
                    for fld, val in ast.iter_fields(parent):
                        if val is node:
                            setattr(parent, fld, ast.Name(id=f"{pre}q", ctx=ast.Load()))
                        elif isinstance(val, list):
                            for j, v in enumerate(val):
                                if v is node:
                                    val[j] = ast.Name(id=f"{pre}q", ctx=ast.Load())
            if store is not None:
                addr, value, enable, st = store
                new = []
                if load is None:
                    new += [f"{pre}a: {at} = {ast.unparse(addr)}", f"{pins['a']}.put({pre}a)"]
                    if port.reads:
                        new += [f"{pre}q: {dt} = {pins['q']}.get()"]
                new += [f"{pre}d: {dt} = {ast.unparse(value)}",
                        f"{pins['d']}.put({pre}d)",
                        f"{pre}e: uint1 = {ast.unparse(enable) if enable is not None else 1}",
                        f"{pins['e']}.put({pre}e)"]
                i = loop.body.index(st)
                loop.body[i:i + 1] = _stmts("\n".join(new))
            if load is None and store is None:  # an unused pin of a count > 1 port
                new = [f"{pins['a']}.put(0)"]
                if port.reads:
                    new += [f"{pre}q: {dt} = {pins['q']}.get()"]
                if port.writes:
                    new += [f"{pins['d']}.put(0)", f"{pins['e']}.put(0)"]
                loop.body[0:0] = _stmts("\n".join(new))

    def _server(self, m, loop_iter, storage_attr):
        """The generated kernel that holds a `server`-lowered memory: per
        iteration every port's address, then every read (latency 0: a
        combinational put; L >= 1: a pipe of L registers, as data), then every
        write -- so a read sees writes of earlier iterations only (visible=1)."""
        dt, rows = m.dtype, m.rows
        at = _addr_type(self._rows(m))
        attr = storage_attr if not m.reset else ""
        head = ["@df.kernel(mapping=[1])", f"def {m.name}_mem():",
                f"    mem: {dt}[{rows}]{attr}"]
        body, writes = [], []
        for p in m.ports:
            for k in range(p.count):
                pk = p.name + (str(k) if p.count > 1 else "")
                ch = f"{m.name}_{pk}"
                body += [f"_{pk}_a: {at} = {ch}_a.get()", f"_{pk}_i: int32 = _{pk}_a"]
                if p.kind == "rw" and p.latency and m.reset:
                    # RAM-mappable storage: one access per port per cycle, a
                    # write OR a read, as ONE if/else -- what Catapult needs to
                    # see the two as exclusive and use one RAM port, not two
                    # (u2_word_array_2026-10-02.rst). A read of a write cycle
                    # is undefined (MiniTPU masks it); a read of a word another
                    # port writes this cycle is the collision obligation.
                    head.append(f"    _{pk}_p: {dt}[{p.latency}]")
                    body += [f"_{pk}_d: {dt} = {ch}_d.get()",
                             f"_{pk}_e: uint1 = {ch}_e.get()"]
                    body += [f"_{pk}_p[{j}] = _{pk}_p[{j - 1}]"
                             for j in range(p.latency - 1, 0, -1)]
                    body += [f"if _{pk}_e:", f"    mem[_{pk}_i] = _{pk}_d",
                             "else:", f"    _{pk}_p[0] = mem[_{pk}_i]",
                             f"{ch}_q.put(_{pk}_p[{p.latency - 1}])"]
                    continue
                if p.reads and p.latency == 0:
                    body.append(f"{ch}_q.put(mem[_{pk}_i])")
                elif p.reads:
                    head.append(f"    _{pk}_p: {dt}[{p.latency}]")
                    body += [f"_{pk}_p[{j}] = _{pk}_p[{j - 1}]"
                             for j in range(p.latency - 1, 0, -1)]
                    body += [f"_{pk}_p[0] = mem[_{pk}_i]",
                             f"{ch}_q.put(_{pk}_p[{p.latency - 1}])"]
                if p.writes:
                    writes += [f"_{pk}_d: {dt} = {ch}_d.get()",
                               f"_{pk}_e: uint1 = {ch}_e.get()",
                               f"if _{pk}_e:", f"    mem[_{pk}_i] = _{pk}_d"]
        lines = head + [f"    for _ in {loop_iter}:"] + [f"        {x}" for x in body + writes]
        return textwrap.indent("\n".join(lines), "    ")

    def memory_manifest(self, target=None, lowering=None) -> dict:
        """What each memory with ports was lowered to: ``memory.json``
        (README D-12), written beside ``latency.json``."""
        plan = self.plan(target, lowering)
        links = "stream" if target == "simulator" else "declared"
        out = {}
        for m in self._ported():
            choice = plan[m.name]
            owners = self._owners(m)
            nread = sum(p.count for p in m.ports if p.reads)
            storage = ("registers (sc_signal array), unreset: clock-edge write "
                       "SC_METHOD, no reset action (D-14)"
                       if not m.reset and links != "stream" and target != "vhls"
                       else "the backend's array (reset storage)")
            impl = {
                "local": f"kernel-local array of {sorted({u.name for u, _ in owners.values()})[0]}"
                         " (one owner of every port: today's array)",
                "replica": f"write-broadcast replica x{nread}: one copy per read port, "
                           "inside its owner; every write sent to every copy",
                "server": f"one storage in generated kernel {m.name}_mem; each port a "
                          "bundle of links (address, data, enable) to its owner",
                "shared": "one region-scope Stateful the owners address (D-11 refuses "
                          "it unless premised; no HLS backend builds it)",
            }[choice]
            ports = {}
            for p in m.ports:
                d = p.manifest()
                d["owner"] = owners[p.name][0].name
                if choice in {"replica", "server"} and links != "stream":
                    if p.reads:
                        d["read"] = ("combinational (D-13 comb links)" if p.latency == 0
                                     else f"registered link + {p.latency}-deep pipe as data")
                    if p.writes:
                        d["write"] = "comb pins into a clock-edge write: seen next cycle"
                ports[p.name] = d
            out[m.name] = {
                "rows": self._rows(m), "dtype": m.dtype, "reset": m.reset,
                "collision": m.collision, "lowering": choice,
                "implementation": impl, "storage": storage,
                "target": target or "declared", "links": links, "ports": ports,
                "status": "untimed" if target == "simulator" else "lowered",
            }
            if m.name in self.obligations:
                out[m.name]["obligation"] = self.obligations[m.name]
        return out

    def build(self, target="simulator", lowering=None, schedule=None, **kwargs):
        """Customize, schedule and build the region for ``target``; with a
        ``project``, write ``memory.json`` there (README D-12)."""
        region = self.region(target, lowering)
        manifest = self.memory_manifest(target, lowering)
        if target == "simulator":
            for name, d in manifest.items():
                print(f"[memory] {name}: {d['lowering']}, untimed (the simulator "
                      f"keeps the order of accesses, not latency or visibility)")
        if target == "simulator":
            assert schedule is None, "the simulator takes no schedule"
            mod = df.build(region, target="simulator", **kwargs)
        else:
            s = df.customize(region)
            if schedule is not None:
                schedule(s)
            mod = s.build(target=target, **kwargs)
        project = kwargs.get("project")
        if project and manifest:
            os.makedirs(project, exist_ok=True)
            with open(os.path.join(project, "memory.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
        return mod

    def source(self, target=None, lowering=None) -> str:
        """The region's text. ``target="simulator"`` emits every link as a
        Stream (the simulator is untimed and refuses Wire); a memory with
        ports is lowered as ``plan`` says."""
        links = "stream" if target == "simulator" else None
        plan = self.plan(target, lowering)
        srcs, extra, region_decls, servers = (
            self._lower(plan, links or "declared") if plan else ({}, [], [], []))
        head = [f"def {self.name}("]
        head += [f"    {m.declaration}" for m in self.memories if not m.ported]
        head += ["):"]
        decls = [c.declaration_as(links or c.kind) for c in self.channels]
        decls += extra + region_decls
        parts = ["@df.region()", "\n".join(head),
                 "\n".join(f"    {d}" for d in decls)]
        for u in self.units:
            parts += ["", srcs.get(u.name) or u.kernel_source()]
        for s in servers:
            parts += ["", s]
        return "\n".join(parts) + "\n"

    def region(self, target=None, lowering=None):
        """The ``@df.region()``-decorated function, built from `source`.

        Registered in ``linecache`` under a stable pseudo-path because Allo
        reads a region back with ``inspect.getsourcelines``.
        ``ALLO_DUMP_COMPOSED=<dir>`` also writes the text out, to read or diff.
        """
        key = (target, tuple(sorted((lowering or {}).items())))
        if key == (None, ()) and self._region is not None:
            return self._region
        if key not in self._regions:
            src = self.source(target, lowering)
            path = f"<composed {self.name}>" if key == (None, ()) else \
                f"<composed {self.name} {target} {dict(key[1])}>"
            linecache.cache[path] = (len(src), None, src.splitlines(True), path)
            dump = os.environ.get("ALLO_DUMP_COMPOSED")
            if dump:
                suffix = "" if key == (None, ()) else \
                    "_" + "_".join([str(target)] + [f"{a}-{b}" for a, b in key[1]])
                with open(os.path.join(dump, f"{self.name}{suffix}.py"), "w",
                          encoding="utf-8") as f:
                    f.write(src)
            namespace = dict(FRONTEND_NAMES)
            namespace.update(self.parameters)
            namespace.update(self._engine_namespace())
            namespace.update(self._call_namespace())
            exec(compile(src, path, "exec"), namespace)  # pylint: disable=exec-used
            self._regions[key] = namespace[self.name]
            if key == (None, ()):
                self._region = self._regions[key]
        return self._regions[key]

    def machine(self, name=None):
        """This architecture's STRUCTURE as an ``allo.actions.Machine``: the
        units, the ports its channels and memories imply, the state they own
        and the channels' lane counts, with no instruction yet.

        The behavioural model is a separate file on purpose -- it imports
        nothing from Allo, so an ISA can be composed and checked without the
        front end -- and this is the one join between them. What it does NOT
        carry is arithmetic: a composed region declares what a unit is wired
        to and never what it computes, so a compute port is the Action
        layer's to add. :doc:`/developer/actions` has the measurement of how
        much of a hand-written machine this replaces, and of the one thing it
        cannot: a compose ``Unit`` is a kernel, replicated by ``instances``,
        and an Action ``Unit`` is one dispatch domain.
        """
        from allo.actions import structure  # noqa: PLC0415  -- one direction

        return structure(self, name)

    def directives(self, s):
        """Apply every unit's Vitis directives to a built schedule; after each
        unit's own, the directives of every engine the unit binds (README
        D-15, D-18), ONCE per distinct engine: an engine's directive names a
        loop of one of its functions, which the front end builds as one
        ``func.func`` shared by every caller (``leading_zeros19:offset``), so
        one application covers every unit binding it. ``ctx.unit`` and
        ``ctx.engine`` name the first binding. A directive naming a function
        the module does not hold as its own ``func.func`` (absent, or marked
        to be inlined) is refused naming the function, never dropped (D-1)."""
        ctx = Directives(top=s.top_func_name, parameters=self.parameters)
        applied = set()
        for u in self.units:
            if u.directives is not None:
                u.directives(s, ctx)
            for slot in sorted(u.engine_slots):
                eng = self.engines[slot]
                if eng.directives is None or id(eng) in applied:
                    continue
                applied.add(id(eng))
                eng.directives(_CarriedSchedule(s, eng.name),
                               Directives(top=ctx.top, parameters=ctx.parameters,
                                          unit=u.name, engine=slot))
        return s


class _CarriedSchedule:
    """The schedule an engine's directives see (README D-18): every
    primitive is the schedule's own, but a loop or array it names must sit
    in a function the module holds as its own ``func.func`` -- a carried
    directive names the function that needs it, never the region's top or
    an inlined helper, whose loops would vanish with the inlining."""

    def __init__(self, s, engine):
        self._s = s
        self._engine = engine

    def _function(self, target):
        where = f"engine {self._engine}: directive on {target!r}"
        assert ":" in target, (
            f"{where} names no function; a carried directive names the "
            f"function whose loop it schedules, '<function>:<loop>' (README D-18)")
        fname = target.split(":", 1)[0]
        func = self._s._find_function(fname, error=False)
        assert func is not None, (
            f"{where}: function {fname!r} is not a func.func of this module "
            f"(the front end inlined it, or nothing calls it); a directive on "
            f"an inlined function is refused, not dropped (README D-18, D-1)")
        assert "inline" not in func.attributes, (
            f"{where}: function {fname!r} is marked to be inlined, so its "
            f"loops vanish into every caller; a directive on an inlined "
            f"function is refused, not dropped (README D-18, D-1)")

    def __getattr__(self, attr):
        prim = getattr(self._s, attr)
        if not callable(prim):
            return prim

        def carried(*args, **kwargs):
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, str):
                    self._function(a)
            return prim(*args, **kwargs)

        return carried

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
import linecache
import os
import textwrap
from dataclasses import dataclass, field

import allo
import allo.dataflow as df
from allo.ir.types import Stream, UInt, int8, int16, int32

# Names every unit body may use without declaring them: the front end itself.
FRONTEND_NAMES = {"allo": allo, "df": df, "Stream": Stream, "UInt": UInt,
                  "int8": int8, "int16": int16, "int32": int32}
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

    @property
    def declaration(self) -> str:
        dims = f"[{', '.join(self.shape)}]" if self.shape else ""
        line = f"{self.name}: Stream[{self.dtype}, {self.depth}]{dims}"
        return f"{line}    # {self.carries}" if self.carries else line


@dataclass(frozen=True)
class Memory:
    """One array at the region boundary -- off-chip, one ``m_axi`` port."""

    name: str
    dtype: str

    @property
    def declaration(self) -> str:
        return f"{self.name}: {self.dtype},"


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

    @property
    def name(self) -> str:
        return self.body.__name__

    @property
    def declared_names(self) -> set[str]:
        return set(self.reads) | set(self.writes) | set(self.parameters) \
            | set(self.isa)

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
        free, declared = self.free_names(), self.declared_names
        assert free == declared, (
            f"unit {self.name}: the body uses {sorted(free - declared)} "
            f"without declaring them and declares "
            f"{sorted(declared - free)} without using them")


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

    def instance(self, unit_name: str, *pid: int) -> str:
        return "_".join([unit_name] + [str(p) for p in (pid or (0,))])

    def memory(self, name: str) -> str:
        return f"{self.top}:{name}"


@dataclass
class Architecture:
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
    _region: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._check()

    def _check(self):
        declared = {c.name for c in self.channels}
        shape = {c.name: c.shape for c in self.channels}
        memories = {m.name for m in self.memories}
        writer, reader, owner = {}, {}, {}
        for u in self.units:
            u.check()
            for name in u.parameters + u.isa:
                assert name in self.parameters, (
                    f"{self.name}: unit {u.name} needs {name!r}, which this "
                    f"architecture does not define")
            if u.legality is not None:
                u.legality(self.parameters)
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
            for mem in u.memories:
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

    def source(self) -> str:
        head = [f"def {self.name}("]
        head += [f"    {m.declaration}" for m in self.memories]
        head += ["):"]
        parts = ["@df.region()", "\n".join(head),
                 "\n".join(f"    {c.declaration}" for c in self.channels)]
        for u in self.units:
            parts += ["", u.kernel_source()]
        return "\n".join(parts) + "\n"

    def region(self):
        """The ``@df.region()``-decorated function, built from `source`.

        Registered in ``linecache`` under a stable pseudo-path because Allo
        reads a region back with ``inspect.getsourcelines``.
        ``ALLO_DUMP_COMPOSED=<dir>`` also writes the text out, to read or diff.
        """
        if self._region is None:
            src = self.source()
            path = f"<composed {self.name}>"
            linecache.cache[path] = (len(src), None, src.splitlines(True), path)
            dump = os.environ.get("ALLO_DUMP_COMPOSED")
            if dump:
                with open(os.path.join(dump, f"{self.name}.py"), "w") as f:
                    f.write(src)
            namespace = dict(FRONTEND_NAMES)
            namespace.update(self.parameters)
            exec(compile(src, path, "exec"), namespace)  # pylint: disable=exec-used
            self._region = namespace[self.name]
        return self._region

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
        """Apply every unit's Vitis directives to a built schedule."""
        ctx = Directives(top=s.top_func_name, parameters=self.parameters)
        for u in self.units:
            if u.directives is not None:
                u.directives(s, ctx)
        return s

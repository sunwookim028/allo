# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose an Allo dataflow region out of separately written units.

Allo reaches a ``@df.kernel`` only as a nested ``ast.FunctionDef`` inside the
``@df.region()`` that owns it, and resolves the names in its body against the
region's scope, so a unit cannot be imported into a region the way a Python
function is imported into a module. This module composes the region's *source*
instead: a unit is an ordinary module-level function plus an interface
declaration, and an ``Architecture`` binds that interface -- channels,
memories, parameters, opcodes -- when it emits the region.

What the front end refuses, and why this is the shape that works today, is on
``docs/source/designs/tinytpu_library.rst``.
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
    dtype: str
    depth: str
    shape: tuple[str, ...] = ()
    carries: str = ""

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

    def directives(self, s):
        """Apply every unit's Vitis directives to a built schedule."""
        ctx = Directives(top=s.top_func_name, parameters=self.parameters)
        for u in self.units:
            if u.directives is not None:
                u.directives(s, ctx)
        return s

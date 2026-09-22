# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dataflow region as a netlist over units that declare their stream ports.

A ``@df.unit`` names the streams it reads and writes in its own signature, so
its interface is a property of the unit rather than of the region it was pasted
into. This module holds the type that interface is, and the rules a wiring of
such interfaces has to satisfy. It refuses; it never transforms, and it
constructs no IR.

What is a rule here and what is an obligation is the split
:mod:`allo.dependence` worked out: every rule below is decided from the
signatures and the netlist alone, and **deadlock-freedom is not one of them**
-- it needs per-unit production and consumption rates, which a netlist does not
carry. See ``docs/source/developer/stream_ports.rst``.
"""

import ast
import copy
import inspect
import textwrap
import warnings
from dataclasses import dataclass, field

from ._mlir.exceptions import AlloError
from .ir.types import Stream, TypeAnnotation
from .ir.utils import get_global_vars

READS = ("get", "try_get", "empty")
WRITES = ("put", "try_put", "full")
IN, OUT, CHAIN = "in", "out", "io"

UNDECLARED_PREMISE = "<no premise declared>"


@dataclass(frozen=True)
class Violation:
    rule: str
    where: str
    found: str
    repair: str

    def __str__(self):
        pad = " " * 26
        return f"{self.rule:24s} {self.where}\n{pad}{self.found}\n{pad}repair: {self.repair}"


class NetlistError(AlloError):
    """A unit interface, or a wiring of them, that a rule refuses."""

    def __init__(self, where, found):
        self.violations = tuple(found)
        body = "\n".join(str(violation) for violation in self.violations)
        super().__init__(f"{len(self.violations)} wiring error(s) in {where}:\n{body}")


class UndeclaredPremise(UserWarning):
    """A netlist with feedback whose author has not said why it cannot deadlock."""


@dataclass(frozen=True)
class DeadlockObligation:
    """What an accepted netlist still rests on. Every rule in this module is
    decided from the netlist; whether the units' rates let the loops drain is
    not, so a netlist that carries feedback leaves one of these."""

    region: str
    cycles: tuple
    premise: str = None

    def __str__(self):
        loops = "; ".join(" -> ".join(cycle) for cycle in self.cycles)
        return f"{self.region}: feedback through {loops}\n    {self.premise or UNDECLARED_PREMISE}"


@dataclass(frozen=True)
class Port:
    """One stream port of a unit: the whole of its declared interface."""

    name: str
    direction: str
    dtype: Stream
    array_shape: tuple = ()

    @property
    def signature(self):
        shape = "".join(f"[{s}]" for s in self.array_shape)
        return f"Stream[{self.dtype.dtype}, {self.dtype.depth}]{shape}"

    def __str__(self):
        return f"{self.name}: {self.signature} ({self.direction})"


@dataclass(frozen=True)
class Channel:
    """One stream declared in a region body, as the netlist sees it."""

    name: str
    dtype: Stream = None
    array_shape: tuple = ()

    @property
    def depth(self):
        return 0 if self.dtype is None else self.dtype.depth


@dataclass(frozen=True)
class Instance:
    name: str
    unit: "UnitSpec"
    bindings: dict


def _root_name(node):
    if isinstance(node, ast.Name):
        return node.id, False
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id, True
    return None, False


def stream_uses(body):
    """Every stream method call in ``body``, as ``name -> (directions, subscripted)``."""
    uses = {}
    for statement in body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in READS + WRITES
            ):
                continue
            name, subscripted = _root_name(node.func.value)
            if name is None:
                continue
            directions, seen = uses.setdefault(name, (set(), set()))
            directions.add(IN if node.func.attr in READS else OUT)
            seen.add(subscripted)
    return uses


def _evaluate_annotation(annotation, namespace):
    try:
        return eval(  # pylint: disable=eval-used
            compile(ast.Expression(copy.deepcopy(annotation)), "<annotation>", "eval"),
            dict(namespace),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _mentions_stream(annotation):
    return any(
        isinstance(node, ast.Name) and node.id == Stream.__name__
        for node in ast.walk(annotation)
    )


def as_stream(value):
    """``(element type, array shape)`` if ``value`` annotates a stream, else None."""
    if isinstance(value, Stream):
        return value, ()
    if isinstance(value, TypeAnnotation) and isinstance(value.dtype, Stream):
        return value.dtype, tuple(value.shape)
    return None


def same_stream(left, right):
    """Element type, element shape and depth all agree."""
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and left.depth == right.depth
    )


class UnitSpec:
    """The declared interface of a ``@df.unit``, checked against its body.

    A **scalar** port is legal if and only if the body uses it at least once
    and in one direction only. An **array** port is a chain: its element index
    is a runtime value, so a body that reads ``ch[i]`` and writes ``ch[i + 1]``
    is reading and writing different streams and nothing static can tell. Such
    a port is ``io``, which is the direction the IR already carries for it, and
    the single-valued rule does not apply to it -- neither does
    single-producer-single-consumer on the stream it is wired to.

    Both rules are decided here, at the ``@df.unit`` that declared the port, so
    a unit with a broken interface never reaches a region.
    """

    def __init__(self, func, mapping=None):
        self.func = func
        self.name = func.__name__
        self.mapping = list(mapping) if mapping is not None else [1]
        self.tree = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
        # Captured here, at the definition, and not again at the region: a
        # unit's names are resolved where it was written, which is what makes
        # it independent of wherever it is instantiated.
        self.namespace = get_global_vars(func)
        namespace = self.namespace
        uses = stream_uses(self.tree.body)
        ports, values, found = [], [], []
        for argument in self.tree.args.args:
            stream = as_stream(_evaluate_annotation(argument.annotation, namespace))
            if stream is None:
                if _mentions_stream(argument.annotation):
                    found.append(
                        Violation(
                            "unresolved-port",
                            f"{self.name}({argument.arg})",
                            f"`{ast.unparse(argument.annotation)}` names Stream "
                            "but does not resolve to one where the unit is defined",
                            "define the element type and the depth in the unit's "
                            "own module, so the port does not depend on where it "
                            "is instantiated",
                        )
                    )
                    continue
                values.append(argument.arg)
                continue
            dtype, array_shape = stream
            directions, subscripted = uses.get(argument.arg, (set(), set()))
            if not directions:
                found.append(
                    Violation(
                        "dangling-port",
                        f"{self.name}({argument.arg})",
                        "declared as a stream port and never read or written",
                        "use the port in the body, or drop it from the signature",
                    )
                )
                continue
            if len(directions) > 1 and not array_shape:
                found.append(
                    Violation(
                        "direction-single-valued",
                        f"{self.name}({argument.arg})",
                        "used both as an input (get/try_get/empty) and as an "
                        "output (put/try_put/full)",
                        "split it into one input port and one output port",
                    )
                )
                continue
            if subscripted != {bool(array_shape)}:
                found.append(
                    Violation(
                        "port-arity",
                        f"{self.name}({argument.arg})",
                        f"declared with array shape {array_shape or '()'} and "
                        f"used {'with' if True in subscripted else 'without'} a subscript",
                        "give the port the array shape its uses need",
                    )
                )
                continue
            direction = directions.pop() if len(directions) == 1 else CHAIN
            ports.append(Port(argument.arg, direction, dtype, array_shape))
        if found:
            raise NetlistError(f"unit {self.name!r}", found)
        self.ports = tuple(ports)
        self.values = tuple(values)

    def port(self, name):
        for port in self.ports:
            if port.name == name:
                return port
        return None

    def __repr__(self):
        return f"UnitSpec({self.name}: {', '.join(str(p) for p in self.ports)})"


def wiring_arity(netlist):
    for instance in netlist.instances:
        expected = [port.name for port in instance.unit.ports] + list(
            instance.unit.values
        )
        for name in expected:
            if name not in instance.bindings:
                yield Violation(
                    "wiring-arity",
                    f"{netlist.region}.{instance.name}",
                    f"parameter {name!r} of unit {instance.unit.name!r} is not wired",
                    f"pass {name}=<name declared in the region body>",
                )
        for name in instance.bindings:
            if name not in expected:
                yield Violation(
                    "wiring-arity",
                    f"{netlist.region}.{instance.name}",
                    f"unit {instance.unit.name!r} has no parameter {name!r}",
                    f"the parameters are {', '.join(expected) or '(none)'}",
                )


def wiring_type(netlist):
    for instance in netlist.instances:
        for port in instance.unit.ports:
            target = instance.bindings.get(port.name)
            channel = netlist.channels.get(target)
            if channel is None:
                yield Violation(
                    "wiring-type",
                    f"{netlist.region}.{instance.name}.{port.name}",
                    f"{target!r} is not a stream declared in this region body",
                    "declare it as `name: Stream[dtype, depth]` above the instance",
                )
                continue
            if channel.dtype is None:
                continue
            if not same_stream(port.dtype, channel.dtype) or tuple(
                channel.array_shape
            ) != tuple(port.array_shape):
                yield Violation(
                    "wiring-type",
                    f"{netlist.region}.{instance.name}.{port.name}",
                    f"port is {port.signature}, channel {target!r} is "
                    f"Stream[{channel.dtype.dtype}, {channel.dtype.depth}]"
                    + "".join(f"[{s}]" for s in channel.array_shape),
                    "make the element type, the element shape, the depth and "
                    "the array shape agree",
                )


def single_producer_single_consumer(netlist):
    for name, channel in netlist.channels.items():
        if channel.array_shape:
            continue
        writers, readers = netlist.writers(name), netlist.readers(name)
        for role, who in ((OUT, writers), (IN, readers)):
            if len(who) > 1:
                yield Violation(
                    "single-producer-single-consumer",
                    f"{netlist.region}.{name}",
                    f"{len(who)} {role}-ports on one stream: {', '.join(who)}",
                    "give each writer and each reader a stream of its own",
                )


def unconnected_stream(netlist):
    for name, channel in netlist.channels.items():
        writers, readers = netlist.writers(name), netlist.readers(name)
        if writers and readers:
            continue
        missing = "no writer" if not writers else "no reader"
        yield Violation(
            "unconnected-stream",
            f"{netlist.region}.{name}",
            f"stream {name!r} has {missing}",
            "wire it to a port of some instance, or delete the declaration",
        )
        _ = channel


def zero_capacity_cycle(netlist):
    edges = {}
    for name, channel in netlist.channels.items():
        if channel.depth > 0:
            continue
        for writer in netlist.writers(name):
            for reader in netlist.readers(name):
                edges.setdefault(_instance_of(writer), []).append(
                    (_instance_of(reader), name)
                )
    for cycle in _cycles(edges):
        yield Violation(
            "zero-capacity-cycle",
            f"{netlist.region}",
            "feedback through depth-0 streams only: " + " -> ".join(cycle),
            "give one stream on the loop a depth of at least 1",
        )


RULES = (
    wiring_arity,
    wiring_type,
    single_producer_single_consumer,
    unconnected_stream,
    zero_capacity_cycle,
)


def _instance_of(endpoint):
    return endpoint.split(".", 1)[0]


def _cycles(edges):
    """Every simple cycle of the directed graph ``edges``, as node lists."""
    found, path, on_path = [], [], set()

    def walk(node):
        path.append(node)
        on_path.add(node)
        for successor, _ in edges.get(node, ()):
            if successor in on_path:
                loop = path[path.index(successor) :] + [successor]
                if loop not in found:
                    found.append(loop)
            elif successor not in visited:
                walk(successor)
        on_path.discard(node)
        path.pop()

    visited = set()
    for node in list(edges):
        if node not in visited:
            walk(node)
            visited.update(on_path)
            visited.add(node)
    return found


@dataclass
class Netlist:
    """A region's channels, the instances wired to them, and the rules."""

    region: str
    channels: dict = field(default_factory=dict)
    instances: list = field(default_factory=list)
    lexical: dict = field(default_factory=dict)

    def _endpoints(self, name, direction):
        who = [
            f"{instance.name}.{port.name}"
            for instance in self.instances
            for port in instance.unit.ports
            if port.direction in (direction, CHAIN)
            and instance.bindings.get(port.name) == name
        ]
        who += [
            f"{owner}(lexical)" for owner in self.lexical.get((name, direction), ())
        ]
        return who

    def writers(self, name):
        return self._endpoints(name, OUT)

    def readers(self, name):
        return self._endpoints(name, IN)

    def violations(self):
        return [violation for rule in RULES for violation in rule(self)]

    def feedback(self):
        edges = {}
        for name in self.channels:
            for writer in self.writers(name):
                for reader in self.readers(name):
                    edges.setdefault(_instance_of(writer), []).append(
                        (_instance_of(reader), name)
                    )
        return tuple(tuple(cycle) for cycle in _cycles(edges))

    def check(self, because=None, stacklevel=4):
        """Run every rule, then record what no rule can settle.

        Raises :class:`NetlistError` naming every site it refuses. Returns the
        :class:`DeadlockObligation` left over when the netlist carries
        feedback, and ``None`` when it does not.
        """
        found = self.violations()
        if found:
            raise NetlistError(f"region {self.region!r}", found)
        cycles = self.feedback()
        if not cycles:
            return None
        if because is None and _WARNED.isdisjoint({(self.region, cycles)}):
            _WARNED.add((self.region, cycles))
            warnings.warn(
                f"netlist: {self.region!r} carries feedback through "
                + "; ".join(" -> ".join(cycle) for cycle in cycles)
                + ". Every rule here is decided from the netlist, and "
                "deadlock-freedom is not one of them -- it needs per-unit "
                "rates, which this does not have. Nothing has checked that "
                "these loops drain. Say what makes them drain with "
                "@df.region(deadlock_free_because=...) -- discharging it is "
                "yours, outside the compiler.",
                UndeclaredPremise,
                stacklevel=stacklevel,
            )
        return DeadlockObligation(self.region, cycles, because)


REGISTRY = {}
_WARNED = set()


def netlist_of(region):
    """The netlist last built for ``region`` (a name or a ``@df.region``)."""
    name = region if isinstance(region, str) else region.__name__
    return REGISTRY.get(name)

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python representation of the RTL emitter's module interface manifest"""

from __future__ import annotations

import json

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Control:
    """The fixed control ABI every emitted module carries"""

    clk: str
    rst: str
    start: str
    done: str

    @classmethod
    def from_json(cls, d: dict) -> Control:
        return cls(d["clk"], d["rst"], d["start"], d["done"])


@dataclass(frozen=True)
class Scalar:
    """A scalar input argument (one port, no suffix)."""

    arg: int
    width: int
    name: str

    @classmethod
    def from_json(cls, d: dict) -> Scalar:
        return cls(d["arg"], d["width"], d["name"])


@dataclass(frozen=True)
class FIFO:
    """A FIFO channel interface. An input (a ``get``) reads ``data`` when
    ``valid`` and drives ``ready``; an output (a ``put``) drives ``data`` and
    ``valid`` and reads ``ready``."""

    arg: int  # kernel block-argument index (-1 if not an argument)
    is_input: bool  # get (input) vs put (output)
    depth: int
    width: int  # payload bit width
    base: str
    data: str
    valid: str
    ready: str

    @classmethod
    def from_json(cls, d: dict) -> FIFO:
        return cls(
            d["arg"],
            d["input"],
            d["depth"],
            d["width"],
            d["base"],
            d["data"],
            d["valid"],
            d["ready"],
        )


@dataclass(frozen=True)
class Memory:
    """One physical interface to an argument array (a single bank of it when the
    argument is cyclically partitioned). A read exposes ``{addr(out),
    data(in)}``; a write ``{addr, data, we}``, all out."""

    @dataclass(frozen=True)
    class Axis:
        """One partitioned axis of the argument, mirroring
        ``allo::BankLayout::Axis``: the element-space decomposition the RTL
        addresses with. ``kind`` is "cyclic", "block" or "skew"."""

        dim: int
        factor: int
        kind: str

        @classmethod
        def from_json(cls, d: dict) -> Memory.Axis:
            return cls(d["dim"], d["factor"], d["kind"])

    arg: int
    bank: int  # the bank this interface serves
    factor: int  # total physical banks
    width: int  # element bit width
    latency: int  # access latency
    base: str
    addr: str
    data: str
    we: str | None = None  # None on a read interface
    shape: tuple[int, ...] = ()  # the argument's element shape
    axes: tuple[Axis, ...] = ()  # partitioned axes, mixed-radix order

    @property
    def write(self) -> bool:
        return self.we is not None

    @classmethod
    def from_json(cls, d: dict) -> Memory:
        return cls(
            d["arg"],
            d["bank"],
            d["factor"],
            d["width"],
            d["latency"],
            d["base"],
            d["addr"],
            d["data"],
            d.get("we"),
            tuple(d.get("shape", ())),
            tuple(cls.Axis.from_json(a) for a in d.get("axes", ())),
        )


@dataclass(frozen=True)
class RegisterFile:
    """A completely-partitioned argument array, crossing the boundary as one port
    per element rather than an addressed interface. ``elements`` is flat
    row-major, so element k of the flattened argument drives ``elements[k]``. Not
    a :class:`Memory`: no address, no bank, no access latency."""

    @dataclass(frozen=True)
    class Element:
        """One element's ports: ``in_`` where it arrives, ``out``/``we`` where it
        leaves. A direction the kernel does not use is ``None``, and the names
        differ by that (``A_k`` when one direction is live, ``A_k_in``/``A_k_out``
        when both are)."""

        in_: str | None = None
        out: str | None = None
        we: str | None = None

        @classmethod
        def from_json(cls, d: dict) -> RegisterFile.Element:
            return cls(d.get("in"), d.get("out"), d.get("we"))

    arg: int
    width: int  # element bit width
    shape: tuple[int, ...]  # the argument's element shape
    elements: tuple[Element, ...]

    @property
    def writeback(self) -> bool:
        return any(e.out is not None for e in self.elements)

    @classmethod
    def from_json(cls, d: dict) -> RegisterFile:
        return cls(
            d["arg"],
            d["width"],
            tuple(d["shape"]),
            tuple(cls.Element.from_json(e) for e in d["elements"]),
        )


@dataclass(frozen=True)
class Result:
    """A scalar function result (one output port, driven at ``done``)."""

    width: int
    name: str

    @classmethod
    def from_json(cls, d: dict) -> Result:
        return cls(d["width"], d["name"])


@dataclass(frozen=True)
class Operator:
    """One extern operator module this module instantiates, with the port shape
    it was declared with. ``impl`` + ``predicate`` join it to the device
    operator, and the behavioral model is built from ``ports``."""

    class Role(str, Enum):
        """What a port is for, so a consumer classifies structurally rather than
        matching ``clk`` / ``ce`` by name."""

        DATA = "data"
        CLK = "clk"
        CE = "ce"
        OUT = "out"

    @dataclass(frozen=True)
    class Port:
        name: str
        width: int
        role: Operator.Role
        is_input: bool

        @classmethod
        def from_json(cls, d: dict) -> Operator.Port:
            return cls(d["name"], d["width"], Operator.Role(d["role"]), d["input"])

    module: str  # the extern module's RTL name
    impl: str  # the device operator's sym_name
    predicate: str  # compare predicate; empty for everything else
    ports: tuple[Port, ...]

    @classmethod
    def from_json(cls, d: dict) -> Operator:
        return cls(
            d["module"],
            d["impl"],
            d["predicate"],
            tuple(cls.Port.from_json(p) for p in d["ports"]),
        )


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class ModuleInterface:
    """The whole boundary of one module. ``reads``/``writes`` group by access, an
    inner tuple holding the access's per-bank interfaces: one entry unbanked, N
    when a data-dependent access spans every bank. ``module`` and ``symbol``
    differ whenever the symbol needed legalizing (``top.child`` ->
    ``top_child``), and the simulator only knows the former."""

    module: str
    symbol: str
    control: Control
    scalars: tuple[Scalar, ...]
    streams: tuple[FIFO, ...]
    reads: tuple[tuple[Memory, ...], ...]
    writes: tuple[tuple[Memory, ...], ...]
    registers: tuple[RegisterFile, ...]
    results: tuple[Result, ...]
    operators: tuple[Operator, ...]
    #: composition class: ``counted_static``, ``indeterminate`` or
    #: ``concurrent``.
    determinacy: str
    #: whether ``latency`` is a worst case rather than an exact count.
    latency_is_bound: bool
    #: start->done span in cycles, None when data-dependent.
    latency: int | None = None

    def ports_for_arg(self, arg: int) -> list[Memory]:
        """Every memory interface of argument ``arg``, reads before writes and
        flat across access groups. An argument accessed at several points has
        several groups (read-twice -> two reads, an accumulator -> a read and a
        write), and a partitioned access one interface per bank within its
        group; a caller wiring the argument needs all of them."""
        return [
            m
            for side in (self.reads, self.writes)
            for grp in side
            for m in grp
            if m.arg == arg
        ]

    def scalar_for_arg(self, arg: int) -> Scalar | None:
        """The scalar input port of argument ``arg``, or None if it is not one."""
        return next((s for s in self.scalars if s.arg == arg), None)

    def stream_for_arg(self, arg: int) -> FIFO | None:
        """The stream interface of argument ``arg``, or None if it is not one. A
        stream is single-ended within a module, so it has exactly one."""
        return next((s for s in self.streams if s.arg == arg), None)

    @property
    def latency_is_exact(self) -> bool:
        """Whether ``latency`` is the exact span the hardware realizes, and so a
        figure a measured cycle count may be held to."""
        return (
            self.latency is not None
            and self.determinacy == "counted_static"
            and not self.latency_is_bound
        )

    @classmethod
    def from_json(cls, d: dict) -> ModuleInterface:
        return cls(
            d["module"],
            d["symbol"],
            Control.from_json(d["control"]),
            tuple(Scalar.from_json(s) for s in d["scalars"]),
            tuple(FIFO.from_json(s) for s in d["streams"]),
            tuple(tuple(Memory.from_json(m) for m in acc) for acc in d["reads"]),
            tuple(tuple(Memory.from_json(m) for m in acc) for acc in d["writes"]),
            tuple(RegisterFile.from_json(r) for r in d["registers"]),
            tuple(Result.from_json(r) for r in d["results"]),
            tuple(Operator.from_json(o) for o in d["operators"]),
            d["determinacy"],
            d["latency_bound"],
            d.get("latency"),
        )


class Interfaces(Mapping[str, ModuleInterface]):
    """The manifest of a whole emission: {RTL module name -> its boundary}."""

    def __init__(self, modules: dict[str, ModuleInterface]):
        self._modules = dict(modules)

    def __getitem__(self, module: str) -> ModuleInterface:
        return self._modules[module]

    def __iter__(self):
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        return f"Interfaces({', '.join(self._modules)})"

    def of_symbol(self, symbol: str) -> ModuleInterface:
        """The manifest of the module emitted for the MLIR symbol ``symbol``,
        which differs from the RTL module name this map is keyed by whenever the
        symbol needed legalizing."""
        for iface in self._modules.values():
            if iface.symbol == symbol:
                return iface
        raise KeyError(f"no emitted module for symbol '{symbol}'")

    @classmethod
    def from_json(cls, doc: str | dict) -> Interfaces:
        """Parse the emitter's manifests, the one place they are decoded. Takes
        the ``interfaces`` member of the emit envelope, either as the raw string
        or as an already-decoded object."""
        d = json.loads(doc) if isinstance(doc, str) else doc
        return cls({k: ModuleInterface.from_json(v) for k, v in d.items()})

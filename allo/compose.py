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
import re
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
class Sram:
    """An SRAM macro a ``Memory`` is lowered onto: ``Memory(impl=Sram(...))``,
    the ``sram`` lowering (README D-12; ``asic_memories_2026-10-04.rst``).

    ``module`` is the macro's module and Catapult library name; ``verilog``
    its behavioural model and ``liberty`` its ``.lib`` (area, timing), both
    files; ``rows`` x ``width`` its geometry; ``ports`` the kinds of its
    physical ports (``("rw", "rw")`` for a 2RW macro); ``read_latency`` the
    edges from address to data (>= 1: an SRAM's read is synchronous) and
    ``visible`` the edges until a write is seen by any port. ``catapult_lib``
    is the compiled Catapult library (the Memory Generator's ``.lib``) when
    one is already built; otherwise the SystemC build makes it on a Catapult
    host. ``from_openram`` reads all of it from an OpenRAM ``.v`` + ``.lib``.
    """

    module: str
    verilog: str
    liberty: str
    rows: int
    width: int
    ports: tuple = ("rw", "rw")
    read_latency: int = 1
    visible: int = 1
    catapult_lib: str = None

    def __post_init__(self):
        assert self.module.isidentifier(), f"sram {self.module!r}: not a module name"
        assert self.ports and all(k in PORT_KINDS for k in self.ports), (
            f"sram {self.module}: ports {self.ports} are not kinds from {PORT_KINDS}")
        assert isinstance(self.read_latency, int) and self.read_latency >= 1, (
            f"sram {self.module}: read_latency={self.read_latency!r}; an SRAM's "
            f"read is synchronous (>= 1 edge)")
        assert isinstance(self.visible, int) and self.visible >= 1, (
            f"sram {self.module}: visible={self.visible!r} is not a count of edges")
        assert self.rows > 0 and self.width > 0, f"sram {self.module}: rows x width"

    @property
    def area_um2(self):
        """The Liberty ``area`` (OpenRAM: layout width x height, um^2), or None."""
        try:
            with open(self.liberty, encoding="utf-8") as f:
                m = re.search(r"^\s*area\s*:\s*([\d.]+)\s*;", f.read(), re.M)
        except OSError:
            return None
        return float(m.group(1)) if m else None

    @classmethod
    def from_openram(cls, verilog, liberty, catapult_lib=None):
        """An OpenRAM macro from its behavioural ``.v`` (module name,
        ``DATA_WIDTH``, ``ADDR_WIDTH``, the ``// Port i: RW|R|W`` lines) and
        its ``.lib``. OpenRAM's model registers its inputs at the rising edge
        and reads or writes at the falling edge: read latency 1, a write
        visible to the next cycle's read on any port."""
        with open(verilog, encoding="utf-8") as f:
            v = f.read()
        module = re.search(r"^module\s+(\w+)\s*\(", v, re.M).group(1)
        width = int(re.search(r"parameter DATA_WIDTH = (\d+)", v).group(1))
        aw = int(re.search(r"parameter ADDR_WIDTH = (\d+)", v).group(1))
        kinds = tuple(k.lower() for k in re.findall(r"// Port \d+: (RW|R|W)\n", v))
        assert kinds, f"{verilog}: no `// Port i: RW|R|W` lines; not an OpenRAM model"
        return cls(module, verilog, liberty, 1 << aw, width, kinds, 1, 1, catapult_lib)

    def manifest(self) -> dict:
        return {"module": self.module, "rows": self.rows, "width": self.width,
                "ports": list(self.ports), "read_latency": self.read_latency,
                "visible": self.visible, "area_um2": self.area_um2,
                "verilog": self.verilog, "liberty": self.liberty,
                "catapult_lib": self.catapult_lib}


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
    architecture's parameters. ``impl`` names the lowering the memory is
    built through -- ``"registers"`` (a flop array with combinational read
    muxes; the default), ``"replica"`` (one copy per read port, FPGA-only),
    or an ``Sram`` (the macro path) -- and ``region(lowering=...)`` can still
    override it per build.
    """

    name: str
    dtype: str
    rows: str = None
    ports: tuple = ()
    collision: str = "refuse"
    reset: bool = True
    impl: object = None

    def __post_init__(self):
        if self.rows is None:
            assert not self.ports, (
                f"memory {self.name}: ports need rows= (a memory without rows "
                f"is a boundary array with its one m_axi port)")
            assert self.impl is None, (
                f"memory {self.name}: impl= needs rows= and ports= (a boundary "
                f"array has no on-chip implementation)")
            return
        if self.impl is not None and not isinstance(self.impl, Sram):
            assert self.impl in LOWERINGS or self.impl in LOWERING_ALIASES, (
                f"memory {self.name}: impl={self.impl!r} is not one of "
                f"{LOWERINGS} or an Sram")
            assert self.impl != "sram", (
                f"memory {self.name}: impl='sram' names no macro; pass the "
                f"macro itself, impl=Sram(...) (or Sram.from_openram(...))")
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

    @property
    def lowering(self):
        """The lowering ``impl`` names (``sram`` for an ``Sram``), or None."""
        if isinstance(self.impl, Sram):
            return "sram"
        return LOWERING_ALIASES.get(self.impl, self.impl)

    @property
    def sram(self):
        return self.impl if isinstance(self.impl, Sram) else None

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


# `registers`: one flop array in a generated server kernel, every read a
# combinational mux (latency 0) or a pipe written as data (L >= 1); the ASIC
# form, and the general one (rw ports, any read latency). `sram`: the same
# server with its storage mapped onto a declared SRAM macro (Memory(impl=
# Sram(...))). `replica`: one copy per read port, the one writer broadcasting
# -- the LUTRAM form, FPGA-only: on the regfile both it and `registers`
# matched MiniTPU per cycle on Catapult, but the three copies cost a Catapult
# area score of 8,576 against 3,877 (DC then merges the equal registers,
# OPT-1215, so its area alone hides the cost; u2_d12_prototype_2026-10-04.rst),
# and an ASIC has no LUTRAM to make them cheap. `local`: one owner of every
# port, today's kernel-local array. `shared`: kept to record D-11's refusal.
LOWERINGS = ("local", "replica", "registers", "sram", "shared")
LOWERING_ALIASES = {"server": "registers"}  # the D-12 prototype's name
DEFAULT_LOWERING = "registers"
FPGA_ONLY = ("replica",)
SERVER_LOWERINGS = ("registers", "sram")  # one storage in a generated kernel
TECHNOLOGIES = ("asic", "fpga")
# What a target says about its technology when the caller does not:
# Vitis is an FPGA flow; SystemC's target is Catapult on an ASIC library
# (README D-1); the simulator and an unnamed target say nothing.
TARGET_TECHNOLOGY = {"vhls": "fpga", "systemc": "asic"}


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
    # {memory: why no two write-capable ports write one word in one cycle}:
    # the premise a `collision="obligation"` memory needs (README D-12). Like
    # `deadlock_free_because`, nothing checks it; a stress cosim discharges it.
    obligations: dict = field(default_factory=dict)
    _region: object = field(default=None, init=False, repr=False)
    _regions: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._check()

    def _check(self):
        declared = {c.name for c in self.channels}
        shape = {c.name: c.shape for c in self.channels}
        memories = {m.name for m in self.memories if not m.ported}
        ported = {m.name: m for m in self.memories if m.ported}
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

    def plan(self, target=None, lowering=None, technology=None) -> dict:
        """``{memory: lowering}`` for every memory with ports, checked.

        ``local``: one unit owns every port -- today's kernel-local array, the
        degenerate case. ``replica``: every read port's owner holds a copy and
        the one write port's owner sends each write to every copy (the
        write-broadcast replica; ``r`` ports at latency 0 and one ``w`` port
        only); FPGA-only -- refused unless ``technology="fpga"`` or the target
        is one (Vitis). ``registers`` (alias ``server``): one flop array in a
        generated kernel; each port is a bundle of links (address, data,
        enable) to its owner. ``sram``: that server with its storage mapped
        onto the ``Sram`` the memory declares (``impl=``); each declared port
        must be one the macro can honour, else refused naming the port.
        ``shared``: one region-scope ``Stateful`` the owners address directly
        -- refused by D-11 unless premised, kept to record the refusal.

        The choice is ``lowering[m]`` if given, else the memory's ``impl``,
        else ``local`` for one owner and ``registers`` otherwise. Vitis
        refuses a memory with more than one owner (README D-12).
        """
        lowering = dict(lowering or {})
        names = {m.name for m in self._ported()}
        for k in lowering:
            assert k in names, f"{self.name}: lowering names {k!r}, not a memory with ports"
        assert technology is None or technology in TECHNOLOGIES, (
            f"{self.name}: technology={technology!r} is not one of {TECHNOLOGIES}")
        tech = technology or TARGET_TECHNOLOGY.get(target)
        out = {}
        for m in self._ported():
            owners = sorted({u.name for u, _ in self._owners(m).values()})
            if target == "vhls" and len(owners) > 1:
                raise NotImplementedError(
                    f"{self.name}: memory {m.name} has {len(owners)} owners "
                    f"({', '.join(owners)}); Vitis refuses a memory with more "
                    f"than one owner (README D-12). Build it with "
                    f'target="systemc" or "simulator"')
            choice = lowering.get(m.name) or m.lowering or (
                "local" if len(owners) == 1 else DEFAULT_LOWERING)
            choice = LOWERING_ALIASES.get(choice, choice)
            assert choice in LOWERINGS, (
                f"memory {m.name}: lowering {choice!r} is not one of {LOWERINGS}")
            if choice == "local":
                assert len(owners) == 1, (
                    f"memory {m.name}: the local lowering needs one owner of "
                    f"every port; it has {owners}")
            if choice in FPGA_ONLY:
                assert tech == "fpga", (
                    f"memory {m.name}: the {choice} lowering is FPGA-only (LUTRAM-"
                    f"style copies, one per read port; an ASIC flow pays every copy "
                    f"in flops, u2_d12_prototype_2026-10-04.rst), and "
                    + (f"target {target!r} is an ASIC target" if tech == "asic"
                       else f"target {target!r} says nothing about its technology")
                    + ": build with technology='fpga', or choose registers or an "
                    "Sram (asic_memories_2026-10-04.rst)")
            if choice == "sram":
                self._check_sram(m)
            if choice in {"replica", "registers", "sram"}:
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

    def sram_ports(self, m) -> dict:
        """``{declared port: macro port index}`` -- each declared port on one
        physical port of the macro that can honour it (its own kind first,
        then an ``rw`` port), or an AssertionError naming the port."""
        sram = m.sram
        assert sram is not None, (
            f"memory {m.name}: the sram lowering needs the macro: declare "
            f"Memory(impl=Sram(...)) (or Sram.from_openram(<.v>, <.lib>))")
        free = list(range(len(sram.ports)))
        out = {}
        for p in m.ports:
            assert p.count == 1, (
                f"memory {m.name}: port {p.name} declares count={p.count}; a "
                f"macro port serves one access per cycle (declare {p.count} ports)")
            pick = next((i for i in free if sram.ports[i] == p.kind), None)
            if pick is None:
                pick = next((i for i in free if sram.ports[i] == "rw"), None)
            assert pick is not None, (
                f"memory {m.name}: port {p.name} ({p.kind}) has no port of the "
                f"macro {sram.module} left to honour it: it offers "
                f"{list(sram.ports)} for {[q.name for q in m.ports]}")
            free.remove(pick)
            out[p.name] = pick
        return out

    def _check_sram(self, m):
        """The declaration against the macro; refuses naming the port."""
        sram = m.sram
        self.sram_ports(m)
        assert not m.reset, (
            f"memory {m.name}: an SRAM macro ({sram.module}) is not reset; "
            f"declare reset=False (README D-14) or lower it to registers")
        rows = self._rows(m)
        assert rows <= sram.rows, (
            f"memory {m.name}: {rows} rows do not fit the macro {sram.module} "
            f"({sram.rows} rows); banking is not lowered here")
        width = self._width(m)
        assert width is None or width <= sram.width, (
            f"memory {m.name}: {width}-bit words are wider than the macro "
            f"{sram.module}'s {sram.width}")
        for p in m.ports:
            if p.reads:
                assert p.latency > 0, (
                    f"memory {m.name}: port {p.name} declares latency=0 (an "
                    f"asynchronous read); the macro {sram.module}'s read is "
                    f"synchronous (latency {sram.read_latency}): an SRAM cannot "
                    f"honour it. Declare latency >= {sram.read_latency} or lower "
                    f"to registers")
                assert p.latency >= sram.read_latency, (
                    f"memory {m.name}: port {p.name} declares latency="
                    f"{p.latency}, below the macro {sram.module}'s read "
                    f"latency {sram.read_latency}")
            if p.writes:
                assert p.visible == sram.visible, (
                    f"memory {m.name}: port {p.name} declares visible="
                    f"{p.visible}; the macro {sram.module} shows a write after "
                    f"{sram.visible}")

    def _width(self, m):
        """Bits of one word, or None when the type is not one of ours."""
        try:
            ns = dict(FRONTEND_NAMES)
            ns.update(self.parameters)
            t = eval(str(m.dtype), {}, ns)  # pylint: disable=eval-used
            return int(t.bits)
        except Exception:  # pylint: disable=broad-except
            return None

    def port_kernels(self, target=None, lowering=None, technology=None) -> list:
        """The kernels a ported memory's lowering consists of -- the owners of
        its ports and any generated server -- as emitted module names
        (``<kernel>_0``): the synthesis group of the memory."""
        plan = self.plan(target, lowering, technology)
        out = []
        for u in self.units:
            if any("." in mem for mem in u.memories):
                out.append(f"{u.name}_0")
        out += [f"{m}_mem_0" for m, c in plan.items() if c in SERVER_LOWERINGS]
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
            if plan[m.name] in SERVER_LOWERINGS or plan[m.name] == "replica":
                its = iters.get(m.name, {})
                assert len(its) == 1, (
                    f"memory {m.name}: its owners iterate differently "
                    f"({its}); one iteration is one cycle of every port, so "
                    f"the {plan[m.name]} lowering needs one loop range")
            if plan[m.name] in SERVER_LOWERINGS:
                servers.append(self._server(m, next(iter(iters[m.name])),
                                            storage_attr, plan[m.name] == "sram"))
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

    def _server(self, m, loop_iter, storage_attr, sram=False):
        """The generated kernel that holds a `registers`- or `sram`-lowered
        memory: per iteration every port's address, then every read (latency
        0: a combinational put; L >= 1: a pipe of L registers, as data), then
        every write -- so a read sees writes of earlier iterations only
        (visible=1). ``sram``: the storage is a plain array (the backend maps
        it onto the declared macro; an unreset ``sc_signal`` array is not
        RAM-mappable) and every ``rw`` port is written in the one-access form."""
        dt, rows = m.dtype, m.rows
        at = _addr_type(self._rows(m))
        attr = storage_attr if not m.reset and not sram else ""
        head = ["@df.kernel(mapping=[1])", f"def {m.name}_mem():",
                f"    mem: {dt}[{rows}]{attr}"]
        body, writes = [], []
        for p in m.ports:
            for k in range(p.count):
                pk = p.name + (str(k) if p.count > 1 else "")
                ch = f"{m.name}_{pk}"
                body += [f"_{pk}_a: {at} = {ch}_a.get()", f"_{pk}_i: int32 = _{pk}_a"]
                if p.kind == "rw" and p.latency and (m.reset or sram):
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

    def memory_manifest(self, target=None, lowering=None, technology=None) -> dict:
        """What each memory with ports was lowered to: ``memory.json``
        (README D-12), written beside ``latency.json``."""
        plan = self.plan(target, lowering, technology)
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
            if choice == "sram":
                storage = (f"SRAM macro {m.sram.module} ({m.sram.rows} x {m.sram.width}, "
                           f"ports {list(m.sram.ports)}), unreset by nature; the server's "
                           f"array is mapped onto it (Catapult: MAP_TO_MODULE)")
            impl = {
                "local": f"kernel-local array of {sorted({u.name for u, _ in owners.values()})[0]}"
                         " (one owner of every port: today's array)",
                "replica": f"write-broadcast replica x{nread}: one copy per read port, "
                           "inside its owner; every write sent to every copy (FPGA-only)",
                "registers": f"one flop array in generated kernel {m.name}_mem, reads as "
                             "combinational muxes or a pipe as data; each port a bundle "
                             "of links (address, data, enable) to its owner (the ASIC form)",
                "sram": f"one storage in generated kernel {m.name}_mem mapped onto the "
                        f"macro {m.sram.module if m.sram else '?'}; each port a bundle of "
                        "links (address, data, enable) to its owner, on one macro port",
                "shared": "one region-scope Stateful the owners address (D-11 refuses "
                          "it unless premised; no HLS backend builds it)",
            }[choice]
            macro_ports = self.sram_ports(m) if choice == "sram" else {}
            ports = {}
            for p in m.ports:
                d = p.manifest()
                d["owner"] = owners[p.name][0].name
                if choice in {"replica", "registers"} and links != "stream":
                    if p.reads:
                        d["read"] = ("combinational (D-13 comb links)" if p.latency == 0
                                     else f"registered link + {p.latency}-deep pipe as data")
                    if p.writes:
                        d["write"] = "comb pins into a clock-edge write: seen next cycle"
                if choice == "sram":
                    i = macro_ports[p.name]
                    d["macro_port"] = {"index": i, "kind": m.sram.ports[i]}
                    if p.reads:
                        d["read"] = (f"macro read, latency {m.sram.read_latency}, inside a "
                                     f"{p.latency}-deep pipe as data; the achieved latency "
                                     f"is measured on the RTL (D-10), not assumed")
                    if p.writes:
                        d["write"] = f"macro write, visible after {m.sram.visible}"
                ports[p.name] = d
            out[m.name] = {
                "rows": self._rows(m), "dtype": m.dtype, "reset": m.reset,
                "collision": m.collision, "lowering": choice,
                "implementation": impl, "storage": storage,
                "target": target or "declared", "links": links, "ports": ports,
                "technology": technology or TARGET_TECHNOLOGY.get(target),
                "status": "untimed" if target == "simulator" else "lowered",
            }
            if choice == "sram":
                out[m.name]["macro"] = m.sram.manifest()
            if m.name in self.obligations:
                out[m.name]["obligation"] = self.obligations[m.name]
        return out

    def sram_configs(self, target, lowering=None, technology=None, configs=None) -> list:
        """The Catapult side of every ``sram``-lowered memory: one entry per
        memory with the library to add (``solution library add <module> -file
        <catapult_lib>``) and the server's array resource to map onto it
        (``directive set <rsc> -MAP_TO_MODULE <module>.<module>``). The
        resource path follows what is synthesized: the ``synth_group`` top,
        the server itself as ``synth_top``, or the region's top. A macro
        without a compiled Catapult library is refused here, naming the file
        the Memory Generator makes (``allo.backend.catapult.build_memory_library``)."""
        plan = self.plan(target, lowering, technology)
        configs = configs or {}
        group = (configs.get("synth_group") or {}).get("name")
        out = []
        for m in self._ported():
            if plan[m.name] != "sram":
                continue
            kernel = f"{m.name}_mem_0"
            root = group if group else ("" if configs.get("synth_top") == kernel else self.name)
            rsc = f"/{root}/{kernel}/run/mem:rsc" if root else f"/{kernel}/run/mem:rsc"
            out.append({"memory": m.name, "library": m.sram.module,
                        "file": m.sram.catapult_lib, "module": m.sram.module,
                        "rsc": rsc, "sram": m.sram})
        return out

    def build(self, target="simulator", lowering=None, schedule=None, technology=None,
              **kwargs):
        """Customize, schedule and build the region for ``target``; with a
        ``project``, write ``memory.json`` there (README D-12). An ``sram``
        lowering on the SystemC target adds the macro's Catapult library and
        the ``MAP_TO_MODULE`` directive to ``configs`` (``run.tcl``), building
        the library first when ``Sram.catapult_lib`` is unset."""
        region = self.region(target, lowering, technology)
        manifest = self.memory_manifest(target, lowering, technology)
        if target == "simulator":
            for name, d in manifest.items():
                print(f"[memory] {name}: {d['lowering']}, untimed (the simulator "
                      f"keeps the order of accesses, not latency or visibility)")
        srams = self.sram_configs(target, lowering, technology, kwargs.get("configs"))
        if srams and target == "systemc":
            configs = dict(kwargs.get("configs") or {})
            for e in srams:
                if e["file"] is None:
                    from allo.backend.catapult import build_memory_library  # noqa: PLC0415
                    outdir = os.path.join(kwargs.get("project") or ".", "memgen")
                    e["file"] = build_memory_library(
                        e["sram"], outdir, clock=float(configs.get("clock_period", 3.33)))
            configs["memories"] = [{k: v for k, v in e.items() if k != "sram"} for e in srams]
            kwargs["configs"] = configs
            for e in srams:
                manifest[e["memory"]]["catapult"] = {k: v for k, v in e.items() if k != "sram"}
        elif srams and target not in (None, "simulator"):
            raise NotImplementedError(
                f"{self.name}: the sram lowering of {[e['memory'] for e in srams]} is "
                f"built only by target='systemc' (Catapult maps the server's array onto "
                f"the macro); target {target!r} has no macro path yet")
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

    def source(self, target=None, lowering=None, technology=None) -> str:
        """The region's text. ``target="simulator"`` emits every link as a
        Stream (the simulator is untimed and refuses Wire); a memory with
        ports is lowered as ``plan`` says."""
        links = "stream" if target == "simulator" else None
        plan = self.plan(target, lowering, technology)
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

    def region(self, target=None, lowering=None, technology=None):
        """The ``@df.region()``-decorated function, built from `source`.

        Registered in ``linecache`` under a stable pseudo-path because Allo
        reads a region back with ``inspect.getsourcelines``.
        ``ALLO_DUMP_COMPOSED=<dir>`` also writes the text out, to read or diff.
        """
        key = (target, tuple(sorted((lowering or {}).items())) + (
            (("technology", technology),) if technology else ()))
        if key == (None, ()) and self._region is not None:
            return self._region
        if key not in self._regions:
            src = self.source(target, lowering, technology)
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
        """Apply every unit's Vitis directives to a built schedule."""
        ctx = Directives(top=s.top_func_name, parameters=self.parameters)
        for u in self.units:
            if u.directives is not None:
                u.directives(s, ctx)
        return s

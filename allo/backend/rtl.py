# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTL IP integration: lazy Verilator simulation and Vitis black-box packaging."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .ip import IPModule, STREAM, IP_SIM_INCLUDE_DIR, parse_cpp_function


# Tuple entries are (payload width, signedness).
# pylint: disable=consider-using-namedtuple-or-dataclass
_TYPES = {
    "bool": (1, False),
    "int8_t": (8, True),
    "uint8_t": (8, False),
    "int16_t": (16, True),
    "uint16_t": (16, False),
    "int32_t": (32, True),
    "uint32_t": (32, False),
    "int": (32, True),
    "unsigned int": (32, False),
}


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise ValueError(f"Invalid RTL/C identifier: {value!r}")
    return value


def _positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class Port:
    """Stream handshake pins. Direction is relative to the RTL IP.

    For ``ap_fifo`` inputs: data=dout, valid=empty_n, ready=read.
    For outputs: data=din, valid=write, ready=full_n.
    Only first-word-fall-through, active-high FIFO handshakes are supported.
    """

    name: str
    data: str
    valid: str
    ready: str
    dir: str = "in"
    ctype: str = "int32_t"
    bind: str = "stream"
    size: int | None = None
    protocol: str = "ready_valid"
    kind = "stream"


@dataclass
class MemPort:
    """Simulation RAM: word addressed, one-cycle reads, edge-triggered writes."""

    name: str
    size: int
    ctype: str
    addr: str
    ce: str
    q: str | None = None
    we: str | None = None
    d: str | None = None
    kind = "mem"


@dataclass(frozen=True)
class ReadyValidAdapter:
    """Generate a FIFO/chain wrapper around a clock-enable-aware ready/valid IP.

    All ports require per-call sizes. ``start_mode`` is ``pulse`` or ``none``;
    ``completion`` is ``counts`` or ``done`` (counts plus the core's done).
    The core must retain outputs while stalled and honor clock_enable globally.
    """

    clock_enable: str
    start_mode: str = "pulse"
    completion: str = "counts"


@dataclass(frozen=True)
class HLSBlackBox:
    """Contract of a supplied or generated chain wrapper (Vitis 2023.2 schema).

    ``c_model`` supplies the same C signature for HLS C/RTL co-simulation.
    Latency is the unstalled transaction latency; II=0 forbids pipelining.
    The wrapper must honor clock_enable and hold done until continue_.
    Set adapter to ReadyValidAdapter to generate that wrapper from core pins.
    """

    c_model: str
    latency: int
    ii: int = 0
    ready: str = "ap_ready"
    idle: str = "ap_idle"
    continue_: str = "ap_continue"
    clock_enable: str = "ap_ce"
    adapter: ReadyValidAdapter | None = None


def _run(command, env=None):
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {command!r}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


# The transactor intentionally covers all three prototype bindings.
# pylint: disable=too-many-branches
def _emit(
    top,
    ports,
    clk,
    rst,
    rst_hi,
    start,
    done,
    persistent,
    control=None,
    function_name=None,
):
    """One transactor emitter for every port combination.

    Three bindings, one loop:
      Port(bind="stream") -- FIFO on the Allo side,  handshake on the RTL side
      Port(bind="array")  -- array on the Allo side, handshake on the RTL side
      MemPort             -- array on the Allo side, ap_memory on the RTL side

    Termination is `ap_done` when the IP has one, else "every output port has
    delivered its `size` items" -- a controller's output count depends on the
    program it runs, so counting only works for fixed-size dataflow IPs.
    """
    S = []
    A = S.append
    hs = [p for p in ports if p.kind == "stream"]
    mem = [p for p in ports if p.kind == "mem"]
    if any(p.bind == "stream" for p in hs):
        A("#include <hls_stream.h>")
    A("#include <stdint.h>")
    A("#include <cstdio>")
    A("#include <cstdlib>")
    A("#include <memory>")
    A("#include <thread>")
    A(f'#include "V{top}.h"\n')
    A("namespace {")
    A("const long MAX_STALL = 500000;   // cycles with NO progress")
    A(
        f"void tick(V{top} *m) {{ m->eval(); m->{clk} = 1; m->eval(); m->{clk} = 0; m->eval(); }}"
    )
    A("}  // namespace\n")
    A('extern "C" {\n')

    sig = []
    for p in ports:
        if p.kind == "mem" or p.bind == "array":
            sig.append(f"{p.ctype} {p.name}[{p.size}]")
        else:
            sig.append(f"hls::stream<{p.ctype}> &{p.name}")
    A(f"void {function_name or top}({', '.join(sig)}) {{")

    def reset_lines(ind):
        out = [f"{ind}m->{clk} = 0;", f"{ind}m->{rst} = {1 if rst_hi else 0};"]
        if control:
            out += [
                f"{ind}m->{control.clock_enable} = 1;",
                f"{ind}m->{control.continue_} = 0;",
            ]
        if start:
            out.append(f"{ind}m->{start} = 0;")
        for p in hs:
            out.append(f"{ind}m->{p.valid if p.dir=='in' else p.ready} = 0;")
        out.append(f"{ind}for (int i = 0; i < 8; i++) tick(m);")
        out.append(f"{ind}m->{rst} = {0 if rst_hi else 1};")
        return out

    storage = "static thread_local " if persistent else ""
    A(f"    {storage}auto context = std::make_unique<VerilatedContext>();")
    A("    context->threads(1);")
    if persistent:
        A("    // Persistent: the IP keeps state across separate Allo")
        A("    // invocations (e.g. a loaded program), so reset happens ONCE.")
        A(f"    static thread_local std::unique_ptr<V{top}> owner;")
        A("    auto *m = owner.get();")
        A("    if (!m) {")
        A(f"        owner.reset(new V{top}(context.get())); m = owner.get();")
        for l in reset_lines("        "):
            A(l)
        A("        tick(m);")
        A("    }")
    else:
        A(f"    auto owner = std::make_unique<V{top}>(context.get());")
        A("    auto *m = owner.get();")
        for l in reset_lines("    "):
            A(l)

    for p in hs:
        if p.dir == "in" and p.bind == "stream":
            storage = "static thread_local " if persistent else ""
            A(f"    {storage}{p.ctype} h_{p.name} = 0;")
            A(f"    {storage}bool have_{p.name} = false;")
            A(f"    int i_{p.name} = 0;")
        else:
            A(f"    int i_{p.name} = 0;")
    A("    long stall = 0; bool finished = false;")
    if start:
        A(f"    m->{start} = 1;")
    A("    while (!finished && stall < MAX_STALL) {")
    A("        bool progress = false;")

    A("        // ---- drive ----")
    if control:
        A(f"        bool completing = m->{done};")
    for p in hs:
        if p.dir == "in":
            if p.bind == "stream":
                limit = f"i_{p.name} < {p.size} && " if p.size is not None else ""
                active = "!completing && " if control else ""
                A(
                    f"        if ({active}{limit}!have_{p.name}) have_{p.name} = {p.name}.read_nb(h_{p.name});"
                )
                A(f"        bool g_{p.name} = have_{p.name};")
                A(f"        {p.ctype} s_{p.name} = h_{p.name};")
            else:
                A(f"        bool g_{p.name} = (i_{p.name} < {p.size});")
                A(
                    f"        {p.ctype} s_{p.name} = g_{p.name} ? {p.name}[i_{p.name}] : 0;"
                )
            A(f"        m->{p.valid} = g_{p.name};")
            A(f"        m->{p.data}  = g_{p.name} ? (unsigned)s_{p.name} : 0u;")
        else:
            room = (
                f"!{p.name}.full()"
                if p.bind == "stream"
                else f"(i_{p.name} < {p.size})"
            )
            A(f"        bool room_{p.name} = {room};")
            A(f"        m->{p.ready} = room_{p.name};")

    A("        m->eval();   // settle before sampling")
    A("        // ---- sample THIS cycle, before the edge ----")
    for p in hs:
        if p.dir == "in":
            A(f"        bool f_{p.name} = g_{p.name} && m->{p.ready};")
        else:
            A(f"        bool f_{p.name} = m->{p.valid} && room_{p.name};")
            A(f"        {p.ctype} v_{p.name} = ({p.ctype})m->{p.data};")
    for p in mem:
        if p.we:
            A(f"        bool w_{p.name} = m->{p.ce} && m->{p.we};")
            A(f"        unsigned wa_{p.name} = (unsigned)m->{p.addr};")
            A(f"        {p.ctype} wd_{p.name} = ({p.ctype})m->{p.d};")
        if p.q:
            A(
                f"        bool r_{p.name} = m->{p.ce}"
                + (f" && !m->{p.we}" if p.we else "")
                + ";"
            )
            A(f"        unsigned ra_{p.name} = (unsigned)m->{p.addr};")
    for p in mem:
        for access, address in (("w", "wa"), ("r", "ra")):
            if (access == "w" and p.we) or (access == "r" and p.q):
                A(
                    f"        if ({access}_{p.name} && {address}_{p.name} >= {p.size}) std::abort();"
                )
    A(f"        bool d_now = m->{done};" if done else "        bool d_now = false;")
    if control:
        A(f"        bool accepted = m->{control.ready} && m->{start};")
    A(f"        m->{clk} = 1; m->eval(); m->{clk} = 0; m->eval();")

    if control:
        A(f"        if (accepted) m->{start} = 0;")
    A("        // ---- commit only what actually moved ----")
    for p in hs:
        if p.dir == "in":
            clear = (
                f"have_{p.name} = false; i_{p.name}++;"
                if p.bind == "stream"
                else f"i_{p.name}++;"
            )
            A(f"        if (f_{p.name}) {{ {clear} progress = true; }}")
        else:
            # the counter drives the count-based termination test, so it must
            # advance for a FIFO write too -- not only for an array store
            store = (
                f"{p.name}.write(v_{p.name}); i_{p.name}++;"
                if p.bind == "stream"
                else f"{p.name}[i_{p.name}++] = v_{p.name};"
            )
            A(f"        if (f_{p.name}) {{ {store} progress = true; }}")
    for p in mem:
        if p.we:
            A(
                f"        if (w_{p.name}) {{ {p.name}[wa_{p.name}] = wd_{p.name}; progress = true; }}"
            )
        if p.q:
            # ap_memory read latency is 1: ce asserted on cycle k -> q must be
            # valid at k+1, so write q right after k's edge. Staging it another
            # cycle makes every read return stale data.
            A(
                f"        if (r_{p.name}) {{ m->{p.q} = (unsigned){p.name}[ra_{p.name}];"
                f" progress = true; }}"
            )
    if done:
        A("        if (d_now) finished = true;")
    else:
        outs = [p for p in hs if p.dir == "out"]
        cond = " && ".join(f"i_{p.name} >= {p.size}" for p in outs) or "true"
        A(f"        if ({cond}) finished = true;")
    A("        stall = progress ? 0 : stall + 1;")
    A("        if (!progress) std::this_thread::yield();")
    A("    }")
    A(
        f'    if (!finished) {{ fprintf(stderr, "[allo-rtl] {top}: STALLED\\n"); std::abort(); }}'
    )
    if start:
        A(f"    m->{start} = 0;")
    for p in hs:
        if p.dir == "in":
            A(f"    m->{p.valid} = 0;")
    A("    m->eval();")
    if control:
        A(f"    m->{control.continue_} = 1;")
        for p in hs:
            A(f"    m->{p.valid if p.dir == 'in' else p.ready} = 0;")
    if persistent:
        A("    tick(m);            // let the IP settle back to idle")
    if control:
        A(f"    m->{control.continue_} = 0;")
    A("}")
    A('\n}  // extern "C"')
    return "\n".join(S) + "\n"


# Descriptors and build artifacts live as long as the callable IP instance.
# pylint: disable=too-many-instance-attributes,too-many-arguments,consider-using-with
class RTLModule(IPModule):
    """An RTL implementation callable inside Allo, with lazy simulation builds.

    ``rtl`` is one source or a sequence (top wrapper plus dependencies).
    The prototype's ports/convenience forms are preserved. ``name`` gives an
    independent instance a distinct C symbol; ``top`` always names the RTL.
    For synthesis pass HLSBlackBox and describe either a supplied FIFO wrapper
    or ready/valid core pins with an explicit ReadyValidAdapter contract.
    """

    def __init__(
        self,
        top,
        rtl,
        ports=None,
        clock="ap_clk",
        reset="ap_rst",
        reset_active_high=True,
        start="ap_start",
        done=None,
        persistent=False,
        workdir=None,
        mlir_include=None,
        n=None,
        ctype="int32_t",
        mode=None,
        inputs=(),
        outputs=(),
        *,
        name=None,
        hls=None,
        include_paths=(),
        parameters=None,
        defines=None,
        verilator=None,
    ):
        # pylint: disable=super-init-not-called
        # IPModule's C parser is intentionally bypassed: the signature is typed
        # metadata, and no source generation/compiler invocation is needed here.
        self.rtl_top = _identifier(top)
        auto_adapter = isinstance(hls, HLSBlackBox) and hls.adapter is not None
        self.top = _identifier(name or (f"{top}_allo" if auto_adapter else top))
        self.clock, self.reset = _identifier(clock), _identifier(reset)
        self.start = _identifier(start) if start else None
        self.done = _identifier(done) if done else None
        self.reset_active_high = reset_active_high
        self.persistent = persistent
        if mode not in (None, "stream", "buffer"):
            raise ValueError("mode must be 'stream' or 'buffer'")
        if ports is None:
            _positive(n, "n")
            ports = []
            for direction, group in (("in", inputs), ("out", outputs)):
                for original in group:
                    port = copy.copy(original)
                    port.dir, port.size, port.ctype = direction, n, ctype
                    port.bind = "stream" if mode == "stream" else "array"
                    ports.append(port)
        self.ports = copy.deepcopy(list(ports))
        if not self.ports:
            raise ValueError("RTLModule needs at least one port")
        self.hls = hls
        if hls is not None:
            if not isinstance(hls, HLSBlackBox):
                raise TypeError("hls must be an HLSBlackBox")
            if (
                not isinstance(hls.latency, int)
                or isinstance(hls.latency, bool)
                or hls.latency < 0
            ):
                raise ValueError("latency must be a non-negative integer")
            if hls.ii != 0:
                raise NotImplementedError(
                    "Milestone 1 requires ii=0 (no overlapping transactions)"
                )
            for pin in (hls.ready, hls.idle, hls.continue_, hls.clock_enable):
                _identifier(pin)
            self.c_model = Path(hls.c_model).expanduser().resolve(strict=True)
        files = [rtl] if isinstance(rtl, (str, os.PathLike)) else list(rtl)
        if not files:
            raise ValueError("rtl source list is empty")
        self.rtl_files = tuple(Path(f).expanduser().resolve(strict=True) for f in files)
        self.rtl_include_paths = tuple(
            Path(p).expanduser().resolve(strict=True) for p in include_paths
        )
        self.parameters = dict(parameters or {})
        self.defines = dict(defines or {})
        for key in (*self.parameters, *self.defines):
            _identifier(key)
        for value in self.parameters.values():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("RTL parameter overrides must be integers")
        self.verilator = verilator
        self.mlir_include = mlir_include
        self._validate_description()
        self.args = []
        self.input_idx, self.output_idx = [], []
        for i, p in enumerate(self.ports):
            stream = isinstance(p, Port) and p.bind == "stream"
            self.args.append(
                (
                    f"hls::stream<{p.ctype}>" if stream else p.ctype,
                    STREAM if stream else (p.size,),
                )
            )
            if (isinstance(p, Port) and p.dir == "in") or (
                isinstance(p, MemPort) and p.q
            ):
                self.input_idx.append(i)
            if (isinstance(p, Port) and p.dir == "out") or (
                isinstance(p, MemPort) and p.we
            ):
                self.output_idx.append(i)
        self._temp_dir = tempfile.TemporaryDirectory(prefix="allo_rtl_")
        self.temp_path = self._temp_dir.name
        # Even an explicit workdir gets a private child: no stale models or
        # concurrent same-top builds sharing archives.
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        self._build_dir = tempfile.TemporaryDirectory(prefix="rtl_", dir=workdir)
        self.build_path = Path(self._build_dir.name)
        self.impl = str(self.build_path / "transactor.cpp")
        self.generated_source = self.impl
        self.abs_path = str(self.build_path)
        self.include_paths = [self.abs_path]
        self.lib_name = f"{self.top}_{Path(self.temp_path).name}"
        self.c_wrapper_file = str(Path(self.temp_path) / f"{self.lib_name}.cpp")
        self._prepared = False
        self._shared = {}
        self._adapted_module = None
        self._generated_wrapper_path = None

    def validate_arguments(self, arguments):
        """Check logical types before lowering (MLIR integers lose signedness)."""
        from ..ir.types import Stream, Int, UInt

        if len(arguments) != len(self.ports):
            raise ValueError(
                f"RTLModule {self.top} expects {len(self.ports)} arguments"
            )
        for arg, port in zip(arguments, self.ports):
            stream = isinstance(port, Port) and port.bind == "stream"
            dtype = arg.dtype
            if stream != isinstance(dtype, Stream):
                raise TypeError(
                    f"RTL argument {port.name} has incompatible stream/array binding"
                )
            if stream:
                if dtype.shape not in ((), (1,)):
                    raise TypeError("RTL stream payloads must be scalar integers")
                dtype = dtype.dtype
            elif tuple(arg.shape) != (port.size,):
                raise TypeError(f"RTL array {port.name} must have shape ({port.size},)")
            bits, signed = _TYPES[port.ctype]
            if (
                not isinstance(dtype, (Int, UInt))
                or dtype.bits != bits
                or (bits != 1 and isinstance(dtype, Int) != signed)
            ):
                raise TypeError(
                    f"RTL argument {port.name} requires {port.ctype}, got {dtype}"
                )

    def _validate_description(self):
        names = set()
        for p in self.ports:
            if not isinstance(p, (Port, MemPort)):
                raise TypeError("ports must contain Port or MemPort descriptors")
            _identifier(p.name)
            if p.name in names:
                raise ValueError(f"Duplicate argument name: {p.name}")
            names.add(p.name)
            if p.ctype not in _TYPES:
                raise ValueError(
                    f"Unsupported payload type {p.ctype!r}; use 1/8/16/32-bit integers"
                )
            if isinstance(p, Port):
                if p.dir not in ("in", "out") or p.bind not in ("stream", "array"):
                    raise ValueError("Port requires dir=in/out and bind=stream/array")
                if p.protocol not in ("ready_valid", "ap_fifo"):
                    raise ValueError("Unsupported stream protocol")
                if p.size is not None or p.bind == "array":
                    _positive(p.size, f"{p.name}.size")
            else:
                _positive(p.size, f"{p.name}.size")
                if bool(p.we) != bool(p.d) or not (p.q or p.we):
                    raise ValueError("MemPort needs q and/or a complete we,d pair")
        if self.hls and self.hls.adapter is not None:
            self._validate_adapter()
        pins = [item[0] for item in self._pins()]
        for pin in pins:
            _identifier(pin)
        if len(set(pins)) != len(pins):
            raise ValueError("RTL pins cannot be bound more than once")
        if not self.done:
            outputs = [p for p in self.ports if isinstance(p, Port) and p.dir == "out"]
            if not outputs or any(p.size is None for p in outputs):
                raise ValueError(
                    "Supply done or a positive size for every output stream"
                )
        if not self.persistent:
            if any(
                isinstance(p, Port)
                and p.bind == "stream"
                and p.dir == "in"
                and p.size is None
                for p in self.ports
            ):
                raise ValueError(
                    "Non-persistent input streams need size to bound prefetch"
                )

    def _pins(self):
        pins = [(self.clock, "input", 1), (self.reset, "input", 1)]
        if self.start:
            pins.append((self.start, "input", 1))
        if self.done:
            pins.append((self.done, "output", 1))
        if self.hls and self.hls.adapter is not None:
            pins.append((self.hls.adapter.clock_enable, "input", 1))
        elif self.hls:
            pins += [
                (self.hls.ready, "output", 1),
                (self.hls.idle, "output", 1),
                (self.hls.continue_, "input", 1),
                (self.hls.clock_enable, "input", 1),
            ]
        for p in self.ports:
            width = _TYPES[p.ctype][0]
            if isinstance(p, Port):
                direction = "input" if p.dir == "in" else "output"
                reverse = "output" if p.dir == "in" else "input"
                pins += [
                    (p.data, direction, width),
                    (p.valid, direction, 1),
                    (p.ready, reverse, 1),
                ]
            else:
                pins += [
                    (p.addr, "output", max(1, (p.size - 1).bit_length())),
                    (p.ce, "output", 1),
                ]
                if p.q:
                    pins.append((p.q, "input", width))
                if p.we:
                    pins += [(p.we, "output", 1), (p.d, "output", width)]
        return pins

    def _validate_adapter(self):
        adapter = self.hls.adapter
        if not isinstance(adapter, ReadyValidAdapter):
            raise TypeError("adapter must be a ReadyValidAdapter")
        _identifier(adapter.clock_enable)
        if adapter.start_mode not in ("pulse", "none"):
            raise ValueError("Adapter start_mode must be 'pulse' or 'none'")
        if (adapter.start_mode == "pulse") != bool(self.start):
            raise ValueError(
                "Use start=None with start_mode='none', or name a start pin for 'pulse'"
            )
        if adapter.completion not in ("counts", "done"):
            raise ValueError("Adapter completion must be 'counts' or 'done'")
        if adapter.completion == "done" and not self.done:
            raise ValueError("completion='done' requires the core done pin")
        if not self.persistent:
            raise ValueError("Generated hardware adapters require persistent=True")
        if self.top == self.rtl_top:
            raise ValueError(
                "Generated wrapper name must differ from the original RTL top"
            )
        if self.defines or self.rtl_include_paths:
            raise NotImplementedError(
                "Generated adapters require self-contained RTL; defines/include_paths are not supported"
            )
        if (
            self.hls.ready,
            self.hls.idle,
            self.hls.continue_,
            self.hls.clock_enable,
        ) != ("ap_ready", "ap_idle", "ap_continue", "ap_ce"):
            raise ValueError(
                "Generated adapters use the standard ap_* HLS control names"
            )
        for port in self.ports:
            if (
                not isinstance(port, Port)
                or port.bind != "stream"
                or port.protocol != "ready_valid"
            ):
                raise NotImplementedError(
                    "Generated adapters support only ready_valid stream ports"
                )
            _positive(port.size, f"{port.name}.size")

    def generate_wrapper(self):
        """Return deterministic wrapper Verilog without invoking external tools."""
        if self.hls is None or self.hls.adapter is None:
            raise ValueError(
                "Set HLSBlackBox(adapter=ReadyValidAdapter(...)) to generate a wrapper"
            )
        self._validate_adapter()
        from .rtl_adapter import emit_ready_valid_wrapper

        return emit_ready_valid_wrapper(self, self._fifo_ports())

    def _fifo_ports(self):
        return [
            replace(
                p,
                data=f"p{i}_data",
                valid=f"p{i}_empty_n" if p.dir == "in" else f"p{i}_write",
                ready=f"p{i}_read" if p.dir == "in" else f"p{i}_full_n",
                protocol="ap_fifo",
            )
            for i, p in enumerate(self.ports)
        ]

    def _get_adapted_module(self):
        """Use exactly the generated wrapper for both simulation and synthesis."""
        if self._adapted_module is not None:
            return self._adapted_module
        directory = self.build_path / "adapter"
        directory.mkdir(exist_ok=True)
        root = Path(os.path.commonpath([str(p.parent) for p in self.rtl_files]))
        sources = []
        for source in self.rtl_files:
            dest = directory / "src" / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            sources.append(dest)
        wrapper = directory / f"{self.top}.v"
        wrapper.write_text(self.generate_wrapper(), encoding="utf-8")
        sources.append(wrapper)
        # Validate the original model before renaming its entry to the wrapper.
        model = self._read_c_model()
        model_path = directory / "model.cpp"
        model_path.write_text(
            re.sub(r"\b" + re.escape(self.rtl_top) + r"\b", self.top, model),
            encoding="utf-8",
        )
        adapted = RTLModule(
            top=self.top,
            rtl=sources,
            ports=self._fifo_ports(),
            start="ap_start",
            done="ap_done",
            persistent=True,
            hls=replace(self.hls, adapter=None, c_model=str(model_path)),
            verilator=self.verilator,
            mlir_include=self.mlir_include,
        )
        # The compiler has already recorded this object's JIT entry name.
        adapted._generated_wrapper_path = wrapper
        adapted.lib_name = self.lib_name
        adapted.c_wrapper_file = str(Path(adapted.temp_path) / f"{self.lib_name}.cpp")
        self.generated_source = adapted.generated_source
        self._adapted_module = adapted
        return adapted

    def _tool(self):
        binary = (
            self.verilator or os.environ.get("VERILATOR") or shutil.which("verilator")
        )
        if not binary:
            raise RuntimeError(
                "Verilator is required for RTL simulation/validation; set VERILATOR or PATH"
            )
        binary = str(Path(binary).resolve())
        env = os.environ.copy()
        # Conda's Perl launcher needs its own bundled modules even when another
        # environment supplies python. No machine-specific paths are used.
        perl = Path(binary).parent.parent / "lib/perl5/core_perl"
        if (perl / "FindBin.pm").exists():
            env["PERL5LIB"] = str(perl) + os.pathsep + env.get("PERL5LIB", "")
        return binary, env

    def _rtl_command(self):
        binary, _ = self._tool()
        return (
            [binary, "--top-module", self.rtl_top]
            + [str(p) for p in self.rtl_files]
            + [f"-I{p}" for p in self.rtl_include_paths]
            + [f"-G{k}={v}" for k, v in self.parameters.items()]
            + [
                f"-D{k}" + (f"={v}" if v is not None else "")
                for k, v in self.defines.items()
            ]
        )

    def validate_rtl(self):
        """Elaborate with Verilator and validate top-level names/directions/widths."""
        _, env = self._tool()
        xml = self.build_path / "ports.xml"
        _run(
            self._rtl_command()
            + [
                "--xml-only",
                "-Wno-fatal",
                "--xml-output",
                str(xml),
                "--Mdir",
                str(self.build_path / "xml"),
            ],
            env,
        )
        tree = ET.parse(xml)
        module = next(
            (m for m in tree.findall(".//module") if m.get("topModule") == "1"), None
        )
        if module is None:
            raise ValueError("Cannot locate elaborated RTL top module")
        dtypes = {d.get("id"): d for d in tree.findall(".//typetable/*")}
        actual = {}
        for var in module.findall("var"):
            if var.get("dir") not in ("input", "output", "inout"):
                continue
            dtype = dtypes[var.get("dtype_id")]
            if dtype.tag != "basicdtype":
                raise ValueError(
                    f"Packed/unpacked aggregate pin unsupported: {var.get('name')}"
                )
            width = abs(int(dtype.get("left", "0")) - int(dtype.get("right", "0"))) + 1
            actual[var.get("name")] = (var.get("dir"), width)
        expected = {name: (direction, width) for name, direction, width in self._pins()}
        for name, spec in expected.items():
            if actual.get(name) != spec:
                raise ValueError(
                    f"RTL pin {name}: expected {spec}, found {actual.get(name)}"
                )
        missing_inputs = [
            n
            for n, (d, _) in actual.items()
            if d in {"input", "inout"} and n not in expected
        ]
        if missing_inputs:
            raise ValueError(f"Unbound RTL input pins: {missing_inputs}")
        if self.hls and self.hls.adapter is not None:
            self._get_adapted_module().validate_rtl()
        return actual

    def _prepare_simulation(self):
        if self._prepared:
            return
        self.validate_rtl()
        binary, env = self._tool()
        vgen = self.build_path / "vgen"
        _run(
            self._rtl_command()
            + [
                "--cc",
                "--build",
                "--Mdir",
                str(vgen),
                "-CFLAGS",
                "-fPIC -fvisibility=hidden",
            ],
            env,
        )
        root = Path(_run([binary, "--getenv", "VERILATOR_ROOT"], env).strip())
        self.include_paths += [
            str(vgen),
            str(root / "include"),
            str(root / "include/vltstd"),
        ]
        Path(self.impl).write_text(
            _emit(
                self.rtl_top,
                self.ports,
                self.clock,
                self.reset,
                self.reset_active_high,
                self.start,
                self.done,
                self.persistent,
                self.hls,
                self.top,
            ),
            encoding="utf-8",
        )
        # Compile runtime sources explicitly; Verilator versions differ in
        # whether --build creates a separate libverilated archive.
        self._link_inputs = [
            str(vgen / f"V{self.rtl_top}__ALL.a"),
            str(root / "include/verilated.cpp"),
            str(root / "include/verilated_threads.cpp"),
        ]
        self._prepared = True

    def compile_shared_lib(self, stream_sim=False):
        if self.has_stream_args and not stream_sim:
            self._reject_stream_on_cpu()
        if self.hls and self.hls.adapter is not None:
            self.validate_rtl()
            return self._get_adapted_module().compile_shared_lib(stream_sim=stream_sim)
        self._prepare_simulation()
        if stream_sim in self._shared:
            return self._shared[stream_sim]
        if stream_sim:
            self.generate_stream_sim_wrapper()
        else:
            self.generate_mlir_c_wrapper()
        mlir_include = self.mlir_include or os.environ.get("MLIR_INCLUDE_DIR")
        if not mlir_include:
            llvm = shutil.which("llvm-config")
            if llvm:
                candidates = [
                    Path(_run([llvm, "--includedir"]).strip()),
                    Path(llvm).resolve().parents[2] / "mlir/include",
                ]
                mlir_include = next(
                    (
                        str(p)
                        for p in candidates
                        if (p / "mlir/ExecutionEngine/CRunnerUtils.h").exists()
                    ),
                    None,
                )
        if not mlir_include:
            raise RuntimeError(
                "Set MLIR_INCLUDE_DIR to the directory containing mlir/ExecutionEngine/CRunnerUtils.h"
            )
        includes = (
            ([IP_SIM_INCLUDE_DIR] if stream_sim else [])
            + self.include_paths
            + [mlir_include]
        )
        so = str(Path(self.temp_path) / f"lib{self.lib_name}_{int(stream_sim)}.so")
        _run(
            [
                os.environ.get("CXX", "g++"),
                "-shared",
                "-fPIC",
                "-fvisibility=hidden",
                "-std=c++17",
                "-pthread",
                *[f"-I{p}" for p in includes],
                self.c_wrapper_file,
                *self._link_inputs,
                "-o",
                so,
            ]
        )
        self._shared[stream_sim] = so
        return so

    def compile_nanobind(self):
        raise NotImplementedError(
            "Build an Allo kernel with target='llvm' for RTL arrays, or target='simulator' for streams"
        )

    def validate_target(self, target):
        """Reject backends that cannot preserve RTL external-module semantics."""
        if target not in (None, "llvm", "simulator", "vitis_hls"):
            raise NotImplementedError(
                f"RTLModule does not support target={target!r}; use llvm for arrays, "
                "simulator for dataflow, or vitis_hls for black-box synthesis"
            )

    def validate_hls(self, platform, mode):
        """Check synthesis support before project generation mutates any files."""
        if platform != "vitis_hls" or mode not in (None, "csyn"):
            raise NotImplementedError(
                "RTLModule supports vitis_hls mode='csyn'; use the generated Tcl project for cosim/export"
            )
        if self.hls is None:
            raise ValueError(
                "RTL synthesis requires an HLSBlackBox describing a supplied wrapper"
            )
        if self.hls.adapter is not None:
            self._validate_adapter()
            return
        if (
            not self.start
            or not self.done
            or not self.reset_active_high
            or not self.persistent
        ):
            raise ValueError(
                "HLS wrapper requires start/done, active-high reset, and persistent=True"
            )
        if any(
            not isinstance(p, Port) or p.bind != "stream" or p.protocol != "ap_fifo"
            for p in self.ports
        ):
            raise NotImplementedError(
                "RTL synthesis currently supports only Port(bind='stream', protocol='ap_fifo')"
            )
        if self.parameters or self.defines or self.rtl_include_paths:
            raise NotImplementedError(
                "For synthesis supply self-contained RTL with parameters/defines resolved in the wrapper"
            )

    def _read_c_model(self):
        """Read and check the supplied model before generating any package."""
        model = self.c_model.read_text(encoding="utf-8")
        parsed = parse_cpp_function(model, self.rtl_top)

        def normalize(args):
            return [(re.sub(r"\s+", "", dtype), shape) for dtype, shape in args]

        if parsed is None or normalize(parsed) != normalize(self.args):
            raise ValueError(
                "C model signature must match RTLModule ports in order and type"
            )
        if re.search(r'^\s*#\s*include\s*"', model, re.MULTILINE):
            raise ValueError(
                "C model must be self-contained; local includes are not packaged"
            )
        if not re.search(r"\bvoid\s+" + re.escape(self.rtl_top) + r"\s*\(", model):
            raise ValueError(f"C model must define void {self.rtl_top}(...)")
        return model

    def export_hls(self, project):
        """Stage supplied RTL and C model, emit a typed header and black-box JSON."""
        self.validate_hls("vitis_hls", "csyn")
        if self.hls.adapter is not None:
            return self._get_adapted_module().export_hls(project)
        model = self._read_c_model()
        project = Path(project)
        package = project / f"rtl_{self.top}"
        package.mkdir(parents=True, exist_ok=True)
        # Preserve relative RTL source paths, including explicit .vh files.
        original_sources = [
            p for p in self.rtl_files if p != self._generated_wrapper_path
        ]
        root = Path(os.path.commonpath([str(p.parent) for p in original_sources]))
        sources = []
        digest = hashlib.sha256()
        for source in original_sources:
            digest.update(source.relative_to(root).as_posix().encode())
            digest.update(source.read_bytes())
        source_dir = project / "rtl_sources" / digest.hexdigest()[:16]
        for source in original_sources:
            dest = source_dir / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            sources.append(dest.relative_to(project).as_posix())
        if self._generated_wrapper_path is not None:
            dest = package / f"{self.top}.v"
            shutil.copyfile(self._generated_wrapper_path, dest)
            sources.append(dest.relative_to(project).as_posix())
        # C and RTL names must match in Vitis. An explicit instance name gets
        # a deterministic, purely structural alias around the supplied wrapper.
        if self.top != self.rtl_top:
            declarations = [
                f"    {direction} wire [{width-1}:0] {pin}"
                for pin, direction, width in self._pins()
            ]
            connections = [f".{pin}({pin})" for pin, _, _ in self._pins()]
            alias = package / f"{self.top}.v"
            alias.write_text(
                f"`timescale 1ns / 1ps\nmodule {self.top}(\n"
                + ",\n".join(declarations)
                + f"\n);\n{self.rtl_top} ip(\n"
                + ",\n".join(connections)
                + "\n);\nendmodule\n"
            )
            sources.append(alias.relative_to(project).as_posix())
        prefix = package.relative_to(project).as_posix()
        sig = ", ".join(f"hls::stream<{p.ctype}> &{p.name}" for p in self.ports)
        header = f"{prefix}/{self.top}.h"
        # Preserve C++ stream type information at the vendor black-box boundary,
        # even if a caller includes this header inside an extern "C" block.
        (project / header).write_text(
            f'#pragma once\n#include <stdint.h>\n#include <hls_stream.h>\nextern "C++" void {self.top}({sig});\n'
        )
        # Models use the RTL top name; rename only that identifier for aliases.
        model = re.sub(r"\b" + re.escape(self.rtl_top) + r"\b", self.top, model)
        model_path = f"{prefix}/model.cpp"
        # Vitis requires a function body when extracting black-box information
        # during synthesis. The black-box registration selects the supplied RTL.
        (project / model_path).write_text(f'#include "{self.top}.h"\n{model}\n')
        params = []
        for p in self.ports:
            mapping = (
                {
                    "FIFO_empty_flag": p.valid,
                    "FIFO_read_enable": p.ready,
                    "FIFO_data_read_in": p.data,
                }
                if p.dir == "in"
                else {
                    "FIFO_full_flag": p.ready,
                    "FIFO_write_enable": p.valid,
                    "FIFO_data_write_out": p.data,
                }
            )
            params.append(
                {"c_name": p.name, "c_port_direction": p.dir, "rtl_ports": mapping}
            )
        manifest = {
            "c_function_name": self.top,
            "rtl_top_module_name": self.top,
            "c_files": [{"c_file": model_path, "cflag": "-std=c++14"}],
            "rtl_files": sources,
            "c_parameters": params,
            "rtl_common_signal": {
                "module_clock": self.clock,
                "module_reset": self.reset,
                "module_clock_enable": self.hls.clock_enable,
                "ap_ctrl_chain_protocol_start": self.start,
                "ap_ctrl_chain_protocol_done": self.done,
                "ap_ctrl_chain_protocol_ready": self.hls.ready,
                "ap_ctrl_chain_protocol_idle": self.hls.idle,
                "ap_ctrl_chain_protocol_continue": self.hls.continue_,
            },
            "rtl_performance": {
                "latency": str(self.hls.latency),
                "II": str(self.hls.ii),
            },
        }
        manifest_path = f"{prefix}/blackbox.json"
        (project / manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
        return header, manifest_path

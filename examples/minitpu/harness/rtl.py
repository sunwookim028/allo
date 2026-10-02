# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive one MiniTPU ``.sv`` unit in Verilator with a stimulus array.

A unit is driven through a generated C++ wrapper, built once per unit and
cached by the hash of its sources and its port spec. Stimulus goes in, and
outputs come back, as raw little-endian ``uint64`` files, so a million vectors
cost one process launch.

Three port shapes cover U1:

``comb``
    No clock. Set the inputs, ``eval()``, read the outputs.
``valid``
    ``clk``, active-low reset, ``valid_i`` in and ``valid_o`` out. One vector
    is offered per cycle (II=1). For each output the wrapper records the cycle
    on which it appeared, so the harness measures the latency rather than
    assuming it.
``bare``
    ``clk`` and no valid. The wrapper records the output after every edge
    and ``run()`` reads each vector's result ``latency`` edges after it went
    in, so there the declared latency is an input. ``probe_latency()``
    measures it independently of any reference: a step from one input to
    another, and the edge on which the output first moves.
    With ``reset=True`` the active-low reset is pulsed first and ``warmup``
    idle cycles follow it (a Catapult thread's reset action).
``stream``
    ``clk``, active-low reset, and a latency-insensitive valid/ready
    handshake on every data port (Catapult/Connections: ``<p>_vld``,
    ``<p>_rdy``, ``<p>_dat``). Each input port is an independent stream that
    offers vector k until it is accepted; the output's ready is held high, or
    dropped one cycle in every ``out_ready_period`` to exercise backpressure.
    A transfer happens on a rising edge with valid and ready both high. The
    wrapper stamps each input's accept cycle (the last port's) and each
    output's cycle, so latency and throughput are both measured;
    ``last_stats`` holds the totals of the latest run.
``trace``
    ``clk`` and every other input, reset included, a *command trace*: one
    row of every input port per cycle, which is how a stateful unit (a
    register file, a RAM, a FIFO) is a function -- from a command trace to a
    *response trace*, one row of every output port per cycle (U2).
    ``run_trace()`` drives it. Cycle ``t`` is: drive row ``t`` with the clock
    low, ``eval()``, sample the ``"pre"`` outputs, rising edge ``t + 1``,
    sample the ``"post"`` outputs. An output's sampling is the optional third
    element of its spec, ``(port, width, "pre" | "post")``, default ``"pre"``:
    ``"pre"`` for an asynchronous read and for flags (the value the cycle's
    consumer sees, so a read and a write of one address in one cycle return
    the old value), ``"post"`` for a registered read. So a port of latency
    ``L`` edges shows a cycle-``t`` command in row ``t + L`` if ``"pre"`` and
    in row ``t + L - 1`` if ``"post"``; ``probe_trace()`` measures it.
    Ports may be any width: a port is ``ceil(width / 64)`` little-endian
    ``uint64`` words in the raw files, unpacked from and into Verilator's
    ``VlWide`` (32-bit words) for ports wider than 64 bits.
    The model is built with ``--x-initial unique`` and every run takes a
    seed (``+verilator+seed``, random reset), so state the RTL never
    initialises -- a RAM entry not yet written, a FIFO before reset --
    differs between seeds: the harness masks those slots rather than trust
    zeros, and ``characterize`` checks that every slot it calls defined is
    seed-invariant. With ``assertions=True`` the model is built with
    ``--assert`` and an unlimited error count; ``last_asserts`` gets
    ``(cycle, message)`` for every assertion that fired.

Latency counts rising edges: an input driven before edge 1 is captured by it,
and a unit of latency ``L`` shows the result after edge ``L`` -- the number of
registers on the path, which is how MiniTPU's RTL comments and ``localparam``
s state it. A combinational unit has latency 0.

MiniTPU's own ``tb/`` stays the second check (D-7); this is the first.
"""

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np


def minitpu_home():
    """The pinned MiniTPU clone: ``$MINITPU_HOME``, else the known places."""
    for p in (
        os.environ.get("MINITPU_HOME"),
        "/work/shared/users/phd/sk3463/minitpu",  # zhang-21 (README D-8)
        os.path.expanduser("~/core/minitpu"),  # ace-01
    ):
        if p and os.path.isdir(p):
            return p
    raise RuntimeError("Set MINITPU_HOME to a MiniTPU clone at b3ba0a4d.")


def _cxx():
    """A g++ new enough for Verilator 5's runtime headers.

    RHEL 8's g++ 8.5 is too old; gcc-toolset-13 is used when present.
    """
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    gts = "/opt/rh/gcc-toolset-13/root/usr/bin/g++"
    return gts if os.path.exists(gts) else "g++"


@dataclass
class RtlUnit:
    top: str
    sources: list  # paths relative to the MiniTPU clone
    inputs: list  # [(port, width)], data ports only
    outputs: list  # [(port, width)]
    shape: str = "comb"  # comb | valid | bare | stream
    latency: int = 0  # declared; sampled at this depth for "bare"
    clk: str = "clk_i"
    rst_n: str = "rst_ni"
    valid_in: str = "valid_i"
    valid_out: str = "valid_o"
    defines: list = field(default_factory=list)
    reset: bool = None  # pulse rst_n first; default: valid/stream yes, bare no
    warmup: int = 0  # idle cycles after reset before the first input
    vld: str = "_vld"  # stream shape: per-port handshake suffixes
    rdy: str = "_rdy"
    dat: str = "_dat"
    out_ready_period: int = 0  # stream: 0 = output always ready
    params: dict = field(default_factory=dict)  # top-level parameters (-G)
    assertions: bool = False  # trace: build with --assert, record firings

    def key(self, home):
        h = hashlib.sha256(repr(self).encode())
        for s in self.sources:
            with open(os.path.join(home, s), "rb") as f:
                h.update(f.read())
        h.update(WRAPPER_VERSION.encode())
        return h.hexdigest()[:16]


WRAPPER_VERSION = "2"


def _wrapper(u):
    """The C++ driver for one unit."""
    if u.shape == "trace":
        return _trace_driver(u)
    ni, no = len(u.inputs), len(u.outputs)
    set_in = "\n".join(
        f"    dut.{p} = (decltype(dut.{p}))in[k * {ni} + {i}];"
        for i, (p, _) in enumerate(u.inputs)
    )
    get_out = "\n".join(
        f"      out[j * {no + 1} + {i}] = (uint64_t)dut.{p};"
        for i, (p, _) in enumerate(u.outputs)
    )
    zero_in = "\n".join(f"    dut.{p} = 0;" for p, _ in u.inputs)
    if u.shape == "stream":
        body = _stream_body(u)
    elif u.shape == "comb":
        body = f"""
  for (uint64_t k = 0; k < n; ++k) {{
{set_in}
    dut.eval();
    {{ uint64_t j = k;
{get_out}
      out[j * {no + 1} + {no}] = 0; }}
  }}"""
    else:
        do_rst = u.reset if u.reset is not None else u.shape == "valid"
        valid_set = (
            f"dut.{u.valid_in} = k < n;" if u.shape == "valid" else ""
        )
        if u.shape == "valid":
            capture = f"""
    if (dut.{u.valid_out}) {{
      if (j >= n) {{ std::fprintf(stderr, "more outputs than inputs\\n"); return 3; }}
{get_out}
      out[j * {no + 1} + {no}] = cyc + 1;
      ++j;
    }}"""
        else:
            # every edge's output: run() aligns it, probe_latency() measures
            capture = f"""
    {{ j = cyc;
{get_out}
      out[j * {no + 1} + {no}] = cyc + 1; }}"""
        body = f"""
  // reset: four cycles low, inputs idle
  dut.{u.clk} = 0;
  {"dut." + u.rst_n + " = 0;" if do_rst else ""}
{zero_in}
  {"dut." + u.valid_in + " = 0;" if u.shape == "valid" else ""}
  for (int r = 0; r < 4; ++r) {{ dut.{u.clk} = 0; dut.eval(); dut.{u.clk} = 1; dut.eval(); }}
  {"dut." + u.rst_n + " = 1;" if do_rst else ""}
  for (int w = 0; w < {u.warmup}; ++w) {{ dut.{u.clk} = 0; dut.eval(); dut.{u.clk} = 1; dut.eval(); }}
  uint64_t j = 0;
  const uint64_t limit = n + {u.latency} + 64;
  for (uint64_t cyc = 0; cyc < limit; ++cyc) {{
    // drive input k = cyc on the falling half, sample after the rising edge
    dut.{u.clk} = 0;
    uint64_t k = cyc;
    if (k < n) {{
{set_in.replace("in[k", "in[k")}
    }} else {{
{zero_in}
    }}
    {valid_set}
    dut.eval();
    dut.{u.clk} = 1;
    dut.eval();
{capture}
  }}
  {"if (j != n) { std::fprintf(stderr, \"%llu of %llu outputs\\n\", (unsigned long long)j, (unsigned long long)n); return 4; }" if u.shape == "valid" else ""}"""
    return f"""// generated by examples/minitpu/harness/rtl.py
#include "V{u.top}.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {{
  Verilated::commandArgs(argc, argv);
  FILE* fi = std::fopen(argv[1], "rb");
  std::fseek(fi, 0, SEEK_END);
  const uint64_t n = std::ftell(fi) / 8 / {ni};
  std::fseek(fi, 0, SEEK_SET);
  const uint64_t rows = {"n + " + str(u.latency) + " + 64" if u.shape == "bare" else "n"};
  std::vector<uint64_t> in(n * {ni}), out(rows * {no + 1});
  if (std::fread(in.data(), 8, n * {ni}, fi) != n * {ni}) return 2;
  std::fclose(fi);
  V{u.top} dut;
{body}
  FILE* fo = std::fopen(argv[2], "wb");
  std::fwrite(out.data(), 8, out.size(), fo);
  std::fclose(fo);
  return 0;
}}
"""


def _stream_body(u):
    """Valid/ready driver: independent input streams, one output stream."""
    ni = len(u.inputs)
    (op, _), = u.outputs  # one output stream
    lines = []
    for i, (p, _) in enumerate(u.inputs):
        lines.append(f"    dut.{p}{u.vld} = k[{i}] < n;")
        lines.append(f"    dut.{p}{u.dat} = k[{i}] < n ? in[k[{i}] * {ni} + {i}] : 0;")
    drive = "\n".join(lines)
    fire = "\n".join(
        f"    fire[{i}] = dut.{p}{u.vld} && dut.{p}{u.rdy};" for i, (p, _) in enumerate(u.inputs)
    )
    per = u.out_ready_period
    ordy = f"(cyc % {per}) != {per} - 1" if per else "1"
    return f"""
  // reset: four cycles low, every valid low, output not ready
  uint64_t k[{ni}] = {{0}};
  bool fire[{ni}];
  std::vector<uint64_t> acc(n, 0);  // cycle the vector's last input was accepted
  dut.{u.rst_n} = 0;
  for (int i = 0; i < {ni}; ++i) k[i] = n;  // idle during reset
{drive}
  dut.{op}{u.rdy} = 0;
  for (int r = 0; r < 4; ++r) {{ dut.{u.clk} = 0; dut.eval(); dut.{u.clk} = 1; dut.eval(); }}
  dut.{u.rst_n} = 1;
  for (int w = 0; w < {u.warmup}; ++w) {{ dut.{u.clk} = 0; dut.eval(); dut.{u.clk} = 1; dut.eval(); }}
  for (int i = 0; i < {ni}; ++i) k[i] = 0;
  uint64_t j = 0, cyc = 0, last = 0, first_out = 0;
  while (j < n) {{
    dut.{u.clk} = 0;
{drive}
    dut.{op}{u.rdy} = {ordy};
    dut.eval();  // ready/valid settle; the transfer happens on the next rising edge
{fire}
    const bool ofire = dut.{op}{u.vld} && dut.{op}{u.rdy};
    const uint64_t odat = (uint64_t)dut.{op}{u.dat};
    dut.{u.clk} = 1;
    dut.eval();
    for (int i = 0; i < {ni}; ++i)
      if (fire[i]) {{ if (acc[k[i]] < cyc) acc[k[i]] = cyc; ++k[i]; last = cyc; }}
    if (ofire) {{
      out[j * 2] = odat;
      out[j * 2 + 1] = cyc - acc[j];
      if (j == 0) first_out = cyc;
      ++j; last = cyc;
    }}
    ++cyc;
    if (cyc - last > 10000) {{
      std::fprintf(stderr, "stalled at cycle %llu: %llu of %llu outputs\\n",
                   (unsigned long long)cyc, (unsigned long long)j, (unsigned long long)n);
      return 4;
    }}
  }}
  std::printf("STREAM cycles=%llu first_out=%llu n=%llu\\n", (unsigned long long)cyc,
              (unsigned long long)first_out, (unsigned long long)n);"""


def nwords(width):
    """``uint64`` words that hold a ``width``-bit port."""
    return (width + 63) // 64


def out_spec(port):
    """``(name, width, sample)`` of an output spec; ``sample`` defaults to pre."""
    return (port[0], port[1], port[2] if len(port) > 2 else "pre")


def pack(values, width):
    """Integers (a list of Python ints, or an integer array) ->
    ``uint64[n, nwords(width)]``, masked to ``width`` bits."""
    nw = nwords(width)
    # a list of Python ints stays exact (numpy would make a mix of big and
    # small ints float64); an integer ndarray takes the fast path
    arr = values if isinstance(values, np.ndarray) else np.array(values, dtype=object)
    assert arr.dtype == object or np.issubdtype(arr.dtype, np.integer), arr.dtype
    if arr.dtype != object and nw == 1:
        m = np.uint64((1 << width) - 1) if width < 64 else np.uint64(2**64 - 1)
        return (arr.astype(np.uint64) & m).reshape(-1, 1)
    out = np.zeros((len(values), nw), dtype=np.uint64)
    m = (1 << width) - 1
    for i, v in enumerate(values):
        v = int(v) & m
        for j in range(nw):
            out[i, j] = (v >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
    return out


def unpack(words):
    """``uint64[n, nw]`` -> list of Python ints (any width)."""
    words = np.asarray(words, dtype=np.uint64).reshape(len(words), -1)
    return [sum(int(w) << (64 * j) for j, w in enumerate(row)) for row in words]


_TRACE_HELPERS = """
template <typename T> static inline void setp(T& p, const uint64_t* w) { p = (T)w[0]; }
template <std::size_t N> static inline void setp(VlWide<N>& p, const uint64_t* w) {
  for (std::size_t i = 0; i < N; ++i) p.m_storage[i] = (EData)(w[i / 2] >> (32 * (i % 2)));
}
template <typename T> static inline void getp(const T& p, uint64_t* w) { w[0] = (uint64_t)p; }
template <std::size_t N> static inline void getp(const VlWide<N>& p, uint64_t* w) {
  for (std::size_t i = 0; i < N; i += 2)
    w[i / 2] = (uint64_t)p.m_storage[i] | (i + 1 < N ? (uint64_t)p.m_storage[i + 1] << 32 : 0);
}
"""


def _trace_driver(u):
    """C++ driver for the ``trace`` shape (module docstring)."""
    win = sum(nwords(w) for _, w in u.inputs)
    wout = sum(nwords(out_spec(p)[1]) for p in u.outputs)
    set_in, zero_in, off = [], [], 0
    for p, w in u.inputs:
        set_in.append(f"    setp(dut.{p}, &in[t * {win} + {off}]);")
        zero_in.append(f"  setp(dut.{p}, zeros);")
        off += nwords(w)
    pre, post, off = [], [], 0
    for spec in u.outputs:
        p, w, smp = out_spec(spec)
        assert smp in ("pre", "post"), spec
        (pre if smp == "pre" else post).append(f"    getp(dut.{p}, &out[t * {wout} + {off}]);")
        off += nwords(w)
    ports = {p for p, _ in u.inputs}
    own_rst = u.rst_n in ports
    do_rst = bool(u.reset) and not own_rst
    idle = "{ ctx->time(0); dut.%s = 0; dut.eval(); dut.%s = 1; dut.eval(); }" % (u.clk, u.clk)
    nl = "\n"
    return f"""// generated by examples/minitpu/harness/rtl.py (trace shape)
#include "V{u.top}.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
{_TRACE_HELPERS}
int main(int argc, char** argv) {{
  VerilatedContext* ctx = Verilated::threadContextp();
  ctx->commandArgs(argc, argv);
  ctx->randReset(2);  // with --x-initial unique: uninitialised state is random
  ctx->randSeed(std::atoi(argv[3]));
  ctx->errorLimit(1 << 30);  // an assertion is recorded, not fatal
  FILE* fi = std::fopen(argv[1], "rb");
  std::fseek(fi, 0, SEEK_END);
  const uint64_t n = std::ftell(fi) / 8 / {win};
  std::fseek(fi, 0, SEEK_SET);
  std::vector<uint64_t> in(n * {win}), out(n * {wout}, 0);
  static const uint64_t zeros[64] = {{0}};
  if (std::fread(in.data(), 8, n * {win}, fi) != n * {win}) return 2;
  std::fclose(fi);
  V{u.top} dut;
  dut.{u.clk} = 0;
{nl.join(zero_in)}
  {f"dut.{u.rst_n} = 0;" if do_rst else ""}
  for (int r = 0; r < {4 if do_rst else 0}; ++r) {idle}
  {f"dut.{u.rst_n} = 1;" if do_rst else ""}
  for (int w = 0; w < {u.warmup}; ++w) {idle}
  for (uint64_t t = 0; t < n; ++t) {{
    ctx->time(t + 1);  // assertion messages carry cycle t + 1
    dut.{u.clk} = 0;
{nl.join(set_in)}
    dut.eval();
{nl.join(pre)}
    dut.{u.clk} = 1;
    dut.eval();
{nl.join(post)}
  }}
  FILE* fo = std::fopen(argv[2], "wb");
  std::fwrite(out.data(), 8, out.size(), fo);
  std::fclose(fo);
  return 0;
}}
"""


def run_trace(u, cmd, seed=1):
    """Run command trace ``cmd`` through a ``trace``-shape unit.

    ``cmd`` maps every input port to its per-cycle values: a 1-D array or list
    of ints, or ``uint64[n, nwords]`` already packed. Returns
    ``{output: uint64[n, nwords]}`` in the row convention of the module
    docstring (``unpack`` turns a column into Python ints). ``seed`` seeds the
    random initial state. ``last_asserts`` gets the assertions that fired.
    """
    assert u.shape == "trace"
    cols = []
    for p, w in u.inputs:
        v = cmd[p]
        if isinstance(v, np.ndarray) and v.ndim == 2:
            assert v.dtype == np.uint64 and v.shape[1] == nwords(w), (p, v.shape)
            cols.append(v)
        else:
            cols.append(pack(v if isinstance(v, np.ndarray) else list(v), w))
    n = len(cols[0])
    assert all(len(c) == n for c in cols), "command columns differ in length"
    stim = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.uint64)
    exe = build(u)
    d = os.path.dirname(exe)
    tag = f"{os.getpid()}.{seed}"
    fi, fo = os.path.join(d, f"in.{tag}"), os.path.join(d, f"out.{tag}")
    stim.tofile(fi)
    try:
        r = subprocess.run([exe, fi, fo, str(seed)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{u.top} trace driver exited {r.returncode}: {r.stderr[-2000:]}")
        wout = sum(nwords(out_spec(p)[1]) for p in u.outputs)
        raw = np.fromfile(fo, dtype=np.uint64).reshape(n, wout)
    finally:
        for p in (fi, fo):
            if os.path.exists(p):
                os.remove(p)
    res, off = {}, 0
    for spec in u.outputs:
        p, w, _ = out_spec(spec)
        k = nwords(w)
        col = raw[:, off : off + k].copy()
        if w % 64:
            col[:, -1] &= np.uint64((1 << (w % 64)) - 1)
        res[p] = col
        off += k
    last_asserts.clear()
    for line in (r.stdout + r.stderr).splitlines():
        if "Assertion failed" in line or "%Error" in line:
            cyc = None
            if line.startswith("["):
                try:
                    cyc = int(line[1 : line.index("]")].split()[0]) - 1
                except ValueError:
                    pass
            last_asserts.append((cyc, line.strip()))
    return res


def probe_trace(u, cmd, port, event, seed=1):
    """Measure a ``trace`` port's latency, in edges, without a reference.

    ``cmd`` is a trace whose ``port`` holds one value up to cycle ``event``
    and moves only because of the command issued in cycle ``event`` (a read
    of a newly different word, a write to the word being read, a push). The
    latency is the number of edges from that command's capture to the first
    row where ``port`` leaves its value of row ``event - 1``: rows count from
    the edge for ``"pre"`` ports and after it for ``"post"`` ports.
    """
    res = run_trace(u, cmd, seed)[port]
    _, _, smp = out_spec(next(o for o in u.outputs if o[0] == port))
    moved = np.flatnonzero((res[event:] != res[event - 1]).any(axis=1))
    assert len(moved), f"{port} never moved after cycle {event}"
    return int(moved[0]) + (1 if smp == "post" else 0)


last_asserts = []  # trace shape: [(cycle, message)] of the latest run_trace()


def build(u, cache=None):
    """Build the unit's driver once; return the binary's path."""
    home = minitpu_home()
    cache = cache or os.environ.get(
        "MINITPU_HARNESS_CACHE",
        os.path.join(home, ".build", "allo_harness"),
    )
    d = os.path.join(cache, f"{u.top}-{u.key(home)}")
    exe = os.path.join(d, f"V{u.top}")
    if os.path.exists(exe):
        return exe
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "driver.cpp"), "w", encoding="utf-8") as f:
        f.write(_wrapper(u))
    verilator = shutil.which("verilator") or "verilator"
    cmd = (
        [verilator, "--cc", "--exe", "--build", "-Wno-fatal", "-O3"]
        + [f"+define+{d_}" for d_ in u.defines]
        + [f"-G{k}={v}" for k, v in u.params.items()]
        + (["--x-initial", "unique"] if u.shape == "trace" else [])
        + (["--assert"] if u.assertions else [])
        + ["--top-module", u.top, "--Mdir", d, "-j", "8"]
        + [os.path.join(home, s) for s in u.sources]
        + [os.path.join(d, "driver.cpp"), "-CFLAGS", "-std=c++17 -O2"]
    )
    env = dict(os.environ, CXX=_cxx())
    r = subprocess.run(cmd, cwd=home, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(
            f"verilator build of {u.top} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )
    return exe


def run(u, stim):
    """Run stimulus ``stim`` (``uint64[n, n_inputs]``) through the RTL unit.

    Returns ``(outputs uint64[n, n_outputs], cycles int64[n])``. ``cycles`` is
    each output's latency in edges (module docstring): measured for
    ``valid``, the declared ``latency`` for ``bare``, zero for ``comb``. For
    ``stream`` it is the output's cycle minus its inputs' accept cycle, and
    ``last_stats`` gets the total cycle count.
    """
    outs, cycles = _execute(u, stim)
    if u.shape == "bare":
        assert u.latency >= 1, "a bare unit is clocked: declare its latency"
        n = len(stim)
        outs = outs[u.latency - 1 : u.latency - 1 + n]
        cycles = np.full(n, u.latency, dtype=np.int64)
    return outs, cycles


def probe_latency(u, x0, x1, hold=32):
    """Measure a clocked unit's latency without a reference.

    Holds input vector ``x0`` for ``hold`` cycles, then ``x1``; the latency is
    the number of edges until the output first leaves its ``x0`` value. The
    two vectors must give different outputs. Works for ``bare`` and
    ``valid`` alike.
    """
    stim = np.array([x0] * hold + [x1] * hold, dtype=np.uint64)
    if u.shape == "valid":
        _, cycles = _execute(u, stim)
        return int(cycles[hold])
    outs, _ = _execute(u, stim)  # outs[c]: after edge c + 1
    settled = outs[hold - 1]
    assert (outs[hold // 2 : hold] == settled).all(), "output not settled on x0"
    moved = np.flatnonzero((outs[hold:] != settled).any(axis=1))
    assert len(moved), "x0 and x1 give the same output: pick another pair"
    # input `hold` is captured by edge hold + 1, so latency = edge - hold
    return int(moved[0]) + 1


def _execute(u, stim):
    """Run the driver; returns its raw outputs and per-row cycle column."""
    exe = build(u)
    stim = np.ascontiguousarray(stim, dtype=np.uint64)
    assert stim.ndim == 2 and stim.shape[1] == len(u.inputs)
    d = os.path.dirname(exe)
    fi, fo = os.path.join(d, f"in.{os.getpid()}"), os.path.join(d, f"out.{os.getpid()}")
    stim.tofile(fi)
    try:
        r = subprocess.run([exe, fi, fo], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{u.top} driver exited {r.returncode}: {r.stderr}")
        raw = np.fromfile(fo, dtype=np.uint64).reshape(-1, len(u.outputs) + 1)
    finally:
        for p in (fi, fo):
            if os.path.exists(p):
                os.remove(p)
    outs = raw[:, :-1]
    for i, (_, w) in enumerate(u.outputs):
        if w < 64:
            outs[:, i] &= np.uint64((1 << w) - 1)
    cycles = raw[:, -1].astype(np.int64)
    if u.shape in ("valid", "bare"):
        cycles = cycles - np.arange(len(cycles), dtype=np.int64)
    last_stats.clear()
    for line in r.stdout.splitlines():
        if line.startswith("STREAM "):
            last_stats.update((k, int(v)) for k, v in (f.split("=") for f in line.split()[1:]))
    return outs, cycles


last_stats = {}  # stream shape: {"cycles", "first_out", "n"} of the latest run()

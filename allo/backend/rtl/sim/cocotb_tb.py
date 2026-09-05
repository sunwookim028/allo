# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic, config-driven cocotb testbench."""

from __future__ import annotations

import json
import os
import random

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge


def _i(sig) -> int:
    """Integer value of a signal, or 0 when unresolvable (X during reset)."""
    v = sig.value
    return int(v) if v.is_resolvable else 0


async def _serve_mem(hdl, clk, mem, readers, writers, size):
    """Service one backing array as a synchronous RAM at each port's declared
    access latency, matching an ``always_ff @(posedge clk)`` model: the address
    and the write enable/addr/data are sampled before the edge, then the read
    data is presented and the write committed at the edge.

    The manifest's ``latency`` is the number the scheduler solved against, and
    honoring it is the driver's half of a contract the RTL does not enforce: the
    module binds its read-data input with no delay elements, so it expects the
    datum that many cycles after the address. A latency of L presents and
    commits L-1 edges later than the 1-cycle base, through a per-port pipeline
    of in-flight values.
    """
    rd = [
        (getattr(hdl, r["addr"]), getattr(hdl, r["data"]), int(r["latency"]))
        for r in readers
    ]
    wr = [
        (
            getattr(hdl, w["we"]),
            getattr(hdl, w["addr"]),
            getattr(hdl, w["data"]),
            int(w["latency"]),
        )
        for w in writers
    ]
    assert all(lat >= 1 for *_, lat in rd) and all(
        lat >= 1 for *_, lat in wr
    ), "a boundary port needs a >= 1 cycle access latency to be edge-triggered"
    # The L-1 results and commits not yet due, per port.
    rd_pipe = [[0] * (lat - 1) for *_, lat in rd]
    wr_pipe = [[None] * (lat - 1) for *_, lat in wr]

    def clamp(addr):
        return addr if 0 <= addr < size else 0

    while True:
        await ReadOnly()  # end of cycle: settled (pre-edge) values
        r_addr = [clamp(_i(addr)) for addr, _, _ in rd]
        w = [(_i(we), clamp(_i(addr)), _i(dat), lat) for we, addr, dat, lat in wr]
        await RisingEdge(clk)  # commit at the edge (NBA-like)
        # A read resolves against pre-write memory, so it is presented before
        # the writes commit below.
        for k, (_, data, lat) in enumerate(rd):
            v = int(mem[r_addr[k]])
            if lat == 1:
                data.value = v
            else:  # due now: the value fetched lat-1 edges ago
                data.value = rd_pipe[k].pop(0)
                rd_pipe[k].append(v)
        for k, (we, addr, dat, lat) in enumerate(w):
            due = (we, addr, dat)
            if lat > 1:  # defer the commit by lat-1 edges
                due = wr_pipe[k].pop(0)
                wr_pipe[k].append((we, addr, dat))
            if due and due[0]:  # a pipe slot is None until the first commit
                mem[due[1]] = due[2]


async def _serve_regfile(hdl, clk, arr, elements):
    """Hold a completely-partitioned argument's registers, which live on this
    side of the boundary: present each element on its ``in`` port and capture the
    module's ``out`` at every edge its ``we`` is high.

    Simpler than ``_serve_mem`` rather than a variation on it: no address to
    decode, no read latency to model, and a write latency of exactly 1, so no
    in-flight pipeline either. An element the module never enables keeps the
    value it was preloaded with.
    """
    ins = [(getattr(hdl, e["in"]), k) for k, e in enumerate(elements) if "in" in e]
    outs = [
        (getattr(hdl, e["we"]), getattr(hdl, e["out"]), k)
        for k, e in enumerate(elements)
        if "out" in e
    ]
    for sig, k in ins:
        sig.value = int(arr[k])
    if not outs:
        return  # read-only: nothing to capture, and the values above are held
    while True:
        await ReadOnly()  # end of cycle: settled (pre-edge) values
        due = [(k, _i(data)) for we, data, k in outs if _i(we)]
        await RisingEdge(clk)  # commit at the edge (NBA-like)
        for k, v in due:
            arr[k] = v
        # A read-write element feeds back through this side, so its input
        # follows the register just captured.
        for sig, k in ins:
            sig.value = int(arr[k])


async def _feed_stream(hdl, clk, s, tokens, gap=0.0):
    """Source a FIFO stream: drive data and valid, advancing to the next token
    only on a cycle the DUT's ready is high at the edge. With ``gap`` > 0,
    randomly withholds valid to starve the DUT, which must stall rather than
    lose or duplicate a token. Holds valid low once the sequence runs out."""
    data = getattr(hdl, s["data"])
    valid = getattr(hdl, s["valid"])
    ready = getattr(hdl, s["ready"])
    i = 0
    while i < len(tokens):
        if gap and random.random() < gap:  # starve: offer nothing this cycle
            valid.value = 0
            await RisingEdge(clk)
            continue
        data.value = int(tokens[i])
        valid.value = 1
        await ReadOnly()  # settled: is the DUT ready this cycle?
        fired = _i(ready) == 1
        await RisingEdge(clk)
        if fired:
            i += 1
    valid.value = 0
    data.value = 0


async def _drain_stream(hdl, clk, s, out, count, gap=0.0):
    """Sink a FIFO stream: capture data on every cycle the DUT drives valid while
    ready is held, until ``count`` tokens are collected. With ``gap`` > 0,
    randomly deasserts ready to back-pressure the DUT, which must freeze rather
    than drop a token."""
    data = getattr(hdl, s["data"])
    valid = getattr(hdl, s["valid"])
    ready = getattr(hdl, s["ready"])
    while len(out) < count:
        stall = bool(gap) and random.random() < gap
        ready.value = 0 if stall else 1
        await ReadOnly()
        if not stall and _i(valid) == 1:
            out.append(_i(data))
        await RisingEdge(clk)
    ready.value = 0


@cocotb.test()
async def cosim(hdl):
    with open(os.environ["ALLO_COSIM_CFG"], encoding="utf-8") as f:
        cfg = json.load(f)

    # Every hardware name comes from the config, none from this harness.
    ctl = cfg["control"]
    clk = getattr(hdl, ctl["clk"])
    rst = getattr(hdl, ctl["rst"])
    start = getattr(hdl, ctl["start"])
    done = getattr(hdl, ctl["done"])

    cocotb.start_soon(Clock(clk, cfg["clock_ps"], unit="ps").start())
    for s in cfg["scalars"]:
        getattr(hdl, s["name"]).value = s["value"]

    # Quiesce the handshake lines during reset, and prepare one capture list
    # per drained stream.
    stream_out: dict[str, list] = {}
    for s in cfg["streams"]:
        if s["input"]:
            getattr(hdl, s["valid"]).value = 0
        else:
            getattr(hdl, s["ready"]).value = 0
            stream_out[s["base"]] = []

    # Preloaded input or RMW seed, else zeros for a pure output.
    arrays = []
    for m in cfg["mems"]:
        if m["file_in"]:
            arr = np.load(m["file_in"]).reshape(-1).astype(np.uint64)
        else:
            arr = np.zeros(m["size"], dtype=np.uint64)
        arrays.append(arr)
    # Always preloaded, so an element the kernel never writes passes through.
    regs = [
        np.load(rf["file_in"]).reshape(-1).astype(np.uint64) for rf in cfg["regfiles"]
    ]

    rst.value = 1
    start.value = 0
    for _ in range(cfg["reset_cycles"]):
        await RisingEdge(clk)
    rst.value = 0
    await RisingEdge(clk)

    for m, arr in zip(cfg["mems"], arrays):
        cocotb.start_soon(
            _serve_mem(hdl, clk, arr, m["readers"], m["writers"], m["size"])
        )
    for rf, arr in zip(cfg["regfiles"], regs):
        cocotb.start_soon(_serve_regfile(hdl, clk, arr, rf["elements"]))
    # A non-zero `stream_gap` starves inputs and back-pressures outputs to
    # exercise the stall shell; KPN determinism makes the result identical.
    gap = cfg.get("stream_gap", 0.0)
    for s in cfg["streams"]:
        if s["input"]:
            toks = np.load(s["file_in"]).reshape(-1).astype(np.uint64)
            cocotb.start_soon(_feed_stream(hdl, clk, s, toks, gap))
        else:
            cocotb.start_soon(
                _drain_stream(hdl, clk, s, stream_out[s["base"]], s["count"], gap)
            )

    start.value = 1
    await RisingEdge(clk)
    start.value = 0

    cycles = 0
    timeout = cfg["timeout"]
    while _i(done) != 1 and cycles < timeout:
        await RisingEdge(clk)
        cycles += 1
    for _ in range(cfg["settle_cycles"]):
        await RisingEdge(clk)

    for m, arr in zip(cfg["mems"], arrays):
        if m["file_out"]:
            np.save(m["file_out"], arr[: m["size"]].astype(np.uint64))
    for rf, arr in zip(cfg["regfiles"], regs):
        if rf["file_out"]:
            np.save(rf["file_out"], arr.astype(np.uint64))
    for s in cfg["streams"]:
        if not s["input"]:
            np.save(s["file_out"], np.array(stream_out[s["base"]], dtype=np.uint64))
    # `done` has settled, so each result port holds its final value.
    results = [_i(getattr(hdl, n)) for n in cfg["result_ports"]]
    with open(cfg["results_out"], "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(cfg["cycles_out"], "w", encoding="utf-8") as f:
        f.write(str(cycles))
    assert cycles < timeout, f"cosim timed out after {timeout} cycles"

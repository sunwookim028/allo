# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12, amended (2026-10-08): a read port's latency ``L`` is counted
in the owner's own iterations on every link kind.

One memory ``m`` (8 words, never reset) with a write port ``w`` and a read
port ``r`` of latency ``L``, both owned by one unit lowered as ``registers``
(the generated server). The owner keeps its own iteration counter ``c``: it
reads word ``(5c) & 7`` and writes word ``c & 7`` every iteration, and puts
the pair (``c + 1``, the data its read port returned this iteration) to a
sink on ONE link kind, so the pair travels together and the sink's own skew
does not matter. The rule: the data the owner sees in iteration ``c`` is the
word its read of iteration ``c - L`` addressed, as of BEFORE that iteration's
write (visible=1). Before the fix a Stream link delivered at ``c - L + 1``
(U4 track A, T-2) -- the post-edge value, which a pre-sampling owner had to
hold one iteration (U4 track C, C8).

Checked at L = 1 and 2 on the simulator (Stream links), in SystemC csim on
Stream links (the simulator's region) and in csim on the Wire links the
SystemC target declares (comb address pins, a registered data pin): values
equal and arriving in the same owner iteration on all three.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

ROWS = 8


@unit(memories=("m.w", "m.r"), writes=("o_i", "o_q"), parameters=("N",))
def owner(mw, mr):
    c: int32 = 0
    for _ in range(N):
        ra: int32 = (c * 5) & 7
        q: UInt(16) = mr[ra]
        wa: int32 = c & 7
        v: UInt(16) = c * 3 + 1
        mw[wa] = v
        tag: int32 = c + 1
        o_i.put(tag)
        o_q.put(q)
        c = c + 1


@unit(memories=("OI", "OQ"), reads=("o_i", "o_q"), parameters=("N",))
def sink(oi: int32[N], oq: UInt(16)[N]):
    for t in range(N):
        i: int32 = o_i.get()  # both links read in the same iteration, before
        q: UInt(16) = o_q.get()  # the (blocking, in csim) output pushes
        oi[t] = i
        oq[t] = q


def architecture(n, latency):
    mem = Memory("m", "UInt(16)", rows=str(ROWS),
                 ports=(Port("w", "w", visible=1), Port("r", "r", latency=latency)),
                 collision="undefined", reset=False)
    return Architecture(
        name=f"port_lat{latency}",
        parameters={"N": n},
        memories=(Memory("OI", "int32[N]"), Memory("OQ", "UInt(16)[N]"), mem),
        channels=(Channel("o_i", "int32", "2", kind="wire"),
                  Channel("o_q", "UInt(16)", "2", kind="wire")),
        units=(owner, sink),
    )


def model(n, latency):
    """``{tag: data}`` the owner must see in iteration ``tag - 1``; a slot
    whose read landed on a never-written word, or that precedes the first
    delivery, is absent (undefined)."""
    mem = [None] * ROWS
    reads = []
    for c in range(n):
        reads.append(mem[(c * 5) & 7])  # before this iteration's write
        mem[c & 7] = (c * 3 + 1) & 0xFFFF
    return {c + 1: reads[c - latency] for c in range(latency, n)
            if reads[c - latency] is not None}


def _pairs(oi, oq):
    return {int(i): int(q) for i, q in zip(oi, oq) if int(i) != 0}


def _check(pairs, want, n, latency, what, edge=0):
    """Every defined slot the sink saw agrees; at least all but ``edge``
    trailing ones were seen (a Wire sink may miss the last few)."""
    seen = [t for t in want if t in pairs]
    assert len(seen) >= len(want) - edge, (what, len(seen), len(want))
    bad = [(t, pairs[t], want[t]) for t in seen if pairs[t] != want[t]]
    if bad:
        # name the offset at which the data does agree, if any (T-2 was -1)
        offs = [k for k in range(-3, 4)
                if all(pairs.get(t + k) == want[t] for t in seen if t + k in pairs)]
        raise AssertionError(f"{what} L={latency}: {len(bad)}/{len(seen)} differ, "
                             f"e.g. {bad[:3]}; agrees at tag offset {offs}")


def _run(mod, n):
    oi = np.zeros(n, np.int32)
    oq = np.zeros(n, np.uint16)
    mod(oi, oq)
    return _pairs(oi, oq)


@pytest.mark.parametrize("latency", [1, 2])
def test_port_latency_simulator(latency):
    n = 48
    a = architecture(n, latency)
    pairs = _run(a.build("simulator", {"m": "registers"}), n)
    want = model(n, latency)
    _check(pairs, want, n, latency, "simulator (Stream links)")
    man = a.memory_manifest("simulator", {"m": "registers"})
    assert f"delivered {latency} owner iterations" in man["m"]["ports"]["r"]["read"]


@pytest.mark.parametrize("latency", [1, 2])
def test_port_latency_manifest_wire(latency):
    man = architecture(8, latency).memory_manifest("systemc", {"m": "registers"})
    assert f"delivered {latency} owner iterations" in man["m"]["ports"]["r"]["read"]


@needs_csim
@pytest.mark.parametrize("latency", [1, 2])
def test_port_latency_csim_stream(latency):
    n = 48
    a = architecture(n, latency)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(a.region("simulator", {"m": "registers"}), target="systemc",
                       mode="csim", project=tmp)
        pairs = _run(mod, n)
    _check(pairs, model(n, latency), n, latency, "csim, Stream links")


@needs_csim
@pytest.mark.parametrize("latency", [1, 2])
def test_port_latency_csim_wire(latency):
    n = 48
    a = architecture(n, latency)
    with tempfile.TemporaryDirectory() as tmp:
        mod = a.build("systemc", {"m": "registers"}, mode="csim", project=tmp)
        pairs = _run(mod, n)
    # a Wire sink is not cycle-locked to the owner (limitation 22): it may
    # miss the last few pairs, never mispair one
    _check(pairs, model(n, latency), n, latency, "csim, Wire links", edge=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

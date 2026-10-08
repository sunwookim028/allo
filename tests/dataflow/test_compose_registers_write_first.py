# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track D finding D-6: a same-cycle write and read of one row of a
``registers``-lowered memory must return the OLD word (``visible=1``).

Track D measured MiniTPU's IRAM form on Catapult RTL returning the word being
written in the same cycle on 31 slots (write-first). The server was not the
cause: on the RTL it reads its flops in the address's own cycle and writes at
the same edge (read-first, 3/3 collisions probed). The cause was the
composition's stimulus kernel. ``src`` pops its array ports and puts each
value to a ``Wire``, interleaved, and Catapult split that loop into two
c-steps (pops 1-3 in c-step 0, the rest in c-step 1). So the writer owner saw
row ``t`` one edge before the reader owner did, and every same-row collision
read the new word. The fix (``EmitSystemC.cpp``, ``planWireDefer``): a kernel
that is not Wire-only publishes all of an iteration's Wire puts together, at
the end of the iteration.

The composition here is that shape, small: ``src`` -> (``writer`` owns
``m.w``, ``reader`` owns ``m.r``, latency 1) -> ``sink``, ``Wire`` channels,
one memory of 8 words, unreset (64-bit words on the simulator and in csim,
whose array ports stop at 64 bits; 128-bit, track D's width, on the RTL). Each row writes ``wa`` and reads ``ra`` with
``ra == wa`` on most rows, so most reads collide with a write. The reader puts
``(tag, q)`` together, so pairing does not depend on the sink's skew. The
rule: the ``q`` paired with tag ``t + 1`` is the word at ``ra[t - 1]`` as of
before row ``t - 1``'s write.

Checked on the simulator (Stream links), in SystemC csim (the declared Wire
links), and -- when ``ALLO_TEST_CATAPULT=1`` with Catapult and Verilator on
PATH -- on the Catapult RTL in Verilator, through track D's ``CatapultRtl``
driver (about 2 minutes).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
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
needs_catapult = pytest.mark.skipif(
    not (os.environ.get("ALLO_TEST_CATAPULT") == "1" and os.environ.get("MGC_HOME")
         and shutil.which("catapult") and shutil.which("verilator")),
    reason="Catapult RTL: set ALLO_TEST_CATAPULT=1 with catapult and verilator on PATH",
)

ROWS = 8


# The stimulus adapter, written as track D's `src`: one 128-bit pop first,
# then each pop followed by its Wire put. Before the fix Catapult scheduled
# these pops in two c-steps, and the Wire puts followed them.
@unit(memories=("WD", "WE", "WA", "RA", "TG", "X1", "X2", "X3"),
      writes=("h_we", "h_wa", "h_wd", "c_ra", "c_tg", "c_x1", "c_x2", "c_x3"),
      parameters=("N", "W"))
def src(xwd: UInt(W)[N], xwe: uint1[N], xwa: UInt(8)[N], xra: UInt(8)[N],
        xtg: int32[N], xx1: uint1[N], xx2: uint1[N], xx3: UInt(8)[N]):
    for t in range(N):
        d: UInt(W) = xwd[t]
        h_we.put(xwe[t])
        h_wa.put(xwa[t])
        h_wd.put(d)
        c_ra.put(xra[t])
        c_tg.put(xtg[t])
        c_x1.put(xx1[t])
        c_x2.put(xx2[t])
        c_x3.put(xx3[t])


@unit(memories=("m.w",), reads=("h_we", "h_wa", "h_wd"), parameters=("N", "W"))
def writer(mw):
    for _ in range(N):
        e: uint1 = h_we.get()
        a_: UInt(8) = h_wa.get()
        a: int32 = a_ & 7
        d: UInt(W) = h_wd.get()
        if e:
            mw[a] = d


@unit(memories=("m.r",), reads=("c_ra", "c_tg", "c_x1", "c_x2", "c_x3"),
      writes=("o_tg", "o_q"), parameters=("N", "W"))
def reader(mr):
    for _ in range(N):
        ra_: UInt(8) = c_ra.get()
        tg: int32 = c_tg.get()
        x1: uint1 = c_x1.get()
        x2: uint1 = c_x2.get()
        x3: UInt(8) = c_x3.get()
        ra: int32 = ra_ & 7
        q: UInt(W) = mr[ra]  # latency 1: the read of the previous iteration
        o_tg.put(tg + ((x1 & x2 & 0) + (x3 & 0)))  # x1..x3 only keep the pins alive
        o_q.put(q)


@unit(memories=("YT", "YQ"), reads=("o_tg", "o_q"), parameters=("N", "W"))
def sink(yt: int32[N], yq: UInt(W)[N]):
    for t in range(N):
        a: int32 = o_tg.get()  # both links before the (blocking, in csim) pushes
        b: UInt(W) = o_q.get()
        yt[t] = a
        yq[t] = b


def architecture(n, w=64):
    mem = Memory("m", "UInt(W)", rows=str(ROWS),
                 ports=(Port("w", "w", visible=1), Port("r", "r", latency=1)),
                 collision="undefined", reset=False)
    chans = [Channel(c, d, "2", kind="wire") for c, d in (
        ("h_we", "uint1"), ("h_wa", "UInt(8)"), ("h_wd", "UInt(W)"),
        ("c_ra", "UInt(8)"), ("c_tg", "int32"), ("c_x1", "uint1"), ("c_x2", "uint1"),
        ("c_x3", "UInt(8)"), ("o_tg", "int32"), ("o_q", "UInt(W)"))]
    arrays = [Memory(nm, dt) for nm, dt in (
        ("WD", "UInt(W)[N]"), ("WE", "uint1[N]"), ("WA", "UInt(8)[N]"),
        ("RA", "UInt(8)[N]"), ("TG", "int32[N]"), ("X1", "uint1[N]"),
        ("X2", "uint1[N]"), ("X3", "UInt(8)[N]"), ("YT", "int32[N]"),
        ("YQ", "UInt(W)[N]"))]
    return Architecture(name=f"reg_wf{w}", parameters={"N": n, "W": w},
                        memories=tuple(arrays) + (mem,), channels=tuple(chans),
                        units=(src, writer, reader, sink))


def stimulus(n, w, seed=5):
    """Rows: every row writes; ``ra == wa`` (a same-cycle collision) on 3 rows
    in 4. Data is random ``w``-bit, so old and new words always differ."""
    rng = np.random.default_rng(seed)
    wd = [int.from_bytes(rng.bytes(w // 8), "little") | 1 for _ in range(n)]
    we = np.ones(n, np.uint8)
    wa = rng.integers(0, ROWS, n).astype(np.uint8)
    ra = np.where(rng.random(n) < 0.75, wa, rng.integers(0, ROWS, n)).astype(np.uint8)
    tg = np.arange(1, n + 1, dtype=np.int32)
    return wd, we, wa, ra, tg


def model(wd, we, wa, ra):
    """``{tag: q}``: tag ``t + 1`` carries the read issued in row ``t - 1``,
    the word as of BEFORE row ``t - 1``'s write (read-first, visible=1).
    Reads of never-written words are absent; so is the number of collisions
    whose old word differs from the new one."""
    mem = [None] * ROWS
    reads, coll = [], 0
    for t in range(len(wd)):
        reads.append(mem[int(ra[t])])
        if we[t] and int(ra[t]) == int(wa[t]) and mem[int(wa[t])] is not None:
            coll += 1
        if we[t]:
            mem[int(wa[t])] = wd[t]
    want = {t + 1: reads[t - 1] for t in range(1, len(wd)) if reads[t - 1] is not None}
    return want, coll


def _lanes(values):
    """128-bit ints -> ``uint32[n, 4]`` (lane 0 = bits 31:0), as the MiniTPU
    harness passes a ``UInt(128)[N]`` column to the RTL driver
    (``ctrl_lanes.split``)."""
    return np.array([[(int(v) >> (32 * k)) & 0xFFFFFFFF for k in range(4)] for v in values],
                    dtype=np.uint32).reshape(len(values), 4)


def _ints(col):
    a = np.asarray(col, dtype=np.uint64)
    if a.ndim == 1:
        return [int(x) for x in a]
    return [sum(int(a[t, k]) << (32 * k) for k in range(a.shape[1])) for t in range(a.shape[0])]


def _arrays(n, w=64):
    wd, we, wa, ra, tg = stimulus(n, w)
    zero8 = np.zeros(n, np.uint8)
    col = _lanes(wd) if w == 128 else np.array(wd, dtype=np.uint64)
    ins = [col, we, wa, ra, tg, zero8.copy(), zero8.copy(), zero8.copy()]
    yq = np.zeros((n, 4), np.uint32) if w == 128 else np.zeros(n, np.uint64)
    return (wd, we, wa, ra), ins, np.zeros(n, np.int32), yq


def _verdict(yt, yq, want, what, edge=0):
    got = {int(t): q for t, q in zip(yt, _ints(yq)) if int(t) != 0}
    seen = [t for t in want if t in got]
    assert len(seen) >= len(want) - edge, (what, len(seen), len(want))
    bad = [t for t in seen if got[t] != want[t]]
    if bad:
        raise AssertionError(f"{what}: {len(bad)}/{len(seen)} reads differ from read-first "
                             f"(the old word), e.g. tags {bad[:5]}")
    return len(seen)


def test_registers_read_first_simulator():
    n = 64
    a = architecture(n)
    (wd, we, wa, ra), ins, yt, yq = _arrays(n)
    want, coll = model(wd, we, wa, ra)
    assert coll >= n // 2, coll  # the trace exercises the collision
    a.build("simulator", {"m": "registers"})(*ins, yt, yq)
    _verdict(yt, yq, want, "simulator (Stream links)")


@needs_csim
def test_registers_read_first_csim():
    n = 64
    a = architecture(n)
    (wd, we, wa, ra), ins, yt, yq = _arrays(n)
    want, _ = model(wd, we, wa, ra)
    with tempfile.TemporaryDirectory() as tmp:
        a.build("systemc", {"m": "registers"}, mode="csim", project=tmp)(*ins, yt, yq)
    # a Wire sink is not cycle-locked to the reader (limitation 22): it may
    # miss the last few pairs, never mispair one
    _verdict(yt, yq, want, "csim (Wire links)", edge=4)


def test_wire_puts_published_at_iteration_end():
    """The emitted SystemC: `src` (not Wire-only) writes every Wire once, at
    the end of its iteration, from a shadow; the Wire-only owners do not."""
    a = architecture(16)
    with tempfile.TemporaryDirectory() as tmp:
        s = df.customize(a.region("systemc", {"m": "registers"}))
        s.build(target="systemc", mode="csyn", project=tmp)
        code = open(os.path.join(tmp, "kernel.cpp"), encoding="utf-8").read()
    blk = code[code.index("SC_MODULE(src_0)"):]
    blk = blk[: blk.index("\n};")]
    body = blk[blk.index("for ("):]
    assert body.count("published together (D-6)") == 8, body
    last_pop = body.rindex(".Pop()")
    first_write = body.index(".write(__wput_")
    assert last_pop < first_write, "a Wire is written before the iteration's last pop"
    for k in ("writer_0", "reader_0", "m_mem_0"):
        kb = code[code.index(f"SC_MODULE({k})"):]
        kb = kb[: kb.index("\n};")]
        assert "__wput_" not in kb, k


def _catapult_rtl_class():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "dev", "records", "minitpu",
                        "u4_track_d_2026-10-08", "scripts", "u4d_check.py")
    spec = importlib.util.spec_from_file_location("u4d_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CatapultRtl


@needs_catapult
def test_registers_read_first_catapult_rtl():
    n = 64
    a = architecture(n, 128)  # track D's word width: a 128-bit pop first in `src`
    (wd, we, wa, ra), ins, yt, yq = _arrays(n, 128)
    want, _ = model(wd, we, wa, ra)
    prj = os.environ.get("ALLO_TEST_CATAPULT_PRJ") or tempfile.mkdtemp(prefix="reg_wf_")
    if os.path.isdir(prj):
        shutil.rmtree(prj)
    s = df.customize(a.region("systemc", {"m": "registers"}))
    for fn in ("src_0", "writer_0", "reader_0", "sink_0", "m_mem_0"):
        loops = s.get_loops(fn)
        band = max(loops.loops, key=lambda b: len(loops[b].loops))
        s.pipeline(f"{fn}:{next(iter(loops[band].loops))}")
    mod = s.build(target="systemc", mode="csyn", project=prj, configs={"clock_period": 3.33})
    mod()
    _catapult_rtl_class()(prj)(*ins, yt, yq)
    _verdict(yt, yq, want, f"Catapult RTL ({prj})", edge=4)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPN-style dataflow container composition: start policies, topologies, and a container's own datapath in the network."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import f32, i32, Stream

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, Mod, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

N = 8
A = np.arange(1, N + 1, dtype=np.int32)


def _inst_start(m, child):
    """The SSA value driving child instance ``child``'s ``start`` port."""
    hits = [l for l in m.text.splitlines() if f"_{child}_i" in l and "hw.instance" in l]
    assert len(hits) == 1, f"expected one instance of {child}, got {hits}"
    sm = re.search(r"start: %([\w.$-]+):", hits[0])
    assert sm, hits[0]
    return sm.group(1)


def _through_regs(m, v):
    """``v`` and everything it feeds back through, registers included.

    ``Mod.cone`` deliberately stops at a register (a control-structure question
    is usually about one cycle); a time-triggered start is the opposite case, a
    shift chain whose whole point is the cycles it crosses.
    """
    byres = {res: inp for _, res, inp in m.regs}
    seen = set()
    work = [v]
    while work:
        x = work.pop()
        if x in seen:
            continue
        seen.add(x)
        work += [byres[x]] if x in byres else [w for w in m.cone(x) if w != x]
    return seen


def _is_and(m, v, *want):
    """``v`` is a ``comb.and`` whose operands are exactly ``want`` (any order)."""
    rhs = m.defs.get(v, "")
    return rhs.startswith("comb.and") and sorted(m.operands(v)) == sorted(want)


def _outlined(mod, container: str) -> list[str]:
    """The processes outlined out of `container`'s own body, each a kernel of
    its own named `<container>.datapath<k>`."""
    return sorted(k for k in Dcp(mod).kernels if k.startswith(f"{container}.datapath"))


# --- the composition operator: one table, three start policies --------------


# One container, all three rows of the start-policy table: sp_src/sp_cons are spawns
# with nothing to wait for, so they take the container's own start verbatim; sp_post
# reads the buffer sp_cons wrote and a spawn has no offset to be placed at, so only its
# real done will do; sp_r reads what sp_w wrote and both are determinate, so it fires
# at a static offset off start, with no handshake anywhere in its cone.
def test_three_start_policies_from_one_table():

    @kernel
    async def sp_src(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def sp_cons(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = s.get() + 1

    @kernel
    def sp_post(tmp: i32[N], o0: i32[N]):
        for i in range(N):
            o0[i] = tmp[i] * 3

    @kernel
    def sp_w(b: i32[N]):
        for i in range(N):
            b[i] = i * 7

    @kernel
    def sp_r(b: i32[N], o1: i32[N]):
        for i in range(N):
            o1[i] = b[i] + 1

    @kernel
    async def sp_top(o0: i32[N], o1: i32[N]):
        f: Stream[i32]
        tmp: i32[N]
        buf: i32[N]
        await sp_src(f)
        await sp_cons(f, tmp)
        sp_post(tmp, o0)
        sp_w(buf)
        sp_r(buf, o1)

    mod = _to_rtl(sp_top)
    m = Mod(mod.mlir, "sp_top")

    # BROADCAST: the spawn takes the container's start with nothing in between.
    assert _inst_start(m, "sp_src") == "start"
    assert _inst_start(m, "sp_cons") == "start"

    # HANDSHAKE: `risingEdge(done)` = `done & ~reg(done)`. The producer's done
    # is an instance result, so it reaches the start cone; `start` does not.
    post = _inst_start(m, "sp_post")
    cone = m.cone(post)
    assert any(v.endswith(".done") for v in cone), cone
    assert "start" not in cone, cone
    assert _is_and(m, post, *m.operands(post)), m.defs[post]

    # TIME-TRIGGERED: a shift chain off `start`, and no `done` on the way.
    r = _inst_start(m, "sp_r")
    assert "compreg" in m.defs.get(
        r, ""
    ), f"sp_r is not time-triggered: {m.defs.get(r)}"
    chain = _through_regs(m, r)
    assert "start" in chain, chain
    assert not any(v.endswith(".done") for v in chain), chain

    o0 = np.zeros(N, np.int32)
    o1 = np.zeros(N, np.int32)
    mod.cosim(o0, o1)
    assert np.array_equal(o0, (np.arange(N) * 2 + 1) * 3), list(o0)
    assert np.array_equal(o1, np.arange(N) * 7 + 1), list(o1)


# --- topologies: linear chains ------------------------------------------------


# Linear SPSC chains of async def processes wired through internal FIFOs, escalating
# from 2 to 4 stages. KPN determinism: cosim == csim golden at any stall rate, and the
# structural top scales past three processes.
def test_dataflow_linear_chain_depth_escalation():

    n = 16

    # Two processes wired producer -> FIFO -> consumer. `await` spawns each
    # concurrently; the enclosing async region forks `start` to both and joins on
    # both `done`. cosim drives only the composed top's boundary (`out`).
    @kernel
    async def spsc_prod(s: Stream[i32]):
        for i in range(n):
            s.put(i * 2)

    @kernel
    async def spsc_cons(s: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = s.get() + 1

    @kernel
    async def spsc_top(out: i32[n]):
        fifo: Stream[i32]
        await spsc_prod(fifo)
        await spsc_cons(fifo, out)

    mod = _to_rtl(spsc_top)
    # A structural top: it instantiates the leaf processes and a FIFO, not a
    # datapath of its own.
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir

    golden = np.zeros(n, np.int32)
    mod.csim(golden)  # CPU dataflow-runtime golden
    exp = np.array([2 * i + 1 for i in range(n)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-stage chain, two internal channels: fork `start` to three, join three
    # `done`s, wire two seq.fifos in a row.
    @kernel
    async def c3_prod(s: Stream[i32]):
        for i in range(n):
            s.put(i)

    @kernel
    async def c3_mid(s: Stream[i32], t: Stream[i32]):
        for i in range(n):
            t.put(s.get() * 2)

    @kernel
    async def c3_cons(t: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = t.get() + 1

    @kernel
    async def c3_top(out: i32[n]):
        s: Stream[i32]
        t: Stream[i32]
        await c3_prod(s)
        await c3_mid(s, t)
        await c3_cons(t, out)

    mod = _to_rtl(c3_top)
    assert mod.mlir.count("hw.instance") >= 3 and mod.mlir.count("seq.fifo") >= 2

    golden = np.zeros(n, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(n)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A deeper 4-stage chain, three internal channels -- the structural top
    # scales past three processes.
    n4 = 12

    @kernel
    async def c4_prod(s: Stream[i32]):
        for i in range(n4):
            s.put(i)

    @kernel
    async def c4_m1(s: Stream[i32], t: Stream[i32]):
        for i in range(n4):
            t.put(s.get() + 3)

    @kernel
    async def c4_m2(t: Stream[i32], u: Stream[i32]):
        for i in range(n4):
            u.put(t.get() * 2)

    @kernel
    async def c4_cons(u: Stream[i32], out: i32[n4]):
        for i in range(n4):
            out[i] = u.get() - 1

    @kernel
    async def c4_top(out: i32[n4]):
        s: Stream[i32]
        t: Stream[i32]
        u: Stream[i32]
        await c4_prod(s)
        await c4_m1(s, t)
        await c4_m2(t, u)
        await c4_cons(u, out)

    mod = _to_rtl(c4_top)
    assert mod.mlir.count("hw.instance") >= 4 and mod.mlir.count("seq.fifo") >= 3

    golden = np.zeros(n4, np.int32)
    mod.csim(golden)
    exp = np.array([(i + 3) * 2 - 1 for i in range(n4)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n4, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# A linear chain's topology comes from stream SSA wiring, not from spawn order or
# payload type: a scrambled spawn order, a float payload carried as its bit pattern,
# and a user-declared FIFO depth all still wire the same producer -> ... -> consumer chain.
def test_dataflow_linear_chain_variants():

    n = 16

    # The topology is defined by stream SSA wiring, not by spawn order: spawn a
    # 3-stage chain scrambled (cons, mid, prod) and it must still wire
    # prod -> mid -> cons.
    @kernel
    async def oo_prod(s: Stream[i32]):
        for i in range(n):
            s.put(i)

    @kernel
    async def oo_mid(s: Stream[i32], t: Stream[i32]):
        for i in range(n):
            t.put(s.get() * 2)

    @kernel
    async def oo_cons(t: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = t.get() + 1

    @kernel
    async def oo_top(out: i32[n]):
        s: Stream[i32]
        t: Stream[i32]
        await oo_cons(t, out)  # spawned before its producer
        await oo_mid(s, t)
        await oo_prod(s)

    mod = _to_rtl(oo_top)
    golden = np.zeros(n, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(n)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-stage f32 chain with an input boundary array: the LI shell / FIFO carry
    # a float payload (as its bit pattern) across a multi-stage pipeline.
    @kernel
    async def fp_prod(a: f32[n], s: Stream[f32]):
        for i in range(n):
            s.put(a[i])

    @kernel
    async def fp_mid(s: Stream[f32], t: Stream[f32]):
        for i in range(n):
            t.put(s.get() * 2.0)

    @kernel
    async def fp_cons(t: Stream[f32], out: f32[n]):
        for i in range(n):
            out[i] = t.get() + 1.0

    @kernel
    async def fp_top(a: f32[n], out: f32[n]):
        s: Stream[f32]
        t: Stream[f32]
        await fp_prod(a, s)
        await fp_mid(s, t)
        await fp_cons(t, out)

    mod = _to_rtl(fp_top)
    fa = np.arange(n, dtype=np.float32)
    fexp = fa * 2.0 + 1.0
    golden = np.zeros(n, np.float32)
    mod.csim(fa, golden)
    assert np.array_equal(golden, fexp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(n, np.float32)
        mod.cosim(fa, out, stall_prob=gap)
        assert np.array_equal(out, fexp), f"gap={gap}: {list(out)}"

    # A user-declared internal FIFO depth (`Stream[i32, D]`) sizes the emitted
    # seq.fifo. The depth is part of the stream type, so it must be spelled
    # consistently on the channel and on every process parameter that touches it
    # (a mismatched depth is a type error). Back-pressure keeps any depth correct.
    @kernel
    async def fd_prod(s: Stream[i32, 4]):
        for i in range(n):
            s.put(i)

    @kernel
    async def fd_mid(s: Stream[i32, 4], t: Stream[i32, 1]):
        for i in range(n):
            t.put(s.get() * 2)

    @kernel
    async def fd_cons(t: Stream[i32, 1], out: i32[n]):
        for i in range(n):
            out[i] = t.get() + 1

    @kernel
    async def fd_top(out: i32[n]):
        s: Stream[i32, 4]  # deep internal FIFO
        t: Stream[i32, 1]  # tight internal FIFO
        await fd_prod(s)
        await fd_mid(s, t)
        await fd_cons(t, out)

    mod = _to_rtl(fd_top)
    # The deep channel keeps its depth; the depth-1 channel is raised to 2 (the
    # seq.fifo minimum, so it never appears as "depth 1") rather than crashing on
    # zero-width pointers, and the design still builds and runs.
    # Anchored: a bare `"depth 1" in` substring test also matches `depth 16`.
    assert re.search(r"\bdepth 4\b", mod.mlir), mod.mlir
    assert not re.search(r"\bdepth 1\b", mod.mlir), mod.mlir
    golden = np.zeros(n, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(n)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- topologies: fan-out / fan-in --------------------------------------------


# Non-linear topologies: a producer branching to two independent consumer chains,
# and elastic joins reading two / three input streams per firing.
def test_dataflow_fanout_fanin():

    n = 16

    # Two output channels feeding two independent consumer chains (a branch, no
    # reconvergence). The producer's two puts share one region, so the out-hazard
    # ORs their back-pressure (all-or-nothing per firing). These are distinct
    # SPSC channels, not a broadcast. Two boundary output arrays exercise
    # multi-output wiring.
    @kernel
    async def split(a: Stream[i32], b: Stream[i32]):
        for i in range(n):
            a.put(i)
            b.put(i * 10)

    @kernel
    async def br_cons_a(a: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = a.get() + 1

    @kernel
    async def br_cons_b(b: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = b.get() - 1

    @kernel
    async def br_top(outa: i32[n], outb: i32[n]):
        a: Stream[i32]
        b: Stream[i32]
        await split(a, b)
        await br_cons_a(a, outa)
        await br_cons_b(b, outb)

    mod = _to_rtl(br_top)
    assert mod.mlir.count("hw.instance") >= 3 and mod.mlir.count("seq.fifo") >= 2

    ga = np.zeros(n, np.int32)
    gb = np.zeros(n, np.int32)
    mod.csim(ga, gb)
    expa = np.array([i + 1 for i in range(n)], np.int32)
    expb = np.array([i * 10 - 1 for i in range(n)], np.int32)
    assert np.array_equal(ga, expa), list(ga)
    assert np.array_equal(gb, expb), list(gb)
    for gap in (0.0, 0.5, 0.8):
        oa = np.zeros(n, np.int32)
        ob = np.zeros(n, np.int32)
        mod.cosim(oa, ob, stall_prob=gap)
        assert np.array_equal(oa, expa), f"gap={gap}: a={list(oa)}"
        assert np.array_equal(ob, expb), f"gap={gap}: b={list(ob)}"

    # A stage reading TWO input streams unconditionally in one region
    # (c = a.get() + b.get()) -- an elastic join. It consumes one token from EACH
    # per firing and pops them together, so under independent random stalls on
    # the two inputs no token is lost (the leading input waits for the lagging).
    @kernel
    async def j2_prodA(a: Stream[i32]):
        for i in range(n):
            a.put(i)

    @kernel
    async def j2_prodB(b: Stream[i32]):
        for i in range(n):
            b.put(i * 10)

    @kernel
    async def j2_join(a: Stream[i32], b: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = a.get() + b.get()

    @kernel
    async def j2_top(out: i32[n]):
        a: Stream[i32]
        b: Stream[i32]
        await j2_prodA(a)
        await j2_prodB(b)
        await j2_join(a, b, out)

    mod = _to_rtl(j2_top)
    golden = np.zeros(n, np.int32)
    mod.csim(golden)
    exp = np.array([i + i * 10 for i in range(n)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-input elastic join -- the all-inputs-pop-together gating scales past
    # two inputs.
    n3 = 12

    @kernel
    async def j3_prodA(a: Stream[i32]):
        for i in range(n3):
            a.put(i)

    @kernel
    async def j3_prodB(b: Stream[i32]):
        for i in range(n3):
            b.put(i * 10)

    @kernel
    async def j3_prodC(c: Stream[i32]):
        for i in range(n3):
            c.put(i * 100)

    @kernel
    async def j3_join(a: Stream[i32], b: Stream[i32], c: Stream[i32], out: i32[n3]):
        for i in range(n3):
            out[i] = a.get() + b.get() + c.get()

    @kernel
    async def j3_top(out: i32[n3]):
        a: Stream[i32]
        b: Stream[i32]
        c: Stream[i32]
        await j3_prodA(a)
        await j3_prodB(b)
        await j3_prodC(c)
        await j3_join(a, b, c, out)

    mod = _to_rtl(j3_top)
    golden = np.zeros(n3, np.int32)
    mod.csim(golden)
    exp = np.array([i + i * 10 + i * 100 for i in range(n3)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(n3, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- topologies: deterministic merge ------------------------------------------


# Deterministic MPSC merge: a stage consuming ONE of two inputs per firing, chosen by
# a data-determined selector -- from a control stream and from a memory array.
# Data-driven, not arrival-driven, so it stays in KPN.
def test_dataflow_deterministic_merge():

    n = 16

    # The selector is a control-stream token read at stage 0, so the chosen
    # `a`/`b` get lands at stage 1 (fifo read latency) -- a MULTI-STAGE join: the
    # selected mid-pipeline get FREEZES the pipeline when its input is empty,
    # while a non-selected empty input never stalls. Producers emit matching
    # counts (rate law): each of a/b is chosen N/2 times.
    pattern = np.array([i % 2 for i in range(n)], np.int32)  # 0,1,0,1,...

    @kernel
    async def cs_prodA(a: Stream[i32]):
        for i in range(n // 2):
            a.put(i)

    @kernel
    async def cs_prodB(b: Stream[i32]):
        for i in range(n // 2):
            b.put(100 + i)

    @kernel
    async def cs_prodSel(p: i32[n], sel: Stream[i32]):
        for i in range(n):
            sel.put(p[i])

    @kernel
    async def cs_merge(sel: Stream[i32], a: Stream[i32], b: Stream[i32], out: i32[n]):
        for i in range(n):
            s: i32 = sel.get()
            x: i32 = 0
            if s == 0:
                x = a.get()
            else:
                x = b.get()
            out[i] = x

    @kernel
    async def cs_top(p: i32[n], out: i32[n]):
        a: Stream[i32]
        b: Stream[i32]
        sel: Stream[i32]
        await cs_prodA(a)
        await cs_prodB(b)
        await cs_prodSel(p, sel)
        await cs_merge(sel, a, b, out)

    exp = np.zeros(n, np.int32)
    ca = cb = 0
    for i in range(n):
        if pattern[i] == 0:
            exp[i] = ca
            ca += 1
        else:
            exp[i] = 100 + cb
            cb += 1

    mod = _to_rtl(cs_top)
    golden = np.zeros(n, np.int32)
    mod.csim(pattern, golden)
    assert np.array_equal(golden, exp), (list(golden), list(exp))
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(pattern, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # The canonical case: the selector is a MEMORY-array read `sel[i]`. The load's
    # read latency puts the predicate at stage 1 and the selected get at stage 2 --
    # deeper than the control-stream form, so the multi-stage freeze must handle
    # arbitrary get depth.
    sel = np.array([i % 2 for i in range(n)], np.int32)

    @kernel
    async def ds_prodA(a: Stream[i32]):
        for i in range(n // 2):
            a.put(i)

    @kernel
    async def ds_prodB(b: Stream[i32]):
        for i in range(n // 2):
            b.put(100 + i)

    @kernel
    async def ds_merge(sel: i32[n], a: Stream[i32], b: Stream[i32], out: i32[n]):
        for i in range(n):
            # One scalar per branch and a single store: two stores to out[i]
            # would need two write ports -> II=2, which the II==1 shell rejects.
            x: i32 = 0
            if sel[i] == 0:
                x = a.get()
            else:
                x = b.get()
            out[i] = x

    @kernel
    async def ds_top(sel: i32[n], out: i32[n]):
        a: Stream[i32]
        b: Stream[i32]
        await ds_prodA(a)
        await ds_prodB(b)
        await ds_merge(sel, a, b, out)

    exp = np.zeros(n, np.int32)
    ca = cb = 0
    for i in range(n):
        if sel[i] == 0:
            exp[i] = ca
            ca += 1
        else:
            exp[i] = 100 + cb
            cb += 1

    mod = _to_rtl(ds_top)
    golden = np.zeros(n, np.int32)
    mod.csim(sel, golden)
    assert np.array_equal(golden, exp), (list(golden), list(exp))
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(sel, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- topologies: nesting -------------------------------------------------------


# A process that is itself a container: the CPU golden across two and three nesting
# levels, then RTL emit of a container-as-callee and of a container whose stream args
# cross its own boundary.
def test_dataflow_nested_containers():

    n = 16

    # The runtime flattens the nest onto one marl scheduler (a nested
    # `allo_df_open` reuses the enclosing scheduler instead of binding a second
    # one to the fiber's thread, which aborts); each level keeps its own
    # WaitGroup so joins are scoped.
    a_nc = (np.arange(n, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def nc_produce(a: i32[n], s: Stream[i32]):
        for i in range(n):
            s.put(a[i])

    @kernel
    async def nc_inner_a(x: Stream[i32], y: Stream[i32]):
        for i in range(n):
            y.put(x.get() + 1)

    @kernel
    async def nc_inner_b(y: Stream[i32], z: Stream[i32]):
        for i in range(n):
            z.put(y.get() * 2)

    @kernel
    async def nc_mid(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await nc_inner_a(x, y)
        await nc_inner_b(y, z)

    @kernel
    async def nc_consume(t: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = t.get()

    @kernel
    async def nc_top(a: i32[n], out: i32[n]):
        s: Stream[i32]
        t: Stream[i32]
        await nc_produce(a, s)
        await nc_mid(s, t)
        await nc_consume(t, out)

    mod = _to_rtl(nc_top)
    exp = (a_nc + 1) * 2
    # csim is deterministic (KPN); repeat to surface any scheduler/WaitGroup race.
    for _ in range(8):
        out = np.zeros(n, np.int32)
        mod.csim(a_nc, out)
        assert np.array_equal(out, exp), list(out)

    # Three container levels: top -> mid -> deep -> {da, db}. The scheduler-reuse
    # flattening holds at arbitrary nesting depth.
    a_dn = (np.arange(n, dtype=np.int32) * 3 + 5) & 0xFF

    @kernel
    async def dn_produce(a: i32[n], s: Stream[i32]):
        for i in range(n):
            s.put(a[i])

    @kernel
    async def dn_da(x: Stream[i32], y: Stream[i32]):
        for i in range(n):
            y.put(x.get() + 3)

    @kernel
    async def dn_db(y: Stream[i32], z: Stream[i32]):
        for i in range(n):
            z.put(y.get() * 2)

    @kernel
    async def dn_deep(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await dn_da(x, y)
        await dn_db(y, z)

    @kernel
    async def dn_mid(x: Stream[i32], z: Stream[i32]):
        await dn_deep(x, z)

    @kernel
    async def dn_consume(t: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = t.get()

    @kernel
    async def dn_top(a: i32[n], out: i32[n]):
        s: Stream[i32]
        t: Stream[i32]
        await dn_produce(a, s)
        await dn_mid(s, t)
        await dn_consume(t, out)

    mod = _to_rtl(dn_top)
    out = np.zeros(n, np.int32)
    mod.csim(a_dn, out)
    assert np.array_equal(out, (a_dn + 3) * 2), list(out)

    # A spawned process that is itself a container, with only MEMREF boundaries
    # -- no stream crosses cc_mid's boundary, isolating container-as-callee from
    # stream boundary ports. Emit must build cc_mid as its own hw.module before
    # the top, keep the outermost (uncalled) container as the DUT, and forward
    # cc_mid's memref boundaries through it.
    a_cc = (np.arange(n, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def cc_inner_p(a: i32[n], s: Stream[i32]):
        for i in range(n):
            s.put(a[i] * 2)

    @kernel
    async def cc_inner_c(s: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = s.get() + 1

    @kernel
    async def cc_mid(a: i32[n], out: i32[n]):
        s: Stream[i32]
        await cc_inner_p(a, s)
        await cc_inner_c(s, out)

    @kernel
    async def cc_top(a: i32[n], out: i32[n]):
        await cc_mid(a, out)

    mod = _to_rtl(cc_top)
    # An emitted module takes its func symbol legalized as a SystemVerilog
    # identifier (`cc_top.cc_mid` -> `cc_top_cc_mid`), so the symbol, the port
    # manifest's key, and the Verilog module name are one name.
    mods = re.findall(r"hw\.module @([\w.]+)", mod.mlir)
    assert mods[0] == "cc_top", mods
    assert "cc_top_cc_mid" in mods, mods
    assert mods.index("cc_top_cc_mid") > mods.index("cc_top"), mods

    golden = np.zeros(n, np.int32)
    mod.csim(a_cc, golden)
    assert np.array_equal(golden, a_cc * 2 + 1), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(a_cc, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A channel crossing a container boundary: sb_mid's two stream args are block
    # args forwarded to its inner processes, so it must expose them as stream
    # ports (data/valid/ready) that look exactly like a leaf's, and the parent
    # wires a FIFO on each side. The full hierarchical composition.
    a_sb = (np.arange(n, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def sb_produce(a: i32[n], s: Stream[i32]):
        for i in range(n):
            s.put(a[i])

    @kernel
    async def sb_inner_a(x: Stream[i32], y: Stream[i32]):
        for i in range(n):
            y.put(x.get() + 1)

    @kernel
    async def sb_inner_b(y: Stream[i32], z: Stream[i32]):
        for i in range(n):
            z.put(y.get() * 2)

    @kernel
    async def sb_mid(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await sb_inner_a(x, y)
        await sb_inner_b(y, z)

    @kernel
    async def sb_consume(t: Stream[i32], out: i32[n]):
        for i in range(n):
            out[i] = t.get()

    @kernel
    async def sb_top(a: i32[n], out: i32[n]):
        s: Stream[i32]
        t: Stream[i32]
        await sb_produce(a, s)
        await sb_mid(s, t)
        await sb_consume(t, out)

    mod = _to_rtl(sb_top)
    ir = mod.mlir
    assert re.findall(r"hw\.module @([\w.]+)", ir)[0] == "sb_top"
    assert "_st_data" in ir and "_st_valid" in ir and "_st_ready" in ir
    assert ir.count("seq.fifo") >= 3, ir.count("seq.fifo")  # s, t, and mid's y

    golden = np.zeros(n, np.int32)
    mod.csim(a_sb, golden)
    assert np.array_equal(golden, (a_sb + 1) * 2), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(a_sb, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- channels seeded with initial tokens --------------------------------------


# A channel initialized with tokens: a feedback cycle seeded with one token so it can
# turn at all, and a channel seeded from a captured NumPy array. Golden (csim) seeds
# the FIFO; RTL (cosim) prepends the tokens on the consumer read port.
def test_dataflow_channel_seeded_with_initial_tokens():

    n = 8

    # A dataflow CYCLE seeded with one token: `fb_emit` reads the feedback channel
    # `t`, records it, and produces x+1 into `s`; `fb_fwd` forwards s -> t. The
    # preloaded token turns the cycle, so out = [0, 1, ..., N-1].
    @kernel
    async def fb_emit(t: Stream[i32], s: Stream[i32], out: i32[n]):
        for i in range(n):
            x = t.get()
            out[i] = x
            s.put(x + 1)

    @kernel
    async def fb_fwd(s: Stream[i32], t: Stream[i32]):
        for i in range(n):
            t.put(s.get())

    @kernel
    async def fb_top(out: i32[n]):
        s: Stream[i32]
        t: Stream[i32] = [0]  # feedback channel, one initial token
        await fb_emit(t, s, out)
        await fb_fwd(s, t)

    mod = _to_rtl(fb_top)
    ir = mod.mlir
    # The seeded channel keeps the plain `seq.fifo` and adds the init-prepend
    # shim (its down-counter) on the consumer side.
    assert "hw.instance" in ir and "seq.fifo" in ir and "_init_rem" in ir

    golden = np.zeros(n, np.int32)
    mod.csim(golden)  # CPU dataflow-runtime golden: seeded, no deadlock
    assert np.array_equal(golden, np.arange(n, dtype=np.int32)), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(n, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # Initial tokens from an externally-defined NumPy array captured into the
    # kernel -- more elements than a hand-written list, exercising the shim's
    # init-ROM mux chain + multi-bit down-counter. Acyclic SPSC: the channel
    # history is [init] ++ [produced], so cons reads the K seeded tokens first,
    # then prod's M values (back-pressure carries the producer tokens through the
    # depth-2 FIFO while the init drains).
    k, m = 8, 8
    init = np.random.default_rng(0).integers(0, 1000, size=k, dtype=np.int32)

    @kernel
    async def cap_prod(c: Stream[i32]):
        for i in range(m):
            c.put(100 + i)

    @kernel
    async def cap_cons(c: Stream[i32], out: i32[k + m]):
        for i in range(k + m):
            out[i] = c.get()

    @kernel
    async def cap_top(out: i32[k + m]):
        c: Stream[i32] = init  # seeded from the captured NumPy array
        await cap_prod(c)
        await cap_cons(c, out)

    mod = _to_rtl(cap_top)
    exp = np.concatenate([init, 100 + np.arange(m, dtype=np.int32)])
    golden = np.zeros(k + m, np.int32)
    mod.csim(golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.7):
        out = np.zeros(k + m, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- a container's own datapath becomes a process -----------------------------


# Compute that neither feeds nor reads the process network still has to go somewhere:
# it becomes a process running concurrently with it.
def test_loose_compute_beside_the_network():

    @kernel
    async def lc_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i] * 2)

    @kernel
    async def lc_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def lc_top(x: i32[N], out: i32[N], aux: i32[N]):
        f: Stream[i32]
        for i in range(N):
            aux[i] = x[i] * 10  # loose: disjoint from the network
        await lc_prod(x, f)
        await lc_cons(f, out)

    mod = _to_rtl(lc_top)
    # The outlined work is a real child module, and the container keeps none of
    # the datapath ops.
    assert _outlined(mod, "lc_top") == ["lc_top.datapath0"]
    assert not Dcp(mod).func("lc_top").has("allo.dcp.compute")
    # And it is spawned, not sequenced: the container stays self-timed with the
    # outlined process in it, which is what "runs concurrently" means here.
    assert mod.schedule().func("lc_top").determinacy == "concurrent"

    out = np.zeros(N, np.int32)
    aux = np.zeros(N, np.int32)
    mod.cosim(A, out, aux)
    assert np.array_equal(out, A * 2 + 1), list(out)
    assert np.array_equal(aux, A * 10), list(aux)


# The outlined prologue writes a buffer the network reads, so the spawns must wait
# for its real done, not a static offset: an await spawn takes the broadcast start
# verbatim, so a determinate producer's latency alone gives it nothing to be held
# back by. Without that gate the producer streams the buffer's reset contents.
def test_prologue_feeding_the_network():

    @kernel
    async def pf_prod(buf: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(buf[i])

    @kernel
    async def pf_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def pf_top(x: i32[N], buf: i32[N], out: i32[N]):
        f: Stream[i32]
        for i in range(N):
            buf[i] = x[i] * 5  # must complete BEFORE pf_prod reads buf
        await pf_prod(buf, f)
        await pf_cons(f, out)

    mod = _to_rtl(pf_top)
    buf = np.zeros(N, np.int32)
    out = np.zeros(N, np.int32)
    mod.cosim(A, buf, out)
    assert np.array_equal(buf, A * 5), list(buf)
    assert np.array_equal(out, A * 5 + 1), list(out)


# The mirror case: the outlined pass reads what the network wrote, so it waits on the
# (indeterminate) consumer's real done.
def test_epilogue_over_the_network_output():

    @kernel
    async def ep_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def ep_cons(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = s.get() + 1

    @kernel
    async def ep_top(tmp: i32[N], out: i32[N]):
        f: Stream[i32]
        await ep_prod(f)
        await ep_cons(f, tmp)
        for i in range(N):
            out[i] = tmp[i] * 3  # must run AFTER the network drains

    mod = _to_rtl(ep_top)
    tmp = np.zeros(N, np.int32)
    out = np.zeros(N, np.int32)
    mod.cosim(tmp, out)
    assert np.array_equal(out, np.array([(2 * i + 1) * 3 for i in range(N)])), list(out)


# Runs are split by the calls between them, so loose work on both sides of a network
# becomes two processes rather than one that would have to run before and after itself.
def test_prologue_and_epilogue_are_separate_processes():

    @kernel
    async def pe_prod(buf: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(buf[i])

    @kernel
    async def pe_cons(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = s.get()

    @kernel
    async def pe_top(x: i32[N], buf: i32[N], tmp: i32[N], out: i32[N]):
        f: Stream[i32]
        for i in range(N):
            buf[i] = x[i] + 1
        await pe_prod(buf, f)
        await pe_cons(f, tmp)
        for i in range(N):
            out[i] = tmp[i] * 7

    mod = _to_rtl(pe_top)
    assert _outlined(mod, "pe_top") == ["pe_top.datapath0", "pe_top.datapath1"]

    buf = np.zeros(N, np.int32)
    tmp = np.zeros(N, np.int32)
    out = np.zeros(N, np.int32)
    mod.cosim(A, buf, tmp, out)
    assert np.array_equal(out, (A + 1) * 7), list(out)


# A scalar the container computes for a process to consume is wired child to child:
# the producer's held result port drives the consumer's scalar input, and the
# consumer starts on the producer's done. That is composition, not compute, so the
# top still has no datapath.
def test_scalar_computed_for_a_process():

    @kernel
    async def sc_prod(y: i32, s: Stream[i32]):
        for i in range(N):
            s.put(y + i)

    @kernel
    async def sc_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def sc_top(x: i32[N], out: i32[N]):
        f: Stream[i32]
        y: i32 = x[0] * 3
        await sc_prod(y, f)
        await sc_cons(f, out)

    mod = _to_rtl(sc_top)
    out = np.zeros(N, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, np.array([A[0] * 3 + i + 1 for i in range(N)])), list(
        out
    )


# A process's scalar result feeding outlined work, whose own result is the
# container's return: the same scalar link in both directions.
def test_process_result_post_processed():

    @kernel
    async def pr_prod(x: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(x[i])

    @kernel
    async def pr_cons(s: Stream[i32], out: i32[N]) -> i32:
        acc: i32 = 0
        for i in range(N):
            v: i32 = s.get()
            out[i] = v + 1
            acc += v
        return acc

    @kernel
    async def pr_top(x: i32[N], out: i32[N]) -> i32:
        f: Stream[i32]
        await pr_prod(x, f)
        r: i32 = await pr_cons(f, out)
        return r * 2 + 1

    mod = _to_rtl(pr_top)
    out = np.zeros(N, np.int32)
    res = mod.cosim(A, out)
    assert res.result == int(A.sum()) * 2 + 1, res.result
    assert np.array_equal(out, A + 1), list(out)


# Loose work touching a STREAM outlines just as happily, and the result is a producer
# process on that channel: the stream is one more captured argument, so the process
# network gains a node rather than a special case.
def test_the_container_itself_can_drive_a_channel():

    @kernel
    async def cd_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def cd_top(x: i32[N], out: i32[N]):
        f: Stream[i32]
        for i in range(N):
            f.put(x[i] * 2)  # the container writes the channel itself
        await cd_cons(f, out)

    mod = _to_rtl(cd_top)
    assert _outlined(mod, "cd_top") == ["cd_top.datapath0"]
    out = np.zeros(N, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, A * 2 + 1), list(out)


# Outlining is for the STRUCTURAL top only. A plain memref/scalar composition lowers
# to the leaf CallUnit path, which hosts loose work beside a call perfectly well, so
# outlining it would cost a module boundary and buy nothing.
def test_a_sequential_composition_is_not_outlined():

    @kernel
    def sq_child(x: i32[N], y: i32[N]):
        for i in range(N):
            y[i] = x[i] * 2

    @kernel
    def sq_top(x: i32[N], y: i32[N], z: i32[N]):
        sq_child(x, y)
        for i in range(N):
            z[i] = x[i] + 1

    assert _outlined(_to_rtl(sq_top), "sq_top") == []


# --- mixing dataflow subnetworks with sequential kernels ----------------------


# A container may mix an await dataflow sub-network with plain sequential kernels: an
# independent one runs concurrently, one that consumes the network's output is gated
# on the producer's real done.
def test_mixed_dataflow_sequential():

    # Data-INDEPENDENT of the dataflow processes (disjoint memory), so all run
    # concurrently: the container broadcasts `start` and joins every `done`, and
    # the df pair streams through its FIFO while the plain kernel runs beside it.
    @kernel
    async def mx_prod(s: Stream[i32]):
        for i in range(16):
            s.put(i * 2)

    @kernel
    async def mx_cons(s: Stream[i32], o1: i32[16]):
        for i in range(16):
            o1[i] = s.get() + 1

    @kernel
    def mx_post(D: i32[16], o2: i32[16]):  # plain (non-async) kernel, disjoint
        for i in range(16):
            o2[i] = D[i] + 100

    @kernel
    async def mx_top(D: i32[16], o1: i32[16], o2: i32[16]):
        fifo: Stream[i32]
        await mx_prod(fifo)
        await mx_cons(fifo, o1)
        mx_post(D, o2)

    mod = _to_rtl(mx_top)
    # A structural top holding both the df processes (+ FIFO) and the seq kernel.
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir

    D = (np.arange(16, dtype=np.int32) + 5) & 0xFF
    o1 = np.zeros(16, np.int32)
    o2 = np.zeros(16, np.int32)
    mod.cosim(D, o1, o2)
    assert np.array_equal(o1, np.array([2 * i + 1 for i in range(16)], np.int32))
    assert np.array_equal(o2, D + 100)

    # A plain kernel that CONSUMES the dataflow network's array output cannot
    # broadcast-start -- an async producer has no static latency, so there is no
    # offset to place it at. Its `start` is gated on the consumer's real `done`,
    # and it shares the `tmp` boundary serially: the writer fully drains first.
    @kernel
    async def rd_prod(s: Stream[i32]):
        for i in range(16):
            s.put(i * 2)

    @kernel
    async def rd_cons(s: Stream[i32], tmp: i32[16]):
        for i in range(16):
            tmp[i] = s.get() + 1

    @kernel
    def rd_post(tmp: i32[16], out: i32[16]):  # plain: consumes the df output
        for i in range(16):
            out[i] = tmp[i] * 3

    @kernel
    async def rd_top(tmp: i32[16], out: i32[16]):
        fifo: Stream[i32]
        await rd_prod(fifo)
        await rd_cons(fifo, tmp)  # the df network writes tmp
        rd_post(tmp, out)  # reads tmp, so it is gated on rd_cons's done

    mod = _to_rtl(rd_top)
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir
    # Every child is a call node, spawn or not: `allo.async` marks the START
    # POLICY, it does not make a different kind of node.
    top = Dcp(mod).func("rd_top")
    assert top.callees(spawned=False) == ["rd_top.rd_post"]
    assert top.callees(spawned=True) == ["rd_top.rd_prod", "rd_top.rd_cons"]
    # ... and rd_post takes the handshake, not the broadcast: its `start` is
    # something the top computed, not the container's own.
    inst = [l for l in mod.mlir.splitlines() if "hw.instance" in l and "rd_post" in l]
    assert len(inst) == 1 and "start: %start:" not in inst[0], inst

    tmp = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    mod.cosim(tmp, out)
    exp = np.array([(2 * i + 1) * 3 for i in range(16)], np.int32)
    assert np.array_equal(out, exp), list(out)

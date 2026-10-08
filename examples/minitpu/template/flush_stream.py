# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 H6 (plan F2): the fetch queue as a Stream, and what flushing it costs.

``sequencer_fetch_queue.sv`` is a 4-deep FIFO between the IRAM's read port and
issue. A taken branch FLUSHES it: at the edge of the branch cycle every
buffered bundle is discarded (``count_q <= 0``), the read landing that cycle is
dropped (``fifo_push = req_pending && !pc_flush``), and the producer restarts
at the target -- so the target is at the head 3 edges later and a body over
``LB_CAP`` pays exactly 2 bubbles per iteration (``control_geometry``). An Allo
``Stream`` has ``put``/``get`` (and ``try_*``/``empty``/``full``) and nothing
that discards its contents. This file is the probe and the prototype of the
README D-n draft in ``u4_track_a_2026-10-08.rst`` ("a flushable stream").

The fetch path is split as the plan's F2 asks: ``fetcher`` (the address
counter, the in-flight read, the IRAM) puts ``{epoch, addr}`` and the bundle into
two 4-deep Streams that move together (``ftag``, ``fifo``: one 141-bit
element corrupts the simulator's heap, finding T-1); ``queue`` holds the head (the RTL's
``fifo_mem[0]``, a flop) and serves issue. Control crosses once per cycle in
both directions (``k_ctl`` queue -> fetcher: occupancy, flush, target;
``k_push`` fetcher -> queue: "I put one"), so iteration = cycle on every
backend and nothing races. The two forms differ only in how the queue gets
rid of what a flush strands in ``fifo``:

``drain``  in the flush's edge, ``get`` and discard every stranded token (a
           bounded loop of conditional gets in ONE iteration). Exact per
           iteration on the simulator and in csim; in hardware a FIFO serves
           one get per cycle, so this loop is the cost hidden from both.
``epoch``  tokens carry the producer's epoch; a flush flips both epochs and
           the queue drops stale tokens as it meets them, one ``get`` per
           cycle (the only per-cycle-honest form with today's Stream). The
           stranded tokens still occupy ``fifo``, so the fetcher's room test
           must count them -- or ``fifo`` must be ``2 * DEPTH - 1`` deep, or
           the producer blocks on ``put`` while the consumer waits on its
           ``k_push``: a DEADLOCK (D-23's condition). Counting them costs
           issue slots after every flush: measured below.

The proposal (``Stream.flush()``): a consumer-side flush that empties the
channel at the edge and drops a put of the same edge; lowered to a FIFO with
a synchronous clear (what the RTL is), it costs nothing and needs no epoch
while producer and consumer are cycle-locked; a self-timed pair needs the
epoch (the producer may put before it hears of the flush), carried by the
channel, not by the bodies.

Run: ``$ALLO_PYTHON -m examples.minitpu.template.flush_stream [--backend systemc]``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from allo.compose import Architecture, Channel, Memory, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

DEPTH = 4  # sequencer_fetch_queue DEPTH


@unit(memories=("RST", "WE", "WA", "WD", "PAUSE", "FLUSH", "RADDR", "POP"),
      writes=("c_rst", "c_we", "c_wa", "c_wd", "c_pause", "q_rst", "q_flush", "q_raddr", "q_pop"),
      parameters=("N",))
def src(xrst: uint1[N], xwe: uint1[N], xwa: UInt(12)[N], xwd: UInt(32)[N, 4], xpause: uint1[N],
        xflush: uint1[N], xraddr: UInt(12)[N], xpop: uint1[N]):
    for t in range(N):
        d: UInt(128) = 0
        d[0:32] = xwd[t, 0]
        d[32:64] = xwd[t, 1]
        d[64:96] = xwd[t, 2]
        d[96:128] = xwd[t, 3]
        c_rst.put(xrst[t])
        c_we.put(xwe[t])
        c_wa.put(xwa[t])
        c_wd.put(d)
        c_pause.put(xpause[t])
        q_rst.put(xrst[t])
        q_flush.put(xflush[t])
        q_raddr.put(xraddr[t])
        q_pop.put(xpop[t])


@unit(reads=("c_rst", "c_we", "c_wa", "c_wd", "c_pause", "k_count", "k_flush", "k_target"),
      writes=("fifo", "ftag", "k_push", "o_rda", "o_rdv"), parameters=("N",))
def fetcher():
    iram: UInt(128)[4096] = 0
    rd_reg: UInt(128) = 0
    next_addr: UInt(12) = 0
    req_pending: uint1 = 0
    req_addr: UInt(12) = 0
    epoch: uint1 = 0
    for _ in range(N):
        rst: uint1 = c_rst.get()
        we: uint1 = c_we.get()
        wa: UInt(12) = c_wa.get()
        wd: UInt(128) = c_wd.get()
        pause: uint1 = c_pause.get()
        if rst == 0:
            next_addr = 0
            req_pending = 0
            req_addr = 0
        count: UInt(3) = k_count.get()
        flush: uint1 = k_flush.get()
        target: UInt(12) = k_target.get()
        rd_valid: uint1 = 0
        if pause == 0 and count < 3:
            rd_valid = 1
        o_rda.put(next_addr)
        o_rdv.put(rd_valid)
        bram: UInt(128) = rd_reg
        ir: int32 = next_addr
        rd_reg = iram[ir]
        if we:
            iw: int32 = wa
            iram[iw] = wd
        pushed: uint1 = 0
        if rst:
            if flush:
                next_addr = target
                req_pending = 0
                epoch = 1 - epoch
            else:
                if req_pending:
                    tag: UInt(13) = 0
                    tag[0:12] = req_addr
                    tag[12] = epoch
                    fifo.put(bram)  # T-1: no stream element wider than 128 bits
                    ftag.put(tag)
                    pushed = 1
                req_pending = rd_valid
                req_addr = next_addr
                if rd_valid:
                    next_addr = next_addr + 1
        else:
            epoch = 1 - epoch  # a reset strands what is buffered, as a flush does
        k_push.put(pushed)


@unit(reads=("q_rst", "q_flush", "q_raddr", "q_pop", "fifo", "ftag", "k_push"),
      writes=("k_count", "k_flush", "k_target", "o_data", "o_valid", "o_baddr", "o_empty", "o_full"),
      parameters=("N", "DEPTH"))
def queue_drain():
    head: UInt(128) = 0
    head_tag: UInt(13) = 0
    head_live: uint1 = 0
    live: UInt(3) = 0  # live tokens in fifo, behind the head
    stale: UInt(3) = 0  # stranded by a flush, not yet discarded
    for _ in range(N):
        rst: uint1 = q_rst.get()
        flush: uint1 = q_flush.get()
        raddr: UInt(12) = q_raddr.get()
        pop_i: uint1 = q_pop.get()
        if rst == 0:
            stale = stale + live
            live = 0
            head_live = 0
        count: UInt(3) = head_live + live
        k_count.put(count)
        k_flush.put(flush & rst)
        k_target.put(raddr)
        o_data.put(head)
        o_baddr.put(head_tag[0:12])
        o_valid.put(head_live)
        empty: uint1 = 1 - head_live
        o_empty.put(empty)
        full: uint1 = (head_live + live) == 4
        o_full.put(full)
        pushed: uint1 = k_push.get()
        if rst:
            if flush:
                stale = stale + live
                live = 0
                head_live = 0
            else:
                if pop_i and head_live:
                    head_live = 0
                live = live + pushed
        # drain: every stranded token goes in this one iteration (one cycle per
        # get in hardware), then the head refills
        for _k in range(2 * DEPTH):
            if stale > 0:
                junk: UInt(128) = fifo.get()
                jtag: UInt(13) = ftag.get()
                stale = stale - 1
        if head_live == 0 and live > 0:
            head = fifo.get()
            head_tag = ftag.get()
            live = live - 1
            head_live = 1


@unit(reads=("q_rst", "q_flush", "q_raddr", "q_pop", "fifo", "ftag", "k_push"),
      writes=("k_count", "k_flush", "k_target", "o_data", "o_valid", "o_baddr", "o_empty", "o_full"),
      parameters=("N",))
def queue_epoch():
    head: UInt(128) = 0
    head_tag: UInt(13) = 0
    head_live: uint1 = 0
    live: UInt(3) = 0
    stale: UInt(3) = 0
    epoch: uint1 = 0
    for _ in range(N):
        rst: uint1 = q_rst.get()
        flush: uint1 = q_flush.get()
        raddr: UInt(12) = q_raddr.get()
        pop_i: uint1 = q_pop.get()
        if rst == 0:
            stale = stale + live
            live = 0
            head_live = 0
        # stranded tokens still occupy fifo: the room test counts them, or
        # fifo overflows and the fetcher's put blocks (deadlock)
        count: UInt(3) = head_live + live + stale
        k_count.put(count)
        k_flush.put(flush & rst)
        k_target.put(raddr)
        o_data.put(head)
        o_baddr.put(head_tag[0:12])
        o_valid.put(head_live)
        empty: uint1 = 1 - head_live
        o_empty.put(empty)
        full: uint1 = (head_live + live) == 4
        o_full.put(full)
        pushed: uint1 = k_push.get()
        if rst:
            if flush:
                stale = stale + live
                live = 0
                head_live = 0
                epoch = 1 - epoch
            else:
                if pop_i and head_live:
                    head_live = 0
                live = live + pushed
        else:
            epoch = 1 - epoch
        # one get per cycle: a stale token (its epoch is not ours) is dropped
        # and the head stays empty this cycle
        if stale > 0 or (head_live == 0 and live > 0):
            tok: UInt(128) = fifo.get()
            tag: UInt(13) = ftag.get()
            if tag[12] != epoch:
                stale = stale - 1
            else:
                head = tok
                head_tag = tag
                live = live - 1
                head_live = 1


@unit(memories=("DATA", "RDA", "RDV", "VALID", "BADDR", "EMPTY", "FULL"),
      reads=("o_data", "o_rda", "o_rdv", "o_valid", "o_baddr", "o_empty", "o_full"), parameters=("N",))
def sink(ydata: UInt(32)[N, 4], yrda: UInt(12)[N], yrdv: uint1[N], yvalid: uint1[N], ybaddr: UInt(12)[N],
         yempty: uint1[N], yfull: uint1[N]):
    for t in range(N):
        d: UInt(128) = o_data.get()
        ydata[t, 0] = d[0:32]  # A5: the 2-D output stored first
        ydata[t, 1] = d[32:64]
        ydata[t, 2] = d[64:96]
        ydata[t, 3] = d[96:128]
        yrda[t] = o_rda.get()
        yrdv[t] = o_rdv.get()
        yvalid[t] = o_valid.get()
        ybaddr[t] = o_baddr.get()
        yempty[t] = o_empty.get()
        yfull[t] = o_full.get()


def architecture(n, form):
    one = [("c_rst", "uint1"), ("c_we", "uint1"), ("c_wa", "UInt(12)"), ("c_wd", "UInt(128)"),
           ("c_pause", "uint1"), ("q_rst", "uint1"), ("q_flush", "uint1"), ("q_raddr", "UInt(12)"),
           ("q_pop", "uint1"), ("k_count", "UInt(3)"), ("k_flush", "uint1"), ("k_target", "UInt(12)"),
           ("k_push", "uint1"), ("o_data", "UInt(128)"), ("o_rda", "UInt(12)"), ("o_rdv", "uint1"),
           ("o_valid", "uint1"), ("o_baddr", "UInt(12)"), ("o_empty", "uint1"), ("o_full", "uint1")]
    ch = [Channel(c, d, "2") for c, d in one] + [Channel("fifo", "UInt(128)", str(DEPTH)),
                                                  Channel("ftag", "UInt(13)", str(DEPTH))]
    mems = [Memory(nm, dt) for nm, dt in (
        ("RST", "uint1[N]"), ("WE", "uint1[N]"), ("WA", "UInt(12)[N]"), ("WD", "UInt(32)[N, 4]"),
        ("PAUSE", "uint1[N]"), ("FLUSH", "uint1[N]"), ("RADDR", "UInt(12)[N]"), ("POP", "uint1[N]"),
        ("DATA", "UInt(32)[N, 4]"), ("RDA", "UInt(12)[N]"), ("RDV", "uint1[N]"), ("VALID", "uint1[N]"),
        ("BADDR", "UInt(12)[N]"), ("EMPTY", "uint1[N]"), ("FULL", "uint1[N]"))]
    q = {"drain": queue_drain, "epoch": queue_epoch}[form]
    return Architecture(name=f"fetch_f2_{form}", parameters={"N": n, "DEPTH": DEPTH}, memories=tuple(mems),
                        channels=tuple(ch), units=(src, fetcher, q, sink))


# ---------------------------------------------------------------------------
# the probe: the RTL against each form, judged on the fetch contract
# ---------------------------------------------------------------------------

def closed_loop_trace(seed=0, n_words=64, n=1500):
    """Issue pops every cycle (II=1: a bundle issues whenever one is offered)
    and branches now and then: the shape of a looping program."""
    from examples.minitpu.harness.traces import Trace, rng_for, word
    from examples.minitpu.units.fetch import _ports

    rng = rng_for("flush-probe", seed)
    t = Trace(_ports("iram"))
    t.idle(2, rst_n=0)
    for a in range(n_words):
        t.cycle(instr_write_en=1, iram_addr=a, dma_iram_din=word(rng, 128), pause=1)
    t.cycle(pc_flush=1, restart_addr=0, pause=1)
    gap = 0
    for _ in range(n):
        gap += 1
        kw = {"bundle_pop": 1}
        if gap >= rng.choice([5, 8, 12, 20, 25]):
            kw.update(pc_flush=1, restart_addr=rng.randrange(n_words))
            gap = 0
        t.cycle(**kw)
    return t.cmd()


def contract(out, cmd):
    """Issued bundles (valid & pop), and flush -> target-at-head latencies."""
    n = len(cmd["rst_n"])
    issued = [(t, int(out["bundle_addr"][t])) for t in range(n)
              if cmd["rst_n"][t] and cmd["bundle_pop"][t] and out["bundle_valid"][t]]
    lat = []
    for e in range(n):
        if cmd["rst_n"][e] and cmd["pc_flush"][e]:
            tgt = cmd["restart_addr"][e]
            for k in range(e + 1, min(n, e + 12)):
                if cmd["pc_flush"][k]:
                    break
                if out["bundle_valid"][k] and out["bundle_addr"][k] == tgt:
                    lat.append(k - e)
                    break
    return issued, lat


def _parts():
    from examples.minitpu.harness.traces import concat
    from examples.minitpu.units import fetch

    return {"program": concat(*[c for _, c, _ in fetch.traces("iram")]),
            "closed-loop": concat(closed_loop_trace(0), closed_loop_trace(1))}


def _rtl(cmd):
    from examples.minitpu.harness import rtl
    from examples.minitpu.units import fetch

    unit = fetch.INSTANCES["iram"]
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    r = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = fetch.REF("iram", packed)
    return {p: rtl.unpack(r[p]) for p in want}, reason


def run_case(label, form, backend, project):
    """One (trace, form, backend) in this process: a verdict line."""
    import allo.dataflow as df

    from examples.minitpu.units import fetch

    cmd = _parts()[label]
    n = len(cmd["rst_n"])
    rtl_out, reason = _rtl(cmd)
    ri, _ = contract(rtl_out, cmd)
    t0 = time.time()
    top = architecture(n, form).region("simulator")
    mod = (df.build(top, target="simulator") if backend == "simulator" else
           df.build(top, target="systemc", mode="csim", project=os.path.join(project, f"{label}_{form}")))
    got = fetch.run_f1(mod, cmd, n, 128)
    got = {p: [int(x) for x in got[p]] for p in rtl_out}
    # per iteration, the slots the queue defines: every control output, and the
    # head while it is valid (a stream has no "head while empty")
    tot = bad = 0
    per = {}
    first = {}
    for p in rtl_out:
        for i in range(n):
            if reason[p][i] or (p in ("bundle_data", "bundle_addr") and not rtl_out["bundle_valid"][i]):
                continue
            tot += 1
            if got[p][i] != rtl_out[p][i]:
                bad += 1
                per[p] = per.get(p, 0) + 1
                first.setdefault(p, i)
    gi, gl = contract(got, cmd)
    same_order = [x for _, x in gi] == [x for _, x in ri][:len(gi)]
    tag = "CONTRACT-MATCH" if bad == 0 else "CONTRACT-DIFF "
    print(f"{tag} fetch f2_{form} {backend} {label}: {tot - bad}/{tot} defined slots; "
          f"{len(gi)}/{len(ri)} bundles issued (order {'= RTL' if same_order else 'DIFFERS'}); "
          f"flush->target {dict(sorted({x: gl.count(x) for x in gl}.items()))} "
          f"({time.time() - t0:.1f}s)" + (f"; differing {per} first at {first}" if per else ""), flush=True)


def main(argv=None):
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", choices=("simulator", "systemc"))
    ap.add_argument("--project", default="/tmp/u4_flush_prj")
    ap.add_argument("--case", nargs=3, metavar=("TRACE", "FORM", "BACKEND"))
    a = ap.parse_args(argv)
    if a.case:
        run_case(*a.case, a.project)
        return 0
    for label, cmd in _parts().items():
        rtl_out, _ = _rtl(cmd)
        ri, rl = contract(rtl_out, cmd)
        print(f"RTL fetch:iram {label}: {len(cmd['rst_n'])} cycles, {len(ri)} bundles issued, flush->target "
              f"{dict(sorted({x: rl.count(x) for x in rl}.items()))}", flush=True)
        for backend in a.backend or ["simulator"]:
            for form in ("drain", "epoch"):
                # one process per build: the simulator's OpenMP teardown can corrupt
                # the heap at exit (pitfalls.rst), which hangs a second build
                r = subprocess.run([sys.executable, "-m", "examples.minitpu.template.flush_stream", "--case",
                                    label, form, backend, "--project", a.project],
                                   capture_output=True, text=True, timeout=1800)
                lines = [x for x in r.stdout.splitlines() if x.startswith("CONTRACT-")]
                print(lines[0] if lines else f"CONTRACT-FAIL  fetch f2_{form} {backend} {label}: "
                      f"exit {r.returncode} {r.stderr.strip().splitlines()[-1:] }", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

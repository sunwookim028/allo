# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Runnable companion to docs/DATAFLOW_LINKS.md.

Two self-contained dataflow designs, each with a golden check:

  1. link_types   -- Stream + Channel side by side (both ORDERED links, so the
                     result is deterministic and verifies bit-exact). No Wire:
                     a standalone Wire has no synchronisation and reads garbage.

  2. wire_sideband -- the ONE idiom in which a Wire is sound: it rides alongside
                     an ordered Stream. The Stream's blocking put/get supplies the
                     barrier; the Wire carries a derived tag for free, letting the
                     FIFO be narrower than a packed word carrying both.

Usage:
    conda activate allo
    export OMP_NUM_THREADS=8

    python docs/dataflow_links_examples.py            # csim both (default)
    python docs/dataflow_links_examples.py link       # csim design 1 only
    python docs/dataflow_links_examples.py wire        # csim design 2 only

csim uses the SystemC backend (g++ host simulation). The JIT "simulator" target has
no Wire support, so design 2 is verified via csim.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys
import numpy as np
import allo
from allo.ir.types import int32, UInt, Stream, Channel, Wire, valid_ready
import allo.dataflow as df


# =====================================================================================
# 1. link_types -- Stream + Channel, both ordered. producer writes each value to BOTH
#    links; consumer sums them back. Result = 2*A, deterministic. (Wire removed: on its
#    own it reads garbage -- see design 2 for the sound Wire idiom.)
# =====================================================================================
M, N = 2, 2


@df.region()
def link_types(A: int32[M, N], B: int32[M, N]):
    fifo: Stream[int32, 4]              # buffered FIFO, depth 4
    chan: Channel[int32, valid_ready]   # handshake, no buffer

    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[M, N]):
        for i, j in allo.grid(M, N):
            fifo.put(a[i, j])
            chan.put(a[i, j])

    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[M, N]):
        for i, j in allo.grid(M, N):
            x: int32 = fifo.get()
            y: int32 = chan.get()
            b[i, j] = x + y


def run_link_types(prj):
    a = np.arange(M * N, dtype=np.int32).reshape(M, N)
    b = np.zeros((M, N), dtype=np.int32)
    want = a + a
    mod = df.build(link_types, target="systemc", mode="csim", project=prj)
    mod(a, b)
    ok = bool((b == want).all())
    print(f"  link_types   Stream+Channel   B={b.ravel()}  "
          f"expected={want.ravel()}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# =====================================================================================
# 2. wire_sideband -- a Wire used correctly: a narrow FIFO carries the payload, a Wire
#    carries a derived tag. The FIFO push/get is the barrier that keeps the wire aligned.
#      gen:  side.put(tg)   THEN  link.put(d)      -> drive wire FIRST, then push
#      proc: d = link.get() THEN  tg = side.get()  -> block on FIFO, THEN read wire
#    Swap either pair and it is racy. (docs/noc/FINDINGS_wire_channel.md has the analysis.)
# =====================================================================================
DW = 16   # payload width (buffered)
TW = 8    # tag width (derived metadata, needed only at the same instant)
NN = 8    # elements pushed through


@df.region()
def wire_sideband(A: int32[NN], C: int32[NN]):
    link: Stream[UInt(DW), 2]             # 2 slots x 16 bits = 32 bits stored
    side: Wire[UInt(TW)]                  # combinational: no depth, no handshake

    @df.kernel(mapping=[1], args=[A])
    def gen(a: int32[NN]):
        for i in range(NN):
            d: UInt(DW) = a[i] & 0xFFFF
            tg: UInt(TW) = a[i] & 3
            side.put(tg)                  # (1) drive the wire FIRST ...
            link.put(d)                   # (2) ... THEN push. The push is the barrier.

    @df.kernel(mapping=[1], args=[C])
    def proc(c: int32[NN]):
        for i in range(NN):
            # A link yields its own UInt type; store through explicit int32 temps.
            dw: UInt(DW) = link.get()     # (3) BLOCKS until gen pushed
            tw: UInt(TW) = side.get()     # (4) safe: wire driven before that push
            d: int32 = dw
            tg: int32 = tw
            c[i] = d * (tg + 1)


def run_wire_sideband(prj):
    a = np.arange(NN, dtype=np.int32)
    c = np.zeros(NN, dtype=np.int32)
    d = (a & 0xFFFF).astype(np.int64)
    tg = (a & 3).astype(np.int64)
    want = (d * (tg + 1)).astype(np.int32)
    mod = df.build(wire_sideband, target="systemc", mode="csim", project=prj)
    mod(a, c)
    ok = bool((c == want).all())
    print(f"  wire_sideband Stream+Wire      C={c}  "
          f"expected={want}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "dataflow_links_out")
    os.makedirs(out, exist_ok=True)

    results = {}
    if what in ("all", "link"):
        results["link_types"] = run_link_types(os.path.join(out, "link_types"))
    if what in ("all", "wire"):
        results["wire_sideband"] = run_wire_sideband(os.path.join(out, "wire_sideband"))

    raise SystemExit(0 if results and all(results.values()) else 1)

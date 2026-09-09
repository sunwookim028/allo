"""Directly measure the executor's per-instruction dispatch cost.

Streams of K identical instructions with no dependency bits and a fixed, known
amount of real work each; the slope of cycles vs K is what one instruction costs
to dequeue and dispatch, independent of what it computes.

Two opcodes at opposite ends of the work scale:
  loadw  -- 4 cycles of work (latch a DIM x DIM tile)
  mm     -- 2*rows + SKEW cycles (stream a panel through the array)
If the slopes agree, the cost is dispatch and not the unit.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.accelerator.tinytpu import microarch_v2 as m2  # noqa: E402
from allo.backend.rtl.sim import shell as _shell  # noqa: E402

_o = _shell._write_sources
_shell._write_sources = lambda *a, **k: (lambda r: (r[0], [*r[1], "-Wno-UNOPTFLAT"]))(_o(*a, **k))

IW = m2.IWIDTH
_rtl = None


def rtl():
    global _rtl
    if _rtl is None:
        _rtl = m2.export_backend("rtl")
    return _rtl


def measure(op, k, rows=4):
    prog = []
    for _ in range(k):
        if op == m2.OP_LOADW:
            prog += [op, m2.GUARD, 0, 0, 0]
        else:
            prog += [op, m2.GUARD, 0, rows, 0]
    imem = np.zeros(m2.IMEM_SIZE, np.int32)
    imem[:len(prog)] = prog
    dmem = np.zeros(m2.DRAM_SIZE, np.float32)
    return rtl().cosim(dmem, imem, k, 0, k, 0).cycles


if __name__ == "__main__":
    # Separate the per-instruction fixed cost from the per-row cost: measure
    # `mm` at several `rows` and fit. A pipelined unit costs
    #     rows * slope + fixed
    # where `fixed` is the array pipeline's fill/drain plus the dequeue and
    # dispatch -- everything that does not scale with the work in the
    # instruction. That fixed part is what a longer stream amortises, and what
    # a short one pays in full.
    print("mm: cost per instruction vs rows streamed")
    prev = None
    for rows in (4, 8, 16):
        c = (measure(m2.OP_MM0, 16, rows) - measure(m2.OP_MM0, 8, rows)) / 8
        if prev:
            pr, pc = prev
            print(f"  rows={pr:2d} -> {rows:2d}   {pc:5.1f} -> {c:5.1f} cyc/instr"
                  f"   slope {(c - pc) / (rows - pr):4.2f} cyc/row")
        else:
            print(f"  rows={rows:2d}          {c:5.1f} cyc/instr")
        prev = (rows, c)
    r0, c0 = 4, (measure(m2.OP_MM0, 16, 4) - measure(m2.OP_MM0, 8, 4)) / 8
    r1, c1 = 16, (measure(m2.OP_MM0, 16, 16) - measure(m2.OP_MM0, 8, 16)) / 8
    sl = (c1 - c0) / (r1 - r0)
    print(f"  fit: {sl:.2f} cyc/row + {c0 - sl * r0:.1f} fixed per instruction\n")

    for name, op, w in (("loadw", m2.OP_LOADW, 4),):
        pts = [(k, measure(op, k)) for k in (4, 8, 16, 32)]
        for a, b in zip(pts, pts[1:]):
            print(f"  {name:11s} K={a[0]:3d}->{b[0]:3d}  {a[1]:5d}->{b[1]:5d}  "
                  f"slope {(b[1] - a[1]) / (b[0] - a[0]):5.1f} cyc/instr "
                  f"(work/instr = {w})")
        k, c = pts[-1]
        print(f"  {name:11s} => dispatch = slope - work = "
              f"{(pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0]) - w:.1f} cyc/instr\n")

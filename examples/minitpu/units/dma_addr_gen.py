# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``dma_addr_gen``, the DMA's row address generator (combinational).

``src/core/dma/dma_addr_gen.sv``: ``word_addr = base_addr + {18'h0, row} *
stride``, truncated to 32 bits; ``dma.sv`` keeps the low ``DRAM_BEAT_ADDR_W``
(29) bits as ``dm_req_addr``. ``row`` is 14 bits, which must equal
``dma.sv``'s ``ROW_BITS`` and ``sequencer_pkg::DESC_BEAT_ROWS_W`` (a
narrower port truncates the row count silently, the file's own comment).

All ports are at most 32 bits, so the harness's ``comb`` shape drives it as it
is: no wrapper (and no dummy clock) is needed. The "IEEE" reference here is
the exact, unbounded ``base + row * stride``; the RTL differs from it exactly
where the 32-bit sum wraps, which is the one deviation class.
"""

import numpy as np

from examples.minitpu.harness import rtl
from examples.minitpu.harness.ref_ctrl_dma import addr_gen

RTL = rtl.RtlUnit(
    top="dma_addr_gen",
    sources=["src/core/dma/dma_addr_gen.sv"],
    inputs=[("base_addr", 32), ("row", 14), ("stride", 32)],
    outputs=[("word_addr", 32)],
    shape="comb",
    latency=0,
)
LATENCY_SOURCE = "combinational (one assign, no clock port)"
PROBE = None
M32 = (1 << 32) - 1


def REF(base, row, stride):
    return np.array([addr_gen(b, r, s) for b, r, s in zip(base, row, stride)], dtype=np.int64)


def IEEE(base, row, stride):
    """Exact ``base + row * stride`` (< 2**47, fits int64)."""
    return base.astype(np.int64) + (row.astype(np.int64) & 0x3FFF) * stride.astype(np.int64)


DEVIATIONS = [("32-bit wrap (base + row*stride >= 2**32)",
               lambda s, ieee, got: (int(ieee) & M32) == int(got))]


def stimulus():
    """Edges crossed (stride 0, 1, 4, 2**k, max; row 0, 1, max; base 0,
    near-wrap, max) plus 200k random, a third with small strides."""
    rng = np.random.default_rng(4)
    bases = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFF00, M32, 0x1000]
    rows = [0, 1, 2, 127, 128, 255, 256, 4095, 8191, 16383]
    strides = [0, 1, 2, 4, 128, 0x10000, 0x40000, 0x7FFFF, M32, 0x80000000]
    edge = np.array([(b, r, s) for b in bases for r in rows for s in strides], dtype=np.uint64)
    n = 200_000
    rb = rng.integers(0, 1 << 32, n, dtype=np.uint64)
    rr = rng.integers(0, 1 << 14, n, dtype=np.uint64)
    rs = rng.integers(0, 1 << 32, n, dtype=np.uint64)
    small = rng.random(n) < 0.33
    rs[small] = rng.integers(0, 1 << 10, int(small.sum()), dtype=np.uint64)
    return np.concatenate([edge, np.stack([rb, rr, rs], axis=1)])


# ---------------------------------------------------------------------------
# Allo variants (U4 track C, ``dev/records/minitpu/u4_track_c_2026-10-08.rst``).
# ``make(n)`` returns a region over ``n`` vectors; the runner takes the built
# module and the ``uint64[n, 3]`` stimulus and returns ``uint32[n]``.
# ---------------------------------------------------------------------------

import allo.dataflow as df  # noqa: E402
from allo.ir.types import UInt, uint32  # noqa: E402

ROW_BITS = 14  # dma.sv ROW_BITS == sequencer_pkg::DESC_BEAT_ROWS_W (dma_desc_adapter.LEGALITY)


def addr(base: UInt(32), row: UInt(14), stride: UInt(32)) -> UInt(32):
    """``dma_addr_gen.sv`` as a function (plan C1): the DMA's issue view calls
    it, as ``dma.sv`` instantiates the module. The product is formed at 46
    bits (14 + 32) and the sum truncated to 32 on return, as the RTL's
    32-bit ``assign`` truncates."""
    p: UInt(46) = row * stride
    w: UInt(32) = base + p
    return w


def bits(n):
    """C1: one kernel calling ``addr`` per vector; region ports are ``uint32``
    (row carried in 32 bits, masked to ``ROW_BITS`` by the narrowing)."""

    @df.region()
    def top(B: uint32[n], R: uint32[n], S: uint32[n], W: uint32[n]):
        @df.kernel(mapping=[1], args=[B, R, S, W])
        def agu(b: uint32[n], r: uint32[n], s: uint32[n], w: uint32[n]):
            for i in range(n):
                bb: UInt(32) = b[i]
                rr: UInt(14) = r[i]
                ss: UInt(32) = s[i]
                w[i] = addr(bb, rr, ss)

    return top


def run_bits(mod, stim_):
    b = np.ascontiguousarray(stim_[:, 0]).astype(np.uint32)
    r = np.ascontiguousarray(stim_[:, 1]).astype(np.uint32)
    s = np.ascontiguousarray(stim_[:, 2]).astype(np.uint32)
    w = np.zeros(len(stim_), dtype=np.uint32)
    mod(b, r, s, w)
    return w


VARIANTS = {"bits": (bits, run_bits)}

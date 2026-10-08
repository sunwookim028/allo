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

import allo.dataflow as df
from allo.ir.types import UInt

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
# Allo (U4 track A, plan C1): the address generator as a plain Allo function,
# the form the DMA unit (track C) calls; held alone through a one-kernel region.
# ---------------------------------------------------------------------------

U32, U14 = UInt(32), UInt(14)


def addr_gen(base_addr: UInt(32), row: UInt(14), stride: UInt(32)) -> UInt(32):
    """``dma_addr_gen.sv``: ``base_addr + {18'h0, row} * stride``, kept to 32 bits."""
    r: UInt(32) = row
    prod: UInt(32) = r * stride
    word_addr: UInt(32) = base_addr + prod
    return word_addr


def c1(n):
    @df.region()
    def top(B: U32[n], R: U14[n], S: U32[n], A: U32[n]):
        @df.kernel(mapping=[1], args=[B, R, S, A])
        def gen(b: U32[n], r: U14[n], s: U32[n], a: U32[n]):
            for i in range(n):
                a[i] = addr_gen(b[i], r[i], s[i])

    return top


def run_c1(mod, stim):
    b = np.ascontiguousarray(stim[:, 0]).astype(np.uint32)
    r = np.ascontiguousarray(stim[:, 1]).astype(np.uint16)
    s = np.ascontiguousarray(stim[:, 2]).astype(np.uint32)
    a = np.zeros(len(stim), dtype=np.uint32)
    mod(b, r, s, a)
    return a


VARIANTS = {"c1": (c1, run_c1)}

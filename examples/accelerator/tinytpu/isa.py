# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cornell TPU ISA semantics, authored against the DSA frontend.

A faithful model of the MODE-based Cornell TPU described in ``todos/isa.md``,
minus the TMA and HALT control instructions. The memory hierarchy is off-chip
``dram`` (where program I/O lives) feeding a 1-D fp32 on-chip word scratchpad
(``bram``) plus an 8-register vector file (``vreg``, 8 lanes x fp32 each). The
instruction set covers:

- the DRAM <-> BRAM block movers (``dma_load`` / ``dma_store``), and
- the VPU vector load / store (``vload`` / ``vstore``, BRAM <-> VREG), and
- the VPU 8-lane SIMD compute (``vadd`` / ``vsub`` / ``vmul`` / ``vrelu`` /
  ``vneg``), and
- the 4x4 systolic matmul ``Z = X @ W^T`` (``matmul``).

The systolic array consumes the weight matrix *transposed*. That is a value
reordering, so it lives in the compute region as ``primitive.transpose`` (semantics),
not in the access patterns -- access stays a value-transparent view. Each
instruction object is a module-level name so it can be called bare inside an
``@tpu.oracle`` body.

This file defines only the ISA *semantics* (the address/compute regions the
compiler backend selects against). The synthesizable *microarchitecture* -- the
``@tpu.unit`` hardware modules and the ``@tpu.entry`` fetch-decode-dispatch top
that actually run on an FPGA -- lives in ``microarch.py``.
"""

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA, scratch
from allo.exp.dsa.errors import ShapeError
from allo.lang.core import f32

tpu = ISA("CornellTPU")

DRAM_SIZE = 65536  # off-chip DRAM words (program I/O lives here)
BRAM_SIZE = 8192  # on-chip scratchpad words
IMEM_SIZE = 4096  # instruction-stream words (IWIDTH words per instruction)
VEC_LANES = 8  # SIMD lanes / vector width
VEC_REGS = 8  # vector register slots

dram = tpu.global_("dram", shape=(DRAM_SIZE,), dtype=f32)  # off-chip I/O pool
bram = tpu.scalar("bram", slots=BRAM_SIZE, dtype=f32)  # on-chip scratchpad
vreg = tpu.vector(
    "vreg", slots=VEC_REGS, shape=(VEC_LANES,), dtype=f32
)  # V0..V7, 8 lanes each


# --- DRAM <-> BRAM block movers. A parametric ``n``-word copy (one identity move
#     instruction handles any block size: an 8-word vector or a 16-word matmul
#     tile), so program I/O staged in ``dram`` is brought on-chip and back. ---
@tpu.instruction(src=dram, dst=bram)
def dma_load(I):
    @I.access
    def _(s, d, n):
        return (contiguous(dram, s, n), contiguous(bram, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@tpu.instruction(src=bram, dst=dram)
def dma_store(I):
    @I.access
    def _(s, d, n):
        return (contiguous(bram, s, n), contiguous(dram, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- VPU type 1 / 2: vector load & store (BRAM <-> VREG, 8 x fp32) ---
@tpu.instruction(src=bram, dst=vreg)
def vload(I):
    @I.access
    def _(s, d):
        return (contiguous(bram, s, 8), contiguous(vreg, d, 1))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@tpu.instruction(src=vreg, dst=bram)
def vstore(I):
    @I.access
    def _(s, d):
        return (contiguous(vreg, s, 1), contiguous(bram, d, 8))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- VPU type 3: 8-lane SIMD compute (VREG x VREG -> VREG) ---
def _vcompute2(I):
    @I.access
    def _(a, b, d):
        return (contiguous(vreg, a, 1), contiguous(vreg, b, 1), contiguous(vreg, d, 1))


@tpu.instruction(src=[vreg, vreg], dst=vreg)
def vadd(I):
    _vcompute2(I)

    @I.compute
    def _(a, b, d):
        return primitive.add(a, b)


@tpu.instruction(src=[vreg, vreg], dst=vreg)
def vsub(I):
    _vcompute2(I)

    @I.compute
    def _(a, b, d):
        return primitive.sub(a, b)


@tpu.instruction(src=[vreg, vreg], dst=vreg)
def vmul(I):
    _vcompute2(I)

    @I.compute
    def _(a, b, d):
        return primitive.mul(a, b)


@tpu.instruction(src=vreg, dst=vreg)
def vrelu(I):
    @I.access
    def _(a, d):
        return (contiguous(vreg, a, 1), contiguous(vreg, d, 1))

    @I.compute
    def _(a, d):
        return primitive.relu(a)


@tpu.instruction(src=vreg, dst=vreg)
def vneg(I):
    """Lane-wise negation.  This matches the TOSA ``negate`` primitive."""
    @I.access
    def _(a, d):
        return (contiguous(vreg, a, 1), contiguous(vreg, d, 1))

    @I.compute
    def _(a, d):
        return primitive.negate(a)

@tpu.instruction(src=vreg, dst=vreg)
def vabs(I):
    @I.access
    def _(a, d):
        return (contiguous(vreg, a, 1), contiguous(vreg, d, 1))

    @I.compute
    def _(a, d):
        return primitive.abs(a)


# --- Systolic (MODE 1): Z = X @ W^T over fixed 4x4 tiles. ADDR_A=W, ADDR_B=X,
#     ADDR_OUT=Z (the assembler order `matmul <w>, <x>, <z>`). The 1-D bram words
#     are expanded to a batched 1x4x4 tile (matching TOSA's batched matmul); the
#     weight transpose lives in the compute region (primitive.transpose), so the access
#     patterns are plain value-transparent views. ---
MATMUL_TILE = 4


@tpu.instruction(src=[bram, bram], dst=bram)
def matmul(I):
    @I.access
    def _(w, x, z):
        return (
            view(bram, w, (1, 4, 4)),
            view(bram, x, (1, 4, 4)),
            view(bram, z, (1, 4, 4)),
        )

    @I.compute
    def _(w, x, z):
        return primitive.matmul(x, primitive.transpose(w, [0, 2, 1]))


# --- Layer lowering: C[M,N] = A[M,K] @ B[K,N] over the fixed 4x4 systolic
# array. This is a compiler macro in the Allo ISA spec: ACT expands it into
# DMA, systolic, and accumulation instructions, then allocates every
# ``scratch`` tile. B is physically transposed while it is loaded because the
# systolic array consumes W = B^T.
@tpu.instruction(
    src=[dram, dram],
    dst=dram,
    cost=lambda M, K, N: (M // MATMUL_TILE)
    * (N // MATMUL_TILE)
    * ((K // MATMUL_TILE) * 21 + max(K // MATMUL_TILE - 1, 0) * 8 + 4),
)
def gemm(I):
    """Whole standard GEMM, lowered to 4x4 Cornell-TPU instructions.

    The source and result use standard TOSA layout: A[M,K], B[K,N], C[M,N].
    Divisible shapes are intentional for this first contract; tails require a
    separately specified padding policy rather than silently reading past a tile.
    """

    @I.access
    def _(a, b, c, M, K, N):
        return (
            view(dram, a, (1, M, K)),
            view(dram, b, (1, K, N)),
            view(dram, c, (1, M, N)),
        )

    @I.compute
    def _(a, b, c):
        return primitive.matmul(a, b)

    @I.expand
    def _(a, b, c, M, K, N):
        for extent, name in ((M, "M"), (K, "K"), (N, "N")):
            if extent % MATMUL_TILE:
                raise ShapeError(
                    f"gemm: {name}={extent} is not divisible by the "
                    f"{MATMUL_TILE}x{MATMUL_TILE} systolic tile"
                )

        x_tile, w_tile, z_tile, partial_tile = (
            scratch((MATMUL_TILE, MATMUL_TILE)) for _ in range(4)
        )
        z_lo, z_hi, partial_lo, partial_hi = (
            scratch((VEC_LANES,)) for _ in range(4)
        )

        for m in range(0, M, MATMUL_TILE):
            for n in range(0, N, MATMUL_TILE):
                for k in range(0, K, MATMUL_TILE):
                    # A's tile rows are contiguous in the host layout.
                    for row in range(MATMUL_TILE):
                        dma_load(
                            s=a + (m + row) * K + k,
                            d=x_tile + row * MATMUL_TILE,
                            n=MATMUL_TILE,
                        )
                    # Transpose B[K,N] while staging it as W[N,K].  Scalar DMA
                    # is deliberate: it is the existing ISA's general transpose
                    # primitive, not a hidden compiler-side data reorder.
                    for row in range(MATMUL_TILE):
                        for col in range(MATMUL_TILE):
                            dma_load(
                                s=b + (k + row) * N + n + col,
                                d=w_tile + col * MATMUL_TILE + row,
                                n=1,
                            )
                    if k == 0:
                        matmul(w=w_tile, x=x_tile, z=z_tile)
                    else:
                        matmul(w=w_tile, x=x_tile, z=partial_tile)
                        # Two 8-lane VPU additions accumulate the 16-word tile.
                        vload(s=z_tile, d=z_lo)
                        vload(s=partial_tile, d=partial_lo)
                        vadd(a=z_lo, b=partial_lo, d=z_lo)
                        vstore(s=z_lo, d=z_tile)
                        vload(s=z_tile + VEC_LANES, d=z_hi)
                        vload(s=partial_tile + VEC_LANES, d=partial_hi)
                        vadd(a=z_hi, b=partial_hi, d=z_hi)
                        vstore(s=z_hi, d=z_tile + VEC_LANES)
                # C's tile rows are likewise scattered with existing linear DMA.
                for row in range(MATMUL_TILE):
                    dma_store(
                        s=z_tile + row * MATMUL_TILE,
                        d=c + (m + row) * N + n,
                        n=MATMUL_TILE,
                    )

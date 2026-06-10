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
- the VPU 8-lane SIMD compute (``vadd`` / ``vsub`` / ``vmul`` / ``vrelu``), and
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
from allo.exp.dsa.core import ISA
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


# --- Systolic (MODE 1): Z = X @ W^T over fixed 4x4 tiles. ADDR_A=W, ADDR_B=X,
#     ADDR_OUT=Z (the assembler order `matmul <w>, <x>, <z>`). The 1-D bram words
#     are expanded to a batched 1x4x4 tile (matching TOSA's batched matmul); the
#     weight transpose lives in the compute region (primitive.transpose), so the access
#     patterns are plain value-transparent views. ---
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

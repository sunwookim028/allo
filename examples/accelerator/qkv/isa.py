# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""QKV accelerator ISA semantics, ported to the DSA frontend.

A model of the ACT-paper QKV attention accelerator (originally written in TAIDL).
The memory hierarchy is an off-chip ``d0`` I/O pool feeding two on-chip bf16
scratchpads: ``d1`` (128 rows x 64, the GEMM operand staging) and ``d2`` (64 x 64,
the GEMM output / softmax buffer). The instruction set covers:

- row/column-major loads ``load_rm`` / ``load_cm`` (d0 -> d1) and stores
  ``store_rm`` / ``store_cm`` (d1 -> d0); the *_cm forms transpose, and
- ``mov`` (d2 -> d1 copy), the 64x64 ``gemm`` (d1 x d1 -> d2, plain ``A @ B``), and
- row ``softmax`` (d2 -> d2, exp / sum-along-columns).

Two deliberate deviations from the original TAIDL spec (functional model, not a
bit-faithful repro):

- **bitcast bypassed.** The original loads/stores reinterpret raw HBM bytes
  (``u8[n,64,2]``) as ``bf16[n,64]`` via ``bitcast_convert``. We have no bitcast
  prim, and the attention program is bf16 end-to-end and never bitcasts, so ``d0``
  is modeled directly as bf16. The loads/stores then reduce to plain value moves
  (``*_rm`` = identity, ``*_cm`` = ``transpose``), and we assign our own addresses.
- **column forms are square (64x64).** A transpose maps ``[n,64] -> [64,n]``, which
  only tiles into ``d1``'s fixed 64-wide slots when ``n == 64`` (the value the
  program always uses); ``load_cm`` / ``store_cm`` therefore drop the ``n`` param.
  ``load_rm`` / ``store_rm`` stay parametric in ``n`` (they are genuine moves).

Like CornellTPU, this file defines only the ISA *semantics* (the access/compute
regions the backend selects against); each instruction is a module-level name so it
can be called bare inside an ``@oracle`` body.
"""

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA
from allo.lang.core import bf16

qkv = ISA("QKV")

D0_SIZE = 65536  # off-chip I/O pool (bf16 words; Q/K/V/O are marshalled here)
N = 64  # the attention tile dimension (head size / sequence tile)

d0 = qkv.global_("d0", shape=(D0_SIZE,), dtype=bf16)  # off-chip I/O pool
d1 = qkv.vector("d1", slots=128, shape=(N,), dtype=bf16)  # GEMM operand staging
d2 = qkv.vector("d2", slots=64, shape=(N,), dtype=bf16)  # GEMM output / softmax


# --- d0 <-> d1 row-major moves (parametric n-row blocks; plain value copies) ---
@qkv.instruction(src=d0, dst=d1)
def load_rm(I):
    """Load ``n`` rows from the d0 pool into d1 (row-major)."""

    @I.access
    def _(addr_in, addr_out, n):
        return (view(d0, addr_in, (n, N)), contiguous(d1, addr_out, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@qkv.instruction(src=d1, dst=d0)
def store_rm(I):
    """Store ``n`` rows from d1 back to the d0 pool (row-major)."""

    @I.access
    def _(addr_in, addr_out, n):
        return (contiguous(d1, addr_in, n), view(d0, addr_out, (n, N)))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- d0 <-> d1 column-major moves (64x64, transposing). The transpose is carried
#     in batched 3-D form (perms [0,2,1]) -- the canonical form `normalize_source`
#     rewrites torch's 2-D transpose + batch-reshape into, and the same rank the
#     GEMM operands use -- so a source `K.T` feeding a matmul matches `load_cm`. ---
@qkv.instruction(src=d0, dst=d1)
def load_cm(I):
    """Load a 64x64 block from d0 into d1, transposed (column-major)."""

    @I.access
    def _(addr_in, addr_out):
        return (
            view(d0, addr_in, (1, N, N)),
            contiguous(d1, addr_out, N).reshape((1, N, N)),
        )

    @I.compute
    def _(a, o):
        return primitive.transpose(a, [0, 2, 1])


@qkv.instruction(src=d1, dst=d0)
def store_cm(I):
    """Store a 64x64 block from d1 to d0, transposed (column-major)."""

    @I.access
    def _(addr_in, addr_out):
        return (
            contiguous(d1, addr_in, N).reshape((1, N, N)),
            view(d0, addr_out, (1, N, N)),
        )

    @I.compute
    def _(a, o):
        return primitive.transpose(a, [0, 2, 1])


# --- mov: copy n rows d2 -> d1 (stage a GEMM result back as a GEMM operand) ---
@qkv.instruction(src=d2, dst=d1)
def mov(I):
    """Copy ``n`` rows from d2 into d1."""

    @I.access
    def _(addr_in, addr_out, n):
        return (contiguous(d2, addr_in, n), contiguous(d1, addr_out, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- gemm: Z = A @ B over 64x64 tiles (d1 x d1 -> d2). Plain matmul, no transpose
#     (the column-major loads already transpose where needed). The 64-wide d1/d2
#     rows are reshaped to a batched 1x64x64 tile, matching TOSA's batched matmul. ---
@qkv.instruction(src=[d1, d1], dst=d2)
def gemm(I):
    """64x64 matrix multiply ``Z = A @ B`` (operands and result in d1/d2)."""

    @I.access
    def _(addr_1, addr_2, addr_out):
        return (
            contiguous(d1, addr_1, N).reshape((1, N, N)),
            contiguous(d1, addr_2, N).reshape((1, N, N)),
            contiguous(d2, addr_out, N).reshape((1, N, N)),
        )

    @I.compute
    def _(a, b, z):
        return primitive.matmul(a, b)


# --- softmax: row softmax of an n x 64 block in place (d2 -> d2). Naive form
#     exp(x) / sum_cols(exp(x)); the column sum keeps its dim ([n,1]) and broadcasts
#     against [n,64] in the divide (mul by reciprocal). ---
@qkv.instruction(src=d2, dst=d2)
def softmax(I):
    """Row softmax of an ``n`` x 64 block in d2 (normalize along the 64 columns)."""

    @I.access
    def _(addr, n):
        return (contiguous(d2, addr, n), contiguous(d2, addr, n))

    @I.compute
    def _(x, o):
        e = primitive.exp(x)
        s = primitive.reduce_sum(e, axis=1)  # [n, 1]
        return primitive.mul(e, primitive.reciprocal(s))  # [n,64] * [n,1] broadcast

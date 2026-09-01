# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""mininpu ISA 0.5.0 semantics, authored against the DSA frontend.

A model of the 32-bit fixed-length RISC-style ISA in ``todos/tpu.md`` -- the frozen
opcode ISA that supersedes the MODE-based encoding of ``todos/isa.md`` (modeled in
``examples/accelerator/cornell_tpu``). The memory hierarchy is off-chip ``dram``
(where program I/O lives) feeding a 32 KiB word-addressable scratchpad (``vmem``,
8192 fp32 words) plus a 32-entry vector register file (``vr``, 16 lanes x fp32);
the MXU hangs off both. Modeled here:

- the DRAM <-> VMEM DMA block movers (``vmemld`` / ``vmemst``), and
- the vector memory path (``vld`` / ``vst``, VMEM <-> VR) plus the register copy
  ``vmov``, and
- the 16-lane SIMD compute (``vadd`` / ``vsub`` / ``vmul`` / ``vrecip`` / ``vrsqrt`` /
  ``vexp``), and the splatting reductions (``vredsum`` / ``vredmax``, which are
  oracle-only as modeled -- see the note at their definition), and
- the matrix path (``vmatload`` / ``vmatpush`` / ``vmatpop``), and
- ``matmul_layer``, which is **not** an ISA 0.5.0 instruction but a *layer-level
  macro* -- see its own section at the bottom of this file.

**The MXU is modeled as two buffers.** ``vmatload`` / ``vmatpush`` / ``vmatpop`` are
three instructions around one piece of hardware state, and the frontend's contract is
per-instruction (access + compute, no implicit machine state), so the state has to be
named: ``mxu_w`` is the stationary weight tile the array holds, ``mxu_q`` the result
queue a push enqueues into. Each instruction then models exactly its own step -- the
weight load and the result pop are plain moves, and the multiply lives in the push.
Two consequences:

- ``mxu_w`` is sized to exactly one 16x16 tile, so its address is always 0 and the
  access carries no address param -- ``vmatload vmem[s]`` keeps its single operand.
- ``mxu_q`` is *addressable* (``vmatpush``/``vmatpop`` take a queue index ``q``),
  whereas the hardware keeps the push/pop pointers implicitly. Assembly that pops in
  push order is faithful; the index is a modeling artifact.

**Assumptions where 0.5.0 is silent.** The doc fixes neither the MXU tile shape nor
the weight orientation. The array is 16-wide (one ``vr`` = one activation row), so the
stationary tile is taken as 16x16, and the orientation is inherited from the same
machine's legacy systolic (``todos/isa.md``): ``Z = X @ W^T``, i.e. ``W`` is stored
row-major in the PyTorch ``nn.Linear`` ``[out, in]`` order. The transpose is a value
reordering, so it lives in the compute region as ``primitive.transpose`` (semantics),
not in the access patterns -- access stays a value-transparent view.

**Not modeled.**

- Control (``lbegin`` / ``lend`` / ``flush`` / ``halt``): an instruction here is a
  pure address+compute contract over buffers, with no notion of control flow,
  pipeline state, or termination. A hardware loop is a program-level construct, and
  ``flush`` / ``halt`` have no data effect to describe. Note ``lbegin N``'s trip count
  is 0.5.0's *only* immediate, and it is out of scope for that reason rather than for
  want of a way to express one: an ``@I.compute`` extra param (a computational
  attribute, ACT's α) models an integer immediate, and no 0.5.0 **data** instruction
  has one — the spec's "Instruction Encoding Format" section is empty.
- ``vgelu``: **no longer blocked by constants** -- ``primitive.const`` now supplies the
  ``0.5`` / ``1/sqrt 2`` that used to be inexpressible (that is how ``vexp`` above is
  modeled). What blocks it is that 0.5.0 does not say *which* GELU: the exact
  ``0.5x(1 + erf(x/sqrt 2))`` and the tanh approximation are different functions, and
  the matcher would act on whichever we picked. Naming it here is the honest state;
  guessing would be the same class of error as calling ``2^x`` an ``exp``.

Like CornellTPU and QKV, this file defines only the ISA *semantics* (the access /
compute regions the compiler backend selects against); each instruction is a
module-level name so it can be called bare inside an ``@npu.oracle`` body.
"""

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA, scratch
from allo.lang.core import f32

npu = ISA("MiniNPU")

DRAM_SIZE = 65536  # off-chip DRAM words (program I/O lives here)
VMEM_SIZE = 8192  # on-chip scratchpad: 32 KiB / 4 B per fp32 word
VEC_LANES = 16  # SIMD lanes / vector width
VEC_REGS = 32  # vr0..vr31 (32 x 16 x fp32 = 2 KiB)
MXU_DIM = VEC_LANES  # the systolic array is one activation row wide
MXU_TILE = MXU_DIM * MXU_DIM  # words in one stationary weight tile
MXU_DEPTH = MXU_DIM  # modeled depth of the MXU result queue (one tile of rows)

dram = npu.global_("dram", shape=(DRAM_SIZE,), dtype=f32)  # host memory
vmem = npu.scalar("vmem", slots=VMEM_SIZE, dtype=f32)  # on-chip scratchpad
vr = npu.vector("vr", slots=VEC_REGS, shape=(VEC_LANES,), dtype=f32)  # vr0..vr31
mxu_w = npu.scalar("mxu_w", slots=MXU_TILE, dtype=f32)  # MXU stationary weight tile
mxu_q = npu.vector("mxu_q", slots=MXU_DEPTH, shape=(VEC_LANES,), dtype=f32)  # results


# --- DMA memory: DRAM <-> VMEM block movers. A parametric ``n``-word copy, so one
#     identity move handles any block (a 16-word vector, a 256-word weight tile). ---
@npu.instruction(src=dram, dst=vmem)
def vmemld(I):
    """``vmemld vmem[d], dram[s]`` -- DMA ``n`` words from host DRAM into VMEM."""

    @I.access
    def _(d, s, n):
        return (contiguous(dram, s, n), contiguous(vmem, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@npu.instruction(src=vmem, dst=dram)
def vmemst(I):
    """``vmemst dram[d], vmem[s]`` -- DMA ``n`` words from VMEM back to host DRAM."""

    @I.access
    def _(d, s, n):
        return (contiguous(vmem, s, n), contiguous(dram, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- Vector memory: VMEM <-> VR, one 16-lane register per instruction ---
@npu.instruction(src=vmem, dst=vr)
def vld(I):
    """``vld vr[d], s(vmem)`` -- load 16 words from VMEM into a vector register."""

    @I.access
    def _(d, s):
        return (contiguous(vmem, s, VEC_LANES), contiguous(vr, d, 1))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@npu.instruction(src=vr, dst=vmem)
def vst(I):
    """``vst vr[s], d(vmem)`` -- store a vector register into 16 VMEM words."""

    @I.access
    def _(s, d):
        return (contiguous(vr, s, 1), contiguous(vmem, d, VEC_LANES))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@npu.instruction(src=vr, dst=vr)
def vmov(I):
    """``vmov vr[d], vr[s]`` -- vector register copy."""

    @I.access
    def _(d, s):
        return (contiguous(vr, s, 1), contiguous(vr, d, 1))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# --- Vector path: 16-lane SIMD, VR x VR -> VR ---
def _vbinary(I):
    @I.access
    def _(d, a, b):
        return (contiguous(vr, a, 1), contiguous(vr, b, 1), contiguous(vr, d, 1))


@npu.instruction(src=[vr, vr], dst=vr)
def vadd(I):
    """``vadd vr[d], vr[a], vr[b]`` -- element-wise ``d = a + b``."""

    _vbinary(I)

    @I.compute
    def _(a, b, d):
        return primitive.add(a, b)


@npu.instruction(src=[vr, vr], dst=vr)
def vsub(I):
    """``vsub vr[d], vr[a], vr[b]`` -- element-wise ``d = a - b``."""

    _vbinary(I)

    @I.compute
    def _(a, b, d):
        return primitive.sub(a, b)


@npu.instruction(src=[vr, vr], dst=vr)
def vmul(I):
    """``vmul vr[d], vr[a], vr[b]`` -- element-wise ``d = a * b``."""

    _vbinary(I)

    @I.compute
    def _(a, b, d):
        return primitive.mul(a, b)


def _vunary(I):
    @I.access
    def _(d, a):
        return (contiguous(vr, a, 1), contiguous(vr, d, 1))


@npu.instruction(src=vr, dst=vr)
def vrecip(I):
    """``vrecip vr[d], vr[a]`` -- element-wise reciprocal ``d = 1 / a``."""

    _vunary(I)

    @I.compute
    def _(a, d):
        return primitive.reciprocal(a)


@npu.instruction(src=vr, dst=vr)
def vrsqrt(I):
    """``vrsqrt vr[d], vr[a]`` -- element-wise ``d = 1 / sqrt(a)``."""

    _vunary(I)

    @I.compute
    def _(a, d):
        return primitive.rsqrt(a)


@npu.instruction(src=vr, dst=vr)
def vexp(I):
    """``vexp vr[d], vr[a]`` -- element-wise base-2 exponential ``d = 2**a``.

    The base is a literal *of the instruction*, not an operand the program supplies,
    which is what ``primitive.const`` is for. Written as ``pow(2, a)`` rather than as
    an ``exp`` of a scaled argument because that is exactly what the hardware does and
    exactly what torch's TOSA backend emits for ``torch.exp2`` (``tosa.pow`` of a
    ``dense<2.0>`` constant), so the two line up node for node -- no base-e/base-2
    approximation anywhere."""

    _vunary(I)

    @I.compute
    def _(a, d):
        return primitive.pow(primitive.const(2.0), a)


# --- Reductions: reduce the 16 lanes to a scalar, then *splat* it back over all 16
#     lanes (the destination is a full vector register). A reduce yields a [1] tensor,
#     so the splat is an add against a zero of the destination's shape -- the [1] + [16]
#     add broadcasts (TOSA/NumPy rule). That zero used to be spelled `a - a`, which was
#     a lie about the dataflow (it made the reduction read `a` a second time);
#     `primitive.const` says what it is.
#
#     Still true, and *not* fixed by having constants: the splat is the **root** of the
#     pattern, so the catalog indexes these under `add` and a source `reduce_sum` never
#     selects them -- as modeled the reductions are oracle-only (hand-written assembly).
#     A compiled program gets a row sum by riding the MXU against a constant tile
#     instead (see `program.rms_norm`). Making the root the reduce itself needs a
#     *broadcast/splat* prim, which is a different thing from a constant. ---
@npu.instruction(src=vr, dst=vr)
def vredsum(I):
    """``vredsum vr[d], vr[a]`` -- row-sum of the 16 lanes, splatted across ``d``."""

    _vunary(I)

    @I.compute
    def _(a, d):
        zero = primitive.const(0.0, shape=(VEC_LANES,))
        return primitive.add(primitive.reduce_sum(a, axis=0), zero)


@npu.instruction(src=vr, dst=vr)
def vredmax(I):
    """``vredmax vr[d], vr[a]`` -- row-max of the 16 lanes, splatted across ``d``."""

    _vunary(I)

    @I.compute
    def _(a, d):
        zero = primitive.const(0.0, shape=(VEC_LANES,))
        return primitive.add(primitive.reduce_max(a, axis=0), zero)


# --- Matrix path. The MXU holds one stationary 16x16 weight tile and streams
#     activation rows through it: `vmatload` fills the tile (a plain VMEM -> MXU
#     move), `vmatpush` multiplies one 16-lane row into the result queue, `vmatpop`
#     moves one queued row back into a vector register. ---
@npu.instruction(src=vmem, dst=mxu_w)
def vmatload(I):
    """``vmatload vmem[s]`` -- load the 16x16 weight tile at ``s`` into the MXU.

    The MXU holds exactly one tile, so the destination address is fixed at 0 and the
    instruction keeps its single operand."""

    @I.access
    def _(s):
        return (contiguous(vmem, s, MXU_TILE), contiguous(mxu_w, 0, MXU_TILE))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@npu.instruction(src=[vr, mxu_w], dst=mxu_q)
def vmatpush(I):
    """``vmatpush vr[x]`` -- push one activation row into the MXU and enqueue the
    compute ``q = x @ W^T`` against the stationary tile (``q`` = queue slot)."""

    @I.access
    def _(x, q):
        return (
            contiguous(vr, x, 1).reshape((1, 1, MXU_DIM)),
            view(mxu_w, 0, (1, MXU_DIM, MXU_DIM)),
            contiguous(mxu_q, q, 1).reshape((1, 1, MXU_DIM)),
        )

    @I.compute
    def _(x, w, q):
        return primitive.matmul(x, primitive.transpose(w, [0, 2, 1]))


@npu.instruction(src=mxu_q, dst=vr)
def vmatpop(I):
    """``vmatpop vr[d]`` -- pop one MXU result row (queue slot ``q``) into ``vr[d]``."""

    @I.access
    def _(d, q):
        return (contiguous(mxu_q, q, 1), contiguous(vr, d, 1))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# ==========================================================================#
# Layer-level macro: one whole matmul layer, lowered by @I.expand
# ==========================================================================#
#
# `matmul_layer` is NOT an ISA 0.5.0 instruction. It is a *layer-level macro*: its
# `@I.compute` states what a whole `Z[M,16] = X[M,16] @ W^T` layer means, and its
# `@I.expand` defines it as the run of real instructions that performs it. Nothing
# below it is new hardware -- the expansion issues only `vmatload` / `vld` /
# `vmatpush` / `vmatpop` / `vst`.
#
# Why this machine is the right home for `@expand`. The mechanism says "one layer op
# *means* N tile ops", the way a vector instruction means N lane ops -- so it fits
# hardware that is genuinely *fixed-size and repeated*, and not hardware whose
# instructions already carry their own loop (MINISA's `ExecuteStreaming` carries the
# streaming-step count `T` inside the instruction, so a faithful encoding of it needs
# no expansion at all -- just a solved shape param, which the frontend has handled
# since `vmemld(d, s, n)`). The MXU is the fixed-and-repeated case exactly: it holds
# ONE 16x16 stationary tile and consumes ONE activation row per push, so a layer is
# necessarily `vmatload` once + M pushes. The weight is loaded once and amortized
# across the whole layer, which is the reuse a per-row selection cannot express.
#
# Selection stays unambiguous without a cost tie-break: at M == 1 the layer costs
# `1 + 4*1 = 5` against `vmatpush`'s 1, so a single row still compiles to the plain
# three-step MXU sequence; at M > 1 `vmatpush`'s fixed 1-row shape does not fit and
# only the layer does.
#
# The registers it stages through are `scratch` tiles: ordinary values, allocated and
# kept live over the run like anything else. (They used to be hand-reserved register
# numbers, because an expansion ran *after* allocation and so had to pick its own —
# a convention that a program under enough register pressure could collide with.)


@npu.instruction(src=[vmem, vmem], dst=vmem, cost=lambda M: 1 + 4 * M)
def matmul_layer(I):
    """``Z[M,16] = X[M,16] @ W[16,16]^T`` -- a whole layer, staged in VMEM.

    Not a hardware instruction: `@I.expand` lowers it to one `vmatload` plus M
    rounds of `vld` / `vmatpush` / `vmatpop` / `vst`. `cost` is that exact emit
    count, so the tree-DP can compare one layer op against the instructions it
    lowers into."""

    @I.access
    def _(x, w, z, M):
        return (
            view(vmem, x, (1, M, MXU_DIM)),
            view(vmem, w, (1, MXU_DIM, MXU_DIM)),
            view(vmem, z, (1, M, MXU_DIM)),
        )

    @I.compute
    def _(x, w, z):
        return primitive.matmul(x, primitive.transpose(w, [0, 2, 1]))

    @I.expand
    def _(x, w, z, M):
        # `M` is the Stage-2-solved row count; `x`/`w`/`z` are the layer's operands as
        # values, so `x + m * MXU_DIM` is an offset *into* `x` and the allocator
        # supplies its base. The three staging slots are `scratch` tiles -- one vector
        # register each way and one queue slot -- allocated and kept live across the
        # loop like any other value, where they used to be hand-picked register
        # numbers. That the weight load is hoisted out of the row loop is not encoded
        # anywhere -- this loop *is* its definition.
        xr, zr, q = (scratch((1, 1, MXU_DIM)) for _ in range(3))
        vmatload(s=w)
        for m in range(M):
            vld(d=xr, s=x + m * MXU_DIM)
            vmatpush(x=xr, q=q)
            vmatpop(d=zr, q=q)
            vst(s=zr, d=z + m * MXU_DIM)

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FeatherX (FEATHER+) MINISA semantics, authored against the DSA frontend.

A functional model of the MINISA virtual-neuron ISA that drives the FEATHER+
NEST array + BIRRD reduction network, at the granularity ``compile_program``
compiles today. The real instruction set (see the RTL traces under
``drafts/feather_tutorial/allo-feather/instr_trace/``) is eight instructions;
this file models the six that carry a GEMM, one modeled name per real one:

    SetIVNLayout / SetWVNLayout / SetOVNLayout   ->  set_ivn / set_wvn / set_ovn
    ExecuteMapping (first / accumulating K-pass) ->  mm / mac
    Load / Store                                 ->  load_i, load_w / store_o

**The setters are configuration, not compute.** Each is declared with
``fx.configures``: it writes a row of the ``cfg`` register file, and what it
assigns is *installed* — every later ``ExecuteMapping`` runs under it until some
instruction assigns the same field again (the epoch fold, ``epoch.epochs``). Its
one field on the word is the VN's ``order``, a **chosen schedule param**: a mode
picked from a finite domain under an ``@I.schedule`` predicate — ACT's e_theta —
and here the predicate pins the host ordering ``o0``, which is also what every
layer of the reference trace ``trace_m24k48n512_16x16.json`` uses. The params are
named ``ivn_order`` / ``wvn_order`` / ``ovn_order`` because a configuration is
keyed by param name: two registers are two registers exactly when their fields
are named apart. The *latch* (a TILE-word read of the tensor the VN describes,
written into ``cfg``) is the register-file fiction this vocabulary needs — every
instruction reads a source and writes a destination — and the same fiction the
frontend's own staged-machine tests use; it is the word, not the data, that
configures.

**What is deliberately not on the setter's word, and why.** The real
``Set*VNLayout`` also carries its ``L1`` tiling factors (``M_L1``/``J_L1``, ...).
Those are immediates, which this vocabulary calls alpha — and an ``@expand`` body
cannot pass alpha on the instructions it issues (``search._ExpandRecorder``
refuses it). So v0 leaves them off the word: they are derivable from the emitted
tile loop itself, so a MINISA encoder can reconstruct them from the program, or a
later relaxation can let an expansion bake body-computed immediates. Likewise the
``ExecuteMapping`` routing fields (``G_r, G_c, r_0, c_0, s_r, s_c``) are absent:
they steer *which PE computes what* and are result-invariant (the output col-map
compensates any order), so they belong to the encoder, not to the semantics.

**ExecuteMapping is a tile, and K is a run of them.** ``mm`` computes one
``TILE x TILE`` output block; ``mac`` accumulates into it, with the accumulator an
*explicit read operand* so the whole K-reduction chain coalesces onto one block
buffer — the on-chip accumulation the hardware performs across consecutive
``ExecuteMapping``s to the same block. The BIRRD network itself is inside the
instruction: intra-array parallelism is the compute region's, invisible to sigma.

**Load/Store are relayouts.** A ``TILE x TILE`` block of row-major ``dram`` is not
contiguous (its rows are ``DRAM_COLS`` apart); the movers read a rank-2 strided
block on one side and a contiguous run on the other, identity compute — a DMA
rearranges by reading with one pattern and writing with another.

**Not modeled, by choice:** padding (extents must divide exactly — the real
M=24-on-16 trace pads to 77% utilization, and nothing in this vocabulary says a
ragged tile), spatial instances (one NEST), and Activation / quantization.
"""

from allo.exp.dsa import primitive
from allo.exp.dsa.access import strided, view
from allo.exp.dsa.core import ISA, scratch
from allo.lang.core import f32

AH = AW = 4  # NEST array height/width
TILE = AH  # one ExecuteMapping computes one TILE x TILE output block
ORDERS = (0, 1, 2, 3, 4, 5)  # the 3-bit VN dimension ordering, o0..o5
HOST_ORDER = 0  # o0 = row-major, the layout the host hands over
DRAM_ROWS, DRAM_COLS = 128, 64  # the row-major host arena

fx = ISA("FeatherX")

# Row-major host memory, where program I/O lives. Declared with two extents, so a
# rank-2 block access is legal on it — which is what the relayout DMA needs.
dram = fx.hbm("dram", shape=(DRAM_ROWS, DRAM_COLS), dtype=f32, is_global=True)
strb = fx.scalar("strb", slots=256, dtype=f32)  # streaming buffer (IVN operands)
stab = fx.scalar("stab", slots=256, dtype=f32)  # stationary buffer (WVN operands)
ob = fx.scalar("ob", slots=256, dtype=f32)  # output block buffer (OVN)
cfg = fx.scalar("cfg", slots=16, dtype=f32)  # the Set*VNLayout register file


# --- Set*VNLayout: one setter per virtual neuron, each writing its own cfg row.
#     `d` is the register row; an expansion passes a fresh `scratch((TILE,))` so
#     the row is placed by the allocator like every other location. ---
def _latch(I):
    """The setter's regions: a TILE-word sliver of the tensor its VN describes,
    latched into a ``cfg`` row. Identity compute — a register write moves bits."""

    @I.access
    def _(r, c, d):
        return (
            strided(dram, [r, c], [1, TILE], [1, 1]),
            view(cfg, d, TILE),
        )

    @I.compute
    def _(s, d):
        return primitive.identity(s)


@fx.instruction(src=[dram], dst=cfg)
def set_ivn(I):
    """SetIVNLayout: install the streaming-input VN layout (order pinned to o0)."""
    _latch(I)

    @I.schedule(ivn_order=ORDERS)
    def _(ivn_order):
        return ivn_order == HOST_ORDER


@fx.instruction(src=[dram], dst=cfg)
def set_wvn(I):
    """SetWVNLayout: install the stationary-weight VN layout (order pinned to o0)."""
    _latch(I)

    @I.schedule(wvn_order=ORDERS)
    def _(wvn_order):
        return wvn_order == HOST_ORDER


@fx.instruction(src=[dram], dst=cfg)
def set_ovn(I):
    """SetOVNLayout: install the output VN layout (order pinned to o0)."""
    _latch(I)

    @I.schedule(ovn_order=ORDERS)
    def _(ovn_order):
        return ovn_order == HOST_ORDER


for _setter in (set_ivn, set_wvn, set_ovn):
    fx.configures(_setter)


# --- Load: gather the TILE x TILE block at (r, c) of row-major dram into a
#     contiguous tile at `d` on chip — a relayout, one per destination buffer
#     because a mover's endpoints are part of its declaration. ---
def _gather(I, dst):
    @I.access
    def _(r, c, d):
        return (
            strided(dram, [r, c], [TILE, TILE], [1, 1]),
            view(dst, d, (TILE, TILE)),
        )

    @I.compute
    def _(s, d):
        return primitive.identity(s)


@fx.instruction(src=[dram], dst=strb)
def load_i(I):
    """Load (streaming): dram block -> contiguous IVN tile in ``strb``."""
    _gather(I, strb)


@fx.instruction(src=[dram], dst=stab)
def load_w(I):
    """Load (stationary): dram block -> contiguous WVN tile in ``stab``."""
    _gather(I, stab)


@fx.instruction(src=[ob], dst=dram)
def store_o(I):
    """Store: scatter a finished output block back into its dram matrix — the
    inverse relayout, so results return in the layout the host handed over."""

    @I.access
    def _(s, r, c):
        return (
            view(ob, s, (TILE, TILE)),
            strided(dram, [r, c], [TILE, TILE], [1, 1]),
        )

    @I.compute
    def _(s, d):
        return primitive.identity(s)


# --- ExecuteMapping, first K-pass: C_block = A_tile @ B_tile. Operands are
#     batched 1 x TILE x TILE tiles, matching TOSA's batched matmul. ---
@fx.instruction(src=[strb, stab], dst=ob)
def mm(I):
    """ExecuteMapping (first K-pass): write ``A_tile @ B_tile`` into a block."""

    @I.access
    def _(a, b, c):
        return (
            view(strb, a, (1, TILE, TILE)),
            view(stab, b, (1, TILE, TILE)),
            view(ob, c, (1, TILE, TILE)),
        )

    @I.compute
    def _(a, b, c):
        return primitive.matmul(a, b)


# --- ExecuteMapping, accumulating K-pass: C_block += A_tile @ B_tile. The block
#     is an explicit read operand, so the K-reduction chain shares one slot. ---
@fx.instruction(src=[strb, stab, ob], dst=ob)
def mac(I):
    """ExecuteMapping (accumulating K-pass): add ``A_tile @ B_tile`` to a block."""

    @I.access
    def _(a, b, c, d):
        return (
            view(strb, a, (1, TILE, TILE)),
            view(stab, b, (1, TILE, TILE)),
            view(ob, c, (1, TILE, TILE)),  # accumulator in
            view(ob, d, (1, TILE, TILE)),  # accumulator out (same slot)
        )

    @I.compute
    def _(a, b, c, d):
        return primitive.add(c, primitive.matmul(a, b))


# --- The layer op: C[M,N] = A[M,K] @ B[K,N] over whole row-major matrices,
#     lowered by @expand into the MINISA stream — the three setters once, then the
#     Load / ExecuteMapping / Store tile loop. Declared *after* the tile ops so a
#     one-tile layer ties at cost 1 and the bare tile instruction wins: no point
#     configuring a layer that is a single ExecuteMapping. ---
@fx.instruction(
    src=[dram, dram],
    dst=dram,
    cost=lambda M, K, N: (M // TILE) * (K // TILE) * (N // TILE),
)
def gemm(I):
    """One whole GEMM layer, from and to row-major host memory."""

    @I.access
    def _(ar, ac, br, bc, cr, cc, M, K, N):
        return (
            strided(dram, [ar, ac], [M, K], [1, 1]).reshape((1, M, K)),
            strided(dram, [br, bc], [K, N], [1, 1]).reshape((1, K, N)),
            strided(dram, [cr, cc], [M, N], [1, 1]).reshape((1, M, N)),
        )

    @I.compute
    def _(a, b, c):
        return primitive.matmul(a, b)

    @I.expand
    def _(ar, ac, br, bc, cr, cc, M, K, N):
        for extent, name in ((M, "M"), (K, "K"), (N, "N")):
            assert extent % TILE == 0, (
                f"gemm: {name}={extent} is not a multiple of the {TILE}-wide NEST "
                f"— padding is not modeled, so a non-dividing layer has no lowering"
            )
        # The layer's configuration, installed once. set_ovn latches from the
        # input too: the output block does not exist yet, and the latch source is
        # arbitrary by design — the word configures, not the data.
        set_ivn(r=ar, c=ac, d=scratch((TILE,)))
        set_wvn(r=br, c=bc, d=scratch((TILE,)))
        set_ovn(r=ar, c=ac, d=scratch((TILE,)))
        a_t, b_t, c_t = (scratch((TILE, TILE)) for _ in range(3))
        for m in range(0, M, TILE):
            for n in range(0, N, TILE):
                for k in range(0, K, TILE):
                    load_i(r=ar + m, c=ac + k, d=a_t)
                    load_w(r=br + k, c=bc + n, d=b_t)
                    if k == 0:
                        mm(a=a_t, b=b_t, c=c_t)
                    else:
                        mac(a=a_t, b=b_t, c=c_t, d=c_t)
                store_o(s=c_t, r=cr + m, c=cc + n)

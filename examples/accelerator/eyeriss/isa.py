# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eyeriss, authored against the DSA frontend -- a deliberate boundary probe.

This is **route 0** of the schedule-ISA work: write a real schedule-ISA machine in
the *current* language and find out mechanically where it breaks, rather than
guessing. The findings are written up in ``todos/schedule-isa.md``; this file is the
evidence, and it stays the evidence as they are closed one by one -- see "Findings"
below for which are and which are not.

Source: Chen, Emer, Sze, *"Eyeriss: A Spatial Architecture for Energy-Efficient
Dataflow for Convolutional Neural Networks"*, ISCA 2016 (Sec. V), plus the chip
paper (ISSCC'16 / JSSC'17) for the physical numbers. Everything asserted about the
dataflow below is from ISCA'16 Sec. V-B and is quoted where it matters.

**The machine.** A 12x14 = 168-PE spatial array with a 108 KB global buffer, 16-bit
fixed point. Each PE has a ~0.5 KB register file and computes exactly one thing: a
**1-D convolution primitive** -- a row of filter weights against a sliding window of
one ifmap row, accumulating one row of psums (ISCA'16 Fig. 5). That primitive is
fixed in silicon and is the machine's *entire* operation repertoire. There is no
opcode anywhere on this chip.

**Row-stationary, in one paragraph.** The 1-D primitives of one 2-D convolution are
grouped into a **logical PE set** whose "height and width are determined by the
filter height (R) and ofmap height (E)". Inside a set, filter rows are reused
horizontally, ifmap rows diagonally, and psums accumulate vertically (Fig. 6). A
whole CONV layer needs ``N x M x C`` such sets. Since that far exceeds 168 PEs, the
logical array is **folded** onto the physical one in two phases: phase 1 folds sets
onto PEs (bounded by the RF and the array size), phase 2 folds whole *processing
passes* in time (bounded by the global buffer). Both phases "happen statically prior
to runtime, so no on-line computation is required" -- i.e. the fold factors are the
compiler's output, and they are what the chip's per-layer configuration carries.

**So what is an "instruction" here?** A **processing pass**: one loading of the
configuration plus the run it enables. That is the unit modeled below as
``conv_pass``. Eyeriss has no instruction stream to speak of -- the configuration is
scanned in per layer -- so a pass is the largest thing that has a single, static
meaning, and the smallest thing that has any meaning at all.

--------------------------------------------------------------------------
Findings  (details, with the error messages, in todos/schedule-isa.md)
--------------------------------------------------------------------------

**Closed by ``@I.schedule``** (capability ①, built because route 0 found this):

1. **The fold factors now have a home.** A pass is configured by (a) the layer shape
   and (b) how the ``N x M x C`` sets split between space (mapped across the array)
   and time (interleaved on one PE). (a) is solved from the source; (b) is the
   ``spatial`` schedule param below -- freely chosen, constrained by the hardware,
   and *not* something the computed value depends on. Before it there were only two
   parameter kinds, one solved and one that the value depends on, and a fold factor
   is neither.

2. **A pass that does not fit is now rejected, at selection time.** ``pass_fits``
   below is the machine's ``e_theta``: the spatially mapped sets must fit the PE
   array and the temporally folded ones each PE's register file. An instruction with
   no legal configuration is not a candidate, so a layer that cannot be run this way
   fails with a message that says so -- where it used to emit one confident,
   impossible ``conv_pass``.

3. **Phase-2 folding works, and needed no frontend change** (capability ⑥). An
   earlier version declared ``conv_layer``'s operands in ``glb`` and the allocator
   then demanded the whole layer on-chip, which read like a limitation but was a
   modeling error: a layer-level operation's operands are off-chip by construction.
   They live in ``dram`` below, the expansion does the staging, and the global-buffer
   bound is stated in the predicate where it belongs. The passes the expansion issues
   are configured individually, by the compiler.

**Still open:**

4. **The multicast NoC is not modeled at all.** "global multicast NoCs for the
   ifmaps and filters, and a local PE-to-PE NoC for the psums" (Sec. V-E). Which PE
   can receive which datum in which cycle is the reachability relation ``R`` from the
   formal model -- the constraint that couples layout to schedule. A buffer here is a
   flat address space that every instruction can reach.

5. **``@I.expand`` still runs after selection, and can only split the outermost dim.**
   The macro must therefore certify in its own ``@I.schedule`` what it is able to
   lower, and its staging area is reserved by convention rather than allocated.
   Eyeriss's real reuse folds are over M and C, both *inner* dims of NHWC / OHWI, so
   those slices are strided -- and a stride is a Stage-2b residence param solved from
   a neighbour's map, never freely chosen.

6. **No time.** A pass's psums are accumulated by *later* passes; the order matters
   and nothing expresses it beyond ordinary value dependence through ``glb``. There
   is no ``t`` in any map here, so the ``(a_h, a_w, t) -> data`` space-time map that
   is the whole content of a schedule ISA has nowhere to live.

Not modeled, for ordinary reasons: sparsity/RLC compression (a separate paper), the
FC and POOL modes (POOL is "swapping the MAC with a MAX comparison", i.e. a second
fixed kernel), and the bias add (folded into the host program).
"""

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous, view
from allo.exp.dsa.core import ISA, scratch
from allo.lang.core import f32

eyeriss = ISA("Eyeriss")

DRAM_SIZE = 1 << 20  # off-chip pool where program I/O lives
GLB_WORDS = 108 * 1024 // 2  # 108 KB global buffer, 16-bit words
PE_ROWS, PE_COLS = 12, 14  # the physical array: 168 PEs
SPAD_WORDS = 512 // 2  # ~0.5 KB per-PE register file

dram = eyeriss.global_("dram", shape=(DRAM_SIZE,), dtype=f32)
glb = eyeriss.scalar("glb", slots=GLB_WORDS, dtype=f32)
# The PE register file, addressed as (pe_row, pe_col) -> a private spad. Declared
# because it is real state and because its capacity bounds the temporal fold
# (`pass_fits`), but no instruction below can name a *single* PE: the whole point of
# the RS dataflow is that a pass addresses the array collectively, through its fold
# factors rather than by naming addresses in it.
spad = eyeriss.buffer("spad", extents=(PE_ROWS, PE_COLS), dtype=f32, slot=(SPAD_WORDS,))


# --- DRAM <-> GLB block movers. Eyeriss's DMA moves a layer's ifmaps/filters into
#     the global buffer and ofmaps back out; a parametric n-word copy covers both. ---
@eyeriss.instruction(src=dram, dst=glb)
def glb_load(I):
    """``glb[d : d+n] = dram[s : s+n]`` -- stage a layer's operands on-chip."""

    @I.access
    def _(s, d, n):
        return (contiguous(dram, s, n), contiguous(glb, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


@eyeriss.instruction(src=glb, dst=dram)
def glb_store(I):
    """``dram[d : d+n] = glb[s : s+n]`` -- write a layer's ofmaps back off-chip."""

    @I.access
    def _(s, d, n):
        return (contiguous(glb, s, n), contiguous(dram, d, n))

    @I.compute
    def _(a, o):
        return primitive.identity(a)


# ==========================================================================#
# The processing pass -- Eyeriss's actual unit of configuration
# ==========================================================================#
#
# One pass consumes an ifmap block, a filter block and the running psum block, and
# produces the updated psum block. The C-dimension is *partial*: a pass covers `Cp`
# of the layer's `C` channels, so its result is a partial sum that a later pass adds
# to. The accumulator is therefore an explicit read operand (`pi`) with a separate
# write operand (`po`) -- the frontend cannot match an instruction that reads its own
# destination, and this is the modeling the search backend prescribes for it.
#
# Every `@I.access` parameter below is a *shape* param, solved from the source: an
# access param that appears in no pattern is refused outright ("never used by the
# access pattern"), which is why the fold factor is not one of them. It is an
# `@I.schedule` param instead -- chosen, not solved, and legal only where the
# hardware can be put into that configuration.


R = S = 3  # filter window -- STATIC, see finding 6
H_TILE = 16  # ifmap rows/cols a pass covers
E_TILE = H_TILE - R + 1  # = 14 ofmap rows: one PE set is R tall x E wide, and
#                          14 == PE_COLS, so a set spans the array's width exactly
MAX_SPATIAL = PE_ROWS // R  # at most 4 sets stack vertically (4*3 = 12 == PE_ROWS)
SET_SPAD = 2 * S + 1  # one folded set costs a PE a filter row, a sliding
#                       ifmap window, and a psum accumulator


def sets_folded(n_sets: int, spatial: int) -> int:
    """How many logical PE sets each physical PE must run in sequence when ``n_sets``
    of them are configured with ``spatial`` mapped across the array. Phase-1 folding,
    ISCA'16 Sec. V-B."""
    return -(-n_sets // spatial)


def pass_fits(n_sets: int, spatial: int) -> bool:
    """Whether one processing pass over ``n_sets`` logical PE sets is a configuration
    this chip can actually be put into: the spatially mapped sets have to fit the PE
    array, and the temporally folded ones have to fit each PE's register file.

    This is the machine's ``e_theta``. Note that it relates a *chosen* quantity
    (``spatial``) to *solved* ones (the block extents) through a ceiling division --
    neither the shape algebra nor a compute region can state it."""
    return (
        spatial * R <= PE_ROWS and sets_folded(n_sets, spatial) * SET_SPAD <= SPAD_WORDS
    )


@eyeriss.instruction(
    src=[glb, glb, glb],
    dst=glb,
    # What this configuration really costs: the number of sets each PE runs back to
    # back. Note the ceiling division -- `cost` and `@schedule` are called with solved
    # ints, so ordinary Python arithmetic is available even though the index algebra
    # (which has only + and *) cannot express any of this.
    cost=lambda Np, Cp, Mp, spatial: float(sets_folded(Np * Cp * Mp, spatial)),
)
def conv_pass(I):
    """One processing pass: ``po = pi + conv(ifm, flt)`` over a ``Cp``-channel slice.

    Blocks are NHWC (ifmap/psum) and OHWI (filter), the frontend's TOSA convention.
    ``Np``/``Cp``/``Mp`` are the images, input channels and filters this pass covers
    -- i.e. how many of the layer's ``N x M x C`` logical PE sets it takes on. The
    spatial extents are fixed constants (finding 6)."""

    @I.access
    def _(ifm, flt, pi, po, Np, Cp, Mp):
        return (
            view(glb, ifm, (Np, H_TILE, H_TILE, Cp)),
            view(glb, flt, (Mp, R, S, Cp)),
            view(glb, pi, (Np, E_TILE, E_TILE, Mp)),
            view(glb, po, (Np, E_TILE, E_TILE, Mp)),
        )

    @I.compute
    def _(ifm, flt, pi, po):
        zero = primitive.const(0.0, f32, (1,))
        return primitive.add(pi, primitive.conv2d(ifm, flt, zero))

    # `spatial` -- how many logical PE sets this pass maps *across* the array, the
    # rest being folded onto the same PEs in time. It is Eyeriss's configuration: a
    # free choice, constrained by the hardware, that the computed value does not
    # depend on. The compiler picks the cheapest legal value and emits it (`@n`).
    @I.schedule(spatial=range(1, MAX_SPATIAL + 1))
    def _(Np, Cp, Mp, spatial):
        return pass_fits(Np * Cp * Mp, spatial)


# ==========================================================================#
# Layer-level macro -- phase-2 folding, and how far `@I.expand` gets
# ==========================================================================#
#
# `conv_layer` states what a whole layer means and lowers it to a run of passes, one
# per image, staging each image through the global buffer. That is Eyeriss's **second
# folding phase**, "at the granularity of processing passes", whose extent ISCA'16
# Sec. V-B says is "determined by the global buffer size" -- which is exactly what
# `stage_words(C, M) <= GLB_WORDS` states below.
#
# Note where the macro's operands live: **`dram`, not `glb`**. That is the whole
# point. An earlier version declared them in `glb`, and the allocator then demanded
# the entire layer resident on-chip (`operands need 109056 unit(s) of 'glb' but it
# holds only 55296`) -- which read like a frontend limitation but was a *modeling*
# error: a layer-level operation's operands are off-chip by construction, and the
# staging is what its expansion is for. Nothing in the frontend had to change.
#
# What does not work is folding over M or C, the reuse dimensions Eyeriss actually
# cares about ("the same filter weights can be shared across N sets, the same ifmap
# pixels across M sets, and the psums across each C sets can be accumulated"). Both
# are *inner* dims of the NHWC / OHWI layouts, so a slice of them is a strided
# access, not a contiguous run -- and a stride is a Stage-2b residence param, solved
# from a neighbour's map, never freely chosen.

# The expansion's staging area, taken from the top of the global buffer downward so
# that best-fit allocation (which packs from 0 up) is the last thing to reach it.
# Reserving expansion scratch properly is a planner change, not a modeling one -- the
# same caveat MiniNPU's `matmul_layer` carries.


def stage_words(C: int, M: int) -> int:
    """The on-chip working set of one processing pass: one image's ifmap, the layer's
    filters, and one image's psums."""
    return H_TILE * H_TILE * C + M * R * S * C + E_TILE * E_TILE * M


@eyeriss.instruction(
    src=[dram, dram, dram],
    dst=dram,
    # The sum of what the expansion issues: N passes, each folding C*M sets.
    cost=lambda N, C, M, spatial: float(N * sets_folded(C * M, spatial)),
)
def conv_layer(I):
    """A whole layer in ``dram``, lowered by ``@I.expand`` to one staged pass per
    image: ``glb_load`` the operands, ``conv_pass``, ``glb_store`` the result."""

    @I.access
    def _(ifm, flt, pi, po, N, C, M):
        return (
            view(dram, ifm, (N, H_TILE, H_TILE, C)),
            view(dram, flt, (M, R, S, C)),
            view(dram, pi, (N, E_TILE, E_TILE, M)),
            view(dram, po, (N, E_TILE, E_TILE, M)),
        )

    @I.compute
    def _(ifm, flt, pi, po):
        zero = primitive.const(0.0, f32, (1,))
        return primitive.add(pi, primitive.conv2d(ifm, flt, zero))

    # Two constraints, and both are `e_theta`: each pass this expands into must be a
    # configuration the array can hold (phase 1), and the working set it stages must
    # fit the global buffer (phase 2). The passes are configured on their own when the
    # expansion issues them, but selection happens first, so the macro states here
    # what it is able to lower.
    @I.schedule(spatial=range(1, MAX_SPATIAL + 1))
    def _(N, C, M, spatial):
        return pass_fits(C * M, spatial) and stage_words(C, M) <= GLB_WORDS

    @I.expand
    def _(ifm, flt, pi, po, N, C, M):
        i_img, w_all = H_TILE * H_TILE * C, M * R * S * C
        p_img = E_TILE * E_TILE * M
        # The staged blocks are `scratch` tiles -- values, so the allocator places
        # them and their live ranges are the run. `ifm + n * i_img` is an offset
        # *into* the layer's operand; the operand's own base is allocation's answer.
        i_tile = scratch((1, H_TILE, H_TILE, C))
        w_tile = scratch((M, R, S, C))
        p_tile = scratch((1, E_TILE, E_TILE, M))
        glb_load(s=flt, d=w_tile, n=w_all)  # filters, hoisted out of the image loop
        for n in range(N):
            glb_load(s=ifm + n * i_img, d=i_tile, n=i_img)
            glb_load(s=pi + n * p_img, d=p_tile, n=p_img)
            conv_pass(ifm=i_tile, flt=w_tile, pi=p_tile, po=p_tile, Np=1, Cp=C, Mp=M)
            glb_store(s=p_tile, d=po + n * p_img, n=p_img)

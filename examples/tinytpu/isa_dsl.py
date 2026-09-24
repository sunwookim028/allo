# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loop-nest generator for TinyTPU-isa: nesting depth IS the AGU level.

**This is a code generator, not a compiler.** It chooses nothing. The author
writes the tiling, the loop order, the scratchpad and vreg layout, which tile
is peeled, and which address term walks which loop; this module derives the
*encoding* of what was written. There is no cost model here, no tile-size
search, no legality repair -- ask it for a bad nest and it raises, it does not
fix. MiniTPU's `board_package/dsl.py` states the same division and this file
keeps it:

    the programmer writes    the tiling, the loop order, the layout, which
                             k-tile is peeled, and which loop an address walks
    the generator derives    every `agu_level`, every AGU term, every loop
                             open/close pair, and the encoded instruction word
    the generator refuses    a nest deeper than the hardware loop stack, an
                             induction variable used outside its own loop, and
                             more address terms than one instruction can carry
    nobody decides           what the best tiling is -- there is no search here

## What it is for

`microarch_isa.gemm_program_handwritten` is the hand-emitted form, and its one
defect is the one MiniTPU's kernels have: the AGU *level* is a hand-typed
integer. Their kernels carry `level = (1 if grouped else 0) + 2` and keep a
trip-count-1 loop alive so the number does not shift; ours carried a comment
reading "level 0 is nb, level 1 is kb" and four `enc_agu` calls that had to
agree with it. Either way the loop structure is stated twice -- once as
`loop`/`endloop` placement, once as an integer -- and nothing checks that the
two agree. Reorder the nest and the addresses silently walk the wrong loop.

Here the level is not written at all. `loop()` is a context manager, the
nesting depth at the moment it opens *is* the level, and the object it yields
is the only way to name that level:

    with k.loop(Nt, "n") as nb:                  # level 0
        with k.loop(Kt, "k") as kb:              # level 1
            k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM).at(kb, T), rows=M)

Swap those two `with` statements and every address term follows, because `nb`
and `kb` carry their levels with them.

## Where this differs from MiniTPU's, and why

Their AGU is one term, applied to one field, as a power-of-two SHIFT of the
induction variable -- so `Ref.shift` is derived from a region whose stride was
padded to a power of two precisely to make that legal, and `load()` passes a
single `(shift, level)` pair. Ours is three `(target, level, stride)` terms
with arbitrary 11-bit strides, any number of which may land on the *same*
field: the `mm` inside the k loop names its weights at `B_SP + nb*MAXDIM +
kb*T`, two terms on `f3`, which their encoding cannot say at all.

So `Ref` here is not a staged region with a shift. It is a base plus an ordered
list of `(induction variable, stride)` terms, `.at()` adds one, and any field
of any instruction may be a `Ref`. The generator walks the fields in order
`f0, f1, f2, f3` and the terms of each field in the order they were written,
which is the order the hand-written program used, and packs them into the
second instruction word.

## What it does not do

No register allocation (`B_SP`, `A_VR`, `AR_C` are the author's names for the
author's layout), no scheduling, no peeling. `gemm_program` below peels the
first k-tile because the *author* decided that carrying `f2=0` on it is how you
say "overwrite the accumulator" without a predicate -- the generator has no
opinion about that and could not form one.

That peel is not forced by `f2` being static. `f2` is an AGU target like any
other field, and one additive term on it expresses acc=[0, 1] over a two-tile k
loop exactly. What it cannot express is the step the k=0 test needs once the
loop is longer, because an AGU term grows additively and monotonically: the
third value is 2, and `check_program` requires f2 in {0, 1}. See
`Unencodable` below, which is where that refusal is named and counted.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from examples.tinytpu.microarch_isa import (  # noqa: E402
    AGU_F0, AGU_F1, AGU_F2, AGU_F3, AGU_TERMS, LOOP_DEPTH,
    OP_DMA_LD, OP_ENDLOOP, OP_LOOP, OP_MM, OP_MVOUT, OP_NOP, OP_VADD, OP_VLD,
    OP_VADDRELU, OP_VRELU, enc, enc_agu,
    A_VR, AR_C, B_SP, DMA_SRC_B, DMA_TO_VR, MAXDIM, MAXROWS, T,
    SPAD_ROWS, NVR, NAR,
)

_TARGETS = (AGU_F0, AGU_F1, AGU_F2, AGU_F3)


class NestError(Exception):
    """A nest the hardware cannot run, named at the point it was written.

    `code` is the constant that refused it, for a caller grouping refusals by
    cause rather than by matching on the message.
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class Unencodable(NestError):
    """A legal loop nest this ISA cannot express, with the reason named first.

    Raised by `gemm_from_nest` below, which is the *encoder* half of the
    co-design loop: a frozen mapper enumerates nests and asks this module
    whether each one can be said in TinyTPU-isa. Every refusal here is a
    statement about the hardware or its encoding, so the reason string leads
    with a short code (`acc-peel:`, `intrinsic:`, `emitter:`, ...) that the
    mapper counts. Widening what the machine can encode is what makes more
    nests available to the mapper; see docs/source/extensions/act.rst.
    """


class Iv:
    """An induction variable, and the only handle on an AGU level.

    It is created by `Program.loop`, which sets `level` from the nesting depth,
    and it stops being usable when its `with` block closes. That expiry is the
    check the hand-written form could not have: an `enc_agu` call naming
    level 1 is legal-looking anywhere in the file, including after the k loop
    has closed, where the sequencer resolves it against a stale `iv_now[1]`.
    """

    __slots__ = ("level", "name", "live", "owner")

    def __init__(self, level, name, owner):
        self.level = level
        self.name = name
        self.owner = owner
        self.live = True

    def __repr__(self):
        return f"Iv({self.name!r}, level={self.level})"


@dataclass(frozen=True)
class Ref:
    """An address: a base, plus the loops it walks.

    `Ref(B_SP).at(nb, MAXDIM).at(kb, T)` is `B_SP + nb*MAXDIM + kb*T`, and it
    encodes as two AGU terms on whichever field it is passed as. Immutable, so
    a base held in a variable can be `.at()`-ed twice without the first
    derivation leaking into the second.
    """

    base: int = 0
    terms: tuple = ()

    def at(self, iv: Iv, stride: int) -> "Ref":
        """Advance this address by `stride` per iteration of `iv`."""
        if not isinstance(iv, Iv):
            raise NestError(
                f"Ref.at() takes the induction variable a `with k.loop(...)` "
                f"yields, not {iv!r}")
        return Ref(self.base, self.terms + ((iv, int(stride)),))


class Program:
    """A program under construction: the emitted words and the open nest."""

    def __init__(self, name="program"):
        self.name = name
        self.words = []          # (instruction word, AGU word) pairs
        self._open = []          # the Ivs of the loops currently open

    # -- what the programmer writes ------------------------------------------

    @contextmanager
    def loop(self, trips, name=None):
        """A hardware loop. **Nesting depth IS the AGU level**, 0 outermost.

        Emits `loop`/`endloop` around the body, which is the only thing the
        sequencer sees; the level exists nowhere in the encoding except inside
        the AGU terms of instructions that asked for it by naming the yielded
        induction variable.
        """
        level = len(self._open)
        name = name if name is not None else f"L{level}"
        if trips < 1:
            # The sequencer is a do-while: `endloop` tests `iv + 1 < trip`,
            # so a trip count of 0 runs the body ONCE. Refuse it rather than
            # emit an instruction whose static and dynamic meanings disagree.
            raise NestError(
                f"{self.name}: loop {name!r} has trip count {trips}. The "
                f"sequencer tests the back edge after the body, so a trip "
                f"count below 1 would still run it once -- guard the `with` "
                f"instead.", code="trip-count")
        if level >= LOOP_DEPTH:
            nest = " > ".join([iv.name for iv in self._open] + [name])
            raise NestError(
                f"{self.name}: loop {name!r} would open level {level}, but "
                f"the sequencer's loop stack is LOOP_DEPTH={LOOP_DEPTH} "
                f"levels deep. The nest is {nest}.", code="LOOP_DEPTH")
        iv = Iv(level, name, self)
        self.words.append((enc(OP_LOOP, nr=trips), 0))
        self._open.append(iv)
        try:
            yield iv
        finally:
            self._open.pop()
            iv.live = False
            self.words.append((enc(OP_ENDLOOP), 0))

    def nop(self):
        self._ins(OP_NOP)

    def dma_ld(self, src, dram_row=0, col_block=0, spad=None, vr=None, rows=0):
        """DRAM -> scratchpad (`spad=`) or operand vregs (`vr=`), exactly one
        of the two. `src` is 0 for A, 1 for B."""
        if (spad is None) == (vr is None):
            raise NestError(f"{self.name}: dma_ld needs exactly one of spad=, vr=")
        f0 = (DMA_SRC_B if src else 0) | (DMA_TO_VR if vr is not None else 0)
        self._ins(OP_DMA_LD, f0, dram_row, col_block,
                  vr if vr is not None else spad, nr=rows)

    def vld(self, vr, spad, rows):
        """Scratchpad -> operand vregs, `rows` packed words."""
        self._ins(OP_VLD, vr, spad, nr=rows)

    def mm(self, vr_a, ar, spad_w, rows, acc=False):
        """One k-tile into the array: `rows` activation rows from the vregs at
        `vr_a`, T weight rows from the scratchpad at `spad_w`. `acc`
        accumulates into `ar`, else overwrites -- which is why a k loop wants
        its first tile peeled."""
        self._ins(OP_MM, vr_a, ar, int(acc), spad_w, nr=rows)

    def vadd(self, ar_d, ar_s1, ar_s2, rows):
        self._ins(OP_VADD, ar_d, ar_s1, ar_s2, nr=rows)

    def vrelu(self, ar_d, ar_s, rows):
        self._ins(OP_VRELU, ar_d, ar_s, nr=rows)

    def vaddrelu(self, ar_d, ar_s1, ar_s2, rows):
        """`vadd` then `vrelu`, in one pass of the accumulator."""
        self._ins(OP_VADDRELU, ar_d, ar_s1, ar_s2, nr=rows)

    def mvout(self, ar, dram_row=0, col_block=0, rows=0):
        """Accumulator -> DRAM, clipped to int8 on the way."""
        self._ins(OP_MVOUT, ar, dram_row, col_block, nr=rows)

    def emit(self):
        """The `(word, agu_word)` list `assemble()` takes."""
        if self._open:
            raise NestError(
                f"{self.name}: loop "
                f"{' > '.join(iv.name for iv in self._open)} is still open")
        return list(self.words)

    # -- what the generator derives ------------------------------------------

    def _ins(self, op, *fields, nr=0):
        """One instruction. Every field is an int or a `Ref`; a `Ref`'s base
        goes in the field and its terms go in the AGU word, targeted at that
        field. Fields are walked `f0..f3` and each field's terms in the order
        `.at()` was called, so the AGU word reads in the order it was written.
        """
        bases = [0, 0, 0, 0]
        terms = []
        for i, v in enumerate(fields):
            if isinstance(v, Ref):
                bases[i] = v.base
                for iv, stride in v.terms:
                    self._check(iv, op)
                    terms.append((_TARGETS[i], iv.level, stride))
            else:
                bases[i] = int(v)
        if len(terms) > AGU_TERMS:
            raise NestError(
                f"{self.name}: instruction at index {len(self.words)} needs "
                f"{len(terms)} address terms; one instruction word carries "
                f"AGU_TERMS={AGU_TERMS}. The terms are "
                f"{[(t, l, s) for t, l, s in terms]}.", code="AGU_TERMS")
        self.words.append((enc(op, *bases, nr=nr), enc_agu(*terms)))

    def _check(self, iv, op):
        """An address may only walk a loop that is open around it."""
        if iv.owner is not self:
            raise NestError(
                f"{self.name}: induction variable {iv.name!r} belongs to "
                f"another program")
        if not iv.live:
            raise NestError(
                f"{self.name}: induction variable {iv.name!r} (level "
                f"{iv.level}) is used outside its own loop. The sequencer "
                f"would resolve it against a stale iv_now[{iv.level}].")
        if self._open[iv.level] is not iv:
            raise NestError(
                f"{self.name}: induction variable {iv.name!r} is not the loop "
                f"open at level {iv.level}")


# ---------------------------------------------------------------- programs ---
#: Which GEMM program order `gemm_program` emits when the caller names none.
#: `shipped` is the published program; the others are the parity baseline's
#: candidates (docs/source/designs/gemmini_comparison.rst, "The parity
#: baseline"). Every order computes the same result with the same per-unit
#: work counts on the same netlist -- only the issue ORDER differs.
GEMM_ORDERS = ("shipped", "interleaved", "b_per_tile")


def gemm_program(M, K, N, relu=False, order=None):
    """Tiled GEMM, generated. `order` (default: `$TPU_PROGRAM`, else
    `shipped`) selects the program order; see `GEMM_ORDERS`.

    The shipped program; `bench_isa.py` asserts it is
    word-for-word what `microarch_isa.gemm_program_handwritten` emits.

    Every decision here is the author's and is visible in the source: A goes
    straight into the vregs one column block per k-tile, B into the
    scratchpad one column block per n-tile, and each `mm` names its T weight
    rows there; the output loop is n outermost and k innermost; the first
    k-tile is peeled so it can carry `acc=False` (overwrite the accumulator)
    without a predicate on the induction variable.

    What is *not* here is any AGU level. `nb` and `kb` are objects yielded by
    the nest, and `B_SP + nb*MAXDIM + kb*T` is written as
    `Ref(B_SP).at(nb, MAXDIM).at(kb, T)` -- reorder the two `with` statements
    and the encoding follows.
    """
    order = order or os.environ.get("TPU_PROGRAM", "shipped")
    if order not in GEMM_ORDERS:
        raise NestError(f"TPU_PROGRAM/order={order!r}; one of {GEMM_ORDERS}")
    if order == "interleaved":
        return gemm_program_interleaved(M, K, N, relu)
    if order == "b_per_tile":
        return gemm_program_b_per_tile(M, K, N, relu)
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    k = Program(f"gemm{'.relu' if relu else ''} {M}x{K}x{N}")

    # --- A: one dma_ld per column block, straight into the vregs ---
    with k.loop(Kt, "kA") as kb:
        k.dma_ld(src=0, dram_row=0,
                 col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=M)
    # --- B: one per column block, into the scratchpad ---
    with k.loop(Nt, "nB") as nb:
        k.dma_ld(src=1, dram_row=0,
                 col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=K)

    # --- the output loop ---
    with k.loop(Nt, "n") as nb:
        #   peeled first k-tile: overwrite the accumulator
        k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM), rows=M, acc=False)
        if Kt > 1:
            with k.loop(Kt - 1, "k") as kb:
                k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                     Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)
        if relu:
            k.vrelu(AR_C, AR_C, rows=M)
        k.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=M)
    return k.emit()


def _gemm_checks(M, K, N):
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    return K // T, N // T


def gemm_program_b_per_tile(M, K, N, relu=False):
    """A parity candidate, the smaller change: A hoisted exactly as shipped,
    B's column block loaded inside the n loop just before the tile that uses
    it.

    Mechanism: the first `mm` waits for one B block instead of all `Nt` of
    them, because `spm` runs its instructions in order and the shipped program
    puts every B load ahead of every `mm`."""
    Kt, Nt = _gemm_checks(M, K, N)
    k = Program(f"b_per_tile gemm{'.relu' if relu else ''} {M}x{K}x{N}")
    with k.loop(Kt, "kA") as kb:
        k.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=M)
    with k.loop(Nt, "n") as nb:
        k.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=K)
        k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM), rows=M, acc=False)
        if Kt > 1:
            with k.loop(Kt - 1, "k") as kb:
                k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                     Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)
        if relu:
            k.vrelu(AR_C, AR_C, rows=M)
        k.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=M)
    return k.emit()


def gemm_program_interleaved(M, K, N, relu=False):
    """A parity candidate: every operand column block is loaded just before the
    first `mm` that reads it, instead of all of them before the first `mm`.

    Mechanism: `spm` and `vru` each run their instructions in order, so in the
    shipped program the first `mm` waits behind all `Kt*M + Nt*K` load rows;
    here it waits behind one B block and one A block (`K + M` rows), and the
    rest of the loads stream in between `mm`s while the array computes.

    Same vreg and scratchpad layout, same instructions, same per-unit work
    counts as the shipped program. The first n iteration is peeled because a
    hardware loop has no predicate: A's blocks are loaded on the first pass
    over k and not on the others. At most 18 static instructions (with relu),
    nesting depth 2, at most 3 AGU terms per instruction."""
    Kt, Nt = _gemm_checks(M, K, N)
    k = Program(f"interleaved gemm{'.relu' if relu else ''} {M}x{K}x{N}")

    # --- n = 0, peeled: A's blocks are loaded here, one per k-tile ---
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=K)
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=A_VR, rows=M)
    k.mm(A_VR, AR_C, B_SP, rows=M, acc=False)
    if Kt > 1:
        with k.loop(Kt - 1, "k0") as kb:
            k.dma_ld(src=0, dram_row=0, col_block=Ref(1).at(kb, 1),
                     vr=Ref(A_VR + MAXDIM).at(kb, MAXDIM), rows=M)
            k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                 Ref(B_SP + T).at(kb, T), rows=M, acc=True)
    if relu:
        k.vrelu(AR_C, AR_C, rows=M)
    k.mvout(AR_C, dram_row=0, col_block=0, rows=M)

    # --- n = 1 .. Nt-1: A is resident; B's block is loaded per tile ---
    if Nt > 1:
        with k.loop(Nt - 1, "n") as nb:
            k.dma_ld(src=1, dram_row=0, col_block=Ref(1).at(nb, 1),
                     spad=Ref(B_SP + MAXDIM).at(nb, MAXDIM), rows=K)
            k.mm(A_VR, AR_C, Ref(B_SP + MAXDIM).at(nb, MAXDIM),
                 rows=M, acc=False)
            if Kt > 1:
                with k.loop(Kt - 1, "k") as kb:
                    k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                         Ref(B_SP + MAXDIM + T).at(nb, MAXDIM).at(kb, T),
                         rows=M, acc=True)
            if relu:
                k.vrelu(AR_C, AR_C, rows=M)
            k.mvout(AR_C, dram_row=0, col_block=Ref(1).at(nb, 1), rows=M)
    return k.emit()


# ------------------------------------------------ a nest in, a program out ---
#: The ranks of a GEMM, in the order the intrinsic's fields are named. A nest
#: the mapper offers is a sequence of loops over these.
RANKS = ("M", "K", "N")


def split_body(nest):
    """The trailing *intrinsic* loops of `nest`, and the rest.

    The innermost level is the tile one instruction performs by itself, so it
    is a property of the ARRAY, not of the mapping: one `mm` drives `rows`
    activation rows against a TxT weight block. The tail must therefore be one
    loop per rank at the innermost level with K = N = T and M free (`rows` is a
    static field, so the mapper may pick it, up to MAXROWS).

    Returns `(emitted_loops, rows)`. Widening the intrinsic -- a wider array, a
    multi-block `mm` -- is a hardware change that shows up here.
    """
    if not nest:
        raise Unencodable("intrinsic: empty nest")
    inner = nest[-1].level
    cut = len(nest)
    while cut and nest[cut - 1].level == inner:
        cut -= 1
    tail = {l.rank: l.factor for l in nest[cut:]}
    if sorted(tail) != sorted(RANKS):
        raise Unencodable(
            f"intrinsic: the innermost level carries {sorted(tail)}, but one "
            f"`mm` performs all three of {list(RANKS)}")
    if tail["K"] != T or tail["N"] != T:
        raise Unencodable(
            f"intrinsic: one `mm` performs a {T}x{T} weight block, the nest "
            f"asks for {tail['K']}x{tail['N']}")
    if not 1 <= tail["M"] <= MAXROWS:
        raise Unencodable(f"intrinsic: rows={tail['M']} exceeds MAXROWS={MAXROWS}")
    return tuple(nest[:cut]), tail["M"]


def gemm_from_nest(nest, M, K, N, relu=False):
    """Emit a TinyTPU-isa GEMM for `nest`, or say why this ISA cannot say it.

    This is the *encoder* the co-design loop drives. A loop nest -- anything
    whose elements have `.rank`, `.factor`, `.level` and `.spatial`, which is
    ACT's `mapping.Loop` interface (docs/source/extensions/act.rst) -- decides
    the order of the emitted loops and how each rank is split across them. What
    it does not decide is the machine's staging: B into the scratchpad one
    block per n-tile, A into the operand vregs one block per k-tile, one
    accumulator region drained per output tile.

    `gemm_program` above is the nest `canonical()` describes, hand-written; this
    function re-emits that program word for word and generalises it.

    Every `raise Unencodable` below is a hardware or encoding limit, and the
    mapper counts them by code. Today they are:

      `acc-peel`   `acc` cannot be made to follow the k loop, so the k=0
                   tile must be a peelable prefix -- which pins K innermost and
                   unsplit and kills every permutation that moves it. This is
                   the binding constraint: 1,150 of 1,226 nests at 16x16x16.
                   **Not because the field is static.** `acc` is `f2` and `f2`
                   IS an AGU target: one additive term on it gives acc=[0] at
                   Kt=1 and acc=[0,1] at Kt=2 -- a genuine no-peel GEMM, exact
                   against `isa_ref` -- and dies at Kt>=3, where the term's
                   third value is 2 and `check_program` requires f2 in {0, 1}.
                   The obstacle is that an AGU term is ADDITIVE and MONOTONE
                   where the k=0 test needs a step. It also costs one of the
                   three AGU terms, so at Nt>1 the accumulating `mm` needs a
                   fourth and does not fit at all: this constraint and
                   `AGU_TERMS` are coupled, not alternatives.
                   Whatever resolves it belongs in the AGU resolution -- the
                   sequencer, with `expand` in lockstep -- so that what reaches
                   a unit, and what the frozen `isa_ref` sees, is still 0 or 1
                   and the instruction keeps its architectural meaning.
      `emitter`    a limit of this function, not of the machine: it re-stages A
                   per m-tile and does not know how to interleave that with an
                   n loop.
      `intrinsic`  the array's own tile (see `split_body`).

    Two more come from below this function: `AGU_TERMS` (one instruction word's
    address-term budget, raised by `Program._ins`) and the accumulator RAW
    distance (raised by `microarch_isa.check_program`). The 3-term AGU does not
    merely forbid nests, it CHOOSES the reuse strategy: an m-tiled nest is
    encodable only if A is re-staged per m-tile, because keeping it resident
    across m needs a fourth term on the accumulating `mm`.
    """
    if not (M <= MAXDIM and K <= MAXDIM and N <= MAXDIM):
        raise Unencodable(f"shape: {M}x{K}x{N} exceeds MAXDIM={MAXDIM}")
    emitted, Mt = split_body(nest)

    ks = [i for i, l in enumerate(emitted) if l.rank == "K"]
    if len(ks) > 1:
        raise Unencodable(
            "acc-peel: K is split across two emitted loops, so the k=0 tile is "
            "not a peelable prefix, and `acc` cannot follow two induction "
            "variables at once -- an AGU term is additive and monotone where "
            "the k=0 test needs a step")
    if ks and ks[0] != len(emitted) - 1:
        raise Unencodable(
            f"acc-peel: the emitted order is "
            f"{'>'.join(l.rank for l in emitted)}, but the k=0 tile must be a "
            f"peelable prefix, which needs K innermost")
    ms = [l for l in emitted if l.rank == "M"]
    if ms and [l.rank for l in emitted[:len(ms)]] != ["M"] * len(ms):
        raise Unencodable(
            "emitter: m loops must be outermost -- this one re-stages A per "
            "m-tile and does not know how to interleave that with an n loop. "
            "A limit of this file, not of the machine")

    Kt = emitted[ks[0]].factor if ks else 1
    order = ">".join(f"{l.rank}{l.factor}" for l in emitted) or "-"
    k = Program(f"gemm{'.relu' if relu else ''} {M}x{K}x{N} [{order}]")
    rest = emitted[len(ms):len(emitted) - len(ks)]          # the n loops

    def steps(ivs, rank, unit):
        """An iv's stride, in units of the thing it indexes: n-tiles for N,
        rows for M. Innermost-first mixed radix -- these are the AGU strides."""
        out, step = [], unit
        for r, iv, factor in reversed(ivs):
            if r == rank:
                out.append((iv, step))
                step *= factor
        return out

    def ref(base, terms, scale=1):
        r = Ref(base)
        for iv, s in terms:
            r = r.at(iv, s * scale)
        return r

    def stage_a(m_terms):
        """A: one k-block per k-tile, into the operand vregs."""
        with k.loop(K // T, "kA") as kb:
            k.dma_ld(src=0, dram_row=ref(0, m_terms), col_block=Ref().at(kb, 1),
                     vr=Ref(A_VR).at(kb, MAXDIM), rows=Mt)

    def stage_b():
        with k.loop(N // T, "nB") as nb:
            k.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1),
                     spad=Ref(B_SP).at(nb, MAXDIM), rows=K)

    def body(ivs):
        m_terms = steps(ivs, "M", Mt)
        n_terms = steps(ivs, "N", 1)
        # The peeled k=0 tile overwrites the accumulator; the rest accumulate.
        k.mm(A_VR, AR_C, ref(B_SP, n_terms, MAXDIM), rows=Mt, acc=False)
        if Kt > 1:
            with k.loop(Kt - 1, "k") as kb:
                k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                     ref(B_SP + T, n_terms, MAXDIM).at(kb, T),
                     rows=Mt, acc=True)
        if relu:
            k.vrelu(AR_C, AR_C, rows=Mt)
        k.mvout(AR_C, dram_row=ref(0, m_terms), col_block=ref(0, n_terms),
                rows=Mt)

    def walk(loops, ivs):
        if not loops:
            body(ivs)
            return
        lv = loops[0]
        # Named `ivar`, not `iv`: `spec_policy`'s monkeypatch rule tracks names
        # derived from an imported module without knowing about scopes, and a
        # second `iv` here would taint `Program.loop`'s own `iv.live = False`.
        with k.loop(lv.factor, f"{lv.rank.lower()}{len(ivs)}") as ivar:
            nxt = ivs + [(lv.rank, ivar, lv.factor)]
            if lv.rank == "M" and len(nxt) == len(ms):
                stage_a(steps(nxt, "M", Mt))     # innermost m loop reached
            walk(loops[1:], nxt)

    if ms:
        stage_b()                                # B is m-independent: hoist it
    else:
        stage_a([])                              # the shipped prologue order
        stage_b()
    walk(list(ms) + list(rest), [])
    return k.emit()


def vector_program(M=8):
    """Every field a GEMM leaves constant, varied -- a TEST program, not a
    kernel. The shipped GEMM always has `dma_ld`/`mvout` DRAM row 0, one
    accumulator region, A only in the vregs, B only in the scratchpad, no
    `vld`, and `vrelu` in place (`f0 == f1`); a unit that ignored any of those
    would pass `bench_isa` exactly. Here:

      * `dma_ld` from nonzero DRAM rows and column blocks, both sources into
        both memories (A into the scratchpad, B into the vregs too);
      * `vld` moving scratchpad rows into the vregs as activations;
      * `mm` into two accumulator regions away from 0, one of them accumulated,
        with weights from two scratchpad regions, one of them A's rows;
      * `vadd` with three distinct regions, `vrelu` with a distinct destination;
      * `mvout` from nonzero `ar`, to nonzero DRAM rows and column blocks, and
        through a loop whose AGU walks `f0`, `f1` and `f2` at once.

    Its gold is `isa_ref.run`, not a formula.

    **What it requires of the build, and why each is here.** These used to be
    unstated, and at T=8 the program silently asked `mm` to read weight rows
    no `dma_ld` had written -- `check_program` caught it, but as a confusing
    rejection of a shipped test program rather than as a configuration limit:

      * `M >= T`, because the weight `dma_ld` loads M rows and the third
        `mm` reads T weight rows from that same region. With M < T the tail is
        unwritten.
      * `MAXDIM // T >= 4`, because it names column block 3 to vary a field
        the GEMM leaves at 0. That is the constraint behind "T=8 needs
        MAXDIM >= 32".
      * room in DRAM for `dram_row=5` plus its `2 * T` rows, and in `ar` for
        the four derived regions, whose top row is `ar_4 + M` (see STRIDE
        below).
    """
    assert M % 2 == 0 and 2 * M <= MAXDIM
    assert M >= T, (
        f"vector_program({M}) at T={T}: `mm` reads T={T} weight rows from the "
        f"region a {M}-row dma_ld filled, so M must be at least T")
    assert MAXDIM // T >= 4, (
        f"vector_program needs column block 3, so MAXDIM // T >= 4; "
        f"MAXDIM={MAXDIM}, T={T} gives {MAXDIM // T}")
    assert 5 + 2 * T <= MAXDIM, (
        f"vector_program reads {2 * T} DRAM rows from row 5, past "
        f"MAXDIM={MAXDIM}")
    # THE REGION ADDRESSES ARE DERIVED, NOT TYPED IN. They used to be the
    # literals 40 / 100 / 10 / 30 / 20 / 40 / 60 / 80, which fit only because
    # the memories were themselves literals (spad 512, vr 256, ar 128). Once
    # the memories are sized from MAXDIM (`microarch_isa`), spad row 100 is
    # off the end of a MAXDIM=16 build and this program stopped assembling.
    # Nothing about it needs a particular address: the regions only have to be
    # DISTINCT, NON-ZERO and in range, which is what makes it a test of the
    # fields the GEMM leaves at zero. `STRIDE` is the widest region any single
    # instruction here touches, so consecutive bases cannot overlap.
    STRIDE = max(M, 2 * T)
    sp_a, sp_w = 1, 1 + STRIDE               # A rows; the two weight blocks
    vr_1, vr_2 = 1, 1 + STRIDE               # vld destination; B from DRAM
    ar_1, ar_2, ar_3, ar_4 = (1 + i * STRIDE for i in range(4))
    assert sp_w + 2 * T <= SPAD_ROWS, (
        f"vector_program needs {sp_w + 2 * T} spad rows, have {SPAD_ROWS}")
    assert vr_2 + M <= NVR, (
        f"vector_program needs {vr_2 + M} vregs, have {NVR}")
    assert ar_4 + M <= NAR, (
        f"vector_program needs {ar_4 + M} accumulator rows, have {NAR}")
    k = Program(f"vector {M}")
    k.dma_ld(src=0, dram_row=3, col_block=1, spad=sp_a, rows=M)  # A -> spad
    k.dma_ld(src=1, dram_row=2, col_block=2, vr=vr_2, rows=M)    # B -> vr
    k.dma_ld(src=1, dram_row=5, col_block=3, spad=sp_w, rows=2 * T)
    k.vld(vr_1, sp_a, rows=M)                # activations 1: A rows via spad
    k.mm(vr_1, ar_1, sp_w, rows=M)           # ar_1 = act1 @ W1
    k.mm(vr_2, ar_2, sp_w + T, rows=M)       # ar_2 = act2 @ W2
    k.mm(vr_1, ar_2, sp_a, rows=M, acc=True)  # ar_2 += act1 @ (the A rows)
    k.vadd(ar_3, ar_1, ar_2, rows=M)         # ar_3 = ar_1 + ar_2
    k.vrelu(ar_4, ar_3, rows=M)              # ar_4 = relu(ar_3)
    k.mvout(ar_3, dram_row=0, col_block=1, rows=M)
    k.mvout(ar_1, dram_row=1, col_block=0, rows=M)
    h = M // 2
    with k.loop(2, "half") as i:             # ar_4.. -> C rows M.., blocks 2, 3
        k.mvout(Ref(ar_4).at(i, h), dram_row=Ref(M).at(i, h),
                col_block=Ref(2).at(i, 1), rows=h)
    return k.emit()


def ar_distance_program(dist):
    """The accumulator distance contract, exercised AT its edge -- a TEST
    program. Reads of `ar` land exactly `dist` `accu` iterations after the
    write they depend on, through every kind of read `accu` has: an
    accumulating `mm`, `vrelu`, `vadd`'s first and second source, and `mvout`,
    and after every kind of write, including `vadd`'s (two iterations a row).

    With `dist = AR_RAW_DIST` it is the tightest program `check_program`
    accepts, and it is what shows the RTL honours the dependence claim
    `schedule()` makes: a pipeline whose read-to-write window reached `dist`
    reads a stale row here, where no GEMM ever would. `check_program` must
    reject `dist - 1`.

    `accu` iterations below are counted from the first `mm`: one per row, two
    per `vadd` row. An `n`-row instruction's first row is read by the next
    instruction `n` iterations after it was written."""
    n = dist
    assert 1 <= n and 2 * n <= MAXDIM
    k = Program(f"ar distance {dist}")
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=0, rows=n)
    k.dma_ld(src=1, dram_row=0, col_block=1, spad=0, rows=T)
    k.mm(0, 10, 0, rows=n)                   # ar10+i written at i
    k.mm(0, 10, 0, rows=n, acc=True)         # acc read at n+i: distance n
    k.vrelu(30, 10, rows=n)                  # read at 2n+i: distance n
    k.vadd(40, 30, 10, rows=n)               # 1st source row 0: distance n
    k.mvout(40, dram_row=0, col_block=0, rows=n)   # last row: distance n
    if n >= 2:
        k.mm(0, 50, 0, rows=n - 1)           # ar50+i written n-1 rows long
        k.vadd(60, 40, 50, rows=n - 1)       # 2nd source row 0: distance n
        k.mvout(10, dram_row=n, col_block=1, rows=n)   # spacer
        k.mvout(60, dram_row=0, col_block=2, rows=n - 1)
    return k.emit()


def assert_matches_handwritten(shapes):
    """The generator is right exactly when it emits the same bits.

    Word-by-word against `gemm_program_handwritten`, both words of every
    instruction, at every shape and both relu settings. Any difference is a bug
    in the generator -- or in the hand-written reference, which is also a
    result; the message says which instruction and which word.
    """
    from examples.tinytpu.microarch_isa import (
        gemm_program_handwritten)
    for (M, K, N) in shapes:
        for relu in (False, True):
            tag = f"{'gemm.relu' if relu else 'gemm'} {M}x{K}x{N}"
            got = gemm_program(M, K, N, relu, order="shipped")
            ref = gemm_program_handwritten(M, K, N, relu)
            assert len(got) == len(ref), (
                f"{tag}: generated {len(got)} instructions, hand-written "
                f"{len(ref)}")
            for i, ((g0, g1), (r0, r1)) in enumerate(zip(got, ref)):
                assert g0 == r0, (
                    f"{tag}: instruction {i} word 0 differs: generated "
                    f"0x{g0:016x} (op {g0 & 0x3F}), hand-written 0x{r0:016x} "
                    f"(op {r0 & 0x3F})")
                assert g1 == r1, (
                    f"{tag}: instruction {i} AGU word differs: generated "
                    f"0x{g1:016x}, hand-written 0x{r1:016x}")
    return True


if __name__ == "__main__":
    from examples.tinytpu.bench_isa import SHAPES
    assert_matches_handwritten(SHAPES)
    print(f"  generated == hand-written, word for word, at all "
          f"{len(SHAPES)} shapes x {{gemm, gemm.relu}}")
    for (M, K, N) in SHAPES:
        p = gemm_program(M, K, N, True)
        print(f"  gemm.relu {M:2d}x{K:2d}x{N:2d}  {len(p):3d} static "
              f"instructions")

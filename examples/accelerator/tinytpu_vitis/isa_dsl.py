# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loop-nest generator for TinyTPU-isa: nesting depth IS the AGU level.

**This is a code generator, not a compiler.** It chooses nothing. The author
writes the tiling, the loop order, the VMEM and vreg layout, which tile
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
            k.mm(A_VR, AR_C, Ref(B_VM).at(nb, MAXDIM).at(kb, T), rows=M)

Swap those two `with` statements and every address term follows, because `nb`
and `kb` carry their levels with them.

## Where this differs from MiniTPU's, and why

Their AGU is one term, applied to one field, as a power-of-two SHIFT of the
induction variable -- so `Ref.shift` is derived from a region whose stride was
padded to a power of two precisely to make that legal, and `load()` passes a
single `(shift, level)` pair. Ours is three `(target, level, stride)` terms
with arbitrary 11-bit strides, any number of which may land on the *same*
field: the `mm` inside the k loop names its weights at `B_VM + nb*MAXDIM +
kb*T`, two terms on `f3`, which their encoding cannot say at all.

So `Ref` here is not a staged region with a shift. It is a base plus an ordered
list of `(induction variable, stride)` terms, `.at()` adds one, and any field
of any instruction may be a `Ref`. The generator walks the fields in order
`f0, f1, f2, f3` and the terms of each field in the order they were written,
which is the order the hand-written program used, and packs them into the
second instruction word.

## What it does not do

No register allocation (`B_VM`, `A_VR`, `AR_C` are the author's names for the
author's layout), no scheduling, no peeling. `gemm_program` below peels the
first k-tile because the *author* decided that carrying `f2=0` on it is how you
say "overwrite the accumulator" without a predicate -- the generator has no
opinion about that and could not form one.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    AGU_F0, AGU_F1, AGU_F2, AGU_F3, AGU_TERMS, LOOP_DEPTH,
    OP_DMA_LD, OP_ENDLOOP, OP_LOOP, OP_MVOUT, OP_NOP, OP_VADD, OP_VLD,
    OP_VRELU, OP_VMATLOAD, OP_VMATPUSH, OP_VMATPOP, enc, enc_agu,
    A_VM, A_VR, AR_C, AR_P, B_VM, B_VR, DMA_SRC_B, MAXDIM, MAXROWS, T, WPR,
    VEC_M,
)

_TARGETS = (AGU_F0, AGU_F1, AGU_F2, AGU_F3)


class NestError(Exception):
    """A nest the hardware cannot run, named at the point it was written."""


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

    `Ref(B_VM).at(nb, MAXDIM).at(kb, T)` is `B_VM + nb*MAXDIM + kb*T`, and it
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
                f"instead.")
        if level >= LOOP_DEPTH:
            nest = " > ".join([iv.name for iv in self._open] + [name])
            raise NestError(
                f"{self.name}: loop {name!r} would open level {level}, but "
                f"the sequencer's loop stack is LOOP_DEPTH={LOOP_DEPTH} "
                f"levels deep. The nest is {nest}.")
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

    def dma_ld(self, src, dram_row=0, col_block=0, vmem=None, rows=0):
        """DRAM -> VMEM, the only place DMA writes. `src` is 0 for A, 1
        for B."""
        if vmem is None:
            raise NestError(f"{self.name}: dma_ld needs vmem=")
        self._ins(OP_DMA_LD, DMA_SRC_B if src else 0, dram_row, col_block,
                  vmem, nr=rows)

    def vld(self, vr, vmem, rows):
        """VMEM -> operand vregs, `rows` packed words."""
        self._ins(OP_VLD, vr, vmem, nr=rows)

    def vmatload(self, vr_w):
        """T weight rows from the vregs at `vr_w` into the array; they take
        effect from the next `vmatpush` (MiniTPU's semantics)."""
        self._ins(OP_VMATLOAD, vr_w, nr=T)

    def vmatpush(self, vr_a, rows):
        """`rows` activation rows from the vregs at `vr_a` into the array."""
        self._ins(OP_VMATPUSH, vr_a, nr=rows)

    def vmatpop(self, ar, rows):
        """The oldest `rows` un-popped result rows into `ar`."""
        self._ins(OP_VMATPOP, ar, nr=rows)

    def vadd(self, ar_d, ar_s1, ar_s2, rows):
        self._ins(OP_VADD, ar_d, ar_s1, ar_s2, nr=rows)

    def vrelu(self, ar_d, ar_s, rows):
        self._ins(OP_VRELU, ar_d, ar_s, nr=rows)

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
                f"{[(t, l, s) for t, l, s in terms]}.")
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
def gemm_program(M, K, N, relu=False):
    """Tiled GEMM, generated. The shipped program; `bench_isa.py` asserts it is
    word-for-word what `microarch_isa.gemm_program_handwritten` emits.

    Every decision here is the author's and is visible in the source: A and B
    go DRAM -> VMEM -> vregs one column block at a time; each (n, k) tile is
    a `vmatload` of T weight rows, a `vmatpush` of M activation rows and a
    `vmatpop` of M result rows; k-tiles after the first are popped into
    `AR_P` and summed into `AR_C` with `vadd` (MiniTPU's split: the array
    accumulates T deep, deeper is the VPU's). The first k-tile is peeled so it
    can pop straight into `AR_C`.

    What is *not* here is any AGU level. `nb` and `kb` are objects yielded by
    the nest, and `B_VR + nb*MAXDIM + kb*T` is written as
    `Ref(B_VR).at(nb, MAXDIM).at(kb, T)` -- reorder the two `with` statements
    and the encoding follows.
    """
    assert M <= MAXDIM and K <= MAXDIM and N <= MAXDIM, (
        f"{M}x{K}x{N} exceeds the built MAXDIM={MAXDIM}")
    assert M % T == 0 and K % T == 0 and N % T == 0
    assert M <= MAXROWS and K <= MAXROWS
    Kt, Nt = K // T, N // T
    k = Program(f"gemm{'.relu' if relu else ''} {M}x{K}x{N}")

    # --- A: one dma_ld per column block into VMEM, then vld into the vregs ---
    with k.loop(Kt, "kA") as kb:
        k.dma_ld(src=0, dram_row=0,
                 col_block=Ref().at(kb, 1),
                 vmem=Ref(A_VM).at(kb, MAXDIM), rows=M)
        k.vld(Ref(A_VR).at(kb, MAXDIM), Ref(A_VM).at(kb, MAXDIM), rows=M)
    # --- B: the same ---
    with k.loop(Nt, "nB") as nb:
        k.dma_ld(src=1, dram_row=0,
                 col_block=Ref().at(nb, 1),
                 vmem=Ref(B_VM).at(nb, MAXDIM), rows=K)
        k.vld(Ref(B_VR).at(nb, MAXDIM), Ref(B_VM).at(nb, MAXDIM), rows=K)

    # --- the output loop ---
    with k.loop(Nt, "n") as nb:
        #   peeled first k-tile: pop straight into the accumulator region
        k.vmatload(Ref(B_VR).at(nb, MAXDIM))
        k.vmatpush(A_VR, rows=M)
        k.vmatpop(AR_C, rows=M)
        if Kt > 1:
            with k.loop(Kt - 1, "k") as kb:
                k.vmatload(Ref(B_VR + T).at(nb, MAXDIM).at(kb, T))
                k.vmatpush(Ref(A_VR + MAXDIM).at(kb, MAXDIM), rows=M)
                k.vmatpop(AR_P, rows=M)
                k.vadd(AR_C, AR_C, AR_P, rows=M)
        if relu:
            k.vrelu(AR_C, AR_C, rows=M)
        k.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=M)
    return k.emit()


def vector_program(M=None):
    """Every field a GEMM leaves constant, varied -- a TEST program, not a
    kernel. The shipped GEMM always has `dma_ld`/`mvout` DRAM row 0, one
    accumulator region, A only as activations, B only as weights, one push
    per load popped at once, and `vrelu` in place (`f0 == f1`); a unit that
    ignored any of those would pass `bench_isa` exactly. Here:

      * `dma_ld` from nonzero DRAM rows and column blocks, both sources, to
        VMEM rows away from the GEMM layout;
      * `vld` moving B's rows in as activations;
      * two loads and two pushes before the first pop (the pending-weight
        queue and the output FIFO both hold more than one tile), and one
        push drained by two pops;
      * `vadd` with three distinct regions, `vrelu` with a distinct destination;
      * `mvout` from nonzero `ar`, to nonzero DRAM rows and column blocks, and
        through a loop whose AGU walks `f0`, `f1` and `f2` at once.

    Every address is derived from T and MAXDIM (M defaults to VEC_M), so the
    program exercises the same things at every array size. Its gold is
    `isa_ref.run`, not a formula."""
    M = M or VEC_M
    assert 2 * M <= MAXDIM and M % 2 == 0
    h = M // 2
    c = lambda i: i % WPR                    # noqa: E731  column blocks
    va, vb, vw = 2 * MAXDIM + 8, 3 * MAXDIM + 16, 4 * MAXDIM + 24   # VMEM
    r1 = 10                                  # one vreg file: act1, act2,
    r2 = r1 + M + 5                          # W1 | W2, then five result
    rw = r2 + M + 7                          # regions, none overlapping
    S = M + 8
    a1 = rw + 2 * T + 6
    a2, a3, a4, a5 = a1 + S, a1 + 2 * S, a1 + 3 * S, a1 + 4 * S
    k = Program(f"vector {M}")
    k.dma_ld(src=0, dram_row=min(3, MAXDIM - M), col_block=c(1), vmem=va, rows=M)
    k.dma_ld(src=1, dram_row=min(2, MAXDIM - M), col_block=c(2), vmem=vb, rows=M)
    if 2 * T <= MAXDIM:                      # W1 | W2: 2T rows of B
        k.dma_ld(src=1, dram_row=min(5, MAXDIM - 2 * T), col_block=c(3),
                 vmem=vw, rows=2 * T)
    else:                                    # only T rows exist: two blocks
        k.dma_ld(src=1, dram_row=0, col_block=c(3), vmem=vw, rows=T)
        k.dma_ld(src=0, dram_row=0, col_block=c(0), vmem=vw + T, rows=T)
    k.vld(r1, va, rows=M)                    # act1: A rows
    k.vld(r2, vb, rows=M)                    # act2: B rows
    k.vld(rw, vw, rows=2 * T)                # W1 at rw, W2 at rw + T
    k.vmatload(rw)                           # W1
    k.vmatpush(r1, rows=M)                   #   act1 @ W1
    k.vmatload(rw + T)                       # W2, pending while act1 runs
    k.vmatpush(r2, rows=M)                   #   act2 @ W2
    k.vmatpop(a1, rows=h)                    # a1 = act1 @ W1, in two pops
    k.vmatpop(a1 + h, rows=M - h)
    k.vmatpop(a2, rows=M)                    # a2 = act2 @ W2
    k.vmatload(rw)                           # W1 again
    k.vmatpush(r2, rows=M)
    k.vmatpop(a3, rows=M)                    # a3 = act2 @ W1
    k.vadd(a2, a2, a3, rows=M)               # a2 += a3
    k.vadd(a4, a1, a2, rows=M)               # a4 = a1 + a2
    k.vrelu(a5, a4, rows=M)                  # a5 = relu(a4)
    k.mvout(a4, dram_row=0, col_block=c(1), rows=M)
    k.mvout(a1, dram_row=1, col_block=c(0), rows=M)
    with k.loop(2, "half") as i:             # a5.. -> C rows M.., blocks 2, 3
        k.mvout(Ref(a5).at(i, h), dram_row=Ref(M).at(i, h),
                col_block=Ref(c(2)).at(i, 1 if WPR > c(2) + 1 else 0), rows=h)
    return k.emit()


def ar_distance_program(dist):
    """The accumulator distance contract, exercised AT its edge -- a TEST
    program. Reads of `ar` land exactly `dist` `accu` iterations after the
    write they depend on, through every kind of read `accu` has -- `vrelu`,
    `vadd`'s first and second source, and `mvout` -- and after every kind of
    write: a pop, a `vrelu` and a `vadd` (two iterations a row).

    With `dist = AR_RAW_DIST` it is the tightest program `check_program`
    accepts, and it is what shows the RTL honours the dependence claim
    `schedule()` makes: a pipeline whose read-to-write window reached `dist`
    reads a stale row here, where no GEMM ever would. `check_program` must
    reject `dist - 1`.

    `accu` iterations below are counted from the first pop: one per row, two
    per `vadd` row. An `n`-row instruction's first row is read by the next
    instruction `n` iterations after it was written."""
    n = dist
    assert 1 <= n and 2 * n <= MAXDIM
    k = Program(f"ar distance {dist}")
    k.dma_ld(src=0, dram_row=0, col_block=0, vmem=0, rows=n)
    k.vld(0, 0, rows=n)
    k.dma_ld(src=1, dram_row=0, col_block=1 % WPR, vmem=64, rows=T)
    k.vld(100, 64, rows=T)                   # clear of every region below
    k.vmatload(100)
    k.vmatpush(0, rows=n)
    k.vmatpush(0, rows=n)
    k.vmatpop(10, rows=n)                    # ar10+i written at i
    k.vrelu(30, 10, rows=n)                  # read at n+i: distance n
    k.vadd(40, 30, 10, rows=n)               # 1st source row 0: distance n
    k.mvout(40, dram_row=0, col_block=0, rows=n)   # last row: distance n
    if n >= 2:
        k.vmatpop(50, rows=n - 1)            # ar50+i written n-1 rows long
        k.vadd(60, 40, 50, rows=n - 1)       # 2nd source row 0: distance n
        k.vmatpop(70, rows=1)                # drain the queue
        k.mvout(10, dram_row=n, col_block=1 % WPR, rows=n)   # spacer
        k.mvout(60, dram_row=0, col_block=2 % WPR, rows=n - 1)
        k.mvout(70, dram_row=2 * n - 1, col_block=3 % WPR, rows=1)
    else:
        k.vmatpop(50, rows=n)
        k.mvout(50, dram_row=n, col_block=1 % WPR, rows=n)
    return k.emit()


def assert_matches_handwritten(shapes):
    """The generator is right exactly when it emits the same bits.

    Word-by-word against `gemm_program_handwritten`, both words of every
    instruction, at every shape and both relu settings. Any difference is a bug
    in the generator -- or in the hand-written reference, which is also a
    result; the message says which instruction and which word.
    """
    from examples.accelerator.tinytpu_vitis.microarch_isa import (
        gemm_program_handwritten)
    for (M, K, N) in shapes:
        for relu in (False, True):
            tag = f"{'gemm.relu' if relu else 'gemm'} {M}x{K}x{N}"
            got = gemm_program(M, K, N, relu)
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
    from examples.accelerator.tinytpu_vitis.microarch_isa import SCORED_SHAPES as SHAPES
    assert_matches_handwritten(SHAPES)
    print(f"  generated == hand-written, word for word, at all "
          f"{len(SHAPES)} shapes x {{gemm, gemm.relu}}")
    for (M, K, N) in SHAPES:
        p = gemm_program(M, K, N, True)
        print(f"  gemm.relu {M:2d}x{K:2d}x{N:2d}  {len(p):3d} static "
              f"instructions")

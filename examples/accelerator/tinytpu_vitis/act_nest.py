# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loop nest in, a TinyTPU-isa program out -- the ACT-as-a-mapper seam.

Kai Shao's ACT flow (https://github.com/kkkaishao/allo, branch ``act``; see
``docs/source/extensions/act.rst`` and ``ATTRIBUTION.md`` at tag
``chia-codesign-final``) has a loop-nest mapper whose chosen object is a frozen
tuple of ``Loop(rank, factor, level, spatial)``, outermost-first. Its own
backend then *fully unrolls* that nest and throws the structure away. Our
machine has hardware loops and a 3-term AGU, so the nest is the thing we want
and the unrolled stream is the thing we do not.

This module is that seam, and nothing else. It takes a nest -- any sequence
whose elements have ``.rank``, ``.factor``, ``.level`` and ``.spatial``, so a
real ``allo.exp.dsa.mapping.Mapping.loops`` can be passed straight in -- and
emits a program through ``isa_dsl.Program``. **No ACT code is copied here**:
``Loop`` below is a four-field record that matches ACT's field *names* so the
two sides plug together, and ``nests()`` is a deliberately obvious stand-in
enumerator so the seam can be tested without ACT's environment (which does not
exist on this host -- see ``scripts/act-test-recipe.sh``).

Run it and it reports three things:

1. **The seam is real.** The nest that describes the tiling
   ``isa_dsl.gemm_program`` hardcodes re-emits that program *word for word*, at
   every shape in ``bench_isa.SHAPES`` and both relu settings.
2. **The seam is narrow, and the hardware says where.** Of the nests a
   loop-nest mapper would offer, most are refused, and the refusals are counted
   by cause. The binding one is not ``LOOP_DEPTH=4``; it is ``AGU_TERMS=3``,
   one instruction word's address-term budget, and it decides *which data-reuse
   choices exist* -- an m-tiled nest is encodable only if A is re-staged per
   m-tile, because keeping it resident would need a fourth term.
3. Every encodable nest is checked functionally against numpy
   (``isa_ref.run``), and ranked by the only pure-python proxy we have. The
   proxy is a regression over five cosim points, not a measurement.

Point 2 is why this file exists: a compiler-side search bumping into a hardware
parameter is what a co-design loop is supposed to surface.
"""

import itertools
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    NestError, Program, Ref,
)
from examples.accelerator.tinytpu_vitis import isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    A_VR, AR_C, B_SP, MAXDIM, MAXROWS, ProgramError, T, assemble, expand,
)

# ACT's `mapping.Loop`, by field name only. A `spatial` loop is Timeloop's
# fanout across instances; this machine has one array and no instance index, so
# a spatial loop is refused rather than silently serialised.
Loop = namedtuple("Loop", "rank factor level spatial")
Loop.__new__.__defaults__ = (False,)

DRAM, ARRAY = "dram", "array"
RANKS = ("M", "K", "N")


class Unencodable(NestError):
    """A legal nest this ISA cannot express, with the reason named first."""


# ------------------------------------------------------------------ the nest --
def split_body(nest):
    """ACT's ``_split_body``: the trailing intrinsic loops, and the rest.

    The innermost level is the tile one instruction performs by itself. For our
    ``mm`` that is ``rows`` activation rows against a TxT weight block, so the
    tail must be one loop per rank at the innermost level, with K = N = T and M
    free (``rows`` is a static field, so the mapper may pick it).
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


def check(nest, M, K, N):
    """ACT's constraint 1 (exact coverage) plus the one our machine adds."""
    extents = {"M": M, "K": K, "N": N}
    for loop in nest:
        if loop.rank not in extents:
            raise Unencodable(f"coverage: unknown rank {loop.rank!r}")
        if loop.spatial:
            raise Unencodable(
                f"spatial: loop {loop.rank!r} fans out across instances, and "
                f"this machine has one array and no instance index")
    for rank, extent in extents.items():
        got = 1
        for loop in nest:
            if loop.rank == rank:
                got *= loop.factor
        if got != extent:
            raise Unencodable(
                f"coverage: rank {rank} factors to {got}, extent {extent}")


# --------------------------------------------------------------- the emitter --
def gemm_from_nest(nest, M, K, N, relu=False):
    """Emit a TinyTPU-isa GEMM for ``nest``, or say why the nest is unencodable.

    What the nest decides is the order of the emitted loops and how each rank is
    split across them -- what a mapper chooses. What it does not decide is the
    machine's staging: B into the scratchpad one block per n-tile, A into the
    operand vregs one block per k-tile, one accumulator region drained per
    output tile.
    """
    if not (M <= MAXDIM and K <= MAXDIM and N <= MAXDIM):
        raise Unencodable(f"shape: {M}x{K}x{N} exceeds MAXDIM={MAXDIM}")
    check(nest, M, K, N)
    emitted, Mt = split_body(nest)

    ks = [i for i, l in enumerate(emitted) if l.rank == "K"]
    if len(ks) > 1:
        raise Unencodable(
            "acc-peel: K is split across two emitted loops, so the k=0 tile is "
            "not a peelable prefix -- `acc` is a static field and cannot be "
            "predicated on an induction variable")
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
    k = Program(f"act{'.relu' if relu else ''} {M}x{K}x{N} [{order}]")
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
        """A: one k-block per k-tile, into the operand vregs. Re-staged per
        m-tile -- keeping it resident across m would need a fourth AGU term on
        the accumulating `mm`."""
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
        with k.loop(lv.factor, f"{lv.rank.lower()}{len(ivs)}") as iv:
            nxt = ivs + [(lv.rank, iv, lv.factor)]
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


def canonical(M, K, N):
    """The nest describing the tiling ``isa_dsl.gemm_program`` hardcodes."""
    return (Loop("N", N // T, DRAM), Loop("K", K // T, DRAM),
            Loop("M", M, ARRAY), Loop("K", T, ARRAY), Loop("N", T, ARRAY))


def nests(M, K, N, slots=2):
    """The nests a two-level loop-nest mapper would offer for this GEMM.

    Each rank's residual after the intrinsic tile is split over ``slots``
    emitted slots and the result permuted -- the factorisations x permutations a
    mapper enumerates, with the innermost level pinned to the intrinsic. No
    spatial slots: one array, no instance index. Deliberately small and
    obvious; this is a stand-in, not a port of ACT's ``mapspace``.
    """
    def divisors(n):
        return [d for d in range(1, n + 1) if n % d == 0]

    def splits(n, parts):
        if parts == 1:
            return [(n,)]
        return [(d,) + r for d in divisors(n) for r in splits(n // d, parts - 1)]

    seen = set()
    for Mt in divisors(M):
        tail = (Loop("M", Mt, ARRAY), Loop("K", T, ARRAY), Loop("N", T, ARRAY))
        residual = {"M": M // Mt, "K": K // T, "N": N // T}
        for combo in itertools.product(*(splits(residual[r], slots)
                                         for r in RANKS)):
            factors = dict(zip(RANKS, combo))
            loops = [Loop(r, f, DRAM) for r in RANKS for f in factors[r] if f != 1]
            for perm in itertools.permutations(loops):
                nest = tuple(perm) + tail
                if nest not in seen:
                    seen.add(nest)
                    yield nest


# The cycle fit docs/source/designs/tinytpu_isa.rst reports over the five cosim
# points. A REGRESSION OVER FIVE MEASUREMENTS: use it to rank, never to quote.
def proxy_cycles(prog):
    return 74.5 + 21.70 * len(expand(prog))


def functional(prog, M, K, N, relu, seed=0):
    """True when the program computes ``relu(A @ B)``, by numpy."""
    rng = np.random.default_rng(seed)
    A = np.zeros((MAXDIM, MAXDIM), np.int8)
    B = np.zeros((MAXDIM, MAXDIM), np.int8)
    A[:M, :K] = rng.integers(-4, 5, (M, K))
    B[:K, :N] = rng.integers(-4, 5, (K, N))
    gold = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    got = isa_ref.run(prog, A.reshape(-1), B.reshape(-1),
                      np.zeros(MAXDIM * MAXDIM, np.int8))
    return np.array_equal(got.reshape(MAXDIM, MAXDIM)[:M, :N],
                          np.clip(gold, -128, 127).astype(np.int8))


def cause_of(exc):
    text = str(exc)
    if "address terms" in text:
        return "AGU_TERMS=3"
    if "loop stack" in text:
        return "LOOP_DEPTH=4"
    if isinstance(exc, ProgramError):
        return "microarchitecture: " + text.split(":")[0][:44]
    return text.split(":")[0]


def main():
    from examples.accelerator.tinytpu_vitis.bench_isa import SHAPES

    print("1. the seam: a nest re-emits the shipped program, word for word")
    from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program
    for (M, K, N) in SHAPES:
        for relu in (False, True):
            got = gemm_from_nest(canonical(M, K, N), M, K, N, relu)
            ref = gemm_program(M, K, N, relu)
            assert len(got) == len(ref), (
                f"{M}x{K}x{N} relu={relu}: {len(got)} instructions vs "
                f"{len(ref)}")
            for i, (g, r) in enumerate(zip(got, ref)):
                assert g == r, (
                    f"{M}x{K}x{N} relu={relu}: instruction {i} differs "
                    f"({g} vs {r})")
    print(f"   ok -- {len(SHAPES)} shapes x {{gemm, gemm.relu}}, bit-identical")

    M, K, N = 16, 16, 16
    print(f"\n2. the mapspace at {M}x{K}x{N}, and what the hardware refuses")
    ok, refused = [], {}
    for nest in nests(M, K, N):
        try:
            prog = gemm_from_nest(nest, M, K, N, relu=True)
            assemble(prog)
        except (NestError, ProgramError, AssertionError) as e:
            cause = cause_of(e)
            refused[cause] = refused.get(cause, 0) + 1
            continue
        ok.append((nest, prog))
    total = len(ok) + sum(refused.values())
    print(f"   {total} nests enumerated, {len(ok)} encodable")
    for cause, n in sorted(refused.items(), key=lambda kv: -kv[1]):
        print(f"     {n:5d}  {cause}")
    peel = refused.get("acc-peel", 0)
    print(f"\n   {peel} of {total} are refused by ONE static instruction "
          f"field: `acc`.\n   It has no predicate on an induction variable, so "
          f"the k=0 tile must be a\n   peelable prefix, which pins K innermost "
          f"and unsplit and kills every\n   permutation that moves it. The "
          f"3-term AGU is the next constraint\n   ({refused.get('AGU_TERMS=3', 0)} "
          f"nests), and LOOP_DEPTH=4 binds nothing here.")

    print("\n3. the survivors: numpy-checked, ranked by the proxy")
    rows = []
    for nest, prog in ok:
        emitted = ">".join(f"{l.rank}{l.factor}" for l in nest
                           if l.level == DRAM) or "-"
        rows.append((len(expand(prog)), emitted,
                     f"rows={[l.factor for l in nest if l.level == ARRAY and l.rank == 'M'][0]}",
                     len(prog), len(assemble(prog)),
                     functional(prog, M, K, N, True)))
    for dyn, emitted, mt, static, words, good in sorted(rows):
        print(f"   {emitted:14s} {mt:9s} {static:3d} instr {words:3d} words "
              f"{dyn:4d} dynamic  ~{74.5 + 21.70 * dyn:7.0f} cy  "
              f"{'correct' if good else 'WRONG'}")
    bad = [r for r in rows if not r[-1]]
    print(f"\n   {len(rows) - len(bad)}/{len(rows)} functionally correct "
          f"against numpy.")
    print("   The cycle column is 74.5 + 21.70/dynamic instruction, a "
          "regression over the\n   five cosim points in "
          "docs/source/designs/tinytpu_isa.rst. Only cosim.py measures.")
    assert not bad, f"{len(bad)} encodable nests compute the wrong thing"


if __name__ == "__main__":
    main()

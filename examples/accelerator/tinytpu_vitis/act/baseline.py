# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference submission: one hand-chosen mapping per corpus spec.

It is what ACT has to beat, and what the judge is calibrated on. Prose:
docs/source/extensions/act_specs.rst."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_dsl import Program, Ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    AR_RAW_DIST, A_VR, AR_C, B_SP, MAXDIM, T,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402


class Unsupported(Exception):
    """A spec this submission has no mapping for."""


def tiles(n):
    return (n + T - 1) // T


def array_rows(m):
    """`mm` rows, raised to the accumulator's RAW distance when M is below it."""
    return max(m, AR_RAW_DIST)


def load_activations(p, k, rows):
    with p.loop(tiles(k), "kA") as kb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=rows)


def load_weights(p, n, rows):
    with p.loop(tiles(n), "nB") as nb:
        p.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=rows)


def contract_k(p, k, peeled_weights, looped_weights, rows):
    """The peeled first k-tile, which overwrites `AR_C`, then the rest."""
    p.mm(A_VR, AR_C, peeled_weights, rows=rows)
    if tiles(k) > 1:
        with p.loop(tiles(k) - 1, "k") as kb:
            p.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                 looped_weights(kb), rows=rows, acc=True)


def epilogue(p, sp, ar, rows):
    if "relu" in sp["epilogue"]:
        p.vrelu(ar, ar, rows=rows)


def gemm(sp):
    d = sp["dims"]
    m, k, n = d["m"], d["k"], d["n"]
    rows = array_rows(m)
    p = Program(sp["name"])
    load_activations(p, k, rows)
    load_weights(p, n, tiles(k) * T)
    with p.loop(tiles(n), "n") as nb:
        contract_k(p, k, Ref(B_SP).at(nb, MAXDIM),
                   lambda kb: Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows)
        epilogue(p, sp, AR_C, rows)
        p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=m)
    return p.emit()


def batched_gemm(sp):
    d = sp["dims"]
    b, m, k, n = d["b"], d["m"], d["k"], d["n"]
    if tiles(k) > 1 or tiles(n) > 1:
        raise Unsupported(
            f"{sp['name']}: a batch loop over {tiles(k)} k-tiles and "
            f"{tiles(n)} n-tiles needs more than AGU_TERMS terms on one mm")
    rows = array_rows(m)
    p = Program(sp["name"])
    load_activations(p, k, b * rows)
    load_weights(p, n, b * tiles(k) * T)
    with p.loop(b, "b") as ib:
        acc = Ref(AR_C).at(ib, m)
        p.mm(Ref(A_VR).at(ib, m), acc, Ref(B_SP).at(ib, k), rows=rows)
        epilogue(p, sp, acc, rows)
        p.mvout(acc, dram_row=Ref().at(ib, m), col_block=0, rows=m)
    return p.emit()


def pointwise(sp):
    """No contraction: `mm` against the host-placed identity is the only way in."""
    d = sp["dims"]
    m, n = d["m"], d["n"]
    rows = array_rows(m)
    p = Program(sp["name"])
    load_activations(p, n, rows)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    with p.loop(tiles(n), "n") as nb:
        p.mm(Ref(A_VR).at(nb, MAXDIM), AR_C, B_SP, rows=rows)
        epilogue(p, sp, AR_C, rows)
        p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=m)
    return p.emit()


def row_reduce(sp):
    """One weight tile, the host-placed ones column, reused by every k-tile."""
    d = sp["dims"]
    m, k = d["m"], d["k"]
    rows = array_rows(m)
    p = Program(sp["name"])
    load_activations(p, k, rows)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    contract_k(p, k, B_SP, lambda kb: B_SP, rows)
    p.mvout(AR_C, dram_row=0, col_block=0, rows=m)
    return p.emit()


MAPPINGS = {"mk,kn->mn": gemm, "bmk,bkn->bmn": batched_gemm,
            "mn->mn": pointwise, "mk->m": row_reduce}


def program(sp):
    """The submission for one spec: the `(word, agu_word)` list `assemble` takes."""
    spec_mod.validate(sp)
    for t in sp["inputs"] + [sp["output"]] + sp.get("constants", []):
        if list(t["origin"]) != [0, 0]:
            raise Unsupported(
                f"{sp['name']}: {t['name']} is declared at origin "
                f"{t['origin']}; this submission only maps tensors at [0, 0]")
    build = MAPPINGS.get(sp["einsum"])
    if build is None:
        raise Unsupported(
            f"{sp['name']}: no mapping for einsum {sp['einsum']!r}; this "
            f"submission maps {sorted(MAPPINGS)}")
    return build(sp)


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.microarch_isa import expand
    for sp in spec_mod.corpus():
        try:
            p = program(sp)
            print(f"  {sp['name']:28s} {len(p):3d} static -> "
                  f"{len(expand(p)):3d} dynamic instructions")
        except Unsupported as e:
            print(f"  {sp['name']:28s} UNSUPPORTED: {e}")

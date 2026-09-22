# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Several legal mappings of one spec, so the cheap gate can be tested on rank.

A gate does not need the right cycle count; it needs the right order. These are
the alternatives a mapper would actually consider -- a different loop order, a
different residency for A, a different output blocking -- all bit-exact, all
legal, and measurably different in cost. Prose:
docs/source/extensions/act_specs.rst."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_dsl import Program, Ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    A_VR, AR_C, B_SP, MAXDIM, T,
)
from examples.accelerator.tinytpu_vitis.act import baseline  # noqa: E402

A_SP = MAXDIM * MAXDIM // T


def gemm_dims(sp):
    if sp["einsum"] != "mk,kn->mn":
        raise baseline.Unsupported(
            f"{sp['name']}: the variants are GEMM mappings only")
    d = sp["dims"]
    return d["m"], d["k"], d["n"]


def a_through_the_scratchpad(sp):
    """A lands in the scratchpad and is copied into the vregs by `vld`, which
    is what the design did before weights became scratchpad-addressed."""
    m, k, n = gemm_dims(sp)
    rows = baseline.array_rows(m)
    p = Program(sp["name"] + " / A via spad")
    with p.loop(baseline.tiles(k), "kA") as kb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 spad=Ref(A_SP).at(kb, MAXDIM), rows=rows)
        p.vld(Ref(A_VR).at(kb, MAXDIM), Ref(A_SP).at(kb, MAXDIM), rows=rows)
    baseline.load_weights(p, n, baseline.tiles(k) * T)
    with p.loop(baseline.tiles(n), "n") as nb:
        baseline.contract_k(p, k, Ref(B_SP).at(nb, MAXDIM),
                            lambda kb: Ref(B_SP + T).at(nb, MAXDIM).at(kb, T),
                            rows)
        baseline.epilogue(p, sp, AR_C, rows)
        p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=m)
    return p.emit()


def row_blocked(sp, block=T):
    """The output is walked in row blocks, with the k loop unrolled because a
    row-block term plus a k term plus an n term is four AGU terms, not three."""
    m, k, n = gemm_dims(sp)
    if m % block or m // block < 2:
        raise baseline.Unsupported(
            f"{sp['name']}: {m} rows do not split into blocks of {block}")
    rows = baseline.array_rows(block)
    p = Program(sp["name"] + f" / row blocks of {block}")
    with p.loop(baseline.tiles(k), "kA") as kb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=m)
    baseline.load_weights(p, n, baseline.tiles(k) * T)
    with p.loop(baseline.tiles(n), "n") as nb:
        with p.loop(m // block, "mb") as mb:
            acc = Ref(AR_C).at(mb, block)
            for kt in range(baseline.tiles(k)):
                p.mm(Ref(A_VR + kt * MAXDIM).at(mb, block), acc,
                     Ref(B_SP + kt * T).at(nb, MAXDIM), rows=rows,
                     acc=kt > 0)
            baseline.epilogue(p, sp, acc, rows)
            p.mvout(acc, dram_row=Ref().at(mb, block),
                    col_block=Ref().at(nb, 1), rows=block)
    return p.emit()


def weights_reloaded_per_output(sp):
    """No weight residency: each output column block re-fetches its weights
    from DRAM right before using them, which is the mapping a scheduler with
    no reuse model produces."""
    m, k, n = gemm_dims(sp)
    rows = baseline.array_rows(m)
    p = Program(sp["name"] + " / weights reloaded per output tile")
    with p.loop(baseline.tiles(k), "kA") as kb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=rows)
    with p.loop(baseline.tiles(n), "n") as nb:
        p.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1), spad=B_SP,
                 rows=baseline.tiles(k) * T)
        baseline.contract_k(p, k, B_SP, lambda kb: Ref(B_SP + T).at(kb, T),
                            rows)
        baseline.epilogue(p, sp, AR_C, rows)
        p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=m)
    return p.emit()


VARIANTS = {"reference": baseline.program,
            "a_via_spad": a_through_the_scratchpad,
            "row_blocked": row_blocked,
            "weights_reloaded": weights_reloaded_per_output}


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.act import cycles
    from examples.accelerator.tinytpu_vitis.act import spec as spec_mod
    for name in (sys.argv[1:] or ["gemm_16x16x16"]):
        sp = spec_mod.by_name(name)
        print(f"{name}:")
        for tag, make in VARIANTS.items():
            try:
                print(f"  {tag:18s} {cycles.report(make(sp))}")
            except baseline.Unsupported as e:
                print(f"  {tag:18s} UNSUPPORTED: {e}")

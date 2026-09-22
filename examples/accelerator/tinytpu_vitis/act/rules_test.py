# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every rejection reaches a named rule with a remedy, not a generic fallback.

The ISA fixtures are `stress_isa._bad_programs`, one crafted violation per
contract; the rest are built here because no shipped program breaks them."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_dsl import Program, Ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    AGU_F3, A_VR, AR_C, B_SP, DMA_TO_VR, LOOP_DEPTH, MAXDIM, T, enc, enc_agu,
    OP_DMA_LD, OP_ENDLOOP, OP_LOOP, OP_MM, OP_MVOUT, OP_NOP,
)
from examples.accelerator.tinytpu_vitis.stress_isa import _bad_programs  # noqa: E402
from examples.accelerator.tinytpu_vitis.act import baseline, legality  # noqa: E402
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402

FALLBACKS = ("isa.contract", "encoding.limit")


def unrolled_past_imem():
    p = Program("unrolled past imem")
    p.dma_ld(src=0, dram_row=0, col_block=0, vr=A_VR, rows=T)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    for _ in range(MAXDIM * MAXDIM):
        p.mm(A_VR, AR_C, B_SP, rows=T)
    p.mvout(AR_C, dram_row=0, col_block=0, rows=T)
    return p.emit()


def rows_padded_into_c(sp):
    """M=6 rounded up to 8 everywhere, including the `mvout`."""
    d = sp["dims"]
    padded = (d["m"] + T - 1) // T * T
    p = Program("rows padded into C")
    with p.loop(d["k"] // T, "kA") as kb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=padded)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=d["k"])
    with p.loop(d["n"] // T, "n") as nb:
        p.mm(A_VR, AR_C, B_SP, rows=padded)
        p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=padded)
    return p.emit()


def loop_nest_over(instruction, trips=127):
    """A two-deep nest whose dynamic work overflows a header count while the
    static program stays well inside imem."""
    return [(enc(OP_DMA_LD, f0=DMA_TO_VR, nr=T), 0),
            (enc(OP_DMA_LD, f0=1, nr=T), 0),
            (enc(OP_MM, nr=T), 0),
            (enc(OP_LOOP, nr=trips), 0), (enc(OP_LOOP, nr=trips), 0),
            instruction,
            (enc(OP_ENDLOOP), 0), (enc(OP_ENDLOOP), 0)]


def operand_past_dram():
    return [(enc(OP_DMA_LD, f0=0, f1=MAXDIM - 2, f3=0, nr=T), 0)]


def field_past_encoding():
    return [(enc(OP_LOOP, nr=2), 0),
            (enc(OP_DMA_LD, f0=DMA_TO_VR, nr=T),
             enc_agu((AGU_F3, 0, 2047), (AGU_F3, 0, 2))),
            (enc(OP_ENDLOOP), 0)]


def nest_past_loop_stack():
    return ([(enc(OP_LOOP, nr=2), 0)] * (LOOP_DEPTH + 1)
            + [(enc(OP_NOP), 0)]
            + [(enc(OP_ENDLOOP), 0)] * (LOOP_DEPTH + 1))


def endloop_with_nothing_open():
    return [(enc(OP_NOP), 0), (enc(OP_ENDLOOP), 0)]


def mm_acc_field_out_of_range():
    return [(enc(OP_DMA_LD, f0=DMA_TO_VR, nr=T), 0),
            (enc(OP_DMA_LD, f0=1, nr=T), 0),
            (enc(OP_MM, f2=2, nr=T), 0)]


def output_left_unwritten(sp):
    p = Program("output left unwritten")
    p.dma_ld(src=0, dram_row=0, col_block=0, vr=A_VR, rows=sp["dims"]["m"])
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    p.mm(A_VR, AR_C, B_SP, rows=sp["dims"]["m"])
    p.mvout(AR_C, dram_row=0, col_block=0, rows=sp["dims"]["m"])
    return p.emit()


def cases():
    gemm44 = spec_mod.by_name("gemm_4x4x4")
    rows68 = spec_mod.by_name("gemm_rows_6x8x8")
    wide = spec_mod.by_name("gemm_16x16x16")
    out = [(f"isa: {name}", None, prog)
           for name, prog in _bad_programs().items()]
    out += [("isa: empty program", None, []),
            ("isa: operand load past the last DRAM row", None,
             operand_past_dram()),
            ("isa: AGU pushes a field past the encoding", None,
             field_past_encoding()),
            ("isa: nest deeper than the loop stack", None,
             nest_past_loop_stack()),
            ("isa: endloop with nothing open", None,
             endloop_with_nothing_open()),
            ("isa: mm f2 neither overwrite nor accumulate", None,
             mm_acc_field_out_of_range()),
            ("encoding: unrolled past imem", gemm44, unrolled_past_imem()),
            ("encoding: looped past the array's header counts", None,
             loop_nest_over((enc(OP_MM, nr=T), 0))),
            ("encoding: looped past a unit's header count", None,
             loop_nest_over((enc(OP_MVOUT, nr=T), 0))),
            ("spec: rows padded into C", rows68, rows_padded_into_c(rows68)),
            ("spec: output left unwritten", wide, output_left_unwritten(wide))]
    return out


def main():
    fails = []
    hit = {}
    for name, sp, prog in cases():
        r = legality.check(sp, prog)
        if r is None:
            fails.append(f"ACCEPTED a program that breaks a contract: {name}")
            continue
        if r.rule.name in FALLBACKS:
            fails.append(f"{name}: fell through to {r.rule.name}, no remedy -- "
                         f"reason was: {r.reason}")
            continue
        if not r.rule.remedy or not r.rule.statement:
            fails.append(f"{name}: rule {r.rule.name} has no remedy")
        hit[r.rule.name] = hit.get(r.rule.name, 0) + 1
    for sp in spec_mod.corpus():
        if spec_mod.fits_build(sp) is not None:
            continue
        try:
            prog = baseline.program(sp)
        except baseline.Unsupported:
            continue
        r = legality.check(sp, prog)
        if r is not None:
            fails.append(f"REJECTED the baseline submission for {sp['name']}:\n{r}")
    unreached = sorted(set(legality.ALL_RULES) - set(hit))
    print(f"  {len(cases())} crafted violations -> {len(hit)} distinct rules, "
          f"every one with a remedy")
    print(f"  rules no fixture reaches: {unreached or 'none'}")
    for line in fails:
        print("  FAIL " + line)
    print("  RULES OK" if not fails else "  RULES FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

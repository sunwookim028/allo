# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price the deficit with PROGRAMS, on ONE unchanged RTL build.

The design is instruction-programmable, so a whole class of attribution
question can be asked without touching `microarch_isa.py` at all: the workload
is data, so two programs that differ in exactly one work count run on the
*identical netlist*, and the cycle difference is that work count's price. No
re-synthesis, no resource delta, no clock delta, nothing to disclose.

    python pyrun.py prog_probe.py            # the default probe set
    python pyrun.py prog_probe.py --list     # names only, no Vitis

Each probe is `(name, program, A, B, gold-or-None)`. `gold` comes from
`isa_ref.run`, so a probe that is meant to compute something is still checked
bit-exact; a probe that is deliberately nonsense (an inflated burst span over
rows the reference does read) is checked too, because `isa_ref` executes the
same resolved stream the sequencer does.

`csynth_design` runs ONCE and every probe re-runs `cosim_design` on that
solution, exactly as `cosim.py` sweeps shapes.
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from allo.dataflow import customize  # noqa: E402
from examples.accelerator.tinytpu_vitis import cosim as C  # noqa: E402
from examples.accelerator.tinytpu_vitis import isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, assemble, expand, schedule, MAXDIM, T, WPR, IMEM_SIZE,
    A_VR, AR_C, B_SP, OP_DMA_LD, OP_MM,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import Program, Ref, gemm_program  # noqa: E402

PRJ = os.path.abspath(os.environ.get("TPU_PRJ", "probe.prj"))


def operands(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, MAXDIM * MAXDIM).astype(np.int8)
    B = rng.integers(-4, 5, MAXDIM * MAXDIM).astype(np.int8)
    return A, B


# --------------------------------------------------------------- programs ---
def p_gemm(M, K, N, relu=False):
    return gemm_program(M, K, N, relu)


def p_loads_only(M, K, N):
    """The shipped program's two load loops and nothing else: prices the
    serial load prefix on its own (no `mm`, no `mvout`, so `accu` and the
    array do no work at all)."""
    Kt, Nt = K // T, N // T
    k = Program(f"loads {M}x{K}x{N}")
    with k.loop(Kt, "kA") as kb:
        k.dma_ld(src=0, dram_row=0, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=M)
    with k.loop(Nt, "nB") as nb:
        k.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=K)
    return k.emit()


def p_oneblock(M, K, N):
    """The shipped program with the A and B loads collapsed to ONE column
    block each, every `mm` pointed at that one block. `accu`, the array,
    `dma_st` and the instruction stream all do the same work as the real
    program; only `dma_ld`'s row count falls, from `Kt*M + Nt*K` to `M + K`.
    The answer is wrong on purpose -- `isa_ref` computes the same wrong answer,
    so it is still checked bit-exact."""
    Kt, Nt = K // T, N // T
    k = Program(f"oneblock {M}x{K}x{N}")
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=A_VR, rows=M)
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=K)
    with k.loop(Nt, "n") as nb:
        k.mm(A_VR, AR_C, B_SP, rows=M, acc=False)
        if Kt > 1:
            with k.loop(Kt - 1, "k"):
                k.mm(A_VR, AR_C, B_SP, rows=M, acc=True)
        k.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=M)
    return k.emit()


def p_span(M, K, N, a_top=False, b_top=False):
    """The shipped program with the A and/or B loads moved to the LAST rows of
    the operand, which pushes `dma_ld`'s burst span from `M` (or `K`) up to
    `MAXDIM` while changing no other work count. Prices the burst per word."""
    Kt, Nt = K // T, N // T
    ar, br = (MAXDIM - M if a_top else 0), (MAXDIM - K if b_top else 0)
    k = Program(f"span {M}x{K}x{N} a_top={a_top} b_top={b_top}")
    with k.loop(Kt, "kA") as kb:
        k.dma_ld(src=0, dram_row=ar, col_block=Ref().at(kb, 1),
                 vr=Ref(A_VR).at(kb, MAXDIM), rows=M)
    with k.loop(Nt, "nB") as nb:
        k.dma_ld(src=1, dram_row=br, col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=K)
    with k.loop(Nt, "n") as nb:
        k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM), rows=M, acc=False)
        if Kt > 1:
            with k.loop(Kt - 1, "k") as kb:
                k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                     Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)
        k.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=M)
    return k.emit()


def p_interleaved(M, K, N, relu=False):
    """THE CANDIDATE. Same instructions, same work, different ORDER: each
    operand column block is loaded at the point it is first needed instead of
    all of them before the first `mm`, so `dma_ld`'s row loop overlaps the
    array instead of running ahead of it.

    B's first block and A's first block are the only loads that precede the
    first `mm`; every later block is issued between two `mm`s. The n loop's
    first iteration is peeled because a hardware loop has no predicate: the A
    blocks must be loaded on the first pass over k and not on the others.
    """
    Kt, Nt = K // T, N // T
    k = Program(f"interleaved gemm{'.relu' if relu else ''} {M}x{K}x{N}")

    # --- peeled n = 0: the A blocks are loaded here, inside the k loop ---
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

    # --- n = 1 .. Nt-1: A is resident, B's block is loaded per iteration ---
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


# ------------------------------------------------------------- the harness ---
def p_b_per_tile(M, K, N, relu=False):
    """A hoisted as shipped, B's column block loaded inside the n loop. Keeps
    `vru`'s A words in one uninterrupted run (so the array is never starved by
    a load) while removing `Nt-1` of B's blocks from the serial head."""
    Kt, Nt = K // T, N // T
    k = Program(f"b_per_tile {M}x{K}x{N}")
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


def p_kouter(M, K, N, relu=False):
    """THE CANDIDATE: k outermost, n innermost, A's vreg block overwritten in
    place, and the `mvout`s folded into the last k-tile.

    Same instructions and the same work count as the shipped program, in a
    different order. Three things fall out of the loop interchange:

      * one A column block is loaded per k-tile and then used by ALL Nt `mm`s
        of that tile, so `vru` spends one load word per Nt activation words
        instead of loading every block before the first `mm`;
      * with k outermost a block is live for one k-tile only, so every A load
        targets the SAME vreg rows. That is what makes the nest fit the
        hardware: the inner `mm` then needs three AGU terms (`ar` by n, and
        the weights by n and by k) rather than four, and AGU_TERMS is 3;
      * the last k-tile is peeled so each `mvout` sits next to the `mm` that
        completes its tile, which keeps `accu`'s retire work interleaved with
        its accumulate work instead of leaving it as a tail.

    n outermost cannot do the first of these: every A block is needed inside
    the first n iteration, so the loads either all precede the first `mm` (the
    shipped program) or sit one-for-one between `mm`s (`p_interleaved`).
    """
    Kt, Nt = K // T, N // T
    assert Nt * M <= 128, "Nt*M accumulator rows must fit NAR"
    k = Program(f"kouter gemm{'.relu' if relu else ''} {M}x{K}x{N}")

    def out_tile(nb):
        if relu:
            k.vrelu(Ref(AR_C).at(nb, M), Ref(AR_C).at(nb, M), rows=M)
        k.mvout(Ref(AR_C).at(nb, M), dram_row=0,
                col_block=Ref().at(nb, 1), rows=M)

    # --- k = 0, peeled: acc=False, and B's Nt blocks are loaded here ---
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=A_VR, rows=M)
    with k.loop(Nt, "n0") as nb:
        k.dma_ld(src=1, dram_row=0, col_block=Ref().at(nb, 1),
                 spad=Ref(B_SP).at(nb, MAXDIM), rows=K)
        k.mm(A_VR, Ref(AR_C).at(nb, M), Ref(B_SP).at(nb, MAXDIM),
             rows=M, acc=False)
        if Kt == 1:
            out_tile(nb)
    if Kt == 1:
        return k.emit()

    # --- k = 1 .. Kt-2: accumulate only ---
    if Kt > 2:
        with k.loop(Kt - 2, "k") as kb:
            k.dma_ld(src=0, dram_row=0, col_block=Ref(1).at(kb, 1),
                     vr=A_VR, rows=M)
            with k.loop(Nt, "n") as nb:
                k.mm(A_VR, Ref(AR_C).at(nb, M),
                     Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)

    # --- k = Kt-1, peeled: the last accumulate and the retire, together ---
    k.dma_ld(src=0, dram_row=0, col_block=Kt - 1, vr=A_VR, rows=M)
    with k.loop(Nt, "nL") as nb:
        k.mm(A_VR, Ref(AR_C).at(nb, M),
             Ref(B_SP + (Kt - 1) * T).at(nb, MAXDIM), rows=M, acc=True)
        out_tile(nb)
    return k.emit()


def counts(prog):
    w = assemble(prog)
    na, nb = w[7] & 0xFFFF, (w[7] >> 16) & 0xFFFF
    return dict(static=len(prog), dyn=len(list(expand(prog))),
                dld=w[1], spm=w[2], vru=w[3], accu=w[5], dst=w[6],
                burst=(na + nb) * WPR)


def tb_for(prog, A, B, gold):
    words = assemble(prog)
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    n = MAXDIM * MAXDIM
    chk = ("" if gold is None else f"""
  for (int i = 0; i < {n}; i++) if (C[i] != gold[i]) bad++;""")
    src = ["#include <cstdio>\n#include <cstdint>\n",
           'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
           C.carr("imem", imem, "uint64_t"),
           C.carr("A", A, "int8_t"), C.carr("B", B, "int8_t"),
           C.carr("C0", np.zeros(n, np.int8), "int8_t"),
           ("" if gold is None else C.carr("gold", gold, "int8_t")),
           f"static alignas(64) int8_t C[{n}];\n",
           f"""
int main() {{
  int bad = 0;
  for (int i = 0; i < {n}; i++) C[i] = C0[i];
  tinytpu_isa(imem, A, B, C);{chk}
  printf("PROBE mismatches = %d\\n", bad);
  return bad == 0 ? 0 : 1;
}}
"""]
    return "".join(src)


def probes():
    """(name, program, A, B, gold-or-None). `gold=None` only where the program
    is a pure timing probe."""
    A, B = operands()
    out = []

    def add(name, prog, checked=True):
        gold = isa_ref.run(prog, A, B, np.zeros(MAXDIM * MAXDIM, np.int8)) \
            if checked else None
        out.append((name, prog, A, B, gold))

    sel = os.environ.get("TPU_PROBE_SHAPES", "4x4x4,16x16x16")
    shapes = [tuple(int(x) for x in t.split("x")) for t in sel.split(",")]
    which = os.environ.get("TPU_PROBE_SET", "attrib").split(",")
    builders = {
        "gemm": p_gemm, "loads_only": p_loads_only, "oneblock": p_oneblock,
        "span_a_top": lambda *s: p_span(*s, a_top=True),
        "span_both_top": lambda *s: p_span(*s, a_top=True, b_top=True),
        "interleaved": p_interleaved, "b_per_tile": p_b_per_tile,
        "kouter": p_kouter,
    }
    sets = {"attrib": ["gemm", "loads_only", "oneblock", "span_a_top",
                       "span_both_top", "interleaved"],
            "order": ["gemm", "interleaved", "b_per_tile", "kouter"],
            "final": ["gemm", "kouter"]}
    names = sets.get(which[0], which)
    for (M, K, N) in shapes:
        tag = f"{M}x{K}x{N}"
        for nm in names:
            add(f"{nm} {tag}", builders[nm](M, K, N))
    return out


def main():
    ps = probes()
    if "--list" in sys.argv:
        for name, prog, *_ in ps:
            print(f"  {name:28s} {counts(prog)}")
        return 0

    syn = os.path.join(PRJ, "out.prj/solution1/syn/report/csynth.rpt")
    if os.environ.get("TPU_REUSE_SYN") == "1" and os.path.exists(syn):
        print(f"  reusing the csynth already in {PRJ}", flush=True)
    else:
        s = customize(tinytpu_isa)
        schedule(s)
        s.build(target="vitis_hls", mode="csyn", project=PRJ, wrap_io=False,
                configs={"align_value": 64})
        C.patch_axi_depths(PRJ)
        open(os.path.join(PRJ, "tb.cpp"), "w").write(tb_for(*ps[0][1:2], *ps[0][2:]))
        print(f"  synthesizing once into {PRJ} ...", flush=True)
        C.vitis(PRJ, C.TCL_SYN, "csynth.log")
        # Vitis has been seen to die seconds into set_part when several agents
        # run it at once; a silent failure here would make every probe below
        # report None, so it is an error rather than a row of blanks.
        assert os.path.exists(syn), (
            f"csynth produced no report; see {PRJ}/csynth.log")

    rpt = os.path.join(PRJ, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
    res = {}
    for name, prog, A, B, gold in ps:
        open(os.path.join(PRJ, "tb.cpp"), "w").write(tb_for(prog, A, B, gold))
        if os.path.exists(rpt):
            os.remove(rpt)
        log = "cosim_" + re.sub(r"\W+", "_", name) + ".log"
        text = C.vitis(PRJ, C.TCL_COSIM, log)
        n = C.cycles(PRJ)
        mm = [l.strip() for l in text.splitlines() if "PROBE mismatches" in l]
        ok = bool(mm) and mm[-1].endswith("= 0")
        res[name] = n
        c = counts(prog)
        print(f"  {name:28s} cycles={n}   "
              f"dld={c['dld']:4d} spm={c['spm']:4d} vru={c['vru']:4d} "
              f"accu={c['accu']:4d} burst={c['burst']:4d} dyn={c['dyn']:3d}"
              f"   {'exact' if ok else (mm[-1] if mm else 'UNCHECKED')}",
              flush=True)
    print("\n  summary")
    for k, v in res.items():
        print(f"    {k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

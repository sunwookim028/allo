# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programs that pass every cheap check and then do not complete in RTL.

A parametric family around the two the judge found, so the fault can be
bisected rather than described. Every case here is accepted by
`check_program`, reports no deadlock in `kpn_model`, and agrees bit-exactly
with `isa_ref` on the Allo simulator; what differs is whether Vitis cosim
finishes. Prose: docs/source/developer/limitations.rst.

    python act/rtl_hang.py check                 the cheap checks only, seconds
    TPU_PRJ=... python act/rtl_hang.py cosim     one csynth, then one cosim each
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu import isa_ref, kpn_model  # noqa: E402
from examples.tinytpu.isa_dsl import Program, Ref  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    A_VR, AR_C, B_SP, MAXDIM, T, check_program,
)
from examples.tinytpu.act import baseline, cycles  # noqa: E402
from examples.tinytpu.act import spec as spec_mod  # noqa: E402


def tiled(n_out, n_block, n_reduce, rows):
    """`n_out` output tiles x `n_block` row blocks, each a `n_reduce`-deep
    accumulation of `rows` rows followed by an `mvout`. Fully unrolled: the
    hardware loop is not involved in the fault."""
    p = Program(f"tiled n{n_out} b{n_block} k{n_reduce} r{rows}")
    for kt in range(n_reduce):
        p.dma_ld(src=0, dram_row=0, col_block=kt,
                 vr=A_VR + kt * MAXDIM, rows=n_block * rows)
    for nb in range(n_out):
        p.dma_ld(src=1, dram_row=0, col_block=nb,
                 spad=B_SP + nb * MAXDIM, rows=n_reduce * T)
    for nb in range(n_out):
        for mb in range(n_block):
            ar = AR_C + mb * rows
            for kt in range(n_reduce):
                p.mm(A_VR + kt * MAXDIM + mb * rows, ar,
                     B_SP + nb * MAXDIM + kt * T, rows=rows, acc=kt > 0)
            p.mvout(ar, dram_row=mb * rows, col_block=nb, rows=rows)
    return p.emit()


def pointwise(n_out, rows, relu):
    """`mm` against a host-placed identity, optionally with `vrelu`, once per
    output tile. Fully unrolled."""
    p = Program(f"pointwise n{n_out} r{rows} relu={relu}")
    for nb in range(n_out):
        p.dma_ld(src=0, dram_row=0, col_block=nb, vr=A_VR + nb * MAXDIM,
                 rows=rows)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    for nb in range(n_out):
        p.mm(A_VR + nb * MAXDIM, AR_C, B_SP, rows=rows)
        if relu:
            p.vrelu(AR_C, AR_C, rows=rows)
        p.mvout(AR_C, dram_row=0, col_block=nb, rows=rows)
    return p.emit()


def family():
    """The bisection. The two at the top are what the judge found, reduced to
    unrolled form; the rest halve one knob at a time."""
    return [
        ("tiled n2 b2 k2 r4", tiled(2, 2, 2, 4)),
        ("tiled n1 b2 k2 r4", tiled(1, 2, 2, 4)),
        ("tiled n2 b1 k2 r4", tiled(2, 1, 2, 4)),
        ("tiled n1 b1 k2 r4", tiled(1, 1, 2, 4)),
        ("tiled n2 b2 k1 r4", tiled(2, 2, 1, 4)),
        ("tiled n2 b2 k2 r8", tiled(2, 2, 2, 8)),
        ("pointwise n4 r16 relu", pointwise(4, 16, True)),
        ("pointwise n1 r16 relu", pointwise(1, 16, True)),
        ("pointwise n4 r16 norelu", pointwise(4, 16, False)),
        ("pointwise n4 r4 relu", pointwise(4, 4, True)),
    ]


def contrast():
    """Programs that are not in the bisection but bound it: the shipped
    mappings whose RTL does complete, including one with a `vrelu` in a
    hardware loop and two that stage a transfer inside the output nest."""
    from examples.tinytpu.act import variants
    return [(name, spec_mod.by_name(spec), make(spec_mod.by_name(spec)))
            for name, spec, make in (
                ("gemm_relu_16x16x16 shipped", "gemm_relu_16x16x16",
                 baseline.program),
                ("gemm_8x8x8 weights in-nest", "gemm_8x8x8",
                 variants.weights_reloaded_per_output),
                ("gemm_16x16x16 weights in-nest", "gemm_16x16x16",
                 variants.weights_reloaded_per_output))]


def host_spec():
    """A spec whose declared output is the whole of `C`, so the judge's
    testbench compares every byte and the family may write anywhere in it."""
    return spec_mod.by_name("relu_16x16")


def cheap_checks(prog, mod=None):
    """Everything short of RTL: the validator, the protocol model, and the
    simulator against `isa_ref` when a build is given."""
    out = []
    check_program(prog)
    out.append("check_program ACCEPTS")
    good, report = kpn_model.run(prog)
    lo = next((d for d in (1, 2, 3, 4, 8) if kpn_model.run(prog, d)[0]), None)
    out.append(f"kpn_model {'OK' if good else 'DEADLOCK ' + str(report)}, "
               f"minimum depth {lo}")
    if mod is not None:
        from examples.tinytpu.stress_isa import execute
        A, B, C0 = spec_mod.buffers(host_spec(), "full", 900)
        want = isa_ref.run(prog, A, B, C0)
        got = execute(mod, prog, A, B, C0)
        bad = int((np.asarray(got) != np.asarray(want)).sum())
        out.append(f"simulator vs isa_ref {'BIT-EXACT' if not bad else str(bad) + ' wrong'}")
    return "; ".join(out)


def main(argv):
    mode = argv[0] if argv else "check"
    mod = None
    if mode == "cosim" or "--sim" in argv:
        from examples.tinytpu.act import correctness
        mod = correctness.build_module()
    jobs = [(name, host_spec(), prog) for name, prog in family()]
    if "--contrast" in argv:
        jobs += contrast()
    rows = [(name, sp, prog, cheap_checks(prog, mod)) for name, sp, prog in jobs]
    for name, _, prog, line in rows:
        print(f"  {name:28s} {len(prog):3d} instrs; {cycles.report(prog)}")
        print(f"  {'':28s} {line}")
    if mode != "cosim":
        return 0
    from examples.tinytpu.act import measure
    prj = measure.PRJ
    print(f"  synthesizing once into {prj} ...", flush=True)
    measure.synthesize(prj)
    for name, sp, prog, _ in rows:
        n, line = measure.measure(sp, prog, prj, tag=name.replace(" ", "_"))
        print(f"  {name:28s} cosim={n}   {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

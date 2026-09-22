# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four runs that bracket the `relu_16x16` RTL hang.

One csynth, then the submission as generated, the same program with its single
`vrelu` deleted, the same program with the loop unrolled, and
`gemm_relu_16x16x16` for contrast. Prose:
docs/source/extensions/act_specs.rst, "What the expensive judge found"."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.isa_dsl import Program, Ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    A_VR, AR_C, B_SP, MAXDIM, T,
)
from examples.accelerator.tinytpu_vitis.act import (  # noqa: E402
    baseline, cycles, measure,
)
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402


def pointwise(sp, relu, unrolled):
    m, n = sp["dims"]["m"], sp["dims"]["n"]
    p = Program(f"{sp['name']} relu={relu} unrolled={unrolled}")
    with p.loop(n // T, "load") as nb:
        p.dma_ld(src=0, dram_row=0, col_block=Ref().at(nb, 1),
                 vr=Ref(A_VR).at(nb, MAXDIM), rows=m)
    p.dma_ld(src=1, dram_row=0, col_block=0, spad=B_SP, rows=T)
    if unrolled:
        for nb in range(n // T):
            p.mm(A_VR + nb * MAXDIM, AR_C, B_SP, rows=m)
            if relu:
                p.vrelu(AR_C, AR_C, rows=m)
            p.mvout(AR_C, dram_row=0, col_block=nb, rows=m)
    else:
        with p.loop(n // T, "n") as nb:
            p.mm(Ref(A_VR).at(nb, MAXDIM), AR_C, B_SP, rows=m)
            if relu:
                p.vrelu(AR_C, AR_C, rows=m)
            p.mvout(AR_C, dram_row=0, col_block=Ref().at(nb, 1), rows=m)
    return p.emit()


def cases():
    relu = spec_mod.by_name("relu_16x16")
    gemm_relu = spec_mod.by_name("gemm_relu_16x16x16")
    return [("relu_looped_relu", relu, pointwise(relu, True, False)),
            ("relu_looped_norelu", relu, pointwise(relu, False, False)),
            ("relu_unrolled_relu", relu, pointwise(relu, True, True)),
            ("gemm_relu_retry", gemm_relu, baseline.program(gemm_relu))]


if __name__ == "__main__":
    prj = measure.PRJ
    print(f"synthesizing once into {prj} ...", flush=True)
    measure.synthesize(prj)
    for tag, sp, prog in cases():
        print(f"  {tag:22s} {cycles.report(prog)}", flush=True)
        n, line = measure.measure(sp, prog, prj, tag=tag)
        print(f"  {tag:22s} cosim={n}   {line}", flush=True)

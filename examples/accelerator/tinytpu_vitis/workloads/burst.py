# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the burst-widening candidate is worth to a program, before measuring.

`TPU_DMA_WIDEN=1` reads `DMA_WORDS` packed words per burst iteration instead of
one, so a burst of `span * WPR` words costs a factor of `DMA_WORDS` fewer
iterations. Two laws fit that mechanism, and the published shape table tells
them apart: charging every removed iteration predicts 1440 and 1920 cycles at
48x48x48 and 64x64x64, against the 720 and 960 measured, while charging only
the longer of the two bursts predicts both exactly.

So `predict` is the overlapped law. It is still an extrapolation onto a model:
both fitted shapes have *equal* activation and weight spans, and every MLP
layer has unequal ones, which is the one regime where the law's shape is doing
work rather than arithmetic. That is why the suite measures a model on RTL
instead of adding up this column. Prose:
docs/source/designs/workload_suite.rst."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    WPR, assemble,
)

MEASURED = {(48, 48, 48): 720, (64, 64, 64): 960}


def spans(prog):
    """The DRAM row spans `dma_ld` must burst for A and for B, from the header
    `assemble` writes, after the AGU is resolved."""
    word = assemble(prog)[7]
    return word & 0xFFFF, word >> 16


def iterations(prog, words):
    a, b = spans(prog)
    return ((a * WPR + words - 1) // words, (b * WPR + words - 1) // words)


def charge_every_iteration(prog, words):
    before, after = iterations(prog, 1), iterations(prog, words)
    return sum(before) - sum(after)


def predict(prog, words):
    before, after = iterations(prog, 1), iterations(prog, words)
    return max(before) - max(after)


def check_measured(words):
    """`predict` against the two shapes the grid measured, and the law it rules
    out. Returns one row per shape."""
    from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program
    rows = []
    for shape, measured in MEASURED.items():
        prog = gemm_program(*shape)
        rows.append((shape, measured, predict(prog, words),
                     charge_every_iteration(prog, words)))
    return rows

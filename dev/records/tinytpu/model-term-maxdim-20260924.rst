..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

######################################################################
The model term is a MAXDIM=64 measurement, and it is flat at MAXDIM=16
######################################################################

:date: 2026-09-24
:what: RTL cosim of ``mlp_tiny`` and ``mlp_deep``, every layer, at
       ``TPU_T=4 TPU_MAXDIM=16 TPU_QD=16``, at ``DMA_WORDS`` 1 and 4.
:why:  Before the CHIA loop's objective is changed to rank on model cycles,
       check that the model/GEMM divergence the change rests on exists **at
       the configuration the loop scores at**.

The question
============

:doc:`/designs/workload_suite` measures the operand-burst widening at
**4.3-7.0 %** of a GEMM shape's runtime and **25.0 / 27.5 / 34.5 %** of a
model's, and that 4-8x gap is the whole argument for putting a model in the
objective. Every one of those numbers was taken at ``TPU_MAXDIM=64``.

The loop scores at ``TPU_MAXDIM=16`` (``evaluate.SCORED``), where ``WPR`` is 4
instead of 16 and every operand burst is therefore **four times shorter**. The
page says as much about the static estimator --- "the fit was taken at
MAXDIM=16 ... so the constant absorbed a burst term that no longer exists" ---
which is the same observation pointed at a different instrument. Nobody had
asked it of the measurement.

The measurement
===============

One ``csynth`` per burst width, every layer of both models through
``act.measure`` on that one build, all bit-exact (``mismatches = 0 / 256`` on
every layer, both widths). ``TPU_DMA_WIDEN=1`` selects ``DMA_WORDS=4`` at
MAXDIM=16, which is the widest the 64-byte beat holds there.

.. list-table:: T=4, MAXDIM=16, QD=16
   :header-rows: 1

   * - layer
     - ``DMA_WORDS=1``
     - ``DMA_WORDS=4``
     - saving
   * - ``mlp_tiny_l0``
     - 442
     - 442
     - **0**
   * - ``mlp_tiny_l1``
     - 419
     - 419
     - **0**
   * - ``mlp_tiny`` (2 layers)
     - **861**
     - **861**
     - **0**
   * - ``mlp_deep_l0``
     - 442
     - 442
     - **0**
   * - ``mlp_deep_l1``
     - 442
     - 442
     - **0**
   * - ``mlp_deep_l2``
     - 377
     - 377
     - **0**
   * - ``mlp_deep_l3``
     - 269
     - 269
     - **0**
   * - ``mlp_deep`` (4 layers)
     - **1 530**
     - **1 530**
     - **0**

Not "small". **Zero, layer by layer, to the cycle.** The same change is worth
287 and 583 cycles on the same two models at MAXDIM=64.

What it means
=============

**The 25-34 % is not a property of models. It is a property of models at
MAXDIM=64.** A 4x16x16 layer bursts 4 activation rows against 16 weight rows;
at MAXDIM=64 a row is 16 packed words, so the weight burst is 256 word
iterations and dominates everything. At MAXDIM=16 a row is 4 packed words, the
weight burst is 64 iterations, and the rest of the per-layer prologue covers
all of it. Widening a burst that was never on the critical path buys nothing.

**And the relationship inverts.** At MAXDIM=16 the GEMM shapes *do* see the
widening --- CHIA run 1 measured ``0 / 0 / -42 / -59 / -59`` over the five
shapes on exactly this configuration --- while the models see nothing. At
MAXDIM=64 the models see 25-34 % and the shapes see 4.3-7.0 %. So at the
configuration the loop scores at, the model term is not merely less sensitive
to the burst than the GEMM term: it is **the less sensitive of the two**, which
is the opposite of the reason it was added.

Consequence for the objective
=============================

A model term measured at MAXDIM=16 does not buy the sensitivity it was chosen
for, and shipping one while quoting the 25-34 % figure would be quoting a
number from a configuration it was not measured on --- this project's
most-repeated error, and the one ``workloads/claims.json`` exists to stop.

So ``evaluate.SCORED_MODEL_ENV`` measures the model term at **MAXDIM=64**,
beside a GEMM control that stays at MAXDIM=16 because that is the published
row. Two configurations means **two csynths per candidate**, which roughly
doubles the per-candidate cost, and that is the price of the term being worth
anything.

The alternative --- move the whole scored point to MAXDIM=64 --- is cheaper per
candidate and changes what the published five-shape row means, so it is a
decision for a person and not for this record.

Reproduce
=========

.. code-block:: bash

    export TPU_T=4 TPU_MAXDIM=16 TPU_QD=16
    python workloads/run.py --cosim mlp_tiny mlp_deep --json a.json
    TPU_DMA_WIDEN=1 python workloads/run.py --cosim mlp_tiny mlp_deep --json b.json
    diff <(jq .cycles a.json) <(jq .cycles b.json)     # empty

About 50 minutes for the pair, two ``csynth`` runs and twelve cosims, no
licence beyond Vitis.

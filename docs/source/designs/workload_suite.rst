..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

##############################################
The Workload Suite: PyTorch MLPs on TinyTPU
##############################################

Every number this project had before this page was a bare GEMM at a shape
somebody chose. That is enough to rank two datapaths and not enough to rank
two *designs*, because a design decision is worth whatever it does to the
mixture of shapes a real program actually issues. This page is about the
smallest thing that fixes that: four PyTorch MLPs, extracted layer by layer
into the workload specs ``act/corpus/`` already uses, mapped onto the fixed
TinyTPU by the ACT flow, verified bit-exact, and measured.

It is deliberately small. Four models that run are worth more than twelve that
are described, and the point of the suite is not coverage --- it is that a
*workload* is now the unit of evaluation.

.. _workload-suite-two-directions:

The sentence to take away first
===============================

**Allo's PyTorch path synthesises a machine for a model. ACT's flow maps a
model onto a machine. The co-design loop needs the second.**

``allo/frontend/pytorch.py``'s ``from_pytorch`` is a real, working PyTorch
front end, and it is not the one this suite uses. Its front half traces the
module; its back half hands the traced graph to ``TorchBuilder``, which emits
**Allo DSL source for that model**, runs it through ``customize``, composes
kernels out of ``allo/library/nn.py`` and calls ``s.build(target=...)``. The
artefact is a bespoke accelerator *for* the MLP. That is a legitimate and
interesting thing to build, and it is the opposite of what an evaluation
needs, because a machine that changes with every model cannot tell you what
changing the machine is worth.

So the suite uses the front half only:

.. code-block:: text

    PyTorch nn.Module
      -> AlloTracer + torch.fx ShapeProp      shapes and op types, nothing more
      -> per-layer workload specs             act/corpus/'s JSON convention
      -> the ACT mapper (act_compile)         onto the FIXED TinyTPU
      -> a TinyTPU program                    verified against isa_ref
      -> cycles                               estimated, then measured

``AlloTracer`` (``allo/frontend/tracer.py``) is an ``fx.Tracer`` subclass and
nothing more; ``ShapeProp`` is stock ``torch.fx``. Neither ``TorchBuilder``
nor ``customize`` nor ``allo/library/nn.py`` is imported by anything under
``workloads/``.

Why not TOSA
------------

The recollection that ACT consumed TOSA extracted from PyTorch is worth
answering directly, because the answer is a scoping decision rather than a
correction. A workload spec in this project needs five things: the einsum, the
extents, the operand and accumulator dtypes, the epilogue, and where each
tensor sits in DRAM. For an MLP, ``fx`` plus ``ShapeProp`` supplies the first
four exactly and the fifth is the mapper's business. Going through TOSA would
mean importing ``torch-mlir``, lowering to a dialect, and then reading M, K and
N back out of a ``tosa.matmul`` --- the same five facts, through two more
tools, with two more versions to pin.

Where TOSA would start to earn its place:

* **a wider op set.** The moment the suite wants convolutions, pooling,
  normalisation or quantisation-aware graphs, hand-written ``fx`` rules stop
  being a page of code and start being a compiler. TOSA has already made those
  choices, and it has made them the same way for everyone.
* **a second front end.** TOSA is an interchange. Today there is one producer
  (PyTorch) and one consumer (ACT), so the interchange carries no traffic. A
  TensorFlow or ONNX front end, or a second target reading the same specs,
  would change that.

Neither is true yet. When one becomes true, the boundary to replace is
``workloads/extract.py`` and nothing downstream of it, because what it emits
is already the corpus's JSON.

The suite
=========

``examples/tinytpu/workloads/models.py``. Ordinary
``nn.Module``\ s with nothing Allo-specific in them, sized so every layer fits
one build (M, K, N all at most ``MAXDIM``), with ``bias=False`` for the reason
in :ref:`workload-suite-what-maps`.

.. list-table::
   :header-rows: 1

   * - model
     - batch
     - layers
     - what it is for
   * - ``mlp_tiny``
     - 4
     - 16 -> 16 -> 16
     - the smallest thing that is still a model; the one measured on RTL
   * - ``mlp_deep``
     - 4
     - 16 -> 16 -> 16 -> 12 -> 8
     - a run of small layers, where almost everything is fixed cost
   * - ``mlp_small``
     - 8
     - 32 -> 32 -> 16
     - a middle shape, and the one where ``Kt = QD``
   * - ``mlp_wide``
     - 64
     - 64 -> 64 -> 64
     - two layers at exactly the steady-state shape the GEMM table quotes
   * - ``mlp_bias``
     - 4
     - 16 -> 16, with bias and a sigmoid
     - the probe: it is meant *not* to map

``mlp_deep`` and ``mlp_wide`` are the two ends of one axis on purpose. They are
what lets the page say something about a design decision that the shape table
could not: see :ref:`workload-suite-widening`.

Weights are drawn in [-8, 8) and used as integers, so "quantising" the module
is a cast rather than a calibration. That is a deliberate limit --- a real
quantisation scheme is a second unvalidated thing between the model and the
number, and this suite is about the mapping.

.. _workload-suite-what-maps:

What maps, and what does not
============================

The design computes ``int8 x int8 -> int32`` with an optional ReLU on the
accumulator before the narrowing. So:

**Maps.** ``nn.Linear`` without bias, at any (M, K, N) within ``MAXDIM``,
including extents that are not multiples of ``T``; and a ReLU that consumes a
Linear's result directly and is its only consumer, which becomes the fused
epilogue rather than a second pass.

**Does not map, and is reported rather than approximated.**

* **``nn.Linear`` with bias.** The ISA has ``vadd`` between accumulator rows,
  but no mapping that broadcasts one bias row across the M rows of the
  accumulator, so the baseline cannot express it. This is a mapper gap, not an
  encoding gap --- ``vadd`` exists.
* **Any other activation.** ``sigmoid``, ``tanh``, ``gelu`` and the rest have
  no unit. ``vru`` does ReLU.
* **A ReLU that is not the sole consumer of a Linear.** With two consumers the
  unfused value is needed as well, and there is no epilogue to ride on.
* **Anything above ``MAXDIM`` in any of M, K or N.** The operands are one
  ``int8[MAXDIM*MAXDIM]`` region, so a shape has to fit the addressable space
  rather than being tiled into it. This is the capability limit
  :doc:`gemmini_comparison` calls worse than the cycle deficit, and the suite
  inherits it exactly.
* **Everything else in a graph** --- reshape, concatenation, attention,
  normalisation, residual adds. The extractor names the node and the op.

``mlp_bias`` exists so that this list is demonstrated and not merely asserted:
it reports two refusals and maps its one remaining layer.

The extractor
=============

``workloads/extract.py``. It walks the ``ShapeProp``\ -annotated graph once and
turns each ``nn.Linear`` into a spec in the **same convention**
``act/corpus/`` uses --- same required fields, same ``einsum`` string, same
epilogue rules, and validated by the same ``act.spec.validate`` the corpus is
held to. There is no second convention. ``run.py --emit`` writes them to
``workloads/specs/``, where they are readable next to the corpus's own.

M is the product of every activation dimension but the last, so a batch is
rows; K and N come from the Linear's ``in_features`` and ``out_features``. The
weight is transposed on the way into DRAM, because ``nn.Linear`` holds
``(out, in)`` and the spec's ``B`` is ``kn``.

Verification
============

Three checks, and a layer that fails any of them is reported, never rounded.

#. **The program against the spec.** ``act.correctness.check`` runs
   ``isa_ref.run`` --- the ISA as numpy --- over four operand distributions and
   compares against ``spec.gold``, the einsum in int64, over the output region,
   with everything outside the write window required to be untouched. This is
   the corpus's own bar and every layer of every model clears it.
#. **The chain against PyTorch.** The suite then runs the *model*: each layer's
   program in turn through ``isa_ref``, with the real weights in DRAM and the
   previous layer's int8 output as the next layer's activation, against
   PyTorch's own evaluation of the same modules with the machine's epilogue
   (ReLU where fused, then the clip to int8) applied in torch. Every model
   matches on every byte. This is what makes the suite a workload rather than
   a shape table: the machine computes the model, not a GEMM of the model's
   size.
#. **The RTL.** Three of the four models' layers go through ``cosim.py``'s
   machinery, below, each compared against ``isa_ref`` over all 4096 bytes of
   ``C`` rather than over the result region alone.

.. _workload-suite-claims:

What is verified, and what is only executed
--------------------------------------------

These are different claims and the suite keeps them apart, on the ladder
``act/judge.py`` already defines --- ``legal``, then ``correct``, then
``confirmed``.

.. list-table::
   :header-rows: 1

   * - model
     - correct (bit-exact vs ``isa_ref`` and vs PyTorch)
     - confirmed (RTL ran it)
     - its cycle figure is
   * - ``mlp_tiny``
     - yes, 2/2 layers
     - **yes**, 2/2 layers
     - **measured**
   * - ``mlp_deep``
     - yes, 4/4 layers
     - **yes**, 4/4 layers
     - **measured**
   * - ``mlp_small``
     - yes, 2/2 layers
     - **yes**, 2/2 layers
     - **measured**
   * - ``mlp_wide``
     - yes, 2/2 layers
     - **no** --- RTL does not complete
     - *estimated only*

So three of four models carry a measured number and one carries a modelled
one, and the page never adds them together. ``mlp_wide`` is **correct and not
confirmed**: its programs agree with ``isa_ref`` and with PyTorch in software,
and Vitis csim reports ``0 / 4096`` mismatches, but no RTL run of it has ever
produced a cycle count (:ref:`why <workload-suite-widening>`).

Two provenance facts that belong with the claims rather than under them:

* **The bit-exactness claims need no bindings.** ``isa_ref`` is numpy and the
  ``act/`` core imports numpy only, so "correct" is established by code that
  is entirely this checkout's.
* **The compiled half is borrowed.** This worktree has no ``mlir/build``, so
  ``allo._mlir`` resolves to ``/home/sk3463/allo-bench`` at ``ff7beaf1`` ---
  ``tests/act/test_bindings.py`` fails here and names both paths, which is
  what it is for. ``git diff ff7beaf1 HEAD -- mlir/`` is **empty**, so the
  emitter that produced the HLS for every cosim below is the same source as
  this branch's. That is the "borrowed, same commit" case ``dev/toolchains.rst``
  describes, said out loud as it asks.

The numbers
===========

.. _workload-suite-layer-by-layer:

Layer by layer, with nothing between layers
-------------------------------------------

**A model's cycle figure is a sum over its layers, and that is the honest
description of what the machine does today, not a simplification of it.**
TinyTPU-isa runs one program per layer: operands in from DRAM, result out to
DRAM. There is no inter-layer fusion, and nothing --- not the activation, not a
weight tile --- stays resident between layers. So the sum is the whole cost,
with one qualification in the other direction: the sum also omits whatever a
host would spend between layers, which on this design is not modelled at all.

``python workloads/run.py`` at the shipped build (``T=4``, ``MAXDIM=64``,
``DMA_WORDS=1``). ``mapping`` is what the ACT search picked, and it is worth
knowing that **the mapper is not the variable here**: on eight of the ten
layers its pick is bit-identical to the hand-written ``gemm_program`` for that
shape. The exception is ``mlp_wide``'s two 64x64x64 layers, where ACT prefers
``M2>N16>K16`` --- a split of the row dimension the hand mapping does not make
--- at *exactly* the same modelled cost, 30 051 and 28 392, so the difference
is a tie broken on the label and not a disagreement about which is better.

.. list-table:: Per layer, ACT's pick, estimated
   :header-rows: 1

   * - layer
     - shape
     - epilogue
     - mapping
     - dynamic
     - critical unit
     - estimate
     - ``isa_ref``
   * - ``mlp_tiny_l0``
     - 4x16x16
     - relu
     - ``N4>K4``
     - 32
     - ``spm`` 144
     - 407
     - bit-exact
   * - ``mlp_tiny_l1``
     - 4x16x16
     - --
     - ``N4>K4``
     - 28
     - ``spm`` 144
     - 407
     - bit-exact
   * - ``mlp_deep_l0``
     - 4x16x16
     - relu
     - ``N4>K4``
     - 32
     - ``spm`` 144
     - 407
     - bit-exact
   * - ``mlp_deep_l1``
     - 4x16x16
     - relu
     - ``N4>K4``
     - 32
     - ``spm`` 144
     - 407
     - bit-exact
   * - ``mlp_deep_l2``
     - 4x16x12
     - relu
     - ``N3>K4``
     - 25
     - ``spm`` 108
     - 348
     - bit-exact
   * - ``mlp_deep_l3``
     - 4x12x8
     - --
     - ``N2>K3``
     - 13
     - ``spm`` 54
     - 261
     - bit-exact
   * - ``mlp_small_l0``
     - 8x32x32
     - relu
     - ``N8>K8``
     - 96
     - ``accu`` 640
     - 1 211
     - bit-exact
   * - ``mlp_small_l1``
     - 8x32x16
     - --
     - ``N4>K8``
     - 48
     - ``vru`` 320
     - 692
     - bit-exact
   * - ``mlp_wide_l0``
     - 64x64x64
     - relu
     - ``M2>N16>K16``
     - 624
     - ``accu`` 18 432
     - 30 051
     - bit-exact
   * - ``mlp_wide_l1``
     - 64x64x64
     - --
     - ``M2>N16>K16``
     - 592
     - ``vru`` 17 408
     - 28 392
     - bit-exact

.. warning::

   **These are estimates and one of them is structurally blind.**
   ``act.cycles.estimate`` is ``173.2 + 1.621 x critical work``, fitted to the
   five published ``MAXDIM=16`` cosim points, and its per-unit work counts come
   from the header ``assemble`` writes --- where ``dma_ld``'s entry is the
   number of instruction *rows*, not the number of burst *iterations*. So the
   estimate cannot see the DMA burst at all. At ``MAXDIM=64`` the burst is a
   large fixed cost at small shapes (:doc:`benchmarks` prices the
   ``MAXDIM=16 -> 64`` move at +193 cycles on 16x16x16 alone), and it is
   exactly the term the burst-widening candidate moves. Every conclusion below
   about the widening therefore rests on cosim, not on this column.

Measured on RTL
---------------

Three of the four models, every layer through ``cosim_design`` on one
``csynth`` of the shipped build, bit-exact against ``isa_ref`` over all 4096
bytes of ``C``. These are **measurements**; the column beside them is the
estimate, for scale.

.. list-table:: ``T=4 MAXDIM=64 DMA_WORDS=1 TPU_QD=16``, ``m_axi_latency`` 0
   :header-rows: 1

   * - model
     - layers
     - estimated
     - **measured**
     - the estimate is
   * - ``mlp_tiny``
     - 584 + 566
     - 813
     - **1 150**
     - 29 % low
   * - ``mlp_deep``
     - 584 + 584 + 542 + 407
     - 1 422
     - **2 117**
     - 33 % low
   * - ``mlp_small``
     - 1 636 + 1 145
     - 1 903
     - **2 781**
     - 32 % low
   * - ``mlp_wide``
     - --
     - 58 443
     - *no completion*
     - --

``mlp_wide`` is dealt with in :ref:`its own section
<workload-suite-widening>`. Note that these needed ``TPU_QD=16``: at the
shipped ``TPU_QD=8`` not even ``mlp_tiny``'s first layer finishes, which is
new information about :ref:`limitations item 24 <limitation-24>` and is
recorded there.

.. _workload-suite-widening:

The measurement that justifies the suite
========================================

A workload changes *which* design decisions look good. The suite's one job,
once it ran, was to test that claim on a decision the project already had in
front of it.

The decision is the **burst-widening candidate**, ``TPU_DMA_WIDEN=1``: the
``dma_ld`` operand burst reads ``DMA_WORDS`` packed words per loop iteration
instead of one, up to the 64-byte beat ``align_value(64)`` lets Vitis widen
the port to. It is parametric, bit-exact at both settings, and costs +123 %
BRAM. It is also the knob in the ``parity-t4`` baseline
(``TPU_T=4 TPU_MAXDIM=64 TPU_DMA_WIDEN=1``), so the widened column below is
that baseline and not a variant invented here. :doc:`benchmarks` measures it
at two shapes:

.. list-table:: What the GEMM table says
   :header-rows: 1

   * - shape
     - shipped
     - widened
     - saving
     - **as a fraction**
   * - 48x48x48
     - 10 289
     - 9 569
     - 720
     - **7.0 %**
   * - 64x64x64
     - 22 123
     - 21 163
     - 960
     - **4.3 %**

Read as a GEMM table, that is a worthwhile but unspectacular few per cent,
shrinking as the shape grows --- the profile of a fixed cost being amortised.
The question the suite exists to answer is what the same change is worth to a
*model*.

Why the shape table cannot be extrapolated
------------------------------------------

The mechanism is that widening removes burst iterations. A burst of
``span x WPR`` words costs ``ceil(span x WPR / DMA_WORDS)`` iterations, and
``dma_ld`` issues one burst for the activations and one for the weights. Two
laws fit that:

* charge **every** removed iteration, activations and weights both;
* charge only the **longer** of the two bursts, the other being hidden.

The first predicts 1440 and 1920 cycles at 48\ :sup:`3` and 64\ :sup:`3`; the
second predicts 720 and 960. The measurements are 720 and 960, so the two
points settle it: only the longer burst is ever critical.

**But both fitted shapes have equal spans.** At 48x48x48 the activation span
is 48 rows and the weight span is ``ceil(48/T) x T`` = 48 rows; at
64x64x64 both are 64. When the two spans are equal the two laws are the same
arithmetic, and the shape table never visits a point where they differ.

**Every MLP layer is such a point.** A 4x16x16 layer bursts 4 activation rows
against 16 weight rows --- 64 word-iterations against 256. The spans differ by
4x, the weight burst dominates, and which law holds decides the answer. That
is not a subtlety the GEMM table could have surfaced, because the GEMM table
only ever asked square questions.

So the suite measures.

What it is worth on a model
---------------------------

Two builds, one ``csynth`` each, every layer through ``cosim_design`` on both.
``T=4``, ``MAXDIM=64``, ``TPU_QD=16``, ``TPU_AXI_LATENCY=0``; the only
difference between the columns is ``DMA_WORDS``. Every run is bit-exact
against ``isa_ref`` over all 4096 bytes of ``C`` at both settings.

.. list-table:: Per layer, measured on RTL
   :header-rows: 1

   * - layer
     - shape
     - shipped
     - widened
     - saving
     - predicted
   * - ``mlp_tiny_l0``
     - 4x16x16
     - 584
     - 443
     - 141
     - 240
   * - ``mlp_tiny_l1``
     - 4x16x16
     - 566
     - 420
     - 146
     - 240
   * - ``mlp_deep_l0``
     - 4x16x16
     - 584
     - 443
     - 141
     - 240
   * - ``mlp_deep_l1``
     - 4x16x16
     - 584
     - 443
     - 141
     - 240
   * - ``mlp_deep_l2``
     - 4x16x12
     - 542
     - 378
     - 164
     - 240
   * - ``mlp_deep_l3``
     - 4x12x8
     - 407
     - 270
     - 137
     - 180
   * - ``mlp_small_l0``
     - 8x32x32
     - 1 636
     - 1 156
     - **480**
     - **480**
   * - ``mlp_small_l1``
     - 8x32x16
     - 1 145
     - 665
     - **480**
     - **480**

.. list-table:: Per model, and against the GEMM table
   :header-rows: 1

   * - model
     - shipped
     - widened
     - saving
     - **as a fraction**
   * - ``mlp_tiny`` (2 layers)
     - 1 150
     - 863
     - 287
     - **25.0 %**
   * - ``mlp_deep`` (4 layers)
     - 2 117
     - 1 534
     - 583
     - **27.5 %**
   * - ``mlp_small`` (2 layers)
     - 2 781
     - 1 821
     - 960
     - **34.5 %**
   * - *GEMM table*, 48x48x48
     - *10 289*
     - *9 569*
     - *720*
     - *7.0 %*
   * - *GEMM table*, 64x64x64
     - *22 123*
     - *21 163*
     - *960*
     - *4.3 %*

**The answer is that the widening is worth four to eight times more on a model
than the shape table says.** 25 to 34 per cent of a model's cycles against 4.3
to 7.0 per cent of a big GEMM's. Read only as a GEMM table, ``TPU_DMA_WIDEN``
is a few per cent that shrinks as the shape grows; read on a model, it is a
quarter to a third of the whole runtime and it does not shrink, because a
model does not get bigger --- it gets *longer*, and every layer pays the
prologue again. That is the argument for the suite existing, and it is a
conclusion the shape table could not have produced, at any level of detail,
because it never asked a question shaped like a model.

Two smaller findings fall out of the same measurement, and both are about our
instruments rather than the design:

**The prediction is exact where the burst is critical and over-predicts where
it is not.** At ``mlp_small``'s two layers the measured saving is 480 cycles
and the predicted saving is 480 cycles, twice, to the cycle. At the 4-row
layers it over-predicts by 40 %: 240 predicted against 141 measured. The
reading is that the burst is fully on the critical path once it is long
enough, and at four activation rows the rest of the prologue overlaps about a
hundred cycles of it. So ``workloads/burst.py`` is a usable screen for which
layers the widening will help and **not** a substitute for measuring how much.

**The static estimate is accidentally calibrated for a machine we do not
ship.** ``act.cycles.estimate`` is 26 to 36 per cent *low* against the shipped
``DMA_WORDS=1`` build and within 8 per cent of the *widened* one --- it fits
the widened machine better than the real one. That is not a coincidence: the
fit was taken at ``MAXDIM=16``, where ``WPR`` is 4 instead of 16 and the burst
is four times shorter, so the constant absorbed a burst term that no longer
matches. An estimate carries its calibration regime with it, and this one's
regime is not the shipped build.

The 64x64x64 point is missing, and why
--------------------------------------

``mlp_wide`` is in the suite precisely so that the steady-state shape could be
measured on *these two builds* rather than compared across published ones, and
**it does not complete in cosim at any channel depth tried**. Six runs, four
builds, both burst widths:

.. list-table::
   :header-rows: 1

   * - ``TPU_QD``
     - bound
     - last RTL progress
     - verdict
   * - 16
     - 600 s
     - ``@ 109000`` ps --- the first periodic report, no second
     - no completion
   * - 32
     - 1500 s
     - ``@ 33400115000`` ps = **33.4 ms, about 10 million cycles**
     - no completion

Its layers have ``Kt = 16``, so ``Kt >= QD`` holds at ``QD=16``, which is the
one condition the parity work has a rule for --- but ``QD=32`` breaks that
condition and does not fix the hang. What ``QD=32`` does change is the
*evidence*: the simulation advanced ten million cycles on a program the model
prices at 22 000 and still did not finish, which is a deadlock and not a slow
run. So ``Kt >= QD`` is **not** the whole rule, and this is a distinct fact
from the ``QD=8``/``QD=16`` cases at the small shapes, where the sim never got
past its first report at all.

One consequence for the headline. The 4.3 % row above is carried from
:doc:`benchmarks`, measured at ``TPU_QD=8`` on the *hand* 64x64x64 mapping,
while the three model rows are measured here at ``TPU_QD=16`` --- so the
comparison crosses a build boundary, and the ratio should be read as
four-to-eight-fold rather than as a precise multiple. The three model rows are
internally controlled: one ``csynth`` per burst width, everything else equal.

Limits
======

* **Layer by layer, no fusion, no residency** --- :ref:`the section above
  <workload-suite-layer-by-layer>`. The sum is what the machine does; it is
  also why a model with many small layers pays the prologue many times.
* **No host time.** Nothing between layers is modelled: no driver, no
  scheduling, no cost for the activation making a round trip through DRAM that
  a fused design would not make.
* **MLPs only, and small ones.** ``MAXDIM`` bounds every extent, which rules
  out the shapes a real network uses. The suite is a way of asking the
  question, not an answer about real networks.
* **Integer weights, not a quantisation scheme.** See above.
* **Cosim has no DRAM model.** Every RTL figure here is at
  ``TPU_AXI_LATENCY=0``, a value we chose; ``m_axi_latency`` is an HLS
  scheduling directive and not a memory latency, as :doc:`benchmarks`
  establishes at length. No row here may be read as behaviour against a memory
  system.
* **Every RTL figure on this page needed ``TPU_QD=16``.** At the shipped
  ``TPU_QD=8`` the very first layer of the smallest model does not finish
  cosim, and it is the *shipped* GEMM mapping at 4x16x16, not an exotic
  one. This is a new instance of :ref:`limitations item 24 <limitation-24>`
  and it is recorded there, because the knob that flips it is new
  information: the earlier bisection never varied a channel depth.
  ``run.py --cosim`` bounds every run and kills by process group; if a layer
  does not finish, raise ``TPU_QD`` before concluding anything from the hang.

Running it
==========

.. code-block:: bash

    cd examples/tinytpu

    python workloads/run.py                      # every model, static, ~1 min
    python workloads/run.py --emit               # and write workloads/specs/
    python workloads/run.py --simulator          # also check the built design
    python workloads/run.py --burst              # what the widening should be worth

    TPU_QD=16 python workloads/run.py --cosim mlp_tiny                   # RTL
    TPU_QD=16 TPU_DMA_WIDEN=1 python workloads/run.py --cosim mlp_tiny   # widened

    pytest tests/act/test_workload_suite.py      # no Vitis, seconds

``--cosim`` synthesises once into ``--project`` (default
``workloads/workload.prj``, removed when the run ends) and measures every named
model's layers on that one build, so a sum over layers is a sum over one
machine. Put the project inside your worktree: ``/home`` is chronically full.

**``torch`` is not a dependency of this repository** and is not in
``requirements.txt`` --- upstream keeps it optional, and
``tests/pytorch/test_linear.py`` guards its import. The suite needs it, and
the ``allo`` environment on this host did not have it until 2026-09-22:

.. code-block:: bash

    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

``pytest tests/act/test_workload_suite.py`` skips rather than fails when
``torch`` is *absent*, so a machine without it still runs the rest of
``tests/act``. It does **not** skip when ``torch`` is installed but broken:
``pytest.importorskip`` re-raises anything that is not a
``ModuleNotFoundError``, which is the behaviour you want and is worth knowing
before you read a red collection error as "the suite is wrong".

.. _workload-suite-loop:

A co-design change, read end to end through the suite
=====================================================

This is the whole point, so here is the loop once, concretely, with the change
that was actually put through it.

**The change.** Widen the ``dma_ld`` operand burst from one packed word per
loop iteration to a whole 64-byte beat. It is already parametric, so the
change is an environment variable rather than a patch, and it costs +123 %
BRAM. **The question a co-design loop has to answer is whether that BRAM buys
anything on software anyone would run.**

**Step 1 --- what does the workload look like to the machine?**

.. code-block:: bash

    python workloads/run.py --emit

Ten layers, all mappable, each one a spec in ``workloads/specs/``. Six of the
ten are four-row GEMMs: the suite is mostly *small* work, which a shape table
chosen by hand would not have told you.

**Step 2 --- cheap screen: which layers could the change even touch?**

.. code-block:: bash

    python workloads/run.py --burst

Every layer, because every layer's weight burst is four to sixteen times its
activation burst. Seconds, no tools. If this had come back zeros the loop would
stop here.

**Step 3 --- measure the two machines on the workload.**

.. code-block:: bash

    TPU_QD=16 python workloads/run.py --cosim mlp_tiny mlp_deep mlp_small
    TPU_QD=16 TPU_DMA_WIDEN=1 python workloads/run.py --cosim mlp_tiny mlp_deep mlp_small

One ``csynth`` per build, every layer bit-exact against ``isa_ref`` on both.
2 117 cycles becomes 1 534 on ``mlp_deep``; 2 781 becomes 1 821 on
``mlp_small``.

**Step 4 --- the number the decision turns on.** **25 to 34 per cent of a
model's runtime**, where the GEMM table said 4.3 to 7.0 per cent. That is the
output of the loop: not "the widening is 720 cycles at 48\ :sup:`3`", which is
true and does not decide anything, but "the widening removes a quarter to a
third of the time our software spends", which does.

**What the loop also returned, unasked.** Three things the GEMM table could
not have produced, and each of them is a correction to an instrument rather
than to the design: the static estimate is calibrated for a machine we do not
ship; the burst law is exact at eight rows and 40 % optimistic at four; and
the shipped channel depth does not run the shipped mapping at 4x16x16. A
workload is a better test of the instruments than a shape is, because it
exercises them where they were never fitted.

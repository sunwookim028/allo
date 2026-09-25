..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#######################################################
What a CHIA candidate costs once T and MAXDIM can move
#######################################################

:date: 2026-09-25
:what: Wall clock for one whole ``evaluate.py`` run at the published row and at
       a proposed ``T=8 MAXDIM=32``, and what the T=8 run could not measure.
:why:  The owner's spend ceiling is arithmetic over the cost of one candidate,
       and both halves of the co-design change move it: two gates were added,
       and a candidate may now propose a bigger machine to synthesise and
       simulate.

What was measured
=================

One ``evaluate.py`` run of the UNMODIFIED design -- ``--shapes
4x4x4,16x16x16``, ``--models mlp_tiny,mlp_deep``, this host (72 cores, one
Vitis HLS 2023.2, one other agent's harness running concurrently) -- against
the same run of the same design with one line added to ``microarch_isa.py``::

    CHIA_CONFIG = {"T": 8, "MAXDIM": 32}

Nothing else differs. Both are ``ok: true``.

.. list-table:: seconds, one candidate end to end
   :header-rows: 1
   :widths: 30 17 17 36

   * - stage
     - T=4 MAXDIM=16
     - T=8 MAXDIM=32
     - what it is
   * - ``gate:pytorch``
     - 2.9
     - 3.2
     - the oracle, ``run.py --verify``
   * - ``gate:isa``
     - 34.9
     - 41.3
     - ``gen_isa.py --conform``
   * - the rest of the gate
     - 66.5
     - 116.7
     - bench, stress, three ``param_check``\ s
   * - GEMM cosim
     - 170.3
     - 388.2
     - one csynth, then one cosim per shape
   * - model cosim
     - 338.9
     - 488.5
     - one csynth at MAXDIM=64, one per layer
   * - **whole candidate**
     - **623.9**
     - **1050.5**
     - ``ok: true`` both
   * - area estimate (um^2)
     - 1,202,984
     - 2,232,922
     - the proxy, both inside its envelope

Reading it
==========

**The two new gates are 37.8 s of 623.9 s (6.1 %) at the published row**, and
``gen_isa.py --conform`` is nearly all of it -- it builds the design down the
HLS path to read the bit ranges the emitted C++ takes. The oracle is 3 s. A
candidate was ~578 s before this work; it is ~624 s now.

**A T=8 candidate is 1.68x a T=4 one**, and the multiplier is in the RTL, not
in the gates: the gate grows 1.55x (bigger builds in the simulator) while the
two csynth+cosim passes grow 2.28x and 1.44x. Note the GEMM term grew 2.28x
while measuring FEWER shapes -- one instead of two -- so per shape it is 4.6x.

For a spend ceiling: a search whose candidates may propose T=8 should be
budgeted at the T=8 figure, not an average. 20 candidates at the published row
is 3.5 hours of machine time; 20 at T=8 is 5.8 hours.

It is a real design point, not just a slower run
================================================

The T=8 run is the first co-design point this loop has been able to produce::

    16x16x16   674 cycles  ->  426 cycles     (-36.8 %)
    mlp_tiny  1150 cycles  ->  783 cycles     (-31.9 %)
    area      1.20 mm^2    ->  2.23 mm^2      (+85.6 %, the proxy's estimate)
    timing    2.431 ns, met at both

Cycles down by a third for area up by nearly a factor of two, with the clock
met -- that is a trade the objective can now *state*, and could not before,
because ``check_invariants`` refused any candidate that was not T=4 MAXDIM=16.

Two things a T=8 candidate cannot measure, and why they are scope
================================================================

Both were found by running it, both are properties of FROZEN machinery rather
than of the candidate, and both are now named in the verdict instead of
failing it:

1. **Two of the five published GEMM shapes cannot run at T=8.** ``cosim.py``
   asserts every dimension is a multiple of T -- its testbench's own
   precondition -- and ``4x4x4`` and ``12x12x12`` are not multiples of 8. So
   the scored GEMM term at T=8 MAXDIM=32 is ``16x16x16`` alone, and
   ``shapes_skipped`` says why. Refusing the candidate for this would have made
   every T but 4 unreachable, which is what the change exists to undo.
2. **The frozen mapper refuses** ``mlp_deep_l2``, a 16x16x12 layer, because 12
   is not a multiple of T=8. ``workloads/scope_map.json`` already records that
   boundary as ``refused-without-a-cause`` on the K and N axes. So the model
   term at T=8 is ``mlp_tiny`` alone (``model_skipped`` names ``mlp_deep``),
   and the oracle's corpus is ``mlp_tiny`` + ``mlp_small`` instead of
   ``mlp_tiny`` + ``mlp_deep``.

The consequence is worth stating plainly: **a T=8 candidate is measured on a
narrower workload than a T=4 one** -- one GEMM shape instead of two, one model
instead of two. Its cycle counts are not comparable with the published row
either way (different machine), and ``accept.py`` reports the cross-check
against the published numbers as ``NOT-COMPARABLE`` rather than
``DISAGREES``. What would widen it is shapes and models chosen to be legal at
more than one T -- every dimension a multiple of 8 -- which is a change to
``shapes.py`` and ``workloads/models.py``, both frozen, and a person's
decision rather than a candidate's.

The corpus the PyTorch gate actually had
========================================

``run.py --verify`` reports it per run, because it is a function of the build
and of the mapper:

============================  =========  ===========  =========================
configuration                 models     bytes        seconds
============================  =========  ===========  =========================
T=4 MAXDIM=16 (the row)       2 of 4     2,688        2.9
T=8 MAXDIM=32                 2 of 4     4,096        3.2
T=4 MAXDIM=64                 4 of 4     71,296       57
============================  =========  ===========  =========================

Eight input/weight draws per model multiply the bytes and not the shapes: the
mapping depends only on the shapes, so each extra draw costs one ``isa_ref``
run and one torch forward. The gate buys an ORACLE -- something outside this
repository on one side of a comparison -- and not breadth. The 8,912-byte
figure quoted on :doc:`/designs/workload_suite` is the MAXDIM=64 corpus, 92 %
of it ``mlp_wide``; at the configuration the loop scores at, the oracle sees
2,688 bytes of six 16x16 layers. That is the honest size of it.

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

####################################
ACT Workload Specs and the Judge
####################################

The two things a compiler needs that :doc:`act` does not supply: a corpus of
workload-level specs to compile, and a way to decide whether what came out is
right, legal and fast. Both live in
``examples/accelerator/tinytpu_vitis/act/`` and both are about the
:doc:`/designs/tinytpu_isa` accelerator; neither contains any part of ACT.

.. note::

   This page is the **input and the judge** side of the interface. ACT's own
   core, mapper and scheduler are Kai Shao's work and are audited from the
   outside on :doc:`act`; nothing here modifies or republishes them.


What a spec says, and what it deliberately does not
===================================================

A spec is a JSON file in ``act/corpus/``. It states the computation and the
calling convention, and nothing about how to perform it: no tiling, no loop
order, no residency, no schedule, no instruction.

.. code-block:: json

   {
     "name": "gemm_16x16x8",
     "stresses": "the measured set's only non-cubic shape: half the output columns of gemm_16x16x16 over the same reduction, so output and reduction cost separate",
     "einsum": "mk,kn->mn",
     "dims": {"m": 16, "k": 16, "n": 8},
     "inputs": [
       {"name": "X", "subscript": "mk", "dtype": "int8", "buffer": "A", "origin": [0, 0]},
       {"name": "W", "subscript": "kn", "dtype": "int8", "buffer": "B", "origin": [0, 0]}
     ],
     "constants": [],
     "output": {"name": "Y", "subscript": "mn", "dtype": "int8", "buffer": "C",
                "origin": [0, 0], "write_window": "exact"},
     "accumulator": "int32",
     "epilogue": ["saturate"],
     "operand_pad": "arbitrary",
     "known_gap": null
   }

JSON, one file per spec, pretty-printed one key to a line: machine-readable
without a dependency, and a new case is a new file rather than a diff inside
an existing one.

``einsum`` is the human-readable statement and ``subscript`` is the
machine-readable one. ``spec.validate`` refuses a spec where they disagree,
which is the same discipline ``isa_dsl.assert_matches_handwritten`` applies to
the generated GEMM: state it twice, check that the two agree, and the
statement stops being able to drift.

Three fields exist because this ISA forces the question into the open
-----------------------------------------------------------------------

**``operand_pad``** -- what the bytes *around* a declared operand hold.
``arbitrary`` means the judge fills them with nonzero garbage; ``zero`` means
the host promises zeros. It matters because ``mm`` consumes a whole ``T``-lane
packed word: a reduction of ``k = 6`` on a ``T = 4`` array has to cover
``k = 0..7``, and whether that is exact depends entirely on what rows 6 and 7
of the weight operand hold. Two corpus specs differ in nothing but this field,
and one of them is satisfiable while the other is not.

**``write_window``** -- how much of ``C`` a program may write. ``exact`` means
the declared output region and not one byte more; ``column_block`` extends the
permission to the end of the enclosing ``T``-lane column block, because
``mvout`` retires whole packed words and an output of ``n = 6`` cannot avoid
touching columns 6 and 7. This is stated per spec rather than assumed,
so "the program clobbered ``C``" and "the machine writes ``T`` lanes at a
time" are different verdicts.

**``constants``** -- compile-time-known matrices the host places in DRAM. The
design has no path that writes DRAM other than ``mvout`` and no opcode that
zeroes an on-chip row, so a program cannot manufacture an identity or a
vector of ones; a pointwise or reduction workload therefore needs one placed
for it. Keeping constants out of ``inputs`` keeps ``einsum`` an honest
statement of the mathematics while the convention that makes it runnable stays
visible in the same file.

Two more fields carry the rest. **``stresses``** is one line saying what this
case tests that the others do not -- ``validate`` rejects a spec without it,
because a corpus of unexplained shapes teaches a compiler nothing.
**``known_gap``** is one line saying why no program on the *current* build can
satisfy the spec, and the judge reports such a spec's failure as ``KNOWN GAP``
rather than ``WRONG``, so the suite stays green while the gap stays visible.

Two layers, one convention
--------------------------

There are now two machine-readable specs in this tree and they are at
different layers, deliberately sharing a foundation rather than inventing two:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - file
     - layer
     - says
   * - ``isa_spec.json``
     - the machine
     - opcodes, field positions and widths, the imem header, the AGU term
       encoding, the loop-stack semantics, the memory map, the build
       parameters and the numerics
   * - ``act/corpus/*.json``
     - the workload
     - what must be computed, at what shapes and precisions, where the bytes
       are, and what the pad and the write window are allowed to be

Both are JSON rather than YAML for the same reason: the validator has to run
wherever ``allo`` runs, and ``json`` is in the standard library while PyYAML is
not in this repository's requirements. Both state the load-bearing facts twice
and check that the two statements agree -- ``gen_isa.py --check`` holds the
design to the ISA table, ``spec.validate`` holds ``einsum`` to the
``subscript`` fields -- which is the discipline
``isa_dsl.assert_matches_handwritten`` already applies to the generated GEMM.
A spec here never restates an ISA fact: opcode numbers, field widths and
``AR_RAW_DIST`` reach the judge by import, never by transcription.

Layout
------

Every tensor lies row-major in a ``MAXDIM x MAXDIM`` int8 DRAM image
(``A``, ``B`` or ``C``): all but the last subscript index rows, the last
indexes columns, and a vector is a column. Origins must start on a column-block
boundary, since every DRAM access the ISA has is a ``T``-lane packed word.
That rule is what lets ``bmk,bkn->bmn`` be placed without a new field.


The corpus
==========

Twenty-two specs, in three groups.

The first six are the shapes this fork measures -- ``bench_isa.py``'s
``LATENCY`` set and the 171 / 261 / 417 / 483 / 685 cosim points on
:doc:`/designs/tinytpu_isa`. The last five are the ``STEADY`` set that
``bench_isa.py`` gains on the ``benchmark-set`` branch, whose accounting is in
``docs/source/designs/benchmarks.rst`` there -- cubic 32, 48 and 64 plus
64x32x64 and
32x64x32, big enough that MACs per cycle characterises the machine rather than
its pipeline fill. Both sets are taken rather than reinvented, so there is one
benchmark set in this tree and not two. The steady shapes do not fit
``main``'s ``MAXDIM = 16`` build and are reported ``SKIPPED`` with the reason
until ``TPU_MAXDIM`` is raised; a workload spec is a property of the workload,
not of one elaboration.

The eleven in the middle exist because a corpus of square GEMMs would teach a
mapper nothing.

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - spec
     - einsum
     - what it stresses
   * - ``gemm_4x4x4``
     - ``mk,kn->mn``
     - one tile in every dimension: the fixed cost of issuing a program with
       nothing to amortise it over
   * - ``gemm_8x8x8``
     - ``mk,kn->mn``
     - the smallest shape with a real loop nest in all three ranks
   * - ``gemm_12x12x12``
     - ``mk,kn->mn``
     - an odd tile count, so a mapper assuming power-of-two trips breaks
   * - ``gemm_16x16x8``
     - ``mk,kn->mn``
     - the measured set's only non-cubic shape
   * - ``gemm_16x16x16``
     - ``mk,kn->mn``
     - the largest shape the operand buffers hold; highest utilisation
   * - ``gemm_relu_16x16x16``
     - ``mk,kn->mn``
     - a fused epilogue: the ReLU is on the int32 accumulator, before the
       narrowing
   * - ``gemm_reuse_m_16x16x4``
     - ``mk,kn->mn``
     - reuse in M: one weight tile serves 16 activation rows
   * - ``gemm_reuse_n_4x4x16``
     - ``mk,kn->mn``
     - reuse in N: activations resident, weights streaming
   * - ``gemm_reuse_k_4x16x4``
     - ``mk,kn->mn``
     - reuse in K: four k-tiles into one accumulator region, which puts every
       accumulate read at exactly ``AR_RAW_DIST = 4`` -- the legality edge
   * - ``gemm_rows_6x8x8``
     - ``mk,kn->mn``
     - ``M = 6`` against ``T = 4``; rows are the one dimension the ISA leaves
       free, so padding M is legal but writes two rows it must not
   * - ``gemm_narrow_2x8x4``
     - ``mk,kn->mn``
     - ``M = 2`` is below ``AR_RAW_DIST``, so the obvious program is rejected
       for *legality*, not arithmetic: pad the ``mm`` to 4 rows, retire 2
   * - ``gemm_cols_8x8x6``
     - ``mk,kn->mn``
     - ``N = 6`` against a ``T``-lane ``mvout``; needs
       ``write_window: column_block``
   * - ``gemm_reduce_8x6x8_zeropad``
     - ``mk,kn->mn``
     - ``K = 6`` with a zero pad: rounding the reduction up is exact and costs
       one extra k-tile
   * - ``gemm_reduce_8x6x8``
     - ``mk,kn->mn``
     - the same with an arbitrary pad. **Known gap**: no opcode zeroes a row,
       so rounding K up sums garbage and nothing else is expressible
   * - ``batched_matmul_2x4x4x4``
     - ``bmk,bkn->bmn``
     - a batch dimension in both inputs and the output, which a GEMM mapper
       has no rank for; the only corpus case whose natural program fills all
       ``AGU_TERMS = 3`` address terms at once
   * - ``relu_16x16``
     - ``mn->mn``
     - no contraction dimension at all: the array is the only path into the
       accumulator, so a pointwise workload goes through ``mm`` against a
       host-placed identity
   * - ``row_reduce_16x16``
     - ``mk->m``
     - a contraction whose output is a vector, written as one column of a
       word the machine can only write whole
   * - ``gemm_steady_32x32x32``
     - ``mk,kn->mn``
     - eight times the headline shape's MACs, so the fixed pipeline term stops
       dominating
   * - ``gemm_steady_48x48x48``
     - ``mk,kn->mn``
     - a tile count that is not a power of two in any rank
   * - ``gemm_steady_64x64x64``
     - ``mk,kn->mn``
     - the largest shape the widened build holds
   * - ``gemm_steady_64x32x64``
     - ``mk,kn->mn``
     - half the reduction at the same output size, which separates the
       wavefront-row term from the tile-count term
   * - ``gemm_steady_32x64x32``
     - ``mk,kn->mn``
     - M and K swapped against the above: same MACs, opposite aspect

``python act/judge.py specs`` prints this list with each ``stresses`` line.

Adding a spec is adding a JSON file. ``python act/spec.py`` validates the
whole corpus; ``python act/baseline.py`` says whether the reference submission
can map it.


The judge
=========

A submission is a program: either ``module:function`` taking a spec and
returning the ``(instruction word, AGU word)`` list ``assemble`` takes, or a
``.json`` file of those word pairs, which is what an agent that emits text
rather than Python should produce (``act/submission.py``). The default
submission is ``act/baseline.py``, one hand-chosen mapping per spec, which
maps all seventeen.

Is it legal?
------------

.. code-block:: bash

   python examples/accelerator/tinytpu_vitis/act/judge.py legal
   python examples/accelerator/tinytpu_vitis/act/judge.py legal --spec gemm_narrow_2x8x4 --program mymod:make

Four layers, reported in the order a program fails them:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - layer
     - what it is
   * - ``isa``
     - ``microarch_isa.check_program``: write-before-read on ``spad`` / ``vr``
       / ``ar``, the ``AR_RAW_DIST = 4`` accumulator distance contract,
       bounds, ``nr >= 1``, resolved field range, loop structure and AGU
       levels
   * - ``encoding``
     - the assertions ``assemble`` makes: imem capacity, the 15-bit header
       counts, the array counts, the DRAM burst span
   * - ``protocol``
     - ``kpn_model.run``: the units' promised work counts against what the
       sequencer dispatches. A valid program cannot reach this layer, which is
       the point -- it is a regression detector for the assembler and the
       microarchitecture, not a program-error check
   * - ``spec``
     - every ``mvout``'s footprint in the dynamic stream against the spec's
       write window, and the spec's output region against that footprint.
       Static, so "you padded M into C" is caught before anything runs

The rejection message is part of the deliverable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A generating agent acts on the message and on nothing else, so every
rejection carries a rule name, what happened, the rule that was broken, what
to change instead, and a disassembly of the offending instruction with its two
neighbours:

.. code-block:: text

   REJECTED [isa / mem.write_before_read]
     what happened: instruction 7 (mm, loop ivs [1]): reads spad row(s) [4, 5, 6, 7]
       as weights before any instruction wrote them. spad is not cleared by the
       hardware; see the write-before-read contract.
     the rule:      spad, vr and ar are not cleared between programs, so a row that
       this program never wrote holds whatever the last one left
     what to change: load the rows with `dma_ld` (or `vld`) before the instruction
       that consumes them. If these rows are a pad you wanted to be zero, note that
       no opcode zeroes a row: `dma_ld` is the only writer of spad and vr and it can
       only copy DRAM the host filled, so a zero pad has to be part of the spec's
       calling convention.
     in context:
         5  endloop  nr=  0 f=[0, 0, 0, 0]
         6  loop     nr=  2 f=[0, 0, 0, 0]
         7  mm       nr=  4 f=[0, 0, 0, 0]  f0+=iv0*4 f1+=iv0*4 f3+=iv0*4   <-- here
         8  mvout    nr=  4 f=[0, 0, 0, 0]  f0+=iv0*4 f1+=iv0*4
         9  endloop  nr=  0 f=[0, 0, 0, 0]

That message is the one this fork's own reference submission earned while it
was being written, and it names the fix.

``python act/judge.py rules`` prints the 23-rule table and runs its self-test.
The ISA fixtures are ``stress_isa._bad_programs`` -- one crafted violation per
contract, reused rather than rewritten -- plus ten built in
``act/rules_test.py`` for rules no shipped program breaks. **Measured in this
session: 29 crafted violations reach 21 of the 23 rules, every one of them
with a remedy, and none falls through to the remedy-less fallback.** The two
unreached rules are ``dma.span`` (``check_program``'s operand-range check
always fires first) and ``kpn.protocol``. The same command also asserts that
no baseline submission is rejected.

Is it correct?
--------------

.. code-block:: bash

   python examples/accelerator/tinytpu_vitis/act/judge.py correct
   python examples/accelerator/tinytpu_vitis/act/judge.py correct --spec relu_16x16 --no-simulator

Bit-exact, against two named anchors, and the verdict says which one moved:

1. **the program against the spec** -- ``isa_ref.run`` (the ISA as numpy,
   already verified and not rewritten here) against ``spec.gold`` (the einsum
   in int64, then the epilogue, then the narrowing). Every byte of the output
   region must match and nothing outside the write window may move. A failure
   here is the mapping.
2. **the design against the ISA** -- the module ``df.build`` produces against
   ``isa_ref.run``, on all ``MAXDIM * MAXDIM`` bytes of ``C`` with no window
   and no tolerance. A failure here is the hardware or the simulator, and the
   message says so.

Four operand distributions per spec (``small``, ``mid``, ``full``, ``corner``,
taken from ``stress_isa.operands``), ``C`` always prefilled with random bytes
so a clobber cannot hide behind a zero, and each case run twice on one build
so state a previous call left in ``spad`` / ``vr`` / ``ar`` shows up. This is
``stress_isa.py``'s discipline applied per spec rather than per shape.

Measured in this session on the reference submission: **16 of the 17 specs
bit-exact against both anchors; the seventeenth is
``gemm_reduce_8x6x8``, reported as its declared known gap** (64 of 64 result
bytes wrong, 0 clobbered -- the arbitrary reduction pad, exactly as the spec
predicts).

Is it fast?
-----------

.. code-block:: bash

   python examples/accelerator/tinytpu_vitis/act/judge.py fast
   python examples/accelerator/tinytpu_vitis/act/judge.py fast --spec gemm_16x16x16 --cosim

The cheap gate costs microseconds and involves no tool. Every unit in this
design is one flat loop over the work count the header promises it
(:doc:`/designs/tinytpu_isa`, the row-flattening note), and the units run
concurrently, so the steady state is the busiest unit's count and everything
else is fixed cost:

.. code-block:: text

   cycles = 173.2 + 1.621 * max(dma_ld, spm, vru, accu, dma_st)

Both counts come straight off the header ``assemble`` writes, so the gate
needs no new model of the machine -- it reads the number the assembler already
had to compute. The two constants are least squares over the five published
cosim points, which are Vitis measurements this fork attributes to
``dev/records/tinytpu/logs/cosim_isa_landed_sweep.log`` at ``e24e433b``. The gate passes a
submission whose estimate is within 10% of the reference submission's, and
``--cosim`` measures only what passes.

The real measurement is ``act/measure.py``: one ``csynth_design``, then one
``cosim_design`` per submission on that same RTL, with a testbench that
prefills ``C`` and compares every byte against ``isa_ref.run``, so a cosim
mismatch is an RTL fault and not a program fault. The Vitis path, the
``-B/usr/bin`` link flag that Vitis 2023.2's binutils needs against this
system's glibc, the ``m_axi`` depths cosim requires and Allo does not emit,
and the ``alignas(64)`` that the ``align_value(64)`` promise obliges the
testbench to keep are all imported from ``cosim.py`` rather than restated.
Point ``TPU_PRJ`` somewhere with room and delete the project afterwards; disk
on this host is tight.

``cosim_design`` runs under a deadline (``ACT_COSIM_TIMEOUT``, default 600 s),
and on expiry the whole process group -- ``vitis_hls`` and the ``xsim`` it
spawned -- is killed and the verdict reads ``RTL DID NOT COMPLETE``, with what
csim said and the last inter-transaction progress line. A submission whose RTL
never finishes is a verdict about the submission; it must not be a hung judge.
That case is not hypothetical: see :ref:`act-specs-rtl-hang`.


.. _act-specs-kpn-verdict:

Is ``kpn_model.py`` good enough to be the cheap gate?
=====================================================

**No, and it does not claim to be.** As shipped it reports no cycle number at
all: it is a deadlock and protocol model with bounded FIFOs, it answers
"minimum channel depth" and "which process is blocked on which channel", and
its own docstring says data values are not modelled. Run over every shipped
program it prints ``minimum depth 1`` and ``KPN OK`` and no cycles
(reproduced in this session).

It can be *made* to produce a cycle number, and ``cycles.kpn_rounds`` does it
without touching the model: ``kpn_model.build`` hands back the process
network, and stepping it barrier-synchronously -- one channel action per
process per round -- counts the cycles a KPN with II=1 links and no memory
latency would take. That number is worse than free and worse than the header
counts. Single-feature least squares over the five published points, all
measured in this session:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - predictor
     - cost to compute
     - max \|error\| in sample, 5 points
   * - ``kpn_rounds`` (barrier-synchronous KPN)
     - 0.6-16 ms
     - **12.6%**
   * - dynamic instruction count
     - 0.1-0.3 ms
     - 12.9%
   * - ``accu`` header count
     - 0.1-0.3 ms
     - 13.1%
   * - ``dma_ld`` header count
     - 0.1-0.3 ms
     - 8.0%
   * - ``vru`` header count
     - 0.1-0.3 ms
     - 8.7%
   * - ``max`` over the five unit counts
     - 0.1-0.3 ms
     - **9.2%**

(Timed in this session: 116-314 us for ``cycles.estimate``, 0.6-15.5 ms for
``cycles.kpn_rounds``, on the same programs.)

So the KPN round count is *less* accurate than a single number the assembler
already computes, and fifty times more expensive: at ``gemm_16x16x16`` it
predicts 1484 rounds against 686 measured cycles, and the 1484 is dominated by
the 4x4 array's 32 weight-loader and PE processes each taking a round to pass a
token along a chain that the RTL walks in one cycle. A barrier-synchronous
scheduler charges a round per *process* per hop; the hardware charges a cycle
per *link*. Fixing that means modelling the array's chains as pipelines rather
than as processes, which is a different model, not a calibration of this one.

It is not a complete deadlock oracle either. On ``relu_16x16`` it reports
``minimum channel depth 1`` and no deadlock, and the RTL does not complete
(:ref:`act-specs-rtl-hang`). It models the channel *counts* -- who is promised
how much work -- and not the latencies and pipeline depths a real hang can
come from, so it catches assembler/microarchitecture mismatches and nothing
else. That is still worth having, and it is worth knowing its edge.

``kpn_model``'s real value is the layer it already occupies in the judge: it
is the ``protocol`` check in ``judge.py legal``, where a promised-work
mismatch is named with the blocked process instead of hanging the simulator.
It should stay there and stay out of the cycle gate.

.. _act-specs-gate-measured:

What the cheap gate is worth
============================

Measured with ``act/calibrate.py``, which runs one ``csynth_design`` and then
one ``cosim_design`` per program on that same RTL. Everything in this section
was measured in this session (``dev/records/tinytpu/logs/cosim_act_corpus_sweep.log`` and
``dev/records/tinytpu/logs/cosim_act_relu_hang.log``); ``PUBLISHED_CYCLES`` in ``cycles.py`` is the
only attributed number on the page.

.. code-block:: bash

   TPU_PRJ=/somewhere/with/room python act/calibrate.py specs
   TPU_PRJ=/somewhere/with/room python act/calibrate.py variants

.. list-table::
   :header-rows: 1
   :widths: 26 10 8 10 9 9 8

   * - spec
     - critical
     - work
     - estimate
     - cosim
     - error
     - in fit
   * - ``gemm_4x4x4``
     - ``spm``
     - 9
     - 188
     - **172**
     - +9.2%
     - yes
   * - ``gemm_8x8x8``
     - ``vru``
     - 48
     - 251
     - **262**
     - -4.2%
     - yes
   * - ``gemm_12x12x12``
     - ``vru``
     - 144
     - 407
     - **418**
     - -2.7%
     - yes
   * - ``gemm_16x16x8``
     - ``vru``
     - 192
     - 484
     - **484**
     - +0.1%
     - yes
   * - ``gemm_16x16x16``
     - ``vru``
     - 320
     - 692
     - **686**
     - +0.9%
     - yes
   * - ``gemm_relu_16x16x16``
     - ``accu``
     - 384
     - 796
     - 750
     - +6.1%
     - no
   * - ``gemm_reuse_m_16x16x4``
     - ``vru``
     - 128
     - 381
     - 383
     - -0.6%
     - no
   * - ``gemm_reuse_n_4x4x16``
     - ``spm``
     - 36
     - 232
     - 269
     - -13.9%
     - no
   * - ``gemm_reuse_k_4x16x4``
     - ``spm``
     - 36
     - 232
     - 256
     - -9.5%
     - no
   * - ``batched_matmul_2x4x4x4``
     - ``spm``
     - 18
     - 202
     - 209
     - -3.2%
     - no
   * - ``row_reduce_16x16``
     - ``vru``
     - 128
     - 381
     - 371
     - +2.6%
     - no
   * - ``relu_16x16``
     - ``accu``
     - 192
     - 484
     - *never*
     - --
     - no

Two things fall out of the left column before the model is even discussed.
**The five published cycle counts reproduce exactly** -- 172 / 262 / 418 /
484 / 686 -- through a testbench that compares all 256 bytes of ``C`` against
``isa_ref.run`` rather than only the ``M x N`` region, which is a stricter
check than the one the published numbers come from. And two of the corpus's
three non-GEMM einsums run on real RTL: ``batched_matmul_2x4x4x4``
(``bmk,bkn->bmn``) at 209 cycles and ``row_reduce_16x16`` (``mk->m``) at 371,
both with 0 of 256 bytes wrong. The third, ``relu_16x16`` (``mn->mn``), is
:ref:`the one that does not <act-specs-rtl-hang>`.

On the model: **in sample, max error 9.2% over the five points it is fitted
to; out of sample, max error 13.9% and mean absolute error 6.0% over the six
held-out specs that produced a cycle number.** The two worst cases are both
ones where ``spm`` is
the critical unit at a small work count, where what the design is actually
doing is filling the array's weight and activation chains -- a term the model
does not have, because ``max`` over the unit counts cannot see a pipeline it
does not model.

A 6% mean is good enough to sort candidates that differ by more than that and
useless for candidates that differ by less. The gate is set at 10% of the
reference submission's estimate for that reason: it is a filter on the
obviously-worse, not a ranking.

.. _act-specs-rtl-hang:

What the expensive judge found that nothing cheaper did
-------------------------------------------------------

**Two of the programs the judge was given pass every cheap check and then do
not complete in RTL**: ``relu_16x16``'s submission, and the ``row_blocked``
variant of ``gemm_8x8x8``. For both, ``check_program`` accepts, ``kpn_model``
reports minimum channel depth 1 and no deadlock, the Allo simulator is
bit-exact against ``isa_ref``, and Vitis csim reports 0 of 256 bytes wrong --
and then ``cosim_design`` never finishes the transaction.

No deadlock is reported by Vitis' own detector, so this page does not call it
one, and nothing here locates the stall. What a hanging run's log looks like
beside a completing one is in `Earlier measurements and corrections`_.

The full characterisation, the bisection, and the two hypotheses it rules out
are :ref:`item 24 <limitation-24>` of the limitations register, with the repro
in ``tests/limits/item24_cosim_hang.py`` and the family in
``act/rtl_hang.py``. Two things from it are worth carrying here, because they
are about how a compiler should be judged rather than about this design:

- **The minimal case is sixteen instructions, fully unrolled, with no
  ``vrelu`` and no epilogue** -- plain int8 GEMM tiling. Nothing exotic has to
  be generated to reach it.
- **The hypothesis that an in-nest transfer is the trigger is measurably
  false.** :doc:`act` leaves open whether "a transfer inside the emitted nest
  rather than hoisted into a prologue" is what the failing mappings share. It
  is not: both non-completing programs here stage every transfer in a
  prologue, and both ``weights_reloaded`` mappings, which issue a ``dma_ld``
  between ``mvout``\ s inside the output nest, complete (256 cycles at 8x8x8
  and 638 at 16x16x16, 0 of 256 bytes wrong). A staging column would pass both
  failures and flag two mappings that run, so it is not the guard to use.

The part that belongs here is what it does to the judge.

Three tiers, and an estimate is not one of them
-----------------------------------------------

The consequence for this judge is that **"is it fast" has a tier above it:
does it run on RTL at all**, and the cheap gate cannot see that tier. So a
result now says which tier it reached:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - tier
     - what has been shown
   * - ``legal``
     - ``check_program``, the ``assemble`` limits, ``kpn_model``, and the
       spec's write window accept the program
   * - ``correct``
     - it computes the spec, bit-exact, on the Allo simulator against
       ``isa_ref``
   * - ``confirmed``
     - Vitis cosim ran it to completion and produced a cycle count

``judge.py fast`` without ``--cosim`` prints in as many words that nothing it
listed is confirmed; with ``--cosim`` it carries a ``tier`` column and exits
nonzero naming anything the RTL did not finish -- **rankable, not pickable**.
``calibrate.py`` marks the same thing. This is the convention ``act_compile.py``
adopted on ``main`` (*encodable* versus *confirmed*), followed rather than
reinvented.

A cycle *estimate* is deliberately not on that ladder. Both hanging programs
have estimates -- 484 and 264 cycles -- and a gate that ranked on them would
have preferred one of them to a program that runs.

This is the reason the judge has an expensive tier at all, and it is a
concrete instance of what :doc:`/designs/tinytpu_isa` calls cosim's role as a
deadlock oracle. It is not a defect in the spec or in the submission: both
pass every contract the machine states.


How to work on this
===================

* ``python act/spec.py`` -- validate the whole corpus and print it.
* ``python act/baseline.py`` -- the reference submission's static and dynamic
  instruction counts per spec.
* ``python act/judge.py rules`` -- the rule table and its self-test. Run it
  after any change to ``check_program``'s messages; it is what keeps a
  rejection from degrading into a bare exception.
* ``python act/judge.py legal`` then ``correct`` then ``fast`` -- the three
  verdicts, cheap to expensive.
* ``python act/cycles.py`` -- the per-unit work table, the estimate and the
  KPN round count for every spec, side by side.
* ``python act/variants.py gemm_16x16x16`` -- several legal mappings of one
  spec with their estimates, which is the fixture the rank test uses.

Environment is the fork's usual one (``CLAUDE.md``): the ``allo`` env,
``LLVM_BUILD_DIR`` exported by hand, ``OMP_NUM_THREADS=8``. Only ``correct``
and ``fast --cosim`` need a build; ``legal``, ``specs`` and ``rules`` are pure
Python over the corpus.


Earlier measurements and corrections
====================================

What a hanging cosim run looks like in the log
----------------------------------------------

What that looks like, and what it does not: a completing run of the same
design prints two progress lines and a ``$finish`` --
``0 / 1 @ "109000"``, then ``1 / 1 @ "777000"``, then
``$finish called at time : 796590 ps`` for a 198-cycle program. A hanging run
prints the **first** line and never the second. ``109000`` is picoseconds and
is simply where Vitis makes its first periodic report, so it is the same
number in every log, passing or hanging; it is not where the design stops, and
nothing here locates the stall.

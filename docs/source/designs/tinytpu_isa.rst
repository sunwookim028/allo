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

############################################
TinyTPU-isa: A Programmable GEMM Accelerator
############################################

TinyTPU-isa (``examples/tinytpu/``) is an int8
instruction-programmable tiled-GEMM accelerator written in grid Allo
(``@df.region`` / ``@df.kernel``) and taken through the Vitis HLS dataflow path to
**RTL co-simulation**. Each of its units is a module of its own under
``ip/units/``, composed into a region by ``allo/compose.py``; that decomposition
is :doc:`tinytpu_library`, and ``microarch_isa.py`` is the instantiation the
published numbers were measured on. It is the machine the project's requirements name: an ISA,
a vector unit, a SIMD scratchpad, vector registers streaming to the array's
ports, and tiled GEMM as a *program* rather than as a fixed-function datapath.

It is the only accelerator design left on ``main``. Its predecessors, the
optimisation history that produced the current numbers, and the superseded
readings of those numbers are on :doc:`tinytpu_history`. The data-type- and
mesh-matched comparison against Gemmini is on :doc:`gemmini_comparison`.
Every cycle count, resource figure, correction and retraction is on
:doc:`tinytpu_isa_results`.

Quick start
-----------

One command takes a clean checkout to the published cycle counts:

.. code-block:: bash

   # One command, from a clean checkout: builds this checkout's MLIR bindings,
   # runs bench_isa + stress_isa, then the default cosim, and checks the five
   # cycle counts against 175 / 265 / 421 / 482 / 674 at the TPU_MAXDIM=16 it
   # pins. Exits nonzero otherwise.
   examples/tinytpu/reproduce.sh            # ~6 min, incl. a fresh mlir build
   examples/tinytpu/reproduce.sh --no-cosim # functional, ~1 min

   # Or by hand, from examples/tinytpu (the env sets neither variable):
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   python bench_isa.py                   # published functional setup: ALL EXACT
   python bench_isa.py 8 8 8             # one shape
   python stress_isa.py                  # correctness gate:          STRESS OK
   python kpn_model.py                   # channel protocol / deadlock model
   python cosim.py                       # default TB: the published cycle counts
   TPU_TB=stress TPU_SHAPES=4x4x4,16x16x16 python cosim.py   # correctness in RTL
   python mutate.py                      # does the harness catch a broken design? (~15 min)
   python mutate.py --no-rtl             # the same without the one cosim (~5 min)
   TPU_WRAP=1 python cosim.py            # the old hoisted-argument variant, for comparison

``bench_isa.py`` and ``cosim.py``'s default testbench use Gemmini's ``[-4, 4]``
operands so the comparison is like for like; they are **performance** checks
and are blind to an int16 accumulator, a wrong clip boundary, or a unit ignoring
a field GEMM never varies. ``stress_isa.py`` and ``TPU_TB=stress`` are the
**correctness** checks; ``mutate.py`` prints which level catches which bug. Any
``TPU_*`` variable left set changes what ``cosim.py`` measures. See
:ref:`tinytpu-isa-verify` for which gate to run after which kind of change.

How it works
------------


Eight kinds of unit, ``2*T*T + 6 = 38`` concurrent processes at ``T=4``.

**Control.** One unit fetches and decodes, and sends each of the five
instruction-executing units only the instructions it executes:

.. code-block:: text

                        +-----------+
     imem ------------->| sequencer |  fetch, decode, resolve the AGU
    (m_axi)             | owns imem |  terms, dispatch -- in dataflow
                        +-----+-----+  order, which is load-bearing
                              |
        c_dld    c_spm    c_vru    c_acc    c_dst    Stream[UInt(64), QD]
          |        |        |        |        |
          v        v        v        v        v
       dma_ld     spm      vru      accu    dma_st

``wld`` and ``pe`` get no control word at all: their trip count rides the
header word down the weight chain, so the array decodes nothing.

**Data.** Every arrow is a declared ``Stream`` one packed word wide --
``UInt(T*8)`` for operands, ``UInt(T*32)`` for accumulator words -- and no
state is shared between units:

.. code-block:: text

    A, B (m_axi)
        |
        v
    +--------+  dma2sp   +-----------+   wcol[0]   +-----------+
    | dma_ld |---------->| spm       |------------>| wld[T][T] |
    | owns   |           | owns spad |  header +   | one per PE|
    | A, B   |           +-----------+  T weights  | wrow east |
    +--------+                 |                   +-----------+
        |                      | sp2vr (vld)             |
        | dma2vr               v                         | wq,
        |                +-----------+                   | depth 4:
        +--------------->| vru       |                   | the
                         | owns vr   |                   | shadow
                         +-----------+                   | weight
                               | acol[0]                 |
                               | activations             |
                               v                         v
                        +--------------------------------+
                        |            pe[T][T]            |
                        | weight-stationary MAC,         |
                        | a_fwd east, p_fwd south        |
                        +---+----------------------------+
                            | cw[T-1]
                            v
                        +--------+   ac2sp   +--------+
                        |  accu  |---------->| dma_st |------> C
                        | owns ar|  int8,    | owns C |     (m_axi)
                        | + the  |  clipped  +--------+
                        | vector |
                        |  ALU   |
                        +--------+

The eight kinds of unit:

.. list-table::
   :header-rows: 1

   * - unit
     - owns
     - does
   * - ``sequencer``
     - ``imem``
     - fetch, decode, dispatch point-to-point; consumes nothing
   * - ``dma_ld``
     - ``A``, ``B``
     - DRAM -> scratchpad or operand vregs, packing T lanes/cycle
   * - ``spm``
     - ``spad``
     - the scratchpad; **pure SIMD access**; the array's weight port
   * - ``vru``
     - ``vr``
     - operand vregs; the array's activation port
   * - ``wld`` x ``T*T`` (16)
     - --
     - one per PE: walks the weight chain, double-buffers its PE's weight
   * - ``pe`` x ``T*T`` (16)
     - its lane
     - weight-stationary MAC, decodes nothing, one flat loop
   * - ``accu``
     - ``ar``
     - accumulator vregs **and the vector ALU**, one flat loop at II=1
   * - ``dma_st``
     - ``C``
     - accumulator -> DRAM, clipped (executes ``mvout``)

**A vector unit that tiled GEMM needed.** As designed, ``mm`` computed the psums
of *one* k-tile and wrote them to accumulator registers, and summing across
k-tiles was an explicit ``vadd`` -- load-bearing, not decoration. This is
Gemmini's split too:
the adds live in ``AccumulatorMem``'s write path, not in the mesh. In the
shipped program ``mm`` carries an overwrite/accumulate field (``f2``) and the
GEMM inner loop no longer uses ``vadd`` at all, so ``vadd_program`` exists to
keep ``vadd``/``vrelu`` exercised; see :ref:`tinytpu-isa-programs`.

**Pure SIMD scratchpad.** A row of ``spad`` *is* one ``UInt(T*8)`` packed word of
T int8 lanes; there is no way to address a lane. That is what lets a
single-ported memory feed T lanes per cycle, and it satisfies ``HLS 200-779``
(single reader, single writer) without a pragma. The diagnostic, as Vitis first
raised it on the single-grid predecessor, is quoted on :doc:`tinytpu_history`.

**The PEs decode nothing.** A header word leads every ``mm`` down the chain the
weights use, carrying just its row count. Each PE's weight loader (``wld``)
forwards it and hands its PE one ``(weight, rows)`` word through a depth-4 FIFO
-- the shadow register, so ``mm`` n+1's weight is latched while ``mm`` n
computes (Gemmini's c1/c2 double buffer); there is no command fan-out to
``T*T`` PEs and no opcode in the array. Gemmini is the same -- its PEs are dumb and ``ExecuteController``
decodes -- and it also means the instruction set can grow without touching the
array.

Chains rather than fan-out
~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight-stationary predecessor (``microarch_ws.py``, removed) had a ``loader``
writing ``a_in[0..T-1]`` and a ``drainer`` reading ``c_out[0..T-1]``, both in fixed
order. That design needs stream depth proportional to ``M * NI`` -- the entire
run -- which means back-pressure never engages (``RESULTS_WS.md`` section 6; both
files at ``e2451b81``). Upstream's ``test_multi_cache_gemm.py`` has no process of
that shape: ``offchip_loadA`` writes exactly one stream and the border PEs
daisy-chain it, ``L2_A[i] -> L2_A[i+1]``. ``test_tiled_systolic.py`` runs on
depth-**4** FIFOs for the same reason.

So every distribution here is a chain, and every chain carries *packed words*:

.. code-block:: text

    spm --hdr,W--> wcol[0] --v--> wld(i,0) --> wcol[i+1]     (down column 0)
                                    |    +--wq--> PE(i,0)    (its own weight)
                                    +-------> wrow[i,0] --> wld(i,1) --> ...
                                                             (east, lane j)

    vru --A------> acol[0] --v--> PE(i,0) --> acol[i+1]      (down column 0)
                                    |  lane i
                                    +-------> a_fwd[i,0] --> east, scalar

    PE(T-1,0) --cw[0]--> PE(T-1,1) --cw[1]--> ... --cw[T-1]--> accu
              (the bottom row packs its psums into one word as it goes east)

One word per cycle per link, so T lanes per cycle, from single-ported memories.

Hazards come from being in-order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each unit consumes the instruction stream in order and every channel is
point-to-point, so ``dma_ld`` before ``mm`` before ``vadd`` before ``mvout`` is
enforced by construction: within a unit by program order, across units by
stream order. That is the whole of the hazard logic *between* units. Inside
``accu`` one more rule holds, and the assembler enforces it: an accumulator
row may not be read within ``AR_RAW_DIST`` rows of being written
(:ref:`tinytpu-isa-dependence`). Gemmini spends a 48-entry
reservation station (``ReservationStation.scala``) to get out-of-order issue on
top of this; this design is strictly in-order and does not pretend otherwise.

Two structural rules that fixed real deadlocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both fixes were the same shape, and both match Gemmini's structure:

1. **One unit doing DMA in both directions deadlocks.** A single ``dma`` unit put
   ``spm`` in a two-process cycle (``dma -> dma2sp -> spm`` *and*
   ``spm -> sp2dma -> dma``). The round trip ``dma_ld; dma_st`` hung with **every**
   body variant tried -- constant trip counts on both sides, no ``meta_for``, no
   store, a constant put. Splitting into one-way ``dma_ld`` and ``dma_st`` units
   fixed it immediately and exactly. Gemmini splits the same way
   (``LoadController.scala`` / ``StoreController.scala``).
2. **Writing results back into the input scratchpad deadlocks.** The original
   ``vst`` (accumulator -> scratchpad) hung even when its body was reduced to
   ``ac2sp.put(123)`` -- no ``ar`` read, no packing, no clipping -- and with
   compile-time trip counts on both sides, and at ``QD=256``. The put side alone
   ran; the get side alone ran; together they hung. It could not be reduced to
   a minimal repro, so the mechanism is still unexplained.

   The fix was architectural: **make the accumulator the output memory.**
   ``mvout`` reads ``ar`` and writes DRAM directly, the scratchpad becomes
   input-only, and the back edge disappears. This is exactly Gemmini:
   ``AccumulatorMem`` is a separate memory from the scratchpad, and
   ``StoreController`` reads it directly -- results never re-enter the input
   scratchpad. It also collapsed two instructions (``vst`` + ``dma_st``) into
   one.

That an architecture-level property (where the output memory lives) is what
makes a design runnable, while the symptom is an unexplained silent hang, is the
strongest argument for a deadlock report in the simulator: naming the blocked
processes and the full channels would have replaced a long bisection with one
run. (The simulator has since gained a tier-0 watchdog; the per-channel report
is still open -- see :ref:`limitation-11`.)

.. _tinytpu-isa-one-owner:

One owner per memory
~~~~~~~~~~~~~~~~~~~~

``spad`` lives in ``spm`` and is written by ``dma_ld`` and read by ``vld`` and
``mm`` (weights); ``vr`` lives in ``vru`` and is written by ``dma_ld`` and
``vld`` and read by ``mm`` (activations); ``ar`` lives in ``accu`` and is
touched by all four of its opcodes.

**The rule is Allo's, not Vitis's.** Allo enforces single reader and single
writer -- it rejects a region-scope ``Stateful`` shared by two kernels
(``EmitVivadoHLS.cpp:3083-3122``) -- so those three units keep every arm of
their opcode dispatch in one process. This rules out a one-process-per-opcode
split; see :ref:`tinytpu-history-rowflat`. Vitis is not what refuses: a Vitis
2023.2 probe (`impact/probe_shared/
<https://github.com/sunwookim028/allo/tree/main/examples/tinytpu/impact/probe_shared>`__)
shows that ``#pragma HLS stream variable=buf type=unsync`` makes it share an
on-chip array between two processes, one per BRAM port (``HLS 200-824``,
``200-755``, ``200-634``), and ``HLS 200-779`` applies only to *synchronized*
arrays; Allo never emits ``stream type=unsync``. Vitis does separately forbid
one ``m_axi`` bundle read by two processes (``HLS 200-1013`` / ``200-984``).

**Measured impact of the rule on this design: 0 cycles** -- every restructure
the gap attribution needed was Allo-legal (:ref:`gemmini-gap-attribution`).
See :ref:`limitation-shared-memory`. This paragraph charged the rule to both
tools until 2026-09-19, and a claim that rested on that reading was withdrawn;
see :doc:`tinytpu_isa_results`.

Data type
~~~~~~~~~

Packing is what makes the SIMD scratchpad work, and packing needs integers, so
unlike ``microarch_ws.py`` (at ``e2451b81``) there is no fp32 switch. The
widths, the exactness of each stage and the output conversion are the
:ref:`tinytpu-isa-spec` numerics table; the configuration shipped is Gemmini's
default (``inputType = SInt(8.W)``, ``accType = SInt(32.W)``) with its
``mvout`` behaviour under ``ACC_SCALE_IDENTITY`` and shift 0, which is what
``allo_cmp.c`` passes.

Reference
---------

The instruction encoding -- bit layout, opcodes, the header, the memory map,
the parameter set and the numerics -- is generated from ``isa_spec.json`` and
is on its own page: :doc:`tinytpu_isa_spec`.

Hardware parameters
~~~~~~~~~~~~~~~~~~~

Every constant is fixed at build time and **independent of the workload**: one
RTL build runs every shape, with M, K and N arriving as instruction fields. This
is the property the Gemmini comparison needs -- Gemmini's numbers come from one
elaboration, and ``allo_cmp.c`` passes ``MAXDIM`` as the stride for every shape.
The parameters, their defaults, their legal ranges, their cross-constraints and
what bounds ``MAXDIM`` are tabulated in :ref:`tinytpu-isa-spec`, generated from
``isa_spec.json``.

``T`` is the one parameter that changes the *shape* of the generated region: the
array is ``T*T`` kernel instances, so ``T=16`` is 262 instances and ~800 streams.
That was unrunnable until the simulator's OpenMP team was sized to the section
count (:ref:`limitation-11`); before that fix it hung with no output.

``A``, ``B`` and ``C`` are **flat** at the region boundary, addressed
``row * MAXDIM + col``. That is what DRAM is, and it is what makes
``wrap_io=False`` legal -- it refuses multi-dimensional arguments to nested
kernels (:ref:`limitation-14`). ``schedule()`` partitions ``A``, ``B`` and ``C``
cyclically by ``T``; this is reachable because ``df.build`` is
``customize(func)`` followed by ``s.build(...)`` (:ref:`limitation-17`).

The build is ``wrap_io=False`` with ``configs={"align_value": 64}`` and
``config_interface -m_axi_max_widen_bitwidth 512``; how those came to be the
build is on :doc:`tinytpu_history`, and the Vitis-side mechanics are in
:doc:`/backends/vitis`.


.. _tinytpu-isa-build:

Files, commands and knobs
~~~~~~~~~~~~~~~~~~~~~~~~~


Files in ``examples/tinytpu/``:

.. list-table::
   :widths: 25 75

   * - ``ip/``
     - the unit library: one module per unit under ``ip/units/``, the
       composition (``compose.py``), the parameter set, the ISA encoder, the
       assembler, the reference programs (:doc:`tinytpu_library`)
   * - ``microarch_isa.py``
     - the shipped instantiation: the parameter set from the environment, and
       the names the harness imports
   * - ``isa_dsl.py``
     - the loop-nest generator (``gemm_program``)
   * - ``reproduce.sh``
     - one command from a clean checkout to the published cycle counts (see
       :ref:`tinytpu-isa-verify`)
   * - ``bench_isa.py``
     - functional sweep on the Allo dataflow simulator, one build -- the
       **performance** setup
   * - ``stress_isa.py``
     - the functional **correctness** gate (full-range operands, every shape,
       prefilled ``C``, non-GEMM and random programs, the validator)
   * - ``isa_ref.py``
     - the ISA as numpy: the reference for any program, not just GEMM
   * - ``mutate.py``
     - mutation testing of the verification harness
   * - ``cosim.py``
     - one csynth, then Vitis ``cosim`` per shape on the same RTL
   * - ``saif_capture.py``
     - a switching-activity file (SAIF) from a project ``cosim.py`` has
       already run in (:ref:`tinytpu-isa-saif`)
   * - ``kpn_model.py``
     - a KPN model of the channel graph with bounded FIFOs and deadlock
       reporting
   * - ``gemmini/``
     - the patches and benchmarks for the matched Gemmini baseline
       (:ref:`gemmini-reproduce`)
   * - ``impact/``
     - the gap attribution's variants (generated from the pre-landing
       baseline), the Vitis shared-array probe, and the RTL probe of the
       accumulator's dependence claim (:ref:`gemmini-attribution-reproduce`)

The csynth/cosim reports and sweep logs behind the numbers on these pages,
and the gap attribution's raw results and timelines, are evidence rather than
part of the design: they live under ``dev/records/tinytpu/logs/`` and
``dev/records/tinytpu/impact-results/`` at the repository root, not in
``examples/``.


Since the fix recorded in :ref:`limitation-11` the simulator sizes its OpenMP
team to the section count itself, so ``OMP_NUM_THREADS=8`` runs the 38-process
design. ``kpn_model.py`` is driven by the assembled header (``ef112868``);
every shipped program completes in it at FIFO depth **1**. See
``dev/toolchains.rst`` for ``LLVM_BUILD_DIR`` and the Vitis install.
Older reproduce instructions, and the thread count the recorded numbers were
produced at, are in :doc:`tinytpu_isa_results`.

``bench_isa.py`` first asserts that the generated program is bit-identical to
the hand-written one, then that the looped and unrolled programs expand to the
identical dynamic stream (every resolved address field, not just the opcode and
row-count stream), then builds the simulator **once** and runs ``gemm`` and
``gemm.relu`` in both looped and flat forms at every shape, plus
``vadd_program``, against a numpy reference *including the int8 clip*. It ends
``ALL EXACT`` or ``FAILURES``.

``cosim.py`` knobs, all environment variables:

.. list-table::
   :widths: 30 70

   * - ``TPU_SHAPES=4x4x4,16x16x16``
     - restrict the sweep (default: every shape in 4x4x4, 8x8x8, 12x12x12,
       16x16x8, 16x16x16 that is a multiple of ``T``)
   * - ``TPU_AXI_LATENCY=<n>``
     - add ``config_interface -m_axi_latency <n>`` to the csynth script
       (default: unset, i.e. Vitis's default of 0)
   * - ``TPU_RANDOM_STALL=1``
     - pass ``-random_stall`` to ``cosim_design``
   * - ``TPU_WRAP=1``
     - build with ``wrap_io=True`` (the hoisted-argument variant)
   * - ``TPU_TB=stress``
     - the **correctness** testbench on the same RTL (below); unset is the
       default performance testbench, the only mode the published cycle counts
       come from

The toolchain fixes ``cosim.py`` applies (a plain C++ testbench, ``-B/usr/bin``,
explicit ``m_axi`` depths, ``alignas(64)`` arrays) are general to any Allo
dataflow design and are documented in :doc:`/backends/vitis`.

.. _tinytpu-isa-saif:

Switching activity from the cosim run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``saif_capture.py`` writes a **SAIF** -- toggle counts and time-in-state per
net -- from the same xsim run whose cycle count is published, so an annotated
power analysis and the cycle figure would describe one workload rather than
two.

**It does not go in a table, and that is settled.** Power is *absent* from the
evaluation, for a reason activity annotation does not touch: under
``sram_mode='none'`` and with no place-and-route, the power from that flow is
dominated by clock power into flip-flop arrays and by guessed wire
capacitance, both larger than the effect being measured. Annotation removes
the default-toggle-rate objection and neither of the other two, so an
annotated number from this flow would be compromised rather than merely
imprecise. The cheapest credible energy axis is SRAM macros plus P&R -- the
same prerequisite that would replace the area methodology wholesale. What this
script buys is that when that day comes, the activity side is already done and
is not what is holding it up.

**Where it injects.** Vitis regenerates ``sim/verilog/tinytpu_isa.tcl`` and
``sim/verilog/run_xsim.sh`` on every ``cosim_design``, so neither is edited:
an edit would be undone by the next run, and it would make the published cycle
counts depend on the instrumentation. Instead the script runs *after* a normal
``cosim.py`` and re-runs **only the xsim step**, from sibling files it writes
itself (``saif_dump.tcl``, ``saif_xsim.sh``) into a separate snapshot. The one
change to the elaboration is ``-debug typical``, which is what makes the
design's internal nets visible to ``get_objects`` at all -- Vitis's
``-trace_level none`` elaboration carries no debug and ``log_saif`` would see
nothing. It adds no logic and changes no timing, and the first check below is
what holds that claim to evidence.

**The DUT instance path**, which a downstream ``read_saif -map_names
-instance_name`` needs and which is not guessable, because Vitis wraps the
design in ``apatb_*`` layers:

.. code-block:: text

   apatb_tinytpu_isa_top/AESL_inst_tinytpu_isa

Read out of the elaborated scope tree, not out of the generator. The script
re-reads it from each SAIF it writes rather than asserting it, since the
wrapper naming is Vitis's to change.

**What it refuses to report as produced.** A SAIF that is empty, that holds
only the testbench's nets, or whose counts are all zero annotates *cleanly*
and yields a beautiful, meaningless power number -- the failure mode this
project keeps catching, an instrument that succeeds without having run. So
four things are checked, and ``--selftest`` feeds the content guards the three
files they exist to reject, in seconds and without Vitis:

1. the latency in ``tinytpu_isa.result.lat.rb`` is byte-identical to the one
   the original cosim measured. If SAIF logging moved the cycle count, that is
   a bug in the instrumentation, not a new result;
2. the RTL output vectors still match the C golden vectors, so the
   re-simulation computed the workload that is quoted;
3. the SAIF names instances *inside* the DUT, and most of the activity is in
   them rather than on its ports;
4. the toggle counts are not all zero.

.. code-block:: bash

   python saif_capture.py --selftest          # the guards, rejecting bad SAIFs

   TPU_MAXDIM=16 TPU_SHAPES=16x16x16 TPU_PRJ=$PWD/saif_t4.prj python cosim.py
   python saif_capture.py saif_t4.prj -o <dir>/run.saif

Measured on the two handoff configurations at **16x16x16**, the largest of the
five published shapes and so the most representative activity:

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 14 14 12 12

   * - configuration
     - cycles
     - window
     - instances
     - nets
     - toggling
     - size
   * - ``T4_MAXDIM16``
     - 685
     - 727.7 clk
     - 548
     - 444 258
     - 67 633
     - 41 MB
   * - ``T8_MAXDIM64``
     - 493
     - 535.1 clk
     - 1 504
     - 965 573
     - 243 424
     - 87 MB

Both cycle counts are byte-identical to the ones their own cosim measured, so
the instrumentation moved nothing. **Both captures predate ``QD=16``**, which
is why the ``T4_MAXDIM16`` row reads 685 rather than the 674 the same
configuration takes today; the activity has not been re-captured against the
shipped RTL. The window is wider than the cycle count
because it spans the ``s_axi_control`` programming around the kernel as well:
42 clocks of it at ``T4``, 42 at ``T8``. Activity is 4 990 787 transitions at
``T4`` and 5 552 502 at ``T8``, 94% and 95% of it inside the DUT's submodules
rather than on its ports, with the accumulator's pipeline the hottest block in
both. Every PE appears with nonzero activity -- all 16 of the 4x4 array, all
64 of the 8x8 -- which is the check that the SAIF describes the array and not
just the wrapper around it. The files are 41 MB and 87 MB: SAIF is a per-net
summary, so size follows the net count, not the run length.

.. _tinytpu-isa-verify:

.. _tinytpu-isa-conformance:

The spec, and holding both consumers to it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``isa_ref.py`` was derived from ``microarch_isa.py``'s own comments and
imported its opcode numbers, its field layout and its address resolution. It
therefore agreed with the design about everything, *because it shared the
design's assumptions* -- a restatement, not an oracle. That is invisible while
the arithmetic is int8 and exact at every stage, and becomes a correctness
hazard as soon as it is not.

So the ISA facts live in one file, ``isa_spec.json``, and the two consumers are
held to it from opposite ends. This is MiniTPU's arrangement
(:doc:`minitpu`), which has already caught real errors there.

**Generated from the spec** (checked in, and ``gen_isa.py --check``
regenerates each one and fails on any byte of difference, so drift is caught
rather than discovered):

* ``isa_encoding.py`` -- the constants, the encoder, a decoder that returns
  operands under the spec's *names*, the control-flow and AGU resolver, the
  header builder and the numerics.
* the tables in :ref:`tinytpu-isa-spec`, between the ``GENERATED`` markers.

**Checked against the spec, not generated.** ``microarch_isa.py`` keeps its
constants and bit slices written out: it is the file Vitis synthesises and the
file a CHIA candidate edits, and ``chia_agent/spec_policy.py`` admits no import
but ``isa_dsl``. MiniTPU's ``board_package/asm.py`` keeps its copies for the
same reason. ``gen_isa.py --check`` holds it to the spec four ways -- every
named constant by value; every bit slice it takes of a 64-bit instruction word
against the field table; ``enc``/``enc_agu``, ``expand`` and ``assemble``'s
header against the spec's own implementations over 73 programs; and every
parameter in range, still read from its environment variable.
``gen_isa.py --conform`` adds the one check that does not read Python at all:
it builds the design down the HLS path and holds **the bit ranges the emitted
C++ actually takes** to the same table.

``allo.encoding.TINYTPU_ISA`` is checked the same way. It is the compiler-side
descriptor of what one instruction word can carry
(:doc:`/developer/extending_allo`) and it imports nothing from ``examples/``,
so its ``address_terms`` and ``loop_depth`` are a third written-out copy of
``AGU_TERMS`` and ``LOOP_DEPTH`` -- held to ``agu.terms`` and
``loop_stack.depth`` by value.

**Same machine, not merely same rules.** A parameter's *default* is part of
the spec, because the reference model has to build the machine the design
built. ``gen_isa.py --check`` re-imports both modules in a fresh interpreter
under four environments -- the bare defaults, ``TPU_MAXDIM`` at 8 and 32, and
every memory overridden at once -- and compares all 16 parameters and derived
constants. A default that moves in the design and not in the spec is named
there, in the one configuration no test varies: the defaults.

**Derived properties.** Four facts *follow* from the encoding rather than
being written in it, and three of them were stated wrongly in this project's
own documents until 2026-09-21. They are recomputed from the spec and
confirmed against the design on every run -- see the generated list in
:ref:`tinytpu-isa-spec`.

The load-bearing one: ``isa_encoding.agu_legal_trip`` derives, from
``legal_values`` alone, how long a field with a restricted value set stays
legal when a loop drives it, and the design is then asked where it actually
stops. ``mm``'s accumulate field driven by the reduce loop at stride 1 is
legal for **two** k-tiles and rejected at the third -- because an address term
is **additively monotone** (``base + iv * stride``, no predication, no
saturation, no wrap), not because the field is static, which is what three
documents here said. A second property prices it: the shipped accumulating
``mm`` already spends all three address terms, so a term on ``acc`` needs a
fourth and the budget refuses the instruction **before** the monotonicity
frontier is ever reached. Widening the address generator and relieving ``acc``
are therefore complements, not alternatives.

The fourth derives what bounds ``MAXDIM``, which is the *encoding* and not the
datapath. There are **two** ceilings and they answer different questions, so
neither is a correction of the other:

* the **addressing** ceiling asks what the operand *layout* can address,
  whatever shape runs -- ``(MAXDIM // T) * MAXDIM <= 2048``, from the 11
  usable bits of an address field. (2048, not 2047: the layout numbers its
  rows ``0 .. OPERAND_ROWS-1``, so it is the highest *address* that must fit.)
  At T=4 that is **88**; at T=8, **128**.
* the **cubic-header** ceiling asks what a *cubic* GEMM's header count can
  promise ``accu`` -- ``MAXDIM**3/T**2 + MAXDIM**2/T <= 32767``, from the
  15-bit header slice. At T=4 that is **76**; at T=8, **120**. It moves with
  the workload: a non-cubic shape gives a different count.

The cubic one binds at both array sizes, but which binds is a property of the
program rather than a constant.
``isa_encoding.maxdim_ceiling`` **computes** each, over multiples of ``T``,
because solving either inequality over the reals gives a number no build can
use -- which is how ``MAXDIM <= 90`` came to be written in three documents
when the answer is 88. The design is then probed at each ceiling and one step
of ``T`` past it, and the two are told apart by the **stage** at which it
refuses: past the addressing ceiling the module will not import at all; past
the cubic-header ceiling it imports and ``assemble`` refuses the GEMM
(``80 refuses in assemble``, ``92 refuses at import``). Without that, either
ceiling could take credit for the other's failure.

**Build limits and program limits are not interchangeable**, and conflating
them cost this file three errors in one sitting. A build's ``MAXDIM`` is
bounded by what the layout can *address*; what a *program* may ask for is
bounded separately, by the rows an instruction's ``nr`` carries and by the
header counts its shape produces. ``isa_encoding.gemm_limits(M, K, N)``
computes which limit refuses a shape **and by how much**, because the margin
is what makes a gap actionable: "over by one in ``nr``" names a design target,
"cannot express" does not. A fifth derived property holds that prediction to
the design over every GEMM shape the build admits.

The distinction is load-bearing. At ``T=8``, ``MAXDIM=128``, the shipped
tiled-GEMM layout is over by **one** on ``max(M, K)`` for ``32x128x128`` and
``64x128x128`` -- it issues one ``dma_ld`` of ``K`` rows per B column block, so
``K`` travels in ``nr``. That is a property of *that layout*, not of the
instruction set: splitting the B load into two ``dma_ld``\ s of 64 rows removes
``K`` from the row count, and both shapes then assemble and compute ``A @ B``
bit-exactly on the design. ``128x128x128`` is over by one even then, and its
``accu`` header count is over by 2049, so it needs both a wider ``nr`` and a
wider header slice.

**The checks have teeth**, and that was measured rather than hoped: fifteen
single-point edits -- an opcode number, a field position, a field width, an
AGU subfield position, a header term's scale, the loop depth,
``AR_RAW_DIST``, the accumulator width, a clip bound, a parameter range, the
AGU budget, ``mm``'s legal ``acc`` values, each of the two ``MAXDIM`` ceiling
predicates, and a changed default in the design -- each produced a named
failure identifying the disagreeing constant, slice,
program or property. The last of those was not a simulation: when ``main``
rebuilt the memories to be derived from ``MAXDIM`` and moved its default to
64, ``gen_isa.py --check`` named all five parameters that had moved, in every
probed configuration, instead of passing.

**What this does not prove.** It is a conformance check, not a proof of
correctness. It says the design and the reference model encode, decode,
resolve and count the same way the spec says; it says nothing about whether a
*unit* does the right thing with a field it decoded correctly -- that is
``stress_isa.py``, ``mutate.py`` and cosim. It checks the emitted HLS's bit
ranges, not its behaviour, and it does not read the Verilog. And the numerics
section is checked only where it is mechanically checkable: the lane widths and
the ``mvout`` saturation bounds. The reference model's *arithmetic* is held to
the spec by construction -- it calls the generated ``acc()`` and
``to_operand()`` -- not by an independent check.

What the split does buy immediately: a mutant of ``microarch_isa.py`` can no
longer move the reference model with it. Before, an opcode renumbered in the
design was renumbered in its own oracle.

Verifying a change
^^^^^^^^^^^^^^^^^^

From a clean checkout, one command builds the checkout's own bindings, runs the
functional gates, runs cosim, and checks the published cycle counts
(175 / 265 / 421 / 482 / 674, at the ``TPU_MAXDIM=16`` it pins), exiting
nonzero if any step fails or any number differs:

.. code-block:: bash

   examples/tinytpu/reproduce.sh              # everything, with cosim
   examples/tinytpu/reproduce.sh --no-cosim   # functional only, ~1 min

It unsets every ``TPU_*`` knob, pins ``TPU_MAXDIM=16``, and checks that
``allo`` resolves to the checkout it is run from (``92fb2f1b``). Added on
``main`` on 2026-09-19; the pristine-worktree runs it has printed, including
the pre-landing design's, are in :doc:`tinytpu_isa_results`.

**The gate is what catches a moved number.** It carries the published row as
an expectation and refuses to pass on a change of one cycle, which is how the
2026-09-22 shift was found minutes after the merge that caused it, on a
refactor whose functional gates were all green.

**Performance vs. correctness.** ``bench_isa.py`` and ``cosim.py`` with its
default testbench are the **performance** setup: seed 0, operands in [-4, 4]
(Gemmini's ``allo_cmp.c`` distribution, kept so the comparison is like for
like), ``C`` zeroed, and only the ``M x N`` region compared. They miss real
bugs -- at T=4 a PE's partial sum never leaves 9 bits, so narrowing the int32
partial sum to int16 still prints ``ALL EXACT``. The correctness gates are
below. Run ``stress_isa.py`` after **any** change to the design, and
``mutate.py`` after any change to the harness.

``examples/tinytpu/e2e_gate.sh``
   About 36 s, no Vitis and no licence. The gate for the claim no single-flow
   check can make: **a PyTorch model through mapping to cycles, and the join
   to area.** It runs the workload suite's claims gate
   (:doc:`workload_suite`), then ``check_pairing.py``, which refuses to let a
   cycle count stand beside an area figure from a different configuration,
   then ``check_numbers.py``. It prints the remote tier -- the Design Compiler
   run -- rather than claiming it, because that needs a licence this
   repository does not have, and it names what it therefore cannot cover. The
   failure paths of both gates are tested in
   ``tests/act/test_gates_negative.py``.

``gen_isa.py --check``
   About 6 s. The ISA conformance check described in
   :ref:`tinytpu-isa-conformance`: both generated artefacts regenerated and
   diffed byte for byte, every consumer of ``isa_spec.json`` held to it, the
   design and the generated module re-imported under four configurations and
   compared, and the derived properties recomputed and confirmed.
   ``--conform`` adds the emitted HLS's own bit ranges (one HLS build).
   It prints ``ISA OK`` or names each disagreement.

``stress_isa.py`` (``ef112868``)
   About 10 s on the Allo simulator, 492 runs: full-range int8 operands with
   -128 and 127 forced; clip and ReLU edge cases landing on 127 / 128 / -128 /
   -129 / 0 / -1; all 64 shapes; ``C`` prefilled with random bytes and compared
   in full; ``isa_dsl.vector_program``, which varies every field GEMM holds
   constant; ``isa_dsl.ar_distance_program`` at the accumulator's distance
   contract; and 200 random valid programs, generated to keep that contract.
   Everything is checked against ``isa_ref.py``, a numpy reference model of the
   ISA that is itself checked against numpy on every GEMM. It also runs the
   validator's negative controls (18 crafted bad programs rejected, 390
   generated ones accepted) and runs ``kpn_model.py`` on every distinct program
   first, so a simulator hang becomes a named report. It prints ``STRESS OK:
   n/n`` or lists each failing run.

``TPU_TB=stress python cosim.py`` (``1ac0d22f``)
   The same idea on the RTL: six calls per shape in one RTL simulation (corner,
   full-range, boundary and mid operands, ``vector_program``, and
   ``ar_distance_program(AR_RAW_DIST)`` -- the one case only RTL can fail),
   each inheriting the previous call's state, with ``C`` prefilled and compared
   in full. Its cycle column is the minimum over those calls, not the headline. The
   default testbench is unchanged byte for byte and is the only mode the
   published cycle counts come from.

**Proof that the split matters.** The int16 partial-sum mutant:

.. list-table::
   :header-rows: 1

   * - check
     - int16 partial-sum mutant
   * - default cosim (performance testbench)
     - **passes**: 0/16 and 0/256 mismatches, at the then-published 252 / 919
       cycles (measured on the pre-landing design)
   * - ``TPU_TB=stress`` cosim
     - **fails**: 3 wrong at 4x4x4, 82 at 16x16x16
   * - ``stress_isa.py``
     - **fails**: 247 of 486 runs (pre-landing); 241 of 492 on the landed design

``mutate.py`` (``c8089332``; re-anchored and extended in ``e24e433b``)
   34 single-point mutants of the design (plus the unmodified
   ``none`` control through the same loader), each run through ``bench_isa``
   and ``stress_isa``, and through the ``TPU_TB=stress`` cosim for the one
   RTL-only mutant (and for any mutant on request). **All 34 are caught**
   (``dev/records/tinytpu/logs/mutate_landed.log``). 20 fail ``bench_isa``. 13 get past it and are
   caught **only** by ``stress_isa``: ``pe_psum_int16`` (int16 partial sum),
   ``clip_hi_off_by_one`` and ``clip_lo_off_by_one`` (both clip bounds),
   ``vadd_dst_is_src1`` and ``vrelu_dst_is_src`` (vadd/vrelu destination),
   ``vadd_src2_is_src1`` (vadd second source), ``vrelu_src_base_ignored``,
   ``mvout_src_base_ignored``, ``dma_ld_row_ignored`` and
   ``mvout_row_ignored`` (base / row fields), ``spm_vld_off_by_one`` (``vld``,
   which the shipped GEMM no longer issues), ``dma_st_accumulates_C``
   (``dma_st`` relying on a zeroed ``C``), and ``ar_contract_unenforced``
   (the assembler no longer enforcing the accumulator distance contract). One,
   ``ar_claim_false``, passes both functional levels by construction and is
   caught **only** by cosim; see :ref:`tinytpu-isa-dependence`.

   The mutants that target the landed code: ``pe_shadow_not_swapped`` and
   ``wld_rows_from_weight`` (the weight double buffer), ``spm_weight_off_by_one``
   (weights from the scratchpad), ``vru_dma_ignores_f3`` and ``vru_act_off_by_one``
   (the A path into and out of the vregs), ``prefetch_lane7_dup`` (the 8-wide
   prefetch), ``vadd_holds_stale_x`` (``vadd``'s two-iteration rows), and the
   two dependence mutants. The pre-landing mutants were re-anchored on the
   equivalent code; ``vrelu_loop_short`` became ``vrelu_rows_short`` (there is
   no per-instruction loop left to shorten). Each mutant's anchor must match
   exactly once, so a refactor of the design makes the script fail loudly
   instead of silently testing nothing.

**The arrays are not cleared by hardware.** ``spad``, ``vr`` and ``ar`` carry no
``= 0`` initialiser (it cost a runtime zero-fill loop; see
:ref:`limitation-g`), so ``assemble()`` enforces a write-before-read contract
(``4357ed59``): ``check_program()`` walks the exact dynamic trace -- the program
has no data-dependent control flow, so this is exact, not conservative -- and
rejects any read of an ``ar`` row (``mm`` accumulate, ``vadd``, ``vrelu``,
``mvout``), a ``vr`` row (``mm`` activations) or a ``spad`` row (``mm``
weights) that no earlier instruction wrote. Written-ness propagates through
``vld``: a ``vld`` may copy an unwritten ``spad`` row, but the copy then counts
as unwritten. The same walk enforces the accumulator distance contract
(:ref:`tinytpu-isa-dependence`). It also rejects out-of-range rows, bad loop nesting
(unbalanced or over-deep loops, trip 0, AGU terms naming a closed loop),
zero-row instructions, and the retired ``dma_st`` opcode.

Two of ``check_program()``'s checks exist only because the walk is exact
rather than conservative: it rejects a resolved field that has climbed past
``2**11`` (the range ``enc()`` admits) -- the sequencer writes the resolved
sum back into the 12-bit field, so an AGU term can push it out of range, and
this walk is the only place that can see it -- and it enforces the structural
shape an instruction must have (``mm``'s ``f2`` in ``{0, 1}``, ``dma_ld``'s
``f0`` in ``{0, 1, 2, 3}``) rather than trusting the encoder. Every rejection
names the static instruction, the loop iteration, and the rows.

Two design facts the hardening found:

* **Zero-row instructions hang the machine** instead of being no-ops: an
  ``nr=0`` data op desynchronises the flat row loops (confirmed on the
  simulator). The validator now rejects them.
* **The hung simulator ignores SIGTERM**; only SIGKILL stops it (it is inside
  a blocking C call; compare the watchdog note in :ref:`limitation-11`).


Limits and known failures
-------------------------

The register of Allo limitations this design runs into, with repros, is
:doc:`/developer/limitations`; the pitfalls of ``@df.region()`` are in
:doc:`/developer/pitfalls`. Two limits belong to the design itself rather than
to the toolchain:

- **The accumulator RAW-distance contract**, below -- the one claim on this
  page that no simulator can check.
- **The csynth latency number is not a cycle count** for a programmable
  design; only cosim measures this machine. See
  :ref:`tinytpu-isa-csynth-bound` on :doc:`tinytpu_isa_results`.

.. _tinytpu-isa-dependence:

The accumulator's dependence claim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``accu`` is one flat row loop at II=1 because ``schedule()`` tells Vitis that
its accumulator carries no dependence across iterations:

.. code-block:: python

   s.dependence(
       "accu_0:x", "ar", dep_type="inter", dependent=False,
       because=f"check_program() rejects any program that reads an ar row "
               f"within AR_RAW_DIST={AR_RAW_DIST} accu iterations of writing "
               f"it (THE ACCUMULATOR DISTANCE CONTRACT); assemble() enforces "
               f"it, the hardware does not, and only TPU_TB=stress cosim can "
               f"see a breach",
   )
   # -> // dependence obligation, checked by no tool: check_program() rejects ...
   #    #pragma HLS dependence variable=ar inter false   (inside accu's row loop)

``s.dependence`` is the schedule primitive added for :ref:`limitation-21`
(``bbea2af0``). Without the claim the flat loop closes at ``Final II = 3``:
the row index is a carried register, so Vitis cannot prove that iteration n's
store and iteration n+1's load of ``ar`` touch different rows.

That same carried register is why this claim is an **obligation** and not
something Allo's legality rule can settle. The rule (``allo/dependence.py``,
added with the 2026-09-22 update to :ref:`limitation-21`) refuses a claim only
when it can *prove* a dependence at a distance the claim denies; ``ar[ra]``
with ``ra`` computed per iteration is not affine in the loop's induction
variable, so nothing is provable and the claim stands -- which is correct, and
is the reason the primitive exists. ``because=`` is where the contract below is
recorded; it is printed above the pragma in ``kernel.cpp`` and listed in
``s.dependence_obligations``.

**The claim is not true of the hardware on its own**, and the branch that
priced it (``v_accudep`` / ``v_design_dep``, which injected the pragma into
``kernel.cpp``) never tested where it fails: only GEMM programs were cosimulated
there. In the synthesized loop (II=1, depth 6) the ``ar`` load issues in
pipeline state 5 and the store lands in state 7, so a row read one or two
iterations after it was written returns its **old** value. Measured in RTL
with ``isa_dsl.ar_distance_program(d)``, every read exactly ``d`` iterations
after its write (``dev/records/tinytpu/logs/cosim_isa_ar_distance.log``,
``impact/ar_distance_probe.py``):

.. list-table::
   :header-rows: 1

   * - distance ``d`` (accu iterations)
     - 1
     - 2
     - 3
     - 4
     - 5
   * - cells of ``C`` wrong, RTL
     - **4**
     - **20**
     - 0
     - 0
     - 0
   * - cells wrong, C simulation
     - 0
     - 0
     - 0
     - 0
     - 0

So the claim is made true by the **assembler**, the way the write-before-read
contract is: ``check_program()`` counts ``accu`` iterations (one per ``mm``,
``vrelu`` or ``mvout`` row, two per ``vadd`` row) and rejects any read of an
``ar`` row fewer than ``AR_RAW_DIST = 4`` iterations after the write it
depends on -- the first safe distance plus one of margin, and exactly ``T``,
the row count of the smallest GEMM, so no GEMM is constrained by it. The
random-program generator keeps the contract, and the ``TPU_TB=stress`` cosim
runs ``ar_distance_program(AR_RAW_DIST)`` on every build, so a re-synthesis
that widened the window fails there.

That test has teeth. ``mutate.py``'s ``ar_claim_false`` sets ``AR_RAW_DIST =
1``: the pragma is then false for programs the assembler accepts. It passes
``bench_isa`` and ``stress_isa`` -- no simulator models a dependence pragma --
and the stress cosim catches it (``ar_distance(1)``: 4 cells wrong).
``ar_contract_unenforced`` (the check disabled) is caught by ``stress_isa``'s
validator controls.

The alternative claim, ``inter RAW distance=4 true``, would have Vitis honour a
distance of 4 itself rather than rely on the schedule; it was not measured.



Results and history
-------------------

Cycle counts, resources, the standard-cell synthesis figure, and every
superseded figure and withdrawn claim are on :doc:`tinytpu_isa_results`. The
design's longer optimisation history is on :doc:`tinytpu_history`.

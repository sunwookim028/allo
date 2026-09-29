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
Catapult SystemC Flow and the Wire
####################################

This page records the fork's work toward an ASIC-oriented HLS path through Siemens Catapult beyond
the in-tree C++ backend (:doc:`/backends/catapult`): the SystemC emitter on the
``choonsik1/allo`` fork, why Catapult is being pursued again, and the RTL-simulation apparatus that
established that the SystemC ``Wire`` is wrong in RTL. The authoritative statement of that last
result is :ref:`limitation-22`; this page is how it was reached and how to re-run it.

Why this flow was revisited at all is recorded in `Earlier measurements and corrections`_.

The SystemC Emitter
-------------------
The SystemC path is a different fork, ``choonsik1/allo``, fetched in this clone as the
``choonsik1`` remote (``https://github.com/choonsik1/allo.git``). Unlike the C++ emitter, whose
only stream type is ``ac_channel``, it emits SystemC over MatchLib Connections with three edge
types:

.. list-table::
   :header-rows: 1

   * - Type
     - Edge
   * - ``Stream[T, depth]``
     - a FIFO
   * - ``Channel``
     - a valid/ready handshake
   * - ``Wire``
     - a non-handshaked combinational edge

``Wire`` is why this path matters: a MiniTPU-class VLIW delay line (see :doc:`/designs/minitpu`)
needs a non-handshaked fixed-latency edge, and Allo's ``Stream`` cannot express one.

The fork is to be integrated rather than only read. ``SystemC-emitter`` is its most complete
branch. ``systemc-ip-integration`` lacks the RAM-pin memory interface (``b92077f5``) and the
free-running-loop fix (``72c70dcb``), so an integration must start from ``SystemC-emitter``. The
``pe_wire`` / ``pe_stream`` / ``pe_channel`` netlists used by ``tests/systemc/rtlsim/`` survive
only in its history, at ``0eff4888:agents/noc/rtl/<design>/rtl.v`` (``REPRO.sh`` names the same
files as ``779e4350^:agents/noc/rtl/{pe_wire,pe_stream,pe_channel}/rtl.v``). Branch layout for all
remotes is in ``dev/fork_maintenance.rst``.

``AlloMemPins``: a dual-port RAM, instantiated once per client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **Corrected 2026-09-19.** This section previously said the hardware for a
   shared memory exists and only Allo's refusal to hand out both ports stood in
   the way. The module is a real 1R1W dual-port RAM, but the emitter does not
   share it: memory instances are keyed per (call, operand) at fork
   ``EmitSystemC.cpp:2539-2545``, so **each client gets a replica**, not a port
   of one memory. The fix is ~50-100 lines binding one writer and one reader to
   one instance's two pin sets. Separately, the one-owner rule on the Vitis path
   is Allo's, not Vitis's: Vitis shares an on-chip array between two processes
   under ``#pragma HLS stream type=unsync`` (:ref:`limitation-shared-memory`).

``AlloMemPins`` **is** an unarbitrated 1R1W dual-port RAM, and it synthesizes. It is **not** in
this checkout's working tree; it lives in ``mlir/lib/Translation/EmitSystemC.cpp`` on
``choonsik1/allo:SystemC-emitter``. Read there (verified 2026-09-18), the module has separate read and
write pin bundles (``radr``/``re``/``q`` and ``wadr``/``d``/``we``) over one ``T mem[SIZE]``, with
both accesses serviced in a single ``wait()``-delimited cycle and both ready lines tied high; its
own comment says "single-cycle, unarbitrated, no bank conflicts". It is instantiated on both sides
of the boundary, the in-design case being described there as "a replicated, multi-client array",
and it is emitted for synthesis (it carries a CIN-233 reset-write workaround for exactly that
case). The valid/ready finding in :ref:`limitation-22` names the older ``AlloMem``, which is a
different module.

Two caveats: this is a **SystemC-path** artifact, so what it demonstrates is that the *hardware* is
unobjectionable, not that the C++ path emits it; and "Allo refuses to hand out both ports" is a
frontend/IR restriction common to both paths (see the classification below). The multi-client
claim is read from source and comments, not from a run -- no SystemC library or MatchLib exists on
``ace-01``, so nothing here was csim'd or synthesized.

What the C++ Path Can and Cannot Express
----------------------------------------
**Scope: this section is about the C++ emitter path** (``EmitCatapultHLS.cpp`` -> ``ac_channel``),
**not** the SystemC path. The C++ emitter is a 14-of-57 override syntax layer over the Vivado
emitter (counted in :doc:`/backends/catapult`), so it can express nothing Vitis cannot. Anything
flagged as a limitation on this path is one of:

- **(A)** genuinely Allo's own -- the frontend/IR cannot express it;
- **(B)** *apparently* Vitis's, but (A) underneath -- Vitis has the construct, Allo has no way to
  reach it;
- **(C)** genuinely the tool's.

**On the C++ path, (B) is nearly always really (A)**, because switching Vitis -> Catapult changes
spelling, not expressiveness. A (B) diagnosis there should be treated as a claim to check, not a
reason to change backend. Six flagged items (2026-09-18):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Item
     - Class
   * - shared multi-ported memory (two ports of one array to two kernels)
     - **(A)**, and the decision-relevant one: Allo will not hand out both ports, so no backend
       switch resolves it. *(Corrected 2026-09-19: Vitis itself accepts two processes on one array
       under* ``stream type=unsync``; *on the SystemC fork each client gets an* ``AlloMemPins``
       *replica. See* :ref:`limitation-shared-memory`. *Measured impact on TinyTPU-isa: 0
       cycles.)*
   * - :ref:`limitation-21`: no ``#pragma HLS dependence``, so a false dependence cannot be
       asserted away
     - **(A)** -- Vitis has the pragma; Allo emits only ``m_axi``/``s_axilite``/``bind_storage``/
       ``array_partition`` (``s_axilite`` was later found not to be emitted on the Vitis path at
       all; see :ref:`limitation-23`)
   * - :ref:`limitation-13`: "no program-controlled DMA"
     - **(B) -> neither** -- retracted: a contiguous runtime-length copy does infer a
       variable-length AXI burst. The cost had been charged to the configuration when the access
       pattern was the variable that differed
   * - :ref:`limitation-18`: non-blocking ``try_get``/``try_put``
     - **(C) + (A)** -- the LOOP-19 ``go compile`` segfault on ``nb_read`` is Catapult's; emitting
       ``success = true`` silently instead of refusing is ours
   * - native ``float`` unsynthesizable (CIN-291)
     - **(C)** -- ``nangate-45nm_beh`` genuinely lacks it, and the ``ac_ieee_float`` override is
       the entire fix
   * - :ref:`limitation-22`: a non-handshaked fixed-latency edge
     - **(C)** *on this path* -- neither ``hls::stream`` nor ``ac_channel`` has one. That is
       precisely why the SystemC path exists, and why the spike is aimed there

The TAPA emitter's non-blocking overrides, added and then removed, are recorded in
`Earlier measurements and corrections`_.

The Wire Investigation: Simulating Catapult's Netlists
------------------------------------------------------
``tests/systemc/rtlsim/`` is the apparatus behind :ref:`limitation-22`. It took
reverse-engineering the netlists' internal signal names (e.g. ``tb.u_mul.mul_0_run_inst.v8_and_cse``)
to build, so it is kept rather than rebuilt.

.. note::

   **Root cause located, 2026-09-19:** ``emitWireGet`` / ``emitWirePut`` at fork
   ``EmitSystemC.cpp:1828-1848`` are bare ``sc_signal`` accesses, with nothing
   tying the reader's loop to the writer's. The fix is the ``SC_METHOD`` comb
   emission mode scoped in ``c7402f9f``. See :ref:`limitation-22`.

An earlier investigation reported ``pe_wire`` as "synthesises clean and fails csim" and concluded
the SystemC thread model could not represent a wire -- that the design was good and the simulator
was lying. **That is backwards**: Catapult's own netlist fails in RTL simulation, so the csim
failure was a true positive.

What it tests
~~~~~~~~~~~~~
Three emitted designs for the same computation -- an element-wise multiply feeding an accumulator,
``C[i] = sum_{j<=i} A[j]*B[j]`` with ``A=1..8``, ``B=2..16`` step 2 -- differing only in the edge
between the two:

.. list-table::
   :header-rows: 1

   * - Design
     - Edge
   * - ``pe_stream``
     - ``Stream[int32, 2]``, a depth-2 FIFO
   * - ``pe_channel``
     - ``Channel``, valid/ready handshake
   * - ``pe_wire``
     - ``Wire``, a non-handshaked combinational edge

Each is driven at 18 producer/consumer pacings (``STALL_IN`` 1..6 x ``STALL_OUT`` 1..3) plus a
baseline, and checked against golden ``2 10 28 60 110 182 280 408`` (numpy ``cumsum``, hardcoded in
the testbench).

The result (``results.txt``, 75 runs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Design
     - PASS
     - FAIL
   * - ``pe_stream``
     - 19
     - 1 -- the injected fault, as intended
   * - ``pe_channel``
     - 19
     - 1 -- the injected fault, as intended
   * - **pe_wire**
     - **2**
     - **27**

``pe_wire`` fails **8/8 elements wrong at every one of the 18 pacings**, producing
``0 0 0 2 2 10 10 28``. The two passes are not pacings; they are the positive control below.

Why it fails, and the positive control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From the netlist: ``acc_0`` has no input handshake and advances on its *consumer's* ready, while
``mul_0`` latches on its *producers'* valid. Nothing couples the two counters, and ``mul`` is ~3x
slower per element, so ``acc`` runs the whole loop before ``mul`` produces anything.

``LOCKSTEP`` + ``ACC_RST_DELAY`` holds ``acc_0`` in reset longer and steps it once per product. On
the **identical** wire RTL:

.. code-block:: text

   ACC_RST_DELAY = 0  1  2  3  4  5  6
                   F  F  F  P  P  F  F

So the wiring and the arithmetic are correct and only the lockstep is missing -- and **the correct
window is 2 cycles wide**, which is the number that matters for anyone trying to schedule against a
``Wire``.

The faults, so the test is known to be able to fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Four injections, all red: ``BREAK_DATA`` on each of the three designs, and ``BREAK_WIRE`` (one
cycle of latency in the wire) on top of the *working* ``ACC_RST_DELAY=3`` configuration. A test
that only ever passes proves nothing; these are why the passes above count.

Running it
~~~~~~~~~~
Needs Vivado's xsim on ``PATH`` (``/opt/xilinx/Vivado/2023.2/settings64.sh``) and the netlists,
which are **not** in this repository -- they are Catapult output from
``choonsik1/allo`` (branch ``SystemC-emitter``; there is no
``choonsik1/SystemC-emitter`` repository) at
``0eff4888:agents/noc/rtl/<design>/rtl.v``, and ``REPRO.sh`` documents where they came from. The scripts
look for them under ``../noc/rtl/<design>/rtl.v`` relative to ``tests/systemc/rtlsim/`` (the
Xcelium runner takes ``RTLDIR`` to override). ``mgc_shim.v`` supplies the Mentor primitives the
netlists instantiate. No SystemC is needed; each case takes about 6 s.

.. code-block:: bash

   cd tests/systemc/rtlsim
   ./run.sh <design>     # one design, tb_top.v testbench
   ./run_mulacc.sh <pe_wire|pe_stream|pe_channel> [-d MACRO[=VAL] ...]
   ./REPRO.sh            # the whole matrix: boundaries, pacing sweep, lockstep control, faults

Both runners print one ``RESULT: PASS|FAIL <tag>`` line per run;
``dev/records/systemc/rtlsim/results.txt`` is that output, sorted. The 76 xsim work
directories are not kept -- they are ~29 MB and regenerable, and the ``RESULT:`` lines
are the evidence.

A second simulator: Xcelium
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``run_mulacc_xrun.sh`` is the Xcelium counterpart of ``run_mulacc.sh`` with the same arguments;
``REPRO.sh`` uses it with ``RUNNER=run_mulacc_xrun.sh``. It unsets ``LD_PRELOAD``, defaults
``CDS_LIC_FILE`` to ``5280@en-license-05.coecis.cornell.edu``, finds ``xrun`` at
``/opt/cadence/XCELIUM2403/tools.lnx86/bin/xrun`` (override with ``XRUN``), and resolves Catapult
library cells from ``$MGC_HOME/pkgs/siflibs``. Xcelium is installed on zhang-21, **not** on
``ace-01`` (see ``dev/toolchains.rst``).

On zhang-21 with Xcelium 24.03 (``710ab138``, 2026-09-18), **all 75 cases in**
``dev/records/systemc/rtlsim/results.txt`` **reproduce**: ``pe_stream`` and ``pe_channel``
pass all 18 pacings each, ``pe_wire`` fails 8/8 at
all 18 in 12 cycles (``0 0 0 2 2 10 10 28``), the lockstep control over ``ACC_RST_DELAY`` 0..6
gives ``F F F P P F F``, and all four fault injections go red. The full ``REPRO.sh`` takes 23 s.

Reference xsim runs
~~~~~~~~~~~~~~~~~~~
``dev/records/systemc/rtlsim/ref_xsim/`` keeps two runs of the **same** ``pe_wire`` **netlist** so
that a run on a different simulator can be compared against a known-good and a known-bad result
rather than against a claim. Trimmed to the logs (``xsim.dir``, ~350 KB per run, is dropped) with
absolute scratch paths rewritten to ``<RUNDIR>``. Produced by ``run_mulacc.sh`` under xsim v2023.2
on ace-01, 2026-09-18.

.. list-table::
   :header-rows: 1

   * -
     - ``fail_default/``
     - ``pass_ACC_RST_DELAY3/``
   * - invocation
     - ``pe_wire``
     - ``pe_wire -d LOCKSTEP -d ACC_RST_DELAY=3``
   * - result
     - FAIL, 8/8 wrong
     - PASS
   * - **cycles**
     - **12**
     - **22**

What to compare, in priority order:

1. **The output pattern.** ``0 0 0 2 2 10 10 28`` is not noise -- it is the golden sequence shifted
   and repeated, which is what a consumer sampling an unhandshaked edge too early and too often
   produces. If another simulator gives this exact pattern, it is reproducing the same defect and
   not merely failing.
2. **The cycle count, which is the sharper tell.** The broken run finishes in **12 cycles and the
   correct one takes 22** -- *the failure is faster*. Nothing throttles ``acc``, so it runs its
   whole loop and finishes early. A simulator that reproduces the wrong values but takes ~22 cycles
   is failing for some other reason and the diagnosis does not transfer.
3. **The positive control.** ``ACC_RST_DELAY=3`` passing on identical RTL is what proves the wiring
   and the arithmetic are correct and only the lockstep is missing. If that does not pass on the
   other simulator, suspect the port of the harness before suspecting the design -- and check the
   fault injections, which must still go red.

Both runs reach ``$finish`` at ``tb_mulacc.v`` line 174 and capture 8/8 elements, so neither is a
hang or a truncated run; the failing one is a complete run with wrong data.

Is it the free-running-loop rewrite? No
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**The hypothesis (2026-09-18,** ``64b0ca86`` **).** ``choonsik1/allo:SystemC-emitter`` carries
``72c70dcb`` (2026-08-20), "SystemC: do not make a port-reading driver loop free-running".
``isSteadyStateLoop`` treated an unused induction variable as proof a kernel runs forever and
rewrote its outermost loop to ``while (1)`` **under** ``__SYNTHESIS__`` -- precisely a bug that
leaves a design correct in csim and wrong in RTL. The netlists above are from ``0eff4888``
(2026-08-01), 19 days older. But ``72c70dcb``'s guard requires *a load from a memory port inside
the loop body*, and ``acc`` is declared ``args=[]`` (``tests/systemc/dot_product_four_links.py``), so
the guard structurally cannot fire for it, while its unused ``i`` does trigger the rewrite. ``mul``
is rewritten too, but it reads Streams, whose handshake makes a free-running loop harmless. The
proposed fix was to extend the guard so that a ``get()`` from a ``Wire`` or ``Channel`` in the loop
body counts as evidence of finite streaming. This was stated as a hypothesis from reading source,
not a measurement.

**The test (2026-09-18,** ``6b84f7b1`` **): a negative result.** ``guard_experiment/guard.patch``
stops the rewrite for any loop that reads a ``Wire``. ``dot_product_four_links.py`` was re-emitted from
``choonsik1/allo:SystemC-emitter`` with and without it, synthesised with Catapult 2024.2, and simulated
under Xcelium 24.03:

.. list-table::
   :header-rows: 1

   * - Netlist
     - ``pe_wire``, 18 pacings
     - Cycles at unit pacing
   * - ``rtl_base`` (emitter as is)
     - FAIL 8/8 at all 18
     - 20
   * - ``rtl_guard`` (with ``guard.patch``)
     - FAIL 8/8 at all 18
     - 20 (identical at every pacing)

- **Emitted code:** exactly as intended. ``acc_0`` loses its free-running loop (and its
  done-on-entry), ``mul_0`` keeps its own, and ``pe_stream``/``pe_channel`` come out byte-identical
  with and without the patch.
- **Controls:** ``rtl_guard/pe_stream`` and ``rtl_guard/pe_channel`` pass all 36 pacings, and
  ``BREAK_DATA`` turns both red.
- **Values:** ``0 8 40 112 240 368 496 624`` -- ``acc`` samples every other product and then holds
  the last one.
- **Why the guard does not help:** removing ``while (1)`` bounds ``acc``'s trip count but gives it
  nothing to wait on. ``acc_0`` still advances on its consumer's handshake and never on its
  producer. The Wire boundary has no synchronisation, with or without ``while (1)``.

So the conclusion of :ref:`limitation-22` stands: the missing piece is synchronisation. Making wires
sound needs the emitter to give cycle-locked kernels a shared advance -- one enable driving every
stage's counter, which is what a VLIW delay line is -- and the SystemC path does not currently
supply a usable non-handshaked edge.

Replaying from the committed netlists (no Allo build needed):

.. code-block:: bash

   cd tests/systemc/rtlsim
   RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_wire
   RTLDIR=$PWD/guard_experiment/rtl_base  ./run_mulacc_xrun.sh pe_wire
   RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_stream -d CONNECTIONS_FIFO
   RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_channel

The lockstep positive control (``-d LOCKSTEP``) is for the August netlists only, because it taps an
internal signal (``v8_and_cse``) that these netlists do not have. So ``REPRO.sh`` section 3 and the
``BREAK_WIRE`` fault cannot be replayed this way: ``RTLDIR`` does pass through ``REPRO.sh``, but
those rows die in elaboration and print ``NO VERDICT``. The shipped netlists cover the three
boundaries and ``BREAK_DATA``, and nothing else.

Regenerating the netlists:

1. Build ``choonsik1/allo:SystemC-emitter`` (``72c70dcb`` or later, LLVM ``6b09f739``).
2. Run ``ALLO_ROOT=<that worktree> python run_sc.py emit.py out_base``. ``run_sc.py`` makes Python
   import that worktree's ``allo`` even when a conda env has an editable install of another one.
3. Apply ``guard.patch``, rebuild, and emit to ``out_guard``.
4. In each project directory, run
   ``MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu catapult -shell -file run.tcl``. The
   netlist lands in ``Catapult/<design>.v1/rtl.v``.

The exploration branch that asked this question, ``sc-wire-guard``, is deleted; its answer is
recorded here and in ``tests/systemc/rtlsim/guard_experiment/``.

Earlier measurements and corrections
------------------------------------
Kept for the record: how the decision to revisit Catapult was reached, and an emitter feature that
was added and then removed.

Why Catapult Again: Decision History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vitis HLS is the primary backend and has the only end-to-end measured results
(:doc:`/designs/tinytpu_isa`).

**2026-09-17: "not pursuing Catapult further, CIRCT is the recommended long-term direction."** The
reasoning then was: market niche (automotive/defense, Siemens-adjacent shops); not a standard
research-community reference tool; Cadence Stratus is the stronger competitor for ASIC research
citations; CIRCT (MLIR-native, Google/Intel-backed) has the better long-term trajectory. The
synthesis results it rested on are in ``dev/records/catapult_decoupled_2x1.rst`` (``top_decoupled_2x1``,
1 MT + 2 CTs: CT latency 295 cycles each, MT 67, 657 sequential, throughput 298 cycles, area scores
CT0=14991, CT1=14991, MT=16180 at Catapult 2024.2, ``nangate-45nm_beh``, 500 MHz).

**Reopened 2026-09-18. Both halves of that decision moved:**

- **Catapult is being pursued again**, as a bounded spike on ``zhang-21`` (the host that has the
  tool). The reason is not the synthesis numbers; it is that the SystemC fork carries a ``Wire``,
  and a MiniTPU-class VLIW delay line needs one. That ``Wire`` is currently **wrong in RTL, not
  merely in csim** (:ref:`limitation-22`): simulating Catapult's own ``pe_wire`` netlist under xsim
  fails 8/8 at all 18 producer/consumer pacings, while ``Stream`` and ``Channel`` pass all 18. The
  failure is diagnosed rather than mysterious -- holding ``acc_0`` in reset 3-4 cycles longer and
  stepping it once per product makes the *identical* RTL produce exact golden output, so only the
  lockstep is missing and the correct window is 2 cycles wide. The spike's first deliverable is
  that test passing under Catapult's own scheduler, not a TPU. A second, independent reason to want
  the tool: it gives **ASIC PPA**, and the Gemmini comparison (:doc:`/designs/gemmini_comparison`)
  is cycles-only today. No SystemC library or MatchLib exists on ``ace-01``, so csim cannot run
  there for any design, which is itself part of why the move is worth making.
- **The CIRCT path is not currently reproducible in this checkout.** Its clone
  (``externals/circt``, 2.3 GB with its build tree) was deleted on 2026-09-18 in the pre-migration
  cleanup. It was untracked, not a submodule, and referenced by nothing. The pin survives only
  because the generated RTL stamps it: **CIRCT** ``af5369d``. The generator lives on
  ``chia-codesign`` (``examples/accelerator/tinytpu/microarch.py``), not on ``main``, and the
  artifacts worth keeping -- the per-unit modules, ``gen_ip.tcl``, and ``manifest.json`` -- are
  committed there under ``examples/accelerator/tinytpu/rtlgen/``. ``manifest.json`` is a
  per-module scheduling model (determinacy class, latency, per-port bank/factor/latency/width) and
  so is directly relevant to :ref:`limitation-22`'s conclusion that the SystemC path lacks one. It is
  a partial answer: the four ``counted_static`` units carry latencies, while the top and both DMA
  units are ``indeterminate`` with none -- the data-dependent units a delay line actually has to
  schedule against. (See also ``dev/fork_maintenance.rst`` on whether ``main`` ever tracked
  ``externals/circt``.)

TAPA Non-Blocking Streams: Added, Then Removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The fork once added ``emitStreamTryGet``, ``emitStreamTryPut``, ``emitStreamEmpty`` and
``emitStreamFull`` overrides to ``EmitTapaHLS.cpp``, mapping to TAPA's ``.try_read()`` /
``.try_write()``, with one test (``test_nb_ops_tapa_codegen``). Both were removed: TAPA is not used
in the mesh research flow, non-blocking stream semantics are validated through Vitis HLS (the
primary target), and the dead codepath was a maintenance burden.

Status (verified 2026-09-17, corrected 2026-09-18): ``EmitTapaHLS.cpp`` has no ``try_write`` /
``try_read`` emission and its visitor dispatches only construct/get/put. The base-class hooks in
``EmitBaseHLS.h`` are empty but never reached: the op falls through to ``visitUnhandledOp`` and
``emitBlock`` reports "can't be correctly emitted", so ``build(target="tapa")`` raises
``RuntimeError`` rather than silently producing nothing. (The earlier "not diagnosed -- it simply
produces nothing" reading was wrong.) The message it prints blames ``wrap_io``, which is unrelated
-- see :ref:`limitation-19`. ``tests/dataflow/test_stream_ops_hls.py::test_tapa_stream_nb`` asserted
``.try_read(`` / ``.try_write(`` and was therefore failing outright; it is now
``xfail(strict=True, raises=RuntimeError)``.

..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##############################################################
U2: ``vpu_word_array`` (VMEM, two ports) across every tool column
##############################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch
   ``u2-word-array`` (from ``origin/u1-pilot`` at ``bf0e60ab``). Bindings: a
   private copy of ``wt-u1``'s build (``ldd`` resolves
   ``libAlloMLIRAggregateCAPI`` inside the worktree). MiniTPU at
   ``b3ba0a4d``, read only. Verilator 5.052, Catapult 2024.2
   (``nangate-45nm_beh``, ``ccs_sample_mem``), DC W-2024.09 (FreePDK45, the
   U1 flow, ``dc_u1.tcl`` byte-identical), RTLGen ``allo-rtlgen`` @
   ``13b55a63``, AMC ``fe60c121``. Nothing under ``allo/`` or ``mlir/``
   changed: every core or emitter defect is recorded with a repro and a
   proposed fix only. Owner's decisions in force: undefined behaviour masked
   and counted; verdicts from single-call traces (D-11); latency reported by
   the backend's manifest and checked on RTL (D-10).

The second U2 unit (README D-9), after the ``vpu_regfile`` pilot
(``u2_regfile_2026-10-02.rst``, whose variants, workarounds and scripts this
record mirrors). The unit is MiniTPU's VMEM storage: one array of whole
words, two symmetric read/write ports (``compute``, ``dma``), **registered
reads** (3 edges on the compute port, 2 on the DMA port), writes visible
one cycle later from either port, no reset. The new element against the
regfile is the read latency; the probe this unit was chosen for is the
two-port question: two independent clients of one memory, same-cycle
read + write on different ports, a same-word collision undefined and masked
-- what does each backend honour, refuse, or silently change (D-1).

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first)::

   $ALLO_PYTHON -m examples.minitpu.harness.check vpu_word_array --inst narrow \
       --variant trace --backend simulator                    # any VARIANTS key / inst
   $ALLO_PYTHON -m examples.minitpu.harness.check vpu_word_array --inst narrow16 \
       --variant trace --backend systemc                      # csim needs words < 64 b (S8)
   $ALLO_PYTHON dev/records/minitpu/u2_word_array_2026-10-02/scripts/emit_csyn.py \
       vpu_word_array wire <prj> --n 64 --inst narrow --width 64 --pipeline wa_0:_ \
       --unroll wa_0:k --unroll wa_0:j --synth-top wa_0 --clock 3.33   # Catapult
   $ALLO_PYTHON dev/records/minitpu/u2_word_array_2026-10-02/scripts/cmp_trace.py <prj>
   $ALLO_PYTHON dev/records/minitpu/u2_word_array_2026-10-02/repros.py   # every finding, ~2 min

RTLGen and AMC: ``rtlgen/run.sh`` and ``amc/run.sh`` beside this file, in
those tools' own environments; ``rtlgen/trace_io.py`` (Allo env) dumps the
trace with MiniTPU's responses and compares their outputs. DC:
``scripts/dc/run_dc.sh`` with ``scripts/dc/dc_runs.txt``.

Instances and what "cycle" means in each column
================================================

The RTL instances (``units/vpu_word_array.py`` ``GEOM``): ``narrow`` (8
words of 64 b, the verdict instance), ``narrow16`` (32 words of 16 b,
``MINITPU_NUM_SUBLANES=1``, added for the csim column: S8), ``mid`` (4,096
words of 64 b, the Catapult RAM-mapping probe), ``full`` (4,096 x 1024 b,
the shipped geometry). Every added instance gives ``REF-MATCH`` and
``LATENCY-OK`` (3 / 2 / 1) in ``characterize``.

The command trace is Phase 0's for the instance, every trace joined end to
end: ``narrow`` 80,609 cycles, 14 traces, **67,717 defined** slots and
93,501 masked (``write cycle`` 48,344, ``no read`` 41,104, ``collision``
2,693, ``ww collision`` 1,353, ``uninit`` 7); ``narrow16`` 68,666 defined.
The mask share is high by construction: ``rdata`` is undefined on every
cycle that holds no read and on every write cycle.

* **Allo simulator**: untimed; cycle ``t`` is iteration ``t``, an *order*
  check, plus the read pipe's row offset when the variant carries it.
* **SystemC csim**: values per iteration, time unchecked.
* **Catapult RTL** (``wire``): driven per clock by the harness ``trace``
  shape with MiniTPU's command trace, compared per clock; read latency and
  write visibility measured by the unit's own step probes on both RTLs.
* **RTLGen / AMC**: per-iteration values and a total cycle count.

Variants (``examples/minitpu/units/vpu_word_array.py``)
=======================================================

All take one array per RTL input port (``uint1`` en/we, ``UInt(AW)``
addresses, ``UInt(W)`` data) and return one per ``rdata`` port, from the
same trace. The question "how is a 3-cycle read pipeline expressed?" has
two answers here, and every column was asked both:

=================  ======================================================================================
variant            expression
=================  ======================================================================================
``trace``          W1, **pipe as data**: one kernel, ``mem: W[WORDS]`` and one shift-register array per
                   port (``pc: W[3]``, ``pd: W[2]``), the RTL's ``*_read_pipe`` transcribed. Per
                   iteration the pipes shift, each enabled port reads (the old word) or writes --
                   one access per port per cycle, as the board's XPM port does -- then
                   ``q[t] = pipe[L - 1]``. The latency is an iteration shift *inside* the kernel
``issue``          W1 as the plan words it, **latency outside the kernel**: no pipe, ``q[t]`` is the
                   read issued at ``t``; ``RESP_SHIFT`` makes ``check.py`` hold ``q[t]`` against the
                   RTL's row ``t + L - 1``. The delay is then a contract (D-10 ``latency=L``, per
                   port), which no backend can state today
``trace_rw``       the simulation model literally: every enabled port reads *and* (if ``we``)
                   writes: two accesses per port per cycle
``ported``         R3: ``@df.unit`` with 8 command + 2 response ``Stream`` ports, lockstep
``shared``         W2: region-scope ``mem @ Stateful``, one kernel per physical port, each with
                   its own pipe, no other link
``shared_sync``    ``shared`` + a per-cycle barrier (one token each way): the clock's order
``annotated``      ``trace`` + ``Memory(resource="URAM", storage_type="RAM_T2P", latency=3,
                   depth=WORDS)``: the D-1 sweep
``wire``           ``trace`` with ``Wire`` ports, unit kernel ``wa`` (``synth_top="wa_0"``); each
                   port's access is one ``if we: write else: read``
=================  ======================================================================================

Workarounds carried from the pilot: B4 (addresses widened to ``int32``
before indexing), B5 (stream/wire addresses read into ``UInt(AW)`` first),
S6 (every port array read unconditionally). New ones this unit needed:
the shift loops run forward with ``p[L - k]`` indices (a negative-step
``range`` is refused, F3 below); ``shared``/``shared_sync`` are two kernel
bodies rather than one body under ``if sync:`` (F4); the DMA shift loop has
its own variable ``j`` so Catapult's ``s.unroll`` can name it (C-W1).

Results, ``narrow`` (64 b) unless marked
========================================

.. list-table::
   :header-rows: 1
   :widths: 13 22 22 43

   * - variant
     - Allo simulator
     - SystemC csim
     - notes
   * - ``trace``
     - **match** 67,717/67,717
     - **finding** 2/67,717 (narrow); **match** 68,666/68,666 (narrow16)
     - csim: **S8** -- the testbench reads every port through ``long long``; the first data
       word >= 2^63 fails extraction and every later value of that file is lost. At 16 b
       the same variant matches
   * - ``trace_rw``
     - **match** 67,717/67,717 (and 93,321/93,501 of the *masked* slots)
     - 2/67,717; narrow16 **match** (92,305/92,552 masked too)
     - the literal model also reproduces the sim model's masked guesses (old word on a write
       cycle, the DMA write winning a write-write collision)
   * - ``issue``
     - **match** 67,717/67,717 (shift 2 / 1 rows)
     - 2/67,717; narrow16 **match**
     - both untimed columns are indifferent to where the latency lives: an iteration shift
       inside or a row offset outside the kernel are the same check
   * - ``ported``
     - **match** 67,717/67,717
     - 2/67,717; narrow16 **match**
     - unit factory per ``(n, w, inst)`` (C9) again
   * - ``shared``
     - **finding** 34,240/67,717, no diagnostic
     - **refused** (both widths)
     - simulator: the two port kernels run unordered (M2 again; deterministic in one run).
       SystemC: "stateful variable ... is used by 2 kernels" (D-1 honoured; G1)
   * - ``shared_sync``
     - **match** 67,717/67,717
     - **refused**
     - the simulator honours two clients of one memory once the per-cycle order is in
       streams; same-word collisions were masked, so the order *within* a cycle never
       mattered. SystemC refuses sharing regardless (G1)
   * - ``annotated``
     - **match** (annotation ignored)
     - 2/67,717; narrow16 **match** (annotation dropped)
     - H3: ``RAM_T2P``/``latency=3`` never reach a backend; SystemC emission identical to
       ``trace``. Nothing refuses
   * - ``wire``
     - **blocked**: ``Failure while creating the ExecutionEngine``
     - **finding** narrow16 7,330/68,666
     - as in the pilot: the simulator does not refuse ``Wire`` by name; csim's src/sink are
       not cycle-locked to the unit (limitation 22). The unit kernel is exact in RTL (next)
   * - ``full`` (1024 b)
     - **blocked** at the host data path
     - **blocked**
     - both backends *build* ``UInt(1024)[4096]``; no numpy dtype carries a 1024-bit
       element (H8/B6), and csim's TB would hit S7 at read-back. Not run

Catapult (``wire``, ``synth_top=wa_0``, ``s.pipeline("wa_0:_")``, 3.33 ns unless marked, n = 64):

.. list-table::
   :header-rows: 1
   :widths: 26 14 26 34

   * - build
     - csyn
     - per-cycle vs MiniTPU RTL
     - measured timing (step probes)
   * - as emitted (shift loops rolled)
     - ok, II=1 *reported*; ``while`` loop **2 c-steps**
     - **19,502/67,717**: the unit samples its Wire inputs every **second** cycle
     - probes never move within 36 cycles; manifest ``status=unreliable`` ("rolled loop(s)
       ['l_S_k_0_k'] merged into the scheduled loop") -- **D-10 worked**: the number was
       flagged, not consumed (C-W1)
   * - + ``s.unroll`` on both shift loops
     - ok, 47 s, II=1, 1 c-step
     - **67,717/67,717** at the same output row as MiniTPU
     - read **3** / **2**, write->read **1** on all four port pairs: **cycle-exact** to
       MiniTPU, on both ports. Manifest: ``latency=2, ii=1, scheduled``; measured kernel
       port-to-port 1 -> ``LATENCY-MISMATCH`` (the manifest counts the reset c-step, C-M1)
   * - + ``s.partition(mem, Complete)``
     - ok, identical RTL area
     - **67,717/67,717**
     - no effect: Catapult already keeps an 8 x 64 b array in registers (CIN-341 acknowledged)
   * - unrolled, 2.0 ns
     - ok, 39 s, II=1
     - (DC below)
     - --
   * - ``mid`` (4,096 x 64 b), default mapping
     - **refused**, SCHD-30
     - --
     - ``mem`` mapped to ``ccs_ram_sync_1R1W`` (MEM-4): two conditional reads and two
       conditional writes per cycle do not fit one read and one write port. Loud, D-1-correct
   * - ``mid``, hand-patch ``MAP_TO_MODULE ccs_ram_sync_dualport``, read/write of a port
       as two ``if`` trees
     - **refused**, SCHD-30
     - --
     - Catapult cannot prove the read and the write of one port exclusive and needs four
       ports of a two-port RAM
   * - ``mid``, same, each port one ``if we: write else: read`` (the committed ``wire``)
     - **refused**, SCHD-30, "could not schedule even with unlimited resources"
     - --
     - not the port count: a *chained feedback data dependency* through ``mem:rsc`` (SCHD-6:
       a ``write_mem`` in one iteration, a ``read_mem`` of the same RAM in the next, at
       II=1). MiniTPU's XPM honours exactly that (write visible next cycle); Catapult's
       sync-RAM model does not at II=1. Asked for **II=2** (``s.pipeline(..., initiation_interval=2)``)
       the same build schedules on the ``ccs_ram_sync_dualport`` (3 c-steps, area score
       1,946): a two-port VMEM exists in Catapult only at half MiniTPU's command rate
   * - ``narrow``, each port one ``if/else`` (the committed ``wire``)
     - ok, 38 s, II=1, 1 c-step
     - **67,717/67,717**
     - read 3 / 2, write->read 1 on all four pairs: the same cycle-exact result; Catapult
       area score 7,958.8 vs 7,416.6 for the two-``if``-trees form (the ``else`` costs a mux)

Same-flow area and timing (DC W-2024.09, FreePDK45, ``dc_u1.tcl`` unchanged; ``narrow``):

.. list-table::
   :header-rows: 1
   :widths: 40 10 14 12 12 12

   * - design
     - clock
     - area (um^2)
     - comb
     - seq
     - slack (ns)
   * - MiniTPU ``vpu_word_array`` narrow, simulation model, writes merged into one block
       (``scripts/dc/vpu_word_array_sim.sv``)
     - 3.33 / 2.0
     - 5,513.9 / 5,515.5
     - 1,711.7 / 1,713.3
     - 3,802.2
     - 2.18 / 0.71
   * - Catapult ``wire`` unrolled, committed form (one ``if/else`` per port; registers)
     - 3.33 / 2.0
     - **7,641.4** (+39 %) / 7,641.4
     - 2,422.5 / 2,422.5
     - 5,218.9 / 5,218.9
     - 2.02 / 0.74
   * - Catapult ``wire`` unrolled, read and write as two ``if`` trees (same behaviour)
     - 3.33 / 2.0
     - 7,189.4 (+30 %) / 7,191.8
     - 1,974.5 / 1,976.9
     - 5,214.9
     - 2.26 / 0.96
   * - Catapult ``wire`` as emitted (merged loops; not the unit's behaviour)
     - 3.33
     - 6,238.5
     - 1,752.4
     - 4,486.1
     - 2.30

Like for like is direct here: both designs are flops with registered read
pipes (the regfile needed an output-register wrapper). Catapult's +30 to
+39 % is sequential (+37 %: its input registers and the 64-bit pipe stages
held as full registers) and combinational (+15 % / +42 %: the ``else`` form
adds a mux per port). **MiniTPU's own simulation model
is not DC-synthesizable** (F6): its two ``always_ff`` blocks both write
``mem`` (ELAB-366, multiple drivers), and DC refuses `` `undef SYNTHESIS``
(VER-402), so the XPM branch is what a synthesis tool sees. The DC copy
merges the two blocks, DMA write last (the model's block order).

RTLGen (``rtlgen/wa_rtlgen.py``, R1 transcribed; one cosim of the whole
trace):

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - build
     - vs MiniTPU (defined)
     - cycles / structure
   * - ``trace``
     - **67,717/67,717**
     - 161,219 cycles (**II=2**). ``mem`` is 8 registers (``mem_0..7``) by RTLGen's own choice
       (no partition asked); the pipes are registers too
   * - ``trace_part`` (``partition(mem, Complete)``)
     - **67,717/67,717**
     - 161,219 cycles: the partition changes nothing (it was already registers)
   * - ``trace_rw`` (4 accesses)
     - **67,717/67,717**
     - 161,219 cycles: the extra reads cost nothing either

AMC (``amc/wa_amc.py``, through AMC's vendored frontend):

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - build
     - vs MiniTPU (defined)
     - cycles / structure
   * - ``llvm`` and ``amc``, unscheduled
     - **67,717/67,717** (amc, full trace); 22/22 at N=256 (both)
     - 403,057 cycles (~5 cycles/iteration)
   * - ``s.pipeline(t)``
     - **finding** 12,425/67,717 -- **wrong, silently** (not a row shift: 18,544 at +1,
       20,286 at +2, 6,299 at 0)
     - prints ``error: problem is infeasible`` (twice), then *returns a design*: 241,842
       cycles (**II=3**). Nothing raised; the caller cannot tell (A1, bug by D-1)
   * - ``s.partition(mem, complete)`` + pipeline
     - **finding** 1/22 (N=256), wrong
     - "banks with more ports than the architecture allows for (2)" x5, then a design
   * - ``trace_rw`` + pipeline
     - 0/22, wrong
     - as pipeline

AMC's allocation (``amc/amc_amc_pipeline_trace_256.alloc.txt``) is the
column's answer to the two-port question: ``amcMemory0`` (``mem``) gets
**five logical ports** ``(w, r, r, w, w)`` -- the compute and DMA reads and
writes, plus the zero-initialisation's write -- each ``dyn ... (1)``
(registered, latency 1); the implementation (``amcMemory0_impl.sv``) has
**two physical ports, p0 read/write and p1 read-only**. VMEM's two ports
are both read/write. So AMC's 2-physical-port RAM is a 1RW + 1R, not
MiniTPU's 2RW, and the five logical ports are time-multiplexed on it: II=3
once pipelined, which is what the regfile saw too (H10). The two read pipes
become memories as well (``amcMemory1`` with seven ports, ``amcMemory2``
with five): no register promotion for a 3-entry array.

The two-port question (D-1), by column
======================================

The unit's own contract: two clients, one memory; each port one access per
cycle; same-cycle read + write on different words defined; a same-word
cross-port access with a write undefined (masked here).

* **Allo simulator.** One kernel owning the array (``trace``) honours it by
  program order. Two kernels on a region ``Stateful`` honour it *only* with an
  explicit per-cycle barrier (``shared_sync``); without one the simulator
  silently interleaves (``shared``, 34,240/67,717). The order within a cycle
  never mattered because the slots where it would are masked.
* **SystemC / Catapult.** Two kernels on one memory: **refused** (G1). One
  kernel with two accesses per cycle: honoured; Catapult keeps a small array
  in registers and, for a RAM-sized one, maps a 1R1W by default and
  **refuses** at II=1 (SCHD-30). A two-port RAM is reachable only by a
  Catapult directive (``MAP_TO_MODULE ccs_ram_sync_dualport``) *and* a source
  shape that makes each port's read and write exclusive, and then only at
  **II=2**: Catapult's sync-RAM model refuses a write in one cycle and a read
  of the same RAM in the next, which is the one thing VMEM's contract
  promises (write visible next cycle). Allo can state neither the port kind
  nor that contract: ``Memory(storage_type="RAM_T2P")`` is dropped (H3).
* **RTLGen.** Registers; two reads and two writes per iteration cost II=2.
* **AMC.** Declares the ports the program uses (2 r + 3 w), all registered,
  on a 1RW + 1R implementation: it neither refuses nor matches VMEM's 2RW;
  pipelined it is wrong (A1).
* **Nowhere** can a port count, a port's read latency, or "this access is
  port A" be stated. The read latency (3 / 2) was reproduced exactly only by
  writing the pipe as data; every backend then carries it as registers.

Findings
========

Repros in ``repros.py`` (output ``logs/repros.txt``) unless noted.

**S8 (SystemC emitter, bug, silent; high).** The csim testbench moves every
port through ``long long _v; _f >> _v``: a ``uint64`` value >= 2^63 fails
extraction (failbit), ``_v`` becomes ``LLONG_MAX``, and **every later value
of that file is lost** (the stream stays failed). Repro: ``[1, 2^63-1,
2^63+5, 7]`` -> csim ``[1, 2^63-1, 2^63-1, 2^63-1]``. Every 64-bit variant
gave 2/67,717; the same variants match at 16 b. Same family as S7 (256 b:
``KeyError 'ui256'``). *Proposed fix:* ``unsigned long long`` (or a width-
aware reader) and ``fail()`` checked per read; refuse a width the TB cannot
carry (D-1).

**C-W1 (SystemC emitter -> Catapult, workaround; high).** A constant-trip
inner loop (the shift-register pipe) is emitted rolled with no unroll
pragma. Catapult then *merges* it into the pipelined steady-state loop
(SCHD-7 "2 c-steps"), reports II=1, and the kernel samples its ``Wire``
inputs every second cycle: 19,502/67,717. With ``s.unroll`` on both loops
the emission carries ``#pragma hls_unroll`` and the RTL is cycle-exact. The
D-10 manifest caught it (``status=unreliable``, the rolled loop named), so
the wrong number was never consumed. *Proposed fix:* unroll (or refuse) a
constant-trip loop nested in a pipelined loop with a ``Wire`` port read in
its body; at least warn at emission.

**A1 (AMC, bug by D-1; high).** ``s.pipeline`` on this kernel prints
``error: problem is infeasible`` and then returns a built, runnable design
that computes **wrong** values (12,425/67,717), with no exception. A failed
schedule must refuse. (Logged in ``amc/amc_runs.log``; no Allo-side repro.)

**C-M1 (Catapult manifest, D-10; medium).** On a ``Wire``-port kernel the
manifest's ``latency`` is the sequential's c-step count (the reset loop plus
the steady-state loop: 2), not the port-to-port latency the harness
measures (1). Same on the regfile (its L=1 matched by coincidence of
reading 1 as "one edge"). *Proposed:* for ``port_style=wire`` report the
steady-state loop's c-steps as the latency, and say that the pipe-as-data
depth is not the manifest's to know.

**F3 (frontend, loud, minor).** ``for k in range(2, 0, -1)`` is refused by
the affine lowering ("expected step to be representable as a positive
signed integer"). Workaround: a forward loop with ``p[L - k]`` indices.

**F4 (frontend, loud, minor).** A Python ``bool`` captured from the
enclosing scope under ``if flag:`` is lowered as an ``i32`` constant and the
verifier refuses the ``scf.if``. Expected: constant-fold, or refuse by name.

**F6 (MiniTPU, FYI).** The simulation model of ``vpu_word_array`` is not
synthesizable: two ``always_ff`` blocks drive ``mem`` (DC ELAB-366). The
board's XPM branch is the only synthesis path. For the same-flow area the DC
copy merges the blocks (``scripts/dc/vpu_word_array_sim.sv``).

**Harness.** ``check.py`` now passes the instance to a variant that takes
one (``make(n, w, inst)``) and honours ``RESP_SHIFT`` (a per-port row
offset) so the ``issue`` form can be checked; ``cmp_trace.py`` measures
each response port's offset and latency separately and prints the manifest
verdict. A 1024-bit instance needs a wider host data path than numpy has
(``_args``: a Python-int list cannot become a ``uint64`` array; H8 stays).

**Minor.** The simulator still reaches ``ExecutionEngine`` with a ``Wire``
instead of refusing it by name. The Catapult-side probe trace must be long
enough for the first read to land (``cmp_trace`` reports "never moved"
instead of asserting).

Hypotheses (plan section 4) that apply to the word array
=========================================================

.. list-table::
   :header-rows: 1
   :widths: 8 18 74

   * - #
     - verdict
     - evidence
   * - H1
     - **refuted** as stated; the *simulator* matched first time
     - simulator ``trace`` 67,717/67,717 on the first run; csim needed a second RTL
       instance (S8), Catapult an unroll (C-W1)
   * - H2
     - as the pilot: SystemC refuses, simulator races
     - ``shared`` refused / 34,240; ``shared_sync`` matches on the simulator
   * - H3
     - **confirmed**
     - ``annotated`` (``RAM_T2P``, ``latency=3``) dropped everywhere, nothing refuses
   * - H4
     - **confirmed, and the registered read is where it stops mattering**
     - the unit's own latency is >= 1, so Catapult's one-edge port latency is absorbed:
       read 3 / 2 and visibility 1 are reproduced exactly with the pipe as data
   * - H5
     - **refuted for a small array**; the RAM-sized case is the SCHD-30 of the pilot again
     - 8 x 64 b: registers without any partition; 4,096 x 64 b: 1R1W, refused
   * - H8
     - **confirmed** (64 b now, not only 256 b)
     - S8 at 64 b; the 1024-bit instance builds and cannot be driven (numpy)
   * - H9
     - **confirmed** (C9)
     - ``ported`` needs a unit factory per (n, w, inst)
   * - H10
     - **confirmed**: AMC has no 2RW
     - 5 logical ports on 1RW + 1R; pipelined wrong (A1), not II=1
   * - H11
     - n/a for the unit; **the simulator does not refuse** the two-client race
     - ``shared`` silently interleaves; ``shared_sync`` is the "port rule" written by hand
   * - H12
     - sim-model half **reproduced by Allo**
     - ``trace_rw`` reproduces 93,321/93,501 of the *masked* slots (old word on a write
       cycle; DMA write wins), which the owner has ruled undefined

What the owner must decide
==========================

1. **S8 now.** Silent, and it hits every 64-bit datapath in csim (TinyTPU's
   int32 is safe; MiniTPU's 64-bit words and anything ``uint64`` are not).
   *[fix with S7: a width-aware TB reader, refusal above what it carries]*
2. **C-W1.** Unroll/refuse a rolled constant-trip loop inside a pipelined
   ``Wire`` kernel at emission, or document ``s.unroll`` as mandatory for
   pipe-as-data? *[emit the unroll pragma for constant-trip inner loops under
   a pipelined loop; keep the manifest flag]*
3. **A1.** Report to AMC (a failed schedule returns a design). *[issue
   upstream; AMC cells stay "unscheduled only" until then]*
4. **The read latency as data.** The pipe-as-data form (``trace``) is what
   made every RTL column cycle-exact, and it is what the hardware is. Adopt
   it as the reference expression for registered-read units, and keep
   ``issue`` + ``latency=L`` as the form for a contract the backend must
   *meet* (triage 8)? *[yes; the memory-port D-n should let a port declare
   its read latency, lowered to exactly this pipe]*
5. **Memory ports (G1, plan Q8), second concrete case.** Two clients of one
   memory are expressible only as one kernel; AMC's typed ports are the one
   place a port *set* exists, and it is a 1RW + 1R. The D-n must say: port
   count, each port's kind (RW), its read latency, and what a same-cycle
   same-word access means (refuse at compose time when static, undefined when
   not). *[draft with the regfile's item 5]*
6. **The simulation model's FYI (F6)** goes to MiniTPU's owner: the model
   cannot be synthesized, so an ASIC VMEM would be written anew.

Files
=====

* ``examples/minitpu/units/vpu_word_array.py``: the variants and the
  ``narrow16``/``mid`` instances; ``examples/minitpu/harness/check.py``:
  ``inst`` and ``RESP_SHIFT``.
* ``scripts/``: ``emit_csyn.py`` (the pilot's + ``--inst``), ``cmp_trace.py``,
  ``run_check.sh``, ``dc/`` (``run_dc.sh``, ``dc_runs.txt``, ``dc_u1.tcl``
  unchanged, ``mtpu_wa_narrow.sv``, ``vpu_word_array_sim.sv``).
* ``logs/`` (verdict lines per variant x backend x instance, ``repros.txt``),
  ``catapult/`` (per build: ``run.tcl``, ``csyn.log.gz``, ``rtl.rpt.gz``,
  ``cycle.rpt``, ``latency.json``; ``cmp_*.txt``; the as-emitted and unrolled
  ``kernel.cpp``), ``dc/`` (area/timing/qor/reference per run), ``rtlgen/``
  (script, runs, cmp, the RTL), ``amc/`` (script, runs, cmp, the allocation,
  ``amcMemory0_impl.sv``).

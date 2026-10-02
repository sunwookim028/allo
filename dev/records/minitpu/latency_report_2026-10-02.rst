..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

################################################################
Unit latency: report first, constrain second
################################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch
   ``latency-report`` (from ``origin/u1-pilot`` at ``bc7399b6``), worktree
   ``scratch/wt-lat`` with a private bindings copy of ``scratch/wt-u1``'s
   ``_mlir`` (``ldd``: every library resolves inside the worktree). MiniTPU
   at ``b3ba0a4d``. Catapult Ultra Synthesis 2024.2/1130128
   (``nangate-45nm_beh``); Vitis HLS 2023.2 (``xcu280-fsvh2892-2L-e``,
   ``/opt/xilinx/Vitis_HLS/2023.2``); Verilator 5.052; RTLGen
   ``kkkaishao/allo@13b55a63`` and AMC ``amc-dialect@fe60c121`` as built in
   ``dev/records/open_hls/``. Units: ``vpu_bf16_add(_pipe)`` ``bits``
   (latency 0 / 2) and ``mxu_acc24_add_pipe`` ``bits`` (latency 3). README
   D-1, D-7, D-9; ``u1_matrix.rst`` triage items 4, 8, 12.

The owner's framing (2026-10-02): from MiniTPU's side, is pinning needed at
all, or can a software compiler consume whatever latency the HLS tool
scheduled? So the order here is **(A)** each RTL backend *reports* each unit's
achieved latency and II in a form a latency table can consume, and the
harness *checks* the report against the RTL; **(B)** an optional exact
constraint where an external contract needs one; **(C)** clockless units.

.. contents::
   :local:
   :depth: 1

Answer
======

* **(A) holds for every tool, with a different file in each.** The reported
  number equals the Verilator-measured RTL latency on every build where the
  tool claims one: Catapult 65/65 historical U1 builds and 10/10 new ones,
  Vitis 10/10, RTLGen 17/17 (agent run), AMC 14/14 (agent run). The one way
  a report goes wrong is known and detectable: a rolled inner loop that
  Catapult merges into the pipelined loop (u1_pipe L5). The prototype flags
  all 3 such builds and reports no number for them.
* **(B) exists only in Catapult and Vitis**, and only Catapult refuses an
  infeasible value. Vitis's ``#pragma HLS latency`` is met exactly (5/5), but
  where the value cannot be met it keeps the latency and misses the clock
  (estimated 8.8 ns against a 2.0 ns target), with a warning and exit 0.
  RTLGen and AMC have no latency constraint. Both formulations could take one
  cheaply (below).
* **(C) Catapult can emit a clockless unit:** ``#pragma hls_design ccore`` with
  ``#pragma hls_ccore_type combinational`` on Allo's own emitted ``add_bits``
  gives a module ``bf16_add_comb(a_i, b_i, result_o)`` with no clock, no
  sequential area, and 251,936/251,936 bit-exact against ``vpu_bf16_add.sv``
  in the harness's ``comb`` shape. RTLGen and AMC cannot emit one: each adds
  ``clk``/``rst``/``start``/``done`` unconditionally.
* **The latency is the tool's, not the source's.** The same ``bits`` adder
  at 5 ns gets these latencies, each of them exact in its own RTL:

  - acc24: Catapult 2, Vitis 5.
  - bf16: Catapult 2, Vitis 5, AMC 1 (its 3 minus two memory cycles).

  A consumer must read the latency from the build. It cannot assume one.

Per-tool table
==============

.. list-table::
   :header-rows: 1
   :widths: 10 26 22 22 20

   * - tool
     - where the latency is reported (machine-readable?)
     - trustworthy vs measured RTL?
     - constrainable? what if infeasible?
     - why (root cause)
   * - **Catapult** (Allo SystemC)
     - ``<sol>/cycle_set.tcl``: the c-step of every op. Latency is
       ``max(Push) - min(Pop)`` in the I/O loop. ``cycle.rpt``'s loop table
       gives iterations, c-steps and ``Init`` (II). The process ``Latency``
       column is ``-1`` for a free-running Connections kernel (F8); for a
       ``Wire`` kernel it is the latency. Text, regex-parsable. Now written
       to ``latency.json`` by Allo.
     - **Yes**, on 65 historical builds (``validate_history.txt``) and 10 new
       ones (``rtl_cmp.txt``), on both Connections and Wire ports, at 2.0,
       3.33 and 5.0 ns. **No** when a rolled loop is merged into the
       pipelined loop: reported 1/3/2, measured 19/21/"2 or 18". It is
       detectable, because cycle.rpt's iterations are more than the loop's
       LOOP-2 trip count (×19, ×17) and the rolled loop is missing from the
       loop table.
     - **Yes, exact**:
       ``cycle set {out.Push()} -from {in.Pop()} -equal L``. Measured = L for
       L = 1-6, now emitted from ``configs["latency"]``. An infeasible L is
       **refused**: ``SCHD-30 ... could not schedule even with unlimited
       resources`` (acc24 L=2 at 2.0 ns). L = 0 is refused (a Push cannot
       share a c-step with its Pop, N3). Wire ports cannot be pinned (N2).
     - The scheduler takes per-op ``CSTEPS_FROM`` constraints, and an I/O op
       on a Connections port is a scheduled op. An ``sc_in`` read is not, so
       a Wire port gives the constraint nothing to anchor to.
   * - **Vitis HLS** 2023.2
     - ``syn/report/<m>_csynth.xml``: ``PipelineDepth``, ``PipelineII``,
       estimated and target clock. ``.autopilot/db/<m>.verbose.sched.rpt``:
       the state ``ST_n`` of every op. Latency is
       ``max(ap_fifo write ST) - min(ap_fifo read ST)``, which equals
       ``PipelineDepth - 2`` for these units. XML and text. Now written to
       ``latency.json``.
     - **Yes**, 10/10 (``vitis/vitis_cmp.txt``), on ap_fifo-port kernels at
       2 and 5 ns, pinned or not. Not determined for Allo's default region
       path: there the unit's arrays are ``ap_memory`` ports with 2-cycle
       loads, and the manifest says ``unknown``.
     - **Partly.** ``#pragma HLS latency min=L max=L`` on the loop body
       gives measured = L (3, 5, 1 for acc24; 2, 4 for bf16). An infeasible
       L is **not refused**. Vitis keeps L and violates timing: acc24 L=1
       estimates 8.803 ns at a 2.0 ns target; ``max=1`` estimates 9.501 ns
       at 5.0 ns; bf16 L=2 estimates 6.008 ns at 5.0 ns. It only prints
       ``WARNING: [HLS 200-871] Estimated clock period ... exceeds the
       target``, and csynth exits 0.
     - The latency directive is a scheduling constraint that ranks above
       the clock. Timing is only an estimate checked after scheduling. The
       manifest marks such a build ``unreliable``, so a D-1 refusal has to
       come from Allo.
   * - **RTLGen** (Kai)
     - ``RTL.schedule()`` (per region: ``interval``, ``trip_count``,
       ``latency``, ``iteration_latency``, ``cost.drain``), ``.interfaces``,
       ``.estimation``, and ``scaffold_project``'s ``manifest.json``
       (``latency``, ``latency_bound``, ``determinacy``). JSON. Kernel
       latency counts ``start`` to ``done``; per element it is ``drain``.
     - **Yes**, 17/17. Measured is exactly ``(N-1)·II + drain + 1`` (bf16
       array and scalar kernels at 10-500 MHz; acc24 at 50-500 MHz).
       ``cosim()`` checks this contract itself and raises "the latency
       contract is UNSOUND" (``core.py:440-476``). At 800 MHz it quietly
       schedules at 509 MHz instead. The cycle count is still right, but
       ``estimation.clock_mhz`` says 800.
     - **No.** ``set_scheduler_opt(latency=2)``: ``ValueError: unknown
       scheduler option(s) ['latency']``. ``pipeline(ii=)`` sets a minimum
       II. ``area_slack`` (CP-SAT) is the only knob that raises latency, and
       it does so indirectly.
     - SDC solves for the least, as-soon-as-possible schedule, with chain
       breaks fixed by the clock before the solve (``SDC.cpp:77-114``,
       ``:491-563``). CP-SAT minimises ``drain``, then area
       (``CPSATScheduler.cpp:1726-1791``). A bound would be one extra edge
       in SDC, or one constraint on ``drain`` in CP-SAT, plus about five
       places of plumbing.
   * - **AMC**
     - ``f.dump_schedule()``: the ``loopschedule.pipeline II = 1 trip_count
       = 16 latency = L`` attributes (``LoopScheduleOps.td:88-103``). MLIR
       text, no JSON. A sequential loop has no ``latency``.
     - **Yes**, 14/14. Measured is exactly ``(trip-1)·II + L + 1``, at
       10-1 ns. L includes the 1-cycle BRAM load and the store commit, so
       the unit's own depth is ``L - 2`` (VCD-checked at 3 periods).
     - **No.** A requested II is an assertion: ``error: Failed to schedule
       for desired II of 2, minimum II is 1``. A period too short gives
       ``Delays of operator type 'comb' exceed maximum cycle time``. Retiming
       by hand after scheduling works (16/16).
     - In CIRCT's ``Problem`` classes, latency belongs to an operator type
       only (``Problems.h:61,188,286``). There is no bound on the total, and
       the simplex scheduler minimises the last op's start
       (``SimplexSchedulers.cpp:57-58``). Pinning the output store with the
       existing ``scheduleAt``/``additionalConstraints`` would take about
       100 lines of C++.

The RTLGen and AMC rows come from two sub-agent runs on this host, with
scripts and logs in ``latency_report_2026-10-02/{rtlgen,amc}/``. Their
scripts still point at ``scratch/{rtlgen,amc}/latency`` and at those tools'
envs (``u1_bf16_add_{rtlgen,amc}/env.sh``). AMC needs a local-disk
``TMPDIR`` (used: ``/scratch/sk3463/amc_latency_tmp``). Code references are
into the tools' own trees (``scratch/rtlgen/kai-allo``,
``scratch/amc/amc-dialect``). The AMC acc24 row was skipped: AMC's frontend
bugs A1-A8 put its transcription over 30 minutes. RTLGen's acc24 port was
almost verbatim and matched 1024/1024 at 7 clocks.

(A) The prototype: a latency manifest, and a check against RTL
==============================================================

**No programming-model change.** Allo reads the reports and writes
``<project>/latency.json`` beside the build:

* ``allo/backend/catapult.py``:

  - ``parse_catapult_schedule`` reads ``cycle.rpt``'s fixed-width tables,
    ``cycle_set.tcl``'s c-steps and the log's LOOP-2/LOOP-4 lines.
  - ``catapult_latency_manifest`` gives each kernel ``latency``, ``ii``,
    ``io`` (each port op's c-step), ``port_style`` (``connections`` or
    ``wire``), ``status`` (``scheduled`` / ``unreliable`` / ``unknown``) and
    ``reason``.
  - ``write_latency_manifest`` writes the file. ``hls.py`` calls it after
    every successful ``csyn``/``ppa`` run and prints one ``[latency]`` line
    per kernel.

* ``allo/backend/vitis.py``: ``vitis_latency_manifest`` does the same from
  ``csynth.xml`` and ``verbose.sched.rpt``. It marks a build ``unreliable``
  when the estimated clock exceeds the target. ``hls.py`` writes it after
  ``vitis_hls`` ``csyn``.
* ``examples/minitpu/harness/latency.py``:

  - ``verdict()`` returns ``LATENCY-MATCH`` / ``-MISMATCH`` /
    ``-UNRELIABLE`` from the manifest and the measured histogram and rate.
  - ``composite()`` handles kernels in series. Their rate is checked
    against the largest II. Their latency is not a manifest number, because
    each FIFO hop adds a cycle: the staged acc24 has 1+1+1 per kernel and
    measured 5.
  - ``latency_report_2026-10-02/cmp_rtl.py`` is the Catapult-track
    comparator with this check appended.
* ``tests/test_latency_manifest.py``: 5 tests on fixtures trimmed from two
  real runs.

A manifest entry (``acc_u_3p33_L3``)::

   "add_0": {"declared": 3, "ii": 1, "io": {"v10.Pop()": 0, "v11.Pop()": 0,
             "v12.Push()": 3}, "iterations": 352116, "latency": 3,
             "loop": "l_S_i_0_i", "loop_c_steps": 4,
             "port_style": "connections", "process": "/top/add_0/run",
             "status": "scheduled"}

**New Catapult builds** (``batch.txt`` via ``build_unit.py``; every RTL is
bit-exact against MiniTPU on the full stimulus, ``rtl_cmp.txt``):

.. list-table::
   :header-rows: 1

   * - build
     - clock
     - manifest
     - measured
     - verdict
   * - acc24 unrolled
     - 3.33
     - 2, II 1
     - 2, 1.000
     - MATCH (unit declares 3)
   * - acc24, ``latency={"add": 3}``
     - 3.33
     - 3, II 1, pinned 3
     - 3
     - MATCH
   * - acc24, ``latency={"add": 5}``
     - 3.33
     - 5
     - 5
     - MATCH
   * - acc24, ``latency={"add": 2}``
     - 2.0
     - refused by Catapult; ``mod()`` now quotes the cause (SCHD-30 ×2,
       "could not schedule partition '/top/add_0/run' ... even with
       unlimited resources")
     - --
     - --
   * - acc24, leading-zero loop rolled
     - 5.0
     - 1, **unreliable** (6,690,204 iterations for a trip count of 352,116;
       ``l_S_offset_0_offset`` merged)
     - **19**, 19 cyc/vector
     - UNRELIABLE (no number consumed)
   * - acc24 ``staged`` (stage 2 not pipelined)
     - 5.0
     - 1/1/1, II 1/2/1
     - 5-9, 2.000 cyc/vector
     - COMPOSITE (rate = max II)
   * - acc24 Wire ports
     - 3.33
     - 2 (cycle.rpt process latency)
     - 2 (``bare``)
     - MATCH
   * - acc24 Wire, ``latency={"add": 3}``
     - 3.33
     - **refused at emit** (N2: no Connections pair)
     - --
     - --
   * - bf16_pipe unrolled
     - 5.0 / 2.0
     - 2 / 3
     - 2 / 3
     - MATCH (unit declares 2)
   * - bf16_pipe, ``latency`` 2 / 4
     - 5.0
     - 2 / 4
     - 2 / 4
     - MATCH
   * - bf16_pipe, ``latency`` 0
     - 5.0
     - **refused at emit** (N3/E1)
     - --
     - --

With the output ready one cycle in three (acc24 pinned 3), the latency is
3-5 at 1.5 cycles per vector, and nothing is lost. A declared latency is a
no-stall property, as before.

**The 65 historical builds** (``validate_history.py``, run against the
projects still in ``scratch/u1_cat2`` and ``scratch/u1_pipe/cat`` and the
measured lines committed with their records):

- 65 OK: 36 Connections builds, 29 Wire builds.
- 3 FLAGGED, each measured unsteady: the rolled-loop builds acc24 at 2.0
  and 5.0 ns, and bf16 at 5.0 ns.
- 2 COMPOSITE (``staged``).
- 0 false flags. The first draft flagged 21 builds because Catapult prints
  LOOP-2 twice, the second time as the trip count plus the pipeline fill.
  The parser takes the smaller.

**(B) in the prototype: Catapult only.** ``configs["latency"] =
{"<kernel>": L}`` makes ``hls.py`` add ``go architect`` plus one ``cycle set
{out.Push()} -from {in.Pop()} -equal L`` per (Out, In) pair before ``go
assembly``. ``io_latency_tcl`` refuses L < 1 and Wire kernels, so the build
fails instead of dropping the request. After the run, a schedule that does
not equal the declared value raises.

**Vitis, measured** (``vitis/``). ``variant.py`` hand-patches Allo's emitted
project: ``set_top add_0``, ``ap_fifo`` on the unit's arrays, and the
``latency`` pragma. ``cmp_vitis.py`` drives the RTL through a 10-line
ap_fifo→valid/ready wrapper. ``ap_start`` is one pulse: held high, the next
call's blocked FIFO read stalls the last call's drain, and 4 outputs never
come out. The ``#0`` initial delays are stripped for Verilator.

.. list-table::
   :header-rows: 1

   * - build (1024 vectors, all bit-exact)
     - target
     - read / write state
     - PipelineDepth
     - est. clock
     - measured
   * - acc24
     - 5.0
     - 2 / 7
     - 7
     - 3.465
     - 5
   * - acc24
     - 2.0
     - 2 / 17
     - 17
     - 1.509
     - 15
   * - acc24, latency 3 / 5
     - 5.0
     - 2 / 5, 2 / 7
     - 5 / 7
     - 4.942 / 3.465
     - 3 / 5
   * - acc24, latency 1
     - 2.0
     - 2 / 3
     - 3
     - **8.803**
     - 1
   * - acc24, latency ``max=1``
     - 5.0
     - 2 / 3
     - 3
     - **9.501**
     - 1
   * - bf16_pipe
     - 5.0 / 2.0
     - 2 / 7, 2 / 14
     - 7 / 14
     - 3.601 / 1.509
     - 5 / 12
   * - bf16_pipe, latency 2 / 4
     - 5.0
     - 2 / 4, 2 / 6
     - 4 / 6
     - **6.008** / 3.694
     - 2 / 4

Allo's own Vitis path, end to end (``mod()`` on the acc24 region, 200 MHz),
writes ``latency.json`` with ``add_0``: depth 7, II 1, latency
``unknown``. The ports there are ``ap_memory`` arrays, which this reader
does not interpret.

(C) Clockless units
===================

**Catapult: yes, as a combinational CCORE.** ``ccore/kernel.cpp`` is Allo's
SystemC emission of ``bf16_add.py::bits`` cut to its ``leading_zeros17`` and
``add_bits`` functions, plus:

- the emitter's two ``ap_int`` aliases;
- a 5-line top marked ``#pragma hls_design top``, ``#pragma hls_design
  ccore`` and ``#pragma hls_ccore_type combinational``.

In the C++ flow (``ccore/run.tcl``, 3.33 ns) this gives:

* ``module bf16_add_comb (a_i, b_i, result_o)``. The top has no ``clk``,
  no ``rst`` and no handshake; ``posedge``/``clk`` occur 0 times in
  ``concat_rtl.v``.
* ``rtl.rpt``: area score 2434.2, all of it combinational (sequential 0.0).
  Max delay 3.305 ns, slack 0.025 at 3.33 ns. ``cycle.rpt`` lists no
  process.
* ``ccore/cmp_comb.py``: **251,936/251,936 bit-exact** against MiniTPU's
  ``vpu_bf16_add.sv``, both in the harness's ``comb`` shape (no clock).

So the clocked interface comes from the top-level port style that Allo
emits (an SC_THREAD with Connections or ``sc_in`` ports), not from Catapult.
Allo already emits each unit function as a plain C++ function, so a ``comb``
unit is one marker on that function plus a C++ (not SystemC) top. Two runs
were not concluded:

- The directive form (``-CCORE_TYPE combinational`` on a ``ccore`` top) used
  a wrong path (``Unknown path '/bf16_add_comb'``).
- The plain-top control failed with ``HIER-55``.

The latency manifest has no entry for a CCORE yet; it should report
``port_style: comb, latency: 0``.

**RTLGen: no.** ``clk``/``rst``/``start``/``done`` are added
unconditionally (``Primitives.cpp:60-62,91``; ``HWEmitter.cpp:643-655``),
and the done latch is a constant cycle (``LatencyModel.h:38``). The least is
1 cycle: at <= 50 MHz the scalar kernel is a combinational cone into a result
register. ``IP(latency=0)`` declares an external black box; it does not emit
your kernel combinationally (``add_rtl_model``: ``NotImplementedError``).

**AMC: no.** Every top module gets ``clk``/``rst``/``start``/``ready``/
``done`` (``LoopScheduleToFSM.cpp:3682-3690, 8301-8322``), and every stage
output is registered (``:10686-10697``). A scalar-port function
(``add16(a, b) -> uint16``) has latency 1 with a start/done handshake. The
scalar bf16 ``bits`` kernel fails to build: ``failed to legalize operation
'seq.hlmem'``. That is a new AMC bug, not reduced here.

Recommendation (proposed README D-10)
=====================================

Triage item 8 proposed ``latency=L, ii=`` everywhere. The evidence here
reverses the order: **report first, constrain only by contract.**

   **D-10 (proposed). A unit's latency is reported by its backend and checked
   on its RTL; it is constrained only where a contract demands.**

   - Every backend that produces RTL writes ``latency.json`` beside the
     build. For each unit it gives ``latency`` (MiniTPU's count: input-accept
     edge to output-visible edge, no stall), ``ii``, the port style, and
     ``status`` with a reason. A number whose status is not ``scheduled``
     must not be consumed. Catapult and Vitis do this on the
     ``latency-report`` branch. RTLGen (``manifest.json``) and AMC
     (``loopschedule.pipeline ... latency``) already report it; their
     adapters are written when they are integrated (M2).
   - A latency table that ACT or an assembler consumes (cf. MiniTPU
     ``docs/isa_latency.json``) is built from manifests, per (unit, backend,
     clock). It is never a constant in the source: the same ``bits`` adder
     is 2 cycles in Catapult and 5 in Vitis at 5 ns.
   - The differential harness checks every RTL unit's measured latency and
     rate against its manifest. ``LATENCY-MISMATCH`` fails a gate;
     ``LATENCY-UNRELIABLE`` blocks the number. A MiniTPU-declared latency is
     reported beside it and does not decide the verdict unless the unit is
     pinned.
   - ``latency=L`` on a kernel is **optional**. It is used where a contract
     outside the tool fixes L: matching MiniTPU at D-7, or cycle-locked
     composition at U3. Catapult honours it through the I/O cycle
     constraint on Connections ports. A Wire kernel, or L < 1, is refused at
     emit. An infeasible L is refused by Catapult, and Allo quotes the cause.
     Vitis meets L by missing the clock, so Allo must refuse when the
     estimated clock exceeds the target. RTLGen and AMC refuse until they
     grow a bound.
   - ``comb`` is a port shape, not ``latency=0``. Catapult can emit it (a
     combinational CCORE, bit-exact here); RTLGen and AMC cannot, so they
     refuse it. Whether Allo grows a ``comb`` unit marker is the owner's call
     at U3 (triage item 4).

   *Amends* triage item 8's proposal: report-and-check is the default, and
   ``latency=`` is the exception. The simulator stays ``untimed`` and SystemC
   csim ``unchecked`` (u1_pipe L1-L3).

Open: the Vitis reader covers ``ap_fifo`` ports only. For Allo's default
``ap_memory`` region path it says ``unknown``; units with ``Stream`` ports
would give ``ap_fifo``. Not tried: a CCORE inside a SystemC region, cosim of
a pinned unit, and Vitis cosim (Verilator was used instead).

Reproduce
=========

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``,
with ``R=dev/records/minitpu/latency_report_2026-10-02`` and
``S=/work/shared/users/phd/sk3463/scratch/lat``::

   $ALLO_PYTHON -m pytest -q tests/test_latency_manifest.py              # 5 tests, ~4 s
   $R/run_batch.sh $R/batch.txt $S 7                                     # 13 Catapult builds, ~1 min each, in parallel
   $R/run_cmp.sh $R/cmp.txt $S > $R/rtl_cmp.txt                         # Verilator + manifest verdicts, ~2 min
   $ALLO_PYTHON $R/validate_history.py \
       dev/records/minitpu/u1_catapult_units_2026-10-02/verilator/rtl_cmp_{stream,wire}.txt \
       dev/records/minitpu/u1_pipe_2026-10-02/rtl_cmp.txt \
       -- /work/shared/users/phd/sk3463/scratch/u1_cat2 /work/shared/users/phd/sk3463/scratch/u1_pipe/cat
   # Vitis: emit, hand-patch, csynth, measure
   $ALLO_PYTHON $R/vitis/build_vitis.py acc24_add_pipe bits $S/vitis/acc_200 200 \
       --unroll leading_zeros19:offset --pipeline add_0:i
   python3 $R/vitis/variant.py $S/vitis/acc_200 $S/vitis/acc24_2p0_L1 add_0 2 --latency 1
   (source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh; cd $S/vitis/acc24_2p0_L1 && vitis_hls -f run.tcl)
   $ALLO_PYTHON $R/vitis/cmp_vitis.py acc24_add_pipe $S/vitis/acc24_2p0_L1
   # CCORE
   mkdir -p $S/ccore/c1/build && cp $R/ccore/{kernel.cpp,run.tcl} $S/ccore/c1/ && \
       (cd $S/ccore/c1/build && catapult -shell -f ../run.tcl)
   $ALLO_PYTHON $R/ccore/cmp_comb.py $S/ccore/c1/build/Catapult/bf16_add_comb.v1

Catapult and Vitis projects stay in scratch and are not kept.

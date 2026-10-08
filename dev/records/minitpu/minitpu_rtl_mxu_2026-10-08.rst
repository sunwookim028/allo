..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##########################################################################
MiniTPU RTL, inverse angle: the Allo MXU in MiniTPU's tree (D-21 probe)
##########################################################################

.. note::

   **Dated measurement record, 2026-10-08.** zhang-21, branch
   ``minitpu-rtl-mxu`` from ``u1-pilot`` at ``3b52c2fd`` (worktree
   ``scratch/wt-mrtl``; bindings copied from ``wt-u1`` at the same commit --
   the ``_allo`` extension's RUNPATH still resolves
   ``libAlloMLIRAggregateCAPI.so`` from ``wt-u1``'s directory, same commit,
   ABI-identical). MiniTPU ``b3ba0a4d`` read only; its tbs ran in a scratch
   copy of ``src/ tb/ docs/ tools/ board_package/`` (``scratch/mrtl_minitpu``,
   ``ORIGIN.txt``). Catapult Ultra 2024.2/1130128, ``nangate-45nm_beh``,
   3.33 ns, ``-IO_MODE super``; Verilator 5.052 (``--binary --timing``, the
   unit suite's flags). ~5 h, owner away: every call is provisional (D-9).
   Nothing under ``allo/`` or ``mlir/`` changed. The plan this record backs:
   ``plan_2026-10-08.rst`` in the record directory (a copy of ``scratch/minitpu_rtl_mxu_plan_2026-10-08.rst``; its sections 1-2, the
   interface map and the tb analysis, are reproduced in section 4 here).

Reproduce (``source examples/minitpu/harness/env-zhang21.sh``, worktree
root, ``R=dev/records/minitpu/minitpu_rtl_mxu_2026-10-08``, ``S`` scratch,
``T`` a scratch copy of the MiniTPU clone with ``R/tb/tb_mxu_single_port_lat.sv``
added under ``tb/``)::

   export PYTHONPATH=$PWD
   $ALLO_PYTHON $R/scripts/u3c_build.py mxu form:mxu_wide $S/mxu2.prj --inst dim2 --n 1048576 \
       --clock 3.33 --pipeline-all --unroll-inner --partition mxu_back_w_0:mem          # ~4 min
   $R/scripts/run_tb.sh $T $S/mxu2.prj tb_mxu_single_port $S/tb2                        # MiniTPU's tb, unchanged
   $R/scripts/run_tb.sh $T $S/mxu2.prj tb_mxu_single_port_lat $S/tb2 +define+MRTL_DIM=2 +define+MRTL_ALLO
   MRTL_PE_DEPTH=4 $ALLO_PYTHON $R/scripts/u3c_build.py mxu form:mxu_wide $S/mxu2q4.prj ... (same flags)
   $ALLO_PYTHON $R/scripts/u3c_build.py mxu form:mxu_wide $S/mxu4.prj --inst dim4 ... \
       --partition mxu_back_w_0:mem --partition mxu_front_w_0:{skd,skv,cbs,wss,hdata}  # ~12 min
   python3 $R/scripts/gen_isa_delta.py --base $T/docs/isa_latency.json --manifest $S/mxu2.prj/latency.json \
       --measured $R/logs/measured_dim2.json --name allo-mxu --write $S/versions_allo-mxu_dim2.json
   python3 $R/scripts/gen_isa_delta.py ... --check $S/versions_allo-mxu_dim2.json     # DELTA-MATCH

Record files: ``forms/mxu_wide.py`` (the packed-port M1), ``scripts/``
(``u3c_build.py`` copied unchanged from track C, ``gen_mxu_allo.py``,
``run_tb.sh``, ``gen_isa_delta.py``, ``collect_logs.sh``), ``tb/`` (the
measuring copy of MiniTPU's tb), ``rtl/`` (the generated ``mxu_allo.sv`` per
build), ``logs/`` (manifests, ``rtl.rpt``/``cycle.rpt``, build lines, tb
outputs, the delta JSON), ``SHA256SUMS.txt``.

.. contents::
   :local:
   :depth: 1

1. Verdicts
===========

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - build
     - Catapult
     - MiniTPU side
   * - ``mxu_wide`` DIM 2, grid links depth 2 (``mxu2``)
     - every kernel **II 1, latency 1** (``pe_unit_1_1`` 0), ``scheduled``; 4 min
     - ``tb_mxu_single_port`` (unchanged, DIM 2): **PASS**. ``tb_mxu_single_port_lat``:
       both tiles **exact**, push->valid **32** (``mxu.sv``: 11), pop->next
       valid **49** (0), but the token lag **grows 1/3 per cycle** (20 at the
       first tile, 48 at the second, 404 after 1,000 idle cycles): the core
       consumes **2/3 token per cycle**. PASS on the check, **lockstep
       violation** by rate (section 2).
   * - same, grid links depth 4 (``mxu2q4``, ``MRTL_PE_DEPTH=4``)
     - every kernel **II 1, latency 1**, ``scheduled``; 10 min
     - ``tb_mxu_single_port_lat``: both tiles **exact**, push->valid **20**
       (``mxu.sv`` 11: the 11 tokens + lag 8 + 1), pop->next valid **9** (lag +
       1), and the lag is **stationary**: ``lag`` 8, ``acc_lag`` 1 at the first
       tile, the second, and after 1,000 and 2,000 idle cycles -- **one token
       per cycle**. PASS, no violation: the first Allo MXU with a declarable
       latency.
   * - ``mxu_wide`` DIM 4 (``mxu4``)
     - first run: ``mxu_front_w`` refused at II 1 -- ``skd`` (52 x 16) mapped to
       ``ccs_ram_sync_1R1W`` (SCHD-6/30), 12 min; with the front's five
       arrays partitioned (``skd``, ``skv``, ``cbs``, ``wss``, ``hdata``):
       every kernel **II 1, latency 1**, ``scheduled``; 25 min (depth 2) /
       17 min (depth 4, ``mxu4q4``)
     - depth 2: both tiles exact, push->valid 92 at the first tile, lag 70
       -> 250 -> 891 (+1,000 idle): **0.4 token per cycle**; the wrapper's
       1,024-deep token FIFO fills during the idle probe and its assertion
       stops the run (**FAIL by the wrapper**). depth 4: both tiles exact,
       push->valid 62, pop->next valid 101, lag 40 -> 100 -> 527 -> 927:
       **0.6 token per cycle**, PASS on the check, violation by rate --
       **depth 4 is enough at DIM 2 and not at DIM 4**.
   * - ``mxu_wide`` DIM 16
     - **not reached**: the SystemC emission of the 256-PE region (one
       ``SC_MODULE`` per PE, ~1,400 lines each) was still in Allo after 11
       min at 3.5 cores and was stopped; DIM 2 -> 4 took the Catapult
       schedule from 4 to 12 min on 16 x the PEs' ops (11,221 real ops at
       DIM 4), so DIM 16 is an hours-scale build, not a one-hour probe
     - --
   * - ``mxu.sv`` (MiniTPU's own, the same tb)
     - --
     - DIM 2 / 4 / 16: push->valid **11 / 21 / 81** (Phase 0's 12/22/82 counted
       from one edge earlier), pop->next valid **0**, pop->valid low 0; PASS

2. The timing answer
====================

**The lockstep MXU's token rate on Catapult RTL is set by the grid links'
depth against the geometry: at DIM 2, depth 2 gives 2/3 token per cycle and
depth 4 gives one token per cycle with a constant lag (the declarable
version); at DIM 4, depth 2 gives 0.4 and depth 4 gives 0.6, and no
stationary lag was reached within the bound.** Every kernel is II 1 in
every build (the manifests); the rate is the composition's. So:

* the result latency MiniTPU sees is ``(2 + DIM (PE + 1))`` **tokens plus
  the lag, and the lag grows without bound** -- 32 cycles at the first
  tile, 49 for the second pop, 737 after 2,000 cycles -- there is no
  ``result_latency.vmatpush`` to declare; the wrapper's token FIFOs (1,024)
  fill after ~3,000 cycles and flag the violation;
* ``tb_mxu_single_port`` passes because it *waits* for ``output_valid``
  (``wait``), the one latency-agnostic MiniTPU tb; a vpu-level tb would
  see every ``vmatpop`` land later than the previous one by a growing
  amount, and the program's writeback calendar (``asm.py``'s
  ``_M_RESULT_LATENCY``, the 7-bit delay field) has no form for that;
* option (b), "declare the longer latency as a version", presumes a
  **stationary** latency: it needs the core at one token per cycle first.
  ``gen_isa_delta.py`` refuses every latency quantity when
  ``tokens_per_cycle < 1`` and emits only the capacities (``WB_W_MPOP_LAST``
  3, ``mxu_output_fifo.depth`` 64; ``logs/versions_allo-mxu_dim2.json``,
  ``--check`` DELTA-MATCH), with the early values under ``unresolved``;
* the 2/3 is the link depth, by a one-variable test: each kernel is II 1
  with ``Pop`` in c-step 0 and ``Push`` in c-step 1 (``loop_c_steps`` 2),
  and a ``Connections::Fifo<T, 2>`` between two such kernels, with the PE's
  two inputs arriving from different depths (west from column ``c``, north
  from row ``r``), sustains two tokens per three cycles; at depth 4 the same
  RTL kernels sustain one per cycle (``mxu2q4``); at DIM 4 the rates are
  0.4 (depth 2) and 0.6 (depth 4): the depth a geometry needs grows with
  it, and the next probe is DIM 4 at depth 8 and 16 (17 min each). The
  link depth is a composition parameter (``Channel.depth``), not a unit
  property: the landed ``pe_channels`` depth "2" is a *latency-neutral*
  choice on the simulator (untimed) and a rate bug on Catapult RTL. At DIM
  16 the arrival-time spread between a PE's two inputs is up to 15 tokens,
  so the depth that keeps the rate at one is geometry-dependent and must be
  found per instance (a legality in D-20's sense: "the grid's links hold the
  skew between a PE's inputs");
* **what the version says at DIM 2 (depth 4)**: ``mxu`` push->valid 20
  (11 + 9), pop->next valid 9, ``acc`` lag 1 (so ``mxu_stream_engine``'s
  commit/accept assertion holds, finding 4). ``gen_isa_delta.py`` on these
  (``logs/versions_allo-mxu_dim2q4.json``) emits the capacities and puts
  the two latencies under ``unresolved`` with ``would_be`` values
  (``result_latency.vmatpush`` 23 = 20 + 4 - 1, ``issue_interval.vmatpop``
  9) **because the profile is 16 lanes and the measurement is DIM 2**: a
  lag is a property of the grid's depth at a geometry, nothing extrapolates
  it, and the generator refuses to (the seam's "pin to a measurement" rule).
  The DIM-16 number is the DIM-16 build's.
* option (a), a cycle-locked grid (A2, Wire links, H8): the PE's token lag
  would be zero by construction (a ``Wire`` is the register it carries,
  D-13), the front and back keep their II 1, and push->valid would be the
  token distance plus the front's and back's own I/O latency (1 + 1 at
  II 1), i.e. within a few cycles of ``mxu.sv``'s 82 -- a *declared*
  version that differs by a constant. What it takes: the ``Wire[T, comb]``
  link for ``lhsx``/``wx``/``px`` (D-13, in the emitter; track C's H8 found
  it "expected blocked" and did not run it), a form of ``pe_unit`` whose
  east/south outputs are registers written at the edge rather than
  ``Push``\ es (the ``bits`` PE row form is that, cycle-equal at L 1), and
  the composition emitted as one SystemC module with one clock domain for
  the 256 PEs -- the SystemC emitter's ``Wire`` path has been run on single
  kernels (``u2_comb_wire_impl_2026-10-02.rst``), never on a D x D grid.
  Estimate: the form is a day; whether Catapult schedules a 256-PE
  Wire-linked module at II 1 in 3.33 ns is the open question, and the
  DIM-16 build time above says the answer costs hours per attempt.

3. Findings
===========

1. **Per-cycle ports exist only for a packed form.** As landed, M1's lane
   arrays become RAM pins (C1); ``forms/mxu_wide.py`` (``UInt(16 D)``,
   ``UInt(16 D SUB)``) is the first MXU form whose region ports are all
   Connections streams, and it made every kernel II 1 -- but only with the
   back's rings **and the front's skew lines partitioned** (``--partition``
   per array; ``skd`` at DIM 4 is 52 x 16 and Catapult maps it to a RAM, at
   DIM 16 it is 976 x 16). Nothing in the source declares that these are
   registers; it is a schedule on the unit (track C C5, D-12's
   ``Memory(ports=)`` would declare it).
2. **Token time vs cycle time is a rate, not a constant** (H7 again, now at
   the integration boundary): the wrapper's lag counters are the
   measurement, and a wrapper is honest only if it flags a growing lag
   (``lockstep_violation_o``) instead of buffering it.
3. **The pop mask is a version quantity.** Even at one token per cycle the
   output side lags the input side by the grid depth, so a pop's effect is
   visible ``lag`` cycles later; the wrapper masks ``output_valid_o`` until
   the post-pop token arrives (``vld_tokens > last_pop_token``) and that
   mask IS ``issue_interval.vmatpop`` for the version. Without it MiniTPU's
   pop engine would write a stale head on a second ``vmatpop``.
4. **``input_accept_o`` lags too** (``acc_lag``): ``mxu_stream_engine.sv:99``
   asserts ``!commit_q || accept`` the cycle after a load's last row; at a
   stationary lag ``L`` that holds only if ``L < DIM`` (the accept stays
   high over the load's rows). A vpu-level run needs either ``L < DIM`` or
   the assertion declared per version.
5. **Which MiniTPU tbs can judge a slower MXU** (plan section 2):
   ``tb_mxu_single_port`` as is; the four ``tb_matrix_*`` tbs read
   ``tb/isa_latency_pkg.sv``, generated from ``docs/isa_latency.json`` --
   so a ``versions``-resolving ``gen_isa_doc.py`` (their master) regenerates
   the package for ``allo-mxu`` and ``tb_matrix_command_ii`` then *measures*
   the version's numbers on the RTL: the seam closes the loop, once there
   is a stationary number to put in it.
6. **Build-time scaling** (the DIM 16 question): emission > 11 min and not
   done at DIM 16; Catapult 4 min (DIM 2) -> 12 min (DIM 4) for 4 x the
   PEs. A DIM-16 Allo MXU through Catapult is an overnight build. The
   ladder's own answer is D-7: unit by unit, with the DIM-2/4 instance as
   the oracle's shape (P-4 "small geometry decides").

4. Interface map and the tbs (from the plan)
============================================

See ``minitpu_rtl_mxu_2026-10-08/plan_2026-10-08.rst`` sections 1-2 for the
port-by-port map (``mxu.sv:5-30`` vs the Catapult region's
``<v>_dat/_vld/_rdy`` ports; what ``mxu_allo.sv`` does: FIFO per In port,
always-ready Out ports, the pop mask, ``rst_ni`` to both the core reset and
the ``rst`` token, the ``DIM``/``NUM_SUBLANES``/``OUTPUT_FIFO_DEPTH``
elaboration checks) and the tb analysis. The generated wrapper for each
build is in ``rtl/``.

Not done
========

* DIM 16 through Catapult (finding 6); the A2 Wire-locked grid (option a).
* A vpu-level tb against the Allo MXU (needs a stationary lag and the
  ``versions``-aware package generator from ``minitpu-comp``'s master).
* The weight-switch span at the ``mxu`` boundary on the Allo core
  (``tb_matrix_weight_pipelining``'s scenario; ``unresolved`` in the delta).
* DC area of ``mxu_wide``; the ``bits`` PE and the ``vpu_fifo`` forms as RTL
  IP inside ``mxu.sv`` (D-21's other direction, ``examples/minitpu-rtl/``).

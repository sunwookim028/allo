U4 findings matrix (README D-9)
===============================

One row per unit, one column per tool, in ``u1_matrix.rst``'s format. Cells:
**match**, **finding** (with its class: bug, missing abstraction, workaround,
semantic mismatch), **blocked**, **refused** (a backend refusing what it cannot
honour, D-1), **open** (not yet tried; the track that owns it), or **n/a**.
Started 2026-10-08 on zhang-21; MiniTPU at ``b3ba0a4d``; branch ``u4-phase0``
from ``u1-pilot``. **Wave 1 filled 2026-10-08** (branch ``u4-wave1-int`` from
``u1-pilot`` at ``bf4e3302``, tracks A, B, C merged): the Allo columns from the
three track records, every verdict re-run once on the merged tree (section
"Wave 1 on the merged tree"); the Catapult, RTLGen and AMC columns are
**open (track D/E)** at wave 1; **track D filled** (Catapult cells,
``u4_track_d_2026-10-08.rst`` section 8) -- not run in wave 1. **Track E filled** (RTLGen, AMC cells;
``u4_track_e_2026-10-08.rst`` section 6) and the closed ``sequencer`` row added
(``u4_seqloop_2026-10-08.rst`` section 8) on 2026-10-08.

The oracle column is Phase 0 (``u4_phase0_2026-10-08.rst``): each RTL unit
against its Python reference on every defined slot, at its declared latency,
with MiniTPU's own tbs as the second check. Every Allo column is held to
that oracle with ``check.py``. Plan, tracks, provisional defaults and the
owner's questions: ``u4_plan_2026-10-08.rst``. The tracks ran with the owner
away: every call in them is provisional (D-9); D-23 and D-24 settled O1 and
the timing order before wave 1 started.

Cells: Sim = Allo simulator (order only); csim = SystemC csim (per
iteration, time unchecked; P-2). Every finding cell names its class and its
id in the track record: ``T-n`` track A, ``F-Bn`` track B, ``Cn`` track C,
``Mn``/``Rn`` the RTLModule records (``minitpu_rtl_m1``/``m2b``).

The calendar (measured, ``calendar.log``)
-----------------------------------------

=================  =====  =====  ==========  ============================================
class              L      W      first RAW   held to
=================  =====  =====  ==========  ============================================
vld                4      6      7           asm ``W_VLD``, json ``WB_W_VLD``, pkg L+2
ALU                3      5      6           ``W_ALU`` / ``WB_W_ALU``
SFU                5      7      8           ``W_SFU`` / ``WB_W_SFU``
vredsum/vredmax    13     15     16          ``W_REDUCE`` / ``WB_W_REDUCE``
vlanered           9      11     12          ``W_LANE_REDUCE`` / ``WB_W_LANE_REDUCE``
vtxout             1      3      4           ``W_TRANSPOSE`` / ``WB_W_TXOUT``
vmatpop            1      3      4           ``W_MPOP_FIRST`` (result waiting)
=================  =====  =====  ==========  ============================================

All **CALENDAR-MATCH**: ``W = L + VPU_WB_STAGES``, the seam agreed with
``minitpu-comp``.

``fetch`` (IRAM + fetch queue)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** ``fq_a8``/``fq``/``iram``: 126,827 / 112,828 / 24,982 defined slots; request->valid 2, flush->target 3, pop 1; ``tb_fetch_queue_shift`` 13,999/13,999 replayed, PASS
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``f1`` (body arrays) on ``iram`` 25,026, ``fq`` 112,864, ``fq_a8`` 126,871; ``f1_d12`` (IRAM a D-12 ``Memory``, ``reset=False``) ``iram`` 25,026. F2 probe (fetch queue as Streams, H6): **finding** T-4 (missing abstraction: no flushable stream), T-1 (bug: >128-bit Stream corrupts the heap), T-5 (semantic: a 1-bit epoch is wrong)
     - ``u4_track_a_2026-10-08.rst`` sections 2, 4; ``f1_d12``: **finding** T-2 (semantic mismatch: a port's ``L`` on Stream links is ``L - 1`` iterations)
   * - SystemC csim
     - **match** (same instances and counts); F2 probe as the simulator (``drain`` match, ``epoch`` 24,348/24,362, one bundle lost when the queue was full at a flush)
     - ``u4_track_a_2026-10-08.rst`` section 4
   * - Catapult RTL + DC
     - **match** per cycle ``fq`` (3.33, 2.0; offset 2, II 1, 0 stalls) and ``iram`` with a 256-row register IRAM (3.33); **blocked** (D-5) the 4,096-row IRAM: SCHD-30 on ``ccs_ram_sync_1R1W`` (pinned and free), and the ``registers`` server at 4,096 rows exceeds 15.7 GB in ``architect`` (D-4). ``f1_d12``: T-2 **confirmed** (t + L on Wire links; the landed body is 2,010 slots late, the amended body matches at shift 4 bar 31 slots), **finding** D-3 (missing abstraction: the comb pins of D-13 refuse the Stream composition), **finding** D-6 (bug: silent wrong answer, write-first IRAM). DC: fq 6,340 um2 vs MiniTPU 3,607 (3.33 ns); IRAM 256 rows 216,231 as registers, 191,510 as the D-12 server composite
     - ``u4_track_d_2026-10-08.rst`` sections 2, 4, 5.3, 5.4, 6, 7
   * - RTLGen
     - n/a (as the matrix)
     - plan section 4 (H3: --)
   * - AMC
     - **match** (IRAM as an ``amc.memory`` ``w(1)`` + ``r(1)``: lowers to an unreset array with a registered read, latency 1 as declared, old word on a same-cycle collision; Verilator 3/3). Frontend: ports inferred, never declared (missing abstraction, d12's gap)
     - ``u4_track_e_2026-10-08.rst`` section 6

``loop_ctrl`` (loop control + loop buffer)
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 527,082 slots; begin->iv 1, branch comb, refetch +2/iteration over ``LB_CAP`` 24; ``tb_bundle_loop``, ``tb_loop_begin_r``, ``tb_loop_buffer_cap`` replayed and PASS. MiniTPU **FYI**: stack overflow/underflow silent in hardware
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``l1`` and ``l1_d12`` (loop buffer a D-12 ``Memory``, two units) 527,446/527,446; D-20 ``ControlGeometry``: 12 derived numbers = Phase 0 and MiniTPU's bookings, 7 wrong declarations **refused**
     - ``u4_track_a_2026-10-08.rst`` sections 2, 3 (H3, H8)
   * - SystemC csim
     - **match** ``l1``, ``l1_d12`` 527,446/527,446
     - 
   * - Catapult RTL + DC
     - **match** per cycle ``l1`` (3.33, 2.0; offset 2, 3 at 2.0, II 1, 0 stalls). ``l1_d12``: not run (track D: the two-unit form meets D-3 as ``f1_d12`` did; the T-2 question was answered there). DC 27,197 vs MiniTPU 21,186 um2 (3.33 ns), 34,846 vs 21,452 (2.0 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6, 9
   * - RTLGen
     - n/a
     - 
   * - AMC
     - **match** (loop buffer ``w(1)`` + ``r(0)``: combinational read; Verilator 3/3). **finding** E-A7 (bug, #126 class: frontend pipelined kernel infeasible -> wrong values); latency 0 not expressible from the frontend (missing abstraction)
     - ``u4_track_e_2026-10-08.rst`` section 6

``seq_decoder`` (decoder)
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 25,416 bundles; ``asm.py`` agrees on every field on 5,255 real bundles; on random words its reads are a safe superset, and S ops 5-7 / IMM ownership disagree (never emitted); ``tb_bundle_encoding`` PASS
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``c1`` 25,458/25,458 (per-slot functions ``decode_v/m/x/d/c/l/s``; no 233-bit port)
     - ``u4_track_a_2026-10-08.rst`` (H1); called by the issue kernel since wave 1 (``seq_issue``, ``vpu_cmd``)
   * - SystemC csim
     - **match** 25,458/25,458
     - 
   * - Catapult RTL + DC
     - **match** per cycle (3.33, 2.0; offset 2, II 1, 0 stalls). **finding** D-1 (missing abstraction: the 233-bit lane boundary needs a track-D form). DC 2,615 vs MiniTPU 255 um2 (3.33 ns): the Connections boundary dominates
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6, 7
   * - RTLGen
     - **match** 25,458/25,458 as written (inlined II=2 on lane ports, II=1 on 1-D ports; called 13N+1). **finding** E-R2 (missing abstraction: lane-array port costs II), E-R3 (workaround: no inline directive)
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **match** 25,458/25,458 after U1's workarounds (II=1, N+2); as written **refused** (A2). **finding** E-A4/E-F1 (typed constants)
     - ``u4_track_e_2026-10-08.rst`` section 6; combinational unit, run because the track listed it

``agu_resolve`` (X-slot address)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 29,216 (every level x shift x valid); ``tb_agu_resolve_width`` 9,216/9,216
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``c1`` 38,432/38,432
     - ``u4_track_a_2026-10-08.rst``; called by the issue kernel since wave 1
   * - SystemC csim
     - **match** 38,432/38,432
     - 
   * - Catapult RTL + DC
     - **match** per cycle (3.33, 2.0; offset 2, II 1, 0 stalls). **finding** D-1 (missing abstraction: lane-array boundary). DC 1,310 vs MiniTPU 209 um2 (3.33 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6
   * - RTLGen
     - **match** 38,432/38,432 (inlined II=1 N+2; called 15N+1)
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **match** 38,432/38,432 inlined (II=1); called **blocked** (A4); E-A5 (testbench, chunks of 4,096)
     - ``u4_track_e_2026-10-08.rst`` section 6

``scalar_agu`` (scalar AGU)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** at ``S_LAT`` 1/2/3; write->visible = ``S_LAT``, no bypass; ``tb_bundle_scalar_agu`` A, B 130/130
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``s1`` at ``S_LAT`` 1/2/3 (200,885 / 201,065 / 200,755), one body; ``S_LAT`` a D-20 booking, refused when built 3 while booked 2
     - ``u4_track_a_2026-10-08.rst`` (H2)
   * - SystemC csim
     - **match** at 1/2/3
     - 
   * - Catapult RTL + DC
     - **match** per cycle at ``S_LAT`` 1 / 2 / 3 (3.33; 2 also at 2.0; offset 2, II 1, 0 stalls), write -> visible = ``S_LAT`` rows (H2). **finding** D-7 (semantic mismatch: ``check_booking`` compares a visibility booking with the manifest's I/O latency 1). DC ``S_LAT`` 2: 9,595 vs MiniTPU 4,576 um2 (3.33 ns), 11,420 vs 5,437 (2.0 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 5.2, 6, 7
   * - RTLGen
     - **finding** E-R1 (bug, silent: 152,372/201,065 as written); **match** 201,065/201,065 per cycle at ``S_LAT`` 2 with E-R1 avoided (``written_e1``, ``regs``, ``regs_split`` II=1 N+4, 249 MHz est.). ``S_LAT`` 1 and 3: **not run** (track E: the pipe is data, ``S_LAT`` only changes the generated depth; P-E5)
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **blocked** E-A6 (crash; ``llvm`` 201,065/201,065). ``S_LAT`` 1 and 3: **not run** (track E: P-E5)
     - ``u4_track_e_2026-10-08.rst`` section 6

``dma_desc_adapter`` (descriptor adapter)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 180,009; start->valid 1, accept->done comb; built with ``SYNTHESIS`` (its check reaches a sibling)
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``bits`` 180,495/180,495 (alone); inside the issue kernel (A1 in ``seq_issue``) **match** as part of 1,042,595. **finding** C3 (workaround: closure-named slice bounds widen to ``UInt(32)``, replaced by a shift)
     - ``u4_track_c_2026-10-08.rst`` section 1; ``u4_track_b_2026-10-08.rst`` section 2
   * - SystemC csim
     - **match** ``bits``; A1 in ``seq_issue`` **match**
     - 
   * - Catapult RTL + DC
     - **match** per cycle (3.33, 2.0; offset 2, II 1, 0 stalls). DC 2,699 vs MiniTPU 664 um2 (3.33 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6
   * - RTLGen
     - **match** 180,495/180,495 per cycle (II=9 on the lane ports, II=1 on 1-D ports) -- the matrix had n/a; run because the task listed it
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **match** 180,495/180,495 per cycle after workarounds (A2, A-D3); as written **refused** (A2)
     - ``u4_track_e_2026-10-08.rst`` section 6

``vpu_adapter`` (``vpu_ctrl_t`` producer)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 24,866; nine valids gated by issue, every other field ungated payload; ``tb_bundle_vpu_adapter`` 34/34
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``slots`` (D-23's three commands ``adapt_v/x/m``) 24,866/24,866; ``slots_gated`` (D-23's allowed deviation): UNIT-DIFF 966/24,866 by declaration, contract **match** 318,113/318,113. **finding** T-6 (semantic mismatch, documentation: README D-23 swaps the X and M names)
     - ``u4_track_a_2026-10-08.rst`` sections 2, 5; called by the issue kernel since wave 1
   * - SystemC csim
     - **match** ``slots``; gated contract **match** 318,113/318,113
     - 
   * - Catapult RTL + DC
     - **match** per cycle ``slots`` (3.33, 2.0; offset 2, II 1, 0 stalls). ``slots_gated``: not run (track D: budget; the contract-judged variant). DC 1,265 vs MiniTPU 72 um2 (3.33 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6, 9
   * - RTLGen
     - **match** ``slots`` 24,866/24,866 (II=2 lanes / II=1 split / called 9N+1); ``slots_gated`` **CONTRACT-MATCH** 318,113/318,113
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **match** ``slots`` 24,866/24,866 after workarounds; as written **refused** (A2)
     - ``u4_track_e_2026-10-08.rst`` section 6

``sequencer`` / ``seq_issue`` (I1 + A1: bundle issue)
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 1,049,131 slots on asm-scheduled programs, both shipped images, three tbs replayed; start->issue 3, II 1, ``delay=N`` N+1, desc round trip 2; every sim-only assertion predicted. MiniTPU **FYI**: F-W2 vtxout unbooked, F-W3 lost release, F-W4 back-edge WAW past ``schedule()``
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``locked`` 1,042,595/1,042,595 on 52,236 cycles, 15 traces, 5,447 issues. Since wave 1 it calls track A's ``decode_*``, ``resolve``, ``adapt_*``; F1's head, L1's ``iv_by_level`` and S1's read ports are still side columns (open-loop replay of Phase 0's references; the closed loop is a composition, not a function swap). **finding** F-B9 (front end: a named slice bound defaults to ``UInt(32)``, warned)
     - ``u4_track_b_2026-10-08.rst`` sections 1, 2 (H4); wave-1 re-run below
   * - SystemC csim
     - **match** 1,042,595/1,042,595; issue cycle for cycle with the RTL (stamped csim, D-23 section 4)
     - 
   * - Catapult RTL + DC
     - **match** per cycle (3.33, 2.0; offset 2, L 1 on all 19 outputs, II 1, 0 stalls), every issue in the RTL's cycle (H4). DC ``seq_issue`` 5,084 um2 (3.33 ns), 5,185 (2.0); no MiniTPU module of that scope
     - ``u4_track_d_2026-10-08.rst`` sections 2, 5.1, 6
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

``sequencer`` (the closed loop: F1 + L1 + S1 + I1/A1 + C1, D-23 commands)
------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** (Phase 0 ``sequencer`` at ``u4_seq_loop.sv``: slots + ``sreg_o``)
     - ``u4_seqloop_2026-10-08.rst`` section 8
   * - Allo simulator
     - **match** 1,286,132/1,286,132 (programs, both images, 3 tbs). **finding** S-2 (bug: the D-12 server serves its ports in declaration order with blocking gets, so the loop buffer declared ``(cap w, replay r)`` deadlocks the simulator with no diagnostic; repro ``tests/limits/new_d12_server_port_order_deadlock.py``; workaround: replay port first), S-1 (missing abstraction: no same-cycle link kind, every exchange is a Stream), S-5 (workaround: literal slice bounds, D-17), S-6 (workaround: one ``rst_n`` boundary copy per unit)
     - ``u4_seqloop_2026-10-08.rst`` sections 6, 8; template ``template/sequencer.py``
   * - SystemC csim
     - **match** 1,286,132/1,286,132; D-23: 0 stall cycles, 5,447/5,447 issues at the RTL cycle at QD 1/2/4 (29 csim cycles per RTL cycle, S-1). **finding** S-4 (semantic mismatch: a data-dependent 2-D boundary read under an ``if`` costs 2 csim cycles, fixed by an unconditional read), S-3 (missing abstraction: the D-12 memories' Wire lowering fails D-13's cone rule for all four owners, so csim runs the Stream lowering), S-7 (deviation: fetch-queue entries are reset arrays, C2)
     - ``u4_seqloop_2026-10-08.rst`` sections 6, 8
   * - Catapult RTL + DC
     - **blocked** (S-3, D-3): the D-12 memories' Wire lowering fails D-13's cone rule and the exchanges are not wires (S-1), so the closed loop was not built for Catapult; its parts are in the rows ``fetch``, ``loop_ctrl``, ``scalar_agu``, ``sequencer`` / ``seq_issue``. DC of the four units summed: 48,215 vs MiniTPU ``sequencer`` (IRAM cut to 2 rows) 28,783 um2 (3.33 ns), 57,791 vs 31,072 (2.0 ns)
     - ``u4_track_d_2026-10-08.rst`` section 6; needs the exchanges as wires (S-1)
   * - RTLGen
     - n/a (composition)
     - 
   * - AMC
     - n/a (composition)
     - 

``vpu_cmd`` (D-23: the command boundary as three Streams + four resources)
--------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** (``sequencer`` traces through ``u4_seq_cmd.sv``; payload gating per ``vpu_ctrl_gating.log``)
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``streams`` (depth 2), ``streams_d1``, ``streams_d4``: 888,217/888,217 (154,378 payload slots masked by D-23's gate). The four resources compose; three wrong owners **refused** at composition, naming the port (H12, H7)
     - ``u4_track_b_2026-10-08.rst`` sections 2, 4
   * - SystemC csim
     - **match** (all three depths); rate: depth >= 2 stall-free, issue at the RTL's cycle 5,447/5,447; depth 1 **finding** F-B3 (488 stall cycles: the depth is a legality, >= 2)
     - ``u4_track_b_2026-10-08.rst`` section 4; re-measured on the merged tree (below)
   * - Catapult RTL + DC
     - **match** per cycle; **D-23 confirmed** on Catapult RTL at depth 2 and 4 (0 stalls, 5,447/5,447 issues on time, 888,217/888,217 values; 3.33 and 2.0 ns); depth 1 **finding** F-B3 (808 stall cycles, 2/5,447 on time). F-B10 resolved (the unrolled harness pad loop). DC depth 2: 8,771 um2 (3.33 ns), 8,873 (2.0)
     - ``u4_track_d_2026-10-08.rst`` sections 3, 6
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

``vpu_writeback`` / ``vpu_wb`` (W1: the write-port calendar)
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 121,000 slots; ``W = L + 2`` for all seven classes, three ways, = asm = ``isa_latency.json`` = ``sequencer_pkg``; 586-case pair sweep: RTL merges = asm refusals (15); payload ignored outside valid (32/32 VREGs, 260/260 beats). MiniTPU **FYI**: F-W1 ``$onehot0`` blind to vredsum + vlanered
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``locked`` and ``units`` (W1 the one owner of ``vreg.w``) 121,075/121,075; a second writer **refused** (H7). **finding** F-B6 (missing abstraction: the one-claim-per-cycle obligation cannot be declared on converging channels; discharged by the stress traces)
     - ``u4_track_b_2026-10-08.rst`` sections 2, 6
   * - SystemC csim
     - **match** ``locked``, ``units`` 121,075/121,075
     - 
   * - Catapult RTL + DC
     - **match** per cycle ``locked`` and ``units`` (3.33, 2.0; ``locked`` offset 2, L 1, ``units`` offset 4, L 3; II 1, 0 stalls). **finding** D-2 (workaround: ``s.unroll`` by name reaches only the first loop of that name). DC ``locked`` 4,696 um2 (3.33 ns), ``units`` 6,665; no MiniTPU module of that scope. W2 (H13): not run (track D: no variant landed)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6, 7, 9
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

``vpu_wb:selftimed`` (D-24 probe: self-timed write port)
--------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - -- (judged against the locked calendar)
     - 
   * - Allo simulator
     - **finding** F-B4 (bug: the polling merge finishes while class units are blocked on full streams; the simulator waits forever, no diagnostic; repro ``tests/limits/new_sim_unfinished_producer_hang.py``)
     - ``u4_track_b_2026-10-08.rst`` sections 2, 5
   * - SystemC csim
     - **semantic mismatch** by design (D-24): UNIT-DIFF 94,493/121,075; every claim at issue + 5 (W' = 7 every class); 77 RAW hazards on 11 legal programs. F-B7: W' is a property of the build
     - ``u4_track_b_2026-10-08.rst`` section 5
   * - Catapult RTL + DC
     - **finding** D-8 (the ``empty()``-polling merge stalls on the RTL after 33 claims, no verdict). DC 4,137 um2 vs ``locked`` 4,696 (3.33 ns). Manifests for the N=64 tree, VMEM's compute port and the MXU pop: not run (track D: no variant landed)
     - ``u4_track_d_2026-10-08.rst`` sections 5.7, 5.8, 6, 9
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

``Calendar`` (template) + ``gen_isa_delta``
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** (``calendar.log``)
     - 
   * - Allo simulator
     - n/a (composition time): **CALENDAR-MATCH** 33/33; 5/5 wrong declarations **refused**; ``allo-locked`` DELTA-MATCH ``v1`` 8/8. F-B1 (pin naming) **resolved**: ``b3ba0a4d`` is ``v1``
     - ``u4_track_b_2026-10-08.rst`` sections 3, 10
   * - SystemC csim
     - n/a
     - 
   * - Catapult RTL + DC
     - n/a
     - 
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

``dma_addr_gen`` (DMA address)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 200,700/200,700 (32-bit wrap)
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``bits`` (C) and ``c1`` (A) 200,700/200,700
     - ``u4_track_c_2026-10-08.rst``, ``u4_track_a_2026-10-08.rst``
   * - SystemC csim
     - **match** both, 200,700/200,700
     - 
   * - Catapult RTL + DC
     - **match** per cycle ``bits`` (3.33, 2.0) and ``c1`` (3.33); 1.000 cyc/vector, L 1. DC 2,516 vs MiniTPU 1,407 um2 (3.33 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 6
   * - RTLGen
     - **match** 200,700/200,700, ``bits`` and ``c1``, inlined (II=1 N+4) and called (7N+1)
     - ``u4_track_e_2026-10-08.rst`` section 6
   * - AMC
     - **match** 200,700/200,700 inlined (II=1 N+6); called **blocked** (A4)
     - ``u4_track_e_2026-10-08.rst`` section 6

``dma`` (D1: DMA engine)
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** per cycle at OUTSTANDING 1/2/4 (2.28 M slots), ``tb_dma_bandwidth`` 174,508/174,508; accept comb, ->request 2, last beat->done 2, VMEM read 2; bridge/loader/copy/CDMA/launch tbs PASS. MiniTPU **FYI**: reuse without clear hangs; >2^29 address truncated; stride-0 store accepted
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator
     - **match** ``bits``, ``bits_reset``, ``streams``, ``streams_reset`` on ``core`` 805,652, ``o1`` 764,284, ``o4`` 317,520, ``bw`` 394,464. **finding** C1 (bug: a local named ``done`` breaks the SystemC build; the simulator runs it; worked around as ``cdone``), C5 (workaround: 8 x 32-bit lanes per beat)
     - ``u4_track_c_2026-10-08.rst`` sections 1, 2, 5 (H9)
   * - SystemC csim
     - **match** ``*_reset`` on all four instances; **refused** ``bits``, ``streams`` (C2: ``@ Stateful(reset=False)`` lowered only in all-``Wire`` kernels; missing abstraction). C4 (workaround: the refusal's text blames ``wrap_io``)
     - 
   * - Catapult RTL + DC
     - **match** per cycle ``bits_reset`` core/o1/o4/bw (3.33; offset 2, II 1, 0 stalls; landing FIFO partitioned, D-5); **blocked** (D-5) at 2.0 ns (SCHD-30: a one-cycle recurrence does not fit) and with the landing FIFO unpartitioned (SCHD-30 on ``ccs_ram_sync_1R1W``); ``streams_reset`` **semantic mismatch** per cycle (1.333 cyc/row, 13,238 stalls: **finding** D-9, depth-2 channels sustain 2 tokens per 3 cycles); ``bits`` / ``streams`` **blocked** (C2). DC core 35,620 vs MiniTPU 17,645 um2 (3.33 ns)
     - ``u4_track_d_2026-10-08.rst`` sections 2, 5.5, 6, 7
   * - RTLGen
     - **not run** (track E: H9, optional, left out of the list)
     - ``u4_track_e_2026-10-08.rst`` section 9
   * - AMC
     - n/a
     - 

``dma_vmem`` (D2: the DMA on the VMEM DMA port, D-12 ``rw`` latency 2)
----------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (track C)
     - **match** new wrapper ``u4_dma_vmem.sv`` against Phase 0's ``DmaModel`` closed on ``VmemDmaSide``: 138,855 / 128,260 / 88,216 (core/o1/o4); mutation checked (284 and 896 differing slots)
     - ``u4_track_c_2026-10-08.rst`` section 1
   * - Allo simulator
     - **match** ``d12``, ``d12_reset``, ``d12_reset_r256`` on core/o1/o4. **finding** C6 (missing abstraction: P-3's one latency checked by hand in the architecture), C8 (semantic mismatch: the server's read token is post-edge; a pre-sampling owner needs a ``hold`` register)
     - ``u4_track_c_2026-10-08.rst`` sections 2, 5
   * - SystemC csim
     - ``d12`` **refused** (C2); ``d12_reset`` **finding** C7 (bug: a 512 KB local array on the SC_THREAD stack segfaults); ``d12_reset_r256`` **match**
     - 
   * - Catapult RTL + DC
     - **blocked** (D-5; H11 confirmed): SCHD-30 in ``vmem_mem``, the two-``rw``-port server beside the compute port at II=1. No RTL, so no DC
     - ``u4_track_d_2026-10-08.rst`` sections 2, 5.6
   * - RTLGen
     - n/a
     - 
   * - AMC
     - **refused** E-A3 (bug: ``rw(3,1)`` + ``rw(2,1)`` not lowered); workarounds lower (2R2W split: latencies 3/2 verified; ``rw(1,1)`` x 2: collision silently port 0 -- semantic mismatch with ``collision="obligation"``)
     - ``u4_track_e_2026-10-08.rst`` section 6

``dma_selftimed`` (D2 probe, plan's self-timed form, H10)
---------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle
     - contract: the request stream equal to ``dma.sv``'s on the same descriptors; device memory equal to a sequential reference
     - ``u4_track_c_2026-10-08.rst`` section 6
   * - Allo simulator
     - **CONTRACT-MATCH** 9/9 (core/o1/o4 x 3 programs). **finding** C9 (semantic mismatch: no peek, the credit frees ``len`` beats early), C10 (semantic mismatch: the D-12 server's pipe advances only with accesses; store-then-load deadlocked; a flush access works around it), C11 (missing abstraction: no reliable ``try_get``, so no per-cycle arbitration)
     - 
   * - SystemC csim
     - **CONTRACT-MATCH** 9/9
     - 
   * - Catapult RTL + DC
     - not run (track D: budget; a probe, contract-judged; the 2.0 ns core refusal applies to the same body)
     - ``u4_track_d_2026-10-08.rst`` section 9
   * - RTLGen
     - n/a
     - 
   * - AMC
     - n/a
     - 

Wave 1 on the merged tree
-------------------------

``u4-wave1-int`` = ``u1-pilot`` ``bf4e3302`` (tracks A, B, C and Phase 0
merged) + the C1 stub swap (``1a0d9e52``), its own bindings. Every verdict
line of the three track records re-run once (``u4_wave1_int_2026-10-08/``:
``scripts/run_ac.sh``, ``logs/``, ``logs/verdict_compare.txt``) and compared
with the record's log, run times stripped:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - record
     - re-run on the merged tree
     - against the record
   * - A: ``dma_addr_gen``, ``agu_resolve``, ``seq_decoder``, ``vpu_adapter``
       (+ gated contract x2), ``scalar_agu`` lat1/2/3, ``fetch`` iram/fq/fq_a8/
       iram ``f1_d12``, ``loop_ctrl``, ``control_geometry``
     - every line UNIT-MATCH / CONTRACT-MATCH / DERIVED as recorded; 7 refusals
     - **equal** (15/15 logs; ``dma_addr_gen`` now also prints C's ``bits``
       lines beside A's ``c1``: the union, both match)
   * - B: ``seq_issue``, ``vpu_cmd`` x3 depths, ``vpu_wb`` locked/units/selftimed
     - UNIT-MATCH everywhere it was; ``selftimed`` csim UNIT-DIFF 94,493 (by
       design); ``selftimed`` simulator not re-run (F-B4 hangs it)
     - **equal** (3/3), **after** the stub swap
   * - B: ``d23_rate.py`` (csim, stamped)
     - depth 2 and 4: 0 stall cycles, 5,447/5,447 issues at the RTL cycle;
       depth 1: 488 stall cycles
     - **equal**
   * - C: ``dma_desc_adapter``, ``dma`` core/o1/o4/bw, ``dma_vmem`` core/o1/o4,
       ``dma_params``, ``dma_selftimed``
     - as recorded: ``*_reset`` match, C2 refusals, C7 segfault, r256 match,
       18/18 contracts, 8 DERIVED-OK
     - **equal** (10/10)
   * - C: ``tests/limits/new_systemc_reserved_local_names.py`` (C1),
       ``new_systemc_local_array_stack.py`` (C7)
     - REPRODUCES both
     - as the record states (no log kept there)

**No verdict changed on the merged tree.** The stub swap (step 1 of wave 1)
replaced track B's Python ``decode`` and its own ``resolve`` with track A's
``decode_v/m/x/d/c``, ``resolve`` and ``adapt_v/x/m`` inside the issue
kernels of ``seq_issue`` and ``vpu_cmd``; both matched the first time. F1's
head, L1's ``iv_by_level`` and S1's read ports remain open-loop side columns:
those are track A's stateful kernels, and closing the loop is a four-unit
composition with several same-cycle exchanges, not a function swap (more
than a day; for the owner below).

D-23 and D-24 (from ``u4_track_b_2026-10-08.rst``)
--------------------------------------------------

**D-23 (the command boundary as three slot Streams + four resources): holds
at depth >= 2.** In stamped csim the sequencer issues cycle for cycle as the
RTL on every Phase 0 program (5,447/5,447, 0 stall cycles, depth 2 and 4;
re-measured on the merged tree); depth 1 stalls 488 cycles (F-B3), so the
depth is a legality, not a default. **Catapult RTL confirms it** (track D
section 3): the issue kernel + three command Streams + three receivers hold
D-23 at depth 2 and 4, 0 stalls, 5,447/5,447 issues at the RTL's cycle at 3.33
and 2.0 ns; depth 1 stalls 808 cycles (F-B3's >= 2 holds there, larger than
csim's 488). (B §4.)

**D-24 (cycle-locked first, self-timed compared):** cycle-locked W1 books
``W = L + 2`` for all seven classes, equal to the RTL (DELTA-MATCH ``v1``
8/8), and MiniTPU's 11 legal programs run unchanged with 0 hazards. The
self-timed form makes W a property of the build (7 for every class in csim,
9-21 from Catapult manifests, 5 of 8 unresolved): 77-165 RAW hazards on the
same legal programs, so it needs a re-schedule, and it hangs the simulator
(F-B4). On Catapult (track D section 5.8) the self-timed ``empty()``-polling
merge stalls on the RTL after 33 claims (D-8), so no RTL verdict; cycle-locked
W1 matches per cycle (``locked``, ``units``). (B §5.)

Triage: the findings, ranked
----------------------------

Order: a silent wrong answer first, then a crash or hang, then a refusal,
then a workaround or missing abstraction. Ids as in the track records.

**Silent wrong answer**

1. **T-1** (bug, simulator; A). A ``Stream`` element wider than 128 bits
   corrupts the heap: the data arrive intact and the process later aborts,
   segfaults or hangs -- in 6 of 14 runs at 129-160 bits; the other 8 ended
   with no symptom. Memory corruption with no reliable signal ranks first. Proposal: the simulator refuses > 128-bit elements
   until the lowering is fixed.
2. **D-6** (bug, D-12 server on Catapult RTL; D). The ``registers`` server of an
   unreset memory (``reset=False``) returns the word being written in the same
   cycle (write-first); the declaration says ``visible=1`` and MiniTPU's IRAM
   returns the old word. 31 IRAM slots differ after a reset, proven against a
   write-first reference. On the Catapult path only: csim and the simulator are
   not affected on these traces. Proposal: the server reads storage in the
   address's own c-step, and a same-cycle read/write test per link kind.
3. **E-R1** (bug, RTLGen frontend; E). Inside a nested ``@kernel``, a narrow
   unsigned parameter compared with a literal whose top bit (at the parameter's
   width) is set is never equal (``op: u3``, ``op == 4``): on RTLGen's ``cpu``
   target and in the RTL alike, no warning. In S1 every ``SSHL`` writes 0:
   48,693 wrong defined slots. Workaround: pass ``op`` as ``u32`` and narrow
   inside. Draft issue: track E section 5.
4. **E-F1** (bug, Allo front end; E). An untyped literal is ``int32``, so ``y:
   UInt(64) = x | (1 << 35)`` leaves bit 35 clear on ``llvm`` and in the
   simulator, no warning (``tests/limits/new_wide_literal_shift.py``). No landed
   U4 unit is affected (slice stores, not literal shifts). Proposal: type a
   literal shift by its annotated target, or refuse a shift amount >= the
   literal's width.
5. **F-B4** (bug, simulator / semantic mismatch, csim; B). A consumer that
   finishes while its producer is blocked on a full stream: the simulator
   never returns, no diagnostic; csim returns normally with the blocked
   producer invisible. Proposal: report "finished with kernel X blocked on
   stream S".
6. **T-2** (semantic mismatch, D-12 lowering; A). A port's ``L`` delivers at
   ``t + L - 1`` on Stream links but "registered link + L-deep pipe" on Wire
   links: one cycle-locked body is off by a register on one of the two, and
   nothing reports it. Owner decision below.
7. **C8** (semantic mismatch, D-12 server; C). The server's read token is
   post-edge; a pre-sampling owner is one cycle off unless it adds a ``hold``
   register that is not hardware.
8. **F-B9 / T-9 / C3** (front end; A, B, C). A slice with non-literal bounds
   defaults to ``UInt(32)`` with only a warning: a wider slice is truncated
   silently (no wrong result yet; every unit now uses literal bounds or
   shifts).
9. **C9** (semantic mismatch, self-timed DMA; C). No peek on a Stream: the
   credit frees ``len`` beats early, OUTSTANDING + 1 in flight against the
   RTL's OUTSTANDING; the contract still matched.
10. **M5** (missing abstraction, RTLModule; ``minitpu_rtl_m1``). ``$readmem``
    paths resolve against the process's cwd, silent when wrong.
11. **T-5** (semantic, the epoch workaround; A). A 1-bit epoch issues stale
    bundles after two close flushes; the committed probe uses the stale count.

**Crash or hang**

12. **S-2** (bug, D-12 server lowering; seqloop). The server serves its ports
    in declaration order with blocking gets; the loop buffer declared ``(cap w,
    replay r)`` deadlocks the simulator with no diagnostic (the RTL's ports have
    no order). Repro ``tests/limits/new_d12_server_port_order_deadlock.py``.
    Proposal: answer every read port before waiting on a write port, or
    ``compose`` refuses a port order its channels make cyclic.
13. **E-A6** (bug, AMC crash; E). S1 in the register discipline aborts the ``amc``
    build (``'loopschedule.await' op operation destroyed but still has uses``)
    on both schedules; reduced to 55 lines (``amc/repro_sagu_await.py``); A-D2 is
    reachable at depth 2. Blocks S1 on AMC.
14. **C7** (bug, SystemC emitter; C). A large kernel-local array lands on the
    SC_THREAD's ~64 KB stack: csim segfaults (VMEM at 4,096 rows).
15. **C10** (semantic mismatch, D-12 server, self-timed; C). The read pipe
    advances only with accesses: store-then-load deadlocked; a flush access
    after every store works around it.
16. **C1** (bug, SystemC emitter; C). A local named ``done`` collides with the
    process's ``done`` port: g++ error (the simulator runs it).
17. **F-B10** (Catapult, budget; B). ``vpu_cmd:streams`` csyn ran away in
    ``architect`` (86 GB, stopped); likely the unrolled harness pad loop.
    Track D's.

**Refusal**

18. **D-5** (refusal, Catapult memory inference; D). A body array read and written in
    one II=1 iteration maps to a single-port RAM model (``ccs_ram_sync_1R1W``) and
    is refused (SCHD-30): blocks the full 4,096-row IRAM, the DMA landing FIFO and
    VMEM's two ``rw`` ports (H11), and ``dma`` at 2.0 ns. Workaround: ``s.partition``
    (+ ``REGISTER_THRESHOLD``) for small arrays, or declare the memory as a D-12
    ``registers`` / ``sram`` (owner decision below).
19. **E-A3** (bug, AMC; E). A ``static rw(R, W)`` port with ``R != W`` is legal at
    the type level but fails ``seq.hlmem`` legalization (only a "failed to
    legalize" message): blocks the VMEM with its latency-3 compute and
    latency-2 DMA ports (H11) on AMC.
20. **C2** (missing abstraction, honest refusal; C). ``@ Stateful(reset=False)``
    is refused in any kernel with a non-Wire port, so P-7's declared form
    has no csim (or Catapult) path in token time. Owner decision below.
21. **F-B2** (missing abstraction, spurious refusal; B). ``compose`` refuses
    ``uint8/16/32/64`` (not in ``FRONTEND_NAMES``); one-line fix.
22. **T-3** (workaround; A). A closed-over Python ``bool`` compiles to an
    ``i32`` ``scf.if`` and fails; two functions instead of a flag.
23. **F-B5** (repro rot; B). ``item18_catapult_try_ops_blocking.py`` is now
    refused by the netlist rules before reaching the backend.

**Workaround / missing abstraction**

24. **D-2** (workaround, schedule; D). ``s.unroll("k:name")`` reaches only the first
    loop of that name, so ``vpu_wb``'s 12 ``k`` loops stayed rolled and the manifest
    was ``unreliable``. Workaround: ``unroll=0`` on every nested ``affine.for``.
    Proposal: ``s.unroll`` of a loop object.
25. **D-9** (workaround, Catapult composites; D). Depth-2 channels between II=1
    kernels that move a token every cycle sustain 2 tokens per 3 cycles (D1 over
    Streams: 1.333 cyc/row); sparse channels (D-23) lose nothing. Workaround:
    depth >= 3 on every-cycle channels, measured per backend.
26. **S-1** (missing abstraction; seqloop). A same-cycle link kind: every one-cycle
    exchange between cycle-locked units must be a Stream (29 csim cycles per RTL
    cycle) because a unit ``Wire`` port is not wired, a region Wire is not
    aligned in csim and a ``comb`` link is refused beside a Stream port.
    D-n candidate below.
27. **T-4 / H6** (A). No stream can be flushed; ``drain`` hides a stretched
    cycle, ``epoch`` costs a bundle when the queue was full. D-n draft below.
28. **C11** (C). No reliable ``try_get`` (item 18, #23): no per-cycle
    arbitration between two blocking request streams.
29. **C6** (C). An owner cannot read its port's declared latency; P-3's one
    number is checked by hand in the architecture builder.
30. **F-B6** (B). A per-cycle claim obligation on converging channels
    cannot be declared (``obligations`` takes multi-write memories only).
31. **F-B3** (B). Command-stream depth is a legality (>= 2 in csim).
32. **M2, M3, M4, M6, R1** (RTLModule; ``minitpu_rtl_m1``/``m2b``): compose
    has no IP declaration category (M2); payloads <= 32 bits, a 256-bit word
    as eight ``MemPort``\ s (M3); fixed transfer counts (M4); no Verilator
    build cache (M6); a dual port only by aliasing one array to two
    ``MemPort``\ s (R1).
33. **C4, C5** (C): the refusal text blames ``wrap_io``; beats as 8 x 32-bit
    lanes.

Recorded, not ranked: T-6 (README D-23 swaps the X/M names: wording for the
owner), T-7, T-8 (by design), F-B1 (resolved: the pin is ``v1``), F-B7 (by
design, D-24), F-B8 (harness rule kept), M7 (by design).

What the owner decides
----------------------

1. **H6: the flushable stream** (track A section 4 D-n draft):
   ``Stream[T, D, flush=True]`` with ``s.flush()`` by the one consumer,
   lowered as a FIFO synchronous clear, an epoch on the channel only for a
   self-timed producer, refusal where a backend cannot build the clear.
   Until then ``drain`` / ``epoch`` stay the recorded workarounds.
2. **T-2: "a port's L means iterations on every link kind"** (deliver at
   ``t + L``; the Wire link's register is one of the ``L`` stages), to be
   checked on Catapult by track D.
3. **C2's lowering**: for a kernel whose ports are Streams, emit unreset
   storage as a module member with no reset action (as reset ``Stateful``
   members, minus the reset); keep the clock-edge ``SC_METHOD`` for all-Wire
   kernels. Until then csim and Catapult run the ``*_reset`` deviation.
4. **Tracks D (Catapult + DC) and E (RTLGen/AMC): approved 2026-10-08; E
   landed (``u4_track_e_2026-10-08.rst``), D running.** (Was: both now, on the landed
   variants?** Every Allo column above is landed and matches; D would start
   with ``seq_issue``/``vpu_cmd`` at depth 2 and 4 (D-23's open Catapult
   half; drop the pad loop, F-B10), ``vpu_wb``, ``dma`` ``bits_reset``,
   ``dma_vmem`` (H11), ``scalar_agu`` (``latency=S_LAT``) and the T-2 check;
   E with C1, S1, D1 and AMC on IRAM / loop buffer.
5. **Closing the sequencer loop** (done: row ``sequencer``, 1,286,132/1,286,132):
   F1, L1 and S1 composed with the issue
   kernel (instead of side columns) -- a wave-2 task of about a day, or
   leave the issue unit held by replay.
6. **T-6**: correct README D-23's slot names (X = memory, M = matrix).
7. **File the six draft issues of track E section 5**
   (``u4_track_e_2026-10-08.rst``; none filed): RTLGen **E-R1**; AMC
   **E-A3**; AMC **E-A6 + A-D2**; AMC minor trio **E-A1/E-A2/E-A5**; a
   comment on AMC **#126** for **E-A7**; Allo **E-F1** upstream (repro
   ``tests/limits/new_wide_literal_shift.py``).
8. **S-1: a declared same-cycle link kind for cycle-locked compositions** as
   a D-n candidate (one put and one get per iteration, required in the cone
   order the composition checks for acyclicity; Stream depth 1 on the
   simulator and csim, a wire on Catapult). Until then every one-cycle
   exchange is a Stream at 29 csim cycles per RTL cycle
   (``u4_seqloop_2026-10-08.rst`` section 6, S-1).

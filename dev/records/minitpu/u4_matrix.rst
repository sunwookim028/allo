U4 findings matrix (README D-9)
===============================

One row per unit, one column per tool, in ``u1_matrix.rst``'s format. Cells:
**match**, **finding** (with its class: bug, missing abstraction, workaround,
semantic mismatch), **blocked**, **refused** (a backend refusing what it cannot
honour, D-1), **open** (not yet tried; the track that owns it), or **n/a**.
Started 2026-10-08 on zhang-21; MiniTPU at ``b3ba0a4d``; branch ``u4-phase0``
from ``u1-pilot``.

The oracle column is Phase 0 (``u4_phase0_2026-10-08.rst``): each RTL unit
against its Python reference on every defined slot, at its declared latency,
with MiniTPU's own tbs as the second check. Every Allo column will be held to
that oracle with ``check.py``. Plan, tracks, provisional defaults and the
owner's questions: ``u4_plan_2026-10-08.rst``. **No Allo column is open for
work until the owner has reviewed the plan** (D-9: expression before coding);
O1 (``vpu_ctrl_t`` as a declared interface) comes first.

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
   * - Allo simulator / SystemC csim
     - **open** (F1 (A), F2 probe: flushable stream (H6))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (L1 (A))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (C1 (A))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (C1 (A))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (S1 (A), ``latency=S_LAT`` pinned)
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (A1 (B))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

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
   * - Allo simulator / SystemC csim
     - **open** (C1 (A); interface Q1 (F))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

``sequencer`` (whole: bundle issue)
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 1,049,131 slots on asm-scheduled programs, both shipped images, three tbs replayed; start->issue 3, II 1, ``delay=N`` N+1, desc round trip 2; every sim-only assertion predicted. MiniTPU **FYI**: F-W2 vtxout unbooked, F-W3 lost release, F-W4 back-edge WAW past ``schedule()``
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator / SystemC csim
     - **open** (I1 (B), I2 probe (H5))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

``vpu_writeback`` (``vpu.sv`` at ``vpu_ctrl_t``: the write-port calendar)
-------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 121,000 slots; ``W = L + 2`` for all seven classes, three ways, = asm = ``isa_latency.json`` = ``sequencer_pkg``; 586-case pair sweep: RTL merges = asm refusals (15); payload ignored outside valid (32/32 VREGs, 260/260 beats). MiniTPU **FYI**: F-W1 ``$onehot0`` blind to vredsum + vlanered
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator / SystemC csim
     - **open** (W1 (B), W2 Option (D); Calendar record (B))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
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
   * - Allo simulator / SystemC csim
     - **open** (C1 (C))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

``dma`` (DMA engine)
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** per cycle at OUTSTANDING 1/2/4 (2.28 M slots), ``tb_dma_bandwidth`` 174,508/174,508; accept comb, ->request 2, last beat->done 2, VMEM read 2; bridge/loader/copy/CDMA/launch tbs PASS. MiniTPU **FYI**: reuse without clear hangs; >2^29 address truncated; stride-0 store accepted
     - ``u4_phase0_2026-10-08.rst``
   * - Allo simulator / SystemC csim
     - **open** (D1 (C), D2 probe (H10), VMEM DMA port (H11, D))
     -
   * - Catapult RTL + DC
     - **open** (track D)
     -
   * - RTLGen / AMC
     - **open** (track E)
     -

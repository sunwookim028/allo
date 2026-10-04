U3 findings matrix (README D-9)
===============================

One row per unit, one column per tool, in ``u1_matrix.rst``'s format. Cells:
**match**, **finding** (with its class: bug, missing abstraction, workaround,
semantic mismatch), **blocked**, **refused** (a backend refusing what it cannot
honour, D-1), **open** (not yet tried; the track that owns it), or **n/a**.
Started 2026-10-04 on zhang-21; MiniTPU at ``b3ba0a4d``; branch ``u3-phase0``
from ``u1-pilot``. The owner is away and has authorised U3 in full: every call
below is *provisional* (D-9) and listed for review.

The oracle column is Phase 0 (``u3_phase0_2026-10-04.rst``): each RTL unit
against its numpy reference on every defined slot, at its declared latency,
with MiniTPU's own tbs as the second check. Every Allo column is held to that
oracle with ``check.py``. Plan, tracks and provisional decisions:
``u3_plan_2026-10-04.rst``.

For the owner's review (on return)
----------------------------------

Provisional decisions in force (plan section 6): **P-1** expressions S1, T1
(+T2), X1, P1 -> P2, A1 (A2 a probe), M1; **P-2** ``mxu`` judged on its
contract trace, PE/array on the cycle model only for cycle-locked RTL; **P-3**
latency reported, pinned only for PE 4 / SFU 5 / tree 13, 9; **P-4** small
geometry for verdicts, shipped once; **P-5** MXU accumulate order bit-exact,
acc24 only through U1's integer units; **P-6** ``compose.unit`` for
parameterised units; **P-7** an engine declares its accumulate order (D-n
draft); **P-8** lane-array ports; **P-9** ``reset=False`` payload; **P-10**
MiniTPU findings recorded, not filed. Open questions O1-O4 (plan section 7).

``sfu``
-------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 462,144/462,144 (every op x every operand, and an op change
       every cycle); latency 5 = ``VPU_SFU_LATENCY``; ``tb_sfu_equiv``,
       ``tb_sfu_math_sweep``, ``tb_sfu_direct_lut`` PASS unchanged
     - ``u3_phase0_2026-10-04.rst``; IEEE deviations all classified (13 causes,
       278,601 vectors, none unexplained). MiniTPU **FYI**: ``vrecip`` top binade
       gives a non-canonical NaN; no subnormal flush (findings 1-2)
   * - Allo simulator / SystemC csim
     - **open** (track A: S1, S2)
     -
   * - Catapult RTL + DC
     - **open** (track C: S2; ROM inference, H2)
     -
   * - RTLGen / AMC
     - **open** (track D)
     -

``xlu_reduction_tree``
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** at N=64 (65,336 defined slots) and N=16 (204,120); latency
       13 / tap 9 (N=64), 9 / 5 (N=16); op tag travels with its wavefront;
       ``tb_reduce_pipe`` 2,370/2,370 and ``tb_xlu_lane_tap`` 184/184 replayed
     - MiniTPU **elaborates and is bit-exact at N=16**, which ``UNITS.md`` §1
       says was never built; its two tree tbs are not geometry-generic (they
       fail at N=16 on hard-coded 16x4 expectations, finding 4)
   * - Allo simulator / SystemC csim
     - **open** (track A: T1, then T2 units with derived legality)
     -
   * - Catapult RTL + DC
     - **open** (track C: H3 depth vs 13, H4 C10 recurrence)
     -
   * - RTLGen / AMC
     - **open** (track D)
     -

``xlu_transpose``
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** 95,779 defined slots; read 1 = ``VPU_TXOUT_LATENCY``, write ->
       read 2; the tile survives reset
     - 16 x 4 only (the row select hard-codes it). No unit-level MiniTPU tb
       exists (only ``vpu``-level ones)
   * - Allo simulator / SystemC csim
     - **open** (track A: X1; X2 asks whether D-12 can state a transposing read, H13)
     -
   * - Catapult RTL + DC
     - **open** (track C)
     -

``mxu_pe``
----------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** per cycle, every output, 60,090 cycles (360,202 slots); psum 4 =
       ``MXU_PE_LATENCY``, commit -> psum 5, forwards 1; the PE's own depth
       assertion never fired
     - cycle model ``ref_mxu.mxu_pe_trace``; unreset payload tainted (338
       "reset fill" slots: the adder's special class is forced by reset while
       its payload flows, so 21 of them differ from the data function)
   * - Allo simulator / SystemC csim
     - **open** (track B: P1 bits from U1's ``mul_acc24`` + ``acc24_add_pipe``)
     -
   * - Catapult RTL + DC
     - **open** (track C: per-cycle against the cycle model at ``latency=4``, H6)
     -
   * - RTLGen / AMC
     - **open** (track D)
     -

``mxu_systolic_array``
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** per cycle at DIM 2, 4 and 16, on random per-cycle stimulus and
       on skewed tiles; row r -> result ``4*(DIM-r)`` edges
     - a grid of the PE model wired as the ``.sv`` wires it. The array has no
       matrix contract of its own: the skew is ``mxu.sv``'s
   * - Allo simulator / SystemC csim
     - **open** (track B: A1 Stream grid, judged at ``mxu``; P-2)
     -
   * - Catapult RTL
     - **open** (track C: A2 Wire-locked grid, expected **blocked**, H8)
     -

``mxu``
-------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - RTL oracle (Phase 0)
     - **match** against the push / commit / pop **contract** at DIM 2, 4, 16,
       on 7 legal directed programs, 3 random legal programs per DIM and 3
       illegal ones (overflow: one group dropped on every lane, as the
       reference's drop; bank overwrite; early commit); push -> valid
       ``2 + 5*DIM`` (82 at 16, ISA 85); switch span measured by function =
       ``5*(DIM-1)`` (75 at 16); ``tb_mxu_single_port`` replayed 85/85;
       ``tb_matrix_*`` (4) PASS unchanged
     - ``ref_mxu.mxu_trace`` models no ``mxu.sv`` register. Method finding: a
       span probe on random data read 74 at DIM 16 -- one corrupted weight of
       256 rounds away; sensitive data gives 75 exactly
   * - Allo simulator / SystemC csim
     - **open** (track B: M1 Streams per ``u3_fifo_composed``)
     -
   * - Catapult RTL + DC
     - **open** (track C: M1 against the contract; push->valid reported, not
       forced, P-3 / O2)
     -

Composition (the U3 upgrade)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Question
     - Cell
     - Evidence / note
   * - MAC plug-in (Q1)
     - **open** (track E; H10)
     - C1/C2 fixed on ``main``; C9/C10 open; ``compose.unit`` avoids both
   * - Matrix engine swap (Q2)
     - **open** (track E; H11)
     - systolic and tree are different functions at bf16: order must be declared (P-7)
   * - Optional modules (Q3)
     - **open** (track E; H15)
     -
   * - Derived parameters as legality (Q4)
     - **open** (tracks A, E)
     - tap level, max delay, span, push->valid

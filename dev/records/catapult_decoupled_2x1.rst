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

##################################################
Catapult Synthesis Record: ``top_decoupled_2x1``
##################################################

.. note::

   **Dated measurement record.** Catapult Ultra 2024.2 (build 1130128),
   ``nangate-45nm_beh``, 2.0 ns clock (500 MHz), block synthesis. The design is
   ``top_decoupled_2x1`` (1 memory tile + 2 compute tiles). The source note was
   rescued onto the fork on 2026-07-15 (``4ee5bdde``, from the since-deleted
   ``feature/mesh-accelerator`` branch) and retired 2026-09-17; the synthesis
   date itself is not recorded in it. Nothing here is maintained -- treat every
   claim as dated.

This page preserves the parts of the former ``notes/archive/CATAPULT.md`` that
exist nowhere else: the structural diagram, execution timeline and concurrency
analysis (§0), and the only per-loop latency and per-module / per-FIFO area
numbers for ``top_decoupled_2x1`` (§3-4). §1-2 are kept as the context needed to
read them. The tile totals alone (295 / 67 / 657 / 298 cycles; CT=14991,
CT=14991, MT=16180) also appear in ``docs/source/extensions/catapult_systemc.rst``. The
"not pursuing Catapult further" decision these numbers originally fed was
reopened on 2026-09-18, so they are live reference again.

The error cookbook (§5) and synthesis methodology (§6) of the original are
covered by ``docs/source/backends/catapult.rst``.

0. Design Visualization
-----------------------

0.1 Structural Diagram
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                          top_decoupled_2x1                                  │
   │                                                                             │
   │  External Memory                                                            │
   │  ┌──────────────────┐      ┌────────────────────────────┐                  │
   │  │ v0.d [A] 512-bit ├─────►│                            │                  │
   │  │ v1.d [B] 512-bit ├─────►│   memory_tile_2x1_0 (MT)  │                  │
   │  │ v2.d [out0]      │◄─────┤    Lat=67 / Thru=69 cyc   │                  │
   │  │ v3.d [out1]      │◄─────┤    Datapath:  2,971 score  │                  │
   │  └──────────────────┘      │    Register: 13,209 score  │                  │
   │                             └──────┬──────────────┬──────┘                  │
   │                                    │              │                         │
   │          ┌─────────────────────────┘              └──────────────────────┐  │
   │          │  ←── 4 channel types per CT ────────────────────────────────► │  │
   │          │                                                                │  │
   │    Data  │ ████ depth=16, 32-bit, area=4279 (v184:cns)                  │  │
   │    Ctrl  │ ▓▓▓  depth= 3, 32-bit, area= 862 (v186:cns)                  │  │
   │  Result  │ ░░░  depth= 2, 32-bit, area= 594 (v188:cns)                  │  │
   │   Token  │ ·    depth= 1, 32-bit, area= 326 (v191:cns)                  │  │
   │          │                                                                │  │
   │          ▼                                                                ▼  │
   │  ┌──────────────────────┐              ┌──────────────────────┐            │
   │  │ compute_tile_2x1_0   │              │ compute_tile_2x1_1   │            │
   │  │       (CT0)          │              │       (CT1)          │            │
   │  │  spad[16] FP32       │              │  spad[16] FP32       │            │
   │  │  Lat=295 / Thru=298  │              │  Lat=295 / Thru=298  │            │
   │  │  Datapath:  4,247    │              │  Datapath:  4,247    │            │
   │  │  Register: 10,744    │              │  Register: 10,744    │            │
   │  └──────────────────────┘              └──────────────────────┘            │
   └─────────────────────────────────────────────────────────────────────────────┘

    8 FIFOs total (4 types × 2 CTs):
      Data  ×2: MT→CT0, MT→CT1          4,279 each → total  8,558 (69% of interconnect)
      Ctrl  ×2: MT↔CT0, MT↔CT1            862 each → total  1,725
      Result×2: CT0→MT, CT1→MT             594 each → total  1,188
      Token ×2: CT↔MT (done/ack)        594 + 326  → total    920
      ─────────────────────────────────────────────────────────────
      Total interconnect:                               12,391 (19% of design area)


0.2 Execution Timeline
~~~~~~~~~~~~~~~~~~~~~~

Based on the Loop Execution Profile table in ``cycle.rpt``.
1 invocation = 298 cycles = 596 ns @ 500 MHz (``Thru=298``).

.. code-block:: text

   Cycle:  0        32       64  69   99      198       297 298
           |         |        |   |    |        |         |   |

   MT:     [══A×16══][══B×16══][5]
           l_S_i_2_i  l_S_i_12_i2           (32 cyc each, 45.7% each)
           └── done at cycle 69 ────────────────────────────────

   CT0:    [3][──RX──][────Compute (48)────][──TX──] ← while iter 1 (99 cyc)
                16                 48          32
           [3][──RX──][────Compute────────][──TX──]   ← while iter 2
           [3][──RX──][────Compute────────][──TX──]   ← while iter 3
                                                       298 cyc total

   CT1:    (identical to CT0, runs in parallel)

   Legend:
     [3]         = while handshake (try_get grant token)
     [──RX──]    = l_S_i_1: receive 16 FP32 from data FIFO (16 cyc, 1 c-step/elem)
     [──Compute] = l_S_i_2: FP accumulate 16 elements  (48 cyc, 3 c-steps/elem)
     [──TX──]    = l_S_i_5: send 16 FP32 to result FIFO (32 cyc, 2 c-steps/elem)


0.3 Concurrency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

MT ↔ CT: Pipeline Overlap
^^^^^^^^^^^^^^^^^^^^^^^^^

**MT (69 cyc) and CTs (298 cyc) run concurrently.** The Data FIFOs (depth=16) absorb the producer-consumer speed mismatch.

- MT ``l_S_i_2_i`` (32 cyc) and CT ``l_S_i_1`` RX (16 cyc) **overlap via the FIFO**: as soon as MT writes the first element, the CT begins reading it (streaming overlap from cycle ~1).
- MT finishes at cycle 69; CTs run until cycle 298. **Cycles 70–298: CTs run alone** — this window covers most of Compute (144 cyc) and TX (96 cyc).
- Area perspective: Data FIFO depth=16 is needed to buffer the 4.3× speed ratio (298/69). Catapult's automatically inferred depth=16 is a full-burst buffering choice, accepting the area cost (4,279 each, 69% of total FIFO area) to avoid stalls.

CT0 ↔ CT1: Full Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**CT0 and CT1 run the same 298-cycle schedule in parallel.**

- ``cycle.rpt``: both modules report ``Latency=295, Throughput=298`` with identical loop structure.
- MT broadcasts the same data to CT0 and CT1 via independent FIFOs, so there is no data dependency between the two CTs.
- The reported design total latency of 657 cycles is a sequential sum; the **actual critical path when run concurrently is 298 cycles** (CT-dominated).

Within a CT: RX → Compute → TX is Sequential (No Intra-Tile Pipeline)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per ``while`` iteration (99 cycles):

===================== ====== ==============================================
Phase                 Cycles Overlap with other modules
===================== ====== ==============================================
Handshake (Ctrl)      3      —
RX (``l_S_i_1``)      16     Overlaps with MT ``l_S_i_2_i`` writes via FIFO
Compute (``l_S_i_2``) 48     MT already done or sending next burst
TX (``l_S_i_5``)      32     Overlaps with MT result-receive loop via FIFO
===================== ====== ==============================================

RX → Compute → TX are **sequential within a tile** (no HLS pipeline pragma applied).
Future optimization: applying double-buffering to ``l_S_i_1`` (RX) and ``l_S_i_2`` (Compute) would overlap them, theoretically reducing CT throughput by up to 48+32 = 80 cycles.


1. Design Overview
------------------

**Design:** ``top_decoupled_2x1`` — Decoupled 2×1 mesh accelerator (1 Memory Tile + 2 Compute Tiles)
**Source:** ``tests/dataflow/test_decoupled_mesh.py`` → ``catapult_decoupled_2x1.prj/kernel.cpp``
**Architecture:** Valid-ready handshake between MT and CTs via ``ac_channel`` FIFOs; MT streams matrix rows to CTs, CTs compute FP dot-products and stream results back.

Constants
~~~~~~~~~

=============== ===== =================================================
Parameter       Value Meaning
=============== ===== =================================================
``M, N, K``     2     Matrix dimension
``P0, P1``      4     Partition factors
``MEM_SIZE``    256   External memory words
``IMEM_SIZE``   8     Instruction/config memory
``BW``          16    Bit-width per element (effective: 32-bit FP used)
``TILE_M``      2     Tiles in M dimension
Elements/stream 16    Payload size per streaming burst
=============== ===== =================================================


2. Synthesis Setup
------------------

===================== =======================================================================
Parameter             Value
===================== =======================================================================
Tool                  Catapult Ultra 2024.2 (build 1130128)
Target library        ``nangate-45nm_beh``
Clock                 ``clk``, rising edge, 2.0 ns period (500 MHz)
Clock uncertainty     0%
Synthesis mode        **Block synthesis** (``solution design set <fn> -block`` for each tile)
Reported timing slack 0.041 ns (timing closure met)
Max delay             1.959 ns
===================== =======================================================================

TCL Flow (catapult_decoupled_2x1.prj/run.tcl)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   solution options set /Input/CppStandard c++11
   solution options set /Input/CompilerFlags {{-D_GLIBCXX_USE_CXX11_ABI=0}}
   solution file add kernel.cpp -type C++
   directive set -DESIGN_HIERARCHY top_decoupled_2x1
   directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0}}
   solution options set /Output/OutputVerilog true
   solution library add nangate-45nm_beh

   go analyze
   solution design set memory_tile_2x1_0 -block
   solution design set compute_tile_2x1_0 -block
   solution design set compute_tile_2x1_1 -block

   go compile
   solution library add ccs_sample_mem
   go assembly
   go extract


3. Latency Results
------------------

3.1 Summary
~~~~~~~~~~~

====================== ==================== ====================== ==============
Module                 Latency (cycles)     Throughput (cycles)    Time @ 500 MHz
====================== ==================== ====================== ==============
``compute_tile_2x1_0`` 295                  298                    596 ns
``compute_tile_2x1_1`` 295                  298                    596 ns
``memory_tile_2x1_0``  67                   69                     138 ns
**Design Total**       **657** (sequential) **298** (CTs dominate) **596 ns**
====================== ==================== ====================== ==============

Tiles run **concurrently**: MT sends data in 69 cycles while CTs compute in parallel.
Critical path: CT throughput (298 cycles) determines overall system throughput.


3.2 Compute Tile (CT) — Latency Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each CT processes 16 FP elements from MT, accumulates a dot-product, and streams the result back. The outer ``while`` loop runs **3 handshake rounds** (matching the data/token protocol).

Per-Iteration Loop Breakdown (single ``while`` iteration = 99 cycles)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

================== ================================================= ===== ============ ===========
Loop               Role                                              Iters C-steps/iter Cycles/iter
================== ================================================= ===== ============ ===========
``while`` header   Handshake control (try_get grant)                 —     3            3
``l_S_i_1``        **Receive**: read 16 FP elements from data stream 16    1            16
``l_S_i_2``        **Compute**: FP add/accumulate 16 elements        16    3            48
``l_S_i_5``        **Send**: write 16 FP results to result stream    16    2            32
**Total per iter**                                                                      **99**
================== ================================================= ===== ============ ===========

Full CT Execution (3 ``while`` iterations = 297 cycles + 1 overhead = 298)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------+----------------------+---------+-----------------+-------------------------+
| Category             | Loop(s)              | Cycles  | % of Throughput | Notes                   |
+======================+======================+=========+=================+=========================+
| **Communication      | ``l_S_i_1`` × 3      | **48**  | 16.1%           | Stream-read 16 FP32     |
| (RX)**               |                      |         |                 | from MT data FIFO       |
+----------------------+----------------------+---------+-----------------+-------------------------+
| **Compute (FP)**     | ``l_S_i_2`` × 3      | **144** | 48.3%           | FP accumulate: 3        |
|                      |                      |         |                 | c-steps/element (FP add |
|                      |                      |         |                 | latency)                |
+----------------------+----------------------+---------+-----------------+-------------------------+
| **Communication      | ``l_S_i_5`` × 3      | **96**  | 32.2%           | Stream-write 16 FP32 to |
| (TX)**               |                      |         |                 | result FIFO             |
+----------------------+----------------------+---------+-----------------+-------------------------+
| **Control            | ``while`` header × 3 | **9**   | 3.0%            | ``try_get``/``try_put`` |
| (handshake)**        |                      |         |                 | on grant tokens         |
+----------------------+----------------------+---------+-----------------+-------------------------+
| **Overhead**         | ``main``             | **1**   | 0.3%            | Loop prologue           |
+----------------------+----------------------+---------+-----------------+-------------------------+
| **Total**            |                      | **298** | 100%            | Throughput (II)         |
+----------------------+----------------------+---------+-----------------+-------------------------+

**Key insight**: FP compute dominates at 48%. Communication accounts for 48% total (RX + TX),
confirming the design is **compute-memory balanced** at this datapath width.


3.3 Memory Tile (MT) — Latency Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MT reads two 16-element arrays from external memory (``v0.d`` = matrix A, ``v1.d`` = matrix B) and streams each to both CTs simultaneously. It simultaneously receives result streams from CTs and writes them to external memory (``v2.d``, ``v3.d``).

MT Loop Breakdown
^^^^^^^^^^^^^^^^^

=============== ============================================ ===== ============ ============ =====
Loop            Role                                         Iters C-steps/iter Total Cycles %
=============== ============================================ ===== ============ ============ =====
``l_S_i_2_i``   **Storage→Comm**: read A + stream to CT0/CT1 16    2            **32**       45.7%
``l_S_i_12_i2`` **Storage→Comm**: read B + stream to CT0/CT1 16    2            **32**       45.7%
``main``        Overhead + control token management          —     5            **5**        7.1%
``core:rlp``    Ring-loop reset overhead                     —     1            **1**        1.4%
**Total**                                                                       **69** → 70  100%
=============== ============================================ ===== ============ ============ =====

MT Subcategory Breakdown
^^^^^^^^^^^^^^^^^^^^^^^^

================================ ====== ==== =======================================================
Category                         Cycles %    Notes
================================ ====== ==== =======================================================
**Storage→Comm (read A + send)** **32** 46%  16 reads × 2 c-steps (mem read + channel write fused)
**Storage→Comm (read B + send)** **32** 46%  Same pattern for B matrix
**Control + overhead**           **6**  9%   Token generation, output memory write control, prologue
**Total**                        **69** 100% 
================================ ====== ==== =======================================================

**Key insight**: MT is almost entirely I/O-bound. The 2-cycle per element cost reflects the
external memory read (1 cycle) + FIFO broadcast write to 2 CTs (1 cycle). MT is **not**
a bottleneck — it completes in 69 cycles vs. CT's 298 cycles.


4. Area Results
---------------

4.1 Design-Level Area Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*(Catapult area score units — scheduling metric, not physical nm²)*

================================== ===================== ===
Category                           Post-Assignment Score %
================================== ===================== ===
REG (pipeline registers, arrays)   35,029                54%
FUNC (datapath logic: adders, MUL) 14,551                22%
MUX (data path multiplexers)       14,313                22%
LOGIC (control logic)              1,408                 2%
**Total (excl. I/O + FIFOs)**      **65,301**            
FSM (finite state machine regs)    161                   —
**Design Total post-assignment**   **65,462**            
================================== ===================== ===

Per-Module Area (CRAAS-12 final schedule)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============================== ========= ========= =============
Module                         Datapath  Register  Total
============================== ========= ========= =============
``compute_tile_2x1_0``         4,246.68  10,744.27 **14,990.96**
``compute_tile_2x1_1``         4,246.68  10,744.27 **14,990.96**
``memory_tile_2x1_0``          2,971.23  13,208.50 **16,179.73**
Sum of tiles                   11,464.59 34,697.04 **46,161.65**
Interconnect FIFOs (top-level) —         —         **12,390.81**
Other (lib primitives, FSM)    —         —         **~910**
**Design Total**                                   **~65,462**
============================== ========= ========= =============


4.2 Compute Tile (CT) — Area Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each CT contains FP arithmetic units and a scratchpad register array (``spad[16]``).
FIFOs are instantiated at the top level and not counted in tile area.

+------------------------+---------------+---------+-----------------------------------------------+
| Subcategory            | Score         | % of CT | What it contains                              |
+========================+===============+=========+===============================================+
| **Compute** (Datapath) | **4,246.68**  | 28%     | FP adder/accumulator (3 c-step FP add); loop  |
|                        |               |         | counter logic; address arithmetic for spad    |
|                        |               |         | indexing; MUX for operand selection           |
+------------------------+---------------+---------+-----------------------------------------------+
| **Storage** (Register) | **10,744.27** | 72%     | ``spad[16]``: 16 × 32-bit FP scratchpad (512  |
|                        |               |         | bits); accumulator register; pipeline hold    |
|                        |               |         | registers for stream I/O; loop induction      |
|                        |               |         | variables                                     |
+------------------------+---------------+---------+-----------------------------------------------+
| **Communication**      | **0**         | —       | Channels (FIFOs) are top-level;               |
|                        |               |         | ``ccs_in_wait``/``ccs_out_wait`` I/O ports    |
|                        |               |         | are 0-area interface wrappers                 |
+------------------------+---------------+---------+-----------------------------------------------+
| **CT Total**           | **14,990.96** | 100%    |                                               |
+------------------------+---------------+---------+-----------------------------------------------+

**Observation**: Storage dominates at 72%, driven by the 16-element FP scratchpad.
For larger tile sizes, storage cost would grow linearly with scratchpad depth.


4.3 Memory Tile (MT) — Area Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MT manages 4 external 512-bit memory interfaces (v0..v3, each 16×32-bit = 512-bit wide)
and internal control logic for handshaking.

+------------------------+---------------+---------+-----------------------------------------------+
| Subcategory            | Score         | % of MT | What it contains                              |
+========================+===============+=========+===============================================+
| **Compute** (Datapath) | **2,971.23**  | 18%     | Address increment logic (4 counters for       |
|                        |               |         | v0..v3 indices); MUX for port arbitration;    |
|                        |               |         | comparators for loop bounds; index arithmetic |
|                        |               |         | for 2D tile mapping                           |
+------------------------+---------------+---------+-----------------------------------------------+
| **Storage** (Register) | **13,208.50** | 82%     | I/O staging registers for 4 × 512-bit         |
|                        |               |         | external ports; control state registers       |
|                        |               |         | (handshake token counters); address           |
|                        |               |         | registers; intermediate pipeline buffers for  |
|                        |               |         | memory-to-FIFO path                           |
+------------------------+---------------+---------+-----------------------------------------------+
| **Communication**      | **0**         | —       | Top-level FIFOs; ``ccs_in``/``ccs_out`` ports |
|                        |               |         | are 0-area                                    |
+------------------------+---------------+---------+-----------------------------------------------+
| **MT Total**           | **16,179.73** | 100%    |                                               |
+------------------------+---------------+---------+-----------------------------------------------+

**Observation**: MT register area (13K) exceeds its datapath (3K) by 4.4×, reflecting the
high register cost of staging 4 wide external memory buses (4 × 512 bits = 2 Kbits of staging).


4.4 Interconnect — FIFO Area Breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All 8 ``ccs_pipe`` FIFOs are at the top level (``/top_decoupled_2x1``). Each is 32-bit wide.

============ ========= ===== ============= ==================================================
FIFO         Direction Depth Area Score    Notes
============ ========= ===== ============= ==================================================
``v184:cns`` MT → CT0  16    4,279.155     **Data stream** (16-element burst, full buffering)
``v185:cns`` MT → CT1  16    4,279.155     **Data stream** (same for CT1)
``v186:cns`` MT ↔ CT0  3     862.281       **Control** (grant token, try_put/try_get)
``v187:cns`` MT ↔ CT1  3     862.281       **Control** (grant token, try_put/try_get)
``v188:cns`` CT0 → MT  2     594.043       **Result stream** (CT0 output)
``v189:cns`` CT1 → MT  2     594.043       **Result stream** (CT1 output)
``v190:cns`` CT ↔ MT   2     594.043       **Handshake** (done/ack)
``v191:cns`` CT ↔ MT   1     325.805       **Token** (small control)
**Total**                    **12,390.81** 19% of design total
============ ========= ===== ============= ==================================================

Communication Area Subcategory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------+----------------------+---------------+-------------+----------------------+
| Type                 | FIFOs                | Total Area    | % of Design | Notes                |
+======================+======================+===============+=============+======================+
| **Data** (MT→CT)     | 2 × depth-16         | **8,558.31**  | 13%         | Dominant; 1 per CT,  |
|                      |                      |               |             | full pipeline depth  |
+----------------------+----------------------+---------------+-------------+----------------------+
| **Result** (CT→MT)   | 2 × depth-2          | **1,188.09**  | 1.8%        | Small; results sent  |
|                      |                      |               |             | sequentially         |
+----------------------+----------------------+---------------+-------------+----------------------+
| **Control/Token**    | 2×depth-3 +          | **2,644.41**  | 4.0%        | Handshake grant/ack  |
|                      | 1×depth-2 +          |               |             | protocol             |
|                      | 1×depth-1            |               |             |                      |
+----------------------+----------------------+---------------+-------------+----------------------+
| **Total              |                      | **12,390.81** | **18.9%**   |                      |
| interconnect**       |                      |               |             |                      |
+----------------------+----------------------+---------------+-------------+----------------------+

**Key insight**: Data FIFOs (depth-16) account for 69% of total FIFO area because they must
buffer a full 16-element burst. Reducing burst size or using a shared bus would cut FIFO area
significantly. Control FIFOs are cheap (depth 1–3) since they carry single tokens.


4.5 Whole-Design Area Summary by Subcategory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==================== =========== ==== =================================================
Subcategory          Area Score  %    Sources
==================== =========== ==== =================================================
**Compute**          ~15,465     24%  CT Datapath (×2) + MT Datapath + FUNC + LOGIC
**Storage**          ~34,697     53%  CT Register (×2) + MT Register + pipeline buffers
**Communication**    ~12,391     19%  All 8 ccs_pipe FIFOs (data + control + result)
**Other** (lib, FSM) ~909        1%   leading_sign primitive, FSM state, mgc_io_sync
**Total**            **~65,462** 100% 
==================== =========== ==== =================================================

Fixes that survive only as code
-------------------------------

Three of the original §5 fixes are not restated elsewhere in the documentation;
they survive in the emitter source:

- **CRD-413** (``ac_ieee_float<binary32>`` assigned to a ``float[]`` stateful
  global): fixed by the virtual hook ``emitStatefulGlobalElementType``,
  overridden in ``CatapultModuleEmitter`` (``EmitVivadoHLS.h:102`` /
  ``EmitCatapultHLS.cpp:143``).
- **Scalar float literals** ``(float)1.000000`` -> ``1.000000f``: Catapult reads
  the double literal as ``double`` before the cast, which EDG rejects; the
  ``f`` suffix is emitted in ``mlir/lib/Translation/Utils.cpp`` (``Utils.cpp:63``).
- **CIN-319** (``hls_design dataflow`` attribute not recognized): recorded as
  a benign informational message -- the Catapult backend expresses parallelism
  through block synthesis rather than ``#pragma HLS DATAFLOW``. The 2026-09-18
  archive audit located what survives of it as code at the ``static`` in
  ``EmitCatapultHLS.cpp:308``.

Original file layout
--------------------

The original record also listed the files modified for the bring-up and the
commands of a driver script, ``tests/dataflow/catapult_synth_decoupled_2x1.py``,
which is not in the tree (it stayed on the deleted ``feature/mesh-accelerator``
branch, last tip ``06ce561``; the nearest in-tree equivalent is
``tests/dataflow/hls_synth_decoupled.py``). Outputs were under
``catapult_decoupled_2x1.prj/Catapult_25/top_decoupled_2x1.v1/`` (``rtl.v``,
``cycle.rpt``, ``rtl.rpt``). The complete original is recoverable with:

.. code-block:: bash

   git show a2d92cd8:notes/archive/CATAPULT.md

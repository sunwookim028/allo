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
TinyTPU-align: Aligning TinyTPU-isa with MiniTPU
##################################################

This page is the alignment spec for **TinyTPU-align**, a variant of
TinyTPU-isa (:doc:`tinytpu_isa`) built for experiments on branch
``tinytpu-align``. It is aligned with MiniTPU (:doc:`minitpu`), another
engineer's machine, at the ISA and memory-hierarchy level. The parked design is
tag ``tinytpu-isa-v1`` (``476a70d8``; 172 / 262 / 418 / 484 / 686 cycles), and
this work does not modify it.

The page has three parts:

* MiniTPU's architectural contract, extracted from its RTL.
* Each element mapped onto TinyTPU-isa and classified as implementable in Allo
  on Vitis today, needing an Allo fix, or not expressible today.
* The increments that implement the implementable part, with what each one
  costs in cycles (:ref:`align-increments`).

**Sources.** MiniTPU at ``77b0bcc`` (``sunwookim028/minitpu-tmp``), read-only.
The single source of truth for its ISA is ``docs/isa_latency.json`` and
``docs/isa_slots.json``. ``tools/gen_isa_doc.py`` generates
``docs/ISA_AND_INTERFACES.md`` and the testbench packages from them, and
``tb/tb_isa_conformance.sv`` holds the RTL constants to them. Every statement
below was checked against ``src/`` and cites it (paths relative to
``src/core/``). Where :doc:`minitpu` or MiniTPU's own ISA doc disagree with the
RTL, the RTL wins and the disagreement is listed in
:ref:`align-corrections`.


.. _align-contract:

1. MiniTPU's architectural contract
===================================

Geometry and memories
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - element
     - contract (RTL)
   * - lanes / sublanes
     - ``NUM_LANES = 16``, ``NUM_SUBLANES = 4`` (``vpu/vpu_pkg.sv``). A **beat**
       is one row across the 16 lanes (256 bits, 32 B, one DDR bus word). A
       **word** is 4 beats, 128 B, exactly one VREG.
   * - VREG file
     - 32 registers of 64 BF16 (4 sublanes x 16 lanes), 4 KiB. **Three read
       ports** (A: the V slot; B: the ALU's second operand; C: a fixed mux the
       matrix stream engine wins over ``vst``) and **one write port**, shared by
       the ALU, SFU, reduction, ``vld`` and ``vmatpop`` through a registered
       writeback (``vpu/vpu_regfile.sv``, ``vpu/vpu.sv:426-430``). Never reset.
   * - VMEM
     - 4096 words = 16,384 beats = 512 KiB, **one flat true-dual-port array**:
       port A is compute (``vld``/``vst``), port B is DMA (``vpu/vpu_word_array.sv``).
       Whole words only on the compute port, no banking, **no arbiter**:
       same-word collisions are undefined. Read latency 3 (compute) and 2 (DMA).
   * - MXU
     - 16x16 weight-stationary BF16 array (``mxu/``). Output FIFO **64 entries
       per lane**; overflow **drops results silently** (``mxu/mxu.sv:40``).
       Input FIFO depth 4. Two pending weight banks per PE.
   * - DRAM (DMEM)
     - PL DDR, 16 GiB, addressed in 32 B beats (29-bit beat address).
       Descriptors address it with ``base + i*stride`` in beats.
   * - IRAM
     - 4096 bundles of 128 bits, loaded by the host, then self-fetched.

The bundle
----------

One 128-bit bundle is issued per cycle. The fields are fixed
(``sequencer/sequencer_pkg.sv``, ``encoded_bundle_t``):

.. code-block:: text

   V   [127:108] 20  op[5] rd[5] ra[5] rb[5]           vector op (16 of 32 opcodes spare)
   M   [107:100]  8  subop[3] reg[5]                   vmatload / vmatpush / vmatpop
   MEM [ 99: 66] 34  kind[2] + 32-bit payload          LDST (vld/vst) or DESC (vmemld/vmemst)
   S   [ 65: 53] 13  valid op[3] rd[2] rs[2] use_iv level[2] spare[2]
   C   [ 52: 46]  7  op[3] operand[4]                  loop.begin(.r), loop.end, wait.channel, halt
   IMM [ 45: 22] 24  one constant shared by S, C and DESC (exactly one claimant)
   DELAY[21: 15]  7  cycles to wait after this bundle, 0..127
   reserved [14:0]

   LDST: dir vreg[5] word[12] agu_valid agu_level[2] agu_shift[3] spare[8]
   DESC: dir channel[2] vmem_word[12] rows_m1[12] has_disp base_sreg[2] stride_sreg[2]

Slot rules the encoding cannot break:

* one op per slot;
* ``vld``/``vst`` and a descriptor are both MEM, so they never share a bundle;
* a descriptor never shares a bundle with a C op (both are next-state decisions);
* ``vst`` never shares a bundle with ``vmatload``/``vmatpush`` (both read port C);
* two writes to one VREG, or on one write-port cycle, never share a bundle.

Instruction semantics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - op
     - semantics (RTL)
     - W (write offset) / occupancy
   * - ``vadd``/``vsub``/``vmul``/``vmax``/``vmin``/``vmov``
     - elementwise on 64 BF16. ``vadd``/``vsub``: round-to-nearest-even (RNE),
       gradual subnormals, exact cancellation gives +0. ``vmul``: RNE,
       **flush-to-zero on input and output**. ``vmax``/``vmin``: ordering key
       ``bf16_gt``.
     - W=5, II=1
   * - ``vgelu``/``vexp``/``vrecip``/``vrsqrt``
     - SFU, table-based (``sfu/*.mem``)
     - W=7
   * - ``vredsum``/``vredmax``, ``vlanered``
     - 64-wide tree result broadcast, or a per-sublane tap two levels earlier
     - W=15 / W=11
   * - ``vtxin``/``vtxout``
     - 4 beats in, 4 beats out of a 16x16 transpose tile
     - W=3 (out)
   * - ``vld v, word``
     - VMEM word -> VREG. Word = ``literal + (iv[level] << shift)``, 12-bit and
       wrapping (``sequencer/sequencer_agu_resolve.sv``); ``shift`` is 0..7 in
       **words**
     - W=6
   * - ``vst v, word``
     - VREG (port C, sampled at issue) -> VMEM word, RAM write at +1
     - --
   * - ``vmatload base``
     - reads ``base .. base+3`` over 16 cycles. PE(r, c) ends up holding
       W[r][c], where row r of W is VREG ``base + r/4``, sublane ``r%4``, lane
       ``c``. The weights take effect from the **next** ``vmatpush``: each PE
       switches as that push's first activation passes it. Loads alternate
       between the two pending banks.
     - 17 cycles to the next stream command
   * - ``vmatpush vs``
     - 4 beats (sublanes 0..3 of ``vs``), each an activation row whose lane k
       enters PE row k. **Y = X W, no transpose.** The partial sum starts at 0
       in PE row 0 (``mxu/mxu_systolic_array.sv:54``), so **nothing
       accumulates across pushes**.
     - 5 cycles; first result at +82
   * - ``vmatpop vd``
     - 4 masked beats: output rows of the **oldest un-popped push** into
       sublanes 0..3 of ``vd``, each beat gated on the output FIFO (the one
       place the hardware waits)
     - W=3..6, 4 cycles
   * - ``vmemld``/``vmemst``
     - descriptor: VMEM beat ``{vmem_word, 00}``, ``(rows_m1+1)*4`` beats,
       DRAM beat i at ``base + i*stride`` with ``base = sreg[b] (+ sext(disp))``,
       ``stride = sreg[s]``, both snapshotted at issue. So one VMEM word
       gathers DRAM beats base, base+S, base+2S and base+3S into sublanes 0..3.
       A burst only when ``stride == 1``. **Asynchronous**: accepted only if
       its channel is idle. **Two channels.** A load completes when its data is
       in VMEM, a store when its write response retires.
     - fence
   * - ``wait.channel mask``
     - waits until ``done & mask == mask`` on the sticky per-channel done bits,
       then clears them. **A mask naming an idle channel hangs.** A second
       descriptor on a busy channel hangs the sequencer.
     - >= 1 cycle
   * - S: ``smovi``/``smov.arg``/``saddi``/``smac``/``sshl``
     - four 32-bit SREGs, wrapping. ``smac``: ``rd = rs + (use_iv ? iv[level]
       : rs) * imm``. Written back after **S_LAT = 2** with no bypass.
     - reader >= 2 bundles later
   * - ``loop.begin lo, step, hi``
     - iv = lo, lo+step, ... while ``iv < hi``; **do-while** (always one pass);
       4-deep stack, iv indexed by absolute level
     - --
   * - ``loop.begin.r``
     - ``hi`` from an SREG or one of 4 kernel-argument CSRs, saturated to
       16 bits; ``hi <= lo`` skips the body (``skip`` bundles)
     - --
   * - ``halt``
     - waits for DMA idle only, **not** for VREG writes or the MXU in flight
     - --

Timing, hazards and the no-interlock rule
-----------------------------------------

* **DELAY.** Holding ``delay = N`` issues the next bundle at ``t + 1 + N``;
  ``delay = 0`` is back to back. The countdown freezes during a descriptor's
  detour and a wait.
* **Nothing interlocks.** Issue stalls only for DELAY, the descriptor and
  wait detours, halt drain, and an empty fetch queue. ``matrix_busy`` is
  simulation-only. VREG RAW/WAW, write-port collisions, SREG RAW, matrix
  command spacing and VMEM ownership are all the assembler's job
  (``board_package/asm.py schedule()``, cross-checked against
  ``tb/bundle_scheduler.svh``). The one exception is ``vmatpop``, which gates
  each beat on the output FIFO.
* **Consequence for semantics.** A bundle reads its sources at issue and
  writes W cycles later, and the scheduler only ever raises delays so that
  every read lands after the write it depends on and writes land in program
  order. **A correctly scheduled program therefore means exactly the
  sequential program: bundles in order, and within a bundle every read sees
  the value from before the bundle.** That is the property the interlocked
  implementation is built on (:ref:`align-vliw`).

Loop buffer, AGU, fences
------------------------

* **Replay buffer**: 24 bundles, innermost loop only, captured on the first
  pass. A longer body refetches (2 cycles an iteration). Time, not semantics.
* **AGU**: one term per load/store, ``iv[level] << shift``, no multiply.
  Descriptors do not use it; their addresses come from SREGs, which ``smac``
  can walk with an induction variable.
* **Fences**: ``wait.channel(mask)`` is the only synchronisation between DMA
  and compute. Which of the two owns a VMEM region is the program's job.

Accumulation and the output path
--------------------------------

* **In-array**: BF16 x BF16 products are exact, with inputs **flushed to zero**
  if subnormal. They are added down each column into a **24-bit float
  accumulator** (1 + 8 + 15), **rounded to nearest even at every add**, with
  subnormal results kept (``mxu/mxu_acc24_add_pipe.sv:199-224``). The sum is
  serial in k = 0..15, so a contraction is exactly 16 deep per push.
* **Out of the array**: acc24 -> BF16 by RNE on the bit pattern, with no flush
  to zero (``mxu/mxu.sv:75-89``). A result is therefore **rounded twice**.
* **Deeper contractions**: ``vadd`` in BF16 in the VPU, one popped tile at a
  time (``board_package/kernels.py:_accumulate``). MiniTPU tried accumulation
  across weight loads and reverted it (README, "Decided against").
* **Output path**: ``vmatpop`` -> VREG -> (``vadd`` ...) -> ``vst`` -> VMEM ->
  ``vmemst`` -> DRAM.

Measured reference points
-------------------------

**Simulation (Verilator)**: the six workloads of ``tb/tb_e2e_unified_runner.sv``,
run by ``tb/run_all.sh --quick``.

.. list-table::
   :header-rows: 1
   :widths: 8 42 12 38

   * - WL
     - workload (builder)
     - cycles
     - program
   * - 1
     - C = A + B, 32x64 BF16 (``tb/wl12_programs.svh:fill_workload1``)
     - 119
     - modulo schedule at II=3, 53 bundles, ``loop.begin hi=6``
   * - 2
     - C = A*B + A, 32x64 (``fill_workload2``)
     - 155
     - II=4, 69 bundles
   * - 3
     - 16x16x16 GEMM on the MXU (``build_workload3_gemm_mxu``)
     - **168**
     - 8 ``vld``, ``vmatload``, 4 ``vmatpush``, 4 ``vmatpop``, 4 ``vst``, halt
   * - 4
     - 32x32 C = A*B + A, DMA-streamed, 4 tiles (``build_workload4_gemm_32x32``)
     - 929
     - per tile: 2 ``vmemld``, ``wait.channel``, 4-trip loop of vld/vld/vmul/vadd/vst, ``vmemst``, wait
   * - 5
     - 16x16 GEMM -> GELU (``build_workload5_gemm_gelu``)
     - **155**
     - WL3 with ``vgelu`` co-issued beside the pops
   * - 6
     - LayerNorm, width 768 (``build_workload6_layernorm_768``)
     - 941 / 218 bundles
     - 12 tiles of 64: sum, variance, ``vrsqrt``, normalise

**Board** (ZCU104 at 187.5 MHz, reported by memh, the session that owns
MiniTPU; ``minitpu-run <fixture> --board`` on ``compiler/tests/fixtures/``):

* GEMM [32,256] x [256,64]: 15,955 cycles on the board, 12,970 in simulation;
* softmax: 1,544 (2 launches);
* FFN: 15,671;
* flash attention: 32,988 (2 launches).

**GEMM cycle model**, exact over 20 simulated points (``tools/gemm_cycle_model.py``),
with n output column tiles of 16, B contraction blocks of 32, M = 32:
``cycles = 46 + 265 n + 64.5 B + 119 n B``, which is 53.8% of peak at the
margin.

The only **board-validated cycle counts** in the tree are the bring-up kernels
(``tb/run_verilator_bringup.sh``: halt 4, vreg 12, scalar 8). Everything in
the table above is simulation. Simulated counts describe the schedule, not DDR
(``docs/TESTING.md``).


.. _align-classification:

2. Mapping onto TinyTPU-isa, and classification
===============================================

Classes:

**(a)** implementable in Allo on Vitis today;

**(b)** needs an Allo fix, named by its register item
(:doc:`/developer/limitations`) or marked **NEW** if this work found it;

**(c)** not expressible on Vitis/Allo today.

.. list-table::
   :header-rows: 1
   :widths: 18 26 26 30

   * - MiniTPU element
     - TinyTPU-isa v1 today
     - aligned form
     - class
   * - VMEM: one flat memory, DMA only into it
     - ``spad`` (input only) **plus** a DMA path straight into ``vr`` (the A
       bypass)
     - one ``VMEM``; ``dma_ld`` writes only VMEM
     - **(a)**
   * - VMEM dual port: compute and DMA
     - one owner process per memory
     - one owner process serving both ports. A DMA process writing and a
       compute process reading would be two owners
     - **(a)** as one owner. As two owners, **(b)**:
       :ref:`shared memory <limitation-shared-memory>` (Allo refuses a
       Stateful shared by two kernels and never emits ``stream type=unsync``)
   * - ``vld``/``vst`` VMEM <-> VREG
     - ``vld`` spad -> vr only; no ``vst`` (results never re-enter the input
       memory)
     - both directions
     - **(a)**, with the cyclic-graph caveats below
   * - one VREG file (3R 1W)
     - three memories: ``vr`` (operands), ``spad`` (weights), ``ar``
       (accumulators)
     - one VREG file in one owner process; its ports become a per-iteration
       read/write budget
     - **(a)**, but it makes the process graph **cyclic** (VREG -> array ->
       VREG, VREG <-> VMEM), which needs the two fixes below
   * - cyclic process graph in **csim/cosim**
     - acyclic by construction
     - --
     - **NEW, tooling**: Vitis runs a dataflow region's C model sequentially,
       so a cycle reads an empty stream and **cosim aborts before RTL
       simulation** (``COSIM 212-360``). Fixed outside Allo by
       ``threaded_csim.py``, which runs each process on a ``std::thread``
       under ``#ifndef __SYNTHESIS__`` (Vitis's ``hls::stream`` model is
       already thread-safe and blocking). Csynth accepts the cycle unchanged.
   * - cyclic process graph in **RTL**
     - --
     - --
     - **(b) NEW**: a process that puts a request and later gets its response
       in one pipelined loop **deadlocks in RTL**. The blocked read of
       iteration i+k freezes the whole default (stall) pipeline, including
       iteration i's put. ``#pragma HLS pipeline style=flp`` (flushable)
       fixes it (``align_probes/probe_cycle.py``: cosim PASS; ``frp`` still
       deadlocks); **Allo's
       ``s.pipeline`` has no ``style`` option**. Neither the Allo simulator nor
       csim can show it. Fix: about 10 lines beside ``rewind``.
   * - MXU ops ``vmatload``/``vmatpush``/``vmatpop``
     - one ``mm``: weights from ``spad``, activations from ``vr``, result
       deposited or accumulated in ``ar``
     - three ops. ``vmatload`` reads VREGs, the weights switch on the next
       push (a flag riding the activation wavefront, the same mechanism as
       MiniTPU's), and pops go to VREGs in push order
     - **(a)**
   * - two weight banks
     - ``wld`` double buffer (depth-4 ``wq``)
     - same
     - **(a)**
   * - output FIFO, 64 entries, silent drop
     - ``cw`` at depth ``QD``
     - a bounded FIFO with back-pressure; the validator refuses a program
       that leaves more rows un-popped than it holds, because under
       back-pressure that is a deadlock rather than a drop
     - **(a)**
   * - accumulation **location**: T deep in the array, deeper by ``vadd``
     - T deep in the array; deeper in ``accu``'s write path (``mm`` f2=1)
     - T deep in the array; k-tiles summed by ``vadd`` on VREGs
     - **(a)**
   * - accumulation **format**: acc24 (1+8+15), RNE per add; BF16 out
     - int32
     - datatype-parametric; exercised at int8 x int8 -> int32
     - BF16 through Allo's float types: **(b) NEW**, the Vivado emitter aborts
       on bf16 (below). acc24: **no MLIR type exists**. Bit-exact to MiniTPU:
       **(a) only as integer soft-float** written in Allo (see
       :ref:`align-datatype`)
   * - output path: pop -> VREG -> ``vst`` -> VMEM -> ``vmemst``
     - ``mvout``: ``ar`` -> clip -> DRAM
     - as MiniTPU
     - **(a)**, cyclic (see above)
   * - DMA descriptors (base/stride SREGs, rows)
     - ``dma_ld`` (row, column block), bursts; separate ``m_axi`` per operand
     - descriptor fields on DRAM **rows**; one reader process for inputs, one
       writer for C (Vitis forbids two processes on one ``m_axi`` bundle,
       ``HLS 200-1013``/``200-984``)
     - **(a)**
   * - async DMA, 2 channels, ``wait.channel``
     - synchronous, in program order
     - DMA rows land asynchronously. VMEM's owner enforces descriptor order
       against earlier compute, and waits at a fence
     - **(a)**, with non-blocking stream reads (``Stream.empty()``, which the
       Vivado emitter supports)
   * - loop stack (4 deep, lo/step/hi)
     - 4 deep, 0..trip-1
     - add lo/step
     - **(a)**
   * - ``loop.begin.r``, kernel-arg CSRs
     - --
     - a register-bounded loop, arguments as an extra region input
     - **(a)**
   * - AGU (1 term, shift)
     - 3 terms, arbitrary 11-bit strides: a superset
     - keep; MiniTPU's form is a special case
     - **(a)**
   * - scalar unit (4 SREGs)
     - --
     - in the sequencer
     - **(a)**
   * - replay buffer (24 bundles)
     - the whole program is prefetched on-chip
     - not needed (timing only)
     - n/a
   * - V-op set (vsub, vmul, vmax, vmin, vmov, reductions, transpose)
     - ``vadd``, ``vrelu``
     - add as needed; ``vrelu`` = ``vmax`` with a zero register
     - **(a)** for integers; SFU transcendentals in float: soft-float, or
       ``hls_math`` after the bf16 fix (the simulator lacks math lowering,
       :ref:`limitation-a`)
   * - 128-bit bundle, slots, IMM sharing
     - two 64-bit words, one op per instruction
     - decode the bundle; issue its slots to the units together
     - **(a)**, Phase 3
   * - **DELAY and no interlocks** (statically timed, fixed-latency,
       unhandshaked communication between units)
     - every edge is a FIFO; nothing is timed
     - DELAY honoured as a **minimum issue spacing**; hazards interlocked
     - timing-exact: **(c)**. A fixed-latency edge is a ``Wire``, which does
       not exist on the Vitis path, and on the SystemC fork it is wrong in RTL
       (:ref:`limitation-22`). Semantics: **(a)**, by the working decision
       below

No step is blocked. The one Allo fix the aligned machine needs to be correct
in RTL, ``pipeline style=flp``, is small and is a schedule primitive of the
same kind as ``s.dependence`` (:ref:`limitation-21`).


.. _align-vliw:

3. The VLIW decision: MiniTPU's semantics on interlocked hardware
==================================================================

**Working decision (from the user; memh agrees):** implement MiniTPU's ISA
**semantics** on **interlocked, decoupled** hardware. As
:ref:`align-contract` shows, a correctly scheduled no-interlock program means
the sequential program. A machine that preserves sequential semantics through
FIFOs and in-order units therefore runs it correctly; only the timing
differs. DELAY is honoured as a **minimum issue spacing**: the sequencer never
issues a bundle sooner than MiniTPU would, and may issue it later.

What the aligned machine guarantees (verified by the harness, not assumed):

* **VREG**: in program order within its one owner process. A dependence
  pragma, made true by an assembler distance contract, carries over from v1's
  accumulator (:ref:`tinytpu-isa-dependence`).
* **VMEM**: compute accesses in program order. A DMA write is applied only
  after every compute access that precedes its descriptor in program order
  (WAR is interlocked), and a compute access after a ``wait.channel`` sees
  the DMA it names (RAW through the fence). A compute access that races an
  un-fenced descriptor is refused by the validator, as
  ``tools/dma_fence_report.py`` refuses it on MiniTPU.
* **MXU**: pops wait for results (MiniTPU's pop also waits).

**What it costs in cycle-comparability:**

1. **A MiniTPU cycle count is a property of the schedule** (memh: a correctly
   scheduled program is timing-exact on MiniTPU). On TinyTPU-align it is a
   property of the microarchitecture. Running the same program on both gives
   the two machines' times for the **same semantics**; it does not give a
   bundle-by-bundle timing correspondence.
2. **DELAY as a minimum spacing** makes TinyTPU-align's issue timeline at best
   MiniTPU's and never ahead of it, so on a MiniTPU program ours can only tie
   or lose. With DELAY ignored, the count measures our machine's own limits.
   Both are reported where they differ (Phase 3).
3. **Interlocks cost cycles MiniTPU does not pay.** FIFO handshakes, pipeline
   fill per unit, and serialisation in the one process that owns the VREG
   file all add cycles. MiniTPU pays in assembler complexity instead. The
   increments below measure the price on TinyTPU's own programs.
4. **Interlocks also forgive** what MiniTPU would get wrong: an under-delayed
   schedule is still correct here. Differential testing against MiniTPU is
   therefore meaningful only for programs MiniTPU's assembler accepts.


.. _align-datatype:

4. Datatype: BF16 on Allo's Vitis path
======================================

The datatype is the **user's decision**. Phase 2 is written
datatype-parametric and exercised at **int8 x int8 -> int32**, TinyTPU's type.
memh suggests BF16 as the common type, plus int8 as a second, GEMM-only mode
on both machines. What was measured (``align_probes/probe_bf16.py``, under
``examples/accelerator/tinytpu_vitis/``):

.. list-table::
   :header-rows: 1

   * - Allo path
     - bf16 today
   * - frontend (``bfloat16`` type, ``arith`` ops on it)
     - works
   * - LLVM backend (``s.build()``)
     - **works**: ``a*b + a`` on 8 values, bit-exact against ``ml_dtypes``
   * - dataflow simulator (``df.build(target="simulator")``, a bf16 ``Stream``)
     - **works**, bit-exact
   * - Vivado HLS emitter (``target="vhls"`` / ``vitis_hls``)
     - **aborts the process**: ``EmitVivadoHLS.cpp:115 getTypeName``,
       ``Assertion '1 == 0 && "Got unsupported type."'``. There is no
       ``BF16Type`` case, and no bf16 case in the three dense-constant
       emitters (``:1624``, ``:2322``, ``:3203``).

**Vitis 2023.2 can represent both MiniTPU formats**: its ``ap_float.h``
provides ``ap_float<W, E>``, and ``ap_float<16, 8>`` is BF16 while
``ap_float<24, 8>`` is exactly acc24 (both pass the header's static checks).

**Size of the emitter fix.** About 60-100 lines of C++ plus a csim/cosim test:
a ``getTypeName`` case (``ap_float<16,8>``), bf16 in the three
constant emitters, ``#include <ap_float.h>``, and conversions (``ap_float``
converts to ``float`` only through ``ap_float<32,8>``, so ``extf``/``truncf``
need an explicit path). ``arith.bitcast`` also needs a non-``union`` form,
because ``ap_float`` is not trivially constructible. Small and verifiable, but
**it does not buy bit-exactness with MiniTPU**:

* ``ap_float`` arithmetic is the Xilinx Floating-Point Operator model:
  RNE, **subnormal operands flushed to zero** (checked:
  ``align_probes/apf.cpp``, a subnormal addend of an ``ap_float<32,8>`` add is
  treated as 0). MiniTPU flushes only
  in ``vmul`` and the MXU multiplier. ``vadd`` and the acc24 adds keep
  subnormals.
* The LLVM simulator computes bf16 through f32, which is correctly rounded
  for single bf16 ops but keeps subnormals everywhere. So **simulator and RTL
  would disagree on subnormals**, the same kind of divergence v1's signed
  bit-slice bug was.
* **acc24 has no MLIR type.** Allo's floats are bf16, f16, f32 and f64
  (``allo/ir/types.py:188-195``). Emulating acc24 in f32 then rounding is
  **not** innocuous double rounding (f32's 24-bit significand is below the
  2 x 16 + 2 = 34 bits that would make it so); in f64 it is.

**Recommendation for the decision.** If the aligned machine must be
**bit-exact to MiniTPU** in BF16, the PE multiply-accumulate, the output
rounding and the V ops should be written as integer soft-float in Allo,
transcribed from ``mxu_bf16_mul_acc24.sv`` / ``mxu_acc24_add_pipe.sv`` /
``vpu_bf16_add_pipe.sv``. That is class **(a)** with no Allo fix, identical on
the simulator and in RTL. It costs LUTs and PE pipeline depth, not an Allo
change. The emitter fix is worth doing anyway (bf16 is a supported frontend
type that crashes the Vitis path), but it gives ``ap_float`` semantics, which
are close to MiniTPU's and not equal to them.


.. _align-bench:

5. A common microbenchmark suite
================================

Each entry names the exact program on each machine. MiniTPU's programs exist
at ``77b0bcc``. TinyTPU-align's are ``isa_dsl.py`` generators on branch
``tinytpu-align``. A shape beyond the ``MAXDIM = 16`` build needs a larger
``TPU_MAXDIM`` build (the hardware is shape-independent; only the DRAM
arrays and VMEM grow).

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - benchmark
     - MiniTPU program (cycles)
     - TinyTPU-align program
   * - **G1** GEMM 16x16x16
     - WL3 ``build_workload3_gemm_mxu`` (168, sim); one 16x16 tile. BF16
     - ``isa_dsl.gemm_program(16, 16, 16)``: 16 tiles at T=4, k-tiles summed
       by ``vadd``
   * - **G2..G5** GEMM 4x4x4, 8x8x8, 12x12x12, 16x16x8
     - WL3-shaped single tile, zero-padded to 16 (not yet built on MiniTPU)
     - ``gemm_program(M, K, N)``; v1: 172 / 262 / 418 / 484
   * - **G6** GEMM [32,256] x [256,64]
     - ``compiler/tests/fixtures/gemm`` (12,970 sim, 15,955 board)
     - ``gemm_program(32, 256, 64)`` on a ``TPU_MAXDIM = 256`` build
   * - **E1** elementwise add 32x64
     - WL1 ``fill_workload1`` (119)
     - ``isa_dsl.eltwise_program("add", 32, 64)``: ``vld``, ``vld``, ``vadd``,
       ``vst`` (needs ``TPU_MAXDIM = 64``)
   * - **E2** multiply-add 32x32, DMA-streamed
     - WL4 ``build_workload4_gemm_32x32`` (929)
     - needs ``vmul``; the same four-tile ``vmemld``/``wait``/``vmemst``
       structure
   * - **R1** GEMM -> GELU, **R2** LayerNorm
     - WL5 (155), WL6 (941)
     - need the SFU and reductions; not planned

The five GEMM shapes (G1-G5) are the gate for every increment below, measured
in Vitis cosim against v1's 172 / 262 / 418 / 484 / 686.


.. _align-corrections:

6. Corrections to :doc:`minitpu` and to MiniTPU's ISA doc
=========================================================

* "16x16 GEMM 283 cy, GEMM->GELU 253 cy, board-validated targets":
  **stale, and not board figures**. They are Verilator workloads, now 168 and
  155 (MiniTPU ``72c8b78``: ``vmatpop``'s occupancy corrected from 7 to 4).
  The only board-validated cycle counts are the bring-up kernels.
* "accumulates exactly 16 deep in hardware": only the **products** are exact.
  Every add rounds to 15 fraction bits, and the result is rounded again to
  BF16.
* "MXU output FIFO 32/lane": **64** per lane.
* The GEMM cycle model is now ``46 + 265 n + 64.5 B + 119 n B`` (was
  ``46 + 194 n + 64.5 B + 303 n B``), after the second weight bank.
* AGU: ISA_AND_INTERFACES.md's "shift >= 2" is the assembler's **beat** shift.
  The RTL shift is 0..7 in **words**, with no minimum.
* Descriptor stride: the RTL walks 32 B **beats** with ``stride`` in beats. A
  burst needs ``stride == 1``, not "stride equals the row size".
* Plain ``loop.begin`` with ``hi <= lo`` still runs once; only ``loop.begin.r``
  skips.
* Not documented there: ``vmul`` and the MXU multiplier flush subnormals,
  ``vadd`` and the acc24 adds do not, and a descriptor's DELAY countdown is
  frozen during its detour.


.. _align-increments:

7. Phase 2: the increments and what each costs
==============================================

Each increment is one commit on ``tinytpu-align``, verified before the next
(:ref:`align-verification`). Cycle counts are Vitis ``cosim`` (xsim),
``-m_axi_latency 0``, the default testbench, T = 4, MAXDIM = 16, the same
setup as v1's published numbers.

.. list-table::
   :header-rows: 1
   :widths: 30 9 9 9 9 34

   * - design
     - 4x4x4
     - 16x16x16
     - vs previous (4 / 16)
     - vs v1 (4 / 16)
     - what it changed
   * - ``tinytpu-isa-v1`` (parked)
     - 172
     - 686
     - --
     - --
     - --
   * - **inc 1**: one VMEM, DMA only into VMEM (``b3793f85``)
     - 176
     - 750
     - +4 / +64
     - +4 / +64
     - A reaches the vregs by ``vld`` from VMEM, not by the DMA bypass. The
       +64 is exactly v1's measured A-bypass worth (``v_design``).
   * - **inc 2**: MiniTPU's M slot; accumulation by ``vadd``
     - 186
     - 1216
     - +10 / +466
     - +14 / +530
     - ``mm`` split into ``vmatload`` (weights from VREGs), ``vmatpush`` and
       ``vmatpop``; the weight switch rides the activation wavefront; k-tiles
       summed by ``vadd`` in ``accu``, two iterations a row. ``accu``
       becomes the critical unit: 704 iterations at 16x16x16, against 320
       for v1's accumulate-in-``mm``.
   * - **inc 3**: one VREG file
     - 197
     - 1904
     - +11 / +688
     - +25 / +1218
     - ``vru`` (operands) and ``accu`` (accumulators) merge into one ``vpu``
       that owns one file of int32 lanes (``vld`` sign-extends; the array
       takes each lane's low 8 bits). The graph becomes **cyclic** (vpu ->
       array -> vpu). The in-order ``vpu`` now pays the array's
       push-to-pop latency on every tile, which inc 2's separate ``accu``
       hid, and serialises what MiniTPU's three read ports overlap.
   * - **inc 3b**: the GEMM program, software-pipelined
     - 199
     - 1514
     - +2 / -390
     - +27 / +828
     - Program only: tile k+1 is loaded and pushed before tile k is popped,
       so the in-order ``vpu`` spends the array latency pushing (MiniTPU's
       GEMM kernels overlap the same way). The program is 25 static
       instructions, so the imem grows from 24 to 32 slots, which is the +2
       prefetch cycles at 4x4x4.
   * - **inc 4**: output via VREG -> ``vst`` -> VMEM -> ``vmemst``
     - 206
     - 1519
     - +7 / +5
     - +34 / +833
     - ``mvout`` retired. ``vst`` saturates each int32 lane to int8 on its
       way into VMEM (the narrowing ``mvout`` did), and ``vmemst`` moves VMEM
       rows to C. VMEM's owner now also sits on a cycle with the ``vpu``
       (``vld`` out, ``vst`` back), so its loop is ``style=flp`` too. Nearly
       free: the ``vpu`` did the same row work for ``mvout``.
   * - **inc 5a**: MiniTPU's DMA ISA (descriptors, channels, ``wait``)
     - 216
     - 1521
     - +10 / +2
     - +44 / +835
     - ``vmemld``/``vmemst`` are MiniTPU descriptors: beat ``base + r *
       stride`` of a flat beat space (A then B for loads, C for stores) and
       a channel, with ``wait`` fencing a channel mask. ``check_program``
       enforces MiniTPU's contract: no descriptor on a busy channel, no wait
       on an idle one, no ``vld``/``vst``/descriptor racing an unfenced
       descriptor. The machine runs each descriptor at its issue point in
       VMEM's program order, so a fenced program means the same here. The
       cost is the waits themselves: a ``wait`` reaches no unit but still
       takes a sequencer issue slot, three more dynamic instructions at
       4x4x4.

Reading the table: the whole cost so far is where MiniTPU does work that v1
did not have to. A DMA into VMEM and then a ``vld`` replaces v1's DMA straight
into the vregs. Summing k-tiles with explicit ``vadd`` instructions replaces
v1's accumulate-on-write (MiniTPU's own README records trying accumulation
across weight loads and reverting it).


**The aligned design at every scored shape** (after inc 5a, the default
testbench, one csynth; ``logs/align_final_cosim.log``):

.. list-table::
   :header-rows: 1

   * - shape
     - v1
     - TinyTPU-align
     - ratio
   * - 4x4x4
     - 172
     - 216
     - 1.26x
   * - 8x8x8
     - 262
     - 408
     - 1.56x
   * - 12x12x12
     - 418
     - 809
     - 1.94x
   * - 16x16x8
     - 484
     - 933
     - 1.93x
   * - 16x16x16
     - 686
     - 1521
     - 2.22x

The ratio grows with the work because nearly all of it is per-row work that
v1 spread over three processes and TinyTPU-align serialises in one ``vpu``.
At 16x16x16 the ``vpu`` runs about 1,150 iterations: every ``vld``,
``vmatload``, push, pop, ``vst`` and two per ``vadd`` row. The price
decomposes as:

* the A bypass (inc 1): +64;
* accumulate-by-``vadd`` (inc 2): +466;
* one VREG file (inc 3 and 3b): +298;
* the output path and the DMA ISA (inc 4 and 5a): +7.

**At T=8** (MAXDIM=16, ``logs/align_final_cosim_t8.log``): 8x8x8 302,
16x16x8 474, 16x16x16 699 cycles, 0 wrong. ``TPU_TB=stress`` is 0 wrong over
six calls at 8x8x8 and 16x16x16. T=16 was not cosimulated: 518 processes; v1's
T=16 csynth is on :doc:`tinytpu_isa`.

Where the cycles would come back, none of it built yet:

* issue V, M and MEM work of one bundle concurrently. This is MiniTPU's
  three read ports, and Phase 3's bundle semantics;
* ``vadd`` in one iteration a row, which needs a second read port, i.e. a
  replicated file;
* overlapping DMA with compute (inc 5b).

**What the cycle costs in RTL, beyond cycles.** Two things only RTL shows,
both found by increment 3's cosim:

* Vitis runs a dataflow region's C model one process after another, so cosim's
  C-testbench pass aborts on a cyclic graph (``threaded_csim.py`` runs the C
  model's processes on threads).
* A pipelined loop that blocks on a read under Vitis's default *stall*
  pipeline freezes its older iterations' puts. The ``vpu``, every PE and
  every weight loader must be pipelined ``style=flp``. The PEs are the
  subtle case. A PE blocked reading its *next* activation row, which the
  ``vpu`` will not push until it has popped, holds the previous row's
  forwarding puts, and the pop never completes. With only the ``vpu`` flushable,
  cosim deadlocked at 16x16x16. It passed at 4x4x4 only because the last
  pushed rows end the PE loops, which drains them. ``s.pipeline(style=)``
  was added to Allo for this (``0038833c``).

.. _align-verification:

Verification of every increment
-------------------------------

Every increment passes, before the next is started:

* ``bench_isa.py``: ALL EXACT (looped and flat GEMM, with and without relu, at
  every scored shape, plus ``vadd_program``);
* ``stress_isa.py``: STRESS OK. This covers full-range, corner and boundary
  operands, every GEMM shape, ``C`` prefilled and compared in full,
  ``vector_program``, ``ar_distance_program`` at the contract, and 200 random
  valid programs, all against ``isa_ref.py``. It also includes the validator's
  controls (every crafted bad program rejected, every generated one accepted)
  and ``kpn_model.py`` on every distinct program;
* ``mutate.py --no-rtl``: every mutant caught (the anchors are re-targeted to
  each increment's code, and new mutants are added for each new mechanism);
* the RTL-only dependence-claim mutant (``ar_claim_false`` / ``vr_claim_false``),
  caught by ``TPU_TB=stress`` cosim;
* Vitis cosim at 4x4x4 and 16x16x16: default testbench (the cycle counts) and
  ``TPU_TB=stress`` (six calls per shape on one RTL instance, ``C``
  prefilled, 0 wrong).

**Parametric in T and MAXDIM.** The design and the harness were made to derive
every shape, address, memory size and crafted program from ``T`` and
``MAXDIM`` (``SCORED_SHAPES`` = T, 2T, 3T, 4T x 4T x 2T, 4T; at T=4 the five v1
shapes). From increment 2 on, the functional gates are run at **T=4 /
MAXDIM=16, T=8 / MAXDIM=16 and T=8 / MAXDIM=32**, and cosim at T=8 as well.
Doing so found a real T-dependence in increment 2 (``aw[0:VW] = vv``: Allo
infers a slice's width from ``upper - lower`` with global names as free
symbols, so the width silently became 32 bits, right only at T=4), and a
simulator bug (a ``Stream`` of ``UInt(65)`` corrupts the simulator's heap at
T=8; 72 and 128 bits run, 96 hangs), now avoided by carrying the weight-switch
flag on its own 8-bit chain.

.. _align-status:

Status: done, pending, and the decisions needed
-----------------------------------------------

**Done on** ``tinytpu-align``: increments 1, 2, 3, 3b, 4 and 5a, each verified
as above. Also done:

* ``s.pipeline(style=)`` in Allo (``0038833c``);
* ``threaded_csim.py``, for cosim of a cyclic region;
* a design and harness parametric in T and MAXDIM;
* limitations-register items P1-P4 (:ref:`limitations-align`).

**Pending (not started):**

* **Inc 5b, asynchronous DMA.** The VMEM owner would run its DMA port
  concurrently with compute. That means a ``while`` loop with non-blocking
  stream I/O, a two-entry descriptor queue activated at its issue point (WAR
  interlocked), and ``wait`` stalling only compute. Everything it needs exists
  in Allo (``empty``/``full``/``try_get``/``try_put``) except loop directives
  on a ``while`` loop (P1, still open). It changes timing only: 5a already has
  the contract.
* **V-op set.** MiniTPU has ``vsub``, ``vmul``, ``vmax``/``vmin``, ``vmov``,
  reductions, transpose and the SFU; this design keeps ``vadd`` and
  ``vrelu``. ``vrelu`` is ``vmax`` against a zero register.
* **VMEM/VREG geometry.** Rows here are T lanes, one sublane (MiniTPU: a word
  is 4 sublanes). Instructions carry row counts where MiniTPU's ops are one
  VREG each. The per-op row granularity belongs to Phase 3's encoding.
* **Phase 3** (MiniTPU's bundle and slot semantics, DELAY as minimum issue
  spacing) was **not started**, as instructed: Phase 2 took most of the
  effort.

**Decisions needed:**

1. **Whether to switch** to TinyTPU-align for experiments at a cost of
   1.26x-2.22x v1's cycles (table above). The cost is the one-VREG-file
   serialisation plus MiniTPU's ``vadd`` accumulation, and both could be
   bought back by bundle-level issue.
2. **The datatype** (:ref:`align-datatype`). BF16 bit-exact to MiniTPU is
   integer soft-float in Allo, with no Allo fix. BF16 through Allo's float
   types needs the emitter fix (60-100 lines) and gives ``ap_float``
   rounding, which is not MiniTPU's.
3. **Whether Phase 3 goes ahead**, which is also where the cycles above
   would come back.

**What broke at T != 4 on v1** (read-only diagnosis of ``tinytpu-isa-v1``):
v1's **hardware is T-parametric**: its GEMM is exact on the dataflow
simulator at T=8 with MAXDIM 16 and 32, up to 32x32x32. What fails is the
harness, and each failure is small:

* ``bench_isa.py`` and ``cosim.py`` hard-code the five T=4 shapes;
  4x4x4 and 12x12x12 trip ``gemm_program``'s ``M % T`` assertion;
* ``isa_dsl.vector_program`` and ``ar_distance_program`` hard-code column
  blocks 2 and 3, past ``WPR = MAXDIM / T = 2``, so ``check_program`` rejects
  them;
* ``stress_isa.py``'s random ``mvout_loop`` walks up to column block 2.

None of it is structural.

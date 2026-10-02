..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###########################################################################
U3 prep: peek check on the RTL, and the two-kernel ``Stream`` as the MXU FIFO
###########################################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch
   ``u2-fifo-composed`` from ``origin/u1-pilot`` at ``8982bd4c``. Bindings: a
   private copy of ``wt-u1``'s build (``ldd`` resolves
   ``libAlloMLIRAggregateCAPI`` inside the worktree). MiniTPU at ``b3ba0a4d``,
   read only. Verilator 5.052, Catapult 2024.2 (``nangate-45nm_beh``), DC
   W-2024.09 (FreePDK45, ``dc_u1.tcl`` byte-identical to the U1/U2 records').
   Nothing under ``allo/`` or ``mlir/`` changed. Owner's decisions in force:
   checkpoint 6 (``u1_matrix.rst``: the composed MXU measures a two-kernel
   ``Stream`` against ``vpu_fifo`` before the form is chosen; ``peek()`` is
   decided by whether the MXU controller holds the head word), D-10 (latency
   from the manifest, checked on the RTL), D-11 (single-call traces).

Two measurements for the U3 composite, both on the two ``vpu_fifo`` instances
the MXU owns: the **input FIFO** (``mxu.sv:97``, ``1 + DIM*16 = 257`` b x 4,
shared by all lanes) and the **output FIFO** (``mxu.sv:154``, ``64`` b x 16,
one per lane). Part A asks the RTL whether either consumer *peeks* (reads
the head without popping it, or pops because of what the head holds). Part
B models each instance as an Allo ``Stream[T, depth]`` **between two
kernels** -- the form the MXU composite would use -- and takes it through
every column: simulator, SystemC csim, Catapult csyn, RTL against
``vpu_fifo.sv`` in Verilator, the latency manifest, and same-flow DC area
beside MiniTPU's FIFO and beside the explicit ring of the U2 FIFO record.

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first)::

   R=dev/records/minitpu/u3_fifo_composed_2026-10-02
   $ALLO_PYTHON $R/scripts/peek_check.py                                  # Part A, both tbs
   $ALLO_PYTHON $R/scripts/cmp_composed.py --inst w32d4 --backend simulator --backend systemc
   $R/scripts/run_csyn.sh composed_w32_3p33_n40057 composed 32 3.33 --n 40057   # whole region
   $ALLO_PYTHON $R/scripts/hs_cmp.py scratch/u3fc/cat/composed_w32_3p33_n40057.prj --inst w32d4
   $ALLO_PYTHON $R/scripts/hs_cmp.py scratch/u3fc/cat/composed_w32_3p33.prj --trace overflow --dump
   $R/scripts/dc/run_dc.sh c_fifo_w32_3p33 AlloFifoC_ac_int_32_false_4 clk 3.33 <concat_rtl.v>
   $ALLO_PYTHON $R/repros.py

Part A: does any consumer peek?
===============================

**Verdict: ``try_get`` suffices for both instances. Neither consumer holds
the head word across cycles before popping, and neither decides whether or
when to pop from the head's value.**

Input FIFO (``mxu.sv:97``, consumer: the array load path in ``mxu.sv``)
-----------------------------------------------------------------------

* The pop is unconditional on the head: ``consume_input = !input_empty``
  (``mxu.sv:111``), ``input_accept_o = consume_input`` (``:112``). The FIFO
  is popped in every cycle it is not empty; no field of the head word enters
  that decision.
* The head is consumed in the same cycle as the pop, into registers clocked
  by the same edge that moves ``rd_ptr_q``: ``{lane_kind, input_fifo_data} =
  input_payload`` (``:106``, ``input_payload`` is ``pop_data_o``, ``:102``);
  ``lhs_skew_data_q[lane][0] <= lane_input_data[lane]`` (``:220``) and
  ``lhs_skew_valid_q[lane][0] <= consume_input && lane_kind == MXU_INPUT_LHS``
  (``:211``) for activations; ``rhs_edge = lane_input_data`` (``:231``) with
  ``rhs_edge_valid = rhs_beat && load_bank_q == bank`` (``:233``),
  ``rhs_beat = consume_input && lane_kind == MXU_INPUT_RHS`` (``:172``), into
  the array's weight registers; ``tile_starts = weights_waiting_q &&
  consume_input && lane_kind == MXU_INPUT_LHS`` (``:170``). The head's
  ``kind`` field steers *where* the word goes in the pop's own cycle, never
  *whether* it is popped.
* The producer never checks ``full``: ``mxu_input_push_o = busy_o``
  (``mxu_stream_engine.sv:63``); the stream engine's own comment is "the
  input FIFO drains every cycle, so a row is accepted the cycle after it is
  pushed" (``:56``), and an overflow is an assertion in the controller
  (``mxu_matrix_ctrl.sv:93-94``), not a stall.
* Trace (``logs/peek_check.txt``): ``tb_mxu_single_port`` (``mxu`` alone,
  DIM = 2): 6 pops, each in the first cycle the head was visible, push to
  pop 1 cycle. ``tb_matrix_command_ii`` (the whole ``vpu``: stream engine,
  pop engine and MXU): 144 pops, all 144 in the first visible cycle, push to
  pop 1 cycle; ``full_o`` never asserted; the FIFO was non-empty for exactly
  144 cycles -- it never holds more than one word. The 4-deep FIFO is, in
  use, a one-entry pipeline register.

Output FIFO (``mxu.sv:154``, consumer: ``mxu_pop_engine.sv`` through ``vpu.sv``)
--------------------------------------------------------------------------------

* The pop is ``beat_fires = popping_q && mxu_output_valid_i``
  (``mxu_pop_engine.sv:35``), ``mxu_output_pop_o = beat_fires`` (``:41``):
  a vmatpop has been issued (``popping_q``, ``:52-54``) and every lane's
  FIFO is non-empty (``output_valid_o = &(~lane_output_empty)``,
  ``mxu.sv:109``; ``consume_output = output_pop_i && output_valid_o``,
  ``:110``). The head's *value* is not in the decision.
* The data is consumed in the pop's cycle: ``vreg_write_data_o =
  mxu_output_data_i`` and ``vreg_write_valid_o = beat_fires``
  (``mxu_pop_engine.sv:45``, ``:42``), combinational from the FIFO's
  ``pop_data_o`` (wired continuously, ``mxu.sv:159``, ``:165-166``; the
  ``mxu_deserializer`` between them is a pass-through,
  ``mxu_serializer.sv:29-30``), written into the VREG by the same edge.
* Trace (``tb_matrix_command_ii``, lane 0): 28 pops; ``mxu_output_pop_o ==
  popping_q && mxu_output_valid_i`` on every one; vmatpop issue to pop 1
  cycle for 23 of them, 2-4 cycles for 3 (a vmatpop issued before its
  result: ``popping_q`` waited on ``valid`` and popped in the cycle it rose,
  as ``:34`` documents). The head word sat *in the FIFO* unread for 0 to 115
  cycles (505 cycles in all) before its vmatpop: that gap is the program's,
  not a hold by the consumer, which never samples ``output_data`` outside
  the pop cycle.
* Caveat for anyone reading ``tb_mxu_single_port``: the *testbench* does
  peek (it checks ``output_data`` after ``wait (output_valid)`` and raises
  ``output_pop`` a cycle later, ``tb_mxu_single_port.sv:60-68``). That is
  the tb's convenience, not the hardware consumer's behaviour.

Consequence for ``Stream.peek()``: not demanded by the MXU. The input FIFO
is "``try_get`` every cycle"; the output FIFO is "``get`` when a vmatpop is
pending and every lane's stream is non-empty" -- the all-lanes condition is
an ``empty()`` poll per lane stream, not a peek.

Part B: the two-kernel ``Stream`` against ``vpu_fifo``
======================================================

The form (``examples/minitpu/units/vpu_fifo.py``, variants ``composed`` and
``composed_try``): one ``Stream[W, d]`` between ``producer`` (the push half
of the trace: ``rst``/``push``/``push_data`` per cycle, a blocking ``put``
on every push, ``full`` = ``q.full()`` per iteration) and ``consumer`` (the
pop half: a blocking ``get`` on every pop, ``pop_data`` = the word got on
that cycle and 0 otherwise, ``empty`` = ``q.empty()`` per iteration). No
head register: Part A says no consumer needs one. ``composed_try`` is the
same with ``try_put``/``try_get`` (the illegal-program demonstration). No
workaround beyond S6 (every port read unconditional).

**The verdict trace** (``traces_composed``): the legal directed traces and
two random legal traces per instance with **reset only at the start**, each
segment **drained** before the next (``_drained``), plus the
``tb_mxu_single_port`` seeds for ``output``. Two things differ from the U2
record's legal set, both consequences of what a Stream is: a mid-stream
``rst_ni`` is a pointer clear on ``vpu_fifo`` and nothing on a Stream (the
two kernels are not even cycle-locked to the trace), so it is a separate
demonstration (3 below), not a verdict slot; and a segment that ends with
words inside followed by a reset is the same thing in disguise (the first
joined trace did exactly that and hung the simulator and stalled the RTL at
the end, both correctly: ``logs/hs_w32d4_n40044_undrained.txt``). w32d4:
40,057 cycles, 11,694 pops; output: 40,206 cycles, 13,120 pops; input:
10,055 cycles, 2,767 pops.

**What is compared** (the consumer-visible contract after Part A):
``pop_data`` on the pop cycles only, aligned exactly (the k-th pop of the
trace is the consumer's k-th ``get``); ``empty`` (the consumer's view) and
``full`` (the producer's) on every defined cycle at the shift ``s`` in
[-8, 8] where ``got[t] == rtl[t + s]`` agrees best, with the agreement at
``s = 0`` beside it. The head-visible slots the U2 record counted
(``pop_data`` on non-pop cycles) are not in the contract.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 10 11 21 21 37

   * - instance
     - column
     - ``pop_data`` on pop cycles
     - ``empty`` / ``full``
     - notes
   * - w32d4
     - simulator
     - **match** 11,694/11,694
     - 25,201/40,056 at s=+5; 23,315/40,056 at s=+4
     - flags are order-only on the untimed simulator (each kernel runs as far ahead as the
       stream lets it)
   * -
     - SystemC csim
     - **match** 11,694/11,694
     - **38,647/40,056 at s=0** (96.5 %); 27,219/40,056 at s=-1
     - ``full`` is the producer's own view and drifts with its stalls (below)
   * -
     - Catapult RTL
     - **match** 11,694/11,694
     - **39,909/40,056 at s=-2** (33,438 at s=0); 36,209/40,056 at s=-2
     - 40,295 cycles for 40,057 iterations: 238 stall cycles; push-accept to pop-deliver
       **3 cycles** on all 2,670 pops that follow their push by one trace cycle
   * - output (64x16)
     - simulator
     - **match** 13,120/13,120
     - order-only
     -
   * -
     - SystemC csim
     - **finding S8** 0/13,120
     - 39,403/40,205 at s=0; 28,881/40,205 at s=-1
     - every 64-bit datum reads back ``0x7fffffffffffffff`` (the U2 record's S8, unchanged);
       d16w48 (48 b x 16) stands in: **match** 3,475/3,475, ``empty`` 9,922/10,137 at s=0
   * -
     - Catapult RTL
     - **match** 13,120/13,120
     - **40,174/40,205 at s=-2**; 37,251/40,205 at s=-2
     - the 64-bit port reads back exactly in Verilator (S8 is the csim testbench's);
       40,315 cycles / 40,206 iterations; 3 cycles on all 1,835 one-cycle pairs
   * - input (257x4)
     - simulator / csim
     - **blocked** (B6 / S7)
     - --
     - as the U2 record: no numpy dtype (refused by ``_args``), ``KeyError 'ui257'`` class
   * -
     - Catapult RTL
     - **match** 2,767/2,767
     - **10,017/10,054 at s=-2**; 9,066/10,054 at s=-2
     - csyn after the S7 testbench hand-patch (3 casts, ``catapult/*/hand_patches.diff``);
       10,111 cycles / 10,055 iterations; 3 cycles on all 675 one-cycle pairs

Catapult, whole region (``producer_0`` + ``AlloFifoC`` + ``consumer_0``, both
loops ``s.pipeline`` at II=1, n = the trace length so the RTL runs the whole
trace -- a kernel with array args runs exactly its baked-in iteration count,
unlike the arg-less ``fifo_0`` of the U2 record):

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - build
     - csyn
     - manifest (D-10) and timing
   * - w32d4, 3.33 ns
     - ok, 41-49 s
     - ``producer_0`` latency 1, ii 1; ``consumer_0`` latency 2, ii 1; both ``scheduled``,
       ``connections``. Composite 1 + 2 = 3 = measured 3. Catapult slack 2.38 ns
   * - w32d4, 2.0 ns
     - ok, 39 s
     - the same schedule; slack 1.05 ns
   * - output 64x16, 3.33 ns
     - ok, 41-42 s
     - the same schedule; slack 2.22 ns
   * - input 257x4, 3.33 ns
     - ok, 44-47 s
     - the same schedule (S7 TB patch); slack 2.32 ns
   * - ``composed_try`` w32d4, 3.33 ns
     - ok, 39 s
     - ``producer_0`` latency 2 (``PushNB`` at c-step 1, ``full`` out at 2), ``consumer_0``
       latency 2 (``PopNB`` at 1); both ``scheduled``

**(1) Does Catapult schedule the two-kernel Stream?** Yes: five builds, three
widths, two clocks, blocking and non-blocking ends, all II=1 with no SCHD
warning. C1 of the U2 record (SCHD-30 on the *self*-FIFO) is confirmed to
be the self-loop, not the ``AlloFifoC`` primitive: the same ``AlloFifoC``
between two modules is an ordinary Connections FIFO to Catapult.

**(2) Ports and handshake wires.** The Stream becomes one RTL module
(``AlloFifoC_ac_int_32_false_4``: ``clk, rst, enq_vld, enq_rdy, enq_dat,
deq_vld, deq_rdy, deq_dat, empty_o, full_o``). Against ``vpu_fifo``'s
``push_i, push_data_i, pop_i, pop_data_o, empty_o, full_o`` it adds two
ready/valid pairs: ``enq_rdy`` (= ``!full``) and ``deq_vld`` (= ``!empty``)
go back to the kernels as back-pressure, and ``enq_vld``/``deq_rdy`` are
what ``push``/``pop`` become. The ``empty_o``/``full_o`` sideband is emitted
only because the kernels poll ``empty()``/``full()`` (``AlloFifoC``; a
Stream nobody polls is a plain ``Connections::Fifo``). What the wires
*mean* is the difference: ``vpu_fifo`` takes every push and pop as a
command and the schedule keeps them legal; the Stream holds a push the
FIFO cannot take and a pop it cannot serve. Two consequences measured on
the legal trace: a push into a full FIFO with a pop in the same cycle
(pass-through on full, legal in ``vpu_fifo.sv:52``) is **refused for one
cycle** by Connections (``enq_rdy`` is the registered ``!full``), so the
producer stalls -- the 238 / 109 / 56 extra cycles above; and the
push-to-pop path is **3 cycles** (producer ``Push`` at c-step 1, the FWFT
FIFO presents it the next cycle, consumer ``Pop`` at c-step 1 and
``pop_data`` out at c-step 2) where ``vpu_fifo`` shows the word one edge
after the push. Hence the per-port offsets are not constant: ``cycle -
iteration`` runs 0, 1, 2, ... for both kernels, growing at every stall
(``logs/hs_verdicts.txt``), and the flags agree best two cycles late
(``s = -2``) with ``empty`` at 99.6 % and ``full`` at 90 % (``full`` is the
producer's view at the producer's own time, which lags the trace by its
accumulated stalls).

**(3) Reset.** ``vpu_fifo`` clears ``rd_ptr_q``/``wr_ptr_q``/``count_q`` on
``!rst_ni`` (``vpu_fifo.sv:47-50``) and the words are gone. A Stream has no
such operation: ``rst_ni`` low mid-trace is a data input the kernels ignore,
and the words stay. Demonstrated on the RTL (``reset-mid-stream``,
``logs/illegal_demos.txt``): three words pushed before the reset, the first
pop after it returns the stale first word (``7d280fc3``) where ``vpu_fifo``
returns the word pushed after the reset; ``pop_data`` 1/2. The only reset a
Stream has is the region's ``rst`` port, which resets ``AlloFifoC``'s
pointers (pointer-only, as MiniTPU's) **and** both kernels. Draining
(``while not empty: get()``, as the U2 ``stream`` variant did) is the
consumer's act, racy against a producer that is not cycle-locked. For the
MXU this is moot in practice: the controller's reset is the core's reset.

**(4) Illegal programs** (``w32d4`` n = 64 builds, ``logs/illegal_demos.txt``):

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - trace
     - ``vpu_fifo`` (RTL)
     - ``composed`` (``put``/``get``)
     - ``composed_try`` (``try_put``/``try_get``)
   * - overflow (7 pushes into depth 4, then 5 pops)
     - 3 pushes **dropped** silently (assertion in simulation)
     - producer **stalls** 4 cycles (rows 8-11: no input accepted) until the pops, nothing
       dropped, all 7 words delivered in order; a deadlock only if no pop ever comes
     - 3 pushes **refused** (``full`` = 1 on iterations 8-10) and dropped as the RTL's,
       ``pop_data`` 4/4 on the defined pops
   * - underflow (pops on empty)
     - pop ignored, stale ``pop_data``
     - consumer **deadlocks** on ``get`` (STALL after 100 idle cycles: ``pop_i`` accepted
       5/64, ``pop_data`` delivered 3)
     - ``try_get`` returns nothing (``empty`` = 1, ``pop_data`` = 0), no stall
   * - push + pop on empty (H12)
     - both pointers move, the word is **lost**
     - the word is kept (M3), consumer waits 1 cycle for it
     - the same
   * - reset mid-stream
     - pointers cleared
     - words kept (3)
     - the same

**(5) Area and timing** (DC W-2024.09, FreePDK45, same flow and script as
the U2 record; every design meets its clock; the MiniTPU baseline was re-run
today and reproduces to the digit):

.. list-table::
   :header-rows: 1
   :widths: 42 8 14 12 12 12

   * - design
     - clock
     - area (um^2)
     - comb
     - seq
     - slack (ns)
   * - MiniTPU ``vpu_fifo`` 32x4 (U2 record) / + output registers (re-run)
     - 3.33
     - 790.0 / **943.8**
     - 159.6
     - 630.4 / 784.2
     - 2.62 / 2.67
   * - U2 explicit ring, Catapult ``wire`` 32x4 (U2 record)
     - 3.33
     - 1,905.6
     - 620.8
     - 1,284.8
     - 0.04
   * - **Stream alone**, ``AlloFifoC_ac_int_32_false_4``
     - 3.33 / 2.0
     - **867.4** (-8 % vs oreg, +10 % vs bare; -54 % vs ring)
     - 143.9
     - 723.5
     - 2.72 / 1.41
   * - MiniTPU 64x16 / + oreg (U2 record)
     - 3.33
     - 5,823.0 / 6,121.5
     - 1,057.9
     - 4,765.1 / 5,063.6
     - 2.25 / 2.32
   * - U2 ring, ``wire`` 64x16
     - 3.33
     - 9,741.5
     - 3,272.3
     - 6,469.1
     - 0.03
   * - **Stream alone**, ``AlloFifoC_ac_int_64_false_16``
     - 3.33
     - **6,640.2** (+8 % vs oreg, +14 % vs bare; -32 % vs ring)
     - 1,072.8
     - 5,567.4
     - 2.30
   * - MiniTPU 257x4 / + oreg (U2 record)
     - 3.33
     - 5,675.6 / 6,846.3
     - 975.4
     - 4,700.2 / 5,871.4
     - 2.23 / 2.25
   * - U2 ring, ``wire`` 257x4
     - 3.33
     - 11,679.5
     - 3,212.7
     - 8,466.8
     - 0.09
   * - **Stream alone**, ``AlloFifoC_ap_uint_257_4``
     - 3.33
     - **6,509.8** (-5 % vs oreg, +15 % vs bare; -44 % vs ring)
     - 998.3
     - 5,511.5
     - 2.21
   * - whole composed region ``top`` 32x4 / 64x16 / 257x4 (n = 64 builds)
     - 3.33
     - 2,086.5 / 8,657.8 / 13,418.1
     - 390.8 / 1,441.2 / 2,146.4
     - 1,695.8 / 7,216.6 / 11,271.8
     - 2.59 / 2.25 / 2.17

Like for like is between the bare ``vpu_fifo`` and the Stream alone: both
present the head combinationally from registered storage (FWFT), and the
Stream's extra 10-15 % is its handshake (``enq_rdy``/``deq_vld`` registers,
the status method) plus a wider pointer/count. The whole-region ``top`` adds
the two trace-replaying kernels and Catapult's port wrappers and is not a
FIFO number. The Stream has 2.2-2.7 ns of slack at 3.33 ns where the U2 ring
had none: the ring was scheduled to the clock, the Stream is a fixed
Connections component.

Findings
========

**C1 narrowed (Catapult, U2 record).** The self-FIFO's SCHD-30 is the
self-loop. The same primitive between two kernels schedules at II=1 in every
build here, with ``empty()``/``full()`` polled and with non-blocking ends.
Proposed resolution of the U2 decision 2: refuse a *self*-FIFO under csyn
(or lower it to a ring); the cross-kernel Stream needs nothing.

**M4 (semantic mismatch, measured): no pass-through on full.** Connections'
``enq_rdy`` is the registered ``!full``, so a push with a simultaneous pop
on a full FIFO is held one cycle (``vpu_fifo.sv:52`` takes it). Cost on the
legal traces: 0.6 % extra cycles at w32d4 (238/40,057), 0.3 % at 64x16,
0.6 % at 257x4. For the MXU input FIFO it cannot occur (it never holds more
than one word, Part A); for the output FIFO it is the overflow case, below.

**M5 (semantic mismatch, measured): the output FIFO's overflow is a drop,
not back-pressure.** ``mxu.sv:44``: "Output full is never checked and
nothing stalls on it"; a push past full drops the result row
(``isa_latency.json`` ``mxu_output_fifo.note``: "Overflow drops results
silently; there is no replay"). A blocking ``put`` there would back-pressure
the gather and, behind it, the systolic array, which cannot stall; the MXU
composite must therefore use ``try_put`` on the output streams (refuse =
drop, with the refusal observable), or keep the explicit ring. ``put`` is
right for the input FIFO only because its producer never overflows
(``mxu_stream_engine.sv:56``).

**The simulator's ``try_*`` are scheduling-dependent, not a bug.** The
``composed_try`` smoke run on the simulator refused every push after the
fourth and then read ``full`` as 1 on every later iteration: the producer
thread had run all its iterations before the consumer started. The ordered
probe in ``repros.py`` (the consumer released by a second stream after the
producer's six ``try_put``) gives exactly ``[1,1,1,1,0,0]`` for the puts and
four words then nothing for the gets. B7 (unused ``try_put`` dropped) did
not recur because every result here is consumed.

**S8 recurs** on the 64-bit ``output`` instance in csim (0/13,120), as
recorded in U2; Verilator reads the same RTL port exactly, so the Catapult
column carries the verdict for that instance.

**Harness.** ``hs_cmp.py`` is a Connections-port driver for a whole region
(every input offered until accepted, every output always ready, per-cycle
record of fires and data, wide ports through ``VlWide``) with a stall
detector; ``rtl.py``'s ``stream`` shape drives one output stream of <= 64 b
and could take this as a generalisation. ``emit_csyn.py`` gained
``--patch-s7`` (the TB cast fix for ports wider than 64 b, which the U2
record applied by hand).

Recommendation for U3's MXU FIFOs
=================================

1. **Input FIFO: a ``Stream[UInt(257), 4]`` between the stream-engine
   kernel and the array-load kernel, consumed with ``try_get`` every
   cycle.** No peek, no head register, no ring. Its depth is a parameter
   that never matters (one word in flight, Part A); keep MiniTPU's 4 so the
   pointer widths match the RTL's. The push-to-pop path is 3 cycles against
   MiniTPU's 1: the composite's latency contract (D-10, ``latency=`` on the
   kernels) must absorb 2 cycles here or the array-load kernel must take the
   word through a ``Channel``/``Wire`` instead -- to be decided at U3 with
   the cycle counts, not here.
2. **Output FIFOs: ``Stream[UInt(64), 16]`` per lane with ``try_put`` from
   the gather and a blocking ``get`` in the pop engine gated on a vmatpop and
   on every lane's ``empty() == 0``.** ``try_put`` reproduces the RTL's drop
   (M5) with the refusal visible; ``put`` would back-pressure the array.
   The area is within 8 % of MiniTPU-plus-output-registers and a third
   below the ring, and Catapult schedules it (1).
3. **Reset**: the region's reset is the FIFO's reset (pointer-only on
   ``AlloFifoC``), matching MiniTPU; no drain code in any kernel.
4. **``Stream.peek()``: not needed by the MXU.** Keep H6 as a proposal only.
5. **The explicit ring stays the standalone-verdict unit** (checkpoint 6)
   and the fallback for any consumer that does turn out to peek.

Files
=====

* ``examples/minitpu/units/vpu_fifo.py``: ``composed``, ``composed_try``,
  ``traces_composed`` (with ``_drained``), ``random_trace(resets=)``.
* ``scripts/``: ``peek_check.py`` (Part A), ``cmp_composed.py`` (simulator /
  csim), ``hs_cmp.py`` (Catapult RTL through its Connections ports),
  ``emit_csyn.py`` (U2's plus ``--patch-s7``), ``run_csyn.sh``, ``dc/``
  (``run_dc.sh``, ``dc_runs.txt``, ``dc_u1.tcl``, the MiniTPU wrappers),
  ``hang_bisect.py`` (the undrained-trace bisection, kept as the method).
* ``repros.py``: the simulator ``put``-past-full and ``try_*`` probes.
* ``catapult/<build>/``: ``run.tcl``, ``csyn.log.gz``, ``rtl.rpt.gz``,
  ``cycle.rpt``, ``latency.json``, ``hand_patches.diff`` (257 b);
  ``composed_w32_3p33/`` also ``kernel.cpp`` and ``concat_rtl.v.gz``.
* ``dc/<run>/``: area, qor, timing, reference reports and the wall time.
* ``logs/``: ``peek_check.txt``, ``cmp_composed_*.txt``, ``hs_verdicts.txt``,
  ``hs_w32d4_n40044_undrained.txt``, ``illegal_demos.txt``,
  ``latency_manifests.txt``, ``csyn_runs*.txt``, ``dc_runs.txt``.

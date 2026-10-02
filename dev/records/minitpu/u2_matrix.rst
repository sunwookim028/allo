U2 findings matrix (README D-9)
===============================

One row per unit, one column per tool, in ``u1_matrix.rst``'s format. Cells:
**match**, **finding** (with its class: bug, missing abstraction, workaround,
semantic mismatch), **blocked**, **refused** (a backend refusing what it
cannot honour, D-1's required behaviour), or **n/a**. Started 2026-10-02 on
zhang-21 with the ``vpu_regfile`` pilot (branch ``u2-regfile``); MiniTPU at
``b3ba0a4d``. A storage unit is checked per cycle on the slots the trace
reference calls *defined*; undefined slots are masked and counted (owner
decision).

For the owner's triage (pilot checkpoint)
-----------------------------------------

The full list, with the agent's provisional defaults, is "What the owner must
decide before the rest of U2" in ``u2_regfile_2026-10-02.rst``. In short:

1. **B4** (core, silent): unsigned array indices are sign-extended on the
   simulator path. Fix now, B1-style. *[fix]*
2. **S6** (SystemC emitter, silent): a conditional argument-array read becomes
   a conditional stream ``Pop()``. Plus B5, B6, S7 (loud or crash). *[fix, S6
   first]*
3. **Asynchronous read = latency 1** on every generated RTL (Catapult
   measured: a uniform one-edge shift). Accept as a recorded deviation?
   *[yes]*
4. **``@ Stateful`` across calls** differs between simulator and csim/cosim
   (M1). *[single-call variants for verdicts]*
5. **Memory ports** (G1): "one memory, N ports, several state machines" has no
   expression any backend accepts. Draft the ``compose`` memory-port D-n?
   *[yes]*
6. **``Memory(latency=, depth=)``** dropped everywhere. *[refuse now]*
7. **Expressions for the rest of U2.** *[``trace``, ``ported``, ``wire``]*
8. **Joined traces** shrink the undefined census (3 vs 484). *[keep joined]*

``vpu_regfile`` (U2 pilot, w16; w256 where marked)
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``trace``/``ported``/``stateful``/``annotated``/``shared_sync``
     - **match** 180,780/180,780 defined (3 masked, ``uninit``), **after workarounds for two
       core bugs**: **finding, bug B4** (silent: an unsigned index is sign-extended, so
       ``UInt(5)`` 16..31 and ``uint8`` >= 128 index out of bounds; the as-written ``trace``
       matched only by consistent aliasing, and ``shared_sync`` segfaulted); **finding, bug
       B5** (loud: ``x: int32 = s.get()`` on a ``UInt(5)`` stream has no cast)
     - ``u2_regfile_2026-10-02.rst``; ``repros.py`` B4/B5. Untimed: cycle = iteration order,
       which is exactly what the RTL's read-old/write-next rule needs.
   * - Allo simulator, ``shared``
     - **finding, semantic mismatch** (M2): 8,502/180,780, no diagnostic; four kernels on a
       region-scope Stateful run unordered (readers first)
     - ``logs/simulator_w16_shared*.txt``. Ordered by token streams (``shared_sync``) it
       matches.
   * - Allo simulator, ``wire``
     - **blocked**: ``Wire`` reaches LLVM translation and fails (``missing
       LLVMTranslationDialectInterface ... func.func``) instead of being refused
     - SystemC-only construct, by design.
   * - Allo simulator, w256
     - **finding, bug B6**: a ``uint64`` array for ``UInt(256)`` is accepted with a warning
       and read past its end (heap corruption / SIGABRT); no way to pass > 64-bit elements (H8)
     - ``logs/w256_sim_csim.txt``; ``repros.py`` B6.
   * - SystemC csim, ``trace``/``ported``/``annotated``
     - **match** 180,780/180,780, **after a workaround for an emitter bug**: **finding, bug
       S6** (silent: ``if we[t]: mem[wa[t]] = wd[t]`` pops the ``wa``/``wd`` streams only when
       ``we``; as written 1,266/180,780)
     - ``logs/systemc_w16_trace_preS6.txt``; ``repros.py`` S6. Values per iteration, time
       unchecked.
   * - SystemC csim, ``stateful``
     - **finding, semantic mismatch** (M1): 170,093/180,780; each call is a fresh process that
       resets the Stateful
     - ``logs/systemc_w16_stateful_clean.txt``.
   * - SystemC csim, ``shared``/``shared_sync``
     - **refused** (D-1 honoured): "stateful variable ... is used by 4 kernels"; mixed
       region-argument sharing is refused too. Plan H2's "silent replica" is gone. **finding,
       missing abstraction G1**: no expression of one memory with several ports is accepted
     - ``repros.py`` H2.
   * - SystemC csim, ``wire``
     - **finding, semantic mismatch**: values right, per-port skew +4/+3/+3 cycles and lost
       tail slots; the src/sink kernels are not cycle-locked to the unit (limitation 22)
     - ``logs/wire_csim_offset.txt``.
   * - SystemC csim, w256
     - **finding, bug S7** (S5 class): ``KeyError 'ui256'`` reading back; the TB moves every
       port through ``long long``
     - ``logs/w256_sim_csim.txt``.
   * - Catapult, ``wire``
     - **finding, workaround** (H5): as emitted, ``mem`` maps to a 1R1W sync RAM and fails
       SCHD-30 at II=1. With Allo's ``s.partition(mem, Complete)`` (the emitter translates it
       to a ``[Register]`` map; no hand-patch): **match at read latency 1**, 180,780/180,780
       per cycle at 2.0 and 3.33 ns; w256 45,744/45,744 (after an S7 TB hand-patch).
       **finding, missing abstraction G2**: read latency 1 / write->read 2 measured against
       MiniTPU's 0 / 1 -- a uniform one-edge shift. Area +38 % (w16), +29 % (w256) over
       MiniTPU + output registers; all designs close both clocks
     - ``u2_regfile_2026-10-02/catapult/``, ``dc/``; ``scripts/cmp_trace.py``.
   * - Catapult, ``annotated``
     - **finding, bug (D-1)** (H3): ``Memory(LUTRAM, RAM_1WNR, latency=0, depth=32)`` is
       dropped -- emission byte-identical to ``trace``; nothing refuses
     - ``repros.py`` H3. Vitis keeps resource/storage only; latency/depth never reach the IR.
   * - RTLGen
     - **match** 180,780/180,780 as written (II=3: RTLGen itself builds a two-copy
       write-broadcast replica of ``mem`` with registered reads) and with ``partition``
       (II=1, 60,262 cycles). **finding, semantic mismatch M1**: ``Stateful`` does not
       persist across ``cosim()`` calls (179,534/180,780 chunked)
     - ``u2_regfile_2026-10-02/rtlgen/``.
   * - AMC
     - **match** 180,780/180,780 (as written ~4.1 cycles/iter; pipelined II=3, 180,821
       cycles). **finding, missing abstraction** (H10): the allocator declares 5 logical
       ports ``(w, r, r, r, w)``, all registered, and builds 2 physical (1RW + 1R); complete
       partition **blocked** (``problem is infeasible`` / ``exposed read port produced no
       rd_data signal``)
     - ``u2_regfile_2026-10-02/amc/``.

``vpu_fifo`` and the MXU output FIFO (w32d4; output = 64x16; input = 257x4)
-----------------------------------------------------------------------------

Record: ``u2_fifo_2026-10-02.rst``. The probe question -- is ``vpu_fifo``
a ``Stream`` with its ports? -- is answered there: a Stream is a link, not
a unit; behind a head register it is bit-exact in simulation (``stream``),
but Catapult cannot schedule the self-FIFO (C1), so every column that
reaches RTL uses the explicit ring (``trace``/``ported``/``wire``).

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``trace``/``ported``/``stream``
     - **match** on every instance it can run: w32d4 213,166/213,166, output 219,756/219,756,
       d16w48 55,028/55,028 (``empty`` masked and counted, 27,105 at w32d4). ``stream`` =
       ``Stream[W, d-1]`` + head register, one special case for H12 (M3)
     - ``u2_fifo_2026-10-02/logs/simulator_*``. The regfile's B4/S6 workarounds carried over.
   * - Allo simulator, ``stream_raw``
     - **finding, missing abstraction** (H6, no peek): 176,457/213,166; flags-only ``empty``
       76,248/80,091, ``full`` 79,710/80,091, all misses after an H12 event or a reset with
       words inside (M3). **finding, bug B7** (silent): a ``try_put`` with an unused result is
       dropped
     - ``logs/flags_only_w32d4.txt``; ``repros.py`` B7.
   * - Allo simulator, ``wire``; 257 b
     - **blocked**: ``Wire`` fails in LLVM translation (as the regfile); 257 b has no numpy
       dtype (B6, refused by the unit's ``_args``)
     - --
   * - SystemC csim, ``trace``/``ported``/``stream``
     - **match** w32d4 and d16w48 (same counts as the simulator). **finding, bug S8** (silent)
       on the 64-bit ``output`` instance: 160,478/219,756 for every variant -- the testbench
       parses a port through ``long long``, a value >= 2^63 and everything after it on that
       port read as ``0x7fffffffffffffff``
     - ``logs/systemc_*``; ``repros.py`` S8. 257 b: S7 (``KeyError`` class), as the regfile.
   * - SystemC csim, ``stream_raw``
     - **finding, missing abstraction** H6: 176,457/213,166, identical to the simulator.
       **finding, bug S9** (silent): a self-FIFO's ``full()`` counter advances on a refused
       ``try_put`` and never reads full again
     - ``repros.py`` S9.
   * - SystemC csim, ``wire``
     - **finding, semantic mismatch**: 150,379/213,166, 146,159/219,756; src/sink not
       cycle-locked to ``fifo_0`` (limitation 22)
     - as the regfile.
   * - Catapult, ``wire``
     - **match at +1 edge** on all three instances: w32d4 213,166/213,166 (3.33 and 2.0 ns;
       with and without partition: a 4- or 16-entry array is registers either way), output
       219,756/219,756, input 53,862/53,862 (after the S7 TB hand-patch). Probes: push to
       ``empty``/``pop_data``/``full``, pop to ``pop_data`` all 2 edges vs MiniTPU's 1 (G2).
       ``latency.json``: ``latency 2, ii 1, scheduled, wire`` = **LATENCY-MATCH** (D-10).
       Area (DC, same flow) vs MiniTPU + output registers: +102 % (32x4), +59 % (64x16),
       +71 % (257x4); B4's ``int32`` pointers are 351 um^2 of the 962 excess at 32x4
       (``wire_np``: +65 %). Every design meets 3.33 and 2.0 ns
     - ``u2_fifo_2026-10-02/catapult/``, ``dc/``, ``logs/cmp_wire.txt``,
       ``logs/latency_manifests.txt``.
   * - Catapult, ``stream``/``stream_raw``/``stream_nodrain``
     - **blocked; finding, bug C1 (D-1)**: SCHD-30 "could not schedule even with unlimited
       resources", pipelined or not, drain or not: the self-FIFO lowering (``AlloFifoC`` bound
       as a self-loop, ``_enq``/``_deq`` ports, synchronous ``_cnt``) is a chained feedback
       path; csim accepts it. Not refused by the emitter
     - ``catapult/stream_*/csyn.log.gz``.
   * - RTLGen
     - **match** 11,212/11,212 (4,096 cycles) at **II=1** as written and partitioned (identical
       netlists): 4 registers and a **combinational** read mux -- ``pop_data`` is asynchronous
       there (H4 refuted for RTLGen; port timing not observable through array ports)
     - ``u2_fifo_2026-10-02/rtlgen/``.
   * - AMC
     - **match** unpipelined 11,212/11,212 (16,392 cycles, 4/iteration). **finding, bug A1**
       (AMC): pipelined II=3 (12,299 cycles) and partitioned + pipelined both 9,467/11,212 --
       flags exact, ``pop_data`` wrong from the first fill; ``amcMemory0`` declares
       ``(w, r, w)`` on ``4xi32``, all registered
     - ``u2_fifo_2026-10-02/amc/``.

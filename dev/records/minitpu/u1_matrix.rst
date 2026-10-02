U1 findings matrix (README D-9)
===============================

One row per unit, one column per tool. Cells: **match**, **finding** (with its
class: bug, missing abstraction, workaround, semantic mismatch), **blocked**,
or **n/a**. Each finding links to its evidence. Started 2026-10-02 on
zhang-21; MiniTPU at ``b3ba0a4d``; harness on branch ``u1-pilot``.

For the owner's triage (D-9, second checkpoint)
-----------------------------------------------

What the pilot settled, and the calls that wait for the owner. Each call has
the agent's provisional default in brackets.

**What the pilot showed.** Every tool column has been tried on
``vpu_bf16_add``. The integer ``bits`` expression matches the RTL bit for bit
in the Allo simulator, RTLGen and AMC; Catapult's RTL from the ``native``
expression matches except on NaN encoding and one signed zero, and is 8 %
smaller than MiniTPU's RTL under the same DC flow. Nothing that works needed a
change to Allo's programming model. What failed were bugs (Allo core: unsigned
compares; SystemC emitter: seven; AMC: two silent miscompiles) and four gaps
in what Allo can say.

1. **Merge the unsigned-compare fix** (``core-uint-compare``, B1-B3)? Zero
   measured impact on every gate, test and TinyTPU emission. *[merge; file
   the drafted upstream issue]*
2. **Merge the SystemC emitter fixes** (``systemc-u1-fixes``, eight bugs,
   each with a compile-and-run regression test)? EVA bit-exact; TinyTPU
   SystemC csim unchanged; TinyTPU's Vitis and Catapult emission is
   byte-identical to ``main`` (sha256 ``6bc774bc...``/``ade1ab5d...``);
   ``tests/test_vhls.py`` gives the same 7 failures (host ``libstdc++``) and
   28 passes on both trees. With them, SystemC csim of ``bf16_add`` runs the
   full stimulus in 5.5 s: ``bits`` matches 251,936/251,936; ``native``
   differs only on NaN encodings (Catapult ``ac::bfloat16`` gives
   ``0x7FFF``/``0xFFFF``) and ``(+0)+(-0)``. *[merge]*
3. **NaN and signed-zero rules.** MiniTPU, Allo's simulator, Catapult and
   RTLGen give four different NaN encodings and two answers to ``(+0)+(-0)``.
   Options: (a) treat NaN payload/sign and the zero sign as don't-care in the
   harness; (b) give Allo a way to state a unit's NaN/zero policy on its float
   type; (c) require ``bits`` expressions wherever the RTL's rules matter.
   *[(a) for U1 verdicts, with the classification kept; (b) as a D-n
   proposal, since every machine has its own policy]*
4. **A combinational unit.** No backend can express "ports a, b, result, no
   clock": Allo's nearest is ``Wire`` ports (still clocked), RTLGen and AMC
   always add start/done. Is a zero-latency unit a programming-model feature
   to add, or is latency >= 1 acceptable as a recorded deviation for leaves?
   *[recorded deviation for U1; revisit at U3, where leaves compose]*
5. **Bit-level notation gaps** (concatenation, reduction-OR, ``[hi:lo]``
   slices; expression widths sized bottom-up; shifts by >= width). Add them,
   or keep them as recorded workarounds? *[proposal only; they cost
   readability, not correctness]*
6. **Pipeline flush mode for Catapult** (default stall leaves the last
   element of a finite stream stuck; ``style=`` refused for SystemC).
   *[emitter fix: emit the flush style for finite streams]*
7. **Report AMC's silent miscompiles** (A5 ``a or b or c`` drops ``c``; A7 a
   two-scalar loop exits with the wrong one) to AMC's author? *[yes, after the
   owner's go-ahead]*

``vpu_bf16_add`` (pilot)
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 250,955/251,936 match; the 981 others
       are the RTL's two non-IEEE rules (NaN always ``+0x7FC0``; ``(+0)+(-0) =
       -0``)
     - ``harness/check.py``; ``docs/source/designs/minitpu.rst`` ("The unit
       ladder"). Allo's ``bfloat16`` has IEEE semantics with no way to state
       a unit's own NaN/zero rules.
   * - Allo simulator, ``bits``
     - **match** 251,936/251,936, **after a workaround for a core bug**:
       **finding, bug B1** -- every ``<``/``<=``/``>``/``>=`` between unsigned
       integers is lowered as a *signed* ``arith.cmpi`` (``allo/ir/builder.py``
       picks the predicate from the MLIR type string, which is signless), so
       ``uint8`` ``200 > 100`` is 0 in the simulator and LLVM backend but 1 in
       HLS C++. Also B2 (bug: comparison typed as its operands, not ``uint1``),
       M1-M3 (missing abstractions: bit concatenation, reduction-OR, ``[lo:hi)``
       slices against SV's ``[hi:lo]``), M4-M5 (latent semantic mismatches:
       expression widths sized bottom-up; shifts by >= width)
     - ``u1_bf16_add_bits_2026-10-02.rst``. B1 confirmed independently. Fixed
       with B2 and a third bug in the same function (B3: every signed
       ``Fixed`` compare took unsigned predicates) on branch
       ``core-uint-compare`` (``716c7baf``, 17 new tests, 12 fail on main):
       no gate, test or TinyTPU emission changes against ``main``. Upstream
       has the same code. Held for the owner's review
       (``dev/records/limitations/uint_compare_2026-10-02.rst`` on that
       branch).
       Everything else in the RTL transcribed directly, including the nested
       ``leading_zeros17`` function.
   * - SystemC csim, ``native``
     - **finding, bug** x3 (emitter): the ``ac::bfloat16`` ``sc_trace``
       overload is in the global namespace, invisible to ADL, so any bf16
       Connections port fails to compile; the testbench feeds all of input 0
       before input 1 from one thread, so any unit with two interleaved
       streamed inputs deadlocks (0 outputs, even at n=200); float values
       cross the testbench as decimal text (``nan``/``inf``). For ``bits``
       (``uint16`` ports): S1, unsigned ports emitted as signed ``ac_int`` (any
       ``uint16`` region fails to compile); S2, a nested function returning
       ``UInt`` gets a signed result buffer
     - Fixes on branch ``systemc-u1-fixes``. The existing bf16 test only
       checked emission, so the compile error was never seen.
   * - Catapult csyn / RTL, ``native``
     - **finding, semantic mismatch** after two emitter bugs worked around:
       RTL 249,907/251,936 vs MiniTPU (249,908 vs IEEE; ties and subnormals
       exact); the rest are NaN encodings (Catapult ``0x7fff``/``0xffff``)
       and ``(+0)+(-0)`` (+0). II=1 latency 2 with ``s.pipeline``; 3 cy/vector
       without. Same-flow DC (FreePDK45, 3.33 ns, output-registered, wire
       ports): Catapult **813.4** vs MiniTPU **883.9** um^2. Bugs: include
       order (CRD-135 ``Marshall``), Wire-only kernel lacks ``wait()``
       (CIN-123). Missing abstractions: a combinational unit (closest is
       ``Wire`` ports, still clocked); pipeline stall/flush mode (default
       leaves the last element stuck; ``style=`` refused for SystemC)
     - ``u1_bf16_add_catapult_2026-10-02/README.md`` (nine findings). Emitter
       bugs passed to ``systemc-u1-fixes``. Catapult's own area score ranks
       the variants differently from DC.
   * - RTLGen, ``native``
     - **finding, semantic mismatch + missing abstraction**: 249,995/251,936
       match. The others are NaN encodings (RTLGen keeps sign and payload)
       and ``(+0)+(-0)``. The adder is an extern Vivado IP with no RTL body,
       so cosim checks a DPI-C model, and ``add_rtl_model`` is unimplemented
     - ``u1_bf16_add_rtlgen_2026-10-02.rst`` (F1, F2)
   * - RTLGen, ``bits``
     - **match**: 251,936/251,936 bit-exact, II=1, N+6 cycles. Findings: no
       combinational kernel (latency >= 1 with start/done); a ``for`` loop
       like the ``.sv``'s gives II=48 unless it is unrolled; 32-bit
       temporaries are not narrowed
     - ``u1_bf16_add_rtlgen_2026-10-02.rst`` (F3-F5)
   * - AMC, ``native``
     - **blocked**: no bf16 in its frontend or its operator library; f32
       needs DesignWare models that are not in the repository
     - ``dev/records/open_hls/amc_exploration_2026-10-02.rst``
   * - AMC, ``bits``
     - **match** 251,936/251,936 bit-exact, through AMC's own frontend and
       through our frontend's ``df.region`` MLIR. II=1 with N+2 cycles needs
       ``s.unroll`` + ``s.pipeline``; as written it takes 21 cycles per
       element (P1). Synthesis (N=16, 10 ns): 268 LUT, 21 FF. This holds
       **only after 7 kernel edits**, which come from these findings:
       **bugs** A1 (no ``not``), A2 (``x[k]`` dead on Python >= 3.9), A3, A4
       (scalar-returning call aborts), **A5 and A7 (silent miscompiles:
       ``a or b or c`` drops ``c``; a two-scalar loop exits with the wrong
       value)**, A8 (slice assignment lowers to a bit-serial loop, then a
       crash); **B1 is present in AMC's frontend too** (A6). Our MLIR also
       needs 4 text edits: R1-R3 are **semantic mismatches** between the
       forks (rank-0 scalars, ``pipeline_ii`` type, ``unroll`` vs
       ``loopschedule.parallel``), and R4 is a ``top`` name clash. One is
       ours: O1, invalid ``trunci`` in plain ``customize`` (B2)
     - ``u1_bf16_add_amc_2026-10-02.rst``; repros in
       ``u1_bf16_add_amc/repros.py``. These are AMC defects, to file with
       AMC (D-2) after triage. Also T1: at a 3.333 ns target AMC's delay
       model misses by 1.24 ns

``vpu_alu`` (composition probe)
-------------------------------

The ALU is built out of Allo engine units, as ``vpu_alu.sv`` is built out of
its adder and multiplier: the pilot's ``bits`` adder is *reused*, hoisted out
of its kernel into ``bf16_add.add_bits``. 5,079,552 vectors (all 16 op codes).
Composition findings C1-C11 are in ``u1_alu_2026-10-02.rst``, with repros in
``u1_alu/repros.py``. Two of them are **silent miscompiles in the front end**
that any cross-module reuse is exposed to (C3, C4). *[provisional: fix them
before more units are composed; proposals 2-3 in the record]*

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``bits`` / ``bits_dispatch`` / ``netlist``
     - **match** 5,079,552/5,079,552 each, **after workarounds**: B1 spare
       bits (``mul_bits``, ``bf16_gt``), B2 (C6), engine selection by def-name
       factories (C11, forced by **bugs** C1, C5 and the **missing
       abstractions** C2, C9)
     - Function reuse across modules works; an engine passed as a value does
       not link (C1). ``netlist`` (engines as ``@df.unit`` on streams) swaps
       engines by value, but a unit cannot be sized at instantiation and a
       stale size deadlocks (C9). **C3, C4: bugs, silent** (a reused
       function reads the caller's same-named globals; a module-level numpy
       array shadows a kernel parameter).
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 5,051,836 match. The rest: NaN
       encodings 3,810, MUL flush 6,065, ``(+0)±0`` 2, MAX/MIN with NaN 4,859
       (``maximumf``: NaN wins), and **C7**: 12,980 MOV/default-arm NaNs lose
       their payload through a ``phi bfloat`` (x86 ``__truncsfbf2``)
     - Engine swaps isolate each cause: ``engines_native`` 5,069,695,
       ``add_bits_mul_native`` 5,072,213 (MUL only), ``netlist_native``
       5,069,696.
   * - SystemC csim, ``bits`` / ``bits_dispatch`` / ``netlist``
     - **match** 5,079,552/5,079,552 each
     - 24.9-88.3 s. The emitter fixes merged in ``u1-pilot`` are enough;
       the engine hierarchy survives as C++ functions.
   * - SystemC csim, ``native``
     - **finding, bug + semantic mismatch**: 5,064,720 match. NaN 6,249
       (``ac::bfloat16`` ``0x7fff``), MUL flush 6,065, zeros 2, and **C8**:
       Allo ``max``/``min`` (``arith.maximumf``) are emitted as ``std::max``,
       which returns ``a`` on NaN or ``±0``, so 2,516 MAX/MIN vectors differ
       from the simulator's own answer
     - ``engines_native``/``netlist_native`` 5,067,236;
       ``add_bits_mul_native`` 5,072,282. C7 does not occur here.
   * - Catapult csyn, ``bits`` vs ``bits_dispatch``
     - **finding, missing abstraction** (C10): II=1 needs
       ``s.unroll("leading_zeros17:offset")``, the reused adder's internal
       loop, named by the ALU's build (``SCHD-30`` without it). Then both
       pipeline at II=1: area score **4138.1** (compute all, mux) vs
       **5211.9** (dispatch, +26 %: one adder per call site, no sharing
       across exclusive branches). rtl.rpt slack -1.34/-1.28 ns at 2.0 ns
     - ``u1_alu/catapult/``. csyn only; not simulated as RTL, no DC. The
       RTL's own form is the cheaper one to write, and Allo keeps whichever
       is written.
   * - RTLGen, AMC
     - **not tried**
     - Out of this session's scope.

Environment findings met on the way
-----------------------------------

* ``LLVM_BUILD_DIR``: CLAUDE.md told sessions on zhang-21 to export a ``/home``
  LLVM build that no longer exists there; the simulator then fails with
  ``Unknown function <top>``. Fixed in ``4904eadc``.
* The emitted SystemC testbench and Catapult's g++ 10.3: the ``allo`` env's
  activate script puts ``gcc-toolset-13`` first and Catapult's module puts its
  own ``python`` first; ``harness/env-zhang21.sh`` fixes the order.

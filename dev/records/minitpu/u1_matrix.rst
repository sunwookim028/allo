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
8. **A pipelined leaf** (``vpu_bf16_mul_pipe``, U1 multipliers). In Allo it
   is the comb unit: the simulator is untimed, ``s.pipeline`` gives II not
   depth, and a ``Stream`` between stage kernels is a FIFO, not a register
   (in csim it halves throughput at depth 1). Add a register/latency form
   (``Reg[T]``, ``delay``, or ``latency=`` on a kernel or unit), or accept
   "latency not expressible" as a recorded deviation like item 4?
   *[recorded deviation for U1; the proposal goes with item 4 to U3]*
9. **Two more SystemC bugs from the multipliers** (S4: ``bf16 -> f32``
   widening does not compile; S5: a ``UInt(24)`` port cannot be read back).
   *[fix on ``systemc-u1-fixes`` with a regression test each]*

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

``vpu_bf16_mul``
----------------

6,019,104 vectors (``bf16_mul.stimulus()``: corners crossed, ties, random,
every ``a`` against every corner). Evidence for every row:
``u1_mul_2026-10-02.rst``.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 5,011,918 match; the 1,007,186 others
       are the RTL's rules: flush-to-zero on input and output (603,316) and
       NaN always ``+0x7FC0`` (403,870). Outside NaN the simulator is IEEE bit
       for bit
     - N2: ``bfloat16`` has no flush-to-zero mode. N1: the simulator keeps
       the NaN operand's sign and gives ``Inf x 0`` the x86 ``-NaN``
   * - Allo simulator, ``bits``
     - **match** 6,019,104/6,019,104, no workaround (the RTL's exponent is
       ``logic signed``, so B1 cannot apply)
     - ``units/bf16_mul.py``; M1 (concat as slice stores) and M3 only
   * - SystemC csim, ``native``
     - **finding, semantic mismatch** x2: 5,011,890 match; same flush classes
       as the simulator; NaN sign is ``a ^ b`` (``ac::bfloat16``), so **the
       simulator and csim disagree on 403,397** of the 1,067,862 NaN/Inf pairs
       of the same Allo program
     - N1 (``u1_mul/repros.py n1``). The NaN a bf16 op returns depends on the
       op and the backend; the simulator is no stand-in for csim on NaNs
   * - SystemC csim, ``bits``
     - **match** 6,019,104/6,019,104 (30 s)
     -
   * - Catapult, RTLGen, AMC
     - not tried
     - out of this session's scope

``vpu_bf16_mul_pipe``
---------------------

Same function as ``vpu_bf16_mul``, two register banks, latency 2 (RTL
measured 2). The question was whether a pipelined unit differs from the comb
one in Allo at all: on the simulator and SystemC side **it does not** (L1-L3).

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator / SystemC, ``native``, ``bits``
     - as ``vpu_bf16_mul`` (the same Allo programs) + **finding, missing
       abstraction** L1: latency is not expressible. The simulator is
       untimed; ``s.pipeline`` states II, not depth; no primitive says
       "register here" or "latency 2"
     - Recorded deviation for U1, as triage item 4 (latency 0) -- here
       latency 2
   * - Allo simulator, ``bits_pipe`` (the pipe's own stage-2 text)
     - **match** 6,019,104, **after the B1 workaround** at a new site:
       ``exp_sum_s2 <= exp_bias_s2`` on ``logic [8:0]``; without the spare bit
       1,229,630 vectors are wrong. The comb module's text needs none: the two
       RTL forms of one unit differ in whether Allo computes them right
     - ``units/bf16_mul_pipe.py``
   * - SystemC csim, ``bits_pipe``
     - **match** 6,019,104
     -
   * - Allo simulator / SystemC, ``stages`` (two kernels, one ``Stream`` per
       ``*_s1_q`` register)
     - **match** 6,019,104 on both (317 s / 137 s) + **finding, semantic
       mismatch** L2: in csim a depth-1 Stream is not a register -- latency
       4 and **II 2** (depth 2: II 1); comb, pipe and ``s.pipeline``'d comb
       all take 2 cycles in csim. **Missing abstraction** L3: no link type is
       a pipeline register
     - ``u1_mul/latency_probe.py``. A declared latency can be checked only on
       RTL a tool wrote from Allo (Catapult track)

``mxu_bf16_mul_acc24``
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, workaround + semantic mismatch**: no acc24 type, so a
       ``float32`` product + bitcast + RNE to 15 bits in a ``uint32`` (W1,
       missing abstraction). 4,668,187 match; the 1,350,917 others are flush
       (674,119) and NaN (676,798; float32 also keeps the payload). W1
       double-rounds float32 subnormals: 31 vectors off IEEE acc24 by an ulp,
       invisible here only because the RTL flushes them
     - ``units/mul_acc24.py``; ``u1_mul_2026-10-02.rst`` (W1, N1, N2)
   * - Allo simulator, ``bits``
     - **match** 6,019,104/6,019,104, no workaround (compares are against
       literals)
     - The ``UInt(24)`` port works in the simulator; committed with ``uint32``
       for S5
   * - SystemC csim, ``native``
     - **finding, bug** S4 (emitter): ``bf16 -> f32`` is emitted as
       copy-initialisation of ``ac_ieee_float<binary32>`` from
       ``ac::bfloat16``, whose constructor is ``explicit``; g++ refuses.
       With the widening done on bit patterns (``native_bitext``,
       **workaround**): 4,941,087 match, same flush classes, NaN 403,898
     - Fix proposed: emit ``T v = T(x);`` for ``ExtFOp``/``TruncFOp``
       between ac floats (``EmitVivadoHLS.cpp:2613`` ``emitCast``)
   * - SystemC csim, ``bits``
     - **match** 6,019,104/6,019,104 (34 s) with a ``uint32`` port; with
       ``UInt(24)``, **finding, bug** S5: csim runs, then reading the output
       raises ``KeyError: 'ui24'`` (``np_supported_types``)
     - ``u1_mul/repros.py s5``
   * - Catapult, RTLGen, AMC
     - not tried
     - out of this session's scope

Environment findings met on the way
-----------------------------------

* ``LLVM_BUILD_DIR``: CLAUDE.md told sessions on zhang-21 to export a ``/home``
  LLVM build that no longer exists there; the simulator then fails with
  ``Unknown function <top>``. Fixed in ``4904eadc``.
* The emitted SystemC testbench and Catapult's g++ 10.3: the ``allo`` env's
  activate script puts ``gcc-toolset-13`` first and Catapult's module puts its
  own ``python`` first; ``harness/env-zhang21.sh`` fixes the order.

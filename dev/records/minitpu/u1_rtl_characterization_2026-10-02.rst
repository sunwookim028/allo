..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#####################################################
U1 RTL characterization: MiniTPU's arithmetic leaves
#####################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch ``u1-rtl``.
   MiniTPU at ``b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1``, Verilator 5.052
   (conda-forge), g++ 13.3.1 (gcc-toolset-13), numpy 2.4.0. RTL side only: no
   Allo build is involved. Each unit's numpy reference states the RTL's
   semantics where they leave IEEE 754 and is held to the RTL bit for bit; the
   Allo variants (README D-7, D-9) come later and are explained against these
   references.

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first)::

   $ALLO_PYTHON -m examples.minitpu.harness.characterize \
       bf16_add_pipe bf16_mul bf16_mul_pipe mul_acc24 acc24_add_pipe alu  # ~2.5 min
   # all 2^32 pairs, 11-35 min each on 1 core; bf16_add needs Allo importable
   MINITPU_HARNESS_CACHE=<local disk> \
       $ALLO_PYTHON -m examples.minitpu.harness.exhaustive bf16_mul

Unit specs: ``examples/minitpu/units/<unit>.py``. References:
``examples/minitpu/harness/ref.py``. Stimulus: ``harness/stimulus.py``.

What "latency" means here
=========================

The number of rising edges from the one that captures an input to the one
after which its result is visible: the number of registers on the path, which
is how MiniTPU's comments and ``localparam`` s count. ``valid`` units are
measured twice, from ``valid_o`` on every vector and by a step probe (hold one
input, switch to another, count edges until the output moves); ``bare`` units
(no valid) by the probe, and their outputs are then read at the declared depth
and must match the reference.

**Harness fix.** ``rtl.py`` as the bf16_add pilot left it read sequential units
one edge early: ``vpu_bf16_add_pipe`` measured 1, and the ``bare`` sampler
would have paired each input with its successor's result. The pilot only drove
a combinational unit, so nothing it reported changes. The wrapper now records
``cycle + 1`` and keeps every edge's output for ``bare``
(``WRAPPER_VERSION = "2"``).

Results
=======

.. list-table::
   :header-rows: 1

   * - unit (top)
     - shape
     - latency declared / measured
     - reference vs RTL
     - RTL differs from IEEE on
   * - ``vpu_bf16_add_pipe``
     - valid
     - 2 (header "Two-stage"; no localparam) / 2 (``valid_o``, probe)
     - 251,936 / 251,936 (comb twin: 2^32 / 2^32)
     - 981
   * - ``vpu_bf16_mul``
     - comb
     - 0 / 0
     - 6,019,104 / 6,019,104; 2^32 / 2^32 exhaustive
     - 1,007,214
   * - ``vpu_bf16_mul_pipe``
     - bare (clock only)
     - 2 ("Two-stage form"; no localparam) / 2 (probe)
     - 6,019,104 / 6,019,104
     - 1,007,214
   * - ``mxu_bf16_mul_acc24``
     - comb
     - 0 / 0
     - 6,019,104 / 6,019,104; 2^32 / 2^32 exhaustive
     - 1,078,037
   * - ``mxu_acc24_add_pipe``
     - valid
     - 3 (``vpu_pkg::MXU_ACC_ADD_LATENCY``) / 3 (``valid_o``, probe)
     - 352,116 / 352,116; plus 7,002,116 / 7,002,116 (seed 12345)
     - 1,544
   * - ``vpu_alu``
     - valid, ``op_i`` 4 bit
     - 3 (``vpu_pkg::VPU_ALU_LATENCY``) / 3 (``valid_o``, probe), every op
     - 5,079,552 / 5,079,552 (all 16 op codes)
     - 959,178

Stimulus: bf16 units take ``binary_bf16`` (44 corners crossed, 50k rounding
ties, 200k random); the multipliers add ``bf16_sweep_a`` (every ``a`` against
every corner, both orders, 5.77M). acc24 takes ``binary_acc24``: 46 corners
crossed, 100k ties and near-cancellations, 50k pairs at the subnormal and
overflow ends (as MiniTPU's ``tb_acc24_add_pipe.cpp`` biases), 200k random.
The 7M-vector acc24 run produced 60,261 subnormal and 33,829 infinite results.
The ALU takes ``binary_bf16`` plus all 2^16 ``a`` (random ``b``) under each of
the 16 op codes.

Exhaustive runs (``harness/exhaustive.py``, all 2^32 operand pairs):

.. list-table::
   :header-rows: 1

   * - unit
     - reference
     - result
     - wall
   * - ``vpu_bf16_add`` (comb twin of ``vpu_bf16_add_pipe``)
     - ``ref.vpu_bf16_add``
     - 4,294,967,296 / 4,294,967,296
     - 654 s
   * - ``vpu_bf16_mul``
     - ``ref.vpu_bf16_mul``
     - 4,294,967,296 / 4,294,967,296
     - 2,106 s
   * - ``mxu_bf16_mul_acc24``
     - ``ref.mxu_bf16_mul_acc24``
     - 4,294,967,296 / 4,294,967,296
     - 2,097 s

``vpu_bf16_add_pipe`` and ``vpu_bf16_mul_pipe`` inherit these through
MiniTPU's own 2^32 pipe-vs-comb equivalence benches. Since
``ref.vpu_bf16_add`` is IEEE RNE except NaN sign and the ``(+0)+(-0)`` zero,
the adder is correctly rounded on every pair, including the
exponent-difference >= 10 short-circuit: this closes ARITHMETIC.md section
10's first open question by exhaustive simulation.

Deviations from IEEE, by unit
=============================

Each count is the number of stimulus vectors where the RTL and the IEEE
reference (``ref.ieee_*``) differ for that cause; ``unexplained`` was empty
for every unit. IEEE NaN sign is unspecified, so the NaN rows count only a
negative IEEE NaN that the RTL returns positive.

``vpu_bf16_add_pipe`` (same as the pilot's ``vpu_bf16_add``, which MiniTPU
holds bit-identical over 2^32 pairs):

* 980 -- every NaN result is ``+0x7FC0``.
* 1 -- ``(+0) + (-0) = -0`` (``(-0) + (+0) = +0``): an exact-zero ``a`` returns
  ``b`` bit for bit. IEEE RNE gives ``+0`` for both.

``vpu_bf16_mul`` and ``vpu_bf16_mul_pipe`` (identical counts):

* 514,904 -- a subnormal operand is zero, so the result is the signed zero
  even where IEEE's product is a *normal* number (``2^-133 x 2^127 = 2^-6``
  gives ``0``).
* 36,401 -- the same flush where IEEE's product is subnormal.
* 51,958 -- a product below ``2^-126`` is flushed, where IEEE gives a
  subnormal.
* 53 -- a product below ``2^-126`` that IEEE rounds *up* to ``2^-126``
  (``0x0080``) is flushed too: the RTL tests underflow before rounding.
* 403,898 -- NaN is always ``+0x7FC0``.
* ``Inf x subnormal = Inf`` matches IEEE (the RTL's Inf-x-0 test reads the bit
  pattern before the flush): no deviation, and ARITHMETIC.md section 6 is
  right that a flush-then-IEEE model would get NaN here.

``mxu_bf16_mul_acc24`` (output acc24):

* 514,840 / 70,862 -- subnormal operand flushed, IEEE product normal / an acc24
  subnormal.
* 88,437 -- a product below ``2^-126`` is flushed although acc24 has
  subnormals that could hold it (exactly, or rounded).
* 403,898 -- NaN is always ``+0x7FC000``.
* Otherwise the product is exact, and overflow is signed Inf: no deviation.

``mxu_acc24_add_pipe``:

* 1,543 -- NaN is always ``+0x7FC000``.
* 1 -- ``(+0) + (-0) = -0``, the same zero bypass as the bf16 adder.
* Otherwise a correctly rounded (once, RNE) add with gradual underflow and
  overflow to Inf, on 7.35M vectors. This settles, by measurement and not
  proof, ARITHMETIC.md section 10's open question on the acc24 adder.

``vpu_alu`` (op codes are ``vpu_pkg::vpu_alu_op_e``: ADD 0, SUB 1, MUL 2,
MOV 3, MAX 4, MIN 5, AND 6, OR 7, XOR 8):

* 944,445 -- **AND, OR and XOR are not implemented**: they fall to the result
  mux's ``default`` arm and return ``a`` (a ``mov``). So do the unused codes
  9-15 (agreeing with the reference; not counted as deviations).
* 4,946 -- MAX/MIN with a NaN operand select by ``vpu_pkg::bf16_gt``, a total
  order on bit patterns (``-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN``):
  a positive NaN wins MAX and a negative NaN wins MIN with sign and payload
  kept, and a NaN on the losing side is dropped. That is neither IEEE
  754-2019 ``maximum`` (NaN always) nor ``maximumNumber`` (NaN never).
  ``-0 < +0`` agrees with ``maximum``.
* 4,745 / 1,320 -- MUL's two flushes, as in ``vpu_bf16_mul_pipe``.
* 3,720 -- ADD/SUB/MUL NaN always ``+0x7FC0``.
* 1 -- ADD ``(+0) + (-0) = -0``.
* 1 -- SUB ``(+0) - (+0) = -0``: SUB flips ``b``'s sign and adds, so it
  inherits the adder's zero bypass. IEEE gives ``+0``; ``x - x`` for non-zero
  ``x`` is ``+0`` as in IEEE.

Findings
========

1. **``docs/UNITS.md`` says ``vpu_alu`` does "add/sub/mul/max/min/mov/
   and/or/xor"; the RTL does not do and/or/xor.** ``vpu_pkg`` declares
   ``VPU_ALU_AND/OR/XOR`` but ``vpu_alu``'s result mux has no arm for them, so
   they return ``a``. Unreachable from the ISA today (``sequencer_decoder.sv``
   issues only ADD..MIN), so it is a doc/RTL disagreement, not a bug a program
   can hit. For the owner (D-7: changing MiniTPU is the owner's call).
2. **ARITHMETIC.md agrees with the RTL everywhere it was tested**: FTZ in both
   multipliers on input and on output, ``Inf x subnormal = Inf``, NaN
   canonicalization, the zero bypass's signed zero, acc24 RNE with
   subnormals, the ``vmax``/``vmin`` NaN order. Two consequences it does not
   spell out, now measured: the multipliers flush a product that would round
   up to ``2^-126`` (53 of 6M here), and ``vsub`` turns the zero bypass into
   ``(+0) - (+0) = -0``. Both of section 10's open questions on the adders
   are answered by simulation: the bf16 adder is correctly rounded on all
   2^32 pairs, and the acc24 adder on 7.35M targeted and random pairs.
3. **Latency**: every declared value is the measured one. Two units state
   theirs only in prose (``vpu_bf16_add_pipe``, ``vpu_bf16_mul_pipe``: no
   ``localparam``). ``docs/isa_latency.json``'s ``w: 5`` for the ALU ops is the
   ISA writeback offset, not ``vpu_alu``'s own latency (``VPU_ALU_LATENCY =
   3``).
4. The harness read sequential units one edge early (above); fixed before any
   sequential result was recorded.

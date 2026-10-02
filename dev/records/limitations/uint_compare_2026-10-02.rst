..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#####################################################
Unsigned compares lowered as signed (B1), 2026-10-02
#####################################################

Branch ``core-uint-compare`` (review branch, not ``main``), base
``06650725``. Host zhang-21. Found while expressing MiniTPU's
``vpu_bf16_add`` in Allo (``dev/records/minitpu/u1_bf16_add_bits_2026-10-02.rst``
on branch ``u1-bits``, items B1 and B2).

The bugs
========

**B1** (``allo/ir/builder.py``, ``build_Compare``). The integer predicate was
chosen from the *MLIR* type string::

    dtype = str(rhs_res.type)
    ATTR_MAP["int" if dtype.startswith("i") else "uint"]

MLIR integers are signless (``i8``, never ``ui8``), so every ``<``, ``<=``,
``>``, ``>=`` between ``UInt`` values became a signed ``arith.cmpi``
(``sgt``/``slt``/...). The LLVM backend and the dataflow simulator then
compute ``uint8 200 > 100 == False``; ``uint1 1 > 0 == False`` (``i1`` 1 is
-1). The HLS C++ emitter ignores the predicate and compares the C types
(``uint8_t``), so **the simulator and the hardware disagree** on the same
program.

**B3** (same function, found here). The fixed-point branch tested
``dtype.startswith("f")`` on ``"!allo.Fixed<...>"``, which is always false, so
*signed* ``Fixed`` compares always got the unsigned ``CmpFixedOp`` predicates
(``ult``...): ``Fixed(8,2)`` ``-2 < 3 == False`` on LLVM/simulator.
``UFixed`` happened to be right.

**B2** (``allo/ir/infer.py``, ``visit_Compare``). A comparison's ``dtype`` was
its operand type, not ``uint1``. ``r: uint1 = a == b`` (``uint8`` operands)
emitted ``arith.trunci i1 -> i1`` and failed to verify; ``r: int32 = a < b``
(``int32`` operands) stored an ``i1`` into an ``i32`` memref and failed too.

Minimal repro (B1)::

    def k(A: uint8[2], B: uint8[2]) -> uint8[2]:
        C: uint8[2] = 0
        for i in range(2):
            C[i] = 1 if A[i] > B[i] else 0
        return C
    allo.customize(k).build()(np.array([200, 100], np.uint8),
                              np.array([100, 200], np.uint8))
    # main: [0 1], MLIR `arith.cmpi sgt ... : i8`;   fixed: [1 0], `ugt`

The fix (Python only, no C++)
=============================

* ``infer.py``: the common operand type goes to ``node.operand_dtype``;
  ``node.dtype = uint1``. ``Compare`` nodes have no other consumer of their
  ``dtype`` in the front end (grep), so B2 is local.
* ``builder.py``: both operands are cast to ``node.operand_dtype`` and the
  predicate family is chosen by ``isinstance`` on it, the way div/rem/shift/
  ``min``/``max`` already do: ``UInt`` -> unsigned; ``Int``/``Index`` ->
  signed; ``UFixed`` -> ``ufixed``; ``Fixed`` -> ``fixed``; ``Float`` ->
  ``cmpf``.

Semantics now stated
--------------------

* Mixed ``Int(m)`` vs ``UInt(n)`` is compared in ``Int(max(m, n+1))``
  (``typing_rule.cmp_rule``): by value. Unchanged by this fix, and it was
  already right on all backends (the operands are widened before the signed
  compare). It **differs from C** at equal widths: C converts ``int32`` to
  ``unsigned``, so ``-1 < 1u`` is false in C and true in Allo. The emitted HLS
  carries the widening (``ap_int<33>``), so Allo's hardware agrees with Allo.
* ``Index`` vs ``Index`` stays signed (MLIR's convention for ``index``).
  ``Index`` vs ``UInt(n)`` is compared as ``UInt(max(64, n))``: a negative
  index would compare as huge. Unchanged; noted.

Other occurrences of the bug class
==================================

``grep 'startswith("ui")'`` over ``allo/``: the remaining hits
(``allo/utils.py``, ``allo/backend/llvm.py``, ``allo/backend/tapa.py``) read
*Allo* type strings built from the ``itypes``/``otypes`` hints
(``get_signed_type_by_hint`` yields ``"ui8"``), not MLIR types, so they are
correct. One related signedness bug, **not fixed** here (it can change emitted
HLS, so it wants its own review): ``builder.py`` ``bitcast`` tests
``isinstance(node.func.value.dtype, UInt) or (node.dtype, UInt)`` -- the second
operand is a tuple, always true, so every bitcast is marked ``unsigned``.

Tests
=====

``tests/test_compare.py`` (17 tests): all six operators at ``UInt`` widths
1/8/16/32/64 on boundary values (0, 1, 2^(w-1)-1, 2^(w-1), 2^w-1, all pairs)
on the LLVM backend, and at 1/8/16/32 under ``df.build(target="simulator")``;
signed ``int8`` stays ``slt``; mixed ``int32``/``uint32`` (value semantics,
``i33``); signed ``Fixed`` and ``UFixed`` compares; float; B2 (``uint1``
result, and a compare widened to ``int32`` is 1 not -1); the HLS emitter still
prints ``uint8_t``. On ``main`` 12 of the 17 fail; on the branch all pass.

Impact against ``main``
=======================

Baseline: a second worktree at ``origin/main`` (``06650725``), same host,
same bindings (both trees symlink the primary checkout's ``mlir/build``),
Vitis removed from ``PATH`` for the pytest runs so HLS-tool tests skip.

.. list-table::
   :header-rows: 1

   * - Gate
     - ``main``
     - branch
   * - ``gen_isa.py --check``
     - ISA OK
     - ISA OK
   * - ``lift_units.py --check``
     - UNITS OK
     - UNITS OK
   * - ``bench_isa.py`` (``TPU_MAXDIM=16``)
     - ALL EXACT
     - ALL EXACT
   * - ``stress_isa.py``
     - STRESS OK 492/492
     - STRESS OK 492/492
   * - ``act_compile.py --gate``
     - ACT GATE OK 12/12
     - ACT GATE OK 12/12
   * - TinyTPU MLIR (pre-schedule) and emitted ``kernel.cpp``
     - --
     - byte-identical to ``main``
   * - ``pytest tests/dataflow --ignore=aie`` (5m40s)
     - 140 pass, 20 skip, 1 xfail
     - identical
   * - ``pytest tests/act`` (3m10s)
     - 193 pass, 2 fail, 4 skip
     - identical
   * - ``pytest tests/test_*.py`` (~3.5m)
     - 459 pass, 13 fail, 10 skip
     - identical + 17 new pass
   * - ``pytest tests/ip tests/ip_integration``
     - 40 pass, 4 fail, 2 skip, 1 xfail
     - identical
   * - ``tests/limits`` repros (all but cosim/Catapult/TAPA ones)
     - --
     - identical verdicts

Per-test outcomes were diffed; the only difference is the 17 new tests. The
pre-existing failures are the same on both trees: ``act/test_bindings.py`` (2,
because the worktrees borrow another checkout's ``mlir/build``),
``test_verify.py`` (8), ``test_stateful.py`` (2), ``test_pynq.py`` (2),
``test_builder.py::test_minmax_cast``, ``ip_integration/test_external.py``
(4). ``tests/dataflow/aie`` fails at collection on both (no AIE toolchain).

**TinyTPU's published numbers cannot move**: the design has no unsigned or
fixed-point compare (its 87 ``cmpi`` are ``eq``/``sge``/``sgt``/``slt`` on
``int32``/``index``, the same set before and after), and the HLS it emits is
byte-identical, so cosim (175/265/421/482/674) is untouched. Cosim was not
re-run.

Upstream status
===============

``cornell-zhang/allo`` ``main`` (``3f2ea5d4``, 2026-09-30) has the same code in
``build_Compare`` (B1 and B3) and ``visit_Compare`` (B2). No upstream issue or
PR found (searched "unsigned", "cmpi", "unsigned compare").

Draft upstream issue (not filed)
--------------------------------

    **[Bug] Unsigned integer comparisons are lowered as signed `arith.cmpi`
    (LLVM backend and simulator disagree with HLS)**

    `ASTTransformer.build_Compare` picks the integer predicate from the MLIR
    type string: `ATTR_MAP["int" if str(rhs_res.type).startswith("i") else
    "uint"]`. MLIR integer types are signless (`i8`, never `ui8`), so the
    `uint` branch is unreachable and every `<`/`<=`/`>`/`>=` between `UInt`
    values becomes `sgt`/`slt`/...:

    ```python
    import numpy as np, allo
    from allo.ir.types import uint8

    def k(A: uint8[2], B: uint8[2]) -> uint8[2]:
        C: uint8[2] = 0
        for i in range(2):
            C[i] = 1 if A[i] > B[i] else 0
        return C

    s = allo.customize(k)
    print(s.module)   # arith.cmpi sgt, ... : i8
    print(s.build()(np.array([200, 100], np.uint8),
                    np.array([100, 200], np.uint8)))   # [0 1], expected [1 0]
    ```

    The Vivado/Vitis HLS emitter ignores the predicate and compares `uint8_t`,
    so it gives `[1 0]`: the LLVM backend (and the dataflow simulator) disagree
    with the generated hardware. Two related defects in the same code:

    * the fixed-point branch tests `dtype.startswith("f")` on
      `"!allo.Fixed<...>"` (always false), so signed `Fixed` compares get
      unsigned predicates (`Fixed(8,2)`: `-2 < 3` is False);
    * `TypeInferer.visit_Compare` gives the comparison its operand type
      instead of `uint1`, so `r: uint1 = a == b` on `uint8` operands emits
      `arith.trunci i1 -> i1` and fails verification.

    Fix: choose the predicate by `isinstance` on the common operand type (as
    div/shift/min/max do), and keep that type on a separate attribute while
    typing the comparison `uint1`. A patch with tests is on
    `sunwookim028/allo` branch `core-uint-compare`.

..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##################################################################
U2: ``@ Stateful(reset=False)`` implemented (README D-14), measured
##################################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch
   ``backend-unreset`` (worktree ``scratch/wt-unrst`` from ``origin/u1-pilot``
   at ``38f2083d``), its own ``mlir/build`` (configured as ``main``'s:
   ``LLVM_DIR``/``MLIR_DIR`` under ``/work/shared/common/llvm-project-main/
   build-rhel8``, gcc-toolset-13, the ``allo`` env's python and ninja; 336
   steps clean). MiniTPU at ``b3ba0a4d``, read only. Catapult 2024.2
   (``nangate-45nm_beh``), DC W-2024.09 (FreePDK45, ``dc_u1.tcl`` of
   ``u2_regfile_2026-10-02/scripts/dc``, unchanged), Verilator 5.052. Scratch
   (Catapult, DC and csim projects, not kept): ``scratch/unrst/``. Record files:
   ``u2_unreset_impl_2026-10-02/``.

``u2_comb_wire_impl_2026-10-02.rst`` (F3) found the one form in which Catapult
leaves storage unreset -- the write as a clock-edge ``SC_METHOD`` with no reset
action, under ``-RESET_CLEARS_ALL_REGS no`` (its form f, hand-written) -- and
README D-14 made unreset storage a declaration. This record is the
implementation: the declaration, the emitter's write method and its rule, the
``run.tcl`` directive, the refusals, ``latency.json``, and the register file
through Catapult and DC at both widths with **no hand patch on the design**.

**Result.** ``mem: UInt(16)[32] @ Stateful(reset=False)`` reaches the IR as
``allo.unreset = "mem"`` on the ``memref.global``. The SystemC emitter writes
it from ``SC_METHOD(wr); sensitive << clk.pos();`` with no reset action, and
``run.tcl`` gets ``directive set -RESET_CLEARS_ALL_REGS no``. On the
``comb_unreset`` variant of ``vpu_regfile``: **0 ``if ( rst )`` in the RTL, all
512 storage flops ``DFF_X1``, bit-exact against MiniTPU at read 0 / write ->
read 1 (w16 180,780/180,780, w256 45,744/45,744)**. DC w16 **4,033.9 um^2**
against MiniTPU's 4,021.7 (+12.2, +0.30 %). The whole difference is the
kernel's two reset control flops (FSM state and ``done``, 2 x ``DFFR_X1``),
which F3's hand form f (4,021.9) did not have. The reset storage of D-13 costs
4,463.2 here, so the declaration recovers 429 of its 441 um^2.

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first; ``R`` is
this record's directory, ``S`` a scratch directory, ``C`` =
``dev/records/minitpu/u2_comb_read_2026-10-02/scripts``)::

   $ALLO_PYTHON -m pytest tests/dataflow/test_systemc_unreset.py -q    # 9 tests, csim included
   # Catapult w16 / w256 (same scripts as the D-13 record; no partition, no design patch)
   $ALLO_PYTHON $C/emit_csyn.py vpu_regfile comb_unreset $S/w16 --n 64 --pipeline rf_0:_ \
       --synth-top rf_0 --clock 3.33 [--width 256 --no-run]
   #   w256 only: the pilot's S7 testbench patch, then catapult -shell -f ../run.tcl in build/
   #   sed -i -E 's/\(long long\)\((ch_v[0-9]+\.Pop\(\))\)/(\1).to_int64()/' $S/w256/kernel.cpp
   $ALLO_PYTHON $C/cmp_rf.py $S/w16/build/Catapult/rf_0.v1/concat_sim_rtl.v --top rf_0 \
       --ports v18,v19,v20,v21,v22,v23:v24,v25,v26 [--inst w256]
   $R/dc/run_dc.sh unrst_w16 rf_0 clk 3.33 $S/w16/build/Catapult/rf_0.v1/concat_rtl.v
   # csim, per iteration and offset-aligned
   $ALLO_PYTHON examples/minitpu/harness/check.py vpu_regfile --variant comb_unreset \
       --variant comb --backend systemc --project $S/prj
   $ALLO_PYTHON $C/csim_offset.py $S/prj/vpu_regfile_w16_comb_unreset_systemc
   # latency.json: Allo's csyn mode, as in the D-13 record with u.comb_unreset

Design
======

**Declaration** (``allo/ir/types.py``, ``infer.py``, ``builder.py``).
``Stateful`` gained ``__init__(self, reset=True)``: ``@ Stateful`` and
``@ Stateful()`` are reset storage as before. ``@ Stateful(reset=False)`` gives
the inferred dtype ``unreset = True``, and the builder puts
``allo.unreset = "<variable>"`` on the global, beside ``static``. ``customize``
clones globals per instance with their attributes, so the marker survives
instancing. Two frontend fixes were needed on the way:

- ``Stateful(reset=<not a bool>)``: ``ASTResolver`` swallows a constructor's
  exception and returns ``None``. The declaration then **silently became an
  ordinary local array**, and so did any typo in the call. ``infer.py`` now
  raises ``Stateful(...) takes only `reset=True` or `reset=False` (a literal)``.
  (The ``isinstance(reset, bool)`` check in ``types.py`` itself is written as
  ``reset is not True and reset is not False``: ``bool`` is ``UInt(1)`` in that
  module.)
- A ``Stateful`` wider than 64 bits (``UInt(256)[32]``, the w256 register file)
  failed in ``DenseElementsAttr.get`` (NumPy has no 256-bit integer), so the
  w256 ``wire_stateful`` form could never be built either. A scalar
  initialiser of a wide integer now becomes a splat attribute directly.

**Emitter** (``mlir/lib/Translation/EmitSystemC.cpp``, ``planUnreset`` /
``unresetThreadDead`` and a ``WriteMethod`` emission mode beside D-13's
``Thread``/``Method``). For a kernel that touches unreset storage:

1. The storage is ``sc_signal<T> name[N]`` (flattened), the same signal storage
   as comb storage, and the reset action leaves it alone (it prints a comment
   saying so). Loads are ``.read()`` and stores ``.write()``.
2. Every store to it, together with its cone, is re-emitted in ``void wr()``,
   which is registered as ``SC_METHOD(wr); sensitive << clk.pos();`` plus
   ``dont_initialize()`` under ``#ifndef __SYNTHESIS__``, so csim does not run
   it once at time 0 (the flop never does). The cone is the address, the data,
   the conditions of the ``if``\ s the store sits under, their ``wire_get``\ s,
   and the scalar temporaries with the store that reaches each. The thread drops
   the stores and then, by the same dead-code pass as D-13, extended to nested
   ``if``\ s, whatever only they needed. In the regfile the thread is left with
   its loop of ``wait()``\ s and the ``done`` flag. Emitted ``rf_0``
   (``catapult/w16/kernel.cpp``)::

      SC_METHOD(wr);
      sensitive << clk.pos();
      ...
      void wr() {  // clock-edge write, no reset action (D-14): __stateful_rf_0_mem_1
        ac_int<5, false> v45 = v21.read(); ... bool v49 = v23.read(); ...
        if (v50) { ... __stateful_rf_0_mem_1[((v53))].write(v51); }
      }

   This is F3 form f's ``wr`` up to value numbering.
3. The constructor carries ``// allo unreset storage: <global>``. ``hls.py``
   reads that marker and sets ``configs["unreset_storage"]``, and
   ``codegen_tcl_catapult`` then adds ``directive set -RESET_CLEARS_ALL_REGS no``
   after ``DESIGN_HIERARCHY``. The value is ``no``, not F3's ``false``, which
   only drew a SIF-24 deprecation warning. A design without unreset storage
   gets a byte-identical ``run.tcl``.
4. Everything else is emitted as before: reset storage, comb cones (D-13; a
   comb cone may read unreset storage, as the regfile's do), and every other
   kernel.

**The rule** (refused at build, naming the storage:
``unreset storage `mem` (rf_0): <why> (README D-14: its write is a clock-edge
process with no reset action)``). ``wr`` runs at *every* clock edge: under
reset, before the kernel's first iteration and after its last. So it may
compute only what is a function of this cycle's inputs:

- the writing kernel is Wire-only: every argument a ``Wire``, and no stream or
  channel op. Only then is one iteration one clock cycle;
- each store is in the iteration block (the kernel-level loop body), under
  nothing but ``if``\ s (``scf.if`` / ``affine.if``), with no inner loop;
- its address, data and conditions read only ``Wire`` inputs, constants,
  arithmetic and scalar iteration temporaries written unconditionally earlier
  in the iteration. That rules out any storage load (no read-modify-write:
  ``mem[x] = mem[x] + d`` is refused), the induction variable, and anything
  loop-carried;
- the iteration reads the storage only before its stores (signal storage reads
  old until the next edge).

Tested refusals: read-modify-write (``... reads mem; the address, data and
enable of a clock-edge write may read only Wire inputs, constants and scalar
iteration temporaries, not storage``), a stream kernel (``... the kernel has a
non-Wire argument (`s_wa`) ...``), and a store in an inner loop (``... a store
to it is inside a loop within the iteration ...``).

**Other backends.** Vitis (``vhls``) and the Catapult C++ flow raise
``NotImplementedError: unreset storage `mem` (__stateful_rf_0_mem_1)
(`@ Stateful(reset=False)`, README D-14) is only lowered by the SystemC backend
...``. Choice: refuse, because neither has a measured unreset form. Vitis'
default ``config_rtl -reset control`` plausibly leaves a static array unreset,
but nothing here measures it, and D-1 asks for a refusal over an unmeasured
claim. **The simulator** runs it as ordinary storage (the LLVM path ignores the
attribute): the array-port regfile with ``reset=False`` gives the model's
values on every defined slot (``test_unreset_other_backends``).

**``latency.json``** (``catapult.py`` ``unreset_storage_in``). The kernel entry
gains ``"storage": {"__stateful_rf_0_mem_1": "unreset"}``, and the build log
gains ``unreset storage [...]`` (``catapult/latency.json``)::

   "rf_0": {"ii": 1, "latency": 1, "loop": "while", "port_style": "wire",
            "ports": {"v24": "comb", "v25": "comb", "v26": "comb"},
            "status": "scheduled", "storage": {"__stateful_rf_0_mem_1": "unreset"}, ...}

``latency``/``ii`` are still the thread's loop, which in the regfile is now
empty. The write is visible one edge after it is presented, by construction of
the clock-edge method, and the trace confirms it.

Validation
==========

``examples/minitpu/units/vpu_regfile.py`` gained ``comb_unreset``, the ``comb``
body with ``mem: W[32] @ Stateful(reset=False)``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - check
     - result
   * - Catapult w16
     - ok, 40 s, ``-RESET_CLEARS_ALL_REGS no`` in the emitted ``run.tcl``. Modules
       ``rf_0_comb`` (clockless), ``rf_0_wr``, ``rf_0_run``, ``rf_0``. ``if ( rst )``
       occurs **0** times; ``if ( ~ rst )`` occurs twice, both in ``rf_0_run`` (its
       FSM state and ``done``). Every ``rf_0_wr`` block is ``always @(posedge clk)``.
       ``Warning: Input port 'rst' is never used. (OPT-4)`` is for the ``wr`` process,
       as in F3 f. Area score **3,800.3** (D-13 ``comb``: 4,219.4), max delay 0.465 ns
   * - trace w16 (``logs/cmp_w16.txt``)
     - **180,780/180,780 at read latency 0**, 3 masked (uninitialised); probes on all
       three ports: read 0, **write -> read 1** (MiniTPU 0 / 1)
   * - DC w16 (``dc/unrst_w16``)
     - **4,033.9 um^2** (comb 1,580.3, seq 2,453.6), slack 2.80 ns at 3.33 ns. 514 flops:
       **512 ``DFF_X1`` (the storage)** + 2 ``DFFR_X1`` (``rf_0_run``'s FSM and
       ``done``; ``logs/flops_w16.txt``). MiniTPU 4,021.7 (1,578.7 / 2,442.9): seq
       +10.7 = the two control flops, comb +1.6. D-13 ``comb`` (reset storage):
       4,463.2
   * - Catapult w256
     - ok, 37 s. The pilot's S7 testbench patch (3 lines, TB only,
       ``catapult/w256/tb_patch_S7.diff``) was applied; the design is untouched. 0
       ``if ( rst )``. Area score **58,110.7** (D-13 ``comb``: 64,658.4), max delay
       0.522 ns
   * - trace w256 (``logs/cmp_w256.txt``)
     - **45,744/45,744 at read 0; write -> read 1** on all three ports
   * - DC w256 (``dc/unrst_w256``)
     - **60,204.6 um^2** (comb 23,022.0, seq 37,182.5), slack 1.95 ns; 8,192 ``DFF_X1`` +
       2 ``DFFR_X1``. MiniTPU w256 82,738.8 (45,566.9 / 37,171.9): seq **+10.6**, the same
       two control flops. The combinational -49 % is the read network, as in the D-13
       record (-18.9 % in total there, -27.2 % here); D-13 ``comb``: 67,084.7
   * - csim per iteration (``logs/check_comb.txt``)
     - ``UNIT-DIFF vpu_regfile:w16 comb_unreset systemc 23010/180780``, the same verdict
       as ``comb`` (limitation 22: a ``Wire`` link is not cycle-locked in csim)
   * - csim offset-aligned (``logs/csim_offset.txt``)
     - **+3/+2/+2** (60/61/61 of 63 defined), the same as ``comb``
   * - ``latency.json``
     - above; ``LATENCY-`` verdicts unchanged in kind (``harness/latency.py`` reads
       ``latency``/``ii``/``status``)

Regression
==========

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - check
     - result
   * - ``tests/dataflow/test_systemc_unreset.py`` (new: marker, bad argument, emit
       shape, ``run.tcl``, three refusals, other backends + simulator, csim
       compile-and-run at offset +3)
     - 9 passed
   * - ``pytest tests/dataflow/test_systemc*.py`` (backend, csim_regress, comb,
       unreset) + ``test_stateful_systemc.py``, ``test_region_stateful.py``,
       ``test_df_unit.py``
     - 97 passed, 2 failed (6:32). The two failures are
       ``test_stateful_systemc.py::test_stateful_{scalar,array}_systemc``, which run
       Xcelium cosim: ``xmsim: *F,VSPLIC: VSP Licensing Failure`` (licence checkout, no
       RTL output). They are environmental: neither design has unreset storage, so
       neither emission changes (``logs/regress.txt``)
   * - TinyTPU emission, ``customize(tinytpu_isa)`` + ``schedule``,
       ``build(target=...)``, sha256 of ``hls_code`` (systemc: ``mode="csim"``, the
       project path replaced), base (this tree before the change, same bindings)
       vs branch (``logs/tinytpu_hash.py``, ``logs/tinytpu_hash.txt``)
     - vhls ``6bc774bc...``, catapult ``ade1ab5d...``, systemc ``ba271c9b...`` on both:
       **identical**. The vhls/catapult prefixes are the ones the D-13 record could
       not reproduce on disk. This recipe (``build(target=...)`` with no project)
       gives them
   * - ``examples/tinytpu/systemc_csim.py 3``
     - 3 runs, each 3x ``wrong=0/16`` (``clobbered_outside`` 4057 / 4057 / 4065, the
       recorded baseline)
   * - ``examples/eva/cosim_eva_systemc.py``
     - ``COSIM RESULT : PASS (bit-exact)``; ``examples/eva/generated`` restored after
   * - U1 units, ``check.py <unit> --backend systemc``
     - all 24 SystemC verdict lines (seven units, every variant) identical to
       ``u1_integration_2026-10-02/<unit>.log``

Findings
========

**H1 (Catapult, match).** A declared ``@ Stateful(reset=False)`` gives
MiniTPU's storage: plain ``DFF_X1``, bit-exact at both widths, with sequential
area equal to MiniTPU's apart from the kernel's two reset control flops. The
D-13 reset deviation is closed for storage that declares it.

**H2 (frontend, bug, fixed).** A ``Stateful(...)`` call that raised was resolved
to ``None`` and **silently dropped**: the storage became a kernel-local array
with no diagnostic. Now refused.

**H3 (frontend, bug, fixed).** A ``Stateful`` wider than 64 bits could not be
built (NumPy buffer), so no w256 ``@ Stateful`` form existed before this. Fixed
for a scalar initialiser.

**H4 (SystemC emitter, info, pre-existing).** A ``UInt`` ``Stateful`` global
carries no ``unsigned`` attribute, so its member is declared
``ac_int<16, true>``. Loads and stores convert at full width, so the bits are
unchanged (the traces above are bit-exact), but the declared type is wrong. Not
fixed here (it changes every ``UInt`` ``Stateful`` emission).

**H5 (Catapult, info).** ``-RESET_CLEARS_ALL_REGS no`` is global to the design.
A register is then reset only when a reset action sets it. The kernel's FSM and
``done`` still are, and so is any ``@ Stateful`` without ``reset=False`` (it is
written in the reset action). An internal register that the C semantics do not
require to be reset could lose its reset in a mixed design. None did here.

Classification (D-9)
====================

.. list-table::
   :header-rows: 1
   :widths: 8 24 68

   * - id
     - class
     - one line
   * - H1
     - match (Catapult)
     - ``reset=False`` storage is MiniTPU's: 0 reset flops, bit-exact, DC +0.30 %
   * - H2
     - bug, fixed (frontend)
     - a ``Stateful(...)`` call that raised was silently dropped
   * - H3
     - bug, fixed (frontend)
     - ``Stateful`` wider than 64 bits could not be built
   * - H4
     - info, pre-existing (SystemC)
     - ``UInt`` ``Stateful`` declared signed; bits unchanged
   * - H5
     - info (Catapult)
     - ``RESET_CLEARS_ALL_REGS no`` is design-wide

Files
=====

* ``catapult/w16/``, ``catapult/w256/``: ``kernel.cpp`` as emitted (``w256``:
  ``kernel.cpp.emitted`` plus the TB patch), ``run.tcl`` as emitted,
  ``csyn.log.gz``, ``rtl.rpt.gz``; ``catapult/latency.json``.
* ``dc/``: ``run_dc.sh`` (paths), ``unrst_w16``, ``unrst_w256`` (``area``, ``qor``,
  ``reference`` reports).
* ``logs/``: ``cmp_w16.txt``, ``cmp_w256.txt``, ``flops_w16.txt``,
  ``flops_w256.txt``, ``check_comb.txt``, ``csim_offset.txt``,
  ``tinytpu_hash.py``, ``tinytpu_hash.txt``, ``regress.txt``.

..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##########################################################
U2: ``Wire[T, comb]`` implemented (README D-13), measured
##########################################################

.. note::

   **Dated measurement record, 2026-10-02.** zhang-21, branch
   ``backend-comb-wire`` (worktree ``scratch/wt-comb`` from ``origin/u1-pilot``
   at ``b029bc94``), its own ``mlir/build`` (configured as ``main``'s:
   ``LLVM_DIR``/``MLIR_DIR`` under ``/work/shared/common/llvm-project-main/
   build-rhel8``, gcc-toolset-13, the ``allo`` env's python and ninja). MiniTPU
   at ``b3ba0a4d``, read only. Catapult 2024.2 (``nangate-45nm_beh``), DC
   W-2024.09 (FreePDK45, ``dc_u1.tcl`` byte-identical to the U1/U2 flow),
   Verilator 5.052. Scratch (Catapult, DC and csim projects, not kept):
   ``scratch/u2comb/``. Record files: ``u2_comb_wire_impl_2026-10-02/``.

``u2_comb_read_2026-10-02.rst`` found the form Catapult builds a clockless
read from (its form e: one clocked ``SC_THREAD`` for the write, one
``SC_METHOD`` for the reads, over an array of ``sc_signal`` zeroed in the reset
action) and proved it with a 121-line hand patch on Allo's emission. README
D-13 chose the explicit marker. This record is the implementation: the
marker, the emitter, the refusals, ``latency.json``, the validation of the
register file at both widths with **no hand patch on the design**, and the
bounded F3 study of an unreset storage.

**Result.** ``Wire[T, comb]`` (or ``Wire(T, (), comb=True)``) is carried in
the IR type (``!allo.wire<ui16, comb>``); the SystemC emitter builds each
comb port's cone as an ``SC_METHOD`` over signal storage and refuses, naming
the port, every cone the method cannot be. The regfile's ``comb`` variant
gives Catapult RTL that is **bit-exact against MiniTPU at read latency 0,
write visible after 1 edge: w16 180,780/180,780, w256 45,744/45,744**, with
the same Catapult and DC numbers as the record's hand patch (w16 DC 4,463.2
um^2, +11.0 % over MiniTPU, all of it reset flops). **F3:** Catapult *can*
synthesize unreset signal storage, but only when its writer is a clock-edge
``SC_METHOD`` (no reset action) under ``-RESET_CLEARS_ALL_REGS no`` -- then
the DC area is MiniTPU's to the um^2 (4,021.9 vs 4,021.7). No directive
exempts a thread's reset action from CIN-233. The emitter keeps the reset
(a thread writes the storage); the clocked-method write form is proposed.

Reproduce (``source examples/minitpu/harness/env-zhang21.sh`` first; ``R`` is
this record's directory, ``S`` a scratch directory)::

   # frontend, emitter, refusals, csim
   $ALLO_PYTHON -m pytest tests/dataflow/test_systemc_comb.py -q
   $ALLO_PYTHON examples/minitpu/harness/check.py vpu_regfile --variant comb --variant wire \
       --backend systemc --project $S/prj                     # per-iteration (limitation 22)
   $ALLO_PYTHON dev/records/minitpu/u2_comb_read_2026-10-02/scripts/csim_offset.py \
       $S/prj/vpu_regfile_w16_comb_systemc                   # offset-aligned
   # Catapult, w16 and w256 (the comb-read record's emit_csyn.py; no --partition, no patch)
   $ALLO_PYTHON dev/records/minitpu/u2_comb_read_2026-10-02/scripts/emit_csyn.py vpu_regfile comb \
       $S/w16 --n 64 --pipeline rf_0:_ --synth-top rf_0 --clock 3.33 [--width 256 --no-run]
   #   w256 only: the pilot's S7 testbench patch before running Catapult (the TB, not the design):
   #   sed -i -E 's/\(long long\)\((ch_v[0-9]+\.Pop\(\))\)/(\1).to_int64()/' $S/w256/kernel.cpp
   $ALLO_PYTHON dev/records/minitpu/u2_comb_read_2026-10-02/scripts/cmp_rf.py \
       $S/w16/build/Catapult/rf_0.v1/concat_sim_rtl.v --top rf_0 \
       --ports v18,v19,v20,v21,v22,v23:v24,v25,v26 [--inst w256]
   $R/dc/run_dc.sh comb_w16 rf_0 clk 3.33 $S/w16/build/Catapult/rf_0.v1/concat_rtl.v
   # latency.json: Allo's own csyn mode
   $ALLO_PYTHON -c "import allo.dataflow as df; from examples.minitpu.units import vpu_regfile as u; \
       s = df.customize(u.comb_read(64, 16)); s.pipeline('rf_0:_'); \
       s.build(target='systemc', mode='csyn', project='$S/csyn', configs={'clock_period': 3.33, 'synth_top': 'rf_0'})()"
   # F3 forms: copy $R/f3/<a..f>/{kernel.cpp,run.tcl}, run catapult -shell -f ../run.tcl in build/

F3: can Catapult leave the storage unreset? (bounded, 1 h)
===========================================================

All on the comb-read record's hand-written form d1 (``catapult/d1_sigarr``:
``sc_signal<data_t> mem[32]``, ``SC_THREAD(wr)`` under ``clk.pos()`` with
``async_reset_signal_is(rst, false)``, ``SC_METHOD(rd)``), 3.33 ns. Messages
verbatim from ``f3/<id>/csyn.log``.

.. list-table::
   :header-rows: 1
   :widths: 4 34 62

   * - id
     - change
     - Catapult
   * - a
     - the reset action no longer writes ``mem`` (control)
     - **refused**: ``Error: kernel.cpp(42): 'mem(0)' must be set in reset action,
       preserving state across reset is not supported (CIN-233)`` x32, ``Compilation
       aborted (CIN-5)``; exit 2
   * - b
     - a + ``directive set -RESET_CLEARS_ALL_REGS false`` before ``go analyze``
     - ``Warning: Boolean directive values of RESET_CLEARS_ALL_REGS directive is
       deprecated (SIF-24)``, ``/RESET_CLEARS_ALL_REGS no``; then **CIN-233 x32 as a**.
       The directive does not reach a SystemC thread's reset-action check
   * - c
     - the write as a **clock-edge ``SC_METHOD``** (``SC_METHOD(wr); sensitive <<
       clk.pos();``; body ``if (we) mem[wa].write(wd);``; no reset action anywhere)
     - **synthesizes** (exit 0; ``Final schedule of SEQUENTIAL '/rf_comb/wr': Latency =
       1``). But Catapult adds a reset itself: ``rf_comb_wr`` gets ``input rst`` and
       every flop ``if ( rst ) mem_k <= 16'b0`` -- active HIGH, so the harness (which
       holds the generated side's active-low ``rst`` high) read zeros:
       8,502/180,780 (``f3/cmp_c.txt``, a polarity artefact of the comparison, not a
       verdict). Area score 3,268.6
   * - d
     - ``SC_THREAD(wr)`` with no reset signal and no reset action
     - **refused**: ``Error: kernel.cpp(36): Expected at least one reset to be defined
       for process 'wr' (CIN-194)``, ``Compilation aborted (CIN-5)``
   * - e
     - a + per-path ``directive set /rf_comb/mem -RESET_CLEARS_ALL_REGS false`` and
       ``/rf_comb/wr/mem:rsc ...`` after ``go compile``
     - **not reached**: CIN-233 aborts ``go compile`` before any per-path directive
       can be set; the check is in the front end, not the scheduler. Same log as a
   * - **f**
     - c + ``directive set -RESET_CLEARS_ALL_REGS false``
     - **synthesizes, unreset**: ``if ( rst )`` occurs 0 times in ``concat_rtl.v``;
       ``rst`` stays a top-level input (``Warning: Input port 'rst' is never used.
       (OPT-4)``). Verilator trace vs MiniTPU: **180,780/180,780 at read 0, write->read
       1** (``f3/cmp_f.txt``). DC (``dc/f3_f``): **4,021.9 um^2** (comb 1,579.0, seq
       2,442.9, slack 2.80 ns), all storage ``DFF_X1`` -- MiniTPU's 4,021.7 (1,578.7 /
       2,442.9) to the um^2. Catapult area score 2,315.3

So an unreset storage is not "unsupported" in Catapult; it is unsupported
**in a thread**. A signal written by an ``SC_THREAD`` must be set in its
reset action (CIN-233), and no directive, option or declaration site tried
here exempts it; a thread cannot be reset-less (CIN-194). A signal written by
a clock-edge ``SC_METHOD`` has no reset action to satisfy, and with
``RESET_CLEARS_ALL_REGS no`` the flops come out plain.

**Decision (provisional, D-9).** The emitter keeps the reset: the storage's
writer is the kernel's thread, which is where every other store, handshake
and ``wait()`` of the kernel lives, and D-13 names the reset as the recorded
deviation. The unreset form needs a *third* process -- the stores to comb
storage as a clocked ``SC_METHOD``, whose cone (address, data, enable) would
have to be classified like the read cone (Wire inputs and temporaries only;
no stream, no state) -- and a ``run.tcl`` directive. That is the proposal
``F3-unreset`` below; it is not needed for a match and was not built.

Design
======

**Marker** (``allo/ir/types.py``, ``infer.py``, ``builder.py``; the dialect).
``comb`` is a marker object in ``allo.ir.types``; ``Wire[T, comb]`` in a body
annotation (read off the AST by ``infer.py``) and ``Wire[T, comb]`` as a value
(``Wire.__class_getitem__``, new -- ``Wire[...]`` did not evaluate before) both
give ``Wire(dtype, shape, comb=True)``. The IR type carries it:
``!allo.wire<ui16, comb>`` (``WireType`` gained a ``comb`` parameter with a
custom assembly format; CAPI ``alloMlirWireTypeGet(ctx, base, comb)`` and the
``WireType.get(base_type, comb=False)`` / ``.comb`` bindings). Every kernel
argument, call operand and put/get sees the flag with no further plumbing.
``@df.unit`` ports remain ``Stream``-typed in the netlist (``as_stream``); a
``Wire`` port of a unit is a pre-existing gap, now written down in
``stream_ports.rst``.

**Emitter** (``mlir/lib/Translation/EmitSystemC.cpp``, ``planComb`` and the
``combMode`` hook; ``EmitVivadoHLS`` gained a ``skipOp`` hook its
``emitBlock`` consults). For a kernel with comb ``Wire`` outputs:

1. *Storage* a cone reads -- a kernel-local array or an ``@ Stateful`` global
   -- is declared as a module member ``sc_signal<T> name[N]`` (flattened),
   read with ``.read()`` everywhere and written with ``.write()`` in the
   thread, and written in the reset action (zero for a local, the initial
   value for a Stateful). A plain member cannot feed an ``SC_METHOD``
   (CIN-197); an ``sc_signal`` array is registers by construction, so the
   pilot's ``s.partition`` / ``hls_resource [Register]`` is not needed.
2. *One* ``SC_METHOD(comb)`` per kernel, sensitive to the ``Wire`` inputs
   the cones read and to every storage element, emits the cones in program
   order: the put, its value's ``wire_get``\ s, storage loads, the
   iteration's scalar temporaries with the store that reaches each, pure
   arithmetic. The thread is emitted first (``combMode = Thread``: it skips
   the puts and, by a dead-code pass over the cone, whatever only they
   consumed -- the regfile's thread keeps exactly the write), then the method
   (``combMode = Method``: the body block with everything outside the cone
   skipped; the cone's values are taken off the name table for that emission
   and restored after, since the method is another C++ scope). A comb
   ``sc_out`` has no reset-action write: the method is its only driver.
   The emitted ``rf_0`` is the hand patch's ``e_wire_patch/kernel.cpp``
   line for line, up to value numbering.
3. *The rule* (refused at build, naming the port): exactly one unconditional
   put per iteration at the top of the iteration block; the cone reads
   storage only before any store to it in the iteration (program order,
   nested regions included); no stream/channel op, no memory port, no
   control flow, nothing defined outside the iteration (induction variable,
   loop-carried values), no constant array, no view; the thread's own loads
   of comb storage must also precede its stores (a signal reads old until
   the next delta). Diagnostics: ``comb port `w_qa` (rf_0): its value reads
   mem after a store to it in the same iteration; a combinational output sees
   only the state loaded before the iteration's stores (README D-13). Read
   first, then store``; ``... the put is under control flow ...``; ``... reads
   a stream or channel ...``. ``hls.py`` now captures the emitter's stderr
   during emission and puts it in the ``RuntimeError`` (a Python diagnostic
   handler on the context segfaulted), so the port and the reason reach the
   caller.

**Other backends.** Vitis and the Catapult C++ flow already refuse every
``Wire`` (``hls.py``, ``NotImplementedError``), so they refuse ``comb``. The
simulator could only run a comb port as an ordinary value (it is untimed:
``untimed`` is what the harness reports for it) -- but it has no wire
semantics at all: a ``Wire`` design failed inside the ExecutionEngine with
``missing LLVMTranslationDialectInterface registration``. ``df.build(...,
target="simulator")`` now refuses ``!allo.wire`` / ``!allo.channel`` with a
``NotImplementedError`` that says so. Choice: refuse, because "run as a
value" would need the simulator to lower wires first, which is a separate
piece of work, and a silent fallback is what D-1 forbids.

**``latency.json``** (``allo/backend/catapult.py`` ``write_latency_manifest``,
``comb_ports_of``). The emitter leaves ``// allo comb ports: v24 v25 v26`` in
the module's constructor; the manifest adds ``"ports": {"v24": "comb", ...}``
to the kernel's entry and ``[latency] rf_0: latency=1 ii=1 scheduled comb
ports ['v24', 'v25', 'v26']`` to the build log. ``latency``/``ii`` stay the
clocked thread's (the write path); ``comb`` is not ``0`` (D-10). Measured
(``catapult/latency.json``)::

   "rf_0": {"ii": 1, "latency": 1, "loop": "while", "loop_c_steps": 1, "port_style": "wire",
            "ports": {"v24": "comb", "v25": "comb", "v26": "comb"}, "process": "/rf_0/run",
            "status": "scheduled"}

Validation
==========

``examples/minitpu/units/vpu_regfile.py`` gained the ``comb`` variant: the
``wire`` body with ``w_qa/w_qb/w_qc: Wire[UInt(w), comb]``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - check
     - result
   * - csim, per iteration (``check.py --backend systemc``)
     - ``UNIT-DIFF vpu_regfile:w16 comb systemc 23010/180780`` -- the ``wire`` control
       gives ``21217/180780``: a ``Wire`` link is not cycle-locked in csim (limitation
       22; the comb-read record's F7). Offset-aligned (``csim_offset.py``, 64 cycles):
       **comb +3/+2/+2** cycles (60/61/61 of 63 defined slots agree), wire +4/+3/+3
       (59/60/60) -- the record's F7 numbers exactly, one cycle less than the thread
       form (``logs/csim_offset.txt``)
   * - Catapult w16 (``cat_w16``)
     - ok, 36 s, no ``MEM-74``, no partition. ``rf_0_comb`` (clockless) + ``rf_0_run``
       + ``rf_0``; area score **4,219.4**, max delay 0.465 ns (hand patch e: 4,219.4 /
       0.465). Trace: **180,780/180,780 at read latency 0; write -> read 1** on all
       three ports (``logs/cmp_w16.txt``)
   * - DC w16 (``dc/comb_w16``)
     - **4,463.2 um^2** (comb 1,601.1, seq 2,862.2), slack 2.80 ns at 3.33 ns --
       identical to the record's e (4,463.2 / 1,601.1 / 2,862.2). +11.0 % over MiniTPU
       (4,021.7): the sequential +419 um^2 (+17.2 %) is ``DFFR_X1`` for ``DFF_X1``
       (``reference.rpt``), the reset of F3
   * - Catapult w256 (``cat_w256``)
     - ok. The emitted TB prints a 256-bit port through ``long long`` (the pilot's S7,
       ``u2_regfile_2026-10-02.rst``; pre-existing, ``d6877a2f``), which Catapult's
       ``go analyze`` refuses (CRD-413) because the TB is in the file; the pilot's
       recorded TB patch (``.to_int64()``, ``catapult/w256/tb_patch_S7.diff``, 3 lines,
       testbench only) was applied, the design untouched. Area score **64,658.4**, max
       delay 0.522 (record d1 w256: 64,642.0 / 0.522). Trace: **45,744/45,744 at read
       0; write -> read 1** (``logs/cmp_w256.txt``)
   * - DC w256 (``dc/comb_w256``)
     - **67,084.7 um^2** (comb 23,364.9, seq 43,719.8), slack 1.96 ns -- the record's d1
       w256 (67,071.1 / 23,362.0 / 43,709.1 / 1.96); -18.9 % vs MiniTPU's 82,738.8, as there
   * - ``latency.json``
     - above; ``LATENCY-`` verdicts unchanged in kind (``harness/latency.py`` reads
       ``latency``/``ii``/``status``; ``ports`` is beside them)

Regression
==========

Own bindings for the worktree (``mlir/build``, ninja, 336 steps clean).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - check
     - result
   * - ``tests/dataflow/test_systemc_comb.py`` (new: marker, emit shape, three
       refusals, other backends, csim compile-and-run at offset +3)
     - 7 passed
   * - ``tests/dataflow/test_systemc_backend.py`` +
       ``test_systemc_csim_regress.py``
     - 60 passed (3:54)
   * - TinyTPU emission, ``df.customize(U.tinytpu_isa).build(target=...)``,
       sha256 of the text, base (``origin/u1-pilot``, same bindings) vs branch
     - vhls ``9a0eb0a3...``, catapult ``962f31d3...`` on both: **identical**. The
       prefixes the task names (``6bc774bc``/``ade1ab5d``) were taken on ``main`` at
       ``06650725`` with a recipe that is not on disk; ``df.build(..., wrap_io=False,
       project=tmp)`` is not even stable run to run (the project path is in the text).
       Identity between the two trees under one recipe is the claim
   * - ``examples/tinytpu/systemc_csim.py 3``
     - 3x ``wrong=0/16`` (``clobbered_outside`` 4057 / 4057 / 4065, the recorded baseline of
       ``dev/records/catapult_handoff/zhang21_compare_2026-10-02``)
   * - ``examples/eva/cosim_eva_systemc.py``
     - ``COSIM RESULT : PASS (bit-exact)``; ``examples/eva/generated`` restored after
   * - U1 units, ``check.py <unit> --backend systemc`` (bf16_add bf16_add_pipe
       bf16_mul bf16_mul_pipe mul_acc24 acc24_add_pipe alu)
     - all 24 verdict lines (every variant of the seven units) identical to
       ``u1_integration_2026-10-02/<unit>.log`` -- the ``bits`` forms match, the ``native``
       forms differ on the recorded NaN / signed-zero slots only (``logs/regress.txt``)

Findings
========

**G1 (Catapult, match).** Allo's own emission of a ``comb``-marked register
file is the comb-read record's form e with no hand patch: read 0 / write ->
read 1, bit-exact at both widths, same Catapult and DC numbers.

**G2 (Catapult, info; amends the comb-read record's F3).** Unreset signal
storage synthesizes when its writer is a clock-edge ``SC_METHOD`` and
``RESET_CLEARS_ALL_REGS`` is ``no`` (form f: MiniTPU's area to the um^2). A
thread's reset action admits no exemption (CIN-233, CIN-194; b, d, e). The
reset stays in the emitted form and is reported as the reset-flop share.

**G3 (SystemC emitter, limitation 22, unchanged).** csim of a Wire design is
offset, not wrong: +3/+2/+2 for the comb form.

**G4 (SystemC emitter, bug, pre-existing: S7 of the U2 pilot).** A port wider
than 64 bits crosses the testbench as ``long long``; Catapult refuses the
file at ``go analyze``. Worked around for w256 with the pilot's TB patch;
unfixed here (the proposed fix is in the pilot record).

**G5 (Allo frontend, gap).** A ``@df.unit`` port cannot be a ``Wire``: the
netlist types ports as ``Stream`` (``as_stream``), and an instance binding a
``Wire`` is refused as a scalar value argument. The marker therefore lives
on region-scope links, which is where the regfile's ports are.

**Proposals.**

- ``F3-unreset``: emit the stores to comb storage as a clocked
  ``SC_METHOD`` (their cone classified like the read cone) and set
  ``RESET_CLEARS_ALL_REGS no`` in ``run.tcl``, behind an explicit option,
  for a storage that must not reset (MiniTPU's). Expected: -419 um^2 at w16.
- ``unit-wire-ports``: generalise ``as_stream`` / ``Port`` to the three link
  kinds so ``Wire[T, comb]`` can be a unit port as D-13 words it.

Classification (D-9)
====================

.. list-table::
   :header-rows: 1
   :widths: 8 24 68

   * - id
     - class
     - one line
   * - G1
     - match (Catapult)
     - the comb marker reproduces form e without a hand patch, both widths
   * - G2
     - info (Catapult)
     - unreset storage needs a clocked SC_METHOD writer + RESET_CLEARS_ALL_REGS no
   * - G3
     - limitation 22 (csim)
     - offset +3/+2/+2, values right
   * - G4
     - bug, pre-existing (SystemC testbench)
     - 256-bit ports through ``long long`` (S7)
   * - G5
     - gap (frontend)
     - unit ports are Streams; a Wire port is refused

Files
=====

* ``f3/<a..f>/``: ``kernel.cpp``, ``run.tcl``, ``csyn.log.gz``; ``f3/cmp_c.txt``,
  ``f3/cmp_f.txt``.
* ``catapult/w16/``, ``catapult/w256/``: the emitted ``kernel.cpp``
  (``w256`` also ``kernel.cpp.emitted`` and ``tb_patch_S7.diff``),
  ``run.tcl``, ``csyn.log.gz``, ``rtl.rpt.gz``; ``catapult/latency.json``.
* ``logs/``: ``cmp_w16.txt``, ``cmp_w256.txt``, ``check_comb.txt`` (csim
  per-iteration), ``csim_offset.txt``, ``regress.txt``, ``tinytpu_hash.txt``.
* ``dc/``: ``run_dc.sh`` (paths), ``comb_w16``, ``comb_w256``, ``f3_f``
  (``area``, ``qor`` reports).

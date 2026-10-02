..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#########################################################################
Unsigned index sign-extended, and eleven more from the U2 unit tracks
#########################################################################

:Date: 2026-10-02
:Host: zhang-21
:Branch: ``core-uint-index`` (review branch, based on ``origin/main``
   ``db184ebc``; eleven commits, one per bug), and
   ``latency-manifest-fix`` (one commit, based on ``origin/u1-pilot``
   ``8c48499f``, for C-M1 whose code is not on ``main``)
:Found by: the MiniTPU U2 ``vpu_regfile`` pilot
   (``dev/records/minitpu/u2_regfile_2026-10-02.rst`` on ``u1-pilot``: B4,
   B5, B6, S6, S7), the comb-read probe
   (``u2_comb_read_2026-10-02.rst``: F5), the ``vpu_word_array`` track
   (``u2_word_array_2026-10-02.rst``: S8, C-W1, C-M1) and the FIFO track
   (``u2_fifo_2026-10-02.rst``: B7, S9, C1); repros in each record's
   ``repros.py``
:Bindings: the worktree's own ``mlir/build`` (configured like ``main``'s:
   LLVM/MLIR ``/work/shared/common/llvm-project-main/build-rhel8``,
   gcc-toolset-13, the ``allo`` env's python and ninja); three of the six
   fixes are C++

The bugs
========

**B4** (core, silent, high). ``build_cast_op`` mapped ``(UInt, Index)`` to
``arith.index_cast``, which sign-extends. On the LLVM backend and in the
dataflow simulator a ``uint8`` index of 200 read ``mem[-56]``, a ``UInt(5)``
address of 20 read ``mem[-12]``: out of bounds, consistently, and silently
(the pilot's regfile *matched* as written because 16..31 aliased 32 distinct
words; the same expression on a region-scope global segfaulted). The HLS
emitters were right by accident -- they index with the C type,
``uint8_t``. Same family as B1 (``uint_compare_2026-10-02.rst``): the
builder read signedness off a signless MLIR type::

    def k8(ra: uint8[4], q: uint16[4]):
        mem: uint16[256] = 0
        for i in range(256):
            mem[i] = i
        for t in range(4):
            q[t] = mem[ra[t]]
    # main: [1, 65528, 0, 65528] for idx [1, 200, 255, 128]; `arith.index_cast %1 : i8 to index`
    # fixed: [1, 200, 255, 128];                              `arith.index_castui`

**B5** (core, loud). ``x: int32 = s.get()`` on a ``Stream[UInt(5)]`` stored
the ``i5`` into an ``i32`` memref with no cast: ``build_assign_stmt``
skipped the conversion for every get op, and the verifier refused the module
(``'affine.store' op value to store must have the same type as memref
element type``). Workaround was a two-step widen through a ``UInt(5)``
local.

**B6** (simulator/LLVM, silent corruption). A numpy array narrower than an
``i<N>``/``ui<N>`` argument with N > 64 was passed as is after an "Input
type mismatch" *warning*; the LLVM element is N rounded up to a power of
two bits, so a ``uint64`` array for ``UInt(256)`` was read 4x past its end
(SIGABRT at n=4; heap corruption and a hang at the regfile). An object array
of Python ints -- the only container for such values -- was refused.

**S6** (SystemC emitter, silent). A kernel's 1-D argument array read once per
iteration becomes a Connections stream and its ``a[i]`` a ``Pop()``.
``isSeqStreamable`` checked the index map and the loop count but not what
sits between the access and its loop: ``if we[t]: mem[wa[t]] = wd[t]``
popped ``wa``/``wd`` only when ``we``, and the streams fell out of step with
the loop (``trace_raw`` csim 1,266/180,780; the six-element repro gives
``[10,10,10,11,11,12]`` for the simulator's ``[10,10,10,13,13,15]``).

**S7** (SystemC emitter, loud). The csim testbench moved every integer port
and memory through ``long long``: a ``UInt(256)`` port failed at read-back
(``KeyError 'ui256'``), and Catapult refused the file (CRD-413: no
conversion from a plain ``ac_int<256>`` to ``long long`` under
``__SYNTHESIS__``; the tb is in the file even when ``synth_top`` is the
unit). w256 needed a hand-patched tb.

**F5** (core, loud). ``s.partition("rf_0:mem", Complete)`` on a ``mem: T[32]
@ Stateful`` crashed in ``find_buffer`` (``'NoneType' object has no
attribute 'operations'``: the kernel's ``if we:`` is an ``scf.if`` with no
else block). Past the crash the array could not be found -- its global is
``__stateful_rf_0_mem_1``, not ``mem`` -- and, had it been, the get_global's
type would have changed but not the global's, and the LLVM lowering refuses
a global with a layout (``failed to legalize operation 'memref.global'``).

**S8** (SystemC emitter, silent, high; S7's twin). The tb read a <= 64-bit
port through ``long long``: a value >= 2^63 overflows the extraction, which
stores ``LLONG_MAX`` and sets failbit, and failbit sticks, so every later
value of that file was lost: 2/67,717 slots right on each 64-bit word-array
variant, the 16-bit instance matching. Repro ``[1, 2^63-1, 2^63+5, 7] ->
[1, 2^63-1, 2^63-1, 2^63-1]``.

**C-W1** (SystemC emitter -> Catapult, silent, high). A constant-trip inner
loop (the shift-register pipe) in a Wire kernel's steady-state loop was
emitted rolled, and Catapult merged it into the pipelined loop::

    Loop '/wa_0/run/l_S_k_0_k' is left rolled. (LOOP-4)
    Prescheduled LOOP '/wa_0/run/while' (2 c-steps) (SCHD-7)

so the unit sampled its Wire inputs every second cycle while reporting
II=1 (19,502/67,717). The latency manifest flagged ``unreliable`` ("rolled
loop(s) ['l_S_k_0_k'] merged into the scheduled loop"), so the number was
not consumed, but the design was still built. ``s.unroll`` on both shift
loops gave cycle-exact RTL.

**C-M1** (latency manifest, ``u1-pilot`` only). For a Wire kernel the
manifest took cycle.rpt's process latency, which counts the reset action's
c-step (``run:rlp``) with the steady-state loop: latency 2 where the RTL
measures 1 (LATENCY-MISMATCH on the unrolled word array; the regfile's L=1
matched by coincidence).

**B7** (simulator, silent). ``junk: uint1 = s.try_put(x)`` pushed nothing:
``empty()`` stayed 1 after four of them; with the flag read, the same code
works. Not the canonicalizer and not the stream lowering (both keep the
op): ``cleanUpUnusedOps`` in ``MemRefDCE.cpp``, run twice by the
composite-type lowering on every LLVM/simulator build, erased every op with
results and no uses by ``use_empty()`` alone -- the lowered ``try_put`` is
an ``scf.if`` whose result is the success flag and whose then-block pushes
the word.

**S9** (SystemC emitter, silent). A self-FIFO's ``full()`` reads a
synchronous counter that ``try_put`` advanced by its ``PushNB`` result; the
enq Combinational holds one value in flight beyond the AlloFifo's depth, so
the fifth push into ``Stream[int32, 4]`` still succeeded, the counter ran
to depth + 1 and ``full()`` was never true again: ``[0,0,0,1,0,0]`` for the
simulator's ``[0,0,0,1,1,1]``.

**C1** (SystemC -> Catapult, D-1). The self-FIFO lowering (an AlloFifo
self-loop with ``_enq``/``_deq`` ports and the counter) cannot be scheduled
by Catapult in any loop that puts and gets conditionally: ``could not
schedule partition '/top/fifo_0/run' even with unlimited resources``
(SCHD-30), with and without pipelining, with and without the reset drain.
csim runs it bit-exactly; the build failed late in csynth instead of being
refused.

The fixes
=========

``[Builder] Zero-extend an unsigned array index (index_castui)``
   ``(UInt, Index)`` builds ``arith.index_castui``; so does the second step
   of float -> index (its ``fptoui`` result is unsigned). ``(Index, UInt)``
   stays ``index_cast``: a 64-bit index into a <= 64-bit UInt truncates
   identically either way, and the loop-transform passes match
   ``index_cast`` on an induction variable. Every UInt -> Index site goes
   through ``build_cast_op`` (indices, slice bounds, loop bounds, affine
   maps), so the one table entry covers them. The HLS emitters learn
   ``arith::IndexCastUIOp`` (``Visitor.h``; the Vivado, Intel and TAPA
   expression visitors -- Catapult and SystemC reuse Vivado's), emitted
   like ``index_cast``: ``int v = x;`` from an unsigned C variable.
   Residual, noted: the HLS index type is C ``int``, so a ``uint32`` index
   at or above 2^31 is negative in the generated C++ too (no design has
   one).
``[Builder] Cast a scalar stream get to the assigned type``
   A scalar ``get`` (Stream, Wire, Channel) is cast with ``build_cast_op``
   like any other scalar right-hand side; array gets and construct ops are
   untouched.
``[Backend][LLVM] Pack integer arguments wider than 64 bits, never read past the buffer``
   ``LLVMModule.__call__`` packs a > 64-bit argument element by element from
   Python ints (``utils.pack_wide_int_array``: any integer dtype or
   ``dtype=object``, little-endian two's complement, range-checked, a
   non-integer dtype refused) and reads it back as Python ints
   (``unpack_wide_int_array``; ``struct_array_to_int_array`` returns the
   same above 64 bits instead of a struct array and a warning): whole into
   an object array, into a narrower numpy array only if the values fit
   (``OverflowError`` otherwise -- loud, never truncated). A void-dtype
   array must be the element's full size. <= 64 bits unchanged. This is
   also the answer to H8 ("how to pass > 64-bit elements"): an object array
   of Python ints, on the simulator and (below) in SystemC csim.
``[Backend][SystemC] Move ports wider than 64 bits through the testbench as decimal text``
   An element wider than 64 bits (``wideIntWidth``) moves through the data
   files as decimal text of any length, by digit arithmetic on the ac_int
   itself (``_rdwide``/``_wrwide`` in the emitted header, every step an
   explicit ``T(...)`` so the csim shim and the bare ``ac_int`` alias both
   construct from the wider intermediate); stream source and sink, memory
   preload and replica-summing read-out. ``read_data`` parses it into Python
   ints. Narrower ports emit exactly as before; the Catapult-compiled tb no
   longer needs a hand-patch.
``[Backend][SystemC] Give a conditionally read argument a memory port, not a stream``
   An access with any non-loop region op (``scf.if``, ``affine.if``, ...)
   between it and its loop is not sequentially streamable; the array takes
   the memory-port path, which reads the element the body asks for when it
   asks. An array read unconditionally stays a stream.
``[Builder] Partition a @ Stateful array like any other``
   ``find_buffer`` guards the missing else block and matches a get_global by
   the variable name the builder now tags it with (``stateful_name``);
   ``partition`` gives the Stateful's ``memref.global`` the layout too;
   ``removeStrideMap`` strips it for the LLVM path as for an alloc; the
   SystemC emitter spells a complete partition of a Stateful member as
   Catapult's ``#pragma hls_resource <sym>_rsc ... [Register]`` before the
   member -- the pragma the comb-read probe hand-patched (form b2). Not
   done: the Vivado emitter prints a partitioned global as a plain static
   array with no ``array_partition`` pragma (pre-existing for every
   partitioned global).
``[Backend][SystemC] Read every ac_int port through the width-agnostic reader``
   S8. Ports are Catapult-native ac_int at every width (``int32``
   included), so every ac_int port and memory takes the digit
   reader/writer (``tbIntKind`` 'a'); a native unsigned goes through
   ``unsigned long long``, a native signed through ``long long``; an
   extraction that fails aborts the tb with the file and index. The writer
   takes the magnitude in a type one bit wider (``ac_int<8,true>`` -128
   otherwise wrapped and printed digits below '0' -- caught by TinyTPU's
   csim). This changes the tb's src/snk loops and memory preload/read-out
   for every integer port; the kernels are untouched.
``[Backend][SystemC] Refuse a self-FIFO try_put at full()``
   S9. ``ok = (cnt < depth) && enq.PushNB(v); cnt += ok;``: refused exactly
   when ``full()``, as in the simulator. Two non-blocking puts in one thread
   iteration still depend on the enq handshake register (no clock between
   them); that is the port's timing, not the counter's, and the test says so.
``[Backend][SystemC] Unroll a constant-trip loop inside a Wire kernel's step``
   C-W1. ``emitLoopDirectivesPreheader`` writes ``#pragma hls_unroll``
   before a constant-trip loop nested in a steady-state loop of a kernel
   with Wire ports, unless the schedule has unroll/parallel on it. A
   Connections kernel is left alone (its merged loop is what the manifest
   catches).
``[Backend][SystemC] Refuse a self-FIFO under every synthesis mode``
   C1. ``HLSModule`` refuses at emission for every SystemC mode but
   ``csim``, naming the stream and kernel and the SCHD-30 text, and saying
   what to write instead (an in-kernel ring, or the two ends in two
   kernels). The ring lowering is the record's proposal and a larger
   change; this is the D-1 half.
``[Pass] Memref DCE keeps an unused-result op that has side effects``
   B7. An op with regions is erased only when ``isOpTriviallyDead`` says
   so; region-less ops keep the old rule (allo's struct ops carry no effect
   interface and ``isLegal`` relies on unused ones being swept).
``[Backend][Catapult] Report a Wire kernel's latency without the reset c-step``
   C-M1, on ``latency-manifest-fix`` (``origin/u1-pilot`` + one commit): for
   ``port_style=wire`` the latency is the ``while`` loop's c-steps; the
   process count stays beside it as ``process_latency``. Worktree with its
   own copy of ``wt-u1``'s bindings (``ldd`` resolves
   ``libAlloMLIRAggregateCAPI`` inside the worktree). ``tests/test_latency_manifest.py``
   7/7 there (2 new); not merged into ``core-uint-index`` because the
   manifest code is not on ``main``.

Tests
=====

.. list-table::
   :header-rows: 1

   * - file
     - covers
     - on ``origin/main``
   * - ``tests/test_uint_index.py`` (9)
     - uint8 / UInt(5) / UInt(3) / uint16 indices at and above half range on
       LLVM and in the simulator, a uint8 loop bound of 200, int8 stays
       signed, the HLS text, float -> index
     - 7 fail
   * - ``tests/dataflow/test_df_stream_get_cast.py`` (5)
     - UInt(5)/Int(5) into int32, int32 into uint8, same type, a Channel
     - 4 fail
   * - ``tests/test_wide_int_args.py`` (10)
     - pack/unpack round trips and refusals, UInt(256)/Int(128) on LLVM and
       the simulator past 2^64, the MiniTPU uint64 case, OverflowError for a
       result that does not fit, UInt(9) unchanged
     - fails at collection (the helpers do not exist); the wide cases abort
       the process
   * - ``tests/dataflow/test_df_systemc_tb_wide.py`` (3 emit-only + 6 csim)
     - S6: one Pop for the unconditional array, read pins for the other,
       csim == simulator; S7: UInt(256)/Int(128)/UInt(72) stream ports and
       UInt(256) memory ports round-trip in csim, the tb text, a narrow
       port's tb unchanged. csim ones skip without ``MGC_HOME`` +
       ``SYSTEMC_HOME``
     - 2 emit-only fail (6 skip)
   * - ``tests/test_partition_stateful.py`` (7)
     - the layout on the global and get_global, LLVM runs (complete and
       block), Vitis/Catapult emit, the Register pragma only when
       partitioned, unknown name refused
     - 6 fail
   * - ``tests/dataflow/test_df_systemc_tb_wide.py`` (S8 additions: 1
       emit-only + 2 csim)
     - UInt(64) at and past 2^63 (the record's uint64 repro) and UInt(1024)
       round-trip in csim; a 64-bit and an int32 port's tb text
     - the emit-only one fails
   * - ``tests/dataflow/test_df_systemc_self_fifo.py`` (4 + 3 csim)
     - S9: the gate in the emitted kernel, six try_puts in the simulator
       and in csim, refused puts then gets then an accepted put; C1: csyn
       and ppa refuse with the SCHD-30 text, csim still builds
     - 3 fail (the gate, csyn, ppa)
   * - ``tests/dataflow/test_df_systemc_wire_unroll.py`` (3)
     - one pragma directly above the shift loop in a Wire kernel, not
       doubled by ``s.unroll``, none in the stream-ported form
     - 1 fail
   * - ``tests/dataflow/test_df_try_put_unused.py`` (3)
     - an unread try_put pushes (both forms of the repro); five unread
       try_puts fill a depth-4 stream with the first four words
     - the unread forms fail (the run did not finish on main)

All pass on the branch, the csim ones with the SystemC environment of
``dev/toolchains.rst`` (Catapult 2024.2's SystemC).

Impact vs main
==============

Baseline: a detached worktree at ``origin/main`` ``db184ebc`` borrowing the
primary checkout's ``mlir/build`` (same commit); the branch worktree builds
its own. Same host, same env (Vitis on ``PATH`` as the ``allo`` env sets
it).

.. list-table::
   :header-rows: 1

   * - check
     - origin/main
     - core-uint-index
   * - TinyTPU emitted Vitis (sha256 of ``str(s.build("vhls"))``)
     - ``6bc774bc...b2ef95``
     - identical
   * - TinyTPU emitted Catapult
     - ``ade1ab5d...88e038``
     - identical
   * - TinyTPU emitted SystemC (``hls_code``)
     - ``ba271c9b...``
     - ``208dc9d4...`` (37 lines differ): the header gains the ``tbIntKind`` helpers and every
       integer port's tb loop reads/writes through ``_rdwide``/``_wrwide``
       (S8); the kernels and the top are byte-identical
   * - ``systemc_csim.py 3`` (TinyTPU, SystemC csim)
     - SYSTEMC CSIM OK, 3 cases wrong=0/16
     - SYSTEMC CSIM OK, same three lines (clobbered_outside 4057/4057/4065); an earlier build of S8 failed here (the digit writer's magnitude at ``ac_int<8,true>`` -128), which is how that bug was caught before the commit
   * - ``cosim_eva_systemc.py`` (EVA)
     - PASS (bit-exact)
     - PASS (bit-exact); ``generated/`` restored after
   * - ``gen_isa.py --check``
     - ISA OK
     - ISA OK
   * - ``lift_units.py --check``
     - UNITS OK
     - UNITS OK
   * - ``bench_isa.py``
     - ALL EXACT
     - ALL EXACT
   * - ``stress_isa.py``
     - STRESS OK 640/640
     - STRESS OK 640/640
   * - ``act_compile.py --gate``
     - ACT GATE OK 12/12
     - ACT GATE OK 12/12
   * - ``pytest tests/dataflow --ignore=tests/dataflow/aie``
     - 123 pass, 22 fail, 50 skip, 1 xfail
     - 142 pass (+20 new), 23 fail (+1), 62 skip (+11 new), 1 xfail
   * - ``pytest tests/act``
     - 193 pass, 2 fail, 4 skip
     - 195 pass, 0 fail, 4 skip
   * - ``pytest tests/test_*.py``
     - 466 pass, 37 fail, 10 skip
     - 491 pass (+25 new), 37 fail, 10 skip

Per-test outcomes were diffed from the JUnit XML. Three pre-existing tests
changed outcome, all explained:

* ``tests/dataflow/test_systemc_backend.py::test_systemc_mem_port_store_emit``
  pass -> fail: a text expectation on the tb's memory read-out line
  (``_mem.mem[f];``), which S8 rewrote to the digit writer
  (``_s = ac_int<32, true>(_s + ..._mem.mem[f]);``). The test's intent (the
  tb reads the memory out) still holds; its assertion is updated in the S8
  commit, so on the branch as pushed it passes and the dataflow count is
  22 fail.
* ``tests/act/test_bindings.py`` (2) fail -> pass: the branch worktree has
  its own ``mlir/build``; the baseline borrows the primary checkout's.

The 23 dataflow failures on the branch are the baseline's 22 plus that one.

The pre-existing failures are environmental and the same on both sides
(Vitis HLS csim/csynth project builds, ``GLIBCXX_3.4.32`` for compiled
wrappers, ``past.verify`` absent, a ``/tmp/allo_test_pynq_prj`` owned by
another user, ``tests/act/test_bindings.py`` on the borrowed build) plus the
two ``test_stateful.py`` name-collision tests recorded in
``frontend_scoping_2026-10-02.rst``.

Why the published numbers cannot move: TinyTPU's Vitis and Catapult C++ are
byte-identical, so cosim (175/265/421/482/674) is untouched and was not
re-run. The SystemC emission changes only in the testbench (S8) and the
header; its csim and EVA's were re-run on the final build.

The U2 regfile itself (``u1-pilot``) was not re-run here: that branch
carries other emitter changes. What this batch changes for it: the ``trace``
variant no longer needs its B4/S6 workarounds, ``ported`` needs no two-step
widen, the w256 instance runs in the simulator with an object array and in
csim/Catapult with no hand-patch, and ``wire_stateful`` can be
``s.partition``\ ed (Register pragma emitted).

Upstream
========

``cornell-zhang/allo`` ``main`` at ``3f2ea5d4`` (2026-09-30) has B4
(``allo/ir/builder.py`` lines 551 and 589: ``(UInt, Index):
arith_d.IndexCastOp`` and the float -> index second step), B5 (line 1045:
``isinstance(rhs, (StreamConstructOp, StreamGetOp))`` -> no cast) and B6
(``allo/backend/llvm.py`` line 194: ``if bitwidth <= 64:`` with no else).
S6, S7 and F5's Stateful are fork-only. No upstream issue or PR matches
("index_cast", "unsigned index", "sign-extend" searched; #595/#612 are the
unrelated bit-slice sign fix). Draft below, not filed.

Draft upstream issue
--------------------

    **[Bug] An unsigned array index is sign-extended (`arith.index_cast`
    for `UInt`): silent out-of-bounds reads on the LLVM backend**

    `ASTTransformer.build_cast_op` maps `(UInt, Index)` to
    `arith.IndexCastOp`, which sign-extends. Any unsigned index at or above
    half its range becomes negative::

        import numpy as np, allo
        from allo.ir.types import uint8, uint16

        def k(ra: uint8[4], q: uint16[4]):
            mem: uint16[256] = 0
            for i in range(256):
                mem[i] = i
            for t in range(4):
                q[t] = mem[ra[t]]

        s = allo.customize(k)
        print(s.module)   # %2 = arith.index_cast %1 : i8 to index
        q = np.zeros(4, np.uint16)
        s.build()(np.array([1, 200, 255, 128], np.uint8), q)
        print(q)          # [1 65528 0 65528]; expected [1 200 255 128]

    The HLS emitters index with the C type (`uint8_t`), so the generated
    hardware is right and the LLVM backend disagrees with it. Two related
    defects in the same area:

    * `x: int32 = s.get()` on a `Stream[UInt(5)]` stores the `i5` into an
      `i32` memref with no cast (`build_assign_stmt` skips the conversion
      for every get op): `'affine.store' op value to store must have the
      same type as memref element type`.
    * `LLVMModule.__call__` passes a numpy array for an integer argument
      wider than 64 bits as is, after a warning only; the LLVM element is
      the next power of two bits, so a `uint64` array for `UInt(256)` is
      read 4x past its end (heap corruption).

    Fix: `arith.IndexCastUIOp` for `(UInt, Index)` (the HLS emitters need a
    `visitOp` for it, emitted like `index_cast`); cast a scalar get like
    any other right-hand side; pack > 64-bit arguments from Python ints
    with a range check and hand results back as Python ints. A patch with
    tests is on `sunwookim028/allo` branch `core-uint-index`.

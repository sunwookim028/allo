..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

###################################################################
``Stateful`` semantics (README D-11): M1 and M2, 2026-10-02
###################################################################

Branch ``core-stateful`` (review branch, not ``main``), base ``db184ebc``.
Host zhang-21, bindings built from the branch's own ``mlir/``. Evidence:
``dev/records/minitpu/u2_regfile_2026-10-02.rst`` (on ``origin/u1-pilot``),
findings M1 and M2, and its ``repros.py``. The decision is README D-11 on
``origin/u1-pilot``: *a* ``Stateful`` *persists across calls on every backend,
and one* ``Stateful`` *has one kernel*.

The findings
============

**M1 (semantic mismatch, csim/cosim).** The simulator keeps a ``Stateful``
across calls of one built module (a global of the JIT-compiled module).
SystemC csim started a new process per call -- ``allo/backend/hls.py``
compiles and runs the emitted testbench on every ``mod(...)`` -- so the reset
action re-initialised the state (``stateful`` variant: 170,093/180,780).
RTLGen's ``cosim()`` re-instantiates the RTL per call.

**M2 (silent, simulator).** Four kernels sharing one region-scope ``Stateful``
with nothing ordering them gave 8,502/180,780 and no diagnostic: the kernels
of a region run concurrently. Vitis, Catapult and SystemC each refused the
sharing in their emitter; the simulator did not, so the one backend that runs
first gave the wrong answer silently.

Design
======

**M2: refuse before any backend.** ``allo.dataflow.check_stateful_sharing``
runs in ``_build_top`` (so the simulator, every HLS target and AIE give the
same answer) and walks every ``df.kernel`` function for ``memref.get_global``
of a non-constant ``__stateful_`` global; a global with two or more kernels
raises ``RuntimeError`` naming the Stateful (source name and symbol) and the
kernels::

   Stateful `acc` (`__stateful_top_acc_1`) is used by 2 kernels: producer_0,
   reader_0. One Stateful has one kernel (README D-11): ...

A mapped kernel's PEs are separate kernels (``pe_0``, ``pe_1``): a
region-scope Stateful touched by ``mapping=[2]`` is shared. What can never
trip it: a kernel-local Stateful (one global per PE, ``__stateful_pe_0_spad``)
and a ``@df.unit`` instance's Stateful (one global per instance,
``__stateful_first_0_acc`` / ``__stateful_second_0_acc``, from the
``_stateful_seen`` renaming in ``customize.py``). Both were probed before the
check was written, and both are in the tests. The emitters' own refusals stay
as a second line; ``test_region_stateful.py``'s emitter tests now carry the
premise below so they still reach the emitters.

*The one escape, and why it exists.* ``examples/minitpu/microarch.py`` shares
``vregs`` across ``port_c`` / ``write_port`` / ``alu`` and ``vmem`` across
``vmem_compute`` / ``vmem_dma``, ordered by request/response and retirement
streams (``pc_req``/``pc_rsp``, ``wp_req``, ``wb_tok``, ``st_tok``,
``fill_done``) -- the record's ``shared_sync`` shape, which the simulator
honours (180,780/180,780) and which is the modelled form of "one memory,
several ports" until the memory-port proposal lands (G1). An unconditional
refusal would have broken the MiniTPU gate, and the task asked that in-tree
``compose``/MiniTPU designs not be caught wrongly. So the region can state the
premise, per Stateful, with a reason::

   @df.region(shared_stateful={"vregs": "every access serves a request the "
                                        "sequencer issued in program order"})

Nothing checks it -- exactly as nothing checks ``deadlock_free_because``
(``docs/source/developer/stream_ports.rst``, the obligation that is not a
rule). The simulator honours it; the HLS backends refuse the sharing
regardless, since they cannot express it. A name that is not a region-scope
Stateful of the region is refused. In-tree users: ``microarch.py`` and the
limits repro ``tests/limits/item01_region_stateful.py`` (whose sharing is the
item's shape, and whose driver output was never checked).

**M1: SystemC csim resumes from files, one process per call.** Alternatives
weighed: keeping one testbench process alive across calls (a different runner
protocol for every argument, and a different process model from the other
csims); patching the emitted ``kernel.cpp`` from Python (text surgery on an
emitter's output). Chosen: the emitter (``EmitSystemC.cpp``,
``emitStatefulStateIO``) gives every kernel ``SC_MODULE`` that owns stateful
members three methods under ``#ifndef __SYNTHESIS__``:
``__allo_state_save(std::ostream&)``, ``__allo_state_load(std::istream&)``
and ``__allo_state_resume()``. ``sc_main`` saves every stateful instance
(``t.dut.u<k>``) to ``allo_state_u<k>.data`` at exit -- after the run is
quiescent, not at the end of each thread's pass, because a stream sink's
``sc_stop()`` can land before a producer's thread reaches the end of its body.
The thread calls ``__allo_state_resume()`` right after its reset action's
``wait()``: the thread is out of reset once ``wait()`` returns, so a reload
can never race the reset that re-initialises the members (a poke from
``sc_main`` between ``t.rst = 1`` and the first clock could, depending on
whether the posedge at the end time is processed in the same ``sc_start``).
It reloads only when ``ALLO_STATE_RESUME`` is set, which the runner
(``HLSModule._csim_state_env``) sets on every call after the first; the first
call also deletes any ``allo_state_*.data`` a previous build left in the
project. ``mod.reset()`` clears the files and the call count. Values cross
as the data files do: integers as decimal ``long long``, floats as raw bits
(``_fbits`` / ``_ffrombits``), bit-exact. The RTL is unchanged: under
``__SYNTHESIS__`` none of it exists, and the contract there is "between
resets". The emitted file of a design with no Stateful changes by one line
(``#include <cstdlib>`` in the preamble).

``LLVMOMPModule.reset()`` (simulator) re-creates the execution engine from the
same lowered module, whose globals carry the initial values.

*Not done, documented in* ``dataflow_semantics.rst``: SystemC **cosim**
(SCVerify drives its own run; out of scope), the Vitis and Catapult csim
hosts (restart per call, as before), RTLGen/AMC cosim. Ladder verdicts keep
single-call traces there (D-11, last bullet).

Tests
=====

``tests/dataflow/test_stateful_semantics.py`` (7): persistence over three
calls, ``reset()``, and resumption after it, on the simulator and in SystemC
csim (region-scope int array, kernel-local scalar, region-scope float array;
the csim test also checks the state file's shape and that ``reset()`` removes
it; skipped cleanly without ``MGC_HOME``/``SYSTEMC_HOME``); the refusal
naming ``acc``, ``producer_0`` and ``reader_0`` from ``df.build`` and
``df.customize``; mapped PEs counted as kernels; the premise honoured by the
simulator and still refused by ``vhls``; an unknown premise name refused; a
single kernel owning a region-scope Stateful beside two ``@df.unit``
instances owning their own, built and persisting with per-instance globals.
``tests/dataflow/test_region_stateful.py``: the two simulator sharing tests
and the three parametrised emitter tests now carry the premise (their
assertions are unchanged). ``tests/limits/item01_region_stateful.py``:
premise added, verdict unchanged (``FIXED``).

Impact vs main
==============

Baseline: the detached worktree at origin/main ``db184ebc`` with ``main``'s
``mlir/build`` symlinked; the branch with its own build of the same sources
plus the emitter change; same host, same env (Catapult 2024.2 g++ for csim).

.. list-table::
   :header-rows: 1

   * - check
     - origin/main ``db184ebc``
     - core-stateful
   * - TinyTPU emitted Vitis (sha256 of ``str(s.build("vhls"))``, ``schedule(s)`` applied, ``TPU_MAXDIM`` unset)
     - ``6bc774bc...b2ef95``
     - ``6bc774bc...b2ef95`` (identical)
   * - TinyTPU emitted Catapult
     - ``ade1ab5d...88e038``
     - ``ade1ab5d...88e038`` (identical)
   * - the same two at ``TPU_MAXDIM=16``
     - ``6504a7c6...`` / ``3637396f...``
     - identical
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
     - STRESS OK 492/492
     - STRESS OK 492/492
   * - ``act_compile.py --gate``
     - ACT GATE OK 12/12
     - ACT GATE OK 12/12
   * - ``examples/tinytpu/systemc_csim.py 3 --project <scratch>``
     - 3x wrong=0/16, SYSTEMC CSIM OK
     - 3x wrong=0/16, SYSTEMC CSIM OK
   * - ``examples/eva/cosim_eva_systemc.py`` (``examples/eva/generated`` restored after)
     - COSIM RESULT: PASS (bit-exact)
     - COSIM RESULT: PASS (bit-exact)
   * - ``examples/minitpu/run.py --quick``
     - PASS
     - PASS (with the premise)
   * - ``pytest tests/dataflow --ignore=tests/dataflow/aie``
     - 26 failed, 165 passed, 4 skipped, 1 xfailed
     - 26 failed, 172 passed, 4 skipped, 1 xfailed: the **same 26** test by
       test; the 7 new tests pass (the csim one with the Catapult env)
   * - ``pytest tests/act``
     - 2 failed, 193 passed, 4 skipped
     - 195 passed, 4 skipped: the 2 are ``test_bindings`` (the baseline's
       build is ``main``'s symlink, so "the extension comes from this
       checkout" fails there by construction)
   * - ``pytest tests/test_*.py``
     - 37 failed, 474 passed, 2 skipped
     - 37 failed, 474 passed, 2 skipped: the same 37 test by test
   * - ``tests/limits/item01_region_stateful.py``
     - FIXED
     - FIXED (with the premise)

Every changed outcome: the 7 added tests, and the 2 baseline-only
``test_bindings`` failures explained above. No test that passed on
``origin/main`` fails on the branch. (A first branch run showed 13
``test_region_stateful.py`` failures with ``SystemExit`` from type inference;
that run overlapped a ``black`` reformat of the test file -- Allo reads kernel
source through ``inspect``/``linecache``, which reloads a changed file at
shifted offsets. The clean rerun above and an isolated run of the file, 23
passed, settle it.)

Not clean: ``scripts/lint/git-clang-format.sh`` rejects
``EmitSystemC.cpp`` as a whole (its ``indent(); os << ...`` convention
predates this branch, on untouched lines too); the new code keeps the file's
convention. ``black`` and the licence check pass.

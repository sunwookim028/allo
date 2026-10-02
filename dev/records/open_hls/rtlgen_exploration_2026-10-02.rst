RTLGen exploration (Kai Shao, ``kkkaishao/allo``)
==================================================

:Date: 2026-10-02 (work started 2026-10-01 22:40 EDT)
:Host: zhang-21.ece.cornell.edu (RHEL 8.10, glibc 2.28, 64 cores, 376 GB RAM)
:Scope: Read-only exploration. Nothing in ``/work/shared/users/phd/sk3463/allo``
        was modified. All clones, envs and builds are under
        ``/work/shared/users/phd/sk3463/scratch/rtlgen/``.
:Commits inspected:

   * ``kkkaishao/allo`` branch **``allo-rtlgen``** tip ``13b55a63``
     (2026-09-06, "Expand lit test coverage for Transforms passes and
     TransformOps"), 947 commits total. Note: the branch is named
     ``allo-rtlgen`` on Kai's remote; ``kai/allo-rtlgen`` is only the name
     under our ``kai`` remote. ``git clone --branch kai/allo-rtlgen`` fails.
   * ``kkkaishao/allo`` ``allov2`` tip ``b22ed847`` (2026-07-10);
     ``act`` ``3c1ad38d``; ``main`` ``a098603f``.
   * ``9e2d5716`` (2026-08-29) -- the ``allo-rtlgen`` commit imported verbatim
     into our ``chia-codesign`` (per ``ATTRIBUTION.md`` at tag
     ``chia-codesign-final``), i.e. the RTLGen the 2026-09-09 assessment
     measured. It is 23 commits behind the tip.
   * cornell-zhang/allo ``main`` ``3f2ea5d4`` (2026-09-30).
   * our ``main`` ``9fd02275`` (2026-10-02); ``35c38494`` (BACKEND_CHOICE.md).

Clone: ``scratch/rtlgen/kai-allo`` (remotes ``origin`` = Kai, ``upstream`` =
cornell-zhang, ``ours`` = local read-only fetch of our ``main``).


1. Input and output
-------------------

**Input.** A Python function in Kai's re-architected frontend (``allo.lang``,
not our ``allo.customize``): ``@kernel`` with fully annotated parameters, nested
``@kernel`` helpers, ``async def`` processes launched with ``await`` and wired
by ``Stream[T, depth]`` locals (KPN), ``Stateful[T]`` module state. A schedule
is a transform-dialect script built by ``kernel.schedule()``
(``allo/schedule/core.py``: ``pipeline``, ``unroll``, ``partition``,
``bind_storage``, ``split``, ``tile``, ``flatten``, ``compute_at``,
``reuse_at``, ``outline``, ``streamline``, ``compose`` ...). The RTL handle is
``kernel.schedule().export("rtl", device=..., freq_mhz=...)``
(``allo/backend/rtl/core.py``, class ``RTL``).

**Pipeline** (``allo/backend/rtl/schedule.py``, ``core.py``)::

   Python AST -> allo/func/affine/memref MLIR
     -> RTL_PREPARE_PIPELINE (grid-mapping, materialize-topology,
        convert-allo-to-func, float-to-int, outline-loose-processes, ...)
     -> raise-to-affine, loop-canonicalization, fold-if-statements,
        tree-height-reduction, rotate-reductions, narrow-demanded-bits,
        reconcile-array-directives, assign-banks, legalize-arith, ...
     -> run_sdc_scheduling: SDC heuristic or CP-SAT exact (OR-Tools),
        against a Device (operator rows with latency/delay/price)
        -> allo.dcp.* ops (DCP = "datapath/control plan" dialect ops)
     -> dcp-resolve-banking -> emit_datapath_to_hw  (mlir/lib/allo/Microarch)
        -> CIRCT hw/comb/seq MLIR
     -> CIRCT ExportVerilog -> SystemVerilog (single string or split files)

**Outputs** on the handle: ``.schedule()`` (per-region II/latency/start
times), ``.mlir`` (hw/comb/seq), ``.verilog``, ``.interfaces`` (port
manifest JSON), ``.microarch`` (units/muxes/storage report), ``.estimation``
(QoR model: span, LUT/FF/DSP/BRAM price, fmax), ``.csim()`` (LLVM-JIT golden),
``.cosim()`` (cocotb + Verilator, with ``stall_prob`` back-pressure), and
``.scaffold_project(dir)`` (split RTL + manifest for Vivado).

**Devices** (``allo/backend/rtl/devices/``) are FPGA only: ``u55c`` (default),
Alveo, Kria, Series-7, UltraScale+, Versal, plus Vivado IP operator rows
(float units are Xilinx IP, simulated by ``sim/ip_models.py``). There is no
ASIC/standard-cell device; one would have to be written as a ``Device`` spec.

**Small example** (``tests/rtl/test_basics.py::test_elementwise_and_addressing``)::

   @kernel
   def vand(A: i32[16], B: i32[16], out: i32[16]):
       for i in range(16):
           out[i] = A[i] & B[i]

   out = np.zeros(16, np.int32)
   vand.schedule().export("rtl").cosim(A16, B16, out)
   assert np.array_equal(out, A16 & B16)

Run result: section 2 ("Runs on the tip") -- cosim PASS, 17 cycles.

Size: ``allo/`` 87 Python files / ~28.8 kLoC; ``mlir/`` ~61.7 kLoC C++/TD
(``Scheduling`` 15.1 k, ``Microarch`` 11.6 k, ``Transforms`` 9.5 k,
``TransformOps`` 5.7 k). Tests: 527 ``test_*`` functions under ``tests/rtl``
(cosim needs Verilator), 36 lit files under ``mlir/test``.


2. Build dependencies, pins, and the zhang-21 build
---------------------------------------------------

Pins (``git submodule status`` at ``13b55a63``; dates from the GitHub API):

=====================  ============  ==========================================
dependency             pin           notes
=====================  ============  ==========================================
llvm-project           ``040a6419``  2026-06-18, LLVM main (post-22). Identical
                                     to CIRCT ``af5369d7``'s own llvm pin.
                                     Our main pins ``6b09f739`` (2025-12-21),
                                     so the two cannot share an LLVM build.
circt                  ``af5369d7``  2026-07-06. Built with
                                     ``-DOR_TOOLS_DISABLE=ON``, no Python
                                     bindings.
or-tools               ``bcd257ff``  2026-08-28, OR-Tools **main**, not a
                                     release. ``af465c47`` "Adapt CP-SAT
                                     scheduler to or-tools main" renamed a type
                                     and dropped 9.15 crash workarounds. CI
                                     (``.github/workflows/ci.yml``) still
                                     downloads the 9.15.6755 release tarball --
                                     inconsistent with the submodule.
marl                   ``b8406ab0``  fiber runtime for the CPU dataflow sim.
past-python-bindings   ``65f989b8``  same pin as our main.
spdlog                 ``v1.17.0``   FetchContent at configure (needs network).
=====================  ============  ==========================================

Python: ``requires-python >= 3.12``; runtime ``rich, sympy, numpy,
ml_dtypes``; build ``nanobind>=2.10, scikit-build-core>=0.10,
setuptools_scm>=8``; dev ``pytest, pytest-xdist, cocotb, sccache,
clang-format==22.1.8, black==24.8.0, pylint==3.0.2``. ``pyproject.toml``
hard-codes ``CMAKE_C(XX)_COMPILER=clang(++)`` and ``LLVM_USE_LINKER=lld``.
Kai's own docs assume a conda env ``allo-rtlgen`` and Docker for Vitis.

zhang-21 has no clang, lld, cmake, ninja or Verilator on ``PATH`` (system
g++ 8.5; gcc-toolset-13 has no lld). So a new conda env was created at
``scratch/rtlgen/env`` (conda-forge: python 3.12, clang/clang++ 20.1.8,
lld 20, cmake 4.4.3, ninja, ccache, verilator 5.052, nanobind, pyyaml,
numpy; 3.5 min). No existing env was touched.

Build attempt (logs: ``scratch/rtlgen/build_llvm.log``,
``build_rest.log``, ``build_allo_pip.log``):

**It builds and runs on zhang-21**, in about 22 minutes of build time on 60
jobs with a cold ccache (the submodule clone took longer than the build):

==============================  ==========================================  ========
step                            what                                        wall
==============================  ==========================================  ========
submodule clone                 ``git submodule update --init --depth 1``   ~25 min
                                llvm-project, circt, marl. On NFS; the
                                shallow clone fetched LLVM ``main`` first
                                and then the pinned SHA
LLVM/MLIR ``040a6419``          CI's flags (Release, ``mlir``, Native,      13.1 min
                                Python bindings on), conda clang 20 + lld,
                                ``ninja -j60``; 5507 steps, exit 0
CIRCT ``af5369d7``              CI's flags, ``-DOR_TOOLS_DISABLE=ON``;      5.8 min
                                1141 steps, exit 0
OR-Tools                        **not built from source.** Used the         <1 min
                                prebuilt ``v9.15.6755``
                                ``x86_64_AlmaLinux-8.10`` C++ tarball
                                (needs GLIBC_2.26; ``ldd`` resolves all
                                105 libraries on glibc 2.28)
Allo (``pip -e .[dev]``)        ``CMAKE_PREFIX_PATH`` = the OR-Tools        2.8 min
                                tarball; 273 steps; installed
                                ``allo-1.0.0``
==============================  ==========================================  ========

Caveats:

* The Allo configure step prints two non-fatal ``CMake Error: Error
  evaluating generator expression $<TARGET_OBJECTS:ortools_math_opt_core>``
  (and ``..._constraints_indicator``). They come from the prebuilt
  OR-Tools CMake config. The build still finished, and every test I ran
  passed.
* The tip's CP-SAT scheduler was adapted to OR-Tools **main**
  (``af465c47``). Linking it against **9.15** compiled, and the default
  scheduler ran. I did **not** exercise the CP-SAT (exact) scheduler path
  against 9.15. Kai's commit says 9.15 had presolve crashes there, so
  running that path may need the pinned ``bcd257ff`` built from source.
  That build was not attempted: ``build-ortools.sh`` with ``BUILD_DEPS=ON``
  also builds abseil and protobuf, an estimated 20-40 min.
* Disk used: LLVM build 3.6 GB, CIRCT build 1.2 GB, env 2.3 GB. ccache
  is at ``scratch/rtlgen/ccache``. ``TMPDIR`` and ``XDG_CACHE_HOME`` point
  under ``scratch/rtlgen``. Nothing under ``/home`` was written.

Runs on the tip (env ``scratch/rtlgen/env``, ``XILINX_VITIS=/nonexistent``):

* ``pytest tests/rtl/test_basics.py -n 8``: **2 passed** in 45.7 s.
* ``scratch/rtlgen/demo/vand_demo.py`` (the example above). Schedule
  latency 17, II=1. The Verilog is 70 lines, headed ``// Generated by CIRCT
  af5369d``, with ``module vand(input clk, rst, start, input [31:0]
  A_rd0_data, B_rd0_data, output done, output [31:0] A_rd0_addr,
  B_rd0_addr, out_wr0_addr, out_wr0_data, output out_wr0_we)``. Arrays
  are external memory ports and control is ``start``/``done``. Cosim
  (Verilator 5.052 + cocotb 2.1.0) gave ``cycles=17``, and the result
  matched ``A & B``. The QoR model on u55c @ 300 MHz estimates 46 LUT,
  14 FF and fmax 493 MHz. Output is in ``demo/vand.sv``.


3. Distance between ``allov2``/``allo-rtlgen`` and our core
-----------------------------------------------------------

**Merge-base.** ``git merge-base origin/allo-rtlgen upstream/main`` =
``76130c63`` (2026-03-13, upstream #555 "Add Allo operations for SPMW").
Same merge-base with our ``main`` and with Kai's own ``main``. From there:

* ``allo-rtlgen`` is 519 commits ahead (all Kai's: 217+211+70+21 under four
  identities); upstream ``main`` is 15 ahead; our ``main`` is 964 ahead.
* ``allo-rtlgen`` contains ``allov2`` entirely (``allov2`` tip ``b22ed847`` is
  an ancestor; ``allo-rtlgen`` is +300 on top of it).
* ``git diff --shortstat 76130c63 origin/allo-rtlgen``: 907 files, +146,100
  / -99,819. ``mlir/`` alone vs our main: 308 files, +64,849 / -31,415.

**Directory overlap.** Of Kai's 87 ``allo/**/*.py`` and our 178, only five
paths coincide (``allo/__init__.py``, ``backend/__init__.py``,
``backend/utils.py``, ``library/__init__.py``, ``logging.py``), and their
contents differ. Kai has ``compiler/ lang/ operators/ schedule/
backend/{cpu,vitis,rtl}``; we have ``customize.py dataflow.py ir/ passes.py
_mlir/ autoscheduler/ act/ compose.py netlist.py backend/{hls,simulator,
systemc,catapult,aie,tapa,xls,asic,...}``. MLIR: Kai ``mlir/lib/allo/{IR,
Transforms,TransformOps,Conversion,Scheduling,Microarch,Translation,Support}``;
ours ``mlir/lib/{Dialect,Transforms,TransformOps,Conversion,Translation,
CAPI,Bindings,Support}``. Only the Vivado HLS emitter exists on both sides,
and it is a different file.

**Dialects are disjoint in practice.** Op mnemonics (from ``*.td``):

* Kai (48 ops): ``kernel invoke return stream.create stream.get stream.put
  get_wid get_nw assume.nodep assume.ssa volatile muladd bit.get_slice
  bit.set_slice`` + ``dcp.*`` (``module unit pipeline sequential chain comb
  compute load store mux select storage resource operator device instance
  output condition uncondition stream_timing``).
* Ours (88 ops): customization ops (``split reorder tile pipeline unroll
  partition reuse_at buffer_at compute_at fuse outline reform ...``), fixed
  point, struct, ``stream_*``/``channel_*``/``wire_*`` with
  ``try_get/try_put/empty/full``, ``*_global`` streams, ``print``.

**Key API differences.**

.. list-table::
   :header-rows: 1

   * - concern
     - ours (upstream lineage)
     - Kai (``allov2`` / ``allo-rtlgen``)
   * - kernel
     - plain ``def`` + ``allo.customize``
     - ``@kernel`` (annotated), nested ``@kernel``, ``consteval``,
       ``Template``, ``constexpr``
   * - schedule
     - ``s = allo.customize(f)``; ``s.split/reorder/pipeline/...``; the
       customizations are ops in the allo dialect
     - ``f.schedule()`` builds a transform-dialect script, using
       ``LoopRef`` / ``BufferRef`` handles; ``.apply()``
   * - build
     - ``s.build(target="vhls"|"llvm"|...)``, ``df.build(region, target=...)``
     - ``.export("cpu"|"vitis"|"rtl")``
   * - dataflow
     - ``@df.region()``, ``@df.kernel(mapping=[...])``, ``@df.unit``,
       ``df.pipe``, ``get_pid``; OpenMP simulator
     - ``async def`` + ``await`` spawn, ``Stream[T, D]`` locals,
       ``get_wid`` / ``get_nw`` grids; marl-fiber CPU simulator
   * - non-blocking stream ops
     - ``try_get`` / ``try_put`` / ``empty`` / ``full``
     - none in the frontend (only cosim ``stall_prob``)

``allov2``'s own ``AGENTS.md``: "This is not the upstream version of Allo. DO
NOT assume the project structure, APIs, or compiler behavior is the same as
the upstream Allo project." Our ``dev/fork_maintenance.rst`` already records
that a trial merge of ``chia-codesign`` gave 26 conflicts, 24 of them
"deleted in chia-codesign and modified in main".


4. Rough cost of porting onto our ``main``
------------------------------------------

Estimate, not measured. The RTL backend is not a pass that can be bolted on
behind our frontend; it is the bottom of a stack whose frontend, dialect and
LLVM pin all differ.

Would carry over with little change (in principle; target-independent C++
over upstream dialects):

* Scheduling core over ``affine/arith/memref/scf``: SDC, CP-SAT,
  dependence/memory/latency models, ``Device`` operator library
  (~15 k LoC) -- but 52 references to Kai's ``allo`` ops and 40 to ``dcp`` ops.
* Microarch emission to CIRCT ``hw/comb/seq`` (~11.6 k LoC), 79 ``dcp``
  references, 11 ``allo`` op references.
* The Python ``allo/backend/rtl`` package (device specs, QoR, reports,
  cocotb harness) -- depends on Kai's ``Kernel``/``ShapedType`` and his
  CAPI bindings (``emit_datapath_to_hw``, ``run_sdc_scheduling``).

Would need rewriting or a bridge:

* **LLVM bump** from ``6b09f739`` (Dec 2025) to ``040a6419`` (Jun 2026) for
  our whole MLIR tree (~half a year of upstream API churn), or pinning CIRCT
  to an older commit that matches our LLVM (and back-porting Kai's CIRCT use).
* **A lowering from our IR** into the form RTLGen's prepare pipeline expects:
  our streams (``allo.stream_*``, ``channel_*``, ``wire_*``, try-ops) and
  ``df.region``/``df.kernel`` structure into his ``kernel/invoke/stream.*``
  + spawn semantics; our ``customize`` primitives into whatever directive
  attributes his ``reconcile-array-directives``/``loop-canonicalization`` read.
  Non-blocking ops have no counterpart in RTLGen's front end at all.
* ~9.5 k LoC of his ``Transforms`` (93 references to his allo ops) and his
  ``TransformOps`` scheduling language (5.7 k, 50 refs) assume his dialect.
* New dependencies in our build: CIRCT, OR-Tools (needs main or a 9.15
  compatibility check), marl, spdlog FetchContent, clang+lld, Python >= 3.12
  (our ``allo`` env's Python version should be checked before assuming).
* An ASIC ``Device`` (operator delays/areas for a standard-cell library), since
  only FPGA devices ship.

Order of magnitude: the scheduler+emitter is ~27 k LoC of C++ that would have
to be re-homed onto a different dialect and LLVM; realistically a multi-week
to multi-month port, not a merge. The cheaper alternative the evidence points
to is consuming RTLGen out of tree (its own env, as built here) at an IR or
Python boundary -- recorded as an observation only; planning is out of scope.


5. The 2026-09-09 assessment (``35c38494``) vs the current tip
---------------------------------------------------------------

What ``examples/accelerator/tinytpu_grid/BACKEND_CHOICE.md`` at ``35c38494``
found about "chia RTLGen":

* Fan-out (one array read/written by N processes) is **allowed** (Vitis
  rejects it: HLS 200-779 / 200-979).
* **Unit occupancy: "a ``func.call``: fills and drains per instruction"**
  versus persistent processes in Vitis dataflow and SystemC.
* Non-blocking: only ``stall_prob`` cosim.
* Link primitives: ``Stream`` only.
* Cycle counts available (cocotb + Verilator).
* Quantified via ``FINDINGS_v2.md`` G.4/H (at ``5a6fee3d``): K back-to-back
  ``mm`` cost ``43.0*K + 4`` cycles (zero overlap); 712 of the executor's
  1448 cycles at 16x16x16 (37 %) are per-instruction fill, drain and
  dispatch. G.4's own breakdown splits that into 288 cycles of ``mm``
  fill/drain and 396 of dispatch. The compiler's
  own message: ``WARN: [PREP] Conditional left as an opaque scheduling unit
  because 'func.call' cannot be predicated; the enclosing loop cannot pipeline
  across it`` and ``INFO: [SCHED] Detected imperfect nest, decomposing into
  sub-regions scheduled in program order.``
* Projection (not measured): persistent units would take 16x16x16 from 1734
  to ~800 cycles (0.70x Gemmini).

Status at tip ``13b55a63``. The source evidence is below, plus one
measured probe. The TinyTPU design itself was **not** rebuilt on the tip:

* The two mechanisms behind the per-instruction cost are **unchanged**.
  ``mlir/lib/allo/Transforms/FoldIfStatements.cpp:416-427`` still emits the
  "opaque scheduling unit because '<op>' cannot be predicated" warning, and
  ``mlir/lib/allo/Scheduling/Scheduler.cpp:1505-1527`` still decomposes an
  imperfect nest (a "Container") into children scheduled in program order:
  "Fusing the level over its inner loops into one modulo problem is not
  implemented: the container sequences its children and runs no schedule of
  its own." Neither file changed in substance in the 23 commits since
  ``9e2d5716`` (only comment edits in ``4b7d90eb``, ``7c3bc50f``,
  ``3cf6b253``).
* ``loop-canonicalization{perfectize=...}`` can sink prologue/epilogue of an
  imperfect nest with a single counted inner loop
  (``LoopCanonicalization.cpp:207-260``); it bails on sibling inner loops and
  on ``while`` -- an instruction-dispatch loop with several unit bodies is
  exactly the sibling-loop case. (Inference from code.)
* Persistent-ish processes **do exist** in the frontend and did already at
  ``9e2d5716``: ``async def`` processes spawned with ``await`` start together
  off the container's ``start`` (BROADCAST policy), run concurrently and talk
  over FIFOs (``tests/rtl/test_dataflow.py::test_three_start_policies_from_one_table``).
  So "no persistent units" was a property of how the TinyTPU executor was
  written (each instruction a ``func.call`` in a dispatch loop) combined with
  the two scheduler limits above, not an absence of concurrent processes.
  Whether a TinyTPU restructured as ``async`` processes fed by an instruction
  ``Stream`` would overlap consecutive instructions is **unverified**; inside
  each process the same imperfect-nest rule would still serialize per-
  instruction inner loops unless they are perfectized or unrolled.

**Measured on the tip (a small probe, not TinyTPU).**
``scratch/rtlgen/demo/dispatch_probe.py`` issues K "instructions". Each one
is a nested ``@kernel`` call that runs a 16-row II=1 loop, dispatched under
``if prog[k] == 0`` inside ``for k in range(K)``. The comparison is the same
work as a plain ``k``/``r`` loop nest with no call. Cosim cycles:

=====  ============================  ===========================
K      dispatch (call under ``if``)  flat nest (no call)
=====  ============================  ===========================
1      22                            18
2      44                            34
4      87                            66
8      173                           130
=====  ============================  ===========================

The dispatch version costs about ``21.6*K``: 16 cycles of work plus about
5.6 cycles of fill, drain and dispatch for **every** instruction, with no
overlap between instructions. The flat nest costs ``16*K + 2``, so its fill
is paid once. The tip still emits ``WARN: [PREP] Conditional left as an
opaque scheduling unit because 'func.call' cannot be predicated; the
enclosing loop cannot pipeline across it``. **So the per-instruction
fill/drain property the 2026-09-09 assessment measured still holds at
``13b55a63``.** This probe did not check output values.

* Other changes since ``9e2d5716``: cross-region functional-unit sharing
  (``15107b45``..``c76eca1b``), shared address-stride registers
  (``fcd4f78c``), shared read/write port factoring, OR-Tools as a submodule
  pinned to main (``65eea1b9``, ``af465c47``), lit tests. None addresses
  fill/drain across calls.
* Still no non-blocking frontend ops (``try_get``/``empty``/``full`` do not
  exist in Kai's dialect); still only ``Stream`` links (no ``Wire``/
  ``Channel``).

Our own README (``D-1``, 2026-10-01) already supersedes the 2026-09-09
recommendation and names RTLGen + AMC as the long-term open target; the
fill/drain property it cited still holds on the tip by the code evidence above.

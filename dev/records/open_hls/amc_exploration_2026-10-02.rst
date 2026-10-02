AMC (Accelerator Memory Compiler) exploration on zhang-21
=========================================================

:Date: 2026-10-01 22:40 to 23:46 EDT (the file name uses the requested 2026-10-02 stamp)
:Host: zhang-21.ece.cornell.edu (RHEL 8.10, glibc 2.28, 64 cores, 376 GB RAM)
:Repo: github.com/cornell-zhang/amc-dialect (private). Default branch ``main``, last pushed 2026-08-25.
:Commits inspected:
  - amc-dialect ``fe60c1219df3c838947794dcf69d18f3e59c0334`` (2026-08-24, "circt submodule tracks amc-hls")
  - circt submodule ``a470836d08c630cea25beb5de18f587c9fd37c97`` (andrewb1999/circt, branch ``amc-hls``. The pin is that branch's tip: GitHub compare reports "identical")
  - llvm submodule (inside circt) ``6279700538792da0c5a08e17babfe9b6e824c69f`` (llvm-project, committed 2026-08-21)
  - our allo checkout ``9fd022759198d54f0f4b3842c0da2238d286c797`` (read only)
:Scratch: ``/work/shared/users/phd/sk3463/scratch/amc/`` holds the clone, the build dirs, a new env ``env/``, the logs and the ``kernels/`` scripts.

Summary
-------

* The full stack (LLVM/MLIR, then CIRCT fork, then AMC with Python bindings) **builds on
  zhang-21** in about 37 minutes of wall time at ``-j32`` with ``gcc-toolset-13``. ``ninja check-amc``:
  226 tests, 181 passed, 45 unsupported, 0 failed. A subset of the AMC Allo pytest suite passed:
  44 passed and 1 skipped.
* AMC ships **its own vendored, forked Allo** (``amc-dialect/allo/``). That fork imports ``amc_mlir`` and
  hosts the ``allo`` dialect inside AMC. It is not upstream Allo and not our fork.
* AMC emits **SystemVerilog** through CIRCT's FSM/HW/Seq/SV ExportVerilog. The output is one module for
  the kernel, one ``*_fsm`` controller, and per-argument BRAM adapter modules. The output also includes
  hand-written primitives (``ram_*.sv``, ``int_arith_pipe.v``).
* **int32 kernels from our repo go through.** Our Allo's ``str(s.module)`` text is parsed by AMC's
  ``AMCModule``, built to SystemVerilog, and simulated in Verilator 5.052 with a bit-exact numpy match
  (vadd, mac). **BF16 does not go through:** no operator library has a bf16 descriptor.
  ``f32`` builds only on the ``amc-designware`` target, and that target cannot be simulated because
  ``float_dw.v`` is not in the repo.

1. Build and pins
-----------------

Submodules (``.gitmodules``)::

   [submodule "circt"]  url = git@github.com:andrewb1999/circt.git  branch = amc-hls
   circt/.gitmodules:   llvm -> https://github.com/llvm/llvm-project.git (shallow = true)

``git submodule status`` after the clone::

    a470836d08c630cea25beb5de18f587c9fd37c97 circt (pycde-0.0.3-9524-ga470836d0)
    6279700538792da0c5a08e17babfe9b6e824c69f llvm (627970053)        # inside circt

Other dependencies:

* **OR-Tools** is optional. ``find_package(ortools CONFIG)``, ``OR_TOOLS_DISABLE`` and ILP partitioning
  live in ``ExploreMemPartitioning.cpp``. It was not found, and the build proceeded without it.
* **Python bindings** need ``-DMLIR_ENABLE_BINDINGS_PYTHON=ON`` on LLVM and
  ``-DAMCHLS_PYTHON_BINDINGS_ENABLED=ON`` on AMC. Note the ``AMCHLS_`` prefix.
  ``utils/buildHelper.sh`` misspells it as ``AMC_PYTHON_BINDINGS_ENABLED``.
  The MLIR Python pin is ``circt/llvm/mlir/python/requirements.txt``: ``nanobind>=2.9,<3.0``,
  ``PyYAML<=6.0.1``, ``numpy>=2.1.0,<=2.1.2``, ``ml_dtypes>=0.5,<=0.6``. The env got
  nanobind 2.15.0 and numpy 2.5.3. Only the numpy version is outside the pin, and nothing broke at runtime.
* The vendored Allo needs numpy, tabulate, xmltodict, psutil, pandas and matplotlib.
* Simulation needs ``verilator`` on ``PATH``. The flags are plain ``--cc --exe --build``, with no ``--timing``.
  Verilator 5.052 at ``/work/shared/users/phd/sk3463/tools/verilator/bin`` worked.
* CI (``.github/workflows/buildAndTest.yml``) builds in the container ``ghcr.io/circt/images/circt-ci-build:20240213211952``
  with clang, Release, assertions OFF and lld. CI runs on a self-hosted runner.

**Allo's LLVM cannot be reused.** Our pin is ``6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb``, committed
2025-12-21, from our ``externals/llvm-project``. AMC's pin ``62797005`` is **30,287 commits ahead** of it
(``gh api repos/llvm/llvm-project/compare/6b09f739...62797005``: ahead=30287, behind=0). In any case,
``/home/sk3463/llvm-allo-6b09f739`` did **not exist on zhang-21** when it was first checked at
about 22:41 (``ls`` gave "No such file or directory"). There is no prebuilt CIRCT or MLIR on the host:
``firtool``, ``circt-opt``, ``mlir-opt`` and ``cmake`` are not on ``PATH``.

What was run (``build_all.sh``, log ``build.log``):

* A new conda env, ``scratch/amc/env``, from conda-forge: python 3.12, cmake 4.4.3, ninja, lld,
  nanobind (pinned to 2.15.0 via pip), numpy, pybind11 and pyyaml. No existing env was modified.
* Compiler: ``scl enable gcc-toolset-13`` (g++ 13.3.1), linked with ``-DLLVM_USE_LINKER=lld``, using
  ``nice -n 10 ninja -j 32``.
* LLVM: ``-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release
  -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON``. This deviates from the README, which
  uses DEBUG and X86;RISCV. Release was chosen to bound time and disk.
* CIRCT and AMC: Release, ``-DAMCHLS_PYTHON_BINDINGS_ENABLED=ON`` on AMC.

Timings (from ``build.log``), on a host shared with other users (load average 70-115 during the build):

======================  ===================  ==========  ========
stage                   start                duration    size
======================  ===================  ==========  ========
LLVM/MLIR (5900 steps)  22:58:48             ~29.5 min   5.1 GB
CIRCT (1266 steps)      23:28:19             ~5 min      1.7 GB
AMC                     23:33:19             ~2 min      666 MB
======================  ===================  ==========  ========

Result: ``build/bin/{amctool,amc-opt,amc-axi-gen}`` and
``build/tools/amc/python_packages/amc_core/amc_mlir``. Results of ``ninja check-amc``
(``check_amc.log``)::

    Total Discovered Tests: 226
      Unsupported:  45 (19.91%)
      Passed     : 181 (80.09%)

Allo pytest subset, ``pytest -m amc tests/test_amc_{kernels,asic,schedule,partition}.py``
(``pytest_subset.log``)::

    44 passed, 1 skipped, 1 warning in 435.48s (0:07:15)

Not run: ``test_amc_axi.py``, ``test_amc_stream.py``, ``test_amc_dyn_*.py``, ``test_amc_backend.py``.

2. What Allo input it accepts
-----------------------------

* AMC does **not** consume upstream Allo or our fork as a library. It carries a **vendored Allo fork**
  at ``amc-dialect/allo/``. It was moved into the repo in ``e697d88`` (2026-04-10, "Move allo fork into
  amc-hls"), and the tree has 127 commits since. All of its MLIR imports are
  ``from amc_mlir... import ...``. The Allo dialect itself (``include/amchls/Dialect/Allo/*.td``,
  ``let name = "allo"``, ``cppNamespace = ::circt::allo``) is compiled into AMC.
* Era (inferred, **unverified**): the fork aliases the dialect as ``hcl_d`` and exposes the pre-``dataflow``
  surface (``allo.customize``, ``allo.grid``, ``allo.reduction``). In our history, upstream used
  ``hcl_mlir`` / ``hcl as hcl_d`` until ``cec32446`` (2024-09-27, "Update to LLVM 19.1 and merge MLIR
  dialect"). So the fork appears to descend from upstream Allo of roughly mid/late 2024, renamed to
  ``allo``. It has **no** ``allo.dataflow`` / ``@df.region`` / ``@df.kernel``, no ``Stream`` / ``Stateful``
  types and no ``bfloat16`` type.
* The front-end API in its tests (``allo/tests/test_amc_*.py``) is
  ``s = allo.customize(fn); f = s.build(target=...)``. The targets are ``"amc"`` (Vivado operator
  library), ``"amc-designware"`` and ``"loopschedule"``, plus ``vhls``/``vitis_hls`` and ``llvm``
  (``allo/allo/customize.py:1036``). ``f(np_args...)`` runs a Verilator simulation. ``f.dump_verilog(dir)``,
  ``f.dump_schedule(file)`` and ``f.get_resource_estimates()`` produce artefacts.
  Schedule primitives used include ``partition``, ``pipeline``, ``unroll``, ``parallel``, ``split``,
  ``reorder``, ``interface``, ``control_interface("axil_handshake")`` and ``allo.label`` (new in the fork).
* Types: ``Int``/``UInt`` of any width, ``Fixed``/``UFixed``, ``float32`` and ``Struct``. ``bool`` is ``Int(1)``.
* Allo-dialect op overlap with our fork (from the ``.td`` files): AMC has every non-dataflow op ours has, plus
  ``BurstCopyOp`` and ``BitcastOp``. It lacks our ``Stream*``, ``GlobalStream*``, ``Wire*``, ``Channel*``,
  ``TransformLayoutOp`` and ``GridMapOp``.
* ``AMCModule.__init__`` (``allo/allo/backend/amc.py:313``) takes **MLIR text** (``Module.parse(str(mod))``)
  and the top function name. It reads the ``itypes``/``otypes`` attributes to type the testbench. That makes a
  textual hand-off from another Allo possible, as section 4 shows.
* ``amctool <file.mlir>`` takes func/affine/scf/memref/arith MLIR directly.

3. What it emits
----------------

The output is SystemVerilog via CIRCT ExportVerilog. The lowering chain runs
allo, then core dialects, then ``amc`` memory allocation (banking/ports), then ``loopschedule``,
then ``fsm`` plus ``hw``/``seq``, then SV. ``dump_verilog`` writes per-module files:
``<kernel>.sv``, ``<kernel>_fsm.sv``, ``memN_bram.sv`` and ``fsm_enum_typedefs.sv``, plus the hand-written
library ``hdl/systemverilog/{ram_*,rom_*,int_arith_pipe.v,ieeefpmult_l5.sv}``.
The default kernel interface is ``clk/rst/start/ready/done`` with BRAM-style ``addr/en/we/din/dout``
per memref argument. With ``s.interface`` / ``control_interface`` it can instead use m_axi plus s_axilite
control.

Example: ``amctool --target-device=xcv80 --emit-verilog vadd.mlir``. The input is our Allo's vadd; the output
is ``kernels/amctool_vadd.sv``, 452 lines, with modules ``vadd_arg{0,1,2}_bram``, ``vadd`` and ``vadd_fsm``.
Excerpt::

    // Generated by CIRCT a470836d0
    typedef enum bit [2:0] {vadd_fsm_state_t_IDLE, vadd_fsm_state_t_FRAME_0,
        vadd_fsm_state_t_loop0_FRAME_0_0, vadd_fsm_state_t_loop0_FRAME_0_1,
        vadd_fsm_state_t_DONE} vadd_fsm_state_t;
    module vadd(
      input         clk, rst, start,
      input  [31:0] vadd_arg0_bram0_dout, vadd_arg1_bram0_dout, vadd_arg2_bram0_dout,
      output        ready, done,
      output [3:0]  vadd_arg0_bram0_addr,
      output        vadd_arg0_bram0_en, vadd_arg0_bram0_we,
      output [31:0] vadd_arg0_bram0_din,
      ...

(``amctool -o`` was ignored with ``--emit-verilog``: the output went to stdout.)

4. Kernels from our repo through AMC
------------------------------------

Method: there are two processes because the MLIR bindings differ (``allo._mlir`` at LLVM ``6b09f739`` and
``amc_mlir`` at ``62797005``).

1. ``kernels/emit_from_our_allo.py`` runs in our ``allo`` conda env with
   ``PYTHONPATH=/work/shared/users/phd/sk3463/allo``, because ``allo`` is not pip-installed in that env on
   this host. It writes ``str(allo.customize(fn).module)`` to ``kernels/{vadd,mac,bf16_add,f32_add}.mlir``
   (N=16).
2. ``kernels/run_amc.py`` runs in ``scratch/amc/env``. It feeds each text to AMC's
   ``AMCModule(txt, top_func_name=..., use_designware=...)``, then calls ``dump_verilog`` and runs the
   Verilator simulation against numpy. ``native_vadd`` is the same kernel written in AMC's own vendored Allo,
   as a control.

Results (``kernels/run_amc.log``)::

    native_vadd  cycle time: 33   sim match: True
    vadd         cycle time: 33   sim match: True     (our Allo's MLIR, target amc)
    mac          cycle time: 98   sim match: True -2613 -2613   (our Allo's MLIR; i64 product, i65 add, trunc to i32)
    f32_add      built on amc-designware -> SV instantiates `fadd_32_dw_l1 fp32add_hw_inst_0`; not simulated
    bf16_add     FAILED: RuntimeError: Unsupported type  (Python frontend dtype decode, allo/utils.py:200)

The textual MLIR from our Allo (LLVM 22 era) parsed without change in AMC's newer MLIR for these kernels.
**Verified only for these plain func/affine/arith/memref kernels.** Anything that uses our fork-only ops
(streams, ``@df.region``, wires/channels, ``GridMapOp``) will not parse, because AMC's dialect lacks them.

BF16 blockers. Two layers fail, both verified:

* **Frontend:** ``get_func_inputs_outputs``/``get_dtype_and_shape_from_type`` rejects ``bf16`` memrefs.
* **Compiler:** after bypassing only the frontend decoder (``kernels/run_bf16_dw.py``, which
  monkeypatches the testbench typing), the DesignWare target fails with::

    error: unsupported operation "arith.addf"(...) : (bf16, bf16) -> bf16
    Failed to run scheduling pipeline   (lower_amc_to_loopschedule_designware failed)

  ``amctool`` reports the same error for both f32 and bf16 on the default (Vivado) operator library.
  The operator libraries (``oplib/backends/*.json``) list fp descriptors only for ``f32``.
  ``designware.json`` has ``addf``/``subf``/``mulf``/``divf``/``cmpf``/``absf``/``negf`` on f32.
  ``vitis.json`` and ``flopoco.json`` have only ``mulf:f32``. None has a bf16 entry.
  Even on f32, the DesignWare modules (``float_dw.v``) are **not in the repo**, so fp kernels build but
  cannot be simulated. AMC's own ``CLAUDE.md`` says the same.
* Unverified option: a bf16 adder written as integer bit manipulation over ``uint16`` (no ``arith.addf``)
  should fall on the supported integer path. This was not tried.

Reproduce
---------

Use these exports for the run steps. The build step only needs ``env/bin`` on ``PATH``, and the script adds it::

   export PATH=/work/shared/users/phd/sk3463/scratch/amc/env/bin:/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH
   export PYTHONPATH=/work/shared/users/phd/sk3463/scratch/amc/amc-dialect/allo:/work/shared/users/phd/sk3463/scratch/amc/amc-dialect/build/tools/amc/python_packages/amc_core
   export TMPDIR=/work/shared/users/phd/sk3463/scratch/amc/tmp     # AMCModule uses tempfile.mkdtemp

Then run, in order:

* ``cd scratch/amc && scl enable gcc-toolset-13 -- ./build_all.sh`` builds everything (about 37 min).
* ``cd scratch/amc/kernels && scl enable gcc-toolset-13 -- python run_amc.py`` runs the kernels. Generate
  the ``.mlir`` inputs first with ``emit_from_our_allo.py``, in the ``allo`` env.

Notes
-----

* Nothing was written to ``/work/shared/users/phd/sk3463/allo``. It was only read, plus a read-only
  ``customize()`` import of it.
* Incidental: ``pip install`` into the scratch env may have used ``~/.cache/pip`` for the first installs.
  The later installs used ``PIP_NO_CACHE_DIR=1``.

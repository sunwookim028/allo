..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

######################################
Development Environment and Toolchains
######################################

The environment every result in the fork's documentation was produced with:
the conda environment, the LLVM/MLIR builds, the EDA tools on the development
host, and the golden simulator tests. None of the tools below is installed by
the repository, and the ``allo`` conda environment sets none of them.

Environment
-----------

``LLVM_BUILD_DIR`` is **not** set by the conda environment -- neither
``conda activate allo`` nor ``conda run`` sets it (verified 2026-09-17) -- and
the simulator asserts ``LLVM_BUILD_DIR is not set`` without it. Export it
explicitly (see ``docs/source/developer/pitfalls.rst``).


.. code-block:: bash

   conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # the env does NOT set this

   # OMP_NUM_THREADS no longer has to exceed the kernel-instance count: the
   # simulator now sets the OpenMP team size to the section count itself.
   # Before that fix a PE blocked on a stream spun in its section, so a team
   # smaller than the section count never started the sections that would
   # unblock it and the region hung SILENTLY. See docs/source/developer/limitations.rst, item 11.
   export OMP_NUM_THREADS=8

.. warning::

   **The LLVM git history on this machine was deleted on 2026-09-24**, with the
   owner's approval, to reclaim 12 GB on a ``/home`` that was 99 % full.

   ``/home/sk3463/allo/externals/llvm-project`` and
   ``/home/sk3463/llvm-allo-6b09f739`` are now **plain directories, not git
   repositories**. The second was a git *worktree* of the first, so its
   ``.git`` pointer was removed before the history, and nothing dangles.

   What still works: the sources, and the build at ``LLVM_BUILD_DIR``
   (``mlir-opt --version`` runs). What no longer works: any ``git`` command in
   either tree, and any rebuild step that stamps a revision from git.

   To restore history, re-clone ``llvm-project`` and check out
   ``6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb`` -- the pin recorded in
   ``scripts/act-test-recipe.sh``, and the commit the build directory's name
   encodes. Prefer ``--filter=blob:none``; the full history is what filled the
   disk.

Which bindings a worktree loads
-------------------------------

``allo/_mlir`` is a **tracked symlink** to ``../mlir/build/tools/allo/_mlir``,
so it resolves only inside a checkout whose own ``mlir/`` has been built. In a
fresh worktree it dangles -- and importing ``allo`` still works, because the
editable install maps the name ``allo`` to one checkout and its finder answers
the submodule from there. Measured 2026-09-22 in an unbuilt worktree:

.. code-block:: text

   allo        <this worktree>/allo/__init__.py
   allo._mlir  /home/sk3463/allo-bench/mlir/include/allo/Bindings/allo/__init__.py
   extension   /home/sk3463/allo-bench/mlir/build/tools/allo/_mlir/_mlir_libs/_allo.cpython-312-x86_64-linux-gnu.so

So the python half comes from the worktree under test and the **compiled half
from a different checkout**, silently: nothing looks stale, because cmake
preserves mtimes, and the symptom surfaces later as an unexplained ABI or
behaviour mismatch. A local symlink takes precedence when it resolves, so the
fix is to make it resolve.

**Setting up a fresh worktree.** Either build that worktree's own bindings --

.. code-block:: bash

   ninja -C mlir/build -j"$(nproc)"        # or: examples/tinytpu/reproduce.sh

-- which is what ``reproduce.sh`` does before it runs anything, and it then
checks that ``allo`` resolves inside the checkout and exits nonzero if it does
not. Note that check is on ``allo`` itself, not on the extension, which is the
half that leaks; closing that is what the test below is for. Or, for a
read-only session that only needs to *run* something and accepts
another checkout's build, point the symlink at a build **of the same commit**
and say so in whatever you report:

.. code-block:: bash

   ln -sfn /path/to/other/mlir/build/tools/allo/_mlir allo/_mlir   # borrowed, not built

``tests/act/test_bindings.py`` asserts this rather than leaving it to be
noticed: it fails naming both paths when the extension comes from another
checkout, and skips when no bindings are reachable at all. The ``allo/act/``
core itself imports numpy only, but it moved under ``allo/`` on 2026-09-24, so
``import allo.act`` runs ``allo/__init__.py``: ``pytest tests/act`` now needs
this checkout's bindings like everything else, where ``act/`` at the root used
to run in a worktree with no build.

Golden test for dataflow simulator
----------------------------------

.. code-block:: bash

   # `conda run` does not source the activate scripts, so export the env first.
   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   python tests/dataflow/test_df_unit.py
   python tests/dataflow/test_region_stateful.py

Toolchains on the development host
----------------------------------

Verified 2026-09-18 on the fork's development host (``ace-01``). None of this is installed by the repo, and
the ``allo`` conda env sets none of it; a migration (e.g. to zhang-21) has to
reproduce or re-point every row. The conda env, ``LLVM_BUILD_DIR`` and
``OMP_NUM_THREADS`` are covered above and are not repeated here.

+------------------------------------------------+---------------------------------------------+-------------------------------------------------+
| Tool                                           | Location                                    | On ``PATH`` by default?                         |
+================================================+=============================================+=================================================+
| Vivado 2023.2 — ``vivado``, and the ``xsim``   | ``/opt/xilinx/Vivado/2023.2/settings64.sh`` | **Yes** — ``which xvlog`` already resolves to   |
| trio ``xvlog``/``xelab``/``xsim``              |                                             | ``/opt/xilinx/Vivado/2023.2/bin/xvlog``, so the |
|                                                |                                             | login profile sources ``settings64.sh``. Do not |
|                                                |                                             | assume that on the new host.                    |
+------------------------------------------------+---------------------------------------------+-------------------------------------------------+
| Vitis HLS 2023.2                               | ``/opt/xilinx/Vitis_HLS/2023.2``            | **No.** ``which vitis_hls`` finds nothing;      |
|                                                | (``settings64.sh`` present)                 | scripts source ``settings64.sh`` themselves —   |
|                                                |                                             | ``examples/tinytpu/cosim.py``                   |
|                                                |                                             | hardcodes the path in its ``VITIS`` constant.   |
+------------------------------------------------+---------------------------------------------+-------------------------------------------------+
| Verilator 5.051                                | ``VERILATOR_ROOT`` tree at                  | Yes. Leave ``VERILATOR_ROOT`` **unset** — the   |
| (``devel rev vUNKNOWN-built20260904-2286359``) | ``~/.local/share/verilator``; driver at     | driver derives it and warns if an inconsistent  |
|                                                | ``~/.local/bin/verilator``                  | one is exported.                                |
+------------------------------------------------+---------------------------------------------+-------------------------------------------------+
| Chipyard                                       | ``~/chipyard/env.sh``                       | No; ``source`` it. It ``conda activate``\ s     |
|                                                |                                             | ``~/chipyard/.conda-env``, so it                |
|                                                |                                             | **replaces** the ``allo`` env — source it in a  |
|                                                |                                             | separate shell.                                 |
+------------------------------------------------+---------------------------------------------+-------------------------------------------------+
| Cadence Xcelium                                | —                                           | **Not installed here.** ``/opt/cadence`` does   |
|                                                |                                             | not exist and no ``xrun`` is on ``PATH``;       |
|                                                |                                             | ``/opt`` holds only ``xilinx`` among EDA        |
|                                                |                                             | vendors. Earlier notes describing               |
|                                                |                                             | ``/opt/cadence/XCELIUM2403`` and an             |
|                                                |                                             | ``unset LD_PRELOAD`` workaround do not apply to |
|                                                |                                             | this host. (:ref:`limitation-22`                |
|                                                |                                             | cites Xcelium cosim results from elsewhere.)    |
+------------------------------------------------+---------------------------------------------+-------------------------------------------------+

Toolchains on zhang-21
~~~~~~~~~~~~~~~~~~~~~~

Surveyed read-only by the ``zhang21`` agent session on 2026-10-02, for hosting the
MiniTPU unit ladder (README, D-7). Catapult, Xcelium and the licence variables
are in ``dev/records/catapult_handoff/zhang21_inventory_2026-10-01.md``.

- **Host:** RHEL 8.10, glibc 2.28, 64 cores. The system g++ is 8.5, too old for
  Verilator 5 ``--timing``; ``gcc-toolset-13`` (g++ 13.3.1,
  ``/opt/rh/gcc-toolset-13``) is available.
- **allo checkout:** ``/work/shared/users/phd/sk3463/allo``. Its bindings are
  current, built by ``reproduce.sh`` at ``c3de83f3``. Run scripts from the
  physical path (``cd -P``), not through the ``/home/sk3463/work/allo``
  symlink.
- **Disk:** ``/work/shared/users`` is NFS with 9.4 TB free; put clones and
  projects there. ``/scratch`` is local, 195 GB free. ``/home`` is full, so
  avoid ``$HOME``.
- **Verilator:** 5.052 installed 2026-10-02 (below).
- **Vivado:** 2019.2, 2022.1, 2023.2 and 2024.2 under ``/opt/xilinx/Vivado``, and
  2026.1 under ``/opt/xilinx/2026.1``. All have the ``xczu7ev`` part. The 2026.1
  module sets ``XILINXD_LICENSE_FILE``; a licence checkout has not been tested.
- **Python (allo env):** 3.12.12, numpy 2.4.0, torch 2.10.0+cu128. The pinned
  ``torch==2.14.0`` CPU is in a separate env, ``allo-torch214`` (below).
- **LLVM_BUILD_DIR:** on this host ``conda activate allo`` *does* set it, to the
  shared build ``/work/shared/common/llvm-project-main/build-rhel8``
  (``activate.d/env_vars.sh``; llvm-project ``6b09f739``, Release,
  ``clang;mlir;openmp``). That is also the build this checkout's bindings link
  against (``mlir/build/CMakeCache.txt``). **Do not export**
  ``/home/sk3463/llvm-allo-6b09f739/build`` here: that directory no longer
  exists on zhang-21 (already gone at 2026-10-01 22:41 EDT; who removed it is not
  established, ``/home`` is full and was being cleaned that evening), and
  exporting it makes the simulator fail with ``Failed to create MemoryBuffer
  for .../libmlir_runner_utils.so`` followed by ``RuntimeError: Unknown
  function <top>``.

Installed 2026-10-02
^^^^^^^^^^^^^^^^^^^^

Both pinned installs of README D-8, by the ``zhang21`` session.

**Verilator 5.052** (conda-forge ``verilator==5.052=py312pl5321h9d6c286_0``),
in its own prefix so no shared env changes::

   ALLO_VERILATOR_HOME=/work/shared/users/phd/sk3463/tools/verilator scripts/verilator-setup.sh
   eval "$(ALLO_VERILATOR_HOME=/work/shared/users/phd/sk3463/tools/verilator scripts/verilator-setup.sh --env)"
   # -> Verilator 5.052 2026-09-05 rev conda-forge build (82 s; conda's package
   #    cache is already on /work: .../miniconda3/pkgs)

Verilator compiles its generated C++ with the host g++; use ``gcc-toolset-13``
(``scl enable gcc-toolset-13 -- bash -c '...'``). A ``--binary --timing``
smoke test builds and runs with g++ 13.3.1. MiniTPU, cloned at
``/work/shared/users/phd/sk3463/minitpu`` (``b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1``):

.. code-block:: text

   $ scl enable gcc-toolset-13 -- bash -c 'export PATH=/work/shared/users/phd/sk3463/tools/verilator/bin:$PATH; bash tb/run_verilator_unit_suite.sh'
   PASS tb_copy_abi_v9 ... PASS tb_isa_conformance
   == unit suite complete: 14 testbenches          # 14/14 PASS, 80 s wall

**torch 2.14.0 CPU**, in a new env cloned from ``allo`` rather than in ``allo``
itself: every session on this host shares ``allo``, and Allo's PyTorch
frontend imports torch, so swapping its torch under them was not worth the
risk (the host has no GPU; nothing in ``allo`` requires torch by package)::

   conda create -n allo-torch214 --clone allo          # ~30 min on NFS, 8 GB
   PIP_CACHE_DIR=/work/shared/users/phd/sk3463/.cache/pip \
     $CONDA_PREFIX/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
   python -c "import torch; print(torch.__version__)"   # 2.14.0+cpu

The clone keeps the CUDA wheels' ``nvidia-*`` packages; they are unused.
Verified with ``conda activate allo-torch214`` and no ``LLVM_BUILD_DIR``
export (see above):

.. code-block:: text

   $ cd examples/tinytpu && make mlp                    # 28 s
     mlp_small_l0   8x32x32   design vs isa_ref over all 4096 bytes of C: 0 differ
     mlp_small_l1   8x32x16   design vs isa_ref over all 4096 bytes of C: 0 differ
   [✓] Against PyTorch
     0 of 384 output bytes differ

The Makefile probes whether the shell's ``python`` imports allo with the
checkout on ``sys.path`` (allo is not pip-installed in this env; the scripts
put the repo on ``sys.path``), and only otherwise falls back to
``conda run -n allo`` -- the *other* env, with torch 2.10 and its activation
banner.

hlslibs ``ac_types`` (no Catapult licence needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Catapult itself is not installed here, but its *data types* are open source,
and that is enough to compile an emitted Catapult ``kernel.cpp`` with plain
``g++``. This host has an hlslibs checkout (29.3.0) at::

    ~/.cache/allo/ac_types/include        # holds ac_int.h, ac_fixed.h, ac_std_float.h, ...

which is the third place the Catapult emit gate looks (the gate is documented
on the published Catapult page, ``docs/source/backends/catapult.rst``), after ``$ALLO_AC_TYPES_INCLUDE`` and ``$MGC_HOME/shared/include``. It is
a cache, not a checked-in copy -- recreate it with::

    git clone https://github.com/hlslibs/ac_types ~/.cache/allo/ac_types
    git -C ~/.cache/allo/ac_types checkout f542cd681bf388f98bc5676e9a8d12952c3e65db   # 4.9.0, Catapult 2024.2's

Without it, ``s.build(target="catapult", ...)`` still runs the gate's text
stage but skips the compile stage with a banner on stderr. Any script that
produces a handoff should export ``ALLO_REQUIRE_AC_TYPES=1`` so the skip is a
failure instead.

There is also a 2016-vintage vendored copy at
``~/chipyard/generators/nvdla/src/main/resources/hw/cmod/hls/include``. It is
**not** a substitute: it has no ``ac_std_float.h``, so any kernel with an
``f32`` operand (which the emitter maps to ``ac_ieee_float<binary32>``) will
not compile against it.

SystemC csim without Catapult: a pinned stand-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``target="systemc"`` emits SystemC **for Catapult**. On a Catapult host, csim is
compiled against Catapult's own copies of SystemC, MatchLib Connections and
``ac_types`` (``$MGC_HOME/shared``). ``scripts/systemc-csim-setup.sh`` builds a
stand-in for hosts without Catapult. It assembles the open-source equivalents
at pinned commits in the same layout, and prints the variables to export.

Its library versions match what Catapult 2024.2/1130128 bundles, read on
zhang-21 on 2026-10-01: SystemC 2.3.3, Connections 2.2.0, ``ac_types`` 4.9.0
and ``ac_simutils`` 1.6.0. Only the compiler differs: the host's g++ against
Catapult's 10.3.0. It is a functional pre-check and nothing more:

- nothing in it synthesizes, schedules or produces RTL;

On zhang-21 on 2026-10-02, Catapult's own csim libraries gave results identical
to the stand-in's for TinyTPU and EVA
(``dev/records/catapult_handoff/zhang21_compare_2026-10-02/``). There, csim must
use Catapult's g++ 10.3.0, which ``module load catapult-2024`` puts first on
``PATH``. The system g++ 8.5 fails to link (``GLIBCXX_3.4.26``).
On 2026-10-01, TinyTPU (three ``stress_isa`` cases) and EVA
(``cosim_eva_systemc.py``) gave the same results on it as on the unpinned
probe.

Pinned external dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every external dependency of a flow is pinned by commit, version or checksum.
A row marked **unpinned** is an open defect.

.. list-table::
   :header-rows: 1

   * - dependency
     - pin
     - where recorded
   * - LLVM
     - ``6b09f739`` (submodule)
     - ``.gitmodules``, ``externals/``
   * - past-python-bindings
     - ``65f989b`` wheel
     - ``requirements.txt``
   * - hlslibs ``ac_types`` / ``ac_simutils`` / ``matchlib_connections``
     - tags 4.9.0 / 1.6.0 / 2.2.0 (``f542cd6`` / ``9aada6f`` / ``6a3003b``), the
       versions Catapult 2024.2 bundles
     - ``scripts/systemc-csim-setup.sh``
   * - Accellera SystemC
     - 2.3.3 (``38b8a2c``), built from source with the host g++ (Catapult's is
       2.3.3 built with its g++ 10.3.0)
     - ``scripts/systemc-csim-setup.sh``
   * - Vitis HLS / Vivado
     - 2023.2, by install path
     - this page
   * - Verilator
     - 5.052, conda-forge build ``py312pl5321h9d6c286_0`` (the ladder's pin).
       ace-01's ``~/.local/bin/verilator`` is an older build, development commit
       ``228635918ed0`` ("5.051")
     - ``scripts/verilator-setup.sh``
   * - PyTorch (the TinyTPU walk-through)
     - ``torch==2.14.0`` CPU
     - ``examples/tinytpu/README.md``
   * - ``ucb-bar/chia``, opencode
     - ``16c35e9``, ``opencode-ai@1.18.25``
     - ``examples/tinytpu/chia_agent/requirements.txt``, ``package-lock.json``
   * - Julian Bushlow's ASIC flow
     - vendored at ``e903e36``
     - ``allo/backend/asic/PROVENANCE.md``
   * - FreePDK45 standard cells
     - ``stdcells.db`` md5 in each run's settings snapshot
     - ``dev/records/tinytpu/``
   * - Design Compiler, mflowgen (zhang-21)
     - W-2024.09; mflowgen 0.8.0 at ``aee0e5d6``
     - ``examples/tinytpu/asic_synthesis/``
   * - Catapult (zhang-21)
     - 2024.2/1130128
     - ``dev/records/catapult_handoff/``
   * - Chipyard / Gemmini
     - ``e0207441`` / ``25809f7`` / rocc-tests ``1a1a1c6``
     - ``examples/tinytpu/gemmini/CONFIG_DELTA.txt``
   * - Kai Shao's ACT (reference)
     - ``kai/act`` ``3c1ad38``
     - ``ATTRIBUTION.md``
   * - OpenRAM (zhang-21, the SRAM-macro path)
     - ``b2b069ce119d1488cbe6883b2240bceb5c7ce29a`` (``v1.2.48-41``, 2026-08-16),
       cloned at ``/work/shared/users/phd/sk3463/tools/OpenRAM``; FreePDK45 as
       bundled with it (``technology/freepdk45``; no NCSU PDK, DRC/LVS off);
       its conda env ``tools/envs/openram`` (Python 3.11.16, numpy 2.4.6,
       scipy 1.17.1, scikit-learn 1.9.1, ngspice 41) pinned by URL and md5 in
       ``dev/records/minitpu/asic_memories_2026-10-04/openram_env/``
     - ``dev/records/minitpu/asic_memories_2026-10-04.rst``
   * - OpenRAM bank macro ``sram_2rw_64x512_freepdk45`` (``mid`` = 8 banks)
     - generated once (OpenRAM as above, ``openram/cfg_2rw_64x512.py``, 1,151 s)
       and reused for every bank; sha256 ``.v`` ``bc3ceb63fd91…``, ``.lib``
       ``f207b9246b42…``, ``.lef`` ``d5cc4f848f79…`` (all three committed in
       ``asic_memories_2026-10-04/openram/``), ``.gds`` ``96c2b751fd07…``
       (14.4 MB, not committed: regenerate and compare); its ``lc_shell``
       ``.db`` md5 ``8b619e1feb7f…``
     - ``asic_memories_2026-10-04.rst`` s.5.1
   * - Catapult Memory Generator (``/MemGen/MemoryGenerator_BuildLib``), Library Compiler
     - part of Catapult 2024.2/1130128 (plain Catapult licence);
       ``lc_shell`` W-2024.09-SP5-3 (``/opt/synopsys/lc/W-2024.09-SP5-3``; the
       W-2024.09 and V-2023.12 builds fail on this host's ``libkrb5``)
     - ``asic_memories_2026-10-04/scripts/memgen_spec.py``, ``dc/``
   * - MiniTPU (``~/core/minitpu``)
     - ``b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1`` (2026-09-30)
     - README (design target)
   * - Python packages other than the above
     - **unpinned** (``requirements.txt`` has ranges or nothing; upstream practice)
     - ``requirements.txt``

Vitis binutils vs. glibc ``.relr.dyn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vitis 2023.2 ships binutils 2.37, which cannot read this system's glibc:
``unknown type [0x13] section '.relr.dyn'``, then ``cannot find libm.so.6``. Both
the csim and the cosim link fail without it.

**zhang-21 is the opposite case (2026-10-08).** Its ``/usr/bin/ld`` is 2.30
(``2.30-128.el8_10``, glibc 2.28), and with ``-B/usr/bin`` the cosim
testbench link fails on every object Vitis's gcc 8.3.0 compiles::

   /usr/bin/ld: obj/kernel.cpp_pre.cpp.tb.o: unable to initialize decompress status for section .debug_info
   collect2: error: ld returned 1 exit status

``cosim.py`` (``linker_dir()``) therefore chooses per host: ``/usr/bin`` when
the system ``ld`` is at least 2.37 (what Vitis bundles), otherwise the
``allo`` env's binutils **2.44** (``$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ld``,
conda-forge ``binutils_impl_linux-64``), linked as ``ld`` into
``$TMPDIR/tinytpu_ld_<uid>`` and passed as ``-B``; ``TPU_LD_DIR=<dir with an
ld>`` overrides both. Every TinyTPU cosim entry point (``cosim.py``,
``workloads/run.py --cosim``, ``mutate.py``'s cosim level, ``reproduce.sh``)
goes through that one string. Before this, ``reproduce.sh``'s cosim stage had
never passed on zhang-21, and ``mutate.py`` reported its RTL-only mutant as
"caught" by a link failure. Measured with it on 2026-10-08, zhang-21 reproduces
175 / 265 / 421 / 482 / 674 (``examples/tinytpu/README.md``, section 4). Wall
times on this host, load average about 30: ``make mlp-cosim MODEL=mlp_small``
4 min 40 s (csynth 2 min, two layer cosims; 2 781 cycles, the published
figure); full ``reproduce.sh`` 10 min 36 s (the cosim stage 8 min);
``TPU_TB=stress python cosim.py`` 6 min 52 s (``COSIM OK``, 6 stress calls at
each of 5 shapes, ``ar_distance`` included). The same mechanism was first
written for the TinyTPU instance
(``examples/minitpu/template/instances/tinytpu/run_gates.py``).

.. warning::

   **Do not source ``settings64.sh`` in a shell you then build the bindings in.**
   Vitis prepends its own libraries, which shadow the system ones, and
   ``cmake`` dies before configuring anything:

   .. code-block:: text

      cmake: error while loading shared libraries: libidn.so.11:
      cannot open shared object file: No such file or directory

   You do not need to source it at all -- the scripts do it themselves (row
   above; ``cosim.py`` hardcodes the path in its ``VITIS`` constant). Observed
   2026-09-25 while verifying the published cycle counts with cosim in a fresh
   worktree: sourcing it broke the build, and running ``reproduce.sh`` with the
   Vitis environment untouched works. Putting only
   ``/opt/xilinx/Vitis_HLS/2023.2/bin`` on ``PATH`` also works, but it is
   redundant rather than required.

   This is worth a warning because the obvious way to make ``vitis_hls``
   available is the one that makes the build fail first, and the failure names
   a library rather than the cause.

The fix in tree is **not** a ``PATH`` override — it is a compiler-driver flag.
``examples/tinytpu/cosim.py`` sets ``LDFLAGS = "-B/usr/bin"`` and
splices it into the generated Vitis script, pointing the driver at the system
linker (2.42) while leaving the rest of the Vitis toolchain in place. Same
story in ``docs/source/designs/tinytpu_isa.rst``. Any new Vitis flow needs the
equivalent.

Python for cosim vs. Python for ``allo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``allo`` conda env is **python 3.12** (``3.12.13``); the miniconda **base** env
is **python 3.14** (``3.14.6``). This matters because CMake picks them
independently: the retired ``chia-codesign`` worktree's ``build/CMakeCache.txt`` recorded
``Python3`` as the env's 3.12 but ``Python`` as base 3.14, and nanobind took its
suffix from the latter — ``NB_SUFFIX=.cpython-314-x86_64-linux-gnu.so``. The
chia worktree's bindings are therefore tagged ``cpython-314`` and the 3.12
interpreter will not import them. Pass an explicit ``-DPython_EXECUTABLE=`` as
well as ``-DPython3_EXECUTABLE=`` when configuring.

The python 3.14 venv that carried ``cocotb 2.1.0`` + ``ml_dtypes`` for chia cosim
**is gone** — it lived under ``/tmp`` and nothing matching it survives. The only
cocotb on the host now is **2.0.1** in the ``mininpu`` conda env (python 3.11);
neither base 3.14 nor the ``allo`` env has cocotb or ``ml_dtypes``. Rebuilding that
venv is a prerequisite for ``allo/backend/rtl/sim/`` on ``chia-codesign``.

LLVM/MLIR builds and worktrees
------------------------------

One LLVM build, at the pinned revision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since 2026-09-19 there is **one** LLVM/MLIR build that matters:
``/home/sk3463/llvm-allo-6b09f739/build`` (LLVM ``22.0.0git`` at ``6b09f739``), the
revision ``main`` pins for ``externals/llvm-project``. ``LLVM_BUILD_DIR`` points
at it, and every worktree's ``mlir/build`` is configured against it. ``git
status`` on ``main`` is clean: the submodule checkout is at the pin.

Before that date the submodule checkout was deliberately left at ``040a6419``
(LLVM 23), because the ``chia-codesign`` worktree linked against an in-tree
LLVM 23 build of it (``externals/llvm-project/build``, 3.9 GB) and imported
four Python binding files through absolute symlinks into ``main``'s submodule
checkout. That made ``M externals/llvm-project`` the *correct* state, and
checking the pin out silently put LLVM 22 sources under LLVM 23 binaries --
which happened once, on 2026-09-18, and was reverted. Retiring
``chia-codesign`` (tag ``chia-codesign-final``) removed the only consumer, so
the in-tree LLVM 23 build was deleted and the pin restored. If a future branch
needs a different LLVM, give it its own out-of-tree build rather than checking
out the shared submodule.

Never point one worktree's build at another worktree's tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cmake preserves mtimes, so a cross-worktree build dependency leaves **nothing
looking stale**. No rebuild is triggered, no warning is printed, and the symptom
surfaces much later as an unexplained ABI or dialect mismatch.

The worked example is the arrangement that caused the 2026-09-18 incident. It
no longer exists -- the ``chia-codesign`` worktree was retired on 2026-09-19
(tag ``chia-codesign-final``) -- and is kept here because the rule below was
learned from it:

+----------------------+----------------------------------------------------------+-------------------------------------------------------------------+
|                      | ``<main worktree>`` (``main``)                           | ``<other worktree>`` (``chia-codesign``)                          |
+======================+==========================================================+===================================================================+
| ``allo/_mlir``       | symlink ``-> ../mlir/build/tools/allo/_mlir``,           | a **real directory** of installed output, not a symlink           |
|                      | **relative, stays inside the worktree**                  |                                                                   |
+----------------------+----------------------------------------------------------+-------------------------------------------------------------------+
| extension ABI tag    | ``_allo.cpython-312-*.so``                               | ``_allo.cpython-314-*.so``                                        |
+----------------------+----------------------------------------------------------+-------------------------------------------------------------------+
| runtime soname       | ``libAlloDataflowRuntime.so.22.0git``                    | ``libAlloDataflowRuntime.so.23.0git``                             |
+----------------------+----------------------------------------------------------+-------------------------------------------------------------------+
| build's ``LLVM_DIR`` | ``/home/sk3463/llvm-allo-6b09f739/build/lib/cmake/llvm`` | ``<main worktree>/externals/llvm-project/build/lib/cmake/llvm``   |
|                      |                                                          | -- **the other worktree**                                         |
+----------------------+----------------------------------------------------------+-------------------------------------------------------------------+

Worse, individual files inside ``<other worktree>/allo/_mlir`` are
absolute symlinks out of the worktree: ``ir.py``, ``passmanager.py``, ``rewrite.py``
and ``execution_engine.py`` all point into
``<main worktree>/externals/llvm-project/mlir/python/mlir/``, i.e. into ``main``'s
submodule checkout. Only ``schedule.py`` stays local
(``-> <other worktree>/mlir/python/allo/schedule.py``).

**This is the mechanism by which the 2026-09-18 mistake above did its damage.**
Checking out a different revision of ``main``'s submodule swapped four of
``chia-codesign``'s Python binding files to a different LLVM version, with no
build step, no warning, and nothing in either worktree's ``git status`` pointing
at it. The failure this produces is an ABI or dialect error that appears to come
from the *other* branch's code.

Rules:

1. A worktree's ``allo/_mlir`` symlink target must be **relative** and must stay
   inside that worktree. ``main``'s is correct; copy that shape.
2. Each worktree gets its own LLVM/MLIR build directory, or they share a
   read-only external one (like ``llvm-allo-6b09f739``) that **neither** worktree's
   ``externals/`` can be checked out from under.
3. Never run a build in worktree A that names a path under worktree B in
   ``LLVM_DIR``/``MLIR_DIR``. Checking out a branch in B then silently changes A's
   sources.
4. On any "impossible" ABI or dialect error, first ``readlink -f allo/_mlir``,
   then ``grep -E '^(LLVM|MLIR)_DIR' <build>/CMakeCache.txt``, and check the
   soname version suffix in ``allo/_mlir/_mlir_libs/``. Those three answer it
   faster than any rebuild.

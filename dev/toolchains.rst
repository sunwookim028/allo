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

Vitis binutils vs. glibc ``.relr.dyn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vitis 2023.2 ships binutils 2.37, which cannot read this system's glibc:
``unknown type [0x13] section '.relr.dyn'``, then ``cannot find libm.so.6``. Both
the csim and the cosim link fail without it.

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

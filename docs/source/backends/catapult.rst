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

############################
Siemens Catapult HLS (FPGA)
############################

The `Catapult HLS <https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/>`_ backend enables Allo to generate hardware accelerators using Siemens' high-level synthesis toolchain. Catapult HLS uses Algorithmic C (AC) data types (``ac_int``, ``ac_fixed``, ``ac_channel``) and provides industry-leading quality of results for ASIC and FPGA designs.

Prerequisites
-------------
To use the Catapult HLS backend, you need:

1. **Siemens Catapult HLS** installed and licensed
2. **MGC_HOME** environment variable set to your Catapult installation directory
3. **AC Datatypes** headers available (typically included with Catapult)

.. code-block:: bash

   # Example environment setup
   export MGC_HOME=/path/to/catapult
   export PATH=$MGC_HOME/bin:$PATH

Kernel Definition
-----------------
Define your kernel using Allo's Python-embedded DSL. Here's an example of a vector addition kernel:

.. code-block:: python

   import allo
   from allo.ir.types import int32

   def vvadd(a: int32[100], b: int32[100]) -> int32[100]:
       c: int32[100]
       for i in range(100):
           c[i] = a[i] + b[i]
       return c

   s = allo.customize(vvadd)

Code Generation for Catapult HLS
--------------------------------
Allo supports two modes for Catapult HLS:

1. **C Simulation (csim)**:
   Compiles the generated C++ code with g++ and runs functional simulation. This mode is useful for verifying the correctness of your design before synthesis.

   .. code-block:: python

      import numpy as np

      mod = s.build(target="catapult", mode="csim", project="vvadd.prj")

      # Prepare test data
      np_a = np.random.randint(0, 100, size=(100,)).astype(np.int32)
      np_b = np.random.randint(0, 100, size=(100,)).astype(np.int32)
      np_c = np.zeros((100,), dtype=np.int32)

      # Run simulation
      mod(np_a, np_b, np_c)

      # Verify results
      np.testing.assert_array_equal(np_c, np_a + np_b)

2. **C Synthesis (csyn)**:
   Runs Catapult HLS synthesis to generate RTL. This mode invokes the Catapult tool to synthesize your design.

   .. code-block:: python

      mod = s.build(target="catapult", mode="csyn", project="vvadd.prj")

      # Run synthesis (no arguments needed for csyn mode)
      mod()

Generated Code Features
-----------------------
The Catapult backend generates C++ code with Catapult-specific features:

**AC Datatypes**

Allo automatically maps data types to Catapult's AC datatypes:

- Integer types map to ``ac_int<W, S>`` for non-standard widths
- Fixed-point types map to ``ac_fixed<W, I, S>``
- Streams map to ``ac_channel<T>``

.. code-block:: cpp

   // Generated headers
   #include <ac_int.h>
   #include <ac_fixed.h>
   #include <ac_channel.h>

**Catapult Pragmas**

Allo's scheduling primitives are translated to Catapult-specific pragmas:

.. code-block:: python

   s = allo.customize(kernel)
   s.pipeline("i")      # Generates: #pragma hls_pipeline_init_interval 1
   s.unroll("j")        # Generates: #pragma hls_unroll
   s.unroll("k", 4)     # Generates: #pragma hls_unroll 4

Project Structure
-----------------
The generated project (e.g., ``vvadd.prj``) includes:

- **kernel.cpp**: The synthesizable kernel code with AC datatypes
- **kernel.h**: Header file for the kernel interface
- **host.cpp**: Host code for C simulation (csim mode only)
- **run.tcl**: TCL script for Catapult synthesis
- **Makefile**: Build scripts for the project

Example: Matrix Multiplication
------------------------------
Here's a complete example of matrix multiplication with Catapult HLS:

.. code-block:: python

   import allo
   from allo.ir.types import int32
   import numpy as np

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

   s = allo.customize(gemm)

   # Apply optimizations
   s.pipeline("j")
   s.unroll("k", 4)

   # Build for Catapult
   with tempfile.TemporaryDirectory() as tmpdir:
       mod = s.build(target="catapult", mode="csim", project=tmpdir)

       # Test the design
       np_A = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
       np_B = np.random.randint(0, 10, size=(32, 32)).astype(np.int32)
       np_C = np.zeros((32, 32), dtype=np.int32)

       mod(np_A, np_B, np_C)
       np.testing.assert_array_equal(np_C, np.matmul(np_A, np_B))

Configuration Options
---------------------
You can customize the synthesis through the ``configs`` dictionary:

.. code-block:: python

   mod = s.build(
       target="catapult",
       mode="csyn",
       project="gemm.prj",
       configs={
           "frequency": 500,  # Target frequency in MHz (default: 300)
       },
   )

The frequency setting affects the clock period constraint in the generated TCL script.

Comparison with Vitis HLS
-------------------------
While both Catapult and Vitis HLS are high-level synthesis tools, they have different characteristics:

.. list-table::
   :header-rows: 1

   * - Feature
     - Catapult HLS
     - Vitis HLS
   * - Data Types
     - AC datatypes (``ac_int``, ``ac_fixed``)
     - AP datatypes (``ap_int``, ``ap_fixed``)
   * - Streams
     - ``ac_channel<T>``
     - ``hls::stream<T>``
   * - Pipeline Pragma
     - ``#pragma hls_pipeline_init_interval``
     - ``#pragma HLS pipeline``
   * - Unroll Pragma
     - ``#pragma hls_unroll``
     - ``#pragma HLS unroll``
   * - Vendor
     - Siemens
     - AMD/Xilinx

Host Setup (zhang-21)
---------------------
The rest of this page records the fork's working knowledge of Catapult, gathered while synthesizing
dataflow designs with it. Catapult is installed on one server the project uses,
``zhang-21.ece.cornell.edu`` (RHEL 8.10, glibc 2.28), and all Catapult synthesis runs there. It is
**not** installed on ``ace-01``, the host where most other work in this fork is done (see
:doc:`/developer/toolchains`).

Installation
~~~~~~~~~~~~
Two versions are installed under ``/opt/siemens/catapult/``:

.. list-table::
   :header-rows: 1

   * - Version
     - Path
   * - 2024.2 (recommended)
     - ``/opt/siemens/catapult/2024.2/``
   * - 2024.1
     - ``/opt/siemens/catapult/2024.1_2-1117371/``

Use the environment module if it is available, and set the path by hand otherwise:

.. code-block:: bash

   # Option A: module (sets MGC_HOME and updates PATH)
   module load mentor-Catapult_synthesis_10.5a

   # Option B: manual path, if the module is not installed
   export MGC_HOME=/opt/siemens/catapult/2024.2
   export PATH=$MGC_HOME/bin:$PATH

   catapult -version
   # Expected: Catapult Ultra 2024.2 (build ...)

If ``MGC_HOME`` is unset, the Allo backend also searches ``/opt/siemens/catapult/<version>/`` and
picks the most recent version (``_find_catapult_binary`` in ``allo/backend/hls.py``).

Catapult ships its own AC datatypes headers under ``$MGC_HOME/shared/include/`` (``ac_int.h``,
``ac_fixed.h``, ``ac_channel.h``, ``ac_std_float.h``, ``ac_math/``), so no separate install is
needed. Catapult analyzes the input C++ with its own EDG front end; the system GCC 8 on zhang-21 is
sufficient for supporting scripts.

Licences
~~~~~~~~
**Set these yourself. Do not assume the host is configured.** An earlier revision of these notes
claimed the server was "pre-configured on zhang-21 via ``CATAPULT_LICENSE_FILE``". That was checked
on zhang-21 on 2026-09-18 and is false: the variable is empty, and it is the wrong variable name
besides.

.. code-block:: bash

   export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu   # Siemens / Catapult
   export CDS_LIC_FILE=5280@en-license-05.coecis.cornell.edu        # Cadence / Xcelium
   unset LD_PRELOAD                                                 # the login env breaks xrun

**Verified on zhang-21, 2026-09-18.** Catapult 2024.2 prints "Connected to license server (LIC-13)"
and "Catapult product license successfully checked out (LIC-14)"; ``xrun`` 24.03 compiles and runs
a module to ``$finish``. ``LD_PRELOAD`` was already unset in that login environment, so the
``unset`` above is belt-and-braces rather than required. Keep it, since the note that flagged it
(``67559cad``) says some login environments do set it.

Provenance, for when these stop working: both exports were recovered from the commits that
produced the Catapult synthesis and Xcelium cosim results on record (``c7402f9f``, ``0eff4888``,
``67559cad`` on ``choonsik1/SystemC-emitter``; ``git log --all -S"MGLS_LICENSE_FILE"`` finds them).

**Both tools run** ``-version`` **without checking a licence out**, so a successful ``-version``
proves nothing. Check a real checkout before concluding the host is ready. Failure modes observed
on zhang-21 on 2026-09-18, all with the *wrong* server:

.. list-table::
   :header-rows: 1

   * - Symptom
     - Meaning
   * - Catapult ``mgls_errno 515``
     - no Siemens licence reachable
   * - Xcelium ``LMC-01902``
     - no ``CDS_LIC_FILE`` set at all
   * - Xcelium ``LMF-03097``
     - server answered, but has no Xcelium feature

The last one is the trap: ``27020@en-license-05`` is the **Synopsys** vendor daemon on the same
host. Right machine, wrong port, so the server replies and the checkout still fails. Siemens is
1717, Cadence is 5280, Synopsys is 27020, all on ``en-license-05.coecis.cornell.edu``.

Running Catapult by Hand
------------------------
A minimal Catapult project needs two files, ``kernel.cpp`` (the top function and its
sub-functions) and ``run.tcl``. On zhang-21, put projects under ``/scratch/$USER/`` (1.8 TB local
SSD), not ``/work/``, to avoid quota:

.. code-block:: bash

   mkdir -p /scratch/$USER/catapult_projects/my_design.prj
   cd /scratch/$USER/catapult_projects/my_design.prj
   catapult -shell -f run.tcl 2>&1 | tee catapult.log

``-shell`` runs headless. Synthesis typically takes 1-5 minutes for small designs. A minimal
``run.tcl``:

.. code-block:: text

   # --- Input ---
   solution options set /Input/CppStandard c++11
   solution options set /Input/CompilerFlags {-D_GLIBCXX_USE_CXX11_ABI=0}
   solution file add kernel.cpp -type C++

   # --- Design hierarchy ---
   directive set -DESIGN_HIERARCHY top_function_name
   # 2.0 ns = 500 MHz
   directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0}}

   # --- Interface scheduling (see "Recommended Scheduling Directives") ---
   directive set -IO_MODE super
   directive set -SPECULATE true

   # --- Target library ---
   solution options set /Output/OutputVerilog true
   solution library add nangate-45nm_beh

   # --- Synthesis steps ---
   go analyze
   go compile
   solution library add ccs_sample_mem
   go assembly
   go extract

Recommended Scheduling Directives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``-IO_MODE super`` and ``-SPECULATE true`` are not cosmetic. Catapult's **default**
``-IO_MODE fixed`` **pins each port's** ``vld``/``dat`` **to a fixed cycle offset**, so a kernel
issuing more than one non-blocking handshake per loop body (every NoC router does) has its
handshakes collide, and fails with SCHD-67 / SCHD-30 (see `Troubleshooting`_). ``super`` lets the
scheduler place each handshake anywhere in the loop window; ``-SPECULATE true`` covers conditional
pushes. Put them in before the first run rather than after the first failure.

**Measured effect: 2/32 -> 22/32 dataflow designs csynth**, with 1 residual scheduling failure in
the whole set. (The remaining 10 fail for reasons other than scheduling.) A 38-``PushNB`` +
38-``PopNB`` Channel router csynths clean with these two.

**Cost -- not a free win.** Under ``super`` the scheduler's placement space is far larger:
``router_rvn_chan`` ran **40 minutes at 49 GB RSS without finishing**. Small designs still finish in
minutes. Budget accordingly on a shared machine, and reach for ``super`` on a block that will not
schedule rather than applying it blanket.

Where this came from, because it is the part that cost the most time: these are MatchLib's own
required settings, in ``hls/run_hls_global_setup.tcl``. They are *not* in
``eva_router/go_hls.tcl``, which is where one looks first. A MatchLib-style Tcl copied from
``go_hls.tcl`` was tried, reported as "did not help", and sent the earlier diagnosis down a
structural-conflict / ``SC_METHOD``-emitter path that was the wrong remedy for a correctly
identified mechanism. If a Connections design will not schedule, open
``run_hls_global_setup.tcl`` first.

Provenance: measured on the ``choonsik1/SystemC-emitter`` fork, on the **SystemC/Connections**
flow. ``allo/backend/catapult.py`` there emits both lines under ``platform == "systemc"``, and
``docs/noc/FINDINGS_wire_channel.md`` §7 on that branch carries the numbers. The directives are
solution-level and apply to any Catapult run; the 2/32 -> 22/32 figure is specific to that
Connections corpus and has not been re-measured for ``ac_channel`` designs.

.. note::

   **This repository's** ``allo/backend/catapult.py`` **emits neither directive** (verified
   2026-09-18), so a run driven from the in-tree Allo backend gets the default ``fixed``.

Block Synthesis for Hierarchical Designs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Catapult has two modes for sub-functions:

- **Inline** (default): the sub-function is merged into the caller's schedule. Local channels are
  treated as variables -- HIER-10 fires if they cross source-level boundaries, and depth is
  inferred as 0.
- **Block** (``-block``): the sub-function is compiled as a separate RTL module with real FIFO
  ports. Catapult infers correct depths based on the producer/consumer II mismatch.

Use block synthesis whenever ``ac_channel`` objects are declared in one function and passed by
reference to another:

.. code-block:: tcl

   go analyze
   solution design set sub_func_a -block
   solution design set sub_func_b -block
   go compile
   solution library add ccs_sample_mem
   go assembly
   go extract

From Allo, pass the sub-function names as ``configs={"sub_funcs": [...]}``; ``codegen_tcl`` in
``allo/backend/catapult.py`` then emits one ``solution design set <fn> -block`` line per name.

**FIFO depth inference.** With block synthesis Catapult selects FIFO depth automatically from the
throughput ratio of producer and consumer. Example: MT throughput 69 cycles, CT throughput 298
cycles -> ratio 4.3x -> Catapult infers depth=16 to buffer a full burst without stalling. The full
per-module results for that design are in :doc:`/records/catapult_decoupled_2x1`.

Reading the Reports
~~~~~~~~~~~~~~~~~~~
After a successful run, outputs appear in ``<prj>/Catapult_<N>/<top>.v1/`` (``Catapult/<top>.v1/``
for a project generated by Allo):

.. list-table::
   :header-rows: 1

   * - File
     - Contents
   * - ``rtl.v``
     - Generated Verilog RTL
   * - ``cycle.rpt``
     - Latency / throughput per loop
   * - ``rtl.rpt``
     - Area report (Catapult score units)
   * - ``schedule.rpt``
     - Detailed schedule, resource binding

In ``cycle.rpt``, *Latency* is the first-output latency in clock cycles and *Throughput* is the
initiation interval (II), the cycles between successive invocations. Areas in ``rtl.rpt`` are
Catapult **score units** (a scheduling metric, not physical nm²), in four categories: ``REG``
(pipeline registers, arrays), ``FUNC`` (datapath logic: adders, multipliers), ``MUX`` (datapath
multiplexers) and ``LOGIC`` (control logic).

PPA Analysis
------------
Besides ``csim`` and ``csyn``, the Catapult target accepts ``mode="ppa"``. It runs the same
synthesis flow as ``csyn`` and then extracts metrics from the reports:

.. code-block:: python

   s = allo.customize(top)
   # s.partition(...), s.pipeline(...)
   mod = s.build(target="catapult", mode="ppa", project="my_ppa_project")
   stats = mod()   # runs synthesis, prints a metrics table, returns a dict
   print(stats)

``stats`` contains ``Latency (cycles)`` (``Max Latency`` from ``cycle.rpt``), ``Area``
(``Total Area`` from ``area.rpt``, unit depends on the technology library) and ``Power`` (from
``power.rpt`` or ``power_summary.rpt`` if present; ``N/A`` otherwise), all read from
``<project>/Catapult/<top>.v1/``. It also carries a ``hierarchical`` entry: a per-module area
(and, where reported, power) breakdown parsed from ``area.rpt``, so that per-PE and interconnect
cost can be separated. That breakdown needs a hierarchical design; Catapult preserves the hierarchy
in the generated RTL unless it is fully flattened.

Technology library selection is by ``configs={"device": ...}``: anything containing ``45nm`` or
``nangate`` selects ``nangate-45nm_beh`` (the default), anything containing ``sky130`` selects
``sky130``, and any other string is passed to ``solution library add`` unchanged.

.. note::

   The mode, the parsed report fields and the library selection above were checked against
   ``allo/backend/catapult.py`` and ``allo/backend/hls.py`` in this tree; no ``ppa`` run is recorded
   here. Note also that ``codegen_tcl`` defaults ``frequency`` to **100** MHz when it is not given,
   not the 300 MHz stated under `Configuration Options`_.

**RTL-to-GDSII with OpenROAD.** For physical PPA, the synthesizable RTL at
``<project>/Catapult/<top>.v1/rtl.v`` can be passed to
`OpenROAD-flow-scripts <https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts>`_: set up
OpenROAD, write a ``config.mk`` pointing at the generated Verilog, run ``make`` in the OpenROAD
directory, and use OpenROAD's reports for post-route timing and power. This flow is described, not
exercised, in this repository.

Using the Allo Backend on RHEL 8
--------------------------------
The Allo MLIR backend (``.so`` files) requires GCC 13's ``libstdc++``. RHEL 8 (zhang-21) ships
GCC 8 only. The conda ``allo`` env includes the newer ``libstdc++``, but it must be on
``LD_LIBRARY_PATH`` *before* activating the env so that subprocesses inherit it:

.. code-block:: bash

   export LD_LIBRARY_PATH="/path/to/miniconda3/envs/allo/lib:$LD_LIBRARY_PATH"
   conda activate allo

For a permanent fix, add to ``~/.bashrc`` on zhang-21:

.. code-block:: bash

   if [[ "$(hostname)" == "zhang-21.ece.cornell.edu" ]]; then
       CONDA_ENV_LIB="/path/to/miniconda3/envs/allo/lib"
       export LD_LIBRARY_PATH="$CONDA_ENV_LIB:$LD_LIBRARY_PATH"
   fi

Or use the wrapper at the repository root, ``./run_allo.sh python my_script.py``, which sets
this automatically (it also prepends a hard-coded ``mlir/build/tools/allo/_mlir`` path under
``/path/to/allo``; edit both paths for another checkout).

**LLVM build.** The ``.so`` files in ``allo/_mlir/_mlir_libs/`` must be built against an LLVM that
is compatible with RHEL 8's glibc (the ``build-rhel8`` LLVM), not one compiled on a glibc 2.35 host;
a build made against a newer glibc than the host's fails at simulator init with a GLIBC_2.33 error.
An earlier revision of these notes said the conda ``allo`` env "already sets ``LLVM_BUILD_DIR``" to
``build-rhel8`` and must not be overridden. **That was corrected on 2026-09-18**: neither
``conda activate allo`` nor ``conda run`` sets ``LLVM_BUILD_DIR``, and the simulator asserts
``LLVM_BUILD_DIR is not set`` without it. Export it yourself, pointing at the build appropriate
for the host (see :doc:`/developer/pitfalls` and :doc:`/developer/toolchains`), and build the
project with ``pip install -v -e .``.

The end-to-end driver the earlier bring-up used, ``tests/dataflow/catapult_synth_decoupled_2x1.py``
(``--mode codegen | csyn | ppa``), is not in the tree; it stayed on the deleted
``feature/mesh-accelerator`` branch (last tip ``06ce561``). The nearest in-tree equivalent is
``tests/dataflow/hls_synth_decoupled.py``.

C++ Emitter vs. SystemC Emitter
-------------------------------
Two Catapult code generators exist for Allo, and they should not be confused:

- **The C++ emitter** (this backend: ``mlir/lib/Translation/EmitCatapultHLS.cpp``,
  ``allo/backend/catapult.py``) came from upstream PR #543 (Feb 2026). This fork added
  non-blocking stream support to it (2026-04-14) and the synthesis bring-up recorded in
  :doc:`/records/catapult_decoupled_2x1`. It emits ``ac_channel`` C++.
- **The SystemC emitter** lives on a separate fork, ``choonsik1/allo:SystemC-emitter``, with a
  different type set -- ``Stream`` / ``Channel`` / ``Wire`` over MatchLib Connections. It is
  described in :doc:`/extensions/catapult_systemc`.

The C++ emitter is a thin syntax layer over the Vivado emitter:
``class CatapultModuleEmitter : public allo::hls::VhlsModuleEmitter`` (``EmitCatapultHLS.cpp:109``).
Counted 2026-09-18, ``EmitCatapultHLS.cpp`` is 683 lines against ``EmitVivadoHLS.cpp``'s 3428
(20%), and it overrides **14** of the 57 member functions ``VhlsModuleEmitter`` declares (12
out-of-line, 2 inline in the class body): ``emitModule``, ``emitFunction``,
``emitFunctionDirectives``, ``emitValue``, ``emitArrayDecl``, ``emitArrayDirectives``,
``emitLoopDirectives``, ``emitStreamConstruct``, ``emitStreamTryGet``, ``emitStreamTryPut``,
``emitStreamEmpty``, ``emitStreamFull``, ``emitStatefulGlobalElementType``,
``emitFloatArrayElement``.

Every one substitutes *syntax* at a point where Vivado already emits something: type spellings
(``ac_ieee_float<binary32>``, ``ac_int<W,S>``), ``ac_channel`` for ``hls::stream``, directive
comments for pragmas, ``static`` on local channels, an ``f`` suffix on float literals. None adds a
construct; the other 43 emitters are inherited verbatim. So whatever the frontend cannot say, both
backends fail to say identically -- and where they differ, Catapult says *less*:

- ``try_get`` / ``try_put`` are emitted as **blocking** ``ch.read(v)`` / ``ch.write(v)`` with
  ``bool success = true;`` hard-coded, because ``nb_read`` / ``nb_write`` inside a spin-while loop
  segfaults Catapult's ``go compile`` (LOOP-19). That is harmless for the
  ``while not S.try_put(x): pass`` idiom but silently turns any design that branches on failure
  into a different, lock-step circuit, with no warning. See :ref:`limitation-18`.
- ``empty()`` is emitted as ``!ch.available(1)``, because ``ac_channel`` has no ``.empty()`` in the
  synthesizable subset (EDG CIN-59). That one is a faithful translation.

The one non-handshaked edge in play -- the SystemC ``Wire`` -- exists only on the SystemC path,
and it is wrong in RTL, not merely in csim. The investigation is in
:doc:`/extensions/catapult_systemc` and the conclusion in :ref:`limitation-22`.

Troubleshooting
---------------

**MGC_HOME not set**

If you see an error about MGC_HOME not being set, ensure the environment variable points to your Catapult installation:

.. code-block:: bash

   export MGC_HOME=/path/to/catapult

**AC headers not found**

If compilation fails due to missing AC headers, verify that the AC datatypes are installed:

.. code-block:: bash

   ls $MGC_HOME/shared/include/ac_int.h

**Synthesis fails**

Check the Catapult log files in your project directory for detailed error messages. Common issues include:

- Unsupported C++ constructs
- Memory access patterns that cannot be synthesized
- Timing constraints that cannot be met

**Error cookbook**

The following errors were hit and fixed while synthesizing Allo-generated designs (Catapult 2024.2,
``nangate-45nm_beh``).

*CIN-291: float not synthesizable*

.. code-block:: text

   Error: Type 'float' is not synthesizable with library 'nangate-45nm_beh'

The target library does not synthesize native C++ ``float`` or ``double``. Use
``ac_ieee_float<binary32>`` and add ``#include <ac_std_float.h>``. The Allo C++ emitter already
maps F32 to ``ac_ieee_float<binary32>``.

*HIER-47 / ASSERT-1: FIFO depth = 0*

.. code-block:: text

   Assertion failed: cap >= 0 (sif_ap_bif.cxx:1745)

Caused by flat (inline) synthesis of a design that passes ``ac_channel`` objects across function
boundaries. Use block synthesis for each sub-function (see `Block Synthesis for Hierarchical
Designs`_).

*HIER-6: non-static ac_channel*

.. code-block:: text

   Warning HIER-6: ac_channel variable must be static

Declare local ``ac_channel`` variables ``static``, e.g. ``static ac_channel<int> my_fifo;``. The
Allo emitter adds the ``static`` prefix.

*CRD-415: double literal conversion*

.. code-block:: text

   Error CRD-415: Cannot convert 'double' to 'ac_ieee_float<binary32>'

Use ``f``-suffixed float literals: ``0.0f`` not ``0.0``. The Allo emitter emits ``0.000000f``
(this also covers the related CRD-413 assignment issue).

*SCHD-67 / SCHD-30: will not schedule even with unlimited resources*

.. code-block:: text

   Error SCHD-67: ... could not schedule even with unlimited resources
   Error SCHD-30: ... loop cannot be scheduled at the requested II

Not a resource problem: the default ``-IO_MODE fixed`` makes multiple handshakes in one loop body
collide. Add ``directive set -IO_MODE super`` and ``directive set -SPECULATE true`` (see
`Recommended Scheduling Directives`_, including their cost).

*HIER-23: possible deadlock (warning)*

Usually a false positive for valid ``try_put`` / ``try_get`` designs. Synthesis completes and the
RTL is correct.

Conclusion
----------
The Catapult HLS backend provides an alternative synthesis path for Allo designs, leveraging Siemens' industry-leading HLS technology. It supports both functional simulation (csim) and RTL synthesis (csyn), making it suitable for designs targeting both ASIC and FPGA implementations.

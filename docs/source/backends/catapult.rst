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
``dev/toolchains.rst``).

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
   export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home   # on zhang-21; the /Mgc_home component is part of the root
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
``67559cad`` on ``choonsik1/allo:SystemC-emitter``; ``git log --all -S"MGLS_LICENSE_FILE"`` finds them).

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

Provenance: measured on the ``choonsik1/allo:SystemC-emitter`` fork, on the **SystemC/Connections**
flow; ``dev/systemc/noc/FINDINGS_wire_channel.md`` §7 carries the numbers. The directives are
solution-level and apply to any Catapult run; the 2/32 -> 22/32 figure is specific to that
Connections corpus and has not been re-measured for ``ac_channel`` designs.

.. note::

   That emitter is now **in this repository**: ``allo/backend/catapult.py`` emits both lines,
   but only under ``platform == "systemc"``. A ``target="catapult"`` run still gets the default
   ``fixed`` -- the directives are attached to the SystemC/Connections flow, not to Catapult in
   general. (Before the ``SystemC-emitter`` merge this repository emitted neither, which is what
   the note here said when it was verified 2026-09-18.)

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
per-module results for that design are in ``dev/records/catapult_decoupled_2x1.rst``.

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
``mode="ppa"`` runs the ``csyn`` flow and then the two steps that produce power --
``go switching`` (simulate the C++ testbench against the pre-power RTL, convert the VCD to SAIF)
and ``flow run /PowerAnalysis/report_pre_pwropt_Verilog`` (annotate that SAIF, write
``power.rpt``) -- and prints latency, area and **measured** power. It needs a C++ testbench that
SCVerify can drive and an Xcelium install; with no testbench there is no activity and therefore no
power, and the build fails at ``build()`` rather than reporting zeros.

Quick start
~~~~~~~~~~~

.. code-block:: bash

   export MGC_HOME=/opt/siemens/catapult/2024.2/Mgc_home
   export PATH=$MGC_HOME/bin:$PATH
   export MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu   # Catapult + PowerPro
   export CDS_LIC_FILE=...                                          # Xcelium, for the switching sim

.. important::

   **Expect to pass** ``configs={"ncsim_root": ...}``. The switching simulation runs in Xcelium,
   and the backend looks for it in ``configs["ncsim_root"]``, then ``$NC_ROOT`` / ``$XCELIUM_HOME``
   / ``$CDS_INST_DIR``, then ``xrun`` on ``PATH``. On zhang-21 **none of those are set**, so the
   explicit config is what works there; on a host configured like it, a run without one fails
   immediately at ``build()``. Nothing is guessed and no path is compiled in: rather than emit a
   tcl with an empty ``/NCSim/NC_ROOT`` -- which would run no simulation and report zero power --
   the build raises.

.. code-block:: python

   s = allo.customize(top)
   mod = s.build(
       target="catapult",
       mode="ppa",
       project="my_ppa_project",
       configs={
           "testbench": "tb.cpp",       # REQUIRED: CCS_MAIN + CCS_DESIGN(top)(...)
           "ncsim_root": "/opt/cadence/XCELIUM2403",   # Xcelium root; zhang-21's, as an example
           "clock_period": 5.0,         # ns
           "library": "nangate-45nm_beh",
       },
   )
   stats = mod()   # runs synthesis + switching + power, prints a table, returns a dict

56 s for the 16-tap MAC below on zhang-21. The table it prints carries latency, area, total /
dynamic / static power, the SAIF annotation coverage, a per-instance power breakdown, and the
caveat below.

The testbench
~~~~~~~~~~~~~
**A design with no C++ testbench gets no power number.** ``go switching`` measures activity by
simulating; with nothing driving the design the report is zeros or a default-toggle guess -- a
power figure that is wrong in the safe-looking direction. ``mode="ppa"`` therefore refuses to run
without ``configs["testbench"]``.

The file is ordinary Catapult SCVerify C++: ``CCS_MAIN(...)`` calling ``CCS_DESIGN(<top>)(...)``
over representative stimulus. The worked example is
``dev/records/catapult_handoff/zhang21_power_2026-09-24/mac_tb.cpp`` (200 transactions):

.. code-block:: cpp

   #include <ac_int.h>
   #include <mc_scverify.h>
   void mac(ac_int<8,true> a[16], ac_int<8,true> b[16], ac_int<24,true> &out);
   CCS_MAIN(int argc, char **argv) {
     ac_int<8,true> a[16], b[16]; ac_int<24,true> o; int errs = 0;
     for (int t = 0; t < 200; ++t) { /* stimulus */ CCS_DESIGN(mac)(a, b, o); }
     CCS_RETURN(errs != 0);
   }

The SystemC flow's emitted ``kernel.cpp`` does **not** count: its testbench is an ``sc_main``,
which SCVerify does not drive. A SystemC design needs a separate SCVerify testbench for ppa; use
``mode="csyn"`` for area and latency only.

**A** ``@df.region()`` **is drivable the same way.** A region reaches this backend as a *void*
function whose parameters are the region's arrays, marked ``#pragma hls_design top`` with a
``#pragma hls_design dataflow`` body -- the same shape SCVerify wrapped for ``mac16``, only wider.
So the testbench declares the top itself, with **C++ linkage** (the Catapult emitter writes no
``extern "C"``, unlike the Vitis emitter, whose testbenches do declare the top ``extern "C"``),
prefills its output array, calls ``CCS_DESIGN(<region>)`` and checks the result. Neither
``host.cpp`` nor ``kernel.h`` is involved in ``mode="ppa"``, so the csim-only defect noted under
`Limits and known failures`_ is not on this path. The worked example is
``dev/records/catapult_handoff/ppa_tinytpu/``, generated by
``examples/tinytpu/ppa_catapult.py``.

Two things that generator learned, both worth copying:

* **Do not write** ``static alignas(64) T x[N]``. ``cosim.py``'s testbench generator does, to keep
  the ``align_value(64)`` promise the *Vitis* ``m_axi`` path makes; that promise has no Catapult
  counterpart, and the form is ill-formed C++ (a standard attribute in the middle of the
  decl-specifiers) which g++ rejects under ``-std=c++11`` -- the standard the emitted tcl sets.
  Vitis only accepts it because it compiles testbenches with ``-std=gnu++0x``.
* **Compile the testbench before handing it over**, against two-line stand-ins for ``ac_int.h``
  and ``mc_scverify.h`` (``CCS_MAIN`` -> ``main``, ``CCS_DESIGN(x)`` -> ``x``,
  ``CCS_RETURN(x)`` -> ``return (x)``) and a stand-in design that replays a reference dump. That
  costs nothing on a host with no Catapult and turns "syntax error after a licence checkout"
  into a local failure. ``ppa_catapult.py`` does this on every regeneration, and also runs the
  doctored cases that prove its ``errors=`` counter can go nonzero.

Reference
~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``configs["testbench"]``
     - **Required.** Path to the SCVerify C++ testbench. Copied next to ``kernel.cpp`` and added
       with ``-exclude true`` (compiled for verification, never synthesized).
   * - ``configs["ncsim_root"]``
     - Xcelium install root (the directory holding ``bin/xrun``). Falls back to ``$NC_ROOT``,
       ``$XCELIUM_HOME``, ``$CDS_INST_DIR``, then ``xrun`` on ``PATH``. No default path is
       compiled in; the build fails if none resolves.
   * - ``configs["switching_activity_type"]``
     - ``saif`` by default. Catapult's own default, FSDB, needs Verdi (``$NOVAS_INST_DIR``).
   * - ``configs["use_ccs_block"]``
     - ``False``. Set it only when your own sources mark the DUT with the ``CCS_BLOCK()`` macro;
       Allo emits ``#pragma hls_design top``, which SCVerify handles without it.
   * - ``CDS_LIC_FILE``
     - Environment, not configuration: Xcelium's licence. Unset produces a warning, not an error
       (a site may keep the licence in the default ``~/cds.lic``).
   * - PowerPro licence
     - Nothing to acquire. ``PProBase``, ``PProAnalysis``, ``PProCGopt``, ``PProWriteRTL`` and
       ``PProPAWorker`` check out from the same server as Catapult itself with nothing extra set.

The simulator is Xcelium, not Questa. Catapult defaults to QuestaSim, and pointing it at an
install without ``vsim`` is how a power step "succeeds" having simulated nothing -- the emitted tcl
therefore sets ``/SCVerify/USE_QUESTASIM false``, ``/SCVerify/USE_NCSIM true`` and
``/NCSim/NC_ROOT <resolved root>``. Host-specific paths (that root, the licence files) are resolved
from configuration and environment at ``build()``, never hardcoded, so the tcl is specific to the
host that generated it while the backend is not.

``stats`` keys: ``Latency (cycles)`` (from ``cycle.rpt``), ``Area``
(``TOTAL AREA (After Assignment)`` from ``rtl.rpt`` -- this flow writes no ``area.rpt``), ``Power``,
``Power dynamic``, ``Power static``, ``Power clock net`` (all µW, from ``power.rpt``),
``SAIF annotated``, ``hierarchical`` (per-instance power) and ``caveat``. All reports are under
``<project>/Catapult/<top>.v1/``.

How to read the power number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Every figure this mode reports carries, and needs, this caveat: the activity is **SAIF from
simulation**, with 100 % of primary inputs and flop outputs annotated and activity propagated to
internal nets by PowerPro (not simulated). It is therefore **workload-specific -- only as
representative as the testbench** -- and **pre-layout**: a wireload model, PowerPro's clock-tree
model, no place-and-route parasitics. A power figure without that attached is not usable evidence.

Limits and known failures
~~~~~~~~~~~~~~~~~~~~~~~~~
* No testbench, no power: ``build()`` raises. This is the intended behaviour.
* No Xcelium root resolvable: ``build()`` raises rather than emitting a tcl with an empty
  ``NC_ROOT`` that silently runs no simulation. Discovery from the environment finds nothing on a
  host like zhang-21, where none of the variables are set -- pass ``ncsim_root``.
* Zero dynamic power, a missing ``power.rpt``, or 0 % SAIF annotation after a run: ``mod()`` raises
  instead of printing a zero.
* ``synth_top`` (synthesizing a submodule) is untested with ppa: SCVerify wraps the design top, so
  a testbench must drive that submodule directly. The one attempt on record --
  ``stream_boundary``'s ``compute_0`` -- produced no activity.
* Technology library selection is by ``configs={"library": ...}`` (``device`` is still accepted);
  see `Configuration Options`_. ``codegen_tcl`` defaults ``frequency`` to **100** MHz when neither
  ``clock_period`` nor ``frequency`` is given.
* **A** ``@df.region()`` **cannot be csim'd through this backend**, and this is not fixed. The
  host generator derives its outputs from the top function's *results* (``func.type.results``),
  and a region top is ``void``, so every argument is treated as an input: ``host.cpp`` writes no
  ``output<i>.data``, ``mod(*args)`` reads none back, and the caller's arrays come back unchanged
  with no error. The ``systemc`` platform already does the right thing -- it splits arguments by
  direction with ``analyze_arg_load_store`` -- so the fix is to adopt that split here, in
  ``codegen_host`` and in the read-back in ``HLSModule.__call__``. It is deliberately not done
  blind: it cannot be exercised without a Catapult install, and a change that *looks* right while
  silently reading the wrong file is the failure mode this backend has already had once.
  ``mode="ppa"`` is unaffected -- its testbench is the caller's own and does its own checking.

Results and history
~~~~~~~~~~~~~~~~~~~
**The worked example** is ``dev/records/catapult_handoff/ppa_mac16/`` with its result in
``ppa_mac16/zhang21_run_2026-09-24/``: the first ``mode="ppa"`` run to go end to end through Allo's
own emitted tcl, **unmodified**, on zhang-21 (Catapult 2024.2/1130128, Xcelium 24.03-s005, exit 0
in 56 s). An Allo-emitted 16-tap int8 MAC driven by a 200-transaction SCVerify testbench:
**272.69 µW = 252.19 dynamic + 20.50 static**, 100.00 % annotation on both flop outputs and user
nets, ``MAC16 TB errors=0`` and SCVerify ``Simulation PASSED``; latency 16 / throughput 18, area
1180.529 score units, ``mac16_core_inst`` 272.09 µW with its FSM at 4.07 µW. ``use_ccs_block`` was
**not** needed -- SCVerify wrapped the ``#pragma hls_design top`` DUT directly. That directory also
holds ``check_ppa.py``, the success criterion written before the run, which was then shown to go
red on three doctored copies of the passing outputs (testbench errors, a zeroed dynamic row, 0 %
annotation).

**The first real design** is ``dev/records/catapult_handoff/ppa_tinytpu/``: TinyTPU-isa at
``TPU_T=4 TPU_MAXDIM=16`` -- ``reproduce.sh``'s pinned build -- with the same ``nangate-45nm_beh``
library and 5 ns clock as ``ppa_mac16``, so the two differ in the design and not in the technology.
Its testbench makes four ``CCS_DESIGN(tinytpu_isa)`` calls on one instance: three 16x16x16 GEMMs
(one with ReLU) and one ``isa_dsl.vector_program(8)``, all with **uniform full-range int8**
operands rather than the ``[-4, 4]`` the cycle benchmarks use -- that distribution exists to match
Gemmini's ``allo_cmp.c`` for a like-for-like *cycle* comparison, and it would hold the top five
bits of every operand constant, biasing power low. Each call prefills the whole result array and
compares all of it, counting cells the program should write separately from cells it should not
touch; ``errors=0`` is the criterion and the clobber count is informative. **It has not been run:**
ace-01 has no Catapult, so what exists is the handoff -- project, testbench, tcl and a
``check_ppa.py`` written before the run and shown to go red on eight doctored copies of the
``ppa_mac16`` outputs. What the resulting number will and will not represent is set out in that
directory's ``RUNME.md``; the short version is one shape, one operand distribution, at
``MAXDIM=16``, pre-layout. Regenerating at the shipped ``MAXDIM=64`` is one environment variable
and changes only array extents, not structure.

The earlier measured run, by hand rather than through this mode, is ``dev/records/catapult_handoff/zhang21_power_2026-09-24/`` (zhang-21,
Catapult 2024.2/1130128 with PowerPro, nangate45 ``typical`` 1.1 V, 5 ns clock): a 16-tap int8 MAC
driven by 200 transactions, **248.47 µW total -- 229.67 dynamic, 18.80 static**. The control with
every input zero gives 83.80 µW dynamic, with combinational power falling 147.96 → 15.03 µW while
the clock network stays at 19.93 µW. The number tracks activity, which is what makes it a
measurement rather than a default-toggle estimate.

That one had to be run by hand, because ``mode="ppa"`` did not work: it emitted the ``csyn`` tcl
(stopping at ``go extract``, so no power step ever ran), and its parser looked for a
``Total Power:`` string that PowerPro's report does not contain, so it would have printed ``N/A``
even for a successful run. Both are fixed, the parser is tested against that committed report, and
the ``ppa_mac16`` run above is the confirmation on the licence host. The two figures differ
(272.69 vs 248.47 µW) because the Allo-emitted top has a different port interface and netlist from
the hand-written ``mac.cpp``; the stimulus is the same.

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
for the host (see :doc:`/developer/pitfalls` and ``dev/toolchains.rst``), and build the
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
  ``dev/records/catapult_decoupled_2x1.rst``. It emits ``ac_channel`` C++.
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

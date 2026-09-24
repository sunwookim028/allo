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

#################################
SystemC emitter (Catapult, ASIC)
#################################

``target="systemc"`` turns an Allo ``@df.region`` into synthesizable **SystemC**: one
``SC_MODULE`` per ``@df.kernel``, MatchLib ``Connections`` for the links between them, and a
self-contained ``sc_main`` testbench. Siemens Catapult then synthesizes that SystemC to
Verilog, and Cadence Xcelium re-runs the emitted testbench against the synthesized RTL for a
bit-exact check. Emission needs nothing but Allo; everything past it needs Catapult
(``MGC_HOME``) and, for cosim, Xcelium.

This backend came from the ``choonsik1/allo`` fork's ``SystemC-emitter`` branch and was merged
with its history, so ``git log`` on ``mlir/lib/Translation/EmitSystemC.cpp`` attributes it
correctly.

Quick start
-----------

Emission runs anywhere the ``allo`` env runs:

.. code-block:: bash

   # the env sets neither of these
   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

   cd examples/systemc
   python pc_channel.py systemc      # print the generated SystemC
   python pc_channel.py mlir         # the MLIR it was emitted from

In Python the whole surface is one call:

.. code-block:: python

   import allo.dataflow as df
   from allo.ir.types import int32, Stream

   @df.region()
   def top(A: int32[8], B: int32[8]):
       fifo: Stream[int32, 4][1]

       @df.kernel(mapping=[1], args=[A])
       def producer(a: int32[8]):
           for i in range(8):
               fifo[0].put(a[i])

       @df.kernel(mapping=[1], args=[B])
       def consumer(b: int32[8]):
           for i in range(8):
               b[i] = fifo[0].get() + 1

   print(df.build(top, target="systemc").hls_code)          # emit only
   mod = df.build(top, target="systemc", mode="csim",       # compile + run
                  project="pc_project")                     #   (needs Catapult)

Emitting a design of this size takes under a second. ``csim`` compiles the emitted
``kernel.cpp`` with ``g++`` against Catapult's bundled ``libsystemc`` and runs it; ``csyn``
and ``cosim`` invoke Catapult.

The larger worked example is **EVA** (``examples/eva/``), a mesh of fused router + PE nodes
emitted at 1x1 as 22 ``SC_MODULE``\ s and 32 FIFOs. ``examples/eva/README.md`` has its
environment and the two commands.

How it works
------------

Three emitters form a subclass chain, each overriding only what differs:

.. code-block:: text

   VhlsModuleEmitter        ap_int, #pragma HLS ...          mlir/lib/Translation/EmitVivadoHLS.cpp
     └── CatapultModuleEmitter   ac_int, #pragma hls_...      mlir/lib/Translation/EmitCatapultHLS.cpp
           └── SystemCModuleEmitter   SC_MODULE + Connections mlir/lib/Translation/EmitSystemC.cpp

So arithmetic, control flow and most of a kernel body are the Vitis emitter's; Catapult swaps
the vendor types and pragmas; SystemC replaces the *structure*. Emission proceeds in four
stages: the region hierarchy is flattened (an ``SC_THREAD`` cannot structurally instantiate a
sub-region), each kernel becomes an ``SC_MODULE`` with a clocked ``SC_THREAD run()``, the
region top becomes a wiring module that declares one channel per link and binds every port,
and finally the device header and testbench are emitted.

The three modes exercise **the same emitted source** three ways:

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - mode
     - what runs
     - what it proves
   * - ``csim``
     - ``g++`` compiles and runs the SystemC on the host
     - functional correctness, in seconds
   * - ``csyn``
     - Catapult synthesizes the SystemC to Verilog
     - that it synthesizes; area and Fmax
   * - ``cosim``
     - the synthesized RTL runs in Xcelium against the csim golden
     - the RTL is bit-exact with the C simulation

csim and synthesis do not compile the same code -- the emitter uses several
``#ifdef __SYNTHESIS__`` splits, the largest being that a kernel's outermost repeat loop
becomes ``while (1)`` under synthesis so Catapult pipelines it rather than treating the body
as reset-action setup. **A green csim therefore does not prove the RTL is right.** Run
``cosim``.

Reference
---------

Link primitives
~~~~~~~~~~~~~~~

The link type is a design decision, not a label: each maps to different RTL. ``Wire`` and
``Channel`` are **SystemC-only** -- every other target raises ``NotImplementedError``.

.. list-table::
   :header-rows: 1
   :widths: 26 30 22 22

   * - Allo link
     - RTL
     - handshake
     - buffered
   * - ``Wire[T]``
     - ``sc_signal<T>``
     - none
     - no
   * - ``Channel[T, valid_ready]``
     - ``Connections::Combinational<T>``
     - full valid/ready
     - no
   * - ``Channel[T, valid_only]``
     - ``_dat`` + ``_vld`` signal pair
     - valid only
     - no
   * - ``Stream[T, depth]``
     - ``Connections::Fifo`` (``AlloFifoC`` when ``empty()``/``full()`` are queried)
     - credit/handshake
     - yes

A ``Wire`` has zero storage **and** zero alignment: it reads garbage unless the two kernels are
cycle-locked. ``valid_only`` drops a datum if the consumer is not looking that cycle. Both are
deliberate performance points -- take them only when you own the timing.

Boundary arrays are not all alike either. A 1-D array scanned strictly in order becomes a
stream port (``a[i]`` lowers to ``.Pop()``); anything 2-D, strided, random or re-read becomes an
addressable memory port behind a request/response channel.

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - key / variable
     - meaning
   * - ``mode``
     - ``csim`` | ``csyn`` | ``cosim`` (see above); ``ppa`` is the Catapult backend's
   * - ``configs["library"]``
     - the Catapult standard-cell library, e.g. ``nangate-45nm_beh``. **Canonical** -- the
       older ``device`` key means an FPGA part to ``vitis_hls`` and a cell library here, and
       passing an FPGA part now raises rather than failing inside Catapult
   * - ``configs["clock_period"]``
     - clock period in ns. Takes precedence over ``frequency`` (MHz). Catapult bakes the
       period into the schedule, so this is the main area/timing knob
   * - ``configs["synth_top"]``
     - synthesize one ``<kernel>_0`` submodule instead of the whole region. **Any area number
       you report needs this**: a region contains its testbench kernels and their memories
   * - ``MGC_HOME``
     - Catapult install. Emission does not need it; ``csim``/``csyn``/``cosim`` do
   * - ``SYSTEMC_HOME``
     - SystemC headers and ``libsystemc`` for the standalone ``csim`` compile
   * - ``ALLO_SYNC_RESET``
     - synchronous instead of asynchronous reset (smaller flops on the ASIC path)

Where things live
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - path
     - what
   * - ``mlir/lib/Translation/EmitSystemC.cpp``
     - the emitter; ``EmitSystemC.md`` beside it is its internal documentation
   * - ``allo/backend/hls.py``
     - the ``systemc`` platform: modes, the csim runner, the ``Wire``/``Channel`` guard
   * - ``allo/backend/catapult.py``
     - the TCL generator both Catapult flows share
   * - ``examples/systemc/``
     - runnable examples, ``VERDICTS.md``, and ``reports/`` for what was measured
   * - ``examples/eva/``
     - the EVA design and its emitted project
   * - ``examples/systemc_rtlsim/``
     - this fork's SystemC-vs-RTL cross-check harness (:ref:`limitation-22`)
   * - ``tests/dataflow/test_systemc_backend.py``
     - the suite; emit cases run anywhere, csim/cosim cases skip without ``MGC_HOME``
   * - ``dev/systemc/``
     - the emitter author's working notes, kept whole and not rewritten

Limits and known failures
-------------------------

- **Neither Catapult nor Xcelium is installed on this fork's development host**
  (``dev/toolchains.rst``). Everything past emission is therefore unverified here: the
  emit-only tests pass, and every ``csim``/``csyn``/``cosim`` case skips. The evidence that
  they pass elsewhere is in ``examples/systemc/reports/`` and ``tests/dataflow/COSIM_REGRESSION.md``.
- **A ``Wire`` is not a cheap ``Channel``.** With no storage and no alignment it reads garbage
  unless the producer and consumer are cycle-locked; ``examples/systemc_rtlsim/`` reproduces
  that on two simulators, and :ref:`limitation-22` records it.
- **Bit-slicing a packed value wider than 64 bits** works in csim and can fail at ``csyn``:
  Catapult rejects subclassing its builtin ``ac_int`` (CIN-15), and the emitted ``ap_int`` shim
  is such a subclass.
- **``csyn`` needs a build subdirectory.** Running Catapult in the directory holding
  ``kernel.cpp`` degrades ``Connections::In``/``Out`` ports to raw ``sc_signal``\ s (CIN-124,
  SCHD-30). ``examples/systemc/csyn_subdir.py`` works around it.
- **``cosim`` does not work against a ``synth_top`` submodule.** SCVerify wraps the design top,
  and the stimulus path into the region's memories disappears. Measure and verify separately.
- **The timed dataflow simulator is not here.** ``SystemC-emitter`` also carries a per-PE
  simulated-clock simulator with a ``get_cycles()`` read-out; it was not merged, because it
  rewrites the same code this fork rewrote for OpenMP team sizing. ``dev/systemc/SIMULATOR.md``
  describes it.

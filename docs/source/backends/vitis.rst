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
AMD Vitis HLS (FPGA)
############################

The `Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ FPGA backend leverages the Vivado/Vitis HLS toolchain to generate hardware accelerators for FPGA devices. This document demonstrates how to define a general matrix multiplication (GEMM) kernel using the Allo ADL and generate HLS code for FPGA synthesis. For more details on kernel customizations and scheduling optimizations, please refer to the `Allo-HLS tutorial <https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html>`_.

Kernel Definition
-----------------
The GEMM kernel is implemented with `float32` precision using pre-defined constants for the matrix dimensions. The kernel utilizes the `allo.grid` API to iterate over output indices and the `allo.reduction` API to designate the reduction axis for accumulating the dot-product.

.. code-block:: python

   import allo
   from allo.ir.types import float32
   import numpy as np

   # Define matrix dimensions
   M, N, K = 32, 32, 32

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

Code Generation for Vivado/Vitis HLS
------------------------------------
Allo supports several approaches to generate HLS code:

1. **Direct HLS Code Generation**:  
   Set the target to `"vhls"` to produce HLS code as a string. This code includes the necessary headers and pragmas for Vitis HLS synthesis.

    .. code-block:: python

       code = s.build(target="vhls")
       print(code)

2. **HLS Emulation, Synthesis, and Execution**:  
   Specify the target as `"vitis_hls"` along with a synthesis mode and project name to generate a complete HLS project. The supported modes are:

   - ``sw_emu``: Software emulation mode, which is similar to C simulation that compiles the program using C compiler and runs it on the CPU. Depending on the size of your input data, this mode may take within one minute.
   - ``hw_emu``: Hardware emulation mode, which is similar to co-simulation that compiles the program into RTL design using HLS compiler and runs the RTL with the test bench on the FPGA emulator. Since it needs to go through the HLS synthesis flow, it may take several minutes to finish.
   - ``hw``: Full hardware synthesis mode, which compiles the program into RTL design using HLS, goes through placement and routing, generates the bitstream, and finally executes on FPGA. This mode may take several hours to finish.

    .. code-block:: python

       mod = s.build(target="vitis_hls", mode="hw_emu", project="gemm.prj")

3. **(Legacy) HLS Synthesis**:  
   Set the target to `"vivado_hls"` to generate a legacy HLS project. This approach is similar to the previous one but uses ``run.tcl`` for the project script. The supported modes are:

   - ``csim``: C simulation mode, using the gcc compiler to compile the program and runs it on the CPU.
   - ``csyn``: C synthesis mode, using Vivado HLS compiler to synthesize the program. **Note: This mode only synthesize the program and generate the RTL design but does not execute the program!**
   - ``cosim``: Co-simulation mode, using Vivado HLS compiler to synthesize the program and generate the RTL design, then runs the RTL with the test bench on the CPU.
   - ``impl``: Implementation mode, using Vivado HLS compiler to synthesize the program, generate the RTL design, and go through placement and routing to the bitstream.

   .. code-block:: python

      mod = s.build(target="vivado_hls", mode="csim", project="gemm.prj")
      # For csim
      mod(np_A, np_B, allo_C)
      # For csyn
      mod()


Project Structure and Execution
-------------------------------
The generated HLS project (e.g., in the folder ``gemm.prj``) typically includes:

- **host.cpp**: The host-side (CPU) code that invokes the accelerator.
- **kernel.cpp**: The accelerator kernel code.
- **Makefile**: Build scripts to compile the project.

To run the design, prepare the input matrices using NumPy and allocate an output array for the result:

.. code-block:: python

   np_A = np.random.random((M, K)).astype(np.float32)
   np_B = np.random.random((K, N)).astype(np.float32)
   allo_C = np.zeros((M, N), dtype=np.float32)
   mod(np_A, np_B, allo_C)
   np.testing.assert_allclose(allo_C, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)

Note:
  Ensure that the Vitis HLS and XRT environments are correctly configured before running the HLS flow. For further environment setup and synthesis mode details, please consult the `Vitis HLS <https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html>`_ documentation.


HBM/DDR Memory Mapping
----------------------
For designs targeting platforms with High Bandwidth Memory (HBM) or multiple DDR banks (such as Alveo U280), Allo provides an easy way to specify memory channel mappings for kernel arguments. This is done through the ``hbm_mapping`` configuration option in the ``configs`` dictionary.

**Basic Usage**

The ``hbm_mapping`` dictionary maps function argument names to memory channels. You can specify:

- **Integer values**: Interpreted as HBM channel numbers (e.g., ``0`` → ``HBM[0]``)
- **String values**: Full memory specification (e.g., ``"HBM[0]"``, ``"DDR[1]"``)

.. code-block:: python

   from allo.ir.types import int32

   def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
       C: int32[32, 32] = 0
       for i, j, k in allo.grid(32, 32, 32):
           C[i, j] += A[i, k] * B[k, j]
       return C

   s = allo.customize(gemm)

   # Define HBM channel mapping using argument names
   hbm_mapping = {
       "A": 0,              # Input A -> HBM channel 0
       "B": "HBM[1]",       # Input B -> HBM channel 1
       "output_0": "DDR[0]", # Return value -> DDR bank 0
   }

   mod = s.build(
       target="vitis_hls",
       mode="hw",
       project="gemm.prj",
       configs={"hbm_mapping": hbm_mapping},
   )

**Argument Naming Convention**

- **Input arguments**: Use the same names as in your function definition (e.g., ``"A"``, ``"B"``)
- **Return values**: Use ``"output_0"``, ``"output_1"``, etc. for functions with return values

**Generated Configuration File**

Allo automatically generates a ``.cfg`` file in the project directory with the connectivity settings. For example, the above configuration would generate a file like:

.. code-block:: text

   [connectivity]

   sp=gemm.v15:HBM[0]
   sp=gemm.v16:HBM[1]
   sp=gemm.v17:DDR[0]

Note that Allo automatically translates your user-friendly argument names (``A``, ``B``, ``output_0``) to the actual HLS-generated argument names (``v15``, ``v16``, ``v17``).

**Complex Example with Multiple Arguments**

For larger designs with many memory interfaces, you can specify different HBM channels for each argument:

.. code-block:: python

   hbm_mapping = {
       "inp_0": 0,           # HBM[0]
       "inp_1": 0,           # HBM[0] - can share channels
       "weight_0": 1,        # HBM[1]
       "weight_1": 2,        # HBM[2]
       "bias": "HBM[3]",     # HBM[3]
       "output_0": "DDR[0]", # DDR bank 0
   }

This feature is particularly useful for:

- **Memory bandwidth optimization**: Distribute data across multiple HBM channels to maximize bandwidth
- **Bank conflict avoidance**: Place frequently accessed data on separate memory banks
- **Platform-specific tuning**: Match memory assignments to your target FPGA platform's memory architecture


Memory Implementation Customization
------------------------------------
Allo provides the ``Memory`` class to specify on-chip memory implementation details for arrays, similar to the `Vitis HLS bind_storage pragma <https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-bind_storage>`_. This allows fine-grained control over how arrays are mapped to FPGA memory resources (BRAM, URAM, LUTRAM, etc.).

**Basic Usage**

Use the ``@`` operator to annotate function arguments or local variables with memory specifications:

.. code-block:: python

   from allo import Memory
   from allo.ir.types import int32, float32

   # Define memory specifications
   MemUram = Memory(resource="URAM")
   MemBram = Memory(resource="BRAM", storage_type="RAM_2P")

   def kernel(a: int32[32] @ MemUram, b: float32[16, 16] @ MemBram) -> int32[32]:
       # Local variable with memory annotation
       buf: int32[32] @ Memory(resource="BRAM")
       for i in range(32):
           buf[i] = a[i] * 2
       c: int32[32]
       for i in range(32):
           c[i] = buf[i] + 1
       return c

   s = allo.customize(kernel)
   mod = s.build(target="vhls")
   print(mod.hls_code)

This generates HLS code with ``bind_storage`` pragmas:

.. code-block:: cpp

   void kernel(int32_t v0[32], float v1[16][16], int32_t v2[32]) {
     #pragma HLS bind_storage variable=v0 impl=uram
     #pragma HLS bind_storage variable=v1 type=ram_2p impl=bram

     int32_t buf[32];
     #pragma HLS bind_storage variable=buf impl=bram
     // ... kernel body ...
   }

**Memory Class Parameters**

The ``Memory`` class accepts the following parameters:

- **resource** (str): Memory resource type

  - ``"BRAM"``: Block RAM - the most common on-chip memory
  - ``"URAM"``: Ultra RAM - larger capacity, available on UltraScale+ devices
  - ``"LUTRAM"``: LUT-based RAM - faster but smaller
  - ``"SRL"``: Shift Register LUT - efficient for FIFOs
  - ``"AUTO"``: Let the HLS tool decide (default)

- **storage_type** (str, optional): RAM access pattern

  - ``"RAM_1P"``: Single-port RAM
  - ``"RAM_2P"``: Simple dual-port RAM (one read, one write port)
  - ``"RAM_T2P"``: True dual-port RAM (two read/write ports)
  - ``"RAM_1WNR"``: Single write, N read ports
  - ``"RAM_S2P"``: Simple dual-port (alias for RAM_2P)
  - ``"ROM_1P"``: Single-port ROM
  - ``"ROM_2P"``: Dual-port ROM
  - ``"ROM_NP"``: N-port ROM

- **latency** (int, optional): Memory access latency in cycles
- **depth** (int, optional): Depth of the memory (useful for streams/FIFOs)

**Examples**

1. **URAM for large buffers**:

   .. code-block:: python

      # URAM is ideal for large arrays on UltraScale+ FPGAs
      LargeBuffer = Memory(resource="URAM")

      def process(data: float32[1024, 1024] @ LargeBuffer):
          ...

2. **Dual-port BRAM for concurrent access**:

   .. code-block:: python

      # RAM_2P allows simultaneous read and write
      DualPort = Memory(resource="BRAM", storage_type="RAM_2P")

      def pipeline(inp: int32[256] @ DualPort) -> int32[256]:
          ...

3. **LUTRAM for small, fast buffers**:

   .. code-block:: python

      # LUTRAM is faster but uses more LUTs
      FastBuffer = Memory(resource="LUTRAM")

      def compute(weights: int8[64] @ FastBuffer):
          ...

4. **Multiple memory types in one kernel**:

   .. code-block:: python

      InputMem = Memory(resource="BRAM", storage_type="RAM_1P")
      WeightMem = Memory(resource="URAM")
      OutputMem = Memory(resource="BRAM", storage_type="RAM_2P")

      def neural_layer(
          inp: float32[128] @ InputMem,
          weights: float32[128, 64] @ WeightMem,
          out: float32[64] @ OutputMem
      ):
          ...

**Best Practices**

- Use **URAM** for large arrays (>36Kb) on UltraScale+ devices to save BRAM resources
- Use **BRAM with RAM_2P** when you need concurrent read/write access
- Use **LUTRAM** for small lookup tables that require low latency
- Use **ROM** types for constant data that never changes
- Let the tool decide (``resource="AUTO"``) when you don't have specific requirements


Device and Frequency Configuration
----------------------------------
You can specify the target device and clock frequency through the ``configs`` dictionary:

.. code-block:: python

   mod = s.build(
       target="vitis_hls",
       mode="hw",
       project="gemm.prj",
       configs={
           "device": "u280",     # Target device (default: "u280")
           "frequency": 300,     # Target frequency in MHz (default: 300)
       },
   )

**Supported Devices**

- **Alveo**: ``u200``, ``u250``, ``u280``
- **Zynq UltraScale+**: ``zcu102``, ``zcu104``, ``zcu106``, ``zcu111``
- **Versal**: ``vck190``, ``vhk158``
- **Embedded**: ``ultra96v2``, ``pynqz2``, ``zedboard``


Aligned ``m_axi`` Pointers and Port Widening
--------------------------------------------
Vitis HLS can widen an ``m_axi`` port so that a long burst of narrow elements
(e.g. ``int8``) moves many bytes per beat instead of one. The Tcl setting is
``config_interface -m_axi_max_widen_bitwidth <bits>``, but Vitis only widens a
port whose pointer it knows to be aligned. By default Allo emits the top
function's array arguments as plain ``T *name`` pointers, so Vitis assumes
1-byte alignment and **declines to widen, silently**: the setting is accepted,
csynth completes, and the ports stay at the element width with no diagnostic.

Allo can emit an alignment promise on every array argument of the top function.
It is opt-in, through the ``align_value`` key of ``configs``:

.. code-block:: python

   mod = s.build(
       target="vitis_hls",
       mode="csyn",
       project="design.prj",
       configs={"align_value": 64},   # bytes
   )

With it, each array parameter of the top function is emitted as

.. code-block:: cpp

   int8_t *__attribute__((align_value(64))) v0,

next to the usual ``#pragma HLS interface m_axi port=v0 offset=slave
bundle=gmem0``. The attribute is added in ``postprocess_hls_code``
(``allo/backend/vitis.py``) when a ``vitis_hls`` project is generated; scalar
arguments are unaffected, and without the key the output is unchanged.

Allo does not add ``-m_axi_max_widen_bitwidth`` itself; put it in your own
synthesis script alongside the alignment:

.. code-block:: tcl

   config_interface -m_axi_max_widen_bitwidth 512

**Measured effect.** On the TinyTPU-isa accelerator (:doc:`/designs/tinytpu_isa`),
``align_value`` 64 plus ``-m_axi_max_widen_bitwidth 512`` took gmem0 to **bit
width 512** and gmem1/gmem2 to 32, and two burst loops from II=4 to II=1;
``csynth.log`` shows one port at 512, two at 32, and zero ``HLS 214-307``
messages. Without the alignment, the same setting left every port at 8 bits.
Together with removing a hidden memset it took the design from 1457 to 919
cycles at 16x16x16 (:ref:`tinytpu-history-prefixes`). The 512-bit gmem0 also
lets the program prefetch move 8 instruction words per cycle, one of the
changes that took the design on to 686 (:ref:`tinytpu-isa-landing`).

.. note::

   ``[HLS 214-307] Could not widen since type i8 size is greater than or equal
   to alignment 1(bytes)`` is what small standalone probes report. On the full
   design it did **not** appear -- the refusal was silent. Probes also showed
   that ``__attribute__((aligned(N)))`` on the element type is the wrong lever
   (Vitis tests the pointer *parameter*), and that ``align_value`` alone is
   necessary but not sufficient to widen every probe kernel. See
   :ref:`limitation-23`.

**It is a promise.** ``align_value`` tells Vitis the pointer *is* N-byte
aligned; the host must keep that promise, which is why it is opt-in rather than
always emitted. XRT buffers are 4 KB aligned, so it holds on hardware for free.
In C/RTL co-simulation the testbench *is* the host, so testbench arrays must be
declared aligned too (``static alignas(64) int8_t A[...]``). A design that wants
wide accesses without the promise can instead declare its operands with a wider
element type (e.g. ``UInt(32)`` words holding four ``int8`` lanes), which reaches
II=1 with no alignment and no widen setting at all.


RTL Co-Simulation of Dataflow Designs
-------------------------------------
``df.build(target="vitis_hls", mode=...)`` handles ``csim`` and ``csyn``; the
other ``vitis_hls`` modes route to the ``XDEVICE`` Makefile flow, and the emitted
``host.cpp`` is an OpenCL/XRT host -- which is not what Vitis ``cosim_design``
wants. ``mode="cosim"`` is not wired into ``df.build`` (:ref:`limitation-16`).
Running C/RTL co-simulation on an Allo dataflow design therefore needs a small
external driver. ``examples/accelerator/tinytpu_vitis/cosim.py`` (~180 lines) is
a working one and shows the four things any such driver has to do:

1. **A plain C++ testbench.** Build the project with ``mode="csyn"`` to get
   ``kernel.cpp``, then write a ``tb.cpp`` whose ``main`` calls the top function
   directly on static arrays and compares against a reference. ``cosim.py``
   generates it from the same inputs and numpy reference the Allo simulator
   run uses, so the vectors cannot drift from the design.
2. **Explicit** ``m_axi`` **depths.** Cosim has to know how much memory to model
   behind each port and fails otherwise (``A depth specification is required
   for MAXI interface port 'gmem0' for cosimulation``). Allo emits the interface
   pragmas without a depth, so the driver patches ``depth=<words>`` into each
   ``#pragma HLS interface m_axi ... bundle=gmemN`` line, in the port order of
   the top function.
3. **Aligned testbench arrays** if the kernel was built with ``align_value``
   (above): ``static alignas(64)``.
4. **A linker Vitis can use on this host** (next section): pass
   ``-ldflags "-B/usr/bin"`` to ``cosim_design``.

The synthesis and co-simulation scripts ``cosim.py`` drives, with the
project's part and clock:

.. code-block:: tcl

   # run.tcl for csynth -- run once
   open_project out.prj -reset
   open_solution -reset solution1 -flow_target vivado
   set_top tinytpu_isa
   add_files kernel.cpp
   add_files -tb tb.cpp -cflags "-std=gnu++0x"
   set_part {xcu280-fsvh2892-2L-e}
   create_clock -period 3.33
   config_interface -m_axi_max_widen_bitwidth 512
   # config_interface -m_axi_latency <n>     (optional, see below)
   csynth_design
   exit

.. code-block:: tcl

   # run.tcl for cosim -- run once per testbench
   open_project out.prj
   open_solution solution1
   set_top tinytpu_isa
   cosim_design -trace_level none -rtl verilog -ldflags "-B/usr/bin"
   exit

Because ``cosim_design`` runs one testbench per invocation, a sweep over
workloads can rewrite ``tb.cpp`` and re-run only the cosim script against the
same project: ``csynth_design`` runs once, so the RTL under test is identical for
every workload. The cycle count is read from
``out.prj/solution1/sim/report/<top>_cosim.rpt``.

**Memory-model knobs.** Cosim's AXI slave answers immediately unless told
otherwise: ``-m_axi_latency`` defaults to 0. ``config_interface -m_axi_latency
<n>`` (at csynth time) makes HLS schedule against an ``n``-cycle read latency,
and ``cosim_design -random_stall`` stalls the top-level interfaces at random.
State which one a number was measured with; on TinyTPU-isa, latency 64 costs 75%
at the smallest shape and 23% at the largest (:doc:`/designs/gemmini_comparison`).
In ``cosim.py`` these are ``TPU_AXI_LATENCY=<n>`` and ``TPU_RANDOM_STALL=1``.

**Why cosim rather than the csynth report.** When loop trip counts are runtime
data -- any instruction-programmable design -- csynth can only report a
worst-case bound derived from the index ranges, and it can be off by orders of
magnitude (91,407 vs 4,133 and 2.259e+08 vs 1,176 on TinyTPU-isa; see
:ref:`tinytpu-isa-csynth-bound`). Cosim is the cycle count. Cosim also compiles
an ``AESL_deadlock_detect_unit`` into the RTL testbench, so it reports
deadlocks.

**C simulation runs dataflow processes in declaration order.** In ``csim`` the
processes of a ``dataflow`` region are called sequentially, in source order, so a
consumer declared before its producer reads an empty stream and aborts with
``ERROR [HLS SIM]: an hls::stream is read while empty``. Kernel declaration order
inside a ``@df.region()`` is therefore load-bearing for ``csim`` (not for RTL),
and designs with bidirectional handshakes cannot pass ``csim`` at all. See
:ref:`limitation-15` and :doc:`/developer/dataflow_semantics`.


Vitis 2023.2 Linker vs. Newer glibc
-----------------------------------
Vitis HLS 2023.2 ships binutils 2.37, which cannot read the glibc of newer Linux
distributions. Both the ``csim`` and ``cosim`` links fail with

.. code-block:: text

   unknown type [0x13] section '.relr.dyn'
   ...
   cannot find libm.so.6

The fix is a compiler-driver flag, not a ``PATH`` override: ``-B/usr/bin`` points
the driver at the system linker (2.42 on the host these results were produced
on) while leaving the rest of the Vitis toolchain in place. Pass it to
``cosim_design -ldflags "-B/usr/bin"`` (and equivalently to any ``csim_design``
link). Any new Vitis flow on such a host needs the equivalent. Host toolchain
details are on :doc:`/developer/toolchains`.


Conclusion
----------
This example illustrates the process of defining a GEMM kernel using the Allo ADL and generating HLS code for FPGA acceleration with the Vitis HLS backend. The approach supports various synthesis modes (sw_emu, hw_emu, hw) to cater to different design and verification needs.

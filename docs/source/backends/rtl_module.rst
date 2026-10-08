.. Copyright Allo authors. All Rights Reserved.
.. SPDX-License-Identifier: Apache-2.0

######################################
RTLModule and ``hls::stream`` IP calls
######################################

.. note::

   **Status:** fork-local, from fork PR #48 (``vincent-yeet:feature/ipmodule-rtlmodule``,
   head ``d83a9887``), merged on the fork's ``pr48-merge`` branch with one follow-up
   commit (Verilator 5 port, flag pass-through, stall knob, this page). Not upstream.
   The PR carried three Markdown notes (``docs/RTL_MODULE.md``,
   ``docs/IP_STREAM_INTEGRATION.md``, ``docs/IP_STREAM_SIM_SHIM.md``); they are the
   three parts below, converted to reST with their content unchanged. Where they say
   "line N" of ``simulator.py`` they mean the file at the PR head ``d83a9887``, not
   this tree. Records: ``dev/records/minitpu/pr48_probe_2026-10-08.md`` (the
   feasibility probe) and ``dev/records/minitpu/pr48_merge_2026-10-08.rst`` (the
   merge and its gates).

Three things land together:

- ``allo.RTLModule`` (:ref:`rtl-module-rtlmodule`): Verilog/SystemVerilog with
  ready/valid (or ``ap_fifo``) stream pins, or an ``ap_memory`` RAM port, called
  from an Allo kernel. Simulated by Verilator through a generated C++ transactor;
  for Vitis it becomes an RTL black box.
- ``IPModule`` with ``hls::stream<T>`` ports (:ref:`rtl-module-stream-ip`): a
  hand-written HLS C++ block whose interface is streams, for the Vitis targets.
- The stream shim (:ref:`rtl-module-sim-shim`): the same stream IPs, and
  ``RTLModule``, run under ``df.build(..., target="simulator")``.

On this fork
============

Verilator
---------

**Version.** The fork pins Verilator **5.052** (conda-forge build, prefix
``/work/shared/users/phd/sk3463/tools/verilator`` on zhang-21; ``dev/toolchains.rst``).
Its generated C++ needs a C++17 compiler; on zhang-21 that is ``gcc-toolset-13``
(g++ 13.3.1): put it first on ``PATH`` and export ``CXX`` to it, since the system
g++ 8.5 is too old and Catapult's environment puts its own g++ 10.3 first.

**Which binary.** ``RTLModule(verilator=...)`` if given, else ``$VERILATOR``, else
``verilator`` on ``PATH``. The pinned prefix is not on ``PATH`` by default, so
either export ``VERILATOR=/work/shared/users/phd/sk3463/tools/verilator/bin/verilator``
or *append* its ``bin`` to ``PATH`` -- prepending it shadows the ``allo`` env's
``python`` with the conda prefix's own. When the binary sits in a conda prefix,
``PERL5LIB`` is pointed at that prefix's Perl modules automatically.

**validate_rtl reads --json-only.** As submitted, the PR elaborated the RTL
with ``verilator --xml-only``; Verilator 5.052 no longer has that option
(``%Error: Invalid option: --xml-only``), and since every simulation path validates
first, nothing simulated on the pinned Verilator (9 test failures). The follow-up
commit reads the ``--json-only`` tree instead: the elaborated top is the ``MODULE``
at ``level`` 1, a pin is a ``VAR`` with a ``direction``, and its width comes from
the ``BASICDTYPE`` ``range``. The checks and their error texts are the same as
before; a pin of any other type (packed/unpacked array, struct, enum) is still
refused with "Packed/unpacked aggregate pin unsupported".

**Flags.** ``RTLModule(..., verilator_args=[...])`` (a list, or one shell-quoted
string) is appended to both Verilator runs -- the ``--json-only`` validation and the
``--cc --build`` model build. For example ``["-Wno-fatal", "--build-jobs", "16"]``.
Without it the build step is exactly the PR's: ``--cc --build --Mdir vgen -CFLAGS
"-fPIC -fvisibility=hidden"``, serial, warnings fatal. On 5.052 immediate
assertions are compiled in by default (there is no ``--assert`` to add;
``--no-assert`` removes them), and ``--timing`` is not needed by RTL without timing
constructs.

**Stall limit.** ``RTLModule(..., max_stall=N)``: the transactor aborts with
``[allo-rtl] <top>: STALLED`` after ``N`` consecutive cycles without handshake
progress (default 500000, the PR's hard-coded value). Raise it for an IP that
computes for long stretches without touching a stream; it is not a timeout on the
whole call.

Limits found by the probe
-------------------------

Wrapping MiniTPU's ``mxu.sv`` (``dev/records/minitpu/pr48_probe_2026-10-08.md``
section 4) found what the descriptors cannot express. Each needs a hand-written
RTL shim around the core today:

- **No scalar or wire sideband pins.** A ``Port`` is a stream (data + valid +
  ready) and a ``MemPort`` is a RAM; there is no descriptor for a plain input
  wire, a one-cycle pulse, or an informational output. Every top-level *input*
  must be bound, so such a pin must be driven from inside a shim (e.g. derived
  from word counts).
- **No aggregate pins.** Packed/unpacked arrays, structs and enums at the top
  are refused by ``validate_rtl``.
- **Payloads are at most 32 bits** (``bool`` and 8/16/32-bit integers). Anything
  wider must be serialised into 32-bit beats by a shim.
- **Fixed transfer counts per object.** Without ``done``, a call ends when every
  output port has delivered its ``size`` tokens, and every output needs a
  positive ``size`` or a ``done`` pin -- so a call that produces nothing, or a
  data-dependent number of tokens, needs a ``done`` protocol in the shim.
- **One instance per object.** The persistent Verilator model belongs to the
  generated function, i.e. to the ``RTLModule`` object, and the builder allows one
  static call site per object. Two objects are two separate pieces of hardware,
  not two operations on one; instruction-level granularity on one model is not
  expressible.
- **ii=0 only.** A call blocks its kernel until it completes; overlapping
  transactions (e.g. loading the next tile while the current one streams out)
  cannot be exercised.
- **Functional, not cycle-accurate.** The transactor is untimed with respect to
  the rest of the region (8 reset cycles once per model).
- **Vitis side:** ``target="vitis_hls"`` needs ``hls=HLSBlackBox(c_model=...)``;
  the generated ready/valid adapter needs an active-high clock enable on the core;
  Vitis 2023.2 only, ``mode="csyn"`` only.

Whole-core use (MiniTPU M-R1)
-----------------------------

``examples/minitpu/rtl/`` wraps MiniTPU's whole ``minitpu_core`` as one
``RTLModule`` (``dev/records/minitpu/minitpu_rtl_m1_2026-10-08.rst``): all 52
oracle launches drain bit-identical to MiniTPU's own testbench. What it
established about the descriptors:

- **A ``MemPort`` and stream ``Port``\ s on one IP work under
  ``target="simulator"``**, including eight ``MemPort``\ s of 2**20 words bound
  to region boundary arrays.
- A 256-bit memory word is served as eight interleaved 32-bit ``MemPort``\ s
  (one access per bank per cycle); there is no wide or laned ``MemPort``.
- A non-persistent input stream needs a fixed ``size``, so a variable-length
  program is padded to a fixed length.
- ``$readmem`` paths in the RTL resolve against the *Python process's* working
  directory, and a missing file is only a warning: the caller must ``chdir``
  and check the model's output (there is no run-directory or plusarg option).
- The model is rebuilt per ``RTLModule`` object (no cache): ~70 s per process
  for the MiniTPU core at ``--build-jobs 16``.
- An ``RTLModule`` composes in ``compose.Architecture`` only as a *parameter*
  (``calls=`` takes functions), unchecked against its ports.
- One boundary array may be passed to two ``MemPort``\ s of one call (a write
  port and a read port): the transactor commits memory ports in list order, so
  listing the write port first gives a write-first dual-ported RAM. M-R2b
  (``minitpu_rtl_m2b_2026-10-08.rst``) relies on it; it is not a documented
  contract of ``MemPort``.

Effects on projects without an RTLModule
----------------------------------------

- **vitis.py adds depth= to every m_axi pragma** of every
  ``target="vitis_hls"`` project whose top-level array arguments have static
  shapes: ``depth=`` is the product of the static dimensions (Vitis cosim needs
  it). This is kept as the PR has it. It changes the ``kernel.cpp`` of such
  projects (it does not touch the ``vhls``, ``catapult`` or ``systemc``
  emission: TinyTPU's three emission hashes are unchanged). TinyTPU's
  ``cosim.py`` ``patch_axi_depths`` used to *append* a ``depth=`` and now
  replaces the one Allo emits (the values are the same).
- **config_compile -pipeline_loops 0** is written into ``run.tcl`` of every
  Vitis project that contains an ``RTLModule`` (Vitis 2023.2 refuses
  ``ap_ctrl_chain`` black boxes inside pipelined regions). It switches off
  automatic loop pipelining for the whole solution, not only around the IP call;
  explicit pipeline directives still apply.
- **The region wiring check counts IP calls.** A plain-kernel region refuses a
  stream with only one blocking end (``allo/ir/units.py``,
  ``_check_kernel_streams``). An ``IPModule``/``RTLModule`` call ``ip(a, c)`` now
  counts as a reader of the streams in the IP's ``input_idx`` and a writer of
  those in ``output_idx``; before the follow-up, every region calling a stream IP
  was refused as "unconnected-stream".

.. _rtl-module-rtlmodule:

RTLModule: simulation and Vitis black-box integration
=====================================================

``allo.RTLModule`` calls Verilog/SystemVerilog IP from an Allo kernel. It extends
and integrates the earlier ``allo_rtl.py`` prototype. No LLM, API key, or wrapper
generation service is involved.

The simulator runs a Verilator model through a generated C++ transactor. The
Vitis backend emits a black-box description and registers a supplied or
automatically generated RTL wrapper. HLS synthesizes the surrounding Allo computation, preserving the
external call as hardware implemented by that wrapper.

Supported paths
---------------

.. list-table::
   :header-rows: 1

   * - Binding
     - Allo software simulation
     - Vitis black-box export
   * - ``Port(bind="stream")``
     - Dataflow simulator; ready/valid or active-high FIFO
     - Supplied FIFO/chain wrapper or generated ready/valid adapter
   * - ``Port(bind="array")``
     - LLVM; array traversed through ready/valid
     - Not yet supported
   * - ``MemPort``
     - LLVM or mixed dataflow simulation; host array acts as RAM
     - Not yet supported

Payloads are ``bool``, signed/unsigned 8-, 16-, and 32-bit integers. Floating-point,
wide packed values, AXI interfaces, multiple clocks, and overlapping transactions
are not supported. Array buffers crossing dataflow kernel boundaries retain the
existing compiler limitations; the array regression uses the LLVM path. C ``int`` and ``unsigned int`` are 32-bit aliases. Stream payloads
must be scalar; directions and signedness must match the Allo declarations.

The first synthesis implementation targets the Vitis 2023.2 black-box JSON/Tcl
flow. Vendor synthesis, co-simulation, and export require manual validation on
your installation. Emitting a project successfully is not proof of hardware
correctness. Other HLS vendors and Vitis ``hw``/emulation builds are rejected.

Supplied-wrapper example
------------------------

See ``examples/ip_integration/rtl_accumulator.py``, ``.v``, and ``.cpp``. The supplied
RTL accepts one integer per invocation and returns a running sum. It demonstrates
state across repeated calls inside one Allo kernel and bounded FIFO backpressure.

.. code-block:: python

   from allo import RTLModule, Port, HLSBlackBox

   ip = RTLModule(
       top="accumulator",               # top of the supplied wrapper
       rtl="rtl_accumulator.v",         # or [wrapper, dependency, ...]
       ports=[
           Port("A", "a_dout", "a_empty_n", "a_read",
                size=1, protocol="ap_fifo"),
           Port("C", "c_din", "c_write", "c_full_n",
                dir="out", size=1, protocol="ap_fifo"),
       ],
       done="ap_done",
       persistent=True,
       hls=HLSBlackBox(c_model="rtl_accumulator.cpp", latency=3),
   )
   # Inside a @df.kernel:
   # ip(input_stream, output_stream)

``top`` names the wrapper, not an unadapted inner IP. Include the original IP's
sources in ``rtl`` if the wrapper instantiates it. The supplied C model implements
the same ordered signature for Vitis C simulation and C/RTL comparison. It must
be self-contained apart from standard/HLS headers. It is not used by Allo's
Verilator simulation, which runs the wrapper itself. Model correctness remains
the author's responsibility, checked by vendor co-simulation.

Automatic ready/valid wrappers (Milestone 2)
--------------------------------------------

Use ``HLSBlackBox(adapter=ReadyValidAdapter(...))`` to generate the hardware glue.
Your ``Port`` descriptors now refer to the **original IP**, not HLS-facing pins.
No LLM or handwritten Verilog wrapper is needed for this supported contract.

.. code-block:: python

   from allo import RTLModule, Port, HLSBlackBox, ReadyValidAdapter

   ip = RTLModule(
       top="ready_valid_accumulator",
       rtl="ready_valid_accumulator.v",
       ports=[
           Port("A", "a_data", "a_valid", "a_ready", size=4),
           Port("B", "b_data", "b_valid", "b_ready", size=4),
           Port("C", "c_data", "c_valid", "c_ready", dir="out", size=4),
       ],
       clock="clk", reset="rst_n", reset_active_high=False,
       start=None, persistent=True,
       hls=HLSBlackBox(
           c_model="ready_valid_accumulator.cpp",
           latency=8,  # Measured for this example's complete wrapper, without stalls.
           adapter=ReadyValidAdapter(clock_enable="ce", start_mode="none"),
       ),
   )
   print(ip.generate_wrapper())   # Pure generation: no compiler or model API call.

The example is ``examples/ip_integration/rtl_adapter.py``, with the original RTL
and C model in ``ready_valid_accumulator.v`` and ``.cpp``. Each call consumes four
items on both inputs and produces four running sums. Two calls demonstrate
state persisting across transactions.

The generated Verilog contains:

- One holding register per input stream. HLS FIFO reads fetch data into it;
  ready/valid transfers consume it. A stalled core sees stable pending data.
- Independent fetched/consumed counters per input and accepted-output counters.
  Prefetch never counts as core consumption, and counts prevent fetching tokens
  from the next transaction.
- Output FIFO write strobes qualified by core valid, FIFO room, and clock enable.
  The core itself must retain output data and valid while stalled.
- A four-state controller: idle, launch, run, completed. Completion remains
  asserted until ``ap_continue`` is sampled on an enabled edge.
- Clock-enable and reset-polarity adaptation. No generated/gated clock is used.

The **same generated wrapper** is compiled by Verilator for Allo simulation and
packaged for Vitis. ``validate_rtl()`` checks both the original and generated top.
The wrapper defaults to ``<original_top>_allo``; ``name=`` selects a different name,
which must differ from the original top. Generated boundary names are standard
``ap_*`` controls and numbered ``p0_data``, ``p0_read``, etc.; the JSON records them.
Different wrappers around the same source set share the staged core definitions.
The generated build is cached on the RTLModule object; construct a new object
after changing source files or interface descriptions.

Required adapter contract
~~~~~~~~~~~~~~~~~~~~~~~~~

- One clock; scalar ready/valid stream arguments with a positive ``size`` for
  **every** input and output. Per-port counts may differ.
- An active-high core clock-enable input (``clock_enable="ce"`` above). When it is
  low, all core state and transfers must stop. The adapter freezes the core
  outside active transactions; reset overrides that enable so synchronous reset
  still works. A core without clock-enable support needs a custom wrapper.
- ``persistent=True``. Internal state survives calls; system reset clears it.
- ``start_mode="none"`` requires ``RTLModule(start=None)`` for a free-running core.
  ``start_mode="pulse"`` requires a named core start pin and generates one enabled
  launch cycle. It does not implement an arbitrary start/ready negotiation.
- ``completion="counts"`` (default) finishes after all declared input transfers
  into the core and output transfers to HLS complete. Choose this only if the
  core is quiescent after those transfers; hidden work must not remain.
- ``completion="done"`` additionally waits for the named ``RTLModule(done=...)``
  signal. A pulse during the run phase is latched, so a late output drain cannot
  lose it. ``done`` must describe the current transaction after launch, not idle.
- ``ii=0``: invocations do not overlap. ``latency`` must describe the **complete
  generated wrapper** under unstalled conditions. It is supplied, not inferred.

Parameters may be integer overrides; they are emitted in the core instantiation.
Preprocessor defines, external include paths, arrays, memory ports, AXI sidebands,
multiple clocks, variable token counts, and non-stallable cores remain custom
wrapper cases. Unsupported combinations are rejected. Standard generation does
not prove an arbitrary core honors its declared protocol: retain IP-specific
verification and run the generated project's vendor co-simulation.

A supplied C model is still required for Vitis C simulation/co-simulation. It must
implement the original top's logical signature and whole per-call transaction,
including persistent state. Allo renames that entry for the generated wrapper;
Allo's dataflow simulator runs the RTL instead of this C model.
The C model body must remain visible with ``__SYNTHESIS__`` defined: Vitis needs
it to extract black-box information. The JSON/Tcl black-box registration selects
the RTL implementation; do not hide the model body behind a synthesis guard.
The generated black-box declaration uses C++ linkage to preserve stream type
information for Vitis. Define the supplied model with ordinary C++ linkage,
without ``extern "C"``; the surrounding Allo kernel can retain C linkage.

Run and inspect
~~~~~~~~~~~~~~~

.. code-block:: bash

   conda activate allo
   export OMP_NUM_THREADS=8
   python examples/ip_integration/rtl_adapter.py --wrapper /tmp/adapter.v --simulate
   python examples/ip_integration/rtl_adapter.py --project /tmp/allo_rtl_milestone2.prj
   # Manual vendor validation, from the generated project:
   cd /tmp/allo_rtl_milestone2.prj
   vitis_hls -f validate.tcl

The Vitis backend emits array element counts directly in the top-level AXI
memory interface pragmas for co-simulation. This example uses depth 8 for each
array. Multidimensional arrays use the product of their static dimensions.

In the local shell, the existing ``allo()`` helper initializes conda, Vitis HLS
2023.2, LLVM, and XRT. The Python library does not source shell setup scripts or
modify that configuration. ``VERILATOR`` can point to a binary in another conda
environment; ``MLIR_INCLUDE_DIR`` is optional when LLVM discovery succeeds.

For interfaces outside this generator's contract, use Milestone 1's supplied
wrapper route: keep ``adapter=None``, describe its FIFO pins with
``protocol="ap_fifo"``, and pass its chain-control configuration. This is also the
extension boundary for any future external wrapper-generation tool.

Port and control contracts
--------------------------

`Port(name, data, valid, ready, dir="in", ctype="int32_t", bind="stream",
size=None, protocol="ready_valid")` preserves the prototype's positional API.
``dir`` is relative to the IP. ``size`` is a per-invocation token count, **not FIFO
depth**. Allo's ``Stream[T, depth]`` specifies FIFO depth.

For ``protocol="ap_fifo"``, use:

.. list-table::
   :header-rows: 1

   * - Direction
     - ``data``
     - ``valid``
     - ``ready``
   * - Input to IP
     - ``dout``
     - ``empty_n``
     - ``read``
   * - Output from IP
     - ``din``
     - ``write``
     - ``full_n``

These are active-high availability signals and zero-latency/front-visible FIFO
data. A synchronous FIFO that returns data a cycle after a read needs a supplied
adapter. Setting the protocol label does not generate that adapter.

``MemPort(name, size, ctype, addr, ce, q=None, we=None, d=None)`` models word-addressed
RAM: reads return data after the requesting edge; writes commit at the edge.
``we`` and ``d`` must appear together. Addresses must fit the declared size, and
out-of-bounds accesses abort simulation. Byte enables, dual ports, and alternate
read latencies need future descriptors/adapters.

For synthesis with ``adapter=None``, the supplied wrapper must expose:

- ``clock`` and active-high ``reset`` (defaults ``ap_clk``, ``ap_rst``).
- ``start`` and ``done`` (set ``done="ap_done"`` explicitly).
- ``HLSBlackBox.ready``, ``.idle``, ``.continue_``, ``.clock_enable`` (defaults
  ``ap_ready``, ``ap_idle``, ``ap_continue``, ``ap_ce``).
- FIFO ports for every logical argument.

The wrapper must honor clock enable, accept start according to the chain
protocol, and hold completion until continue. ``persistent=True`` is required:
hardware resets at system reset, not on every function call. ``ii=0`` explicitly
marks the initial implementation as non-pipelined. Supply the true unstalled
latency; it is not inferred or validated from port names. Backpressure can extend
elapsed time. Resources are not estimated by Allo.

The transactor resets for eight cycles. Without ``done``, simulation ends when all
output ports reach their declared sizes. Non-persistent input streams require
sizes to bound prefetch. Persistent stream transactors retain pending input tokens
across calls. A prolonged lack of progress aborts with an RTL stall diagnostic.
This is functional integration simulation, not a globally synchronized cycle
simulation of every Allo kernel. Simulator FIFO capacity includes any transactor
holding registers and need not match HLS timing.

Instances and state
-------------------

Use one static call site per RTLModule object. A runtime loop around that call
is supported. Multiple static call sites, including replicated mapped kernels,
currently require distinct RTLModule objects and distinct ``name=`` values. The
backend emits structural aliases when the C name differs from the RTL top.

Vitis 2023.2 rejects ``ap_ctrl_chain`` black boxes inside pipeline regions.
Projects containing RTLModule therefore emit ``config_compile -pipeline_loops 0``
to disable automatic loop pipelining throughout the solution. Explicit pipeline
directives elsewhere remain effective, but do not pipeline a loop or function
containing an RTLModule call. This conservative setting can reduce performance
of other loops that previously relied on automatic pipelining.

Persistent simulator state belongs to the compiled model on its executing thread.
Do not depend on state surviving a change of simulator worker thread or rebuild.
Independent modules compile into isolated libraries. Automatic shared-instance
arbitration and overlapping calls are outside this milestone.

Build and validation
--------------------

Activate the Allo environment first. For software simulation install Verilator
and a C++17 compiler. Set ``VERILATOR`` to its executable if it is not on PATH.
Configure ``LLVM_BUILD_DIR`` as required by the existing Allo simulator.
``MLIR_INCLUDE_DIR`` may specify the directory containing
``mlir/ExecutionEngine/CRunnerUtils.h``; otherwise Allo attempts LLVM discovery.
The optional ``verilator=`` and ``mlir_include=`` arguments provide per-module values.

.. code-block:: bash

   conda activate allo
   export OMP_NUM_THREADS=8
   python examples/ip_integration/rtl_accumulator.py --validate-rtl --simulate
   python examples/ip_integration/rtl_accumulator.py --project /tmp/rtl_accumulator.prj

Construction and project emission do not run Verilator or Vitis. ``validate_rtl()``
explicitly elaborates RTL with Verilator, checking top-level pin names, widths,
directions, and unbound inputs. Simulation invokes this automatically. This
structural check does not prove protocol compliance.

The emitted package contains a typed header, C model, original RTL,
optional instance alias, and ``blackbox.json``; ``run.tcl`` registers it with
``add_files -blackbox``. Sources use project-relative paths; identical RTL source sets share a content-addressed directory. Run tools from the
project directory. Source dependencies must be listed explicitly; synthesis
requires self-contained RTL. Supplied wrappers resolve their own parameters and
preprocessor configuration; generated adapters support explicit integer parameters. Simulation supports integer
``parameters``, ``defines``, and ``include_paths`` passed to Verilator.

Generated wrappers and aliases declare a ``timescale 1ns / 1ps`` directive for RTL
co-simulation. Supplied RTL should declare its own appropriate timescale (or
SystemVerilog time units); Allo copies those sources without modifying them.
XSIM rejects modules with missing timescales when other design modules have one.

For **manual Vitis validation**, the example emits a deterministic C testbench and
``validate.tcl``:

.. code-block:: bash

   cd /tmp/rtl_accumulator.prj
   vitis_hls -f validate.tcl

This requests C simulation, synthesis, RTL co-simulation, and IP export. Inspect
logs and the generated hierarchy to confirm that ``accumulator`` is present. The
example uses the default board configuration; select a supported part in the Tcl
if your installation requires a different device.

Python's ``df.build(..., target="vitis_hls", mode="csyn")`` emits the project;
calling its returned module starts the vendor build. This milestone deliberately
supports only that Python synthesis mode. Direct sequential ``csim`` stream calls
are not supported; use Allo's dataflow simulator or the manual Tcl testbench.

Migrating the prototype
-----------------------

Change ``from allo_rtl import RTLModule, Port, MemPort`` to
``from allo import RTLModule, Port, MemPort``. The ``ports=[...]`` and
``n=..., inputs=..., outputs=..., mode="stream"/"buffer"`` forms remain available.
The convenience form still defaults to array binding. Descriptors are copied
rather than mutated. The RTL object now retains its description rather than
returning a monkey-patched IPModule.

Verilator runs lazily, so ``generated_source`` names a file that appears only after
simulation preparation. Temporary build artifacts are isolated and cleaned up
with their owning object. ``workdir`` chooses the parent for an isolated build
directory, not a shared cache. Direct ``ip(...)`` from ordinary Python is not a
nanobind interface; call it from an Allo kernel.

The original local prototype is left untouched. No LLM dependency has been added. Standard ready/valid wrappers are generated
deterministically; custom protocol wrappers remain user-supplied.

Implementation map
------------------

- ``allo/backend/rtl.py``: descriptors, validation, transactor, simulation compiler,
  and black-box export.
- ``allo/backend/rtl_adapter.py``: deterministic ready/valid adapter emitter.
- ``allo/ir/infer.py``: logical argument validation before integer signedness is lost.
- ``allo/ir/builder.py``: external-call reuse and symbol-collision checks.
- ``allo/backend/hls.py``: backend eligibility, header inclusion, and registration.
- ``allo/backend/vitis.py``: static array depths in AXI memory interface pragmas.
- ``allo/passes.py``: recursive external-call rewriting, required for array/memory
  RTLModule calls nested in runtime loops (an existing local prerequisite).
- ``allo/customize.py`` and ``allo/dataflow.py``: reject unsupported target dispatch.
- ``tests/ip_integration/test_rtl.py``: supplied-wrapper and simulation regressions.
- ``tests/ip_integration/test_rtl_adapter.py``: deterministic export, Allo simulation,
  randomized stalls/reset, and pulse-start/delayed-done adapter checks.

Protocol reference: AMD UG1399,
`JSON File for RTL Blackbox <https://docs.amd.com/r/2023.1-English/ug1399-vitis-hls/JSON-File-for-RTL-Blackbox>`__.

Reproducing software checks
---------------------------

Vendor validation status (2026-10-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ready/valid accumulator example passed C simulation, HLS synthesis, and
XSIM C/RTL co-simulation with Vitis HLS 2023.2 targeting
``xcu280-fsvh2892-2L-e``. Vivado IP export also completed; ``export.zip`` contains
both ``ready_valid_accumulator.v`` and ``ready_valid_accumulator_allo.v``, and
``component.xml`` registers both sources.

The initial verified run used one top-level transaction (two RTLModule calls).
The extended testbench also passed C/RTL co-simulation and IP export with three
top-level calls, checking the persistent running sum across all six RTLModule
calls.
This does not establish post-route timing or board-level operation.

Software regression
~~~~~~~~~~~~~~~~~~~

On 2026-10-01 the command below passed **51 tests**, with the vendor synthesis
test deliberately deselected. The three-transaction example also passed a
standalone C-model/testbench run. Changed Python files passed Black, new C++
models passed clang-format, and the RTL backends passed pylint. Repository-wide
lint still encounters a pre-existing blank-line formatting issue in
``examples/ip_integration/vadd_ip.py``.

With the Allo/LLVM environment active and ``VERILATOR`` available:

.. code-block:: bash

   python -m pytest tests/ip_integration/test_rtl_adapter.py \
     tests/ip_integration/test_rtl.py \
     tests/ip_integration/test_stream_ip.py \
     tests/ip_integration/test_stream_ip_sim.py \
     tests/utils/test_backend_utils.py \
     --deselect tests/ip_integration/test_stream_ip.py::test_stream_ip_csynth -q

The deselection is intentional even when Vitis is on PATH: these are software
checks. Run the generated ``validate.tcl`` separately for vendor verification.
The adapter tests exercise FIFO sources changing immediately after reads,
independent input stalls, output stalls, clock-enable stalls, repeated calls,
excess offered input tokens, reset mid-transaction, and delayed completion.

.. _rtl-module-stream-ip:

Integrating HLS IPs that use ``hls::stream`` interfaces
=======================================================

This document explains a change that lets ``allo.IPModule`` integrate a
hand-written HLS C++ block ("IP") whose interface uses ``hls::stream<T>``
ports, not just arrays and scalars. It is written for someone new to the Allo
compiler and to compilers in general, so it starts with background and builds up
to the specific edits.

If you only want the "what changed" list, jump to :ref:`The five edits <rtl-module-five-edits>`.

1. Background you need first
----------------------------

1.1 What an ``IPModule`` is
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An IP is a ``.cpp`` file you wrote by hand (or got from a vendor) containing one
HLS function, e.g.:

.. code-block:: cpp

   void vadd(int A[32], int B[32], int C[32]) { ... }

``allo.IPModule(top="vadd", impl="vadd.cpp")`` lets an Allo design *call* that
function. Allo does **not** read the function body. It only:

1. **parses the signature** to learn the argument types,
2. **emits a call** to it in the generated code, and
3. **stitches the IP source back in** with ``#include`` at the end.

So the IP is a black box wired in by name — the same idea as an ``extern``
declaration in C, or a foreign-function binding (``ctypes``).

1.2 A 60-second MLIR primer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allo compiles your Python down through **MLIR**, an intermediate representation.
A few facts are enough to read the rest of this doc:

- Everything is an **operation** named ``dialect.opname``. ``func.func`` is the
  "define a function" op; ``func.call`` calls one; ``allo.stream_construct`` creates
  a FIFO. ``func``, ``allo``, ``memref``, ``affine`` are *dialects* (namespaced groups of
  ops).
- ``%x`` is a **value** (data flowing through the function). ``@name`` is a
  **symbol** (a global name, e.g. a function). So ``call @vadd(%0)`` means "call
  the function named ``vadd``, passing value ``%0``".
- **Types**: ``memref<32xi32>`` is a buffer of 32 ints (Allo's array type);
  ``!allo.stream<i32, 4>`` is a FIFO of ``i32`` with depth 4 (the leading ``!`` means
  "a type defined by a dialect").
- A ``func.func private @vadd(...)`` with no body is a *declaration* — "this
  function exists somewhere, here is how to call it". This is exactly how the IP
  appears inside MLIR.

1.3 The pipeline a call goes through
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a dataflow (``@df.region``) design, an IP call passes through these stages, in
order:

.. code-block:: text

   Python  ──parse──►  IPModule.args     (backend/ip.py)
           ──build──►  func.func private + func.call in MLIR   (ir/builder.py)
           ──hoist──►  streams moved onto kernel interfaces     (dataflow.py)
           ──emit───►  HLS C++ text                             (mlir/.../EmitVivadoHLS.cpp)
           ──splice─►  #include the IP .cpp into kernel.cpp      (backend/hls.py)

The key discovery behind this change: **the emit and splice stages already
handle streams correctly.** A hand-written
``func.func private @my_ip(memref<8xi32>, !allo.stream<i32, 4>)`` plus a ``call``
emits exactly the C++ we want with **zero backend changes**. So the whole task
was getting the *frontend* (parse → build → hoist) to produce that IR.

1.4 Why streams were the hard part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Parsing:** ``hls::stream<int>&`` has a namespace (``::``), a template (``<>``), and
  a reference (``&``). The old type regex could not match it.
- **Direction is ambiguous:** in C++ you read a stream with ``.read()`` and write
  it with ``.write()``, but **both are declared the same way**: ``hls::stream<T>&``.
  So the tool cannot tell, from the signature alone, whether the IP *consumes* or
  *produces* a given stream. The user must say.
- **The hoisting pass rejected the call:** ``move_stream_to_interface()`` figures
  out each stream's direction by looking at *how it is used*. It understood
  ``StreamPut``/``StreamGet`` but hit a hard ``raise`` on anything else — including a
  ``func.call`` to an IP.

1.5 What "hoisting" means (the stage most people haven't seen)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your Python you declare a stream once at region scope and use it inside
kernels, as if it were shared. But MLIR/HLS functions can only touch what is
*passed into them*. So a pass called ``move_stream_to_interface()`` rewrites:

.. code-block:: text

   // before: each kernel has its own local copy of the stream
   func.func @producer() { %s = allo.stream_construct {name="fifo"} ; put(%s, ...) }
   func.func @consumer() { %s = allo.stream_construct {name="fifo"} ; get(%s) }

into:

.. code-block:: text

   // after: one stream in `top`, passed into each kernel as an argument
   func.func @producer(%s: !allo.stream<i32,4>) { put(%s, ...) }
   func.func @consumer(%s: !allo.stream<i32,4>) { get(%s) }
   func.func @top() { %s = allo.stream_construct {name="fifo"} ; producer(%s); consumer(%s); }

"Hoisting" = lifting the declaration out of the kernel body and onto the kernel's
**interface** (argument list), then creating one real stream in ``top`` and
threading it to everyone. This is what turns three disconnected copies into one
connected FIFO. An IP call has to survive this pass so its stream operand gets
retargeted to the shared stream too.

2. The design decisions
-----------------------

- Direction is declared with ``input_idx`` / ``output_idx`` on ``IPModule``. These
  list which argument positions the IP *reads* (input) vs *writes* (output). This
  mirrors the existing AIE ``ExternalModule`` API, so it is not a new concept in the
  codebase. For ``vadd_stream(A, B, C)`` where the IP reads A, B and writes C:
  ``input_idx=[0, 1], output_idx=[2]``.
- **Kernel-scope only.** The IP is called from inside a ``@df.kernel`` body. (An
  earlier experiment showed calling it directly in the ``@df.region`` body crashes a
  different pass, ``_build_top``; supporting that is a separate, larger change.)
- **HLS targets only** *(as of this change)*. A stream IP works for
  ``vitis_hls``/``vivado_hls`` (csyn and beyond). It cannot run on the CPU targets
  (``llvm``, ``simulator``) or in ``csim`` mode, because those link the IP as a shared
  library through raw pointers and have no way to represent a FIFO. We reject
  those paths with a clear error instead of failing confusingly later.
  **Since then**, ``target="simulator"`` *is* supported, through a stream shim
  that makes the IP drive Allo's ring buffers directly — see
  :ref:`rtl-module-sim-shim`. The plain ``llvm`` target
  and ``csim`` remain fenced off, because they run the kernels sequentially and a
  blocking stream read could never be satisfied.

.. _rtl-module-five-edits:

3. The five edits
-----------------

Edit 1 — Parser recognizes ``hls::stream`` — ``allo/backend/ip.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*(This edit was made first, before the others.)*

``parse_cpp_function()`` returns a list of ``(type, shape)`` tuples describing each
argument. The shape slot encodes the kind: ``()`` scalar, ``None`` pointer, a tuple
of dims for an array. A new sentinel ``STREAM`` (an instance of ``_StreamShape``)
marks a stream, and for a stream the *type* slot holds the full stream type
string as written, e.g. ``('hls::stream< int8_t >', STREAM)``.

Why a sentinel in the shape slot: the 2-tuple shape is unpacked positionally in
several places and is also shared with the AIE ``ExternalModule``. Keeping the
2-tuple intact and marking streams with a distinct shape value means existing
code keeps working, and any shape-dispatching code (``len(shape)``) fails loudly on
a stream rather than silently doing the wrong thing.

Supporting pieces: new regexes (``_TYPE_TOKEN``, ``_STREAM_TOKEN``) that match a
namespaced template whole; the comma-splitter now tracks ``<``/``>`` depth (so a
template argument's internal comma does not split a parameter); and inline
``/* ... */`` comments are stripped first (the HLS emitter annotates stream
parameters with ``/* v0[2] */``, which would otherwise look like array dims).

**Known limitation:** the template regex allows one level of ``<>`` nesting, so
``hls::stream<int>`` and ``hls::stream<ap_int<8>>`` parse, but a double-nested
element like ``hls::stream<vector<ap_int<8>,4>>`` does not.

Edit 2 — ``IPModule`` accepts ``input_idx`` / ``output_idx`` — ``allo/backend/ip.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IPModule.__init__`` gained optional ``input_idx=None, output_idx=None``
parameters, stored on the object. They declare per-argument direction — required
for stream ports since the C++ signature cannot express it. Default ``None``
preserves the historic behavior for array/scalar IPs. This matches
``ExternalModule``, and the IR builder already reads ``obj.input_idx``.

Also added a small helper ``IPModule.has_stream_args`` (True if any argument is a
stream) used by the fencing in Edit 4.

Edit 3 — Builder types the stream call — ``allo/ir/builder.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``build_Call``, the ``IPModule``/``ExternalModule`` branch (~line 3183) now has a
case for ``shape is STREAM``. For a stream argument it:

1. Clones the ``stream_construct`` op into the calling kernel, exactly as
   ``put``/``get`` do. A stream referenced by name resolves to the construct op that
   was created where the stream was declared — possibly in another function.
   Cloning makes a local copy so the call is a use of a construct *inside this
   kernel* (the hoisting pass later deduplicates by the stream's ``name``).
2. **Adopts the operand's own stream type** (``!allo.stream<T, depth>``) as the
   declaration's argument type, rather than building a ``memref``. The operand
   carries both the element type and the FIFO depth; the C++ signature has
   neither (it never states a depth).
3. **Skips the alloc/copy dance.** Streams pass by reference, so there is nothing
   to copy in or out.
4. **Records direction on the call.** It builds a ``stream_dirs`` string — one
   character per operand, ``i``/``o``/``_`` — from ``input_idx``/``output_idx``, and
   attaches it as a string attribute on the ``func.call``. This is how the
   direction the *user* declared in Python reaches the hoisting pass, which only
   sees MLIR. (If a stream arg is in neither list, the builder raises a clear
   error telling the user to declare it.)

The result, for ``vadd_stream(sA, sB, sC)``:

.. code-block:: text

   func.func private @vadd_stream(!allo.stream<i32, 4>, !allo.stream<i32, 4>, !allo.stream<i32, 4>)
   ...
   call @vadd_stream(%2, %0, %1) {stream_dirs = "iio"} : (...) -> ()

Edit 4 — Fence the CPU / simulator paths — ``allo/backend/ip.py``, ``allo/passes.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*(Partly superseded: the dataflow simulator now runs stream IPs through a shim;
see :ref:`rtl-module-sim-shim`. The fences described
below still stand for the plain ``llvm`` target and for ``csim``.)*

A stream IP cannot run on the CPU. The two wrapper generators
(``generate_nanobind_wrapper``, ``generate_mlir_c_wrapper``) and the shared-library
compile call them, so a guard at the top of each raises a clear
``NotImplementedError`` if the IP has stream args. ``call_ext_libs_in_ptr`` (used by
both the LLVM JIT and the OMP dataflow simulator) also checks up front, so the
error surfaces early with a helpful message rather than as a cryptic g++ or JIT
failure. ``csim`` mode reaches these guards transitively (it re-parses the
generated ``kernel.cpp`` through the nanobind path).

Edit 5 — Hoisting pass accepts the IP call — ``allo/dataflow.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the linchpin. In ``move_stream_to_interface()``, the loop that classifies a
stream's direction by walking its uses previously understood only
put/get/empty/full and hit ``raise ValueError("Stream is not used correctly")`` on
anything else. A new branch handles a ``func.CallOp``:

.. code-block:: python

   elif isinstance(use.owner, func_d.CallOp):
       dirs = use.owner.attributes["stream_dirs"].value   # written by Edit 3
       direction = "in" if dirs[use.operand_number] == "i" else "out"

``use.operand_number`` is the stream's position among the call's operands, so we
read the matching character from the ``stream_dirs`` string Edit 3 attached.
Everything after classification is already generic: the pass appends the stream
to the kernel's signature and rewrites every use (including this call's operand)
to the new argument, and ``_build_top`` threads one shared stream through the whole
graph. No further changes were needed.

**Constraint (inherited, not introduced):** direction is a single value per
stream per kernel. A given kernel either reads or writes a given stream —
separate in/out streams are fine, a bidirectional port on one stream is not
expressible. This is exactly how put/get already behave.

4. What the finished pipeline produces
--------------------------------------

For a region with feeder kernels, a wrapper kernel that calls the IP, and a drain
kernel, the emitted HLS C++ is:

.. code-block:: cpp

   void ip_wrap_0(
     hls::stream< int32_t >& v8,
     hls::stream< int32_t >& v9,
     hls::stream< int32_t >& v10
   ) {
     vadd_stream(v10, v8, v9);
   }
   ...
   void top(int32_t *v20, int32_t *v21, int32_t *v22) {
     #pragma HLS dataflow
     hls::stream< int32_t > v33;   // one real FIFO per stream, declared in top
     hls::stream< int32_t > v34;
     hls::stream< int32_t > v35;
     feedA_0(...); feedB_0(...); ip_wrap_0(v34, v35, v33); drain_0(...);
   }

and ``vadd_stream.cpp`` is copied into the project, ``#include``\ d in ``kernel.cpp``,
and ``add_files``'d into ``run.tcl``.

5. How to use it
----------------

.. code-block:: python

   import allo
   from allo.ir.types import int32, Stream
   import allo.dataflow as df

   vadd_stream = allo.IPModule(
       top="vadd_stream",
       impl="vadd_stream.cpp",
       input_idx=[0, 1],   # the IP reads streams A and B
       output_idx=[2],     # the IP writes stream C
   )

   @df.region()
   def top(A: int32[32], B: int32[32], C: int32[32]):
       sA: Stream[int32, 4]
       sB: Stream[int32, 4]
       sC: Stream[int32, 4]

       @df.kernel(mapping=[1], args=[A])
       def feedA(a: int32[32]):
           for i in range(32): sA.put(a[i])

       @df.kernel(mapping=[1], args=[B])
       def feedB(b: int32[32]):
           for i in range(32): sB.put(b[i])

       @df.kernel(mapping=[1])
       def ip_wrap():
           vadd_stream(sA, sB, sC)      # the IP call, kernel scope

       @df.kernel(mapping=[1], args=[C])
       def drain(c: int32[32]):
           for i in range(32): c[i] = sC.get()

   mod = df.build(top, target="vitis_hls", mode="csyn", project="out.prj")

See ``tests/ip_integration/test_stream_ip.py`` and
``tests/ip_integration/vadd_stream.cpp`` for the runnable version.

6. Verification performed
-------------------------

- **Parser:** returns ``('hls::stream< int32_t >', STREAM)`` for stream args and
  unchanged tuples for memref/pointer/scalar/``ap_int`` (regression).
- **IR (pre-hoist):** the private decl carries ``!allo.stream`` types and the call
  carries ``stream_dirs``.
- **IR (post-hoist):** ``df.customize`` no longer aborts; the IP-calling kernel
  gets the streams on its interface (``stypes`` updated) and the graph is wired
  through one shared construct per stream in ``top``.
- **Codegen:** emitted C++ has ``hls::stream< int32_t >&`` on the wrapper signature
  and the ``vadd_stream(...)`` call; the IP ``.cpp`` is copied, ``#include``\ d, and
  ``add_files``'d.
- **Fencing:** ``target="simulator"`` (and the LLVM path) raise a clear
  ``NotImplementedError``. *(The simulator fence was later lifted — see
  :ref:`rtl-module-sim-shim`; the LLVM one stands.)*
- **Regressions:** existing ``tests/ip_integration/test_external.py`` (7 passed, 1
  skipped) and ``tests/dataflow/test_df_unit.py`` pass.

End-to-end csynth requires ``vitis_hls`` on PATH (the `test_stream_ip.py::
test_stream_ip_csynth` case is skipped automatically when it is not available).

7. Files changed
----------------

.. list-table::
   :header-rows: 1

   * - File
     - Edit
   * - ``allo/backend/ip.py``
     - Parser (Edit 1); ``input_idx``/``output_idx`` + ``has_stream_args`` (Edit 2); CPU fences (Edit 4)
   * - ``allo/ir/builder.py``
     - Type the stream call, attach ``stream_dirs`` (Edit 3)
   * - ``allo/dataflow.py``
     - Direction classifier accepts ``func.call`` (Edit 5)
   * - ``allo/passes.py``
     - Fence ``call_ext_libs_in_ptr`` (Edit 4)
   * - ``tests/ip_integration/vadd_stream.cpp``
     - Example stream IP
   * - ``tests/ip_integration/test_stream_ip.py``
     - Tests

.. _rtl-module-sim-shim:

Running an ``hls::stream`` IP under the CPU dataflow simulator
==============================================================

This document explains the change that lets a hand-written HLS C++ block ("IP")
whose interface uses ``hls::stream<T>`` ports run under Allo's **CPU dataflow
simulator** (``df.build(top, target="simulator")``), not only on the FPGA path.

It is a companion to :ref:`rtl-module-stream-ip`,
which describes how such an IP is integrated for ``vitis_hls``/``vivado_hls``. Read
that one first if you want the vocabulary; this one is self-contained enough to
follow on its own. It assumes no knowledge of MLIR or compilers — every piece of
jargon is explained the first time it appears.

**Before this change**

.. code-block:: python

   mod = df.build(top, target="simulator")
   # NotImplementedError: IP 'vadd_stream' has hls::stream<T> arguments, which are
   # only supported for the vitis_hls/vivado_hls targets...

**After**

.. code-block:: python

   mod = df.build(top, target="simulator")
   mod(A, B, C)          # runs on the CPU; C == A + B

1. Background: the three things you need to know
------------------------------------------------

1.1 What Allo's dataflow simulator is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allo lets you describe an accelerator as a **dataflow region**: a set of
concurrent kernels connected by FIFO channels.

.. code-block:: python

   @df.region()
   def top(A: int32[32], B: int32[32], C: int32[32]):
       sA: Stream[int32, 4]        # a FIFO channel of int32, depth 4

       @df.kernel(mapping=[1], args=[A])
       def feedA(a: int32[32]):
           for i in range(32):
               sA.put(a[i])        # blocking write

``target="simulator"`` compiles this to run **on your CPU** so you can check the
numerics without waiting for hardware synthesis. It does that by turning each
``@df.kernel`` into its own **OpenMP thread**. (OpenMP is the standard C/C++
threading runtime; "each kernel becomes an ``omp.section`` inside an
``omp.parallel``" simply means each kernel body gets its own thread that runs
concurrently with its peers.) The kernels really do run at the same time, so a
kernel that blocks waiting on a FIFO gets unblocked by a peer that is running
right now. This matters a lot below.

1.2 What a ``Stream`` becomes on the CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no hardware FIFO on a CPU, so the simulator builds one in memory: a
**ring buffer** (a fixed array plus two indices that wrap around).

For ``Stream[int32, 4]`` the simulator allocates this object (this is MLIR type
syntax; ``memref<5xi32>`` means "a 5-element array of 32-bit integers", and
``memref<i32>`` means "a single 32-bit integer in memory"):

.. code-block:: text

   !allo.struct< memref<5xi32>, memref<i32>, memref<i32> >
   //              data           head          tail
   //            5 = depth + 1   read index   write index

* ``data`` — the storage. Its length is ``depth + 1``, not ``depth``. The extra
  slot is what makes the two states unambiguous: ``head == tail`` means *empty*,
  and ``(tail + 1) % cap == head`` means *full*. Without the spare slot those two
  conditions would be the same test.
* ``head`` — the index the *consumer* reads from. Only the consumer advances
  it.
* ``tail`` — the index the *producer* writes to. Only the producer advances
  it.

Both start at 0. Because each index has exactly one writer, no lock is needed;
what *is* needed is careful ordering, which section 3 covers in detail.

Every ``sA.put(x)`` / ``sA.get()`` in your Allo code is rewritten into a fixed
sequence of loads, stores and memory fences over this ring buffer. That rewrite
lives in ``allo/backend/simulator.py``, function ``_process_function_streams``.

1.3 What an ``IPModule`` is
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``allo.IPModule`` wraps a hand-written C++ function so you can call it from an
Allo kernel:

.. code-block:: python

   vadd_stream = allo.IPModule(
       top="vadd_stream", impl="vadd_stream.cpp",
       link_hls=False, input_idx=[0, 1], output_idx=[2],
   )

Allo parses the C++ signature, and **compiles the source itself** — that last
part is the hinge this whole design turns on.

For CPU targets, Allo generates a small C++ **wrapper** around the IP,
``g++``-compiles wrapper + IP into a shared library (``.so``), and hands that ``.so``
to the JIT (just-in-time compiler) that runs the Allo module. The JIT then calls
the wrapper like any other function. That machinery is
``IPModule.generate_mlir_c_wrapper`` and ``IPModule.compile_shared_lib`` in
``allo/backend/ip.py``.

2. Why the CPU path could not already do this
---------------------------------------------

On the **FPGA path** it just works, and it is worth being precise about why: the
IP's ``A.read()`` and Allo's ``sA.get()`` both compile down to *Vitis's own*
``hls::stream`` primitive. Vitis owns the representation on both sides, so the two
halves meet in the middle automatically.

On the **CPU** there is no shared primitive:

.. list-table::
   :header-rows: 1

   * -
     - Allo's kernels
     - the IP
   * - written in
     - Python → MLIR
     - C++
   * - compiled by
     - LLVM JIT, in-process
     - ``g++``, into a ``.so``
   * - a stream is
     - the ring buffer of §1.2
     - whatever ``hls_stream.h`` says

Two different FIFO implementations in two different compilation units. A
``put`` from an Allo kernel would land in the ring buffer; a ``read()`` in the IP
would look in a Vitis ``hls::stream`` object that nobody ever fills. So stream IPs
were fenced off on the CPU with a clear ``NotImplementedError`` rather than left
to fail confusingly inside ``g++``.

3. The idea: a one-sided stream shim
------------------------------------

Allo compiles the IP from source. Therefore Allo controls which
``hls_stream.h`` the IP sees.

So: ship a *shim* ``hls::stream<T>`` whose ``read()`` and ``write()`` **are** the ring
buffer handshake, put its directory first on the include path, and hand it
Allo's ring buffers. The IP's body is not modified in any way — ``A.read()`` in
the unmodified IP now performs exactly the operation an Allo kernel's
``sA.get()`` performs, on exactly the same buffer.

.. code-block:: text

     ┌───────────┐   put     ┌──────────────────────┐   read()   ┌──────────────┐
     │ feedA     │──────────►│  ring buffer (sA)    │◄───────────│ vadd_stream  │
     │ (Allo,    │           │  data / head / tail  │            │ (IP, g++,    │
     │  JIT'd)   │           │  in shared memory    │            │  shim header)│
     └───────────┘           └──────────────────────┘            └──────────────┘
        thread 1                  one buffer, no copies              thread 3

Four properties of this design are worth stating explicitly, because each one is
a decision that could have gone otherwise:

* **One-sided.** Only the IP side is adapted. Allo's kernels, its MLIR lowering,
  and the ring-buffer format are untouched — which means the shim cannot
  possibly regress the existing simulator.
* **Zero-copy.** The shim does not own a FIFO and does not marshal data between
  two FIFOs. It holds *pointers into Allo's buffer*. There is exactly one queue,
  so there is no "who has the real data" question and no extra latency.
* **Simulator-only.** The shim is used for ``target="simulator"`` and nothing
  else. The plain ``llvm`` target runs the kernels **sequentially**, one call
  after another; an IP that blocks waiting for a FIFO that a not-yet-run kernel
  will fill would hang forever. That path keeps its ``NotImplementedError``. The
  same reasoning applies to Vitis ``csim``.
* **The IP must be a concurrent process.** For the same reason, the IP has to be
  called from inside its own ``@df.kernel``, so the simulator gives it its own
  thread. This is checked and reported (§6), not left to deadlock.

4. The protocol, line by line
-----------------------------

This is the part that must be exactly right. A ring buffer shared by two threads
is a classic place for a data race that shows up as "works 999 times, corrupts
once". The shim is therefore a **literal translation** of what
``allo/backend/simulator.py`` emits, not an independent reimplementation.

File: ``allo/backend/ip_sim/allo_fifo.h``.

.. code-block:: c

   template <typename T> struct AlloFifo {
     T *data;        // cap slots
     int32_t cap;    // depth + 1
     int32_t *head;  // read index  — only the consumer advances it
     int32_t *tail;  // write index — only the producer advances it
   };

4.1 ``put`` (what the IP's ``write()`` calls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Line numbers are ``allo/backend/simulator.py`` as of this change.

.. list-table::
   :header-rows: 1

   * - step
     - shim (``allo_fifo_put``)
     - simulator.py
     - why
   * - 1
     - ``__atomic_thread_fence(SEQ_CST)``
     - ``openmp_d.FlushOp`` — line 870
     - Before reading ``head`` we must see the consumer's latest value, not one cached in this thread's registers/cache.
   * - 2
     - load ``tail``
     - ``memref_d.LoadOp`` — line 871
     - We are the only producer, so nobody else can move ``tail``.
   * - 3
     - ``next = (tail + 1) % cap``
     - ``AddIOp`` + ``RemUIOp`` — lines 875–882
     - The slot we would publish.
   * - 4
     - ``while (head == next) backoff();``
     - ``scf.while``, condition at lines 915–920, re-fence each iteration at line 903
     - ``head == next`` is the FULL test. The fence *inside* the loop is essential: without it the compiler is free to hoist the load of ``head`` out of the loop and spin forever on a stale value.
   * - 5
     - ``data[tail] = value;``
     - ``memref_d.StoreOp`` — lines 989–994
     - **Before** the publish.
   * - 6
     - ``__atomic_store_n(tail, next, SEQ_CST)``
     - ``omp.critical`` + store, lines 996–999, later converted to ``omp.atomic.write`` by ``convert_critical_write_to_atomic_write`` (line 1783, called at line 1871)
     - This single store is what makes the element visible. Atomic so the consumer can never observe a half-written index.
   * - 7
     - ``__atomic_thread_fence(SEQ_CST)``
     - ``openmp_d.FlushOp`` — line 1000
     - Trailing fence, mirroring the emitted code.

**The ordering rule, stated plainly:** step 5 must happen before step 6. The
consumer decides "there is an element" purely by looking at ``tail``. If ``tail``
became visible first, the consumer could read a slot that has not been written
yet — it would read stale garbage, and no test would reliably catch it.

4.2 ``get`` (what the IP's ``read()`` calls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact mirror image:

.. list-table::
   :header-rows: 1

   * - step
     - shim (``allo_fifo_get``)
     - simulator.py
     - why
   * - 1
     - fence
     - line 869 (shared prologue)
     - See the producer's latest ``tail``.
   * - 2
     - load ``head``, compute ``next``
     - lines 885–895
     - We are the only consumer.
   * - 3
     - ``while (head == tail) backoff();``
     - condition at lines 1002–1007, fence at 903
     - ``head == tail`` is the EMPTY test.
   * - 4
     - ``value = data[head];``
     - ``memref_d.LoadOp`` — lines 1062–1065
     - **Before** the publish.
   * - 5
     - ``__atomic_store_n(head, next, SEQ_CST)``
     - ``omp.critical`` + store — lines 1088–1090
     - Releases the slot.

**The mirrored ordering rule:** step 4 must happen before step 5. The producer
decides "that slot is free" purely by looking at ``head``. If ``head`` were
published first, the producer could overwrite the slot while we are still
reading it.

The MLIR ``get`` path emits no trailing ``omp.flush``; the shim performs one anyway,
purely for symmetry with ``put``. A sequentially-consistent store already implies
a full fence, so this is not an extra ordering constraint, just an explicit one.

4.3 Why sequential consistency, and why ``usleep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``__ATOMIC_SEQ_CST`` is the strongest and simplest ordering C++ offers: every
  thread observes all seq-cst operations in one consistent global order. It is
  what ``omp.flush`` + ``omp.atomic.write`` give on the MLIR side, so using it keeps
  the two sides literally equivalent. A weaker acquire/release pairing would be
  sufficient in theory and faster, but this is a *simulator* — matching the
  reference exactly is worth far more than the cycles.
* ``usleep(1)`` in the spin loop mirrors the ``usleep(1)`` the simulator injects
  (line 913). Its job is to stop a blocked kernel from burning a core at 100%
  and starving the very peer it is waiting for — which, with more kernels than
  cores, is the difference between "slow" and "hangs".
* ``#pragma omp taskyield`` (line 908 in the emitted code) is present in the
  shim but compiled only when ``ALLO_IP_SIM_OPENMP=1``. It is a *scheduling hint*
  with no bearing on correctness. It is off by default because ``g++`` would link
  the IP against GNU's ``libgomp`` while the JIT'd Allo code uses LLVM's ``libomp``,
  and hosting two OpenMP runtimes in one process is unsafe. ``usleep(1)`` does the
  actual yielding.

4.4 The shim header itself
~~~~~~~~~~~~~~~~~~~~~~~~~~

``allo/backend/ip_sim/hls_stream.h`` is a thin ``namespace hls`` wrapper over those
operations:

.. code-block:: cpp

   template <typename T, int DEPTH = 0> class stream {
   public:
     explicit stream(AlloFifo<T> *fifo) : fifo_(*fifo), owned_(nullptr) {}  // bound to Allo's buffer
     T    read()                  { return allo_fifo_get(fifo_); }
     void write(const T &value)   { allo_fifo_put(fifo_, value); }
     bool read_nb(T &value)       { return allo_fifo_try_get(fifo_, &value); }
     bool write_nb(const T &value){ return allo_fifo_try_put(fifo_, value); }
     bool empty() const;  bool full() const;  std::size_t size() const;
     // plus the free operators `is >> value` and `os << value`
   };

It covers the operations an IP performs on a *port*; it is not a full
reimplementation of Vitis's class. Two details:

* It is non-copyable, matching Vitis — and also because copying a FIFO *view*
  would silently duplicate a queue's state.
* A stream the IP declares **internally** (one Allo never passed in, so no Allo
  buffer exists) gets its own heap ring buffer of
  ``ALLO_SIM_LOCAL_STREAM_DEPTH`` slots, so such an IP still compiles and runs.

5. The ABI: how Allo's FIFO reaches C++
---------------------------------------

"ABI" (application binary interface) is just: what the arguments physically look
like when one compiled function calls another. This section is what lets the
generated wrapper be written mechanically instead of guessed at.

5.1 What the JIT actually passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After lowering, an Allo stream argument is a pointer to the
``!allo.struct<memref<5xi32>, memref<i32>, memref<i32>>`` from §1.2. The struct is
**not** passed as a struct pointer: MLIR unpacks each ``memref`` into plain
scalars, a so-called *memref descriptor*.

* ``memref<Nxi32>`` → ``(allocated_ptr, aligned_ptr, offset, size, stride)``, with
  element ``i`` living at ``aligned_ptr[offset + i]``.
* ``memref<i32>`` (rank 0) → ``(allocated_ptr, aligned_ptr, offset)``, value at
  ``aligned_ptr[offset]``.

Note ``cap`` is simply the data memref's ``size`` field.

5.2 Why the wrapper does not hand-match that layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than reconstructing five scalars per memref by hand — fragile, and it
would silently rot if MLIR's layout ever changed — the wrapper uses the **same
mechanism the existing memref-IP path already uses**: it declares each argument
as an *unranked* memref (``memref<*xi32>``) on the MLIR side, which crosses into
C++ as a ``(int64_t rank, void *descriptor)`` pair, and then hands that pair to
``DynamicMemRefType`` from ``mlir/ExecutionEngine/CRunnerUtils.h``. That class is
MLIR's own descriptor reader; it knows the layout so we don't have to.

Each stream therefore becomes **three** unranked-memref arguments (data, head,
tail). A three-stream IP has nine MLIR operands, as seen below.

5.3 Before and after, in MLIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the rewrite — the IP is declared with Allo stream types, which no CPU ABI
can express:

.. code-block:: text

   func.func private @vadd_stream(!allo.stream<i32, 4>, !allo.stream<i32, 4>, !allo.stream<i32, 4>)

   func.func @ip_wrap_0(%arg0: !allo.stream<i32, 4>, ...) attributes {df.kernel, ...} {
     call @vadd_stream(%arg2, %arg0, %arg1) {stream_dirs = "iio"} : (...) -> ()

After — the declaration and the call site now speak the wrapper's ABI:

.. code-block:: text

   func.func private @pyvadd_stream_1785273214135020487(
       memref<*xi32>, memref<*xi32>, memref<*xi32>,   // stream 0: data, head, tail
       memref<*xi32>, memref<*xi32>, memref<*xi32>,   // stream 1
       memref<*xi32>, memref<*xi32>, memref<*xi32>)   // stream 2

   func.func @ip_wrap_0(%arg0: memref<!allo.struct<memref<5xi32>, memref<i32>, memref<i32>>>, ...) {
     call @pyvadd_stream_1785273214135020487(%cast, %cast_0, ..., %cast_7) : (...) -> ()

Note the kernel's arguments are now FIFO structs — the simulator retyped them —
and each ``%cast`` is one field of one struct, cast to an unranked memref.

5.4 The generated wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~

For each stream port the wrapper emits (abridged, three ports in the real file):

.. code-block:: cpp

   extern "C" __attribute__((visibility("default")))
   void pyvadd_stream_1785273214135020487(
       int64_t s0_data_rank, void *s0_data_ptr, /* head, tail, then s1, s2 ... */) {
     UnrankedMemRefType<int32_t> s0_data_u = {s0_data_rank, s0_data_ptr};
     DynamicMemRefType<int32_t>  s0_data(s0_data_u);
     /* ... same for s0_head, s0_tail ... */
     assert(s0_data.rank == 1 && "Allo FIFO storage must be a 1-D memref");
     assert(s0_data.strides[0] == 1 && "Allo FIFO storage must be contiguous");

     AlloFifo<int32_t> s0_fifo;
     s0_fifo.data = s0_data.data + s0_data.offset;
     s0_fifo.cap  = (int32_t)s0_data.sizes[0];      // = depth + 1
     s0_fifo.head = s0_head.data + s0_head.offset;
     s0_fifo.tail = s0_tail.data + s0_tail.offset;
     hls::stream<int32_t> s0(&s0_fifo);
     /* ... */
     vadd_stream(s0, s1, s2);
   }

Array and scalar ports are emitted exactly as ``generate_mlir_c_wrapper`` emits
them, so an IP may freely mix array, scalar and stream ports.

6. The changes, file by file
----------------------------

Edit 1 — the shim headers — ``allo/backend/ip_sim/`` *(new)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``allo_fifo.h`` (the protocol of §4) and ``hls_stream.h`` (the ``namespace hls``
wrapper of §4.4). New directory, so nothing existing can be affected by it.

``IP_SIM_INCLUDE_DIR`` in ``allo/backend/ip.py`` points at it, and
``compile_shared_lib`` puts it **first** with ``-I``, so the IP's
``#include <hls_stream.h>`` resolves to the shim even when ``link_hls=True`` also
put Vitis's include directory on the list.

Edit 2 — the wrapper generator — ``allo/backend/ip.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``generate_stream_sim_wrapper()`` — emits the wrapper of §5.4. Named alongside
  the existing ``generate_mlir_c_wrapper()`` and deliberately built the same way
  (``UnrankedMemRefType`` → ``DynamicMemRefType``) so there is one descriptor-reading
  idiom in the codebase, not two.
* ``stream_element_type()`` / ``split_template_args()`` — pull ``T`` out of
  ``hls::stream<T>`` / ``hls::stream<T, DEPTH>`` as written in C++. The optional
  ``DEPTH`` argument is dropped: on the CPU the depth that exists is the one the
  Allo ``Stream`` declaration allocated.
* ``stream_arg_indices`` — positions of the stream ports, in order.
* ``compile_shared_lib(stream_sim=False)`` — the ``stream_sim=True`` flavour adds
  the shim include directory (first), ``-Wno-unknown-pragmas`` (the IP keeps its
  ``#pragma HLS ...`` lines, which are hardware directives that ``g++`` neither
  knows nor needs), and, under ``ALLO_IP_SIM_OPENMP=1``, ``-fopenmp``.
* ``_reject_stream_on_cpu()`` — message updated: the simulator is now supported;
  the plain ``llvm`` target and ``csim`` still are not, and the message now says
  *why* (they call the IP once, sequentially).

Edit 3 — symbol visibility — ``allo/backend/ip.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compile_shared_lib`` now compiles with ``-fvisibility=hidden``, and both
wrapper generators mark the entry point
``__attribute__((visibility("default")))`` (the ``_EXPORT_ATTR`` constant).

This one is worth spelling out, because it was a real, silent deadlock found
during verification, not a precaution.

The wrapper entry point carries a per-instance hash (``pyvadd_stream_<hash>``), so
those never collide. But the **IP's own top function keeps the name the user
wrote** — ``vadd_stream`` — and it was exported from every ``.so`` built from it:

.. code-block:: text

   $ nm -D --defined-only libpyvadd_stream_<hashA>.so
   0000000000001283 T pyvadd_stream_<hashA>
   000000000000121c T vadd_stream          ← global

When two IPModules built from *different* sources but sharing a top name end up
in one process (two tests in one pytest run — exactly what
``test_stream_ip_sim.py`` does), the dynamic linker resolves a global symbol to
the definition it loaded **first**. So the second wrapper called the *first*
IP's body. In the test suite that meant an IP compiled for 256 elements ran a
body that consumed 32 and returned; the feeders then blocked forever on a full
FIFO and the whole run hung, with no error message anywhere.

Hiding everything but the entry point makes each wrapper's call bind inside its
own ``.so``:

.. code-block:: text

   $ nm -D --defined-only libpyvadd_stream_<hashA>.so
   0000000000001203 T pyvadd_stream_<hashA>

This is applied to both wrapper flavours because the hazard is identical for
memref IPs; it changes no behaviour other than removing the interposition.

Edit 4 — let the simulator through the CPU fence — ``allo/passes.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``call_ext_libs_in_ptr`` rewrites each IP call into a call through unranked-memref
pointers. A stream port has no such representation, so this function used to
reject any stream IP outright.

It now takes ``allow_stream_ip=False``. The simulator passes ``True``, which makes
this pass **skip stream IPs entirely** (they are excluded from ``lib_map`` and
their declaration is left standing) and leave them to ``backend/simulator.py``,
which runs later and knows what ring buffer each stream became. The plain ``llvm``
target still passes ``False`` and still raises — with a message that now explains
the sequential-execution reason.

One consequential detail: because a stream IP's *declaration* is now
deliberately left in the module, the loop below it can encounter an external
function, which has no body. The guard became
``isinstance(op, func_d.FuncOp) and not op.is_external``.

Edit 5 — declare the wrapper — ``allo/backend/simulator.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``declare_stream_ip_wrappers()`` replaces
``func.func private @vadd_stream(!allo.stream<...>, ...)`` with
``func.func private @pyvadd_stream_<hash>(memref<*xi32>, ...)`` — the §5.3
"after". ``_plan_stream_ip_wrapper()`` computes both the per-IP-argument plan
(``stream`` / ``memref`` / ``scalar``) and the flattened MLIR operand list.

Only the *declaration* changes here. The call sites cannot be rewritten yet: at
this point the streams are still ``!allo.stream``, and the ring buffers do not
exist.

Edit 6 — rewrite the call sites — ``allo/backend/simulator.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_lower_stream_ip_calls()`` runs from inside ``_process_function_streams``, at the
exact point where each kernel's stream arguments have just been retyped to FIFO
structs and ``arg_stream_table`` (which block argument corresponds to which
stream) is known. Placement is the whole trick: a moment earlier there is no
ring buffer to point at, a moment later the mapping is gone.

For each operand of the call it emits, per stream: an ``affine.load`` of the FIFO
struct, three ``allo.struct_get``\ s (data / head / tail), and a ``memref.cast`` of
each to an unranked memref — the same cast ``call_ext_libs_in_ptr`` uses for array
arguments. Array operands are cast the same way; scalars pass through untouched.

It also enforces the constraints, with messages that say what to do:

* the call must be inside a ``@df.kernel`` (checked via the ``df.kernel`` attribute)
  — otherwise the IP does not get its own thread and would deadlock;
* each stream operand must be a stream passed into that kernel;
* the stream's element type must match the IP's declared element type — the IP
  writes Allo's buffer *in place*, so a mismatch is not convertible, it is a
  type error (raised as ``TypeError``);
* the stream's elements must be scalars.

``_check_no_unlowered_stream_ip_calls()`` then sweeps the module for any call to a
stream IP the rewrite did not reach, and fails loudly. Without it, an unhandled
shape (say, a call in a helper function) would reach the LLVM lowering as a call
to a symbol that no longer exists — a confusing crash far from the cause.

Edit 7 — wire it into the build — ``allo/backend/simulator.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``build_dataflow_simulator(module, top_func_name, ext_libs=None)`` — takes the
  IP list, builds the ``stream_ips`` map and calls Edit 5, then threads the plans
  through ``_process_function_streams`` so Edit 6 can run at the right moment.
* ``_process_function_streams`` also learns to **skip** stream IPs when collecting
  "PE calls": an IP is an opaque external function, not a processing element, and
  it has no body to walk into.
* ``LLVMOMPModule.__init__`` passes ``allow_stream_ip=True`` to ``call_ext_libs_in_ptr``,
  passes ``ext_libs`` to ``build_dataflow_simulator``, and compiles each stream IP
  with ``compile_shared_lib(stream_sim=True)`` (others unchanged), adding the
  resulting ``.so`` to ``shared_libs`` for the JIT.

7. How to use it
----------------

**The IP** (``vadd_stream.cpp``) — ordinary HLS, unmodified:

.. code-block:: cpp

   #include <hls_stream.h>
   #include <stdint.h>

   extern "C" {
   void vadd_stream(hls::stream<int32_t> &A, hls::stream<int32_t> &B,
                    hls::stream<int32_t> &C) {
     for (int i = 0; i < 32; ++i) {
   #pragma HLS pipeline II = 1
       int32_t a = A.read();
       int32_t b = B.read();
       C.write(a + b);
     }
   }
   }

**The Allo program** — the IP gets its own kernel, between feeders and a drain:

.. code-block:: python

   import allo, numpy as np
   import allo.dataflow as df
   from allo.ir.types import int32, Stream

   N = 32
   vadd_stream = allo.IPModule(
       top="vadd_stream", impl="vadd_stream.cpp",
       link_hls=False,                 # the shim replaces Vitis's header,
       input_idx=[0, 1], output_idx=[2],   # so vitis_hls need not be installed
   )

   @df.region()
   def top(A: int32[N], B: int32[N], C: int32[N]):
       sA: Stream[int32, 4]
       sB: Stream[int32, 4]
       sC: Stream[int32, 4]

       @df.kernel(mapping=[1], args=[A])
       def feedA(a: int32[N]):
           for i in range(N):
               sA.put(a[i])

       @df.kernel(mapping=[1], args=[B])
       def feedB(b: int32[N]):
           for i in range(N):
               sB.put(b[i])

       @df.kernel(mapping=[1])          # its own kernel => its own thread
       def ip_wrap():
           vadd_stream(sA, sB, sC)

       @df.kernel(mapping=[1], args=[C])
       def drain(c: int32[N]):
           for i in range(N):
               c[i] = sC.get()

   mod = df.build(top, target="simulator")
   a = np.random.randint(-1000, 1000, N).astype(np.int32)
   b = np.random.randint(-1000, 1000, N).astype(np.int32)
   c = np.zeros(N, dtype=np.int32)
   mod(a, b, c)
   np.testing.assert_array_equal(c, a + b)

Run it with at least as many OpenMP threads as there are kernels:

.. code-block:: bash

   export OMP_NUM_THREADS=8
   python your_script.py

If a kernel count exceeds ``OMP_NUM_THREADS``, two kernels share a thread and
run one after the other — and a blocking IP will hang. This is a property of the
simulator, not of the shim, but stream IPs make it much easier to hit.

Things that are (deliberately) rejected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - you write
     - you get
   * - ``df.build(top, target="llvm")`` with a stream IP
     - ``NotImplementedError``: sequential execution can never satisfy a blocking read
   * - the IP called outside a ``@df.kernel``
     - ``NotImplementedError``: "Wrap the call in its own @df.kernel"
   * - ``hls::stream<int8_t>`` port fed by ``Stream[int32, 4]``
     - ``TypeError``: "The element types must match"
   * - a stream of arrays
     - ``NotImplementedError``: the shim supports streams of scalars
   * - ``vadd_stream.generate_mlir_c_wrapper()`` (memref path)
     - ``NotImplementedError``, unchanged

8. Verification
---------------

Environment used (``zhang-21``):

.. code-block:: bash

   export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8
   export PATH="$LLVM_BUILD_DIR/bin:$PATH"
   export OMP_NUM_THREADS=8

New tests — ``tests/ip_integration/test_stream_ip_sim.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda run -n allo python -m pytest tests/ip_integration/test_stream_ip_sim.py -q
   # 5 passed in 3.56s

.. list-table::
   :header-rows: 1

   * - test
     - what it pins down
   * - ``test_stream_ip_sim``
     - the end-to-end numeric result — this is the case that raised ``NotImplementedError`` before
   * - ``test_stream_ip_sim_repeated_runs``
     - a run leaves every FIFO with ``head == tail``, so the module is reusable
   * - ``test_stream_ip_sim_backpressure``
     - 256 elements through depth-2 FIFOs: both spin paths hit thousands of times
   * - ``test_stream_ip_sim_element_type_mismatch``
     - the type check fires
   * - ``test_stream_ip_sim_wrapper_shape``
     - the generated wrapper includes the shim before the IP and binds all three fields per port

Concurrency soak
~~~~~~~~~~~~~~~~

Deliberately tiny FIFOs and long runs, each configuration in its own process:

.. code-block:: bash

   # N=2048 depth=1 x20 iterations   -> SOAK OK
   # N=4096 depth=2 x10 iterations   -> SOAK OK
   # N=512  depth=8 x40 iterations   -> SOAK OK

``depth=1`` (a two-slot ring) is the most contended case possible: nearly every
``put`` blocks on FULL and nearly every ``get`` blocks on EMPTY. All values matched
``a + b`` exactly on every iteration.

This soak is what surfaced the symbol-interposition deadlock of Edit 3 — it
reproduced only when several IP ``.so``\ s coexisted in one process.

Regressions
~~~~~~~~~~~

.. code-block:: bash

   conda run -n allo python -m pytest tests/ip_integration/test_external.py \
                                      tests/ip_integration/test_stream_ip.py -q
   # 10 passed, 2 skipped in 93.59s      (2 skipped: vitis_hls not on PATH)

   conda run -n allo python tests/dataflow/test_df_unit.py
   # Dataflow Simulator Passed! / Dataflow Simulator Passed! /
   # Dataflow Simulator (Arithmetic) Passed!

   conda run -n allo python tests/dataflow/test_region_stateful.py
   # exit 0

``test_stream_ip.py``'s rejection test was rewritten (it previously asserted the
simulator refuses stream IPs, which is precisely what changed) into
``test_stream_ip_sequential_cpu_paths_rejected``, which asserts the *remaining*
fences — ``generate_mlir_c_wrapper``, ``generate_nanobind_wrapper``, and
``call_ext_libs_in_ptr`` with ``allow_stream_ip=False`` — still raise.

The FPGA codegen tests are unchanged and still pass; the csynth tests skip
because ``vitis_hls`` is not on this shell's PATH.

Lint
~~~~

``black`` and ``pylint --rcfile=./scripts/lint/pylintrc`` report nothing on the new
or changed code. (``allo/backend/simulator.py`` has pre-existing ``black`` and
``pylint`` findings elsewhere in the file, all outside the changed hunks; they were
left alone rather than reformatted into this diff.)

9. Limitations and where to look next
-------------------------------------

* **Scalar element types only.** ``_c_type_to_mlir`` maps the plain C scalar types
  Allo already knows (``allo/utils.py: c2allo_type``). An HLS type such as
  ``ap_int<8>`` has no CPU representation here and is refused.
* **Streams of arrays are not supported** (``stream_type.rank != 1`` is rejected).
  Allo itself can build such FIFOs; the shim's element type would have to become
  an array view.
* **One kernel per IP call.** The IP must sit in its own ``@df.kernel``. Calling
  it directly in the ``@df.region`` body remains unsupported, as on the FPGA path.
* **Depth comes from Allo.** A ``hls::stream<T, DEPTH>`` port's ``DEPTH`` is ignored;
  the depth that exists on the CPU is the one the ``Stream[...]`` declaration
  allocated. On the FPGA path Vitis would honour the port's depth, so a design
  that depends on a deeper IP-side buffer can behave differently between the two
  targets.
* ``ALLO_IP_SIM_OPENMP=1`` compiles the ``taskyield`` hint into the shim. It is
  off by default (two OpenMP runtimes in one process); turn it on only when
  investigating scheduling behaviour.

File map
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - file
     - role
   * - ``allo/backend/ip_sim/allo_fifo.h``
     - the ring-buffer protocol in C (§4)
   * - ``allo/backend/ip_sim/hls_stream.h``
     - the shim ``hls::stream<T>`` (§4.4)
   * - ``allo/backend/ip.py``
     - wrapper generation, include path, ``-fvisibility=hidden`` (Edits 2–3)
   * - ``allo/passes.py``
     - ``call_ext_libs_in_ptr(..., allow_stream_ip)`` (Edit 4)
   * - ``allo/backend/simulator.py``
     - declaration swap, call-site rewrite, build wiring (Edits 5–7)
   * - ``tests/ip_integration/test_stream_ip_sim.py``
     - the simulator-path tests (§8)
   * - :ref:`rtl-module-stream-ip`
     - the FPGA-path counterpart

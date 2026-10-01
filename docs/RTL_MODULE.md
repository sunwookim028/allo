<!-- Copyright Allo authors. All Rights Reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RTLModule: simulation and Vitis black-box integration

`allo.RTLModule` calls Verilog/SystemVerilog IP from an Allo kernel. It extends
and integrates the earlier `allo_rtl.py` prototype. No LLM, API key, or wrapper
generation service is involved.

The simulator runs a Verilator model through a generated C++ transactor. The
Vitis backend emits a black-box description and registers a supplied or
automatically generated RTL wrapper. HLS synthesizes the surrounding Allo computation, preserving the
external call as hardware implemented by that wrapper.

## Supported paths

| Binding | Allo software simulation | Vitis black-box export |
|---|---|---|
| `Port(bind="stream")` | Dataflow simulator; ready/valid or active-high FIFO | Supplied FIFO/chain wrapper or generated ready/valid adapter |
| `Port(bind="array")` | LLVM; array traversed through ready/valid | Not yet supported |
| `MemPort` | LLVM or mixed dataflow simulation; host array acts as RAM | Not yet supported |

Payloads are `bool`, signed/unsigned 8-, 16-, and 32-bit integers. Floating-point,
wide packed values, AXI interfaces, multiple clocks, and overlapping transactions
are not supported. Array buffers crossing dataflow kernel boundaries retain the
existing compiler limitations; the array regression uses the LLVM path. C `int` and `unsigned int` are 32-bit aliases. Stream payloads
must be scalar; directions and signedness must match the Allo declarations.

The first synthesis implementation targets the Vitis 2023.2 black-box JSON/Tcl
flow. Vendor synthesis, co-simulation, and export require manual validation on
your installation. Emitting a project successfully is not proof of hardware
correctness. Other HLS vendors and Vitis `hw`/emulation builds are rejected.

## Supplied-wrapper example

See `examples/ip_integration/rtl_accumulator.py`, `.v`, and `.cpp`. The supplied
RTL accepts one integer per invocation and returns a running sum. It demonstrates
state across repeated calls inside one Allo kernel and bounded FIFO backpressure.

```python
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
```

`top` names the wrapper, not an unadapted inner IP. Include the original IP's
sources in `rtl` if the wrapper instantiates it. The supplied C model implements
the same ordered signature for Vitis C simulation and C/RTL comparison. It must
be self-contained apart from standard/HLS headers. It is not used by Allo's
Verilator simulation, which runs the wrapper itself. Model correctness remains
the author's responsibility, checked by vendor co-simulation.

## Automatic ready/valid wrappers (Milestone 2)

Use `HLSBlackBox(adapter=ReadyValidAdapter(...))` to generate the hardware glue.
Your `Port` descriptors now refer to the **original IP**, not HLS-facing pins.
No LLM or handwritten Verilog wrapper is needed for this supported contract.

```python
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
```

The example is `examples/ip_integration/rtl_adapter.py`, with the original RTL
and C model in `ready_valid_accumulator.v` and `.cpp`. Each call consumes four
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
  asserted until `ap_continue` is sampled on an enabled edge.
- Clock-enable and reset-polarity adaptation. No generated/gated clock is used.

The **same generated wrapper** is compiled by Verilator for Allo simulation and
packaged for Vitis. `validate_rtl()` checks both the original and generated top.
The wrapper defaults to `<original_top>_allo`; `name=` selects a different name,
which must differ from the original top. Generated boundary names are standard
`ap_*` controls and numbered `p0_data`, `p0_read`, etc.; the JSON records them.
Different wrappers around the same source set share the staged core definitions.
The generated build is cached on the RTLModule object; construct a new object
after changing source files or interface descriptions.

### Required adapter contract

- One clock; scalar ready/valid stream arguments with a positive `size` for
  **every** input and output. Per-port counts may differ.
- An active-high core clock-enable input (`clock_enable="ce"` above). When it is
  low, all core state and transfers must stop. The adapter freezes the core
  outside active transactions; reset overrides that enable so synchronous reset
  still works. A core without clock-enable support needs a custom wrapper.
- `persistent=True`. Internal state survives calls; system reset clears it.
- `start_mode="none"` requires `RTLModule(start=None)` for a free-running core.
  `start_mode="pulse"` requires a named core start pin and generates one enabled
  launch cycle. It does not implement an arbitrary start/ready negotiation.
- `completion="counts"` (default) finishes after all declared input transfers
  into the core and output transfers to HLS complete. Choose this only if the
  core is quiescent after those transfers; hidden work must not remain.
- `completion="done"` additionally waits for the named `RTLModule(done=...)`
  signal. A pulse during the run phase is latched, so a late output drain cannot
  lose it. `done` must describe the current transaction after launch, not idle.
- `ii=0`: invocations do not overlap. `latency` must describe the **complete
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
The C model body must remain visible with `__SYNTHESIS__` defined: Vitis needs
it to extract black-box information. The JSON/Tcl black-box registration selects
the RTL implementation; do not hide the model body behind a synthesis guard.
The generated black-box declaration uses C++ linkage to preserve stream type
information for Vitis. Define the supplied model with ordinary C++ linkage,
without `extern "C"`; the surrounding Allo kernel can retain C linkage.

### Run and inspect

```sh
conda activate allo
export OMP_NUM_THREADS=8
python examples/ip_integration/rtl_adapter.py --wrapper /tmp/adapter.v --simulate
python examples/ip_integration/rtl_adapter.py --project /tmp/allo_rtl_milestone2.prj
# Manual vendor validation, from the generated project:
cd /tmp/allo_rtl_milestone2.prj
vitis_hls -f validate.tcl
```

The Vitis backend emits array element counts directly in the top-level AXI
memory interface pragmas for co-simulation. This example uses depth 8 for each
array. Multidimensional arrays use the product of their static dimensions.

In the local shell, the existing `allo()` helper initializes conda, Vitis HLS
2023.2, LLVM, and XRT. The Python library does not source shell setup scripts or
modify that configuration. `VERILATOR` can point to a binary in another conda
environment; `MLIR_INCLUDE_DIR` is optional when LLVM discovery succeeds.

For interfaces outside this generator's contract, use Milestone 1's supplied
wrapper route: keep `adapter=None`, describe its FIFO pins with
`protocol="ap_fifo"`, and pass its chain-control configuration. This is also the
extension boundary for any future external wrapper-generation tool.

## Port and control contracts

`Port(name, data, valid, ready, dir="in", ctype="int32_t", bind="stream",
size=None, protocol="ready_valid")` preserves the prototype's positional API.
`dir` is relative to the IP. `size` is a per-invocation token count, **not FIFO
depth**. Allo's `Stream[T, depth]` specifies FIFO depth.

For `protocol="ap_fifo"`, use:

| Direction | `data` | `valid` | `ready` |
|---|---|---|---|
| Input to IP | `dout` | `empty_n` | `read` |
| Output from IP | `din` | `write` | `full_n` |

These are active-high availability signals and zero-latency/front-visible FIFO
data. A synchronous FIFO that returns data a cycle after a read needs a supplied
adapter. Setting the protocol label does not generate that adapter.

`MemPort(name, size, ctype, addr, ce, q=None, we=None, d=None)` models word-addressed
RAM: reads return data after the requesting edge; writes commit at the edge.
`we` and `d` must appear together. Addresses must fit the declared size, and
out-of-bounds accesses abort simulation. Byte enables, dual ports, and alternate
read latencies need future descriptors/adapters.

For synthesis with `adapter=None`, the supplied wrapper must expose:

- `clock` and active-high `reset` (defaults `ap_clk`, `ap_rst`).
- `start` and `done` (set `done="ap_done"` explicitly).
- `HLSBlackBox.ready`, `.idle`, `.continue_`, `.clock_enable` (defaults
  `ap_ready`, `ap_idle`, `ap_continue`, `ap_ce`).
- FIFO ports for every logical argument.

The wrapper must honor clock enable, accept start according to the chain
protocol, and hold completion until continue. `persistent=True` is required:
hardware resets at system reset, not on every function call. `ii=0` explicitly
marks the initial implementation as non-pipelined. Supply the true unstalled
latency; it is not inferred or validated from port names. Backpressure can extend
elapsed time. Resources are not estimated by Allo.

The transactor resets for eight cycles. Without `done`, simulation ends when all
output ports reach their declared sizes. Non-persistent input streams require
sizes to bound prefetch. Persistent stream transactors retain pending input tokens
across calls. A prolonged lack of progress aborts with an RTL stall diagnostic.
This is functional integration simulation, not a globally synchronized cycle
simulation of every Allo kernel. Simulator FIFO capacity includes any transactor
holding registers and need not match HLS timing.

## Instances and state

Use one static call site per RTLModule object. A runtime loop around that call
is supported. Multiple static call sites, including replicated mapped kernels,
currently require distinct RTLModule objects and distinct `name=` values. The
backend emits structural aliases when the C name differs from the RTL top.

Vitis 2023.2 rejects `ap_ctrl_chain` black boxes inside pipeline regions.
Projects containing RTLModule therefore emit `config_compile -pipeline_loops 0`
to disable automatic loop pipelining throughout the solution. Explicit pipeline
directives elsewhere remain effective, but do not pipeline a loop or function
containing an RTLModule call. This conservative setting can reduce performance
of other loops that previously relied on automatic pipelining.

Persistent simulator state belongs to the compiled model on its executing thread.
Do not depend on state surviving a change of simulator worker thread or rebuild.
Independent modules compile into isolated libraries. Automatic shared-instance
arbitration and overlapping calls are outside this milestone.

## Build and validation

Activate the Allo environment first. For software simulation install Verilator
and a C++17 compiler. Set `VERILATOR` to its executable if it is not on PATH.
Configure `LLVM_BUILD_DIR` as required by the existing Allo simulator.
`MLIR_INCLUDE_DIR` may specify the directory containing
`mlir/ExecutionEngine/CRunnerUtils.h`; otherwise Allo attempts LLVM discovery.
The optional `verilator=` and `mlir_include=` arguments provide per-module values.

```sh
conda activate allo
export OMP_NUM_THREADS=8
python examples/ip_integration/rtl_accumulator.py --validate-rtl --simulate
python examples/ip_integration/rtl_accumulator.py --project /tmp/rtl_accumulator.prj
```

Construction and project emission do not run Verilator or Vitis. `validate_rtl()`
explicitly elaborates RTL with Verilator, checking top-level pin names, widths,
directions, and unbound inputs. Simulation invokes this automatically. This
structural check does not prove protocol compliance.

The emitted package contains a typed header, C model, original RTL,
optional instance alias, and `blackbox.json`; `run.tcl` registers it with
`add_files -blackbox`. Sources use project-relative paths; identical RTL source sets share a content-addressed directory. Run tools from the
project directory. Source dependencies must be listed explicitly; synthesis
requires self-contained RTL. Supplied wrappers resolve their own parameters and
preprocessor configuration; generated adapters support explicit integer parameters. Simulation supports integer
`parameters`, `defines`, and `include_paths` passed to Verilator.

Generated wrappers and aliases declare `` `timescale 1ns / 1ps `` for RTL
co-simulation. Supplied RTL should declare its own appropriate timescale (or
SystemVerilog time units); Allo copies those sources without modifying them.
XSIM rejects modules with missing timescales when other design modules have one.

For **manual Vitis validation**, the example emits a deterministic C testbench and
`validate.tcl`:

```sh
cd /tmp/rtl_accumulator.prj
vitis_hls -f validate.tcl
```

This requests C simulation, synthesis, RTL co-simulation, and IP export. Inspect
logs and the generated hierarchy to confirm that `accumulator` is present. The
example uses the default board configuration; select a supported part in the Tcl
if your installation requires a different device.

Python's `df.build(..., target="vitis_hls", mode="csyn")` emits the project;
calling its returned module starts the vendor build. This milestone deliberately
supports only that Python synthesis mode. Direct sequential `csim` stream calls
are not supported; use Allo's dataflow simulator or the manual Tcl testbench.

## Migrating the prototype

Change `from allo_rtl import RTLModule, Port, MemPort` to
`from allo import RTLModule, Port, MemPort`. The `ports=[...]` and
`n=..., inputs=..., outputs=..., mode="stream"/"buffer"` forms remain available.
The convenience form still defaults to array binding. Descriptors are copied
rather than mutated. The RTL object now retains its description rather than
returning a monkey-patched IPModule.

Verilator runs lazily, so `generated_source` names a file that appears only after
simulation preparation. Temporary build artifacts are isolated and cleaned up
with their owning object. `workdir` chooses the parent for an isolated build
directory, not a shared cache. Direct `ip(...)` from ordinary Python is not a
nanobind interface; call it from an Allo kernel.

The original local prototype is left untouched. No LLM dependency has been added. Standard ready/valid wrappers are generated
deterministically; custom protocol wrappers remain user-supplied.

## Implementation map

- `allo/backend/rtl.py`: descriptors, validation, transactor, simulation compiler,
  and black-box export.
- `allo/backend/rtl_adapter.py`: deterministic ready/valid adapter emitter.
- `allo/ir/infer.py`: logical argument validation before integer signedness is lost.
- `allo/ir/builder.py`: external-call reuse and symbol-collision checks.
- `allo/backend/hls.py`: backend eligibility, header inclusion, and registration.
- `allo/backend/vitis.py`: static array depths in AXI memory interface pragmas.
- `allo/passes.py`: recursive external-call rewriting, required for array/memory
  RTLModule calls nested in runtime loops (an existing local prerequisite).
- `allo/customize.py` and `allo/dataflow.py`: reject unsupported target dispatch.
- `tests/ip_integration/test_rtl.py`: supplied-wrapper and simulation regressions.
- `tests/ip_integration/test_rtl_adapter.py`: deterministic export, Allo simulation,
  randomized stalls/reset, and pulse-start/delayed-done adapter checks.

Protocol reference: AMD UG1399,
[JSON File for RTL Blackbox](https://docs.amd.com/r/2023.1-English/ug1399-vitis-hls/JSON-File-for-RTL-Blackbox).

## Reproducing software checks

### Vendor validation status (2026-10-01)

The ready/valid accumulator example passed C simulation, HLS synthesis, and
XSIM C/RTL co-simulation with Vitis HLS 2023.2 targeting
`xcu280-fsvh2892-2L-e`. Vivado IP export also completed; `export.zip` contains
both `ready_valid_accumulator.v` and `ready_valid_accumulator_allo.v`, and
`component.xml` registers both sources.

The initial verified run used one top-level transaction (two RTLModule calls).
The extended testbench also passed C/RTL co-simulation and IP export with three
top-level calls, checking the persistent running sum across all six RTLModule
calls.
This does not establish post-route timing or board-level operation.

### Software regression

On 2026-10-01 the command below passed **51 tests**, with the vendor synthesis
test deliberately deselected. The three-transaction example also passed a
standalone C-model/testbench run. Changed Python files passed Black, new C++
models passed clang-format, and the RTL backends passed pylint. Repository-wide
lint still encounters a pre-existing blank-line formatting issue in
`examples/ip_integration/vadd_ip.py`.

With the Allo/LLVM environment active and `VERILATOR` available:

```sh
python -m pytest tests/ip_integration/test_rtl_adapter.py \
  tests/ip_integration/test_rtl.py \
  tests/ip_integration/test_stream_ip.py \
  tests/ip_integration/test_stream_ip_sim.py \
  tests/test_backend_utils.py \
  --deselect tests/ip_integration/test_stream_ip.py::test_stream_ip_csynth -q
```

The deselection is intentional even when Vitis is on PATH: these are software
checks. Run the generated `validate.tcl` separately for vendor verification.
The adapter tests exercise FIFO sources changing immediately after reads,
independent input stalls, output stalls, clock-enable stalls, repeated calls,
excess offered input tokens, reset mid-transaction, and delayed completion.

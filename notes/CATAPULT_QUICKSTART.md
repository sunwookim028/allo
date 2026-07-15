# Catapult HLS Quickstart — zhang-21

**Target server**: `zhang-21.ece.cornell.edu` (RHEL 8.10, glibc 2.28)
Catapult is installed only on this server. All synthesis work runs here.

---

## 1. Prerequisites

### 1.1 Catapult Tool

Two versions are installed under `/opt/siemens/catapult/`:

| Version | Path |
|---------|------|
| 2024.2 (recommended) | `/opt/siemens/catapult/2024.2/` |
| 2024.1 | `/opt/siemens/catapult/2024.1_2-1117371/` |

**Option A — module (if available):**

```bash
module load mentor-Catapult_synthesis_10.5a
# Sets MGC_HOME and updates PATH automatically
```

**Option B — manual path (fallback if module is not installed):**

```bash
export MGC_HOME=/opt/siemens/catapult/2024.2
export PATH=$MGC_HOME/bin:$PATH
```

Verify:

```bash
catapult -version
# Expected: Catapult Ultra 2024.2 (build ...)
```

### 1.2 License

The license server is pre-configured on zhang-21 via `CATAPULT_LICENSE_FILE`. Verify it is set:

```bash
echo $CATAPULT_LICENSE_FILE
```

If empty, contact the server admin. Synthesis will fail immediately with a license checkout error if the license is not available.

### 1.3 AC Datatypes (Algorithmic C)

Catapult ships its own AC datatypes headers — no separate install needed. They live under:

```
$MGC_HOME/shared/include/
  ac_int.h          # fixed-width integer: ac_int<W, S>
  ac_fixed.h        # fixed-point: ac_fixed<W, I, S>
  ac_channel.h      # streaming FIFO: ac_channel<T>
  ac_std_float.h    # IEEE-754 float: ac_ieee_float<binary32/binary64>
  ac_math/          # transcendental functions
```

Include them in your kernel C++ as:

```cpp
#include <ac_int.h>
#include <ac_channel.h>
#include <ac_std_float.h>    // required if using floating-point
```

**Do not use `float` or `double` directly** — the target library (`nangate-45nm_beh`) does not
synthesize native C++ float types. Use `ac_ieee_float<binary32>` instead.

### 1.4 System C++ Compiler

Catapult uses its own internal EDG C++ front-end for analysis — you do not need a specific system
compiler for the input C++. The system GCC 8 on zhang-21 is sufficient for any supporting scripts.

---

## 2. Running Synthesis

### 2.1 Project Layout

A minimal Catapult project needs two files:

```
my_design.prj/
  kernel.cpp    # your C++ design (top function + sub-functions)
  run.tcl       # synthesis script
```

Put the project under `/scratch/sk3463/` (1.8 TB local SSD), not `/work/`:

```bash
mkdir -p /scratch/$USER/catapult_projects/my_design.prj
```

### 2.2 Minimal TCL Script (`run.tcl`)

```tcl
# --- Input ---
solution options set /Input/CppStandard c++11
solution options set /Input/CompilerFlags {-D_GLIBCXX_USE_CXX11_ABI=0}
solution file add kernel.cpp -type C++

# --- Design hierarchy ---
directive set -DESIGN_HIERARCHY top_function_name
directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0}}   ;# 2.0 ns = 500 MHz

# --- Target library ---
solution options set /Output/OutputVerilog true
solution library add nangate-45nm_beh

# --- Synthesis steps ---
go analyze
go compile
solution library add ccs_sample_mem
go assembly
go extract
```

Replace `top_function_name` with your actual top-level C++ function name.

**Block synthesis** (required when sub-functions pass `ac_channel` objects by reference):

```tcl
go analyze
solution design set sub_func_a -block
solution design set sub_func_b -block
go compile
solution library add ccs_sample_mem
go assembly
go extract
```

See §5 for why block synthesis is required for hierarchical designs.

### 2.3 Run Non-Interactively

```bash
cd /scratch/$USER/catapult_projects/my_design.prj
catapult -shell -f run.tcl 2>&1 | tee catapult.log
```

The `-shell` flag runs headless (no GUI). Synthesis typically takes 1–5 minutes for small designs.

### 2.4 Outputs

After a successful run, outputs appear in `<prj>/Catapult_<N>/<top>.v1/`:

| File | Contents |
|------|----------|
| `rtl.v` | Generated Verilog RTL |
| `cycle.rpt` | Latency / throughput per loop |
| `rtl.rpt` | Area report (Catapult score units) |
| `schedule.rpt` | Detailed schedule, resource binding |

```bash
ls /scratch/$USER/catapult_projects/my_design.prj/Catapult_1/top_function_name.v1/
```

---

## 3. Reading Reports

### Cycle report (`cycle.rpt`)

Key fields per module:

```
Latency   = first-output latency in clock cycles
Throughput = initiation interval (II) — cycles between successive invocations
```

### Area report (`rtl.rpt`)

Areas are reported as Catapult **score units** (scheduling metric, not physical nm²). Categories:

| Category | Meaning |
|----------|---------|
| `REG` | Pipeline registers, arrays (scratchpads) |
| `FUNC` | Datapath logic: adders, multipliers |
| `MUX` | Data path multiplexers |
| `LOGIC` | Control logic |

---

## 4. Common Errors and Fixes

### CIN-291: `float` not synthesizable

```
Error: Type 'float' is not synthesizable with library 'nangate-45nm_beh'
```

Use `ac_ieee_float<binary32>` and add `#include <ac_std_float.h>`.

### HIER-47 / ASSERT-1: FIFO depth = 0

```
Assertion failed: cap >= 0 (sif_ap_bif.cxx:1745)
```

Caused by flat (inline) synthesis of a design that passes `ac_channel` objects across
function boundaries. Fix: use **block synthesis** for each sub-function (see §2.2).

### HIER-6: Non-static `ac_channel`

```
Warning HIER-6: ac_channel variable must be static
```

Declare local `ac_channel` variables as `static`:

```cpp
static ac_channel<int> my_fifo;
```

### CRD-415: Double literal conversion

```
Error CRD-415: Cannot convert 'double' to 'ac_ieee_float<binary32>'
```

Use `f`-suffixed float literals: write `0.0f` not `0.0`, `1.5f` not `1.5`.

### HIER-23: Possible deadlock (warning)

Usually a false positive for valid try\_put/try\_get designs. Synthesis completes; RTL is correct.

---

## 5. Design Notes

### Why block synthesis is required for `ac_channel` across functions

Catapult has two modes for sub-functions:
- **Inline** (default): sub-function merged into caller's schedule. Local channels treated as
  variables — HIER-10 fires if they cross source-level boundaries, and depth is inferred as 0.
- **Block** (`-block`): sub-function compiled as a separate RTL module with real FIFO ports.
  Catapult infers correct depths based on producer/consumer II mismatch.

Use block synthesis whenever `ac_channel` objects are declared in one function and passed by
reference to another.

### FIFO depth inference with block synthesis

Catapult selects FIFO depth automatically based on the throughput ratio of producer and consumer.
Example: MT throughput 69 cycles, CT throughput 298 cycles → ratio 4.3× → Catapult infers
depth=16 to buffer a full burst without stalling.

---

## 6. Allo Integration (secondary)

This section applies only when using the Allo HLS framework to generate Catapult input.

### 6.1 Additional Dependency: conda env + libstdc++

The Allo MLIR backend (`.so` files) requires GCC 13's `libstdc++`. RHEL 8 (zhang-21) ships GCC 8
only. The conda `allo` env includes the newer `libstdc++`, but it must be on `LD_LIBRARY_PATH`
before activating the env so that subprocesses inherit it:

```bash
# Set LD_LIBRARY_PATH before conda activate so subprocesses inherit it
export LD_LIBRARY_PATH="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib:$LD_LIBRARY_PATH"
conda activate allo
```

Or use the wrapper script which sets this automatically:

```bash
./run_allo.sh python my_script.py
```

Add to `~/.bashrc` on zhang-21 for a permanent fix:

```bash
if [[ "$(hostname)" == "zhang-21.ece.cornell.edu" ]]; then
    CONDA_ENV_LIB="/work/shared/users/phd/sk3463/envs/miniconda3/envs/allo/lib"
    export LD_LIBRARY_PATH="$CONDA_ENV_LIB:$LD_LIBRARY_PATH"
fi
```

### 6.2 MLIR Build for RHEL 8

The `.so` files in `allo/_mlir/_mlir_libs/` must be built against the `build-rhel8` LLVM
(not the default `build/` which was compiled on a glibc 2.35 host). See `notes/ENVIRONMENT.md`
§ "Rebuild for RHEL 8" for the full cmake + ninja + copy procedure.

### 6.3 Codegen, Synthesis, PPA

NOTE (2026-07-15): `tests/dataflow/catapult_synth_decoupled_2x1.py` is not in
the tree; it stayed on the deleted `feature/mesh-accelerator` branch (last tip
`06ce561`) and can be recovered from there if needed. The nearest in-tree
equivalent is `tests/dataflow/hls_synth_decoupled.py`.

```bash
# Generate kernel.cpp only (no Catapult needed)
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode codegen

# Run full synthesis (Catapult required)
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode csyn

# Extract PPA from area.rpt
./run_allo.sh python tests/dataflow/catapult_synth_decoupled_2x1.py --mode ppa
```

Project output goes to `catapult_decoupled_2x1.prj/` (put under `/scratch/` to avoid quota).

---

## 7. Reference

| Resource | Location |
|----------|----------|
| AC datatypes headers | `$MGC_HOME/shared/include/` |
| Catapult 2024.2 install | `/opt/siemens/catapult/2024.2/` |
| Synthesis results & analysis | `notes/CATAPULT.md` |
| Allo Catapult backend | `allo/backend/catapult.py` |

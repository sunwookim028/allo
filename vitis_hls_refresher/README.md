# Vitis HLS Refresher: Understanding `static` and HLS→RTL Mapping

This directory contains a progressive set of experiments to understand how the `static` keyword in Vitis HLS affects hardware generation and RTL mapping.

## Quick Start

```bash
# Navigate to any example directory
cd ex01_basic_static

# Run C simulation (fast iteration)
make csim

# View results
cat ex01.prj/solution1/sim/report/csim.log
```

## Directory Structure

- `ex01_basic_static/` - Basic static vs non-static variable behavior
- `ex02_static_variables/` - Static variables in loops and accumulators
- `ex03_static_arrays/` - Static arrays and memory resource mapping
- `ex04_pipelines/` - Static variables in pipelined loops

Each example contains:
- `kernel.cpp` - HLS kernel code with multiple versions
- `tb.cpp` - Testbench demonstrating behavior
- `run.tcl` - Vitis HLS TCL script
- `Makefile` - Build automation

## Key Concepts

### 1. Static Variables → Registers

**Non-static variables:**
- Synthesized as wires/logic
- Reset on every function call
- No state retention

**Static variables:**
- Synthesized as registers (flip-flops)
- Retain value across function calls
- Require initialization logic
- May need reset control

### 2. Static Arrays → Memory Resources

**Small arrays (< 64 elements):**
- May map to registers (if partitioned)
- Fast access, uses LUTs/FFs

**Large arrays:**
- Map to BRAM (Block RAM)
- Map to URAM (Ultra RAM) for very large arrays
- Slower access but efficient storage

**Static arrays:**
- Persist across function calls
- Shared state between calls
- Useful for shift registers, lookup tables

### 3. Static in Pipelines

**Challenges:**
- Static variables can prevent II=1 pipelining
- Creates dependencies across iterations
- May require stall cycles

**Proper use:**
- Shift registers (tapped delays)
- State machines
- Lookup tables

**Avoid:**
- Static accumulators in pipelined loops
- Static variables that create false dependencies

## Running Experiments

### Example 1: Basic Static Behavior

```bash
cd ex01_basic_static
make csim
```

**What to observe:**
- Non-static resets each call
- Static accumulates across calls
- Check synthesis report for register usage

### Example 2: Static Variables in Loops

```bash
cd ex02_static_variables
make test_a test_b test_c
```

**What to observe:**
- Version A: Clean pipeline, II=1 achievable
- Version B: Static accumulator persists
- Version C: Controlled reset mechanism

### Example 3: Static Arrays

```bash
cd ex03_static_arrays
make test_a test_b test_c
make synth view  # Inspect resource usage
```

**What to observe:**
- Resource report shows BRAM/URAM usage
- Compare static vs non-static array mapping
- Initialization behavior

### Example 4: Pipelines

```bash
cd ex04_pipelines
make test_a test_b test_c test_d
make synth  # Check II values
```

**What to observe:**
- Initiation Interval (II) values
- Pipeline stalls
- Resource usage vs performance tradeoff

## Understanding Synthesis Reports

After running `make synth`, check the report:

```bash
cat <project>/solution1/syn/report/top_csynth.rpt
```

Look for:
- **Initiation Interval (II)**: Cycles between new inputs
- **Latency**: Total cycles
- **Resource Usage**: BRAM, DSP, FF, LUT
- **Pipelining**: Loop pipelining status

## Interpreting Hardware Mapping

### Variable → Hardware Mapping

| C++ Code | RTL Hardware | Notes |
|----------|--------------|-------|
| `int x;` | Wire/Logic | Combinational |
| `static int x;` | Register (FF) | Sequential, stateful |
| `int arr[10];` | Registers (if small) | Depending on partitioning |
| `static int arr[100];` | BRAM | Block RAM |
| `static int arr[1000];` | URAM | Ultra RAM (if available) |

### Pipeline Impact

| Pattern | II Achievable | Notes |
|---------|---------------|-------|
| Non-static accumulator | II=1 | Clean dependency |
| Static accumulator | II>1 | Persists across calls |
| Shift register | II=1 | Regular pattern |
| State machine | II=1 (with stalls) | Conditional |

## Common Patterns

### ✅ Good: Shift Register

```cpp
static int shift_reg[5] = {0};
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    // Shift operation
    for (int j = 4; j > 0; j--) {
        shift_reg[j] = shift_reg[j-1];
    }
    shift_reg[0] = data[i];
}
```

### ✅ Good: State Machine

```cpp
static int state = 0;
static int counter = 0;
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    if (state == 0) {
        // Process state 0
    } else {
        // Process state 1
    }
}
```

### ❌ Avoid: Static Accumulator in Pipeline

```cpp
static int acc = 0;  // Problematic!
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    acc = acc + data[i];  // Cannot pipeline well
}
```

### ✅ Alternative: Non-static Accumulator

```cpp
int acc = 0;  // Better!
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    acc = acc + data[i];  // Clean pipeline
}
```

## Tips for Fast Iteration

1. **Use `csim` only** for quick verification
   ```bash
   make csim  # Fast: ~seconds
   ```

2. **Use `synth` sparingly** for resource analysis
   ```bash
   make synth  # Slow: ~minutes
   ```

3. **Check logs** for immediate feedback
   ```bash
   cat <project>/solution1/sim/report/csim.log
   ```

4. **Compare versions** side-by-side
   ```bash
   make test_a test_b  # Compare A vs B
   ```

## Troubleshooting

### Vitis HLS not found
```bash
# Check if Vitis HLS is in PATH
which vitis_hls

# If not, source setup script (ask user for location)
source /path/to/vitis_hls/settings64.sh
```

### Synthesis errors
- Check for valid pragmas
- Verify array sizes are reasonable
- Review resource constraints

### Simulation mismatches
- Verify initialization values
- Check for uninitialized static variables
- Review testbench expectations

## Next Steps

1. Run each example and observe behavior
2. Modify parameters and see impact
3. Check synthesis reports for resource usage
4. Experiment with different pragmas
5. Compare static vs non-static implementations

## References

- Vitis HLS User Guide (UG1399)
- HLS Optimization Guide
- FPGA Resource Documentation

## Questions to Explore

1. When should you use static vs non-static?
2. How do static arrays affect timing?
3. What's the cost of static variables in pipelines?
4. How to control static variable initialization?
5. When do static variables break II=1?

For each example, modify the code, run simulations, and observe the changes in behavior and hardware mapping!


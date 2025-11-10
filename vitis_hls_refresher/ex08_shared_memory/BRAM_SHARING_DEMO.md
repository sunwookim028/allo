# BRAM Sharing Demonstration in Mini TPU (ex08)

## Overview

This example demonstrates **BRAM sharing** between the MatMul engine and Elemwise engine in a mini TPU architecture. Both kernels share the **same physical BRAM resource**, enabling efficient data sharing and resource utilization.

## BRAM Sharing Architecture

```
┌─────────────────────────────────────────┐
│         shared_memory[4096]              │
│         (Single BRAM Resource)          │
│                                          │
│  ┌──────────┐      ┌──────────┐        │
│  │ MatMul   │◄─────┤  BRAM    │─────►│  │
│  │ Engine   │      │  (Shared)│       │  │
│  └──────────┘      └──────────┘       │  │
│       │                  ▲              │  │
│       │                  │              │  │
│       └──────────────────┘              │  │
│              │                          │  │
│              ▼                          │  │
│  ┌──────────┐                          │  │
│  │ Elemwise │                          │  │
│  │ Engine   │                          │  │
│  └──────────┘                          │  │
└─────────────────────────────────────────┘
```

## Key Code Elements

### 1. Shared BRAM Declaration

```cpp
// File-scope array - maps to SINGLE BRAM resource
int shared_memory[4096];  // Shared by ALL kernels
```

**Critical Point**: This is declared at **file scope**, making it accessible to all kernels. In hardware, this maps to a **single BRAM resource**.

### 2. BRAM Binding in MatMul Engine

```cpp
void matmul_shared(int w_offset, int x_offset, int result_offset, ...) {
    // Specifies shared_memory maps to BRAM
    #pragma HLS bind_storage variable=shared_memory type=ram_2p impl=bram
    // ... uses shared_memory ...
}
```

### 3. BRAM Binding in Elemwise Engine

```cpp
void elemwise_shared(int a_offset, int b_offset, int result_offset, ...) {
    // SAME pragma - uses SAME BRAM resource!
    #pragma HLS bind_storage variable=shared_memory type=ram_2p impl=bram
    // ... uses shared_memory ...
}
```

**Key Insight**: Both kernels use the **same `shared_memory` variable** with the **same pragma**. This means they share the **same physical BRAM** in hardware.

## BRAM Sharing Demonstration Flow

### Test 5: BRAM Sharing Demonstration

```
Step 1: Load data into shared_memory (BRAM)
  - W matrix → shared_memory[0:15]
  - X matrix → shared_memory[16:31]
  - B bias → shared_memory[32:47]

Step 2: MatMul Engine writes to BRAM
  - Reads: W from shared_memory[0], X from shared_memory[16]
  - Writes: W*X to shared_memory[48:63]
  - Uses: SAME BRAM resource

Step 3: Elemwise Engine reads from BRAM
  - Reads: W*X from shared_memory[48] (written by MatMul!)
  - Reads: B from shared_memory[32]
  - Writes: H = W*X + B to shared_memory[64:79]
  - Uses: SAME BRAM resource as MatMul!

Step 4: Verify BRAM Sharing
  - Both kernels accessed the SAME physical BRAM
  - Intermediate result (W*X) persisted in BRAM
  - No data transfer needed between kernels
```

## Hardware Resource Comparison

### With BRAM Sharing (Current Implementation)

```
Kernels: MatMul + Elemwise
BRAM Resources: 1 shared BRAM
Memory Layout: Managed via offsets
Data Transfer: On-FPGA only (no DDR)
```

### Without BRAM Sharing (Hypothetical)

```
Kernels: MatMul + Elemwise
BRAM Resources: 2 separate BRAMs
Memory Layout: Independent
Data Transfer: DDR transfer between kernels
```

**Resource Savings**: 1 BRAM resource saved by sharing!

## BRAM Resource Details

### Dual-Port BRAM (ram_2p)

- **Type**: `ram_2p` (dual-port)
- **Implementation**: `bram` (Block RAM)
- **Size**: 4096 ints = 16KB
- **Ports**: 2 ports enable concurrent read/write
- **Sharing**: Multiple kernels can access simultaneously

### Why Dual-Port?

- MatMul engine may read from one region while writing to another
- Elemwise engine may read while MatMul writes
- Enables better pipelining and throughput

## Mini TPU Use Case

### Operation: H = W * X + B

1. **MatMul Engine**:
   - Reads W, X from shared BRAM
   - Computes W*X
   - Writes result to shared BRAM

2. **Elemwise Engine**:
   - Reads W*X from shared BRAM (same location MatMul wrote!)
   - Reads B from shared BRAM
   - Computes W*X + B
   - Writes result to shared BRAM

**Key Benefit**: Intermediate result (W*X) stays in BRAM - no DDR transfer!

## Verification

Run Test 5 to see BRAM sharing in action:

```bash
cd ex08_shared_memory
make test_shared
```

Or:

```bash
vitis_hls -f test_shared_demo.tcl
```

The test clearly shows:
- MatMul writes to `shared_memory[48]`
- Elemwise reads from `shared_memory[48]` (same location!)
- Both kernels use the same BRAM resource

## Synthesis Verification

After synthesis, check the resource report:

```bash
make synth
make view
```

Look for:
- **BRAM usage**: Should show shared BRAM resource
- **Memory instances**: Should show `shared_memory` mapped to BRAM
- **Resource sharing**: Both kernels reference the same BRAM

## Key Takeaways

1. **File-scope arrays** → Shared BRAM resource
2. **Same variable name** → Same physical BRAM
3. **Same pragma** → Confirms BRAM mapping
4. **Offset management** → Enables multiple kernels to use same BRAM
5. **Resource efficiency** → 1 BRAM instead of 2+

This is the foundation of efficient multi-kernel designs in HLS!

---

## Troubleshooting: Testbench Access to Shared Memory

### Problem: Testbench Shows Zeros Instead of Expected Values

**Symptom:**
```
Step 2: MatMul Engine computes W*X
  First 4 elements: 0 0 0 0  (expected: 1 2 3 4)

Step 3: Elemwise Engine reads intermediate result
  First 4 elements: 0 0 0 0  (expected: 2 3 4 5)
```

**Root Cause:**
The testbench and kernel are accessing **different instances** of `shared_memory`. This happens when:
- Testbench declares `static int shared_memory[4096]` (creates separate instance)
- Testbench doesn't declare `extern int shared_memory[4096]` at file scope
- The `extern` declaration is inside a function instead of at file scope

**Solution:**

1. **In `kernel.cpp`**: Declare `shared_memory` at file scope (non-static):
   ```cpp
   // File scope - shared by all kernels
   int shared_memory[4096];  // Non-static for testbench access
   ```

2. **In `tb.cpp`**: Declare `shared_memory` as `extern` at **file scope** (before `main()`):
   ```cpp
   // Forward declarations
   extern void matmul_shared(...);
   extern void elemwise_shared(...);
   
   // CRITICAL: Must be at file scope, not inside main()!
   extern int shared_memory[4096];
   
   int main() {
       // Now can access shared_memory
       memcpy(&shared_memory[0], data, size);
       matmul_shared(...);
   }
   ```

3. **DO NOT** declare `static int shared_memory[4096]` in the testbench - this creates a separate instance!

**Verification:**
After fixing, the test should show:
```
Step 2: MatMul Engine computes W*X
  First 4 elements: 1 2 3 4  (expected: 1 2 3 4) ✅

Step 3: Elemwise Engine reads intermediate result
  First 4 elements: 2 3 4 5  (expected: 2 3 4 5) ✅
```

**Key Points:**
- `extern` declaration must be at **file scope** (outside any function)
- `extern` tells the compiler to link to the definition in `kernel.cpp`
- `static` in testbench creates a **separate instance** - avoid this!
- In hardware, file-scope variables are shared across kernels automatically

**Compilation Check:**
If you see compilation errors like:
```
error: 'shared_memory' was not declared in this scope
```
This means the `extern` declaration is missing or in the wrong scope.

**Standalone Test:**
To verify linkage works, compile separately:
```bash
g++ -std=c++11 -I/opt/xilinx/Vitis_HLS/2023.2/include tb.cpp kernel.cpp -o test
./test
```
If this works but Vitis HLS doesn't, check that the `extern` is at file scope.


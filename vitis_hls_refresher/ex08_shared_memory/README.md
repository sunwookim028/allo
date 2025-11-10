# Example 8: Shared Static Memory Across Kernels

## Overview

This example demonstrates **properly shared static variables** across multiple kernels. Unlike Example 6 where each kernel had its own static memory, here we show how to create shared memory that multiple kernels can access.

## Key Concept: File-Scope vs Function-Scope Static

### Function-Scope Static (NOT Shared)
```cpp
void kernel1() {
    static int memory[1024];  // Each kernel instance has its own memory
    // ...
}

void kernel2() {
    static int memory[1024];  // Different memory from kernel1!
    // ...
}
```
**Problem**: Each kernel has its own copy. Not shared!

### File-Scope Static (SHARED)
```cpp
// Declared at file scope (outside any function)
static int shared_memory[4096];  // Shared by ALL kernels

void kernel1() {
    // Accesses shared_memory
    shared_memory[0] = 100;
}

void kernel2() {
    // Accesses the SAME shared_memory
    int value = shared_memory[0];  // Gets 100 from kernel1!
}
```
**Solution**: File-scope static is shared across all kernels!

## Architecture

### Kernels

1. **MatMul Kernel** (`matmul_kernel`)
   - Computes matrix multiplication: `C = A * B`
   - Uses shared memory for intermediate storage
   - Example: `H = W * X`

2. **Elemwise Kernel** (`elemwise_kernel`)
   - Computes element-wise operations: `C = A op B`
   - Operations: add (op=0) or multiply (op=1)
   - Example: `H = H + B` (bias addition)

3. **Top Module** (`top_module`)
   - Coordinates multiple operations
   - Demonstrates kernel composition
   - Examples:
     - `H = W * X + B` (matmul + bias)
     - `A = X * Y + W * Z` (two matmuls + add)

### Shared Memory Layout

```
shared_memory[4096]:
┌─────────────────┐
│  W matrix       │  Offset: 0
│  (M*K elements) │
├─────────────────┤
│  X matrix       │  Offset: M*K
│  (K*N elements) │
├─────────────────┤
│  Results        │  Offset: M*K + K*N
│  (M*N elements) │
├─────────────────┤
│  ...            │
└─────────────────┘
```

## Usage Examples

### Example 1: H = W * X + B

```cpp
// Step 1: Matrix multiplication
int temp_H[M*N];
matmul_kernel(W, X, temp_H, M, N, K);  // temp_H = W * X

// Step 2: Add bias
int H[M*N];
elemwise_kernel(temp_H, B, H, M*N, 0);  // H = temp_H + B
```

### Example 2: A = X * Y + W * Z

```cpp
// Step 1: Compute X * Y
int temp_XY[M*N];
matmul_kernel(X, Y, temp_XY, K, N, K);

// Step 2: Compute W * Z
int temp_WZ[M*N];
matmul_kernel(W, Z, temp_WZ, M, N, K);

// Step 3: Add results
int A[M*N];
elemwise_kernel(temp_XY, temp_WZ, A, M*N, 0);  // A = temp_XY + temp_WZ
```

## Direct Shared Memory Access (More Efficient)

For better performance, kernels can access shared memory directly:

```cpp
// Direct access via offsets (no AXI transfers)
matmul_shared(w_offset, x_offset, result_offset, M, N, K);
elemwise_shared(a_offset, b_offset, result_offset, size, op);
```

**Advantages**:
- No AXI transfers needed
- Data already in shared memory
- Lower latency
- Higher throughput

## Interface Pragmas

### MatMul Kernel
```cpp
#pragma HLS INTERFACE m_axi port=W depth=4096      // External memory for W
#pragma HLS INTERFACE m_axi port=X depth=4096      // External memory for X
#pragma HLS INTERFACE m_axi port=result depth=4096 // External memory for result
#pragma HLS INTERFACE s_axilite port=M             // Control register
#pragma HLS INTERFACE s_axilite port=N             // Control register
#pragma HLS INTERFACE s_axilite port=K             // Control register
```

### Elemwise Kernel
```cpp
#pragma HLS INTERFACE m_axi port=A depth=4096      // External memory for A
#pragma HLS INTERFACE m_axi port=B depth=4096      // External memory for B
#pragma HLS INTERFACE m_axi port=result depth=4096 // External memory for result
#pragma HLS INTERFACE s_axilite port=size          // Control register
#pragma HLS INTERFACE s_axilite port=op            // Control register
```

### Shared Memory Storage
```cpp
#pragma HLS bind_storage variable=shared_memory type=RAM_2P_BRAM
```
- **Dual-port BRAM**: Allows simultaneous access from multiple kernels
- **Shared resource**: All kernels access the same BRAM

## Comparison: Shared vs Non-Shared

| Aspect | Function-Scope Static | File-Scope Static |
|--------|----------------------|-------------------|
| **Scope** | Per kernel instance | Shared across kernels |
| **Memory** | Separate copies | Single shared copy |
| **Use Case** | Kernel-local storage | Inter-kernel communication |
| **Hardware** | Multiple BRAMs | Single shared BRAM |

## Key Insights

1. **File-scope static = shared**: Declared outside functions, shared by all kernels
2. **Function-scope static = local**: Declared inside functions, separate per kernel
3. **Dual-port BRAM**: Enables simultaneous access from multiple kernels
4. **Memory layout**: Careful offset management required
5. **Direct access**: More efficient than AXI transfers

## When to Use Shared Memory

✅ **Use shared memory when:**
- Multiple kernels need to share data
- Intermediate results need to persist
- Kernel composition is required
- Data locality is important

❌ **Avoid shared memory when:**
- Kernels are independent
- Data doesn't need to persist
- Simple single-kernel operations
- Memory conflicts would occur

## Running the Example

```bash
cd ex08_shared_memory

# Test all kernels
make all

# Test specific kernel
make test_matmul   # Matrix multiplication
make test_elemwise # Element-wise operations
make test_shared   # Shared memory access

# Run synthesis
make synth
make view
```

## Differences from Example 6

| Feature | Example 6 | Example 8 |
|---------|----------|-----------|
| **Static scope** | Function-scope | File-scope |
| **Sharing** | Not shared | Shared |
| **Use case** | Single kernel | Multiple kernels |
| **Memory** | Per kernel | Shared BRAM |
| **Composition** | Not supported | Supported |

## Summary

Shared static memory enables:
- **Kernel composition**: Multiple kernels working together
- **Data persistence**: Results available across kernel calls
- **Efficient communication**: Direct memory access
- **Complex operations**: Building larger systems from smaller kernels

This is essential for building complex accelerator designs where multiple kernels need to cooperate!


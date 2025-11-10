# AXI Overhead and Interfacing Choices: A Gentle Introduction

## Overview

This experiment compares two approaches to accessing shared memory in HLS designs:
1. **AXI-based access** (`matmul_kernel`) - Uses `m_axi` ports for external memory transfer
2. **Direct shared memory access** (`matmul_shared`) - Uses `s_axilite` ports for offsets only

## Understanding the Two Approaches

### Version 1: AXI-based (`matmul_kernel`)

**Interface:**
```cpp
void matmul_kernel(int* W, int* X, int* result, int M, int N, int K) {
    #pragma HLS INTERFACE m_axi port=W depth=4096
    #pragma HLS INTERFACE m_axi port=X depth=4096
    #pragma HLS INTERFACE m_axi port=result depth=4096
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return
    
    // Copy data from external memory (via AXI) to shared_memory
    for (int i = 0; i < M * K; i++) {
        shared_memory[i] = W[i];  // AXI read transfer
    }
    // ... computation ...
    // Copy results back to external memory (via AXI)
    for (int i = 0; i < M * N; i++) {
        result[i] = shared_memory[i];  // AXI write transfer
    }
}
```

**What happens:**
- Data flows: **Host Memory → AXI Bus → Kernel → shared_memory → Kernel → AXI Bus → Host Memory**
- Requires **AXI Master Interface** logic
- **Overhead**: AXI protocol handling, burst transfers, address management

### Version 2: Direct Shared Memory (`matmul_shared`)

**Interface:**
```cpp
void matmul_shared(int w_offset, int x_offset, int result_offset, 
                   int M, int N, int K) {
    #pragma HLS INTERFACE s_axilite port=w_offset
    #pragma HLS INTERFACE s_axilite port=x_offset
    #pragma HLS INTERFACE s_axilite port=result_offset
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return
    
    // Direct access to shared_memory using offsets
    // No AXI transfers - data already in shared memory!
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            sum += shared_memory[w_offset + i*K + k] * 
                   shared_memory[x_offset + k*N + j];
        }
    }
}
```

**What happens:**
- Data flows: **shared_memory → Kernel → shared_memory** (direct BRAM access)
- Only **AXI-Lite** for control (offsets, dimensions)
- **No overhead**: Direct BRAM access, no protocol handling

## The AXI Overhead Explained

### What is AXI Overhead?

**AXI (Advanced eXtensible Interface)** is a protocol for connecting IP blocks. When you use `m_axi`:

1. **Interface Logic**: 
   - Address generators
   - Burst controllers
   - Read/write data paths
   - Handshaking logic (VALID/READY signals)

2. **Latency**:
   - Protocol overhead (address phase, data phase)
   - Burst transfer setup
   - Data alignment and packing

3. **Resource Usage**:
   - Additional LUTs/FFs for AXI logic
   - FIFOs for data buffering
   - State machines for protocol handling

### Visual Comparison

```
AXI-based (matmul_kernel):
┌─────────┐     AXI Bus      ┌──────────┐     BRAM      ┌─────────┐
│  Host   │ ───────────────> │  Kernel  │ ───────────> │ shared_ │
│ Memory  │   (m_axi port)   │          │   (direct)    │ memory  │
└─────────┘                   └──────────┘               └─────────┘
     ↑                              ↓                          ↑
     └──────────────────────────────┴──────────────────────────┘
                    AXI overhead here!

Direct Shared Memory (matmul_shared):
┌──────────┐     BRAM      ┌─────────┐
│  Kernel  │ ───────────> │ shared_ │
│          │   (direct)    │ memory  │
└──────────┘               └─────────┘
     ↑
     └─── AXI-Lite (control only: offsets, dimensions)
     
     No data transfer overhead!
```

## Synthesis Results Comparison

### Interface Types

**Version 1 (AXI-based):**
- `m_axi` ports for data (W, X, result)
- `s_axilite` ports for control (M, N, K)
- **Hardware**: AXI Master Interface + AXI-Lite Slave

**Version 2 (Direct Shared Memory):**
- `s_axilite` ports for offsets and dimensions only
- **Hardware**: AXI-Lite Slave only (much simpler!)

### Resource Usage

**Expected Differences:**

| Resource | AXI-based | Direct Shared | Difference |
|----------|-----------|---------------|------------|
| **LUTs** | Higher | Lower | AXI logic overhead |
| **FFs** | Higher | Lower | AXI state machines |
| **BRAM** | Similar | Similar | Same shared_memory |
| **DSP** | Similar | Similar | Same computation |

**Why?** AXI Master Interface requires:
- Address calculation logic
- Burst transfer controllers
- Data path FIFOs
- Protocol state machines

### Latency Comparison

**AXI-based:**
```
Total Latency = AXI Read Latency + Computation + AXI Write Latency
              = (burst setup + data transfer) + compute + (burst setup + data transfer)
```

**Direct Shared Memory:**
```
Total Latency = Computation only
              = (direct BRAM access + compute)
```

**Key Insight**: Direct access eliminates AXI transfer overhead!

## When to Use Each Approach

### Use AXI-based (`m_axi`) when:
- ✅ Data comes from **external memory** (DDR, host)
- ✅ Need to **transfer large datasets** from host
- ✅ Data is **not persistent** between kernel calls
- ✅ **Standard IP interface** required

**Example**: Processing image data from host memory

### Use Direct Shared Memory (`s_axilite` offsets) when:
- ✅ Data is **already in shared memory**
- ✅ **Multiple kernels** share the same data
- ✅ Want to **minimize latency** and overhead
- ✅ **Kernel composition** (one kernel feeds another)

**Example**: Pipeline of operations: `matmul → elemwise → matmul`

## Trade-offs Summary

| Aspect | AXI-based | Direct Shared |
|--------|-----------|---------------|
| **Flexibility** | ✅ Can access external memory | ❌ Limited to shared memory |
| **Latency** | ⚠️ Higher (AXI overhead) | ✅ Lower (direct access) |
| **Resources** | ⚠️ More (AXI logic) | ✅ Fewer (simple interface) |
| **Complexity** | ⚠️ More complex | ✅ Simpler |
| **Use Case** | External data | Internal data sharing |

## Key Takeaways

1. **AXI interfaces add overhead**: Protocol handling, state machines, buffering
2. **Direct shared memory is more efficient**: When data is already in shared memory
3. **Choose based on data flow**: External → AXI, Internal → Direct
4. **Kernel composition benefits**: Direct shared memory enables efficient pipelines

## Experiment Results

Run the comparison:
```bash
cd ex08_shared_memory
make synth              # Synthesize both versions
./compare_axi_overhead.sh  # Compare results
```

Check the synthesis log for detailed interface information:
```bash
grep -E "m_axi|s_axilite|ap_" synth_axi_comparison.log
```

## Conclusion

Understanding AXI overhead helps you make informed design choices:
- **Need external access?** → Use AXI (`m_axi`)
- **Data already internal?** → Use direct shared memory (`s_axilite` offsets)

The overhead is **real and measurable** - direct shared memory access is significantly more efficient when data is already in shared memory!


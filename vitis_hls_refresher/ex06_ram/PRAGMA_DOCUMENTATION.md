# Interface and Resource Pragmas in RAM Kernels

## Overview

This document explains the pragmas used in each RAM kernel version and what they specify in terms of physical hardware implementation.

## Interface Pragmas (`#pragma HLS INTERFACE`)

### Purpose
Specifies how function arguments connect to hardware interfaces. This determines how the kernel communicates with the host or other hardware components.

### Common Interface Types

#### `ap_none` (No Interface)
- **Usage**: Direct connection, no protocol
- **Hardware**: Simple wires/logic
- **Use Case**: Scalar inputs/outputs, control signals
- **Example**: `#pragma HLS INTERFACE ap_none port=write_addr`

#### `m_axi` (Memory-Mapped AXI4)
- **Usage**: AXI4 master interface for external memory access
- **Hardware**: AXI4 master interface with burst support
- **Use Case**: Large arrays passed as pointers, external memory access
- **Example**: `#pragma HLS INTERFACE m_axi port=data depth=1024`
- **Note**: Requires `depth` parameter to specify array size

#### `s_axilite` (AXI4-Lite Slave)
- **Usage**: AXI4-Lite slave interface for control/status
- **Hardware**: AXI4-Lite slave interface
- **Use Case**: Control registers, status signals
- **Example**: `#pragma HLS INTERFACE s_axilite port=return`

#### `ap_fifo` / `ap_hs` (Streaming Interfaces)
- **Usage**: FIFO or handshake streaming interfaces
- **Hardware**: FIFO buffers or handshake protocols
- **Use Case**: Streaming data, pipelined designs
- **Example**: `#pragma HLS INTERFACE ap_fifo port=data_stream`

### In Our RAM Kernels

All RAM kernels use `ap_none` for all ports because:
- **Scalar inputs**: `write_addr`, `read_addr`, `in`, `write_en` are simple integers
- **Direct output**: `out` is a pointer to a single integer
- **No external memory**: The static array is internal to the kernel
- **No streaming**: Data is accessed via addresses, not streams

**Example from Version A:**
```cpp
#pragma HLS INTERFACE ap_none port=out
#pragma HLS INTERFACE ap_none port=write_addr
#pragma HLS INTERFACE ap_none port=read_addr
#pragma HLS INTERFACE ap_none port=in
#pragma HLS INTERFACE ap_none port=write_en
```

## Resource Pragmas (`#pragma HLS bind_storage`)

### Purpose
Specifies the physical storage resource (BRAM, URAM, registers) for variables. This directly controls hardware resource allocation.

### Storage Types

#### `ram_1p impl=bram` (Single-Port Block RAM)
- **Hardware**: One Block RAM (18Kb or 36Kb)
- **Access**: One read OR one write per cycle (not simultaneous)
- **Use Case**: Sequential access patterns, single-port memory
- **Resources**: ~1 BRAM per 512-1024 words (32-bit)
- **Example**: `#pragma HLS bind_storage variable=memory type=ram_1p impl=bram`

#### `ram_2p impl=bram` (Dual-Port Block RAM)
- **Hardware**: One Block RAM with two ports
- **Access**: Simultaneous read AND write (different addresses)
- **Use Case**: Parallel access patterns, high throughput
- **Resources**: ~1 BRAM (more efficient than two single-port BRAMs)
- **Example**: `#pragma HLS bind_storage variable=memory type=ram_2p impl=bram`

#### `ram_1p impl=uram` / `ram_2p impl=uram` (Ultra RAM)
- **Hardware**: Ultra RAM (72Kb blocks)
- **Access**: Similar to BRAM but larger capacity
- **Use Case**: Very large arrays (>64KB)
- **Resources**: More efficient for large arrays
- **Example**: `#pragma HLS bind_storage variable=large_array type=ram_1p impl=uram`

### In Our RAM Kernels

#### Version A: Single-Port BRAM
```cpp
#pragma HLS bind_storage variable=memory type=ram_1p impl=bram
```
- **Hardware**: One BRAM block
- **Access**: Sequential (read OR write per cycle)
- **Use Case**: Basic memory operations

#### Version B: Dual-Port BRAM
```cpp
#pragma HLS bind_storage variable=memory type=ram_2p impl=bram
```
- **Hardware**: One BRAM block with two ports
- **Access**: Simultaneous read AND write
- **Use Case**: High-throughput applications

#### Version C: Single-Port BRAM (Initialized)
```cpp
#pragma HLS bind_storage variable=memory type=ram_1p impl=bram
```
- **Hardware**: Same as Version A
- **Difference**: Initialization values stored in BRAM

#### Version D: Single-Port BRAM + Partitioned Array
```cpp
#pragma HLS bind_storage variable=memory type=ram_1p impl=bram
#pragma HLS ARRAY_PARTITION variable=written cyclic factor=32
```
- **Hardware**: BRAM for `memory`, registers for `written` (partitioned)
- **Access**: BRAM for data, parallel access to `written` array

#### Version E: Single-Port BRAM (Shared Address)
```cpp
#pragma HLS bind_storage variable=memory type=ram_1p impl=bram
```
- **Hardware**: Same as Version A
- **Difference**: Read/write share same address

## Physical Implementation Mapping

### Single-Port BRAM (ram_1p impl=bram)
```
┌─────────────────┐
│   BRAM Block    │
│   (18Kb/36Kb)   │
│                 │
│  Address ───────┼─── Addr Decoder
│  Data In ───────┼─── Write Logic
│  Write En ──────┼─── Control
│  Data Out ──────┼─── Read Logic
└─────────────────┘
```

### Dual-Port BRAM (ram_2p impl=bram)
```
┌─────────────────┐
│   BRAM Block    │
│   (Dual-Port)   │
│                 │
│  Port A:        │
│    Addr ────────┼─── Write Port
│    Data In ─────┼───
│    Write En ────┼───
│                 │
│  Port B:        │
│    Addr ────────┼─── Read Port
│    Data Out ────┼───
└─────────────────┘
```

## Deprecated: `#pragma HLS RESOURCE`

**Old Syntax (Deprecated):**
```cpp
#pragma HLS RESOURCE variable=memory core=ram_1p impl=bram
```

**New Syntax (Current):**
```cpp
#pragma HLS bind_storage variable=memory type=ram_1p impl=bram
```

The `RESOURCE` pragma is deprecated. Use `bind_storage` instead.

## Why Remove the `mem` Argument?

### Original (Incorrect) Design
```cpp
void ram_a(int* mem, int write_addr, int read_addr, int in, int write_en, int* out) {
    static int memory[1024];  // Internal static array
    #pragma HLS INTERFACE m_axi port=mem depth=1024  // Wrong!
    // mem argument was never used!
}
```

### Problems:
1. **`mem` was unused**: The function never accessed `mem`, only the internal `memory`
2. **Not shared**: Each kernel call has its own `memory` (static is per-kernel-instance)
3. **Incorrect AXI interface**: Declaring `m_axi` for an unused argument creates unnecessary hardware
4. **Misleading**: Suggests external memory access that doesn't exist

### Corrected Design
```cpp
void ram_a(int write_addr, int read_addr, int in, int write_en, int* out) {
    static int memory[1024];  // Internal static array
    #pragma HLS bind_storage variable=memory type=ram_1p impl=bram
    // No mem argument - memory is internal
}
```

### What the AXI Interface Would Have Meant

If `mem` were actually used with `m_axi`:
- **Hardware**: AXI4 master interface
- **Purpose**: Access external DDR memory
- **Behavior**: Burst transfers, address translation
- **Use Case**: Large arrays stored in DDR, accessed via AXI

But since `mem` was unused, the pragma was meaningless and created unnecessary hardware overhead.

## Summary Table

| Version | Interface Pragmas | Storage Pragma | Hardware |
|---------|------------------|----------------|----------|
| A | `ap_none` (all ports) | `ram_1p impl=bram` | Single-port BRAM |
| B | `ap_none` (all ports) | `ram_2p impl=bram` | Dual-port BRAM |
| C | `ap_none` (all ports) | `ram_1p impl=bram` | Single-port BRAM (init) |
| D | `ap_none` (all ports) | `ram_1p impl=bram` + partition | BRAM + registers |
| E | `ap_none` (all ports) | `ram_1p impl=bram` | Single-port BRAM |

## Key Takeaways

1. **Interface pragmas** control how arguments connect to hardware
2. **Storage pragmas** control physical resource allocation
3. **Static arrays** are internal to each kernel (not shared)
4. **Remove unused arguments** to avoid unnecessary hardware
5. **Use `bind_storage`** instead of deprecated `RESOURCE`


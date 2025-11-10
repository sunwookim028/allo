# Example 6: RAM Module Implementation

## Overview

This example demonstrates how to implement a **RAM (Random Access Memory)** module in Vitis HLS using static arrays. RAM provides random access to stored data, unlike shift registers which only provide sequential access.

## Interface

```cpp
void ram(mem, write_addr, read_addr, in, write_en, out)
```

- **mem**: Internal static memory array (not directly accessed externally)
- **write_addr**: Address to write data to
- **read_addr**: Address to read data from
- **in**: Input data to write
- **write_en**: Write enable signal (1 = write, 0 = no write)
- **out**: Output data (always reads from read_addr)

## Key Concept: Static Arrays as RAM

A **static array** in HLS becomes **BRAM (Block RAM)** in hardware when the array is large enough. BRAM provides:
- Random access (any address, any time)
- Large storage capacity
- Independent read/write ports (dual-port)

### Hardware Implementation

```
Write Port:        Read Port:
write_addr ──┐     read_addr ──┐
in ──────────┤     ────────────┤
write_en ────┤                  │
             ▼                  ▼
         [Address Decoder]  [Address Decoder]
             │                  │
             ▼                  ▼
         [BRAM Memory Array]
             │                  │
             └──────────────────┘
                          │
                          ▼
                         out
```

## Versions

### Version A: Basic RAM (Single-port)

**Features:**
- Separate read and write addresses
- Write only when `write_en=1`
- Read always outputs value
- Undefined values if never written

**Hardware:**
- Single-port BRAM
- Address decoders
- Write enable logic

**Use Case:** Basic memory operations

### Version B: Dual-port RAM

**Features:**
- Can read and write simultaneously
- Independent read/write addresses
- Higher throughput

**Hardware:**
- Dual-port BRAM
- Two address decoders
- Parallel access

**Use Case:** High-throughput applications needing simultaneous access

### Version C: RAM with Initialization

**Features:**
- All locations initialized to 0
- No undefined values
- Predictable behavior

**Hardware:**
- Same as Version A
- Initialization logic

**Use Case:** When you need known initial values

### Version D: RAM with Undefined Tracking

**Features:**
- Tracks which addresses have been written
- Can detect undefined values
- More complex but safer

**Hardware:**
- BRAM for data
- Additional memory for tracking
- More resources

**Use Case:** When you need to detect uninitialized reads

### Version E: Single-port RAM (Shared Address)

**Features:**
- Read and write share same address
- Simpler interface
- More common pattern

**Hardware:**
- Single-port BRAM
- Address multiplexer
- Write enable control

**Use Case:** Standard memory interface

## Behavior Demonstration

### Write Operation

```
write_addr = 5
in = 100
write_en = 1
→ memory[5] = 100
```

### Read Operation

```
read_addr = 5
→ out = memory[5] = 100
```

### Undefined Read

```
read_addr = 10 (never written)
→ out = undefined (likely 0, but not guaranteed)
```

## Running the Example

```bash
cd ex06_ram

# Test all versions
make all

# Test specific version
make test_a  # Basic RAM
make test_b  # Dual-port
make test_c  # With initialization
make test_d  # Undefined tracking
make test_e  # Single-port

# Run synthesis to see BRAM usage
make synth
make view
```

## Hardware Mapping

| Component | Hardware | Notes |
|-----------|----------|-------|
| `static int memory[1024]` | BRAM (1-2 blocks) | Block RAM |
| Address decoding | Logic | Address to memory |
| Write enable | Logic | Controls writes |
| Read port | BRAM output | Always active |

## Key Insights

1. **Static arrays → BRAM**: Large arrays become Block RAM
2. **Write only when enabled**: `write_en=1` required
3. **Read always active**: Outputs value at `read_addr`
4. **Undefined values**: Unwritten locations have undefined values
5. **Dual-port for throughput**: Simultaneous read/write

## Memory Types in HLS

| Type | Hardware | Size | Access |
|------|----------|------|--------|
| Small array (<64) | Registers | Small | Fast |
| Medium array (64-1024) | BRAM | Medium | Medium |
| Large array (>1024) | BRAM/URAM | Large | Slower |

## Common Patterns

### Pattern 1: Simple Read/Write
```cpp
static int mem[1024];
if (write_en) {
    mem[addr] = data_in;
}
data_out = mem[addr];
```

### Pattern 2: Dual-port Access
```cpp
static int mem[1024];
// Write port
if (write_en) {
    mem[write_addr] = data_in;
}
// Read port (independent)
data_out = mem[read_addr];
```

### Pattern 3: Initialized Memory
```cpp
static int mem[1024] = {0};  // All zeros
// Or with specific values
static int lut[256] = {0, 1, 2, ...};
```

## Applications

- **Lookup Tables**: Precomputed values
- **Buffers**: Data buffering
- **Caches**: Temporary storage
- **State Storage**: Storing state information
- **Data Structures**: Implementing arrays, stacks, queues

## Trade-offs

| Aspect | Single-port | Dual-port |
|--------|-------------|-----------|
| Throughput | Lower | Higher |
| Resources | Less BRAM | More BRAM |
| Complexity | Simpler | More complex |
| Use Case | Sequential | Parallel |

## BRAM Resource Usage

- **1 BRAM** = 18Kb (can store ~512 32-bit words)
- **1024 words** = ~2 BRAM blocks
- **Dual-port** uses more BRAM resources
- **Partitioning** can convert BRAM to registers

## Tips

1. **Use BRAM for large arrays**: Efficient storage
2. **Initialize if needed**: Avoid undefined values
3. **Dual-port for throughput**: When you need parallel access
4. **Consider partitioning**: Can improve performance
5. **Watch resource usage**: BRAM is limited resource

## Undefined Value Handling

**Problem:** Reading unwritten locations gives undefined values

**Solutions:**
1. **Initialize array**: `static int mem[1024] = {0};`
2. **Track writes**: Use separate array to track written locations
3. **Always initialize**: Write to all locations before use
4. **Use sentinel values**: Return special "undefined" marker

## Next Steps

- Modify array size and observe BRAM usage
- Try different read/write patterns
- Experiment with dual-port vs single-port
- Check synthesis reports for resource usage
- Compare with shift register implementation


## Interface and Resource Pragmas

See `PRAGMA_DOCUMENTATION.md` for detailed explanation of:
- Interface pragmas (`#pragma HLS INTERFACE`)
- Storage pragmas (`#pragma HLS bind_storage`)
- Physical hardware mapping
- Why deprecated `RESOURCE` pragma was replaced

## Important Note: Static Memory is NOT Shared

**Each kernel has its own static memory** - it is NOT shared across kernel calls:

```cpp
void ram_a(...) {
    static int memory[1024];  // This is LOCAL to ram_a
    // Each call to ram_a uses the SAME memory (persists)
    // But ram_b has a DIFFERENT memory array
}
```

**For shared memory across kernels**, see `ex08_shared_memory` which demonstrates:
- File-scope static variables (shared across kernels)
- Multiple kernels accessing the same memory
- Kernel composition (matmul + elemwise operations)


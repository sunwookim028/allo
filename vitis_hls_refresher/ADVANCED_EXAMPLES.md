# Advanced Examples: History Buffer, RAM, and Stream of Blocks

## Overview

These three examples extend the basic `static` concept to practical hardware modules:
1. **History Buffer** (ex05): Shift registers for delayed access
2. **RAM Module** (ex06): Random access memory with read/write
3. **Stream of Blocks** (ex07): Static with arbitrary C++ objects

## Example 5: History Buffer with Shift Registers

### Concept

A **history buffer** stores past values in a shift register, allowing access to data from N cycles ago. This is essential for:
- Signal processing (FIR filters, delays)
- Control systems (phase alignment)
- Communication (buffering)

### Key Implementation

```cpp
static int shift_reg[16] = {0};  // Static array = register chain

// Shift operation
for (int i = 15; i > 0; i--) {
    shift_reg[i] = shift_reg[i-1];  // Move data
}
shift_reg[0] = in;  // Insert new value

// Output from N cycles ago
*out = shift_reg[time];  // Tap at position 'time'
```

### Hardware Mapping

- **Static array** → **Register chain** (tapped delay line)
- Each element = one register (flip-flop)
- Shift operation = combinational logic
- Output selection = multiplexer (if configurable)

### When to Use

- ✅ Need delayed access to past values
- ✅ Fixed or configurable delay
- ✅ Pipeline-friendly pattern
- ❌ Don't use for random access (use RAM instead)

## Example 6: RAM Module

### Concept

A **RAM module** provides random access to stored data, unlike shift registers which only provide sequential access. Essential for:
- Lookup tables
- Data buffering
- State storage
- General-purpose memory

### Key Implementation

```cpp
static int memory[1024];  // Static array = BRAM

// Write operation
if (write_en && write_addr < 1024) {
    memory[write_addr] = in;  // Write only if enabled
}

// Read operation
*out = memory[read_addr];  // Always reads
```

### Hardware Mapping

- **Large static array** → **BRAM** (Block RAM)
- Address decoding → logic
- Write enable → control logic
- Read port → BRAM output

### RAM Types

| Type | Ports | Use Case |
|------|-------|----------|
| Single-port | 1 (shared) | Sequential access |
| Dual-port | 2 (independent) | Parallel read/write |

### When to Use

- ✅ Need random access
- ✅ Large storage requirements
- ✅ Independent read/write addresses
- ❌ Don't use for sequential access (shift register better)

## Example 7: Stream of Blocks

### Concept

Extends `static` to **arbitrary C++ objects** (structures, classes). Demonstrates that static works universally with any type.

### Key Implementation

```cpp
struct DataBlock {
    int value;
    int timestamp;
    bool valid;
    DataBlock() : value(0), timestamp(0), valid(false) {}
};

// Static structure
static DataBlock buffer;  // Becomes registers

// Static array of structures
static DataBlock history[16];  // Becomes BRAM
```

### Hardware Mapping

| Type | Size | Hardware |
|------|------|----------|
| `static DataBlock x` | Small | Registers (one per field) |
| `static DataBlock arr[16]` | Large | BRAM |
| `static ComplexBlock arr[4]` | Very Large | BRAM |

### HLS Stream Interface

```cpp
void func(hls::stream<DataBlock>& in, 
          hls::stream<DataBlock>& out) {
    #pragma HLS PIPELINE II=1
    DataBlock data = in.read();
    // Process...
    out.write(data);
}
```

**Benefits:**
- FIFO semantics
- Pipeline-friendly
- Hardware-efficient

### When to Use

- ✅ Complex data structures
- ✅ Structured data (packets, frames)
- ✅ State machines with complex state
- ✅ Streaming applications (use HLS Stream)

## Comparison Table

| Feature | History Buffer | RAM | Stream of Blocks |
|---------|---------------|-----|------------------|
| **Access Pattern** | Sequential (shift) | Random | Sequential (stream) |
| **Hardware** | Registers | BRAM | Registers/BRAM |
| **Use Case** | Delays, filters | Lookup, storage | Complex data |
| **Flexibility** | Fixed/configurable delay | Random access | Structured data |
| **Pipeline** | Excellent | Good | Excellent (with streams) |

## Design Patterns

### Pattern 1: Delay Line (History Buffer)
```cpp
static int delay[4] = {0};
delay[3] = delay[2];
delay[2] = delay[1];
delay[1] = delay[0];
delay[0] = input;
output = delay[3];  // 3-cycle delay
```

### Pattern 2: Random Access (RAM)
```cpp
static int mem[1024];
if (write_en) mem[addr] = data;
result = mem[addr];
```

### Pattern 3: Structured Buffer (Stream of Blocks)
```cpp
static DataBlock buffer[8];
for (int i = 7; i > 0; i--) {
    buffer[i] = buffer[i-1];
}
buffer[0] = input;
output = buffer[3];
```

## Choosing the Right Pattern

### Use History Buffer When:
- Need delayed access (N cycles ago)
- Sequential access pattern
- Small to medium delays (< 100 cycles)
- Pipeline-friendly design

### Use RAM When:
- Need random access
- Large storage requirements
- Independent read/write addresses
- Lookup tables, buffers

### Use Stream of Blocks When:
- Complex data structures
- Structured data (packets, frames)
- Streaming applications
- State machines with complex state

## Resource Considerations

### History Buffer
- **Resources**: Registers (FFs)
- **Size**: ~N registers for N-cycle delay
- **Efficient**: Yes, for small delays

### RAM
- **Resources**: BRAM blocks
- **Size**: 1 BRAM ≈ 512 words (32-bit)
- **Efficient**: Yes, for large storage

### Stream of Blocks
- **Resources**: Registers or BRAM (depends on size)
- **Size**: Varies with structure size
- **Efficient**: Depends on structure complexity

## Performance Tips

1. **History Buffer**: Use fixed delay when possible (more efficient)
2. **RAM**: Use dual-port for parallel access
3. **Stream of Blocks**: Use HLS Stream interface for streaming
4. **All**: Partition arrays to improve pipeline performance
5. **All**: Initialize properly to avoid undefined values

## Common Mistakes

### ❌ Mistake 1: Using RAM for Sequential Access
```cpp
// Bad: Using RAM for sequential access
static int mem[100];
for (int i = 0; i < 100; i++) {
    result = mem[i];  // Sequential, should use shift register
}
```

### ✅ Correct: Use Shift Register
```cpp
// Good: Shift register for sequential access
static int delay[100];
// Shift and tap
```

### ❌ Mistake 2: Using Shift Register for Random Access
```cpp
// Bad: Shift register for random access
static int shift[1024];
result = shift[random_addr];  // Can't do this efficiently
```

### ✅ Correct: Use RAM
```cpp
// Good: RAM for random access
static int mem[1024];
result = mem[random_addr];
```

### ❌ Mistake 3: Uninitialized Static Structures
```cpp
// Bad: No default constructor
struct Data {
    int value;
    // No constructor!
};
static Data d;  // Undefined!
```

### ✅ Correct: Initialize
```cpp
// Good: Default constructor
struct Data {
    int value;
    Data() : value(0) {}  // Initialize
};
static Data d;  // Initialized
```

## Next Steps

1. **Experiment**: Modify examples and observe behavior
2. **Combine**: Use multiple patterns together
3. **Optimize**: Try different pragmas and partitioning
4. **Analyze**: Check synthesis reports for resources
5. **Apply**: Use in your own designs

## Summary

These three examples demonstrate that `static` is a **powerful and universal concept** in HLS:

- **History Buffer**: Shows static arrays as shift registers
- **RAM**: Shows static arrays as memory
- **Stream of Blocks**: Shows static works with any type

All three leverage the same fundamental concept: **static variables persist across function calls and become hardware state** (registers or BRAM).

The key is understanding **when to use which pattern** based on your access pattern and requirements!


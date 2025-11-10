# New Examples Summary: History Buffer, RAM, and Stream of Blocks

## Overview

Three new advanced examples have been added to demonstrate practical applications of the `static` keyword in Vitis HLS:

1. **Example 5: History Buffer** - Shift registers for delayed access
2. **Example 6: RAM Module** - Random access memory implementation  
3. **Example 7: Stream of Blocks** - Static with arbitrary C++ objects

## Quick Start

```bash
# Run all new examples
cd vitis_hls_refresher
make ex05 ex06 ex07

# Or run individually
cd ex05_history_buffer && make csim
cd ex06_ram && make csim
cd ex07_stream_blocks && make csim
```

## Example 5: History Buffer

### Purpose
Implements a **history buffer** using shift registers that provides access to data from N cycles ago.

### Interface
```cpp
void history_buffer(buf, reset, in, out, time=1)
```

### Key Features
- **Configurable delay**: Supports delays from 1 to 15 cycles
- **Fixed delay option**: Most efficient for known delays
- **Pipeline-friendly**: Works well with `#pragma HLS PIPELINE`
- **Reset capability**: Clears buffer when needed

### Hardware Mapping
- **Static array** → **Register chain** (shift register)
- Each element = one register (flip-flop)
- Shift operation = combinational logic
- Output selection = multiplexer (if configurable)

### Example Usage
```cpp
// 3-cycle delay
history_buffer(buf, 0, 10, &out, 3);
// Input: 10, 20, 30, 40
// Output: 0, 0, 0, 10 (from 3 calls ago)
```

### Documentation
See `ex05_history_buffer/README.md` for detailed explanation.

## Example 6: RAM Module

### Purpose
Implements a **RAM module** with read/write interface for random access to stored data.

### Interface
```cpp
void ram(mem, write_addr, read_addr, in, write_en, out)
```

### Key Features
- **Write control**: Only writes when `write_en=1`
- **Read always active**: Always outputs value at `read_addr`
- **Dual-port option**: Simultaneous read/write
- **Undefined value handling**: Options for tracking unwritten locations

### Hardware Mapping
- **Large static array** → **BRAM** (Block RAM)
- Address decoding → logic
- Write enable → control logic
- Read port → BRAM output

### Example Usage
```cpp
// Write operation
ram(mem, 5, 0, 100, 1, &out);  // Write 100 to address 5

// Read operation
ram(mem, 0, 5, 0, 0, &out);    // Read from address 5
// out = 100
```

### Documentation
See `ex06_ram/README.md` for detailed explanation.

## Example 7: Stream of Blocks

### Purpose
Demonstrates how `static` works with **arbitrary C++ objects** (structures, classes, arrays of structures).

### Key Features
- **Structures**: Static structures become registers
- **Arrays of structures**: Become BRAM or registers
- **HLS Stream interface**: Efficient streaming with `hls::stream<>`
- **Complex types**: Works with nested structures, arrays, etc.

### Hardware Mapping
| Type | Size | Hardware |
|------|------|----------|
| `static DataBlock x` | Small | Registers (one per field) |
| `static DataBlock arr[16]` | Large | BRAM |
| `static ComplexBlock arr[4]` | Very Large | BRAM |

### Example Usage
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

### Documentation
See `ex07_stream_blocks/README.md` for detailed explanation.

## Key Insights

### 1. Static is Universal
The `static` keyword works with **any C++ type**:
- Primitive types (`int`, `float`)
- Structures (`struct DataBlock`)
- Classes (custom classes)
- Arrays of any type
- Complex nested structures

### 2. Hardware Mapping Depends on Size
- **Small** (< 64 bytes): Registers (FFs)
- **Large** (> 64 bytes): BRAM
- **Very Large** (> 1KB): BRAM/URAM

### 3. Access Pattern Determines Implementation
- **Sequential access** → Shift register (History Buffer)
- **Random access** → RAM
- **Streaming** → HLS Stream interface

### 4. Initialization is Critical
Always initialize static objects:
```cpp
// Good
static DataBlock state = DataBlock(0, 0, false);
// Or
struct DataBlock {
    DataBlock() : value(0) {}  // Default constructor
};
```

## Comparison Table

| Feature | History Buffer | RAM | Stream of Blocks |
|---------|---------------|-----|------------------|
| **Access** | Sequential | Random | Sequential |
| **Hardware** | Registers | BRAM | Registers/BRAM |
| **Use Case** | Delays | Storage | Complex data |
| **Pipeline** | Excellent | Good | Excellent |

## When to Use Each

### Use History Buffer When:
- ✅ Need delayed access (N cycles ago)
- ✅ Sequential access pattern
- ✅ Small to medium delays
- ✅ Pipeline-friendly design

### Use RAM When:
- ✅ Need random access
- ✅ Large storage requirements
- ✅ Independent read/write addresses
- ✅ Lookup tables, buffers

### Use Stream of Blocks When:
- ✅ Complex data structures
- ✅ Structured data (packets, frames)
- ✅ Streaming applications
- ✅ State machines with complex state

## Running the Examples

### Example 5: History Buffer
```bash
cd ex05_history_buffer
make test_a    # Configurable delay
make test_b    # Pipelined version
make test_c    # Fixed delay (most efficient)
make synth     # See hardware implementation
```

### Example 6: RAM Module
```bash
cd ex06_ram
make test_a    # Basic RAM
make test_b    # Dual-port RAM
make test_c    # With initialization
make test_d    # Undefined tracking
make test_e    # Single-port RAM
make synth     # See BRAM usage
```

### Example 7: Stream of Blocks
```bash
cd ex07_stream_blocks
make test_a    # Static structure
make test_b    # Array of structures
make test_c    # Accumulator
make test_e    # Complex structure
make test_f    # With reset
make synth     # See resource usage
```

## Documentation Files

Each example includes comprehensive documentation:

- **ex05_history_buffer/README.md**: History buffer concepts and patterns
- **ex06_ram/README.md**: RAM implementation and memory types
- **ex07_stream_blocks/README.md**: Static with complex types
- **ADVANCED_EXAMPLES.md**: Comparison and design patterns

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

## Common Mistakes to Avoid

### ❌ Using RAM for Sequential Access
```cpp
// Bad: Sequential access with RAM
for (int i = 0; i < 100; i++) {
    result = mem[i];  // Should use shift register
}
```

### ❌ Using Shift Register for Random Access
```cpp
// Bad: Random access with shift register
result = shift_reg[random_addr];  // Should use RAM
```

### ❌ Uninitialized Static Structures
```cpp
// Bad: No default constructor
struct Data {
    int value;
    // No constructor!
};
static Data d;  // Undefined!
```

## Next Steps

1. **Run all examples**: Understand each pattern
2. **Modify code**: Experiment with changes
3. **Check synthesis**: See hardware mapping
4. **Compare versions**: Understand trade-offs
5. **Apply to your designs**: Use patterns in practice

## Summary

These three examples demonstrate that `static` is a **powerful and universal concept** in HLS:

- **History Buffer**: Shows static arrays as shift registers
- **RAM**: Shows static arrays as memory
- **Stream of Blocks**: Shows static works with any type

All leverage the same fundamental concept: **static variables persist across function calls and become hardware state** (registers or BRAM).

The key is understanding **when to use which pattern** based on your access pattern and requirements!


# Example 7: HLS Stream of Blocks - Extending Static to Arbitrary Objects

## Overview

This example demonstrates how the `static` keyword works with **arbitrary C++ objects** (structures, classes, arrays of structures) in Vitis HLS. This extends the static concept beyond simple types to complex data structures, showing that static works universally with any C++ type.

## Key Concept: Static with Complex Types

The `static` keyword in HLS works with **any C++ type**:
- **Primitive types**: `int`, `float`, `bool`
- **Structures**: `struct DataBlock`
- **Classes**: Custom classes
- **Arrays of structures**: `DataBlock history[16]`
- **Nested structures**: Structures containing arrays

### Hardware Mapping

| C++ Type | Small (<64 bytes) | Large (>64 bytes) |
|----------|-------------------|-------------------|
| `static int x` | Register (FF) | Register (FF) |
| `static DataBlock x` | Registers | BRAM |
| `static DataBlock arr[10]` | Registers (if partitioned) | BRAM |
| `static ComplexBlock arr[4]` | BRAM | BRAM |

## Versions

### Version A: Static Structure (Single Object)

**Features:**
- Single static structure
- Simple buffering
- Demonstrates structure persistence

**Hardware:**
- Registers for structure fields
- Each field becomes a register

**Use Case:** Simple state storage

### Version B: Static Array of Structures (History Buffer)

**Features:**
- Array of structures as shift register
- Delayed output
- Similar to Example 5 but with structures

**Hardware:**
- BRAM or registers (depending on size)
- Shift register pattern

**Use Case:** History buffer with structured data

### Version C: Static Accumulator Structure

**Features:**
- Structure accumulates values
- Persists across calls
- Demonstrates stateful processing

**Hardware:**
- Registers for accumulator fields
- Accumulation logic

**Use Case:** Running totals, statistics

### Version D: HLS Stream Interface (Recommended)

**Features:**
- Uses `hls::stream<DataBlock>` interface
- Efficient streaming
- Pipeline-friendly

**Hardware:**
- FIFO-like interface
- Optimized for streaming

**Use Case:** High-throughput streaming applications

### Version E: Complex Structure with Nested Arrays

**Features:**
- Structure contains arrays
- Large memory footprint
- Demonstrates BRAM mapping

**Hardware:**
- BRAM for large structures
- Partitioning possible

**Use Case:** Complex data structures

### Version F: Static Structure with Reset

**Features:**
- Reset capability
- Controlled state
- Safe initialization

**Hardware:**
- Registers with reset logic
- Reset network

**Use Case:** State machines, resettable accumulators

## Structure Definition

```cpp
struct DataBlock {
    int value;
    int timestamp;
    bool valid;
    
    // Default constructor (IMPORTANT for static)
    DataBlock() : value(0), timestamp(0), valid(false) {}
    
    // Parameterized constructor
    DataBlock(int v, int t, bool vld) : value(v), timestamp(t), valid(vld) {}
};
```

**Key Points:**
- **Default constructor required**: Static objects need initialization
- **Member initialization**: Use initializer lists
- **Copy semantics**: Structures are copied, not referenced

## Behavior Demonstration

### Static Structure Persistence

```cpp
static DataBlock state;
// Call 1: state.value = 10
// Call 2: state.value still = 10 (persists!)
// Call 3: state.value = 20 (updated)
```

### Array of Structures as Shift Register

```cpp
static DataBlock history[4];
// Shift operation moves structures
history[3] = history[2];
history[2] = history[1];
history[1] = history[0];
history[0] = new_data;
```

## Running the Example

```bash
cd ex07_stream_blocks

# Test all versions
make all

# Test specific version
make test_a  # Static structure
make test_b  # Array of structures
make test_c  # Accumulator
make test_e  # Complex structure
make test_f  # With reset

# Run synthesis to see resource usage
make synth
make view
```

## Hardware Mapping

| Component | Hardware | Notes |
|-----------|----------|-------|
| `static DataBlock x` | Registers (3 FFs) | One per field |
| `static DataBlock arr[16]` | BRAM | Large array |
| Structure copy | Logic | Copy operations |
| Nested arrays | BRAM | Large structures |

## Key Insights

1. **Static works with any type**: Structures, classes, arrays
2. **Default constructor required**: For static initialization
3. **Size determines mapping**: Small → registers, Large → BRAM
4. **HLS Stream for efficiency**: Better than pointer-based
5. **Initialization matters**: Always initialize static objects

## HLS Stream Interface

### Why Use Streams?

```cpp
// Pointer-based (less efficient)
void func(DataBlock* in, DataBlock* out);

// Stream-based (more efficient)
void func(hls::stream<DataBlock>& in, hls::stream<DataBlock>& out);
```

**Benefits:**
- **FIFO semantics**: Natural for streaming
- **Pipeline-friendly**: Works well with pipelines
- **Hardware-efficient**: Maps to FIFOs in hardware
- **Type-safe**: Template-based

### Stream Usage Pattern

```cpp
void process_stream(hls::stream<DataBlock>& in, 
                   hls::stream<DataBlock>& out) {
    #pragma HLS PIPELINE II=1
    
    DataBlock data = in.read();  // Blocking read
    // Process...
    out.write(data);  // Blocking write
}
```

## Common Patterns

### Pattern 1: Static State Structure
```cpp
struct State {
    int counter;
    bool flag;
    State() : counter(0), flag(false) {}
};

static State state;
state.counter++;
state.flag = true;
```

### Pattern 2: Static Array of Structures
```cpp
static DataBlock buffer[8];
// Shift register pattern
for (int i = 7; i > 0; i--) {
    buffer[i] = buffer[i-1];
}
buffer[0] = input;
```

### Pattern 3: Complex Static Structure
```cpp
struct Packet {
    int header[4];
    int data[64];
    int checksum;
};

static Packet packet_buffer[4];
// Large → BRAM
```

## Applications

- **Packet Processing**: Network packets, data frames
- **Signal Processing**: Complex signal data
- **Image Processing**: Pixels, windows, blocks
- **State Machines**: Complex state structures
- **Data Structures**: Queues, stacks, buffers

## Trade-offs

| Aspect | Small Structure | Large Structure |
|--------|----------------|-----------------|
| Hardware | Registers | BRAM |
| Speed | Fast | Slower |
| Resources | FFs | BRAM blocks |
| Use Case | Simple state | Complex data |

## Tips

1. **Always provide default constructor**: Required for static
2. **Initialize members**: Avoid undefined values
3. **Use streams for streaming**: More efficient
4. **Consider size**: Determines hardware mapping
5. **Partition large arrays**: Can improve performance

## Initialization Best Practices

### ✅ Good: Explicit Initialization
```cpp
static DataBlock state = DataBlock(0, 0, false);
// Or
static DataBlock state;
// Then initialize in code
```

### ❌ Bad: Uninitialized
```cpp
static DataBlock state;  // Members undefined!
// Use without initialization
```

### ✅ Good: Default Constructor
```cpp
struct DataBlock {
    int value;
    DataBlock() : value(0) {}  // Initialize
};
```

## Next Steps

- Create your own structures
- Experiment with different sizes
- Try HLS Stream interface
- Compare register vs BRAM usage
- Build complex data structures

## Extending Further

The static concept extends to:
- **Classes with methods**: Static class instances
- **Templates**: Static template instantiations
- **Nested structures**: Complex hierarchies
- **Unions**: Static unions
- **Any C++ type**: Universally applicable

This demonstrates that **static is a fundamental concept** that works uniformly across all C++ types in HLS!


## Version D: HLS Stream Interface - Efficiency Demonstration

### Purpose

Version D demonstrates the **efficiency advantages** of using HLS Stream interface (`hls::stream<>`) compared to pointer-based interfaces for streaming applications.

### Comparison Test

The testbench compares:
- **Baseline**: Version B (pointer-based) - similar functionality
- **Streaming**: Version D (HLS Stream) - same functionality with stream interface

### Key Advantages Demonstrated

1. **FIFO Semantics**
   - Natural for streaming data
   - Automatic flow control
   - Backpressure handling

2. **Pipeline Efficiency**
   - Achieves II=1 easily
   - Better resource utilization
   - Higher throughput

3. **Hardware Mapping**
   - Maps to FIFOs in hardware
   - More efficient than pointer-based
   - Better for dataflow designs

4. **Code Safety**
   - No manual pointer management
   - Type-safe interface
   - Cleaner code

### Running the Comparison

```bash
cd ex07_stream_blocks
make test_d
```

The testbench will:
1. Run baseline (Version B) with pointer-based interface
2. Run streaming (Version D) with HLS Stream interface
3. Show output comparison
4. Highlight efficiency advantages

### Expected Output

```
Baseline (Version B - pointer-based):
  Call 4: in=40, out=10 (expected: 10)
  Call 5: in=50, out=20 (expected: 20)

Streaming (Version D - hls::stream):
  Processing stream with II=1 pipeline...
  Cycle 4: in=40, out=10 (expected: 10)
  Cycle 5: in=50, out=20 (expected: 20)

Key Advantages of Stream Interface:
  - FIFO semantics: Natural for streaming data
  - Pipeline-friendly: Achieves II=1 easily
  - Hardware-efficient: Maps to FIFOs in hardware
  - Backpressure handling: Automatic flow control
  - No manual pointer management: Safer and cleaner
```

### When to Use Stream Interface

✅ **Use `hls::stream<>` when:**
- Streaming data (continuous flow)
- Need high throughput
- Want II=1 pipelines
- Building dataflow designs
- Need backpressure handling

❌ **Use pointers when:**
- Random access needed
- Single-shot operations
- Simple interfaces
- Legacy code compatibility


# Example 5: History Buffer with Shift Registers

## Overview

This example demonstrates how to implement a **history buffer** using shift registers in Vitis HLS. A history buffer stores past values and allows you to access data from N cycles ago, which is essential for many signal processing and control applications.

## Interface

```cpp
void history_buffer(buf, reset, in, out, time=1)
```

- **buf**: Internal static buffer (shift register) - not directly accessed
- **reset**: Reset signal (clears entire buffer)
- **in**: Input value to store
- **out**: Output value (delayed by 'time' cycles)
- **time**: Delay amount (default 1, meaning output from 1 invocation ago)

## Key Concept: Shift Registers

A **shift register** is a chain of registers where data moves from one register to the next on each clock cycle. This creates a "tapped delay line" where you can access values from different points in time.

### Hardware Implementation

```
Input → [R0] → [R1] → [R2] → [R3] → ... → [R15]
         ↓       ↓       ↓       ↓
        out(t=0) out(t=1) out(t=2) out(t=3)
```

- **Static array** becomes a **register chain**
- Each element stores value from N cycles ago
- Tapping at position `time` gives you delayed output

## Versions

### Version A: Configurable Delay

**Features:**
- Supports delays from 1 to 15 cycles
- Flexible but uses more resources
- Good for applications needing variable delays

**Hardware:**
- 16 registers (for max delay of 15)
- Multiplexer for output selection
- Shift logic for moving data

**Use Case:** When delay amount changes at runtime

### Version B: Pipelined Version

**Features:**
- Same as Version A but with pipeline pragma
- Achieves II=1 (one new input per cycle)
- Optimized for high throughput

**Hardware:**
- Same as Version A
- Pipeline registers for timing
- Unrolled shift operations

**Use Case:** High-throughput streaming applications

### Version C: Fixed Delay (Most Efficient)

**Features:**
- Fixed 3-cycle delay
- Minimal hardware (only 4 registers needed)
- Most efficient for fixed delays

**Hardware:**
- 4 registers only
- Simple shift chain
- No multiplexer needed

**Use Case:** When delay is known at compile time

## Behavior Demonstration

### Example: 3-Cycle Delay

```
Input sequence:  10, 20, 30, 40, 50
Output (time=3):  0,  0,  0, 10, 20
                  ↑   ↑   ↑   ↑   ↑
                fill fill fill valid valid
```

**Explanation:**
- First 3 calls: Buffer is filling, output is 0 (or undefined)
- Call 4: Output is 10 (from 3 calls ago)
- Call 5: Output is 20 (from 3 calls ago)

## Running the Example

```bash
cd ex05_history_buffer

# Test all versions
make all

# Test specific version
make test_a  # Configurable delay
make test_b  # Pipelined
make test_c  # Fixed delay

# Run synthesis to see hardware
make synth
make view
```

## Hardware Mapping

| Component | Hardware | Notes |
|-----------|----------|-------|
| `static int shift_reg[16]` | 16 Registers (FFs) | Register chain |
| Shift operation | Combinational logic | Moves data |
| Output selection | Multiplexer | Selects tap |
| Reset logic | Reset network | Clears all registers |

## Key Insights

1. **Static arrays create shift registers**: Each element is a register
2. **Delay = position in array**: `shift_reg[3]` = 3 cycles ago
3. **Fixed delay is more efficient**: No multiplexer needed
4. **Pipeline-friendly**: Shift registers work well with pipelines
5. **Reset is important**: Clears state when needed

## Common Patterns

### Pattern 1: Simple Delay Line
```cpp
static int delay[4] = {0};
delay[3] = delay[2];
delay[2] = delay[1];
delay[1] = delay[0];
delay[0] = input;
output = delay[3];  // 3-cycle delay
```

### Pattern 2: Tapped Delay Line
```cpp
static int delay[8] = {0};
// Shift...
output1 = delay[1];  // 1-cycle delay
output2 = delay[3];  // 3-cycle delay
output3 = delay[7];  // 7-cycle delay
```

### Pattern 3: Moving Average (Extension)
```cpp
static int history[10] = {0};
// Shift and store
int sum = 0;
for (int i = 0; i < 10; i++) {
    sum += history[i];
}
average = sum / 10;
```

## Applications

- **Signal Processing**: FIR filters, moving averages
- **Control Systems**: Delay compensation, phase alignment
- **Communication**: Buffering, synchronization
- **Image Processing**: Line buffers, window operations

## Trade-offs

| Aspect | Configurable Delay | Fixed Delay |
|--------|-------------------|-------------|
| Flexibility | High | Low |
| Hardware | More (mux) | Less |
| Performance | Good | Best |
| Use Case | Variable delay | Fixed delay |

## Tips

1. **Use fixed delay when possible**: More efficient
2. **Partition arrays**: Improves pipeline performance
3. **Initialize properly**: Avoid undefined values
4. **Consider reset**: May need to clear state
5. **Pipeline for throughput**: Use pragma HLS PIPELINE

## Next Steps

- Modify delay amount and observe behavior
- Try different array sizes
- Experiment with pipeline pragmas
- Check synthesis reports for resource usage


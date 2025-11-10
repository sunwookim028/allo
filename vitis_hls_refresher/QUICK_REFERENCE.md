# Quick Reference: Static Keyword in Vitis HLS

## The `static` Keyword - What It Does

### In Software (C/C++)
- **Scope**: Variable persists across function calls
- **Storage**: Allocated in static memory (not stack)
- **Initialization**: Happens once, before first use

### In Hardware (HLS → RTL)
- **Storage**: Becomes a **register** (flip-flop) or **BRAM**
- **State**: Retains value between function invocations
- **Reset**: Requires explicit reset logic (or uses initialization)

## Quick Decision Tree

```
Do you need state between function calls?
├─ NO → Use non-static (wire/logic)
│   └─ Cleaner pipelines, less hardware
│
└─ YES → Use static (register/BRAM)
    ├─ Small variable (< 64 bits)?
    │   └─ static int → Register (FF)
    │
    └─ Array or large structure?
        ├─ Small array (< 64 elements)?
        │   └─ static int arr[N] → Registers (if partitioned)
        │
        └─ Large array (> 64 elements)?
            └─ static int arr[N] → BRAM/URAM
```

## Code Patterns

### Pattern 1: Accumulator (Pipeline-friendly)
```cpp
int acc = 0;  // Non-static: better for pipelines
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    acc = acc + data[i];
}
```

### Pattern 2: Shift Register (Requires static)
```cpp
static int shift_reg[5] = {0};  // Static: required
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=shift_reg complete
for (int i = 0; i < N; i++) {
    // Shift operation
    for (int j = 4; j > 0; j--) {
        shift_reg[j] = shift_reg[j-1];
    }
    shift_reg[0] = data[i];
}
```

### Pattern 3: State Machine (Requires static)
```cpp
static int state = 0;  // Static: state persists
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    if (state == 0) {
        // Process state 0
        if (condition) state = 1;
    } else {
        // Process state 1
        if (condition) state = 0;
    }
}
```

### Pattern 4: Lookup Table (Requires static)
```cpp
static int LUT[256] = {0, 1, 2, ...};  // Static: initialized once
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    result[i] = LUT[data[i]];
}
```

## Hardware Mapping Summary

| C++ Construct | Small (≤64B) | Large (>64B) | Notes |
|---------------|--------------|--------------|-------|
| `int x;` | Wire/Logic | N/A | Combinational |
| `static int x;` | Register (FF) | Register (FF) | Sequential |
| `int arr[10];` | Registers* | BRAM | *If partitioned |
| `static int arr[10];` | Registers* | BRAM | Persistent |
| `static int arr[1000];` | N/A | BRAM/URAM | Persistent |

## Pipeline Impact Table

| Pattern | Non-static | Static | II Achievable |
|---------|------------|--------|---------------|
| Accumulator | ✅ Good | ❌ Problematic | II=1 (non-static) |
| Shift Register | ❌ Not possible | ✅ Good | II=1 |
| State Machine | ❌ Not possible | ✅ Good | II=1 (with stalls) |
| Lookup Table | ❌ Not efficient | ✅ Good | II=1 |
| Loop Counter | ✅ Good | ⚠️ Rarely needed | II=1 |

## Common Pitfalls

### ❌ Pitfall 1: Static Accumulator in Pipeline
```cpp
static int acc = 0;  // BAD: Prevents II=1
#pragma HLS PIPELINE II=1
for (int i = 0; i < N; i++) {
    acc = acc + data[i];  // Static persists across calls
}
```
**Problem**: Static variable persists across function calls, breaking pipeline assumptions.

**Fix**: Use non-static accumulator
```cpp
int acc = 0;  // GOOD: Resets each call
```

### ❌ Pitfall 2: Uninitialized Static Variable
```cpp
static int state;  // BAD: Uninitialized
// Use without initialization
```
**Problem**: Undefined behavior.

**Fix**: Always initialize
```cpp
static int state = 0;  // GOOD: Initialized
```

### ❌ Pitfall 3: Static Array Without Reset
```cpp
static int history[100];  // Accumulates forever
// No reset mechanism
```
**Problem**: State accumulates indefinitely.

**Fix**: Add reset control
```cpp
static int history[100];
if (reset) {
    for (int i = 0; i < 100; i++) history[i] = 0;
}
```

## Debugging Tips

### Check Behavior
```bash
make csim  # Fast simulation
cat <project>/solution1/sim/report/csim.log
```

### Check Resources
```bash
make synth  # Synthesis
cat <project>/solution1/syn/report/top_csynth.rpt | grep -E "BRAM|FF|LUT|DSP"
```

### Check Pipeline
```bash
cat <project>/solution1/syn/report/top_csynth.rpt | grep "Initiation Interval"
```

## When to Use Static

✅ **Use static when:**
- State needs to persist between function calls
- Building shift registers
- Implementing state machines
- Creating lookup tables
- Need shared state across invocations

❌ **Avoid static when:**
- Building simple accumulators in loops
- Want clean pipelining (II=1)
- Variable should reset each call
- No need for state persistence

## Performance Tips

1. **Small static variables**: Use registers (fast, low area)
2. **Large static arrays**: Use BRAM (slow access, high area efficiency)
3. **Partition static arrays**: Can improve throughput
4. **Initialize statics**: Prevents undefined behavior
5. **Reset mechanism**: Consider if reset is needed

## Quick Commands

```bash
# Run all examples
make all

# Run specific example
cd ex01_basic_static && make csim

# Compare versions
../common/compare_versions.sh ex02_static_variables top_a top_b

# View report
../common/view_report.sh ex01_basic_static/ex01.prj
```


# MLIR ↔ Vitis HLS Stateful Variable Translation

This project demonstrates how Allo's Stateful type is translated from MLIR to Vitis HLS C++ code, specifically showing that stateful variables are correctly emitted as `static` variables **inside** the functions that use them.

## Overview

When you declare a `Stateful[int32]` variable in Allo, it:
1. Creates a `memref.global` operation in MLIR with naming pattern `*_stateful_*`
2. Gets translated to a `static` variable in the generated HLS C++ code
3. Is correctly placed **inside** the function body (not at module level)

## Quick Start

### Run the Test

```bash
cd mlir_to_hls
python3 test_stateful.py
```

This will:
- Show the input MLIR
- Translate it to Vitis HLS C++
- Verify the static variable is inside the function
- Save output to `generated_output.cpp`

### Expected Output

The test should show:
```
✓ PASSED: Static variable declared INSIDE test() function
✓ All tests passed!
✓ Output saved to: generated_output.cpp
```

### Inspect Generated Code

```bash
cat generated_output.cpp
```

You should see:
```cpp
void test(
  int32_t v0,
  int32_t *v1
) {
  static int32_t acc_stateful_5700927378943735518 = {1};  // ← Inside function!
  // ... rest of function body
}
```

## Test Scripts

- **`test_stateful.py`** - Main test script (run this!)
  - Tests MLIR → HLS translation
  - Verifies static variable placement
  - Generates `generated_output.cpp`

- **`demo_stateful.py`** - Demonstration script
  - Shows complete flow: Allo → MLIR → HLS
  - Educational walkthrough

- **`demo_run.py`** - Simple translation demo
  - Basic MLIR to HLS translation example

## Test Files

- **`test_input.mlir`** - MLIR input file for testing
- **`generated_output.cpp`** - Generated HLS C++ output (updated by test script)

## How It Works

### 1. Allo Code
```python
import allo
from allo.ir.types import int32, Stateful

def test(x: int32) -> int32:
    acc: Stateful[int32] = 1
    acc += x
    return acc
```

### 2. Generated MLIR
```mlir
module {
  memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<1>
  func.func @test(%arg0: i32) -> i32 {
    %0 = memref.get_global @acc_stateful_5700927378943735518 : memref<i32>
    ...
  }
}
```

### 3. Generated HLS C++
```cpp
void test(int32_t v0, int32_t *v1) {
  static int32_t acc_stateful_5700927378943735518 = {1};  // ← Static inside function!
  ...
}
```

## Implementation Details

The translation pass (`mlir/lib/Translation/EmitVivadoHLS.cpp`) recognizes stateful variables by their naming pattern (`_stateful_`) and:

1. **Collects** stateful globals used by each function
2. **Emits** them as `static` variables inside the function body
3. **Skips** emitting them at module level

This ensures stateful variables persist across kernel invocations while being scoped to the function that uses them.

## Prerequisites

- Python 3 with Allo installed
- Built Allo MLIR translation library (for Python bindings)

## Running Tests

### Main Test
```bash
python3 test_stateful.py
```

### Unit Test (from project root)
```bash
python3 tests/test_stateful.py
```

## Files

- `test_stateful.py` - Main test script (run this!)
- `demo_stateful.py` - Educational demonstration
- `demo_run.py` - Simple translation demo
- `test_input.mlir` - MLIR test input
- `generated_output.cpp` - Generated HLS output
- `README.md` - This file
- `STATIC_VS_GLOBAL.md` - Documentation on static vs global distinction

## Related Documentation

- `STATIC_VS_GLOBAL.md` - Explains how static vs global variables are distinguished
- `TRANSLATION_WALKTHROUGH.md` - Detailed walkthrough of translation process
- `GET_GLOBAL_WALKTHROUGH.md` - Guide to accessing globals in MLIR

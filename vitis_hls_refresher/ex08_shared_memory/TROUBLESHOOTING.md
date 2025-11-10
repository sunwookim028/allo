# Troubleshooting Guide for ex08: Shared BRAM

## Common Issues and Solutions

### Issue 1: Testbench Shows Zeros Instead of Expected Values

**Symptom:**
When running Test 5, the output shows zeros instead of computed values:
```
Step 2: MatMul Engine computes W*X
  First 4 elements: 0 0 0 0  (expected: 1 2 3 4)

Step 3: Elemwise Engine reads intermediate result
  First 4 elements: 0 0 0 0  (expected: 2 3 4 5)
```

**Root Cause:**
The testbench and kernel are accessing **different instances** of `shared_memory`. This happens when:
- Testbench declares `static int shared_memory[4096]` (creates separate instance)
- Testbench doesn't declare `extern int shared_memory[4096]` at file scope
- The `extern` declaration is inside a function instead of at file scope

**Solution:**

1. **In `kernel.cpp`**: Ensure `shared_memory` is declared at file scope (non-static):
   ```cpp
   // File scope - shared by all kernels
   int shared_memory[4096];  // Non-static for testbench access
   ```

2. **In `tb.cpp`**: Declare `shared_memory` as `extern` at **file scope** (before `main()`):
   ```cpp
   // Forward declarations
   extern void matmul_shared(...);
   extern void elemwise_shared(...);
   
   // CRITICAL: Must be at file scope, not inside main()!
   extern int shared_memory[4096];
   
   int main() {
       // Now can access shared_memory
       memcpy(&shared_memory[0], data, size);
       matmul_shared(...);
   }
   ```

3. **DO NOT** declare `static int shared_memory[4096]` in the testbench - this creates a separate instance!

**Verification:**
After fixing, the test should show:
```
Step 2: MatMul Engine computes W*X
  First 4 elements: 1 2 3 4  (expected: 1 2 3 4) ✅

Step 3: Elemwise Engine reads intermediate result
  First 4 elements: 2 3 4 5  (expected: 2 3 4 5) ✅
```

**Key Points:**
- `extern` declaration must be at **file scope** (outside any function)
- `extern` tells the compiler to link to the definition in `kernel.cpp`
- `static` in testbench creates a **separate instance** - avoid this!
- In hardware, file-scope variables are shared across kernels automatically

**Compilation Check:**
If you see compilation errors like:
```
error: 'shared_memory' was not declared in this scope
```
This means the `extern` declaration is missing or in the wrong scope.

**Standalone Test:**
To verify linkage works, compile separately:
```bash
g++ -std=c++11 -I/opt/xilinx/Vitis_HLS/2023.2/include tb.cpp kernel.cpp -o test
./test
```
If this works but Vitis HLS doesn't, check that the `extern` is at file scope.

---

### Issue 2: Compilation Error: 'shared_memory' was not declared

**Symptom:**
```
../../../../tb.cpp:106:13: error: 'shared_memory' was not declared in this scope
     memcpy(&shared_memory[W_OFFSET], W, M * K * sizeof(int));
             ^~~~~~~~~~~~~
```

**Solution:**
Add `extern int shared_memory[4096];` at file scope in `tb.cpp` (before `main()`).

---

### Issue 3: Multiple Definition Error

**Symptom:**
```
error: multiple definition of 'shared_memory'
```

**Solution:**
- In `kernel.cpp`: Use `int shared_memory[4096];` (definition)
- In `tb.cpp`: Use `extern int shared_memory[4096];` (declaration only)
- Do NOT define `shared_memory` in both files!

---

### Issue 4: Kernels Don't Share Data

**Symptom:**
Kernel 1 writes data, but Kernel 2 reads zeros or uninitialized values.

**Solution:**
- Ensure `shared_memory` is declared at **file scope** in `kernel.cpp` (not inside functions)
- Both kernels must use the same `shared_memory` variable name
- Both kernels must have the same `bind_storage` pragma for `shared_memory`
- Check that offsets are correct (no overlap or out-of-bounds access)

---

## Quick Reference: Correct Setup

### kernel.cpp
```cpp
// File scope - shared by all kernels
int shared_memory[4096];  // Definition (non-static)

void matmul_shared(...) {
    #pragma HLS bind_storage variable=shared_memory type=ram_2p impl=bram
    // Uses shared_memory
}

void elemwise_shared(...) {
    #pragma HLS bind_storage variable=shared_memory type=ram_2p impl=bram
    // Uses shared_memory
}
```

### tb.cpp
```cpp
// Forward declarations
extern void matmul_shared(...);
extern void elemwise_shared(...);

// CRITICAL: File scope extern declaration
extern int shared_memory[4096];

int main() {
    // Can now access shared_memory
    memcpy(&shared_memory[0], data, size);
    matmul_shared(...);
    // Read results from shared_memory
}
```

---

## Debugging Tips

1. **Verify linkage**: Compile standalone to check if extern linkage works
2. **Check scope**: Ensure `extern` is at file scope, not function scope
3. **Check static**: Remove any `static` declarations of `shared_memory` in testbench
4. **Verify offsets**: Print offsets to ensure memory layout is correct
5. **Check pragmas**: Ensure both kernels use the same `bind_storage` pragma

---

## Summary

The key to proper BRAM sharing:
- ✅ File-scope `int shared_memory[4096]` in `kernel.cpp` (definition)
- ✅ File-scope `extern int shared_memory[4096]` in `tb.cpp` (declaration)
- ✅ Same variable name in both files
- ✅ Same `bind_storage` pragma in both kernels
- ❌ NO `static` declaration in testbench
- ❌ NO `extern` inside functions

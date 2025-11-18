# Static vs Global Variables in MLIR to HLS Translation

## Summary

The translation code **CAN** distinguish between static and global variables from MLIR code by checking for the `hls.static` attribute on `memref.global` operations.

## How It Works

In `mlir/lib/Translation/EmitVivadoHLS.cpp` (line 1263):

```cpp
void ModuleEmitter::emitGlobal(memref::GlobalOp op) {
  // ...
  if (op->hasAttr("hls.static")) {
    os << "static ";
  }
  // ...
}
```

- **With `hls.static` attribute**: Emits `static int32_t variable_name = {0};`
- **Without `hls.static` attribute**: Emits `int32_t variable_name = {0};` (global)

## Current Issue

The MLIR text syntax for `hls.static` attribute is causing parsing errors. The following syntaxes have been tested:

1. `memref.global "private" @name : memref<i32> = dense<0> { hls.static }` ❌ Parse error
2. `memref.global "private" @name : memref<i32> = dense<0> : memref<i32> { hls.static }` ❌ Parse error

## Current Behavior

Currently, `test_input.mlir` does NOT have the `hls.static` attribute, so it generates:
```cpp
int32_t acc_stateful_5700927378943735518 = {0};  // Global variable
```

To generate a static variable, the `hls.static` attribute needs to be present, but the correct syntax needs to be determined or the attribute needs to be added programmatically after parsing.

## Next Steps

1. Determine the correct MLIR syntax for `hls.static` attribute
2. Or add the attribute programmatically after MLIR parsing
3. Verify that the attribute is preserved through the MLIR transformation pipeline

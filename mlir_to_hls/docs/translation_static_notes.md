# Extending `emit-vivado-hls` for Static Globals

The stock Allo Vitis HLS translation path does not emit `static` storage when it
encounters mutable `memref.global` definitions. To model a persistent accumulator
we extend `mlir/lib/Translation/EmitVivadoHLS.cpp` so that globals annotated with
the `hls.static` attribute are emitted as `static` variables in the generated
C++.

## Patch Summary

- In `ModuleEmitter::emitGlobal` we now print `static` ahead of the existing
  `const` qualifier check when the `memref.global` carries the unit attribute
  `hls.static`.
- All other behaviour remains unchanged, which keeps the existing code path for
  constant tables and arrays intact.

```diff
--- a/mlir/lib/Translation/EmitVivadoHLS.cpp
+++ b/mlir/lib/Translation/EmitVivadoHLS.cpp
@@
     indent();
     auto arrayType = op.getType().cast<ShapedType>();
     auto type = arrayType.getElementType();
+    if (op->hasAttr("hls.static")) {
+      os << "static ";
+    }
     if (op->hasAttr("constant")) {
       os << "const ";
     }
```

## Rebuilding the Translator

```bash
cd /home/sk3463/allo/mlir/build
ninja mlir-translate
```

After rebuilding, confirm the flag is active by running the verification script
from the playground root (added in a later step):

```bash
cd /home/sk3463/allo/sunwoo_playground/mlir_to_hls
./scripts/verify.py --run-mlir
```





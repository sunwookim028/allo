# MLIR → HLS Translation Walkthrough

## Your Test Program

```mlir
module {
  memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<0>
  func.func @test(%arg0: i32) -> i32 attributes {itypes = "s", otypes = "s"} {
    %0 = memref.get_global @acc_stateful_5700927378943735518 : memref<i32>
    %1 = affine.load %0[] {from = "acc"} : memref<i32>
    %2 = arith.addi %1, %arg0 : i32
    affine.store %2, %0[] {to = "acc"} : memref<i32>
    %3 = affine.load %0[] {from = "acc"} : memref<i32>
    return %3 : i32
  }
}
```

## Code Path Highlights

### 1. Module Entry Point
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:2389-2454`

```cpp
void ModuleEmitter::emitModule(ModuleOp module) {
  // Line 2445: Emit headers
  os << device_header;
  
  // Line 2446-2453: Process operations in order
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<func::FuncOp>(op))
      emitFunction(func);  // ← Processes @test function
    else if (auto cst = dyn_cast<memref::GlobalOp>(op))
      emitGlobal(cst);     // ← Processes @acc_stateful_5700927378943735518
  }
}
```

### 2. Global Variable Emission
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:1253-1316`

**Key Lines:**
- **Line 1263-1265:** Checks for `hls.static` attribute ⚠️ **NOT EXECUTED** (your MLIR doesn't have it)
  ```cpp
  if (op->hasAttr("hls.static")) {
    os << "static ";  // ← This would add 'static' keyword
  }
  ```
- **Line 1269:** Emits type name
  ```cpp
  os << getTypeName(type);  // → "int32_t"
  ```
- **Line 1270:** Emits global name
  ```cpp
  os << " " << op.getSymName();  // → " acc_stateful_5700927378943735518"
  ```
- **Line 1273-1315:** Emits initializer
  ```cpp
  os << " = {0};";  // For dense<0> with i32 element
  ```

**Result:** `int32_t acc_stateful_5700927378943735518 = {0};`

### 3. Function Signature Emission
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:2232-2368`

**Key Lines:**
- **Line 2247:** Function name
  ```cpp
  os << "void " << func.getName();  // → "void test("
  ```
- **Line 2274-2303:** Process input arguments
  - `itypes = "s"` → signed integer
  - Scalar `i32` → `int32_t arg0` (no pointer)
  ```cpp
  emitValue(arg);  // Line 2295: → "int32_t arg0"
  ```
- **Line 2315-2343:** Process return value
  - Return type `i32` → converted to pointer output parameter
  ```cpp
  emitValue(result, /*rank=*/0, /*isPtr=*/true);  // Line 2335: → "int32_t *result"
  ```
- **Line 2359:** Emit function body
  ```cpp
  emitBlock(func.front());  // ← Processes function body operations
  ```

### 4. Function Body Processing
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:2019-2027`

```cpp
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))  // ← Tries expression ops first
      continue;
    if (StmtVisitor(*this).dispatchVisitor(&op))  // ← Then statement ops
      continue;
    emitError(&op, "can't be correctly emitted.");
  }
}
```

**Visitor dispatch:**
- **Line 309:** `AffineLoadOp` → `StmtVisitor` → `emitAffineLoad()`
- **Line 310:** `AffineStoreOp` → `StmtVisitor` → `emitAffineStore()`
- **Line 364:** `arith::AddIOp` → `ExprVisitor` → `emitBinary(op, "+")`
- **Line 322:** `memref::GetGlobalOp` → `StmtVisitor` → `emitGetGlobal()`

### 5. GetGlobal Operation
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:1234-1241`

```cpp
void ModuleEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  indent();
  os << "// placeholder for const ";  // Line 1236
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, 0, false /*isPtr*/, op.getName().str());  // Line 1239
  emitInfoAndNewLine(op);
}
```

**Result:** `// placeholder for const int32_t acc_stateful_5700927378943735518;`

### 6. AffineLoad Operation
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:916-962`

**Key Lines:**
- **Line 918-921:** Extract `from` attribute for name preservation
  ```cpp
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
    // → "acc"
  }
  ```
- **Line 922-924:** Emit result variable
  ```cpp
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);  // → "int32_t %1"
  os << " = ";
  ```
- **Line 926-927:** Emit memref access with preserved name
  ```cpp
  emitValue(memref, 0, false, load_from_name);  // → "acc"
  ```
- **Line 954-959:** Emit array indices
  ```cpp
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);  // Empty for []
    os << "]";
  }
  ```

**Result:** `int32_t %1 = acc[];`

### 7. Addition Operation
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:1393-1406`

```cpp
void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));  // Line 1394: rank = 0 for scalar
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);  // Line 1398: → "int32_t %2"
  os << " = ";
  emitValue(op->getOperand(0), rank);  // Line 1400: → "%1"
  os << " " << syntax << " ";         // → " + "
  emitValue(op->getOperand(1), rank);  // Line 1402: → "%arg0"
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}
```

**Result:** `int32_t %2 = %1 + %arg0;`

### 8. AffineStore Operation
**File:** `mlir/lib/Translation/EmitVivadoHLS.cpp:964-1010`

**Key Lines:**
- **Line 966-969:** Extract `to` attribute
  ```cpp
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
    // → "acc"
  }
  ```
- **Line 970-971:** Emit memref with preserved name
  ```cpp
  emitValue(memref, 0, false, store_to_name);  // → "acc"
  ```
- **Line 1001-1005:** Emit indices
  ```cpp
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);  // Empty for []
    os << "]";
  }
  ```
- **Line 1006-1007:** Emit value assignment
  ```cpp
  os << " = ";
  emitValue(op.getValueToStore());  // → "%2"
  ```

**Result:** `acc[] = %2;`

## Summary of Translation Flow

1. **emitModule()** (line 2389) → Processes module body
2. **emitGlobal()** (line 1253) → `int32_t acc_stateful_5700927378943735518 = {0};`
3. **emitFunction()** (line 2232) → `void test(int32_t arg0, int32_t *result) {`
4. **emitBlock()** (line 2019) → Iterates through operations
5. **emitGetGlobal()** (line 1234) → Maps global symbol
6. **emitAffineLoad()** (line 916) → `int32_t %1 = acc[];`
7. **emitBinary()** (line 1393) → `int32_t %2 = %1 + %arg0;`
8. **emitAffineStore()** (line 964) → `acc[] = %2;`
9. **emitAffineLoad()** (line 916) → `int32_t %3 = acc[];` (second load)
10. Return handled during function signature emission

## Important Note on Static Behavior

⚠️ **Your MLIR program does NOT include the `hls.static` attribute**, so the global variable will be:
- A **file-scope global** (not `static`)
- **Shared across all invocations** if multiple instances exist
- **Initialized once** at program startup

To get `static` storage class (persistent state across kernel invocations), modify the MLIR:

```mlir
memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<0> { hls.static }
```

This would trigger **line 1263-1265** in `emitGlobal()`, emitting: `static int32_t acc_stateful_5700927378943735518 = {0};`




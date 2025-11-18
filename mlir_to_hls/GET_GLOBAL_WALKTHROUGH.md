# `memref.get_global` Translation Walkthrough

## Your MLIR with `get_global`

```mlir
%0 = memref.get_global @acc_stateful_5700927378943735518 : memref<i32>
%1 = affine.load %0[] {from = "acc"} : memref<i32>
```

## Code Flow for `get_global`

### Step 1: `get_global` Operation Processing
**File:** `EmitVivadoHLS.cpp:1234-1241`

```cpp
void ModuleEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  indent();
  os << "// placeholder for const ";  // Line 1236: Comment
  Value result = op.getResult();       // Line 1237: %0 is the result
  fixUnsignedType(result, op->hasAttr("unsigned"));
  // Line 1239: KEY LINE - Maps %0 to global name
  emitValue(result, 0, false /*isPtr*/, op.getName().str());
  emitInfoAndNewLine(op);
}
```

**What happens at line 1239:**
- `result` = `%0` (the memref value returned by get_global)
- `op.getName().str()` = `"acc_stateful_5700927378943735518"` (the global symbol name)
- Calls `emitValue(%0, 0, false, "acc_stateful_5700927378943735518")`

### Step 2: `emitValue` with Name Parameter
**File:** `EmitVivadoHLS.cpp:1883-1905`

```cpp
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  // Line 1888-1892: Check if already declared
  if (isDeclared(val)) {
    os << getName(val);  // ← Use existing name from table
    return;
  }

  os << getTypeName(val) << " ";  // Line 1895: "memref<i32>" → needs handling

  if (name == "") {
    // Line 1899: Generate auto name like "v0"
    os << addName(val, isPtr);
  } else {
    // Line 1903: KEY LINE - Use provided name
    os << addName(val, isPtr, name);
  }
}
```

**At line 1903:** `addName(%0, false, "acc_stateful_5700927378943735518")` is called

### Step 3: `addName` Stores the Mapping
**File:** `mlir/lib/Translation/Utils.cpp:13-35`

```cpp
SmallString<8> AlloEmitterBase::addName(Value val, bool isPtr,
                                        std::string name) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  // Line 21-28: Handle provided name
  if (name != "") {
    if (state.nameConflictCnt.count(name) > 0) {
      state.nameConflictCnt[name]++;
      valName += StringRef(name + std::to_string(state.nameConflictCnt[name]));
    } else { // first time
      state.nameConflictCnt[name] = 0;  // Line 26: Mark name as used
      valName += name;                   // Line 27: Use the global name
    }
  } else {
    valName += StringRef("v" + std::to_string(state.nameTable.size()));
  }
  // Line 32: CRITICAL - Store %0 → "acc_stateful_5700927378943735518" mapping
  state.nameTable[val] = valName;

  return valName;
}
```

**Result:** `%0` is now mapped to `"acc_stateful_5700927378943735518"` in the name table.

**Output at this point:**
```cpp
// placeholder for const int32_t acc_stateful_5700927378943735518;
```

### Step 4: Using `%0` in `affine.load`
**File:** `EmitVivadoHLS.cpp:916-962`

```cpp
void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  // Line 918-921: Extract "from" attribute if present
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
    // → load_from_name = "acc"
  }
  Value result = op.getResult();  // %1 is the load result
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);  // Line 924: Declare %1 variable
  os << " = ";
  auto memref = op.getMemRef();  // Line 926: memref = %0
  
  // Line 927: KEY LINE - Emit %0 with optional name override
  emitValue(memref, 0, false, load_from_name);
  // ↑ If load_from_name="acc", this OVERRIDES the stored name!
```

**Important:** At line 927, `emitValue(%0, 0, false, "acc")` is called with `load_from_name="acc"`.

**But wait!** Since `%0` was already declared via `get_global`, when we call `emitValue(%0, ...)`, let's check what happens:

**Line 1888-1892 in emitValue:**
```cpp
if (isDeclared(val)) {  // %0 is already declared
  os << getName(val);   // Line 1889: Look up %0 in name table
  return;
}
```

**Line 1889 calls `getName(%0)` which looks up in the table:**
**File:** `Utils.cpp:37-67`

```cpp
SmallString<8> AlloEmitterBase::getName(Value val) {
  // ... constant handling ...
  // Line 66: Look up in name table
  return state.nameTable.lookup(val);
}
```

**However**, the `"from"` attribute handling at line 927 passes `"acc"` as a name, which would try to **redeclare** the value. But since `%0` is already declared, it should just use the stored name.

**Actually, looking more carefully at line 927:**
- If `%0` is already declared, `isDeclared(%0)` returns true
- Line 1889: `getName(%0)` → returns `"acc_stateful_5700927378943735518"` from the name table
- The `load_from_name` parameter is **ignored** because we return early at line 1892

**But wait** - there's a comment at line 952 that suggests the name might be used for comments:
```cpp
os << ".read(); // ";
emitValue(memref, 0, false, load_from_name); // comment  // Line 952
```

So the `from` attribute might be primarily for generating helpful comments, not overriding the actual variable name.

## Summary of Name Resolution Flow

1. **`get_global` at line 1239:**
   ```cpp
   emitValue(%0, 0, false, "acc_stateful_5700927378943735518")
   ```
   → Stores `%0` → `"acc_stateful_5700927378943735518"` in name table

2. **`affine.load %0[]` at line 927:**
   ```cpp
   emitValue(%0, 0, false, "acc")  // "acc" from "from" attribute
   ```
   → `isDeclared(%0)` is true → uses stored name `"acc_stateful_5700927378943735518"`

3. **Final output:**
   ```cpp
   int32_t %1 = acc_stateful_5700927378943735518[];
   ```

## Key Code Lines

| Operation | File | Line | What It Does |
|-----------|------|------|--------------|
| `memref.get_global` dispatch | `EmitVivadoHLS.cpp` | 322 | Routes to `emitGetGlobal()` |
| Map result to global name | `EmitVivadoHLS.cpp` | 1239 | `emitValue(result, 0, false, op.getName().str())` |
| Store name mapping | `Utils.cpp` | 32 | `state.nameTable[val] = valName` |
| Look up stored name | `Utils.cpp` | 66 | `state.nameTable.lookup(val)` |
| Use in `affine.load` | `EmitVivadoHLS.cpp` | 927 | `emitValue(memref, 0, false, load_from_name)` |

## Output

```cpp
// placeholder for const int32_t acc_stateful_5700927378943735518;	// L...
int32_t %1 = acc_stateful_5700927378943735518[];	// L...
```

The `get_global` ensures that `%0` resolves to the actual global variable name `acc_stateful_5700927378943735518` when used in subsequent load/store operations.




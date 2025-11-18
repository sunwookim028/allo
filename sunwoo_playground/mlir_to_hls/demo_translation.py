#!/usr/bin/env python3
"""
Demonstration of MLIR -> HLS translation for the user's test program.
Shows the code paths that are executed during translation.
"""

test_mlir_program = """
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
"""

print("=" * 80)
print("MLIR -> HLS TRANSLATION DEMONSTRATION")
print("=" * 80)
print("\nINPUT MLIR PROGRAM:")
print(test_mlir_program)

print("\n" + "=" * 80)
print("TRANSLATION FLOW WITH CODE REFERENCES")
print("=" * 80)

translation_steps = [
    {
        "step": "1. emitModule() - Top-level entry",
        "location": "EmitVivadoHLS.cpp:2389",
        "action": "Processes module operations in order",
        "code": """
void ModuleEmitter::emitModule(ModuleOp module) {
  // ... header includes ...
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<func::FuncOp>(op))
      emitFunction(func);
    else if (auto cst = dyn_cast<memref::GlobalOp>(op))
      emitGlobal(cst);  // ← Called for @acc_stateful_5700927378943735518
  }
}"""
    },
    {
        "step": "2. emitGlobal() - Process memref.global",
        "location": "EmitVivadoHLS.cpp:1253-1316",
        "action": "Emits global variable declaration (NO 'static' since hls.static attr missing)",
        "code": """
void ModuleEmitter::emitGlobal(memref::GlobalOp op) {
  // Line 1263: Check for hls.static attribute
  if (op->hasAttr("hls.static")) {
    os << "static ";  // ← NOT executed (no hls.static in your MLIR)
  }
  // Line 1269: Emit type
  os << getTypeName(type);  // → "int32_t"
  // Line 1270: Emit name
  os << " " << op.getSymName();  // → " acc_stateful_5700927378943735518"
  // Line 1271-1272: Array dimensions (empty for scalar memref)
  // Line 1273-1314: Initializer
  os << " = {0};";  // → "int32_t acc_stateful_5700927378943735518 = {0};"
}"""
    },
    {
        "step": "3. emitFunction() - Process func.func",
        "location": "EmitVivadoHLS.cpp:2232-2368",
        "action": "Emits function signature and body",
        "code": """
void ModuleEmitter::emitFunction(func::FuncOp func) {
  // Line 2247: Function signature
  os << "void " << func.getName();  // → "void test("
  
  // Line 2274-2303: Process arguments
  // %arg0: i32 with itypes="s" → "int32_t arg0" (no pointer, scalar input)
  
  // Line 2315-2343: Process return value
  // Return type i32 → output as pointer parameter
  // Line 2335: emitValue(result, 0, true) → "int32_t *result" 
  
  // Line 2359: emitBlock() to process function body
  emitBlock(func.front());
}"""
    },
    {
        "step": "4. emitBlock() - Process function body operations",
        "location": "EmitVivadoHLS.cpp:2019-2027",
        "action": "Visits each operation in the block",
        "code": """
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))  // Try expression visitor
      continue;
    if (StmtVisitor(*this).dispatchVisitor(&op))  // Try statement visitor
      continue;
    emitError(&op, "can't be correctly emitted.");
  }
}"""
    },
    {
        "step": "5. memref.get_global → emitGetGlobal()",
        "location": "EmitVivadoHLS.cpp:1234-1241",
        "action": "Maps global symbol to variable name",
        "code": """
void ModuleEmitter::emitGetGlobal(memref::GetGlobalOp op) {
  // Line 1236: Comment
  os << "// placeholder for const ";
  // Line 1239: Emit variable declaration
  emitValue(result, 0, false, op.getName().str());
  // → "// placeholder for const int32_t acc_stateful_5700927378943735518;"
  // This creates a reference to the global variable
}"""
    },
    {
        "step": "6. affine.load → emitAffineLoad()",
        "location": "EmitVivadoHLS.cpp:916-962",
        "action": "Emits load from memref",
        "code": """
void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  // Line 918-921: Extract 'from' attribute if present
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
    // → load_from_name = "acc"
  }
  // Line 922-924: Emit result variable
  emitValue(result);  // → "int32_t %1"
  os << " = ";
  // Line 926-927: Emit memref access
  emitValue(memref, 0, false, load_from_name);  // → "acc"
  // Line 954-959: Emit indices
  for (auto index : affineMap.getResults()) {
    os << "[";  // Empty for [] index
    os << "]";
  }
  // → "int32_t %1 = acc[];"
}"""
    },
    {
        "step": "7. arith.addi → emitBinary()",
        "location": "EmitVivadoHLS.cpp:1393-1406",
        "action": "Emits addition operation",
        "code": """
void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  // Line 1394: Handle nested loops if result is array (not needed here)
  auto rank = emitNestedLoopHead(op->getResult(0));  // rank = 0 for scalar
  // Line 1398: Emit result
  emitValue(result, rank);  // → "int32_t %2"
  os << " = ";
  // Line 1400-1402: Emit operands with operator
  emitValue(op->getOperand(0), rank);  // → "%1"
  os << " " << syntax << " ";  // → " + "
  emitValue(op->getOperand(1), rank);  // → "%arg0"
  // → "int32_t %2 = %1 + %arg0;"
}"""
    },
    {
        "step": "8. affine.store → emitAffineStore()",
        "location": "EmitVivadoHLS.cpp:964-1010",
        "action": "Emits store to memref",
        "code": """
void ModuleEmitter::emitAffineStore(AffineStoreOp op) {
  // Line 966-969: Extract 'to' attribute
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
    // → store_to_name = "acc"
  }
  // Line 970-971: Emit memref
  emitValue(memref, 0, false, store_to_name);  // → "acc"
  // Line 1001-1005: Emit indices
  for (auto index : affineMap.getResults()) {
    os << "[";  // Empty for []
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());  // → "%2"
  // → "acc[] = %2;"
}"""
    },
    {
        "step": "9. func.return → ExprVisitor handles it",
        "location": "EmitVivadoHLS.cpp:432",
        "action": "Return operation (minimal processing)",
        "code": """
bool ExprVisitor::visitOp(func::ReturnOp op) { return true; }
// Return values handled during function signature emission"""
    }
]

for i, step in enumerate(translation_steps, 1):
    print(f"\n{step['step']}")
    print(f"Location: {step['location']}")
    print(f"Action: {step['action']}")
    print(f"\nRelevant Code:")
    print(step['code'])

print("\n" + "=" * 80)
print("EXPECTED OUTPUT (without static keyword)")
print("=" * 80)

expected_output = """//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

int32_t acc_stateful_5700927378943735518 = {0};	// L...

void test(
  int32_t arg0,
  int32_t *result
) {	// L...

  // placeholder for const int32_t acc_stateful_5700927378943735518;	// L...
  int32_t %1 = acc[];	// L...
  int32_t %2 = %1 + %arg0;	// L...
  acc[] = %2;	// L...
  int32_t %3 = acc[];	// L...
}
"""

print(expected_output)

print("\n" + "=" * 80)
print("NOTES:")
print("=" * 80)
print("""
1. The global variable is declared WITHOUT 'static' keyword because the
   MLIR program doesn't have the 'hls.static' attribute on memref.global.

2. To get static behavior, modify the MLIR to:
   memref.global "private" @acc_stateful_5700927378943735518 : memref<i32> = dense<0> { hls.static }

3. Variable names like %1, %2 are placeholder names - actual translation
   uses internal name table for proper C++ identifiers.

4. The 'from' and 'to' attributes on load/store are used to preserve
   original variable names in comments.
""")




/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITCATAPULTHLS_H
#define ALLO_TRANSLATION_EMITCATAPULTHLS_H

#include "allo/Translation/EmitVivadoHLS.h" // VhlsModuleEmitter base + AlloEmitterState
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
namespace allo {

// Map an MLIR type to its Catapult / Algorithmic-C C++ name (ac_int, ac_fixed,
// ac_ieee_float<binary32>, int32_t, ...). Shared with the SystemC emitter so
// both flows use Catapult-native types instead of Xilinx ap_int/ap_fixed.
llvm::SmallString<16> getCatapultTypeName(Type valType);

// Catapult ModuleEmitter: emits plain C++ / Algorithmic-C for Catapult HLS.
// Exposed in the header (rather than hidden in the .cpp) so downstream emitters
// can reuse the C++ compute codegen -- e.g. EmitCatapultHLS2 wraps these plain
// C++ functions in a thin SystemC/Connections shell.
class CatapultModuleEmitter : public hls::VhlsModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit CatapultModuleEmitter(AlloEmitterState &state)
      : hls::VhlsModuleEmitter(state) {}

  // Override methods that need Catapult-specific behavior.
  void emitModule(ModuleOp module) override;
  void emitFunctionDirectives(func::FuncOp func,
                              ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false,
                     std::string name = "") override;
  void emitLoopDirectives(Operation *op) override;
  void emitLoopDirectivesPreheader(Operation *op) override;
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitStreamTryGet(allo::StreamTryGetOp op) override;
  void emitStreamTryPut(allo::StreamTryPutOp op) override;
  void emitStreamEmpty(allo::StreamEmptyOp op) override;
  void emitStreamFull(allo::StreamFullOp op) override;
  void emitArrayDirectives(Value memref) override;
  void emitArrayDirectivesPreheader(Value memref) override;
  void emitFunction(func::FuncOp func) override;

protected:
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;
  // Catapult-specific type names (ac_int/ac_fixed/...).
  llvm::SmallString<16> getTypeName(Type valType) {
    return getCatapultTypeName(valType);
  }
  llvm::SmallString<16> getTypeName(Value val) {
    return getCatapultTypeName(val.getType());
  }
  // Stateful globals use ac_ieee_float<binary32> for f32 (nangate-45nm_beh
  // has no native float).
  void emitStatefulGlobalElementType(Type type) override {
    os << getCatapultTypeName(type);
  }
  // Float array elements get an 'f' suffix (ac_ieee_float<binary32> has no
  // double-literal constructor). Defined out-of-line to keep <cmath> out of
  // this header.
  void emitFloatArrayElement(float value) override;
};

LogicalResult emitCatapultHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitCatapultHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITCATAPULTHLS_H

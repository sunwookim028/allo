/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITBASEHLS_H
#define ALLO_TRANSLATION_EMITBASEHLS_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "allo/Dialect/AlloOps.h"

namespace mlir {
namespace allo {
namespace hls {

class ModuleEmitterBase : public AlloEmitterBase {
public:
  using operand_range = Operation::operand_range;

  explicit ModuleEmitterBase(AlloEmitterState &state) : AlloEmitterBase(state) {}
  virtual ~ModuleEmitterBase() = default;

  /// SCF statement emitters.
  virtual void emitScfFor(scf::ForOp op) {}
  virtual void emitScfIf(scf::IfOp op) {}
  virtual void emitScfWhile(scf::WhileOp op) {}
  virtual void emitScfCondition(scf::ConditionOp op) {}
  virtual void emitScfYield(scf::YieldOp op) {}

  /// Affine statement emitters.
  virtual void emitAffineFor(affine::AffineForOp op) {}
  virtual void emitAffineIf(affine::AffineIfOp op) {}
  virtual void emitAffineParallel(affine::AffineParallelOp op) {}
  virtual void emitAffineApply(affine::AffineApplyOp op) {}
  virtual void emitAffineLoad(affine::AffineLoadOp op) {}
  virtual void emitAffineStore(affine::AffineStoreOp op) {}
  virtual void emitAffineYield(affine::AffineYieldOp op) {}

  /// Memref-related statement emitters.
  virtual void emitLoad(memref::LoadOp op) {}
  virtual void emitStore(memref::StoreOp op) {}
  virtual void emitGetGlobal(memref::GetGlobalOp op) {}
  virtual void emitGetGlobalFixed(allo::GetGlobalFixedOp op) {}
  virtual void emitGlobal(memref::GlobalOp op) {}
  virtual void emitSubView(memref::SubViewOp op) {}
  // Whole-array copy (`dst[:] = src`). Every HLS backend lacked this, so any
  // slice assignment aborted with "'memref.copy' op is unsupported operation".
  virtual void emitCopy(memref::CopyOp op) {}
  virtual void emitReshape(memref::ReshapeOp op) {}

  /// Tensor-related statement emitters.
  virtual void emitTensorExtract(tensor::ExtractOp op) {}
  virtual void emitTensorInsert(tensor::InsertOp op) {}
  virtual void emitDim(memref::DimOp op) {}
  virtual void emitRank(memref::RankOp op) {}

  /// Standard operation emitters.
  virtual void emitBinary(Operation *op, const char *syntax);
  virtual void emitUnary(Operation *op, const char *syntax);
  virtual void emitPower(Operation *op);
  virtual void emitMaxMin(Operation *op, const char *syntax);

  /// Special operation emitters.
  virtual void emitCall(func::CallOp op) {}
  virtual void emitSelect(arith::SelectOp op) {}
  virtual void emitConstant(arith::ConstantOp op) {}
  virtual void emitGeneralCast(UnrealizedConversionCastOp op) {}
  // Hook called by emitCast after the operand: a backend may emit a narrowing
  // suffix (e.g. `.to_int64()`) when converting a >64-bit ac_int to a native
  // int/index, which under __SYNTHESIS__ has no implicit conversion. Default: none.
  virtual void emitNarrowCastSuffix(Value src, Value dst) {}
  virtual void emitGetBit(allo::GetIntBitOp op) {}
  virtual void emitSetBit(allo::SetIntBitOp op) {}
  virtual void emitGetSlice(allo::GetIntSliceOp op) {}
  virtual void emitSetSlice(allo::SetIntSliceOp op) {}
  virtual void emitBitReverse(allo::BitReverseOp op) {}
  virtual void emitBitcast(arith::BitcastOp op) {}

  /// Stream operation emitters.
  virtual void emitStreamConstruct(allo::StreamConstructOp op) {}
  virtual void emitStreamGet(allo::StreamGetOp op) {}
  virtual void emitStreamPut(allo::StreamPutOp op) {}
  virtual void emitStreamTryGet(allo::StreamTryGetOp op) {}
  virtual void emitStreamTryPut(allo::StreamTryPutOp op) {}
  virtual void emitStreamEmpty(allo::StreamEmptyOp op) {}
  virtual void emitStreamFull(allo::StreamFullOp op) {}

  // Wire operations (combinational link).
  virtual void emitWireConstruct(allo::WireConstructOp op) {}
  virtual void emitWireGet(allo::WireGetOp op) {}
  virtual void emitWirePut(allo::WirePutOp op) {}

  // Channel operations (handshake link).
  virtual void emitChannelConstruct(allo::ChannelConstructOp op) {}
  virtual void emitChannelGet(allo::ChannelGetOp op) {}
  virtual void emitChannelPut(allo::ChannelPutOp op) {}
  virtual void emitChannelTryGet(allo::ChannelTryGetOp op) {}
  virtual void emitChannelTryPut(allo::ChannelTryPutOp op) {}

  /// Top-level MLIR module emitter.
  virtual void emitModule(ModuleOp module) {}

protected:
  /// C++ component emitters.
  virtual void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                         std::string name = "") {}

  virtual void emitArrayDecl(Value array, bool isFunc = false,
                             std::string name = "") {}

  virtual void emitBufferDecl(Value array, bool isAccessor = false,
                              bool isReadOnly = false, std::string name = "") {}

  virtual unsigned emitNestedLoopHead(Value val) { return 0; }
  virtual void emitNestedLoopTail(unsigned rank);
  virtual void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  virtual void emitBlock(Block &block) {}
  virtual void emitLoopDirectives(Operation *op) {}
  // Loop directives that must be emitted BEFORE the loop header (default: none).
  // Catapult's #pragma hls_pipeline_init_interval / hls_unroll bind to the loop
  // that FOLLOWS them, unlike Vivado's in-body pragmas -- see the Catapult override.
  virtual void emitLoopDirectivesPreheader(Operation *op) {}
  virtual void emitArrayDirectives(Value memref) {}
  // Array directives that must be emitted BEFORE the declaration (default: none).
  // Same asymmetry as emitLoopDirectivesPreheader: Vivado's array pragmas follow the
  // declaration they name, but Catapult's #pragma hls_resource binds to the variable
  // DECLARED AFTER IT. Emitted after the declaration it is silently ineffective, and
  // Catapult then picks a 1R1W RAM for the array -- which serialises every read and,
  // because Genus treats the RAM as an unresolved black box, reports its area as ZERO.
  virtual void emitArrayDirectivesPreheader(Value memref) {}
  virtual void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList) {}

  virtual void emitFunction(func::FuncOp func) {}
  virtual void emitFunctionDeclaration(func::FuncOp func) {}

  virtual void emitFunction(func::FuncOp func, bool isAccessor) {
    (void)isAccessor;
    emitFunction(func);
  }

  virtual void emitHostFunction(func::FuncOp func) {}
};

} // namespace hls
} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITBASEHLS_H
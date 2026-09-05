/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_VIVADOHLSEMITTER_H
#define ALLO_TRANSLATION_VIVADOHLSEMITTER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "allo/IR/AlloOps.h"
#include "allo/Translation/EmitterState.h"

namespace mlir::allo {

struct VivadoHLSEmitter {
  explicit VivadoHLSEmitter(llvm::raw_ostream &os) : state(os) {}

  void emitFunction(func::FuncOp func);
  void emitCall(func::CallOp op);
  void emitReturn(func::ReturnOp op);

  void emitAffineFor(affine::AffineForOp op);
  void emitAffineLoad(affine::AffineLoadOp op);
  void emitAffineStore(affine::AffineStoreOp op);
  void emitAffineIf(affine::AffineIfOp op);
  void emitAffineYield(affine::AffineYieldOp op);
  void emitAffineApply(affine::AffineApplyOp op);

  void emitMemrefAlloc(memref::AllocOp op);
  void emitMemrefAlloca(memref::AllocaOp op);
  void emitMemrefLoad(memref::LoadOp op);
  void emitMemrefStore(memref::StoreOp op);
  void emitMemrefGlobal(memref::GlobalOp op);
  void emitMemrefGetGlobal(memref::GetGlobalOp op);
  void emitDenseInitializer(DenseElementsAttr dense, MemRefType type);

  void emitStreamCreate(allo::StreamCreateOp op);
  void emitStreamGet(allo::StreamGetOp op);
  void emitStreamPut(allo::StreamPutOp op);

  void emitBitGetSlice(allo::BitGetSliceOp op);
  void emitBitSetSlice(allo::BitSetSliceOp op);

  void emitAssumeNoDep(allo::AssumeNoDepOp op);

  void emitFor(scf::ForOp op);
  void emitIf(scf::IfOp op);
  void emitIndexSwitch(scf::IndexSwitchOp op);
  void emitWhile(scf::WhileOp op);
  void emitSCFYield(scf::YieldOp op);

  void emitSelect(arith::SelectOp op);
  void emitConstant(arith::ConstantOp op);
  void emitCmpI(arith::CmpIOp op);
  void emitCmpF(arith::CmpFOp op);

  void emitModule(ModuleOp);

  EmitterState state;

private:
  void emitBlock(Block &block);
  void emitValueDecl(Value val, bool isSigned = false);
  void emitValueRef(Value val);
  void emitFunctionArguments(func::FuncOp func);
  void emitFunctionReturnType(func::FuncOp func);
  void emitFunctionSignature(func::FuncOp func);
  void emitFunctionDirectives(func::FuncOp func);
  void emitTrailingLocation(Operation *op);
  void emitPartitionPragma(allo::PartitionAttr attr, llvm::StringRef varName);
  void emitBindStoragePragma(DictionaryAttr attr, llvm::StringRef varName);
  bool isTopFunc(func::FuncOp func);
  void emitLoopDirectives(Operation *op);
  void emitArraySuffix(ArrayRef<int64_t> shape, Location loc);
  void emitArraySuffix(ShapedType type, Location loc);
  void emitIndexedValue(Value value, ValueRange indices);
  void emitStreamTransferLoops(bool isPut, Value stream,
                               ValueRange streamIndices, ShapedType blockType,
                               ArrayRef<std::string> indices,
                               llvm::StringRef valueName);
  void emitYieldAssignments(Operation *parent, OperandRange operands);
  void emitAffineMapReduction(AffineMap map, OperandRange operands,
                              llvm::StringLiteral functionName);
  void emitBinaryOp(Operation *op, llvm::StringLiteral keyword);
  void emitBinaryOp(Operation *op, llvm::StringLiteral keyword, bool isSigned);
  void emitPrefixBinaryOp(Operation *op, llvm::StringLiteral keyword);
  void emitPrefixBinaryOp(Operation *op, llvm::StringLiteral keyword,
                          bool isSigned);
  void emitUnaryOp(Operation *op, llvm::StringLiteral keyword);
  void emitCastOp(Operation *op);
  void emitBitcastOp(arith::BitcastOp op);
  void emitIntExtOp(Operation *op, bool isSigned);
  void emitFPToIntOp(Operation *op, bool isSigned);
  void emitIntToFPOp(Operation *op, bool isSigned);
  void emitSignedOperand(Value value, bool isSigned);
  std::string getSymbolName(llvm::StringRef name);
  std::string getTemporaryName(llvm::StringRef prefix);
  std::string getTypeName(Type type, bool isSigned = false);

  void dispatch(Operation *op);

  llvm::StringMap<std::string> symbolNameTable;
  llvm::StringSet<> usedSymbolNames;
  unsigned temporaryNameCounter = 0;
};

struct AffineExprEmitter : public mlir::AffineExprVisitor<AffineExprEmitter> {
  explicit AffineExprEmitter(EmitterState &state, OperandRange operands,
                             unsigned numDims)
      : state(state), operands(operands), numDims(numDims) {}

  void visitDimExpr(AffineDimExpr expr) {
    state.os << state.getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    state.os << state.getName(operands[numDims + expr.getPosition()]);
  }
  void visitConstantExpr(AffineConstantExpr expr) {
    state.os << expr.getValue();
  }
  void visitAddExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    visitAffineBinExpr(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    state.os << "((";
    visit(expr.getLHS());
    state.os << " + ";
    visit(expr.getRHS());
    state.os << " - 1) / ";
    visit(expr.getRHS());
    state.os << ")";
  }
  void emitAffineMap(AffineMap map) {
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      if (i > 0)
        state.os << ", ";
      visit(map.getResult(i));
    }
  }

private:
  EmitterState &state;
  OperandRange operands;
  unsigned numDims;
  void visitAffineBinExpr(AffineBinaryOpExpr expr, llvm::StringLiteral op) {
    state.os << "(";
    visit(expr.getLHS());
    state.os << " " << op << " ";
    visit(expr.getRHS());
    state.os << ")";
  }
};

void registerVivadoHLSTranslation();

LogicalResult emitVivadoHLS(ModuleOp mod, llvm::raw_ostream &os,
                            bool enableApFloat, unsigned indexWidth,
                            bool withLocation, StringRef topName = "");

} // namespace mlir::allo

#endif // ALLO_TRANSLATION_VIVADOHLSEMITTER_H

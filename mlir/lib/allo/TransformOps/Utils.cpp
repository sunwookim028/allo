/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::allo;

static bool mapOperandDependsOnValue(mlir::Value operand, mlir::Value needle) {
  if (operand == needle)
    return true;

  auto arithDependsOnNeedle = [&](mlir::Operation *defOp) {
    if (!defOp || defOp->getNumRegions() != 0 || defOp->getNumResults() != 1)
      return false;
    if (auto *dialect = defOp->getDialect()) {
      if (dialect->getNamespace() != "arith")
        return false;
    } else {
      return false;
    }
    return llvm::any_of(defOp->getOperands(), [&](mlir::Value in) {
      return mapOperandDependsOnValue(in, needle);
    });
  };

  auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>();
  if (applyOp) {
    mlir::AffineMap map = applyOp.getAffineMap();
    for (mlir::AffineExpr resultExpr : map.getResults()) {
      if (mlir::allo::affineExprUsesValue(resultExpr, applyOp.getMapOperands(),
                                          map.getNumDims(), needle)) {
        return true;
      }
    }
    return false;
  }

  return arithDependsOnNeedle(operand.getDefiningOp());
}

namespace mlir::allo {
bool affineExprUsesValue(AffineExpr expr, ValueRange mapOperands,
                         unsigned numDims, Value needle) {
  bool used = false;
  expr.walk([&](AffineExpr inner) {
    if (used)
      return;
    if (auto dim = dyn_cast<AffineDimExpr>(inner)) {
      unsigned pos = dim.getPosition();
      if (pos < mapOperands.size() &&
          mapOperandDependsOnValue(mapOperands[pos], needle)) {
        used = true;
      }
      return;
    }
    auto sym = dyn_cast<AffineSymbolExpr>(inner);
    if (!sym)
      return;
    unsigned pos = numDims + sym.getPosition();
    if (pos < mapOperands.size() &&
        mapOperandDependsOnValue(mapOperands[pos], needle)) {
      used = true;
    }
  });
  return used;
}

int findMemRefAxisFromIVs(affine::AffineStoreOp storeOp, Value iv) {
  AffineMap map = storeOp.getAffineMap();
  auto operands = storeOp.getMapOperands();
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (affineExprUsesValue(map.getResult(i), operands, map.getNumDims(), iv))
      return static_cast<int>(i);
  }
  return -1;
}

bool isMemRefCastOrViewLike(Operation *op) {
  return isa<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
             memref::CastOp, memref::TransposeOp>(op);
}

// Follow view-like aliases and resolve to a root buffer value.
Value resolveMemRefValueRoot(Value value) {
  SmallPtrSet<Value, 8> visited;
  while (value && visited.insert(value).second) {
    if (isa<BlockArgument>(value))
      return value;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return value;

    if (isMemRefCastOrViewLike(defOp)) {
      value = defOp->getOperand(0);
      continue;
    }
    return value;
  }
  return value;
}

LogicalResult resolveBufferAttrCarrier(Value value, Operation *&owner,
                                       std::optional<unsigned> &argNumber) {
  owner = nullptr;
  argNumber = std::nullopt;
  if (!isa<MemRefType>(value.getType()))
    return failure();

  Value root = resolveMemRefValueRoot(value);
  if (!isa<MemRefType>(root.getType()))
    return failure();

  if (auto arg = dyn_cast<BlockArgument>(root)) {
    auto func = dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
    if (!func)
      return failure();
    owner = func;
    argNumber = arg.getArgNumber();
    return success();
  }

  Operation *defOp = root.getDefiningOp();
  if (!defOp)
    return failure();
  if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (!global)
      return failure();
    owner = global;
    return success();
  }
  if (!isa<memref::AllocOp, memref::AllocaOp>(defOp))
    return failure();
  owner = defOp;
  return success();
}

} // namespace mlir::allo

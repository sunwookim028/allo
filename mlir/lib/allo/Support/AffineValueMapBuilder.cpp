/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AffineValueMapBuilder.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::allo;

Value mlir::allo::stripCast(Value value) {
  while (true) {
    auto *defOp = value.getDefiningOp();
    if (!defOp)
      return value;
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp>(defOp)) {
      value = defOp->getOperand(0);
    } else {
      return value;
    }
  }
}

AffineExpr AffineValueMapBuilder::addDim(Value v) {
  auto *it = llvm::find(dims, v);
  if (it != dims.end()) {
    return getAffineDimExpr(std::distance(dims.begin(), it), ctx);
  }
  unsigned pos = dims.size();
  dims.push_back(v);
  return getAffineDimExpr(pos, ctx);
}

AffineExpr AffineValueMapBuilder::addSym(Value v) {
  auto *it = llvm::find(syms, v);
  if (it != syms.end()) {
    return getAffineSymbolExpr(std::distance(syms.begin(), it), ctx);
  }
  unsigned pos = syms.size();
  syms.push_back(v);
  return getAffineSymbolExpr(pos, ctx);
}

FailureOr<AffineExpr> AffineValueMapBuilder::importValueInternal(Value v) {
  v = stripCast(v);
  if (exprFailureCache.contains(v))
    return failure();
  IntegerAttr::ValueType cst;
  if (matchPattern(v, m_ConstantInt(&cst))) {
    return getAffineConstantExpr(cst.getSExtValue(), ctx);
  }
  // Allo worker-id / worker-count queries are invariant within a kernel and can
  // be treated as affine symbols (e.g. `fifo[i, j]` where `i = get_wid(0)`).
  if (isa_and_present<GetWorkerIdOp, GetNumWorkersOp>(v.getDefiningOp())) {
    return addSym(v);
  }
  if (affine::isValidDim(v)) {
    return addDim(v);
  }
  if (affine::isValidSymbol(v)) {
    return addSym(v);
  }
  auto *defOp = v.getDefiningOp();
  if (auto applyOp = dyn_cast_if_present<affine::AffineApplyOp>(defOp)) {
    if (failed(importMapAndOperands(
            applyOp.getAffineMap(), applyOp.getDimOperands(),
            applyOp.getSymbolOperands(), /*allowMultiResults=*/false)))
      return cacheFailure(v);
    return exprs.back();
  }
  if (auto addOp = dyn_cast_if_present<arith::AddIOp>(defOp)) {
    auto lhs = importValueInternal(addOp.getLhs());
    auto rhs = importValueInternal(addOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    return *lhs + *rhs;
  }
  if (auto subOp = dyn_cast_if_present<arith::SubIOp>(defOp)) {
    auto lhs = importValueInternal(subOp.getLhs());
    auto rhs = importValueInternal(subOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    return *lhs - *rhs;
  }
  if (auto mulOp = dyn_cast_if_present<arith::MulIOp>(defOp)) {
    auto lhs = importValueInternal(mulOp.getLhs());
    auto rhs = importValueInternal(mulOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // Multiplication is affine only when one side is a symbol: d0 * d1 is not
    // affine, but s0 * d0 and s0 * s1 are.
    auto result = *lhs * *rhs;
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  if (isa_and_present<arith::DivSIOp, arith::DivUIOp>(defOp)) {
    auto lhs = importValueInternal(defOp->getOperand(0));
    auto rhs = importValueInternal(defOp->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // Division is affine only when the divisor is a symbol or a constant.
    auto result = lhs->floorDiv(*rhs);
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  if (isa_and_present<arith::RemSIOp, arith::RemUIOp>(defOp)) {
    auto lhs = importValueInternal(defOp->getOperand(0));
    auto rhs = importValueInternal(defOp->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // Remainder is affine only when the divisor is a symbol or a constant.
    auto result = *lhs % *rhs;
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  // AffineExpr can only be +, -, *, //, %, while max/min is not an AffineExpr.
  return cacheFailure(v);
}

affine::AffineValueMap AffineValueMapBuilder::compose() const {
  SmallVector<Value, 4> operands(dims);
  llvm::append_range(operands, syms);
  auto map = AffineMap::get(dims.size(), syms.size(), exprs, ctx);
  affine::AffineValueMap vMap(map, operands, results);
  vMap.composeSimplifyAndCanonicalize();
  return vMap;
}

void AffineValueMapBuilder::reset() {
  dims.clear();
  syms.clear();
  results.clear(); // exprFailureCache is deliberately kept
  exprs.clear();
}

LogicalResult AffineValueMapBuilder::importMapAndOperands(
    AffineMap map, ValueRange dims, ValueRange syms, bool allowMultiResults) {
  if (map.getNumResults() > 1 && !allowMultiResults)
    return failure();
  SmallVector<AffineExpr, 4> dimExprs;
  SmallVector<AffineExpr, 4> symExprs;

  for (auto dim : dims) {
    auto dimExpr = importValueInternal(dim);
    if (failed(dimExpr))
      return failure();
    dimExprs.push_back(*dimExpr);
  }
  for (auto sym : syms) {
    auto symExpr = importValueInternal(sym);
    if (failed(symExpr))
      return failure();
    symExprs.push_back(*symExpr);
  }
  for (auto result : map.getResults()) {
    // One result is one candidate bound expression in the final min/max set.
    exprs.push_back(result.replaceDimsAndSymbols(dimExprs, symExprs));
  }
  return success();
}

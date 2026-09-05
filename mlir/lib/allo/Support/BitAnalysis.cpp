/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/BitAnalysis.h"

#include "allo/IR/AlloOps.h" // dcp::DCPathComputeOp, CombOpKindEnum

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

/// What an operation does to bits, which is all this file asks of one. Not the
/// operator library's rows: those name the device row an op is priced under,
/// where a signed and an unsigned shift share one row yet move bits
/// differently.
enum class BitOp {
  Unknown,
  ZExt,
  SExt,
  Trunc,
  And,
  Or,
  Xor,
  Shl,
  LShr,
  AShr,
  Select
};

/// Classifies \p op on whichever side of reification it sits: a reified
/// compute names its realization outright, an `arith` one is matched on its
/// type. An operator neither arm recognizes reads as `Unknown`, which costs a
/// conclusion and never correctness.
BitOp bitOpOf(Operation *op) {
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op)) {
    std::optional<CombOpKindEnum> kind = comp.getCombKind();
    if (!kind)
      return BitOp::Unknown;
    switch (*kind) {
    case CombOpKindEnum::Extui:
      return BitOp::ZExt;
    case CombOpKindEnum::Extsi:
      return BitOp::SExt;
    case CombOpKindEnum::Trunci:
      return BitOp::Trunc;
    case CombOpKindEnum::Andi:
      return BitOp::And;
    case CombOpKindEnum::Ori:
      return BitOp::Or;
    case CombOpKindEnum::Xori:
      return BitOp::Xor;
    case CombOpKindEnum::Shli:
      return BitOp::Shl;
    case CombOpKindEnum::Shrui:
      return BitOp::LShr;
    case CombOpKindEnum::Shrsi:
      return BitOp::AShr;
    case CombOpKindEnum::Select:
      return BitOp::Select;
    default:
      return BitOp::Unknown;
    }
  }
  return llvm::TypeSwitch<Operation *, BitOp>(op)
      .Case<arith::ExtUIOp>([](auto) { return BitOp::ZExt; })
      .Case<arith::ExtSIOp>([](auto) { return BitOp::SExt; })
      .Case<arith::TruncIOp>([](auto) { return BitOp::Trunc; })
      .Case<arith::AndIOp>([](auto) { return BitOp::And; })
      .Case<arith::OrIOp>([](auto) { return BitOp::Or; })
      .Case<arith::XOrIOp>([](auto) { return BitOp::Xor; })
      .Case<arith::ShLIOp>([](auto) { return BitOp::Shl; })
      .Case<arith::ShRUIOp>([](auto) { return BitOp::LShr; })
      .Case<arith::ShRSIOp>([](auto) { return BitOp::AShr; })
      .Case<arith::SelectOp>([](auto) { return BitOp::Select; })
      .Default([](auto) { return BitOp::Unknown; });
}

} // namespace

llvm::KnownBits mlir::allo::knownBits(Value v, unsigned depth) {
  auto ty = cast<IntegerType>(v.getType());
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return llvm::KnownBits::makeConstant(cst);
  Operation *op = v.getDefiningOp();
  llvm::KnownBits unknown(ty.getWidth());
  if (!op || !depth)
    return unknown;
  // Every operand reached below is integer-typed, so the recursion stays inside
  // this function's contract. An index cast is not among them: its operand has
  // no width to walk into.
  auto of = [&](unsigned k) { return knownBits(op->getOperand(k), depth - 1); };
  switch (bitOpOf(op)) {
  case BitOp::ZExt:
    return of(0).zext(ty.getWidth());
  case BitOp::SExt:
    return of(0).sext(ty.getWidth());
  case BitOp::Trunc:
    return of(0).trunc(ty.getWidth());
  case BitOp::And:
    return of(0) & of(1);
  case BitOp::Or:
    return of(0) | of(1);
  case BitOp::Xor:
    return of(0) ^ of(1);
  case BitOp::Shl:
    return llvm::KnownBits::shl(of(0), of(1));
  case BitOp::LShr:
    return llvm::KnownBits::lshr(of(0), of(1));
  case BitOp::AShr:
    return llvm::KnownBits::ashr(of(0), of(1));
  // Either arm may run, so only what both agree on is known.
  case BitOp::Select:
    return of(1).intersectWith(of(2));
  case BitOp::Unknown:
    return unknown;
  }
  llvm_unreachable("unhandled BitOp");
}

bool mlir::allo::isBitRename(Operation *op) {
  switch (bitOpOf(op)) {
  case BitOp::Shl:
  case BitOp::LShr:
  case BitOp::AShr: {
    APInt amount;
    return matchPattern(op->getOperand(1), m_ConstantInt(&amount));
  }
  case BitOp::Or:
  case BitOp::Xor: {
    // An `index` operand has no width to reason in.
    if (!isa<IntegerType>(op->getResult(0).getType()))
      return false;
    // Sharing no set bit, the two sides concatenate: `or` and `xor` agree bit
    // for bit wherever one of them is zero.
    return (knownBits(op->getOperand(0)).Zero |
            knownBits(op->getOperand(1)).Zero)
        .isAllOnes();
  }
  default:
    return false;
  }
}

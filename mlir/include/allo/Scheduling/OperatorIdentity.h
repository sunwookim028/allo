/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORIDENTITY_H
#define ALLO_SCHEDULING_OPERATORIDENTITY_H

#include "allo/IR/AlloOps.h" // dcp::DCPathComputeOp

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::allo {

class OperatorLibrary;

/// What one physical operator is: two operations share an identity exactly when
/// one unit can run both. The library's second key, finer than
/// `NodeTiming::typeName`, which names a timing row.
struct OperatorIdentity {
  /// The realization, on exactly one of the two exclusive paths a `dcp.compute`
  /// takes. Both empty when no functional unit is built (a memory or stream
  /// access, a literal, a call).
  std::optional<CombOpKindEnum> comb; // native: emitted inline by `emitCompute`
  std::string ipSymbol;               // IP: a `dcp.operator` symbol
  llvm::SmallVector<Type, 2> argTypes; // operand types, so width is in here
  Type resultType;
  Attribute predicate; // a compare's `predicate`; null otherwise
  Attribute map;       // an `affine.apply`'s `map`; null otherwise
  /// A bit rename (a shift by a literal, a width-kept resize): wiring, priced
  /// at nothing. In the identity so a rename never shares a class with the real
  /// operator its mnemonic spells.
  bool rename = false;

  /// Whether an operation of this identity gets a functional unit.
  bool realized() const { return comb || !ipSymbol.empty(); }

  bool operator==(const OperatorIdentity &o) const {
    return comb == o.comb && ipSymbol == o.ipSymbol &&
           llvm::ArrayRef<Type>(argTypes) == llvm::ArrayRef<Type>(o.argTypes) &&
           resultType == o.resultType && predicate == o.predicate &&
           map == o.map && rename == o.rename;
  }
  bool operator!=(const OperatorIdentity &o) const { return !(*this == o); }

  /// How the realization spells, whichever path it took: the comb mnemonic or
  /// the IP symbol. A display name for a report or debug dump; `Naming.h` owns
  /// the RTL ones (`operatorModuleName`).
  llvm::StringRef realizationName() const {
    return comb ? stringifyCombOpKindEnum(*comb) : llvm::StringRef(ipSymbol);
  }

  /// A stable string spelling of the whole identity, for map keys and reports.
  std::string key() const;
};

/// The identity of a reified compute op, which carries its own realization.
OperatorIdentity operatorIdentity(dcp::DCPathComputeOp comp);

/// The identity \p lib resolves for \p op; empty when \p op has no realization.
/// Dispatches to the overload above for an already-reified op.
OperatorIdentity operatorIdentity(Operation *op, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORIDENTITY_H

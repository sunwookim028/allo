/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_BITANALYSIS_H
#define ALLO_SUPPORT_BITANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/KnownBits.h"

namespace mlir::allo {

/// The bits of \p v a forward walk proves constant. A value it cannot follow,
/// and anything past \p depth, reads as all-unknown, which is the safe answer:
/// an unknown bit forfeits a conclusion, never reaches a wrong one.
///
/// \p v must be integer-typed; an `index` has no width to reason in.
llvm::KnownBits knownBits(Value v, unsigned depth = 8);

/// Whether \p op renames bits rather than computing them, so it costs no
/// logic. Two shapes qualify:
///
///   * a shift by a literal amount, which `comb` canonicalizes into an extract
///     or concat. The device's shift row prices a barrel shifter, the delay a
///     runtime amount pays;
///   * an `or` / `xor` whose operands share no set bit, which concatenates
///     rather than combines: every result bit takes one side while the other
///     contributes a constant zero.
///
/// The bit-level half of `isZeroDelay`, which is what the two pricing sites
/// ask.
bool isBitRename(Operation *op);

} // namespace mlir::allo

#endif // ALLO_SUPPORT_BITANALYSIS_H

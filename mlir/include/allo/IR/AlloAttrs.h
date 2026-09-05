/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ATTRS_H
#define ALLO_ATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "allo/IR/AlloEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.h.inc"

namespace mlir::allo {

/// What the resource vector \p uses spends at \p params: ONE entry per named
/// resource, the sum over that resource's terms of the product of the term's
/// factors with factor `i` evaluated at `params[i]`, rounded to the nearest
/// whole resource once the sum is complete. Resources come out in the order
/// they first appear, and a caller never has to add two entries up itself.
///
/// \p uses is a `dcp.resource`-referencing `ResourceUseAttr` array, and
/// \p params is the parameter tuple of the realization's kind (an operator's
/// operand width; a multiplexer's fan-in and width; a chain's or a storage's
/// depth and width). A null \p uses spends nothing, which is what an undeclared
/// cost means.
///
/// A lone `tiled` factor is the exception to the one-factor-per-parameter rule:
/// it reads the whole tuple, since `ceil(depth*width/bits)` does not separate.
///
/// Nullopt where a factor is not measured at its parameter; `unmeasuredUse`
/// names the offending cost.
std::optional<llvm::SmallVector<std::pair<mlir::SymbolRefAttr, int64_t>>>
evaluateResourceUse(mlir::ArrayAttr uses, llvm::ArrayRef<int64_t> params);

/// The cost inside \p uses whose measured points do not cover its parameter,
/// null wherever `evaluateResourceUse` answers.
CostAttr unmeasuredUse(mlir::ArrayAttr uses, llvm::ArrayRef<int64_t> params);

} // namespace mlir::allo

#endif // ALLO_ATTRS_H

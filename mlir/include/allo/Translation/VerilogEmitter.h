/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_VERILOGEMITTER_H
#define ALLO_TRANSLATION_VERILOGEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::allo {
LogicalResult emitVerilog(ModuleOp mod, llvm::raw_ostream &os);

LogicalResult emitSplitVerilog(ModuleOp mod, StringRef directory);
} // namespace mlir::allo

#endif // ALLO_TRANSLATION_VERILOGEMITTER_H

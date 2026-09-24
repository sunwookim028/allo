/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Minimal SystemC (sc_fifo) backend — Track B / X1.
 */

#ifndef ALLO_TRANSLATION_EMITSYSTEMC_H
#define ALLO_TRANSLATION_EMITSYSTEMC_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace allo {

LogicalResult emitSystemC(ModuleOp module, llvm::raw_ostream &os);
void registerEmitSystemCTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITSYSTEMC_H

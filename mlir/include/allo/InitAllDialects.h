/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_INIT_ALL_DIALECTS_H
#define ALLO_INIT_ALL_DIALECTS_H

namespace mlir {
class DialectRegistry;
namespace allo {
void registerAllDialects(DialectRegistry &registry);
} // namespace allo
} // namespace mlir

#endif // ALLO_INIT_ALL_DIALECTS_H

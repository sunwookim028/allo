/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_INIT_ALL_EXTENSIONS_H
#define ALLO_INIT_ALL_EXTENSIONS_H

namespace mlir {
class DialectRegistry;
namespace allo {
void registerAllExtensions(DialectRegistry &registry);
} // namespace allo
} // namespace mlir

#endif // ALLO_INIT_ALL_EXTENSIONS_H

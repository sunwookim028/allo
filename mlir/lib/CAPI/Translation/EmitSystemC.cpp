/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/EmitSystemC.h"
#include "allo-c/Translation/EmitSystemC.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace allo;

MlirLogicalResult mlirEmitSystemC(MlirModule module,
                                  MlirStringCallback callback, 
                                  void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitSystemC(unwrap(module), stream));
}

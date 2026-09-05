/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Registration here is strictly ADDITIVE on top of the upstream
 * RegisterEverything bundled in the same package: only the `allo` dialect, the
 * Allo transform-dialect extension, and Allo-specific passes. The two share one
 * set of MLIR global registries, so re-registering anything upstream aborts.
 */

#include "allo-c/Registration.h"

#include "allo/IR/AlloOps.h"
#include "allo/InitAllDialects.h"
#include "allo/InitAllExtensions.h"
#include "allo/InitAllPasses.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include <mutex>

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Allo, allo, ::mlir::allo::AlloDialect)

void alloMlirRegisterAllDialects(MlirContext context) {
  DialectRegistry registry;
  allo::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void alloMlirRegisterAllExtensions(MlirContext context) {
  DialectRegistry registry;
  allo::registerAllExtensions(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void alloMlirRegisterAllPasses() {
  static std::once_flag once;
  std::call_once(once, [] { allo::registerAllPasses(); });
}

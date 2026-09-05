/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_EMITDRIVER_H
#define ALLO_MICROARCH_EMITDRIVER_H

#include "allo/Microarch/Datapath.h"

#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::allo::iface {
struct ModuleInterface;
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

struct MicroarchReport;

/// Lower the scheduled `func.func`s reachable from \p top to structural
/// `hw.module`s, erasing the source funcs, and map each emitted module's name
/// to its port-interface JSON (the cosim manifest) in \p interfaces. This is
/// the free function behind the `allo-datapath-to-hw` pass.
/// Emission runs bottom-up over the call DAG, callees before callers.
/// \p cycleTime is the target period in ns and must be the one the scheduler
/// took: `validateDatapath` holds the result to the same clock.
/// \p report collects what each module's allocation DECIDED, in emit order.
LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top, float cycleTime,
                               llvm::StringMap<std::string> &interfaces,
                               MicroarchReport &report);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_EMITDRIVER_H

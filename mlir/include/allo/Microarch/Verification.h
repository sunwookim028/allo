/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_VERIFY_H
#define ALLO_MICROARCH_VERIFY_H

#include "allo/Microarch/Datapath.h"
#include "allo/Microarch/Report.h"           // TimingPath
#include "allo/Scheduling/OperatorLibrary.h" // prices muxes and units

#include "mlir/Support/LLVM.h"

#include <vector>

namespace mlir::allo::uarch {

/// Everything checked between the model being sealed and hardware being built,
/// cut by who is at fault: the design, this backend, or an upstream pass. The
/// one call the emit driver makes. Returns this module's few worst
/// combinational paths, longest first, which the report publishes and the QoR
/// turns into a clock. Structures the operator library does not price are
/// absent from them, so they are estimates and no substitute for place and
/// route. Never empty: a module with no datapath still holds one register hop.
FailureOr<std::vector<TimingPath>>
validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp, float cycleTime,
                 const OperatorLibrary &lib, bool plannedBinding);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H

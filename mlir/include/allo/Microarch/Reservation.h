/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_RESERVATION_H
#define ALLO_MICROARCH_RESERVATION_H

#include "allo/Microarch/Datapath.h"

namespace mlir::allo::uarch {

/// The resource cycles one bound op occupies on its functional unit, within its
/// region's schedule. A pipelined unit contends only its issue slot; a
/// non-pipelined unit is busy for its whole latency. Cyclic regions count
/// residues mod II, acyclic regions absolute cycles.
struct Reservation {
  RegionId region = 0;
  llvm::SmallVector<unsigned, 4> cycles; // occupied resource cycles
};

/// The reservation of an op bound to \p unit at issue \p residue in \p region.
/// \p residue is the value the binder already stored in FuncUnit::boundOps
/// (start mod II for cyclic, absolute start for acyclic).
Reservation reservationOf(const RegionBlock &region, const FuncUnit &unit,
                          unsigned residue);

/// Whether two reservations may coexist on one shared unit: their occupied
/// cycles must not intersect. Both must belong to the same region, a unit being
/// owned by exactly one. With equal `OperatorIdentity` this is the full
/// legality test; the timing side of a fold (the mux it grows must fit
/// `unitSlack`) lives with the policy.
bool reservationsDisjoint(const Reservation &a, const Reservation &b);

/// Assert the binding is legal: no two ops bound to the same unit contend for
/// it in the same resource cycle. Vacuous under the trivial binding.
void verifyBinding(const Datapath &dp);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_RESERVATION_H

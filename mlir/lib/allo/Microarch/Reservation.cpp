/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Reservation.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cassert>

namespace mlir::allo::uarch {

Reservation reservationOf(const RegionBlock &region, const FuncUnit &unit,
                          unsigned residue) {
  Reservation r;
  r.region = region.id;
  // A pipelined unit holds only the issue slot, its stages carrying distinct
  // data; a non-pipelined unit stays busy for its whole latency.
  unsigned len = unit.pipelined ? 1 : std::max(1u, unit.latency);
  // Cyclic regions wrap occupancy mod II, so a latency at or above II marks the
  // unit busy on every residue. Acyclic regions run on a straight timeline.
  unsigned mod =
      region.kind == RegionBlock::Kind::Cyclic ? region.ii.value_or(1) : 0;
  for (unsigned i = 0; i < len; ++i)
    r.cycles.push_back(mod ? (residue + i) % mod : residue + i);
  return r;
}

bool reservationsDisjoint(const Reservation &a, const Reservation &b) {
  assert(a.region == b.region && "cross-region sharing is not modelled");
  llvm::SmallDenseSet<unsigned, 8> cyclesA(a.cycles.begin(), a.cycles.end());
  return llvm::none_of(b.cycles,
                       [&](unsigned c) { return cyclesA.contains(c); });
}

// Checks every unit's bound ops for identity and reservation conflicts.
// Debug-only: the sweep costs O(units * boundOps^2) comparisons.
void verifyBinding([[maybe_unused]] const Datapath &dp) {
#ifndef NDEBUG
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      llvm::SmallVector<Reservation, 4> held;
      for (const FuncUnit::BoundOp &bo : u.boundOps) {
        // The emitter builds one operator from the unit's identity, so a bound
        // op of any other identity would be miscompiled.
        assert(operatorIdentity(cast<dcp::DCPathComputeOp>(bo.op)) ==
                   u.identity &&
               "shared unit binds an op of a different operator identity");
        held.push_back(reservationOf(rb, u, bo.residue));
      }
      for (unsigned i = 0, e = held.size(); i < e; ++i)
        for (unsigned j = i + 1; j < e; ++j)
          assert(reservationsDisjoint(held[i], held[j]) &&
                 "binding hazard: two ops share a unit in the same cycle");
    }
#endif
}

} // namespace mlir::allo::uarch

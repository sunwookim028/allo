/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_BINDINGPOLICY_H
#define ALLO_MICROARCH_BINDINGPOLICY_H

#include "allo/Microarch/Datapath.h"

#include <memory>
#include <vector>

namespace mlir::allo::uarch {

/// Timing data a policy needs beyond the model: the clock the schedule was cut
/// against, and the device rows that price a unit's inputs and a mux level.
struct BindingContext {
  float cycleTime;
  const OperatorLibrary &lib;
};

/// A resource-binding policy. `plan` returns unit groups to merge, each group
/// folding onto its first unit and every unit not named keeping its own; an
/// empty result is the trivial binding. A policy only decides, never mutating
/// `dp`.
struct BindingPolicy {
  virtual ~BindingPolicy() = default;
  virtual std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const = 0;
  /// Whether the fold this policy returns realizes the solve's own allocation
  /// rather than one decided here, so a unit the solve named must be built.
  virtual bool realizesSolvePlan() const { return false; }
};

/// Every op keeps its own unit: `plan` returns no groups.
struct TrivialBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
};

/// Exact within-region sharing: fold same-operator-type units whose MRT
/// reservations are disjoint, decided by one CP-SAT solve per region
/// (`solveSharing`) minimizing modelled area, with every input cone held to the
/// period under the same recursion the emit-side gate walks. A greedy first-fit
/// plan seeds the solve and stands in when its budget expires with nothing
/// usable.
struct ExactShareBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
};

/// Build the allocation the scheduler decided: fold together every unit whose
/// bound op names the same `dcp.compute` `unit` symbol. An op the scheduler
/// left unallocated names no symbol and keeps its own unit.
struct PlannedBinding : BindingPolicy {
  std::vector<llvm::SmallVector<UnitId, 2>>
  plan(const Datapath &dp, const BindingContext &ctx) const override;
  bool realizesSolvePlan() const override { return true; }
};

/// The policy named by a pass option ("trivial" / "exact-share" / "planned");
/// null on an unknown name.
std::unique_ptr<BindingPolicy> bindingPolicyFor(llvm::StringRef name);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_BINDINGPOLICY_H

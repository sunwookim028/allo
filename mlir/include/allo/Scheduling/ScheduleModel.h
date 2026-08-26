/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULE_MODEL_H
#define ALLO_SCHEDULING_SCHEDULE_MODEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlir::allo {

/// One op's place in its region's schedule.
struct OpSchedule {
  /// Start cycle, relative to the region's own start.
  int64_t start = 0;
  /// Sub-cycle start (ns within the cycle) from a chaining solve; empty without
  /// one.
  std::optional<float> startInCycle;
  /// Which allocated instance runs it: an index into
  /// `ScheduleModel::allocatedUnits`. Empty unless an exact solve allocated.
  std::optional<unsigned> unit;
  /// The `dcp.operator` row an exact solve selected, empty where the library's
  /// pick stands. The reify realizes the op on this row, the one the schedule
  /// priced.
  std::string selectedImpl;
};

/// The solved schedule of one region: what the solver decided, nothing the IR
/// already says. Trip count, lower bound and step are re-derived from the loop;
/// only a trip an assumption bounded is here, since reification keeps the
/// loop's runtime bound operand and drops the assumption.
struct RegionSolution {
  /// Initiation interval. Empty for a straight-line span, which issues once.
  std::optional<int64_t> ii;
  /// Schedule depth, the cycle by which every op has completed. Report only: a
  /// span composes from `drain` instead, since the solver may leave slack above
  /// the last commit.
  int64_t length = 0;
  /// The region's terminal cycle: `done` rises `drain + 1` cycles after the
  /// last issue pulse. What `leafSpan` composes.
  int64_t drain = 0;
  /// The region's iteration count for one invocation. Empty for a straight-line
  /// span and for an unbounded data-dependent trip. Per-invocation, so a span
  /// is composed where used (`composeSpan`) not stored.
  std::optional<int64_t> trip;
  /// The trip is a worst case from an `allo.assume.ssa` range, not a
  /// compile-time constant, so any span composed from it is a bound.
  bool tripIsBound = false;
};

/// One scheduled op as the report names it: read off the reified op but named
/// after its source op, not the dcp op standing for it.
struct ScheduledOpReport {
  /// An arith/affine-style mnemonic: `addi`, `mulf`, `load`, `stream.get`.
  std::string kind;
  /// Start cycle, relative to the region's own start.
  int64_t start = 0;
  /// The `dcp.operator` symbol realizing it, empty for a combinational or
  /// memory op. The emitted module name derives from it (`operatorModuleName`).
  std::string impl;
  /// Sub-cycle start (ns within the cycle), from a chaining solve.
  std::optional<float> z;
};

/// One scheduling region as the report names it, with the ops it issues
/// directly. A nested region's ops are reported under it, so an op appears
/// once.
struct RegionReport {
  /// `cyclic` for a pipeline, `acyclic` for a straight-line span, `guard` for a
  /// select, which carries no compute of its own.
  std::string kind;
  /// Program order among its func's regions, and nesting depth among dcp
  /// regions (0 = outermost).
  int64_t order = 0, depth = 0;
  /// Whether a region nests inside it (making it a wrapper not a leaf), and
  /// whether its execution is predicated (a while pipeline or a guard).
  bool container = false, conditional = false;
  /// Vitis vocabulary: `interval` the II, `latency` the whole region's span,
  /// `iterationLatency` one iteration's schedule depth. `drain` is the terminal
  /// cycle `done` composes off, a composition quantity not a latency a reader
  /// compares: a solver may leave slack between it and `iterationLatency`, and
  /// only `drain` is charged.
  std::optional<int64_t> interval, tripCount, iterationLatency, drain, latency;
  /// The latency above is an upper bound, not an exact count.
  bool latencyBound = false;
  /// Mnemonic of the region's determinacy class, the controller family pacing
  /// it. Empty only when the attribute is absent.
  std::string determinacy;
  std::vector<ScheduledOpReport> ops;
};

/// What one region's solve cost: a measurement of the compiler, not the
/// hardware, so never stamped as an IR attribute. Kept separate from
/// `RegionReport`: a solve is keyed by the affine loop that owned the problem,
/// gone once the report is built off the reified `dcp` ops. Both lists are in
/// program order per func.
struct SolveReport {
  /// The func it belongs to, and where its region is, as the log names it.
  std::string func, where;
  /// `cyclic` for a counted loop, `while` for a flushing while, `acyclic` for a
  /// straight-line span.
  std::string kind;
  /// Problem size: ops registered, and how many hold at least one limited unit.
  int64_t ops = 0, limitedOps = 0;
  /// The initiation interval settled, absent for an acyclic span.
  std::optional<int64_t> interval;
  /// Wall time of the whole solve in milliseconds.
  double millis = 0.0;
  /// `simplex` or `cpsat`; the config below applies to a cpsat solve only.
  std::string solver;
  int64_t workers = 0, seed = 0;
  double budgetSeconds = 0.0;
  /// The solver's own verdict; see `SolveTelemetry`.
  bool proven = false, spanProven = false, budgetExhausted = false,
       fallback = false;
  std::optional<int64_t> exhaustedAtII;
  /// The area minimization's incumbent and dual bound; see `SolveTelemetry`.
  std::optional<int64_t> modelArea;
  std::optional<double> modelAreaBound;
  /// Whether re-running the compile reproduces this schedule: workers
  /// interleaved and budget held.
  bool deterministic = true;
};

/// One kernel's schedule: an `allo.dcp.module` and the regions it holds.
struct FuncReport {
  std::string name;
  std::optional<int64_t> latency;
  bool latencyBound = false;
  /// Mnemonic of the kernel's determinacy class, the composition contract a
  /// caller holds its `latency` to. Empty only when the attribute is absent.
  std::string determinacy;
  std::vector<RegionReport> regions;
};

/// One kernel's dependence-analysis residue: what stayed outside the
/// polyhedral test and fell to the conservative path. Filled by the
/// scheduler, like `solves`.
struct DependenceReport {
  std::string func;
  /// Accesses the polyhedral test cannot model (non-affine op, or a nest
  /// that is not all-affine).
  int64_t conservativeAccesses = 0;
  /// Pairs the test accepted but could not decide.
  int64_t undecidedPairs = 0;
};

/// A schedule directive the scheduler did not apply. Only refusals are listed:
/// a directive that lands marks the region it shaped, one that does not is
/// otherwise invisible, its region decomposed by report time.
struct UnhonoredDirective {
  /// The directive as the user spelled it (`pipeline`).
  std::string directive;
  /// Source anchor of the loop it was attached to, as the log names it.
  std::string where;
  /// Why it could not be applied, one stable mnemonic rather than prose.
  std::string reason;
};

/// What the scheduling pipeline knows, in the two forms its phases need: the
/// solution hands `runSDCScheduler`'s start times and region solutions to
/// `runPostScheduleConversion`; the report is what the reify builds from the
/// written module, read back by Python via `toJSON`. Valid at disjoint times:
/// the solution between the phases, the report only after (by then `forget`
/// dropped every erased op). Keyed by `Operation *`, valid across both phases,
/// which run back to back with no pass between to fold or rebuild an op.
class ScheduleModel {
public:
  /// Record \p op's solved start. An op is scheduled once, by the solver or by
  /// the reify for a cone the solver never saw, never both.
  void setStart(Operation *op, int64_t start) {
    bool inserted =
        ops.try_emplace(op, OpSchedule{start, std::nullopt, std::nullopt, {}})
            .second;
    assert(inserted && "an op carries one start time");
    (void)inserted;
  }

  /// Record \p op's sub-cycle start; meaningful and read only alongside a
  /// start.
  void setStartInCycle(Operation *op, float z) {
    auto it = ops.find(op);
    assert(it != ops.end() && "a sub-cycle start belongs to a scheduled op");
    it->second.startInCycle = z;
  }

  /// One functional-unit instance an allocation decided to build. The reify
  /// declares a `dcp.unit` per entry and points its ops at that symbol.
  struct AllocatedUnit {
    std::string name;   // the `dcp.unit` symbol, unique across the module
    std::string opType; // the `dcp.operator` it realizes
  };

  /// Declare \p count instances of \p opType and return the first's index.
  /// Names are minted here, so a `dcp.unit` symbol is unique module-wide.
  unsigned addUnits(llvm::StringRef opType, unsigned count) {
    unsigned base = units.size();
    for (unsigned k = 0; k < count; ++k)
      units.push_back(
          {(opType + "_u" + llvm::Twine(units.size())).str(), opType.str()});
    return base;
  }

  /// Record that \p op runs on `allocatedUnits()[index]`.
  void setUnit(Operation *op, unsigned index) {
    auto it = ops.find(op);
    assert(it != ops.end() && "an instance belongs to a scheduled op");
    it->second.unit = index;
  }

  /// Record that an exact solve realized \p op on row \p symbol rather than the
  /// library's pick.
  void setSelectedImpl(Operation *op, llvm::StringRef symbol) {
    auto it = ops.find(op);
    assert(it != ops.end() && "a realization belongs to a scheduled op");
    it->second.selectedImpl = symbol.str();
  }

  /// Every instance the allocation decided to build, module-wide.
  llvm::ArrayRef<AllocatedUnit> allocatedUnits() const { return units; }

  /// \p op's place in its region's schedule, or null when it has none: a
  /// declaration, a terminator, a region anchor, an op no phase scheduled.
  const OpSchedule *scheduleOf(Operation *op) const {
    auto it = ops.find(op);
    return it == ops.end() ? nullptr : &it->second;
  }

  /// Open the solution owned by \p owner: the innermost loop of a counted band,
  /// a flushing `scf.while`, or a straight-line span's first op. The op both
  /// descents land on, hence the key.
  RegionSolution &addRegion(Operation *owner) {
    auto [it, inserted] = regions.try_emplace(owner);
    assert(inserted && "a region is solved once");
    (void)inserted;
    return it->second;
  }

  /// \p owner's solution, or null when it owns none: a sequential wrapper, a
  /// `while` that cannot flush, an all-constant span the solver skipped.
  RegionSolution *regionOf(Operation *owner) {
    auto it = regions.find(owner);
    return it == regions.end() ? nullptr : &it->second;
  }

  /// Record that the schedule satisfies the RAW dependence from \p store to
  /// \p load only through store->load forwarding: the paired store instance
  /// issues \p offset cycles after the read (0 = same cycle, up to the read
  /// latency while the read is in flight), and the reify stamps the pair
  /// onto the dcp accesses.
  void addForward(Operation *load, Operation *store, int64_t offset) {
    forwards[load].push_back({store, offset});
  }
  /// Every recorded (load -> (store, offset)) forwarding, for the reify to
  /// stamp.
  const llvm::DenseMap<Operation *,
                       llvm::SmallVector<std::pair<Operation *, int64_t>, 1>> &
  allForwards() const {
    return forwards;
  }

  /// Record that an `allo.assume.ssa` range bounds \p loop's trip at \p trip,
  /// when its exact count is not compile-time.
  void setTripBound(Operation *loop, int64_t trip) { tripBounds[loop] = trip; }
  /// The assumption-derived worst-case trip of \p loop, or empty when its trip
  /// is compile-time or nothing bounds it.
  std::optional<int64_t> tripBoundOf(Operation *loop) const {
    auto it = tripBounds.find(loop);
    return it == tripBounds.end() ? std::nullopt : std::optional(it->second);
  }

  /// Drop everything recorded about \p op; every erase of a scheduled op owes
  /// the model this. MLIR reuses a freed op's address, so a stale entry would
  /// answer for an op no phase scheduled.
  void forget(Operation *op) {
    ops.erase(op);
    regions.erase(op);
    tripBounds.erase(op);
    forwards.erase(op);
  }

  /// Read \p module's reified `allo.dcp.*` ops into `report`, and the prep
  /// passes' stamped decisions into `prep`. Called once at the tail of the
  /// reify: before it no dcp ops exist, after it the pipeline is gone.
  void record(ModuleOp module);

  /// The report as the JSON document Python parses. Optional fields are omitted
  /// rather than null, as in the interface manifest, so a consumer tests for a
  /// field not a sentinel.
  std::string toJSON() const;

  /// The reified schedule, whole-module. Plain data: the reify fills it and the
  /// CAPI serializes it.
  std::vector<FuncReport> report;

  /// What each solve cost, in solve order: funcs callees-first, regions in
  /// program order within a func. Not one entry per `report` region (a
  /// container solves nothing of its own, one solve covers a perfect band).
  /// Filled by the solver, so unlike `report` it survives whether or not the
  /// reify runs.
  std::vector<SolveReport> solves;

  /// Directives the scheduler could not apply, in encounter order. Filled by
  /// the solver, like `solves`.
  std::vector<UnhonoredDirective> unhonored;

  /// Each kernel's dependence-analysis residue, in schedule order. Filled by
  /// the solver, like `solves`.
  std::vector<DependenceReport> dependence;

  /// The clock period the schedule holds (ns): the target, or the least period
  /// every device row fits when the target is unreachable (`runSDCScheduler`
  /// derates it). What emission prices and checks against.
  float cycleTimeNs = 0.0f;

  /// What the solved schedule costs in the device's own currency, summed over
  /// every region (`regionArea`). The quantity the area objective minimizes,
  /// evaluated on the settled schedule, so two compiles of one kernel at
  /// different periods compare. Filled by the solver, like `solves`.
  int64_t modeledArea = 0;

private:
  llvm::DenseMap<Operation *, OpSchedule> ops;
  std::vector<AllocatedUnit> units;
  llvm::DenseMap<Operation *, RegionSolution> regions;
  llvm::DenseMap<Operation *, int64_t> tripBounds;
  llvm::DenseMap<Operation *,
                 llvm::SmallVector<std::pair<Operation *, int64_t>, 1>>
      forwards;
};

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULE_MODEL_H

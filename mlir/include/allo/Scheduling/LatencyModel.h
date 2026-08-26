/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_LATENCY_MODEL_H
#define ALLO_SCHEDULING_LATENCY_MODEL_H

#include "allo/IR/AlloAttrs.h"           // DeterminacyEnum
#include "allo/Scheduling/RegionGraph.h" // RegionShape

#include "mlir/IR/Block.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::allo {

/// The cycles a controller family spends at its region's boundaries, outside
/// the region's own schedule. Only structural constants live here; a
/// datapath-derived delay (`tCond`, `drainStage`) is passed in as a parameter.
struct BoundaryCost {
  /// `start` -> the first body pass issues.
  unsigned arm;
  /// A body pass completing -> the next one issues. Meaningless for a run-once
  /// family, which sets it equal to `arm`.
  unsigned reArm;
};

/// A `done` level is a latch, so completion is visible one cycle after the
/// pulse that sets it; every family pays it toward its own `done`. A successor
/// of a conditional region starts on the pulse itself, and no static span
/// composes across such a region.
inline constexpr unsigned kDoneLatchCycles = 1;

/// A container's children read its counter as their own bound and sample it at
/// their own start, so every launch is registered, the first included. A
/// container whose first child is conditional relaunches on the last child's
/// completion pulse and composes no body span, so this constant does not
/// describe it.
inline constexpr BoundaryCost kContainerBoundary{/*arm=*/1, /*reArm=*/1};

/// A call region's first pass launches on `start` itself through a start-cycle
/// bypass. Advances still ride the settled register.
inline constexpr BoundaryCost kCallNodeBoundary{/*arm=*/0, /*reArm=*/1};

/// A sequential-wrapper while re-evaluates its condition on a check pulse one
/// cycle after `start` and after each body drain; the condition cone's `tCond`
/// is added on top by the controller. A while composes no span, so this is not
/// priced.
inline constexpr BoundaryCost kCheckedBoundary{/*arm=*/1, /*reArm=*/1};

/// A guard checks its predicate one cycle after `start`. That also keeps a
/// skipped guard's completion pulse off the `done` latch's start-clear, which
/// would otherwise leave no rising edge.
inline constexpr BoundaryCost kGuardBoundary{/*arm=*/1, /*reArm=*/1};

/// An acyclic region arms on `start` directly, nested or top-level: a
/// container's counter and iter-args commit at least a cycle before the launch
/// pulse, and a top-level one has no outer counter to wait for.
inline constexpr BoundaryCost kAcyclicBoundary{/*arm=*/0, /*reArm=*/0};

/// A pipelined leaf issues its first iteration on `start` itself, the counter
/// and scaled counters reading their reload values through a start-cycle
/// bypass. Iterations then overlap at the solved `ii`, which `leafSpan` takes
/// as a parameter, so `reArm` does not describe this family.
inline constexpr BoundaryCost kPipelinedBoundary{/*arm=*/0, /*reArm=*/0};

/// An empty region, a runtime zero trip or a static `lb >= ub`, never launches
/// a pass: `gateStart` masks the start launch and a register on `start &&
/// isEmpty` feeds the `done` latch. Two cycles, whichever family drives the
/// region. A separate constant from the trip-zero arithmetic below, which is
/// written for `trip >= 1`.
inline constexpr int64_t kEmptyRegionCycles = 2;

/// A done-paced region's whole span, given what one pass of its body costs
/// (\p bodySpan, the sum of its children's spans). Evaluated identically by the
/// scheduler and the reifier.
inline int64_t containerSpan(const BoundaryCost &boundary, int64_t trip,
                             int64_t bodySpan) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * (boundary.reArm + bodySpan) + bodySpan +
         kDoneLatchCycles;
}

/// A leaf's whole span: it arms, issues \p trip iterations at its solved \p ii,
/// then drains. \p drain is the terminal quantity, cycles from the last issue
/// pulse to the deepest output committing, so `done` rises `drain + 1` cycles
/// after that pulse. It is not the schedule depth, above which the solver may
/// leave slack.
inline int64_t leafSpan(const BoundaryCost &boundary, int64_t trip, int64_t ii,
                        int64_t drain) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * ii + drain + kDoneLatchCycles;
}

/// One region as the latency model sees it: enough to compose a span. Built by
/// two structural walks, the scheduler over affine/scf loops and the reifier
/// over the dcp regions built from them, that both feed the composition
/// arithmetic above.
struct SpanNode {
  RegionShape shape = RegionShape::Leaf;
  /// Iterations of this region's body. Empty when data-dependent (a `while`, a
  /// dynamic bound), which leaves every enclosing span unknown rather than
  /// guessed.
  std::optional<int64_t> trip;
  /// A leaf's solved schedule: issue cadence `ii` and terminal cycle `drain`,
  /// the delay from the last issue pulse to the deepest output committing. `ii`
  /// stays empty for an acyclic leaf, which issues once. `drain`, not the
  /// schedule depth, since the solver may leave slack above the last commit.
  std::optional<int64_t> drain, ii;
  /// An instance element's whole start->done contract (see `instance`).
  std::optional<int64_t> contract;
  /// A worst case the scheduler bounded from an `allo.assume.ssa` range, for a
  /// node whose own `trip` is data-dependent. Usable only as a kernel's own
  /// `latency` (flagged `latency_bound`), never as a container's body pass,
  /// which must pace a real counter. Carried here because reification keeps the
  /// bounded latency but drops the assumed trip that produced it.
  std::optional<int64_t> assumedSpan;
  /// A straight-line span rather than a counted loop.
  bool acyclic = false;
  /// Paced by back-pressure rather than its own schedule (`isElastic`), so it
  /// has no static span. Set on the node holding the stream access and on every
  /// node above it.
  bool elastic = false;
  /// An instance element rather than a region of this func: `contract` is the
  /// callee's whole start->done span, counted to its own `done` rising.
  bool instance = false;
  /// The contract composed here (an instance's, or one a leaf priced into its
  /// drain) is a ceiling (`latency_bound`), not an exact count.
  bool contractBound = false;
  /// Body elements of a done-paced region, in program order. `std::vector`
  /// since the element type is this one, still incomplete here, which only
  /// `std::vector` is specified to accept.
  std::vector<SpanNode> children;
  /// A guard's else-arm elements, in program order; `children` holds its
  /// then-arm's. Empty for an absent or empty else, which completes in the
  /// arming cycle.
  std::vector<SpanNode> elseChildren;
};

/// The per-invocation span of \p n: its start pulse to its `done` rising,
/// including the node's arming cost, so a composer only sums spans. A leaf runs
/// its solved schedule and drains it; a done-paced region runs its body in
/// sequence, each element handed on through its own `done` latch; a guard
/// composes the deeper arm, a ceiling rather than an exact count.
///
/// nullopt whenever any element is data-dependent, leaving the enclosing span
/// unknown rather than guessed.
std::optional<int64_t> composeSpan(const SpanNode &n);

/// Whether the span composed of \p n is a ceiling rather than an exact count: a
/// guard runs whichever arm its predicate takes, an assumed trip is a worst
/// case, and a bounded contract hands on its callee's ceiling. Meaningful only
/// where `composeSpan` returned a value; the composition is monotone, so a
/// ceiling composes into a ceiling.
bool spanHoldsBound(const SpanNode &n);

/// A run of nodes composed in program order, the sum of their spans: each
/// starts on its predecessor's `done` edge, which costs nothing (`startFor` is
/// a rising edge, not a register). Used for a done-paced region's body pass and
/// for a func's top-level regions along one path of their DAG.
std::optional<int64_t> composeSequence(llvm::ArrayRef<SpanNode> nodes);

/// For each top-level node, in program order, the earlier nodes it must run
/// after. \p nodeOps gives the ops each node owns, keeping this IR-agnostic
/// across the scheduler's and the reifier's regions.
///
/// Three ordering signals, the same the emitter composes on
/// (`recordSiblingDeps`): a shared memref orders only its hazard pairs (RAW /
/// WAW / WAR), so read-read pairs overlap; a shared stream channel orders its
/// touchers regardless of direction; a cross-region SSA use. Everything else
/// runs concurrently.
///
/// The edges must match `recordSiblingDeps`, since the span is published as an
/// exact contract. `RegionGraph`'s polyhedral refinement is not used here: it
/// drops edges the emitter keeps.
std::vector<llvm::SmallVector<unsigned, 2>>
siblingPredecessors(llvm::ArrayRef<llvm::SmallVector<Operation *>> nodeOps);

/// One node's (or region's) touch of a shared array, for `hazardEdges`.
struct MemTouch {
  unsigned node;
  bool writes;
  /// The statically-resolved bank, or nullopt for a touch that may reach any
  /// (a crossbar access, a child's mastered port).
  std::optional<int64_t> bank;
};

/// Invoke \p addPred(p, c) for each pair of \p touch (sorted by node) the
/// hazard rule orders: per bank, each reader after the last writer and each
/// writer after every reader since. Read-read pairs stay unordered, and touches
/// of different banks never pair; a bank-less touch joins every bank. The one
/// edge rule `siblingPredecessors` and `recordSiblingDeps` both compose on.
void hazardEdges(llvm::ArrayRef<MemTouch> touch,
                 llvm::function_ref<void(unsigned, unsigned)> addPred);

/// A func's top-level span: its regions composed over their dependence DAG, the
/// longest path not the sum, since independent siblings overlap. \p preds is
/// `siblingPredecessors`, indexed alongside \p nodes.
std::optional<int64_t>
composeDag(llvm::ArrayRef<SpanNode> nodes,
           llvm::ArrayRef<llvm::SmallVector<unsigned, 2>> preds);

/// One materialized dcp region as the latency model sees it: the reify-side
/// structural walk over `dcp.pipeline` / `dcp.sequential` / `dcp.select` and
/// the `dcp.instance` elements they hold. `Scheduler.cpp` has the other, over
/// the affine/scf loops these were built from. A region's span and composition
/// class are derived where used, never read back off an attribute. For
/// \p topLevel see `dcpSpanNodes`.
SpanNode dcpSpanNode(Operation *regionOp, bool topLevel);

/// The elements of a reified block, in program order.
///
/// The two scopes differ in what a span is for. \p topLevel is a func's entry
/// block, composing an exported contract, so an assume-bounded region
/// contributes its bound for a caller to wait out. A region body composes one
/// pass of a counter, where a bound cannot pace anything, so a bounded child
/// leaves the container done-paced instead.
std::vector<SpanNode> dcpSpanNodes(Block &block, bool topLevel);

/// How a materialized region is paced: which controller family drives it, and
/// the single-run span a container may time-trigger it against.
struct RegionTiming {
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;
  /// Present only for `counted_static`, and then exact.
  std::optional<int64_t> staticLatency;
  /// A ceiling where a span composes but is not exact (`spanHoldsBound`): a
  /// guard's deeper arm, or a bounded contract beneath. Never set beside
  /// `staticLatency`. Published as `latency_bound`, which a caller may wait out
  /// but nothing may time-trigger against.
  std::optional<int64_t> boundedLatency;
};

/// Derive \p regionOp's pacing from the region itself.
///
/// One definition, called twice: the reifier stamps `latency` / `latency_bound`
/// / `determinacy` from it, the emitter decides a controller family from it, so
/// those attributes are a report of this function, never an input.
///
/// Four classes, tested in order since each shadows the ones after it:
/// concurrent, conditional, counted_static, indeterminate.
RegionTiming dcpRegionTiming(Operation *regionOp);

/// Func-level: whole-kernel latency in cycles, the top-level regions composed
/// over their dependence DAG (`publishKernelLatency`). Set only when every
/// region has a composable span. Exact count versus assume-bounded worst case
/// is not recorded: a bound is an upper one, so it times a caller safely either
/// way.
constexpr llvm::StringLiteral kLatencyAttr = "allo.sched.latency";

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_LATENCY_MODEL_H

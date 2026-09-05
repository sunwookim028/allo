/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_DATAPATHBUILDER_H
#define ALLO_MICROARCH_DATAPATHBUILDER_H

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/Datapath.h"

#include "allo/IR/AlloOps.h"

#include "llvm/ADT/MapVector.h"

#include <deque>

namespace mlir::allo::uarch {

/// The memref a `dcp.load` / `dcp.store` accesses; null for any other op.
Value dcpMemref(Operation *op);
/// Every op of \p regionOp that binds a resource: its body, plus a guard's else
/// branch. A nested region is visited in its own turn, so this does not
/// recurse. The facts sweep and the binding walk must enumerate the same ops in
/// the same order.
void forEachBodyOp(Operation *regionOp,
                   llvm::function_ref<void(Operation *)> fn);

/// The producer of a value plus the register depth a consumer needs to read it.
struct Resolved {
  // The producing cell output; None => producer outside this region or not
  // modelled, read as an unresolved edge.
  Source base;
  Value key;      // register key (the produced SSA value; null => never reg)
  unsigned depth; // pipeline-register depth for this edge
  unsigned ready = 0; // cycle `base` lands at within its iteration
  // Reduction identities of the loop-carried iter_arg this edge reads, one per
  // iteration: `inits[n]` re-injected at iteration n, `base` carrying the edge
  // from `inits.size()` on (that size the recurrence distance). Empty
  // otherwise.
  llvm::SmallVector<Source, 1> inits = {};
};

//===----------------------------------------------------------------------===//
// DatapathBuilder. One instance builds one function's Datapath: `build()`
// drives allocation then interconnect derivation and returns the model.
//===----------------------------------------------------------------------===//
struct DatapathBuilder {
  Datapath &dp;
  dcp::DCPathModuleOp func;

  // Build-time scratch, not part of the result: value/op provenance maps.
  llvm::DenseMap<Value, MemId> memOf;
  llvm::DenseMap<Value, StreamId> streamOf;
  // Keyed by the produced value, not its op: a multi-result producer (a
  // sub-kernel call) drives one Source per result.
  llvm::DenseMap<Value, Source> producerOf;
  llvm::DenseMap<Value, Source> ioOf;
  llvm::DenseMap<Operation *, unsigned> regionIdxOf;

  // Forwarding facts off the dcp attributes: `fwd_id` -> (mem, access), and
  // each forwarded load's (mem, access, ids, per-id window offsets).
  llvm::DenseMap<int64_t, std::pair<MemId, unsigned>> fwdStoreOf;
  llvm::SmallVector<std::tuple<MemId, unsigned, llvm::SmallVector<int64_t, 1>,
                               llvm::SmallVector<int64_t, 1>>>
      fwdLoads;

  // Interconnect-derivation scratch (transient; see resolveEdges).
  // A delay chain is keyed by (held value, consuming region): one value read in
  // several nested regions needs its own chain in each.
  using RegKey = std::pair<::mlir::Value, unsigned>;
  struct Edge {  // an input slot's driver, and the delay it owes before landing
    Source base; // what drives the head of the chain
    RegKey key;  // the chain this slot taps
    unsigned depth; // the tap it reads
    unsigned ready; // cycle `base` lands at within its iteration
    // The address reduction folded this operand into a scaled counter, so no
    // chain owes it a tap and its slot stays empty.
    bool reduced = false;
  };
  struct MuxBuild { // one slot's per-op drivers, muxed after the chains exist
    Source *slot;
    RegionId region;
    Type type;
    llvm::SmallVector<Operation *, 2> ops;
    llvm::SmallVector<Source, 2> sources;    // parallel to ops
    llvm::SmallVector<Mux::Phase, 2> phases; // parallel to ops
  };
  // Keyed by the slot each edge patches; a slot takes at most one. Chains are
  // built in record order.
  llvm::MapVector<Source *, Edge> edges;
  std::deque<MuxBuild> muxBuilds; // a deque so `recordEdge` /
                                  // `recordCarriedEdge` slot pointers into
                                  // `sources` survive later pushes
  // Where the cell containers sat when `resolveEdges` finished. `edges` keys
  // slots by pointer, so the delay passes assert they have not moved.
  const void *unitsBase = nullptr, *memsBase = nullptr, *streamsBase = nullptr;

  const BindingPolicy &policy; // decides resource sharing
  const DeviceModel &dev;      // device storage + operator timing
  float cycleTime;             // the period the schedule was cut against
  const CalleeCtx &callees;    // child modules/ifaces for a dcp.instance
                               // (null for a plain leaf, no calls)

  DatapathBuilder(Datapath &dp, dcp::DCPathModuleOp func,
                  const BindingPolicy &policy, const DeviceModel &dev,
                  float cycleTime, const CalleeCtx &callees)
      : dp(dp), func(func), policy(policy), dev(dev), cycleTime(cycleTime),
        callees(callees) {}

  /// build the datapath model
  void build();

  // Allocation & binding -----------------------------------------
  /// Register every literal as a tie-off ConstCell (func-wide, so a hoisted
  /// constant resolves the same as an in-body one).
  void collectConstants();
  /// Create a RegionBlock for \p regionOp (id \p ridx): kind/ii/length/trip and
  /// the parent/child linkage. Returned by value; pushed by `build`.
  RegionBlock addRegion(Operation *regionOp, RegionId ridx);
  /// Derive every region's `shape` discriminant (see `RegionBlock::Shape`) and
  /// assert the invariants each shape carries. Runs after the region walk, once
  /// parent/child edges and the CallUnits it reads are complete.
  void deriveShapes();
  /// Bind one body op to its resource: one arm per resource kind, plus the
  /// kinds that bind nothing (a nested region, a literal, a declaration). An op
  /// matching none is reported and marks the build infeasible.
  void bindResource(Operation *op, RegionBlock &rb);
  /// A `dcp.instance` -> a CallUnit owned by \p rb: one MemArg per child memory
  /// port, one scalar-input slot per scalar operand (its driver resolved later
  /// by `recordCallScalars`), and a `Source::Call` producer per scalar result.
  void bindCall(dcp::DCPathInstanceOp inv, RegionBlock &rb);
  /// A `stream.get` / `stream.put` -> one StreamChannel access. Both directions
  /// bind identically; only a get produces a token.
  void bindStream(Operation *op, RegionBlock &rb);
  /// A `dcp.load` / `dcp.store` on \p memref -> one MemUnit access. Asserts the
  /// two contracts this binding assumes: no store to a memory classified
  /// read-only, and the scheduled access latency equalling the device model's.
  void bindMemory(Operation *op, Value memref, RegionBlock &rb);
  /// Resolve the `fwd` / `fwd_id` attributes into `MemUnit::forwards`. Runs
  /// after the region walk, once every access index exists.
  void recordForwards();
  /// A `dcp.compute` -> a FuncUnit, combinational or IP-realized, holding the
  /// op at its reservation slot (its issue cycle, modulo II when cyclic).
  void bindCompute(dcp::DCPathComputeOp comp, RegionBlock &rb);
  /// Build one MemUnit per array the function touches, holding what device and
  /// layout say, and classify an initialized array nothing writes as a constant
  /// table. Runs before the region walk and must reach the same ops. Fixes
  /// MemId order; boundary port order follows `m.accesses`, set by the binding
  /// walk.
  void collectStorageFacts(llvm::ArrayRef<Operation *> regionOps);
  /// The MemUnit backing \p memref. A lookup, not a factory:
  /// `collectStorageFacts` has already built them all.
  MemId memIdOf(Value memref);
  /// Allocate (or reuse) a StreamChannel for the `!allo.stream` value \p stream
  /// (a func block arg). \p isInput sets the channel direction on first
  /// touch (a get => input, a put => output).
  StreamId getOrCreateStream(Value stream, bool isInput);
  /// Record how each region produces its results (`rb.results`) and, where it
  /// has one, its control predicate (`rb.condition`). A region result is a
  /// survivor register; a loop's k-th result is its k-th iter-arg's last value.
  void recordRegionResults();
  /// Resolve each `dcp.instance`'s scalar operands into its CallUnit's
  /// `scalarIns`. Separate from `bindResource`: a Source resolution needs the
  /// complete region model (see `resolveValue`).
  void recordCallScalars();
  /// Record every CallUnit's composition predecessors (`cu.predecessors`),
  /// hazard-directed (RAW / WAW / WAR); a read-read pair commutes and overlaps.
  /// A scheduled composition gates only an earlier-placed or indeterminate
  /// hazard producer; a concurrent one has no placement, so an unordered hazard
  /// is the whole rule. Runs after `recordCallScalars`.
  void recordCallDeps();
  /// Derive every cyclic region's `counterType`, the width its iteration
  /// counter and bounds are built at, from that loop's induction range. A
  /// consumer wanting another width adapts at its own end (a datapath read
  /// widens back to `kIndexWidth`, a child's index port takes the port's width,
  /// an address cone the memory's).
  void deriveCounterTypes();
  /// Record each pipeline's induction bounds (lb/ub/step) as Sources on its
  /// RegionBlock: a runtime bound from the `lbBound`/`dynamicBound`/`stepBound`
  /// operand, a compile-time one as a literal cell. Needs `counterType`, the
  /// width those literals are tied in at.
  void recordRegionBounds();
  /// A literal \p v of type \p t as a Source, appending the ConstCell that
  /// holds it. For a value the model needs but no `arith.constant` in the body
  /// produces, such as an induction bound written as an attribute.
  Source constant(int64_t v, Type t);
  /// Enumerate the module's boundary memory ports: `dp.{read,write}Ports`, each
  /// external access's `MemUnit::Access::{portIdx, portBase}` and each
  /// call-mastered argument's `CallUnit::MemArg::topBase`, all off one
  /// per-(memory, role) counter so parent accesses number first and child ports
  /// continue in call order. Runs once every access and call is bound. Owner
  /// names come from `uniqueOwnerOf` over the whole memref list, so two
  /// arguments sharing a source name still differ.
  void enumerateBoundaryPorts();
  /// Bind every memory access and child port to a port of its bank
  /// (`MemUnit::Access::port`, `CallUnit::MemArg::port`) and record how many
  /// ports each bank is built with. Runs after `planAccessPorts`, which settles
  /// how each access reaches its memory.
  void bindMemoryPorts();
  /// Record the instances of its row each bank is held in
  /// (`MemUnit::instances`) and which of them serves each read port
  /// (`MemUnit::readInstance`), by the same arithmetic `bindMemoryPorts`
  /// compares for its pooled decision. Runs over every memory, since a skew
  /// binds its ports by lane and leaves that loop early.
  void assignReadInstances();
  /// Group a skewed memory's accesses into lanes that can share one port per
  /// bank (`MemUnit::skewed`, `Access::lane`), or leave it crossbarring when
  /// they cannot. Runs before `planAccessPorts`, which reads whether the skew
  /// held.
  void assignLanes();
  /// Decide how each access and each child-mastered port reaches its memory
  /// (`MemUnit::Access::plan`, `CallUnit::MemArg::plan`). Runs before
  /// `bindMemoryPorts`, which hands out ports along the plan it settles.
  void planAccessPorts();
  /// Ports one bank comes out of a `planPorts` colouring with: split by
  /// direction, plus `total`, which is below their sum wherever a port carries
  /// both. `colours` is the whole memory's count, where a second colouring of
  /// the other direction starts numbering.
  struct PortCounts {
    unsigned reads = 0, writes = 0, total = 0, colours = 0;
  };
  /// One candidate binding of a memory's ports, held outside the model so two
  /// can be compared before either is committed.
  struct PortAssignment {
    /// The port each vertex of `Datapath::portGraph`'s order takes, already
    /// offset by the `base` it was planned at.
    llvm::SmallVector<unsigned> colour;
    PortCounts counts;
    /// Whether the writes landed on ports separate enough to be separate
    /// `always` blocks. Meaningful only when `writes` covered them.
    bool writesIndependent = false;
    /// The direction this covers, or none for a both-directions colouring.
    /// `commitPorts` replays the same vertex walk from it.
    std::optional<bool> writes;
  };
  /// Colour one memory's port graph. \p writes picks a direction, or nullopt
  /// takes both together. \p base offsets the numbering so a second, separate
  /// colouring cannot collide with the first. Returns nullopt only from a
  /// both-directions pass whose writes did not split; a pass given a direction
  /// always plans.
  std::optional<PortAssignment>
  planPorts(const MemUnit &m, std::optional<bool> writes, unsigned base);
  /// Write \p pa into `MemUnit::Access::port` / `CallUnit::MemArg::port`, and
  /// `MemUnit::writesIndependent` where it covered the writes.
  void commitPorts(MemUnit &m, const PortAssignment &pa);
  /// Record what each memory's ports cost against what its schedule asks:
  /// `MemUnit::{readConcurrency, writeConcurrency, boundaryPorts}`. Measures
  /// only; `checkStorageLegality` decides what is worth reporting. Runs after
  /// `enumerateBoundaryPorts`, whose groups it counts.
  void measurePorts();
  /// Record each top-level region's composition predecessors
  /// (`rb.predecessors`): the earlier siblings it must start after. Runs after
  /// the region walk (bound accesses and region tree ready) and before the port
  /// passes, which read the ordering it establishes (`Datapath::portGraph`).
  void recordSiblingDeps();
  /// Scalar (non-memref) function arguments become input IOPorts.
  void bindIOArgs();
  /// Scalar (non-memref) function results become `dp.results` output ports,
  /// each driven by the Source of its `func.return` operand. Array results are
  /// out-params by this stage (buffer-results-to-out-params).
  void recordResults();

  /// Settle the allocation: fold each group in \p groups onto one unit and
  /// rebuild the table densely, so a unit with no bound op never exists. Runs
  /// right after the region walk, the last point a `UnitId` is held only by
  /// `producerOf` and `dp.opToUnit` (both rewritten here); a later fold would
  /// leave resolved Sources naming a unit with no ops.
  void allocateUnits(llvm::ArrayRef<llvm::SmallVector<UnitId, 2>> groups);

  // Value resolution ---------------------------------------------
  /// The one Value -> Source resolution: the channel through which \p v can be
  /// read, or None if this datapath cannot (a caller reports an unresolved
  /// slot, never drops it). Needs the complete region model, to know which
  /// regions latch iter-args, so every caller runs after the region walk.
  Source resolveValue(Value v);

  // Interconnect derivation ---------------------------------------
  /// Resolve an operand \p v consumed by \p consumer (region II \p ii) to its
  /// producing Source plus register depth, plus the one edge that does not read
  /// \p v (an un-latched own iter-arg, the loop recurrence). An enclosing
  /// region's counter is held for the whole nested pass and ties in with no
  /// chain; on an address slot (\p addressSlot) it keeps its depth, the address
  /// reduction folding counters into scaled strides and withdrawing the chain
  /// where it succeeds.
  Resolved resolveOperand(Value v, Operation *consumer, unsigned ii,
                          bool addressSlot = false);
  /// Resolve every unit input / memory address / store-data / stream driver
  /// into an `Edge`, recording what each slot reads and how late. Materializes
  /// nothing.
  void resolveEdges();
  /// Record a resolved edge into \p slot: a depth-0 edge ties directly, a
  /// deeper one is deferred to `edges` and patched by `insertRegisters`.
  void recordEdge(const Resolved &r, Source &slot, unsigned regionIdx);
  /// `recordEdge` for a slot that is a bare Source with no input port beside it
  /// to hold a recurrence identity (`FuncUnit::inputInits`): an address, a
  /// store datum, a stream token. A recurrence edge on \p operand of \p
  /// consumer is muxed against its identities, phased on that op's issue cycle;
  /// every other edge is recorded unchanged.
  void recordCarriedEdge(const Resolved &r, Value operand, Operation *consumer,
                         Source &slot, unsigned regionIdx);
  /// Resolve every unit input (single, or shared-then-muxed).
  void resolveUnitInputs();
  /// Resolve every memory access's address operands and store datum.
  void resolveMemoryOperands();
  /// Resolve every stream access's put datum and get/put predicate.
  void resolveStreamOperands();
  /// Decide which accesses carry their address in a register that advances
  /// with the loop counters, record the scaled counters that needs, and
  /// withdraw the edge of every operand no residual is left reading.
  void planAddressGenerators();
  /// Build one chain per (value, region) the surviving edges tap, patch their
  /// slots, and materialize the shared-unit muxes.
  void insertRegisters();
  /// Materialize the sharing muxes: one shared driver needs no mux. Runs last
  /// in `insertRegisters`, the sources being final only once the registers are
  /// built and the pending slots patched.
  void materializeMuxes();
  /// Record each region's terminal cycle (`RegionBlock::drainStage`). Runs
  /// after `resolveEdges`, the last pass that moves a stream put's stage.
  void recordDrainStages();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATHBUILDER_H

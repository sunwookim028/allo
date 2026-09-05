/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Per-region emission, split by role along the control/datapath (F/G) seam.
// ControlEmitter is control (G), DatapathEmitter is datapath (F), HWEmitter the
// orchestrator. Elasticity (H) is derived per region by `deriveStallShell` and
// handed to each side: G takes `issueEnable`, F takes `chainEnable`.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_HWEMIT_H
#define ALLO_MICROARCH_HWEMIT_H

#include "allo/IR/AlloOps.h" // kMemPortAttr
#include "allo/Microarch/Naming.h"
#include "allo/Microarch/Primitives.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::allo::iface {
struct ModuleInterface; // each callee's port model, read to wire its instance
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Terminator: what ends a pipelined region's run, a counter reaching an
// iteration bound (`Counted`) or a condition going false (`Conditional`).
//===----------------------------------------------------------------------===//
struct Terminator {
  enum class Kind { Counted, Conditional };
  Kind kind = Kind::Counted;
  // Induction bounds: `lb`/`step` seed the counter register (init lb, +=step),
  // `ub` ends it (iv+step >= ub). A while free-runs a 0-based counter
  // (lb=0/step=1, ub null), terminating on ~cond.
  Value lb, ub, step;
  Value cond; // Conditional: the i1 continue condition (a datapath value)
  /// The counter is unsigned, so its bound compares are unsigned. A negative
  /// `lb` or the 32-bit fallback keeps the signed form.
  bool unsignedCounter = false;

  static Terminator counted(Value lb, Value ub, Value step, bool isUnsigned) {
    return {Kind::Counted, lb, ub, step, Value(), isUnsigned};
  }
  static Terminator conditional(Value cond, Value zero, Value one) {
    return {Kind::Conditional, zero, Value(), one, cond, false};
  }

  /// `a >= b` at the counter's signedness.
  Value ge(EmitContext &c, Value a, Value b) const {
    return unsignedCounter ? c.icmpUgeV(a, b) : c.icmpSgeV(a, b);
  }
  /// The iteration issued at `iv` is the last one: `iv + step` reaches `ub`, or
  /// the continue-condition is false. \p ivStep is `iv + step`.
  Value isLast(EmitContext &c, Value ivStep) const {
    return kind == Kind::Conditional ? c.notBit(cond) : ge(c, ivStep, ub);
  }
  /// The region is empty (issues nothing): lb >= ub. A while is never empty
  /// here; its zero-iteration case is the condition false on iteration 0.
  Value isEmpty(EmitContext &c) const {
    return kind == Kind::Conditional ? c.f1 : ge(c, lb, ub);
  }
  /// The start pulse gated so an empty region issues nothing; a while passes
  /// through unconditionally.
  Value gateStart(EmitContext &c, Value start) const {
    return kind == Kind::Counted ? c.andBits(start, c.notBit(isEmpty(c)))
                                 : start;
  }
};

//===----------------------------------------------------------------------===//
// ControlEmitter (G): a per-region control regime plus its completion signal.
// It consumes a resolved Terminator and never itself walks a Source.
//===----------------------------------------------------------------------===//
struct RegionControl {
  Value issue; // pipeline issue / valid signal (already gated by the stall
               // shell's enable)
  Value
      counter; // iteration index (Source::Counter); null for an acyclic region
  Value wantIssue; // the ungated issue desire (issue before `& enable`): the
                   // stall shell hazards a stage-0 stream access on this, not
                   // the gated issue, to stay combinationally acyclic
  Value running;   // the region is executing: the counter reloads its lower
                   // bound while low. Null for a done-driven controller, whose
                   // counter reloads on `start` instead.
  /// The modulo phase [0, ii), 0 in the cycle the first iteration issues (the
  /// bypassed start cycle for a rigid counted region, the cycle after `start`
  /// for the registered families) and advancing on every enabled cycle after
  /// it, drain included: an op landing at cycle `r` lands at phase `r % ii`.
  /// Null unless a schedule-paced controller runs at II > 1.
  Value phase;
  /// One register per `RegionBlock::addrStrides` entry, holding that multiple
  /// of `counter`, updated by the same expression as the counter.
  llvm::SmallVector<Value> scaledCounters;
};

//===----------------------------------------------------------------------===//
// DatapathFeedback (F -> G): the store timing a control regime consumes to
// compute completion.
//===----------------------------------------------------------------------===//
struct DatapathFeedback {
  // The deepest store's commit cycle as the writes were emitted (a stream put
  // folds in too); 0 if the region stores nothing. `emitRegion` checks it
  // against `RegionBlock::drainStage`, the same number decided on the model.
  unsigned storeDrain = 0;
  // A CallUnit region's completion. For a CallNode (loop-over-call) region it
  // is the child's per-invocation completion pulse, which paces the counter;
  // for a scheduled or concurrent composition it is the joined pass-scoped
  // done level, which forms part of the region's completion. Null for a
  // call-free region.
  Value callDone;
};

//===----------------------------------------------------------------------===//
// IterationControl: the output of a done-driven controller, one whose
// iterations are paced by the body completing rather than by the schedule's own
// cadence. `rc.issue` is the body-launch pulse; `done` is latched here rather
// than by `emitDone`.
//===----------------------------------------------------------------------===//
struct IterationControl {
  RegionControl rc;
  /// The latched completion level, null where the caller declared it unused
  /// and only the pulse below was built.
  Value done;
  /// The pulse that sets `done`: high in the completion cycle itself, one
  /// cycle before the latched level. A successor may start on it directly only
  /// where `handoffSafe` says the region's state has settled; otherwise it is
  /// registered, which lands in the level's own rising-edge cycle.
  Value donePulse;
};

struct ControlEmitter {
  EmitContext &c;
  explicit ControlEmitter(EmitContext &c) : c(c) {}

  /// Pick the control shape for region \p rb, acyclic or a pipelined loop, and
  /// emit it, driven by \p start against the resolved \p term.
  RegionControl emitPipelineControl(const uarch::RegionBlock &rb,
                                    const Terminator &term, Value start,
                                    const StallShell &sh) const;
  /// The scaled counters of \p rb: one register per `addrStrides` entry, each
  /// at its own width (`AddrStride::width`) rather than the counter's. \p
  /// update is the counter's next-value expression with `lb` and `step` scaled,
  /// supplied by the caller since the two controller families disagree about
  /// it. \p bypassStart mirrors a done-driven counter's start-cycle bypass. A
  /// slot flagged `isCounter` builds no register and takes \p counter directly,
  /// the two being the same value at the counter's own width.
  llvm::SmallVector<Value> emitScaledCounters(
      const uarch::RegionBlock &rb, Value bypassStart, Value counter,
      llvm::function_ref<Value(Value cur, Value stepped, Value init)> update)
      const;
  /// The one pipelined control skeleton for the free-running (II==1), modulo
  /// (II>1) and while (flushing) regimes: a `running` flag plus an iteration
  /// counter, differing only in \p term and, for II>1, a phase counter gating
  /// issue. Non-speculative for a conditional terminator (II >= t_cond, so no
  /// doomed iteration issues); no backpressure. \p sh gates issue as
  /// `wantIssue & sh.issueEnable` and runs the phase counter on
  /// `sh.chainEnable`; a rigid shell leaves both ungated.
  /// \p rb names the emitted state cells (`r<id>_run` / `_iv` / `_phase`) and
  /// carries the II the phase counter runs at.
  RegionControl emitPipelined(const uarch::RegionBlock &rb,
                              const Terminator &term, Value start,
                              const StallShell &sh) const;
  /// The straight-line control skeleton: one pass, no counter. The pass is
  /// deferred while `sh.issueEnable` is low rather than dropped, letting a
  /// stage-0 stream access wait for its handshake. A rigid shell issues
  /// unconditionally and builds no state at all.
  RegionControl emitAcyclic(unsigned region, Value start,
                            const StallShell &sh) const;

  /// The counted done-driven controller: `Container` and `CallNode` x
  /// `CountedStatic`. Runs one body pass per iteration of \p term, launching
  /// the next when \p complete pulses (the body's drain edge, a Backedge the
  /// caller resolves once the body has emitted). \p rb's `shape` decides
  /// whether the first pass may launch on \p start itself. With \p chained,
  /// \p complete is the last child's commit pulse and the next pass launches
  /// on it directly. Without \p wantLevel the completion latch is not built and
  /// only `donePulse` is returned.
  IterationControl emitCountedIteration(const uarch::RegionBlock &rb,
                                        const Terminator &term, Value start,
                                        Value complete, bool chained,
                                        bool wantLevel) const;
  /// The conditional done-driven controller: `Container` x `Conditional`, a
  /// sequential-wrapper while. A CHECK pulse one cycle after \p start and after
  /// each body drain (\p complete) re-evaluates \p cond on the settled
  /// iter-args, \p tCond cycles later for a condition that reads memory or an
  /// IP, then forks to continue or finish. The region has no counter, so the
  /// returned `rc.counter` is null. Without \p wantLevel the completion latch
  /// is not built and only `donePulse` is returned.
  IterationControl emitCheckedIteration(unsigned region, Value cond,
                                        unsigned tCond, Value start,
                                        Value complete, bool wantLevel) const;

  /// The region's completion signal, one latched level for every regime: it
  /// rises when \p lastIssue delayed `rb.drainStage` cycles lands, or
  /// immediately on \p emptyDone (null when unreachable). The latch's register
  /// cycle is the store/result commit cycle, so a sibling starting on this done
  /// reads every committed store and survivor. A \p retrig region resets on \p
  /// start and reads 0 on the \p start cycle itself, since a completion pulse
  /// coinciding with \p start would otherwise latch high on the first pass and
  /// never produce a later rising edge. \p sh holds the pulse through
  /// back-pressure, the last store or token not being committed until it is
  /// accepted.
  /// Returns {level, pulse}: the latched level and the pulse that sets it,
  /// which is high in the last commit cycle itself. Without \p wantLevel the
  /// latch is not built and the level comes back null: a successor reading only
  /// the level's rising edge registers the pulse instead, which is the same
  /// cycle for one register rather than two.
  std::pair<Value, Value> emitDone(const uarch::RegionBlock &rb,
                                   Value lastIssue, Value emptyDone,
                                   Value start, bool retrig,
                                   const StallShell &sh, bool wantLevel) const;
};

//===----------------------------------------------------------------------===//
// DatapathEmitter (F): register chains, compute units, memory access, and
// Source resolution. Reads the controller's `issue`/`counter` (setControl) and
// returns the region's store drain.
//===----------------------------------------------------------------------===//
struct DatapathEmitter {
  EmitContext &c;
  // The sealed model. Emission is a pure function of it.
  const uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;
  const llvm::StringMap<Operation *> &opModules;

  // A region's controller outputs, the G->F seam. `counter` is null for an
  // acyclic region; `wantIssue` is null when the region has no stall shell.
  DenseMap<unsigned, RegionControl> controlOf;
  // Each region's counter widened to `kIndexWidth`. The counter register is
  // only as wide as its own induction range needs (`RegionBlock::counterType`);
  // a datapath read of it is an ordinary index, so this is the second, wider
  // wire.
  DenseMap<unsigned, Value> counterIndex;
  // H's output per region. An access or a shared-unit mux is timed against the
  // shell of the region that owns it, which need not be the one currently
  // emitting. An unregistered region is rigid (the default `StallShell`).
  DenseMap<unsigned, StallShell> shellOf;
  DenseMap<uint64_t, Value> streamReadData; // (channel id, access idx) -> the
                                            // input-stream data port value
  DenseMap<uint64_t, Value> survivorOf;    // (region id, result idx) -> latched
                                           // result (accKey-packed)
  DenseMap<uint64_t, Value> callResultVal; // (call id, result idx) -> the child
                                           // instance's scalar result output
                                           // (populated by emitCalls)
  /// Internal mem id -> the hlmem handles holding it, bank-major over the
  /// instances of each bank (`bank * instances + inst`). Empty for a scattered
  /// array, which holds no hlmem. Index it through `memReadCell` /
  /// `memWriteCells` rather than directly.
  DenseMap<unsigned, SmallVector<Value>> memBanks;
  /// The one instance of \p m's bank \p bank that answers read port \p port.
  /// Every instance holds the whole array, so one of them is enough.
  Value memReadCell(const uarch::MemUnit &m, unsigned bank, unsigned port) {
    return memBanks[m.id][bank * m.instances +
                          m.readInstance.lookup(
                              uarch::MemUnit::instanceKey(bank, port))];
  }
  /// Every instance of \p m's bank \p bank: a write reaches all of them, each
  /// copy needing it to stay the same array.
  ArrayRef<Value> memWriteCells(const uarch::MemUnit &m, unsigned bank) {
    return ArrayRef<Value>(memBanks[m.id])
        .slice(bank * m.instances, m.instances);
  }
  /// One backedge per element of a scattered internal array, in flat row-major
  /// order: the register's own output. Declared with the array so a read can
  /// select over the elements before the stores that drive them have emitted,
  /// and resolved by `finalizeScatteredPorts` once they all have.
  DenseMap<unsigned, SmallVector<circt::Backedge>> scatterElems; // by MemId
  DenseMap<unsigned, Value>
      romArray; // ROM mem id -> its hw.aggregate_constant array value
  DenseMap<unsigned, circt::Backedge> regHeadBE; // reg id -> chain head input
  DenseMap<unsigned, ShiftChain> regStages;      // reg id -> its tap chain
  DenseMap<uint64_t, Value> readData;            // (mem,access) -> read data
  DenseMap<unsigned, Value> unitVal;             // unit id -> result
  DenseMap<unsigned, circt::Backedge> unitBE;    // unit id -> result backedge
  DenseMap<unsigned, Value> muxVal;              // mux id -> resolved output

  /// One accessor's drive of a shared physical port: the terms it presents and
  /// the pulse that says it is presenting. A port is reached by several
  /// accessors (the accesses of different regions, a child mastering it) while
  /// each of `hw.output`, `seq.write` and an element register takes it exactly
  /// once, so an arm is built where its accessor emits and only combined by
  /// `commitSink` once every region has emitted.
  struct SinkArm {
    /// This accessor is driving now: a store's commit pulse, a region's
    /// accesses presenting, a child's run window. Null only where the arm holds
    /// the port alone and drives it unconditionally.
    Value fired;
    Value addr; // null on a sink that carries no address
    Value data; // null on a sink that carries no datum
  };
  /// What a shared port carries in a cycle no arm fires.
  enum class Idle {
    DontCare, // nothing samples it: a write whose enable is low
    Hold,     // it must keep the last value: an address bus another region owns
  };
  /// Reduce \p arms onto one driver per term, plus the OR of their pulses. At
  /// most one arm fires in a cycle (`portGraph` separates any two that could
  /// overlap), so the reduction is a one-hot select.
  SinkArm commitSink(ArrayRef<SinkArm> arms, Idle idle);

  /// One channel's port drives, accumulated over every access to it: a FIFO has
  /// a single {data,valid,ready} triple that several accesses time-share, and
  /// `hw.output` takes each port exactly once, so `finalizeStreamPorts` drives
  /// the ports after all regions have emitted.
  struct StreamDrive {
    Value valid;                  // OR of the puts' pulses
    Value ready;                  // OR of the gets' pulses
    SmallVector<SinkArm, 1> puts; // each put's pulse and the token it presents
  };
  SmallVector<StreamDrive> streamDrives; // by StreamId (sized on first use)

  /// Stores to a scattered memory, by MemId: `addr` is the element targeted, at
  /// the memory's address width, and the commit demuxes each arm onto every
  /// element. `finalizeScatteredPorts` drives an argument's element ports or
  /// builds an internal array's registers from them.
  DenseMap<unsigned, SmallVector<SinkArm, 1>> scatterWrites;

  /// A forwarded store's issue-time terms, recorded by `emitWrites` before the
  /// write-latency delays: its commit pulse, its (bank, offset) and its datum.
  struct ForwardStore {
    Value we, bank, offset, data;
  };
  DenseMap<uint64_t, ForwardStore> fwdStores; // accKey(mem, store idx)
  /// One forwarded load awaiting its stores: the RAM datum it would have read,
  /// its own issue-time (bank, offset), and the backedge its consumers hold.
  struct PendingForward {
    unsigned mem, load;
    Value raw, bank, offset;
    circt::Backedge out;
  };
  SmallVector<PendingForward, 1> pendingForwards;
  /// Resolve every pending forward: per paired store, a same-element compare at
  /// issue gated by the store's commit pulse, with the select and the datum
  /// registered to the read latency on the load's shell and muxed over the RAM
  /// datum. Runs once every region has emitted.
  void finalizeForwards();

  /// One store to an internal array, held back so the stores coloured onto the
  /// same write port can be muxed onto one `seq.write`. The colouring spreads
  /// the stores over at most two ports so block-RAM inference holds.
  struct SharedWrite {
    unsigned bank; // the bank this store commits to (0 when unbanked)
    unsigned port; // the write port it was coloured onto
    SinkArm arm;   // `fired` already delayed for the device write latency
  };
  DenseMap<unsigned, SmallVector<SharedWrite, 2>> sharedWrites; // by MemId

  /// One shared read port, keyed by (memory, bank, port). `sharedReadPort`
  /// builds the `seq.read` on the first access to reach it, so its datum is
  /// available before the address that fetches it exists; the address and the
  /// read enable ride backedges `finalizeSharedReadPorts` resolves once every
  /// holder is known. An arm's `fired` is the second of two selects: within a
  /// region `sharedAddress` has already picked between that region's own
  /// accesses.
  struct SharedReadPort {
    Value data;
    circt::Backedge rdEnBE;
    circt::Backedge addr;
    /// One arm per holder: the regions plus the mastering children.
    SmallVector<SinkArm, 1> arms;
    /// The one region holding the port, when a region (not a child) does. The
    /// finalize reads its resolved shell off `shellOf`, the chainEnable at
    /// contribution time being still a promise.
    std::optional<unsigned> ownerRegion;
  };
  /// A MapVector so the finalize's port-driving order is stable, not
  /// hash-ordered.
  llvm::MapVector<std::tuple<unsigned, unsigned, unsigned>, SharedReadPort>
      sharedReads;

  /// Stores to an external array's port group, keyed by the group's base name
  /// (a StringRef into the model's `portBase`/`topBase`) so a child mastering
  /// the same (bank, port) colour joins the accesses' arms. A MapVector for a
  /// stable port-driving order.
  llvm::MapVector<llvm::StringRef, SmallVector<SinkArm, 2>> boundaryWrites;

  /// The same for a boundary read port group's address output. A group several
  /// regions or children share is one module output, so only
  /// `finalizeSharedReadPorts` may drive it.
  llvm::MapVector<llvm::StringRef, SmallVector<SinkArm, 1>> boundaryReads;

  /// A kernel-local channel's body wires: what a boundary channel reads off its
  /// module ports, an internal one reads off its own `seq.fifo`. Backedges,
  /// since the FIFO can only be built once every access has contributed its
  /// drive.
  struct StreamWires {
    circt::Backedge data;  // the FIFO's show-ahead output
    circt::Backedge valid; // a token is available (~empty)
    circt::Backedge ready; // space is available (~full)
  };
  DenseMap<unsigned, StreamWires> streamWires; // internal channels only

  /// Body wires of a channel whose ends are child ports (`callEnds`): the
  /// producer end's `ready` and, per consumer end, its `{data, valid}`. Both
  /// halves are backedges, the child's input ports existing before the FIFO
  /// that drives them and the FIFO needing the child's outputs. One entry per
  /// consumer: several readers fan out, each owning its own FIFO.
  struct ComposedWires {
    circt::Backedge prodReady;
    llvm::SmallVector<circt::Backedge, 1> sinkData, sinkValid;
  };
  DenseMap<unsigned, ComposedWires> composedWires; // by StreamId
  /// Each instantiated child's output ports, by name. The channel realization
  /// reads these to find a producer's `{data, valid}` and a consumer's `ready`,
  /// since `emitCalls` builds the instances before the queues between them.
  DenseMap<unsigned, llvm::StringMap<Value>> callOuts; // by CallId

  // The child modules a `dcp.instance`'s CallUnit instantiates (null for
  // a plain leaf with no calls).
  const uarch::CalleeCtx &callees;

  DatapathEmitter(EmitContext &c, const uarch::Datapath &dp,
                  circt::hw::HWModulePortAccessor &pa,
                  const llvm::StringMap<Operation *> &opModules,
                  const uarch::CalleeCtx &callees)
      : c(c), dp(dp), pa(pa), opModules(opModules), callees(callees) {}

  static uint64_t accKey(unsigned m, unsigned a) {
    return (uint64_t(m) << 32) | a;
  }

  /// Resolve a datapath Source to the SSA value driving it.
  Value resolveSource(const uarch::Source &s);
  /// A shared unit's input mux: the bound ops hold disjoint MRT residues, so
  /// the `activationPulse` selects are one-hot and an AND-OR reduction serves.
  /// With no op issuing the result is zero, which no consumer samples.
  Value resolveMux(uarch::MuxId id);
  /// The window a recurrence input reads its reduction identities in: region
  /// \p rb's counter still inside its first \p dist iterations. A level, valid
  /// when the region issues, which a consumer delays to its own stage.
  Value firstIterations(const uarch::RegionBlock &rb, unsigned dist);
  /// The single iteration that reads identity \p iter: \p rb's counter at
  /// `lb + iter*step`. The same kind of level as `firstIterations`.
  Value atIteration(const uarch::RegionBlock &rb, unsigned iter);
  /// \p rb's counter and its lower bound, resized to \p w bits.
  std::pair<Value, Value> counterAndLb(const uarch::RegionBlock &rb,
                                       unsigned w);
  /// The width a `lb + n*step` gate compares at: the counter register's own,
  /// or wider where a narrowed runtime-bound counter's hull cannot absorb the
  /// offset (`RegionBlock::counterHull`).
  unsigned gateWidth(const uarch::RegionBlock &rb, unsigned n);
  /// The counter value \p rb's n-th iteration holds, `lb + n*step` at \p lb's
  /// width; null for n == 0, which is \p lb itself.
  Value ivAt(const uarch::RegionBlock &rb, unsigned n, Value lb);
  /// One cone \p r of this access's address as hardware at \p width: a
  /// constant, one register per strength-reduced term, and whatever did not
  /// reduce, evaluated.
  Value buildAddr(const uarch::MemUnit::Access &acc,
                  const uarch::MemUnit::Access::Reduced &r, unsigned width);
  /// The address hardware of an access: the element index within the bank it
  /// reaches, plus the bank digit when that is decided at runtime. The runtime
  /// dual of the static split (`dcp-resolve-banking`), routing an element to
  /// the same bank off the cones `planAddressGenerators` reduced.
  BankSplit bankAddress(const uarch::MemUnit &m,
                        const uarch::MemUnit::Access &acc);
  /// Narrow a child's port address to this memory's clog2(depth)-bit index
  /// (hlmem). `bankAddress` already returns an offset at that width.
  Value memAddr(const uarch::MemUnit &m, Value addr);
  /// Which element of a scattered memory an access names, at the memory's own
  /// address width (compared against literal element numbers, not used to
  /// index).
  Value scatterIndex(const uarch::MemUnit &m,
                     const uarch::MemUnit::Access &acc);
  /// The element registers of a scattered internal array, in element order.
  SmallVector<Value> scatterValues(unsigned id);
  /// \p v delayed to land with the datum of a read of \p m: a bank select and a
  /// constant table's own output both have to reach the consumer on the cycle
  /// the data does.
  Value atReadData(const uarch::MemUnit &m, Value v, const StallShell &sh);

  /// Bind external read-data input ports into readData (once, before regions).
  void bindReadPorts();
  /// Instantiate seq.hlmem storage for each internal (non-argument) memory.
  void createInternalMemories();
  /// Wire a region's controller output into the datapath, the G->F seam. Each
  /// field is absent where its controller publishes none, hence the
  /// field-by-field copy rather than a whole-struct assignment.
  void setControl(unsigned region, const RegionControl &rc) {
    RegionControl &slot = controlOf[region];
    if (rc.counter) {
      slot.counter = rc.counter;
      // Widen the counter to the index width a Source::Counter is read at, at
      // the counter's own signedness: an unsigned counter zero-extends, its top
      // bit being magnitude not sign.
      counterIndex[region] =
          resize(c.b, c.loc, rc.counter, kIndexWidth,
                 /*isSigned=*/!dp.regions[region].counterUnsigned);
    }
    slot.issue = rc.issue;
    if (rc.wantIssue)
      slot.wantIssue = rc.wantIssue;
    if (rc.phase)
      slot.phase = rc.phase;
    if (!rc.scaledCounters.empty())
      slot.scaledCounters = rc.scaledCounters;
  }
  /// Record a region's latched result \p port so a sibling reading
  /// Source::Survivor{region, port} resolves to it.
  void setSurvivor(unsigned region, unsigned port, Value v) {
    survivorOf[accKey(region, port)] = v;
  }
  /// Register region \p region's stall shell, the H seam. The orchestrator
  /// registers a promise (two backedges) before F and G emit against it, then
  /// re-registers the derived shell once `deriveStallShell` resolves them.
  void setShell(unsigned region, const StallShell &sh) { shellOf[region] = sh; }
  /// Region \p region's stall shell; rigid for an unregistered region. Stamped
  /// with the owning region and its pass discipline, which a primitive
  /// delaying this region's pulses reads off the shell.
  StallShell shellFor(unsigned region) const {
    StallShell sh = shellOf.lookup(region);
    sh.region = region;
    sh.singlePass = dp.regions[region].singlePass();
    return sh;
  }

  /// The part of \p rb's datapath that precedes the units, for both the leaf
  /// path (`emit`) and a container's condition cone (`emitConditionRegion`):
  /// the delay chains, the unit backedges (a read address may read a unit
  /// emitted later) and the reads, whose data the units consume.
  void emitBeforeUnits(const uarch::RegionBlock &rb, Value issue);
  /// The part that follows the units: the register heads, then the boundary
  /// read addresses, which may be computed by a unit and so need its filled
  /// value rather than its backedge.
  void emitAfterUnits(const uarch::RegionBlock &rb, Value issue);

  void emitRegisters(const uarch::RegionBlock &rb);
  /// Backedge every unit output before any consumer resolves it, so a read
  /// address or another unit input may reference a unit emitted later.
  void declareUnits(const uarch::RegionBlock &rb);
  /// Bind the datum of every read scheduled in region \p rb into `readData`,
  /// before `emitUnits` consumes it. One arm per `PortPlan`; the one it cannot
  /// serve is a boundary port group, whose address may be computed by a unit
  /// (`emitExternalReadAddrs`) and whose datum `bindReadPorts` already bound.
  void emitReads(const uarch::RegionBlock &rb, Value issue);
  /// One read port of \p m per bank on lane \p port, rather than one per bank
  /// per access. The lane's accesses (\p idxs, indexing `m.accesses`) hold
  /// distinct slots, so bank k takes the offset of whichever of them reaches it
  /// and hands its datum back to that one.
  void emitLaneReads(const uarch::MemUnit &m, unsigned port,
                     ArrayRef<unsigned> idxs, const StallShell &sh);
  /// The address one region's accesses on a port present: each drives it on its
  /// own issue cycle, held with the datapath so a read frozen by back-pressure
  /// keeps re-presenting its address. \p idxs indexes `m.accesses`, all in the
  /// region \p sh and \p issue belong to. \p fired, when given, additionally
  /// receives "one of them is presenting now", which a port another region also
  /// holds selects on; a lone region on a port drives it unconditionally.
  Value sharedAddress(const uarch::MemUnit &m, ArrayRef<unsigned> idxs,
                      Value issue, const StallShell &sh, Value *fired);
  /// Stamp an emitted `seq.read`/`seq.write` with the physical port it drives
  /// (`kMemPortAttr`), which puts a port's read and write in one `always`
  /// block and so makes them one port of a dual-port RAM.
  template <typename OpT> OpT atPort(OpT op, unsigned port) {
    op->setAttr(kMemPortAttr, c.b.getI64IntegerAttr(port));
    return op;
  }
  /// Drive the read-address port of each single-interface external port group
  /// in region \p rb (unbanked or statically banked); the data-dependent ones
  /// are `emitReads`. Runs after the units, so an address computed by
  /// one resolves to its filled value rather than a dangling backedge.
  void emitExternalReadAddrs(const uarch::RegionBlock &rb, Value issue);
  /// Region \p rb's compute units: native -> comb, IP -> an instance of the
  /// extern operator module. A loop-carried input re-injects `inputInits[k][n]`
  /// on the n-th iteration; a container's own units carry none
  /// (`assertModelInvariants`).
  void emitUnits(const uarch::RegionBlock &rb);
  /// Emit a sequential (check/run) while's condition cone: the container's own
  /// condition memory reads plus its compute. Returns the settled condition
  /// with its ready latency `t_cond`, the cycles after check-start at which it
  /// is valid (0 for a combinational condition). The read address is the frozen
  /// iter-arg survivor, so the loaded value is a stable wire across the CHECK
  /// window; the caller samples it at `delayValid(checkStart, t_cond)`.
  std::pair<Value, unsigned> emitConditionRegion(const uarch::RegionBlock &rb,
                                                 const uarch::Source &condSrc);
  void resolveRegHeads(const uarch::RegionBlock &rb);
  /// Every write scheduled in region \p rb, gated by \p issue, folding the
  /// deepest store's stage into \p fb.storeDrain. One arm per `PortPlan`, as
  /// `emitReads`.
  void emitWrites(const uarch::RegionBlock &rb, Value issue,
                  DatapathFeedback &fb);
  /// One write port of \p m per bank on one lane. Bank k takes the address and
  /// data of whichever of the lane's accesses (\p idxs, indexing `m.accesses`)
  /// reaches it, and its write-enable is the OR of their demuxed enables, so an
  /// access commits on its own bank and nowhere else. At most one arm of that
  /// OR is live. \p commit builds the store pulse lazily, called only where a
  /// store exists.
  void emitLaneWrites(const uarch::MemUnit &m, ArrayRef<unsigned> idxs,
                      llvm::function_ref<Value()> commit, const StallShell &sh);

  /// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
  /// `hw.instance` and fold the child's `done` into \p fb.callDone. Runs before
  /// the region's own register heads and accesses, since a call's scalar result
  /// is an ordinary datapath Source a register chain or a store may read.
  void emitCalls(const uarch::RegionBlock &rb, Value issue,
                 DatapathFeedback &fb);
  /// Child \p cu's instance inputs by child port name: clk/rst/`start`, each
  /// read's data input, each channel end and each scalar operand. An internal
  /// read consumes a backedge, handed back through \p rdBackedge and resolved
  /// after the instance; a boundary read passes the top's data input straight
  /// through.
  llvm::StringMap<Value>
  childInstanceInputs(const uarch::CallUnit &cu, Value startK,
                      llvm::StringMap<circt::Backedge> &rdBackedge);
  /// Which consumer end of channel \p ch the (\p call, \p arg) stream operand
  /// is: the index of its `{data, valid}` pair and of its own FIFO in the
  /// fan-out tee, counting the input ends only.
  unsigned consumerSlot(const uarch::StreamChannel &ch, uarch::CallId call,
                        unsigned arg) const;
  /// Master each memref operand of child \p cu from its instance outputs
  /// \p outs. One arm per `PortPlan`, as `emitReads` and `emitWrites`.
  /// \p rdBackedge holds the read-data promise each of the child's read ports
  /// waits on; \p runWindow is the window the child owns a port a second
  /// accessor also holds, built on demand.
  void masterCallPorts(const uarch::CallUnit &cu, llvm::StringMap<Value> &outs,
                       llvm::StringMap<circt::Backedge> &rdBackedge,
                       llvm::function_ref<Value()> runWindow,
                       const StallShell &sh);
  /// The start pulse of one child, from the start-policy table read on this
  /// node's contract and its region's composition class.
  Value startForCall(const uarch::CallUnit &cu, Value issue,
                     llvm::ArrayRef<Value> predDones, bool concurrent,
                     const StallShell &sh);
  /// The queue(s) behind a channel whose ends are child ports: one `seq.fifo`
  /// per consumer end (the fan-out tee), each optionally fronted by a seeded
  /// channel's init-prepend shim, and a pass-through where one end is a
  /// boundary port of this module rather than a child.
  void emitComposedChannel(const uarch::StreamChannel &s);
  /// What a seeded channel's shim presents to its consumer, replacing the
  /// FIFO's own triple.
  struct ShimEnd {
    Value data, valid, rdEn;
  };
  /// The init-prepend shim in front of consumer \p k of seeded channel \p s,
  /// one of \p nSinks. `rem` counts the initial tokens still to serve,
  /// nInit down to 1, and the datum is a one-hot select on the running index
  /// idx = nInit - rem.
  ShimEnd initPrependShim(const uarch::StreamChannel &s, unsigned k,
                          unsigned nSinks, Value out, Value notEmpty,
                          Value cReady, Value rdEn);

  /// Declare each kernel-local channel's body wires (`streamWires`) before any
  /// region reads them; `finalizeStreamPorts` builds the FIFO that resolves
  /// them.
  void declareInternalChannels();
  /// One channel's three handshake signals, wherever they live: a boundary
  /// channel's module ports, or a kernel-local channel's own FIFO.
  Value streamData(const uarch::StreamChannel &s);
  Value streamValid(const uarch::StreamChannel &s);
  Value streamReady(const uarch::StreamChannel &s);

  /// Bind each input stream's `_data` module port into `streamReadData` (once,
  /// before any consumer), so a Source::Stream resolves like a memory read.
  void bindStreamReads(const uarch::RegionBlock &rb);
  /// H for one region: wire region \p rb's stream handshakes and return the
  /// stall shell they derive. An input contributes its `_ready` (gated so a
  /// full output holds intake too), an output its `_data` plus `_valid`; the
  /// region's stalls become `{chainEnable, issueEnable}`, split by whether the
  /// blocked handshake belongs to an in-flight iteration or to the pass about
  /// to issue, and each put's stage folds into \p fb.storeDrain. Runs on the
  /// already-emitted (F, G) pair, timing its own deeper pulses against the
  /// region's registered promise; the caller resolves that promise with the
  /// result.
  StallShell deriveStallShell(const uarch::RegionBlock &rb, Value issue,
                              DatapathFeedback &fb);
  /// Drive every boundary channel's module ports, and build every local
  /// channel's `seq.fifo`, from the accumulated `streamDrives`. Call exactly
  /// once, after all regions have emitted and before `hw.output`.
  void finalizeStreamPorts();
  /// Drive each scattered argument's per-element data + write-enable outputs
  /// from the accumulated `scatterWrites`. Call exactly once, after all regions
  /// have emitted and before `hw.output`; a read-only scattered argument has no
  /// output port and drives nothing.
  void finalizeScatteredPorts();
  void finalizeSharedWritePorts();
  /// The `seq.read` of \p m's bank \p bank on read port \p port, built on the
  /// first access to reach it and reused by every later one. Its address is a
  /// backedge: the accesses sharing a port drive it on their own cycles, so it
  /// is only known once they have all emitted.
  Value sharedReadPort(const uarch::MemUnit &m, unsigned bank, unsigned port);
  /// Drive each shared read port's one address bus from the arms the regions
  /// holding it contributed, and each shared boundary read group's address
  /// output likewise. Call exactly once, with the same timing as the write
  /// finalizes.
  void finalizeSharedReadPorts();
  /// Whether read port \p port of \p m's bank \p bank is held by more than one
  /// accessor: a region whose own accesses reach it, or a child that masters
  /// it. Counts holders rather than regions, the two kinds signalling that they
  /// drive in different ways.
  bool portHasSeveralHolders(const uarch::MemUnit &m, unsigned bank,
                             unsigned port) const;
  /// Drive each merged boundary write port group from the stores coloured onto
  /// it. Call exactly once, with the same timing as the two above.
  void finalizeBoundaryWritePorts();
  /// Build one kernel-local channel's `seq.fifo` from its accumulated drives
  /// (\p data is the puts' muxed token) and resolve its `streamWires`.
  void emitInternalChannel(const uarch::StreamChannel &s, Value data);

  /// Emit region \p rb's whole datapath (F) given the controller's \p issue;
  /// returns its store feedback. Times everything against the region's
  /// registered shell; deriving that shell (H) is the orchestrator's separate
  /// step, run on what this emits.
  DatapathFeedback emit(const uarch::RegionBlock &rb, Value issue);
};

//===----------------------------------------------------------------------===//
// HWEmitter: the orchestrator. Owns the context + both emitters and drives the
// region tree (sibling hand-off, container nesting), wiring the typed seam.
//===----------------------------------------------------------------------===//
struct HWEmitter {
  EmitContext ctx;
  ControlEmitter control;
  DatapathEmitter datapath;
  const uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;

  /// Each emitted region's completion pulse (its done latch's set input), high
  /// in the last commit cycle. Absent for a region completing on a child
  /// call's done, which is a level with no usable pulse.
  llvm::DenseMap<unsigned, Value> donePulse;
  /// A leaf result's wire in its own capture cycle, keyed
  /// accKey(region, port), for a consumer sampling before the survivor
  /// register settles.
  llvm::DenseMap<uint64_t, Value> liveResult;
  /// A chained container's iter-arg latch D wires, keyed accKey(region, port):
  /// what the register holds at the end of the current cycle. Its first child
  /// launches in the latch cycle itself and reads the D wire in place of the
  /// stale output.
  llvm::DenseMap<uint64_t, Value> throughValue;
  /// A guard survivor's captured datum, keyed accKey(region, port). The
  /// survivor latches on the guard's completion pulse (`donePulse`) itself, so
  /// a container sampling in that cycle rebuilds the latch's D wire from the
  /// datum and that pulse instead of reading the stale register.
  llvm::DenseMap<uint64_t, Value> guardCapture;

  HWEmitter(OpBuilder &b, Location loc, const uarch::Datapath &dp,
            circt::hw::HWModulePortAccessor &pa,
            const llvm::StringMap<Operation *> &opModules,
            circt::BackedgeBuilder &bb, Type i1, Type i32,
            const uarch::CalleeCtx &callees)
      : ctx(b, loc, bb, i1, i32), control(ctx),
        datapath(ctx, dp, pa, opModules, callees), dp(dp), pa(pa) {
    ctx.countedDelayCycles = dp.countedDelayCycles;
  }

  /// The counted terminator of region \p rb: each bound resolved from its
  /// runtime Source (a dynamic trip) or the constant fast path. Empty for an
  /// acyclic region; a while builds its own Terminator::conditional.
  Terminator terminatorOf(const uarch::RegionBlock &rb);
  /// Emit one region and return its `done`. A leaf runs one imperative path for
  /// every regime (counted / dynamic-trip / while): control -> datapath ->
  /// resolve the F->G condition, capture results, done. A container runs its
  /// children once per outer iteration. With \p levelUnused the caller reads
  /// only `donePulse`, so the completion latch is skipped and the return is
  /// null; a region completing on a call's done has no pulse and keeps its
  /// level regardless.
  Value emitRegion(const uarch::RegionBlock &rb, Value start, bool retrig,
                   bool levelUnused);
  /// A loop-over-call region: a counted `dcp.pipeline` wrapping one
  /// `dcp.instance`. One child instance is fired \p tripCount times, a counter
  /// driving its index and each invocation advancing on the child's real
  /// `done`, so throughput is one iteration per child latency rather than the
  /// pipeline cadence.
  Value emitLoopCall(const uarch::RegionBlock &rb, Value start);
  /// The final iteration's issue pulse: a counted region's last iteration
  /// (counter+1 reaches the bound), a while's condition-false exit, or the
  /// issue pulse itself for an acyclic region. The pulse `emitDone` and
  /// `captureResults` both key off.
  Value lastIssuePulse(const RegionControl &rc, const Terminator &term);
  /// Capture leaf region \p rb's results into the survivor registers a sibling
  /// reads, each at its own ready cycle relative to \p captureOn; returns the
  /// region's result-drain stage. \p captureOn is the last iteration's issue
  /// pulse for a counted loop and each continuing iteration's for a while.
  unsigned captureResults(const uarch::RegionBlock &rb, Value captureOn,
                          Value start, Value phase);
  /// Run \p regions in program order, each starting when its predecessor drains
  /// (the first on \p start); returns the last region's drain pulse. The shared
  /// sequencer for a container's children and a guard's arms. No done level is
  /// wanted anywhere along the chain, so no region in it builds one. With \p
  /// tailOnPulse the tail's drain is its completion pulse itself, which only a
  /// caller that has checked the tail's state has settled may ask for.
  Value sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                 bool retrig, bool tailOnPulse = false);
  /// The cycle a successor of region \p rid may start in: its completion pulse
  /// where `handoffSafe` allows the same cycle, that pulse registered
  /// otherwise, and the rising edge of \p done for a region that recorded no
  /// pulse and so kept its level.
  Value drainPulse(uarch::RegionId rid, Value done);
  /// Whether a successor may start on \p rb's completion pulse rather than on
  /// its latched done. True for a conditional region, whose exit trails every
  /// commit and every iter-arg latch by at least a cycle, and for a
  /// result-less guard, which has no survivor latching on the pulse. A counted
  /// region's pulse is its commit cycle, so it keeps the latch.
  bool handoffSafe(const uarch::RegionBlock &rb) const {
    return rb.conditional ||
           (rb.shape == uarch::RegionBlock::Shape::Guard && rb.results.empty());
  }
  /// Region \p rid's completion pulse where a successor may start on it, null
  /// where the region keeps its latch.
  Value handoffPulse(uarch::RegionId rid) const {
    return handoffSafe(dp.regions[rid]) ? donePulse.lookup(rid) : Value();
  }
  /// Whether container \p rb's advance may ride its last child's completion
  /// pulse, one cycle ahead of the latched done edge. Requires that no exact
  /// span composes through this container and that the last child's results
  /// are settled or handed live at the pulse.
  bool advancesOnPulse(const uarch::RegionBlock &rb) const;
  /// Whether \p rb also relaunches in that same cycle (advance = launch,
  /// boundary {0,0}). Requires the first child to sample the counter and
  /// iter-args no earlier than one cycle after launch, which a conditional
  /// child and a guard child both guarantee. Asked only of a counted container
  /// that already advances on the pulse.
  bool chainsTurnover(const uarch::RegionBlock &rb) const;
  /// The next-value wire container \p rb's iter-arg \p k latches on advance.
  /// Normally the producing child's survivor register. With \p onPulse the
  /// advance falls in the last child's own commit cycle, where a survivor
  /// captured in that cycle still reads stale and the result's live wire
  /// stands in; every other source is settled a cycle earlier and stays the
  /// survivor.
  Value nextValueFor(const uarch::RegionBlock &rb, unsigned k, bool onPulse);
  /// Sequence container \p rb's children from \p issue and resolve the body's
  /// promises: \p lastDrain takes the last child's completion pulse when the
  /// advance rides it (\p onPulse), its done edge otherwise, and each iter-arg
  /// backedge in \p nextBE its next value.
  void resolveIterationBody(const uarch::RegionBlock &rb, Value issue,
                            circt::Backedge &lastDrain,
                            llvm::MutableArrayRef<circt::Backedge> nextBE,
                            bool onPulse);
  /// Compose the func-scope sibling regions by their dependence DAG
  /// (`rb.predecessors`): a predecessor-free region starts with the kernel
  /// \p start (independent siblings run concurrently), the rest on the rising
  /// edge of their predecessors' joined `done`. The returned kernel `done` is
  /// the conjunction of every region's, so it completes when the last does.
  Value composeSiblings(llvm::ArrayRef<uarch::RegionId> regions, Value start);
  /// Set up a container's loop-carried iter-args as frozen survivor registers
  /// (latch each `rb.results[k].init` at \p start, advance on \p advance),
  /// record each as Source::Survivor{rb, k}, and return the per-arg next-value
  /// backedges, set after the children emit since the next value comes from
  /// them. With \p publishThrough each latch's D wire is recorded in
  /// `throughValue`, for a first child that launches in the same cycle these
  /// registers latch and must read what the register is becoming.
  llvm::SmallVector<circt::Backedge>
  setupCarriedIterArgs(const uarch::RegionBlock &rb, Value start, Value advance,
                       bool publishThrough);
  /// A counted container: wire `emitCountedIteration` to a body that sequences
  /// its children, so the outer counter advances when the last child drains. A
  /// cross-region result crosses child-to-child as a survivor register.
  Value emitContainer(const uarch::RegionBlock &rb, Value start,
                      bool levelUnused);
  /// A conditional container (a sequential-wrapper while): the same
  /// per-iteration child sequencing as emitContainer, but the outer iter-args
  /// are frozen survivor registers advanced by the children's results and the
  /// loop terminates on a continue-condition re-evaluated over them, so the
  /// controller is `emitCheckedIteration` rather than the counted one.
  Value emitConditionalContainer(const uarch::RegionBlock &rb, Value start,
                                 bool levelUnused);
  /// A guard region (a dcp.select): a predicated container whose children run
  /// once iff the held predicate (`rb.condition`) holds, else are skipped. The
  /// predicate start-gates child 0 (`start & cond`); a false predicate
  /// completes the region in one cycle (`start & ~cond`) without ever issuing
  /// the children, so their stores never fire.
  Value emitGuard(const uarch::RegionBlock &rb, Value start, bool levelUnused);
  /// Emit the whole module body: preamble + each top-level region in order.
  void emit();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_HWEMIT_H

/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_DATAPATH_H
#define ALLO_MICROARCH_DATAPATH_H

#include "allo/IR/AlloAttrs.h"           // StallContractEnum, DeterminacyEnum
#include "allo/IR/AlloOps.h"             // dcp::DCPathModuleOp
#include "allo/Scheduling/MemoryModel.h" // BankLayout
#include "allo/Scheduling/OperatorIdentity.h" // what one unit realizes
#include "allo/Scheduling/RegionGraph.h"      // RegionShape

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h" // function_ref
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace circt::hw {
class HWModuleOp;
} // namespace circt::hw
namespace mlir::allo::iface {
struct ModuleInterface;
} // namespace mlir::allo::iface
namespace mlir::allo {
struct DeviceModel;
class OperatorLibrary;
} // namespace mlir::allo

namespace mlir::allo::uarch {

struct BindingPolicy;

/// The already-emitted callees a `dcp.instance` lowers against: the child
/// `hw.module`s to instantiate plus their port models. Filled bottom-up by the
/// emit driver, so a callee is present before its caller.
struct CalleeCtx {
  const llvm::StringMap<circt::hw::HWModuleOp> &modules;
  const llvm::StringMap<iface::ModuleInterface> &ifaces;
};

//===----------------------------------------------------------------------===//
// Identifiers. Cells are referenced by ids indexing the Datapath's vectors
// rather than by pointers, so the model stays trivially copyable across a
// rebind.
//===----------------------------------------------------------------------===//

using UnitId = unsigned;
using RegId = unsigned;
using MemId = unsigned;
using MuxId = unsigned;
using IOId = unsigned;
using ConstId = unsigned;
using RegionId = unsigned;
using StreamId = unsigned;
using CallId = unsigned;

//===----------------------------------------------------------------------===//
// A resolved driver of one input port. Exactly one Source feeds each input, so
// a mux forced by sharing is its own cell whose output is the Source.
//
// A shared unit's output carries a different bound op's result in each issue
// cycle, so `outPort` says which one. Its meaning is per kind:
//   Unit    -> which bound op's result this is (index into `boundOps`; 0 under
//              the trivial allocation, where a unit has exactly one)
//   Reg     -> tap level to read (0 = chain head, i.e. the newest sample)
//   Mem     -> index of the read access whose loaded data this is
//   Mux      -> 0
//   IO       -> 0
//   Const    -> 0
//   Counter  -> 0 (id = the RegionBlock whose iteration counter this is)
//   Survivor -> which result of the producing region (id = the RegionBlock),
//               latched when that region completes
//   Stream   -> index of the get access whose loaded token this is
//               (id = the StreamChannel)
//   Call     -> which scalar result of a sub-kernel call (id = the CallUnit),
//               landing at start+latency
//===----------------------------------------------------------------------===//

struct Source {
  enum class Kind {
    None,
    Unit,
    Reg,
    Mem,
    Mux,
    IO,
    Const,
    Counter,
    Survivor,
    Stream,
    Call
  };
  Kind kind = Kind::None;
  unsigned id = 0;
  unsigned outPort = 0;
  /// A resolvable (non-None) source.
  explicit operator bool() const { return kind != Kind::None; }
};

//===----------------------------------------------------------------------===//
// Structural cells.
//===----------------------------------------------------------------------===//

/// A functional-unit instance (adder, multiplier, floating-point core, ...).
/// In the trivial binding every compute op gets its own unit, so `boundOps`
/// holds a single entry and no input needs a mux.
struct FuncUnit {
  UnitId id = 0;
  // What this unit realizes; two units fold only if their identities are equal.
  // Carries the realization, the result type, and the fields the RTL module
  // name is spelled from.
  OperatorIdentity identity;
  unsigned latency = 0;  // result available `latency` cycles after issue
  bool pipelined = true; // accepts a new input every cycle
  /// The delay this unit's inputs must settle within, in ns, from the
  /// `in_delay` stamped beside `z`. Marginal for a combinational unit, whose
  /// `z` already carries the register floor. Zero for an op that renames bits
  /// rather than computing them.
  double inDelay = 0.0;
  // The IP's port/back-pressure contract (from its `dcp.operator`), unused for
  // a combinational unit. Clock-enable is the only one the emitter builds.
  StallContractEnum stall = StallContractEnum::Ce;

  /// One op bound here, and when it issues. `stage` is what a consumer's pulse
  /// is delayed by, `residue` what the reservation table contends on; they
  /// differ in a cyclic region at II > 1.
  struct BoundOp {
    Operation *op = nullptr;
    unsigned stage = 0;   // schedule cycle within its region
    unsigned residue = 0; // `stage % ii` when cyclic, else `stage`
    /// The sub-cycle start the solve proved for this op, in ns. Empty where no
    /// solve placed the cell, marking it unpriced rather than zero.
    std::optional<double> z;
  };
  // Ops bound here; sharing puts several non-conflicting ops in the list.
  // Never empty.
  llvm::SmallVector<BoundOp, 1> boundOps;

  /// The representative bound op, whose operands shape the unit's input ports
  /// and whose location names it. Every site must pick the same one.
  Operation *repOp() const {
    assert(!boundOps.empty() &&
           "a unit with no bound op has no representative");
    return boundOps.front().op;
  }

  // One resolved driver per input operand port (post-binding). A fused
  // recurrence (II == latency, depth II-L == 0) has a self-referential input
  // (`Source::Unit{this.id}`): the IP's own pipeline is the accumulator.
  llvm::SmallVector<Source, 2> inputs;

  // Per-input reduction identities (parallel to `inputs`), one per iteration of
  // the recurrence distance: port k reads `inputInits[k][n]` at iteration n and
  // takes `inputs[k]` from iteration `inputInits[k].size()` on. Empty for a
  // non-recurrence input, and on a shared port, whose identities are arms of
  // the input mux instead (`Mux::Phase`).
  llvm::SmallVector<llvm::SmallVector<Source, 1>, 2> inputInits;
};

/// A shift-register chain carrying one SSA value across cycle boundaries. Its
/// length is the largest delay any consumer needs; consumers read at their own
/// `tap` (see Source).
struct Register {
  RegId id = 0;
  Value value; // the L0 value being held; also its width and name at emit
  Type type;
  unsigned depth = 0; // chain length in cycles (>= 1 for a real register)
  Source input;       // driver of the chain head (the producing cell output)
  /// The cycle within the producing iteration at which `input` carries a fresh
  /// datum (`readyCycleOf` the producer, 0 for a held source): the phase an
  /// II-folded chain captures on. Not re-derivable from `input`, whose shared
  /// unit names a representative op.
  unsigned ready = 0;
  /// The depths consumers read (each `Source::Reg`'s `outPort`), sorted
  /// ascending, zero excluded; the deepest equals `depth`. Shift-register
  /// extraction breaks at every tap, so the ledger charges one SRL run per
  /// maximal inter-tap segment.
  llvm::SmallVector<unsigned, 2> taps;
};

/// How an access reaches its memory, a separate question from what the memory
/// is (`MemUnit::Realization`). Decided once per access by `planAccessPorts`
/// and dispatched on by both emit paths. A port a child masters
/// (`CallUnit::MemArg`) plans like an access.
enum class PortPlan {
  /// No address port at all: one cell (or one boundary port) per element, and
  /// an access picks its element by comparing the index. Serves any number of
  /// accesses at once.
  ElementWise,
  /// A combinational constant table, indexed and then registered to the read
  /// latency the schedule timed the access at. No port to contend for.
  Table,
  /// One address bus per colour, carrying every access `bindMemoryPorts` proved
  /// never issues with the others on it, each driving it on its own activation.
  Coloured,
  /// A skewed array's lane: one port per bank, taken by whichever of the lane's
  /// accesses the rotation sends there.
  Lane,
  /// A bank decided at run time: a port on every bank, the datum selected by
  /// the bank digit aligned with it.
  Crossbar,
};

/// A memref-backed memory with banks and ports. The memory model resolves which
/// `dcp.storage` realizes it; this records the name, leaving physical selection
/// (address decode, per-primitive ports) to lowering.
struct MemUnit {
  MemId id = 0;
  Value memref;
  bool external = false;   // a func-argument memref (bare interface, no AXI)
  unsigned width = 0;      // element width in bits
  unsigned depthWords = 0; // elements per bank
  // The memref's `allo.part` decomposition: which axes are partitioned, by what
  // factor and kind, and the per-bank shape. An unpartitioned memref decodes to
  // a single bank whose `bankShape` is the full shape.
  BankLayout layout;
  unsigned numBanks = 1; // == layout.numBanks (1 = unbanked or registers)
  /// The banks are skewed and every access resolved an `Access::slot`, so they
  /// are read through lane-shared ports instead of routed. False on a skewed
  /// layout whose slots `assign-banks` declined to assign, which crossbars like
  /// any other. An access's index meaning comes from the layout's `skew()`.
  bool skewed = false;
  /// Held as one cell per element rather than an addressed interface, which
  /// `layout.registers` (a complete partition) asks for: unlimited
  /// combinational ports, where an addressed port serves one element per cycle.
  /// A top-level argument's cells are the caller's and arrive as one port per
  /// element
  /// (`elemPorts`); an internal array's are registers here, read by a
  /// combinational select. A callee's array argument is neither: the storage is
  /// the parent's and the child masters an ordinary addressed port on it.
  bool scattered = false;
  /// Whether `bindMemoryPorts` split the writes across ports, done only where
  /// two enabled in one cycle provably address different words. A consumer may
  /// then place each port in its own `always` block; false puts them all in
  /// one, whose priority order resolves the collision.
  bool writesIndependent = false;
  /// The module ports holding one element of a `scattered` argument: the input
  /// it arrives on, and the output plus write-enable it leaves on. An unused
  /// direction has no port, and the live directions decide the names: `A_k` for
  /// one, `A_k_in` / `A_k_out` for both.
  struct ElemPort {
    std::string in, out, we;
  };
  /// One per element, flat row-major, when `scattered`; composed by
  /// `enumerateBoundaryPorts`. Empty for every other memory.
  llvm::SmallVector<ElemPort> elemPorts;
  std::string storage; // resolved `dcp.storage` realization
  /// The vendor attribute pinning the array to `storage`, from that row. Empty
  /// where the row declares none. Stamped on every `Ram` this module declares.
  std::string ramStyle;

  /// Ports one instance of `storage` provides, from its `dcp.storage` row
  /// narrowed by the topology the array asked for; the budget the scheduler
  /// reserved against. Not a count of what a bank is built with: `bindPorts`
  /// colours by what may share an address bus, so `readPortsBuilt` may be many
  /// times it.
  StoragePorts ports;
  /// Ports one bank is built with: the distinct ports `bindMemoryPorts`
  /// assigned to the accesses reaching a bank, maximized over the banks. The
  /// third is not the sum of the first two, a pooled-storage port may carry
  /// both a read and a write. A skewed bank serves a whole lane from one port.
  /// Zero for a scattered memory and a ROM, neither addressed.
  unsigned readPortsBuilt = 0, writePortsBuilt = 0, portsBuilt = 0;

  /// Accesses of one bank the model cannot separate in time, per direction: a
  /// lower bound on what the schedule asks of this array in a cycle, against
  /// which the ports above are what was built. The two differ where the binding
  /// separates a pair the schedule never issues together; `portConcurrency`
  /// bounds them. Zero for a ROM or scattered array, neither addressed.
  unsigned readConcurrency = 0, writeConcurrency = 0;
  /// Module interface groups this array contributes: one per bound boundary
  /// port, one per group a child masters on it, or one per element of a
  /// scattered argument. Zero for an internal array.
  unsigned boundaryPorts = 0;

  /// Whether `storage` can hold \p writes write ports and \p total ports in
  /// all, over as many copies as that takes. Only the writes decide it, every
  /// copy needing them; a further read is a further copy. Takes a candidate
  /// rather than the committed counts.
  bool fitsStorage(unsigned writes, unsigned total) const {
    return ports.holds(writes, total);
  }
  /// The same of the binding this memory committed to.
  bool fitsStorage() const { return fitsStorage(writePortsBuilt, portsBuilt); }

  /// Instances of `storage` this bank is held in, decided by `bindMemoryPorts`.
  /// Reads past one instance's are served by another copy of the whole array,
  /// each copy taking every write; a write cannot be served that way, so the
  /// two directions are not symmetric.
  unsigned instances = 1;

  /// Which instance of a bank serves each read port, by `instanceKey`. A port
  /// index is a colour rather than a dense read index. Empty while `instances`
  /// is 1.
  llvm::DenseMap<uint64_t, unsigned> readInstance;
  static uint64_t instanceKey(unsigned bank, unsigned port) {
    return (uint64_t(bank) << 32) | port;
  }

  /// What this module builds to hold the array, derived from the fields above
  /// rather than stored.
  enum class Realization {
    /// Nothing: the cells are the caller's and this module holds ports on
    /// them. Every argument array, addressed or scattered.
    Boundary,
    /// A combinational constant table (`hw.aggregate_constant`), unlimited-port
    /// and with no storage cost.
    Rom,
    /// One register per element, selected by comparison rather than addressed.
    Scatter,
    /// An addressed array, held in as many copies of its `storage` row as its
    /// read ports take. The only realization an internal addressed array has; a
    /// binding that overruns the row is reported rather than realized
    /// otherwise.
    Ram,
  };
  /// An argument is a boundary whatever shape its cells have upstream: this
  /// module builds none, only holding ports on them.
  Realization realization() const {
    return external    ? Realization::Boundary
           : isRom     ? Realization::Rom
           : scattered ? Realization::Scatter
                       : Realization::Ram;
  }

  // Access latency of `storage`, the numbers the scheduler stamped onto this
  // memref's `dcp.load`/`dcp.store`. The consumer's register depth was solved
  // as `tY - (start + readLatency)`, so ports must be built at exactly these.
  unsigned readLatency = 0;
  unsigned writeLatency = 1;

  // `romInit` is the `initial_value` (a DenseElementsAttr) of the
  // `memref.global` this memref reads through, when it has one. `isRom` is the
  // narrower realizable property, initialized and never written, becoming a
  // combinational `hw.aggregate_constant` with no writable hlmem. Read-only is
  // a property of the use: a mutable global with a power-on value
  // (`allo.lang.Stateful`) has `romInit` and is not a ROM.
  bool isRom = false;
  Attribute romInit;

  /// One bound access. A read's loaded data is referenced by
  /// Source{Mem, id, <index of this access>}; a write consumes `data`.
  struct Access {
    Operation *op = nullptr;
    bool isWrite = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned stage = 0;  // scheduled cycle within the region
    /// The setup delay the schedule priced this access against, in ns
    /// (`accessCharacterization`: the address cone plus the port's own delay).
    /// `portDelay` is the port's own share alone, bounding a select in front of
    /// a registered address.
    double inDelay = 0.0;
    double portDelay = 0.0;
    /// The port of its bank this access drives, assigned by `bindMemoryPorts`.
    /// Two accesses share a port only where the model proves they never issue
    /// in the same cycle, so it carries a select rather than an arbiter. Under
    /// a skewed layout it is the access's `lane`.
    unsigned port = 0;
    /// How this access reaches the storage, and so which emit path it takes.
    PortPlan plan = PortPlan::Coloured;
    /// This access's slot in the module's boundary port list: an index into
    /// `Datapath::readPorts` or `writePorts` by `isWrite`, its port identity at
    /// the boundary. `kNoPort` for an access to an internal memory.
    static constexpr unsigned kNoPort = ~0u;
    unsigned portIdx = kNoPort;
    /// The boundary port group's base name (`A_rd0`), which every field port is
    /// composed from (`A_rd0_addr`); a data-dependent banked access suffixes a
    /// bank as well (`A_rd0_b2`, see `extPorts`). Part of the C++/Python
    /// manifest contract. Empty for an internal memory's access.
    std::string portBase;
    /// Which bank this access routes to, when its memref is partitioned
    /// (`numBanks > 1`): the index `assign-banks` assigned it, or empty to
    /// crossbar over all `numBanks` banks. 0 for an unbanked memref;
    /// `externalBank` pairs it with the bank count. Defaults to crossbar rather
    /// than bank 0. Always empty under a skewed layout, where no compile-time
    /// bank exists and `assign-banks` resolved a `slot`.
    std::optional<unsigned> staticBank;
    /// The compile-time half of a skewed access's bank, which is
    /// `(cls + slot) mod numBanks` with `cls` a runtime value the array's
    /// accesses share: two distinct slots are distinct banks at every rotation
    /// while neither names one. Empty off a skewed layout, and where
    /// `assign-banks` resolved nothing.
    std::optional<unsigned> slot;
    /// Which of a skewed memory's parallel port sets this access uses.
    /// Accesses in one lane hold distinct slots, so they reach distinct banks
    /// and share one port per bank; two accesses of the same slot collide and
    /// must land in different lanes. Always 0 off the skewed path, where
    /// `bindMemoryPorts` also copies it into `port`: a lane-shared access's
    /// port index is its lane.
    unsigned lane = 0;
    AffineMap addrMap; // index map over `addr` operands (identity when the
                       // subscript was not affine)
    llvm::SmallVector<Source, 2> addr; // address operand drivers (delayed IVs)
    Source data;                       // write data driver (writes only)
    /// One strength-reduced term of the address: a scaled counter its region
    /// carries (`RegionBlock::addrStrides`).
    struct ScaledTerm {
      unsigned region;
      unsigned slot; // index into that RegionBlock's `addrStrides`
    };
    /// One address cone after strength reduction: `base`, plus one register per
    /// term (a scaled counter, or a digit of one, that the controller advances
    /// instead of rebuilding arithmetic every cycle), plus `residual` evaluated
    /// over `addr`. Partial by design, a term reducing or not on its own, so
    /// with nothing reduced `terms` is empty and `residual` holds it all.
    struct Reduced {
      llvm::SmallVector<ScaledTerm, 3> terms;
      /// The expression's constant, zero whenever a term exists: the first term
      /// that does not wrap absorbs it (`AddrStride::init`) rather than an
      /// adder carrying it.
      int64_t base = 0;
      AffineExpr residual; // null when the whole expression reduced
      /// Registers the residual reads (`SplitAddress::reads`), in the order it
      /// names them. Appended to the operand list `buildAddr` evaluates the
      /// residual over, so they land on the symbol positions it named.
      llvm::SmallVector<ScaledTerm, 2> reads;
    };
    /// The element index within the bank, and the bank digit when one is
    /// decoded at run time: two cones off the same operands (`addressExprsOf`).
    /// A bank digit is `(counter floordiv D) mod F`, which is a register as
    /// much as a row stride is.
    Reduced offset;
    Reduced bank;
    /// Whether a digit is decoded at all, so the emitter builds `bank` or
    /// leaves it unwired. Not derivable from `bank`: a cone that reduced to the
    /// constant 0 is indistinguishable from no cone.
    bool hasBankCone = false;
    /// How many cycles late this access needs the scaled counters, the delay
    /// its counter operands would otherwise be tapped at. The counters run
    /// live, so their sum is delayed once rather than each operand separately.
    /// The residual's operands arrive already delayed and are not covered.
    unsigned addrDelay = 0;
    /// What the address costs on this access's setup path, in ns, `portDelay`
    /// excluded. Without a delay register that is the whole cone (`inDelay`
    /// less the port's own share); with one it is only what `buildAddr` builds
    /// after that register, the term sum having landed in it a cycle earlier.
    double addrSetup = 0.0;
  };
  llvm::SmallVector<Access, 2> accesses;

  /// One store->load forwarding pair: the two accesses may issue in the same
  /// cycle, where the RAM would still return the old word. The emitter
  /// compares their addresses at issue and muxes the store's datum into the
  /// load's data out, delayed to the read latency.
  struct Forward {
    unsigned load = 0, store = 0; // indices into `accesses`
  };
  llvm::SmallVector<Forward, 1> forwards;
};

/// Which of \p acc's address operands a residual still reads, by position in
/// `Access::addr`. The reduction folded every other operand into a scaled
/// counter, so nothing resolves its slot and the slot stays empty.
llvm::BitVector residualReads(const MemUnit::Access &acc);

/// One bound access, referenced as (owning cell id, access index): a memory
/// access is `dp.mems[id].accesses[idx]`, a stream access
/// `dp.streams[id].accesses[idx]`.
struct AccRef {
  unsigned id, idx;
};

/// A sub-kernel call as a multi-cycle datapath node, built from a
/// `dcp.instance` and owned by the `RegionBlock` it sits in. The child instance
/// masters the memory ports of its memref operands (it drives their
/// addr/data/we; the parent's `MemUnit` supplies the storage), so a shared
/// internal buffer becomes a `seq.read`/`seq.write` the child addresses. Its
/// scalar result lands at `start + latency` as a survivor.
struct CallUnit {
  CallId id = 0;
  Operation *invoke = nullptr; // the dcp::DCPathInstanceOp
  RegionId region = 0;         // the RegionBlock (a dcp.sequential) it sits in
  std::string callee;          // callee symbol (key into CalleeCtx maps)
  // The invoke's `latency`: its start->done depth. Empty when the callee
  // publishes no whole-kernel latency, so it completes on its own `done`.
  std::optional<int64_t> latency;
  unsigned start = 0; // region-relative issue cycle (the invoke `start`)
  /// Whether the child completes at a statically known cycle, so a consumer may
  /// be released by a static offset instead of its real `done`. The invoke's
  /// declared `determinacy`, not `latency.has_value()`: a dynamic-trip callee
  /// publishes a latency bound and stays indeterminate.
  bool determinate = false;
  /// An `await` spawn rather than a scheduled call: it starts with its
  /// container and is ordered thereafter by FIFO back-pressure alone, so it has
  /// no offset to place at and offers a consumer nothing to time-trigger off.
  bool async = false;

  /// One memory port the child drives for a mastered memref operand. A callee
  /// argument accessed at several points exposes several (a read-twice argument
  /// two read ports, a read-modify-write accumulator a read and a write), so
  /// there is one MemArg per child port rather than per operand.
  struct MemArg {
    unsigned calleeArg; // operand position == callee argument index
    MemId mem;          // caller MemUnit backing this array
    bool isBoundary;    // a func BlockArgument vs an internal alloc
    bool isWrite;
    unsigned bank = 0;   // cyclic bank this port serves (0 unbanked)
    unsigned factor = 1; // partition factor (1 unbanked)
    /// The caller-side port of that bank this child drives, from the same
    /// `bindMemoryPorts` assignment the caller's own accesses take
    /// (`MemUnit::Access::port`).
    unsigned port = 0;
    /// How this port reaches the storage. Never `Crossbar`: a child masters one
    /// bank, already indexed in that bank's own space.
    PortPlan plan = PortPlan::Coloured;
    /// The child says its write ports on this argument never collide, so the
    /// array backing them may give each its own `always` block
    /// (`MemUnit::writesIndependent`, `iface::Memory::independent`).
    bool independent = false;
    std::string addr, data, we; // child port names; `we` empty for a read
    std::string topBase; // top boundary port base (indexed); empty = internal
    /// Whether this MemArg opened `topBase`'s group. A child mastered on a
    /// (bank, port) colour another holder already opened shares that group,
    /// and only the opener declares it in the interface.
    bool ownsGroup = true;
  };
  llvm::SmallVector<MemArg, 2> memArgs;

  /// A scalar operand the child consumes: its driver (an IO port, a sibling
  /// survivor, an enclosing counter, a same-region unit, or a constant, all
  /// resolved by `recordCallScalars`) plus the child port it feeds.
  struct ScalarArg {
    Source src;
    std::string port; // child scalar-input port name
    /// The port's width, so the wiring adapts the driver to the child rather
    /// than the child's width propagating back into its producer. An enclosing
    /// counter needs it: an index has no width of its own.
    unsigned width = 0;
  };
  llvm::SmallVector<ScalarArg, 1> scalarIns;

  /// A stream (FIFO) operand: the child is one end of a channel, handshaking on
  /// three ports of its own. A channel crossing a call boundary is a
  /// back-pressured hand-off, not a timed one, so the leaf datapath rejects a
  /// stream-operand call.
  struct StreamArg {
    unsigned calleeArg;             // operand position == callee argument index
    StreamId chan;                  // the channel this port binds
    bool isInput;                   // the child reads the channel
    unsigned depth = 2;             // the child's requested buffering
    std::string base;               // the child's port group
    std::string data, valid, ready; // its three port names
  };
  llvm::SmallVector<StreamArg, 1> streamArgs;

  /// The child result-output port per scalar result. The result's datapath
  /// Source is Source::Call{id, k}, captured into this region's survivor
  /// exactly like a compute result: a sibling reads it as
  /// Source::Survivor{region, k}.
  llvm::SmallVector<std::string, 1> resultPorts;

  /// An earlier sibling call this one must start after: composition
  /// predecessors at call granularity. `recordCallDeps` derives them by how the
  /// owning region composes, a scheduled composition ordering its children by
  /// their placed `start` while a concurrent one reads the hazard directions
  /// instead.
  struct Pred {
    CallId call;
    /// The edge is a scalar result hand-off rather than a shared array, and can
    /// never be time-triggered: the producer's result port only holds from its
    /// `done`.
    bool viaResult = false;
  };
  llvm::SmallVector<Pred, 2> predecessors;

  /// How this call is released. Decided by `recordCallDeps` off the owning
  /// region's composition class and this node's own contract.
  enum class StartPolicy {
    /// The rising edge of the predecessors' joined `done`. The only policy
    /// available when the producer's completion cycle is not statically known:
    /// a spawn, a consumer of a scalar result, or a gate on an indeterminate
    /// producer. A channel-connected pair is never one of these, back-pressure
    /// already being their ordering.
    Handshake,
    /// The container's own start, taken directly: an ungated spawn, ordered
    /// thereafter by back-pressure alone.
    Broadcast,
    /// Released at the scheduled offset `start`. Not at the region's issue
    /// pulse, since an ungated call's operands need not be ready there (a
    /// scalar argument loaded from memory is the reachable case).
    TimeTriggered,
  };
  StartPolicy startPolicy = StartPolicy::TimeTriggered;
};

/// A FIFO channel: a `!allo.stream` value, handshaked (valid/ready) rather than
/// addressed. Either an input (the kernel reads it via `allo.stream.get`) or an
/// output (writes it via `allo.stream.put`); its payload type and depth come
/// from the stream type. A get's loaded token is referenced by
/// Source{Stream, id, <index of the get access>}; a put consumes `data`. A
/// channel carries exactly one access (single-producer / single-consumer).
struct StreamChannel {
  StreamId id = 0;
  Value stream;         // the !allo.stream SSA value (a func block arg)
  Type payload;         // element type carried through the FIFO
  unsigned depth = 2;   // FIFO depth (from the stream type)
  bool isInput = false; // input (get) vs output (put)
  // A channel this kernel owns: defined by an `allo.stream.create` in its own
  // body rather than passed in, so both ends sit inside this module. It takes
  // no boundary port (a `seq.fifo` in the module body carries it) and is the
  // one channel that may be both read and written (a loop-carried delay line),
  // which leaves `isInput` meaningless for it.
  bool internal = false;
  /// Initial tokens (a `stream.create` initializer): the channel's history is
  /// `[init] ++ [produced]`, breaking a feedback cycle's start dependence.
  /// Realized as a consumer-side prepend shim, not as tokens pushed into the
  /// FIFO. Null for an unseeded channel.
  Attribute init;

  /// A channel end that is a child port rather than one of this module's own
  /// `get`/`put` accesses: `(call, index into that CallUnit's streamArgs)`. A
  /// container wires its channels end-to-end between `hw.instance`s and issues
  /// no access of its own; a leaf's channels have accesses and no call ends. A
  /// channel may have several consumer ends, the fan-out realized as one FIFO
  /// per reader pushed in lock-step, but only one producer end: a merge has no
  /// deterministic token interleaving, so `validateDatapath` rejects it.
  struct CallEnd {
    CallId call;
    unsigned arg; // index into `dp.calls[call].streamArgs`
  };
  llvm::SmallVector<CallEnd, 2> callEnds;

  struct Access {
    Operation *op = nullptr; // the stream.get / stream.put op
    bool isPut = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned stage = 0;  // scheduled cycle within the region (dcpStart)
    /// The setup delay the schedule priced this access against, in ns
    /// (`accessCharacterization`), as on `MemUnit::Access`.
    double inDelay = 0.0;
    Source data; // put: the token's data driver (puts only)
    // A predicated access (an i1 `pred` operand from a masked `if`) consumes or
    // produces its token only where this holds. Delayed to `stage` like `data`;
    // None for an unconditional access.
    Source when;
  };
  llvm::SmallVector<Access, 1> accesses;
};

/// A multiplexer inserted where sharing makes several sources contend for one
/// sink input. Empty in the trivial binding; one per shared-unit input port
/// that sees different drivers across the ops bound to it.
struct Mux {
  MuxId id = 0;
  /// Which iterations of its selecting op's run an arm drives. A recurrence of
  /// distance `d` reads one identity per iteration below `d` and the value
  /// carried round from `d` on, taking `d + 1` arms of one select instead of a
  /// mux chain in front of the port. Used wherever `FuncUnit::inputInits`
  /// cannot hold the identities: a time-shared port, which has no cycle of its
  /// own to time such a mux against, and a slot with no input port beside it at
  /// all (an address, a store datum, a stream token).
  struct Phase {
    enum Kind : uint8_t {
      Always, // every iteration: an ordinary operand
      At,     // iteration `iter` alone: that iteration's identity
      From,   // iteration `iter` on: the value carried round
    };
    Kind kind = Always;
    unsigned iter = 0;
  };
  llvm::SmallVector<Source, 2> sources;
  // The op whose issue selects each source (parallel to `sources`): that op's
  // activation pulse, the signal a store's write-enable also uses, narrowed by
  // `phases`. The selects are one-hot because the MRT holds the ops to disjoint
  // residues and `phases` partitions one op's pulse across its arms.
  // `selectStages` is the cycle each pulse is delayed to.
  llvm::SmallVector<Operation *, 2> selectOps;
  llvm::SmallVector<unsigned, 2> selectStages; // parallel to `selectOps`
  llvm::SmallVector<Phase, 2> phases;          // parallel to `sources`
  RegionId region = 0; // region whose issue pulse times the selects
  Type type;           // the muxed value's type, whose width prices the select
};

/// The sub-cycle room \p u's bound ops have left, in ns: the smallest
/// `cycleTime - z - inDelay` over them, less the reduction-identity select a
/// recurrence port carries (priced off \p lib), which sits inside the proved
/// cycle. Bounds the combinational delay binding may add in front of the unit.
/// Empty where any bound op carries no `z`: its room is unknown rather than
/// maximal, so binding must not fold onto it.
/// \p sinkTails, non-null, further charges each bound op the committed delay a
/// non-unit sink it feeds still spends after its result (`sinkTails` below).
std::optional<double>
unitSlack(const FuncUnit &u, const OperatorLibrary &lib, float cycleTime,
          const llvm::DenseMap<Operation *, double> *sinkTails = nullptr);

/// A top-level scalar input port (a scalar kernel argument). Memref arguments
/// become external `MemUnit`s and a scalar function result becomes a `Result`,
/// so every IOPort is an input by construction.
struct IOPort {
  IOId id = 0;
  Value value;
  Type type;
};

/// A literal tied into the datapath.
struct ConstCell {
  ConstId id = 0;
  Attribute value;
  Type type;
};

/// A scalar function result, exposed as an output port driven by `source` (a
/// returning region's survivor, a passthrough scalar input, or a constant) and
/// valid when the function's `done` rises. An array (memref) result becomes a
/// trailing out-param before emit, so only scalars reach here.
struct Result {
  Source source;
  Type type;
  std::string name;
};

//===----------------------------------------------------------------------===//
// Regions. One RegionBlock per dcp region op. Cyclic blocks are II-paced
// pipelined loops; acyclic blocks are straight-line, and blocks run in program
// order with no overlap.
//===----------------------------------------------------------------------===//

/// How a region produces one of its results. One shape covers every regime, so
/// a consumer reads the same three fields whichever controller runs. A region
/// result is always a *survivor register*: the value is latched when it lands
/// and held for whoever reads it (a sibling region, an enclosing container's
/// next iteration, the function's output port).
///
///   counted loop / while | `value` = the loop-carried next (the terminator's
///                        |   `dcp.uncondition` / `dcp.condition` operand),
///                        |   `init` = the matching `inits` operand. The two
///                        |   regimes differ only in the pulse the capture keys
///                        |   off.
///   sequential           | `value` = the yielded value; no recurrence, so
///                        |   `init` is None (it lands exactly once).
///   guard (dcp.select)   | `value` = the then arm's yield, `elseValue` = the
///                        |   else arm's; the survivor is `cond ? then : else`.
///
/// A `None` `value` is an untracked result: no survivor is built, and a
/// consumer that reads it fails at its own slot. A `None` `init` means the
/// result is not a loop-carried recurrence. Its survivor then powers on at zero
/// instead of being preloaded, which is only safe because such a result always
/// lands.
struct RegionResult {
  Source value;
  Source init;
  Source elseValue;
};

struct RegionBlock {
  RegionId id = 0;

  /// The `dcp.pipeline` / `dcp.sequential` / `dcp.select` this block models, so
  /// a region diagnostic anchors on the loop the user wrote rather than the
  /// enclosing function.
  Operation *op = nullptr;

  /// Structural shape, axis 1 of the controller discriminant (shape by
  /// termination class picks the controller).
  ///
  /// The populated cells:
  ///
  ///                | CountedStatic          | Conditional
  ///   -------------+------------------------+---------------------------
  ///   Leaf         | free-running / modulo  | flushing while
  ///   Container    | counted outer + child  | check/run outer
  ///                | sequencer              |
  ///   Guard        | branch-pulse, run-once | (same: run-once either way)
  ///   CallNode     | fire + child `done`    | n/a
  ///
  /// Every other cell is unreachable; `emitRegion` rejects rather than falling
  /// through. Spelled once in `RegionShape`, so the reifier and the emitter
  /// cannot disagree.
  using Shape = allo::RegionShape;
  /// Read off the region op by `dcpRegionShape` in
  /// `DatapathBuilder::deriveShapes`, which re-asks it of the built model
  /// (parent/child edges linked, CallUnits bound) and asserts the two agree.
  Shape shape = Shape::Leaf;

  enum class Kind { Cyclic, Acyclic } kind = Kind::Acyclic;
  std::optional<unsigned> ii; // set iff Cyclic

  /// Whether at most one pass of this region is in flight. A cyclic region
  /// overlaps its iterations at `ii` by construction; every other family runs a
  /// pass to its `done` before the next issues.
  bool singlePass() const { return kind == Kind::Acyclic; }

  // Counted-loop induction: the IV runs `lb, lb+step, ...` up to (excluding)
  // `ub`. Each bound is a datapath `Source`, a data-dependent value or a
  // literal `ConstCell` synthesized by `recordRegionBounds`. Set for a Cyclic
  // region, None for an Acyclic one (no counter).
  //
  // `ubSource` is the one exception: a constant trip over a runtime lb or step
  // (the `for j in range(i, i+K)` window) has `ub = lb + K*step`, derived
  // arithmetic no cell can carry, so `ubSource` is None there and
  // `terminatorOf` builds the expression instead.
  std::optional<int64_t> tripCount; // constant trip iff Cyclic
  /// An upper bound on the trip of a loop that has no constant one, from the
  /// `allo.assume.ssa` range the scheduler distilled (`dcp.pipeline`'s
  /// `trip_bound`). Mutually exclusive with `tripCount`, as the op verifier
  /// enforces.
  std::optional<int64_t> tripBound;
  Source lbSource;   // lower bound (counter init)
  Source ubSource;   // upper bound; see `tripCount` above
  Source stepSource; // step (counter increment)
  // The width the iteration counter is built at.
  Type counterType;
  /// The counter never goes negative, so it is built at an unsigned width (one
  /// bit under the signed hull) and every predicate and resize that reads it is
  /// unsigned. False for a genuinely signed counter (a negative lb) and for a
  /// while, whose 0-based counter keeps the default 32-bit signed form.
  bool counterUnsigned = false;
  /// Signed hull of every value the counter holds or compares, set only when
  /// a runtime bound narrowed below `kIndexWidth`; the recurrence gates widen
  /// `lb + n*step` past `counterType` where it leaves this range.
  std::optional<std::pair<int64_t, int64_t>> counterHull;
  int64_t counterStepHi = 1; // the step's hull top
  // Termination class as the emitter discriminates it, axis 2 of the pair
  // above. A while loop (a `dcp.condition` terminator) is a flushing pipeline
  // whose exit is data-dependent. The declared `determinacy` below agrees in
  // one direction only: a while is always declared `Conditional` (asserted in
  // `deriveShapes`), and so is a `dcp.select`, where `conditional` stays false
  // since a guard is not a flushing loop.
  bool conditional = false;
  // The two raw structural flags `shape` is derived from. Consumers should read
  // `shape`.
  bool guard = false; // this region op is a dcp.select
  /// Nests another dcp region in either arm, so a guard with children is
  /// `container` too and this is not the same as `shape == Container`.
  ///
  /// Read directly for one question only: whether a `dcp.pipeline` latches its
  /// loop-carried values into survivors a nested region can name, rather than
  /// fusing them into a register recurrence only the carrying op reads
  /// (`resolveValue`, `resolveOperand`). Every other site wants `shape`.
  bool container = false;
  /// Whether a value produced by one iteration is read back by a later one out
  /// of a register of this region: a chain tap at `distance * ii + tY - ready`,
  /// or a fused IP whose own pipeline is the accumulator. Such a read holds
  /// only while iterations issue exactly `ii` apart, which is what stops
  /// `deriveStallShell` from deferring an issue without freezing the datapath.
  /// A recurrence carried through a stream or a memory is elastic and is not
  /// one of these.
  bool cycleIndexedState = false;
  std::string counterName; // source loop IV name (its NameLoc), for a readable
                           // iteration-counter wire; empty when the IV carried
                           // no name (best-effort)
  /// A register this region carries beside its own counter, holding
  /// `coeff * digit` of it for a coefficient and a digit an access's address
  /// needs, advanced rather than rebuilt.
  ///
  /// A digit `(x floordiv D) mod K` rides the same register: it advances by
  /// nothing on most iterations and by one where `x` crosses a multiple of `D`,
  /// carried from a companion register holding `x mod D` (itself a stride with
  /// `wrap = D`) and wrapped at `K` by subtracting once.
  ///
  /// One update rule covers both:
  ///
  ///     raw  = cur + step + (carry fired ? bump : 0)
  ///     next = wrap && raw >= wrap ? raw - wrap : raw
  ///
  /// A plain scaled counter is `bump = wrap = 0`. `step + bump <= wrap` holds
  /// by construction, `asDigit` refusing a step that could wrap twice, so the
  /// single subtract is exact. A decreasing digit (`A[N-1-i]`) mirrors it:
  /// `step` and `bump` go negative and the wrap adds on borrow (`raw > cur`
  /// unsigned) instead of subtracting on overflow.
  struct AddrStride {
    int64_t init;       // `coeff * lb`, the value the register loads at start
    int64_t step;       // `coeff * step`, added wherever the counter advances
    int64_t bump = 0;   // added when `carry`'s register wraps
    int64_t wrap = 0;   // subtracted on reaching it (0: a plain accumulator)
    unsigned carry = 0; // slot whose wrap gates `bump`; self means none
    bool hasCarry = false; // whether `carry` names one
    bool down = false;     // counts down, so `wrap` is added on borrow
    /// The width the register is built at. Every field above is compile-time,
    /// so its range is too, rounded up to bits and independent of the counter's
    /// own width: `clog2` of the modulus for a wrapping digit, of the array for
    /// a row stride. `kIndexWidth` when the range is unbounded (`slotFor`).
    unsigned width = kIndexWidth;
    /// This stride is the region's own counter (`init == lb`, `step`, no bump,
    /// wrap or carry), so the emitter reads `rc.counter` for it and builds no
    /// register. Set by `planAddressGenerators`.
    bool isCounter = false;
  };
  /// Deduplicated, since two accesses down the same row share a stride. Some
  /// slots exist only to carry another (the `x mod D` companion of a quotient
  /// digit) and no access names them; a carry always precedes its consumer, so
  /// one pass emits them. Empty when no address follows this counter, or its
  /// bounds are not constant (which is what makes the fields compile-time).
  llvm::SmallVector<AddrStride> addrStrides;

  // Composition class, derived by `dcpRegionTiming` in `addRegion`.
  // `deriveShapes` asserts the one cross-axis invariant, that `conditional`
  // implies `determinacy == conditional`.
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;

  /// The terminal cycle of this region's own datapath, relative to the issue
  /// pulse of the iteration that reaches it: the last of the deepest store to
  /// commit, the deepest survivor to latch and the deepest put to present.
  /// `emitDone` is the only consumer and rises `drainStage + 1` cycles after
  /// the last issue; every shape but a Leaf completes on its children instead.
  unsigned drainStage = 0;
  // The same cycle as the latency model composed it (`drain` on the region op),
  // against which `HWEmitter::emitRegion` checks `drainStage`. A divergence is
  // a consumer placed at an offset the hardware does not honour.
  std::optional<int64_t> modelledDrain;
  /// Cycles `resolveStreamOperands` inserted into this region's stream
  /// schedule (the transient-din bump), maximized over its channels. The span
  /// was composed off the unshifted schedule, so this bounds how far past
  /// `modelledDrain` the built drain may sit.
  unsigned streamShift = 0;

  // Composition predecessors: the earlier top-level sibling regions this one
  // must start after, set by `recordSiblingDeps`. Only top-level regions
  // populate it, container children staying serial. A region depends on an
  // earlier sibling iff they touch a shared memref with a hazard between them
  // (RAW / WAW / WAR; read-read pairs overlap and take separate ports), a
  // shared stream, or a cross-region SSA edge (a scalar survivor); functional
  // units are auto-disjoint under per-region binding. A region with no
  // predecessors starts concurrently with the kernel, one with predecessors on
  // their joined `done`. Producers precede consumers in program order, so the
  // relation is a DAG.
  llvm::SmallVector<RegionId, 2> predecessors;

  // Region nesting. A container region drives its `children` in its body; each
  // child's `parent` is the enclosing container. Top-level regions (no parent)
  // are the func-scope siblings chained by the sequencer; a container runs its
  // child `tripCount` times (hierarchical control, II_outer >= L_inner).
  std::optional<RegionId> parent;
  llvm::SmallVector<RegionId, 2> children;
  // A guard (dcp.select) with a non-empty `else` branch is a dual guard: its
  // `children` are the then-branch sub-schedule (run iff the predicate holds)
  // and `elseChildren` are the else-branch sub-schedule (run iff it does not).
  // Empty for a container loop and for a then-only guard.
  llvm::SmallVector<RegionId, 2> elseChildren;

  // Cells owned by this region (ids are Datapath-global; these record
  // membership and thus which counter drives them).
  llvm::SmallVector<UnitId, 4> units;
  llvm::SmallVector<RegId, 4> regs;
  llvm::SmallVector<MuxId, 2> muxes;
  llvm::SmallVector<CallId, 1> callUnits; // sub-kernel calls
  // The accesses this region issues, driven by its controller and timed against
  // its schedule. Memories and streams are owned Datapath-wide, a buffer
  // written in one region and read in another being one storage cell, so
  // membership is a property of each access and is recorded here. Both lists
  // are in body program order, the order `bindResource` walks.
  llvm::SmallVector<AccRef, 2> memAccesses;
  llvm::SmallVector<AccRef, 1> streamAccesses;

  // The Sources this region's results come from, indexed by result number (see
  // RegionResult). Empty for a result-less region. For a loop this is also its
  // loop-carried recurrence, `results[k]` being iter-arg k, since a counted
  // loop's k-th result is the final value of its k-th iter-arg.
  llvm::SmallVector<RegionResult, 1> results;

  // This region's control predicate, as a resolved i1 Source: a while's
  // per-iteration continue condition, or a guard's (dcp.select) run-once
  // predicate. None for a counted region, which terminates on its counter.
  //
  // A while's condition is a scheduled compute producer (cmpi/cmpf, a
  // Source::Unit); a guard's is that same combinational unit over the enclosing
  // counter (an affine guard `i > j`) or a scheduled prologue region's survivor
  // (a data-dependent `flag[j] > 0`). Either way it is held for the run it
  // gates: a guard start-gates its children by it, so the not-taken arm's
  // stores never fire structurally, with no per-store gate.
  Source condition;
};

//===----------------------------------------------------------------------===//
// The whole microarchitecture of one function.
//===----------------------------------------------------------------------===//

struct Datapath {
  dcp::DCPathModuleOp func;
  /// Whether this function is the top of the emitted design, so its arguments
  /// name storage nobody in the design owns. A callee's array argument is a
  /// port it masters on its caller's storage, so the two answer
  /// `MemUnit::scattered` differently.
  bool atTop = false;

  // Derived structural cells.
  std::vector<FuncUnit> units;
  std::vector<Register> regs;
  std::vector<MemUnit> mems;
  std::vector<StreamChannel> streams;
  std::vector<Mux> muxes;
  std::vector<IOPort> ios;
  std::vector<ConstCell> consts;
  std::vector<Result> results;      // scalar func results, in return order
  std::vector<CallUnit> calls;      // sub-kernel calls
  std::vector<RegionBlock> regions; // program order

  // The module's boundary memory ports: every access to an external (func
  // argument) memref, split by role and ordered by (memref, access). An
  // internal memref is on-chip `seq.hlmem` storage and takes no port. The
  // single enumeration: an access's index here is its port identity, mirrored
  // onto the access as `MemUnit::Access::portIdx` and read by the port
  // declaration, the naming layer, the manifest and the emitter alike.
  llvm::SmallVector<AccRef> readPorts, writePorts;

  // L1 binding decisions the policy writes; the structure above is derived from
  // these plus the schedule. (Memory port binding lives in MemUnit::accesses,
  // co-located with its memref.)
  llvm::DenseMap<Operation *, UnitId> opToUnit;

  /// Set when the builder hit a schedule it cannot realize and has already
  /// reported it, namely a consumer placed before its producer's result is
  /// ready
  /// (`resolveOperand`). The build finishes with placeholder values so it stays
  /// bounded, and `validateDatapath` turns this into a failure before any
  /// hardware is emitted.
  bool infeasible = false;

  /// The pulse-delay depth from which `delayValid` builds a counter instead of
  /// extending a chain: the crossover of the device's chain row against the
  /// counter's own registers and arithmetic, stamped by the builder. The
  /// default applies on a device that prices neither side.
  unsigned countedDelayCycles = 64;

  Datapath() = default;
  /// \p dev is the device the scheduler priced this kernel against: its
  /// storage model resolves each MemUnit's implementation and access latency,
  /// its operator rows let \p policy price a fold's multiplexer against
  /// \p cycleTime, the period the schedule was cut to.
  Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
           const DeviceModel &dev, float cycleTime, const CalleeCtx &callees,
           bool isTop = false);

  /// \p s's compile-time value, when it is an integer literal cell; empty for
  /// every Source whose value is only known at run time.
  std::optional<int64_t> constantOf(const Source &s) const;

  /// The cycle \p s's value lands, relative to the issuing pulse of the
  /// iteration that produced it. Zero for a held source (a literal, a port, a
  /// counter, a survivor), which is settled before the reader issues.
  unsigned readyCycle(const Source &s) const;

  /// The top-level region \p r sits under, walking the container chain to the
  /// root. The granularity `recordSiblingDeps` orders at and `portGraph` reads
  /// that ordering back at.
  RegionId topRegionOf(RegionId r) const;

  void dump(llvm::raw_ostream &os) const;

  /// A vertex of `portGraph` that is a child's port rather than an access of
  /// this function, so it has no index in `MemUnit::accesses`.
  static constexpr unsigned kNoAccess = ~0u;

  /// One vertex of `portGraph`: which access of this function it is
  /// (`kNoAccess` for a child's port), which call masters it (-1 for an access
  /// of this function), whether that call declared its ports collision-free,
  /// which direction it runs in, and which bank it commits to (-1 when it may
  /// reach any).
  struct PortVertex {
    unsigned access = kNoAccess;
    int call = -1;
    bool independent = false;
    bool write = false;
    int bank = -1;
  };

  /// The accesses of one array that hold a port, and the "can issue in one
  /// cycle" relation over them. A child's port counts as an access.
  struct PortRelation {
    llvm::SmallVector<PortVertex> verts;
    /// Which vertices vertex `i` may issue with; never itself.
    llvm::SmallVector<llvm::BitVector> adj;
    unsigned size() const { return verts.size(); }
    void link(unsigned i, unsigned j) {
      adj[i].set(j);
      adj[j].set(i);
    }
  };

  /// The relation over the accesses of \p id. \p writes selects one direction;
  /// nullopt takes both, writes before reads.
  ///
  /// Conservative in the safe direction. Only an ordering the model already
  /// proves separates a pair: two top-level regions are separated by a path in
  /// the sibling DAG (`recordSiblingDeps`, run by `build()` before any caller
  /// of this); two calls by a `recordCallDeps` edge or by disjoint
  /// TimeTriggered contract intervals of one scheduled region; two
  /// region-local accesses at different modulo residues never share a cycle;
  /// and two accesses committing to different banks contend for nothing.
  /// Anything else counts as simultaneous.
  PortRelation portGraph(MemId id, std::optional<bool> writes) const;

  /// A lower bound on the accesses of \p id in direction \p writes that one
  /// cycle has to serve: the largest `portGraph` clique a greedy expansion from
  /// each vertex finds. Never above the true maximum, and exact at a clique of
  /// one or two. Zero for a direction with no access.
  unsigned portConcurrency(MemId id, bool writes) const;
};

//===----------------------------------------------------------------------===//
// The model visitor. `Source`s are scattered across ~20 slots of the model, so
// `forEachSource` is the one traversal: a new `Source` field is covered by
// adding it here, once.
//===----------------------------------------------------------------------===//

/// One `Source` slot in the model: what it drives, and whether being
/// unresolved (`Source::Kind::None`) is a defect there.
struct SourceSite {
  enum class Slot {
    UnitInput,        // a compute unit's operand port
    UnitInit,         // that port's reduction identity (absent => None)
    RegisterInput,    // a shift chain's head driver
    MuxInput,         // one arm of a derived sharing mux
    MemAddress,       // an address operand of a memory access
    MemWriteData,     // a store's data (a load leaves it None)
    StreamData,       // a put's token data (a get leaves it None)
    StreamPredicate,  // a masked access's `pred` (absent => None)
    CallScalarIn,     // a scalar operand handed to a sub-kernel
    FuncResult,       // a scalar function result's driver
    RegionBound,      // a runtime lb / ub / step (compile-time => None)
    RegionResult,     // a region's yielded value / carried next (untracked
                      // => None)
    RegionResultInit, // that result's loop-carried identity (absent => None)
    RegionElseResult, // a dual guard's else-arm yield
    RegionCondition,  // a while's continue condition / a guard's predicate
                      // (a counted region has none => None)
  };
  Slot slot;
  /// Which port / operand / result of the owner this is; a RegId for
  /// `RegisterInput` and a RegionId for `RegionBound` / `RegionCondition`.
  unsigned index = 0;
  /// The dcp op this slot belongs to, for a located diagnostic. Null for a slot
  /// owned by a region or by the function rather than by one op.
  Operation *op = nullptr;
  /// Whether an unresolved Source here is a defect. False for the slots where
  /// `None` is the legitimate encoding of "absent" (see the comments above).
  bool required = true;

  /// A noun phrase naming this slot, for a diagnostic ("operand 1 of a compute
  /// unit"). Built only on failure.
  std::string describe() const;
};

/// Visit every `Source` slot in \p dp exactly once, in model order.
void forEachSource(
    const Datapath &dp,
    llvm::function_ref<void(const Source &, const SourceSite &)> fn);

/// The committed delay each op's result still spends inside its own cycle at a
/// non-unit sink it feeds directly: a memory port's setup past a store's data
/// or an unregistered address, or a stream port's, none of which any unit's
/// room accounts for. Keyed by the producing op and read off the schedule's
/// own access stamps, so it is the same map before binding (each unit one op)
/// and at emit (per bound op); a port-colour select is deliberately not in it,
/// being a reported quality finding rather than a refusable binding cone.
llvm::DenseMap<Operation *, double> sinkTails(const Datapath &dp);

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR, for the builder only: every cycle
// the emitter needs is frozen onto the model (`FuncUnit::BoundOp::stage`,
// `MemUnit::Access::stage`, `StreamChannel::Access::stage`,
// `Mux::selectStages`, `CallUnit::start`). `readyCycleOf` is the single
// authority for the cycle a producing op's result lands, relative to its
// issuing pulse.
//===----------------------------------------------------------------------===//

/// Region-relative schedule cycle of a dcp compute/load/store op (its `start`).
unsigned dcpStart(Operation *op);
/// Result latency of a producing dcp op (0 if uncharacterized): a load's own
/// `latency`, or an IP compute's `latency` (stamped at emit from its operator).
unsigned dcpLatency(Operation *op);
/// The cycle a producing op's result is ready: `dcpStart + dcpLatency` (a
/// stream get is a combinational front-read, latency 0). Zero for an at-issue
/// value with no producing op (a constant, the iteration counter).
unsigned readyCycleOf(Operation *op);

/// The drain cycle a store to \p m contributes, relative to its region's
/// issue: `stage + writeLatency - 1`, one less than the commit cycle because
/// the done latch registers a cycle after it. Model side and emitter side each
/// reduce their own maximum over this per-store number.
unsigned storeDrainCycle(const MemUnit &m, const MemUnit::Access &acc);

/// A region's controller shape as one lower-case word, the single spelling
/// shared by the debug dump and the microarch report.
llvm::StringRef shapeName(RegionBlock::Shape s);

/// The banking of an external (argument) memory access, so the boundary
/// presents one interface per bank. `factor == 1` is an unbanked memory
/// (`bank == 0`); a banked access is either statically routed (`bank` set) or
/// data-dependent (`bank` empty, a crossbar over all `factor` interfaces). A
/// skewed argument reaches the data-dependent arm, `staticBank` being empty
/// under a skew.
///
/// Both halves are stored on the model (`MemUnit::numBanks` and
/// `Access::staticBank`); this pairs them under the name consumers ask by.
struct ExternalBanking {
  unsigned factor = 1;          // physical banks (1 = unbanked)
  std::optional<unsigned> bank; // static bank, or empty = data-dependent
};
inline ExternalBanking externalBank(const MemUnit &m,
                                    const MemUnit::Access &acc) {
  return {m.numBanks, acc.staticBank};
}
// (`extPorts` in Naming.h names the resulting per-bank interfaces.)

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATH_H

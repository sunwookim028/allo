/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MEMORYMODEL_H
#define ALLO_SCHEDULING_MEMORYMODEL_H

#include "allo/IR/AlloAttrs.h"         // MemoryPortEnum
#include "allo/Scheduling/Scheduler.h" // OccupancyProblem

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::allo {

/// The width an `index` value widens back to when read as an ordinary index,
/// though a counter or address register may be built narrower
/// (`RegionBlock::counterType`, `RegionBlock::AddrStride::width`). Emitter and
/// scheduler both price against it.
inline constexpr unsigned kIndexWidth = 32;

/// The bits a value of type \p t occupies in the datapath: `index` at
/// `kIndexWidth`, a float as its bit pattern, an integer verbatim. One answer
/// shared by operator pricing (`combParamWidth`), the emitter
/// (`uarch::datapathType`) and the boundary port model (`iface`).
unsigned datapathWidth(mlir::Type t);

//===----------------------------------------------------------------------===//
// Memory timing library: the `memory:` section of the device file. Read/write
// latency and delay per storage implementation, plus one FIFO (stream) timing.
//===----------------------------------------------------------------------===//

/// Read/write latencies (cycles) of one storage kind.
struct RWLatency {
  unsigned read = 0;
  unsigned write = 0;
};

/// Read/write combinational delays (ns) of one storage kind.
struct RWDelay {
  double read = 0.0;
  double write = 0.0;
};

/// Timing of one storage realization (or of the stream FIFO): latency and
/// delay, each split by direction.
struct MemKindTiming {
  RWLatency latency;
  RWDelay delay;
};

/// Instances of its storage row one array may be held in. A compiler policy,
/// not a hardware limit: each copy costs the row's area again and buys one
/// instance's worth of reads.
constexpr unsigned kStorageCopies = 2;

/// Ports of one instance of a storage realization, per bank. Nullopt is no
/// limit on that axis. The three together give what an array may be given in a
/// cycle, which the scheduler reserves and the datapath binds against.
///
/// A block RAM instance's two ports each read or write, so two writers and a
/// concurrent reader take three, stated by `instPool`. A row whose directions
/// are independent declares no pool, as a LUT RAM's write port against its
/// addressed read does.
struct StoragePorts {
  std::optional<unsigned> instReads;
  std::optional<unsigned> instWrites;
  std::optional<unsigned> instPool;
  /// Whether the counts describe the whole array rather than one copyable
  /// instance. Set by an `allo.bind.storage type=` topology and by a stream's
  /// two ends, neither leaving room for an added copy.
  bool stated = false;

  /// Instances an array may be spread over in a cycle, what the scheduler
  /// issues against. Not a bound on the copies built: the port binding colours
  /// by what may share an address bus, so two issued reads can still need three
  /// buses.
  unsigned copies() const { return stated ? 1 : kStorageCopies; }

  /// The tighter of the two budgets on every axis. A nullopt is no limit and
  /// yields to whatever the other side declares.
  StoragePorts meet(const StoragePorts &other) const;

  /// Whether this row can hold an array of \p writes write ports over \p ports
  /// address buses, given as many copies as it takes. Buses are not writes plus
  /// reads: one bus may carry a read and a write that never issue together.
  ///
  /// Reads never disqualify a row (a further read is a further copy); writes
  /// do, since every copy needs every write, so one instance's write ports are
  /// the ceiling at any copy count. A pooled row's writes also spend a port of
  /// every copy. Where the ports are `stated` the whole array fits one
  /// instance.
  bool holds(unsigned writes, unsigned ports) const;

  /// Whether it can serve the topology \p want names, asked of ports a
  /// directive requested rather than ones a binding built.
  bool holds(const StoragePorts &want) const;

  /// One instance's ports as a diagnostic phrase ("2 read / 1 write over 2
  /// shared ports"), an unlimited axis spelled as such.
  std::string describe() const;
};

/// One `dcp.storage` row: a structure the device can hold an array in, named by
/// the device's own vocabulary rather than by a case of a closed enum.
struct StorageRealization {
  std::string name;
  MemKindTiming timing;
  StoragePorts ports;
  /// Vendor attribute pinning an array to this structure, stamped on the
  /// emitted declaration. Empty where the part has none, and the synthesizer
  /// chooses.
  std::string ramStyle;
  /// Whether the structure comes up holding contents. False for one that powers
  /// up undefined, as an UltraRAM does.
  bool canInit = true;
  /// The row that is not a memory: one cell per element, no address, where a
  /// complete partition goes.
  bool scatter = false;
  /// The row that is a constant lookup built from logic: no address bus, no
  /// port limit, where a read-only initialized array goes.
  bool table = false;
  /// What one instance spends over `(depth, width)`, the row's `uses` verbatim,
  /// held as the attribute since the price is only meaningful at an array's own
  /// shape. Null where the device left the row unpriced.
  mlir::ArrayAttr uses;
  /// Read delay curve over the array's depth, and the factor its width scales
  /// it by. Null where delay is the same at every shape and `timing.delay.read`
  /// stands alone.
  CostAttr rdDelayDepth, rdDelayWidth;
};

/// The storage-timing library, filled from the `dcp.storage` and
/// `dcp.stream_timing` rows of the device.
class MemoryLibrary {
public:
  /// Build the library from a module's injected `dcp.device`: its
  /// `dcp.storage` rows and its `dcp.stream_timing`. A module with no
  /// `dcp.device` yields an empty (all-default) library.
  static MemoryLibrary fromModule(ModuleOp module);

  struct Timing {
    unsigned latency = 0;
    double delay = 0.0;
    // The accessed array's resolved storage realization, empty for an access
    // with no storage axis (a stream is a FIFO timed by its own row). Accesses
    // of different realizations must map to different operator types, or they
    // collapse onto one latency, so this keys the type.
    std::string storage;
  };
  /// Timing for a memory/stream access op; zero latency and delay if \p op is
  /// not one. An array access is timed by its memref's storage realization.
  Timing timing(Operation *op) const;

  /// The device's row for the storage realization \p name, or null where it
  /// declares none. Timing, ports and vendor attribute all come from here.
  const StorageRealization *row(llvm::StringRef name) const;

  /// The timing of storage realization \p name. The device must declare every
  /// realization an array resolves to (`PreVerification` enforces this); an
  /// undeclared one asserts and falls to zero (combinational) timing.
  MemKindTiming timing(llvm::StringRef name) const;

  /// Whether \p storage is the row the device marked `scatter`: one cell per
  /// element, no address, no port limit. False for every row when the device
  /// marks none, leaving a complete partition unrealizable rather than silently
  /// addressed.
  bool isScatter(llvm::StringRef storage) const {
    return !scatterStorage.empty() && storage == scatterStorage;
  }

  /// Whether \p storage is the row the device marked `table`: a combinational
  /// constant lookup, no address bus, no port limit. False for every row when
  /// the device marks none, which realizes every constant table as a memory.
  bool isTable(llvm::StringRef storage) const {
    return !tableStorage.empty() && storage == tableStorage;
  }

  /// The read delay (ns) of \p storage holding \p words x \p width: the row's
  /// own `rd_delay`, or its depth curve scaled by its width factor where it
  /// declares one.
  double readDelay(llvm::StringRef storage, int64_t words,
                   unsigned width) const;

  /// What one bank of \p words x \p width of \p storage spends, as a fraction
  /// of the part: the worst of its resources, the axis a design runs out on.
  /// Nullopt where the row is unpriced, where a cost is not measured at this
  /// shape, or where the device quotes no capacity for what it spends. Prices
  /// one instance, not the copies.
  std::optional<double> fractionOfPart(llvm::StringRef storage, int64_t words,
                                       unsigned width) const;

  /// The row an unbound array of \p words x \p width takes: cheapest by
  /// `fractionOfPart` among the pinnable rows at the least access latency.
  /// Latency ranks first without exception, being the contract the schedule is
  /// built on. \p needsInit excludes a row that powers up undefined. Empty
  /// where the device can neither pin nor price a row, and `defaultStorage`
  /// stands.
  ///
  /// \p canTable admits the `table` row, which only an initialized array
  /// nothing writes may take. Cheapest at every shape, so kept only while its
  /// read is no slower than the memory it displaces; its read delay bounds how
  /// deep it is worth building.
  std::string rowFor(int64_t words, unsigned width, bool needsInit,
                     bool canTable) const;

  // The `dcp.storage` marked `default`, empty where the device marks none and
  // `rowFor` chooses. A name not a handle, so replacing a row leaves it valid.
  std::string defaultStorage;
  // What a completely partitioned array resolves to: the `dcp.storage` marked
  // `scatter`, empty when the device marks none.
  std::string scatterStorage;
  // What a read-only initialized array resolves to where its read delay allows:
  // the `dcp.storage` marked `table`, empty when the device marks none.
  std::string tableStorage;
  std::vector<StorageRealization> storage; // the `dcp.storage` rows
  MemKindTiming fifo;                      // `dcp.stream_timing`
  /// How much of each `dcp.resource` the part has, turning a row's spend into a
  /// fraction so rows spending different primitives compare.
  llvm::StringMap<int64_t> capacity;
};

//===----------------------------------------------------------------------===//
// Per-memref storage predicates, read off the array's `allo.part` /
// `allo.bind.storage` attributes.
//===----------------------------------------------------------------------===//

/// The `memref.global` initializer behind \p memRef, i.e. a constant table's
/// declared contents, or nullopt when it has none.
std::optional<Attribute> globalInitOf(Value memRef);

/// Whether \p memRef is eligible to be a constant table: it has a
/// `memref.global` initializer and nothing writes it, here or through a
/// sub-kernel it is handed to. Read-only is a property of the use: an array
/// stored to even once is a real memory that merely starts with contents.
///
/// Eligibility only; whether the table is built is `recordArrayStorage`'s
/// decision, read through `MemoryChar::constantTable`.
///
/// A constant table lowers to `hw.aggregate_constant` read by one
/// `hw.array_get` per access: combinational, no handshake, unlimited-port. A
/// child that only reads one is served off the parent's table; a child that
/// writes disqualifies the array.
bool isConstantTable(Value memRef);

/// The `allo.bind.storage impl=` written on \p memref: what was asked for,
/// before `characterize` resolves it, empty when nothing was. A complete
/// partition overrides an explicit choice, so this makes the two directives
/// comparable and their disagreement reportable.
llvm::StringRef boundStorageOf(Value memref);

/// The realization `recordArrayStorage` resolved \p memref to (`kStorageAttr`),
/// matching what was asked for only where the user bound it. A lookup, so two
/// carriers of one array agree by reading one record, not by repeating a
/// derivation.
std::string resolvedStorageOf(Value memref);

/// The two orthogonal axes of one `allo.bind.storage` directive: its `type`
/// string (port topology) and its `impl` string (storage realization). The
/// RAM/ROM half of a `type` spelling is not an axis; read-only is a property of
/// the use, decided by `isConstantTable`.
struct BindStorage {
  /// The topology asked for, empty where the directive names none. Absent is
  /// not the dual-port default: an array that asked for nothing takes whatever
  /// its realization has, and only an explicit topology narrows it.
  std::optional<MemoryPortEnum> port;
  llvm::StringRef storage; // empty: no explicit choice, not "no storage"
};

/// The axes \p bind states, all defaulted for a null dictionary.
BindStorage parseBindStorage(mlir::DictionaryAttr bind);

/// Whether topology \p a serves everything \p b asks for. The three form a
/// chain, `1p` under `s2p` under `t2p`, so two carriers of one array reconcile
/// by taking the one that covers the other.
bool topologyCovers(MemoryPortEnum a, MemoryPortEnum b);

//===----------------------------------------------------------------------===//
// Partition and static-bank queries. A DCP banking pass reuses these facts so
// it materializes the *same* banks the scheduler bound its ResII against.
//===----------------------------------------------------------------------===//

/// The bank decomposition of a partitioned memref, in element space: which bank
/// holds element `(i_0 .. i_{r-1})` and where inside it. The single definition
/// of "which bank", shared by the port model, the static split, the emitter's
/// crossbar and the host layout.
///
/// A cyclic axis of factor F puts element `i_d` in bank `i_d mod F` at local
/// coordinate `i_d floordiv F`. A block axis puts it in bank
/// `i_d floordiv extent` at `i_d mod extent`, `extent = ceil(S_d / F)`. A
/// skewed axis puts it in bank `(sum over all k of i_k) mod F`, keeping `i_d
/// floordiv F` on its distribution dimension `d` and every other subscript
/// whole. Axes compose in mixed radix, in `allo.part` order:
/// `((b_1 * F_2) + b_2) * F_3 + ...`. An axis with `dim == 0` means every
/// dimension, contributing one `Axis` each (`numBanks` is `F^rank`, not `F`); a
/// skew is never spelled that way, its bank already reading every subscript.
///
/// A skew buys conflict freedom, not a compile-time bank, where block and
/// cyclic (functions of one subscript) serve an array read only one way:
/// `A[i][Fj+k]` and `A[Fj+k][i]` each reach F distinct banks as `k` runs over
/// the factor, so an unrolled group takes one port per bank instead of F (see
/// `skewSlotOf`).
struct BankLayout {
  /// How an axis maps a subscript onto banks. `Cyclic` interleaves, `Block`
  /// chunks, `Skew` reads every subscript (see the type comment).
  enum class Kind { Cyclic, Block, Skew };
  struct Axis {
    unsigned dim; // 0-based memref dimension (the distribution dim for a skew)
    int64_t factor; // banks along this dimension
    Kind kind = Kind::Cyclic;
    int64_t extent; // per-bank extent of `dim` == ceil(shape[dim] / factor)
  };
  llvm::SmallVector<Axis, 2> axes; // mixed-radix order, most significant first
  llvm::SmallVector<int64_t, 4> bankShape; // per-bank extents, full memref rank
  unsigned numBanks = 1;                   // product of the axis factors
  bool registers = false;                  // complete partition: no banks

  /// Elements in one bank (the product of `bankShape`).
  int64_t bankWords() const;

  /// The single skewed axis, or null. At most one is allowed: the slot analysis
  /// reasons about one rotation of the bank index.
  const Axis *skew() const;
};

/// `kind` as the interface manifest spells it, the name the host reproduces the
/// decomposition from.
llvm::StringRef bankKindName(BankLayout::Kind kind);

/// Decode a memref's `allo.part` attribute into its element-space bank
/// decomposition (a single unpartitioned bank when there is no attribute). The
/// one decoder of that attribute: a consumer wanting only the bank count or the
/// complete-partition flag reads it here rather than parsing again.
BankLayout bankLayoutOf(Value memRef);

/// One array's storage shape: how it banks, what ports one bank has, and which
/// `dcp.storage` realization it resolves to. The one characterization, billed
/// by the scheduler's port model (`MemoryBankModel`) and built by the microarch
/// datapath (`MemUnit`), so a schedule's reservation and the emitter's wiring
/// cannot drift apart.
struct MemoryChar {
  BankLayout layout; // element-space banks (one when unpartitioned)
  /// Ports one instance of the row holding one bank has: the resolved `storage`
  /// row's, narrowed by the `allo.bind.storage type=` topology. One budget for
  /// the scheduler and the emitter both.
  StoragePorts ports;
  /// Realized as a combinational constant array: the resolved row is the
  /// device's `table` and this carrier owns the array. A parameter is not; its
  /// cells are the caller's and this side holds an addressed port on them.
  bool constantTable = false;
  /// The `dcp.storage` realization recorded for this array (`kStorageAttr`),
  /// read rather than re-resolved. Empty only for the array with nowhere to go,
  /// a complete partition on a device marking no `scatter` row, which
  /// `PreVerification` reports.
  std::string storage;

  /// Whether there is no port here to contend for: a constant table is
  /// combinational, and a complete partition scattered the array into
  /// registers.
  bool unlimited() const { return layout.registers || constantTable; }
};

/// Where an array carries the `dcp.storage` realization it resolved to. On the
/// array's carrier, so `dcp-resolve-banking` copies it onto every bank alloc
/// and a per-bank array answers what the whole one did. Empty names the array
/// with nowhere to go, a complete partition on a device marking no `scatter`
/// row, which `PreVerification` reports.
constexpr llvm::StringLiteral kStorageAttr = "allo.storage";

/// Resolve every array of \p module to a `dcp.storage` realization and record
/// it under `kStorageAttr`. Runs once, before any layer asks what an array was
/// realized as, so no consumer re-runs the cost model and reaches a different
/// answer.
void recordArrayStorage(ModuleOp module, const MemoryLibrary &lib);

/// Characterize a memref's storage shape from its partition attributes and the
/// realization `recordArrayStorage` resolved for it, independent of any
/// scheduling region. \p lib supplies what the device states about that
/// realization, and has to be the same device the access latencies were stamped
/// from, or the two disagree.
MemoryChar characterize(Value memref, const MemoryLibrary &lib);

/// The ports the `allo.bind.storage type=` topology on \p memref asks for, or
/// nullopt where it names none. A constraint rather than a budget: the array's
/// realization decides what it has and this only narrows it. `PreVerification`
/// reports a topology the row cannot meet.
std::optional<StoragePorts> requestedPortsOf(Value memref);

/// The canonical spelling of \p part for a memref of \p type: a complete
/// partition collapses to its one whole-array axis, a `dim == 0` block or
/// cyclic axis expands into one axis per dimension, and the axes are sorted by
/// dimension. Null canonicalizes to null.
///
/// `bankLayoutOf` folds the axes in order into a mixed-radix bank index, so a
/// spelling is part of the bank index function, not presentation: two
/// attributes describing the same banking must be spelled identically before a
/// caller and callee agree (a sub-kernel masters port group `k` of exactly the
/// caller's bank `k`).
PartitionAttr canonicalizePartition(PartitionAttr part, MemRefType type);

/// The coarsest banking of a memref of \p type that satisfies both \p a and
/// \p b, canonical; failure with \p why set when the two cannot be reconciled.
///
/// The order is refinement: `a` is below `b` when every pair of elements `a`
/// places in distinct banks `b` does too. A partition directive is a lower
/// bound on the bank distinctness its kernel needs, so a kernel scheduled
/// against the join still sees every access group it asked to be conflict-free.
/// A complete partition is the top, an absent attribute the bottom (one bank).
///
/// Axes on different dimensions compose in mixed radix with no reconciling. On
/// one dimension the join must remain a single axis (`allo.part` admits no
/// duplicate dimension), so it exists only when one factor divides the other
/// (and, for a block axis, the finer chunk boundaries fall on the coarser
/// ones). A block axis against a cyclic axis has no common single-axis
/// refinement.
llvm::FailureOr<PartitionAttr> joinPartitions(PartitionAttr a, PartitionAttr b,
                                              MemRefType type,
                                              std::string &why);

/// An access's bank index and in-bank offset, as affine expressions over the
/// address map's operands: each partitioned axis contributes its mixed-radix
/// digit to `bank`, and what remains of the subscripts re-linearizes over the
/// per-bank shape into `offset`. \p map is in element space, one result per
/// memref dimension; linearizing happens at the point of use, never in the IR.
///
/// Deriving this on the expression rather than on emitted values makes common
/// banked idioms free: `A[2*i]` under cyclic-2 has bank `(2*i) mod 2` and
/// offset
/// `(2*i) floordiv 2`, which fold to `0` and `i` (no hardware), where the same
/// derivation on emitted values leaves a multiply/mask/shift nothing downstream
/// can fold away.
struct BankSplitExpr {
  AffineExpr bank;   // which of `layout`'s banks, mixed radix in axis order
  AffineExpr offset; // the element's row-major index inside that bank
  /// `offset` before it is linearized: the element's coordinate on each
  /// dimension of the per-bank shape. The static split rewrites an access map
  /// in element space and so needs these rather than their row-major fold.
  llvm::SmallVector<AffineExpr, 4> coords;
};
BankSplitExpr bankSplitOf(const BankLayout &layout, AffineMap map,
                          llvm::ArrayRef<int64_t> shape);

/// The values a map operand takes when a caller knows them: inclusive bounds on
/// the dim standing for it. `known == false` is "anything", for an operand the
/// caller cannot bound.
struct DimRange {
  int64_t lo = 0, hi = 0;
  bool known = false;
};

/// The compile-time bank of an access whose address map is \p map over a memref
/// of \p shape, or nullopt when the bank varies at runtime.
///
/// This is `bankSplitOf(...).bank` when that expression is one value, so the
/// bank a consumer routes to and the bank the port model bills cannot drift
/// apart. A cyclic digit is one value when every variable coefficient of its
/// subscript vanishes modulo the factor.
///
/// \p ranges bounds the dims (the map's own numbering), which a block digit
/// needs: `A[i]` under block-2 of an `i32[16]` is `i floordiv 8`, which folds
/// for no `i` but is constant over every `i` a loop on `[0,8)` produces, so the
/// standard idiom (a loop per block) resolves nothing without it. An empty
/// \p ranges asks the folding question alone.
std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    llvm::ArrayRef<int64_t> shape,
                                    llvm::ArrayRef<DimRange> ranges = {});

/// A skewed access's bank, split into the part it shares with the array's other
/// accesses and the part that distinguishes it.
///
/// A skewed bank is `(sum of the subscripts) mod F`, splitting into a runtime
/// `cls` plus a compile-time constant `slot`, so the bank is `(cls + slot) mod
/// F`. Two accesses whose `cls` agree reach the same bank exactly when their
/// slots do, at every runtime `cls` (the bank index is one rotation of the slot
/// index, a bijection).
///
/// A slot is billable the way a static bank is: `assign-banks` records it in
/// `kBankAttr` and the port model bills a port on it, so F accesses with F
/// distinct slots take one port per bank. The emitter must not route to it
/// directly: the physical bank is the slot rotated by `cls`, known only at run
/// time. `BankLayout::skew()` tells the two readings of `kBankAttr` apart.
struct SkewSlot {
  AffineExpr cls;    // the runtime part of the bank's linear form
  unsigned slot = 0; // its constant part, modulo the factor
};

/// \p map's `SkewSlot` over a skewed \p layout, or nullopt when the layout is
/// not skewed or the sum does not split (a non-affine or dynamic subscript).
/// The caller must check that every access agrees on `cls` before billing the
/// slots, since accesses of different `cls` can collide.
std::optional<SkewSlot> skewSlotOf(const BankLayout &layout, AffineMap map,
                                   llvm::ArrayRef<int64_t> shape);

/// Where an access carries the bank `assign-banks` decided for it, before the
/// schedule is reified. Afterwards the fact lives in the `dcp.load`/`dcp.store`
/// op's own `bank` attribute, which a rewrite cannot silently drop the way a
/// discardable one can.
constexpr llvm::StringLiteral kBankAttr = "allo.bank";

/// The bank \p op was assigned, or nullopt when it reaches every bank of its
/// memref: a roaming subscript, a non-affine index, or an `assign-banks` that
/// never ran. Reads whichever carrier the IR layer uses, so every consumer sees
/// one recorded decision. Nullopt is the conservative answer everywhere (bill,
/// route and address through all banks).
std::optional<unsigned> assignedBankOf(Operation *op);

/// \p map, in element space, rewritten as the single simplified row-major
/// linear element index it addresses. Applied at the point of use by everything
/// needing a flat address, so pricing, strength reduction and the emitter
/// cannot disagree.
///
/// Nothing rewrites the IR with it, deliberately: element space carries
/// per-dimension structure the linear form cannot be simplified back into
/// (`(6i+j) floordiv 6` does not fold to `i` without knowing `j < 6`), which
/// the bank split needs.
///
/// Working on the expression cancels the delinearize/linearize pair of a
/// coalesced nest: `iv -> (iv floordiv N, iv mod N)` composed with
/// `(r, c) -> r*N + c` simplifies back to `iv`, where the same round trip built
/// from `comb` ops is a divider, a modulo and a multiplier.
AffineMap linearizeAccessMap(AffineMap map, llvm::ArrayRef<int64_t> shape);

} // namespace mlir::allo

namespace mlir::allo {

/// Per-bank memory-port model. `observe` every memory access in a scheduling
/// region, `finalize` to `characterize` the arrays behind them, then
/// `resources` gives the port resources one access holds. Each `allo.part` bank
/// is a separate limited resource carrying the array's ports.
///
/// An access holds one port on every bank it can reach: the bank `assign-banks`
/// assigned it, or all of them when assigned none. The latter is not a
/// conservative bound but the crossbar the emitter builds, so a partitioned
/// array under a roaming access sustains one bank's ports, not that times the
/// bank count.
class MemoryBankModel {
public:
  void observe(Operation *op);
  void finalize(const MemoryLibrary &lib);

  /// What one access holds: the port resources, as {resource key, slots per
  /// bank}, one entry per bank it reaches. The limit repeats because it is a
  /// property of the bank, not of the access.
  struct PortDemand {
    llvm::SmallVector<std::pair<std::string, unsigned>> units;
    /// Slots taken on each bank. A read takes one, a write one of every copy
    /// the array is spread over, since the copies hold the same data.
    unsigned slots = 1;
  };
  /// The ports \p op holds at once. Empty when \p op is not a memory access, or
  /// when its storage has no port to contend for (a constant table, a complete
  /// partition).
  PortDemand resources(Operation *op) const;

private:
  llvm::DenseMap<Value, MemoryChar> byMemref;
};

} // namespace mlir::allo

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory resource model: applies the per-memref port/bank model to a scheduling
// problem, the storage twin of `populateOperatorTypes`.
//===----------------------------------------------------------------------===//

/// Assign per-memref memory-port resources to every memory access \p problem
/// holds. A port is a one-cycle reservation whatever its latency
/// (`getResourceCycles`'s default), so no occupancy window is set.
///
/// Two passes over the same operations: the bank model has to see every access
/// of an array before it can say what one holds.
///
/// \p lib is what `characterize` resolves an array's storage row against. The
/// ports billed here do not depend on that row, but drawing them from the same
/// characterization the emitter builds from keeps the two in step.
template <class ProblemT>
void populateMemoryResources(ProblemT &problem, const MemoryLibrary &lib) {
  using namespace circt::scheduling;
  MemoryBankModel banks;
  for (Operation *op : problem.getOperations())
    banks.observe(op);
  banks.finalize(lib);
  for (Operation *op : problem.getOperations()) {
    MemoryBankModel::PortDemand held = banks.resources(op);
    SmallVector<Problem::ResourceType> units;
    for (auto &[key, limit] : held.units) {
      assert(held.slots <= limit &&
             "an access takes more slots than its own budget has, which no "
             "cycle can hold and the greedy placement would search forever "
             "for; every limit is one instance's ports once per copy and a "
             "write takes one of each, so it cannot outgrow the budget");
      Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
      problem.setLimit(rsrc, limit);
      units.push_back(rsrc);
    }
    if (units.empty()) // non-memory, or storage with no port to contend for
      continue;
    problem.setLinkedResourceTypes(op, units);
    problem.setResourceDemand(op, held.slots);
  }
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYMODEL_H

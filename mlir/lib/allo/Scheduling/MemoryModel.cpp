/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h"

#include "allo-c/Schedule.h" // kPartitionAttr, kBindStorageAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloOps.h" // dcp::DCPathStoreOp (post-reification)
#include "allo/Scheduling/AddressModel.h" // simplifiedForHardware
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (the access substrate)

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GlobalOp / GetGlobalOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::allo;

// The storage root an access operates on (views peeled), or null for a
// non-access. Arrays and streams are BOTH port-limited storage: an array by its
// memory ports, a stream by its handshake, a FIFO carrying exactly one transfer
// per end per cycle.
static Value storageOf(Operation *op) {
  auto a = asMemAccess(op);
  return a ? a->root : Value();
}

// Look up attribute \p name on \p memRef's carrier: its defining op, else the
// function-argument attrs if it is a func argument. A `memref.get_global` is a
// REFERENCE to storage, so its carrier is the `memref.global` that declares it,
// which is where the schedule primitives write.
template <typename AttrT>
static AttrT carrierAttr(Value memRef, StringRef name) {
  if (Operation *def = memRef.getDefiningOp()) {
    if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          get, get.getNameAttr());
      assert(global && "get_global references an undefined memref.global");
      return global->getAttrOfType<AttrT>(name);
    }
    return def->getAttrOfType<AttrT>(name);
  }
  // Asked of the `func.func` the scheduler works on and of the `dcp.module` it
  // closes into, so it keys on the interface rather than on either op.
  if (auto barg = dyn_cast<BlockArgument>(memRef))
    if (auto func =
            dyn_cast<FunctionOpInterface>(barg.getOwner()->getParentOp()))
      return llvm::dyn_cast_or_null<AttrT>(
          func.getArgAttr(barg.getArgNumber(), name));
  return {};
}

// The tighter of two limits on one axis; nullopt is no limit and yields.
static std::optional<unsigned> tighter(std::optional<unsigned> a,
                                       std::optional<unsigned> b) {
  if (!a)
    return b;
  if (!b)
    return a;
  return std::min(*a, *b);
}

StoragePorts StoragePorts::meet(const StoragePorts &other) const {
  return {tighter(instReads, other.instReads),
          tighter(instWrites, other.instWrites),
          tighter(instPool, other.instPool), stated || other.stated};
}

bool StoragePorts::holds(unsigned nwrites, unsigned nports) const {
  assert(nports >= nwrites && "a write port is one of the ports built");
  if ((instWrites && nwrites > *instWrites) ||
      (instPool && nwrites > *instPool))
    return false;
  // Buses carrying no write: the reads needing a port of their own somewhere.
  // A copy hosts every write and has what is left over for them.
  unsigned own = nports - nwrites;
  unsigned perCopy = instPool ? *instPool - nwrites : instReads.value_or(own);
  // Unless the ports are stated another copy is always available, so the reads
  // fail only where the writes fill one instance on their own.
  return stated ? own <= perCopy : own == 0 || perCopy > 0;
}

bool StoragePorts::holds(const StoragePorts &want) const {
  unsigned nwrites = want.instWrites.value_or(0);
  // A pooled request spends its pool over both directions; without one they are
  // separate structures and it asks for both at once.
  return holds(nwrites,
               want.instPool.value_or(want.instReads.value_or(0) + nwrites));
}

std::string StoragePorts::describe() const {
  auto axis = [](std::optional<unsigned> n) {
    return n ? std::to_string(*n) : std::string("unlimited");
  };
  std::string s = axis(instReads) + " read / " + axis(instWrites) + " write";
  if (instPool)
    s += " over " + axis(instPool) +
         (*instPool == 1 ? " shared port" : " shared ports");
  return s;
}

BindStorage allo::parseBindStorage(DictionaryAttr bind) {
  BindStorage bs;
  if (!bind)
    return bs;
  // Both vocabularies mirror a Python enum the frontend validates, so every
  // string reaching here is a known case; the optional makes a drifted
  // vocabulary a loud bug instead of a silent fall to the default.
  if (auto ty = bind.getAs<StringAttr>("type")) {
    auto t = ty.getValue();
    auto port =
        llvm::StringSwitch<std::optional<MemoryPortEnum>>(t)
            .Cases({"ram_1p", "rom_1p"}, MemoryPortEnum::SinglePort)
            .Cases({"ram_2p", "ram_s2p"}, MemoryPortEnum::SimpleDualPort)
            // 2 shared R/W ports. `fifo` is not a topology, but a stream is
            // never characterized through here.
            .Cases({"ram_t2p", "ram_1wnr", "rom_2p", "rom_np", "fifo"},
                   MemoryPortEnum::TrueDualPort)
            .Default(std::nullopt);
    assert(port && "unknown allo.bind.storage type= (the frontend's "
                   "BindStorageType vocabulary drifted from this switch)");
    bs.port = port;
  }
  // `impl` NAMES a `dcp.storage` of the device, so there is no table here to
  // drift: a name the device does not declare is reported by `PreVerification`
  // against the array, which is where the user can act on it.
  if (auto im = bind.getAs<StringAttr>("impl"))
    bs.storage = im.getValue();
  return bs;
}

bool allo::topologyCovers(MemoryPortEnum a, MemoryPortEnum b) {
  auto rank = [](MemoryPortEnum p) -> unsigned {
    switch (p) {
    case MemoryPortEnum::SinglePort:
      return 0;
    case MemoryPortEnum::SimpleDualPort:
      return 1;
    case MemoryPortEnum::TrueDualPort:
      return 2;
    }
    llvm_unreachable("unhandled MemoryPortEnum");
  };
  return rank(a) >= rank(b);
}

std::optional<StoragePorts> allo::requestedPortsOf(Value memref) {
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memref, kBindStorageAttr));
  if (!bs.port)
    return std::nullopt;
  // Per bank. A SimpleDualPort (S2P) RAM has one dedicated port of each
  // direction, so its two ends never contend and it declares no pool; every
  // other topology shares its ports between the directions, a ROM spelling
  // asking for the same ports as the RAM it mirrors. `stated` because the
  // request names the ports the array is to have, leaving no room for a copy.
  switch (*bs.port) {
  case MemoryPortEnum::SinglePort:
    return StoragePorts{1u, 1u, 1u, true};
  case MemoryPortEnum::SimpleDualPort:
    return StoragePorts{1u, 1u, std::nullopt, true};
  case MemoryPortEnum::TrueDualPort:
    return StoragePorts{2u, 2u, 2u, true};
  }
  llvm_unreachable("unhandled MemoryPortEnum");
}

std::optional<Attribute> mlir::allo::globalInitOf(Value memRef) {
  auto gg = memRef.getDefiningOp<memref::GetGlobalOp>();
  if (!gg)
    return std::nullopt;
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      gg, gg.getNameAttr());
  assert(global && "get_global references an undefined memref.global");
  if (auto init = global.getInitialValue())
    return *init;
  return std::nullopt;
}

static bool calleeWrites(func::CallOp call, Value memRef);

bool mlir::allo::isConstantTable(Value memRef) {
  if (!globalInitOf(memRef))
    return false;
  // A write is an `affine`/`memref` store before reification and a `dcp.store`
  // after, so cover both. A child that only reads is served off the parent's
  // table, so only a child that writes disqualifies it; a reified instance is
  // opaque here and always does.
  return llvm::none_of(memRef.getUsers(), [&](Operation *u) {
    if (isa<dcp::DCPathStoreOp, dcp::DCPathInstanceOp>(u))
      return true;
    if (auto call = dyn_cast<func::CallOp>(u))
      return calleeWrites(call, memRef);
    auto a = asMemAccess(u);
    return a && a->isWrite;
  });
}

// Whether anything may read this array. An argument and an array crossing into
// a sub-kernel are always taken as read, which keeps the port budget the same
// number on both sides of a call, where the two hold ports on one structure.
static bool mayBeRead(Value memRef) {
  if (isa<BlockArgument>(memRef))
    return true;
  return llvm::any_of(memRef.getUsers(), [](Operation *u) {
    if (isa<func::CallOp, dcp::DCPathInstanceOp>(u))
      return true;
    auto a = asMemAccess(u);
    return a && !a->isWrite;
  });
}

static bool writtenThrough(Value memRef);

// Whether \p call may write \p memRef: a callee with no body may write anything
// it is handed, and one with a body writes it where a parameter it lands on is
// written through.
static bool calleeWrites(func::CallOp call, Value memRef) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee || callee.isExternal())
    return true;
  for (auto [k, actual] : llvm::enumerate(call.getArgOperands()))
    if (actual == memRef && writtenThrough(callee.getArgument(k)))
      return true;
  return false;
}

// Whether \p memRef is written anywhere at or below this point: a store here,
// or a call that writes it. The call graph is acyclic (`PreVerification`
// refuses a cyclic one), so the walk terminates.
static bool writtenThrough(Value memRef) {
  return llvm::any_of(memRef.getUsers(), [&](Operation *u) {
    if (auto call = dyn_cast<func::CallOp>(u))
      return calleeWrites(call, memRef);
    auto a = asMemAccess(u);
    return a && a->isWrite;
  });
}

// Sub-kernel calls that write \p memRef. Two of them run together wherever
// their footprints do not collide, and the array then needs a write port each.
// This function's own stores are not counted: they belong to one accessor the
// schedule serializes against whatever budget the row gives.
static unsigned writingCalls(Value memRef) {
  unsigned n = 0;
  for (Operation *u : memRef.getUsers())
    if (auto call = dyn_cast<func::CallOp>(u))
      n += calleeWrites(call, memRef);
  return n;
}

// The name of the storage realization a memref resolves to, the input to
// per-realization access timing. Five sources, in the order they outrank each
// other:
//
//   * a complete partition takes the device's `scatter` row whatever
//     `bind.storage impl` says, since once every bank holds one word there is
//     no addressed structure left;
//   * else an explicit `bind.storage impl`;
//   * else the `scatter` row again for a local array several sub-kernels write,
//     since an addressed structure has no more write ports than it has ports
//     and only registers take a write decoder per writer. Not for an argument,
//     whose cells this module only holds ports on;
//   * else the device's `default` row where it marks one. Not for a constant
//     table, which is resolved by cost, the `table` row being a realization
//     only it can take;
//   * else the row `rowFor` derives from what the array costs on this part.
//
// An empty result is a device that can hold this array nowhere, which
// `PreVerification` reports.
//
// Called once per array, by `recordArrayStorage`; every later layer reads the
// record it leaves.
static std::string deriveStorage(Value memRef, const MemoryLibrary &lib) {
  BankLayout layout = bankLayoutOf(memRef);
  if (layout.registers)
    return lib.scatterStorage;
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
  if (!bs.storage.empty())
    return bs.storage.str();
  if (!isa<BlockArgument>(memRef) && writingCalls(memRef) > 1)
    return lib.scatterStorage;
  bool canTable = isConstantTable(memRef);
  if (!lib.defaultStorage.empty() && !canTable)
    return lib.defaultStorage;
  auto type = dyn_cast<MemRefType>(memRef.getType());
  if (!type)
    return {};
  // Priced per bank, the structure that gets built: the bank count is a common
  // factor across the rows and cannot reorder them, but a row's tiling minimum
  // is charged once per bank and can.
  return lib.rowFor(layout.bankWords(), datapathWidth(type.getElementType()),
                    globalInitOf(memRef).has_value(), canTable);
}

std::string allo::resolvedStorageOf(Value memRef) {
  auto rec = carrierAttr<StringAttr>(memRef, kStorageAttr);
  assert(rec && "`recordArrayStorage` resolves every array before any layer "
                "asks what it was realized as");
  return rec.str();
}

// Write \p value onto the carrier `carrierAttr` reads.
static void setCarrierAttr(Value memRef, StringRef name, Attribute value) {
  if (Operation *def = memRef.getDefiningOp()) {
    if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          get, get.getNameAttr());
      assert(global && "get_global references an undefined memref.global");
      global->setAttr(name, value);
      return;
    }
    def->setAttr(name, value);
    return;
  }
  auto barg = cast<BlockArgument>(memRef);
  cast<FunctionOpInterface>(barg.getOwner()->getParentOp())
      .setArgAttr(barg.getArgNumber(), name, value);
}

// The parameter each memref argument of \p call binds, with the array passed
// to it. Empty for a callee with no body.
static void calleeParams(func::CallOp call,
                         llvm::SmallVectorImpl<std::pair<Value, Value>> &out) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee || callee.isExternal())
    return;
  for (auto [k, actual] : llvm::enumerate(call.getArgOperands()))
    if (isa<MemRefType>(actual.getType()))
      out.emplace_back(actual, callee.getArgument(k));
}

void allo::recordArrayStorage(ModuleOp module, const MemoryLibrary &lib) {
  // A parameter a call binds does not own its array, and inputs to the
  // resolution such as who writes it are only all visible at the owner. Its
  // record is carried in rather than derived, keeping the two sides of a call
  // one decision.
  llvm::DenseSet<void *> bound;
  llvm::SmallVector<std::pair<Value, Value>> params;
  module.walk([&](func::CallOp call) { calleeParams(call, params); });
  for (auto [actual, param] : params)
    bound.insert(param.getAsOpaquePointer());

  llvm::SmallVector<std::pair<Value, std::string>> work;
  auto own = [&](Value memRef) {
    if (isa<MemRefType>(memRef.getType()) &&
        !bound.contains(memRef.getAsOpaquePointer()))
      work.emplace_back(memRef, deriveStorage(memRef, lib));
  };
  // Every root an access can name: an allocation, a reference to a declared
  // global, or an array a function was handed from outside the module.
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(op))
      own(op->getResult(0));
    if (auto fn = dyn_cast<FunctionOpInterface>(op); fn && !fn.isExternal())
      for (BlockArgument arg : fn.getArguments())
        own(arg);
  });

  while (!work.empty()) {
    auto [memRef, row] = work.pop_back_val();
    setCarrierAttr(memRef, kStorageAttr,
                   StringAttr::get(module.getContext(), row));
    llvm::SmallVector<std::pair<Value, Value>> reached;
    for (Operation *u : memRef.getUsers())
      if (auto call = dyn_cast<func::CallOp>(u))
        calleeParams(call, reached);
    for (auto [actual, param] : reached)
      if (actual == memRef) {
        // A parameter naming its own `impl=` keeps what that resolves to, so
        // one array bound to two rows stays a disagreement
        // `checkArgumentAgreement` reports rather than one silently overridden.
        StringRef bind = boundStorageOf(param);
        work.emplace_back(param, bind.empty() ? row : bind.str());
      }
  }
}

void allo::recordPortSelectArms(ModuleOp module, const MemoryLibrary &lib) {
  // One port bus, keyed by the array it serves, the bank it addresses and the
  // direction it runs in: what `bindMemoryPorts` colours accesses onto.
  using Bus = std::tuple<Value, unsigned, unsigned>;
  llvm::DenseMap<Value, MemoryChar> chars;
  llvm::DenseMap<Bus, unsigned> holders;
  llvm::SmallVector<std::pair<Operation *, Bus>> accesses;
  auto charOf = [&](Value root) {
    auto [it, fresh] = chars.try_emplace(root);
    if (fresh)
      it->second = characterize(root, lib);
    return it->second;
  };

  module.walk([&](Operation *op) {
    // A child holds ports of its own on each array it is handed, and drives
    // them as one more arm of the caller's bus. It may master several groups
    // per direction, which this counts as one.
    if (auto call = dyn_cast<func::CallOp>(op)) {
      llvm::SmallVector<std::pair<Value, Value>> params;
      calleeParams(call, params);
      for (Value actual : llvm::make_first_range(params)) {
        MemoryChar mc = charOf(actual);
        if (mc.unlimited() || mc.layout.skew())
          continue;
        for (unsigned b = 0; b < mc.layout.numBanks; ++b)
          for (unsigned write : {0u, 1u})
            ++holders[{actual, b, write}];
      }
      return;
    }
    std::optional<MemAccess> a = asMemAccess(op);
    if (!a || a->kind != AccessKind::Array)
      return;
    MemoryChar mc = charOf(a->root);
    // A scattered array and a constant table hold no port to share, and a
    // skewed one is read through lanes whose slots already separate its
    // accesses.
    if (mc.unlimited() || mc.layout.skew())
      return;
    std::optional<unsigned> bank = mc.layout.numBanks > 1
                                       ? assignedBankOf(op)
                                       : std::optional<unsigned>(0);
    // An access with no bank of its own reaches every one through the
    // crossbar, which shares its bus with nothing.
    if (!bank)
      return;
    Bus bus{a->root, *bank, static_cast<unsigned>(a->isWrite)};
    ++holders[bus];
    accesses.emplace_back(op, bus);
  });

  Builder builder(module.getContext());
  for (auto [op, bus] : accesses)
    if (unsigned arms = holders.lookup(bus); arms > 1)
      op->setAttr(kSelectArmsAttr, builder.getI32IntegerAttr(arms));
}

unsigned allo::portSelectArmsOf(Operation *op) {
  auto arms = op->getAttrOfType<IntegerAttr>(kSelectArmsAttr);
  return arms ? static_cast<unsigned>(arms.getInt()) : 1;
}

StringRef allo::boundStorageOf(Value memRef) {
  return parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr))
      .storage;
}

void MemoryBankModel::observe(Operation *op) {
  if (Value root = storageOf(op))
    byMemref.try_emplace(root);
}

void MemoryBankModel::finalize(const MemoryLibrary &lib) {
  for (auto &[root, info] : byMemref) {
    // A stream channel is a FIFO, not an array: one transfer per end per cycle,
    // no banking or storage-impl axis to characterize, two independent ends
    // (one port each, no pool) and nothing that copies it. `characterize` would
    // cast its type to MemRefType.
    if (isa<StreamType>(root.getType())) {
      info.ports = {1u, 1u, std::nullopt, true};
      // Its default `layout` is the single unbanked one, and it resolves no
      // `storage` realization, which nothing here asks a stream for.
      continue;
    }
    info = characterize(root, lib);
  }
}

MemoryBankModel::PortDemand MemoryBankModel::resources(Operation *op) const {
  auto memRef = storageOf(op);
  if (!memRef)
    return {};
  auto it = byMemref.find(memRef);
  if (it == byMemref.end())
    return {};
  const MemoryChar &info = it->second;
  if (info.unlimited())
    return {};

  // The pools this access draws from: its own direction's, where the row caps
  // that direction, and the shared pool, where the row has one. An access holds
  // both at once, which makes two writers and a concurrent reader three ports
  // of a block RAM rather than two writes plus one read.
  //
  // Billed in slots, one instance's ports once per copy: a read takes a slot of
  // the one copy that serves it, a write a slot of every copy.
  auto a = asMemAccess(op);
  assert(a && "storageOf named a storage root, so this is a memory access");
  unsigned copies = info.ports.copies();
  SmallVector<std::pair<StringRef, unsigned>> pools;
  if (auto dir = a->isWrite ? info.ports.instWrites : info.ports.instReads)
    pools.emplace_back(a->isWrite ? "_w" : "_r", *dir * copies);
  if (info.ports.instPool)
    pools.emplace_back("_rw", *info.ports.instPool * copies);
  std::string base = "mem_" + std::to_string(hash_value(memRef));

  // The banks this access occupies: its assigned bank alone, or every one of
  // them when it has none and reaches the emitter's crossbar. READ rather than
  // derived, so the ports billed and the routing built are one fact.
  unsigned numBanks = info.layout.numBanks;
  std::optional<unsigned> bank;
  if (numBanks > 1)
    bank = assignedBankOf(op);
  PortDemand out;
  out.slots = a->isWrite ? copies : 1;
  auto take = [&](unsigned k) {
    for (auto &[dir, limit] : pools)
      out.units.emplace_back(base + "_b" + std::to_string(k) + dir.str(),
                             limit);
  };
  if (numBanks == 1 || bank)
    take(bank.value_or(0));
  else
    for (unsigned k = 0; k < numBanks; ++k)
      take(k);
  return out;
}

unsigned mlir::allo::datapathWidth(Type t) {
  if (isa<IndexType>(t))
    return kIndexWidth;
  if (auto f = dyn_cast<FloatType>(t))
    return f.getWidth();
  return cast<IntegerType>(t).getWidth();
}

namespace mlir::allo {

int64_t BankLayout::bankWords() const {
  int64_t n = 1;
  for (int64_t e : bankShape)
    n *= e;
  return n;
}

StringRef bankKindName(BankLayout::Kind kind) {
  switch (kind) {
  case BankLayout::Kind::Cyclic:
    return "cyclic";
  case BankLayout::Kind::Block:
    return "block";
  case BankLayout::Kind::Skew:
    return "skew";
  }
  llvm_unreachable("unhandled bank layout kind");
}

const BankLayout::Axis *BankLayout::skew() const {
  const Axis *found = nullptr;
  for (const Axis &a : axes)
    if (a.kind == Kind::Skew) {
      assert(!found && "a layout carries at most one skewed axis");
      found = &a;
    }
  return found;
}

// The banking a memref's `allo.part` implies, in element space: each block or
// cyclic axis splits its dimension into `factor` banks of
// `ceil(extent/factor)`, the axes composing in mixed radix; a complete
// partition scatters into registers. See BankLayout for the full definition.
BankLayout bankLayoutOf(Value memRef) {
  BankLayout l;
  auto mt = cast<MemRefType>(memRef.getType());
  ArrayRef<int64_t> shape = mt.getShape();
  l.bankShape.assign(shape.begin(), shape.end());
  auto part = carrierAttr<PartitionAttr>(memRef, kPartitionAttr);
  if (!part)
    return l;
  for (PartitionAxisAttr axis : part.getPartitions()) {
    // A complete partition leaves no banked storage to describe, so drop any
    // axis seen so far.
    if (axis.getKind() == PartitionKindEnum::CompletePartition) {
      l.axes.clear();
      l.bankShape.assign(shape.begin(), shape.end());
      l.numBanks = 1;
      l.registers = true;
      return l;
    }
    int64_t f = axis.getFactor();
    BankLayout::Kind kind = axis.getKind() == PartitionKindEnum::BlockPartition
                                ? BankLayout::Kind::Block
                            : axis.getKind() == PartitionKindEnum::SkewPartition
                                ? BankLayout::Kind::Skew
                                : BankLayout::Kind::Cyclic;
    auto addAxis = [&](unsigned d) {
      int64_t extent = (l.bankShape[d] + f - 1) / f;
      l.axes.push_back({d, f, kind, extent});
      l.bankShape[d] = extent;
      l.numBanks *= static_cast<unsigned>(f);
    };
    // `dim == 0` partitions every dimension by this factor (never a skew, whose
    // verifier requires a named distribution dimension).
    if (axis.getDim() == 0)
      for (unsigned d = 0, e = mt.getRank(); d < e; ++d)
        addAxis(d);
    else
      addAxis(static_cast<unsigned>(axis.getDim() - 1));
  }
  return l;
}

//===--------------------------------------------------------------------===//
// The partition lattice: the canonical spelling of a banking, and the coarsest
// banking that satisfies two of them.
//===--------------------------------------------------------------------===//

static bool isCompletePartition(PartitionAttr part) {
  return part && llvm::any_of(part.getPartitions(), [](PartitionAxisAttr a) {
           return a.getKind() == PartitionKindEnum::CompletePartition;
         });
}

static bool hasSkewAxis(PartitionAttr part) {
  return part && llvm::any_of(part.getPartitions(), [](PartitionAxisAttr a) {
           return a.getKind() == PartitionKindEnum::SkewPartition;
         });
}

// The whole-array top, spelled once. `bankLayoutOf` scatters into registers on
// ANY complete axis whatever dimension it names, so normalizing the dimension
// away is what lets two spellings of "registers" compare equal.
static PartitionAttr completePartition(MLIRContext *ctx) {
  return PartitionAttr::get(
      ctx, {PartitionAxisAttr::get(ctx, PartitionKindEnum::CompletePartition,
                                   /*factor=*/0, /*dim=*/0)});
}

PartitionAttr canonicalizePartition(PartitionAttr part, MemRefType type) {
  if (!part)
    return {};
  MLIRContext *ctx = part.getContext();
  if (isCompletePartition(part))
    return completePartition(ctx);
  // `dim == 0` means every dimension, which `bankLayoutOf` expands in
  // increasing dimension order; expanding it here lets an axis list be compared
  // one dimension at a time.
  SmallVector<PartitionAxisAttr, 4> axes;
  for (PartitionAxisAttr axis : part.getPartitions()) {
    if (axis.getDim() != 0) {
      axes.push_back(axis);
      continue;
    }
    for (int64_t d = 1, e = type.getRank(); d <= e; ++d)
      axes.push_back(
          PartitionAxisAttr::get(ctx, axis.getKind(), axis.getFactor(), d));
  }
  llvm::sort(axes, [](PartitionAxisAttr x, PartitionAxisAttr y) {
    return x.getDim() < y.getDim();
  });
  return PartitionAttr::get(ctx, axes);
}

// The single axis refining both \p x and \p y on the dimension they share, of
// static extent \p extent. A cyclic residue class modulo F is a union of the
// classes modulo kF, so a multiple of the factor always refines; a block chunk
// of `ceil(extent / F)` splits into finer chunks only where the division leaves
// no remainder, else a finer chunk straddles a coarser boundary.
static llvm::FailureOr<PartitionAxisAttr> joinAxis(PartitionAxisAttr x,
                                                   PartitionAxisAttr y,
                                                   int64_t extent,
                                                   std::string &why) {
  assert(x.getDim() == y.getDim() && "joining axes of different dimensions");
  assert(x.getKind() != PartitionKindEnum::SkewPartition &&
         y.getKind() != PartitionKindEnum::SkewPartition &&
         "a skew is handled whole-attribute, being its array's only axis");
  llvm::raw_string_ostream os(why);
  if (x.getKind() != y.getKind()) {
    os << "dimension " << x.getDim() << " is "
       << ConvertToPartitionString(x.getKind()) << "-partitioned on one side, "
       << ConvertToPartitionString(y.getKind())
       << " on the other; a chunked layout and an interleaved one place the "
          "same elements in different banks, so no single banking serves both. "
          "Give both sides the same kind, or partition the array with a Skew, "
          "which stays conflict-free along either axis";
    return failure();
  }
  int64_t lo = std::min(x.getFactor(), y.getFactor());
  int64_t hi = std::max(x.getFactor(), y.getFactor());
  if (hi % lo != 0) {
    os << "dimension " << x.getDim() << " is partitioned by " << lo
       << " on one side and by " << hi
       << " on the other; the factors must divide, so that the finer banking "
          "keeps apart everything the coarser one does";
    return failure();
  }
  if (x.getKind() == PartitionKindEnum::BlockPartition &&
      (ShapedType::isDynamic(extent) || extent % hi != 0)) {
    os << "dimension " << x.getDim() << " is block-partitioned by " << lo
       << " on one side and by " << hi << " on the other, but its extent ("
       << extent
       << ") is not a multiple of the larger factor, so the two chunkings cut "
          "the dimension at different points";
    return failure();
  }
  return PartitionAxisAttr::get(x.getContext(), x.getKind(), hi, x.getDim());
}

llvm::FailureOr<PartitionAttr> joinPartitions(PartitionAttr a, PartitionAttr b,
                                              MemRefType type,
                                              std::string &why) {
  a = canonicalizePartition(a, type);
  b = canonicalizePartition(b, type);
  if (!a || a == b)
    return b;
  if (!b)
    return a;
  MLIRContext *ctx = type.getContext();
  // A complete partition is the top: every element its own register, which
  // distinguishes every pair and so serves every consumer.
  if (isCompletePartition(a) || isCompletePartition(b))
    return completePartition(ctx);
  if (hasSkewAxis(a) || hasSkewAxis(b)) {
    llvm::raw_string_ostream(why)
        << "a skew partition must be an array's only axis (its bank already "
           "reads every subscript), so "
        << a << " and " << b << " cannot be combined";
    return failure();
  }
  // Axes on different dimensions compose in mixed radix; only a shared
  // dimension folds into one axis. The ordered map also puts the result in
  // canonical order.
  std::map<int64_t, PartitionAxisAttr> byDim;
  for (PartitionAxisAttr axis : a.getPartitions())
    byDim.emplace(axis.getDim(), axis);
  for (PartitionAxisAttr axis : b.getPartitions()) {
    auto [slot, fresh] = byDim.try_emplace(axis.getDim(), axis);
    if (fresh)
      continue;
    auto joined =
        joinAxis(slot->second, axis, type.getDimSize(axis.getDim() - 1), why);
    if (failed(joined))
      return failure();
    slot->second = *joined;
  }
  SmallVector<PartitionAxisAttr, 4> axes;
  for (auto &[dim, axis] : byDim)
    axes.push_back(axis);
  return PartitionAttr::get(ctx, axes);
}

// The linear form a skewed axis reads its bank digit from: every subscript,
// summed. A skew is the only axis of its layout (`PartitionAttr::verify`), so
// this sees the access's own coordinates rather than a partly peeled set, which
// lets `skewSlotOf` reproduce it from the map alone.
static AffineExpr skewSum(ArrayRef<AffineExpr> coord) {
  AffineExpr s = coord.front();
  for (AffineExpr c : coord.drop_front())
    s = s + c;
  return s;
}

BankSplitExpr bankSplitOf(const BankLayout &layout, AffineMap map,
                          ArrayRef<int64_t> shape) {
  assert(map && "bank split of an access with no address map");
  assert(map.getNumResults() == shape.size() &&
         "an address map is in element space, one result per memref dimension");
  // The per-bank strides below are products of the trailing extents, so a
  // dynamic non-leading dim poisons them.
  assert((shape.empty() ||
          llvm::none_of(shape.drop_front(),
                        [](int64_t d) { return ShapedType::isDynamic(d); })) &&
         "banked addressing needs static non-leading memref dims");
  unsigned rank = shape.size();
  unsigned nd = map.getNumDims(), ns = map.getNumSymbols();
  MLIRContext *ctx = map.getContext();

  SmallVector<AffineExpr> coord(map.getResults());

  // Peel each axis's digit off its own subscript, cyclic taking the residue and
  // block the quotient; a skew reads every subscript and divides only its
  // distribution dimension. The digits compose in mixed radix.
  AffineExpr bank;
  for (const BankLayout::Axis &a : layout.axes) {
    AffineExpr ci = coord[a.dim];
    AffineExpr digit;
    switch (a.kind) {
    case BankLayout::Kind::Block:
      digit = ci.floorDiv(a.extent);
      coord[a.dim] = ci % a.extent;
      break;
    case BankLayout::Kind::Cyclic:
      digit = ci % a.factor;
      coord[a.dim] = ci.floorDiv(a.factor);
      break;
    case BankLayout::Kind::Skew:
      digit = skewSum(coord) % a.factor;
      coord[a.dim] = ci.floorDiv(a.factor);
      break;
    }
    bank = bank ? bank * a.factor + digit : digit;
  }
  if (!bank)
    bank = getAffineConstantExpr(0, ctx); // unpartitioned: the one bank

  // What remains linearizes over the PER-BANK extents, the address space one
  // bank actually has.
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * layout.bankShape[k + 1];
  AffineExpr offset = getAffineConstantExpr(0, ctx);
  for (unsigned k = 0; k < rank; ++k) {
    offset = offset + coord[k] * stride[k];
    coord[k] = simplifiedForHardware(coord[k], nd, ns);
  }

  return {simplifiedForHardware(bank, nd, ns),
          simplifiedForHardware(offset, nd, ns), std::move(coord)};
}

// The interval \p e takes over \p ranges, or nullopt when an operand in it is
// unbounded. Endpoint arithmetic, exact since every operator is monotone in its
// argument; the one over-approximation is a residue whose argument straddles a
// multiple of the divisor, widened to the whole residue class. Widening is
// SOUND: a caller acts on `lo == hi`, so a wider interval only declines.
static std::optional<std::pair<int64_t, int64_t>>
rangeOf(AffineExpr e, ArrayRef<DimRange> ranges) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return std::pair{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    if (d.getPosition() >= ranges.size() || !ranges[d.getPosition()].known)
      return std::nullopt;
    const DimRange &r = ranges[d.getPosition()];
    return std::pair{r.lo, r.hi};
  }
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return std::nullopt; // a symbol: loop-invariant, but not bounded here
  std::optional<std::pair<int64_t, int64_t>> l = rangeOf(bin.getLHS(), ranges),
                                             r = rangeOf(bin.getRHS(), ranges);
  if (!l || !r)
    return std::nullopt;
  if (bin.getKind() == AffineExprKind::Add)
    return std::pair{l->first + r->first, l->second + r->second};
  // Every other operator's right operand is a constant in a well-formed map.
  if (r->first != r->second)
    return std::nullopt;
  int64_t k = r->first;
  if (bin.getKind() == AffineExprKind::Mul)
    return k >= 0 ? std::pair{l->first * k, l->second * k}
                  : std::pair{l->second * k, l->first * k};
  if (k <= 0)
    return std::nullopt;
  int64_t qlo = llvm::divideFloorSigned(l->first, k),
          qhi = llvm::divideFloorSigned(l->second, k);
  if (bin.getKind() == AffineExprKind::FloorDiv)
    return std::pair{qlo, qhi};
  if (bin.getKind() == AffineExprKind::CeilDiv)
    return std::pair{llvm::divideCeilSigned(l->first, k),
                     llvm::divideCeilSigned(l->second, k)};
  assert(bin.getKind() == AffineExprKind::Mod && "unhandled affine operator");
  if (qlo != qhi)
    return std::pair{int64_t{0}, k - 1}; // straddles: the whole residue class
  return std::pair{l->first - qlo * k, l->second - qlo * k};
}

std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    ArrayRef<int64_t> shape,
                                    ArrayRef<DimRange> ranges) {
  if (!map)
    return std::nullopt;
  // "Statically banked" is the bank expression taking ONE value, which a
  // constant fold covers directly and a bounded iteration domain (a block
  // partition's digit) can cover too.
  AffineExpr bank = bankSplitOf(layout, map, shape).bank;
  if (auto cst = dyn_cast<AffineConstantExpr>(bank))
    return cst.getValue();
  if (std::optional<std::pair<int64_t, int64_t>> r = rangeOf(bank, ranges))
    if (r->first == r->second)
      return r->first;
  return std::nullopt;
}

std::optional<SkewSlot> skewSlotOf(const BankLayout &layout, AffineMap map,
                                   ArrayRef<int64_t> shape) {
  const BankLayout::Axis *ax = layout.skew();
  if (!map || !ax)
    return std::nullopt;
  assert(layout.axes.size() == 1 && "a skew is its layout's only axis");
  assert(map.getNumResults() == shape.size() &&
         "an address map is in element space, one result per memref dimension");
  unsigned nd = map.getNumDims(), ns = map.getNumSymbols();
  AffineExpr sum = skewSum(map.getResults());
  // The constant part is what the sum reads with every operand at zero, so the
  // rest is the runtime part. One substitution rather than a walk, which no
  // shape of affine sum can defeat.
  AffineExpr zero = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> zeroDims(nd, zero), zeroSyms(ns, zero);
  auto cst = dyn_cast<AffineConstantExpr>(
      simplifyAffineExpr(sum.replaceDimsAndSymbols(zeroDims, zeroSyms), 0, 0));
  if (!cst)
    return std::nullopt;
  int64_t f = ax->factor;
  return SkewSlot{simplifyAffineExpr(sum - cst.getValue(), nd, ns),
                  static_cast<unsigned>(((cst.getValue() % f) + f) % f)};
}

AffineMap linearizeAccessMap(AffineMap map, ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  assert(map.getNumResults() == rank &&
         "an address map is in element space, one result per memref dimension");
  // Row-major strides are a product of the TRAILING extents, so a dynamic
  // non-leading dim poisons every stride; shape[0] is never read, so a leading
  // dynamic dim is safe.
  assert((shape.empty() ||
          llvm::none_of(shape.drop_front(),
                        [](int64_t d) { return ShapedType::isDynamic(d); })) &&
         "row-major linearization needs static non-leading memref dims");
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * shape[k + 1];
  AffineExpr lin = getAffineConstantExpr(0, map.getContext());
  for (unsigned k = 0; k < rank; ++k)
    lin = lin + map.getResult(k) * stride[k];
  lin = simplifiedForHardware(lin, map.getNumDims(), map.getNumSymbols());
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), lin,
                        map.getContext());
}

std::optional<unsigned> assignedBankOf(Operation *op) {
  // Two carriers, one fact: a discardable attribute while the access is still
  // affine, the reified op's own attribute afterwards.
  IntegerAttr bank;
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    bank = l.getBankAttr();
  else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
    bank = s.getBankAttr();
  else
    bank = op->getAttrOfType<IntegerAttr>(kBankAttr);
  if (!bank)
    return std::nullopt;
  return static_cast<unsigned>(bank.getInt());
}

} // namespace mlir::allo

//===----------------------------------------------------------------------===//
// Memory timing library
//===----------------------------------------------------------------------===//

MemoryLibrary MemoryLibrary::fromModule(ModuleOp module) {
  MemoryLibrary m;
  dcp::DCPathDeviceOp device;
  module.walk([&](dcp::DCPathDeviceOp d) { device = d; });
  if (!device)
    return m;
  // Both rows carry the same four fields under the same accessor names, so one
  // template reads a `dcp.storage` and a `dcp.stream_timing` alike.
  auto timing = [](auto row) {
    MemKindTiming t;
    t.latency.read = (unsigned)row.getRdLatency();
    t.latency.write = (unsigned)row.getWrLatency();
    t.delay.read = row.getRdDelay().convertToDouble();
    t.delay.write = row.getWrDelay().convertToDouble();
    return t;
  };
  auto limit = [](std::optional<int64_t> n) {
    return n ? std::optional<unsigned>((unsigned)*n) : std::nullopt;
  };
  Block &body = device.getBody().front();
  for (auto r : body.getOps<dcp::DCPathResourceOp>())
    m.capacity[r.getSymName()] = r.getCapacity();
  for (auto s : body.getOps<dcp::DCPathStorageOp>()) {
    m.storage.push_back({s.getSymName().str(),
                         timing(s),
                         {limit(s.getInstReads()), limit(s.getInstWrites()),
                          limit(s.getInstPorts())},
                         s.getRamStyle().value_or("").str(),
                         !s.getNoInit(),
                         s.getIsScatter(),
                         s.getIsTable(),
                         s.getUsesAttr(),
                         s.getRdDelayDepthAttr(),
                         s.getRdDelayWidthAttr()});
    if (s.getIsDefault())
      m.defaultStorage = s.getSymName().str();
    if (s.getIsScatter())
      m.scatterStorage = s.getSymName().str();
    if (s.getIsTable())
      m.tableStorage = s.getSymName().str();
  }
  for (auto st : body.getOps<dcp::DCPathStreamTimingOp>()) {
    m.fifo = timing(st);
    // The emitter builds one FIFO shape: `seq.fifo` is show-ahead (the head is
    // on the wire while `valid` is high) and a put commits in one cycle, with
    // no presentation pipeline for anything deeper.
    assert(m.fifo.latency.read == 0 && m.fifo.latency.write == 1 &&
           "a stream row's latencies must match the built FIFO: read 0 "
           "(show-ahead), write 1");
  }
  return m;
}

const StorageRealization *MemoryLibrary::row(StringRef name) const {
  for (const StorageRealization &s : storage)
    if (s.name == name)
      return &s;
  return nullptr;
}

std::optional<double> MemoryLibrary::fractionOfPart(StringRef storage,
                                                    int64_t words,
                                                    unsigned width) const {
  const StorageRealization *s = row(storage);
  if (!s || !s->uses)
    return std::nullopt;
  auto spent = evaluateResourceUse(s->uses, {words, int64_t(width)});
  if (!spent || spent->empty())
    return std::nullopt;
  double worst = 0.0;
  for (auto &[resource, count] : *spent) {
    auto cap = capacity.find(resource.getLeafReference().getValue());
    if (cap == capacity.end() || cap->second <= 0)
      return std::nullopt;
    worst = std::max(worst, double(count) / double(cap->second));
  }
  return worst;
}

double MemoryLibrary::readDelay(StringRef storage, int64_t words,
                                unsigned width) const {
  const StorageRealization *s = row(storage);
  if (!s)
    return 0.0;
  if (!s->rdDelayDepth)
    return s->timing.delay.read;
  // Outside the measured depths the curve holds at its end: a deeper table
  // than was ever built is priced at the deepest one that was.
  auto clampEval = [](CostAttr c, int64_t p) {
    auto [lo, hi] = c.measuredDomain();
    return *c.evaluate(std::clamp(p, lo, hi));
  };
  double d = clampEval(s->rdDelayDepth, words);
  if (s->rdDelayWidth)
    d *= clampEval(s->rdDelayWidth, width);
  return d;
}

std::string MemoryLibrary::rowFor(int64_t words, unsigned width, bool needsInit,
                                  bool canTable) const {
  // Only a row the device can pin is a candidate, so the structure chosen here
  // is the one built. A row with no vendor attribute stays reachable through
  // `bind_storage impl=`; the `table` row carries none, being no array
  // declaration the synthesizer is told anything about.
  llvm::SmallVector<std::pair<const StorageRealization *, double>> viable;
  for (const StorageRealization &s : this->storage) {
    if (s.scatter || (needsInit && !s.canInit))
      continue;
    if (s.table ? !canTable : s.ramStyle.empty())
      continue;
    if (auto cost = fractionOfPart(s.name, words, width))
      viable.emplace_back(&s, *cost);
  }
  if (viable.empty())
    return {};
  // Capacity before latency: a row one bank overflows cannot realize it
  // however fast it reads, so a deeper row that fits (uram past bram's
  // capacity) stays reachable. With nothing fitting, every row stands and the
  // utilization accounting reports the overflow.
  auto fits = [](const auto &v) { return v.second <= 1.0; };
  if (llvm::any_of(viable, fits))
    llvm::erase_if(viable, [&](const auto &v) { return !fits(v); });
  // Read latency first, then write: a deeper row is out however cheap it is.
  // Strictly cheaper to displace, so rows priced the same break by declaration
  // order.
  auto latency = [](const StorageRealization *s) {
    return std::pair(s->timing.latency.read, s->timing.latency.write);
  };
  auto cheapestAtLeastLatency =
      [&](llvm::ArrayRef<std::pair<const StorageRealization *, double>> rows) {
        const StorageRealization *pick = nullptr;
        double best = 0.0;
        auto fastest = latency(llvm::min_element(rows, [&](auto &a, auto &b) {
                                 return latency(a.first) < latency(b.first);
                               })->first);
        for (auto &[s, cost] : rows)
          if (latency(s) == fastest && (!pick || cost < best)) {
            pick = s;
            best = cost;
          }
        return pick;
      };
  // What bounds a constant table's depth. The table is the cheapest row at
  // every shape these parts hold, so cost alone would take it at any size;
  // what grows with the array is its read delay, where an addressed row's is
  // flat. Take it only while it is no slower than the memory that would
  // otherwise hold it.
  llvm::SmallVector<std::pair<const StorageRealization *, double>> memories(
      llvm::make_filter_range(viable,
                              [](const auto &v) { return !v.first->table; }));
  if (!memories.empty()) {
    const StorageRealization *mem = cheapestAtLeastLatency(memories);
    double bar = readDelay(mem->name, words, width);
    llvm::erase_if(viable, [&](const auto &v) {
      return v.first->table && readDelay(v.first->name, words, width) > bar;
    });
  }
  return cheapestAtLeastLatency(viable)->name;
}

MemKindTiming MemoryLibrary::timing(StringRef name) const {
  const StorageRealization *s = row(name);
  // `PreVerification` rejects an array resolving to a realization the device
  // does not declare, so reaching here means that check was bypassed and the
  // access would schedule at latency 0.
  assert(s && "storage realization not declared by the device -> silent "
              "latency-0 access");
  return s ? s->timing : MemKindTiming{};
}

MemoryLibrary::Timing MemoryLibrary::timing(Operation *op) const {
  auto a = asMemAccess(op);
  if (!a)
    return {};
  // A stream is a FIFO, timed by its own row rather than by a realization
  // (which also returns empty for an array with no declared realization).
  // Branch on the access kind, not the resolved name, or the two cases collide.
  std::string name;
  MemKindTiming t = fifo;
  if (a->kind != AccessKind::Stream) {
    name = resolvedStorageOf(a->root);
    // The one way a resolution comes back empty is a completely partitioned
    // array on a device marking no `scatter` row, which `PreVerification`
    // rejects; reaching here means that check was bypassed.
    assert(!name.empty() &&
           "an array access resolves to a storage realization");
    t = timing(name);
    // Priced at the array's own shape where the row's read delay depends on it.
    if (auto mt = dyn_cast<MemRefType>(a->root.getType()))
      t.delay.read = readDelay(name, bankLayoutOf(a->root).bankWords(),
                               datapathWidth(mt.getElementType()));
  }
  return a->isWrite ? Timing{t.latency.write, t.delay.write, name}
                    : Timing{t.latency.read, t.delay.read, name};
}

MemoryChar allo::characterize(Value memref, const MemoryLibrary &lib) {
  MemoryChar c;
  c.layout = bankLayoutOf(memref);
  c.storage = resolvedStorageOf(memref);
  // An argument is never the table itself: the cells are the caller's and this
  // side masters an addressed port on them, which is a port to contend for.
  c.constantTable = lib.isTable(c.storage) && !isa<BlockArgument>(memref);
  // The realization decides what the array has and a `type=` topology narrows
  // it, the meet keeping the tighter of the two. `PreVerification` reports a
  // topology asking for more than the row has. An argument is held in the same
  // budget as the structure backing it upstream, the boundary publishing one
  // interface per port bound here.
  if (const StorageRealization *row = lib.row(c.storage))
    c.ports = row->ports;
  if (auto want = requestedPortsOf(memref))
    c.ports = c.ports.meet(*want);
  // Writes filling a pooled row's ports would leave a read no port on any copy
  // and the array would fit nowhere. Reserve one port of the pool for the reads
  // and let the schedule serialize the writes against it. Skipped where the
  // ports are `stated`, the user having named the topology.
  if (c.ports.instPool && !c.ports.stated && mayBeRead(memref))
    c.ports.instWrites =
        std::max(1u, std::min(c.ports.instWrites.value_or(*c.ports.instPool),
                              *c.ports.instPool - 1));
  return c;
}

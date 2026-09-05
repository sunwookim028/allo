/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // addressDelayOf (per-site address)
#include "allo/Support/BitAnalysis.h"     // isBitRename
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Native realizations: the one table the three views below are generated from.
//
// A row is (`CombOpKindEnum` case, abstract `OpKind` case, the MLIR ops it
// realizes). One table rather than three switches, because the three have to
// agree: `classify(op) == opKindOf(*combKindOf(op))` wherever an op has a
// native lowering, and nothing outside the table could enforce that. Adding a
// native operator is one row here plus one `emitCompute` case.
//
// A row's kind is FINER than its abstract kind, which is what the two columns
// are for: a device characterizes "an integer add", while the emitter has to
// know it is emitting `addi` and not `subi`.
//===----------------------------------------------------------------------===//

#define ALLO_COMB_KINDS(X)                                                     \
  X(Addi, Add, arith::AddIOp)                                                  \
  X(Subi, Sub, arith::SubIOp)                                                  \
  X(Muli, Mul, arith::MulIOp)                                                  \
  X(Divsi, Div, arith::DivSIOp)                                                \
  X(Divui, Div, arith::DivUIOp)                                                \
  X(Remsi, Rem, arith::RemSIOp)                                                \
  X(Remui, Rem, arith::RemUIOp)                                                \
  X(Andi, And, arith::AndIOp)                                                  \
  X(Ori, Or, arith::OrIOp)                                                     \
  X(Xori, Xor, arith::XOrIOp)                                                  \
  X(Shli, Shl, arith::ShLIOp)                                                  \
  X(Shrsi, Shr, arith::ShRSIOp)                                                \
  X(Shrui, Shr, arith::ShRUIOp)                                                \
  X(Cmpi, Cmp, arith::CmpIOp)                                                  \
  X(Select, Select, arith::SelectOp)                                           \
  X(Extsi, ICastI, arith::ExtSIOp)                                             \
  X(Extui, ICastI, arith::ExtUIOp)                                             \
  X(Trunci, ICastI, arith::TruncIOp)                                           \
  X(IndexCast, ICastI, arith::IndexCastOp)                                     \
  X(IndexCastUi, ICastI, arith::IndexCastUIOp)                                 \
  X(Negf, Neg, arith::NegFOp)                                                  \
  X(Minsi, Min, arith::MinSIOp)                                                \
  X(Minui, Min, arith::MinUIOp)                                                \
  X(Maxsi, Max, arith::MaxSIOp)                                                \
  X(Maxui, Max, arith::MaxUIOp)                                                \
  /* An address expression: no device row covers a whole affine map, so    */  \
  /* `lookup` prices its cone through the address model (`addressCost`)    */  \
  X(Apply, Unknown, affine::AffineApplyOp)

std::optional<CombOpKindEnum> mlir::allo::combKindOf(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<CombOpKindEnum>>(op)
#define X(comb, abstract, ...)                                                 \
  .Case<__VA_ARGS__>([](auto) { return CombOpKindEnum::comb; })
      ALLO_COMB_KINDS(X)
#undef X
          .Default([](auto) { return std::nullopt; });
}

OpKind mlir::allo::opKindOf(CombOpKindEnum kind) {
  switch (kind) {
#define X(comb, abstract, ...)                                                 \
  case CombOpKindEnum::comb:                                                   \
    return OpKind::abstract;
    ALLO_COMB_KINDS(X)
#undef X
  }
  llvm_unreachable("every comb realization names an abstract kind or Unknown");
}

//===----------------------------------------------------------------------===//
// Classification: concrete IR op -> abstract kind
//
// Total, so it also covers what has no native lowering: float arithmetic, the
// float casts, and the composite integer kinds `legalize-arith` expands.
// `Unknown` for everything else, an access included: an access is timed by its
// storage (`accessCharacterization`), so no operator row answers for it.
//===----------------------------------------------------------------------===//

OpKind mlir::allo::classify(Operation *op) {
  return llvm::TypeSwitch<Operation *, OpKind>(op)
#define X(comb, abstract, ...)                                                 \
  .Case<__VA_ARGS__>([](auto) { return OpKind::abstract; })
      ALLO_COMB_KINDS(X)
#undef X
          .Case<arith::AddFOp>([](auto) { return OpKind::Add; })
          .Case<arith::SubFOp>([](auto) { return OpKind::Sub; })
          .Case<arith::MulFOp>([](auto) { return OpKind::Mul; })
          .Case<arith::DivFOp>([](auto) { return OpKind::Div; })
          .Case<arith::RemFOp>([](auto) { return OpKind::Rem; })
          .Case<arith::MaximumFOp>([](auto) { return OpKind::Max; })
          .Case<arith::MinimumFOp>([](auto) { return OpKind::Min; })
          .Case<arith::MaxNumFOp>([](auto) { return OpKind::MaxNum; })
          .Case<arith::MinNumFOp>([](auto) { return OpKind::MinNum; })
          .Case<arith::CeilDivSIOp, arith::CeilDivUIOp>(
              [](auto) { return OpKind::CeilDiv; })
          .Case<arith::FloorDivSIOp>([](auto) { return OpKind::FloorDiv; })
          .Case<arith::CmpFOp>([](auto) { return OpKind::Cmp; })
          .Case<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                arith::FPToUIOp>([](auto) { return OpKind::FCastI; })
          .Case<arith::ExtFOp, arith::TruncFOp>(
              [](auto) { return OpKind::FCastF; })
          .Default([](auto) { return OpKind::Unknown; });
}

//===----------------------------------------------------------------------===//
// Matching helpers
//===----------------------------------------------------------------------===//

namespace {

// The element type of each shaped type in `types`, else the type itself: what
// an IP row is matched against.
llvm::SmallVector<Type> elementTypes(TypeRange types) {
  llvm::SmallVector<Type> out;
  for (Type t : types) {
    if (auto sh = dyn_cast<ShapedType>(t))
      t = sh.getElementType();
    out.push_back(t);
  }
  return out;
}

// Whether every data operand of `op` has integer element type: what an
// integer-arithmetic comb row matches on. An `index` counts, and has to: a
// bound, a counter and an address are index-typed, and a row that skipped them
// would leave the device's own adder and divider priced at the DEFAULT row,
// which is 0.1 ns whatever it builds.
bool allIntegerOperands(Operation *op) {
  auto ts = elementTypes(op->getOperandTypes());
  return !ts.empty() && llvm::all_of(ts, [](Type t) {
    return isa<IntegerType, IndexType>(t);
  });
}

// The significant bits \p op's operands actually carry: an extension counts
// its source (a zero-extension plus the sign bit it pins), a constant its own
// significant bits, anything else its full width. Gates a `fed_width` row,
// which holds only where synthesis sees the extensions.
unsigned fedWidthOf(Operation *op) {
  unsigned fed = 1;
  for (Value v : op->getOperands()) {
    unsigned w = datapathWidth(v.getType());
    APInt cst;
    if (matchPattern(v, m_ConstantInt(&cst)))
      w = cst.getSignificantBits();
    else if (Operation *d = v.getDefiningOp()) {
      if (isa<arith::ExtSIOp>(d))
        w = datapathWidth(d->getOperand(0).getType());
      else if (isa<arith::ExtUIOp>(d))
        w = datapathWidth(d->getOperand(0).getType()) + 1;
    }
    fed = std::max(fed, w);
  }
  return fed;
}

// Every library row that could realize \p op, in declaration order: an advanced
// row matches its raw mnemonic and exact element-type list, an IP row its
// abstract kind and exact element-type list, a comb row its abstract kind.
// A `fedWidth` row also needs the operands proven that narrow.
// `selectImplementation` ranks the result.
llvm::SmallVector<const OperatorEntry *, 2>
matchEntries(const std::vector<OperatorEntry> &advanced,
             const std::vector<OperatorEntry> &entries, Operation *op) {
  llvm::SmallVector<const OperatorEntry *, 2> out;
  auto kind = classify(op);
  auto mnem = op->getName().stripDialect();
  auto aTys = elementTypes(op->getOperandTypes());
  auto rTys = elementTypes(op->getResultTypes());
  ArrayRef<Type> a = aTys, r = rTys;
  for (const OperatorEntry &e : advanced)
    if (e.mlirOp == mnem && ArrayRef<Type>(e.argTypes) == a &&
        ArrayRef<Type>(e.resTypes) == r)
      out.push_back(&e);
  for (const OperatorEntry &e : entries) {
    if (e.kind != kind)
      continue;
    if (e.comb) {
      // `select`/`neg` comb rows match any operand type: a mux over any
      // datatype, a float sign flip. Every other comb kind is integer
      // arithmetic.
      if (kind == OpKind::Select || kind == OpKind::Neg ||
          allIntegerOperands(op))
        out.push_back(&e);
    } else if (ArrayRef<Type>(e.argTypes) == a &&
               ArrayRef<Type>(e.resTypes) == r) {
      if (!e.fedWidth || fedWidthOf(op) <= e.fedWidth)
        out.push_back(&e);
    }
  }
  return out;
}

// Whether \p op needs an IP realization: a float arithmetic op or compare
// other than neg/select, any cast to or from float, or a math.* advanced op.
bool needsIP(Operation *op) {
  auto isFloat = [](Type t) { return isa<FloatType>(t); };
  bool floaty = llvm::any_of(elementTypes(op->getOperandTypes()), isFloat) ||
                llvm::any_of(elementTypes(op->getResultTypes()), isFloat);
  switch (classify(op)) {
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Rem:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::MaxNum:
  case OpKind::MinNum:
  case OpKind::Cmp:
    return floaty;
  case OpKind::CeilDiv:
  case OpKind::FloorDiv:
    // No native comb realization; `legalize-arith` expands these unless the
    // device provides an IP, so one reaching the scheduler must be an IP.
    return true;
  case OpKind::FCastI:
  case OpKind::FCastF:
    return true;
  case OpKind::Unknown:
    // `allo.muladd` exists only where a device row matched it, so it has no
    // combinational fallback.
    return isa<math::MathDialect>(op->getDialect()) || isa<MulAddOp>(op);
  default:
    return false;
  }
}

// The identity of the unit \p op runs on: a native \p comb realization or the
// `dcp.operator` \p symbol, exactly one of which a caller gives. Empty without
// either, or when \p op is not the single-result compute a `FuncUnit` is built
// from.
OperatorIdentity identityOf(Operation *op, std::optional<CombOpKindEnum> comb,
                            StringRef symbol) {
  assert((!comb || symbol.empty()) && "a compute takes one realization path");
  OperatorIdentity id;
  if ((!comb && symbol.empty()) || op->getNumResults() != 1)
    return id;
  id.comb = comb;
  id.ipSymbol = symbol.str();
  id.argTypes.assign(op->getOperandTypes().begin(),
                     op->getOperandTypes().end());
  id.resultType = op->getResult(0).getType();
  id.predicate = op->getAttr("predicate");
  id.map = op->getAttr("map");
  // On an arith op the rename is decided here; a `dcp.compute` carries the
  // decision as the attribute `reify` stamped.
  id.rename = op->hasAttr("rename") || isZeroDelay(op);
  return id;
}

} // namespace

//===----------------------------------------------------------------------===//
// Building the library from injected `dcp.device` / `dcp.operator` IR
//===----------------------------------------------------------------------===//

OperatorLibrary OperatorLibrary::fromModule(ModuleOp module) {
  OperatorLibrary lib;
  // The default row: ops that match nothing (constants, address arithmetic) are
  // 0-latency combinational.
  lib.defaultEntry.latency = 0;
  lib.defaultEntry.inDelay = lib.defaultEntry.outDelay = 0.1;

  dcp::DCPathDeviceOp device;
  module.walk([&](dcp::DCPathDeviceOp d) { device = d; });

  // Comb and IP rows share `entries`, both candidates for an operation of their
  // kind. The last comb row of a kind wins over an earlier one;
  // `selectImplementation` decides the rest.
  if (device) {
    lib.regFloor = device.getRegDelay().convertToDouble();
    for (dcp::DCPathCombOp comb :
         device.getBody().getOps<dcp::DCPathCombOp>()) {
      OperatorEntry e;
      e.kind = comb.getKind();
      e.comb = true;
      e.latency = 0;
      // Left as a curve: the width to evaluate it at is the matched
      // operation's, which `lookup` knows and this does not.
      e.delay = comb.getDelayAttr();
      e.uses = comb.getUsesAttr();
      lib.entries.push_back(std::move(e));
    }

    // The currency: the most plentiful resource sets the scale, so a price is
    // how scarce a resource is relative to the one the part has most of. A
    // declared weight scales that price further.
    int64_t widest = 1;
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>())
      widest = std::max<int64_t>(widest, r.getCapacity());
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>()) {
      int64_t price =
          std::max<int64_t>(1, llvm::divideNearest<int64_t>(
                                   kPriceResolution * widest, r.getCapacity()));
      if (auto w = r.getWeight())
        price =
            std::max<int64_t>(1, std::llround(price * w->convertToDouble()));
      lib.resourcePrices[r.getSymName()] = price;
    }
    for (auto m : device.getBody().getOps<dcp::DCPathMuxOp>()) {
      lib.muxUses = m.getUsesAttr();
      lib.muxDelay = m.getDelayAttr();
      lib.muxDelayWidth = m.getDelayWidthAttr();
    }
    for (auto c : device.getBody().getOps<dcp::DCPathChainOp>())
      lib.chainUses = c.getUsesAttr();
  }

  // IP rows in injection order, built-in then user. Two cores of one kind and
  // signature are both candidates; the ranking, not the order, picks one.
  module.walk([&](dcp::DCPathOperatorOp op) {
    OperatorEntry e;
    e.latency = (uint32_t)op.getLatency();
    e.pipelined = op.getPipelined();
    e.inDelay = op.getInDelay().convertToDouble();
    e.outDelay = op.getOutDelay().convertToDouble();
    e.minPeriod = op.getMinPeriod().convertToDouble();
    e.symbol = op.getSymName().str();
    e.uses = op.getUsesAttr();
    e.fedWidth = (unsigned)op.getFedWidth().value_or(0);
    auto sig = op.getSignature();
    e.argTypes = elementTypes(sig.getInputs());
    e.resTypes = elementTypes(sig.getResults());
    if (std::optional<OpKind> kind = symbolizeOpKindEnum(op.getKind())) {
      e.kind = *kind;
      lib.entries.push_back(std::move(e));
    } else {
      e.mlirOp = op.getKind().str(); // advanced: matched by stripped mnemonic
      lib.advancedEntries.push_back(std::move(e));
    }
  });
  return lib;
}

//===----------------------------------------------------------------------===//
// Lookup
//===----------------------------------------------------------------------===//

const OperatorEntry *OperatorLibrary::selectImplementation(
    ArrayRef<const OperatorEntry *> candidates, int64_t width) const {
  // What a row needs for a cycle of its own, in the same float arithmetic the
  // derate walk prices it at (`minSchedulablePeriod`).
  auto needOf = [floor = static_cast<float>(regFloor)](const OperatorEntry *e) {
    return periodNeed(floor, e->inDelay, e->outDelay, e->minPeriod);
  };
  // Fitting the selection period ranks first, so a deep pipeline beats a
  // shallow core the clock cannot hold; among misses the least need wins,
  // which is what the period derates to. Then shortest, then cheapest at this
  // width, then the first symbol: a total order over the IPs, independent of
  // injection order. A core the device did not measure at this width ranks
  // last rather than free.
  auto rank = [&](const OperatorEntry *e) {
    float need = needOf(e);
    bool unfit = need > selectionPeriodNs;
    return std::make_tuple(
        unfit, unfit ? need : 0.0f, e->latency,
        priceOf(e->uses, {width}).value_or(std::numeric_limits<int64_t>::max()),
        StringRef(e->symbol));
  };
  const OperatorEntry *best = nullptr;
  for (const OperatorEntry *e : candidates) {
    assert((e->comb || !e->symbol.empty()) && "an IP row carries its symbol");
    if (!e->comb && (!best || rank(e) < rank(best)))
      best = e;
  }
  if (best)
    return best;
  // No IP: the combinational row, the last of its kind the device declared,
  // which is what `combEntry` reads.
  for (const OperatorEntry *e : candidates)
    if (e->comb)
      best = e;
  return best;
}

const OperatorEntry *OperatorLibrary::combEntry(OpKind kind) const {
  // `dcp.device.comb` is a dictionary, so there is at most one row per kind in
  // practice.
  const OperatorEntry *found = nullptr;
  for (const OperatorEntry &e : entries)
    if (e.comb && e.kind == kind)
      found = &e;
  return found;
}

double OperatorLibrary::combDelay(CombOpKindEnum kind, int64_t width) const {
  const OperatorEntry *e = combEntry(opKindOf(kind));
  if (!e)
    return defaultEntry.outDelay;
  std::optional<double> d = e->delay.evaluate(width);
  assert(d && "a realization the scheduler priced through `lookup` is inside "
              "the row's measured widths");
  return *d;
}

double OperatorLibrary::combDelay(OpKind kind, int64_t width) const {
  const OperatorEntry *e = combEntry(kind);
  if (!e)
    return 0.0;
  assert(!e->delay.unmeasuredAt(width) &&
         "the compiler builds no structure wider than the widths a device row "
         "was measured at; a program width is checked in `lookup`");
  return *e->delay.evaluate(width);
}

double OperatorLibrary::combMarginalDelay(OpKind kind, int64_t width) const {
  return std::max(0.0, combDelay(kind, width) - regFloor);
}

std::optional<double> OperatorLibrary::measuredCombDelay(OpKind kind,
                                                         int64_t width) const {
  const OperatorEntry *e = combEntry(kind);
  if (!e || e->delay.unmeasuredAt(width))
    return std::nullopt;
  return *e->delay.evaluate(width);
}

unsigned OperatorLibrary::maxPipelinedMulWidth() const {
  unsigned w = 0;
  for (const OperatorEntry &e : entries)
    if (e.kind == OpKind::Mul && !e.comb && e.pipelined &&
        !e.argTypes.empty() &&
        llvm::all_of(e.argTypes, [](Type t) { return isa<IntegerType>(t); }))
      w = std::max(w, cast<IntegerType>(e.argTypes[0]).getWidth());
  return w;
}

unsigned OperatorLibrary::smallestAdvancedRowWidth(llvm::StringRef mnem,
                                                   unsigned width) const {
  unsigned best = 0;
  for (const OperatorEntry &e : advancedEntries) {
    if (e.mlirOp != mnem || e.argTypes.empty())
      continue;
    auto ity = dyn_cast<IntegerType>(e.argTypes[0]);
    if (!ity || ity.getWidth() < width)
      continue;
    if (!best || ity.getWidth() < best)
      best = ity.getWidth();
  }
  return best;
}

bool OperatorLibrary::hasAdvancedRow(llvm::StringRef mnem, TypeRange args,
                                     TypeRange results) const {
  auto aTys = elementTypes(args);
  auto rTys = elementTypes(results);
  return llvm::any_of(advancedEntries, [&](const OperatorEntry &e) {
    return e.mlirOp == mnem &&
           ArrayRef<Type>(e.argTypes) == ArrayRef<Type>(aTys) &&
           ArrayRef<Type>(e.resTypes) == ArrayRef<Type>(rTys);
  });
}

std::optional<int64_t> OperatorLibrary::advancedRowPrice(llvm::StringRef mnem,
                                                         TypeRange args,
                                                         TypeRange results,
                                                         int64_t width) const {
  auto aTys = elementTypes(args);
  auto rTys = elementTypes(results);
  std::optional<int64_t> best;
  for (const OperatorEntry &e : advancedEntries) {
    if (e.mlirOp != mnem ||
        ArrayRef<Type>(e.argTypes) != ArrayRef<Type>(aTys) ||
        ArrayRef<Type>(e.resTypes) != ArrayRef<Type>(rTys))
      continue;
    if (std::optional<int64_t> p = priceOf(e.uses, {width}))
      best = best ? std::min(*best, *p) : *p;
  }
  return best;
}

double OperatorLibrary::combMarginalDelay(CombOpKindEnum kind,
                                          int64_t width) const {
  return std::max(0.0, combDelay(kind, width) - regFloor);
}

// The datapath's width for \p t, or 0 for a type it carries no value of (a
// stream handle). Pricing an `index` at `datapathWidth` is exact where the
// carrier is full width and conservative where the emitter narrows it
// (`counterWidth`, `AddrStride::width`), which never widens it.
static int64_t paramWidthOf(Type t) {
  return t.isIntOrIndexOrFloat() ? datapathWidth(t) : 0;
}

int64_t mlir::allo::combParamWidth(Operation *op) {
  int64_t width = 0;
  for (Type t : elementTypes(op->getOperandTypes()))
    width = std::max(width, paramWidthOf(t));
  if (width)
    return width;
  for (Type t : elementTypes(op->getResultTypes()))
    width = std::max(width, paramWidthOf(t));
  return width ? width : 1;
}

bool mlir::allo::isZeroDelay(Operation *op) {
  if (isBitRename(op))
    return true;
  // A resize the datapath carries at one width on both sides is the identity:
  // `uarch::resize` hands the value back and builds nothing. An `index` is
  // carried at `kIndexWidth`, so an `index_cast` to or from an integer that
  // wide is a wire; a real narrowing or widening is not.
  if (!isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
           arith::ExtUIOp, arith::TruncIOp>(op))
    return false;
  return paramWidthOf(op->getOperand(0).getType()) ==
         paramWidthOf(op->getResult(0).getType());
}

std::optional<int64_t>
OperatorLibrary::priceOf(ArrayAttr uses, ArrayRef<int64_t> params) const {
  auto spent = evaluateResourceUse(uses, params);
  if (!spent)
    return std::nullopt;
  int64_t total = 0;
  for (auto [resource, count] : *spent) {
    auto it = resourcePrices.find(resource.getLeafReference().getValue());
    assert(it != resourcePrices.end() &&
           "a realization spends a resource the device does not declare, which "
           "the dialect verifier resolves before this point");
    assert(count >= 0 && "a realization spends a negative resource count");
    total += it->second * count;
  }
  return total;
}

int64_t OperatorLibrary::muxPrice(int64_t sources, int64_t width) const {
  std::optional<int64_t> price = priceOf(muxUses, {sources, width});
  assert(price && "a multiplexer row holds over every fan-in a region can "
                  "share an operator over");
  return *price;
}

int64_t OperatorLibrary::instancePrice(const OperatorIdentity &identity,
                                       int64_t width) const {
  // A rename is wiring: pricing it as its mnemonic's row would make folding
  // two renames look like saving a real structure.
  if (identity.rename)
    return 0;
  const OperatorEntry *e = nullptr;
  if (identity.comb) {
    e = combEntry(opKindOf(*identity.comb));
  } else {
    // An IP symbol is unique across the device, so a match is the row.
    for (const OperatorEntry &row : advancedEntries)
      if (row.symbol == identity.ipSymbol)
        e = &row;
    for (const OperatorEntry &row : entries)
      if (row.symbol == identity.ipSymbol)
        e = &row;
  }
  if (!e)
    return 0;
  std::optional<int64_t> price = priceOf(e->uses, {width});
  assert(price && "a realization the scheduler priced through `lookup` is "
                  "inside the row's measured widths");
  return *price;
}

unsigned mlir::allo::muxLevels(unsigned sources) {
  return sources <= 1 ? 0 : llvm::Log2_32_Ceil(sources);
}

double mlir::allo::muxCone(const OperatorLibrary &lib, unsigned sources,
                           unsigned width) {
  if (sources <= 1)
    return 0.0;
  CostAttr row = lib.muxDelayRow();
  if (!row)
    // Unmeasured device: the OR row per level with a margin, which over-counts
    // a routed cone two- to three-fold on the measured fabrics.
    return muxLevels(sources) * kMuxDelayMargin * lib.combDelay(OpKind::Or, 1);
  auto clampEval = [](CostAttr c, int64_t p) {
    auto [lo, hi] = c.measuredDomain();
    return *c.evaluate(std::clamp(p, lo, hi));
  };
  double d = clampEval(row, sources);
  if (CostAttr wf = lib.muxDelayWidthRow())
    d *= clampEval(wf, width);
  return d;
}

int64_t OperatorLibrary::chainPrice(int64_t depth, int64_t width) const {
  // A chain of no stages is a wire. The device row characterizes a structure
  // that exists, so its head and tail terms are not zero at depth zero.
  if (depth <= 0)
    return 0;
  std::optional<int64_t> price = priceOf(chainUses, {depth, width});
  assert(price && "a delay chain row holds over every depth a schedule can "
                  "carry a value across");
  return *price;
}

int64_t OperatorLibrary::pulsePrice() const {
  return chainPrice(2, 1) - chainPrice(1, 1);
}

double mlir::allo::portSelectDelay(Operation *op, const OperatorLibrary &lib) {
  unsigned arms = portSelectArmsOf(op);
  if (arms < 2)
    return 0.0;
  std::optional<MemAccess> a = asMemAccess(op);
  assert(a && a->kind == AccessKind::Array &&
         "only an array access is coloured onto a port bus");
  auto type = cast<MemRefType>(a->root.getType());
  // The address bus is as wide as one bank's word count; a write also selects
  // the datum, and one delay covers the wider of the two.
  unsigned width = llvm::Log2_64_Ceil(
      std::max<int64_t>(2, bankLayoutOf(a->root).bankWords()));
  if (a->isWrite)
    width = std::max(width, datapathWidth(type.getElementType()));
  return quantizeCone(muxCone(lib, arms, width));
}

NodeTiming mlir::allo::accessCharacterization(Operation *op,
                                              const OperatorLibrary &opLib,
                                              const MemoryLibrary &memLib) {
  std::optional<MemAccess> a = asMemAccess(op);
  assert(a && "accessCharacterization was handed something that is not an "
              "access");
  MemoryLibrary::Timing t = memLib.timing(op);
  NodeTiming c;
  bool stream = a->kind == AccessKind::Stream;
  c.typeName = stream ? (a->isWrite ? "srm.wr" : "srm.rd")
                      : (a->isWrite ? "mem.wr" : "mem.rd");
  if (!stream) {
    assert(!t.storage.empty() &&
           "an array access resolves to a storage realization");
    c.typeName += t.storage;
  }
  c.latency = t.latency;
  c.inDelay = c.outDelay = t.delay;
  // A cone carries no dependence, so charge its delay to the port it feeds and
  // to the type name (else two sites costing differently share one row). A
  // registered port takes it on its input side alone; a zero-latency port has
  // none, and CIRCT requires its two delays to agree, so there it lands on
  // both.
  auto addCone = [&](double d, std::string suffix) {
    c.inDelay += d;
    if (c.latency == 0)
      c.outDelay += d;
    c.typeName += suffix;
  };
  // The address cone in front of the port.
  if (double addr = quantizeCone(addressDelayOf(op, opLib)))
    addCone(addr, "@" + llvm::formatv("{0:F2}", addr).str());
  // The port-select cone the port colouring grows in front of the bus, reserved
  // here so the cut leaves room for it.
  if (double sel = portSelectDelay(op, opLib))
    addCone(sel,
            llvm::formatv("/{0}:1@{1:F2}", portSelectArmsOf(op), sel).str());
  return c;
}

OperatorChar OperatorLibrary::lookup(Operation *op) const {
  // Neither a sub-kernel call nor a memory access is an operator: no device
  // row, identity, or price. A call's length is its callee's own schedule
  // (`scheduledCallLatency`), an access's is its storage's
  // (`accessCharacterization`); each caller decides what that means for its
  // own question.
  assert(!isSyncSubKernelCall(op) &&
         "the operator library was asked to time a sub-kernel call");
  assert(!asMemAccess(op) &&
         "the operator library was asked to time a memory access");

  // Every row is characterized over one parameter, an operand width.
  int64_t width = combParamWidth(op);
  const OperatorEntry *e =
      selectImplementation(matchEntries(advancedEntries, entries, op), width);
  if (!e) {
    // A standalone apply's hardware is its map's cone (`evalAffine`), so it is
    // priced by the address model: marginal delay at the index width, area
    // from the operators the cone instantiates. The type name carries the
    // quantized delay; allocation still groups by identity, which holds the
    // map, so only same-map applies share a unit.
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
      AddressCost cost =
          addressCost(applyExprOf(apply.getAffineMap()), addressDelaysOf(*this),
                      AddressDelays::refWidth);
      double delay = quantizeCone(cost.delay);
      auto combPrice = [&](OpKind kind) -> int64_t {
        const OperatorEntry *row = combEntry(kind);
        if (!row)
          return 0;
        std::optional<int64_t> p =
            priceOf(row->uses, {AddressDelays::refWidth});
        assert(p && "a comb row an address cone builds from is measured at the "
                    "index width");
        return *p;
      };
      OperatorChar c;
      c.timing.typeName = "comb.apply@" + llvm::formatv("{0:F2}", delay).str();
      c.timing.latency = 0;
      c.timing.inDelay = c.timing.outDelay = delay;
      c.price = cost.adders * combPrice(OpKind::Add) +
                cost.multipliers * combPrice(OpKind::Mul) +
                cost.dividers * combPrice(OpKind::Div);
      c.identity = identityOf(op, combKindOf(op), "");
      return c;
    }
    // Matching nothing is ordinary: a constant or a yield terminator costs
    // nothing real here. A float->float arith op reaching here would
    // miscompile at latency 0, so assert instead; extend
    // `classify()`/`needsIP()` (`validateDatapath` repeats this check for a
    // release build).
    auto isFloat = [](Type t) { return isa<FloatType>(t); };
    bool floatIn = llvm::any_of(elementTypes(op->getOperandTypes()), isFloat);
    bool floatOut = llvm::any_of(elementTypes(op->getResultTypes()), isFloat);
    assert((needsIP(op) || combKindOf(op) || !(floatIn && floatOut)) &&
           "unrecognized arith float->float op fell through to the latency-0 "
           "default row (no IP requirement, no comb lowering); add it to "
           "classify()/needsIP(). This is an early duplicate of the operator "
           "realizability check in validateDatapath, which is where a release "
           "build reports it");
    e = &defaultEntry;
  }

  // A row measured over other widths gives this operation neither a delay nor
  // an area, so it comes back unrealized and the pre-schedule realizability
  // check refuses the program.
  std::optional<double> delay =
      e->comb ? e->delay.evaluate(width) : std::optional<double>(0.0);
  std::optional<int64_t> price = priceOf(e->uses, {width});
  if (!delay || !price) {
    CostAttr bad = delay ? unmeasuredUse(e->uses, {width}) : e->delay;
    auto [lo, hi] = bad.measuredDomain();
    std::string row =
        e->symbol.empty() ? stringifyOpKindEnum(e->kind).str() : e->symbol;
    logging::error(logging::Stage::Prep,
                   logging::Code::DeviceDeclarationInvalid, op)
        << "Operation '" << op->getName() << "' is " << width
        << " bits wide, and the device's '" << row << "' row is measured over "
        << lo << ".." << hi
        << " bits. Measure the fabric at this width, or narrow the operands to "
           "one it covers";
    return OperatorChar{};
  }
  return characterize(op, *e, width);
}

OperatorChar OperatorLibrary::characterize(Operation *op,
                                           const OperatorEntry &e,
                                           int64_t width) const {
  std::optional<double> delay =
      e.comb ? e.delay.evaluate(width) : std::optional<double>(0.0);
  std::optional<int64_t> price = priceOf(e.uses, {width});
  assert(delay && price &&
         "the caller checked the row is measured at this "
         "width");

  // The stable Problem::OperatorType key: an IP row's symbol, a comb row's
  // `comb.<kind>.w<N>`, else `default`.
  OperatorChar c;
  c.timing.typeName =
      !e.symbol.empty() ? e.symbol
      : e.comb
          ? ("comb." + stringifyOpKindEnum(e.kind) + ".w" + Twine(width)).str()
          : std::string("default");
  c.timing.latency = e.latency;
  // A comb row carries its marginal delay: what the operator adds to a path
  // that already left a register. The floor the measurement also saw is paid
  // once per cycle, as the lower bound on every sub-cycle start time, so
  // charging it per operator would cost an N-deep chain N floors.
  //
  // Incoming and outgoing hold the same number because
  // `ChainingProblem::checkDelays` rejects a zero-latency operator whose two
  // delays differ: for a combinational cell they describe one path.
  if (e.comb)
    c.timing.inDelay = c.timing.outDelay = std::max(0.0, *delay - regFloor);
  else {
    c.timing.inDelay = e.inDelay;
    c.timing.outDelay = e.outDelay;
    c.timing.minPeriod = e.minPeriod;
  }
  c.pipelined = e.pipelined;
  // An IP's signature pins the width, so there the factors are constants and
  // this is the measured core.
  c.price = *price;
  // A shift by a literal is wiring, not a shifter, and so is a resize that
  // changes no width. It takes a type name of its own because the problem
  // registers timing per name: sharing the row would make two spellings of that
  // row disagree. It builds no logic, so it prices at nothing.
  if (isZeroDelay(op)) {
    c.timing.typeName = "rename." + c.timing.typeName;
    c.timing.inDelay = c.timing.outDelay = 0.0;
    c.price = 0;
  }
  // The realization is the row's own symbol when it is an IP, else the native
  // lowering the reifier picks; the default row reaches the comb arm too.
  if (!e.symbol.empty())
    c.identity = identityOf(op, std::nullopt, e.symbol);
  else
    c.identity = identityOf(op, combKindOf(op), "");
  return c;
}

OperatorChar OperatorLibrary::lookup(Operation *op, StringRef symbol) const {
  int64_t width = combParamWidth(op);
  // A comb row has no symbol of its own; a decided comb realization travels
  // as its characterization's type name, matched on the kind's last comb row.
  const OperatorEntry *comb = nullptr;
  for (const OperatorEntry *e : matchEntries(advancedEntries, entries, op)) {
    if (e->comb)
      comb = e;
    else if (e->symbol == symbol)
      return characterize(op, *e, width);
  }
  if (comb) {
    OperatorChar c = characterize(op, *comb, width);
    if (c.timing.typeName == symbol)
      return c;
  }
  llvm_unreachable("a decided realization names one of its op's candidates");
}

SmallVector<OperatorChar, 2>
OperatorLibrary::candidateChars(Operation *op) const {
  SmallVector<OperatorChar, 2> out;
  int64_t width = combParamWidth(op);
  for (const OperatorEntry *e : matchEntries(advancedEntries, entries, op)) {
    if (e->comb)
      continue; // a comb row is never a selection candidate under this order
    // The same float fit test `selectImplementation` ranks by, so the set here
    // and the row `lookup` picks never disagree about what fits.
    float need = periodNeed(static_cast<float>(regFloor), e->inDelay,
                            e->outDelay, e->minPeriod);
    if (need > selectionPeriodNs)
      continue;
    if (!priceOf(e->uses, {width}))
      continue; // unmeasured at this width: it cannot realize the op
    out.push_back(characterize(op, *e, width));
  }
  return out;
}

std::string OperatorIdentity::key() const {
  std::string s = (rename ? "rename." : "") + realizationName().str();
  llvm::raw_string_ostream os(s);
  os << '(';
  llvm::interleaveComma(argTypes, os);
  os << ")->" << resultType;
  if (predicate)
    os << " p" << predicate;
  if (map)
    os << " m" << map;
  return os.str();
}

OperatorIdentity mlir::allo::operatorIdentity(dcp::DCPathComputeOp comp) {
  return identityOf(comp, comp.getCombKind(),
                    comp.getOpType().value_or(StringRef()));
}

OperatorIdentity mlir::allo::operatorIdentity(Operation *op,
                                              const OperatorLibrary &lib) {
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return operatorIdentity(comp);
  return lib.lookup(op).identity;
}

bool OperatorLibrary::requiresUnmatchedIP(Operation *op) const {
  return needsIP(op) && matchEntries(advancedEntries, entries, op).empty();
}

bool OperatorLibrary::hasDirectRealization(Operation *op) const {
  return !matchEntries(advancedEntries, entries, op).empty();
}

SmallVector<StringRef, 2> OperatorLibrary::candidateIPs(Operation *op) const {
  SmallVector<StringRef, 2> symbols;
  for (const OperatorEntry *e : matchEntries(advancedEntries, entries, op))
    if (!e->comb)
      symbols.push_back(e->symbol);
  return symbols;
}

SmallVector<OperatorChar, 2>
mlir::allo::selectionCandidates(Operation *op, const OperatorLibrary &lib,
                                bool cyclic) {
  if (isSyncSubKernelCall(op) || asMemAccess(op) || isZeroDelay(op))
    return {};
  OperatorChar own = lib.lookup(op);
  if (own.identity.ipSymbol.empty() &&
      !StringRef(own.timing.typeName).starts_with("comb."))
    return {}; // the default realization stays the library's
  SmallVector<OperatorChar, 2> cands = lib.candidateChars(op);
  llvm::erase_if(cands, [&](const OperatorChar &c) {
    return (cyclic && !c.pipelined) ||
           (c.timing.latency == 0 && c.timing.inDelay != c.timing.outDelay);
  });
  // Latency, the two cones and the price are all a schedule can tell apart, so
  // a row another candidate matches or beats on all four only costs a selection
  // variable. Ties break on the name so two equal rows do not drop each other.
  auto asGood = [](const OperatorChar &d, const OperatorChar &c) {
    return d.timing.latency == c.timing.latency &&
           d.timing.inDelay <= c.timing.inDelay &&
           d.timing.outDelay <= c.timing.outDelay && d.price <= c.price;
  };
  SmallVector<OperatorChar, 2> distinct;
  for (const OperatorChar &c : cands)
    if (llvm::none_of(cands, [&](const OperatorChar &d) {
          return asGood(d, c) &&
                 (!asGood(c, d) || d.timing.typeName < c.timing.typeName);
        }))
      distinct.push_back(c);
  cands = std::move(distinct);
  if (cands.size() < 2)
    return {};
  // The library's pick has to be among them, or the schedule and the emitter
  // would resolve different rows.
  if (llvm::none_of(cands, [&](const OperatorChar &c) {
        return c.timing.typeName == own.timing.typeName;
      }))
    return {}; // the pick fell to a scope limit above
  return cands;
}

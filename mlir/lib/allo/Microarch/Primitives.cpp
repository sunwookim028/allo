/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Primitives.h"

#include "allo/Microarch/Interface.h"     // iface::ModuleInterface (the ports)
#include "allo/Microarch/Naming.h"        // regionSignal
#include "allo/Scheduling/AddressModel.h" // applyExprOf

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpIPredicate
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Module boundaries: the ports one is declared with, and wiring one up.
//===----------------------------------------------------------------------===//

llvm::SmallVector<hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b) {
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  auto *ctx = b.getContext();
  Type i1 = b.getI1Type();
  // A data port's hw width is its field bit width, so `iType(w)` reproduces
  // `datapathType`/`memElemType` for the data ports.
  auto iType = [&](unsigned w) -> Type { return b.getIntegerType(w); };
  SmallVector<PortInfo> ports;
  // The port names are the manifest, authored before CIRCT's LegalizeNames
  // runs, so a name ExportVerilog would rewrite or uniquify desyncs cosim from
  // the Verilog. These check the composed result.
  llvm::StringSet<> seen;
  auto port = [&](const Twine &n, Type t, Dir d) {
    std::string s = n.str();
    assert(sv::isNameValid(s, /*caseInsensitiveKeywords=*/false) &&
           "module port name is not a legal SystemVerilog identifier; the JSON "
           "manifest would desync from the emitted Verilog");
    bool fresh = seen.insert(s).second;
    assert(fresh && "duplicate module port name; the JSON manifest would "
                    "desync from the emitted Verilog");
    (void)fresh;
    ports.push_back(PortInfo{{StringAttr::get(ctx, s), t, d}});
  };
  port(kClk, i1, Dir::Input);
  port(kRst, i1, Dir::Input);
  port(kStart, i1, Dir::Input);
  // Scalar kernel arguments; memref args become memory ports instead. One
  // named after a control port trips the duplicate check above.
  for (const iface::Scalar &s : model.scalars)
    port(s.name, iType(s.width), Dir::Input);
  // Stream FIFO ports, input side. Module inputs must stay contiguous at the
  // front, since HWModulePortAccessor maps body args to the first `numInputs`
  // ports positionally, so {data, valid} / {ready} go here.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.data, iType(s.width), Dir::Input);
      port(s.valid, i1, Dir::Input);
    } else {
      port(s.ready, i1, Dir::Input);
    }
  }
  // A partitioned argument presents one interface per bank (a data-dependent
  // access spans all of them, a static access one); `model.reads[i]` holds an
  // access's per-bank interfaces.
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.data, iType(r.width), Dir::Input);
  // A fully-partitioned argument gets one input per element, no address or
  // latency, read combinationally in any number at once. A write-only argument
  // has no input side.
  for (const iface::RegisterFile &rf : model.registers)
    for (const iface::RegisterFile::Element &e : rf.elements)
      if (!e.in.empty())
        port(e.in, iType(rf.width), Dir::Input);
  port(kDone, i1, Dir::Output);
  // Stream FIFO ports, output side: an input stream's back-pressure {ready}, an
  // output stream's {data, valid}.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.ready, i1, Dir::Output);
    } else {
      port(s.data, iType(s.width), Dir::Output);
      port(s.valid, i1, Dir::Output);
    }
  }
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.addr, iType(kDatapathAddressWidth), Dir::Output);
  for (const auto &acc : model.writes)
    for (const iface::Memory &w : acc) {
      port(w.addr, iType(kDatapathAddressWidth), Dir::Output);
      port(w.data, iType(w.width), Dir::Output);
      port(w.we, i1, Dir::Output);
    }
  // A written scattered argument leaves on one data + write-enable pair per
  // element: the storage is the driver's, so an element commits only where the
  // module says it did.
  for (const iface::RegisterFile &rf : model.registers)
    for (const iface::RegisterFile::Element &e : rf.elements)
      if (!e.out.empty()) {
        port(e.out, iType(rf.width), Dir::Output);
        port(e.we, i1, Dir::Output);
      }
  // Scalar function results: one output port each, driven by the returning
  // region's survivor and valid when `done` rises (emit()).
  for (const iface::Result &r : model.results)
    port(r.name, iType(r.width), Dir::Output);
  return ports;
}

llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins) {
  using Dir = hw::ModulePort::Direction;
  SmallVector<Value> operands(mod.getNumInputPorts());
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Input) {
      auto it = ins.find(p.name.getValue());
      assert(it != ins.end() && "unwired child input port");
      operands[p.argNum] = it->second;
    }
  auto inst =
      hw::InstanceOp::create(b, loc, mod, b.getStringAttr(name), operands);
  llvm::StringMap<Value> outs;
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Output)
      outs[p.name.getValue()] = inst.getResult(p.argNum);
  return outs;
}

//===----------------------------------------------------------------------===//
// Shared free helpers.
//===----------------------------------------------------------------------===//

IntegerType datapathType(Type t, OpBuilder &b) {
  return b.getIntegerType(datapathWidth(t));
}

IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b) {
  return datapathType(cast<MemRefType>(m.memref.getType()).getElementType(), b);
}

Value resize(OpBuilder &b, Location loc, Value v, unsigned width,
             bool isSigned) {
  auto want = b.getIntegerType(width);
  unsigned have = cast<IntegerType>(v.getType()).getWidth();
  if (have == width)
    return v;
  if (have > width)
    return comb::ExtractOp::create(b, loc, want, v, 0).getResult();
  return isSigned ? comb::createOrFoldSExt(b, loc, v, want)
                  : comb::createZExt(b, loc, v, width);
}

unsigned declaredDepth(unsigned words) { return std::max(2u, words); }

unsigned memAddrWidth(const uarch::MemUnit &m) {
  return llvm::Log2_64_Ceil(declaredDepth(m.depthWords));
}

SmallVector<APInt> initWords(ElementsAttr init, unsigned width,
                             unsigned depth) {
  SmallVector<APInt> words;
  if (isa<FloatType>(init.getElementType()))
    for (const APFloat &v : init.getValues<APFloat>())
      words.push_back(v.bitcastToAPInt().zextOrTrunc(width));
  else
    for (const APInt &v : init.getValues<APInt>())
      words.push_back(v.zextOrTrunc(width));
  assert(words.size() <= depth &&
         "an array's declared depth must cover its initializer");
  words.resize(depth, APInt(width, 0));
  return words;
}

void recordMemoryInit(seq::HLMemOp mem, ArrayRef<APInt> words) {
  Type elemTy = mem.getMemType().getElementType();
  SmallVector<Attribute> vals;
  for (const APInt &w : words)
    vals.push_back(IntegerAttr::get(elemTy, w));
  mem->setAttr(kMemoryInitAttr, ArrayAttr::get(mem.getContext(), vals));
}

//===----------------------------------------------------------------------===//
// Memory-banking crossbar: one access reaching a bank decided at run time.
//===----------------------------------------------------------------------===//

Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank) {
  c.muxLedger.add(MuxRole::Crossbar, bankValues.size(),
                  datapathWidth(bankValues[0].getType()));
  // The selects are exclusive by construction, so the reduction is the
  // log-depth AND-OR `muxLevels` prices rather than an N-1 priority chain. A
  // no-match select reads 0, a don't-care: the access is out of bounds.
  return c.oneHotSelect(bankValues, oneHotDecode(c, bank, bankValues.size()));
}

SmallVector<Value> oneHotDecode(EmitContext &c, Value idx, unsigned n) {
  if (n == 1)
    return {c.t1}; // one target: an in-bounds index can only name it
  Value sel = addrAt(c.b, c.loc, idx, llvm::Log2_32_Ceil(n));
  SmallVector<Value> lines;
  for (unsigned k = 0; k < n; ++k)
    lines.push_back(c.icmpEq(sel, k));
  return lines;
}

Value writeDemux(EmitContext &c, Value we, Value bank, unsigned k) {
  if (!bank)
    return we;
  // One decoder arm: a compare at the bank tag's width plus a 1-bit gate,
  // charged as a 2:1 cone at that width, which prices about the same fabric.
  c.muxLedger.add(MuxRole::Crossbar, 2, datapathWidth(bank.getType()));
  return c.andBits(we, c.icmpEq(bank, k));
}

//===----------------------------------------------------------------------===//
// Address arithmetic. An address, a bank digit and a scaled counter are all
// non-negative by construction, so every width change on the address path is
// the unsigned resize and every divisor a compile-time constant.
//===----------------------------------------------------------------------===//

// A literal of \p v's own width. Address arithmetic is carried at whatever
// width the addressed memory needs, so every operand of a `comb` op below has
// to be built against the value it accompanies rather than a fixed i32.
static Value konstLike(OpBuilder &b, Location loc, Value v, int64_t k) {
  return hw::ConstantOp::create(b, loc, v.getType(), k).getResult();
}

Value addrAt(OpBuilder &b, Location loc, Value v, unsigned width) {
  return resize(b, loc, v, width, /*isSigned=*/false);
}

// Multiply by a compile-time constant. A power-of-two coefficient is a shift;
// anything else stays a `comb.mul` deliberately, since synthesis recodes a
// constant multiplier into a shift-add network better than a decomposition
// emitted here could.
static Value mulConst(OpBuilder &b, Location loc, Value v, int64_t k) {
  if (k == 1)
    return v;
  if (k > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(k)))
    return comb::ShlOp::create(b, loc, v,
                               konstLike(b, loc, v, llvm::Log2_64(k)), false)
        .getResult();
  return comb::MulOp::create(b, loc, v, konstLike(b, loc, v, k), false)
      .getResult();
}

// Unsigned divide by a compile-time constant: a shift for a power of two,
// else the reciprocal multiply, `(v * M) >> shift` at the product's width,
// priced by `addressCost` from the same `magicMultiplier`. A divisor past the
// operand's range leaves a zero quotient.
static Value divConst(OpBuilder &b, Location loc, Value v, int64_t d) {
  if (d == 1)
    return v;
  if (llvm::isPowerOf2_64(d))
    return comb::ShrUOp::create(b, loc, v,
                                konstLike(b, loc, v, llvm::Log2_64(d)), false)
        .getResult();
  unsigned w = cast<IntegerType>(v.getType()).getWidth();
  assert(w <= 62 && "the reciprocal multiplier of a wider operand overflows");
  if (static_cast<uint64_t>(d) >= (uint64_t(1) << w))
    return konstLike(b, loc, v, 0);
  unsigned shift;
  uint64_t magic = magicMultiplier(d, w, shift);
  Value wide = addrAt(b, loc, v, 2 * w + 1);
  Value prod = mulConst(b, loc, wide, static_cast<int64_t>(magic));
  Value q =
      comb::ShrUOp::create(b, loc, prod, konstLike(b, loc, prod, shift), false);
  return addrAt(b, loc, q, w);
}

// A power-of-two divisor never reaches here: `evalAffine` builds that subtree
// narrow instead, which is the same mask. Everything else is
// `v - (v / d) * d` over the reciprocal quotient, in the operand's own width,
// which the product never exceeds.
static Value modConst(OpBuilder &b, Location loc, Value v, int64_t d) {
  if (d == 1)
    return konstLike(b, loc, v, 0);
  unsigned w = cast<IntegerType>(v.getType()).getWidth();
  assert(w <= 62 && "the reciprocal multiplier of a wider operand overflows");
  if (static_cast<uint64_t>(d) >= (uint64_t(1) << w))
    return v;
  Value qd = mulConst(b, loc, divConst(b, loc, v, d), d);
  return comb::SubOp::create(b, loc, v, qd, false).getResult();
}

// Evaluate an affine index expression to a hw value \p width bits wide,
// emitting comb ops. `idx` holds the resolved value of each map operand (dims
// then symbols), each at the datapath width. Shared by the two places a map
// reaches the datapath: a memory access's address (bankAddress) and a
// standalone affine.apply (emitCompute).
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims, unsigned width) {
  Type t = b.getIntegerType(width);
  if (auto cst = dyn_cast<AffineConstantExpr>(e))
    return hw::ConstantOp::create(b, loc, t, cst.getValue()).getResult();
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return addrAt(b, loc, idx[d.getPosition()], width);
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return addrAt(b, loc, idx[numDims + sym.getPosition()], width);
  auto bin = cast<AffineBinaryOpExpr>(e);
  if (e.getKind() == AffineExprKind::Add)
    return comb::AddOp::create(
               b, loc, evalAffine(b, loc, bin.getLHS(), idx, numDims, width),
               evalAffine(b, loc, bin.getRHS(), idx, numDims, width), false)
        .getResult();
  if (e.getKind() == AffineExprKind::Mul) {
    Value lhs = evalAffine(b, loc, bin.getLHS(), idx, numDims, width);
    // An affine coefficient is always constant, so this is a shift-or-multiply
    // rather than a general multiplier. A semi-affine map is representable
    // though, so a non-constant one still lowers.
    if (auto k = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return mulConst(b, loc, lhs, k.getValue());
    return comb::MulOp::create(
               b, loc, lhs,
               evalAffine(b, loc, bin.getRHS(), idx, numDims, width), false)
        .getResult();
  }
  // floordiv/mod by a constant is delinearization left by a coalesced nest over
  // a non-negative index. Neither is congruent modulo 2^width, so both compute
  // wide and narrow afterwards.
  auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS());
  assert(rc && rc.getValue() > 0 &&
         "affine div/mod by a non-constant or non-positive divisor");
  int64_t f = rc.getValue();
  // With one congruent exception, the one a bank digit always ends in: `x mod
  // 2^k` is the low k bits, so that subtree is built k bits wide and the mask
  // disappears with it. `addressCost` prices it at the same narrowed width.
  if (e.getKind() == AffineExprKind::Mod && f > 1 &&
      llvm::isPowerOf2_64(static_cast<uint64_t>(f))) {
    unsigned k =
        std::min<unsigned>(width, llvm::Log2_64(static_cast<uint64_t>(f)));
    return addrAt(b, loc, evalAffine(b, loc, bin.getLHS(), idx, numDims, k),
                  width);
  }
  Value lhs =
      evalAffine(b, loc, bin.getLHS(), idx, numDims, kDatapathAddressWidth);
  if (e.getKind() == AffineExprKind::FloorDiv)
    return addrAt(b, loc, divConst(b, loc, lhs, f), width);
  if (e.getKind() == AffineExprKind::Mod)
    return addrAt(b, loc, modConst(b, loc, lhs, f), width);
  llvm_unreachable("unexpected affine op");
}

static comb::ICmpPredicate combICmpPredicate(arith::CmpIPredicate p) {
  using A = arith::CmpIPredicate;
  using C = comb::ICmpPredicate;
  switch (p) {
  case A::eq:
    return C::eq;
  case A::ne:
    return C::ne;
  case A::slt:
    return C::slt;
  case A::sle:
    return C::sle;
  case A::sgt:
    return C::sgt;
  case A::sge:
    return C::sge;
  case A::ult:
    return C::ult;
  case A::ule:
    return C::ule;
  case A::ugt:
    return C::ugt;
  case A::uge:
    return C::uge;
  }
  llvm_unreachable("unknown arith::CmpIPredicate");
}

Value emitCompute(OpBuilder &b, Location loc, const OperatorIdentity &id,
                  ValueRange operands, Type resultType) {
  using E = CombOpKindEnum;
  assert(id.comb && "emitCompute realizes the native path of an identity");
  CombOpKindEnum kind = *id.comb;
  // A constant affine map takes no operands, and the unary kinds take one, so
  // neither read is unconditional.
  Value lhs = operands.empty() ? Value() : operands[0];
  Value rhs = operands.size() > 1 ? operands[1] : Value();
  // A compare feeds a mux: what an integer min/max is.
  auto minmax = [&](comb::ICmpPredicate p) -> Value {
    Value c = comb::ICmpOp::create(b, loc, p, lhs, rhs, false)->getResult(0);
    return comb::MuxOp::create(b, loc, c, lhs, rhs)->getResult(0);
  };
  // No `default`, so a new `CombOpKind` case fails to compile here rather than
  // reaching the unreachable below.
  switch (kind) {
  // affine.apply: the map rides on the op, left by loop-canonicalization when
  // an IV is read outside an address. Built as `applyExprOf`, the form the
  // schedule priced; via evalAffine, so a power-of-two divisor stays
  // shift+mask.
  case E::Apply: {
    assert(id.map && "an apply identity must carry the original affine map");
    AffineMap map = cast<AffineMapAttr>(id.map).getValue();
    return evalAffine(b, loc, applyExprOf(map), operands, map.getNumDims());
  }
  // Width-changing unary casts resize operand[0] via a comb sign/zero-extend or
  // a low-bit extract; 0-latency, so they slot into the schedule like any comb.
  case E::Extsi:
    return comb::createOrFoldSExt(b, loc, lhs, resultType);
  case E::Extui:
    return comb::createZExt(b, loc, lhs,
                            cast<IntegerType>(resultType).getWidth());
  case E::Trunci:
    return comb::ExtractOp::create(b, loc, resultType, lhs, 0).getResult();
  case E::IndexCast:
    return resize(b, loc, lhs, cast<IntegerType>(resultType).getWidth(),
                  /*isSigned=*/true);
  case E::IndexCastUi:
    return resize(b, loc, lhs, cast<IntegerType>(resultType).getWidth(),
                  /*isSigned=*/false);
  // Float negate: the float rides as its integer bit pattern, so flipping its
  // sign bit is a single XOR, no IP.
  case E::Negf: {
    unsigned w = cast<IntegerType>(resultType).getWidth();
    // The width's signed minimum is exactly the top bit set, at any width. An
    // APInt, not `1 << (w-1)`, which shifts into an int64's sign bit at
    // w == 64 (UB before C++20) and past it beyond.
    Value signBit = hw::ConstantOp::create(
        b, loc, IntegerAttr::get(resultType, APInt::getSignedMinValue(w)));
    return comb::XorOp::create(b, loc, lhs, signBit, false)->getResult(0);
  }
  // 3-input value mux: arith.select(cond, t, f) == comb.mux (cond ? t : f).
  case E::Select:
    return comb::MuxOp::create(b, loc, operands[0], operands[1], operands[2])
        ->getResult(0);
  // Width-preserving binary integer/logic ops.
  case E::Addi:
    return comb::AddOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Subi:
    return comb::SubOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Muli:
    return comb::MulOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Andi:
    return comb::AndOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Ori:
    return comb::OrOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Xori:
    return comb::XorOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Shli:
    return comb::ShlOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Shrsi:
    return comb::ShrSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Shrui:
    return comb::ShrUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  // Signed / unsigned divide, emitted for a flattened guard's delinearization;
  // a scheduled data divide is multi-cycle IP instead.
  case E::Divsi:
    return comb::DivSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Divui:
    return comb::DivUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  // Signed / unsigned remainder (int rem is combinational under the operator
  // model).
  case E::Remsi:
    return comb::ModSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Remui:
    return comb::ModUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  case E::Minsi:
    return minmax(comb::ICmpPredicate::slt);
  case E::Maxsi:
    return minmax(comb::ICmpPredicate::sgt);
  case E::Minui:
    return minmax(comb::ICmpPredicate::ult);
  case E::Maxui:
    return minmax(comb::ICmpPredicate::ugt);
  // Integer compare, with the predicate carried from arith.cmpi.
  case E::Cmpi: {
    auto pred = cast<arith::CmpIPredicateAttr>(id.predicate).getValue();
    return comb::ICmpOp::create(b, loc, combICmpPredicate(pred), lhs, rhs,
                                false)
        ->getResult(0);
  }
  }
  // Not an `assert`: under NDEBUG that would fall through and hand back a null
  // Value to wire into the datapath.
  llvm_unreachable("CombOpKind outside the enum reached emitCompute");
}

//===----------------------------------------------------------------------===//
// RegLedger: what the emission spent in flip-flops.
//===----------------------------------------------------------------------===//

llvm::StringRef roleName(RegRole role) {
  switch (role) {
  case RegRole::Value:
    return "value";
  case RegRole::Pulse:
    return "pulse";
  case RegRole::Counted:
    return "counted";
  case RegRole::Survivor:
    return "survivor";
  case RegRole::Counter:
    return "counter";
  case RegRole::Control:
    return "control";
  case RegRole::Storage:
    return "storage";
  }
  llvm_unreachable("every RegRole has a name");
}

std::vector<RegClass> RegLedger::classes() const {
  std::vector<RegClass> out;
  out.reserve(runs.size());
  for (const auto &[key, count] : runs) {
    auto [role, width, depth, reset, enable] = key;
    out.push_back({role, width, depth, count, reset, enable});
  }
  return out;
}

unsigned RegLedger::bits() const {
  unsigned total = 0;
  for (const auto &[key, count] : runs)
    total += std::get<1>(key) * std::get<2>(key) * count;
  return total;
}

llvm::StringRef muxRoleName(MuxRole role) {
  switch (role) {
  case MuxRole::Address:
    return "address";
  case MuxRole::Commit:
    return "commit";
  case MuxRole::Crossbar:
    return "crossbar";
  }
  llvm_unreachable("every MuxRole has a name");
}

std::vector<MuxCone> MuxLedger::classes() const {
  std::vector<MuxCone> out;
  out.reserve(cones.size());
  for (const auto &[key, count] : cones) {
    auto [role, fanin, width] = key;
    out.push_back({role, fanin, width, count});
  }
  return out;
}

//===----------------------------------------------------------------------===//
// EmitContext: the shared builder substrate.
//===----------------------------------------------------------------------===//

Value EmitContext::konst(Type t, int64_t v) {
  return R(hw::ConstantOp::create(b, loc, t, v));
}

// Only control state needs a synchronous reset. A value run's data is
// don't-care until its valid pulse arrives, and a pulse run only needs a
// defined power-on 0, which the fabric's INIT carries for free. The reset is
// what blocks shift-register extraction.
static bool holdsReset(RegRole role) {
  return role != RegRole::Value && role != RegRole::Pulse;
}

Value EmitContext::initialFor(Value rstVal) {
  auto k = rstVal.getDefiningOp<hw::ConstantOp>();
  assert(k && "a reset-free register powers on to a constant");
  Value &v = initials[k.getValueAttr()];
  if (!v)
    v = seq::createConstantInitialValue(b, loc, k.getValueAttr());
  return v;
}

Value EmitContext::reg(Value in, Value rstVal, RegRole role) {
  if (!inChainRun)
    ledger.add(role, datapathWidth(in.getType()), 1, holdsReset(role));
  if (holdsReset(role))
    return R(seq::CompRegOp::create(b, loc, in, clk, rst, rstVal));
  return R(seq::CompRegOp::create(b, loc, in, clk, /*reset=*/Value(),
                                  /*rstValue=*/Value(), initialFor(rstVal)));
}

Value EmitContext::enabledReg(Value in, Value ce, Value rstVal, RegRole role) {
  // `seq.compreg.ce` is not self-referential, so CSE merges identical runs
  // before the export pipeline lowers it.
  if (!inChainRun)
    ledger.add(role, datapathWidth(in.getType()), 1, holdsReset(role),
               /*enable=*/true);
  if (holdsReset(role))
    return R(seq::CompRegClockEnabledOp::create(
        b, loc, in.getType(), in, clk, ce, StringAttr(), rst, rstVal,
        /*initialValue=*/Value(), hw::InnerSymAttr()));
  return R(seq::CompRegClockEnabledOp::create(
      b, loc, in.getType(), in, clk, ce, StringAttr(), /*reset=*/Value(),
      /*resetValue=*/Value(), initialFor(rstVal), hw::InnerSymAttr()));
}

Value EmitContext::shellReg(Value in, Value rstVal, const StallShell &sh,
                            RegRole role) {
  return sh ? enabledReg(in, sh.chainEnable, rstVal, role)
            : reg(in, rstVal, role);
}

Value EmitContext::stallHold(Value in, const StallShell &sh) {
  if (!sh)
    return in; // rigid: the address is just the live index
  Value held =
      enabledReg(in, sh.chainEnable, konst(in.getType(), 0), RegRole::Control);
  return mux(sh.chainEnable, in, held);
}

Value EmitContext::latchReg(Value init, Value next, Value load, Value advance,
                            RegRole role, Value *dWire) {
  assert(holdsReset(role) && "a latch is control state and keeps its reset");
  if (!inChainRun)
    ledger.add(role, datapathWidth(init.getType()), 1, /*reset=*/true,
               /*enable=*/true);
  // Built in the self-holding spelling: a latch is a recurrence, one per region
  // result, so the clock-enabled form's CSE gain does not apply.
  llvm::SaveAndRestore charged(inChainRun, true);
  Backedge selfNext = bb.get(init.getType());
  Value self = reg(selfNext, konst(init.getType(), 0), role);
  Value d = mux(load, init, mux(advance, next, self));
  selfNext.setValue(d);
  if (dWire)
    *dWire = d;
  return self;
}

Value EmitContext::mux(Value sel, Value t, Value f) {
  return R(comb::MuxOp::create(b, loc, sel, t, f));
}

Value EmitContext::oneHotSelect(ArrayRef<Value> values,
                                ArrayRef<Value> selects) {
  assert(values.size() == selects.size() && !values.empty() &&
         "one select per value, and a select over nothing is not a value");
  if (values.size() == 1)
    return values.front();
  auto type = cast<IntegerType>(values.front().getType());
  SmallVector<Value> terms;
  terms.reserve(values.size());
  for (auto [value, sel] : llvm::zip(values, selects)) {
    assert(value.getType() == type &&
           "one select drives one port, so every source is that port's width");
    // A 1-bit port needs no replication, and `comb.replicate` by 1 is illegal.
    Value mask = type.getWidth() == 1
                     ? sel
                     : R(comb::ReplicateOp::create(b, loc, sel,
                                                   (int32_t)type.getWidth()));
    terms.push_back(R(comb::AndOp::create(b, loc, value, mask, false)));
  }
  // One variadic OR, so the synthesizer balances the reduction tree.
  return R(comb::OrOp::create(b, loc, type, terms, false));
}

// Charge one chain run split at its consumed taps: extraction breaks a shift
// register at every tap, so each maximal inter-tap segment is its own run and a
// short segment falls back to flip-flops.
static void chargeChainRuns(RegLedger &ledger, RegRole role, unsigned width,
                            unsigned depth, ArrayRef<unsigned> taps, bool reset,
                            bool enable) {
  if (taps.empty()) {
    ledger.add(role, width, depth, reset, enable);
    return;
  }
  assert(taps.back() == depth && "the deepest tap is the chain's depth");
  unsigned prev = 0;
  for (unsigned t : taps) {
    assert(t > prev && "taps are sorted, unique and start past the source");
    ledger.add(role, width, t - prev, reset, enable);
    prev = t;
  }
}

ShiftChain EmitContext::shiftChain(Value in, unsigned depth,
                                   const StallShell &sh, RegRole role,
                                   ArrayRef<unsigned> taps) {
  ShiftChain chain;
  chain.stages.push_back(in); // stage 0 = the source (a depth-0 tap reads it)
  Value rz = konst(in.getType(), 0);
  Value cur = in;
  {
    llvm::SaveAndRestore charged(inChainRun, true);
    for (unsigned s = 1; s <= depth; ++s) {
      // An elastic shell advances every stage only while enabled, so all taps
      // freeze together; a rigid shell is a plain unconditional shift.
      cur = shellReg(cur, rz, sh, role);
      chain.stages.push_back(cur);
    }
  }
  chargeChainRuns(ledger, role, datapathWidth(in.getType()), depth, taps,
                  holdsReset(role), /*enable=*/bool(sh));
  return chain;
}

ShiftChain EmitContext::foldedChain(Value in, unsigned depth, unsigned ii,
                                    Value phase, unsigned ready,
                                    const StallShell &sh,
                                    ArrayRef<unsigned> taps) {
  assert(ii > 1 && "a fold at II 1 is the plain chain, one register per tap");
  // A stall freezes the phase, so the capture term stays high across it and
  // would otherwise shift the chain once per stalled cycle.
  Value capture = icmpEq(phase, ready % ii);
  Value ce = sh ? andBits(sh.chainEnable, capture) : capture;
  Value rz = konst(in.getType(), 0);
  llvm::SmallVector<Value> held;
  Value cur = in;
  unsigned n = (depth + ii - 1) / ii;
  {
    llvm::SaveAndRestore charged(inChainRun, true);
    for (unsigned j = 0; j < n; ++j) {
      cur = enabledReg(cur, ce, rz, RegRole::Value);
      held.push_back(cur);
    }
  }
  // The run is the registers built, not the cycles spanned: a fold holds the
  // same `depth` taps in `n` of them, and cycle tap k reads register
  // ceil(k / ii), which is where the run splits.
  llvm::SmallVector<unsigned> regTaps;
  for (unsigned t : taps) {
    assert(t >= 1 && "a zero tap reads the source, not the chain");
    regTaps.push_back((t - 1) / ii + 1);
  }
  regTaps.erase(std::unique(regTaps.begin(), regTaps.end()), regTaps.end());
  chargeChainRuns(ledger, RegRole::Value, datapathWidth(in.getType()), n,
                  regTaps, /*reset=*/false, /*enable=*/true);
  ShiftChain chain;
  chain.stages.push_back(in); // stage 0 = the source, as in a plain chain
  for (unsigned k = 1; k <= depth; ++k)
    chain.stages.push_back(held[(k - 1) / ii]); // register ceil(k / ii)
  return chain;
}

Value EmitContext::delayPulseCounted(Value pulse, unsigned n,
                                     const StallShell &sh) {
  assert(sh.singlePass && "a counted delay drops every pulse but the first, "
                          "so its owning region must issue one pass");
  assert(n >= 1 && "a zero-cycle delay is the signal itself");
  // `pulse` arms the counter at 0; it counts every advancing cycle and fires at
  // n-1, so the output rises exactly n cycles after the input, as a chain tap
  // does. Under an elastic shell it counts only while enabled.
  IntegerType cntTy = b.getIntegerType(std::max(1u, llvm::Log2_64_Ceil(n)));
  Backedge armedNext = bb.get(i1);
  Backedge countNext = bb.get(cntTy);
  const RegRole role = RegRole::Counted;
  Value cz = konst(cntTy, 0);
  Value armed = shellReg(armedNext, f1, sh, role);
  Value count = shellReg(countNext, cz, sh, role);
  Value fire = andBits(armed, icmpEq(count, n - 1));
  armedNext.setValue(mux(pulse, t1, mux(fire, f1, armed)));
  countNext.setValue(mux(
      pulse, cz,
      mux(armed, R(comb::AddOp::create(b, loc, count, konst(cntTy, 1), false)),
          count)));
  std::string tag = sh.region ? regionTagOf(*sh.region) : regionTag;
  if (!tag.empty()) {
    nameValue(armed, regionSignal(tag, "wait" + std::to_string(n)));
    nameValue(count, regionSignal(tag, "wait" + std::to_string(n) + "_c"));
  }
  return fire;
}

Value EmitContext::delayValid(Value sig, unsigned n, const StallShell &sh) {
  assert(sig.getType() == i1 && "a valid is one bit");
  if (n == 0)
    return sig;
  if (n >= countedDelayCycles && sh.singlePass) {
    if (Value hit = countedPulses.lookup({sig, n, sh.chainEnable}))
      return hit;
    Value fire = delayPulseCounted(sig, n, sh);
    countedPulses[{sig, n, sh.chainEnable}] = fire;
    return fire;
  }
  ShiftChain &chain = pulseChains[{sig, sh.chainEnable}];
  if (chain.stages.empty())
    chain.stages.push_back(sig); // stage 0 = the source itself
  if (unsigned have = chain.depth(); have < n) {
    std::string tag = sh.region ? regionTagOf(*sh.region) : regionTag;
    Value cur = chain.stages.back();
    llvm::SaveAndRestore charged(inChainRun, true);
    for (unsigned k = have + 1; k <= n; ++k) {
      cur = shellReg(cur, f1, sh, RegRole::Pulse);
      // Label each stage with the cycle it is valid at, so a waveform reads
      // `r1_v3`: region 1, three cycles after issue. Named after the owning
      // region, so a chain another region extends keeps one name family.
      if (!tag.empty())
        nameValue(cur, regionSignal(tag, "v" + std::to_string(k)));
      chain.stages.push_back(cur);
    }
    ledger.extend(RegRole::Pulse, 1, have, n, holdsReset(RegRole::Pulse),
                  /*enable=*/bool(sh));
  }
  return chain.stages[n];
}

Value EmitContext::activationPulse(Value pulse, unsigned stage,
                                   const StallShell &sh) {
  return delayValid(pulse, stage, sh);
}

Value EmitContext::icmpEq(Value a, int64_t cst) {
  return R(comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, a,
                                konstLike(b, loc, a, cst), false));
}

Value EmitContext::icmpEqV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, lhs, rhs, false));
}

Value EmitContext::icmpSgeV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::sge, lhs, rhs, false));
}

Value EmitContext::icmpUgeV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::uge, lhs, rhs, false));
}

Value EmitContext::notBit(Value v) {
  return R(comb::XorOp::create(b, loc, v, t1, false));
}

Value EmitContext::andBits(Value lhs, Value rhs) {
  return R(comb::AndOp::create(b, loc, lhs, rhs, false));
}

Value EmitContext::orBits(Value lhs, Value rhs) {
  return R(comb::OrOp::create(b, loc, lhs, rhs, false));
}

Value EmitContext::risingEdge(Value level) {
  Value prev = reg(level, f1);
  return R(comb::AndOp::create(
      b, loc, level, R(comb::XorOp::create(b, loc, prev, t1, false)), false));
}

Value EmitContext::startFor(Value regionStart, ArrayRef<Value> predDones) {
  if (predDones.empty())
    return regionStart;
  Value ready = predDones.front();
  for (Value d : predDones.drop_front())
    ready = andBits(ready, d);
  return risingEdge(ready);
}

Value EmitContext::holdDone(Value setPulse, Value start) {
  circt::Backedge doneNext = bb.get(i1);
  Value done = reg(doneNext, f1);
  doneNext.setValue(mux(start, f1, mux(setPulse, t1, done)));
  return done;
}

Value EmitContext::completedSince(Value level, Value passStart) {
  return completedSinceEdge(risingEdge(level), passStart);
}

Value EmitContext::completedSinceEdge(Value edge, Value passStart) {
  return andBits(orBits(holdDone(edge, passStart), edge), notBit(passStart));
}

std::pair<Value, Value> EmitContext::branchPulse(Value when, Value cond) {
  return {andBits(when, cond), andBits(when, notBit(cond))};
}

void EmitContext::initLiterals() {
  zero32 = konst(i32, 0);
  one32 = konst(i32, 1);
  f1 = konst(i1, 0);
  t1 = konst(i1, 1);
}

} // namespace mlir::allo::uarch

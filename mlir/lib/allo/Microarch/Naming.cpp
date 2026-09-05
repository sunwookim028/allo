/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Naming.h"
#include "allo/Microarch/Datapath.h"       // externalBank
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid
#include "circt/Dialect/Seq/SeqOps.h"   // seq::CompRegOp

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpFPredicate

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

namespace {
// The suffix vocabulary: the one place these strings are spelled.
constexpr llvm::StringLiteral kAddr = "_addr";
constexpr llvm::StringLiteral kData = "_data";
constexpr llvm::StringLiteral kWe = "_we";
constexpr llvm::StringLiteral kValid = "_valid";
constexpr llvm::StringLiteral kReady = "_ready";
constexpr llvm::StringLiteral kRead = "_rd";
constexpr llvm::StringLiteral kWrite = "_wr";
constexpr llvm::StringLiteral kStream = "_st";
constexpr llvm::StringLiteral kIn = "_in";
constexpr llvm::StringLiteral kOut = "_out";
constexpr llvm::StringLiteral kBank = "_b";

std::string join(llvm::StringRef base, llvm::StringLiteral suffix) {
  return base.str() + suffix.str();
}
} // namespace

std::string verilogName(llvm::StringRef name) {
  std::string s = sanitizeCppIdentifier(name); // charset + leading digit
  // ExportVerilog renames a reserved word (`input` -> `input_0`), desyncing the
  // manifest from the Verilog, so escape it here instead.
  while (!sv::isNameValid(s, /*caseInsensitiveKeywords=*/false))
    s.push_back('_');
  return s;
}

//===----------------------------------------------------------------------===//
// Owner tokens.
//===----------------------------------------------------------------------===//

std::string argOwner(unsigned argNo) { return "a" + std::to_string(argNo); }
std::string memOwner(MemId m) { return "m" + std::to_string(m); }
std::string unitOwner(UnitId u) { return "u" + std::to_string(u); }
std::string chanOwner(StreamId s) { return "ch" + std::to_string(s); }
std::string regOwner(RegId r) { return "reg" + std::to_string(r); }
std::string regionTagOf(unsigned r) { return "r" + std::to_string(r); }

std::string ownerOf(Location loc, llvm::StringRef fallback) {
  // Charset only: the keyword escape belongs to the composed name, so an
  // `input` array yields `input_rd0` rather than `input__rd0`.
  if (auto name = nameFromLoc(loc))
    return sanitizeCppIdentifier(*name);
  return fallback.str();
}

std::string ownerOf(Value v, llvm::StringRef fallback) {
  // An unnamed value keys on its own identity, the argument position, never
  // on where its port lands in the port list.
  auto ba = dyn_cast<BlockArgument>(v);
  return ownerOf(v.getLoc(), ba ? argOwner(ba.getArgNumber()) : fallback.str());
}

std::string uniqueOwnerOf(Value v, llvm::ArrayRef<Value> siblings,
                          llvm::StringRef fallback) {
  std::string own = ownerOf(v, fallback);
  // Only one versus more than one matters, so stop at the second tie.
  unsigned ties = 0;
  for (Value s : siblings)
    if (ownerOf(s, fallback) == own && ++ties > 1)
      break;
  if (ties <= 1)
    return own;
  // Two values sharing a source name would collide in port/cell naming. Each
  // tied value takes a suffix unique by construction: its argument position,
  // else the caller's per-cell fallback.
  auto ba = dyn_cast<BlockArgument>(v);
  return own + "_" + (ba ? argOwner(ba.getArgNumber()) : fallback.str());
}

//===----------------------------------------------------------------------===//
// Fields and bases.
//===----------------------------------------------------------------------===//

std::string portAddr(llvm::StringRef base) { return join(base, kAddr); }
std::string portData(llvm::StringRef base) { return join(base, kData); }
std::string portWe(llvm::StringRef base) { return join(base, kWe); }
std::string portValid(llvm::StringRef base) { return join(base, kValid); }
std::string portReady(llvm::StringRef base) { return join(base, kReady); }

std::string memBase(llvm::StringRef owner, bool write, unsigned group) {
  return verilogName(join(owner, write ? kWrite : kRead) +
                     std::to_string(group));
}
std::string streamBase(llvm::StringRef owner) {
  return verilogName(join(owner, kStream));
}
std::string scalarBase(llvm::StringRef owner) {
  return verilogName(join(owner, kIn));
}
std::string resultBase(llvm::StringRef owner) {
  return verilogName(join(owner, kOut));
}
std::string bankBase(llvm::StringRef base, unsigned bank) {
  return verilogName(join(base, kBank) + std::to_string(bank));
}
std::string elemBase(llvm::StringRef owner, unsigned index, ElemDir dir) {
  std::string base = owner.str() + "_" + std::to_string(index);
  if (dir == ElemDir::In)
    base += kIn.str();
  else if (dir == ElemDir::Out)
    base += kOut.str();
  return verilogName(base);
}

llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const MemUnit &m, const MemUnit::Access &acc) {
  assert(!acc.portBase.empty() &&
         "an access with no port base is not on the module boundary");
  ExternalBanking eb = externalBank(m, acc);
  if (eb.factor == 1)
    return {{0u, acc.portBase}};
  if (eb.bank)
    return {{*eb.bank, acc.portBase}}; // statically routed to one interface
  // Data-dependent: one interface per bank (the crossbar drives every bank).
  llvm::SmallVector<std::pair<unsigned, std::string>> all;
  for (unsigned k = 0; k < eb.factor; ++k)
    all.push_back({k, bankBase(acc.portBase, k)});
  return all;
}

std::string streamPortBase(const Datapath &dp, const StreamChannel &s) {
  assert(!s.internal && "a kernel-local channel has no boundary port to name");
  auto own = [](const StreamChannel &c) {
    return streamBase(ownerOf(c.stream, chanOwner(c.id)));
  };
  std::string base = own(s);
  // Count the siblings this name ties with. A tie would give two handshakes one
  // set of port names, which ExportVerilog uniquifies, desyncing the manifest.
  unsigned sameBase = 0, sameDir = 0;
  for (const StreamChannel &o : dp.streams) {
    if (o.internal || own(o) != base)
      continue;
    ++sameBase;
    sameDir += o.isInput == s.isInput;
  }
  if (sameBase == 1)
    return base;
  base += (s.isInput ? kIn : kOut).str(); // the systolic shape: a get, a put
  return sameDir == 1 ? base : base + "_s" + std::to_string(s.id);
}

std::string scalarPortName(const Datapath &dp, const IOPort &io) {
  llvm::SmallVector<Value> siblings;
  for (const IOPort &o : dp.ios)
    siblings.push_back(o.value); // every IOPort is a scalar input
  return scalarBase(
      uniqueOwnerOf(io.value, siblings, "s" + std::to_string(io.id)));
}

std::string resultPortName(unsigned i, unsigned n) {
  // A result is 1:1 with the source signature, so it carries an index only
  // when the signature itself declares several.
  return resultBase(n == 1 ? "ret" : "ret" + std::to_string(i));
}

//===----------------------------------------------------------------------===//
// Internal cells.
//===----------------------------------------------------------------------===//

std::string memCellName(llvm::StringRef owner, unsigned numBanks, unsigned bank,
                        unsigned instances, unsigned inst) {
  // The only name with no role suffix, so it escapes itself: a buffer named
  // `buf` collides with the Verilog gate primitive.
  std::string name = numBanks > 1 ? bankBase(owner, bank) : verilogName(owner);
  return instances > 1 ? name + "_c" + std::to_string(inst) : name;
}

// The memrefs of the module are the sibling namespace the tie-break runs in; an
// internal memory has no boundary port carrying an already-resolved owner.
std::string memOwnerName(const Datapath &dp, const MemUnit &m) {
  llvm::SmallVector<Value> siblings;
  for (const MemUnit &o : dp.mems)
    siblings.push_back(o.memref);
  return uniqueOwnerOf(m.memref, siblings, memOwner(m.id));
}

std::string memCellName(const Datapath &dp, const MemUnit &m, unsigned bank,
                        unsigned inst) {
  return memCellName(memOwnerName(dp, m), m.numBanks, bank, m.instances, inst);
}

std::string memArrayName(const Datapath &dp, const MemUnit &m) {
  return memCellName(memOwnerName(dp, m), m.numBanks, /*bank=*/0);
}

std::string memElemName(const Datapath &dp, const MemUnit &m, unsigned k) {
  return elemBase(memOwnerName(dp, m), k);
}

std::string regionSignal(llvm::StringRef tag, llvm::StringRef sig) {
  return verilogName(tag.str() + "_" + sig.str());
}

std::string regionSignal(unsigned region, llvm::StringRef sig) {
  return regionSignal(regionTagOf(region), sig);
}

std::string regTapName(llvm::StringRef owner, unsigned k) {
  return verilogName(owner.str() + "_d" + std::to_string(k));
}

std::string survivorName(unsigned region, unsigned k) {
  return regionSignal(region, "sv" + std::to_string(k));
}

std::string unitInstanceName(const FuncUnit &u) {
  std::string own = ownerOf(u.repOp()->getLoc(), "");
  return verilogName(own.empty() ? unitOwner(u.id)
                                 : own + "_" + unitOwner(u.id));
}

std::string childInstanceName(llvm::StringRef callee, unsigned n) {
  return verilogName(callee.str() + "_i" + std::to_string(n));
}

std::string channelSignal(llvm::StringRef chan, llvm::StringRef sig) {
  return verilogName(chan.str() + "_" + sig.str());
}

std::string operatorPredicate(const FuncUnit &u) {
  // A compare is the only IP carrying a `predicate`. Integer compare is
  // combinational, so an IP compare is floating-point.
  if (auto pred =
          dyn_cast_if_present<arith::CmpFPredicateAttr>(u.identity.predicate))
    return arith::stringifyCmpFPredicate(pred.getValue()).str();
  return "";
}

std::string operatorModuleName(const FuncUnit &u) {
  std::string pred = operatorPredicate(u);
  return pred.empty() ? u.identity.ipSymbol : u.identity.ipSymbol + "_" + pred;
}

void nameValue(Value v, llvm::StringRef name) {
  if (name.empty())
    return;
  Operation *op = v.getDefiningOp();
  if (!op) // a block argument / unresolved backedge is named elsewhere
    return;
  // Pick the channel ExportVerilog reads: a register names from its own `name`
  // attr, since sv.namehint is ignored on a reg; any other value uses namehint.
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    reg.setNameAttr(StringAttr::get(op->getContext(), name));
  else if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(op))
    ce.setNameAttr(StringAttr::get(op->getContext(), name));
  else
    op->setAttr("sv.namehint", StringAttr::get(op->getContext(), name));
}

void nameValue(Value v, Location loc) {
  if (auto name = nameFromLoc(loc))
    nameValue(v, sanitizeCppIdentifier(*name));
}

bool isNamedValue(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return true; // a block argument is named by its port
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    return reg.getName() && !reg.getName()->empty();
  if (auto ce = dyn_cast<seq::CompRegClockEnabledOp>(op))
    return ce.getName() && !ce.getName()->empty();
  return op->hasAttr("sv.namehint");
}

} // namespace mlir::allo::uarch

/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Report.h"

#include "allo/Microarch/Datapath.h"
#include "allo/Microarch/Naming.h"        // operatorModuleName, ownerOf
#include "allo/Scheduling/AddressModel.h" // applyExprOf, addressCost
#include "allo/Scheduling/MemoryModel.h"  // bankKindName

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

namespace {

/// How an array's banks decompose, as one word. Several axes of one kind read
/// as that kind; a mix has no single name and says so rather than picking the
/// first, which would report a block-and-cyclic array as either one.
std::string layoutName(const MemUnit &m) {
  if (m.layout.registers)
    return "complete";
  if (m.layout.axes.empty())
    return "none";
  BankLayout::Kind first = m.layout.axes.front().kind;
  for (const BankLayout::Axis &a : m.layout.axes)
    if (a.kind != first)
      return "mixed";
  return bankKindName(first).str();
}

/// The structure the module built for an array, as one word.
llvm::StringRef realizationName(MemUnit::Realization r) {
  switch (r) {
  case MemUnit::Realization::Boundary:
    return "boundary";
  case MemUnit::Realization::Rom:
    return "rom";
  case MemUnit::Realization::Scatter:
    return "scatter";
  case MemUnit::Realization::Ram:
    return "ram";
  }
  llvm_unreachable("every realization is named");
}

/// A signed hull [lo, hi] of the values a cell can carry.
using Hull = std::pair<int64_t, int64_t>;

/// The hull, when it fits int64; unknown on overflow.
std::optional<Hull> hull(__int128 lo, __int128 hi) {
  assert(lo <= hi && "a hull is ordered");
  if (lo < std::numeric_limits<int64_t>::min() ||
      hi > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return Hull{(int64_t)lo, (int64_t)hi};
}

/// Significant bits of a hull, the signed convention `counterWidth` sizes by.
unsigned bitsOfHull(Hull h) {
  auto bits = [](int64_t v) {
    return (unsigned)APInt(64, (uint64_t)v, /*isSigned=*/true)
        .getSignificantBits();
  };
  return std::max(bits(h.first), bits(h.second));
}

// The bits a chain register holds its proven range in, at the register's own
// signedness: a non-negative range drops the sign bit, matching the unsigned
// width the counter is built at, so the reported range never exceeds it.
unsigned storedBitsOfHull(Hull h) {
  if (h.first >= 0)
    return std::max(1u,
                    (unsigned)APInt(64, (uint64_t)h.second).getActiveBits());
  return bitsOfHull(h);
}

std::optional<Hull> rangeOfSource(const Datapath &dp, const Source &s,
                                  unsigned fuel);

/// Interval-evaluate an affine expr; dims and symbols read \p u's inputs.
std::optional<Hull> rangeOfExpr(const Datapath &dp, const FuncUnit &u,
                                AffineExpr e, unsigned numDims, unsigned fuel) {
  auto operand = [&](unsigned pos) -> std::optional<Hull> {
    return pos < u.inputs.size() ? rangeOfSource(dp, u.inputs[pos], fuel)
                                 : std::nullopt;
  };
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return Hull{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return operand(d.getPosition());
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return operand(numDims + sym.getPosition());
  auto bin = cast<AffineBinaryOpExpr>(e);
  auto lhs = rangeOfExpr(dp, u, bin.getLHS(), numDims, fuel);
  auto rhs = rangeOfExpr(dp, u, bin.getRHS(), numDims, fuel);
  if (!lhs || !rhs)
    return std::nullopt;
  auto [a, b] = *lhs;
  auto [c, d] = *rhs;
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return hull((__int128)a + c, (__int128)b + d);
  case AffineExprKind::Mul: {
    __int128 p[] = {(__int128)a * c, (__int128)a * d, (__int128)b * c,
                    (__int128)b * d};
    return hull(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
  }
  case AffineExprKind::FloorDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideFloorSigned(a, c), llvm::divideFloorSigned(b, c)};
  case AffineExprKind::CeilDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideCeilSigned(a, c), llvm::divideCeilSigned(b, c)};
  case AffineExprKind::Mod:
    if (c != d || c <= 0)
      return std::nullopt;
    return a >= 0 && b < c ? lhs : std::optional<Hull>(Hull{0, c - 1});
  default:
    return std::nullopt;
  }
}

std::optional<Hull> rangeOfUnitMath(const Datapath &dp, const FuncUnit &u,
                                    unsigned fuel);

/// The hull of unit \p u's result, transferred through its comb kind. An IP,
/// a kind with no monotone transfer, and a hull the result carrier could WRAP
/// answer unknown.
std::optional<Hull> rangeOfUnit(const Datapath &dp, const FuncUnit &u,
                                unsigned fuel) {
  std::optional<Hull> h = rangeOfUnitMath(dp, u, fuel);
  if (h && bitsOfHull(*h) > datapathWidth(u.identity.resultType))
    return std::nullopt;
  return h;
}

std::optional<Hull> rangeOfUnitMath(const Datapath &dp, const FuncUnit &u,
                                    unsigned fuel) {
  using K = CombOpKindEnum;
  if (!u.identity.comb)
    return std::nullopt;
  K kind = *u.identity.comb;
  if (kind == K::Apply) {
    AffineMap map = cast<AffineMapAttr>(u.identity.map).getValue();
    return rangeOfExpr(dp, u, applyExprOf(map), map.getNumDims(), fuel);
  }
  auto in = [&](unsigned k) -> std::optional<Hull> {
    return k < u.inputs.size() ? rangeOfSource(dp, u.inputs[k], fuel)
                               : std::nullopt;
  };
  if (kind == K::Select) {
    auto x = in(1), y = in(2);
    if (!x || !y)
      return std::nullopt;
    return Hull{std::min(x->first, y->first), std::max(x->second, y->second)};
  }
  auto x = in(0);
  if (!x)
    return std::nullopt;
  auto [a, b] = *x;
  switch (kind) {
  case K::Extsi:
  case K::IndexCast:
    return x;
  case K::Extui:
  case K::IndexCastUi: // reinterprets the bits unsigned: sound only proven >= 0
    return a >= 0 ? x : std::nullopt;
  case K::Trunci: // the wrapper refuses a hull the narrower carrier wraps
    return x;
  case K::Shli:
  case K::Shrsi:
  case K::Shrui: {
    std::optional<int64_t> sh =
        u.inputs.size() > 1 ? dp.constantOf(u.inputs[1]) : std::nullopt;
    if (!sh || *sh < 0 || *sh > 62 || (kind == K::Shrui && a < 0))
      return std::nullopt;
    int64_t p = int64_t(1) << *sh;
    if (kind == K::Shli)
      return hull((__int128)a * p, (__int128)b * p);
    // An arithmetic (or proven-non-negative logical) shift floors either sign.
    return Hull{llvm::divideFloorSigned(a, p), llvm::divideFloorSigned(b, p)};
  }
  case K::Addi:
  case K::Subi:
  case K::Muli:
  case K::Minsi:
  case K::Maxsi:
  case K::Minui:
  case K::Maxui: {
    auto y = in(1);
    if (!y)
      return std::nullopt;
    auto [c, d] = *y;
    switch (kind) {
    case K::Addi:
      return hull((__int128)a + c, (__int128)b + d);
    case K::Subi:
      return hull((__int128)a - d, (__int128)b - c);
    case K::Muli: {
      __int128 p[] = {(__int128)a * c, (__int128)a * d, (__int128)b * c,
                      (__int128)b * d};
      return hull(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
    }
    case K::Minui:
    case K::Maxui:
      if (a < 0 || c < 0)
        return std::nullopt;
      [[fallthrough]];
    case K::Minsi:
    case K::Maxsi:
      return kind == K::Minsi || kind == K::Minui
                 ? Hull{std::min(a, c), std::min(b, d)}
                 : Hull{std::max(a, c), std::max(b, d)};
    default:
      llvm_unreachable("the outer case narrowed the kind");
    }
  }
  default:
    return std::nullopt;
  }
}

/// The hull of the value \p s carries: a forward interval walk over the
/// model's cells. Unknown is always sound; \p fuel bounds recursion through
/// recurrences.
std::optional<Hull> rangeOfSource(const Datapath &dp, const Source &s,
                                  unsigned fuel) {
  if (!fuel--)
    return std::nullopt;
  switch (s.kind) {
  case Source::Kind::Const:
    if (auto c = dp.constantOf(s))
      return Hull{*c, *c};
    return std::nullopt;
  case Source::Kind::Reg: // a chain holds an older sample of the same value
    return rangeOfSource(dp, dp.regs[s.id].input, fuel);
  case Source::Kind::Counter: {
    const RegionBlock &rb = dp.regions[s.id];
    // A narrowed runtime-bound counter published its hull at derivation.
    if (rb.counterHull)
      return hull(rb.counterHull->first, rb.counterHull->second);
    auto pipe = dyn_cast_or_null<dcp::DCPathPipelineOp>(rb.op);
    std::optional<int64_t> trip = rb.tripCount ? rb.tripCount : rb.tripBound;
    if (!pipe || rb.conditional || !trip || pipe.getLbBound() ||
        pipe.getStepBound())
      return std::nullopt;
    // hull{lb, lb + trip*step}: the one-past value included, the same numbers
    // `counterWidth` sizes the register by.
    __int128 lb = pipe.getLb().value_or(0);
    __int128 last = lb + (__int128)*trip * pipe.getStep().value_or(1);
    return hull(std::min(lb, last), std::max(lb, last));
  }
  case Source::Kind::Unit:
    return rangeOfUnit(dp, dp.units[s.id], fuel);
  default: // IO, Survivor, Mem, Mux, Stream, Call
    return std::nullopt;
  }
}

/// The driving cell of a chain, as one word; a unit spells its realization.
std::string sourceClassOf(const Datapath &dp, const Source &s) {
  switch (s.kind) {
  case Source::Kind::Unit:
    return dp.units[s.id].identity.realizationName().str();
  case Source::Kind::Reg:
    return "reg";
  case Source::Kind::Mem:
    return "mem";
  case Source::Kind::Mux:
    return "mux";
  case Source::Kind::IO:
    return "io";
  case Source::Kind::Const:
    return "const";
  case Source::Kind::Counter:
    return "counter";
  case Source::Kind::Survivor:
    return "survivor";
  case Source::Kind::Stream:
    return "stream";
  case Source::Kind::Call:
    return "call";
  case Source::Kind::None:
    return "none";
  }
  llvm_unreachable("every source kind is named");
}

/// The multiplexers of one region, aggregated by (fan-in, width).
std::vector<MuxClass> muxClasses(const Datapath &dp, const RegionBlock &rb) {
  std::map<std::pair<unsigned, unsigned>, unsigned> byClass;
  for (MuxId mid : rb.muxes)
    ++byClass[{(unsigned)dp.muxes[mid].sources.size(),
               datapathWidth(dp.muxes[mid].type)}];
  std::vector<MuxClass> out;
  out.reserve(byClass.size());
  for (const auto &[key, count] : byClass)
    out.push_back({key.first, key.second, count});
  return out;
}

} // namespace

FuncUarch::FuncUarch(const Datapath &dp, llvm::StringRef symbol,
                     llvm::StringRef module, const RegLedger &ledger,
                     const MuxLedger &muxes,
                     std::vector<TimingPath> criticalPaths)
    : func(symbol.str()), module(module.str()), top(dp.atTop),
      regs(ledger.classes()), muxCones(muxes.classes()),
      readPorts(dp.readPorts.size()), writePorts(dp.writePorts.size()),
      criticalPaths(std::move(criticalPaths)) {
  for (const RegionBlock &rb : dp.regions) {
    RegionUarch r;
    r.order = rb.id;
    r.shape = shapeName(rb.shape).str();
    r.kind = rb.kind == RegionBlock::Kind::Cyclic ? "cyclic" : "acyclic";
    if (rb.ii)
      r.interval = (int64_t)*rb.ii;
    for (const RegionBlock::AddrStride &s : rb.addrStrides) {
      // The counter-aliased stride builds no register, so it does not count
      // among the ones riding beside the counter.
      if (!s.isCounter)
        ++r.cost.addrStrides;
      r.cost.strides.push_back(
          {s.width, s.step != 0, s.hasCarry, s.wrap != 0, s.isCounter});
    }
    if (rb.counterType)
      r.cost.counterWidth = datapathWidth(rb.counterType);
    // The phase counter exists exactly where `emitPipelined` builds one: a
    // schedule-paced leaf issuing once per II.
    if (rb.shape == RegionBlock::Shape::Leaf &&
        rb.kind == RegionBlock::Kind::Cyclic && rb.ii && *rb.ii > 1)
      r.cost.phaseWidth = llvm::Log2_64_Ceil(*rb.ii);
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      r.computeOps += u.boundOps.size();
      UnitReport ur{u.identity.key(),
                    u.identity.ipSymbol,
                    u.identity.comb ? std::string() : operatorModuleName(u),
                    datapathWidth(u.identity.resultType),
                    u.latency,
                    (unsigned)u.boundOps.size(),
                    u.identity.comb.has_value(),
                    u.pipelined};
      if (u.identity.comb == CombOpKindEnum::Apply) {
        AffineMap map = cast<AffineMapAttr>(u.identity.map).getValue();
        AddressCost cone = addressCost(applyExprOf(map), AddressDelays{},
                                       AddressDelays::refWidth);
        ur.adders = cone.adders;
        ur.multipliers = cone.multipliers;
        ur.dividers = cone.dividers;
      }
      r.units.push_back(std::move(ur));
    }
    r.muxes = muxClasses(dp, rb);
    for (const MuxClass &m : r.muxes) {
      r.cost.muxInputs += m.count * m.fanin;
      // A k:1 mux costs about (k-1) 2:1 muxes per bit.
      r.cost.muxBits += m.count * m.width * (m.fanin - 1);
    }
    for (RegId rid : rb.regs) {
      const Register &rg = dp.regs[rid];
      ChainReport cr;
      cr.region = rb.id;
      cr.width = datapathWidth(rg.type);
      llvm::raw_string_ostream(cr.carried) << rg.type;
      cr.depth = rg.depth;
      cr.ii = rb.ii.value_or(1);
      cr.taps = rg.taps.size();
      cr.source = sourceClassOf(dp, rg.input);
      if (auto h = rangeOfSource(dp, rg.input, /*fuel=*/16))
        cr.rangeBits = storedBitsOfHull(*h);
      chains.push_back(std::move(cr));
    }
    regions.push_back(std::move(r));
  }

  for (const MemUnit &m : dp.mems) {
    MemReport mr;
    mr.owner = memArrayName(dp, m);
    auto mt = cast<MemRefType>(m.memref.getType());
    mr.shape.assign(mt.getShape().begin(), mt.getShape().end());
    mr.width = m.width;
    mr.banks = m.numBanks;
    mr.layout = layoutName(m);
    mr.storage = m.storage;
    mr.depthWords = m.depthWords;
    mr.readLatency = m.readLatency;
    mr.writeLatency = m.writeLatency;
    for (const MemUnit::Access &acc : m.accesses)
      (acc.isWrite ? mr.writes : mr.reads)++;
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id)
          (ma.isWrite ? mr.cost.callWrites : mr.cost.callReads)++;
    mr.cost.readPorts = m.readPortsBuilt;
    mr.cost.writePorts = m.writePortsBuilt;
    mr.cost.ports = m.portsBuilt;
    mr.cost.instances = m.instances;
    mr.cost.copiesBudget = m.ports.copies();
    mr.cost.rowReads = m.ports.instReads.value_or(0);
    mr.cost.rowWrites = m.ports.instWrites.value_or(0);
    mr.cost.readConcurrency = m.readConcurrency;
    mr.cost.writeConcurrency = m.writeConcurrency;
    mr.cost.boundaryPorts = m.boundaryPorts;
    mr.realization = realizationName(m.realization());
    mr.external = m.external;
    mr.scattered = m.scattered;
    mr.writesIndependent = m.writesIndependent;
    mr.rom = m.isRom;
    mr.skewed = m.skewed;
    // A skew resolves a slot rather than a bank, which still counts as
    // resolved: the array shares one port per lane.
    mr.partitionResolved =
        m.numBanks <= 1 ||
        llvm::all_of(m.accesses, [](const MemUnit::Access &a) {
          return a.staticBank || a.slot;
        });
    mems.push_back(std::move(mr));
  }

  for (const StreamChannel &s : dp.streams)
    streams.push_back({ownerOf(s.stream, chanOwner(s.id)),
                       datapathWidth(s.payload), s.depth, !s.callEnds.empty(),
                       s.internal});

  std::map<std::string, CallReport> byCallee;
  for (const CallUnit &cu : dp.calls) {
    CallReport &c = byCallee[cu.callee];
    c.callee = cu.callee;
    ++c.count;
    c.spawns += cu.async;
    c.latency = cu.latency;
    switch (cu.startPolicy) {
    case CallUnit::StartPolicy::Handshake:
      ++c.handshake;
      break;
    case CallUnit::StartPolicy::Broadcast:
      ++c.broadcast;
      break;
    case CallUnit::StartPolicy::TimeTriggered:
      ++c.timed;
      break;
    }
  }
  for (auto &[name, c] : byCallee)
    calls.push_back(std::move(c));
}

std::string MicroarchReport::toJSON() const {
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::OStream j(os);
  j.object([&] {
    j.attribute("version", (int64_t)kVersion);
    j.attribute("binding", binding);
    j.attribute("cycle_time", cycleTime);
    j.attributeArray("funcs", [&] {
      for (const FuncUarch &f : funcs)
        j.object([&] {
          j.attribute("func", f.func);
          j.attribute("module", f.module);
          j.attribute("top", f.top);
          j.attribute("read_ports", (int64_t)f.readPorts);
          j.attribute("write_ports", (int64_t)f.writePorts);
          j.attribute("critical_ns", f.criticalPath());
          j.attributeArray("critical_paths", [&] {
            for (const TimingPath &p : f.criticalPaths)
              j.object([&] {
                j.attribute("total_ns", p.total);
                j.attribute("slack_ns", p.slack);
                j.attribute("endpoint", p.endpoint);
                if (!p.where.empty())
                  j.attribute("where", p.where);
                j.attributeArray("steps", [&] {
                  for (const TimingStep &s : p.steps)
                    j.object([&] {
                      j.attribute("what", s.what);
                      j.attribute("ns", s.delay);
                    });
                });
              });
          });
          j.attributeArray("regions", [&] {
            for (const RegionUarch &r : f.regions)
              j.object([&] {
                j.attribute("order", r.order);
                j.attribute("shape", r.shape);
                j.attribute("kind", r.kind);
                if (r.interval)
                  j.attribute("interval", *r.interval);
                j.attribute("compute_ops", (int64_t)r.computeOps);
                j.attributeObject("cost", [&] {
                  j.attribute("mux_inputs", (int64_t)r.cost.muxInputs);
                  j.attribute("mux_bits", (int64_t)r.cost.muxBits);
                  j.attribute("counter_width", (int64_t)r.cost.counterWidth);
                  j.attribute("phase_width", (int64_t)r.cost.phaseWidth);
                  j.attribute("addr_strides", (int64_t)r.cost.addrStrides);
                  j.attributeArray("strides", [&] {
                    for (const StrideCost &s : r.cost.strides)
                      j.object([&] {
                        j.attribute("width", (int64_t)s.width);
                        j.attribute("step", s.step);
                        j.attribute("carry", s.carry);
                        j.attribute("wrap", s.wrap);
                        j.attribute("is_counter", s.isCounter);
                      });
                  });
                });
                j.attributeArray("units", [&] {
                  for (const UnitReport &u : r.units)
                    j.object([&] {
                      j.attribute("identity", u.identity);
                      if (!u.impl.empty())
                        j.attribute("impl", u.impl);
                      if (!u.module.empty())
                        j.attribute("module", u.module);
                      j.attribute("width", (int64_t)u.width);
                      j.attribute("latency", (int64_t)u.latency);
                      j.attribute("bound_ops", (int64_t)u.boundOps);
                      j.attribute("comb", u.comb);
                      j.attribute("pipelined", u.pipelined);
                      if (u.adders)
                        j.attribute("adders", (int64_t)u.adders);
                      if (u.multipliers)
                        j.attribute("multipliers", (int64_t)u.multipliers);
                      if (u.dividers)
                        j.attribute("dividers", (int64_t)u.dividers);
                    });
                });
                j.attributeArray("muxes", [&] {
                  for (const MuxClass &m : r.muxes)
                    j.object([&] {
                      j.attribute("fanin", (int64_t)m.fanin);
                      j.attribute("width", (int64_t)m.width);
                      j.attribute("count", (int64_t)m.count);
                    });
                });
              });
          });
          j.attributeArray("regs", [&] {
            for (const RegClass &c : f.regs)
              j.object([&] {
                j.attribute("role", roleName(c.role));
                j.attribute("width", (int64_t)c.width);
                j.attribute("depth", (int64_t)c.depth);
                j.attribute("count", (int64_t)c.count);
                j.attribute("reset", c.reset);
                j.attribute("enable", c.enable);
              });
          });
          j.attributeArray("chains", [&] {
            for (const ChainReport &c : f.chains)
              j.object([&] {
                j.attribute("region", c.region);
                j.attribute("width", (int64_t)c.width);
                j.attribute("carried", c.carried);
                j.attribute("depth", (int64_t)c.depth);
                j.attribute("ii", (int64_t)c.ii);
                j.attribute("taps", (int64_t)c.taps);
                j.attribute("source", c.source);
                if (c.rangeBits)
                  j.attribute("range_bits", (int64_t)*c.rangeBits);
              });
          });
          j.attributeArray("mux_cones", [&] {
            for (const MuxCone &m : f.muxCones)
              j.object([&] {
                j.attribute("role", muxRoleName(m.role));
                j.attribute("fanin", (int64_t)m.fanin);
                j.attribute("width", (int64_t)m.width);
                j.attribute("count", (int64_t)m.count);
              });
          });
          j.attributeArray("mems", [&] {
            for (const MemReport &m : f.mems)
              j.object([&] {
                j.attribute("owner", m.owner);
                j.attributeArray("shape", [&] {
                  for (int64_t d : m.shape)
                    j.value(d);
                });
                j.attribute("width", (int64_t)m.width);
                j.attribute("banks", (int64_t)m.banks);
                j.attribute("layout", m.layout);
                j.attribute("storage", m.storage);
                j.attribute("depth_words", (int64_t)m.depthWords);
                j.attribute("read_latency", (int64_t)m.readLatency);
                j.attribute("write_latency", (int64_t)m.writeLatency);
                j.attribute("reads", (int64_t)m.reads);
                j.attribute("writes", (int64_t)m.writes);
                j.attributeObject("cost", [&] {
                  j.attribute("call_reads", (int64_t)m.cost.callReads);
                  j.attribute("call_writes", (int64_t)m.cost.callWrites);
                  j.attribute("read_ports", (int64_t)m.cost.readPorts);
                  j.attribute("write_ports", (int64_t)m.cost.writePorts);
                  j.attribute("ports", (int64_t)m.cost.ports);
                  j.attribute("instances", (int64_t)m.cost.instances);
                  j.attribute("copies_budget", (int64_t)m.cost.copiesBudget);
                  j.attribute("row_reads", (int64_t)m.cost.rowReads);
                  j.attribute("row_writes", (int64_t)m.cost.rowWrites);
                  j.attribute("read_concurrency",
                              (int64_t)m.cost.readConcurrency);
                  j.attribute("write_concurrency",
                              (int64_t)m.cost.writeConcurrency);
                  j.attribute("boundary_ports", (int64_t)m.cost.boundaryPorts);
                });
                j.attribute("external", m.external);
                j.attribute("scattered", m.scattered);
                j.attribute("writes_independent", m.writesIndependent);
                j.attribute("realization", m.realization);
                j.attribute("rom", m.rom);
                j.attribute("skewed", m.skewed);
                j.attribute("partition_resolved", m.partitionResolved);
              });
          });
          j.attributeArray("streams", [&] {
            for (const StreamReport &s : f.streams)
              j.object([&] {
                j.attribute("owner", s.owner);
                j.attribute("width", (int64_t)s.width);
                j.attribute("depth", (int64_t)s.depth);
                j.attribute("crosses_call", s.crossesCall);
                j.attribute("internal", s.internal);
              });
          });
          j.attributeArray("calls", [&] {
            for (const CallReport &c : f.calls)
              j.object([&] {
                j.attribute("callee", c.callee);
                j.attribute("count", (int64_t)c.count);
                j.attribute("spawns", (int64_t)c.spawns);
                j.attribute("handshake", (int64_t)c.handshake);
                j.attribute("broadcast", (int64_t)c.broadcast);
                j.attribute("timed", (int64_t)c.timed);
                if (c.latency)
                  j.attribute("latency", *c.latency);
              });
          });
        });
    });
  });
  return out;
}

} // namespace mlir::allo::uarch

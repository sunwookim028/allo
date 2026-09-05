/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Interface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::uarch;

namespace mlir::allo::iface {

namespace {
int argOf(Value v) {
  auto ba = dyn_cast<BlockArgument>(v);
  return ba ? (int)ba.getArgNumber() : -1;
}

// The element-space bank decomposition of \p mu, in the manifest's shape: the
// host reproduces it to shard an argument across the bank interfaces exactly as
// the emitted address arithmetic does.
std::pair<std::vector<int64_t>, std::vector<Memory::Axis>>
layoutOf(const uarch::MemUnit &mu) {
  auto shape = cast<MemRefType>(mu.memref.getType()).getShape();
  std::vector<Memory::Axis> axes;
  for (const BankLayout::Axis &a : mu.layout.axes)
    axes.push_back({(int)a.dim, a.factor, bankKindName(a.kind).str()});
  return {{shape.begin(), shape.end()}, std::move(axes)};
}

// One boundary interface on \p mu; every field but the five passed in is
// derived from the memory.
Memory memPort(const uarch::MemUnit &mu, bool write, bool independent,
               unsigned bank, unsigned factor, llvm::StringRef base) {
  auto [shape, axes] = layoutOf(mu);
  return {argOf(mu.memref),
          write,
          write && independent,
          (int)bank,
          (int)factor,
          mu.width,
          write ? mu.writeLatency : mu.readLatency,
          base.str(),
          portAddr(base),
          portData(base),
          write ? portWe(base) : std::string(),
          std::move(shape),
          std::move(axes)};
}
} // namespace

ModuleInterface::ModuleInterface(const uarch::Datapath &dp) {
  auto fn = dp.func;
  // Legalized here so the manifest key is the emitted Verilog module name;
  // ExportVerilog would otherwise rewrite a nested callee like `top.child`.
  symbol = fn.getSymName().str();
  module = uarch::verilogName(symbol);

  // The timing contract, republished from the op that declares it.
  if (std::optional<uint64_t> lat = fn.getLatency())
    latency = (int64_t)*lat;
  latencyBound = latency.has_value() && fn.getLatencyBound();
  determinacy = stringifyDeterminacyEnum(fn.getDeterminacy()).str();

  // Every IOPort is a scalar kernel argument; a scalar result is a `dp.results`
  // entry, declared further down.
  for (const uarch::IOPort &io : dp.ios)
    scalars.push_back(
        {argOf(io.value), datapathWidth(io.type), scalarPortName(dp, io)});

  for (const uarch::StreamChannel &s : dp.streams) {
    if (s.internal)
      continue; // kernel-local: a seq.fifo in the body, not a boundary port
    auto base = streamPortBase(dp, s);
    streams.push_back({argOf(s.stream), s.isInput, (int)s.depth,
                       datapathWidth(s.payload), base, portData(base),
                       portValid(base), portReady(base)});
  }

  // A scattered argument is declared per element, off the memory rather than
  // off its accesses, so it appears in neither `reads` nor `writes`. A
  // scattered internal array is registers in the body and reaches no port.
  for (const uarch::MemUnit &mu : dp.mems) {
    if (!mu.scattered || !mu.external)
      continue;
    auto mt = cast<MemRefType>(mu.memref.getType());
    auto shape = mt.getShape();
    std::vector<RegisterFile::Element> elems;
    for (const uarch::MemUnit::ElemPort &p : mu.elemPorts)
      elems.push_back({p.in, p.out, p.we});
    registers.push_back({argOf(mu.memref),
                         datapathWidth(mt.getElementType()),
                         {shape.begin(), shape.end()},
                         std::move(elems)});
  }

  // Each external access expands to one interface per boundary bank: one when
  // unbanked or statically routed, N for a data-dependent access.
  auto group = [&](uarch::AccRef r, bool write) {
    const auto &mu = dp.mems[r.id];
    const auto &acc = mu.accesses[r.idx];
    unsigned factor = externalBank(mu, acc).factor;
    std::vector<Memory> g;
    for (const auto &[bank, base] : extPorts(mu, acc))
      g.push_back(memPort(mu, write, mu.writesIndependent, bank, factor, base));
    return g;
  };
  for (uarch::AccRef r : dp.readPorts)
    reads.push_back(group(r, /*write=*/false));
  for (uarch::AccRef r : dp.writePorts)
    writes.push_back(group(r, /*write=*/true));

  // A CallUnit-mastered boundary argument has no MemUnit::Access (the child
  // drives the port), so it is declared here with the same `<name>_<role><i>`
  // naming as a normal port.
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (!ma.isBoundary || !ma.ownsGroup)
        continue;
      // One port group per (bank, port) colour, declared by the child that
      // opens it: concurrent accessors keep separate groups backed by the same
      // array, and a cyclic argument gets one group per bank.
      const auto &mu = dp.mems[ma.mem];
      Memory m = memPort(mu, ma.isWrite, ma.independent, ma.bank, ma.factor,
                         ma.topBase);
      (ma.isWrite ? writes : reads).push_back({m});
    }

  for (const uarch::Result &r : dp.results)
    results.push_back({datapathWidth(r.type), r.name});

  // One entry per extern operator module this kernel instantiates, with the
  // ports it is declared with, which `declareOperatorModules` builds the extern
  // from. A native (comb) unit emits inline and declares nothing. Deduplicated
  // by module name, which also decides whether two units may share a module.
  llvm::StringMap<const allo::OperatorIdentity *> listed;
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.identity.comb)
      continue;
    std::string modName = uarch::operatorModuleName(u);
    auto [claim, fresh] = listed.try_emplace(modName, &u.identity);
    assert(*claim->second == u.identity &&
           "two operator identities share one module name");
    if (!fresh)
      continue;
    Operator entry{
        modName, u.identity.ipSymbol, uarch::operatorPredicate(u), {}};
    // Widths come from the identity, which is what the module name separates.
    for (auto [k, argType] : llvm::enumerate(u.identity.argTypes))
      entry.ports.push_back({std::string(1, static_cast<char>('a' + k)),
                             datapathWidth(argType), Operator::Role::Data});
    entry.ports.push_back({uarch::kClk.str(), 1, Operator::Role::Clk});
    // `ce == 0` freezes the IP in lockstep with the shell; a free-running one
    // has no such port.
    if (u.stall == allo::StallContractEnum::Ce)
      entry.ports.push_back({uarch::kCe.str(), 1, Operator::Role::Ce});
    entry.ports.push_back({uarch::kOpOut.str(),
                           datapathWidth(u.identity.resultType),
                           Operator::Role::Out});
    operators.push_back(std::move(entry));
  }
}

llvm::SmallVector<const Memory *, 2>
ModuleInterface::portsForArg(int arg) const {
  llvm::SmallVector<const Memory *, 2> out;
  for (const std::vector<std::vector<Memory>> *side : {&reads, &writes})
    for (const std::vector<Memory> &grp : *side)
      for (const Memory &m : grp)
        if (m.arg == arg)
          out.push_back(&m);
  return out;
}

const FIFO *ModuleInterface::streamForArg(int arg) const {
  for (const FIFO &s : streams)
    if (s.arg == arg)
      return &s;
  return nullptr;
}

const Scalar *ModuleInterface::scalarForArg(int arg) const {
  for (const Scalar &s : scalars)
    if (s.arg == arg)
      return &s;
  return nullptr;
}

std::string ModuleInterface::toJSON() const {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  auto mems = [](const std::vector<std::vector<Memory>> &accs) {
    Array out;
    for (const auto &acc : accs) {
      Array banks;
      for (const Memory &p : acc) {
        Object o{{"arg", p.arg},
                 {"bank", p.bank},
                 {"factor", p.factor},
                 {"width", (int64_t)p.width},
                 {"latency", (int64_t)p.latency},
                 {"base", p.base},
                 {"addr", p.addr},
                 {"data", p.data}};
        if (!p.we.empty())
          o["we"] = p.we;
        // The bank decomposition, published only for a partitioned argument:
        // the host shards its numpy array with it (see `plan_mems`).
        if (!p.axes.empty()) {
          Array shape;
          for (int64_t d : p.shape)
            shape.push_back(d);
          o["shape"] = std::move(shape);
          Array axes;
          for (const Memory::Axis &a : p.axes)
            axes.push_back(
                Object{{"dim", a.dim}, {"factor", a.factor}, {"kind", a.kind}});
          o["axes"] = std::move(axes);
        }
        banks.push_back(std::move(o));
      }
      out.push_back(std::move(banks));
    }
    return out;
  };

  Array scalars;
  for (const Scalar &s : this->scalars)
    scalars.push_back(
        Object{{"arg", s.arg}, {"width", (int64_t)s.width}, {"name", s.name}});
  Array streams;
  for (const FIFO &s : this->streams)
    streams.push_back(Object{{"arg", s.arg},
                             {"input", s.isInput},
                             {"depth", s.depth},
                             {"width", (int64_t)s.width},
                             {"base", s.base},
                             {"data", s.data},
                             {"valid", s.valid},
                             {"ready", s.ready}});
  Array registers;
  for (const RegisterFile &rf : this->registers) {
    Array shape, elements;
    for (int64_t d : rf.shape)
      shape.push_back(d);
    // An unused direction has no port, so its key is absent rather than empty.
    for (const RegisterFile::Element &e : rf.elements) {
      Object o;
      if (!e.in.empty())
        o["in"] = e.in;
      if (!e.out.empty()) {
        o["out"] = e.out;
        o["we"] = e.we;
      }
      elements.push_back(std::move(o));
    }
    registers.push_back(Object{{"arg", rf.arg},
                               {"width", (int64_t)rf.width},
                               {"shape", std::move(shape)},
                               {"elements", std::move(elements)}});
  }
  Array results;
  for (const Result &r : this->results)
    results.push_back(Object{{"width", (int64_t)r.width}, {"name", r.name}});
  Array operators;
  for (const Operator &o : this->operators) {
    Array ports;
    for (const Operator::Port &p : o.ports) {
      llvm::StringRef role = p.role == Operator::Role::Data  ? "data"
                             : p.role == Operator::Role::Clk ? "clk"
                             : p.role == Operator::Role::Ce  ? "ce"
                                                             : "out";
      ports.push_back(Object{{"name", p.name},
                             {"width", (int64_t)p.width},
                             {"role", role},
                             {"input", p.isInput()}});
    }
    operators.push_back(Object{{"module", o.module},
                               {"impl", o.impl},
                               {"predicate", o.predicate},
                               {"ports", std::move(ports)}});
  }

  Object root{{"module", module},
              {"symbol", symbol},
              // The start->done contract. `latency` is omitted, not null, when
              // the span is data-dependent.
              {"latency_bound", latencyBound},
              {"determinacy", determinacy},
              // The fixed control ABI, published so no consumer hard-codes it.
              {"control", Object{{"clk", uarch::kClk},
                                 {"rst", uarch::kRst},
                                 {"start", uarch::kStart},
                                 {"done", uarch::kDone}}},
              {"scalars", std::move(scalars)},
              {"streams", std::move(streams)},
              {"reads", mems(reads)},
              {"writes", mems(writes)},
              {"registers", std::move(registers)},
              {"results", std::move(results)},
              {"operators", std::move(operators)}};
  if (latency)
    root["latency"] = *latency;
  std::string s;
  llvm::raw_string_ostream os(s);
  os << Value(std::move(root));
  return s;
}

} // namespace mlir::allo::iface

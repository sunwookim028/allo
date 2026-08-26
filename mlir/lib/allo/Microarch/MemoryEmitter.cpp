/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The memory subsystem's emit half: how an access reaches its storage. Three
// dispatches on `PortPlan` (`emitReads`, `emitWrites`, `masterCallPorts`), plus
// the address hardware they share and the finalizers a port shared between
// regions needs. What the storage is, and which ports each access holds, is
// decided in Memory.cpp.
//===----------------------------------------------------------------------===//

#include "allo/IR/AlloOps.h" // kIndependentWritesAttr
#include "allo/Microarch/HWEmitter.h"
#include "allo/Support/Logging.h" // logging::debug

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// An address on its way out of the module. A boundary address port is
// `kDatapathAddressWidth` wide for every argument, the contract the manifest
// and the cosim harness are written against, so a narrow in-bank address widens
// back here.
static Value boundaryAddr(EmitContext &c, Value addr) {
  return addrAt(c.b, c.loc, addr, kDatapathAddressWidth);
}

// Which of \p vals bank \p k takes, each tagged by the bank it reaches
// (\p banks, parallel to it): the inverse of `readCrossbar`. At most one tag
// equals `k`, since a lane holds distinct slots and distinct slots are distinct
// banks at every rotation, so the selects are one-hot; with no tag on `k` the
// result is 0, a don't-care behind the port's own enable.
static Value laneSelect(EmitContext &c, ArrayRef<Value> banks,
                        ArrayRef<Value> vals, unsigned k) {
  assert(banks.size() == vals.size() &&
         "a lane's bank tags are parallel to its values");
  if (vals.size() == 1)
    return vals.front();
  c.muxLedger.add(MuxRole::Crossbar, vals.size(),
                  datapathWidth(vals.front().getType()));
  SmallVector<Value> sels;
  for (Value bank : banks)
    sels.push_back(c.icmpEq(bank, k));
  return c.oneHotSelect(vals, sels);
}

// Build one cone \p r of this access's address as hardware at \p width, out of
// the parts `planAddressGenerators` split it into: a constant, one register per
// strength-reduced term (`RegionBlock::addrStrides`, advanced by the
// controller), and whatever did not reduce. The residual is added after the
// delay chain, its operands arriving already delayed where the counters run
// live, which puts both halves in the access's own cycle.
Value DatapathEmitter::buildAddr(const uarch::MemUnit::Access &acc,
                                 const uarch::MemUnit::Access::Reduced &r,
                                 unsigned width) {
  StallShell sh = shellFor(acc.region);
  Value phase = controlOf.lookup(acc.region).phase;
  unsigned ii = dp.regions[acc.region].ii.value_or(1);
  // The counters a delayed cone reads are fresh at cycle 0 of their iteration,
  // so under a published phase the delay folds onto it: `ceil(addrDelay / ii)`
  // registers instead of one per cycle of the access's stage.
  auto delayed = [&](Value v) {
    if (!acc.addrDelay)
      return v;
    return phase && acc.addrDelay > 1
               ? c.foldedChain(v, acc.addrDelay, ii, phase, /*ready=*/0, sh)
                     .last()
               : c.shiftChain(v, acc.addrDelay, sh).last();
  };
  Value addr;
  auto add = [&](Value v) {
    addr =
        addr ? comb::AddOp::create(c.b, c.loc, addr, v, false).getResult() : v;
  };
  if (r.base)
    add(c.konst(c.b.getIntegerType(width), r.base));
  for (const uarch::MemUnit::Access::ScaledTerm &t : r.terms) {
    const uarch::RegionControl &rc = controlOf.lookup(t.region);
    assert(t.slot < rc.scaledCounters.size() &&
           "a reduced address term has no scaled counter in its region");
    add(addrAt(c.b, c.loc, rc.scaledCounters[t.slot], width));
  }
  if (addr)
    addr = delayed(addr);
  if (r.residual) {
    // A register the residual reads runs live like a counter, so each is
    // delayed on its own. Appended at the datapath width, which is what
    // `evalAffine` reads its operands at.
    SmallVector<Value> idx; // the access's own index sources, dims then symbols
    // An operand the reduction folded into a scaled counter has an empty slot
    // and no position in this residual, so nothing reads the gap.
    for (const uarch::Source &s : acc.addr)
      idx.push_back(s ? resolveSource(s) : Value());
    for (const uarch::MemUnit::Access::ScaledTerm &t : r.reads) {
      const uarch::RegionControl &rc = controlOf.lookup(t.region);
      assert(t.slot < rc.scaledCounters.size() &&
             "a residual's digit has no scaled counter in its region");
      // Delayed at the counter's own width and widened after: the chain then
      // costs the digit's bits, not the datapath's.
      Value v = delayed(rc.scaledCounters[t.slot]);
      idx.push_back(addrAt(c.b, c.loc, v, kDatapathAddressWidth));
    }
    add(evalAffine(c.b, c.loc, r.residual, idx, acc.addrMap.getNumDims(),
                   width));
  }
  // Nothing at all: the access sits at a fixed element of a one-word bank.
  return addr ? addr : c.konst(c.b.getIntegerType(width), 0);
}

// The address hardware of one access: the element index within the bank it
// reaches, plus the bank digit when that is decided at runtime. Uniform over
// banked and unbanked, since an unpartitioned memref is a one-bank layout whose
// offset is the flat index and whose digit nothing builds. Both halves are the
// `Reduced` cones `planAddressGenerators` already split.
BankSplit DatapathEmitter::bankAddress(const uarch::MemUnit &m,
                                       const uarch::MemUnit::Access &acc) {
  assert(acc.addrMap && "dcp memory access without an affine map");
  Value offset = buildAddr(acc, acc.offset, memAddrWidth(m));
  // The digit's cone is built at the datapath width so its intermediates keep
  // their range, then narrowed to clog2(numBanks), the width `icmpEq` compares
  // it against literal bank numbers at. It reduces like the offset: `counter
  // mod F` is a register that wraps, not a `mod` on the setup path.
  Value bank =
      acc.hasBankCone
          ? addrAt(c.b, c.loc, buildAddr(acc, acc.bank, kDatapathAddressWidth),
                   std::max(1u, llvm::Log2_64_Ceil(m.numBanks)))
          : Value();
  // The hold is elsewhere: a boundary read address is held once where it leaves
  // the module (`sharedAddress`, the crossbar read), and an internal port keeps
  // its in-flight datum through its read enable.
  return {bank, offset};
}

// Narrow a child's port address to the clog2(depth)-bit index `seq.hlmem` /
// `hw.array_get` expects. This module's own accesses need no narrowing:
// `bankAddress` already carries its arithmetic at that width.
Value DatapathEmitter::memAddr(const uarch::MemUnit &m, Value addr) {
  return addrAt(c.b, c.loc, addr, memAddrWidth(m));
}

// Which element of a scattered memory \p acc names, at the memory's own
// address width. The crossbar and the write demux compare it against literal
// element numbers (`icmpEq` builds those at its width).
Value DatapathEmitter::scatterIndex(const uarch::MemUnit &m,
                                    const uarch::MemUnit::Access &acc) {
  assert(m.scattered && "an element index belongs to a scattered memory");
  return bankAddress(m, acc).offset;
}

// The element registers of scattered internal array \p id, in element order.
// They are backedges until `finalizeScatteredPorts` resolves them, so a reader
// takes them without waiting for the stores that drive them.
SmallVector<Value> DatapathEmitter::scatterValues(unsigned id) {
  auto it = scatterElems.find(id);
  assert(it != scatterElems.end() && "no element registers for this array");
  return {it->second.begin(), it->second.end()};
}

// Bind the read-data input ports into readData, once, before the per-region
// loop (external memories only; internal ones read via seq.read below). Every
// access of a port group takes the same data input, since they never issue
// together. A data-dependent banked read has one data port per bank and is
// bound by emitReads, which muxes them in-region.
void DatapathEmitter::bindReadPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.external)
      continue;
    for (auto [i, acc] : llvm::enumerate(m.accesses))
      // A write, or a plan whose datum is a select over several ports rather
      // than one port's: both are bound by `emitReads`. A Coloured access is
      // statically banked and holds the one interface `acc.portBase` names.
      if (!acc.isWrite && acc.plan == PortPlan::Coloured)
        readData[accKey(m.id, i)] = pa.getInput(portData(acc.portBase));
  }
}

// Instantiate on-chip storage for each internal (non-argument) memory: one
// seq.hlmem, or one per bank when the array reached emit still partitioned (a
// data-dependent bank `dcp-resolve-banking` could not split statically). The
// handles are module-scope so writes and reads in different regions share them.
void DatapathEmitter::createInternalMemories() {
  using R = uarch::MemUnit::Realization;
  for (const uarch::MemUnit &m : dp.mems) {
    R realization = m.realization();
    if (realization == R::Boundary)
      continue;
    IntegerType elemTy = memElemType(m, c.b);
    unsigned depth = declaredDepth(m.depthWords);
    if (realization == R::Rom) {
      // A constant table: one hw.aggregate_constant holding the global's
      // initializer, read combinationally by hw.array_get and registered to the
      // read latency in emitReads. No writable hlmem and no write ports.
      SmallVector<Attribute> fields;
      for (const APInt &w :
           initWords(cast<ElementsAttr>(m.romInit), m.width, depth))
        fields.push_back(IntegerAttr::get(elemTy, w));
      // A hw.array indexes element 0 as the last aggregate_constant field, so
      // the natural-order initializer is reversed to make array_get(i) ==
      // data[i].
      std::reverse(fields.begin(), fields.end());
      romArray[m.id] = hw::AggregateConstantOp::create(
          c.b, c.loc, hw::ArrayType::get(elemTy, depth),
          c.b.getArrayAttr(fields));
      continue;
    }
    // A completely partitioned array is one register per element rather than an
    // addressed memory, which is what buys the unlimited combinational ports
    // the scheduler was billed against. Only the backedges here; the registers
    // need every store, so `finalizeScatteredPorts` builds them. Exactly
    // `depthWords` of them, not `declaredDepth`: the padding word only keeps an
    // hlmem's address one bit wide, and an element is selected by comparison.
    if (realization == R::Scatter) {
      SmallVector<Backedge> elems;
      for (unsigned k = 0; k < m.depthWords; ++k)
        elems.push_back(c.bb.get(elemTy));
      scatterElems[m.id] = std::move(elems);
      continue;
    }
    // One cell per instance of each bank, bank-major. Reads past what one
    // instance of the row has are served by another copy of the whole array.
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k)
      for (unsigned i = 0; i < m.instances; ++i) {
        auto mem = seq::HLMemOp::create(c.b, c.loc, c.clk, c.rst,
                                        memCellName(dp, m, k, i),
                                        {static_cast<int64_t>(depth)}, elemTy);
        // The port binding proved these writes never collide, which lets the
        // lowering put each in its own `always` block and build a true dual
        // port. Without it they share one block, which arbitrates.
        if (m.writesIndependent)
          mem->setAttr(kIndependentWritesAttr, c.b.getUnitAttr());
        // Pin the array to the row it is realized in. Leaving it unsaid hands
        // the structure to the synthesizer, which then builds something the
        // cost model did not price. Arbitrated writes are the exception: they
        // share one `always` block, no RAM is inferred, and the pin sits on a
        // register array the cost model priced as the row.
        if (!m.ramStyle.empty()) {
          mem->setAttr(kRamStyleAttr, c.b.getStringAttr(m.ramStyle));
          if (m.writePortsBuilt > 1 && !m.writesIndependent)
            logging::debug(logging::Stage::Emit)
                << memCellName(dp, m, k, i) << " is pinned to ram_style '"
                << m.ramStyle
                << "' but its writes are arbitrated in one block, which "
                   "infers no RAM; the style row's price does not match what "
                   "synthesis builds";
        }
        // An initialized array the kernel also writes is a real memory that
        // starts with contents. `seq.hlmem` carries no initializer, so the
        // words ride to the seq->SV pipeline, which gives the backing reg an
        // `initial` block. Every copy starts with them.
        if (m.romInit)
          recordMemoryInit(
              mem, initWords(cast<ElementsAttr>(m.romInit), m.width, depth));
        banks.push_back(mem.getHandle());
      }
    memBanks[m.id] = std::move(banks);
  }
}

Value DatapathEmitter::atReadData(const uarch::MemUnit &m, Value v,
                                  const StallShell &sh) {
  return c.shiftChain(v, m.readLatency, sh).last();
}

// Emit region \p rb's reads, one arm per `PortPlan`. Read latency is the
// memory's device-resolved `readLatency`, the number the scheduler timed the
// access at, so the datum lands on exactly the cycle the consumer's register
// depth was solved against.
void DatapathEmitter::emitReads(const uarch::RegionBlock &rb, Value issue) {
  StallShell sh = shellFor(rb.id);
  // The two plans that serve several accesses from one port, collected here and
  // built below, once the region's whole demand on the port is known.
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  llvm::MapVector<std::tuple<unsigned, unsigned, unsigned>,
                  SmallVector<unsigned>>
      shared;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (acc.isWrite)
      continue;
    switch (acc.plan) {
    case PortPlan::ElementWise: {
      // No address port: a read selects over the cells, and a constant
      // subscript folds the select away. An argument's cells arrive on its
      // element input ports, an internal array's are this module's registers.
      // Either way they are timed at read latency 0.
      SmallVector<Value> elems;
      if (m.external)
        for (const uarch::MemUnit::ElemPort &p : m.elemPorts)
          elems.push_back(pa.getInput(p.in));
      else
        elems = scatterValues(m.id);
      readData[accKey(m.id, r.idx)] =
          readCrossbar(c, elems, scatterIndex(m, acc));
      break;
    }
    case PortPlan::Table: {
      // A constant table read: index the aggregate_constant combinationally,
      // then register to the scheduled read latency so timing matches a RAM.
      Value idx = bankAddress(m, acc).offset;
      readData[accKey(m.id, r.idx)] = atReadData(
          m, c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id], idx)), sh);
      break;
    }
    case PortPlan::Coloured:
      // A compile-time bank reads its own memory: no crossbar, and no read port
      // on the other banks. An unbanked memref is the same case at bank 0. An
      // argument's group is not built here: its datum is the port's, bound by
      // `bindReadPorts`, and its address by `emitExternalReadAddrs`.
      if (!m.external)
        shared[{r.id, *acc.staticBank, acc.port}].push_back(r.idx);
      break;
    case PortPlan::Lane:
      lanes[{r.id, acc.lane}].push_back(r.idx);
      break;
    case PortPlan::Crossbar: {
      // Read every bank at the (bank-independent) offset, then select by the
      // runtime bank, aligned with the read data. Such an access reaches every
      // bank, so it holds a port of its own on each. A boundary address is held
      // against back-pressure before it widens; an internal port freezes
      // through its read enable.
      auto bs = bankAddress(m, acc);
      SmallVector<Value> vals;
      if (m.external) {
        Value addr = boundaryAddr(c, c.stallHold(bs.offset, sh));
        for (const auto &[bank, base] : extPorts(m, acc)) {
          pa.setOutput(portAddr(base), addr);
          vals.push_back(pa.getInput(portData(base)));
        }
      } else {
        Value addr = bs.offset;
        for (unsigned k = 0; k < m.numBanks; ++k)
          vals.push_back(c.R(atPort(
              seq::ReadPortOp::create(
                  c.b, c.loc, memReadCell(m, k, acc.port), ValueRange{addr},
                  /*rdEn=*/sh ? sh.chainEnable : Value(), m.readLatency),
              acc.port)));
      }
      readData[accKey(m.id, r.idx)] =
          readCrossbar(c, vals, atReadData(m, bs.bank, sh));
      break;
    }
    }
  }
  for (auto &[key, idxs] : lanes)
    emitLaneReads(dp.mems[key.first], key.second, idxs, sh);
  // Reads coloured onto one port of one bank: `bindMemoryPorts` proved they
  // never issue in the same cycle, so one bus carries them all under a select
  // on their own activation.
  for (auto &[key, idxs] : shared) {
    auto [id, bank, port] = key;
    const uarch::MemUnit &m = dp.mems[id];
    Value rd = sharedReadPort(m, bank, port);
    for (unsigned i : idxs)
      readData[accKey(m.id, i)] = rd;
    // This region's own accesses select between themselves here, where their
    // addresses and their shell are. The bus itself is driven by
    // `finalizeSharedReadPorts`, once every region holding the port has
    // contributed its arm.
    Value fired;
    Value addr =
        sharedAddress(m, idxs, issue, sh,
                      portHasSeveralHolders(m, bank, port) ? &fired : nullptr);
    SharedReadPort &p = sharedReads[key];
    p.arms.push_back({fired, addr, Value()});
    p.ownerRegion = rb.id;
  }
  // A forwarded load's consumers read the shadow mux rather than the RAM
  // datum: promise it now, resolved by `finalizeForwards` once the paired
  // stores (emitted after the reads) have recorded their issue terms.
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (acc.isWrite ||
        llvm::none_of(m.forwards, [&](const uarch::MemUnit::Forward &f) {
          return f.load == r.idx;
        }))
      continue;
    uint64_t key = accKey(m.id, r.idx);
    Value raw = readData.lookup(key);
    assert(raw && "a forwarded load's RAM datum exists before the shadow");
    BankSplit bs = bankAddress(m, acc);
    Backedge out = c.bb.get(raw.getType());
    pendingForwards.push_back({m.id, r.idx, raw, bs.bank, bs.offset, out});
    readData[key] = out;
  }
}

void DatapathEmitter::emitLaneReads(const uarch::MemUnit &m, unsigned port,
                                    ArrayRef<unsigned> idxs,
                                    const StallShell &sh) {
  SmallVector<Value> banks, offs;
  for (unsigned i : idxs) {
    BankSplit bs = bankAddress(m, m.accesses[i]);
    banks.push_back(bs.bank);
    offs.push_back(bs.offset);
  }
  // Untagged: a lane is assigned by the skew rather than by the port graph,
  // so it proves nothing about what else touches this bank.
  SmallVector<Value> vals;
  for (unsigned k = 0; k < m.numBanks; ++k)
    vals.push_back(c.R(seq::ReadPortOp::create(
        c.b, c.loc, memReadCell(m, k, port),
        ValueRange{laneSelect(c, banks, offs, k)},
        /*rdEn=*/sh ? sh.chainEnable : Value(), m.readLatency)));
  // Each access picks its own bank's datum back out, delayed with it.
  for (auto [i, bank] : llvm::zip(idxs, banks))
    readData[accKey(m.id, i)] = readCrossbar(c, vals, atReadData(m, bank, sh));
}

bool DatapathEmitter::portHasSeveralHolders(const uarch::MemUnit &m,
                                            unsigned bank,
                                            unsigned port) const {
  // A region is one holder however many of its accesses reach the port, since
  // they have already selected between themselves; a call is another.
  llvm::SmallDenseSet<uint64_t> holders;
  for (const uarch::MemUnit::Access &acc : m.accesses)
    if (!acc.isWrite && acc.staticBank.value_or(0) == bank && acc.port == port)
      holders.insert(uint64_t(acc.region) << 1);
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs)
      if (!ma.isWrite && ma.mem == m.id && ma.bank == bank && ma.port == port)
        holders.insert((uint64_t(cu.id) << 1) | 1);
  return holders.size() > 1;
}

Value DatapathEmitter::sharedReadPort(const uarch::MemUnit &m, unsigned bank,
                                      unsigned port) {
  SharedReadPort &p = sharedReads[{m.id, bank, port}];
  if (!p.data) {
    p.addr = c.bb.get(c.b.getIntegerType(memAddrWidth(m)));
    // The read enable is a backedge: whether one owner's shell may freeze the
    // port is known only once every holder has contributed.
    p.rdEnBE = c.bb.get(c.i1);
    p.data = c.R(
        atPort(seq::ReadPortOp::create(c.b, c.loc, memReadCell(m, bank, port),
                                       ValueRange{Value(p.addr)},
                                       Value(p.rdEnBE), m.readLatency),
               port));
  }
  return p.data;
}

DatapathEmitter::SinkArm DatapathEmitter::commitSink(ArrayRef<SinkArm> arms,
                                                     Idle idle) {
  assert(!arms.empty() && "a shared port was built for no driver");
  // One unconditional arm is the port: nothing to select between, and nothing
  // for an idle cycle to take it away from.
  if (arms.size() == 1 && !arms.front().fired)
    return arms.front();
  SinkArm out;
  for (const SinkArm &a : arms) {
    assert(a.fired && "an arm sharing a sink has to say when it is presenting");
    out.fired = out.fired ? c.orBits(out.fired, a.fired) : a.fired;
  }
  auto reduce = [&](llvm::function_ref<Value(const SinkArm &)> term) -> Value {
    if (!term(arms.front()))
      return {}; // a term this sink does not carry
    // A held sink has one more arm than the drivers: the idle register.
    c.muxLedger.add(MuxRole::Commit, arms.size() + (idle == Idle::Hold ? 1 : 0),
                    datapathWidth(term(arms.front()).getType()));
    // The arms are exclusive by construction (the binding proved two drivers
    // never enabled together), so the reduction is a log-depth AND-OR rather
    // than an arms-1 priority chain. With nothing fired it reads 0, a
    // don't-care behind `out.fired`.
    SmallVector<Value> vals, sels;
    for (const SinkArm &a : arms) {
      vals.push_back(term(a));
      sels.push_back(a.fired);
    }
    Value hot = c.oneHotSelect(vals, sels);
    if (idle == Idle::DontCare)
      return hot;
    // Between drives the bus holds its last value: a read frozen by
    // back-pressure re-presents its address, and an idle region must not put a
    // stale one back on a bus another region has taken.
    Type ty = term(arms.front()).getType();
    Backedge next = c.bb.get(ty);
    Value held = c.reg(next, c.konst(ty, 0));
    Value res = c.mux(out.fired, hot, held);
    next.setValue(res);
    return res;
  };
  out.addr = reduce([](const SinkArm &a) { return a.addr; });
  out.data = reduce([](const SinkArm &a) { return a.data; });
  return out;
}

void DatapathEmitter::finalizeSharedReadPorts() {
  auto address = [&](ArrayRef<SinkArm> arms) {
    assert(!arms.empty() && "a read port was built for no access");
    assert((arms.size() > 1 || !arms.front().fired) &&
           "a port the binding gave to two regions got one arm, so a region "
           "holding it never emitted its accesses");
    return commitSink(arms, Idle::Hold).addr;
  };
  for (auto &[key, p] : sharedReads) {
    // The port freezes with its owner where that is unambiguous: a lone
    // region's chain enable keeps the in-flight datum in the port's own
    // register. Several holders read every cycle off the held bus instead, a
    // constant-true enable the hlmem lowering folds away.
    StallShell sh = p.arms.size() == 1 && p.ownerRegion
                        ? shellFor(*p.ownerRegion)
                        : StallShell{};
    p.rdEnBE.setValue(sh ? sh.chainEnable : c.t1);
    p.addr.setValue(address(p.arms));
  }
  for (auto &[base, arms] : boundaryReads)
    pa.setOutput(portAddr(base), address(arms));
}

// The address one region's accesses on a read port present: each drives it on
// its own issue cycle, and the select is held with the datapath so a read
// frozen by back-pressure keeps re-presenting its address until its datum is
// taken. A port with one access here is that access's address.
Value DatapathEmitter::sharedAddress(const uarch::MemUnit &m,
                                     ArrayRef<unsigned> idxs, Value issue,
                                     const StallShell &sh, Value *fired) {
  auto addrOf = [&](unsigned i) {
    return bankAddress(m, m.accesses[i]).offset;
  };
  // Selected and held at the bank's own address width, a boundary port widening
  // only after, so neither runs at the 32-bit boundary contract.
  auto out = [&](Value addr) {
    addr = c.stallHold(addr, sh);
    return m.external ? boundaryAddr(c, addr) : addr;
  };
  // Every pulse below says when its access is presenting; only an access alone
  // on a port no one else holds needs none and drives it unconditionally.
  assert((issue || (idxs.size() == 1 && !fired)) &&
         "a region with no issue pulse cannot say when it is driving a port; "
         "`bindMemoryPorts` leaves such a read alone on one");
  if (idxs.size() == 1) {
    if (fired)
      *fired = c.activationPulse(issue, m.accesses[idxs.front()].stage, sh);
    return out(addrOf(idxs.front()));
  }
  SmallVector<Value> addrs, sels;
  for (unsigned i : idxs) {
    addrs.push_back(addrOf(i));
    sels.push_back(c.activationPulse(issue, m.accesses[i].stage, sh));
  }
  c.muxLedger.add(MuxRole::Address, addrs.size(),
                  datapathWidth(addrs.front().getType()));
  // Any of them presenting is this region driving the port, which is what a
  // port held by another region as well selects on.
  if (fired)
    for (Value s : sels)
      *fired = *fired ? c.orBits(*fired, s) : s;
  return out(c.oneHotSelect(addrs, sels));
}

// Drive the read-address port of each single-interface external read in region
// \p rb: the in-bank offset for a statically-banked argument (the boundary
// presents one interface per bank), the flat element index for an unbanked one.
// A data-dependent banked read spans every interface, and emitReads drives all
// of its addresses.
void DatapathEmitter::emitExternalReadAddrs(const uarch::RegionBlock &rb,
                                            Value issue) {
  StallShell sh = shellFor(rb.id);
  // One address per port group, the accesses sharing it selecting on their own
  // activation as they do on an internal port.
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> shared;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    // A scattered argument has no address port to drive, and a data-dependent
    // banked one spans every interface (`emitReads`).
    if (!m.external || acc.isWrite || acc.plan != PortPlan::Coloured)
      continue;
    shared[{r.id, acc.portIdx}].push_back(r.idx);
  }
  for (auto &[key, idxs] : shared) {
    const uarch::MemUnit &m = dp.mems[key.first];
    const uarch::MemUnit::Access &acc = m.accesses[idxs.front()];
    // The group is one module output, so a second holder (another region's
    // accesses, or a child mastered on the colour) cannot drive it here;
    // `finalizeSharedReadPorts` does, once every holder has presented.
    Value fired;
    bool held = portHasSeveralHolders(m, acc.staticBank.value_or(0), acc.port);
    Value addr = sharedAddress(m, idxs, issue, sh, held ? &fired : nullptr);
    boundaryReads[acc.portBase].push_back({fired, addr, Value()});
  }
}

void DatapathEmitter::emitWrites(const uarch::RegionBlock &rb, Value issue,
                                 DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
  // A store's write-enable is the issue pulse delayed to its stage. A leaf
  // while's doomed exit iteration still issues, so its store is additionally
  // gated by the continue-condition.
  Value gatedIssue;
  auto commitPulse = [&]() -> Value {
    if (!rb.conditional)
      return issue;
    if (!gatedIssue) {
      assert(rb.condition &&
             "a conditional (while) region has no continue condition; it is "
             "required to gate in-loop store commits");
      gatedIssue = c.andBits(issue, resolveSource(rb.condition));
    }
    return gatedIssue;
  };
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (!acc.isWrite)
      continue;
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainCycle(m, acc));
    // A forwarded store's issue-time terms, taken before the write-latency
    // delays: the shadow compares and captures at issue.
    if (llvm::any_of(m.forwards, [&](const uarch::MemUnit::Forward &f) {
          return f.store == r.idx;
        })) {
      BankSplit bs = bankAddress(m, acc);
      fwdStores[accKey(m.id, r.idx)] = {
          c.activationPulse(commitPulse(), acc.stage, sh), bs.bank, bs.offset,
          resolveSource(acc.data)};
    }
    if (acc.plan == PortPlan::Lane) {
      lanes[{r.id, acc.lane}].push_back(r.idx);
      continue;
    }
    // A `seq.hlmem` write port realizes exactly one cycle, so an internal
    // memory whose device latency is deeper presents address, data and enable
    // `writeLatency - 1` cycles late; the datum still lands at `stage +
    // writeLatency` (see `storeDrainCycle`). A boundary port takes its terms at
    // its stage.
    unsigned pre = m.external ? 0 : m.writeLatency - 1;
    auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
    Value we =
        c.delayValid(c.activationPulse(commitPulse(), acc.stage, sh), pre, sh);
    Value data = late(resolveSource(acc.data));
    switch (acc.plan) {
    case PortPlan::ElementWise:
      // The cells are shared by every store, so this only records the terms:
      // `finalizeScatteredPorts` drives an argument's element ports, or builds
      // an internal array's registers, once every region and call has
      // contributed.
      scatterWrites[m.id].push_back({we, scatterIndex(m, acc), data});
      break;
    case PortPlan::Table:
      llvm_unreachable("a constant table has no write port; an array "
                       "anything writes is never classified as one");
    case PortPlan::Coloured: {
      // A compile-time bank writes its own memory: no demux, and no write port
      // on the other banks. An unbanked memref is the same case at bank 0. One
      // interface carries every store bound to the port, driven once all of
      // them have emitted, by `finalizeBoundaryWritePorts` or
      // `finalizeSharedWritePorts`.
      auto bs = bankAddress(m, acc);
      if (m.external)
        boundaryWrites[acc.portBase].push_back(
            {we, boundaryAddr(c, bs.offset), data});
      else
        sharedWrites[m.id].push_back(
            {*acc.staticBank, acc.port, {we, late(bs.offset), data}});
      break;
    }
    case PortPlan::Lane:
      llvm_unreachable("a lane's stores are delayed and demuxed together, "
                       "below, so they leave the loop above this");
    case PortPlan::Crossbar: {
      // Drive every bank; the runtime bank gates each write-enable so only the
      // target bank commits (an N-way demux).
      auto bs = bankAddress(m, acc);
      if (m.external) {
        Value addr = boundaryAddr(c, bs.offset);
        for (const auto &[bank, base] : extPorts(m, acc)) {
          pa.setOutput(portAddr(base), addr);
          pa.setOutput(portData(base), data);
          pa.setOutput(portWe(base), writeDemux(c, we, bs.bank, bank));
        }
        break;
      }
      Value addr = late(bs.offset);
      Value bank = late(bs.bank);
      for (unsigned k = 0; k < m.numBanks; ++k)
        for (Value cell : memWriteCells(m, k))
          atPort(seq::WritePortOp::create(c.b, c.loc, cell, ValueRange{addr},
                                          data, writeDemux(c, we, bank, k),
                                          c.b.getI64IntegerAttr(1)),
                 acc.port);
      break;
    }
    }
  }
  for (auto &[key, idxs] : lanes)
    emitLaneWrites(dp.mems[key.first], idxs, commitPulse, sh);
}

void DatapathEmitter::emitLaneWrites(const uarch::MemUnit &m,
                                     ArrayRef<unsigned> idxs,
                                     llvm::function_ref<Value()> commit,
                                     const StallShell &sh) {
  unsigned pre = m.writeLatency - 1;
  auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
  SmallVector<Value> addrs, datas, wes, bankOf;
  for (unsigned i : idxs) {
    const uarch::MemUnit::Access &acc = m.accesses[i];
    BankSplit bs = bankAddress(m, acc);
    Value bank = late(bs.bank);
    bankOf.push_back(bank);
    addrs.push_back(late(bs.offset));
    datas.push_back(late(resolveSource(acc.data)));
    wes.push_back(
        c.delayValid(c.activationPulse(commit(), acc.stage, sh), pre, sh));
  }
  auto wlat = c.b.getI64IntegerAttr(1);
  for (unsigned k = 0; k < m.numBanks; ++k) {
    Value we = writeDemux(c, wes[0], bankOf[0], k);
    for (unsigned i = 1; i < idxs.size(); ++i)
      we = c.orBits(we, writeDemux(c, wes[i], bankOf[i], k));
    // Untagged: a skew assigns its ports by lane rather than by the port
    // graph, so nothing proves this store and a read of the same bank stay
    // out of each other's cycle, and only that proof lets the two share one
    // address.
    for (Value cell : memWriteCells(m, k))
      seq::WritePortOp::create(c.b, c.loc, cell,
                               ValueRange{laneSelect(c, bankOf, addrs, k)},
                               laneSelect(c, bankOf, datas, k), we, wlat);
  }
}

// Drive each boundary write port group from the stores bound to it: a one-hot
// select over them, or a single store's own terms where it has the group to
// itself.
void DatapathEmitter::finalizeBoundaryWritePorts() {
  for (auto &[base, writes] : boundaryWrites) {
    SinkArm out = commitSink(writes, Idle::DontCare);
    pa.setOutput(portAddr(base), out.addr);
    pa.setOutput(portData(base), out.data);
    pa.setOutput(portWe(base), out.fired);
  }
}

// Drive an array's shared write ports from the stores coloured onto each. Two
// stores on one port are provably never enabled in the same cycle, which is
// what lets `commitSink` reduce them as a one-hot select.
void DatapathEmitter::finalizeSharedWritePorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    auto it = sharedWrites.find(m.id);
    if (it == sharedWrites.end())
      continue;
    ArrayRef<SharedWrite> writes = it->second;
    unsigned ports = 0;
    for (const SharedWrite &w : writes)
      ports = std::max(ports, w.port + 1);
    for (unsigned k = 0; k < m.numBanks; ++k)
      for (unsigned p = 0; p < ports; ++p) {
        SmallVector<SinkArm, 2> onPort;
        for (const SharedWrite &w : writes)
          if (w.bank == k && w.port == p)
            onPort.push_back(w.arm);
        if (onPort.empty())
          continue;
        SinkArm out = commitSink(onPort, Idle::DontCare);
        // The same port on every instance of the bank: a copy that missed a
        // write would stop holding the same array.
        for (Value cell : memWriteCells(m, k))
          atPort(seq::WritePortOp::create(c.b, c.loc, cell,
                                          ValueRange{out.addr}, out.data,
                                          out.fired, c.b.getI64IntegerAttr(1)),
                 p);
      }
  }
}

// Settle each scattered memory's elements from every store recorded against it:
// per element the datum is a one-hot select over the stores that reach it (at
// most one is live per cycle, since two stores to one element are ordered by
// the dependence analysis), and the write-enable the OR of their decoded
// pulses. An argument's cells are the caller's and this drives its element
// ports; an internal array's are this module's and this builds them, one
// enabled register per element.
void DatapathEmitter::finalizeScatteredPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.scattered)
      continue;
    ArrayRef<SinkArm> writes;
    if (auto it = scatterWrites.find(m.id); it != scatterWrites.end())
      writes = it->second;
    // Nothing stores to it: an argument's cells arrive from the caller and
    // never leave, and an internal array's elements hold their reset value for
    // the whole run, so each is that constant rather than a register.
    if (writes.empty()) {
      if (!m.external)
        for (Backedge &be : scatterElems[m.id])
          be.setValue(c.konst(memElemType(m, c.b), 0));
      continue;
    }
    // One narrow decode per store, shared by every element, rather than a
    // compare per (store, element) pair at the index's carried width, which
    // dominates the scatter's LUT bill.
    SmallVector<SmallVector<Value>> hot;
    for (const SinkArm &w : writes) {
      hot.push_back(oneHotDecode(c, w.addr, m.depthWords));
      c.muxLedger.add(MuxRole::Crossbar, m.depthWords, memAddrWidth(m));
    }
    // Demuxed onto element k first, so the select is the pulse and not the
    // index: two stores in different regions can name element k at once (an
    // idle region's stale address register), and only the enabled one may
    // drive.
    auto driveOf = [&](unsigned k) {
      SmallVector<SinkArm, 1> at;
      for (auto [s, w] : llvm::enumerate(writes))
        at.push_back({c.andBits(w.fired, hot[s][k]), Value(), w.data});
      return commitSink(at, Idle::DontCare);
    };
    if (m.external) {
      for (auto [k, p] : llvm::enumerate(m.elemPorts)) {
        SinkArm out = driveOf(k);
        pa.setOutput(p.out, out.data);
        pa.setOutput(p.we, out.fired);
      }
      continue;
    }
    IntegerType elemTy = memElemType(m, c.b);
    for (auto [k, be] : llvm::enumerate(scatterElems[m.id])) {
      SinkArm out = driveOf(k);
      Value zero = c.konst(elemTy, 0);
      Value cell = c.enabledReg(out.data, out.fired, zero, RegRole::Storage);
      nameValue(cell, memElemName(dp, m, k));
      be.setValue(cell);
    }
  }
}

// Resolve every pending forward: per paired store, a same-element compare at
// the store's issue cycle (against the load's address delayed by the pair's
// window offset), the select and the store's datum registered out to the
// read's data cycle on the load's shell, muxed over the RAM datum. Arms stack
// oldest first: within one offset at most one select fires per cycle (a WAW
// pair holding one address is kept a cycle apart, and same-cycle stores are
// element-disjoint), and across offsets the younger (larger-offset) write is
// muxed outermost and wins.
void DatapathEmitter::finalizeForwards() {
  for (PendingForward &p : pendingForwards) {
    const uarch::MemUnit &m = dp.mems[p.mem];
    const uarch::MemUnit::Access &load = m.accesses[p.load];
    StallShell sh = shellFor(load.region);
    // A bank cone is absent where the digit folded to a constant; the compare
    // then reads the assigned bank (0 when unbanked).
    auto bankOf = [&](const uarch::MemUnit::Access &acc, Value cone) {
      if (cone)
        return cone;
      unsigned w = std::max(1u, llvm::Log2_64_Ceil(m.numBanks));
      return c.konst(c.b.getIntegerType(w), acc.staticBank.value_or(0));
    };
    // The load's bank digit is one cone for every paired store.
    Value loadBank = m.numBanks > 1 ? bankOf(load, p.bank) : Value();
    // The load's issue-time address, delayed to each armed offset's store
    // cycle; one chain per offset however many stores share it.
    llvm::SmallDenseMap<unsigned, std::pair<Value, Value>> addrAt;
    auto loadAddrAt = [&](unsigned off) {
      auto [it, fresh] = addrAt.try_emplace(off);
      if (fresh) {
        it->second.first =
            off ? c.shiftChain(p.offset, off, sh).last() : p.offset;
        it->second.second =
            loadBank && off ? c.shiftChain(loadBank, off, sh).last() : loadBank;
      }
      return it->second;
    };
    SmallVector<const uarch::MemUnit::Forward *> arms;
    for (const uarch::MemUnit::Forward &f : m.forwards)
      if (f.load == p.load)
        arms.push_back(&f);
    llvm::stable_sort(arms, [](const auto *a, const auto *b) {
      return a->offset < b->offset;
    });
    Value muxed = p.raw;
    for (const uarch::MemUnit::Forward *f : arms) {
      ForwardStore st = fwdStores.lookup(accKey(p.mem, f->store));
      assert(st.we && "a forwarded store recorded no issue terms");
      assert(f->offset <= m.readLatency && "an arm lies in the read's flight");
      const uarch::MemUnit::Access &store = m.accesses[f->store];
      auto [lOff, lBank] = loadAddrAt(f->offset);
      Value match = c.icmpEqV(lOff, st.offset);
      if (m.numBanks > 1)
        match = c.andBits(match, c.icmpEqV(lBank, bankOf(store, st.bank)));
      unsigned toData = m.readLatency - f->offset;
      Value sel = c.delayValid(c.andBits(match, st.we), toData, sh);
      Value data = c.shiftChain(st.data, toData, sh).last();
      c.muxLedger.add(MuxRole::Crossbar, 2, m.width);
      muxed = c.mux(sel, data, muxed);
    }
    p.out.setValue(muxed);
  }
}

// Master each buffer from child \p cu's addr/data/we outputs (\p outs): a
// boundary argument passes straight through to the top port, an internal one
// reaches its storage the way the parent's own accesses do. One arm per
// `PortPlan`.
void DatapathEmitter::masterCallPorts(
    const uarch::CallUnit &cu, llvm::StringMap<Value> &outs,
    llvm::StringMap<circt::Backedge> &rdBackedge,
    llvm::function_ref<Value()> runWindow, const StallShell &sh) {
  for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
    if (ma.isBoundary) {
      // The child's drive is one arm of its colour's boundary group: a holder
      // it provably never issues with (another child, or a region's own
      // accesses) shares the bus, selected on the run window. Concurrent
      // masters carry distinct colours and keep distinct groups.
      if (ma.isWrite) {
        boundaryWrites[ma.topBase].push_back(
            {outs[ma.we], outs[ma.addr], outs[ma.data]});
      } else {
        Value fired;
        if (portHasSeveralHolders(dp.mems[ma.mem], ma.bank, ma.port))
          fired = runWindow();
        boundaryReads[ma.topBase].push_back({fired, outs[ma.addr], Value()});
      }
      continue;
    }
    const uarch::MemUnit &m = dp.mems[ma.mem];
    switch (ma.plan) {
    case PortPlan::ElementWise: {
      // A scattered array holds no addressable port, so the child's addressed
      // one is served off the element registers: a select for its read, a term
      // per store for its write. The child keeps the ordinary port ABI.
      assert(ma.bank == 0 && "a scattered array is one bank, so a child "
                             "masters it in whole-array element space");
      Value idx = addrAt(c.b, c.loc, outs[ma.addr], kDatapathAddressWidth);
      if (ma.isWrite)
        scatterWrites[m.id].push_back({outs[ma.we], idx, outs[ma.data]});
      else
        rdBackedge[ma.data].setValue(readCrossbar(c, scatterValues(m.id), idx));
      break;
    }
    case PortPlan::Table: {
      // A constant table the child only reads: one `hw.array_get` registered
      // to the latency the child was timed against, so the datum lands where
      // a RAM's would.
      Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id],
                                              memAddr(m, outs[ma.addr])));
      rdBackedge[ma.data].setValue(atReadData(m, elem, sh));
      break;
    }
    case PortPlan::Coloured: {
      // One hlmem per bank: the child masters bank `ma.bank`, already indexed
      // in that bank's own space via `allo.part`, so this routes straight to it
      // with no crossbar.
      assert(ma.bank < m.numBanks &&
             "child bank index exceeds the buffer's bank count; "
             "validateDatapath must have rejected the partition mismatch");
      Value addr = memAddr(m, outs[ma.addr]);
      // The child was compiled against this buffer's device latency, read here
      // from the MemUnit since the parent never accesses the buffer itself. A
      // deeper write pipelines into the fixed 1-cycle port, as in emitWrites.
      if (ma.isWrite) {
        unsigned pre = m.writeLatency - 1;
        Value a = c.shiftChain(addr, pre, sh).last();
        Value d = c.shiftChain(outs[ma.data], pre, sh).last();
        Value w = c.delayValid(outs[ma.we], pre, sh);
        // The binding settles a call's write port too, so two ports of one
        // child that declared them independent land in separate `always`
        // blocks and the array still infers a true dual port.
        sharedWrites[m.id].push_back({ma.bank, ma.port, {w, a, d}});
        break;
      }
      // The port may also be held by a sibling call or by the parent's own
      // accesses, so the datum comes off the one `seq.read` they share and the
      // address joins its arms. A child paces itself and brings no read enable,
      // so as an owner it keeps the port unfrozen.
      rdBackedge[ma.data].setValue(sharedReadPort(m, ma.bank, ma.port));
      Value fired;
      if (portHasSeveralHolders(m, ma.bank, ma.port))
        fired = runWindow();
      SharedReadPort &p = sharedReads[{m.id, ma.bank, ma.port}];
      p.arms.push_back({fired, addr, Value()});
      break;
    }

    case PortPlan::Lane:
      llvm_unreachable("a child masters a port on a skewed array; a lane is "
                       "assigned from this module's own accesses and the "
                       "child holds none. `checkEmitterSubset` refuses it");
    case PortPlan::Crossbar:
      llvm_unreachable("a child masters one bank, indexed in that bank's own "
                       "space, so it never crossbars");
    }
  }
}

} // namespace mlir::allo::uarch

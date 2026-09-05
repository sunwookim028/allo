/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The memory subsystem's model half: one `MemUnit` per array, what each access
// reaches it by (`PortPlan`), which port of which bank it drives, what holds it
// (`MemUnit::Realization`), and the boundary port groups an argument publishes.
// The hardware for those decisions is built in MemoryEmitter.cpp.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h"
#include "allo/Microarch/Naming.h"
#include "allo/Scheduling/MemoryModel.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Support/AliasAnalysis.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

MemId DatapathBuilder::memIdOf(Value memref) {
  // Key on the storage root, not the operand as written, so a buffer threaded
  // out of a region is the same memory to its producer and its consumer.
  auto it = memOf.find(resolveRoot(memref));
  assert(it != memOf.end() &&
         "`collectStorageFacts` builds a MemUnit for every array the function "
         "touches, so a lookup here cannot miss");
  return it->second;
}

// Build the MemUnit for \p memref, or hand back the one it already has.
static MemId createMem(Datapath &dp, llvm::DenseMap<Value, MemId> &memOf,
                       const DeviceModel &dev, Value memref) {
  memref = resolveRoot(memref);
  if (auto it = memOf.find(memref); it != memOf.end())
    return it->second;
  MemId id = dp.mems.size();
  MemUnit m;
  m.id = id;
  m.memref = memref;
  m.external = isa<BlockArgument>(memref);
  auto mt = cast<MemRefType>(memref.getType());
  m.width = mt.getElementTypeBitWidth();
  // Banking and ports from the same storage model the scheduler binds against
  // (allo.part / allo.bind.storage), so the ports billed and the ports built
  // cannot disagree.
  MemoryChar mc = allo::characterize(memref, dev.memory);
  // Power-on contents, when the array reads through an initialized global. The
  // resolved row decides whether they become a combinational table or a memory
  // that starts with them.
  if (auto init = allo::globalInitOf(memref))
    m.romInit = *init;
  m.isRom = mc.constantTable;
  m.layout = mc.layout;
  m.numBanks = m.layout.numBanks;
  // The device row says whether the array is held one cell per element. A
  // callee's array argument never is: the storage is the parent's and the child
  // masters an addressed port on it.
  m.scattered = dev.memory.isScatter(mc.storage) && (!m.external || dp.atTop);
  m.storage = mc.storage;
  // Timing and style from the same device row the scheduler timed this memref's
  // accesses against. The emitter builds ports at these latencies.
  const StorageRealization *sr = dev.memory.row(m.storage);
  assert(sr && "`PreVerification` rejects an array whose storage realization "
               "the device does not declare");
  m.readLatency = sr->timing.latency.read;
  m.writeLatency = sr->timing.latency.write;
  m.ramStyle = sr->ramStyle;
  m.ports = mc.ports;
  assert(mt.hasStaticShape() &&
         "datapath memory requires a static shape (a dynamic memref sizes to "
         "depthWords 0)");
  // Per-bank depth: a bank's address space is exactly the elements it holds
  // (`ceil` per partitioned dim, not of the total).
  m.depthWords = static_cast<unsigned>(m.layout.bankWords());
  dp.mems.push_back(std::move(m));
  memOf[memref] = id;
  return id;
}

void DatapathBuilder::collectStorageFacts(ArrayRef<Operation *> regionOps) {
  // Whether anything writes each array, indexed by MemId, which checks the
  // recorded realization holds: a table with a writer would drop every store.
  llvm::SmallVector<bool> written;
  auto touch = [&](Value memref, bool isWrite) {
    MemId id = createMem(dp, memOf, dev, memref);
    written.resize(dp.mems.size(), false);
    written[id] = written[id] || isWrite;
  };
  for (Operation *regionOp : regionOps)
    forEachBodyOp(regionOp, [&](Operation *op) {
      if (Value memref = dcpMemref(op)) {
        touch(memref, isa<dcp::DCPathStoreOp>(op));
        return;
      }
      auto inv = dyn_cast<dcp::DCPathInstanceOp>(op);
      if (!inv)
        return;
      // A child's array operand, and the direction of every port it masters on
      // it. The callee interface is registered before this caller is built.
      auto it = callees.ifaces.find(inv.getCallee());
      assert(it != callees.ifaces.end() &&
             "the callee interface must be registered (emitted bottom-up)");
      for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        bool isWrite = false;
        for (const iface::Memory *p : it->second.portsForArg(int(k)))
          isWrite |= p->write;
        touch(operand, isWrite);
      }
    });

  for ([[maybe_unused]] MemUnit &m : dp.mems) {
    assert(!(m.romInit && m.external) &&
           "an argument array reads through no initialized global");
    assert((!m.isRom || (m.romInit && !written[m.id])) &&
           "an array resolved to the device's `table` row is initialized and "
           "written by nobody, which `isConstantTable` decided before the "
           "storage record was taken");
  }
}

// Plan \p m: which port of its bank each access would drive, and how many ports
// one bank would therefore be built with.
//
// Two accesses share a port only where `portGraph` has no edge between them,
// which proves they never issue in the same cycle, so the port carries a select
// over them rather than an arbiter. Two shapes take a port of their own: an
// access `contendsWithAll` relates to, and every write of an array whose
// splitting is not proven safe.
std::optional<DatapathBuilder::PortAssignment>
DatapathBuilder::planPorts(const MemUnit &m, std::optional<bool> writes,
                           unsigned base) {
  Datapath::PortRelation rel = dp.portGraph(m.id, writes);
  ArrayRef<Datapath::PortVertex> verts = rel.verts;
  unsigned n = rel.size();
  llvm::SmallVector<unsigned> colour(n, 0);
  unsigned used = 0;

  // Two shapes share with nothing. An access with no bank of its own is routed
  // to every bank by a crossbar, so it reaches whatever any other reaches. And
  // a read with no activation pulse cannot be selected between: a container's
  // own reads are live on every cycle its children run, and a guard sequences
  // its arms rather than running a datapath at all.
  auto contendsWithAll = [&](unsigned i) {
    if (verts[i].bank < 0)
      return true;
    if (verts[i].write || verts[i].call >= 0)
      return false;
    RegionBlock::Shape s = dp.regions[m.accesses[verts[i].access].region].shape;
    return s == RegionBlock::Shape::Container || s == RegionBlock::Shape::Guard;
  };
  for (unsigned i = 0; i < n; ++i)
    if (contendsWithAll(i))
      for (unsigned j = 0; j < n; ++j)
        if (j != i)
          rel.link(i, j);
  // Greedy in vertex order, taking the lowest port no neighbour holds.
  for (unsigned i = 0; i < n; ++i) {
    llvm::BitVector taken(n);
    for (unsigned j = 0; j < i; ++j)
      if (rel.adj[i].test(j))
        taken.set(colour[j]);
    colour[i] = taken.find_first_unset();
    used = std::max(used, colour[i] + 1);
  }
  // Greedy first fit bounds the ports, it does not minimize them: two read
  // ports with no edge across them merge into one carrying both selects. Write
  // ports stay as they are, since dropping below two clears
  // `writesIndependent`, which puts every write in one `always` block and
  // infers no RAM at all.
  llvm::SmallVector<llvm::BitVector> members(used, llvm::BitVector(n));
  llvm::SmallVector<llvm::BitVector> nbrs(used, llvm::BitVector(n));
  llvm::BitVector reads(used, true);
  for (unsigned i = 0; i < n; ++i) {
    members[colour[i]].set(i);
    nbrs[colour[i]] |= rel.adj[i];
    if (verts[i].write)
      reads.reset(colour[i]);
  }
  // Merge into the lowest port that will take it. A port merged away is never a
  // target afterwards, so one pass over the ports in order is a fixed point.
  for (unsigned b = 1; b < used; ++b)
    if (reads[b])
      for (unsigned a = 0; a < b; ++a)
        if (reads[a] && members[a].any() && !nbrs[a].anyCommon(members[b])) {
          members[a] |= members[b];
          nbrs[a] |= nbrs[b];
          members[b].reset();
          break;
        }
  used = 0;
  for (llvm::BitVector &group : members)
    if (group.any()) {
      for (unsigned i : group.set_bits())
        colour[i] = used;
      ++used;
    }

  // Whether the writes may go on separate ports, which are separate `always`
  // blocks with nothing between them to resolve a collision. Only a pair proven
  // to address different words may: two accesses of one region, or two write
  // ports of one child that declared them independent.
  bool split = true;
  auto proven = [&](unsigned i, unsigned j) {
    if (verts[i].call < 0 && verts[j].call < 0)
      return m.accesses[verts[i].access].region ==
             m.accesses[verts[j].access].region;
    return verts[i].call >= 0 && verts[i].call == verts[j].call &&
           verts[i].independent;
  };
  for (unsigned i = 0; split && used > 1 && i < n; ++i)
    for (unsigned j = i + 1; j < n; ++j)
      if (verts[i].write && verts[j].write && rel.adj[i].test(j) &&
          !proven(i, j)) {
        split = false;
        break;
      }
  // An unsplittable set of writes stays on one `always` block, which arbitrates
  // the collision it might have. Each still keeps a port of its own so the
  // block holds one assignment per write: two writes to different words in one
  // cycle must both commit, and a select would drop one. That block is per
  // direction, so a both-directions pass cannot express it and declines.
  if (!split) {
    if (!writes)
      return std::nullopt;
    for (unsigned i = 0; i < n; ++i)
      colour[i] = i;
    used = n;
  }
  PortAssignment out;
  out.writes = writes;
  // Only a colouring that included the writes has anything to say about them.
  if (!writes || *writes) {
    llvm::SmallDenseSet<unsigned> writeColours;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].write)
        writeColours.insert(colour[i]);
    out.writesIndependent = split && writeColours.size() > 1;
  }

  // Ports one bank is built with: a bank is its own `seq.hlmem` and only the
  // accesses reaching it take its ports.
  out.counts.colours = used;
  for (unsigned k = 0; k < m.numBanks; ++k) {
    llvm::SmallDenseSet<unsigned> all, rd, wr;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].bank < 0 || verts[i].bank == int(k)) {
        all.insert(colour[i]);
        (verts[i].write ? wr : rd).insert(colour[i]);
      }
    out.counts.total = std::max<unsigned>(out.counts.total, all.size());
    out.counts.reads = std::max<unsigned>(out.counts.reads, rd.size());
    out.counts.writes = std::max<unsigned>(out.counts.writes, wr.size());
  }
  for (unsigned c : colour)
    out.colour.push_back(base + c);
  return out;
}

void DatapathBuilder::commitPorts(MemUnit &m, const PortAssignment &pa) {
  // The vertex order `portGraph` builds: writes before reads, and within each
  // this function's accesses before the ports its children master.
  unsigned v = 0;
  for (bool dir : {true, false}) {
    if (pa.writes && *pa.writes != dir)
      continue;
    for (MemUnit::Access &acc : m.accesses)
      if (acc.isWrite == dir)
        acc.port = pa.colour[v++];
    for (CallUnit &cu : dp.calls)
      for (CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && ma.isWrite == dir)
          ma.port = pa.colour[v++];
  }
  assert(v == pa.colour.size() &&
         "the port binding walks `portGraph`'s vertex order");
  if (!pa.writes || *pa.writes)
    m.writesIndependent = pa.writesIndependent;
}

// Group a skewed memory's accesses into lanes: within a lane the slots are
// distinct, so the accesses reach distinct banks and share one port on each.
// Same-slot accesses always collide, so each takes the next lane. Numbered per
// region and reads apart from writes, the granularity a port is contended at.
void DatapathBuilder::assignLanes() {
  for (MemUnit &m : dp.mems) {
    // A constant table has no ports to share (it is combinational), and an
    // argument's ports are boundary interfaces the manifest already published,
    // one set per access, so `assign-banks` gives it no slot either.
    if (!m.layout.skew() || m.external || m.isRom)
      continue;
    // One access without a slot and the array is back to crossbarring: a lane
    // shares a port on the strength of every user holding a distinct slot.
    if (llvm::any_of(m.accesses,
                     [](const MemUnit::Access &a) { return !a.slot; }))
      continue;
    m.skewed = true;
    llvm::DenseMap<std::tuple<unsigned, unsigned, unsigned>, unsigned> used;
    for (MemUnit::Access &acc : m.accesses) {
      assert(*acc.slot < m.numBanks && "a slot indexes the skew's banks");
      acc.lane = used[{acc.region, acc.isWrite, *acc.slot}]++;
    }
  }
}

void DatapathBuilder::planAccessPorts() {
  // What the storage or the layout decides, which every access of the array
  // then takes; empty where the access's own bank decides it.
  auto uniform = [](const MemUnit &m) -> std::optional<PortPlan> {
    if (m.isRom)
      return PortPlan::Table;
    if (m.scattered)
      return PortPlan::ElementWise;
    if (m.skewed)
      return PortPlan::Lane;
    return std::nullopt;
  };
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses)
      acc.plan = uniform(m).value_or(acc.staticBank ? PortPlan::Coloured
                                                    : PortPlan::Crossbar);
  for (CallUnit &cu : dp.calls)
    for (CallUnit::MemArg &ma : cu.memArgs)
      ma.plan = uniform(dp.mems[ma.mem]).value_or(PortPlan::Coloured);
}

// Copies of its row a bank bound with (\p reads, \p writes, \p total) ports is
// held in, and the reads one copy serves (`per`). Every copy takes every write
// and serves `instReads` reads; past the pool the reads share what the writes
// leave, and a read riding a pooled write bus is served wherever that bus lands
// at no port of its own. Not bounded by the copies budget: that budget is what
// a cycle may issue, and a binding needing more address buses still builds one
// copy per bus.
static std::pair<unsigned, unsigned> instancesFor(const StoragePorts &ports,
                                                  unsigned reads,
                                                  unsigned writes,
                                                  unsigned total) {
  unsigned per = ports.instReads.value_or(0);
  if (ports.instPool) {
    if (total <= *ports.instPool)
      return {1, per};
    unsigned riding = reads + writes - total;
    // Measured on the part: 1024x32 is one tile at one read and two at two.
    per =
        std::min(per, *ports.instPool > writes ? *ports.instPool - writes : 0u);
    reads -= riding;
  }
  if (!per || reads <= per)
    return {1, per};
  return {(reads + per - 1) / per, per};
}

void DatapathBuilder::bindMemoryPorts() {
  for (MemUnit &m : dp.mems) {
    // Neither is addressed, so neither has a port to contend for: a scattered
    // array is one cell per element and a constant table is combinational.
    if (m.scattered || m.isRom)
      continue;
    // A skew binds by lane, which already holds distinct slots.
    if (m.skewed) {
      llvm::SmallDenseSet<unsigned> lanes[2];
      for (MemUnit::Access &acc : m.accesses) {
        acc.port = acc.lane;
        lanes[acc.isWrite].insert(acc.lane);
      }
      m.readPortsBuilt = lanes[0].size();
      m.writePortsBuilt = lanes[1].size();
      m.portsBuilt = m.readPortsBuilt + m.writePortsBuilt;
      continue;
    }
    // A direction at a time, reads numbered past the writes so no port carries
    // both. On a row whose directions are separate structures, merging them
    // buys an address multiplexer and nothing else.
    PortAssignment w = planPorts(m, /*writes=*/true, /*base=*/0).value();
    PortAssignment r =
        planPorts(m, /*writes=*/false, /*base=*/w.counts.colours).value();
    unsigned separateTotal = w.counts.writes + r.counts.reads;
    // Where the row's ports are a pool, each serving either direction, a read
    // may ride a write's port and one address bus carries both. Possible only
    // where the writes were split.
    std::optional<PortAssignment> pooled;
    if (m.ports.instPool && !m.external &&
        (w.counts.writes <= 1 || w.writesIndependent)) {
      // The shared bus carries the write's address on the cycle it commits, so
      // a write that presents its terms early would drive the read's cycle too.
      assert(m.writeLatency == 1 &&
             "a pooled row's write port realizes in one cycle");
      pooled = planPorts(m, /*writes=*/std::nullopt, /*base=*/0);
    }
    if (pooled) {
      if (m.fitsStorage(w.counts.writes, separateTotal)) {
        // Both bindings fit, so the outcomes decide: a shared bus buys an
        // addrWidth multiplexer while a copy is a whole tile, so pooled stands
        // only where it builds strictly fewer copies.
        unsigned sep = instancesFor(m.ports, r.counts.reads, w.counts.writes,
                                    separateTotal)
                           .first;
        unsigned pool =
            instancesFor(m.ports, pooled->counts.reads, pooled->counts.writes,
                         pooled->counts.total)
                .first;
        if (pool >= sep)
          pooled.reset();
      } else if (pooled->counts.total >= separateTotal) {
        // The separate binding does not fit the row, and a pooled one that
        // saves no port fits no better: the multiplexer for nothing.
        pooled.reset();
      }
    }
    if (pooled) {
      commitPorts(m, *pooled);
      m.readPortsBuilt = pooled->counts.reads;
      m.writePortsBuilt = pooled->counts.writes;
      m.portsBuilt = pooled->counts.total;
      continue;
    }
    commitPorts(m, w);
    commitPorts(m, r);
    m.writePortsBuilt = w.counts.writes;
    m.readPortsBuilt = r.counts.reads;
    m.portsBuilt = separateTotal;
  }
  assignReadInstances();
}

void DatapathBuilder::assignReadInstances() {
  for (MemUnit &m : dp.mems) {
    auto [instances, per] = instancesFor(m.ports, m.readPortsBuilt,
                                         m.writePortsBuilt, m.portsBuilt);
    if (instances == 1)
      continue;
    m.instances = instances;
    // Each bank ranks the read ports that reach it and hands them out a whole
    // instance at a time. Per bank, not over the memory: `readPortsBuilt` is
    // the largest any one bank holds, so ranking every colour together would
    // put more reads on an instance than it has. A read on a write's bus goes
    // to the first instance, where the port it rides already exists.
    llvm::SmallDenseSet<unsigned> writePorts;
    llvm::SmallVector<llvm::SmallVector<unsigned>> byBank(m.numBanks);
    auto reaches = [&](std::optional<unsigned> bank, unsigned port) {
      if (bank)
        byBank[*bank].push_back(port);
      else
        for (auto &ports : byBank)
          ports.push_back(port);
    };
    for (const MemUnit::Access &acc : m.accesses)
      if (acc.isWrite)
        writePorts.insert(acc.port);
      else
        reaches(acc.staticBank, acc.port);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs) {
        if (ma.mem != m.id)
          continue;
        if (ma.isWrite)
          writePorts.insert(ma.port);
        else
          reaches(ma.bank, ma.port);
      }
    for (auto [k, ports] : llvm::enumerate(byBank)) {
      llvm::sort(ports);
      ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
      unsigned rank = 0;
      for (unsigned port : ports) {
        unsigned inst = writePorts.contains(port) ? 0 : rank++ / per;
        assert(inst < m.instances && "a read ranked past the instances");
        m.readInstance[MemUnit::instanceKey(k, port)] = inst;
      }
    }
  }
}

void DatapathBuilder::measurePorts() {
  for (MemUnit &m : dp.mems) {
    // Neither is addressed, so neither contends for a port.
    if (!m.scattered && !m.isRom) {
      m.readConcurrency = dp.portConcurrency(m.id, /*writes=*/false);
      m.writeConcurrency = dp.portConcurrency(m.id, /*writes=*/true);
    }
    // A scattered argument publishes its cells rather than an address bus, so
    // its groups are the elements. Every other array publishes one per bound
    // port, plus one per group a child masters on it.
    m.boundaryPorts = m.elemPorts.size();
  }
  for (AccRef r : dp.readPorts)
    ++dp.mems[r.id].boundaryPorts;
  for (AccRef r : dp.writePorts)
    ++dp.mems[r.id].boundaryPorts;
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      dp.mems[ma.mem].boundaryPorts += ma.isBoundary && ma.ownsGroup;
}

void DatapathBuilder::enumerateBoundaryPorts() {
  auto key = [](MemId mem, bool write) {
    return (uint64_t(mem) << 1) | unsigned(write);
  };
  llvm::DenseMap<uint64_t, unsigned> group;
  // The group already opened for a (bank, port) colour, so a child mastered
  // on the colour joins it instead of opening a second interface on the same
  // bus. Coloured plans only: a data-dependent access spans every bank and
  // shares with nobody.
  llvm::DenseMap<std::tuple<uint64_t, unsigned, unsigned>, std::string>
      colourBase;

  for (MemUnit &m : dp.mems) {
    if (!m.external)
      continue;
    std::string owner = memOwnerName(dp, m);
    // A scattered argument's ports are per element, enumerated once for the
    // memory rather than per access, since every access reads them all and
    // selects. Its accesses keep the default portIdx/portBase.
    if (m.scattered) {
      // The directions used decide the names: an argument used one way takes
      // the bare `A_k`, used both ways its two ports need telling apart.
      bool reads = false, writes = false;
      for (const MemUnit::Access &acc : m.accesses)
        (acc.isWrite ? writes : reads) = true;
      for (unsigned k = 0, e = m.depthWords; k < e; ++k) {
        MemUnit::ElemPort p;
        if (reads)
          p.in = elemBase(owner, k, writes ? ElemDir::In : ElemDir::Only);
        if (writes) {
          p.out = elemBase(owner, k, reads ? ElemDir::Out : ElemDir::Only);
          p.we = portWe(p.out);
        }
        m.elemPorts.push_back(std::move(p));
      }
      continue;
    }
    // One boundary port group per bound port: accesses that provably never
    // issue together share a port, and so share the interface the caller backs
    // the array with, driving it through a select on their own activation.
    //
    // Keyed by bank as well as port, since a port index is one per bank and two
    // accesses routed to different banks are different interfaces. One map per
    // direction, since the two number their groups in their own port list.
    llvm::SmallDenseMap<std::pair<unsigned, unsigned>, unsigned> groupOfPort[2];
    for (auto [a, acc] : llvm::enumerate(m.accesses)) {
      auto &ports = acc.isWrite ? dp.writePorts : dp.readPorts;
      auto [it, isNew] = groupOfPort[acc.isWrite].try_emplace(
          {acc.staticBank.value_or(~0u), acc.port}, ports.size());
      if (!isNew) {
        acc.portIdx = it->second;
        acc.portBase = m.accesses[ports[acc.portIdx].idx].portBase;
        continue;
      }
      acc.portIdx = ports.size();
      acc.portBase =
          memBase(owner, acc.isWrite, group[key(m.id, acc.isWrite)]++);
      if (acc.plan == PortPlan::Coloured)
        colourBase[{key(m.id, acc.isWrite), acc.staticBank.value_or(0),
                    acc.port}] = acc.portBase;
      ports.push_back({m.id, unsigned(a)});
    }
  }
  // Open a new boundary group for an argument a child masters: the next index
  // on that (memory, direction) counter.
  auto openGroup = [&](CallUnit::MemArg &ma) {
    return memBase(memOwnerName(dp, dp.mems[ma.mem]), ma.isWrite,
                   group[key(ma.mem, ma.isWrite)]++);
  };
  // One port group per (bank, port) colour: holders of one colour provably
  // never drive it in the same cycle (`bindMemoryPorts`), so they share the
  // interface the caller backs the bus with, as coloured accesses already do.
  // Concurrent masters carry distinct colours and keep distinct groups.
  for (CallUnit &cu : dp.calls)
    for (CallUnit::MemArg &ma : cu.memArgs) {
      if (!ma.isBoundary)
        continue;
      if (ma.plan != PortPlan::Coloured) {
        ma.topBase = openGroup(ma);
        continue;
      }
      auto [it, isNew] =
          colourBase.try_emplace({key(ma.mem, ma.isWrite), ma.bank, ma.port});
      if (isNew)
        it->second = openGroup(ma);
      ma.ownsGroup = isNew;
      ma.topBase = it->second;
    }
}

} // namespace mlir::allo::uarch

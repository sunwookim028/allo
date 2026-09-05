/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h" // kMemoryInitAttr, kIndependentWritesAttr, kMemPortAttr

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LOWERHLMEMPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace {

/// The accesses of one physical port: its read and its write, or every write
/// of the memory where the writes could not be split. They share one `always`
/// block, the shape a RAM port infers from.
struct MemPort {
  SmallVector<seq::WritePortOp> writes;
  seq::ReadPortOp read; // null where the port only writes
  unsigned readIdx = 0; // its index among the memory's reads, for the reg name
};

void lowerMemory(seq::HLMemOp mem) {
  seq::HLMemType memType = mem.getMemType();
  assert(memType.getShape().size() == 1 &&
         "an hlmem is emitted one dimensional");
  Type elemTy = memType.getElementType();
  auto arrayTy = hw::UnpackedArrayType::get(elemTy, memType.getShape()[0]);
  SmallVector<seq::ReadPortOp> reads;
  SmallVector<seq::WritePortOp> writes;
  for (Operation *user : mem.getHandle().getUsers()) {
    if (auto read = dyn_cast<seq::ReadPortOp>(user)) {
      reads.push_back(read);
      continue;
    }
    auto write = cast<seq::WritePortOp>(user);
    assert(write.getLatency() == 1 && "a write port is emitted at latency 1");
    writes.push_back(write);
  }

  OpBuilder b(mem);
  Location loc = mem.getLoc();
  Value clk = mem.getClk();
  StringRef name = mem.getName();
  auto array = sv::RegOp::create(b, loc, arrayTy, mem.getNameAttr());
  // Pin the array to the structure the device priced it as; without this the
  // synthesizer chooses.
  if (auto style = mem->getAttrOfType<StringAttr>(kRamStyleAttr)) {
    std::string expr = ("\"" + style.getValue() + "\"").str();
    sv::setSVAttributes(
        array, {sv::SVAttributeAttr::get(b.getContext(), "ram_style", expr)});
  }

  // BLOCKING assignments: only those does synthesis read back as a block-RAM
  // INIT, and only the emitted text tells the two apart.
  if (auto init = mem->getAttrOfType<ArrayAttr>(kMemoryInitAttr)) {
    assert(init.size() == arrayTy.getNumElements() &&
           "an initializer must cover exactly the declared words");
    Type addrTy = b.getIntegerType(llvm::Log2_64_Ceil(init.size()));
    sv::InitialOp::create(b, loc, [&] {
      for (auto [i, word] : llvm::enumerate(init)) {
        Value idx = hw::ConstantOp::create(b, loc, addrTy, i);
        Value slot = sv::ArrayIndexInOutOp::create(b, loc, array, idx);
        sv::BPAssignOp::create(
            b, loc, slot,
            hw::ConstantOp::create(b, loc, cast<IntegerAttr>(word)));
      }
    });
  }

  // One `always` block per physical port (`allo.mem.port`): the write and the
  // read bound to one port address the array in a single process off one
  // address, the shape a dual-port RAM infers from. An access carrying no tag
  // shares with nothing, a FIFO's two ends being able to push and pop in one
  // cycle.
  SmallVector<MemPort> ports;
  llvm::DenseMap<unsigned, unsigned> byTag;
  auto portOf = [&](Operation *op) -> MemPort & {
    auto tag = op->getAttrOfType<IntegerAttr>(kMemPortAttr);
    if (!tag) {
      ports.emplace_back();
      return ports.back();
    }
    auto [it, isNew] = byTag.try_emplace(unsigned(tag.getInt()), ports.size());
    if (isNew)
      ports.emplace_back();
    return ports[it->second];
  };
  // Writes not proven collision-free share one block whatever ports they hold,
  // its priority order resolving the collision they might have. Nothing infers
  // a RAM from that shape.
  bool splitWrites = writes.size() <= 1 || mem->hasAttr(kIndependentWritesAttr);
  if (splitWrites)
    for (seq::WritePortOp write : writes)
      portOf(write).writes.push_back(write);
  else
    ports.emplace_back().writes.assign(writes.begin(), writes.end());
  SmallVector<seq::ReadPortOp> combReads;
  for (auto [i, read] : llvm::enumerate(reads)) {
    // A latency-0 read is combinational and belongs to no clocked process.
    if (read.getLatency() == 0) {
      combReads.push_back(read);
      continue;
    }
    MemPort &port = portOf(read);
    assert(!port.read && "the accesses bound to one read port are merged into "
                         "one `seq.read` before this, so a port has at most "
                         "one");
    port.read = read;
    port.readIdx = i;
  }

  Value hwClk = seq::FromClockOp::create(b, clk.getLoc(), clk);
  for (MemPort &group : ports) {
    // A port has one address bus: its write owns it on the cycle it commits and
    // its read takes it the rest of the time. What the read returns on a write
    // cycle is unsampled, the two never issuing together.
    Value addr, reg;
    std::string rdName;
    // A constant-true enable is no enable and is dropped.
    Value rdEn = group.read ? group.read.getRdEn() : Value();
    if (auto k = rdEn ? rdEn.getDefiningOp<hw::ConstantOp>() : nullptr)
      if (k.getValue().isOne())
        rdEn = Value();
    if (group.read) {
      rdName = (name + "_rd" + Twine(group.readIdx)).str();
      assert(group.writes.size() <= 1 &&
             "a port carrying a read carries at most one write, or the two "
             "would have no single address to share");
      addr = group.read.getAddresses()[0];
      if (!group.writes.empty())
        addr =
            comb::MuxOp::create(b, loc, group.writes.front().getWrEn(),
                                group.writes.front().getAddresses()[0], addr);
      reg = sv::RegOp::create(b, group.read.getLoc(), elemTy,
                              b.getStringAttr(rdName + "_reg"));
    }
    sv::AlwaysFFOp::create(b, loc, sv::EventControl::AtPosEdge, hwClk, [&] {
      for (seq::WritePortOp write : group.writes) {
        Location wloc = write.getLoc();
        Value wa = addr ? addr : write.getAddresses()[0];
        sv::IfOp::create(b, wloc, write.getWrEn(), [&] {
          Value slot = sv::ArrayIndexInOutOp::create(b, wloc, array, wa);
          sv::PAssignOp::create(b, wloc, slot, write.getInData());
        });
      }
      // The read enable is the port's ENB: low holds the read register.
      if (group.read) {
        Location rloc = group.read.getLoc();
        auto readSlot = [&] {
          Value slot = sv::ArrayIndexInOutOp::create(b, rloc, array, addr);
          sv::PAssignOp::create(b, rloc, reg,
                                sv::ReadInOutOp::create(b, rloc, slot));
        };
        if (rdEn)
          sv::IfOp::create(b, rloc, rdEn, readSlot);
        else
          readSlot();
      }
    });
    if (!group.read)
      continue;
    // The port reads at latency 1; anything deeper is an output pipeline
    // register on the data, which is the register a block RAM or an UltraRAM
    // has. The enable gates every stage.
    Location rloc = group.read.getLoc();
    Value data = sv::ReadInOutOp::create(b, rloc, reg);
    for (unsigned d = 1; d < group.read.getLatency(); ++d) {
      auto name = b.getStringAttr(rdName + "_dly" + Twine(d));
      data = rdEn
                 ? seq::CompRegClockEnabledOp::create(
                       b, rloc, data.getType(), data, clk, rdEn, name,
                       /*reset=*/Value(), /*resetValue=*/Value(),
                       /*initialValue=*/Value(), hw::InnerSymAttr())
                       .getResult()
                 : seq::CompRegOp::create(b, rloc, data, clk, name).getResult();
    }
    group.read.replaceAllUsesWith(data);
    group.read.erase();
  }
  // A combinational read has no register to enable, so `rdEn` is dropped.
  for (seq::ReadPortOp read : combReads) {
    Location rloc = read.getLoc();
    Value slot =
        sv::ArrayIndexInOutOp::create(b, rloc, array, read.getAddresses()[0]);
    read.replaceAllUsesWith(sv::ReadInOutOp::create(b, rloc, slot).getResult());
    read.erase();
  }

  for (seq::WritePortOp write : writes)
    write.erase();
  mem.erase();
}

struct LowerHLMemPass : public allo::impl::LowerHLMemPassBase<LowerHLMemPass> {
  void runOnOperation() override {
    SmallVector<seq::HLMemOp> mems;
    getOperation().walk([&](seq::HLMemOp mem) { mems.push_back(mem); });
    for (seq::HLMemOp mem : mems)
      lowerMemory(mem);
  }
};

} // namespace

/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_PRIMITIVES_H
#define ALLO_MICROARCH_PRIMITIVES_H

#include "allo/Microarch/Datapath.h"
#include "allo/Microarch/Naming.h" // regionTagOf
#include "allo/Microarch/RegLedger.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h" // seq::HLMemOp
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace mlir::allo::iface {
struct ModuleInterface; // the port model both emitters declare their ports from
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Type / width / storage-declaration rules, the comb lowering of a compute
// unit, and the module-boundary ABI. Shared by every emitter.
//===----------------------------------------------------------------------===//

/// The bit vector an MLIR datapath type is carried as: `datapathWidth` bits.
IntegerType datapathType(Type t, OpBuilder &b);
/// The element type of a memory's memref.
IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b);
/// The depth on-chip storage holding \p words elements is DECLARED with. All
/// three realizations (`seq.hlmem`, `hw.array_get`, `seq.fifo`) address their
/// storage with exactly clog2(depth) bits, so a single-element store would need
/// a 0-bit address, a width `hw`/`comb` cannot carry. One spare word makes the
/// address 1 bit; the spare is never read. One rule, since a leaf's accesses
/// and a composed container's child ports must agree on the address width.
unsigned declaredDepth(unsigned words);
/// The bits an address of \p m carries: clog2 of its declared depth. Buses,
/// backedges and the muxes priced against them share this width.
unsigned memAddrWidth(const uarch::MemUnit &m);
/// The element bit patterns of a compile-time array initializer, in NATURAL
/// order (element 0 first), each resized to \p width and padded with zero to
/// exactly \p depth words. A float table carries its values as their IEEE bit
/// patterns, the convention the datapath gives every float (`datapathType`).
llvm::SmallVector<llvm::APInt> initWords(ElementsAttr init, unsigned width,
                                         unsigned depth);
/// Record \p words as the power-on contents of \p mem (`kMemoryInitAttr`).
/// `seq.hlmem` has no initializer, so the contents ride as a discardable
/// attribute until the seq->SV pipeline turns them into an `initial` block,
/// which a synthesis tool reads back as a BRAM INIT.
void recordMemoryInit(circt::seq::HLMemOp mem,
                      llvm::ArrayRef<llvm::APInt> words);
/// The datapath's width for an index value, seen from the address side: an
/// address expression may be carried narrower than this (see `evalAffine`), but
/// its operands arrive at this width and a divider is computed at it. The same
/// number as `kIndexWidth`.
inline constexpr unsigned kDatapathAddressWidth = kIndexWidth;

/// \p v resized to \p width bits: truncated when narrowing, sign- or
/// zero-extended when widening, and returned unchanged at equal width.
///
/// Truncation IS reduction modulo `2^width`, which `+`, `-` and `*` commute
/// with, so narrowing is exact wherever the value itself fits. An index is
/// signed (a loop counter runs under signed compares, and a lower bound may be
/// negative); an address, a bank digit and a scaled counter are not.
Value resize(OpBuilder &b, Location loc, Value v, unsigned width,
             bool isSigned);

/// \p v resized to \p width bits as an address: unsigned, an address, a bank
/// digit and a scaled counter all being non-negative by construction.
Value addrAt(OpBuilder &b, Location loc, Value v, unsigned width);

/// Evaluate an affine index expression to a \p width -bit hw value, emitting
/// comb ops. \p idx holds the resolved value of each map operand (dims then
/// symbols), each `kDatapathAddressWidth` wide.
///
/// `+`, `-` and `*` commute with truncation, so carrying an address at the
/// `clog2(depth)` bits it needs is exact. `floordiv` / `mod` do not, so they
/// are computed at `kDatapathAddressWidth` and their result narrowed.
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims, unsigned width = kDatapathAddressWidth);
/// The comb op realizing a combinational compute unit, reading as many of
/// \p operands as its kind's arity needs. Every `CombOpKind` has a case here,
/// which is what makes the enum the whole native vocabulary.
/// \p id is the unit's identity, which carries the kind and the two
/// op-specific attributes a kind can need (a compare's `predicate`, an
/// `affine.apply`'s `map`). \p resultType is the unit's hw result type: the
/// width-preserving binary ops ignore it, the unary casts resize to it.
Value emitCompute(OpBuilder &b, Location loc, const allo::OperatorIdentity &id,
                  ValueRange operands, Type resultType);

/// Declare a module's boundary ports from its port model, in the canonical ABI
/// order: clk/rst/start, then scalar + stream-input + read-data *inputs*, done,
/// then stream-output + read-addr + write + result *outputs*. All module inputs
/// stay contiguous at the front, as HWModulePortAccessor requires.
llvm::SmallVector<circt::hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b);

/// Instantiate module \p mod (as instance \p name), wiring its input ports by
/// name from \p ins and returning its output ports by name.
llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        circt::hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins);

//===----------------------------------------------------------------------===//
// Memory-banking crossbar: routing an access to one of a cyclic-partitioned
// array's N banks when the bank is not statically known.
//===----------------------------------------------------------------------===//
struct EmitContext;

/// An element address split into its bank index and in-bank offset, per the
/// memref's `BankLayout` (see `DatapathEmitter::bankAddress`).
struct BankSplit {
  Value bank;   // which of the layout's banks holds the element; null when the
                // access is statically banked and the caller routes it itself
  Value offset; // its linear index inside that bank (over `bankShape`)
};
/// N:1 result mux: a one-hot select of `bankValues[bank]`, bank in [0,N).
/// Values are pre-read from every bank; the caller aligns \p bank with the
/// read latency.
Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank);

/// Decode \p idx into one line per target: line k = (idx == k), compared at the
/// index's live clog2(n) bits so consumers share one narrow compare.
SmallVector<Value> oneHotDecode(EmitContext &c, Value idx, unsigned n);

/// The 1:N write mirror of `readCrossbar`: the write-enable of bank \p k when
/// one address/datum is broadcast to every bank interface and only the
/// addressed bank may commit. \p bank is null for a statically-routed or
/// unbanked write, whose single interface takes \p we verbatim. Caller aligns
/// \p bank and \p we to the same cycle.
Value writeDemux(EmitContext &c, Value we, Value bank, unsigned k);

//===----------------------------------------------------------------------===//
// ShiftChain: the taps of one shift-register chain. The index carries timing:
// `stages[k]` is the input delayed exactly k cycles, `stages[0]` the input.
//===----------------------------------------------------------------------===//
struct ShiftChain {
  llvm::SmallVector<Value> stages;
  /// The input delayed \p k cycles (k-cycle latency).
  Value tap(unsigned k) const {
    assert(k < stages.size() && "shift-chain tap out of range");
    return stages[k];
  }
  /// The deepest tap (delayed `depth()` cycles).
  Value last() const { return stages.back(); }
  /// The chain length in cycles (deepest delay).
  unsigned depth() const { return stages.size() - 1; }
};

//===----------------------------------------------------------------------===//
// StallShell (H): the elasticity derivation's one output object. It stretches a
// rigid region's time base, so a stalled cycle advances nothing and taps align.
//
// Both fields null is a rigid shell, the identity every primitive below reduces
// to. `DatapathEmitter::deriveStallShell` builds it from the region's stream
// handshakes, and a caller takes it from the region that owns the cell
// (`shellFor`), not necessarily the region currently emitting.
//===----------------------------------------------------------------------===//
struct StallShell {
  // F's half: every shift stage, held read address and clock-enabled IP `ce`
  // advances only while high, so the datapath freezes together.
  Value chainEnable; // F consumes; null => rigid
  // G's half: the controller issues only while high and defers the denied pass
  // rather than dropping it. Implies `chainEnable`, strictly stronger wherever
  // the region may drain a deferred pass away, leaving a bubble.
  Value issueEnable; // G consumes; null => rigid (issue ungated)
  // Owning region, stamped by `shellFor`: whose pass discipline the delayed
  // pulses obey. Names the delay cells; unset on a bare rigid shell.
  std::optional<unsigned> region;
  // Whether the owner has at most one pass in flight
  // (`RegionBlock::singlePass`), which lets `delayValid` time a long delay with
  // a counter: a pipelined region's chain taps each hold a distinct in-flight
  // iteration, so a counter would drop every pulse but the first.
  bool singlePass = false;
  /// Whether this region is latency-insensitive at all (has a stall shell).
  explicit operator bool() const { return chainEnable != Value(); }
  /// This shell with the enables dropped: the raw clock as the time base, the
  /// owner's pass discipline kept.
  StallShell rigid() const { return {Value(), Value(), region, singlePass}; }
};

//===----------------------------------------------------------------------===//
// EmitContext: the shared builder substrate. The clock/reset/constants and the
// low-level combinational and sequential helpers both emitters build on.
//===----------------------------------------------------------------------===//
struct EmitContext {
  OpBuilder &b;
  Location loc;
  Type i1, i32;
  circt::BackedgeBuilder &bb;

  Value clk;    // seq.clock form (for compregs / hlmem)
  Value clkRaw; // i1 form (for extern operator instances)
  Value rst;
  Value zero32, one32, f1, t1; // set by initLiterals()

  // Region being emitted, as a naming prefix (`r3`). Naming only: a delay cell
  // is named after the shell that owns it, and this is the fallback for a shell
  // carrying no region.
  std::string regionTag;

  // Pulse-delay depth at which `delayValid` builds a counter instead of
  // extending a chain, from `Datapath::countedDelayCycles`. Unstamped never
  // counts.
  unsigned countedDelayCycles = std::numeric_limits<unsigned>::max();

  // Every register this module's emission builds, by run. `reg` below is the
  // one `seq.compreg` creation, so this counts the emitted design rather than
  // modelling it; `checkRegLedger` holds the two together on every emission.
  RegLedger ledger;
  // Every select cone built around storage after the binding (shared-port
  // selects, commit sinks, crossbars), charged where built. The allocation's
  // own mux cells are priced from the model, not here.
  MuxLedger muxLedger;
  // Set while a chain builder runs: it charges the whole run at once, so its
  // stages must not each charge one of their own.
  bool inChainRun = false;
  // Power-on immutables by constant, shared across every reset-free register of
  // the module.
  llvm::DenseMap<Attribute, Value> initials;
  // Pulse-delay memos: one chain per (source, chain enable) extended to the
  // deepest requested stage and tapped, one counter per exact
  // (source, depth, enable), so consumers share stages instead of each building
  // a private chain the ledger would recount.
  llvm::DenseMap<std::pair<Value, Value>, ShiftChain> pulseChains;
  llvm::DenseMap<std::tuple<Value, unsigned, Value>, Value> countedPulses;

  EmitContext(OpBuilder &b, Location loc, circt::BackedgeBuilder &bb, Type i1,
              Type i32)
      : b(b), loc(loc), i1(i1), i32(i32), bb(bb) {}

  Value R(Operation *op) { return op->getResult(0); }
  /// Combinational (0-cycle) constant.
  Value konst(Type t, int64_t v);

  /// The power-on value a reset-free register carries, as the shared
  /// `seq.initial` immutable for \p rstVal's constant. One per distinct
  /// constant, so a 64-stage chain builds one initial region.
  Value initialFor(Value rstVal);

  /// Registered (1-cycle): out[t+1] = in[t], sampled unconditionally. A control
  /// role resets to `rstVal` (`seq.compreg` with a synchronous reset); a Value
  /// or Pulse role powers on to it and carries no reset, since a reset blocks
  /// shift-register extraction on the fabric.
  ///
  /// The one place a register is built (with `enabledReg`), which makes
  /// `ledger` exact. \p role is charged as a run of one unless a chain builder
  /// is already charging the whole run it belongs to.
  Value reg(Value in, Value rstVal, RegRole role = RegRole::Control);
  /// Clock-enabled register (1-cycle when enabled): out[t+1] = ce[t] ? in[t] :
  /// out[t], built as `seq.compreg.ce` so identical runs stay CSE-able. Reset
  /// vs power-on follows the same role split as `reg`. Edge-triggered, not a
  /// level-sensitive latch.
  Value enabledReg(Value in, Value ce, Value rstVal,
                   RegRole role = RegRole::Control);
  /// A register on \p sh's time base: clock-enabled by `chainEnable` under an
  /// elastic shell, an unconditional `reg` under a rigid one.
  Value shellReg(Value in, Value rstVal, const StallShell &sh, RegRole role);
  /// Stall-hold: transparent (combinational passthrough) while \p sh's
  /// `chainEnable` is high, holds its last enabled value while low. out = ce ?
  /// in : held; held[t+1] = out[t]. Unlike `enabledReg` it adds no latency when
  /// enabled, so a read address stays == the counter in steady state but
  /// freezes on back-pressure, keeping the in-flight read alive. A no-op
  /// (returns `in`) under a rigid shell.
  Value stallHold(Value in, const StallShell &sh);
  /// A while iter-arg's frozen result register: out[t+1] = load ? init :
  /// (advance ? next : out[t]). Frozen once the loop exits, so it holds the
  /// loop's final carried value, or `init` for a zero-iteration loop.
  /// \p dWire, when given, receives the register's D input: the value held at
  /// the end of the current cycle. A consumer sampling in the latch cycle
  /// itself reads it in place of the not-yet-settled output.
  Value latchReg(Value init, Value next, Value load, Value advance,
                 RegRole role = RegRole::Survivor, Value *dWire = nullptr);
  /// Combinational (0-cycle) 2:1 mux: out = sel ? t : f.
  Value mux(Value sel, Value t, Value f);
  /// Combinational (0-cycle) k:1 select over mutually exclusive selects:
  /// `OR over i of (values[i] & replicate(selects[i]))`, `ceil(log2 k)` levels
  /// deep (`muxLevels`). With no select high the result is zero, which every
  /// caller must treat as a don't-care.
  Value oneHotSelect(ArrayRef<Value> values, ArrayRef<Value> selects);
  /// Shift register on \p sh's time base: each tap advances every clock under a
  /// rigid shell, and only while `chainEnable` is high under an elastic one, so
  /// the taps freeze together and the "index == cycles delayed" contract holds
  /// under stall too. Returns the taps: `stages[k]` = `in` delayed k cycles,
  /// each stage powering on to 0 (or reset to 0 for a control-state role),
  /// `stages[0]` = `in` itself.
  ///
  /// Charges the ledger one run per maximal inter-tap segment of \p taps (the
  /// consumed depths, sorted, deepest == \p depth), since extraction breaks at
  /// every tap; an empty \p taps charges one run of `depth`. \p role comes from
  /// the caller, which is what knows whether the run carries a datum or a
  /// pulse.
  ShiftChain shiftChain(Value in, unsigned depth, const StallShell &sh,
                        RegRole role = RegRole::Value,
                        ArrayRef<unsigned> taps = {});
  /// `shiftChain`'s tap table folded to an initiation interval: with a fresh
  /// datum landing on \p in only every \p ii cycles, `ceil(depth / ii)`
  /// registers hold every live value, each capturing once per iteration when
  /// \p phase reaches the landing phase `ready % ii`. `stages[k]` is register
  /// `ceil(k / ii)`, so the taps index by the same cycle count a plain chain
  /// uses.
  ///
  /// \p phase must be a free-running time base (`RegionControl::phase`), never
  /// a pulse: a datum reaches register `j` only on the `j`-th capture after it
  /// lands, so a chain that stops capturing at the last issue strands its last
  /// `ceil(depth / ii) - 1` iterations. Sound only where one iteration issues
  /// every `ii` cycles, i.e. a schedule-paced cyclic leaf. \p taps splits the
  /// ledger charge as in `shiftChain`, mapped onto the registers built.
  ShiftChain foldedChain(Value in, unsigned depth, unsigned ii, Value phase,
                         unsigned ready, const StallShell &sh,
                         ArrayRef<unsigned> taps = {});
  /// A 1-bit signal delayed `n` cycles (issue -> a store's pipeline stage):
  /// tap `n` of the one pulse chain per (signal, time base), extended on demand
  /// so every consumer shares its stages. Powers on to 0, so no spurious valid.
  /// A delay past `countedDelayCycles` under a single-pass owner
  /// (`sh.singlePass`) is built as `delayPulseCounted` instead, memoized per
  /// exact depth since a counter admits no taps.
  Value delayValid(Value sig, unsigned n, const StallShell &sh);
  /// One pulse delayed `n` cycles by a counter: `log2(n)` registers instead of
  /// `n`, at the cost of admitting only one pulse at a time. Sound exactly
  /// where the pulse's owner is single-pass (`sh.singlePass`), and asserts it.
  Value delayPulseCounted(Value pulse, unsigned n, const StallShell &sh);
  /// A scheduled cell's activation pulse: \p pulse delayed to \p stage, the
  /// cycle within its region the cell issues at. The one name for "this cell
  /// fires now", used for a store's write-enable, a shared-unit input's mux
  /// select and an accumulator's init gate alike.
  Value activationPulse(Value pulse, unsigned stage, const StallShell &sh);
  /// Combinational (0-cycle) equality of `a` against a constant built at
  /// `a`'s own width, so a narrow counter compares narrow.
  Value icmpEq(Value a, int64_t c);
  /// Combinational (0-cycle) equality of two same-width values (a runtime
  /// compare, e.g. a counter against a data-dependent trip bound).
  Value icmpEqV(Value lhs, Value rhs);
  /// Combinational (0-cycle) signed `lhs >= rhs` of two same-width values (the
  /// induction bound test `iv+step >= ub`): signed so a negative compile-time
  /// lower bound (`affine.for %i = -4 to 4`) compares correctly. Identical to
  /// the unsigned test for a non-negative counter.
  Value icmpSgeV(Value lhs, Value rhs);
  /// Combinational (0-cycle) unsigned `lhs >= rhs`, the bound test of a counter
  /// built at an unsigned width, whose top bit is a magnitude bit not a sign.
  Value icmpUgeV(Value lhs, Value rhs);
  /// Combinational (0-cycle) logical NOT of an i1 (`v XOR 1`).
  Value notBit(Value v);
  /// Combinational (0-cycle) AND of two i1s.
  Value andBits(Value lhs, Value rhs);
  /// Combinational (0-cycle) OR of two i1s.
  Value orBits(Value lhs, Value rhs);
  /// A 1-cycle pulse in the same cycle `level` rises 0->1 (out = level &
  /// ~(level delayed one cycle); 0 added latency). The delay reg resets to 0,
  /// so a level held high straight out of reset pulses on cycle 0.
  Value risingEdge(Value level);
  /// The start pulse of a schedulable node: its region-entry `regionStart` when
  /// it has no predecessors, else the rising edge of its predecessors' joined
  /// `done` (the node waits for all predecessors).
  Value startFor(Value regionStart, ArrayRef<Value> predDones);
  /// A completion-latch level: set to 1 by \p setPulse, cleared to 0 by
  /// \p start (so a retriggered region re-edges each pass). out[t+1] = start ?
  /// 0 : (setPulse ? 1 : out[t]).
  Value holdDone(Value setPulse, Value start);
  /// "\p level has risen since \p passStart": 0 from \p passStart (that cycle
  /// included, so the flag re-edges however short the pass) until \p level's
  /// rising edge, 1 from that edge on (0 added latency; it rises the same cycle
  /// \p level does).
  Value completedSince(Value level, Value passStart);
  /// `completedSince` with the level's rising-edge pulse already in hand, so a
  /// caller that also consumes the pulse shares its register.
  Value completedSinceEdge(Value edge, Value passStart);
  /// Split a one-cycle \p when pulse by predicate \p cond into {taken,
  /// notTaken} = {when & cond, when & ~cond}: `taken` (re)starts a container's
  /// children, `notTaken` completes the region without issuing them.
  std::pair<Value, Value> branchPulse(Value when, Value cond);
  /// Materialize the shared literals (0/1 as i32, false/true as i1).
  void initLiterals();
};

/// Scoped setter for `EmitContext::regionTag`, restoring the enclosing
/// container's value on exit.
struct RegionTag {
  EmitContext &c;
  std::string saved;
  RegionTag(EmitContext &c, unsigned region) : c(c), saved(c.regionTag) {
    c.regionTag = regionTagOf(region);
  }
  ~RegionTag() { c.regionTag = saved; }
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_PRIMITIVES_H

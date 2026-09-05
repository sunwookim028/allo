/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/LatencyModel.h"
#include "circt/Dialect/Comb/CombOps.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// Whether a schedule-paced region launches through the start-cycle bypass
// (`kPipelinedBoundary`'s zero arm). The scaled counters must bypass exactly
// when the counter does.
static bool bypassesStart(const Terminator &term, const StallShell &sh) {
  return term.kind == Terminator::Kind::Counted && !sh;
}

RegionControl ControlEmitter::emitPipelineControl(const uarch::RegionBlock &rb,
                                                  const Terminator &term,
                                                  Value start,
                                                  const StallShell &sh) const {
  if (rb.kind == uarch::RegionBlock::Kind::Acyclic)
    return emitAcyclic(rb.id, start, sh);
  assert(rb.ii && "a pipelined region reached control emission with no II");
  auto rc = emitPipelined(rb, term, start, sh);
  // The same start-cycle bypass and update as the counter (`emitPipelined`),
  // with `lb` and `step` scaled.
  bool chained = bypassesStart(term, sh);
  Value first = chained ? term.gateStart(c, start) : Value();
  rc.scaledCounters = emitScaledCounters(
      rb, /*bypassStart=*/first, rc.counter,
      [&](Value cur, Value stepped, Value init) {
        return chained ? c.mux(rc.issue, stepped, c.mux(rc.running, cur, init))
                       : c.mux(rc.running, c.mux(rc.issue, stepped, cur), init);
      });
  return rc;
}

// The scaled address counters of region \p rb, one register per stride slot.
// \p update is passed rather than re-derived so that each family's scaled
// counters are written beside the counter they have to track: drifting from
// that counter is the only way these can be wrong.
llvm::SmallVector<Value> ControlEmitter::emitScaledCounters(
    const uarch::RegionBlock &rb, Value bypassStart, Value counter,
    llvm::function_ref<Value(Value, Value, Value)> update) const {
  llvm::SmallVector<Value> scaled;
  // Whether each slot's register wraps THIS advance, which is what a digit
  // above it advances on. A carry slot always precedes its consumer, so the
  // signal exists by the time it is read.
  llvm::SmallVector<Value> wrapped(rb.addrStrides.size());
  for (auto [slot, s] : llvm::enumerate(rb.addrStrides)) {
    // The stride that is the counter reads the counter register: same value,
    // same start-cycle bypass, one register for both. It never carries a digit
    // above it (no wrap), so `wrapped[slot]` stays unset with no consumer.
    if (s.isCounter) {
      assert(counter && "an identity stride in a region with no counter");
      scaled.push_back(counter);
      continue;
    }
    // Each register at the width ITS OWN range needs, not what the counter
    // happens to be: a cyclic-4 bank digit is 3 bits beside a 32-bit iteration
    // counter.
    auto ty = c.b.getIntegerType(s.width);
    Backedge next = c.bb.get(ty);
    Value init = c.konst(ty, s.init);
    Value reg = c.reg(next, init, RegRole::Counter);
    nameValue(reg, regionSignal(rb.id, "addr" + std::to_string(slot)));
    // The same start-cycle bypass the counter takes, for the same reason: a
    // call region's first pass reads its index on `start` itself.
    Value cur = bypassStart ? c.mux(bypassStart, init, reg) : reg;
    Value raw = cur;
    if (s.step)
      raw =
          c.R(comb::AddOp::create(c.b, c.loc, raw, c.konst(ty, s.step), false));
    if (s.hasCarry) {
      assert(wrapped[s.carry] &&
             "a digit's carry slot is not emitted before it");
      raw = c.R(comb::AddOp::create(
          c.b, c.loc, raw,
          c.mux(wrapped[s.carry], c.konst(ty, s.bump), c.konst(ty, 0)), false));
    }
    Value stepped = raw;
    if (s.wrap) {
      // Unsigned throughout: a stride register holds an index, and a digit is a
      // residue, so neither is ever negative. Counting DOWN, the register goes
      // out of range by wrapping around zero, which is exactly `raw > cur`.
      Value wrapKonst = c.konst(ty, s.wrap);
      wrapped[slot] = c.R(comb::ICmpOp::create(
          c.b, c.loc,
          s.down ? comb::ICmpPredicate::ugt : comb::ICmpPredicate::uge, raw,
          s.down ? cur : wrapKonst, false));
      Value fixed =
          s.down ? c.R(comb::AddOp::create(c.b, c.loc, raw, wrapKonst, false))
                 : c.R(comb::SubOp::create(c.b, c.loc, raw, wrapKonst, false));
      stepped = c.mux(wrapped[slot], fixed, raw);
    }
    next.setValue(update(cur, stepped, init));
    scaled.push_back(cur);
  }
  return scaled;
}

// The one pipelined control skeleton, covering three regimes that differ only
// in their `Terminator` and (for II>1) a phase counter:
//   * free-running (II==1, counted): one iteration issued every cycle;
//   * modulo (II>1, counted): one issued every II cycles, gated by a [0,II)
//     phase counter (in-flight drain via the valid chain);
//   * while (II==1, conditional): a non-speculative flushing pipeline,
//     terminated by the condition going false.
// A rigid counted region issues its first iteration in the `start` cycle: the
// counter and phase read their reload values through a start-cycle bypass, so
// the arm cost is zero (`kPipelinedBoundary`). A while must first latch the
// iter-args its condition reads, and an elastic region may not issue on a
// pulse it cannot hold, so both launch a cycle later through `running`. The
// counter advances on issue, feeding the counted bound test and the datapath's
// iteration-0 recurrence-init injection. A conditional terminator
// is non-speculative (II >= t_cond, so no doomed iteration issues and nothing
// squashes) and stall-free (fixed-latency memory, no FIFO).
RegionControl ControlEmitter::emitPipelined(const uarch::RegionBlock &rb,
                                            const Terminator &term, Value start,
                                            const StallShell &sh) const {
  // G's half of H: a rigid region issues unconditionally. The phase counter
  // below takes F's half instead, being a time base rather than a gate.
  Value enable = sh ? sh.issueEnable : c.t1;
  static_assert(kPipelinedBoundary.arm == 0,
                "the start-cycle bypass below is what a zero arm cost means in "
                "hardware; a different declared arm would have to be built "
                "here, not just written down");
  Value first = bypassesStart(term, sh) ? term.gateStart(c, start) : Value();
  auto runNext = c.bb.get(c.i1);
  Value running = c.reg(runNext, c.f1);
  nameValue(running, regionSignal(rb.id, "run"));
  // The ungated per-cycle issue desire: a [0,II) phase counter gates it to once
  // per II; II==1 (and a while) wants to issue every running cycle.
  Value wantIssue = running;
  Value phase;
  if (*rb.ii > 1) {
    // The phase runs [0, ii), so it is built at clog2(ii) bits.
    IntegerType phaseTy = c.b.getIntegerType(llvm::Log2_64_Ceil(*rb.ii));
    auto phaseNext = c.bb.get(phaseTy);
    Value pz = c.konst(phaseTy, 0);
    Value phaseReg = c.reg(phaseNext, pz, RegRole::Counter);
    nameValue(phaseReg, regionSignal(rb.id, "phase"));
    // The effective phase, 0 in the bypassed start cycle whatever the register
    // held between runs; every consumer (the issue gate, the folded chains)
    // reads this one.
    phase = first ? c.mux(first, pz, phaseReg) : phaseReg;
    wantIssue = c.R(
        comb::AndOp::create(c.b, c.loc, running, c.icmpEq(phase, 0), false));
    Value phasep1 =
        c.R(comb::AddOp::create(c.b, c.loc, phase, c.konst(phaseTy, 1), false));
    Value phaseAdv = c.mux(c.icmpEq(phase, *rb.ii - 1), pz, phasep1);
    // The phase is the region's time base rather than an issue gate: it
    // free-runs with the chains folded onto it, so a frozen cycle holds them
    // together and the cadence resumes where it paused. A pass deferred by a
    // starved input stays a whole `ii` late, keeping the modulo reservation
    // intact. Advancing from the effective phase re-times the cadence at the
    // bypassed start; the registered families reload on `start`.
    Value adv = c.mux(sh ? sh.chainEnable : c.t1, phaseAdv, phase);
    phaseNext.setValue(first ? adv : c.mux(term.gateStart(c, start), pz, adv));
  }
  if (first)
    wantIssue = c.orBits(wantIssue, first);
  // Gated issue: a stalled cycle (enable low) issues nothing, so the counter,
  // `running`, and (with the enabled shift chains) the whole datapath hold.
  Value issue = c.andBits(wantIssue, enable);
  nameValue(issue, regionSignal(rb.id, "issue"));
  // The counter IS the source IV, holding `lb` at start and advancing by `step`
  // on each gated issue, so a `lb != 0` / `step != 1` loop needs no body
  // rewriting. The bypassed family reads `lb` combinationally in the start
  // cycle, when a bound just committed by the predecessor is not yet in the
  // register.
  auto iterNext = c.bb.get(term.lb.getType());
  Value ivReg = c.reg(iterNext, term.lb, RegRole::Counter);
  // Label the counter register after the source loop variable; the bypass mux
  // below stays anonymous.
  nameValue(ivReg, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                          : rb.counterName);
  Value iv = first ? c.mux(first, term.lb, ivReg) : ivReg;
  Value ivStep = c.R(comb::AddOp::create(c.b, c.loc, iv, term.step, false));
  iterNext.setValue(first ? c.mux(issue, ivStep, c.mux(running, ivReg, term.lb))
                          : c.mux(running, c.mux(issue, ivStep, iv), term.lb));
  // Terminate on the last issued iteration (the next induction value reaches
  // the bound, or the condition is false), clearing running the next cycle. A
  // bypassed single-iteration run terminates in its own start cycle, so
  // `running` must not be set behind it.
  Value terminate = c.R(
      comb::AndOp::create(c.b, c.loc, issue, term.isLast(c, ivStep), false));
  Value runAfterLast = c.mux(terminate, c.f1, running);
  runNext.setValue(c.mux(term.gateStart(c, start),
                         first ? c.notBit(terminate) : c.t1, runAfterLast));
  return {/*issue=*/issue,         /*counter=*/iv,
          /*wantIssue=*/wantIssue, /*running=*/running,
          /*phase=*/phase,         /*scaledCounters=*/{}};
}

// The one counted done-driven skeleton, covering the two cells whose iterations
// are paced by the body draining rather than by a schedule:
//   * Container: the body is a sequence of child regions;
//   * CallNode: the body is one instantiated sub-kernel.
// Both keep the same four cells: an induction register advancing on `advance`,
// the `isLast` test against the bound, the launch pulse, and a done latch
// cleared on `start`. They differ only in when the FIRST pass launches, spelled
// as the two families' `arm` in `LatencyModel.h`.
IterationControl ControlEmitter::emitCountedIteration(
    const uarch::RegionBlock &rb, const Terminator &term, Value start,
    Value complete, bool chained, bool wantLevel) const {
  assert(term.lb && "a counted iteration controller needs induction bounds");
  // A chained container turns over for free: `complete` is the last child's
  // commit pulse and the next pass launches on it. Its first child samples
  // the counter and iter-args only a cycle later, so the registers settle in
  // the launch cycle itself.
  const BoundaryCost boundary =
      chained ? BoundaryCost{/*arm=*/0, /*reArm=*/0}
              : (rb.shape == uarch::RegionBlock::Shape::CallNode
                     ? kCallNodeBoundary
                     : kContainerBoundary);
  // Launching on `start` itself means reading the counter combinationally
  // there, before its register settles, so the bypass is a consequence of a
  // zero arm cost and moves with it.
  bool launchAtStart = boundary.arm == 0;

  Backedge ivNext = c.bb.get(term.lb.getType());
  Value ivReg = c.reg(ivNext, term.lb, RegRole::Counter);
  nameValue(ivReg, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                          : rb.counterName);
  Value iv = launchAtStart ? c.mux(start, term.lb, ivReg) : ivReg;
  Value ivStep = c.R(comb::AddOp::create(c.b, c.loc, iv, term.step, false));
  // An empty region never advances at all: its body never runs, so `advance`
  // stays low and the counter holds `lb`.
  Value last = term.isLast(c, ivStep);
  Value advance = c.andBits(complete, c.notBit(last));
  ivNext.setValue(c.mux(start, term.lb, c.mux(advance, ivStep, iv)));
  llvm::SmallVector<Value> scaled = emitScaledCounters(
      rb, /*bypassStart=*/launchAtStart ? start : Value(), iv,
      [&](Value cur, Value stepped, Value init) {
        // Exactly `ivNext` above, with `lb` and `step` scaled.
        return c.mux(start, init, c.mux(advance, stepped, cur));
      });

  // `gateStart` masks the start launch of an empty region (a runtime zero trip
  // or a static lb >= ub), which completes through `empty` below instead.
  Value first = term.gateStart(c, start);
  // Each launch path delayed by its own boundary cost, sharing whatever the two
  // have in common so a family paying the same on both keeps one register.
  unsigned shared = std::min(boundary.arm, boundary.reArm);
  Value launch = c.delayValid(
      c.orBits(c.delayValid(first, boundary.arm - shared, StallShell{}),
               c.delayValid(advance, boundary.reArm - shared, StallShell{})),
      shared, StallShell{});
  nameValue(launch, regionSignal(rb.id, "fire"));
  // An empty region completes one cycle after `start`, not on it: `done` is a
  // level cleared by `start`, so a pulse landing there would leave it high with
  // no 0->1 edge for the next node to start on.
  static_assert(kEmptyRegionCycles == 1 + kDoneLatchCycles,
                "an empty region is one registered start pulse feeding the "
                "done latch; a different declared cost must be built here");
  Value empty = c.reg(c.andBits(start, term.isEmpty(c)), c.f1);
  Value donePulse = c.orBits(empty, c.andBits(complete, last));
  Value done;
  if (wantLevel) {
    done = c.holdDone(donePulse, start);
    nameValue(done, regionSignal(rb.id, "done"));
  }
  return {{/*issue=*/launch, /*counter=*/iv, /*wantIssue=*/Value(),
           /*running=*/Value(), /*phase=*/Value(),
           /*scaledCounters=*/std::move(scaled)},
          done,
          donePulse};
}

// The conditional done-driven skeleton: a sequential-wrapper while. Same
// boundary/continue/launch/done shape as the counted one, but the continue test
// is not available AT the boundary. The condition reads the iter-arg survivor
// registers, which only settle the cycle after a body pass drains, and may
// itself take `tCond` cycles (a memory- or IP-dependent condition). So the
// decision is a delayed CHECK pulse rather than a combinational test, forking
// into launch / finish. The zero-iteration case needs no separate empty term:
// the first CHECK already answers it, a cycle after `start`, which is exactly
// the edge hygiene `done` needs.
IterationControl
ControlEmitter::emitCheckedIteration(unsigned region, Value cond,
                                     unsigned tCond, Value start,
                                     Value complete, bool wantLevel) const {
  static_assert(kCheckedBoundary.arm == kCheckedBoundary.reArm,
                "one CHECK register serves both the start and the drain path; "
                "differing costs would need them split as in the counted "
                "controller");
  Value check = c.delayValid(c.orBits(start, complete), kCheckedBoundary.arm,
                             StallShell{});
  nameValue(check, regionSignal(region, "check"));
  // A container derives no stall shell of its own, since its stream-touching
  // work sits in a child leaf under that leaf's shell, so the CHECK window is
  // rigid.
  Value settled = c.delayValid(check, tCond, StallShell{});
  auto [launch, finish] = c.branchPulse(settled, cond);
  nameValue(launch, regionSignal(region, "fire"));
  Value done;
  if (wantLevel) {
    done = c.holdDone(finish, start);
    nameValue(done, regionSignal(region, "done"));
  }
  return {{/*issue=*/launch, /*counter=*/Value(), /*wantIssue=*/Value(),
           /*running=*/Value(), /*phase=*/Value(), /*scaledCounters=*/{}},
          done,
          finish};
}

// Acyclic (straight-line) region: a single pass, armed after its family's `arm`
// cost (`LatencyModel.h`). There is no iteration index of its own.
//
// Under an elastic shell the arming pulse is LATCHED into `pend`, the acyclic
// counterpart of the pipelined regime's `running`: a single one-shot pulse
// cannot be gated, only dropped, so a stage-0 stream access would sample its
// `_data` at the arming cycle whatever `_valid` said, and a stage-0 put would
// drop its token and never complete. `pend` is combinationally ORed with the
// arming pulse rather than replacing it, so an available token still issues at
// the arming cycle. A rigid region has nothing to defer and stays a bare pulse.
RegionControl ControlEmitter::emitAcyclic(unsigned region, Value start,
                                          const StallShell &sh) const {
  Value armed = c.delayValid(start, kAcyclicBoundary.arm, StallShell{});
  if (!sh) {
    // At zero arm cost `armed` is the caller's start wire, which may already
    // be named for its first role (a container's `fire`); keep that name.
    if (!isNamedValue(armed))
      nameValue(armed, regionSignal(region, "issue"));
    return {armed,
            /*counter=*/Value(),
            /*wantIssue=*/Value(),
            /*running=*/Value(),
            /*phase=*/Value(),
            /*scaledCounters=*/{}};
  }
  auto pendNext = c.bb.get(c.i1);
  Value pending = c.reg(pendNext, c.f1);
  nameValue(pending, regionSignal(region, "pend"));
  Value wantIssue = c.orBits(armed, pending);
  Value issue = c.andBits(wantIssue, sh.issueEnable);
  nameValue(issue, regionSignal(region, "issue"));
  // Hold the pass pending until it actually issues; the pass is a single one,
  // so `wantIssue` falls the cycle after and the latch stays down.
  pendNext.setValue(c.andBits(wantIssue, c.notBit(sh.issueEnable)));
  return {issue,
          /*counter=*/Value(),
          /*wantIssue=*/wantIssue,
          /*running=*/Value(),
          /*phase=*/Value(),
          /*scaledCounters=*/{}};
}

// The region's completion signal: one latched level for every regime (cyclic,
// while, acyclic). It rises when the last iteration's deepest output has
// drained, `lastIssue` delayed by `drainStage` cycles, or immediately on
// `emptyDone`. The latch's register cycle is the LAST commit cycle, so a
// sibling starting on this done's edge reads every committed store and
// survivor. Keying on `lastIssue` rather than a store-retire count keeps a
// region that retires several stores in one cycle from completing early. A
// `retrig` region resets its completion state on `start`. A caller that wants
// no level takes the pulse alone and registers it where it needs the edge.
std::pair<Value, Value>
ControlEmitter::emitDone(const uarch::RegionBlock &rb, Value lastIssue,
                         Value emptyDone, Value start, bool retrig,
                         const StallShell &sh, bool wantLevel) const {
  Value fire = c.delayValid(lastIssue, rb.drainStage, sh);
  // The final put is not committed until accepted, so gate the completion pulse
  // on the region's clock-enable: `done` holds through back-pressure on the
  // last token. A no-op under a rigid shell.
  if (sh)
    fire = c.andBits(fire, sh.chainEnable);
  if (emptyDone)
    fire = c.orBits(emptyDone, fire);
  static_assert(kDoneLatchCycles == 1,
                "completion is one latch register below; a different declared "
                "cost would have to be built here, not just written down");
  if (!wantLevel)
    return {Value(), fire};
  auto dNext = c.bb.get(c.i1);
  Value done = c.reg(dNext, c.f1);
  nameValue(done, regionSignal(rb.id, "done"));
  // `retrig` clears the held `done` on `start`, giving a fresh 0->1 edge each
  // pass.
  Value held = retrig ? c.mux(start, c.f1, done) : done;
  dNext.setValue(c.mux(fire, c.t1, held));
  if (!retrig)
    return {done, fire};
  // `fire` wins over that clear and the two can coincide, so mask the start
  // cycle out of the LEVEL: it reads 0 there whether or not the clear landed,
  // which is what lets every pass re-edge.
  return {c.andBits(done, c.notBit(start)), fire};
}

} // namespace mlir::allo::uarch

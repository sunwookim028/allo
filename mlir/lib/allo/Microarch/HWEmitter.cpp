/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/LatencyModel.h"

#include "circt/Dialect/Comb/CombOps.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// HWEmitter: the orchestrator.
//===----------------------------------------------------------------------===//

// The counted induction bounds (lb/ub/step) of region \p rb, each resolved to
// the region's `counterType`: the IV runs `lb, lb+step, ...` and terminates on
// `iv+step >= ub`. Empty for an acyclic region (no counter) or a while, which
// builds its own Terminator::conditional from the resolved condition.
//
// The counter counts up through SIGNED compares, so a negative lb is fine but
// the step must be positive. A runtime step's sign is a contract with the
// caller, since no static check settles it: step <= 0 would hang the loop and
// write out of bounds on the way.
Terminator HWEmitter::terminatorOf(const uarch::RegionBlock &rb) {
  if (!rb.lbSource)
    return {}; // acyclic: no counter, hence no bounds
  assert(dp.constantOf(rb.stepSource).value_or(1) > 0 &&
         "counted-loop counter is up-counting; a statically non-positive step "
         "must have been rejected by the frontend or the op verifier");
  auto ivType = cast<IntegerType>(rb.counterType);
  // A bound resized to the counter's width: identity for a literal bound, which
  // `recordRegionBounds` already tied in at that width, a real resize for a
  // runtime bound, which arrives as an ordinary index. The counter's own
  // signedness picks the extension; a non-negative runtime bound is the same
  // either way.
  bool isUnsigned = rb.counterUnsigned;
  auto at = [&](const uarch::Source &s) {
    return resize(ctx.b, ctx.loc, datapath.resolveSource(s), ivType.getWidth(),
                  /*isSigned=*/!isUnsigned);
  };
  Value lb = at(rb.lbSource), step = at(rb.stepSource);
  if (rb.ubSource)
    return Terminator::counted(lb, at(rb.ubSource), step, isUnsigned);
  // No ubSource (see `RegionBlock::ubSource`): a constant trip K over a runtime
  // lb or step, so `ub = lb + K*step`. A literal step still folds its span.
  int64_t trip = *rb.tripCount;
  std::optional<int64_t> kstep = dp.constantOf(rb.stepSource);
  Value span = kstep ? ctx.konst(ivType, trip * *kstep)
                     : ctx.R(comb::MulOp::create(ctx.b, ctx.loc, step,
                                                 ctx.konst(ivType, trip),
                                                 /*twoState=*/false));
  return Terminator::counted(
      lb, ctx.R(comb::AddOp::create(ctx.b, ctx.loc, lb, span, false)), step,
      isUnsigned);
}

// Check region \p rb's built drain against the span its consumers were placed
// against. A determinate call is priced into the drain at its contract, which
// the emitted child honours, so folding the same term back in makes the bounds
// hold for a call-holding region too. An indeterminate call has no span and
// leaves the drain unchecked.
static void checkDrainAgainstComposedSpan(const uarch::Datapath &dp,
                                          const uarch::RegionBlock &rb) {
#ifndef NDEBUG
  int64_t builtDrain = rb.drainStage;
  bool doneTimed = false;
  for (uarch::CallId cid : rb.callUnits) {
    const uarch::CallUnit &cu = dp.calls[cid];
    if (cu.determinate)
      builtDrain =
          std::max(builtDrain,
                   int64_t(cu.start) + std::max<int64_t>(*cu.latency, 1) - 1);
    else
      doneTimed = true;
  }
  // Draining past the composed span is a fault: a consumer released at that
  // offset samples before this region has committed. `resolveStreamOperands`
  // re-stamps a stream put's stage, so a region it reached may sit
  // `streamShift` cycles out and no further.
  assert((!rb.modelledDrain || doneTimed ||
          builtDrain <= *rb.modelledDrain + int64_t(rb.streamShift)) &&
         "the built datapath drains past the composed span; a consumer placed "
         "against it samples before this region has committed");
  // Draining early is pessimism rather than a fault: the composed latency
  // claims cycles the hardware does not take and every consumer waits them.
  // With the bound above, a region with no stream shift is pinned to equality.
  assert((!rb.modelledDrain || doneTimed || builtDrain >= *rb.modelledDrain) &&
         "the built datapath drains before the composed span, so the composed "
         "latency is longer than the hardware takes");
#endif
}

// Emit one region: control -> datapath -> resolve the F->G condition, capture
// results, done. The leaf regimes (counted / dynamic-trip / while) differ only
// in the Terminator and the survivor mechanism.
Value HWEmitter::emitRegion(const uarch::RegionBlock &rb, Value start,
                            bool retrig, bool levelUnused) {
  RegionTag tag(ctx, rb.id); // naming scope for this region's cells
  // The controller is selected by (shape x termination): one switch over the
  // table in `RegionBlock::Shape`. `Leaf` falls out and is built inline below.
  switch (rb.shape) {
  case uarch::RegionBlock::Shape::Guard:
    // Run-once under the predicate, either termination class.
    return emitGuard(rb, start, levelUnused);
  case uarch::RegionBlock::Shape::Container:
    return rb.conditional ? emitConditionalContainer(rb, start, levelUnused)
                          : emitContainer(rb, start, levelUnused);
  case uarch::RegionBlock::Shape::CallNode:
    // A counted loop whose body is one CallUnit, advancing on the child's real
    // `done` rather than on the per-cycle pipeline cadence.
    assert(!rb.conditional && "CallNode x Conditional is not a producible "
                              "shape; see RegionBlock::Shape");
    return emitLoopCall(rb, start);
  case uarch::RegionBlock::Shape::Leaf:
    break;
  }

  // A while's continue-condition is a datapath value not emitted yet, so it
  // rides a backedge resolved after the datapath; a counted bound resolves
  // here.
  Backedge condBE;
  Terminator term;
  if (rb.conditional) {
    condBE = ctx.bb.get(ctx.i1);
    term = Terminator::conditional(condBE, ctx.zero32, ctx.one32);
  } else {
    term = terminatorOf(rb);
  }

  // H (elasticity): a stream region's enables depend on handshakes not yet
  // emitted, so it registers a promise (two backedges) that G, F and the done
  // drain wire against, RAUWed at the end. A stream-free region is rigid, its
  // shell carrying only the owner stamp `shellFor` gives every shell.
  Backedge chainEnableBE, issueEnableBE;
  StallShell shell = datapath.shellFor(rb.id);
  if (!rb.streamAccesses.empty()) {
    chainEnableBE = ctx.bb.get(ctx.i1);
    issueEnableBE = ctx.bb.get(ctx.i1);
    shell.chainEnable = chainEnableBE;
    shell.issueEnable = issueEnableBE;
    datapath.setShell(rb.id, shell);
  }

  auto rc = control.emitPipelineControl(rb, term, start, shell);
  datapath.setControl(rb.id, rc); // seam G -> F (counter + issue)

  // This also emits a while's condition and its next-value producers.
  auto fb = datapath.emit(rb, rc.issue);
  // H runs on the emitted (F, G) pair, deriving the two promised enables.
  StallShell derived = datapath.deriveStallShell(rb, rc.issue, fb);

  // `setValue` RAUWs and erases the placeholder, so re-point the terminator: a
  // later `term.cond` read (lastIssuePulse's exit test) needs the real value.
  if (rb.conditional) {
    Value cond = datapath.resolveSource(rb.condition);
    condBE.setValue(cond);
    term.cond = cond;
  }

  Value lastIssue = lastIssuePulse(rc, term);
  // The one thing the two leaf terminations disagree about: a counted loop's
  // recurrence is final only on the last iteration, while a while advances on
  // every CONTINUING iteration (the doomed exit iteration must not commit).
  Value captureOn =
      rb.conditional ? ctx.andBits(rc.issue, term.cond) : lastIssue;
  [[maybe_unused]] unsigned resultDrain =
      captureResults(rb, captureOn, start, rc.phase);
  assert(std::max(fb.storeDrain, resultDrain) == rb.drainStage &&
         "the built datapath's terminal cycle is not the one the model "
         "recorded");
  checkDrainAgainstComposedSpan(dp, rb);

  // An empty counted leaf (lb >= ub) issues nothing, so it completes on
  // `start`, delayed one cycle so the pulse doesn't land on `start` itself:
  // `done` is a level and retrigger needs a real 0->1 edge.
  static_assert(kEmptyRegionCycles == 1 + kDoneLatchCycles,
                "an empty region is one registered start pulse feeding the "
                "done latch; a different declared cost must be built here");
  Value emptyDone =
      (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional)
          ? ctx.delayValid(ctx.andBits(start, term.isEmpty(ctx)), 1, shell)
          : Value();
  // A CallUnit region completes on the child's `done`; one that also has loose
  // datapath waits for both, ANDing two held levels so the later wins. Only a
  // call-free region records its completion pulse: composed with a call's done
  // level, the pulse no longer marks completion.
  bool looseWork = !rb.streamAccesses.empty() || !rb.units.empty() ||
                   !rb.memAccesses.empty();
  Value done = fb.callDone;
  if (!fb.callDone || looseWork) {
    // A composed call done is a level with no pulse to substitute for it, so
    // only a call-free region may drop its latch.
    bool wantLevel = fb.callDone || !levelUnused;
    auto [drained, pulse] = control.emitDone(rb, lastIssue, emptyDone, start,
                                             retrig, shell, wantLevel);
    if (fb.callDone) {
      done = ctx.andBits(fb.callDone, drained);
    } else {
      done = drained;
      donePulse[rb.id] = pulse;
    }
  }
  // Resolving the promise RAUWs every consumer and erases the placeholders, so
  // re-register the region with the resolved values; a later region must not
  // read the placeholders.
  if (shell) {
    assert(derived && "a stream region must derive its shell");
    chainEnableBE.setValue(derived.chainEnable);
    issueEnableBE.setValue(derived.issueEnable);
    datapath.setShell(rb.id, derived);
  }
  return done;
}

// The final iteration's issue pulse: a counted region's last iteration (iv+step
// reaches the bound) or a while's exit; an acyclic region has no counter, so
// its single issue pulse is itself the last. Both `emitDone` and the survivor
// captures key off it.
Value HWEmitter::lastIssuePulse(const RegionControl &rc,
                                const Terminator &term) {
  if (!rc.counter)
    return rc.issue; // acyclic: a single pass
  Value ivStep =
      ctx.R(comb::AddOp::create(ctx.b, ctx.loc, rc.counter, term.step, false));
  return ctx.andBits(rc.issue, term.isLast(ctx, ivStep));
}

// Capture each of a result-yielding LEAF region's results into its own survivor
// register on the cycle it lands, while the result is still on its Source: a
// free-running datapath overwrites it once the run ends. \p captureOn is the
// issue pulse the capture keys off; a result produced at a later stage delays
// its capture to match. Returns the latest-landing result's stage, one of the
// terms of `RegionBlock::drainStage`, which `emitRegion` checks against it. A
// store-ful region yields no result and returns 0.
unsigned HWEmitter::captureResults(const uarch::RegionBlock &rb,
                                   Value captureOn, Value start, Value phase) {
  StallShell sh = datapath.shellFor(rb.id);
  unsigned ii = rb.ii.value_or(1);
  unsigned maxStage = 0;
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    if (!r.value)
      continue; // an untracked result: no survivor (asserts if read)
    if (r.value.kind == uarch::Source::Kind::Call)
      continue; // a call result: emitCalls sets the survivor from the child's
                // held output port (self-timed by `done`), not a static capture
    unsigned stage = dp.readyCycle(r.value);
    Value cap = ctx.delayValid(captureOn, stage, sh);
    Value res = datapath.resolveSource(r.value);
    // A rotated reduction's slot k is the datum delayed k iterations. The shift
    // chain holds every live partial sum (a pulse-clocked shift would strand
    // the last ones), so tap `k * ii`, captured on the head pulse so every slot
    // latches its own iteration's value.
    if (r.shiftTap) {
      unsigned depth = r.shiftTap * ii;
      ShiftChain sc = phase && depth > 1
                          ? ctx.foldedChain(res, depth, ii, phase, stage, sh)
                          : ctx.shiftChain(res, depth, sh);
      res = sc.tap(depth);
    }
    // A loop-carried result preloads its init at `start`, so a run that never
    // captures keeps the identity rather than a stale value. An init-less
    // result always lands: it powers on at 0.
    Value survivor =
        r.init ? ctx.latchReg(datapath.resolveSource(r.init), res, start, cap)
               : ctx.enabledReg(res, cap, ctx.konst(res.getType(), 0),
                                RegRole::Survivor);
    nameValue(survivor, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, survivor);
    liveResult[DatapathEmitter::accKey(rb.id, k)] = res;
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// Run `regions` in program order, each region starting when its predecessor
// drains (the first on `start`); returns the last region's drain pulse. A
// `handoffSafe` predecessor hands its successor its completion pulse directly,
// a cycle ahead of the latched done, since its state has already settled.
// Nothing here reads a done level, so no region along the chain builds one.
Value HWEmitter::sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                          bool retrig, bool tailOnPulse) {
  Value drain;
  Value startK = start;
  for (auto [i, rid] : llvm::enumerate(regions)) {
    const auto &rb = dp.regions[rid];
    Value done = emitRegion(rb, startK, retrig, /*levelUnused=*/true);
    bool tail = i + 1 == regions.size();
    if (tail && tailOnPulse) {
      drain = donePulse.lookup(rid);
      assert(drain && "a pulse-advanced tail recorded no completion pulse");
    } else {
      drain = drainPulse(rid, done);
    }
    startK = drain;
  }
  return drain;
}

// A region's completion as an edge: the pulse itself where a successor may
// start in the commit cycle, otherwise that pulse registered, which is the
// cycle the done latch would have risen in. Only a region completing on a
// call's done has no pulse, and it keeps its level for this.
Value HWEmitter::drainPulse(uarch::RegionId rid, Value done) {
  if (Value pulse = handoffPulse(rid))
    return pulse;
  if (Value pulse = donePulse.lookup(rid)) {
    Value edge = ctx.reg(pulse, ctx.f1);
    nameValue(edge, regionSignal(rid, "drain"));
    return edge;
  }
  assert(done && "a region recording no completion pulse must keep its level");
  return ctx.risingEdge(done);
}

// Compose the func-scope siblings by their dependence DAG (rb.predecessors): a
// region with no predecessors starts with the kernel `start` (independent
// siblings run concurrently), the rest on the rising edge of their
// predecessors' joined `done`. The kernel `done` is the conjunction of every
// region's `done`. Emission is in program order, so a predecessor's `done` is
// already built when its consumer reads it.
Value HWEmitter::composeSiblings(llvm::ArrayRef<uarch::RegionId> regions,
                                 Value start) {
  // Nothing to compose: complete a cycle after `start`, the shape an empty
  // counted region's `done` already takes.
  if (regions.empty())
    return ctx.holdDone(ctx.reg(start, ctx.f1), start);

  llvm::DenseMap<uarch::RegionId, Value> doneOf;
  Value allDone;
  for (uarch::RegionId rid : regions) {
    const auto &rb = dp.regions[rid];
    llvm::SmallVector<Value, 2> predDones;
    for (uarch::RegionId p : rb.predecessors) {
      Value d = doneOf.lookup(p);
      assert(d && "a predecessor's done must be emitted before its consumer");
      predDones.push_back(d);
    }
    // The same conditional-predecessor hand-off as `sequence`; a join over
    // several predecessors needs the latched levels.
    Value startK = rb.predecessors.size() == 1
                       ? handoffPulse(rb.predecessors.front())
                       : Value();
    if (!startK)
      startK = ctx.startFor(start, predDones);
    // Every sibling's level is read: by the joins below and by the kernel's own
    // conjunction.
    Value done = emitRegion(rb, startK, /*retrig=*/true, /*levelUnused=*/false);
    // A lone region is its own conjunction and has no consumer to hand a stale
    // level to, so it keeps the raw done.
    Value completed =
        regions.size() > 1 ? ctx.completedSince(done, start) : done;
    doneOf[rid] = completed;
    allDone = allDone ? ctx.andBits(allDone, completed) : completed;
  }
  return allDone;
}

// Set up a container's loop-carried iter-args as frozen survivor registers:
// each latches its `results[k].init` at `start` and advances to a next-value on
// `advance`, and is recorded as Source::Survivor{rb, k}. Returns the per-arg
// next-value backedges the caller sets to `resolveSource(results[k].value)`
// once the children have produced them; the recurrence splits in two halves
// because the register must exist before the children that feed it emit.
SmallVector<circt::Backedge>
HWEmitter::setupCarriedIterArgs(const uarch::RegionBlock &rb, Value start,
                                Value advance, bool publishThrough) {
  SmallVector<circt::Backedge> nextBE;
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    assert(r.init && "a container iter-arg has no resolvable init");
    // An init sourced from a chained enclosing container's survivor: this
    // latch fires in the very cycle that survivor latches, so read its D wire.
    Value init;
    if (r.init.kind == uarch::Source::Kind::Survivor)
      init = throughValue.lookup(
          DatapathEmitter::accKey(r.init.id, r.init.outPort));
    if (!init)
      init = datapath.resolveSource(r.init);
    circt::Backedge nb = ctx.bb.get(init.getType());
    nextBE.push_back(nb);
    Value through;
    Value carried = ctx.latchReg(init, nb, start, advance, RegRole::Survivor,
                                 publishThrough ? &through : nullptr);
    nameValue(carried, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, carried);
    if (publishThrough)
      throughValue[DatapathEmitter::accKey(rb.id, k)] = through;
  }
  return nextBE;
}

// A loop-over-call region (a counted `dcp.pipeline` wrapping one CallUnit): the
// counter is `rc.counter` and the child start is `rc.issue`, so one child
// instance fires N times, each invocation advancing on its real `done`, a held
// level cleared on its start whose rising edge marks each completion.
Value HWEmitter::emitLoopCall(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id);
  // A loop-over-call body is one child instance and nothing else
  // (`validateDatapath`), so the region is rigid: it derives no stall shell.
  assert(rb.streamAccesses.empty() &&
         "a loop-over-call region with stream accesses would need a stall "
         "shell, which this controller does not build");
  // Bounds are at the child's index-port width (this region's `counterType`).
  // The controller is paced by the child's per-invocation completion pulse
  // (`fb.callDone`), a backedge since emitCalls needs the counter first.
  Backedge callDone = ctx.bb.get(ctx.i1);
  IterationControl ic = control.emitCountedIteration(
      rb, terminatorOf(rb), start, callDone, /*chained=*/false,
      /*wantLevel=*/true);

  // An empty loop never fires the child, whose own run gating keeps every
  // write-enable low.
  datapath.setControl(rb.id, ic.rc);
  auto fb = datapath.emit(rb, ic.rc.issue);
  assert(fb.callDone && "a loop-over-call region produced no child done");
  callDone.setValue(fb.callDone);
  return ic.done;
}

bool HWEmitter::advancesOnPulse(const uarch::RegionBlock &rb) const {
  // A container composing an exact span keeps its wiring: consumers are placed
  // against that span, so the hardware may not run ahead of it. A conditional
  // container, a dynamic trip or a non-static child yields only a ceiling.
  bool exactSpan =
      !rb.conditional && rb.tripCount &&
      llvm::all_of(rb.children, [&](uarch::RegionId c) {
        return dp.regions[c].determinacy == DeterminacyEnum::CountedStatic;
      });
  if (exactSpan)
    return false;
  // The last child must hand its results over safely at the pulse: a call-free
  // leaf exposes each result's live wire, a conditional container's iter-args
  // settle at least a cycle before its exit, and a guard's survivor is sampled
  // through its capture D wire.
  const uarch::RegionBlock &last = dp.regions[rb.children.back()];
  bool lastOk = false;
  switch (last.shape) {
  case uarch::RegionBlock::Shape::Leaf:
    lastOk = last.callUnits.empty();
    break;
  case uarch::RegionBlock::Shape::Container:
    lastOk = last.conditional;
    break;
  case uarch::RegionBlock::Shape::Guard:
    lastOk = true;
    break;
  case uarch::RegionBlock::Shape::CallNode:
    break;
  }
  if (!lastOk)
    return false;
  // Every carried next-value must be a settled survivor or a constant; a
  // container-level unit would be sampled in the cycle it still computes.
  return llvm::all_of(rb.results, [](const uarch::RegionResult &r) {
    return r.value.kind == uarch::Source::Kind::Survivor ||
           r.value.kind == uarch::Source::Kind::Const;
  });
}

bool HWEmitter::chainsTurnover(const uarch::RegionBlock &rb) const {
  assert(!rb.conditional &&
         "a conditional container never chains its turnover");
  // A counted container's first child samples the counter and iter-args at its
  // own start, so collapsing launch onto the commit cycle needs a first child
  // that samples a cycle later: a conditional child's CHECK or a guard child's
  // predicate test.
  const uarch::RegionBlock &first = dp.regions[rb.children.front()];
  return first.conditional || first.shape == uarch::RegionBlock::Shape::Guard;
}

Value HWEmitter::nextValueFor(const uarch::RegionBlock &rb, unsigned k,
                              bool onPulse) {
  const uarch::RegionResult &r = rb.results[k];
  if (onPulse && r.value.kind == uarch::Source::Kind::Survivor &&
      r.value.id == rb.children.back()) {
    const uarch::RegionBlock &last = dp.regions[r.value.id];
    // A counted leaf result captured in the commit cycle itself: its survivor
    // register settles only at the turnover's end, so latch from the same wire
    // its own capture takes. Any other capture has settled a cycle earlier.
    if (last.shape == uarch::RegionBlock::Shape::Leaf && !last.conditional &&
        dp.readyCycle(last.results[r.value.outPort].value) == last.drainStage) {
      Value live = liveResult.lookup(
          DatapathEmitter::accKey(r.value.id, r.value.outPort));
      assert(live && "a pulse-advanced container's live result was not "
                     "recorded");
      return live;
    }
    // A guard survivor latches on the pulse itself, so rebuild its latch's D
    // wire: the captured datum in a capture cycle, the register otherwise.
    if (last.shape == uarch::RegionBlock::Shape::Guard) {
      Value en = donePulse.lookup(r.value.id);
      Value datum = guardCapture.lookup(
          DatapathEmitter::accKey(r.value.id, r.value.outPort));
      assert(en && datum &&
             "a pulse-advanced container's guard capture was not recorded");
      return ctx.mux(en, datum, datapath.resolveSource(r.value));
    }
  }
  return datapath.resolveSource(r.value);
}

void HWEmitter::resolveIterationBody(const uarch::RegionBlock &rb, Value issue,
                                     Backedge &lastDrain,
                                     llvm::MutableArrayRef<Backedge> nextBE,
                                     bool onPulse) {
  lastDrain.setValue(
      sequence(rb.children, issue, /*retrig=*/true, /*tailOnPulse=*/onPulse));
  for (auto [k, nb] : llvm::enumerate(nextBE))
    nb.setValue(nextValueFor(rb, k, onPulse));
}

// A container region: a cyclic loop whose body nests one or more child regions,
// run once per outer iteration. The outer counter is materialized first, then
// the children are sequenced within one outer iteration, and the counter
// advances when the LAST child drains. Non-overlapping (II_outer >= sum of
// child latencies), so the outer index is stable across one pass. A value
// handed child-to-child crosses as a survivor register. Returns a latched
// completion level. Where no exact span composes through the container, the
// advance rides the last child's completion pulse (`advancesOnPulse`), and
// with a conditional or guard first child the relaunch collapses onto that
// same cycle (`chainsTurnover`).
Value HWEmitter::emitContainer(const uarch::RegionBlock &rb, Value start,
                               bool levelUnused) {
  RegionTag tag(ctx, rb.id);
  bool onPulse = advancesOnPulse(rb);
  bool chained = onPulse && chainsTurnover(rb);
  // The controller is paced by `lastDrain`: the last child's done edge, or its
  // completion pulse when the advance may ride it, resolved once the children
  // emit. `chained` additionally collapses the relaunch onto that cycle.
  Backedge lastDrain = ctx.bb.get(ctx.i1);
  IterationControl ic =
      control.emitCountedIteration(rb, terminatorOf(rb), start, lastDrain,
                                   chained, /*wantLevel=*/!levelUnused);
  donePulse[rb.id] = ic.donePulse;
  // The counter must be live while the children emit: it is their outer index,
  // and (for a variable-trip child) its own bound.
  datapath.setControl(rb.id, ic.rc);

  // Loop-carried iter-args, advancing on each outer-iteration drain; the final
  // value is this region's survivor. When chained, the first child launches in
  // the latch cycle, so the D wires are published for its init latches.
  SmallVector<Backedge> nextBE =
      setupCarriedIterArgs(rb, start, lastDrain, /*publishThrough=*/chained);

  // The container's own combinational units (a nested guard's predicate over
  // this counter) emit once the counter and iter-arg survivors are live, so a
  // guard child reads its predicate as a Source::Unit when it emits below.
  datapath.declareUnits(rb);
  datapath.emitUnits(rb);

  resolveIterationBody(rb, ic.rc.issue, lastDrain, nextBE, onPulse);
  return ic.done;
}

// A conditional container: a sequential-wrapper while whose body nests child
// regions. Each outer iteration runs the children once (as emitContainer), but
// the loop is data-dependent: the outer iter-args are frozen survivor registers
// advanced by the children's results, and a done-based CHECK/RUN FSM re-checks
// the continue-condition on the settled iter-args after each drain, ending the
// loop when it goes false. No squash or stall: the same non-speculative
// flushing family as a leaf while.
Value HWEmitter::emitConditionalContainer(const uarch::RegionBlock &rb,
                                          Value start, bool levelUnused) {
  RegionTag tag(ctx, rb.id);
  bool onPulse = advancesOnPulse(rb);

  // The outer iter-arg registers are this region's survivors, advanced when an
  // outer iteration drains (`lastDrain`, resolved after the children emit).
  // No D wires to publish: the children launch behind the CHECK window, at
  // least a cycle after these registers latch.
  Backedge lastDrain = ctx.bb.get(ctx.i1);
  SmallVector<Backedge> nextBE =
      setupCarriedIterArgs(rb, start, lastDrain, /*publishThrough=*/false);

  // The condition cone yields the continue-condition and its ready latency
  // t_cond (0 when combinational, several cycles when memory- or IP-dependent).
  // It reads only the frozen iter-args, so it emits before its sampler.
  auto [cond, tCond] = datapath.emitConditionRegion(rb, rb.condition);
  IterationControl ic = control.emitCheckedIteration(
      rb.id, cond, tCond, start, lastDrain, /*wantLevel=*/!levelUnused);
  donePulse[rb.id] = ic.donePulse;

  // The last child's drain advances the iter-args and drives the next CHECK,
  // which samples a cycle later, once the advanced iter-args have settled.
  resolveIterationBody(rb, ic.rc.issue, lastDrain, nextBE, onPulse);
  return ic.done;
}

// A guard region (a dcp.select): the then-arm (`children`) runs iff the
// predicate holds, the else-arm (`elseChildren`) iff it does not. The predicate
// is a held value, valid at `start`. The not-taken arm's children never issue,
// so the predicate reaches every store write-enable structurally, via the
// missing issue pulse, not a per-store gate. An empty arm completes in one
// cycle, its start pulse IS its drain, so the region produces a done edge in
// both branches. Run-once: no iteration or iter-args, since the predicate is
// independent of the children.
Value HWEmitter::emitGuard(const uarch::RegionBlock &rb, Value start,
                           bool levelUnused) {
  RegionTag tag(ctx, rb.id);
  // The predicate as a Source: a scheduled condition region's survivor (a
  // data-dependent scf guard), or the parent container's combinational
  // predicate unit (an affine guard, emitted by the container beforehand).
  Value cond = datapath.resolveSource(rb.condition);
  // CHECK after the guard's arm cost decouples the completion pulse from the
  // start-clear below: a skipped guard's done would otherwise coincide with
  // `start` and be masked.
  Value checkTime = ctx.delayValid(start, kGuardBoundary.arm, StallShell{});
  nameValue(checkTime, regionSignal(rb.id, "check"));
  auto [thenStart, elseStart] = ctx.branchPulse(checkTime, cond);
  // Each arm runs its children once, retrig so a re-entered guard presents
  // fresh edges each enclosing pass. The arm's completion follows the sibling
  // hand-off rule: a `handoffSafe` tail hands its pulse, everything else the
  // done edge, so the capture below reads settled arm results.
  auto armDrained = [&](llvm::ArrayRef<uarch::RegionId> arm,
                        Value armStart) -> Value {
    return arm.empty() ? armStart : sequence(arm, armStart, /*retrig=*/true);
  };
  Value thenDrained = armDrained(rb.children, thenStart);
  Value elseDrained = armDrained(rb.elseChildren, elseStart);
  // Exactly one arm runs a pass, so the region completes on whichever drains.
  // The set pulse is recorded for the hand-off and advance chains, and is the
  // capture enable of every survivor below.
  Value pulse = ctx.orBits(thenDrained, elseDrained);
  donePulse[rb.id] = pulse;
  // Each yielded result is the taken arm's value, latched into one survivor
  // when that arm drains. The two drain pulses are disjoint, so `thenDrained`
  // selects the datum while `pulse` enables the capture.
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    Value tv = datapath.resolveSource(r.value);
    Value ev = datapath.resolveSource(r.elseValue);
    Value datum = ctx.mux(thenDrained, tv, ev);
    Value surv = ctx.enabledReg(datum, pulse, ctx.konst(tv.getType(), 0),
                                RegRole::Survivor);
    nameValue(surv, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, surv);
    // The captured datum lets a pulse-advanced container rebuild this latch's
    // D wire and sample the survivor in the capture cycle itself.
    guardCapture[DatapathEmitter::accKey(rb.id, k)] = datum;
  }
  // Latch done (a level) for a caller that reads one; clear on start so a
  // retriggered guard re-edges. `handoffSafe` restricts a sibling hand-off to a
  // result-less guard, since a successor's datapath samples survivor registers
  // that settle a cycle after the pulse; a pulse-advanced container instead
  // takes the result through the capture D wire.
  if (levelUnused)
    return Value();
  Value done = ctx.holdDone(pulse, start);
  nameValue(done, regionSignal(rb.id, "done"));
  return done;
}

// Emit the whole module body: preamble (literals, read ports, internal memories
// and channels) once, then the func-scope sibling regions composed by their
// dependence DAG. Nested regions emit inside their container.
void HWEmitter::emit() {
  ctx.clk = ctx.R(seq::ToClockOp::create(ctx.b, ctx.loc, pa.getInput(kClk)));
  ctx.clkRaw = pa.getInput(kClk);
  ctx.rst = pa.getInput(kRst);
  ctx.initLiterals();
  datapath.bindReadPorts();
  datapath.createInternalMemories();
  datapath.declareInternalChannels();
  SmallVector<uarch::RegionId> top;
  for (const uarch::RegionBlock &rb : dp.regions)
    if (!rb.parent) // a child region emits inside its container
      top.push_back(rb.id);
  // retrig keeps the module re-invocable with a fresh `done` edge each drive.
  pa.setOutput(kDone, composeSiblings(top, pa.getInput(kStart)));
  // Stream ports and the internal FIFOs last: a channel's single handshake is
  // shared by every access to it, so it can only be driven once every region
  // has contributed.
  datapath.finalizeStreamPorts();
  // Same reason: a scattered argument's N element outputs are shared by every
  // store to it.
  datapath.finalizeScatteredPorts();
  // And an internal array's write ports, each shared by the stores coloured
  // onto it so the array still infers a block RAM, and an external array's
  // boundary groups, merged onto the same colours so its OWNER can.
  datapath.finalizeSharedWritePorts();
  datapath.finalizeBoundaryWritePorts();
  // An internal read port is one address bus for every read coloured onto it;
  // only now has each of them built the address it presents.
  datapath.finalizeSharedReadPorts();
  // The store->load shadows: a forwarded load's mux needs the paired stores'
  // issue terms, recorded as each region's writes emitted.
  datapath.finalizeForwards();
  // Scalar results: the returning region's survivor register, stable once its
  // region (and thus `done`) has risen; the cosim samples it at `done`.
  for (const uarch::Result &r : dp.results)
    pa.setOutput(r.name, datapath.resolveSource(r.source));
}

} // namespace mlir::allo::uarch

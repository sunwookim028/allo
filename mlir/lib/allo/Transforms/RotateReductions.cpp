/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h" // lookup, latency at a period
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_ROTATEREDUCTIONSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// The reductions a loop carries, one entry per iter_arg in order: a valid step
// is a rotatable reduction, a null step a plain carried value the rebuild
// leaves as a single accumulator.
struct LoopReductions {
  affine::AffineForOp loop;
  SmallVector<ReductionStep> steps;
};

// Classify iter_arg `k`: is its yielded value a reduction step combining that
// iter_arg with a leaf, with the iter_arg and the step result each used once?
ReductionStep matchReductionArg(affine::AffineForOp loop, unsigned k) {
  Value acc = loop.getRegionIterArgs()[k];
  auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
  Value yielded = yield.getOperand(k);
  ReductionStep step = matchReductionStep(yielded);
  if (!step || !yielded.hasOneUse() || !acc.hasOneUse())
    return {};
  // `acc` must be exactly one of the combined operands (the other is the leaf).
  // Its single use also keeps it out of any other reduction's leaf, so the
  // reductions in one loop stay independent.
  auto [lhs, rhs] = reductionOperands(step);
  if ((lhs == acc) == (rhs == acc))
    return {};
  return step;
}

std::optional<LoopReductions> matchReductions(affine::AffineForOp loop) {
  unsigned nArgs = loop.getNumRegionIterArgs();
  if (nArgs == 0)
    return std::nullopt;
  // Only a leaf loop's reduction rotates: the emitter realizes the rotated
  // shift register on a childless modulo loop. A reduction carried by a
  // container loop (`total += inner_sum`) stays at the operator's latency.
  WalkResult nested = loop.getBody()->walk(
      [](affine::AffineForOp) { return WalkResult::interrupt(); });
  if (nested.wasInterrupted())
    return std::nullopt;
  LoopReductions red{loop, SmallVector<ReductionStep>(nArgs)};
  bool any = false;
  for (unsigned k = 0; k < nArgs; ++k)
    any |= bool(red.steps[k] = matchReductionArg(loop, k));
  if (!any)
    return std::nullopt;
  return red;
}

// The identity element of `step`'s operator (0 for add, 1 for mul) as a
// constant of the accumulator's (narrow) type.
Value identityFor(OpBuilder &b, Location loc, const ReductionStep &step) {
  Type ty = step.type();
  if (ty.isIntOrIndex())
    return arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(ty, step.isMul() ? 1 : 0));
  return arith::ConstantOp::create(
      b, loc, b.getFloatAttr(ty, step.isMul() ? 1.0 : 0.0));
}

void rotate(LoopReductions red, unsigned n) {
  affine::AffineForOp loop = red.loop;
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  Value oldIv = loop.getInductionVar();
  auto oldArgs = loop.getRegionIterArgs();
  auto oldInits = loop.getInits();
  unsigned nArgs = oldArgs.size();

  // Lay out the new slots: each reduction expands to N accumulators (its init,
  // then N-1 identities), each plain carried value keeps its one.
  // `start`/`size` locate each old iter_arg's slots.
  SmallVector<Value> inits;
  SmallVector<unsigned> start(nArgs), size(nArgs);
  for (unsigned k = 0; k < nArgs; ++k) {
    start[k] = inits.size();
    size[k] = red.steps[k] ? n : 1;
    inits.push_back(oldInits[k]);
    if (red.steps[k])
      inits.append(n - 1, identityFor(b, loc, red.steps[k]));
  }

  auto oldYield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
  auto newLoop = affine::AffineForOp::create(
      b, loc, loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), inits,
      [&](OpBuilder &nb, Location nloc, Value niv, ValueRange slots) {
        IRMapping map;
        map.map(oldIv, niv);
        // A reduction reads its oldest slot (the last of its group), a plain
        // value its single slot.
        for (unsigned k = 0; k < nArgs; ++k)
          map.map(oldArgs[k], slots[start[k] + size[k] - 1]);
        for (Operation &o : loop.getBody()->without_terminator())
          nb.clone(o, map);
        SmallVector<Value> yields;
        for (unsigned k = 0; k < nArgs; ++k) {
          if (!red.steps[k]) {
            yields.push_back(map.lookupOrDefault(oldYield.getOperand(k)));
            continue;
          }
          // Rotate: the new partial enters slot 0, the rest shift down one.
          yields.push_back(map.lookupOrDefault(red.steps[k].result()));
          for (unsigned s = 0; s + 1 < size[k]; ++s)
            yields.push_back(slots[start[k] + s]);
        }
        affine::AffineYieldOp::create(nb, nloc, yields);
      });

  b.setInsertionPointAfter(newLoop);
  unsigned nRot = 0;
  for (unsigned k = 0; k < nArgs; ++k) {
    Value total;
    if (red.steps[k]) {
      SmallVector<Value> groupResults;
      for (unsigned s = 0; s < size[k]; ++s)
        groupResults.push_back(newLoop.getResult(start[k] + s));
      total = buildBalancedTree(b, red.steps[k], groupResults);
      ++nRot;
    } else {
      total = newLoop.getResult(start[k]);
    }
    loop.getResult(k).replaceAllUsesWith(total);
  }
  info(Stage::Prep, newLoop)
      << "Rotating " << nRot << " reduction(s) across " << n << " accumulators";
  loop.erase();
}

// The auto count for one loop: the largest reduction-operator latency at the
// selection period, so RecII = ceil(L/N) reaches 1 for every step.
unsigned autoAccumulators(const OperatorLibrary &lib,
                          const LoopReductions &red) {
  unsigned n = 0;
  for (const ReductionStep &step : red.steps)
    if (step)
      n = std::max(n, lib.lookup(step.core).timing.latency);
  return n;
}

struct RotateReductionsPass
    : public allo::impl::RotateReductionsPassBase<RotateReductionsPass> {
  using RotateReductionsPassBase::RotateReductionsPassBase;

  void runOnOperation() override {
    if (accumulators == 0)
      return; // off
    assert(accumulators >= -1 &&
           "accumulators is -1 (auto), 0 (off), or a forced count");
    // Auto reads each operator's latency at the period; the library ranks its
    // rows against that period exactly as the scheduler will.
    std::optional<OperatorLibrary> lib;
    if (accumulators < 0) {
      assert(periodNs > 0.0 && "auto accumulators needs the target period");
      lib = OperatorLibrary::fromModule(
          getOperation()->getParentOfType<ModuleOp>());
      lib->setSelectionPeriod((float)periodNs);
    }
    SmallVector<std::pair<LoopReductions, unsigned>> targets;
    getOperation().walk([&](affine::AffineForOp loop) {
      std::optional<LoopReductions> red = matchReductions(loop);
      if (!red)
        return;
      unsigned n = accumulators > 0 ? (unsigned)accumulators
                                    : autoAccumulators(*lib, *red);
      if (n < 2)
        return; // a single accumulator rotates nothing
      // Rotate only a trip known to reach N: a runtime trip below N drains the
      // shift register wrong, so an unknown trip is skipped and a known-short
      // one warns.
      std::optional<uint64_t> trip = affine::getConstantTripCount(loop);
      if (!trip)
        return;
      if (*trip < n) {
        warn(Stage::Prep, loop)
            << "Reduction not rotated because its trip count " << *trip
            << " is below the " << n << " accumulators it would take";
        return;
      }
      targets.push_back({*red, n});
    });
    for (auto &[red, n] : targets)
      rotate(red, n);
  }
};

} // namespace

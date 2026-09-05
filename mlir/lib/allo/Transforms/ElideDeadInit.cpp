/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::allo {
#define GEN_PASS_DEF_ELIDEDEADINITPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using affine::FlatAffineValueConstraints;
using presburger::PresburgerSet;
using presburger::PresburgerSpace;
using presburger::VarKind;

namespace {

// --- element sets ------------------------------------------------------------

// Which region of `guard` holds `access`.
bool inThenRegion(affine::AffineIfOp guard, Operation *access) {
  for (Region *region = access->getParentRegion(); region;
       region = region->getParentRegion())
    if (region->getParentOp() == guard.getOperation())
      return region == &guard.getThenRegion();
  llvm_unreachable("access is not below the guard");
}

// The elements `access` touches, in the array's own index space, symbolic in
// the `depth` induction variables enclosing the array's declaration. Enclosing
// loop domains and guards constrain the iteration space, and negating the
// condition an else region escaped splits one system into several, so this
// appends rather than returning one.
LogicalResult elementSets(Operation *access, unsigned depth,
                          SmallVectorImpl<FlatAffineValueConstraints> &out) {
  SmallVector<Operation *, 8> enclosing;
  affine::getEnclosingAffineOps(*access, &enclosing);
  SmallVector<Operation *, 8> loops;
  llvm::copy_if(enclosing, std::back_inserter(loops), [](Operation *op) {
    return isa<affine::AffineForOp, affine::AffineParallelOp>(op);
  });

  FlatAffineValueConstraints domain;
  if (failed(affine::getIndexSet(loops, &domain)))
    return failure();

  SmallVector<FlatAffineValueConstraints, 1> systems{std::move(domain)};
  for (Operation *op : enclosing) {
    auto guard = dyn_cast<affine::AffineIfOp>(op);
    if (!guard)
      continue;
    if (inThenRegion(guard, access)) {
      for (FlatAffineValueConstraints &system : systems)
        system.addAffineIfOpDomain(guard);
      continue;
    }
    IntegerSet set = guard.getIntegerSet();
    SmallVector<Value> operands(guard.getOperands());
    affine::canonicalizeSetAndOperands(&set, &operands);
    SmallVector<FlatAffineValueConstraints, 1> negated;
    for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
      // Not reaching the then region means failing one of its constraints:
      // `expr >= 0` fails as `-expr - 1 >= 0`, an equality on either side.
      SmallVector<AffineExpr, 2> alternatives{-set.getConstraint(i) - 1};
      if (set.isEq(i))
        alternatives.push_back(set.getConstraint(i) - 1);
      for (AffineExpr alternative : alternatives) {
        AffineExpr constraints[] = {alternative};
        bool equalities[] = {false};
        IntegerSet one = IntegerSet::get(set.getNumDims(), set.getNumSymbols(),
                                         constraints, equalities);
        for (FlatAffineValueConstraints system : systems) {
          FlatAffineValueConstraints piece(one, operands);
          system.mergeAndAlignVarsWithOther(0, &piece);
          system.append(piece);
          negated.push_back(std::move(system));
        }
      }
    }
    systems = std::move(negated);
  }

  affine::MemRefAccess touch(access);
  affine::AffineValueMap accessMap;
  touch.getAccessMap(&accessMap);
  auto type = cast<MemRefType>(touch.memref.getType());
  unsigned rank = type.getRank();
  SmallVector<Value, 4> outer;
  affine::getAffineIVs(*access, outer);
  assert(depth <= outer.size() && "asked for a depth the access is not inside");
  outer.resize(depth);

  for (FlatAffineValueConstraints &system : systems) {
    for (Value operand : accessMap.getOperands())
      if (failed(system.addInductionVarOrTerminalSymbol(operand)))
        return failure();
    if (failed(system.composeMap(&accessMap)))
      return failure();
    system.setDimSymbolSeparation(system.getNumDimAndSymbolVars() - rank);
    // Induction variables below `depth` are existentially quantified, not
    // projected out: Fourier-Motzkin would over-approximate, and an
    // over-claimed write drops a live initializer.
    SmallVector<Value, 4> vars;
    system.getValues(rank, system.getNumDimAndSymbolVars(), &vars);
    for (Value var : vars) {
      if (!affine::isAffineInductionVar(var) || llvm::is_contained(outer, var))
        continue;
      unsigned pos;
      system.findVar(var, &pos);
      system.convertToLocal(VarKind::Symbol, pos - rank, pos - rank + 1);
    }
    system.constantFoldVarRange(rank, system.getNumSymbolVars());
    for (auto [dim, size] : llvm::enumerate(type.getShape())) {
      system.addBound(presburger::BoundType::LB, dim, 0);
      system.addBound(presburger::BoundType::UB, dim, size - 1);
    }
    out.push_back(std::move(system));
  }
  return success();
}

// Every element of `type`, in the array's own index space.
PresburgerSet wholeArray(MemRefType type) {
  presburger::IntegerPolyhedron box(
      PresburgerSpace::getSetSpace(type.getRank()));
  for (auto [dim, size] : llvm::enumerate(type.getShape())) {
    box.addBound(presburger::BoundType::LB, dim, 0);
    box.addBound(presburger::BoundType::UB, dim, size - 1);
  }
  return PresburgerSet(box);
}

// Give a set computed without symbols the walk's symbol columns, leaving them
// unconstrained, so the two live in one space.
PresburgerSet withSymbols(PresburgerSet set, unsigned numSymbols) {
  if (numSymbols)
    set.insertVarInPlace(VarKind::Symbol, 0, numSymbols);
  return set;
}

// Existentially quantify the last symbol column, which is the induction
// variable of the loop the walk is leaving. Quantifying keeps the set exact
// where projecting it out would widen it.
PresburgerSet quantifyLastSymbol(PresburgerSet &set) {
  unsigned last = set.getSpace().getNumSymbolVars() - 1;
  PresburgerSpace shrunk = set.getSpace();
  shrunk.removeVarRange(VarKind::Symbol, last, last + 1);
  PresburgerSet out = PresburgerSet::getEmpty(shrunk);
  for (presburger::IntegerRelation disjunct : set.getAllDisjuncts()) {
    disjunct.convertToLocal(VarKind::Symbol, last, last + 1);
    out.unionInPlace(disjunct);
  }
  return out;
}

// Drop a callee summary's symbol columns, or nullopt when the body constrains
// one, since such a column names something the caller cannot see. The subset
// test is what establishes that a column falls away cleanly.
std::optional<PresburgerSet> dropSymbols(PresburgerSet &set,
                                         unsigned numSymbols) {
  PresburgerSet flat = set;
  for (unsigned i = 0; i < numSymbols; ++i)
    flat = quantifyLastSymbol(flat);
  if (!withSymbols(flat, numSymbols).isSubsetOf(set))
    return std::nullopt;
  return flat;
}

// Values a region built below `root` can carry as a symbol column. The walk
// fixes its symbol list before it starts, so a symbol missing from this list
// makes the region unplaceable and the array opaque.
void collectSymbols(Operation *root, SmallVectorImpl<Value> &out) {
  auto note = [&](Value value) {
    SmallVector<Value, 4> worklist{value};
    while (!worklist.empty()) {
      Value candidate = worklist.pop_back_val();
      if (auto apply = candidate.getDefiningOp<affine::AffineApplyOp>()) {
        llvm::append_range(worklist, apply.getOperands());
        continue;
      }
      if (!affine::isAffineInductionVar(candidate) &&
          !llvm::is_contained(out, candidate))
        out.push_back(candidate);
    }
  };
  root->walk([&](Operation *op) {
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      for (Value operand : loop.getLowerBoundOperands())
        note(operand);
      for (Value operand : loop.getUpperBoundOperands())
        note(operand);
    } else if (auto guard = dyn_cast<affine::AffineIfOp>(op)) {
      for (Value operand : guard.getOperands())
        note(operand);
    } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      for (Value operand : load.getIndices())
        note(operand);
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      for (Value operand : store.getIndices())
        note(operand);
    }
  });
}

// --- coverage by an earlier write of the same access -------------------------

// Whether `op` sits in the else region of some `affine.if`.
// `getEnclosingAffineOps` collects that guard whichever region `op` is in and
// `getIndexSet` then adds its THEN condition, so the iteration domain comes
// back as the complement of the truth. Such accesses stay out of the pairing.
bool inElseRegion(Operation *op) {
  for (Region *region = op->getParentRegion(); region;
       region = region->getParentRegion()) {
    Operation *parent = region->getParentOp();
    if (!parent || parent->hasTrait<OpTrait::AffineScope>())
      return false;
    if (auto guard = dyn_cast<affine::AffineIfOp>(parent))
      if (region == &guard.getElseRegion())
        return true;
  }
  return false;
}

// Bring a batch of constraint systems into one symbol space. The first round
// grows `systems[0]` to the union of every symbol, the second aligns the rest.
unsigned alignSymbols(MutableArrayRef<FlatAffineValueConstraints *> systems) {
  for (int round = 0; round < 2; ++round)
    for (unsigned i = 1; i < systems.size(); ++i)
      systems[0]->mergeSymbolVars(*systems[i]);
  return systems.empty() ? 0 : systems[0]->getNumSymbolVars();
}

// Whether every instance of `load` is preceded by a write to the same element,
// so it never sees what the array held before. `checkMemrefAccessDependence` at
// depth d reports the (write, read) iteration pairs that touch one element with
// the write earlier at loop d, which settles coverage in the read's iteration
// space. Depths at or outside `initDepth` are unusable: there the initializer
// re-ran between the two.
bool coveredByEarlierWrite(Operation *load, ArrayRef<Operation *> stores,
                           unsigned initDepth) {
  if (inElseRegion(load))
    return false;
  affine::MemRefAccess read(load);
  presburger::IntegerRelation readRel(PresburgerSpace::getRelationSpace());
  if (failed(read.getAccessRelation(readRel)))
    return false;
  FlatAffineValueConstraints domain(readRel.getDomainSet());
  unsigned numIvs = domain.getNumDimVars();

  SmallVector<FlatAffineValueConstraints, 1> covers;
  for (Operation *store : stores) {
    if (inElseRegion(store))
      continue;
    affine::MemRefAccess write(store);
    unsigned common = affine::getInnermostCommonLoopDepth({store, load});
    for (unsigned depth = initDepth + 1; depth <= common + 1; ++depth) {
      FlatAffineValueConstraints pairs;
      if (affine::checkMemrefAccessDependence(write, read, depth, &pairs,
                                              /*dependenceComponents=*/nullptr)
              .value != affine::DependenceResult::HasDependence)
        continue;
      // The pairs are laid out as the writing iteration then the reading one.
      // Quantifying the writer away is exact; projecting it out would
      // over-approximate, and over-claimed coverage drops a live initializer.
      assert(pairs.getNumDimVars() >= numIvs && "dependence lost the reader");
      unsigned numWriterIvs = pairs.getNumDimVars() - numIvs;
      pairs.convertToLocal(VarKind::SetDim, 0, numWriterIvs);
      covers.push_back(std::move(pairs));
    }
  }
  if (covers.empty())
    return false;

  SmallVector<FlatAffineValueConstraints *> systems{&domain};
  for (FlatAffineValueConstraints &cover : covers)
    systems.push_back(&cover);
  PresburgerSpace space =
      PresburgerSpace::getSetSpace(numIvs, alignSymbols(systems));
  PresburgerSet covered = PresburgerSet::getEmpty(space);
  for (FlatAffineValueConstraints &cover : covers)
    covered = covered.unionSet(PresburgerSet(cover));
  return PresburgerSet(domain).isSubsetOf(covered);
}

// --- what the pass models ----------------------------------------------------

bool isFillOf(Operation *op, Value buf) {
  auto fill = dyn_cast<linalg::FillOp>(op);
  return fill && fill.getDpsInits().size() == 1 && fill.getDpsInits()[0] == buf;
}

// The array traffic this pass models. Anything else hides an access, and a
// hidden read is what makes an initializer live.
bool isModelledUse(Operation *op, Value buf) {
  return isa<affine::AffineLoadOp, affine::AffineStoreOp, func::CallOp>(op) ||
         isFillOf(op, buf);
}

// --- why an initializer survived ---------------------------------------------

// Debug output only: an array reaching any of these but `Elided` keeps its
// initializer.
enum class Decline {
  Elided,
  DynamicShape,
  UnmodelledUse,   // a view, a return, a non-affine subscript, an opaque callee
  NoInitializer,   // nothing initializes the array: nothing to remove
  LateInitializer, // it has one, but not as the first thing to touch it
  Opaque,          // a region or a callee summary this pass cannot compute
  UncoveredRead,   // a read no earlier write is known to reach
  Count,
};

StringRef describeDecline(Decline reason) {
  switch (reason) {
  case Decline::Elided:
    return "elided";
  case Decline::DynamicShape:
    return "dynamic shape";
  case Decline::UnmodelledUse:
    return "unmodelled use";
  case Decline::NoInitializer:
    return "never initialized";
  case Decline::LateInitializer:
    return "initializer not first or not erasable";
  case Decline::Opaque:
    return "opaque region or callee";
  case Decline::UncoveredRead:
    return "read not covered";
  case Decline::Count:
    break;
  }
  llvm_unreachable("unhandled decline reason");
}

// The source-level name of a local array, which the frontend puts in the
// allocation's location. A pass-built array has none.
std::string nameOf(Operation *alloc) {
  std::string name =
      logging::detail::describe(alloc->getLoc(), /*withFile=*/false);
  return name.empty() ? "<unnamed>" : name;
}

// --- the walk ----------------------------------------------------------------

// Everything one array's walk needs and the next walk must not inherit.
struct Walk {
  Value buf;
  MemRefType type;
  unsigned initDepth = 0; // loops enclosing the array's initializer
  // The symbol columns every set of this walk carries, in order. The tail is a
  // stack: entering a loop pushes its induction variable, leaving quantifies it
  // away again.
  SmallVector<Value> symbols;
  SmallVector<Operation *> stores; // the pairwise fallback's candidate writes
  Operation *exposedBy = nullptr; // the read that first escaped, for the report

  PresburgerSpace space() {
    return PresburgerSpace::getSetSpace(type.getRank(), symbols.size());
  }
  PresburgerSet empty() { return PresburgerSet::getEmpty(space()); }
  bool touches(Operation *op) {
    return llvm::any_of(buf.getUsers(),
                        [&](Operation *user) { return op->isAncestor(user); });
  }
};

// Move `region` into the walk's symbol space, one column per Value in
// `symbols`. Fails on a symbol the setup did not anticipate, since aligning it
// against the wrong column would silently change what the set means.
std::optional<PresburgerSet> place(Walk &walk,
                                   FlatAffineValueConstraints &region) {
  for (auto [index, symbol] : llvm::enumerate(walk.symbols)) {
    unsigned wanted = region.getNumDimVars() + index;
    unsigned pos;
    if (region.findVar(symbol, &pos, wanted) &&
        pos < region.getNumDimAndSymbolVars())
      region.swapVar(wanted, pos);
    else
      region.insertSymbolVar(index, ValueRange(symbol));
  }
  if (region.getNumSymbolVars() != walk.symbols.size())
    return std::nullopt;
  return PresburgerSet(region);
}

// What a function's body reads from and always writes to one of its memref
// parameters, in that parameter's index space.
struct ArgEffect {
  PresburgerSet must, exposed;
};

// One run's state. The per-parameter summaries are shared across every array,
// since a callee is summarized the same way whoever allocated the argument.
struct Elider {
  SymbolTableCollection symbols;
  // Why the last round declined, for the report. Only the driver knows whether
  // these stand for the array or just terminated a chain of elisions.
  Operation *blame = nullptr;
  StringRef detail;
  DenseMap<std::pair<Operation *, unsigned>, std::optional<ArgEffect>> cache;
  DenseSet<Operation *> onStack;

  // The union of the element sets `access` touches, placed in the walk's space.
  std::optional<PresburgerSet> touched(Walk &walk, Operation *access,
                                       unsigned depth) {
    SmallVector<FlatAffineValueConstraints, 1> regions;
    if (failed(elementSets(access, depth, regions)))
      return std::nullopt;
    PresburgerSet set = walk.empty();
    for (FlatAffineValueConstraints &region : regions) {
      std::optional<PresburgerSet> placed = place(walk, region);
      if (!placed)
        return std::nullopt;
      set = set.unionSet(*placed);
    }
    return set;
  }

  // Fold a read into `exposed`, unless what it reaches is already written or an
  // earlier write of the same access covers every one of its iterations. Only
  // an affine load has an access to pair up that way; a call is a set.
  void expose(Walk &walk, Operation *reader, PresburgerSet &read,
              PresburgerSet &written, PresburgerSet &exposed) {
    PresburgerSet fresh = read.subtract(written);
    if (fresh.isIntegerEmpty())
      return;
    if (isa<affine::AffineLoadOp>(reader) &&
        coveredByEarlierWrite(reader, walk.stores, walk.initDepth))
      return;
    if (!walk.exposedBy)
      walk.exposedBy = reader;
    exposed = exposed.unionSet(fresh);
  }

  // Thread the two sets forward over `[begin, end)`: `written` is an
  // under-approximation of what every execution has written by the current op,
  // `exposed` an over-approximation of what is read before a write covered it.
  // Returns false when an effect cannot be modelled at all, which is a
  // different answer from a read that escapes.
  bool scan(Walk &walk, Block::iterator begin, Block::iterator end,
            unsigned depth, PresburgerSet &written, PresburgerSet &exposed) {
    for (Block::iterator it = begin; it != end; ++it)
      if (walk.touches(&*it) && !scanOp(walk, *it, depth, written, exposed))
        return false;
    return true;
  }

  bool scanOp(Walk &walk, Operation &op, unsigned depth, PresburgerSet &written,
              PresburgerSet &exposed) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
      std::optional<PresburgerSet> set = touched(walk, &op, depth);
      if (!set)
        return false;
      if (isa<affine::AffineLoadOp>(op))
        expose(walk, &op, *set, written, exposed);
      else
        written = written.unionSet(*set);
      return true;
    }
    if (isFillOf(&op, walk.buf)) {
      written = written.unionSet(
          withSymbols(wholeArray(walk.type), walk.symbols.size()));
      return true;
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
        if (actual != walk.buf)
          continue;
        std::optional<ArgEffect> effect = calleeEffect(call, k);
        if (!effect)
          return false;
        PresburgerSet read = withSymbols(effect->exposed, walk.symbols.size());
        expose(walk, &op, read, written, exposed);
        written =
            written.unionSet(withSymbols(effect->must, walk.symbols.size()));
      }
      return true;
    }
    // A loop body starts from what was written before the loop and nothing
    // else: an element the body writes at one iteration is not there at the
    // first. Quantifying the induction variable away on the way out unions the
    // body's writes over the iteration space, empty when the loop may not run.
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      walk.symbols.push_back(loop.getInductionVar());
      PresburgerSet body = withSymbols(written, 1);
      PresburgerSet bodyExposed = withSymbols(exposed, 1);
      bool modelled = scan(walk, loop.getBody()->begin(), loop.getBody()->end(),
                           depth + 1, body, bodyExposed);
      written = quantifyLastSymbol(body);
      exposed = quantifyLastSymbol(bodyExposed);
      walk.symbols.pop_back();
      return modelled;
    }
    Block *taken = nullptr, *skipped = nullptr;
    bool affineGuard = false;
    if (auto guard = dyn_cast<affine::AffineIfOp>(op)) {
      taken = guard.getThenBlock();
      skipped = guard.hasElse() ? guard.getElseBlock() : nullptr;
      affineGuard = true;
    } else if (auto guard = dyn_cast<scf::IfOp>(op)) {
      taken = guard.thenBlock();
      skipped = guard.elseBlock();
    }
    if (taken) {
      PresburgerSet thenWritten = written, thenExposed = exposed;
      if (!scan(walk, taken->begin(), taken->end(), depth, thenWritten,
                thenExposed))
        return false;
      PresburgerSet elseWritten = written, elseExposed = exposed;
      if (skipped && !scan(walk, skipped->begin(), skipped->end(), depth,
                           elseWritten, elseExposed))
        return false;
      // Each access below an affine guard carries that guard in its own element
      // set, so the branches add up. A data-dependent `scf.if` leaves no
      // condition to fold, so only what both branches write is certain.
      written = affineGuard ? thenWritten.unionSet(elseWritten)
                            : thenWritten.intersect(elseWritten);
      exposed = thenExposed.unionSet(elseExposed);
      return true;
    }
    // A region this pass does not model may run any number of times, so it
    // writes nothing for certain and every read below it is exposed.
    bool modelled = true;
    op.walk([&](Operation *nested) {
      if (nested == &op || !walk.touches(nested))
        return;
      if (isa<affine::AffineLoadOp>(nested)) {
        std::optional<PresburgerSet> set = touched(walk, nested, depth);
        modelled &= set.has_value();
        if (set)
          expose(walk, nested, *set, written, exposed);
        return;
      }
      auto call = dyn_cast<func::CallOp>(nested);
      if (!call)
        return;
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
        if (actual != walk.buf)
          continue;
        std::optional<ArgEffect> effect = calleeEffect(call, k);
        modelled &= effect.has_value();
        if (effect) {
          PresburgerSet read =
              withSymbols(effect->exposed, walk.symbols.size());
          expose(walk, nested, read, written, exposed);
        }
      }
    });
    return modelled;
  }

  std::optional<ArgEffect> summarize(func::FuncOp callee, unsigned k) {
    auto key = std::make_pair(callee.getOperation(), k);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
    // A cycle has no fixed point to iterate towards here, so it reads as opaque
    // and poisons every caller on the way out.
    if (!onStack.insert(callee.getOperation()).second)
      return std::nullopt;
    std::optional<ArgEffect> effect = computeSummary(callee, k);
    onStack.erase(callee.getOperation());
    cache[key] = effect;
    return effect;
  }

  std::optional<ArgEffect> computeSummary(func::FuncOp callee, unsigned k) {
    if (callee.isExternal())
      return std::nullopt;
    BlockArgument arg = callee.getArgument(k);
    auto type = dyn_cast<MemRefType>(arg.getType());
    if (!type || !type.hasStaticShape())
      return std::nullopt;
    for (Operation *user : arg.getUsers())
      if (!isModelledUse(user, arg))
        return std::nullopt;

    Block &body = callee.getBody().front();
    Walk walk{arg, type, /*initDepth=*/0, {}, {}, nullptr};
    collectSymbols(callee, walk.symbols);
    for (Operation *user : arg.getUsers())
      if (isa<affine::AffineStoreOp>(user))
        walk.stores.push_back(user);
    PresburgerSet written = walk.empty(), exposed = walk.empty();
    if (!scan(walk, body.begin(), body.end(), /*depth=*/0, written, exposed))
      return std::nullopt;
    // A parameter's effect is stated in its own index space.
    std::optional<PresburgerSet> must =
        dropSymbols(written, walk.symbols.size());
    std::optional<PresburgerSet> escaped =
        dropSymbols(exposed, walk.symbols.size());
    if (!must || !escaped)
      return std::nullopt;
    return ArgEffect{*must, *escaped};
  }

  std::optional<ArgEffect> calleeEffect(func::CallOp call, unsigned k) {
    auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee)
      return std::nullopt;
    return summarize(callee, k);
  }

  // An initializer this pass may delete once it is proven dead: a whole-array
  // fill, or a loop nest whose only effect is writing this array. Erasing the
  // nest discards nothing, since a value defined in a region cannot escape it.
  bool erasable(Operation *init, Value buf) {
    if (isFillOf(init, buf))
      return true;
    auto nest = dyn_cast<affine::AffineForOp>(init);
    if (!nest)
      return false;
    bool onlyWrites = true;
    nest.walk([&](Operation *op) {
      if (isa<affine::AffineForOp>(op) || op->hasTrait<OpTrait::IsTerminator>())
        return;
      auto store = dyn_cast<affine::AffineStoreOp>(op);
      if (store && store.getMemRef() == buf)
        return;
      onlyWrites &= isMemoryEffectFree(op);
    });
    return onlyWrites;
  }

  Decline elideOne(Operation *alloc) {
    Value buf = alloc->getResult(0);
    auto type = cast<MemRefType>(buf.getType());
    blame = alloc;
    if (!type.hasStaticShape()) {
      detail = "its shape is dynamic";
      return Decline::DynamicShape;
    }
    for (Operation *user : buf.getUsers())
      if (!isModelledUse(user, buf)) {
        blame = user;
        detail = "this use is not traffic the pass models";
        return Decline::UnmodelledUse;
      }

    // The initializer has to be the first thing that touches the array. A read
    // before it sees whatever the previous execution of this block left behind,
    // which dropping the initializer would change.
    auto below = [&](Operation *op) {
      return llvm::any_of(buf.getUsers(), [&](Operation *user) {
        return op->isAncestor(user);
      });
    };
    Operation *init = nullptr;
    for (Operation *op = alloc->getNextNode(); op; op = op->getNextNode())
      if (below(op)) {
        init = op;
        break;
      }
    if (!init || !erasable(init, buf)) {
      // An array nobody initializes has nothing to remove, which is a different
      // thing from one whose initializer this pass cannot reach.
      if (llvm::none_of(buf.getUsers(),
                        [&](Operation *user) { return isFillOf(user, buf); })) {
        detail = "nothing initializes it";
        return Decline::NoInitializer;
      }
      blame = init ? init : alloc;
      detail = "its initializer is not the first thing to touch it, or is not"
               " one the pass can delete";
      return Decline::LateInitializer;
    }

    // The loops around the DECLARATION stay symbols, so a region like `0..k` is
    // compared against another `0..k` rather than against its hull.
    Walk walk{buf, type, /*initDepth=*/0, {}, {}, nullptr};
    affine::getAffineIVs(*alloc, walk.symbols);
    unsigned depth = walk.symbols.size();
    walk.initDepth = depth;
    for (Operation *op = init->getNextNode(); op; op = op->getNextNode())
      collectSymbols(op, walk.symbols);
    for (Operation *user : buf.getUsers())
      if (isa<affine::AffineStoreOp>(user))
        walk.stores.push_back(user);

    PresburgerSet written = walk.empty(), exposed = walk.empty();
    if (!scan(walk, Block::iterator(init->getNextNode()),
              alloc->getBlock()->end(), depth, written, exposed)) {
      blame = walk.exposedBy ? walk.exposedBy : alloc;
      detail = "an op's effect on it is opaque";
      return Decline::Opaque;
    }
    if (!exposed.isIntegerEmpty()) {
      blame = walk.exposedBy ? walk.exposedBy : alloc;
      detail = "this read may reach an element no earlier write covers";
      return Decline::UncoveredRead;
    }

    info(Stage::Prep, init)
        << "Every element of the local array '" << nameOf(alloc)
        << "' that is read is written first, so its initialization is dead:"
           " dropped it rather than re-running it on every enclosing iteration";
    init->erase();
    return Decline::Elided;
  }
};

struct ElideDeadInitPass
    : public allo::impl::ElideDeadInitPassBase<ElideDeadInitPass> {
  void runOnOperation() override {
    SmallVector<Operation *> allocs;
    getOperation()->walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        allocs.push_back(op);
    });
    Elider elider;
    std::array<unsigned, size_t(Decline::Count)> tally{};
    for (Operation *alloc : allocs) {
      // A chain of initializers collapses one at a time: the survivor of a
      // round may itself be dead against what follows it. The reason from the
      // round that stops the chain stands for the array only when none elided.
      Decline reason;
      unsigned dropped = 0;
      while ((reason = elider.elideOne(alloc)) == Decline::Elided)
        ++dropped;
      if (dropped) {
        ++tally[size_t(Decline::Elided)];
        continue;
      }
      ++tally[size_t(reason)];
      debug(Stage::Prep, elider.blame)
          << "Local array '" << nameOf(alloc)
          << "' keeps its initializer: " << elider.detail;
    }
    if (allocs.empty() || !logging::detail::enabled(logging::Level::Debug))
      return;
    logging::Diagnostic report = debug(Stage::Prep);
    report << "Local arrays seen by elide-dead-init: " << allocs.size();
    for (auto [index, count] : llvm::enumerate(tally))
      if (count)
        report << ", " << count << " " << describeDecline(Decline(index));
  }
};

} // namespace

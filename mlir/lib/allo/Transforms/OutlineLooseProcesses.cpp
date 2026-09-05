/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"             // StreamCreateOp, StreamType
#include "allo/Scheduling/RegionGraph.h" // composesOnStructuralTop
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::allo {
#define GEN_PASS_DEF_OUTLINELOOSEPROCESSESPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A type an outlined process can take as a parameter. Mirrors what the
// structural top can wire: arrays and streams by reference, scalars by value.
bool portableArg(Type t) {
  return isa<MemRefType, StreamType, IndexType>(t) || t.isIntOrFloat();
}

struct OutlineLooseProcessesPass
    : public allo::impl::OutlineLooseProcessesPassBase<
          OutlineLooseProcessesPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto func : llvm::to_vector(module.getOps<func::FuncOp>()))
      if (!func.isExternal() && composesOnStructuralTop(func))
        outlineContainer(func, module);
  }

  // Split \p container's entry block into maximal runs of loose (datapath) ops,
  // separated by the calls that order them, and make each run a process of its
  // own. Run k's call sits where run k was, so program order is preserved.
  void outlineContainer(func::FuncOp container, ModuleOp module) {
    SmallVector<SmallVector<Operation *>> runs;
    SmallVector<Operation *> cur;
    for (Operation &op : container.front()) {
      // Only a call (or the terminator) breaks a run: the declarations and
      // constants left behind carry no ordering of their own.
      if (isa<func::CallOp>(op) || op.hasTrait<OpTrait::IsTerminator>()) {
        if (!cur.empty())
          runs.push_back(std::move(cur));
        cur.clear();
      } else if (!isContainerStructure(op))
        cur.push_back(&op);
    }
    if (!cur.empty())
      runs.push_back(std::move(cur));

    unsigned k = 0;
    for (SmallVector<Operation *> &run : runs)
      outlineRun(run, container, module, k);
  }

  void outlineRun(ArrayRef<Operation *> run, func::FuncOp container,
                  ModuleOp module, unsigned &k) {
    DenseSet<Operation *> inRun(run.begin(), run.end());
    auto inside = [&](Operation *op) {
      for (; op; op = op->getParentOp())
        if (inRun.contains(op))
          return true;
      return false;
    };
    auto definedInside = [&](Value v) {
      if (auto arg = dyn_cast<BlockArgument>(v))
        return inside(arg.getOwner()->getParentOp());
      return inside(v.getDefiningOp());
    };

    // Inputs: everything the run reads from outside it. A constant is cloned
    // into the callee rather than threaded through a port, which would cost the
    // container a datapath to drive it.
    SetVector<Value> inputs;
    SetVector<Operation *> constants;
    for (Operation *op : run)
      op->walk([&](Operation *inner) {
        for (Value v : inner->getOperands()) {
          if (definedInside(v))
            continue;
          Operation *def = v.getDefiningOp();
          if (def && def->hasTrait<OpTrait::ConstantLike>())
            constants.insert(def);
          else
            inputs.insert(v);
        }
      });
    // Outputs: everything defined in the run that anything outside still reads.
    SetVector<Value> outputs;
    for (Operation *op : run)
      for (Value r : op->getResults())
        for (Operation *u : r.getUsers())
          if (!inside(u)) {
            outputs.insert(r);
            break;
          }

    // A run the structural top could not wire anyway is left where it is; the
    // emitter's check reports it.
    auto portable = [](Value v) { return portableArg(v.getType()); };
    if (!llvm::all_of(inputs, portable) || !llvm::all_of(outputs, [](Value v) {
          return v.getType().isIntOrFloat();
        }))
      return;

    OpBuilder b(container);
    SmallVector<Type> argTypes(
        llvm::map_range(inputs, [](Value v) { return v.getType(); }));
    SmallVector<Type> resTypes(
        llvm::map_range(outputs, [](Value v) { return v.getType(); }));
    std::string name;
    do {
      name = (container.getName() + ".datapath" + Twine(k++)).str();
    } while (module.lookupSymbol(name));

    Location loc = run.front()->getLoc();
    auto fn = func::FuncOp::create(b, loc, name,
                                   b.getFunctionType(argTypes, resTypes));
    fn.setPrivate();
    Block *entry = fn.addEntryBlock();

    // Place the call at the run's LAST op: every input is read by some op of
    // the run and every outside user of an output comes after it, so both
    // directions dominate by construction.
    b.setInsertionPointAfter(run.back());
    auto call = func::CallOp::create(b, loc, fn, inputs.getArrayRef());
    for (auto [i, res] : llvm::enumerate(call.getResults())) {
      Value out = outputs[i];
      out.replaceUsesWithIf(
          res, [&](OpOperand &use) { return !inside(use.getOwner()); });
    }

    b.setInsertionPointToStart(entry);
    SmallVector<std::pair<Value, Value>> rebind; // outer value -> callee value
    for (Operation *c : constants)
      rebind.emplace_back(c->getResult(0), b.clone(*c)->getResult(0));
    for (Operation *op : run)
      op->moveBefore(entry, entry->end());
    for (auto [v, arg] : llvm::zip(inputs, entry->getArguments()))
      rebind.emplace_back(v, arg);
    for (auto [outer, in] : rebind)
      outer.replaceUsesWithIf(in, [&, fn = fn](OpOperand &use) {
        return use.getOwner()->getParentOfType<func::FuncOp>() == fn;
      });
    b.setInsertionPointToEnd(entry);
    func::ReturnOp::create(b, loc, outputs.getArrayRef());

    info(Stage::Prep, container)
        << "Outlined a loose datapath span of " << run.size()
        << " op(s) into process '" << name
        << "': a dataflow container composes processes, so its own compute "
           "becomes one more of them";
  }
};

} // namespace

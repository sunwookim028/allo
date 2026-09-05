/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_DROPTRIVIALFUNCPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// Which argument each result of \p func forwards, when the whole body is a
// return of block arguments. An empty body is the zero-result case of that same
// shape, so both fall out of one test. A returned constant or a value the body
// computed is not a forward, and neither is a second block: control flow is
// work.
std::optional<SmallVector<unsigned>> forwardedArguments(func::FuncOp func) {
  if (func.isExternal() || !func.getBody().hasOneBlock())
    return std::nullopt;
  Block &body = func.getBody().front();
  auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!ret || !body.without_terminator().empty())
    return std::nullopt;
  SmallVector<unsigned> forwarded;
  for (Value result : ret.getOperands()) {
    auto arg = dyn_cast<BlockArgument>(result);
    if (!arg || arg.getOwner() != &body)
      return std::nullopt;
    forwarded.push_back(arg.getArgNumber());
  }
  return forwarded;
}

// The `func.call` ops naming \p func, or nullopt when the symbol is reached
// some other way: a use this pass cannot rewrite is a reason to keep the
// function whole.
std::optional<SmallVector<func::CallOp>> callsOf(func::FuncOp func,
                                                 ModuleOp module) {
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(func, module);
  if (!uses)
    return std::nullopt;
  SmallVector<func::CallOp> calls;
  for (SymbolTable::SymbolUse use : *uses) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call)
      return std::nullopt;
    calls.push_back(call);
  }
  return calls;
}

struct DropTrivialFuncPass
    : public allo::impl::DropTrivialFuncPassBase<DropTrivialFuncPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Dropping a call can leave its own caller trivial, so this runs to a
    // fixpoint rather than once over the call graph.
    for (bool changed = true; changed;) {
      changed = false;
      for (auto func :
           llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
        // A public function is the module's interface, so its symbol outlives
        // whatever its body happens to say.
        if (!func.isPrivate())
          continue;
        auto forwarded = forwardedArguments(func);
        if (!forwarded)
          continue;
        auto calls = callsOf(func, module);
        if (!calls)
          continue;
        info(Stage::Prep, func)
            << "Dropping empty/identity function '" << func.getName()
            << "', which computes nothing, and its " << calls->size()
            << " call site(s)";
        for (func::CallOp call : *calls) {
          for (auto [result, arg] : llvm::zip(call.getResults(), *forwarded))
            result.replaceAllUsesWith(call.getOperand(arg));
          call.erase();
        }
        func.erase();
        changed = true;
      }
    }
  }
};

} // namespace

/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- MaterializeApintWrapper.cpp ----------------------------------------===//
//
// Wrap a top kernel whose boundary uses non-standard-width integers with a
// standard-width interface, so numpy/ctypes/the LLVM memref ABI can talk to it.
//
//===----------------------------------------------------------------------===//

#include "allo/IR/AlloOps.h"
#include "allo/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::allo {
#define GEN_PASS_DEF_GENERATEAPINTWRAPPERPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

// Smallest standard byte width >= w, or 0 if w does not fit in 64 bits.
static unsigned stdWidth(unsigned w) {
  if (w <= 8)
    return 8;
  if (w <= 16)
    return 16;
  if (w <= 32)
    return 32;
  if (w <= 64)
    return 64;
  return 0;
}

// If `t` is an integer needing widening (non-standard width), return its width;
// otherwise 0. Standard widths {1,8,16,32,64} and non-integer types return 0.
static unsigned nonStdIntWidth(Type t) {
  auto it = dyn_cast<IntegerType>(t);
  if (!it)
    return 0;
  unsigned w = it.getWidth();
  if (w == 1 || w == 8 || w == 16 || w == 32 || w == 64)
    return 0;
  return w;
}

// The standard-width boundary type for a kernel operand/result type. Returns
// the same type when no widening is needed and sets `changed` when it is.
static Type boundaryType(Type t, bool &changed) {
  if (auto mr = dyn_cast<MemRefType>(t)) {
    unsigned w = nonStdIntWidth(mr.getElementType());
    if (!w)
      return t;
    changed = true;
    return MemRefType::get(mr.getShape(),
                           IntegerType::get(t.getContext(), stdWidth(w)),
                           mr.getLayout(), mr.getMemorySpace());
  }
  if (unsigned w = nonStdIntWidth(t)) {
    changed = true;
    return IntegerType::get(t.getContext(), stdWidth(w));
  }
  return t;
}

// The `allo.signed` marker carries one char per operand then result: 's'
// signed, 'u' unsigned, 'x' non-integer. Missing/short markers default to
// unsigned.
static bool operandIsSigned(KernelOp kernel, unsigned idx) {
  auto attr = kernel->getAttrOfType<StringAttr>(kAlloSignedAttr);
  if (!attr)
    return false;
  StringRef marker = attr.getValue();
  return idx < marker.size() && marker[idx] == 's';
}

// Emit a `dst[i...] = cast(src[i...])` loop nest over the full static shape.
static void buildCopyLoop(OpBuilder &b, Location loc, Value src, Value dst,
                          bool toApint, bool isSigned) {
  ArrayRef<int64_t> shape = cast<MemRefType>(src.getType()).getShape();
  Type dstElem = cast<MemRefType>(dst.getType()).getElementType();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  SmallVector<Value> lbs(shape.size(), zero), steps(shape.size(), one), ubs;
  for (int64_t d : shape)
    ubs.push_back(arith::ConstantIndexOp::create(b, loc, d));

  scf::buildLoopNest(b, loc, lbs, ubs, steps,
                     [&](OpBuilder &nb, Location nl, ValueRange ivs) {
                       Value v = memref::LoadOp::create(nb, nl, src, ivs);
                       Value cvt;
                       if (toApint)
                         cvt = arith::TruncIOp::create(nb, nl, dstElem, v);
                       else if (isSigned)
                         cvt = arith::ExtSIOp::create(nb, nl, dstElem, v);
                       else
                         cvt = arith::ExtUIOp::create(nb, nl, dstElem, v);
                       memref::StoreOp::create(nb, nl, cvt, dst, ivs);
                     });
}

struct MaterializeApintWrapperPass
    : public allo::impl::GenerateApintWrapperPassBase<
          MaterializeApintWrapperPass> {
  using GenerateApintWrapperPassBase::GenerateApintWrapperPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (topName.empty())
      return;
    auto top = module.lookupSymbol<KernelOp>(topName);
    if (!top || top.getBody().empty())
      return;

    FunctionType fnTy = top.getFunctionType();
    bool changed = false;
    SmallVector<Type> stdInputs, stdResults;
    for (Type t : fnTy.getInputs())
      stdInputs.push_back(boundaryType(t, changed));
    for (Type t : fnTy.getResults())
      stdResults.push_back(boundaryType(t, changed));
    if (!changed)
      return; // boundary already standard-width; nothing to do.

    // Reject integer widths that no numpy/ctypes scalar can hold.
    auto tooWide = [](Type t) {
      Type e = isa<MemRefType>(t) ? cast<MemRefType>(t).getElementType() : t;
      unsigned w = nonStdIntWidth(e);
      return w && stdWidth(w) == 0;
    };
    if (llvm::any_of(fnTy.getInputs(), tooWide) ||
        llvm::any_of(fnTy.getResults(), tooWide)) {
      top.emitError("APInt wrapper does not support integer widths > 64 bits");
      return signalPassFailure();
    }

    MLIRContext *ctx = &getContext();
    StringAttr origName = top.getSymNameAttr();
    std::string implName = (origName.getValue() + "__impl").str();

    // Rename the real kernel `<name>__impl` and hide it; the wrapper takes the
    // public name and converts around an invoke of it.
    top.setSymName(implName);
    top.setSymVisibility("private");

    OpBuilder b(top);
    Location loc = top.getLoc();
    auto wrapper = KernelOp::create(
        b, loc, origName,
        TypeAttr::get(FunctionType::get(ctx, stdInputs, stdResults)),
        b.getStringAttr("public"), /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr, b.getDenseI32ArrayAttr({}));
    if (auto marker = top->getAttrOfType<StringAttr>(kAlloSignedAttr))
      wrapper->setAttr(kAlloSignedAttr, marker);

    SmallVector<Location> argLocs(stdInputs.size(), loc);
    Block *entry = b.createBlock(&wrapper.getBody(), wrapper.getBody().end(),
                                 stdInputs, argLocs);
    b.setInsertionPointToStart(entry);

    struct CopyOut {
      Value tmp, stdArg;
      bool isSigned;
    };
    SmallVector<Value> callOperands;
    SmallVector<CopyOut> copyOut;
    for (unsigned i = 0; i < fnTy.getNumInputs(); ++i) {
      Type origTy = fnTy.getInput(i);
      Value stdArg = entry->getArgument(i);
      if (auto mr = dyn_cast<MemRefType>(origTy)) {
        if (!nonStdIntWidth(mr.getElementType())) {
          callOperands.push_back(stdArg);
          continue;
        }
        bool sgn = operandIsSigned(top, i);
        auto alloc = memref::AllocOp::create(b, loc, mr);
        // Tag the temp so the emitter renders its element type with the same
        // signedness as the callee parameter it feeds.
        alloc->setAttr(kAlloSignedAttr, b.getStringAttr(sgn ? "s" : "u"));
        Value tmp = alloc.getResult();
        buildCopyLoop(b, loc, stdArg, tmp, /*toApint=*/true,
                      /*isSigned=*/false);
        callOperands.push_back(tmp);
        copyOut.push_back({tmp, stdArg, sgn});
      } else if (nonStdIntWidth(origTy)) {
        callOperands.push_back(arith::TruncIOp::create(b, loc, origTy, stdArg));
      } else {
        callOperands.push_back(stdArg);
      }
    }

    auto invoke = InvokeOp::create(b, loc, top, callOperands);

    for (const CopyOut &c : copyOut) {
      buildCopyLoop(b, loc, c.tmp, c.stdArg, /*toApint=*/false, c.isSigned);
      memref::DeallocOp::create(b, loc, c.tmp);
    }

    SmallVector<Value> retVals;
    for (unsigned j = 0; j < fnTy.getNumResults(); ++j) {
      Value r = invoke.getResult(j);
      if (!nonStdIntWidth(fnTy.getResult(j))) {
        retVals.push_back(r);
        continue;
      }
      bool sgn = operandIsSigned(top, fnTy.getNumInputs() + j);
      retVals.push_back(
          sgn ? arith::ExtSIOp::create(b, loc, stdResults[j], r).getResult()
              : arith::ExtUIOp::create(b, loc, stdResults[j], r).getResult());
    }
    ReturnOp::create(b, loc, retVals);
  }
};

} // namespace

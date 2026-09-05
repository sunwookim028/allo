/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "allo/TransformOps/AlloTransformOps.h"

using namespace mlir;
using namespace mlir::allo;

// `transform.allo.streamline` converts the memref boundaries between fused
// kernels into on-chip stream hand-offs. apply() dispatches each boundary on
// its (writers, readers) shape: 1->1 a direct hand-off, 1->N a generated `tee`,
// N->1 a generated `merge`. Each side then meets the row-major FIFO contract as
// passthrough (in place), windowed (a K-row line buffer) or staged (a
// full-tensor reorder buffer). `lanes=L` widens a boundary to L parallel FIFOs
// and `depth` sizes them.
namespace {

enum class ArgKind { Unused, ReadOnly, WriteOnly, ReadWrite, NonAnalyzable };

// Read operand `idx` of a kernel's positional `allo.signed` marker ('s' =>
// signed). MLIR integers are signless, so the FIFO element type's signedness
// can only be recovered from this marker.
static bool operandIsSigned(KernelOp kernel, unsigned idx) {
  auto attr = kernel->getAttrOfType<StringAttr>(allo::kAlloSignedAttr);
  if (!attr)
    return false;
  StringRef marker = attr.getValue();
  return idx < marker.size() && marker[idx] == 's';
}

// Classify a kernel block argument by walking its direct uses. Any use that is
// not a direct affine/memref load or store (a view, a nested invoke, ...) makes
// the argument non-analyzable.
static ArgKind classifyArg(BlockArgument arg) {
  bool hasLoad = false, hasStore = false;
  for (OpOperand &use : arg.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<affine::AffineLoadOp, memref::LoadOp>(owner))
      hasLoad = true;
    else if (isa<affine::AffineStoreOp, memref::StoreOp>(owner))
      hasStore = true;
    else
      return ArgKind::NonAnalyzable;
  }
  if (hasLoad && hasStore)
    return ArgKind::ReadWrite;
  if (hasLoad)
    return ArgKind::ReadOnly;
  if (hasStore)
    return ArgKind::WriteOnly;
  return ArgKind::Unused;
}

// Build a perfect row-major affine.for nest over `shape` and call `body` at the
// innermost point with the induction variables. Rank 0 calls `body` once.
static void
buildRowMajorNest(OpBuilder &builder, Location loc, ArrayRef<int64_t> shape,
                  function_ref<void(OpBuilder &, ValueRange)> body) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> ivs;
  affine::AffineForOp innermost;
  for (int64_t dim : shape) {
    auto forOp = affine::AffineForOp::create(builder, loc, 0, dim, 1);
    ivs.push_back(forOp.getInductionVar());
    innermost = forOp;
    builder.setInsertionPointToStart(forOp.getBody());
  }
  if (innermost)
    innermost->setAttr(kPipelineIIAttr, builder.getI64IntegerAttr(1));
  body(builder, ivs);
}

// Cyclic-partition `alloc`'s (1-indexed) `dim` by `factor` for conflict-free
// parallel access along that axis.
static void partitionDimCyclic(Operation *alloc, int64_t factor, unsigned dim) {
  MLIRContext *ctx = alloc->getContext();
  auto axis = PartitionAxisAttr::get(ctx, PartitionKindEnum::CyclicPartition,
                                     factor, /*dim=*/dim);
  alloc->setAttr(kPartitionAttr, PartitionAttr::get(ctx, {axis}));
}

// Build an L-lane row-major copy nest over `shape` (last dim must be a multiple
// of L): outer loops over all-but-last dims, an inner contiguous-tile loop of
// `last/L`, and L unrolled lane bodies. `perLane` receives the affine access
// `buf[i0,...,i_{r-2}, ic*L + lane]` and the constant lane index.
static void buildLanedNest(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> shape, int64_t L,
    function_ref<void(OpBuilder &, AffineMap, ValueRange, int64_t)> perLane) {
  OpBuilder::InsertionGuard guard(builder);
  unsigned r = shape.size();
  SmallVector<Value, 4> outerIVs;
  for (unsigned d = 0; d + 1 < r; ++d) {
    auto f = affine::AffineForOp::create(builder, loc, 0, shape[d], 1);
    outerIVs.push_back(f.getInductionVar());
    builder.setInsertionPointToStart(f.getBody());
  }
  auto cFor = affine::AffineForOp::create(builder, loc, 0, shape[r - 1] / L, 1);
  cFor->setAttr(kPipelineIIAttr, builder.getI64IntegerAttr(1));
  builder.setInsertionPointToStart(cFor.getBody());
  for (int64_t lane = 0; lane < L; ++lane) {
    SmallVector<AffineExpr, 4> exprs;
    for (unsigned d = 0; d + 1 < r; ++d)
      exprs.push_back(builder.getAffineDimExpr(d));
    exprs.push_back(builder.getAffineDimExpr(r - 1) * L + lane);
    AffineMap map = AffineMap::get(r, 0, exprs, builder.getContext());
    SmallVector<Value, 4> operands(outerIVs.begin(), outerIVs.end());
    operands.push_back(cFor.getInductionVar());
    perLane(builder, map, operands, lane);
  }
}

// Update a kernel's FunctionType from its current entry-block argument types
// (after some arg types were mutated in place).
static void refreshKernelSignature(KernelOp kernel) {
  Block &entry = kernel.getBody().front();
  SmallVector<Type, 8> inputs(entry.getArgumentTypes().begin(),
                              entry.getArgumentTypes().end());
  kernel.setFunctionType(FunctionType::get(
      kernel.getContext(), inputs, kernel.getFunctionType().getResults()));
}

//===--------------------------------------------------------------------===//
// Full-tensor staging
//===--------------------------------------------------------------------===//

// Producer side: arg #p (write-only memref) becomes a scalar output stream. Its
// stores are redirected into a full-shape buffer; a row-major re-emit nest at
// the end of the body drains the buffer into the stream.
static void streamifyProducerArg(OpBuilder &rewriter, KernelOp kernel,
                                 unsigned p, StreamType streamTy, int64_t L) {
  Block &entry = kernel.getBody().front();
  BlockArgument arg = entry.getArgument(p);
  auto mt = cast<MemRefType>(arg.getType());
  Location loc = kernel.getLoc();

  rewriter.setInsertionPointToStart(&entry);
  auto alloc = memref::AllocOp::create(rewriter, loc, mt);
  alloc->setAttr(
      allo::kAlloSignedAttr,
      rewriter.getStringAttr(operandIsSigned(kernel, p) ? "s" : "u"));
  Value buf = alloc.getResult();
  arg.replaceAllUsesWith(buf);
  arg.setType(streamTy);

  rewriter.setInsertionPoint(entry.getTerminator());
  if (L == 1) {
    buildRowMajorNest(
        rewriter, loc, mt.getShape(), [&](OpBuilder &b, ValueRange ivs) {
          Value v = affine::AffineLoadOp::create(b, loc, buf, ivs);
          StreamPutOp::create(b, loc, arg, ValueRange{}, v);
        });
    return;
  }
  partitionDimCyclic(alloc, L, mt.getRank());
  buildLanedNest(
      rewriter, loc, mt.getShape(), L,
      [&](OpBuilder &b, AffineMap map, ValueRange operands, int64_t lane) {
        Value v = affine::AffineLoadOp::create(b, loc, buf, map, operands);
        Value laneIdx =
            arith::ConstantIndexOp::create(b, loc, lane).getResult();
        StreamPutOp::create(b, loc, arg, ValueRange{laneIdx}, v);
      });
}

// Consumer side: arg #c (read-only memref) becomes a scalar input stream. A
// row-major drain nest at the start of the body fills a full-shape buffer from
// the stream; the existing loads are redirected to read the buffer.
static void streamifyConsumerArg(OpBuilder &rewriter, KernelOp kernel,
                                 unsigned c, StreamType streamTy, int64_t L) {
  Block &entry = kernel.getBody().front();
  BlockArgument arg = entry.getArgument(c);
  auto mt = cast<MemRefType>(arg.getType());
  Location loc = kernel.getLoc();

  rewriter.setInsertionPointToStart(&entry);
  auto alloc = memref::AllocOp::create(rewriter, loc, mt);
  alloc->setAttr(
      allo::kAlloSignedAttr,
      rewriter.getStringAttr(operandIsSigned(kernel, c) ? "s" : "u"));
  Value buf = alloc.getResult();
  arg.replaceAllUsesWith(buf);
  arg.setType(streamTy);

  rewriter.setInsertionPointAfter(alloc); // drain runs before the original body
  if (L == 1) {
    buildRowMajorNest(
        rewriter, loc, mt.getShape(), [&](OpBuilder &b, ValueRange ivs) {
          Value v =
              StreamGetOp::create(b, loc, arg, ArrayRef<Value>{}).getResult();
          affine::AffineStoreOp::create(b, loc, v, buf, ivs);
        });
    return;
  }
  partitionDimCyclic(alloc, L, mt.getRank());
  buildLanedNest(
      rewriter, loc, mt.getShape(), L,
      [&](OpBuilder &b, AffineMap map, ValueRange operands, int64_t lane) {
        Value laneIdx =
            arith::ConstantIndexOp::create(b, loc, lane).getResult();
        Value v = StreamGetOp::create(b, loc, arg, ArrayRef<Value>{laneIdx})
                      .getResult();
        affine::AffineStoreOp::create(b, loc, v, buf, map, operands);
      });
}

//===--------------------------------------------------------------------===//
// Access-pattern analysis
//===--------------------------------------------------------------------===//

// Enclosing affine.for band of `accessOp`, outer-first (band[d] is depth d).
// The walk stops at the first non-affine.for parent, so an affine.if guard
// between the access and its loops shortens the band.
static SmallVector<affine::AffineForOp, 4> enclosingBand(Operation *accessOp) {
  SmallVector<affine::AffineForOp, 4> band;
  Operation *cur = accessOp->getParentOp();
  while (auto forOp = dyn_cast_or_null<affine::AffineForOp>(cur)) {
    band.push_back(forOp);
    cur = forOp->getParentOp();
  }
  std::reverse(band.begin(), band.end());
  return band;
}

// True iff `accessOp` touches the WHOLE memref in row-major order: a band of
// rank == memref.rank, each loop [0, dim_d) step 1, and an identity access map
// (tensor dim d indexed by the depth-d IV). This is the in-order condition
// under which the boundary is passthrough-safe, producer put sequence ==
// consumer get sequence == row-major, so no reorder buffer is needed.
static bool isCanonicalRowMajorFull(Operation *accessOp, MemRefType mt) {
  unsigned rank = mt.getRank();
  SmallVector<affine::AffineForOp, 4> band = enclosingBand(accessOp);
  if (band.size() != rank)
    return false;
  for (unsigned d = 0; d < rank; ++d) {
    affine::AffineForOp f = band[d];
    if (!f.hasConstantBounds() || f.getConstantLowerBound() != 0 ||
        f.getStepAsInt() != 1 || f.getConstantUpperBound() != mt.getDimSize(d))
      return false;
  }
  affine::MemRefAccess access(accessOp);
  affine::AffineValueMap avm;
  access.getAccessMap(&avm);
  avm.composeSimplifyAndCanonicalize();
  AffineMap map = avm.getAffineMap();
  if (map.getNumResults() != rank)
    return false;
  ArrayRef<Value> operands = avm.getOperands();
  for (unsigned d = 0; d < rank; ++d) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d));
    if (!dimExpr || dimExpr.getPosition() >= operands.size())
      return false;
    if (operands[dimExpr.getPosition()] != band[d].getInductionVar())
      return false;
  }
  return true;
}

// The single affine access of `arg` (exactly one), or null otherwise.
template <typename AccessOp>
static AccessOp uniqueAffineAccess(BlockArgument arg) {
  AccessOp found;
  for (OpOperand &use : arg.getUses())
    if (auto op = dyn_cast<AccessOp>(use.getOwner())) {
      if (found)
        return AccessOp();
      found = op;
    }
  return found;
}

//===--------------------------------------------------------------------===//
// Passthrough (in place, no buffer)
//===--------------------------------------------------------------------===//

// The static last-dim offset (0..L-1) an unrolled access carries relative to
// its now-strided innermost IV, i.e. the parallel lane it belongs to. After
// unrolling the contiguous loop by L the canonical map is `IV` (lane 0) or
// `IV + k` (lane k).
static int64_t laneOffsetOf(Operation *access, unsigned rank) {
  affine::MemRefAccess acc(access);
  affine::AffineValueMap avm;
  acc.getAccessMap(&avm);
  avm.composeSimplifyAndCanonicalize();
  AffineExpr last = avm.getAffineMap().getResult(rank - 1);
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(last))
    if (bin.getKind() == AffineExprKind::Add)
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        return c.getValue();
  return 0;
}

// Producer passthrough: the row-major store streams out in place, no buffer.
// Scalar (L==1) rewrites the lone store to a put; laned (L>1) unrolls the
// innermost (contiguous) loop by L so L elements are put per cycle, one to each
// parallel lane FIFO.
static void passthroughProducerArg(OpBuilder &b, BlockArgument arg,
                                   affine::AffineStoreOp store,
                                   StreamType streamTy, int64_t L) {
  if (L == 1) {
    b.setInsertionPoint(store);
    arg.setType(streamTy);
    StreamPutOp::create(b, store.getLoc(), arg, ValueRange{},
                        store.getValueToStore());
    store.erase();
    return;
  }
  unsigned rank = cast<MemRefType>(arg.getType()).getRank();
  auto inner = cast<affine::AffineForOp>(store->getParentOp());
  LogicalResult unrolled = affine::loopUnrollByFactor(inner, L);
  assert(succeeded(unrolled) && "lane unroll must succeed: L divides the loop");
  (void)unrolled;
  SmallVector<affine::AffineStoreOp, 8> stores;
  inner.getBody()->walk([&](affine::AffineStoreOp s) {
    if (s.getMemRef() == arg)
      stores.push_back(s);
  });
  arg.setType(streamTy);
  for (affine::AffineStoreOp s : stores) {
    b.setInsertionPoint(s);
    Value laneIdx =
        arith::ConstantIndexOp::create(b, s.getLoc(), laneOffsetOf(s, rank));
    StreamPutOp::create(b, s.getLoc(), arg, ValueRange{laneIdx},
                        s.getValueToStore());
    s.erase();
  }
  inner->setAttr(kPipelineIIAttr, b.getI64IntegerAttr(1));
}

// Consumer passthrough: the row-major load streams in place, no buffer.
// Symmetric to the producer: laned unrolls the contiguous loop by L and reads
// L lane FIFOs per cycle.
static void passthroughConsumerArg(OpBuilder &b, BlockArgument arg,
                                   affine::AffineLoadOp load,
                                   StreamType streamTy, int64_t L) {
  if (L == 1) {
    b.setInsertionPoint(load);
    arg.setType(streamTy);
    Value v = StreamGetOp::create(b, load.getLoc(), arg, ArrayRef<Value>{})
                  .getResult();
    load.getResult().replaceAllUsesWith(v);
    load.erase();
    return;
  }
  unsigned rank = cast<MemRefType>(arg.getType()).getRank();
  auto inner = cast<affine::AffineForOp>(load->getParentOp());
  LogicalResult unrolled = affine::loopUnrollByFactor(inner, L);
  assert(succeeded(unrolled) && "lane unroll must succeed: L divides the loop");
  (void)unrolled;
  SmallVector<affine::AffineLoadOp, 8> loads;
  inner.getBody()->walk([&](affine::AffineLoadOp ld) {
    if (ld.getMemRef() == arg)
      loads.push_back(ld);
  });
  arg.setType(streamTy);
  for (affine::AffineLoadOp ld : loads) {
    b.setInsertionPoint(ld);
    Value laneIdx =
        arith::ConstantIndexOp::create(b, ld.getLoc(), laneOffsetOf(ld, rank));
    Value v = StreamGetOp::create(b, ld.getLoc(), arg, ArrayRef<Value>{laneIdx})
                  .getResult();
    ld.getResult().replaceAllUsesWith(v);
    ld.erase();
  }
  inner->setAttr(kPipelineIIAttr, b.getI64IntegerAttr(1));
}

//===--------------------------------------------------------------------===//
// Per-side dispatch
//===--------------------------------------------------------------------===//

// Convert producer arg #p (write-only memref) into a stream output: passthrough
// (in place, widened to L lanes) when it writes the whole tensor once
// row-major, else stage a reorder buffer.
static void convertProducerSide(OpBuilder &b, KernelOp kernel, unsigned p,
                                MemRefType mt, StreamType streamTy, int64_t L) {
  BlockArgument arg = kernel.getBody().front().getArgument(p);
  auto store = uniqueAffineAccess<affine::AffineStoreOp>(arg);
  if (store && isCanonicalRowMajorFull(store, mt))
    passthroughProducerArg(b, arg, store, streamTy, L);
  else
    streamifyProducerArg(b, kernel, p, streamTy, L);
}

//===--------------------------------------------------------------------===//
// Windowed staging (circular line buffer)
//===--------------------------------------------------------------------===//

// Returns (K, outer) if the consumer's reads of `arg` form a valid sliding
// window along dim 0: every read is `arg[i0 + d, <inner not using i0>]` with
// the outer loop i0 streaming dim 0 ([0,T) step 1, T == dim0 - K + 1 so the
// stream drains exactly), d a constant in [0, K), K = max d + 1. Only K rows
// then need to be buffered instead of the whole tensor.
struct WindowInfo {
  int64_t k;                 // window height, rows to buffer
  affine::AffineForOp outer; // the streamed (dim-0) loop
  bool
      vertical; // identity inner access -> the fill fuses into the compute body
};
static std::optional<WindowInfo> detectSlidingWindow(BlockArgument arg,
                                                     MemRefType mt) {
  unsigned rank = mt.getRank();
  if (rank == 0)
    return std::nullopt;
  SmallVector<affine::AffineLoadOp, 4> loads;
  for (Operation *user : arg.getUsers()) {
    auto ld = dyn_cast<affine::AffineLoadOp>(user);
    if (!ld)
      return std::nullopt; // a non-load use is not analyzable
    loads.push_back(ld);
  }
  if (loads.empty())
    return std::nullopt;
  SmallVector<affine::AffineForOp, 4> band0 = enclosingBand(loads[0]);
  if (band0.empty())
    return std::nullopt;
  affine::AffineForOp outer = band0[0];
  if (!outer.hasConstantBounds() || outer.getConstantLowerBound() != 0 ||
      outer.getStepAsInt() != 1)
    return std::nullopt;
  Value i0 = outer.getInductionVar();
  int64_t maxOff = -1;
  bool vertical = true; // every read's inner access is identity (no offsets)
  for (affine::AffineLoadOp ld : loads) {
    SmallVector<affine::AffineForOp, 4> band = enclosingBand(ld);
    if (band.empty() || band[0] != outer)
      return std::nullopt;
    affine::MemRefAccess access(ld);
    affine::AffineValueMap avm;
    access.getAccessMap(&avm);
    avm.composeSimplifyAndCanonicalize();
    AffineMap map = avm.getAffineMap();
    if (map.getNumResults() != rank)
      return std::nullopt;
    ArrayRef<Value> operands = avm.getOperands();
    // dim 0 access must be `i0 + d`, d a non-negative constant.
    AffineExpr e0 = map.getResult(0);
    int64_t d = 0;
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(e0)) {
      auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (bin.getKind() != AffineExprKind::Add || !cst)
        return std::nullopt;
      d = cst.getValue();
      e0 = bin.getLHS();
    }
    auto dim0 = dyn_cast<AffineDimExpr>(e0);
    if (!dim0 || dim0.getPosition() >= operands.size() ||
        operands[dim0.getPosition()] != i0 || d < 0)
      return std::nullopt;
    maxOff = std::max(maxOff, d);
    // inner dims must not depend on the streamed index i0.
    for (unsigned r = 1; r < rank; ++r)
      if (map.getResult(r).isFunctionOfDim(dim0.getPosition()))
        return std::nullopt;
    // A purely vertical window lets the newest-row fill fuse into the compute
    // body: each read of the newest row is exactly the just-filled element. A
    // column offset (e.g. a 2D conv) would read an unfilled cell, so fill and
    // compute stay separate loops.
    if (band.size() != rank)
      vertical = false;
    else
      for (unsigned r = 1; r < rank; ++r) {
        auto dim = dyn_cast<AffineDimExpr>(map.getResult(r));
        if (!dim || dim.getPosition() >= operands.size() ||
            operands[dim.getPosition()] != band[r].getInductionVar())
          vertical = false;
      }
  }
  int64_t K = maxOff + 1;
  if (K < 1 || K > mt.getDimSize(0) ||
      outer.getConstantUpperBound() != mt.getDimSize(0) - K + 1)
    return std::nullopt;
  return WindowInfo{K, outer, vertical};
}

// Fill one streamed row into `cbuf`: a nest over the inner dims storing a get()
// per point at `cbuf[rowExpr, inner...]`. `rowExpr` is over `rowOperands`; the
// inner dims use fresh IVs appended after them.
static void buildRowFill(OpBuilder &b, Location loc, Value cbuf,
                         Value streamArg, MemRefType mt, AffineExpr rowExpr,
                         ValueRange rowOperands) {
  OpBuilder::InsertionGuard g(b);
  unsigned rank = mt.getRank();
  SmallVector<Value, 4> innerIVs;
  affine::AffineForOp innermost;
  for (unsigned d = 1; d < rank; ++d) {
    auto f = affine::AffineForOp::create(b, loc, 0, mt.getDimSize(d), 1);
    innerIVs.push_back(f.getInductionVar());
    innermost = f;
    b.setInsertionPointToStart(f.getBody());
  }
  if (innermost)
    innermost->setAttr(kPipelineIIAttr, b.getI64IntegerAttr(1));
  Value v =
      StreamGetOp::create(b, loc, streamArg, ArrayRef<Value>{}).getResult();
  unsigned nRow = rowOperands.size();
  SmallVector<AffineExpr, 4> exprs{rowExpr};
  for (unsigned d = 0; d < innerIVs.size(); ++d)
    exprs.push_back(b.getAffineDimExpr(nRow + d));
  AffineMap map =
      AffineMap::get(nRow + innerIVs.size(), 0, exprs, b.getContext());
  SmallVector<Value, 4> operands(rowOperands.begin(), rowOperands.end());
  operands.append(innerIVs.begin(), innerIVs.end());
  affine::AffineStoreOp::create(b, loc, v, cbuf, map, operands);
}

// Consumer side, windowed: arg #c becomes an input stream feeding a K-row
// circular line buffer. A warmup fills rows 0..K-2; each output row i streams
// in the newest row (i+K-1) before computing, and every read `arg[i0+d, inner]`
// is redirected to `cbuf[(i0+d) mod K, inner]`.
static void windowedConsumerArg(OpBuilder &b, KernelOp kernel, unsigned c,
                                MemRefType mt, StreamType streamTy,
                                const WindowInfo &win) {
  Block &entry = kernel.getBody().front();
  BlockArgument arg = entry.getArgument(c);
  Location loc = kernel.getLoc();
  MLIRContext *ctx = b.getContext();
  unsigned rank = mt.getRank();
  int64_t K = win.k;
  affine::AffineForOp outer = win.outer;

  SmallVector<int64_t, 4> cshape{K};
  for (unsigned d = 1; d < rank; ++d)
    cshape.push_back(mt.getDimSize(d));
  b.setInsertionPointToStart(&entry);
  auto alloc = memref::AllocOp::create(
      b, loc, MemRefType::get(cshape, mt.getElementType()));
  alloc->setAttr(allo::kAlloSignedAttr,
                 b.getStringAttr(operandIsSigned(kernel, c) ? "s" : "u"));
  Value cbuf = alloc.getResult();
  if (K > 1)
    partitionDimCyclic(alloc, K, /*dim=*/1); // separate the K window rows
  arg.setType(streamTy);

  SmallVector<affine::AffineLoadOp, 4> loads;
  for (Operation *user : arg.getUsers())
    if (auto ld = dyn_cast<affine::AffineLoadOp>(user))
      loads.push_back(ld);
  // The loop band (captured before the loads are erased) drives the fused fill.
  SmallVector<affine::AffineForOp, 4> band = enclosingBand(loads.front());
  for (affine::AffineLoadOp ld : loads) {
    AffineMap m = ld.getAffineMap();
    SmallVector<AffineExpr, 4> results(m.getResults().begin(),
                                       m.getResults().end());
    results[0] = results[0] % K;
    AffineMap nm =
        AffineMap::get(m.getNumDims(), m.getNumSymbols(), results, ctx);
    b.setInsertionPoint(ld);
    Value v =
        affine::AffineLoadOp::create(b, loc, cbuf, nm, ld.getMapOperands());
    ld.getResult().replaceAllUsesWith(v);
    ld.erase();
  }

  // Warm up rows 0..K-2 before the first output row.
  b.setInsertionPoint(outer);
  if (K > 1) {
    auto pre = affine::AffineForOp::create(b, loc, 0, K - 1, 1);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(pre.getBody());
    buildRowFill(b, loc, cbuf, arg, mt, b.getAffineDimExpr(0),
                 ValueRange{pre.getInductionVar()});
  }

  AffineExpr slot = (b.getAffineDimExpr(0) + (K - 1)) % K;
  if (win.vertical) {
    // Fused: emit the get() at the start of the innermost loop, so each element
    // is streamed in right before it is read, one pass over the inner dims.
    b.setInsertionPointToStart(band.back().getBody());
    Value v = StreamGetOp::create(b, loc, arg, ArrayRef<Value>{}).getResult();
    SmallVector<AffineExpr, 4> exprs{slot};
    SmallVector<Value, 4> ivs{outer.getInductionVar()};
    for (unsigned d = 1; d < rank; ++d) {
      exprs.push_back(b.getAffineDimExpr(d));
      ivs.push_back(band[d].getInductionVar());
    }
    affine::AffineStoreOp::create(b, loc, v, cbuf,
                                  AffineMap::get(rank, 0, exprs, ctx), ivs);
  } else {
    // Each output row i streams in the newest needed row (i + K - 1) first.
    b.setInsertionPointToStart(outer.getBody());
    buildRowFill(b, loc, cbuf, arg, mt, slot,
                 ValueRange{outer.getInductionVar()});
  }
}

// Convert consumer arg #c (read-only memref) into a stream input, symmetric to
// convertProducerSide.
static void convertConsumerSide(OpBuilder &b, KernelOp kernel, unsigned c,
                                MemRefType mt, StreamType streamTy, int64_t L) {
  BlockArgument arg = kernel.getBody().front().getArgument(c);
  auto load = uniqueAffineAccess<affine::AffineLoadOp>(arg);
  if (load && isCanonicalRowMajorFull(load, mt)) {
    passthroughConsumerArg(b, arg, load, streamTy, L);
  } else if (L == 1) {
    // A bounded sliding window needs only a K-row line buffer.
    if (auto win = detectSlidingWindow(arg, mt))
      windowedConsumerArg(b, kernel, c, mt, streamTy, *win);
    else
      streamifyConsumerArg(b, kernel, c, streamTy, L);
  } else {
    streamifyConsumerArg(b, kernel, c, streamTy, L);
  }
}

//===--------------------------------------------------------------------===//
// Fan-out (tee)
//===--------------------------------------------------------------------===//

// Body of a `tee`: read each element of the input stream once and broadcast it
// to every output stream, a buffer-free passthrough fan-out. The stream get/put
// order IS the row-major contract, so the loops only set the trip count; with
// L>1 each inner iteration moves L lanes.
static void buildBroadcastBody(OpBuilder &b, Location loc,
                               ArrayRef<int64_t> shape, int64_t L, Value in,
                               ValueRange outs) {
  if (L == 1) {
    buildRowMajorNest(b, loc, shape, [&](OpBuilder &bb, ValueRange) {
      Value v = StreamGetOp::create(bb, loc, in, ArrayRef<Value>{}).getResult();
      for (Value out : outs)
        StreamPutOp::create(bb, loc, out, ValueRange{}, v);
    });
    return;
  }
  // Laned: the broadcast needs only the lane index, not the per-lane map.
  buildLanedNest(
      b, loc, shape, L,
      [&](OpBuilder &bb, AffineMap, ValueRange, int64_t lane) {
        Value li = arith::ConstantIndexOp::create(bb, loc, lane);
        Value v =
            StreamGetOp::create(bb, loc, in, ArrayRef<Value>{li}).getResult();
        for (Value out : outs)
          StreamPutOp::create(bb, loc, out, ValueRange{li}, v);
      });
}

// Create a private dataflow kernel `name` with `numStreams` stream arguments,
// fill its body via `buildBody(entry)`, terminate it, and insert it into the
// module with a uniquified name. Shared by the tee and the merge.
static KernelOp buildStreamKernel(OpBuilder &b, Operation *moduleOp,
                                  Location loc, StringRef name,
                                  StreamType streamTy, unsigned numStreams,
                                  bool isSigned,
                                  function_ref<void(Block *)> buildBody) {
  MLIRContext *ctx = b.getContext();
  SmallVector<Type, 8> inputs(numStreams, streamTy);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&moduleOp->getRegion(0).front());
  auto kernel = KernelOp::create(
      b, loc, b.getStringAttr(name),
      TypeAttr::get(FunctionType::get(ctx, inputs, {})),
      b.getStringAttr("private"),
      /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, b.getDenseI32ArrayAttr({}));
  // All args carry the boundary payload, so the marker is a uniform run of one
  // char.
  kernel->setAttr(
      allo::kAlloSignedAttr,
      b.getStringAttr(std::string(numStreams, isSigned ? 's' : 'u')));
  SymbolTable(moduleOp).insert(kernel); // uniquify name among existing kernels
  SmallVector<Location, 8> argLocs(numStreams, loc);
  Block *entry =
      b.createBlock(&kernel.getBody(), kernel.getBody().end(), inputs, argLocs);
  buildBody(entry);
  b.setInsertionPointToEnd(entry);
  ReturnOp::create(b, loc, ValueRange{});
  return kernel;
}

// Create a `tee` kernel (1 input stream -> N output streams) broadcasting the
// boundary, insert it into the module with a uniquified name, and return it.
static KernelOp buildTeeKernel(OpBuilder &b, Operation *moduleOp, Location loc,
                               ArrayRef<int64_t> shape, StreamType streamTy,
                               unsigned n, int64_t L, bool isSigned) {
  return buildStreamKernel(b, moduleOp, loc, "streamline_tee", streamTy, n + 1,
                           isSigned, [&](Block *entry) {
                             b.setInsertionPointToStart(entry);
                             buildBroadcastBody(
                                 b, loc, shape, L, entry->getArgument(0),
                                 entry->getArguments().drop_front());
                           });
}

//===--------------------------------------------------------------------===//
// Fan-in (merge)
//===--------------------------------------------------------------------===//

// If `store` writes a contiguous row-major block of `mt` (inner dims full and
// identity, dim 0 a sub-range [c, c+Nk) via the access `iv0 + c`), return the
// block's (offset, length) in elements. This is the fan-in producer contract:
// each producer fills one such block, drained in order to rebuild the tensor.
static std::optional<std::pair<int64_t, int64_t>>
contiguousOuterBlock(Operation *store, MemRefType mt) {
  unsigned rank = mt.getRank();
  SmallVector<affine::AffineForOp, 4> band = enclosingBand(store);
  if (band.size() != rank || rank == 0)
    return std::nullopt;
  affine::MemRefAccess access(store);
  affine::AffineValueMap avm;
  access.getAccessMap(&avm);
  avm.composeSimplifyAndCanonicalize();
  AffineMap map = avm.getAffineMap();
  if (map.getNumResults() != rank)
    return std::nullopt;
  ArrayRef<Value> operands = avm.getOperands();
  auto isIdentityDim = [&](unsigned d, AffineExpr e) {
    auto dim = dyn_cast<AffineDimExpr>(e);
    return dim && dim.getPosition() < operands.size() &&
           operands[dim.getPosition()] == band[d].getInductionVar();
  };
  // Inner dims (1..r-1): full [0, dim) and identity-indexed.
  int64_t inner = 1;
  for (unsigned d = 1; d < rank; ++d) {
    affine::AffineForOp f = band[d];
    if (!f.hasConstantBounds() || f.getConstantLowerBound() != 0 ||
        f.getStepAsInt() != 1 || f.getConstantUpperBound() != mt.getDimSize(d))
      return std::nullopt;
    if (!isIdentityDim(d, map.getResult(d)))
      return std::nullopt;
    inner *= mt.getDimSize(d);
  }
  // Dim 0: loop [0, Nk) step 1, access expr `iv0 + c` (c >= 0, block in
  // bounds).
  affine::AffineForOp f0 = band[0];
  if (!f0.hasConstantBounds() || f0.getConstantLowerBound() != 0 ||
      f0.getStepAsInt() != 1)
    return std::nullopt;
  int64_t Nk = f0.getConstantUpperBound();
  AffineExpr e0 = map.getResult(0);
  int64_t c = 0;
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e0)) {
    auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (bin.getKind() != AffineExprKind::Add || !cst)
      return std::nullopt;
    c = cst.getValue();
    e0 = bin.getLHS();
  }
  if (!isIdentityDim(0, e0) || c < 0 || c + Nk > mt.getDimSize(0))
    return std::nullopt;
  return std::make_pair(c * inner, Nk * inner);
}

// Create a `merge` kernel (N block streams -> 1 output stream): drain input k
// for `lens[k]` elements into the output, so the blocks concatenate in order to
// the tensor's row-major sequence.
static KernelOp buildMergeKernel(OpBuilder &b, Operation *moduleOp,
                                 Location loc, StreamType streamTy,
                                 ArrayRef<int64_t> lens, bool isSigned) {
  unsigned n = lens.size();
  return buildStreamKernel(
      b, moduleOp, loc, "streamline_merge", streamTy, n + 1, isSigned,
      [&](Block *entry) {
        Value out = entry->getArgument(n);
        b.setInsertionPointToStart(entry);
        for (unsigned k = 0; k < n; ++k) {
          auto f = affine::AffineForOp::create(b, loc, 0, lens[k], 1);
          f->setAttr(kPipelineIIAttr, b.getI64IntegerAttr(1));
          {
            OpBuilder::InsertionGuard fg(b);
            b.setInsertionPointToStart(f.getBody());
            Value v = StreamGetOp::create(b, loc, entry->getArgument(k),
                                          ArrayRef<Value>{})
                          .getResult();
            StreamPutOp::create(b, loc, out, ValueRange{}, v);
          }
          b.setInsertionPointAfter(f);
        }
      });
}

//===--------------------------------------------------------------------===//
// Reconvergence (deadlock) analysis
//===--------------------------------------------------------------------===//

// Direction of a stream-typed invoke operand: does the callee PUT or GET
// through the matching arg? Orients the dataflow graph for the reconvergence
// analysis.
enum class StreamDir { None, Producer, Consumer };
static StreamDir streamArgDir(InvokeOp invoke, unsigned idx) {
  auto kernel = SymbolTable::lookupNearestSymbolFrom<KernelOp>(
      invoke, invoke.getCalleeAttr());
  if (!kernel)
    return StreamDir::None;
  for (Operation *user : kernel.getBody().front().getArgument(idx).getUsers()) {
    if (isa<StreamPutOp>(user))
      return StreamDir::Producer;
    if (isa<StreamGetOp>(user))
      return StreamDir::Consumer;
  }
  return StreamDir::None;
}

} // namespace

DiagnosedSilenceableFailure
transform::StreamlineOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  // The Python frontend gives every call site a unique callee copy
  // ({hierarchy}.{counter}), so each invoke resolves to exactly one kernel.
  struct Node {
    InvokeOp invoke;
    KernelOp kernel;
  };
  auto resolve = [&](Value handle, StringRef which,
                     Node &out) -> DiagnosedSilenceableFailure {
    auto ops = llvm::to_vector(state.getPayloadOps(handle));
    if (ops.size() != 1)
      return emitSilenceableError()
             << which << " must resolve to exactly one allo.invoke";
    auto invoke = dyn_cast<InvokeOp>(ops[0]);
    if (!invoke)
      return emitSilenceableError() << which << " must be an allo.invoke";
    auto kernel = SymbolTable::lookupNearestSymbolFrom<KernelOp>(
        invoke, invoke.getCalleeAttr());
    if (!kernel)
      return emitSilenceableError()
             << "could not resolve " << which << " callee";
    out = {invoke, kernel};
    return DiagnosedSilenceableFailure::success();
  };

  SmallVector<Node, 4> producers, consumers;
  for (Value h : getProducers()) {
    Node n;
    if (auto f = resolve(h, "producer", n); !f.succeeded())
      return f;
    producers.push_back(n);
  }
  for (Value h : getConsumers()) {
    Node n;
    if (auto f = resolve(h, "consumer", n); !f.succeeded())
      return f;
    consumers.push_back(n);
  }
  if (producers.empty() || consumers.empty())
    return emitSilenceableError()
           << "streamline needs at least one producer and one consumer";

  Block *parent = producers[0].invoke->getBlock();
  for (Node &n : llvm::concat<Node>(producers, consumers))
    if (n.invoke->getBlock() != parent)
      return emitSilenceableError()
             << "all kernels must be invoked in the same region";
  for (Node &p : producers)
    for (Node &c : consumers)
      if (p.kernel == c.kernel)
        return emitSilenceableError()
               << "a kernel cannot be both producer and consumer";

  MLIRContext *ctx = producers[0].kernel.getContext();
  Operation *moduleOp = SymbolTable::getNearestSymbolTable(producers[0].kernel);
  int64_t depth = getDepth();
  Location loc = producers[0].invoke.getLoc();

  // A *boundary* is a memref value shared at the invoke operands: written by
  // some producers (WriteOnly) and read by some consumers (ReadOnly).
  using Refs = SmallVector<std::pair<unsigned, unsigned>, 2>;
  llvm::DenseMap<Value, Refs> writers, readers;
  for (unsigned pi = 0; pi < producers.size(); ++pi) {
    Block &pe = producers[pi].kernel.getBody().front();
    for (unsigned a = 0; a < pe.getNumArguments(); ++a)
      if (isa<MemRefType>(pe.getArgument(a).getType()) &&
          classifyArg(pe.getArgument(a)) == ArgKind::WriteOnly)
        writers[producers[pi].invoke.getOperand(a)].push_back({pi, a});
  }
  for (unsigned ci = 0; ci < consumers.size(); ++ci) {
    Block &ce = consumers[ci].kernel.getBody().front();
    for (unsigned a = 0; a < ce.getNumArguments(); ++a)
      if (isa<MemRefType>(ce.getArgument(a).getType()) &&
          classifyArg(ce.getArgument(a)) == ArgKind::ReadOnly)
        readers[consumers[ci].invoke.getOperand(a)].push_back({ci, a});
  }
  // Boundary values in a stable (producer arg) order.
  SmallVector<Value, 4> boundaries;
  llvm::SmallPtrSet<Value, 4> seen;
  for (Node &p : producers)
    for (unsigned a = 0; a < p.invoke.getNumOperands(); ++a) {
      Value v = p.invoke.getOperand(a);
      if (writers.count(v) && readers.count(v) && seen.insert(v).second)
        boundaries.push_back(v);
    }

  // Tag each boundary stream with its element count so the reconvergence check
  // below can recommend a deadlock-safe depth (worst-case skew = whole tensor).
  auto makeStream = [&](StreamType ty, int64_t elems, bool isSigned) {
    rewriter.setInsertionPointToStart(parent);
    auto op = StreamCreateOp::create(rewriter, loc, ty);
    op->setAttr("allo.fifo.elems", rewriter.getI64IntegerAttr(elems));
    // Carry the payload signedness so the emitter renders a FIFO element type
    // matching the converted producer/consumer parameters.
    op->setAttr(allo::kAlloSignedAttr,
                rewriter.getStringAttr(isSigned ? "s" : "u"));
    return op.getResult();
  };

  unsigned converted = 0;
  for (Value boundary : boundaries) {
    Refs &w = writers[boundary], &r = readers[boundary];
    auto mt = cast<MemRefType>(producers[w[0].first]
                                   .kernel.getBody()
                                   .front()
                                   .getArgument(w[0].second)
                                   .getType());
    if (!mt.hasStaticShape())
      return emitSilenceableError()
             << "streamline boundary has a non-static footprint";
    // The boundary's element signedness comes from the producer kernel's
    // marker; every stream and kernel derived from it inherits it.
    bool sgn = operandIsSigned(producers[w[0].first].kernel, w[0].second);

    // lanes=L widens the boundary to L parallel FIFOs moving L elements/cycle.
    // It requires L to divide the tensor's contiguous dim, and fan-in is scalar
    // only.
    int64_t L = getLanes();
    bool fanIn = w.size() > 1;
    if (L > 1 && (fanIn || mt.getShape().back() % L != 0)) {
      if (!fanIn)
        emitWarning() << "lanes=" << L
                      << " does not divide the last dim of the "
                      << "boundary; using scalar";
      L = 1;
    }
    auto streamTy =
        StreamType::get(ctx, mt.getElementType(), depth,
                        L > 1 ? ArrayRef<int64_t>{L} : ArrayRef<int64_t>{});
    int64_t elems = mt.getNumElements();

    if (w.size() == 1 && r.size() == 1) {
      // Direct hand-off: producer streams straight into the consumer's FIFO.
      auto [pi, pa] = w[0];
      auto [ci, ca] = r[0];
      Value s = makeStream(streamTy, elems, sgn);
      convertProducerSide(rewriter, producers[pi].kernel, pa, mt, streamTy, L);
      convertConsumerSide(rewriter, consumers[ci].kernel, ca, mt, streamTy, L);
      producers[pi].invoke.setOperand(pa, s);
      consumers[ci].invoke.setOperand(ca, s);
    } else if (w.size() == 1) {
      // Fan-out: producer writes one FIFO, a generated tee broadcasts it to
      // each consumer's FIFO (a stream can't be read twice).
      auto [pi, pa] = w[0];
      SmallVector<Value, 4> consumerStreams;
      for (auto [ci, ca] : r) {
        Value s = makeStream(streamTy, elems, sgn);
        convertConsumerSide(rewriter, consumers[ci].kernel, ca, mt, streamTy,
                            L);
        consumers[ci].invoke.setOperand(ca, s);
        consumerStreams.push_back(s);
      }
      Value prodStream = makeStream(streamTy, elems, sgn);
      convertProducerSide(rewriter, producers[pi].kernel, pa, mt, streamTy, L);
      producers[pi].invoke.setOperand(pa, prodStream);
      auto tee = buildTeeKernel(rewriter, moduleOp, loc, mt.getShape(),
                                streamTy, r.size(), L, sgn);
      rewriter.setInsertionPointAfter(producers[pi].invoke);
      SmallVector<Value, 5> teeOps{prodStream};
      teeOps.append(consumerStreams.begin(), consumerStreams.end());
      InvokeOp::create(rewriter, loc, tee, teeOps);
    } else if (r.size() == 1) {
      // Fan-in: each producer must fill a disjoint contiguous row-major block;
      // a generated merge concatenates the blocks in order into the consumer.
      auto [ci, ca] = r[0];
      struct Blk {
        unsigned pi, pa;
        int64_t off, len;
        affine::AffineStoreOp store;
      };
      SmallVector<Blk, 4> blks;
      bool ok = true;
      for (auto [pi, pa] : w) {
        auto store = uniqueAffineAccess<affine::AffineStoreOp>(
            producers[pi].kernel.getBody().front().getArgument(pa));
        auto blk = store ? contiguousOuterBlock(store, mt) : std::nullopt;
        if (!blk) {
          ok = false;
          break;
        }
        blks.push_back({pi, pa, blk->first, blk->second, store});
      }
      if (!ok)
        return emitSilenceableError()
               << "fan-in producer does not write a contiguous row-major block "
                  "of the boundary tensor";
      llvm::sort(blks,
                 [](const Blk &x, const Blk &y) { return x.off < y.off; });
      int64_t expect = 0;
      for (Blk &bk : blks) {
        if (bk.off != expect) {
          expect = -1;
          break;
        }
        expect += bk.len;
      }
      if (expect != elems)
        return emitSilenceableError()
               << "fan-in producer blocks do not tile the boundary tensor";
      SmallVector<Value, 4> blockStreams;
      SmallVector<int64_t, 4> lens;
      for (Blk &bk : blks) {
        Value s = makeStream(streamTy, bk.len, sgn);
        BlockArgument arg =
            producers[bk.pi].kernel.getBody().front().getArgument(bk.pa);
        passthroughProducerArg(rewriter, arg, bk.store, streamTy, /*L=*/1);
        producers[bk.pi].invoke.setOperand(bk.pa, s);
        blockStreams.push_back(s);
        lens.push_back(bk.len);
      }
      Value merged = makeStream(streamTy, elems, sgn);
      auto merge =
          buildMergeKernel(rewriter, moduleOp, loc, streamTy, lens, sgn);
      rewriter.setInsertionPoint(consumers[ci].invoke);
      SmallVector<Value, 5> mergeOps(blockStreams.begin(), blockStreams.end());
      mergeOps.push_back(merged);
      InvokeOp::create(rewriter, loc, merge, mergeOps);
      convertConsumerSide(rewriter, consumers[ci].kernel, ca, mt, streamTy, L);
      consumers[ci].invoke.setOperand(ca, merged);
    } else {
      return emitSilenceableError()
             << "many-to-many boundary (multiple producers and consumers) is "
                "unsupported";
    }
    ++converted;
  }

  for (Node &p : producers)
    refreshKernelSignature(p.kernel);
  for (Node &c : consumers)
    refreshKernelSignature(c.kernel);

  if (converted == 0)
    return emitSilenceableError()
           << "streamline: no convertible memory boundary found between the "
              "given producers and consumers";

  // Deadlock safety: streamline may now have built a reconvergent fork/join,
  // where one value reaches a join both directly and through a longer path. The
  // short branch's FIFO must hold the latency skew or the dataflow deadlocks.
  // The skew is not measurable at the IR level, so it is over-bounded by the
  // whole tensor. The check runs over the whole parent dataflow graph, so it
  // fires on the streamline call that closes the diamond.
  llvm::DenseMap<Value, InvokeOp> producerOf;
  SmallVector<InvokeOp, 8> invokes;
  for (Operation &op : *parent)
    if (auto inv = dyn_cast<InvokeOp>(&op)) {
      invokes.push_back(inv);
      for (unsigned i = 0; i < inv.getNumOperands(); ++i)
        if (isa<StreamType>(inv.getOperand(i).getType()) &&
            streamArgDir(inv, i) == StreamDir::Producer)
          producerOf[inv.getOperand(i)] = inv;
    }

  // Invokes reachable upstream of `start` via stream edges, keyed by min hops.
  auto ancestors = [&](InvokeOp start) {
    llvm::DenseMap<Operation *, int> dist{{start.getOperation(), 0}};
    SmallVector<std::pair<InvokeOp, int>, 8> work{{start, 0}};
    while (!work.empty()) {
      auto [inv, d] = work.pop_back_val();
      for (unsigned i = 0; i < inv.getNumOperands(); ++i) {
        Value v = inv.getOperand(i);
        if (!isa<StreamType>(v.getType()) ||
            streamArgDir(inv, i) != StreamDir::Consumer)
          continue;
        auto pit = producerOf.find(v);
        if (pit == producerOf.end())
          continue;
        int nd = d + 1;
        auto dit = dist.find(pit->second.getOperation());
        if (dit == dist.end() || nd < dit->second) {
          dist[pit->second.getOperation()] = nd;
          work.push_back({pit->second, nd});
        }
      }
    }
    return dist;
  };

  for (InvokeOp join : invokes) {
    SmallVector<Value, 4> ins;
    for (unsigned i = 0; i < join.getNumOperands(); ++i)
      if (isa<StreamType>(join.getOperand(i).getType()) &&
          streamArgDir(join, i) == StreamDir::Consumer)
        ins.push_back(join.getOperand(i));
    if (ins.size() < 2)
      continue;
    for (unsigned a = 0; a < ins.size(); ++a)
      for (unsigned b = a + 1; b < ins.size(); ++b) {
        auto pa = producerOf.find(ins[a]), pb = producerOf.find(ins[b]);
        if (pa == producerOf.end() || pb == producerOf.end())
          continue;
        auto da = ancestors(pa->second), db = ancestors(pb->second);
        // A common ancestor with unequal path lengths is a fork with skew.
        Operation *fork = nullptr;
        bool aShort = true;
        for (auto &kv : da) {
          auto it = db.find(kv.first);
          if (it != db.end() && kv.second != it->second) {
            fork = kv.first;
            aShort = kv.second < it->second;
            break;
          }
        }
        if (!fork)
          continue;
        Value shortStream = aShort ? ins[a] : ins[b];
        int64_t elems = 0;
        if (auto *def = shortStream.getDefiningOp())
          if (auto e = def->getAttrOfType<IntegerAttr>("allo.fifo.elems"))
            elems = e.getInt();
        int64_t d = cast<StreamType>(shortStream.getType()).getDepth();
        if (d < elems)
          emitWarning()
              << "streamline: reconvergent path from @"
              << cast<InvokeOp>(fork).getCalleeAttr().getValue() << " to @"
              << join.getCalleeAttr().getValue() << " has a depth-" << d
              << " FIFO on its short branch; the dataflow may deadlock -- set "
                 "streamline(depth=) >= "
              << elems;
      }
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::StreamlineOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getProducersMutable(), effects);
  transform::onlyReadsHandle(getConsumersMutable(), effects);
  transform::modifiesPayload(effects);
}

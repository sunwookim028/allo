/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SystemC / Catapult-HLS backend. Based on EmitCatapultHLS.cpp; subclasses the Vivado
 * emitter so loop/arith/memref emission is reused. Only the module/thread structure and
 * the MatchLib Connections links are SystemC-specific.
 *
 *   Stream[T,depth] -> Connections::Fifo (AlloFifoC) ; put/get -> Push/Pop
 *   Channel / Wire  -> Connections::Combinational / sc_signal
 *   @df.kernel      -> SC_MODULE + SC_THREAD(run)
 *   @df.region/top  -> wiring SC_MODULE (channel members + kernel instances + port binds)
 */

#include "allo/Translation/EmitSystemC.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/Visitor.h"
#include "allo/Translation/EmitVivadoHLS.h" // reuse the Vhls emitter base
#include "allo/Translation/EmitCatapultHLS.h" // reuse Catapult's ac_int type map
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace allo;

// TODO: REVISIT — audit comments against the code before trusting them. Several
// "not supported / not yet / errored" claims here were STALE (as of 2026-08: empty()/full()
// "not synthesizable", the wide-word ">64 accidentally worked", and memory-port store
// "'o'/'b' not yet" were all false). When you touch a path, update its comment.

//===----------------------------------------------------------------------===//
// Helper functions (file-local): type names, signedness, channel protocol, reset,
// memory-port sizing.
//===----------------------------------------------------------------------===//

// C++ type name for an SC interface (port / channel). Mirrors the Vhls emitter's
// getTypeName so port types match the reused body: i8/16/32/64 -> (u)intN_t, other
// widths -> ap_(u)int<N> (aliased to ac_int in the header), f16 -> half, f32 -> float,
// fixed -> ap_(u)fixed.
static SmallString<32> getSCTypeName(Type valType) {
  // For integers WIDER than 64 bits, use the ap_int shim (an ac_int subclass,
  // defined in the preamble) rather than a plain ac_int: ac_int omits the
  // implicit >64-bit narrowing to a native int (e.g. `int x = <ac_int<66>>`),
  // which the shim provides via operator long long(). (<=64-bit and non-integer
  // types get Catapult-native ac_int/ac_fixed/ac_ieee_float from the shared map.
  // Under __SYNTHESIS__ the shim is a plain ac_int alias, so RTL is unaffected.)
  Type scalar = valType;
  if (auto st = llvm::dyn_cast<ShapedType>(scalar))
    scalar = st.getElementType();
  if (auto it = llvm::dyn_cast<IntegerType>(scalar)) {
    if (it.getWidth() > 64) {
      bool uns =
          it.getSignedness() == IntegerType::SignednessSemantics::Unsigned;
      return SmallString<32>((uns ? "ap_uint<" : "ap_int<") +
                             std::to_string(it.getWidth()) + ">");
    }
  }
  // Delegate to the shared Catapult type mapping so the SystemC flow emits
  // Catapult-native ac_int/ac_fixed/ac_ieee_float<binary32>.
  return SmallString<32>(allo::getCatapultTypeName(valType).str());
}

// A stream/channel/wire/memory link's element UNSIGNEDNESS is not in its MLIR base
// type -- allo builds that signless (i22), and carries the sign as an `unsigned`
// attr instead: on the *construct* op (top level, builder.py sets it for UInt) and
// on every put/get/load/store *user* op (kernel level). Locals recover it via
// fixUnsignedType(hasAttr("unsigned")); a link payload type has no such op inline,
// so recover it here from the defining construct or any user. (Without this every
// UInt payload emits as ac_int<W,true> -> signed, breaking csim of UInt designs.)
static bool linkPayloadUnsigned(Value linkVal) {
  if (Operation *def = linkVal.getDefiningOp())
    if (def->hasAttr("unsigned"))
      return true;
  for (Operation *user : linkVal.getUsers())
    if (user->hasAttr("unsigned"))
      return true;
  return false;
}

// Payload type for a Connections stream/channel/wire (or a memory port). These go
// through the marshaller (Wrapped<T> needs T::width + T::Marshall), which ships
// specializations only for SOME native integers -- `short`/`int`/`long` have them
// but `signed char` (int8_t) does NOT, so an int8 payload fails to synthesize
// (marshaller.h:203, CRD-276). ac_int<W> always carries the marshaller interface, so
// emit EVERY <=64 bit integer payload as ac_int<W> (uniform, a no-op for cosim
// bit-width). `isUnsigned` comes from linkPayloadUnsigned (the base type is signless,
// so its own signedness can't be trusted). Wider ints keep the ap_int/ap_uint shim
// (an ac_int subclass, marshaller-compatible); non-integers marshal as-is.
static SmallString<32> getStreamPayloadTypeName(Type valType, bool isUnsigned) {
  Type scalar = valType;
  if (auto st = llvm::dyn_cast<ShapedType>(scalar))
    scalar = st.getElementType();
  if (auto it = llvm::dyn_cast<IntegerType>(scalar)) {
    bool uns = isUnsigned ||
               it.getSignedness() == IntegerType::SignednessSemantics::Unsigned;
    if (it.getWidth() <= 64)
      return SmallString<32>("ac_int<" + std::to_string(it.getWidth()) + ", " +
                             (uns ? "false" : "true") + ">");
    return SmallString<32>((uns ? "ap_uint<" : "ap_int<") +
                           std::to_string(it.getWidth()) + ">");
  }
  return getSCTypeName(valType);
}

// A Channel's PROTOCOL picks genuinely different hardware, not a cosmetic label:
//   valid_ready -> Connections::Combinational<T>              (full handshake, back-pressure)
//   valid_only  -> raw sc_signal<T> _dat + sc_signal<bool> _vld  (no ready line)
// valid_only never refuses the producer (one-cycle valid pulse); if the consumer is not
// looking that cycle the datum is dropped -- by design, the cheaper link (no ready path or
// transactor, which a DSE cost model may prefer). So it is correct ONLY for lock-step or
// drop-tolerant consumers; it sits between valid_ready (safe at any timing) and Wire (no
// sync at all) -- a race SKIPS a datum rather than reading garbage.
static bool isValidOnlyChannel(Value v) {
  auto ct = llvm::dyn_cast<ChannelType>(v.getType());
  if (!ct)
    return false;
  return ct.getProtocol() == ChannelProtocol::ValidOnly;
}

// Reset style. Default = async (async_reset_signal_is), matching prior behavior.
// ALLO_SYNC_RESET set -> synchronous reset (reset_signal_is): lets the tool use
// plain / sync-reset flops on datapath registers instead of forcing an async-reset
// flop (DFFR) on every register + an async reset tree -- smaller area and cleaner
// DFT/timing on the ASIC path. Reset is held for several cycles by the tb, so the
// one-edge-later semantics of a sync reset are safe. Applies to emitted SC_THREADs
// and (via a string swap at emit) the device_header module templates.
static const char *alloResetFn() {
  static const bool sync = std::getenv("ALLO_SYNC_RESET") != nullptr;
  return sync ? "reset_signal_is" : "async_reset_signal_is";
}

// Address width for a memory of `total` elements: ceil(log2(total)), min 1.
static unsigned scAddrW(int64_t total) {
  unsigned w = 1;
  while ((int64_t(1) << w) < total) // 2^w
    w++;
  return w;
}

// Data width (bits) of a memory element type — used to size the packed request.
static unsigned scDataW(Type elt) {
  if (auto it = llvm::dyn_cast<IntegerType>(elt))
    return it.getWidth() < 1 ? 1 : it.getWidth();
  if (llvm::isa<Float16Type>(elt))
    return 16;
  if (llvm::isa<Float64Type>(elt))
    return 64;
  return 32; // f32 / index / default
}

//===----------------------------------------------------------------------===//
// Hierarchy flattening (pre-pass).
// A @df.kernel body may invoke a `dataflow` SUB-REGION (or a delegating kernel
// that does). C++/HLS keeps such nesting as nested dataflow functions, but a
// SystemC kernel is an SC_THREAD and cannot structurally instantiate a
// sub-region. So before emitting we INLINE every call to a non-leaf callee into
// the top, transitively, until the top is a FLAT set of stream constructs +
// leaf-kernel calls (which emitTopModule already handles). A flat design has no
// non-leaf calls -> this is a no-op.
//===----------------------------------------------------------------------===//

// A leaf compute kernel: a df.kernel whose body calls no other kernel/sub-region
// (only compute + stream get/put, and possibly pure helper functions).
static bool callsKernelOrRegion(func::FuncOp f, ModuleOp m) {
  bool found = false;
  f.walk([&](func::CallOp c) {
    if (auto callee = m.lookupSymbol<func::FuncOp>(c.getCallee()))
      if (callee->hasAttr("df.kernel") || callee->hasAttr("dataflow"))
        found = true;
  });
  return found;
}
static bool isLeafKernel(func::FuncOp f, ModuleOp m) {
  return f->hasAttr("df.kernel") && !callsKernelOrRegion(f, m);
}

static void flattenHierarchy(ModuleOp module) {
  func::FuncOp top;
  for (auto f : module.getOps<func::FuncOp>())
    if (f->hasAttr("top"))
      top = f;
  if (!top)
    return;
  // Repeatedly inline every top-body call to a non-leaf callee, substituting the
  // callee's block args with the actual operands (clone carries the mapping to
  // cloned results, so inner stream/kernel operands are remapped correctly).
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<func::CallOp> toInline;
    top.walk([&](func::CallOp call) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (callee && callee.getBlocks().size() == 1 &&
          !isLeafKernel(callee, module))
        toInline.push_back(call);
    });
    for (auto call : toInline) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      IRMapping map;
      for (auto pair : llvm::zip(callee.getArguments(), call.getOperands()))
        map.map(std::get<0>(pair), std::get<1>(pair));
      OpBuilder builder(call);
      for (auto &op : callee.front().without_terminator())
        builder.clone(op, map);
      call.erase();
      changed = true;
    }
  }
  // Erase the now-dead inlined funcs (delegating kernels + sub-regions); leaf
  // kernels + helpers stay (still referenced by the flattened top).
  bool erased = true;
  while (erased) {
    erased = false;
    for (auto f : llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      if (!f->hasAttr("top") && SymbolTable::symbolKnownUseEmpty(f, module)) {
        f.erase();
        erased = true;
      }
  }
}

//===----------------------------------------------------------------------===//
// SystemC emitter — subclass of CatapultModuleEmitter (reuses Vhls body emission).
//===----------------------------------------------------------------------===//

namespace {

// Re-parented onto CatapultModuleEmitter (instead of VhlsModuleEmitter) so the
// SystemC flow reuses Catapult-native codegen: ac_int/ac_fixed types AND the
// Catapult loop/design pragmas (#pragma hls_pipeline_init_interval / hls_unroll
// / hls_design) via the inherited emitLoopDirectives -- the base Vivado emitter
// would otherwise emit Xilinx "#pragma HLS ..." which Catapult ignores. This
// emitter's own overrides (SC_MODULE structure, Connections/sc_signal links,
// Pop/Push get-put, affine load/store) still win.
class SystemCModuleEmitter : public allo::CatapultModuleEmitter {
public:
  explicit SystemCModuleEmitter(AlloEmitterState &state)
      : allo::CatapultModuleEmitter(state) {
    // SystemC/Catapult-native flow: f16 constants -> explicit half(...) ctor.
    state.acFloatConstCtor = true;
    state.scfWhileWait = true; // clocked SC_THREADs: wait() per while iteration
  }

  void emitModule(ModuleOp module) override;

private:
  void emitKernelModule(func::FuncOp func); // @df.kernel -> SC_MODULE + SC_THREAD
  void emitTopModule(func::FuncOp func);    // @df.region/top -> wiring SC_MODULE

  // Connections: get/put emit .Pop()/.Push() instead of the base .read()/.write().
  void emitStreamGet(allo::StreamGetOp op) override;
  void emitStreamPut(allo::StreamPutOp op) override;
  void emitChannelGet(allo::ChannelGetOp op) override;
  void emitChannelPut(allo::ChannelPutOp op) override;
  void emitWireGet(allo::WireGetOp op) override;
  void emitWirePut(allo::WirePutOp op) override;
  // Non-blocking: try_get/try_put -> Connections .PopNB()/.PushNB() (fire-on-valid).
  // empty()/full() read a synchronous sideband signal, NOT In::Empty()/Out::Full() (which
  // track the port handshake, not the FIFO's logical occupancy) -- see emitStreamEmpty.
  void emitStreamTryGet(allo::StreamTryGetOp op) override;
  void emitStreamTryPut(allo::StreamTryPutOp op) override;
  void emitChannelTryGet(allo::ChannelTryGetOp op) override;
  void emitChannelTryPut(allo::ChannelTryPutOp op) override;
  void emitStreamEmpty(allo::StreamEmptyOp op) override;
  void emitStreamFull(allo::StreamFullOp op) override;
  // Narrow a >64-bit ac_int to a native int/index with an explicit
  // .to_int64()/.to_uint64() (no implicit conversion under __SYNTHESIS__).
  void emitNarrowCastSuffix(Value src, Value dst) override;
  // max/min with both operands cast to the result type (a float literal mixed
  // with an ac_ieee_float operand otherwise fails template deduction).
  void emitMaxMin(Operation *op, const char *syntax) override;

  // Slice assignment (`A[pid, :] = v`) lowers to memref.subview + memref.copy. The base
  // emits the subview as a POINTER into the array (`T *v2 = &v0[0][0];`) -- but a
  // memory-port array is PINS, there is no array to point into. So record the subview's
  // flat offset instead and let the copy address the memory through _rd/_wr.
  void emitSubView(memref::SubViewOp op) override;
  void emitCopy(memref::CopyOp op) override;
  // subview result -> (memory-port base, flat offset expression)
  llvm::DenseMap<Value, std::pair<Value, std::string>> memPortSubviews;

  // Stateful globals (`x: T @ Stateful`). The base emits them as function-scope
  // `static`s -- correct for a C function that is CALLED REPEATEDLY, wrong for an
  // SC_THREAD (see the reset-action block in emitKernelModule).
  void emitGlobalStorageQualifier(memref::GlobalOp op) override;
  // getTypeName isn't virtual, so the declaration would otherwise print Xilinx
  // ap_int/ap_fixed while the body's uses print Catapult-native types.
  void emitStatefulGlobalElementType(Type type) override;
  static bool isStatefulGlobal(memref::GlobalOp g);
  // One element of a dense initializer, as a C++ literal. Mirrors the base emitGlobal's
  // per-element formatting, which is only reachable there inside a full `= {...}` brace
  // list -- a reset action needs the values one at a time.
  void emitDenseElementLiteral(Attribute element, Type type, bool isUnsigned);

  // Loop-shape transform: the kernel's outermost `for t` becomes a free-running while(1)
  // under __SYNTHESIS__ so Catapult pipelines the body (see emitAffineFor).
  bool isSteadyStateLoop(affine::AffineForOp op);
  void emitAffineFor(affine::AffineForOp op) override;

  // Sequential-stream body transform: a boundary memref arg becomes a Connections
  // stream port, so load a[i] -> port.Pop(), store b[i]=v -> port.Push(v).
  void emitAffineLoad(affine::AffineLoadOp op) override;
  void emitAffineStore(affine::AffineStoreOp op) override;

  // The base emitValue uses EmitVivadoHLS's file-local getTypeName (ap_int/
  // ap_fixed). Override so scalar SSA decls in kernel bodies get Catapult-native
  // types (ac_int/ac_fixed) via getSCTypeName, matching the port/signal decls.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;
  // The base emits a `union{ from; to; }` bit-converter for a bitcast; that has
  // a deleted ctor when a member is non-trivial (ac_ieee_float<binary16>). Emit
  // a std::memcpy bit-reinterpret instead (both operands are equal-width POD).
  void emitBitcast(arith::BitcastOp op) override;

  // Bit ops: the base emits the Vitis `ap_int` proxy forms (`x(hi,lo)`, `x[i]`)
  // via the ap_int-subclass shim, which does not interoperate with ac_int
  // (`ap_rng` won't assign to ac_int). Emit the NATIVE ac_int API instead:
  //   get slice -> num.slc<W>(lo)      set slice -> res=num; res.set_slc(lo,val)
  //   get bit   -> num[idx]            set bit   -> res=num; res[idx]=val
  void emitGetBit(allo::GetIntBitOp op) override;
  void emitSetBit(allo::SetIntBitOp op) override;
  void emitGetSlice(allo::GetIntSliceOp op) override;
  void emitSetSlice(allo::SetIntSliceOp op) override;

  // Random-access accesses to a memory-port arg via the memref dialect (dynamic
  // / 2-D indices) -- mirror the affine load/store rewrites (_rd/_wr accessors);
  // non-mem-port memrefs (local %alloc arrays) fall back to the base emitter.
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;
  // If `v` is a df.kernel memref arg turned into a stream, its dir ('i'/'o'); else 0.
  char streamArgDir(Value v);
  // If `v` is a df.kernel memref arg that is directional but NOT sequentially
  // streamable (random/strided/2-D), it becomes a random-access MEMORY PORT.
  // Returns its dir ('i' load, 'o' store, 'b' read+modify+write), else 0.
  char memPortArgDir(Value v);
  // Row-major flatten of direct (memref-dialect) index Values -> one C++ expr.
  void emitFlatIndexMemref(ValueRange indices, ArrayRef<int64_t> shape);
  // Shared memory-port transaction emitters (flat index supplied by callback,
  // so the affine and memref call sites reuse the same req/rsp lowering).
  void emitMemPortLoad(Value memref, Value result, bool isUnsigned,
                       llvm::function_ref<void()> emitIdx);
  void emitMemPortStore(Value memref, Value value,
                        llvm::function_ref<void()> emitIdx);
  // Streamable = 1-D array accessed strictly in order (a[i] by the loop index) -> a FIFO;
  // anything else (2-D / strided / random / reused) needs a random-access memory port.
  bool isSeqStreamable(Value v);

  // Emit a single affine expr (dims/symbols resolved via `operands`, split at
  // `numDims`) as a C++ index expression — a local reimplementation of the
  // base's file-local AffineExprEmitter (not reachable from this file).
  void emitAffineExprSC(AffineExpr e, ValueRange operands, unsigned numDims);
  // Emit the row-major FLAT element index (memory-port address) of an affine
  // load or store: Σ result[k] * stride[k].
  void emitFlatIndexCore(AffineMap map, ArrayRef<int64_t> shape,
                         ValueRange operands);
  void emitFlatIndex(affine::AffineLoadOp op);
  void emitFlatIndex(affine::AffineStoreOp op);
  // For a region boundary arg, the memory-port dir of the kernel arg it feeds.
  char regArgMemPort(func::FuncOp top, Value regArg);

  // direction of stream arg `i` from the func's "stypes" attr: 'i','o', or 0.
  char streamDir(func::FuncOp func, unsigned i);
  // direction of memref arg `i` from the func's "arg_dirs" attr: 'i','o','b', or 0.
  char argDir(func::FuncOp func, unsigned i);

  // Region boundary arrays -> top-level Connections stream ports; the sc_main
  // testbench drives inputs and reads outputs (direction from arg_dirs).
  struct IOArray {
    std::string member; // top-level stream port name (e.g. "v11")
    std::string ctype;  // element C type (for the tb Combinational channel)
    int64_t total;      // flattened element count
    char dir;           // 'i' input (Push), 'o' output (Pop)
    int fileIdx;        // input<fileIdx>.data / output<fileIdx>.data
  };
  SmallVector<IOArray> ioArrays;

  // Region boundary array routed to an internal memory (random-access port):
  //   dir 'i' -> AlloMem  (LOAD, req+rsp), preloaded from input<inIdx>.data
  //   dir 'o' -> write-only RAM pins, read out to output<outIdx>.data
  //   dir 'b' -> AlloMem  (LOAD+STORE, req+rsp): preloaded AND read out (in-place)
  struct MemArray {
    std::string base;   // region arg name (kernel binds base_radr/_re/_q, _wadr/_d/_we)
    std::string ctype;  // element C type
    int64_t total;      // element count (memory depth)
    unsigned addrw, dataw;
    char dir;           // 'i' read-only, 'o' write-only, 'b' read+write
    int inIdx;          // input<inIdx>.data  (preload; -1 if none)
    int outIdx;         // output<outIdx>.data (read-out; -1 if none)
  };
  SmallVector<MemArray> memArrays;

  // One PHYSICAL memory per (kernel instance, memory-port arg). A boundary array
  // shared by several grid replicas is REPLICATED: each client gets its own
  // memory + channels. Reads preload every replica from the same input file;
  // writes (disjoint pid-indexed elements) are summed across replicas at readout.
  // TODO: REVISIT — the write-merge SUMS elements across replicas and ASSUMES each replica
  // writes disjoint pid-indexed elements (rest zero); nothing enforces it, so two replicas
  // writing the same element would silently corrupt the readout. Add a check/assert.
  struct MemInst {
    std::string chan;   // unique base name (mp<call>_<arg>) for channels + memory
    std::string inst;   // kernel instance name (u<call>)
    std::string port;   // kernel port name (callee arg)
    std::string ctype;
    int64_t total;
    unsigned addrw, dataw;
    char dir;           // 'i' read / 'o' write / 'b' read+write
    int inIdx, outIdx;  // the region array's input/output file indices (-1 = none)
    std::string base;   // REGION array this replicates -- the grouping key. Several
                        // MemInsts share a base exactly when the array is replicated
                        // across clients, which decides whether it can be exposed as
                        // a top-level port (see the port-emission comment).
    bool exposed;       // true -> lives in the tb, reachable through top ports;
                        // false -> stays an internal memory (multi-client replica)
  };
  SmallVector<MemInst> memInsts;

  // A stream used by exactly ONE kernel (a self-FIFO: one kernel does put+get+empty+full)
  // can't map to a directional Connections port pair, so it is realized as a bounded MatchLib
  // AlloFifo wired as a self-loop at the top (the kernel gets both an Out enq end and an In
  // deq end), with a synchronous _cnt counter for correct empty()/full().
  llvm::DenseSet<Value> localStreamArgs;       // the kernel block-arg Values
  llvm::DenseSet<Value> localStreamConstructs; // the top-level construct results
  bool isLocalStream(Value v) { return localStreamArgs.count(v) > 0; }

  // A boundary memref array used as a seq-streamable directional arg by MORE THAN
  // ONE kernel call (fan-in output: N producers; fan-out input: N consumers) can't
  // map to a single Connections::Combinational (exactly one writer + one reader; a
  // 2nd bind aborts MatchLib's ConManager). These callee block-args are forced to
  // the random-access memory-port path (per-PE AlloMem[W] replica) instead.
  llvm::DenseSet<Value> forceMemPortArgs;
  bool forceMemPort(Value v) { return forceMemPortArgs.count(v) > 0; }

  // When ONE kernel call passes the SAME stream/channel to MULTIPLE arg positions
  // (e.g. a drain PE that merges two sources with two put-sites into one output
  // stream), the callee has several block-args aliasing one channel. Emitting a
  // port per arg would bind >1 sc_out to one signal (SystemC E115). Map each such
  // duplicate block-arg -> the FIRST (primary) block-arg so they share one port.
  llvm::DenseMap<Value, Value> streamArgAlias;
  Value aliasOf(Value v) { return streamArgAlias.lookup(v); }

  // Number of df.kernel MODULE INSTANCES in the top (calls.size()). The single-
  // shot testbench advances the clock until this many kernels have finished one
  // pass (each bumps the csim-only __allo_done counter) before reading memory.
  int numKernelInsts = 0;
};

} // namespace

// stypes is a string with one char per arg: '_' = not a stream, 'i' = in, 'o' = out.
char SystemCModuleEmitter::streamDir(func::FuncOp func, unsigned i) {  // new (SystemC-only)
  auto attr = func->getAttrOfType<StringAttr>("stypes");
  if (!attr)
    return 0;
  StringRef s = attr.getValue();
  if (i >= s.size())
    return 0;
  char c = s[i];
  return (c == 'i' || c == 'o') ? c : 0;
}

// arg_dirs is a string with one char per arg: 'i'=in, 'o'=out, 'b'=both, else '_'.
char SystemCModuleEmitter::argDir(func::FuncOp func, unsigned i) {  // new (SystemC-only)
  auto attr = func->getAttrOfType<StringAttr>("arg_dirs");
  if (!attr)
    return 0;
  StringRef s = attr.getValue();
  if (i >= s.size())
    return 0;
  char c = s[i];
  return (c == 'i' || c == 'o' || c == 'b') ? c : 0;
}

// Subview of a memory-port array. The base emits `T *p = &arr[i][j];`, which needs an
// actual array; with RAM pins there is none. Emit nothing and remember where the slice
// starts, so the memref.copy that consumes it can drive the pins directly.
void SystemCModuleEmitter::emitSubView(memref::SubViewOp op) {  // override (base emitter)
  Value src = op.getSource();
  if (!memPortArgDir(src)) {
    CatapultModuleEmitter::emitSubView(op);
    return;
  }
  auto srcType = llvm::cast<MemRefType>(src.getType());
  auto shape = srcType.getShape();
  unsigned n = shape.size();
  if (!srcType.hasStaticShape()) {
    emitError(op, "memref.subview of a memory port requires a static shape.");
    return;
  }
  // The offset arithmetic below collapses the slice to `base + flat_offset`, i.e. it
  // assumes a CONTIGUOUS run. The base emitter validates exactly this before taking a
  // pointer, and dropping the checks here would not fail -- it would silently address
  // the wrong elements. So refuse what cannot be linearised, rather than miscompute it.
  for (auto s : op.getMixedStrides()) {
    auto attr = llvm::dyn_cast_or_null<Attribute>(s);
    if (!attr || llvm::cast<IntegerAttr>(attr).getInt() != 1) {
      emitError(op, "only unit-stride memref.subview is supported on a memory port.");
      return;
    }
  }
  auto sizes = op.getMixedSizes();
  for (unsigned k = 1; k < sizes.size() && k < n; ++k) {
    auto attr = llvm::dyn_cast_or_null<Attribute>(sizes[k]);
    if (!attr || llvm::cast<IntegerAttr>(attr).getInt() != shape[k]) {
      emitError(op, "only a contiguous memref.subview is supported on a memory port.");
      return;
    }
  }
  SmallVector<int64_t> stride(n);
  int64_t acc = 1;
  for (int k = (int)n - 1; k >= 0; --k) { stride[k] = acc; acc *= shape[k]; }
  std::string off;
  auto offsets = op.getMixedOffsets();
  for (unsigned k = 0; k < offsets.size() && k < n; ++k) {
    std::string term;
    if (auto attr = llvm::dyn_cast_or_null<Attribute>(offsets[k])) {
      int64_t v = llvm::cast<IntegerAttr>(attr).getInt();
      if (v == 0)
        continue;
      term = std::to_string(v);
    } else {
      term = std::string(getName(llvm::cast<Value>(offsets[k])).str());
    }
    if (stride[k] != 1)
      term = "(" + term + ") * " + std::to_string(stride[k]);
    off = off.empty() ? term : off + " + " + term;
  }
  if (off.empty())
    off = "0";
  memPortSubviews[op.getResult()] = {src, off};
  indent();
  os << "// slice of memory-port array " << getName(src) << " at flat offset " << off;
  emitInfoAndNewLine(op);
}

// Whole-array copy. If either side is a memory port (directly, or through the subview
// recorded above) the element access has to go through the _rd/_wr accessors instead of
// array indexing.
void SystemCModuleEmitter::emitCopy(memref::CopyOp op) {  // override (base emitter)
  Value src = op.getSource(), dst = op.getTarget();
  auto resolve = [&](Value v, Value &base, std::string &off) -> bool {
    auto it = memPortSubviews.find(v);
    if (it != memPortSubviews.end()) {
      base = it->second.first;
      off = it->second.second;
      return true;
    }
    if (memPortArgDir(v)) {
      base = v;
      off = "0";
      return true;
    }
    return false;
  };
  Value sBase, dBase;
  std::string sOff, dOff;
  bool sMem = resolve(src, sBase, sOff);
  bool dMem = resolve(dst, dBase, dOff);
  if (!sMem && !dMem) {
    CatapultModuleEmitter::emitCopy(op);
    return;
  }
  auto cpType = llvm::dyn_cast<MemRefType>(sMem ? dst.getType() : src.getType());
  if (!cpType || !cpType.hasStaticShape()) {
    emitError(op, "memref.copy on a memory port requires a statically shaped operand.");
    return;
  }
  auto shape = cpType.getShape();
  unsigned n = shape.size();
  // Flat index over the copied region, used for whichever side is a memory port.
  SmallVector<int64_t> stride(n);
  int64_t acc = 1;
  for (int k = (int)n - 1; k >= 0; --k) { stride[k] = acc; acc *= shape[k]; }
  auto flat = [&](const std::string &off) {
    std::string e = off;
    for (unsigned k = 0; k < n; ++k) {
      std::string t = "_cp" + std::to_string(k);
      if (stride[k] != 1)
        t = "(" + t + ") * " + std::to_string(stride[k]);
      e += " + " + t;
    }
    return e;
  };
  indent(); os << "{\n";
  addIndent();
  for (unsigned k = 0; k < n; ++k) {
    indent();
    os << "for (int _cp" << k << " = 0; _cp" << k << " < " << shape[k] << "; ++_cp" << k
       << ") {\n";
    addIndent();
  }
  indent();
  if (dMem) {
    os << getName(dBase) << "_wr((ac_int<"
       << scAddrW(llvm::cast<MemRefType>(dBase.getType()).getNumElements())
       << ", false>)(" << flat(dOff) << "), ";
    if (sMem)
      os << getName(sBase) << "_rd((ac_int<"
         << scAddrW(llvm::cast<MemRefType>(sBase.getType()).getNumElements())
         << ", false>)(" << flat(sOff) << "))";
    else {
      emitValue(src);
      for (unsigned k = 0; k < n; ++k)
        os << "[_cp" << k << "]";
    }
    os << ");\n";
  } else { // source is the memory port, destination a plain array
    emitValue(dst);
    for (unsigned k = 0; k < n; ++k)
      os << "[_cp" << k << "]";
    os << " = " << getName(sBase) << "_rd((ac_int<"
       << scAddrW(llvm::cast<MemRefType>(sBase.getType()).getNumElements())
       << ", false>)(" << flat(sOff) << "));\n";
  }
  for (unsigned k = 0; k < n; ++k) { reduceIndent(); indent(); os << "}\n"; }
  reduceIndent();
  indent(); os << "}";
  emitInfoAndNewLine(op);
}

// A `x: T @ Stateful` variable. The frontend lowers it to a private memref.global
// carrying an initial value, tagged BOTH with a `static` attr and a `__stateful_` name
// prefix (allo/ir/builder.py:1965, the only site that sets either).
//
// Keyed on the NAME ALONE, deliberately, even though the base emitter accepts either.
// `static` on its own means "give this static storage duration", which is a different
// request: a global carrying it but NOT stateful would be handed a per-instance member
// and a reset, silently changing its lifetime. Matching the name keeps this to variables
// the frontend actually created from `@ Stateful`.
bool SystemCModuleEmitter::isStatefulGlobal(memref::GlobalOp g) {  // new (SystemC-only)
  return g.getSymName().str().find("__stateful_") != std::string::npos;
}

// Storage class. The base maps stateful -> `static`, which is right for a C function
// called repeatedly but WRONG here on two counts: a function-scope static in run() has
// static storage duration (shared by every instance of the SC_MODULE, not per-instance),
// and it is initialised once at program start, so an RTL reset would NOT clear it and
// csim would silently diverge from cosim. We emit no storage class at all and declare
// the variable in the reset action instead (emitKernelModule), which gives per-instance
// state that the reset re-initialises -- matching the RTL.
void SystemCModuleEmitter::emitGlobalStorageQualifier(memref::GlobalOp op) {  // override (base emitter)
  if (isStatefulGlobal(op) && !op->hasAttr("constant"))
    return;
  CatapultModuleEmitter::emitGlobalStorageQualifier(op);
}

// Element type of a global's declaration. getTypeName is not virtual, so without this
// the declaration prints Xilinx ap_int/ap_fixed while every use of it in the body prints
// the Catapult-native type -- they interoperate only through the ap_int shim.
void SystemCModuleEmitter::emitStatefulGlobalElementType(Type type) {  // override (base emitter)
  os << getSCTypeName(type);
}

// One dense-initializer element as a standalone C++ literal. The base emits these only
// inside a brace list; resetting a member needs them individually.
void SystemCModuleEmitter::emitDenseElementLiteral(Attribute element, Type type,
                                                   bool isUnsigned) {  // new (SystemC-only)
  if (llvm::isa<FloatType>(type)) {
    // f16 needs an explicit half(...) ctor (state.acFloatConstCtor); reuse the base's
    // float formatter so INFINITY / precision handling stays in one place.
    auto fa = llvm::cast<FloatAttr>(element);
    if (type.isF64()) {
      double v = fa.getValue().convertToDouble();
      if (std::isfinite(v))
        os << v;
      else
        os << (v > 0 ? "INFINITY" : "-INFINITY");
    } else {
      emitFloatArrayElement(fa.getValue().convertToFloat());
    }
    return;
  }
  if (type.isInteger(1)) {
    os << (llvm::cast<BoolAttr>(element).getValue() ? "true" : "false");
    return;
  }
  if (type.isIntOrIndex()) {
    auto it = llvm::dyn_cast<IntegerType>(type);
    if (isUnsigned) {
      os << llvm::cast<IntegerAttr>(element).getValue().getZExtValue();
      if (it && it.getWidth() > 64)
        os << "ULL";
    } else {
      os << llvm::cast<IntegerAttr>(element).getValue();
      if (it && it.getWidth() > 64)
        os << "LL";
    }
    return;
  }
  emitError(nullptr, "stateful variable has an unsupported element type.");
}

// Safe-to-stream check: sequential single-pass access only. The stream transform
// ignores the load/store index, so it is correct ONLY if the array is 1-D and
// every access is an identity a[iv] (each element once, in order). Anything else
// (2-D, strided, reversed, gathered, re-read, or a non-load/store use) is NOT
// sequential and must use a memory port instead.
bool SystemCModuleEmitter::isSeqStreamable(Value v) {  // new (SystemC-only)
  auto mt = llvm::dyn_cast<MemRefType>(v.getType());
  if (!mt || mt.getRank() != 1)
    return false;
  for (auto &use : v.getUses()) {
    Operation *op = use.getOwner();
    if (auto ld = llvm::dyn_cast<affine::AffineLoadOp>(op)) {
      if (!ld.getAffineMap().isIdentity())
        return false;
    } else if (auto st = llvm::dyn_cast<affine::AffineStoreOp>(op)) {
      if (!st.getAffineMap().isIdentity())
        return false;
    } else {
      return false; // any other use -> not a clean sequential scan
    }
    // Reject re-reads/re-writes: an identity a[iv] under an OUTER loop touches
    // each element more than once, which a stream (one element per handshake)
    // cannot reproduce. A 1-D single-pass scan sits inside exactly ONE loop.
    unsigned loops = 0;
    for (Operation *p = op->getParentOp();
         p && !llvm::isa<func::FuncOp>(p); p = p->getParentOp())
      if (llvm::isa<affine::AffineForOp>(p))
        loops++;
    if (loops != 1)
      return false;
  }
  return true;
}

// A df.kernel memref arg with a pure in/out direction is stream-ified into a
// Connections port; return that direction ('i'/'o'), else 0 (emit normally).
char SystemCModuleEmitter::streamArgDir(Value v) {  // new (SystemC-only)
  auto barg = llvm::dyn_cast<BlockArgument>(v);
  if (!barg || !llvm::isa<MemRefType>(v.getType()))
    return 0;
  auto func = llvm::dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp());
  if (!func || !func->hasAttr("df.kernel"))
    return 0;
  char d = argDir(func, barg.getArgNumber());
  // stream only pure in/out AND sequentially-safe args ('both' / random stay memref)
  // AND single-driver (multi-producer/consumer is forced to the memory-port path).
  return ((d == 'i' || d == 'o') && isSeqStreamable(v) && !forceMemPort(v)) ? d : 0;
}

// A df.kernel memref arg that is directional but NOT sequentially streamable is a
// random-access memory port. Returns its dir: 'i' load (AlloMem req+rsp), 'o' store
// (write pins only), 'b' read+modify+write (both pin bundles); else 0.
char SystemCModuleEmitter::memPortArgDir(Value v) {  // new (SystemC-only)
  auto barg = llvm::dyn_cast<BlockArgument>(v);
  if (!barg || !llvm::isa<MemRefType>(v.getType()))
    return 0;
  auto func = llvm::dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp());
  if (!func || !func->hasAttr("df.kernel"))
    return 0;
  char d = argDir(func, barg.getArgNumber());
  if (d != 'i' && d != 'o' && d != 'b')
    return 0;
  if (forceMemPort(v))
    return d; // multi-driver boundary: forced off the stream path onto memory
  return isSeqStreamable(v) ? 0 : d; // streamable -> handled by the stream path
}

// A dataflow kernel's outermost bounded loop IS its steady-state loop: `for t in
// range(NUM_IT)` runs one router/crossbar step per iteration. Emitted as a finite loop
// followed by the terminal `while(1) wait();`, Catapult classifies EVERYTHING before
// that while as RESET ACTION -- it never pipelines the body and the real steady-state
// loop is a 1-cycle empty spin. Measured: whvcrouter 5 c-steps/iteration, arbxbar 17,
// against MatchLib's II=1 `while(1){wait(); body;}`. Emitting this loop AS `while(1)`
// under __SYNTHESIS__ gives Catapult MatchLib's shape.
//
// State declarations already sit ABOVE this loop, so nothing moves. (An earlier attempt
// wrapped while(1) around the declarations too -- that made every pass a cold restart
// and cost `buf` its resource path: "Unknown path '/router_0/run/buf:rsc'".)
bool SystemCModuleEmitter::isSteadyStateLoop(affine::AffineForOp op) {  // new (SystemC-only)
  auto func = op->getParentOfType<func::FuncOp>();
  // A KERNEL carries df.kernel; "dataflow" is on the region top (see line ~2314).
  if (!func || (!func->hasAttr("df.kernel") && !func->hasAttr("dataflow")))
    return false;
  // Outermost loop of the kernel body.
  if (op->getParentOp() != func.getOperation())
    return false;
  // Induction variable must be DEAD -- a body that reads `t` would change meaning.
  if (!op.getInductionVar().use_empty())
    return false;
  // Constant trip count only; anything else keeps the ordinary path.
  if (!op.hasConstantBounds())
    return false;
  // A kernel storing to a random-access MEMORY PORT must run ONCE: a free-running
  // body would re-accumulate `C[i] += ...` on every pass. Only PORT arrays count --
  // they arrive as function arguments (region boundary arrays routed to AlloMem).
  // Stores to kernel-LOCAL arrays are the design's own state (buf/occ/cred/...); those
  // are registers and are SUPPOSED to persist across steps, exactly as MatchLib's are.
  // (Checking for any store at all disables the transform on every stateful kernel.)
  bool storesToPort = false;
  func.walk([&](Operation *o) {
    Value target;
    if (auto st = dyn_cast<memref::StoreOp>(o))
      target = st.getMemRef();
    else if (auto st = dyn_cast<affine::AffineStoreOp>(o))
      target = st.getMemRef();
    else
      return;
    // Chase through view-like ops to the root definition.
    while (auto *def = target.getDefiningOp()) {
      if (def->getNumOperands() == 0)
        break;
      if (!isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp>(def))
        break;
      target = def->getOperand(0);
    }
    if (isa<BlockArgument>(target))
      storesToPort = true;   // a func argument == a memory port
  });
  if (storesToPort)
    return false;

  // A kernel that LOADS from a memory port INSIDE the loop is streaming FINITE data out
  // of that array. Its termination lives in a local cursor over the port, not in the
  // loop counter, so a dead induction variable proves nothing about it being
  // free-running.
  //
  // Measured on EVA: rdrv_w/e/n/s walk `sp[r]` through `rdin_w[r, sp[r]]` and never
  // mention `t`, so they passed every other guard. Made free-running they keep injecting
  // neutral packets after the data is exhausted and the design never completes -- the
  // reported "stuck in the while loop". A sibling kernel `drv_w` escaped only by
  // accident, because it happens to write `t >= sp[r]`.
  //
  // Scoped to the loop BODY deliberately, not the whole function: `node` reads its
  // config `pcfg[0,0]` in the PROLOGUE, before the loop, and must KEEP the transform --
  // it is the kernel the pipelining win was measured on (23 -> 4 cycles/iteration).
  // Checking the whole function would disable the transform there and undo that.
  bool loadsFromPortInLoop = false;
  op.walk([&](Operation *o) {
    Value src;
    if (auto ld = dyn_cast<memref::LoadOp>(o))
      src = ld.getMemRef();
    else if (auto ld = dyn_cast<affine::AffineLoadOp>(o))
      src = ld.getMemRef();
    else
      return;
    while (auto *def = src.getDefiningOp()) {
      if (def->getNumOperands() == 0)
        break;
      if (!isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp>(def))
        break;
      src = def->getOperand(0);
    }
    if (isa<BlockArgument>(src))
      loadsFromPortInLoop = true;
  });
  return !loadsFromPortInLoop;
}

void SystemCModuleEmitter::emitAffineFor(affine::AffineForOp op) {  // override (base emitter)
  if (!isSteadyStateLoop(op)) {
    CatapultModuleEmitter::emitAffineFor(op);
    return;
  }
  // Header twice, body ONCE: both branches open exactly one brace.
  //
  // The preheader directives must be emitted INSIDE each branch, immediately above that
  // branch's loop header -- NOT once before the `#ifdef`. A Catapult pragma binds to the
  // NEXT construct, so with the `#ifdef` line (and the `done.write` below) in between it
  // binds to nothing and is dropped SILENTLY: no CIN-203 acknowledgement, and therefore
  // no CIN-319 "cannot bind pragma" either. That made `s.pipeline()` a no-op on every
  // steady-state kernel -- measured on EVA: only the 8 collector loops (which have no
  // `#ifdef` between pragma and header) were pipelined, while the PE node and all 8
  // drivers were skipped. Placing it correctly lets the elastic PE schedule at II=4:
  // 23 -> 4 cycles/iteration, -5.3 % area, and it meets 2.0 ns where unscheduled misses.
  os << "#ifdef __SYNTHESIS__\n";
  // This loop never exits, so the `done.write(true)` that emitFunction places AFTER the
  // body is UNREACHABLE in RTL -- and the region's _agg_done ANDs every kernel's done,
  // so one steady-state kernel keeps the whole region's `done` low forever. A
  // free-running kernel has no "finished"; the honest RTL signal is "running", so raise
  // it on entry. csim takes the #else branch and still completes normally.
  if (auto pf = op->getParentOfType<func::FuncOp>())
    if (pf->hasAttr("df.kernel")) {
      indent();
      os << "done.write(true);  // steady-state: no completion, so assert on entry "
            "(the post-body write is unreachable here)\n";
    }
  emitLoopDirectivesPreheader(op); // must sit directly above the header (see above)
  indent();
  os << "while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step\n";
  os << "#else\n";
  emitLoopDirectivesPreheader(op); // and again for the csim header
  indent();
  os << "l_steady: for (";   // emitValue emits the type on first use
  emitValue(op.getInductionVar(), 0, false, "t");
  os << " = " << op.getConstantLowerBound() << "; ";
  emitValue(op.getInductionVar(), 0, false, "t");
  os << " < " << op.getConstantUpperBound() << "; ";
  emitValue(op.getInductionVar(), 0, false, "t");
  os << " += " << op.getStep() << ") {\n";
  os << "#endif\n";
  addIndent();
  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  // csim only: an SC_THREAD does not yield on its own, so a body issuing non-blocking
  // stream ops needs a per-iteration wait() or peer threads never run. Under synthesis
  // the PushNB/PopNB handshake supplies the cycle boundary.
  os << "#ifndef __SYNTHESIS__\n";
  indent();
  os << "wait();\n";
  os << "#endif\n";
  reduceIndent();
  indent();
  os << "}\n";
}

// Local reimplementation of the base's file-local affine-expr emitter: walk the
// expr, resolving dim/symbol positions to their operand SSA names via emitValue.
void SystemCModuleEmitter::emitAffineExprSC(AffineExpr e, ValueRange operands,
                                            unsigned numDims) {  // new (SystemC-only)
  switch (e.getKind()) {
  case AffineExprKind::Constant:
    os << llvm::cast<AffineConstantExpr>(e).getValue();
    return;
  case AffineExprKind::DimId:
    emitValue(operands[llvm::cast<AffineDimExpr>(e).getPosition()]);
    return;
  case AffineExprKind::SymbolId:
    emitValue(operands[numDims + llvm::cast<AffineSymbolExpr>(e).getPosition()]);
    return;
  default:
    break;
  }
  auto bin = llvm::cast<AffineBinaryOpExpr>(e);
  if (e.getKind() == AffineExprKind::CeilDiv) {
    os << "((";
    emitAffineExprSC(bin.getLHS(), operands, numDims);
    os << " + ";
    emitAffineExprSC(bin.getRHS(), operands, numDims);
    os << " - 1) / ";
    emitAffineExprSC(bin.getRHS(), operands, numDims);
    os << ")";
    return;
  }
  const char *opstr = " + ";
  switch (e.getKind()) {
  case AffineExprKind::Add: opstr = " + "; break;
  case AffineExprKind::Mul: opstr = " * "; break;
  case AffineExprKind::Mod: opstr = " % "; break;
  case AffineExprKind::FloorDiv: opstr = " / "; break;
  default: assert(false && "unexpected affine expr kind"); break;
  }
  os << "(";
  emitAffineExprSC(bin.getLHS(), operands, numDims);
  os << opstr;
  emitAffineExprSC(bin.getRHS(), operands, numDims);
  os << ")";
}

// Row-major flatten of a (multi-dim) affine index -> one C++ expression.
void SystemCModuleEmitter::emitFlatIndexCore(AffineMap map,
                                             ArrayRef<int64_t> shape,
                                             ValueRange operands) {  // new (SystemC-only)
  unsigned n = map.getNumResults();
  SmallVector<int64_t> stride(n);
  int64_t s = 1;
  for (int k = (int)n - 1; k >= 0; --k) {
    stride[k] = s;
    s *= shape[k];
  }
  os << "(";
  for (unsigned k = 0; k < n; ++k) {
    if (k)
      os << " + ";
    os << "(";
    emitAffineExprSC(map.getResult(k), operands, map.getNumDims());
    os << ")";
    if (stride[k] != 1)
      os << " * " << stride[k];
  }
  os << ")";
}
void SystemCModuleEmitter::emitFlatIndex(affine::AffineLoadOp op) {  // new (SystemC-only)
  auto mt = llvm::cast<MemRefType>(op.getMemRef().getType());
  SmallVector<Value> operands(op.getMapOperands().begin(),
                              op.getMapOperands().end());
  emitFlatIndexCore(op.getAffineMap(), mt.getShape(), operands);
}
void SystemCModuleEmitter::emitFlatIndex(affine::AffineStoreOp op) {  // new (SystemC-only)
  auto mt = llvm::cast<MemRefType>(op.getMemRef().getType());
  SmallVector<Value> operands(op.getMapOperands().begin(),
                              op.getMapOperands().end());
  emitFlatIndexCore(op.getAffineMap(), mt.getShape(), operands);
}

// Row-major flatten of direct index Values (memref dialect): Σ idx[k]*stride[k].
// TODO: REVISIT — near-exact duplicate of emitFlatIndexCore (only the per-term emission
// differs: raw getName here vs emitAffineExprSC there). Fold both into one stride+sum helper
// with a per-term callback, as emitMemPortLoad/Store already do for the index.
void SystemCModuleEmitter::emitFlatIndexMemref(ValueRange indices,
                                               ArrayRef<int64_t> shape) {  // new (SystemC-only)
  unsigned n = indices.size();
  SmallVector<int64_t> stride(n);
  int64_t s = 1;
  for (int k = (int)n - 1; k >= 0; --k) {
    stride[k] = s;
    s *= shape[k];
  }
  os << "(";
  for (unsigned k = 0; k < n; ++k) {
    if (k)
      os << " + ";
    os << "(" << std::string(getName(indices[k]).str()) << ")";
    if (stride[k] != 1)
      os << " * " << stride[k];
  }
  os << ")";
}

// LOAD from a random-access memory port -> the RAM-pin accessor:
//   result = <arr>_rd(addr);      (flat index emitted by `emitIdx`)
// The accessor itself is generated per array in emitKernelModule and carries
// `#pragma design modulario`; see there for why the read waits twice.
void SystemCModuleEmitter::emitMemPortLoad(Value memref, Value result,
                                           bool isUnsigned,
                                           llvm::function_ref<void()> emitIdx) {  // new (SystemC-only)
  fixUnsignedType(result, isUnsigned);
  auto mt = llvm::cast<MemRefType>(memref.getType());
  int64_t total = 1;
  for (auto d : mt.getShape())
    total *= d;
  std::string aT = "ac_int<" + std::to_string(scAddrW(total)) + ", false>";
  auto nm = getName(memref);
  indent();
  emitValue(result);
  os << ";\n";
  // One call to the generated accessor; the pin sequence and its two clock edges live
  // in <arr>_rd() (emitKernelModule), which carries `#pragma design modulario`.
  indent();
  emitValue(result);
  os << " = " << nm << "_rd((" << aT << ")(";
  emitIdx();
  os << "));";
}

// STORE to a random-access memory port -> the RAM-pin accessor:
//   <arr>_wr(addr, value);        (flat index emitted by `emitIdx`)
void SystemCModuleEmitter::emitMemPortStore(Value memref, Value value,
                                            llvm::function_ref<void()> emitIdx) {  // new (SystemC-only)
  auto mt = llvm::cast<MemRefType>(memref.getType());
  int64_t total = 1;
  for (auto d : mt.getShape())
    total *= d;
  std::string aT = "ac_int<" + std::to_string(scAddrW(total)) + ", false>";
  auto nm = getName(memref);
  // One call to the generated accessor; the pin sequence lives in <arr>_wr().
  //
  // The value rides its own data pin at its natural type, so the float `_fbits()`
  // round-trip that the packed request used to need is gone -- there is no longer an
  // ac_int to squeeze a half/ac_ieee_float into.
  indent(); os << nm << "_wr((" << aT << ")(";
  emitIdx();
  os << "), ";
  emitValue(value);
  os << ");";
}

// memref.load: mem-port arg -> req/rsp; local %alloc array -> base emitter.
void SystemCModuleEmitter::emitLoad(memref::LoadOp op) {  // override (base emitter)
  if (char d = memPortArgDir(op.getMemRef()); d == 'i' || d == 'b') {
    auto mt = llvm::cast<MemRefType>(op.getMemRef().getType());
    emitMemPortLoad(op.getMemRef(), op.getResult(), op->hasAttr("unsigned"),
                    [&]() { emitFlatIndexMemref(op.getIndices(), mt.getShape()); });
    emitInfoAndNewLine(op);
    return;
  }
  VhlsModuleEmitter::emitLoad(op); // normal local-array load
}

// memref.store: mem-port arg -> req; local %alloc array -> base emitter.
void SystemCModuleEmitter::emitStore(memref::StoreOp op) {  // override (base emitter)
  if (char d = memPortArgDir(op.getMemRef()); d == 'o' || d == 'b') {
    auto mt = llvm::cast<MemRefType>(op.getMemRef().getType());
    emitMemPortStore(op.getMemRef(), op.getValueToStore(),
                     [&]() { emitFlatIndexMemref(op.getIndices(), mt.getShape()); });
    emitInfoAndNewLine(op);
    return;
  }
  VhlsModuleEmitter::emitStore(op); // normal local-array store
}

// For a region boundary arg, look up the kernel arg it feeds and return that
// kernel arg's memory-port direction (0 if it is a normal stream boundary).
char SystemCModuleEmitter::regArgMemPort(func::FuncOp top, Value regArg) {  // new (SystemC-only)
  auto parent = top->getParentOfType<ModuleOp>();
  for (auto &op : top.front())
    if (auto call = llvm::dyn_cast<func::CallOp>(&op))
      for (auto opnd : llvm::enumerate(call.getOperands()))
        if (opnd.value() == regArg) {
          auto callee = parent.lookupSymbol<func::FuncOp>(call.getCallee());
          if (callee)
            if (char d = memPortArgDir(callee.getArgument(opnd.index())))
              return d;
        }
  return 0;
}

// NEAR-COPY of VhlsModuleEmitter::emitValue (the Vivado base): identical structure, the
// ONLY change is routing the type name through getSCTypeName (-> Catapult ac_int/ac_fixed)
// instead of the base's file-local getTypeName (-> Xilinx ap_int/ap_fixed). Overridden
// (not extended) only because getTypeName isn't virtual, so the one line can't be swapped
// in place -- the same copy-to-swap-one-string pattern as Catapult vs Vivado getTypeName.
void SystemCModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                                     std::string name) {  // override (base emitter)
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getSCTypeName(val.getType()) << " ";

  if (name == "") {
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

// bitcast (e.g. fp16 <-> uint16 packing). A std::memcpy over a non-trivial IEEE
// float -- ac_ieee_float<binaryNN>, e.g. 'half' -- takes its address as a void*,
// which Catapult's synthesis front end rejects (CIN-71: "Invalid pointer cast from
// 'half *' to 'void *'"), aborting `go compile`. g++ csim accepts it, so it only
// surfaces at synthesis. When a fp16/fp32 is involved we therefore reinterpret via
// the float type's own bit accessors -- data_ac_int()/set_data() -- which synthesize
// (the same idiom the _fbits helper uses). The generic
// int<->int (and double) path keeps the memcpy, which is fine for trivial PODs.
void SystemCModuleEmitter::emitBitcast(arith::BitcastOp op) {  // override (base emitter)
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  Value operand = op.getOperand();
  fixUnsignedType(operand, op->hasAttr("unsigned"));
  Type resTy = result.getType(), opTy = operand.getType();
  // operand was emitted earlier, so its name is stable now; the RESULT name is only
  // assigned during emitValue(result) below, so capture `rn` AFTER that call.
  std::string on = std::string(getName(operand).str());

  // fp16/fp32 bit width (0 for double / non-float: not handled by the accessors).
  auto floatBits = [](Type t) -> unsigned {
    if (llvm::isa<Float16Type>(t))
      return 16;
    if (llvm::isa<Float32Type>(t))
      return 32;
    return 0;
  };
  unsigned resFB = floatBits(resTy), opFB = floatBits(opTy);

  // raw int bits -> fp16/fp32: set_data() loads the bit pattern (a value-cast would
  // reinterpret the number). ac_int<W,true>(operand) accepts a native or ac_int src.
  if (resFB && !llvm::isa<FloatType>(opTy)) {
    indent();
    emitValue(result);
    std::string rn = std::string(getName(result).str());
    os << "; " << rn << ".set_data(ac_int<" << resFB << ", true>(" << on << "));";
    emitInfoAndNewLine(op);
    return;
  }
  // fp16/fp32 -> raw int bits: _fbits() reads data_ac_int().to_uint() (synthesis-safe),
  // then narrow to the destination integer type.
  if (opFB && !llvm::isa<FloatType>(resTy)) {
    indent();
    emitValue(result);
    os << " = (" << getSCTypeName(resTy) << ")_fbits(" << on << ");";
    emitInfoAndNewLine(op);
    return;
  }

  // int<->int (or anything involving double): same-width memcpy over trivial PODs.
  // TODO: REVISIT -- a double (Float64) bitcast falls here and keeps the memcpy, which
  // would hit the same CIN-71 void*-cast rejection at synthesis that the fp16/fp32
  // branches avoid. Latent only: no current design bitcasts a double.
  indent();
  emitValue(result);
  std::string rn = std::string(getName(result).str());
  os << ";\n";
  indent();
  os << getSCTypeName(opTy) << " _bc_" << rn << " = " << on << ";\n";
  indent();
  os << "std::memcpy(&" << rn << ", &_bc_" << rn << ", sizeof(" << rn << "));";
  emitInfoAndNewLine(op);
}

// --- native ac_int bit ops (replace the Vitis ap_int proxy forms) ---

void SystemCModuleEmitter::emitGetBit(allo::GetIntBitOp op) {  // override (base emitter)
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  Value num = op.getNum();
  unsigned nw = num.getType().getIntOrFloatBitWidth();
  indent();
  emitValue(result); // declares "<T> <res>"
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  // num may be a native C int (int32_t) with no bit-index op -> copy into an
  // ac_int temp first, which always supports [i]/.slc/.set_slc.
  indent();
  os << "ac_int<" << nw << ", true> _bs_" << rn << " = ";
  emitValue(num);
  os << ";\n";
  indent();
  os << rn << " = _bs_" << rn << "[";
  emitValue(op.getIndex());
  os << "];";
  emitInfoAndNewLine(op);
}

void SystemCModuleEmitter::emitSetBit(allo::SetIntBitOp op) {  // override (base emitter)
  Value result = op.getResult();
  Value num = op.getNum();
  unsigned nw = num.getType().getIntOrFloatBitWidth();
  indent();
  emitValue(result); // "<T> <res>"
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << "ac_int<" << nw << ", true> _bs_" << rn << " = ";
  emitValue(num);
  os << ";\n";
  indent();
  os << "_bs_" << rn << "[";
  emitValue(op.getIndex());
  os << "] = ";
  emitValue(op.getVal());
  os << ";\n";
  indent();
  os << rn << " = _bs_" << rn << ";";
  emitInfoAndNewLine(op);
}

void SystemCModuleEmitter::emitGetSlice(allo::GetIntSliceOp op) {  // override (base emitter)
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  Value num = op.getNum();
  unsigned nw = num.getType().getIntOrFloatBitWidth();
  unsigned w = result.getType().getIntOrFloatBitWidth();
  // ac_int::slc<w>(lo) is a member fn; the emitted num may be a plain C int with
  // no .slc -> wrap it in an ac_int temp (mirror of emitSetSlice's set_slc case).
  indent();
  emitValue(result); // "<T> <res>"
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << "ac_int<" << nw << ", true> _bs_" << rn << " = ";
  emitValue(num);
  os << ";\n";
  indent();
  os << rn << " = _bs_" << rn << ".slc<" << w << ">(";
  emitValue(op.getLo());
  os << ");";
  emitInfoAndNewLine(op);
}

void SystemCModuleEmitter::emitSetSlice(allo::SetIntSliceOp op) {  // override (base emitter)
  Value result = op.getResult();
  Value num = op.getNum();
  unsigned nw = num.getType().getIntOrFloatBitWidth();
  // ac_int::set_slc(lo, val) requires an ac_int val (its width = #bits set);
  // the emitted val may be a plain C int -> wrap it in an ac_int of the val's
  // bit width so the right number of bits is written. The dst likewise needs an
  // ac_int temp (a native int32_t has no .set_slc).
  unsigned vw = op.getVal().getType().getIntOrFloatBitWidth();
  indent();
  emitValue(result); // "<T> <res>"
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << "ac_int<" << nw << ", true> _bs_" << rn << " = ";
  emitValue(num);
  os << ";\n";
  indent();
  os << "_bs_" << rn << ".set_slc(";
  emitValue(op.getLo());
  os << ", ac_int<" << vw << ", false>(";
  emitValue(op.getVal());
  os << "));\n";
  indent();
  os << rn << " = _bs_" << rn << ";";
  emitInfoAndNewLine(op);
}

// Sequential-stream read:  <result> = <port>.Pop();   (index ignored — in order)
void SystemCModuleEmitter::emitAffineLoad(affine::AffineLoadOp op) {  // override (base emitter)
  // Random-access INPUT ('i') or read+write ('b') memory port: LOAD via the read pins.
  if (char d = memPortArgDir(op.getMemRef()); d == 'i' || d == 'b') {
    emitMemPortLoad(op.getMemRef(), op.getResult(), op->hasAttr("unsigned"),
                    [&]() { emitFlatIndex(op); });
    emitInfoAndNewLine(op);
    return;
  }
  if (streamArgDir(op.getMemRef()) != 'i') {
    VhlsModuleEmitter::emitAffineLoad(op); // normal array load
    return;
  }
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op.getMemRef(), 0, false);
  os << ".Pop();";
  emitInfoAndNewLine(op);
}

// Sequential-stream write:  <port>.Push(<value>);   (index ignored — in order)
void SystemCModuleEmitter::emitAffineStore(affine::AffineStoreOp op) {  // override (base emitter)
  // Random-access OUTPUT ('o') or read+write ('b') memory port: STORE via the write
  // pins (no response; the memory applies it).
  if (char d = memPortArgDir(op.getMemRef()); d == 'o' || d == 'b') {
    emitMemPortStore(op.getMemRef(), op.getValueToStore(),
                     [&]() { emitFlatIndex(op); });
    emitInfoAndNewLine(op);
    return;
  }
  if (streamArgDir(op.getMemRef()) != 'o') {
    VhlsModuleEmitter::emitAffineStore(op); // normal array store
    return;
  }
  indent();
  emitValue(op.getMemRef(), 0, false);
  os << ".Push(";
  emitValue(op.getValueToStore());
  os << ");";
  emitInfoAndNewLine(op);
}

//===----------------------------------------------------------------------===//
// emitFunction split: kernel module vs top wiring module
//===----------------------------------------------------------------------===//

// fwd decl (defined near emitStreamEmpty): does func query empty()/full() on arg?
static bool streamArgQueried(func::FuncOp func, unsigned argIdx, bool wantEmpty);

void SystemCModuleEmitter::emitKernelModule(func::FuncOp func) {  // new (SystemC-only)
  auto name = func.getName();
  os << "SC_MODULE(" << name << ") {\n";
  addIndent();

  // Clock + reset (required for synthesizable clocked threads).
  indent(); os << "sc_in_clk clk;\n";
  indent(); os << "sc_in<bool> rst;\n";
  // Hardware completion flag: raised (synthesized logic, RTL-observable) after the
  // kernel's single pass. The top ANDs all kernels' `done` into a top-level output
  // port the tb polls -- a real done signal that works in RTL cosim, unlike the
  // C-side __allo_done counter (which the synthesized DUT can't increment, forcing
  // the tb to burn a huge fixed completion cap every cosim).
  indent(); os << "sc_out<bool> done;\n";

  // Ports + members from arguments.
  SmallVector<std::string, 4> streamPorts;
  // sc_out wire ports needing a reset-action write (Catapult CIN-233), each paired
  // with the ZERO EXPRESSION for its type. A plain `0` is wrong for a float payload:
  // sc_out<ac_ieee_float<binary32>>::write takes const T& and there is no implicit
  // int->T conversion, so `.write(0)` fails to compile. (The deleted AlloMemW used
  // _mem_decode<T>(0) for exactly this reason.)
  SmallVector<std::pair<std::string, std::string>, 4> wireOutPorts;
  // Zero literal for a payload type: floats need an explicit construction.
  auto zeroOf = [&](Type et, const std::string &tn) -> std::string {
    return llvm::isa<FloatType>(et) ? (tn + "(0.0f)") : std::string("0");
  };
  // valid_only channel ports: (portName, payloadType, dir). Each gets a pair of
  // modulario-annotated accessor methods, mirroring how Connections implements
  // Push/PushNB/Pop/PopNB -- see emitValidOnlyAccessors.
  SmallVector<std::tuple<std::string, std::string, char>, 4> vonlyPorts;
  // Memory-pin arrays: {port name, address type, data type, dir}. Drives the generated
  // _rd/_wr accessors below -- the RAM pin sequence must live in a METHOD so it can
  // carry `#pragma design modulario`, which is what makes Catapult treat it as a
  // cycle-accurate interface instead of signal writes it may schedule freely.
  SmallVector<std::tuple<std::string, std::string, std::string, char>, 4> memPins;
  // Self-FIFO base name + depth: this kernel maintains a synchronous occupancy
  // counter per self-FIFO so empty()/full() read the LOGICAL fill (put -> ++,
  // get -> --) instead of the clocked AlloFifo's handshake, which lags by a cycle
  // and made empty()/full() read stale in RTL cosim (off vs the functional sim).
  SmallVector<std::pair<std::string, int64_t>, 4> localFifos;
  for (auto arg : llvm::enumerate(func.getArguments())) {
    unsigned i = arg.index();
    Value v = arg.value();
    // Duplicate stream/channel arg: reuse the primary arg's port (already named,
    // since it sits at a lower index). No new port, no reset, no binding.
    if (Value primary = aliasOf(v)) {
      state.nameTable[v] = getName(primary);
      continue;
    }
    indent();
    if (auto st = llvm::dyn_cast<StreamType>(v.getType())) {
      std::string pn = std::string(addName(v, /*isPtr=*/false).str());
      if (isLocalStream(v)) {
        // self-FIFO (one kernel both produces AND queries it): realized as a
        // bounded MatchLib AlloFifo wired as a self-loop at the top. This one
        // kernel therefore needs BOTH ends -- an Out (enq: put/try_put/full) and
        // an In (deq: get/try_get/empty). A bounded FIFO makes Full()/Empty()
        // correct, unlike an unbounded ac_channel.
        std::string T = std::string(getStreamPayloadTypeName(st.getBaseType(), linkPayloadUnsigned(v)).str());
        std::string en = pn + "_enq", dq = pn + "_deq";
        streamPorts.push_back(en);
        streamPorts.push_back(dq);
        localFifos.push_back({pn, st.getDepth()});
        os << "Connections::Out< " << T << " > " << en << ";\n";
        indent();
        os << "Connections::In< " << T << " > " << dq << ";\n";
      } else {
        // stream arg -> Connections::In/Out<T> port (direction from stypes)
        char d = streamDir(func, i);
        streamPorts.push_back(pn);
        os << (d == 'o' ? "Connections::Out< " : "Connections::In< ");
        os << getStreamPayloadTypeName(st.getBaseType(), linkPayloadUnsigned(v)) << " > " << pn << ";\n";
        // Occupancy sideband inputs: added ONLY where this kernel actually queries
        // empty()/full() (gate on usage, not direction -- a producer may query
        // full() on its Out-stream), and only for buffered streams (depth>=1) that
        // have an AlloFifo to source them. sc_in<bool> needs no Reset().
        if (st.getDepth() != 0) {
          if (streamArgQueried(func, i, /*wantEmpty=*/true)) {
            indent();
            os << "sc_in<bool> " << pn << "_empty;\n";
          }
          if (streamArgQueried(func, i, /*wantEmpty=*/false)) {
            indent();
            os << "sc_in<bool> " << pn << "_full;\n";
          }
        }
      }
    } else if (auto ct = llvm::dyn_cast<ChannelType>(v.getType())) {
      char d = streamDir(func, i);
      std::string pn = std::string(addName(v, /*isPtr=*/false).str());
      std::string T = std::string(
          getStreamPayloadTypeName(ct.getBaseType(), linkPayloadUnsigned(v)).str());
      if (isValidOnlyChannel(v)) {
        // valid_only -> raw data + valid ports, NO ready and NO Connections
        // transactor. Not added to streamPorts: sc_in/sc_out have no .Reset().
        // A driven sc_out must still be written in the reset action (CIN-233), so
        // an OUT bundle's two ports are tracked in wireOutPorts like a Wire's.
        if (d == 'o') {
          // A float-payload valid_only channel hits the same int->T problem.
          std::string vzero = zeroOf(ct.getBaseType(), T);
          wireOutPorts.push_back({pn + "_dat", vzero});
          wireOutPorts.push_back({pn + "_vld", "0"});
        }
        os << (d == 'o' ? "sc_out< " : "sc_in< ") << T << " > " << pn << "_dat;\n";
        indent();
        os << (d == 'o' ? "sc_out<bool> " : "sc_in<bool> ") << pn << "_vld;\n";
        vonlyPorts.push_back({pn, T, d});
      } else {
        // valid_ready -> Connections::In/Out<T> port (combinational, no buffer)
        streamPorts.push_back(pn);
        os << (d == 'o' ? "Connections::Out< " : "Connections::In< ");
        os << T << " > " << pn << ";\n";
      }
    } else if (auto wt = llvm::dyn_cast<WireType>(v.getType())) {
      // wire arg -> raw sc_in/sc_out<T> port (combinational, no handshake).
      // NOT added to streamPorts: sc ports have no Connections .Reset(). A driven
      // sc_out must still be set in the reset action (Catapult CIN-233), so an
      // OUT wire is tracked separately in wireOutPorts.
      char d = streamDir(func, i);
      std::string pn = std::string(addName(v, /*isPtr=*/false).str());
      if (d == 'o')
        wireOutPorts.push_back({pn, "0"});
      os << (d == 'o' ? "sc_out< " : "sc_in< ");
      os << getStreamPayloadTypeName(wt.getBaseType(), linkPayloadUnsigned(v)) << " > " << pn << ";\n";
    } else if (auto mt = llvm::dyn_cast<MemRefType>(v.getType())) {
      char d = argDir(func, i);
      if ((d == 'i' || d == 'o') && isSeqStreamable(v) && !forceMemPort(v)) {
        // sequential-stream: boundary array arg -> Connections stream port
        // (body's a[i]/b[i]=v become .Pop()/.Push() via the affine overrides)
        std::string pn = std::string(addName(v, /*isPtr=*/false).str());
        streamPorts.push_back(pn);
        os << (d == 'o' ? "Connections::Out< " : "Connections::In< ");
        os << getStreamPayloadTypeName(mt.getElementType(), linkPayloadUnsigned(v)) << " > " << pn << ";\n";
      } else if (d == 'i' || d == 'b') {
        // random-access INPUT ('i') or read+write ('b') array -> RAM pins. 'i' gets
        // the read bundle; 'b' additionally gets the write bundle below, because it
        // both loads and stores through the same boundary array.
        std::string pn = std::string(addName(v, /*isPtr=*/false).str());
        int64_t total = 1;
        for (auto s : mt.getShape())
          total *= s;
        // RAM PINS, not a Connections req/rsp pair: address+enable out, data in.
        // Matches Catapult's ccs_ramifc_w_handshake_r and the Vitis path's
        // _rsc_radr/_re/_q, so the boundary is a memory an integrator recognises.
        // These are raw sc_ports: no .Reset(), so the driven ones are registered in
        // wireOutPorts, which writes them in the reset action (Catapult CIN-233).
        std::string aT = "ac_int<" + std::to_string(scAddrW(total)) + ", false>";
        std::string dT = std::string(getStreamPayloadTypeName(
            mt.getElementType(), linkPayloadUnsigned(v)).str());
        wireOutPorts.push_back({pn + "_radr", "0"});
        wireOutPorts.push_back({pn + "_re", "0"});
        os << "sc_out< " << aT << " > " << pn << "_radr;\n";
        indent(); os << "sc_out<bool> " << pn << "_re;\n";
        indent(); os << "sc_in< " << dT << " > " << pn << "_q;\n";
        // _rrdy: the memory's "I can serve you" line. INPUT, so it is not in
        // wireOutPorts (which exists to drive sc_outs in the reset action).
        indent(); os << "sc_in<bool> " << pn << "_rrdy;\n";
        if (d == 'b') { // read+write: add the write side of the same memory
          std::string dzero = zeroOf(mt.getElementType(), dT);
          wireOutPorts.push_back({pn + "_wadr", "0"});
          wireOutPorts.push_back({pn + "_d", dzero});
          wireOutPorts.push_back({pn + "_we", "0"});
          indent(); os << "sc_out< " << aT << " > " << pn << "_wadr;\n";
          indent(); os << "sc_out< " << dT << " > " << pn << "_d;\n";
          indent(); os << "sc_out<bool> " << pn << "_we;\n";
          indent(); os << "sc_in<bool> " << pn << "_wrdy;\n";
        }
        memPins.push_back({pn, aT, dT, d});
      } else if (d == 'o') {
        // random-access OUTPUT array -> write-only RAM pins. Body stores become
        // <arr>_wr(addr, val) via the affine store override.
        std::string pn = std::string(addName(v, /*isPtr=*/false).str());
        int64_t total = 1;
        for (auto s : mt.getShape())
          total *= s;
        // RAM PINS (write side), mirroring ccs_ramifc_w_handshake_w: address, data
        // and enable all OUT. No data comes back, so there is no `q` here -- the
        // write-only case is genuinely 3 pins, not a truncated read port.
        std::string aT = "ac_int<" + std::to_string(scAddrW(total)) + ", false>";
        std::string dT = std::string(getStreamPayloadTypeName(
            mt.getElementType(), linkPayloadUnsigned(v)).str());
        std::string dzero = zeroOf(mt.getElementType(), dT);
        wireOutPorts.push_back({pn + "_wadr", "0"});
        wireOutPorts.push_back({pn + "_d", dzero});
        wireOutPorts.push_back({pn + "_we", "0"});
        os << "sc_out< " << aT << " > " << pn << "_wadr;\n";
        indent(); os << "sc_out< " << dT << " > " << pn << "_d;\n";
        indent(); os << "sc_out<bool> " << pn << "_we;\n";
        indent(); os << "sc_in<bool> " << pn << "_wrdy;\n";
        memPins.push_back({pn, aT, dT, d});
      } else {
        // non-directional memref -> internal array member (fallback)
        os << getSCTypeName(mt.getElementType()) << " " << addName(v, /*isPtr=*/false);
        for (auto s : mt.getShape())
          os << "[" << s << "]";
        os << ";\n";
      }
    }
  }

  // Stateful variables (`x: T @ Stateful`) as MODULE MEMBERS, initialised in the reset
  // action below. Not locals in run(): an SC_THREAD body runs on a coroutine stack of
  // only ~64KB, and a large stateful array would overflow it and segfault at run time
  // with no useful message (the same trap the const-array block documents -- test_mlp's
  // 128KB weight array). Members also can't be `static`, which would share the state
  // across every instance of this module and skip the reset. This is the shape MatchLib
  // uses for its own state, and Catapult registers it identically.
  llvm::SmallVector<memref::GlobalOp, 4> statefulGlobals;
  func.walk([&](memref::GetGlobalOp gg) {
    auto g = gg->getParentOfType<ModuleOp>()
                 .lookupSymbol<memref::GlobalOp>(gg.getName());
    if (!g || !isStatefulGlobal(g) || g->hasAttr("constant"))
      return;
    for (auto &e : statefulGlobals)
      if (e.getSymName() == g.getSymName())
        return;
    statefulGlobals.push_back(g);
  });
  for (auto &g : statefulGlobals) {
    auto at = llvm::cast<ShapedType>(g.getType());
    fixUnsignedType(g, g->hasAttr("unsigned"));
    indent();
    emitStatefulGlobalElementType(at.getElementType());
    os << " " << g.getSymName();
    for (auto &s : at.getShape())
      os << "[" << s << "]";
    os << ";  // @ Stateful\n";
  }

  // Constructor: name the ports + register a clocked, reset-aware thread.
  indent(); os << "SC_HAS_PROCESS(" << name << ");\n";
  indent(); os << name << "(sc_module_name n) : sc_module(n), done(\"done\")";
  for (auto &pn : streamPorts)
    os << ", " << pn << "(\"" << pn << "\")";
  os << " {\n";
  addIndent();
  indent(); os << "SC_THREAD(run);\n";
  indent(); os << "sensitive << clk.pos();\n";
  indent(); os << alloResetFn() << "(rst, false);\n";
  reduceIndent();
  indent(); os << "}\n";

  // valid_only accessors. These MIRROR the Connections reference implementation
  // (connections-guide.pdf Listing 3, and the 106 `modular IO` sites in
  // connections.h) with the ready line deleted:
  //
  //   Connections Push : do { val=1; msg=m; wait(); } while (!rdy);  val=0;
  //   valid_only put   :      vld=1; dat=m; wait();                  vld=0;
  //   Connections Pop  : do { rdy=1; wait(); } while (!val); rdy=0; return msg;
  //   valid_only get   : do {        wait(); } while (!vld);         return dat;
  //
  // Two details are load-bearing and were WRONG in a first attempt:
  //  * the wait() comes BEFORE the valid test, and the payload is sampled AFTER
  //    the edge at which valid was seen true -- that edge IS the transaction
  //    ("a transaction is valid when at the clock edge both valid and ready are
  //    true"; with no ready, valid alone decides). Testing before waiting and
  //    adding a trailing wait samples a cycle early and costs an extra state.
  //  * `#pragma design modulario` is what makes Catapult treat these as a
  //    cycle-accurate LI interface instead of ordinary signal accesses it may
  //    schedule freely. Every Connections port method carries it; ours must too.
  // RAM-pin accessors. Same reasoning as the valid_only ones above: the sequence has
  // to be a METHOD carrying `#pragma design modulario`, or Catapult is free to move the
  // signal writes around and the fixed read latency stops holding.
  for (auto &mp : memPins) {
    const std::string &pn = std::get<0>(mp);
    const std::string &aT = std::get<1>(mp);
    const std::string &dT = std::get<2>(mp);
    char d = std::get<3>(mp);
    if (d == 'i' || d == 'b') {
      indent(); os << "#pragma design modulario <in>\n";
      indent(); os << dT << " " << pn << "_rd(" << aT << " addr) {\n";
      indent(); os << "  " << pn << "_radr.write(addr); " << pn << "_re.write(true);\n";
      // FIXED two-edge access. A stall loop here (`do { wait(); } while (!_rrdy)`) was
      // tried and does NOT synthesize: a modulario method is a fixed-protocol C-CORE, and
      // a data-dependent loop inside it fails with ASM-2 / BASIC-25. See EmitSystemC.md.
      indent(); os << "  wait();                    // edge N: address captured\n";
      indent(); os << "  " << pn << "_re.write(false);\n";
      indent(); os << "  wait();                    // data valid on this edge\n";
      indent(); os << "  return " << pn << "_q.read();\n";
      indent(); os << "}\n";
    }
    if (d == 'o' || d == 'b') {
      indent(); os << "#pragma design modulario <out>\n";
      indent(); os << "void " << pn << "_wr(" << aT << " addr, " << dT << " val) {\n";
      indent(); os << "  " << pn << "_wadr.write(addr); " << pn << "_d.write(val);\n";
      indent(); os << "  " << pn << "_we.write(true);\n";
      // FIXED single edge, for the same reason as the read above.
      indent(); os << "  wait();\n";
      indent(); os << "  " << pn << "_we.write(false);\n";
      indent(); os << "}\n";
    }
  }
  for (auto &vp : vonlyPorts) {
    const std::string &pn = std::get<0>(vp);
    const std::string &T = std::get<1>(vp);
    char d = std::get<2>(vp);
    if (d == 'o') {
      indent(); os << "#pragma design modulario <out>\n";
      indent(); os << "void " << pn << "_put(const " << T << " &m) {\n";
      indent(); os << "  " << pn << "_vld.write(true); " << pn << "_dat.write(m);\n";
      indent(); os << "  wait();\n";
      indent(); os << "  " << pn << "_vld.write(false);\n";
      indent(); os << "}\n";
      // A valid_only put can never be refused (no ready line), so the
      // non-blocking form is the blocking form that always reports success.
      indent(); os << "#pragma design modulario <out>\n";
      indent(); os << "bool " << pn << "_try_put(const " << T << " &m) {\n";
      indent(); os << "  " << pn << "_put(m); return true;\n";
      indent(); os << "}\n";
    } else {
      indent(); os << "#pragma design modulario <in>\n";
      indent(); os << T << " " << pn << "_get() {\n";
      indent(); os << "  do { wait(); } while (" << pn << "_vld.read() != true);\n";
      indent(); os << "  return " << pn << "_dat.read();\n";
      indent(); os << "}\n";
      // Mirrors PopNB: one edge is consumed whether or not a datum was there.
      // TODO: REVISIT -- m is written before the vld check, so a caller that
      // ignores the returned bool reads a stale datum. This matches PopNB's "val
      // is undefined when ok is false" contract, but is a silent trap; consider
      // leaving m untouched on a miss once callers are audited.
      indent(); os << "#pragma design modulario <in>\n";
      indent(); os << "bool " << pn << "_try_get(" << T << " &m) {\n";
      indent(); os << "  wait();\n";
      indent(); os << "  m = " << pn << "_dat.read();\n";
      indent(); os << "  return " << pn << "_vld.read();\n";
      indent(); os << "}\n";
    }
  }

  // run(): reset ports, wait, then free-running loop over the REUSED body.
  indent(); os << "void run() {\n";
  addIndent();
  for (auto &pn : streamPorts) {
    indent(); os << pn << ".Reset();\n";
  }
  // Self-FIFO occupancy counters (see localFifos): synchronous fill tracked by
  // this kernel so empty()/full() match the functional simulator in RTL cosim.
  for (auto &lf : localFifos) {
    indent(); os << "int " << lf.first << "_cnt = 0;\n";
  }
  // Stateful variables: INITIALISE the members declared above. This is the reset
  // action, so the RTL reset re-establishes the initial value exactly as csim does --
  // the property that makes cosim meaningful. Uses need no rewriting: the base
  // emitGetGlobal binds the GetGlobalOp's SSA result to the global's SYMBOL NAME, so
  // loads and stores in the body already name this member.
  for (auto &g : statefulGlobals) {
    // Without an initial value there is nothing to reset to, and the body would run on
    // an undefined register. emitGlobal used to swallow this case silently.
    auto init = g.getInitialValue();
    if (!init.has_value()) {
      g.emitError("stateful variable `")
          << g.getSymName()
          << "` has no initial value, so its register cannot be reset; the kernel body "
             "would read undefined state";
      state.encounteredError = true;
      return;
    }
    auto dense = llvm::dyn_cast<DenseElementsAttr>(init.value());
    if (!dense) {
      g.emitError("stateful variable `")
          << g.getSymName() << "` has a non-dense initial value, which is unsupported";
      state.encounteredError = true;
      return;
    }
    auto at = llvm::cast<ShapedType>(g.getType());
    fixUnsignedType(g, g->hasAttr("unsigned"));
    // The frontend only allows a single scalar initialiser (`x: T[N] @ Stateful = 0`),
    // so the attribute is a splat and one loop nest resets the whole array. Emitting an
    // assignment per element instead would put thousands of statements in the reset
    // action of any sizeable buffer.
    if (dense.isSplat()) {
      unsigned rank = at.getRank();
      for (unsigned d = 0; d < rank; ++d) {
        indent();
        os << "for (int _sr" << d << " = 0; _sr" << d << " < " << at.getShape()[d]
           << "; ++_sr" << d << ") {\n";
        addIndent();
      }
      indent();
      os << g.getSymName();
      for (unsigned d = 0; d < rank; ++d)
        os << "[_sr" << d << "]";
      os << " = ";
      emitDenseElementLiteral(dense.getSplatValue<Attribute>(), at.getElementType(),
                              g->hasAttr("unsigned"));
      os << ";\n";
      for (unsigned d = 0; d < rank; ++d) {
        reduceIndent();
        indent();
        os << "}\n";
      }
    } else {
      // Not reachable from the current frontend; kept correct rather than silently wrong.
      SmallVector<int64_t> idx(at.getRank(), 0);
      for (auto element : dense.getValues<Attribute>()) {
        indent();
        os << g.getSymName();
        for (int64_t i : idx)
          os << "[" << i << "]";
        os << " = ";
        emitDenseElementLiteral(element, at.getElementType(), g->hasAttr("unsigned"));
        os << ";\n";
        for (int d = (int)at.getRank() - 1; d >= 0; --d) {
          if (++idx[d] < at.getShape()[d])
            break;
          idx[d] = 0;
        }
      }
    }
  }
  // Raw sc_out wire ports must be driven in the reset action (Catapult CIN-233).
  for (auto &wp : wireOutPorts) {
    indent(); os << wp.first << ".write(" << wp.second << ");\n";
  }
  indent(); os << "done.write(false);  // completion flag low until the pass finishes\n";
  indent(); os << "wait();\n";
  // Baked-in constant arrays (e.g. `W: T[M,N] = np_W` weights) referenced by this
  // kernel: a memref.global holds the data and a GetGlobalOp aliases it, but the
  // SystemC path (unlike Vhls emitFunction) never emitted the global itself, so the
  // body's reads of `W` were undefined at synthesis. Emit each such const array as
  // a local `[static] const T W[...] = {...}` before the body reads it. Stateful
  // (__stateful_) globals are WRITTEN, so they are declared in the reset action above
  // instead -- `static const` here would be both shared and unwritable.
  {
    llvm::SmallVector<memref::GlobalOp, 4> constGlobals;
    func.walk([&](memref::GetGlobalOp gg) {
      auto g = gg->getParentOfType<ModuleOp>()
                   .lookupSymbol<memref::GlobalOp>(gg.getName());
      if (!g || !g.getInitialValue().has_value())
        return;
      // Stateful globals: handled by the reset-action block above.
      if (isStatefulGlobal(g) && !g->hasAttr("constant"))
        return;
      for (auto &e : constGlobals)
        if (e.getSymName() == g.getSymName())
          return;
      constGlobals.push_back(g);
    });
    for (auto &g : constGlobals) {
      // Emit as `static const`. These are read-only baked-in constants (e.g.
      // `W: T[M,N] = np_W` weights). Plain locals live on the SC_THREAD
      // coroutine stack, which is small (~64KB); a large weight array overflows
      // it and segfaults at run time (test_mlp: W0[256][128] = 128KB crashes
      // linear1_0::run in the initializer). `static` moves it to static
      // storage; `const` is correct (never written) and lets multiple kernel
      // instances share one copy. It also synthesizes as a ROM under Catapult.
      // (Reuses emitGlobal's static/const attr hooks; attrs restored after.)
      bool hadStatic = g->hasAttr("static");
      bool hadConst = g->hasAttr("constant");
      if (!hadStatic)
        g->setAttr("static", UnitAttr::get(g->getContext()));
      if (!hadConst)
        g->setAttr("constant", UnitAttr::get(g->getContext()));
      emitGlobal(g);
      if (!hadStatic)
        g->removeAttr("static");
      if (!hadConst)
        g->removeAttr("constant");
    }
  }
  // Single-shot: run the body EXACTLY ONCE, then idle. Free-running (while(1)
  // around the body) is safe for stream kernels (they re-block on an empty input
  // after one pass) but WRONG for a read-modify-write `both` memory accumulator
  // (C[i]+=... re-accumulates every pass). Running once is correct for both, and
  // lets the tb read memory outputs after a real completion instead of guessing a
  // settle time. The idle `while(1) wait()` keeps the clocked thread alive.
  emitBlock(func.front()); // put/get now emit .Push()/.Pop()
  indent(); os << "done.write(true);  // RTL-observable completion (see the done port)\n";
  os << "#ifndef __SYNTHESIS__\n";
  indent(); os << "__allo_done++; // csim: this kernel finished its single pass\n";
  os << "#endif\n";
  indent(); os << "while (1) { wait(); }\n";
  reduceIndent();
  indent(); os << "}\n";

  reduceIndent();
  os << "};\n\n";
}

// Wire get: <result> = <wire>.read();  (raw combinational, no handshake)
void SystemCModuleEmitter::emitWireGet(WireGetOp op) {  // override (base emitter)
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  indent();
  emitValue(result);
  os << " = ";
  emitValue(op->getOperand(0), 0, false);
  os << ".read();";
  emitInfoAndNewLine(op);
}

// Wire put: <wire>.write(<value>);  (raw combinational, no handshake)
void SystemCModuleEmitter::emitWirePut(WirePutOp op) {  // override (base emitter)
  indent();
  emitValue(op->getOperand(0), 0, false);
  os << ".write(";
  emitValue(op->getOperand(1));
  os << ");";
  emitInfoAndNewLine(op);
}

// Connections channel get: <result> = <channel>.Pop();  (scalar handshake link)
void SystemCModuleEmitter::emitChannelGet(ChannelGetOp op) {  // override (base emitter)
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  if (isValidOnlyChannel(op->getOperand(0))) {
    // valid_only blocking get: there is no ready line to assert, so "blocking"
    // means spin until the producer's one-cycle valid pulse is visible, then
    // sample the data wire. The wait() is what makes the spin advance time.
    std::string pn = std::string(getName(op->getOperand(0)).str());
    indent();
    emitValue(result);
    os << " = " << pn << "_get();";
    emitInfoAndNewLine(op);
    return;
  }
  indent();
  emitValue(result);
  os << " = ";
  emitValue(op->getOperand(0), 0, false);
  os << ".Pop();";
  emitInfoAndNewLine(op);
}

// Connections channel put: <channel>.Push(<value>);  (scalar handshake link)
void SystemCModuleEmitter::emitChannelPut(ChannelPutOp op) {  // override (base emitter)
  if (isValidOnlyChannel(op->getOperand(0))) {
    // valid_only put: drive the data, pulse valid for exactly ONE cycle, then
    // drop it. No ready line exists, so the put can never be refused and never
    // blocks -- it costs one cycle unconditionally. Dropping valid after the
    // cycle is what keeps two consecutive puts distinguishable; leaving it high
    // would make one datum look like many.
    std::string pn = std::string(getName(op->getOperand(0)).str());
    indent();
    os << pn << "_put(";
    emitValue(op->getOperand(1));
    os << ");";
    emitInfoAndNewLine(op);
    return;
  }
  indent();
  emitValue(op->getOperand(0), 0, false);
  os << ".Push(";
  emitValue(op->getOperand(1));
  os << ");";
  emitInfoAndNewLine(op);
}

// Non-blocking channel get: <result>; <success> = <channel>.PopNB(<result>);
// Same Connections NB primitive the Stream try_get uses -- a channel lowers to a
// Connections::Combinational just as a stream does, so PopNB applies unchanged.
// (A Wire has no non-blocking form and never reaches here: the frontend rejects
// wire.try_get(), since a wire is always its current value.)
void SystemCModuleEmitter::emitChannelTryGet(ChannelTryGetOp op) {  // override (base emitter)
  Value result = op.getResult(0);
  Value success = op.getResult(1);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto channel = op->getOperand(0);
  if (isValidOnlyChannel(channel)) {
    // valid_only try_get: sample this cycle's wires. `success` is simply whether
    // the producer's valid happens to be asserted right now -- there is no ready
    // to assert back, so looking costs nothing and consumes nothing.
    std::string pn = std::string(getName(channel).str());
    indent();
    emitValue(result);
    os << ";\n";
    indent();
    emitValue(success);
    os << " = " << pn << "_try_get(";
    emitValue(result);
    os << ");";
    emitInfoAndNewLine(op);
    return;
  }
  // PopNB takes Message& (a NON-const reference), so its argument must be EXACTLY the
  // port's payload type. At NATIVE widths (1/8/16/32/64) emitValue declares the result as
  // a plain C type (bool, uint32_t, ...), which will not bind to Combinational<ac_int<W>>:
  //     error: cannot bind non-const lvalue reference of type 'ac_int<32,false>&'
  //            to an rvalue of type 'ac_int<32,false>'
  // Non-native widths happened to work because they already print as ac_int, which is why
  // this went unnoticed -- and why designs were forced onto odd flit widths (a 26-bit flit
  // instead of 32) to dodge it. Same bug and same fix as emitStreamTryGet (CRD-304): pop
  // into a payload-typed temp, then convert to the result. PushNB takes const Message&,
  // so try_put needs no such temp.
  std::string payloadT = std::string(
      getStreamPayloadTypeName(
          llvm::cast<ChannelType>(channel.getType()).getBaseType(),
          linkPayloadUnsigned(channel))
          .str());
  indent();
  emitValue(result); // assigns the result's name; take it AFTER for the temp
  os << ";\n";
  std::string nb = std::string(getName(result).str()) + "_nb";
  indent();
  os << payloadT << " " << nb << ";\n";
  indent();
  emitValue(success);
  os << " = ";
  emitValue(channel, 0, false);
  if (llvm::isa<ShapedType>(channel.getType())) {
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (idx)
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".PopNB(" << nb << "); ";
  emitValue(result);
  os << " = " << nb << ";";
  emitInfoAndNewLine(op);
}

// Non-blocking channel put: <success> = <channel>.PushNB(<value>);
void SystemCModuleEmitter::emitChannelTryPut(ChannelTryPutOp op) {  // override (base emitter)
  Value success = op.getResult();
  auto channel = op->getOperand(0);
  auto value = op->getOperand(1);
  if (isValidOnlyChannel(channel)) {
    // valid_only try_put: ALWAYS succeeds. Without a ready line the consumer has
    // no way to refuse, so a non-blocking put is indistinguishable from a blocking
    // one -- both drive data + a one-cycle valid pulse. `success` is a constant
    // true, which is the honest answer: the producer genuinely cannot be stalled.
    // (Whether anyone RECEIVED the datum is a different question this protocol
    // cannot answer -- that is exactly what the missing ready line buys you.)
    std::string pn = std::string(getName(channel).str());
    indent();
    emitValue(success);
    os << " = " << pn << "_try_put(";
    emitValue(value);
    os << ");";
    emitInfoAndNewLine(op);
    return;
  }
  indent();
  emitValue(success);
  os << " = ";
  emitValue(channel, 0, false);
  if (llvm::isa<ShapedType>(channel.getType())) {
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (idx)
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".PushNB(";
  emitValue(value);
  os << ");";
  emitInfoAndNewLine(op);
}

// Connections get: <result> = <stream>[indices].Pop();
// (Base scalar path, with .read() -> .Pop(); block-streams deferred.)
void SystemCModuleEmitter::emitStreamGet(StreamGetOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO consumer end: <result> = <stream>_deq.Pop(); occupancy--.
    Value result = op.getResult();
    fixUnsignedType(result, op->hasAttr("unsigned"));
    std::string sn = std::string(getName(op->getOperand(0)).str());
    indent();
    emitValue(result);
    os << " = " << sn << "_deq.Pop(); " << sn << "_cnt--;";
    emitInfoAndNewLine(op);
    return;
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);
  int rank = 0;
  if (llvm::isa<StreamType>(stream.getType())) {
    unsigned dimIdx = 0;
    auto sst = llvm::dyn_cast<StreamType>(stream.getType());
    if (auto shapedType = llvm::dyn_cast<ShapedType>(sst.getBaseType())) {
      indent(); emitArrayDecl(result, false); os << ";\n";
      for (auto &shape : shapedType.getShape()) {
        indent();
        os << "for (int iv" << dimIdx << " = 0; iv" << dimIdx << " < " << shape
           << "; ++iv" << dimIdx++ << ") {\n";
        addIndent();
      }
      rank = dimIdx;
    }
  }
  indent();
  emitValue(result, rank);
  os << " = ";
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : idx.asArrayRef())
      os << "[" << v << "]";
  }
  os << ".Pop();";
  if (rank > 0) {
    os << "\n";
    for (int i = 0; i < rank; ++i) { reduceIndent(); indent(); os << "}\n"; }
  }
  emitInfoAndNewLine(op);
}

// Connections put: <stream>[indices].Push(<value>);
// (Base scalar path, with .write(v) -> .Push(v); block-streams deferred.)
void SystemCModuleEmitter::emitStreamPut(StreamPutOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO producer end: <stream>_enq.Push(<value>); occupancy++.
    std::string sn = std::string(getName(op->getOperand(0)).str());
    indent();
    os << sn << "_enq.Push(";
    emitValue(op->getOperand(1));
    os << "); " << sn << "_cnt++;";
    emitInfoAndNewLine(op);
    return;
  }
  auto stream = op->getOperand(0);
  auto data = op->getOperand(1);
  int rank = 0;
  if (llvm::isa<StreamType>(stream.getType())) {
    unsigned dimIdx = 0;
    auto sst = llvm::dyn_cast<StreamType>(stream.getType());
    if (auto shapedType = llvm::dyn_cast<ShapedType>(sst.getBaseType())) {
      for (auto &shape : shapedType.getShape()) {
        indent();
        os << "for (int iv" << dimIdx << " = 0; iv" << dimIdx << " < " << shape
           << "; ++iv" << dimIdx++ << ") {\n";
        addIndent();
      }
      rank = dimIdx;

      // Whole-block put of a random-access memory-port arg (`put(argA)`): the
      // data's name is a req/rsp pair, not an indexable local array. Read each
      // element via the mem-port protocol (mirrors emitMemPortLoad) and Push it,
      // rather than emitting an invalid `<arg>[iv0][iv1]`.
      if (char d = memPortArgDir(data); d == 'i' || d == 'b') {
        auto mt = llvm::cast<MemRefType>(data.getType());
        int64_t total = 1;
        for (auto dim : mt.getShape())
          total *= dim;
        std::string aT = "ac_int<" + std::to_string(scAddrW(total)) + ", false>";
        auto dn = getName(data);
        SmallVector<int64_t> stride(rank);
        int64_t s = 1;
        for (int k = rank - 1; k >= 0; --k) {
          stride[k] = s;
          s *= mt.getShape()[k];
        }
        // Whole-block put of a memory-port array: read each element through the same
        // RAM-pin accessor the ordinary loads use, then push it onto the stream.
        indent();
        emitValue(stream, 0, false);
        os << ".Push( " << dn << "_rd((" << aT << ")(";
        for (int k = 0; k < rank; ++k) {
          if (k)
            os << " + ";
          os << "(iv" << k << ")";
          if (stride[k] != 1)
            os << " * " << stride[k];
        }
        os << ")) );\n";
        for (int i = 0; i < rank; ++i) {
          reduceIndent();
          indent();
          os << "}\n";
        }
        emitInfoAndNewLine(op);
        return;
      }
    }
    indent();
    emitValue(stream, 0, false);
  } else {
    indent();
    emitValue(stream, 0, false);
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    for (int64_t v : idx.asArrayRef())
      os << "[" << v << "]";
  }
  os << ".Push(";
  emitValue(op->getOperand(1), rank);
  os << ");";
  if (rank > 0) {
    os << "\n";
    for (int i = 0; i < rank; ++i) { reduceIndent(); indent(); os << "}\n"; }
  }
  emitInfoAndNewLine(op);
}

// Non-blocking get: <result>; <success> = <stream>[idx].PopNB(<result>);
// (Base try_get path, with .read_nb -> .PopNB.)
void SystemCModuleEmitter::emitStreamTryGet(StreamTryGetOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO consumer end: <result>; <success> = <stream>_deq.PopNB(<result>);
    // occupancy -= success so empty()/full() track the logical fill.
    Value r = op.getResult(0), s = op.getResult(1);
    fixUnsignedType(r, op->hasAttr("unsigned"));
    std::string sn = std::string(getName(op->getOperand(0)).str());
    // PopNB needs a payload-typed temp (non-const ref; see the cross-kernel path).
    std::string payloadT = std::string(
        getStreamPayloadTypeName(
            llvm::cast<StreamType>(op->getOperand(0).getType()).getBaseType(),
            linkPayloadUnsigned(op->getOperand(0)))
            .str());
    indent();
    emitValue(r); // assigns r's name; take it AFTER for the temp
    os << ";\n";
    std::string nb = std::string(getName(r).str()) + "_nb";
    indent();
    os << payloadT << " " << nb << ";\n";
    indent();
    emitValue(s);
    os << " = " << sn << "_deq.PopNB(" << nb << "); ";
    emitValue(r);
    os << " = " << nb << "; " << sn << "_cnt -= ";
    emitValue(s);
    os << ";";
    emitInfoAndNewLine(op);
    return;
  }
  Value result = op.getResult(0);
  Value success = op.getResult(1);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);
  // PopNB takes Message& (a NON-const reference), so its argument must be EXACTLY
  // the port's payload type (ac_int<W>, see getStreamPayloadTypeName) -- a native
  // int32_t result won't bind to In<ac_int<32>> (CRD-304). Pop into a payload-typed
  // temp, then convert to the result. (PushNB takes const Message&, so try_put needs
  // no such temp.)
  std::string payloadT = std::string(
      getStreamPayloadTypeName(llvm::cast<StreamType>(stream.getType()).getBaseType(),
                               linkPayloadUnsigned(stream))
          .str());
  indent();
  emitValue(result); // assigns the result's name; take it AFTER for the temp
  os << ";\n";
  std::string nb = std::string(getName(result).str()) + "_nb";
  indent();
  os << payloadT << " " << nb << ";\n";
  indent();
  emitValue(success);
  os << " = ";
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (idx)
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".PopNB(" << nb << "); ";
  emitValue(result);
  os << " = " << nb << ";";
  emitInfoAndNewLine(op);
}

// Non-blocking put: <success> = <stream>[idx].PushNB(<value>);
void SystemCModuleEmitter::emitStreamTryPut(StreamTryPutOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO producer end: <success> = <stream>_enq.PushNB(<value>);
    // occupancy += success (bool -> 0/1) so empty()/full() track the logical fill.
    Value s = op.getResult();
    std::string sn = std::string(getName(op->getOperand(0)).str());
    indent();
    emitValue(s);
    os << " = " << sn << "_enq.PushNB(";
    emitValue(op->getOperand(1));
    os << "); " << sn << "_cnt += ";
    emitValue(s);
    os << ";";
    emitInfoAndNewLine(op);
    return;
  }
  Value success = op.getResult();
  auto stream = op->getOperand(0);
  auto value = op->getOperand(1);
  indent();
  emitValue(success);
  os << " = ";
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (idx)
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".PushNB(";
  emitValue(value);
  os << ");";
  emitInfoAndNewLine(op);
}

// Does `func` call Stream.empty() (wantEmpty) or .full() on stream arg `argIdx`?
// Used to add the occupancy sideband port only where it is actually queried, so
// unqueried streams keep their old interface. Cross-kernel stream args are scalar
// StreamType BlockArguments (region-level Stream[T,d][P,Q] is distributed to
// per-instance scalar ports via mapping), so the operand is the arg directly.
static bool streamArgQueried(func::FuncOp func, unsigned argIdx, bool wantEmpty) {
  bool found = false;
  func.walk([&](Operation *op) {
    bool isE = llvm::isa<allo::StreamEmptyOp>(op);
    bool isF = llvm::isa<allo::StreamFullOp>(op);
    if ((wantEmpty && !isE) || (!wantEmpty && !isF))
      return;
    if (auto ba = llvm::dyn_cast<BlockArgument>(op->getOperand(0)))
      if (ba.getArgNumber() == argIdx &&
          ba.getOwner()->getParentOp() == func.getOperation())
        found = true;
  });
  return found;
}

// Does ANY kernel that receives this cross-kernel stream call empty()/full() on
// it? If not, the buffered stream needs no occupancy sideband -- it can use a
// plain Connections::Fifo instead of AlloFifoC (saves the status method + ports).
static bool streamValueQueried(Value sv) {
  for (OpOperand &use : sv.getUses()) {
    auto call = llvm::dyn_cast<func::CallOp>(use.getOwner());
    if (!call)
      continue; // only kernel-call uses map a stream to a consumer arg
    auto mod = call->getParentOfType<ModuleOp>();
    auto callee = mod ? mod.lookupSymbol<func::FuncOp>(call.getCallee()) : nullptr;
    if (!callee)
      continue;
    unsigned idx = use.getOperandNumber(); // call operand i == callee arg i
    if (idx < callee.getNumArguments() &&
        (streamArgQueried(callee, idx, /*wantEmpty=*/true) ||
         streamArgQueried(callee, idx, /*wantEmpty=*/false)))
      return true;
  }
  return false;
}

// empty()/full() on a CROSS-kernel buffered stream read the AlloFifo occupancy
// SIDEBAND (<stream>_empty / _full sc_in wires), NOT In<T>.Empty()/Out<T>.Full():
// on a regular Connections In/Out port those return a sim-only latched-data flag
// our channel never sets, so they never reflect FIFO state (data never moves). A
// LOCAL self-FIFO instead reads its synchronous <stream>_cnt (see below).
void SystemCModuleEmitter::emitStreamEmpty(StreamEmptyOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO empty(): read the synchronous occupancy counter, not the clocked
    // AlloFifo handshake (which lags a cycle and reads stale in RTL cosim).
    Value result = op.getResult();
    fixUnsignedType(result, op->hasAttr("unsigned"));
    indent();
    emitValue(result);
    os << " = (" << std::string(getName(op->getOperand(0)).str())
       << "_cnt == 0);";
    emitInfoAndNewLine(op);
    return;
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);
  indent();
  emitValue(result);
  os << " = ";
  emitValue(stream, 0, false);
  os << "_empty";
  if (llvm::isa<ShapedType>(stream.getType()))
    if (auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices"))
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  os << ".read();";
  emitInfoAndNewLine(op);
}
void SystemCModuleEmitter::emitStreamFull(StreamFullOp op) {  // override (base emitter)
  if (isLocalStream(op->getOperand(0))) {
    // self-FIFO full(): counter == depth (synchronous; see emitStreamEmpty).
    Value result = op.getResult();
    fixUnsignedType(result, op->hasAttr("unsigned"));
    int64_t depth = llvm::cast<StreamType>(op->getOperand(0).getType()).getDepth();
    indent();
    emitValue(result);
    os << " = (" << std::string(getName(op->getOperand(0)).str())
       << "_cnt == " << depth << ");";
    emitInfoAndNewLine(op);
    return;
  }
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);
  indent();
  emitValue(result);
  os << " = ";
  emitValue(stream, 0, false);
  os << "_full";
  if (llvm::isa<ShapedType>(stream.getType()))
    if (auto idx = op->getAttrOfType<DenseI64ArrayAttr>("indices"))
      for (int64_t v : idx.asArrayRef())
        os << "[" << v << "]";
  os << ".read();";
  emitInfoAndNewLine(op);
}

// Narrowing a >64-bit ac_int to a native int/index needs an EXPLICIT
// .to_int64()/.to_uint64(). The ap_int shim's implicit narrowing is csim-only (a
// plain ac_int under __SYNTHESIS__ lacks it -> Catapult CRD-413); the explicit
// call compiles in both csim and synthesis. `index` (emitted as `int`) is a
// native signed 64-bit type -- e.g. a bit-slice range endpoint computed wide
// (ap_int<66>) then cast to index would otherwise fail to convert.
void SystemCModuleEmitter::emitNarrowCastSuffix(Value src, Value dst) {  // override (base emitter)
  auto si = llvm::dyn_cast<IntegerType>(src.getType());
  if (!si || si.getWidth() <= 64)
    return;
  Type dt = dst.getType();
  if (auto di = llvm::dyn_cast<IntegerType>(dt)) {
    if (di.getWidth() <= 64)
      os << (di.getSignedness() == IntegerType::SignednessSemantics::Unsigned
                 ? ".to_uint64()"
                 : ".to_int64()");
  } else if (llvm::isa<IndexType>(dt)) {
    os << ".to_int64()";
  }
}

// max/min are std::max<T>(const T&, const T&) -- both args must be the SAME type.
// A ReLU `max(x, 0.0)` emits max(<ac_ieee_float>, 0.0f), and the float literal
// vs ac_ieee_float mismatch fails template deduction (Catapult CRD-304). Cast
// both operands to the result type so deduction succeeds; the cast is a no-op for
// an operand already of that type and invokes the element ctor for a literal.
void SystemCModuleEmitter::emitMaxMin(Operation *op, const char *syntax) {  // override (base emitter)
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result, rank);
  std::string T = std::string(getSCTypeName(result.getType()).str());
  os << " = " << syntax << "((" << T << ")";
  emitValue(op->getOperand(0), rank);
  os << ", (" << T << ")";
  emitValue(op->getOperand(1), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void SystemCModuleEmitter::emitTopModule(func::FuncOp func) {  // new (SystemC-only)
  auto parent = func->getParentOfType<ModuleOp>();
  os << "SC_MODULE(" << func.getName() << ") {\n";
  addIndent();

  // Clock + reset (fanned out to every submodule).
  indent(); os << "sc_in_clk clk;\n";
  indent(); os << "sc_in<bool> rst;\n";
  // Top-level completion port = AND of every kernel's `done` (see below). The tb
  // polls this to stop exactly when the design finishes -- valid in RTL cosim,
  // unlike the C-side __allo_done counter.
  indent(); os << "sc_out<bool> done;\n";

  // Region boundary arrays -> top-level Connections stream ports (In=input,
  // Out=output; direction from arg_dirs). A random-access array is routed to an
  // internal AlloMem instead (memArrays). Input/output file indices are global
  // over ALL 'i'/'o' args in arg order, matching hls.py's input/output split.
  int inCount = 0, outCount = 0;
  for (auto arg : llvm::enumerate(func.getArguments())) {
    auto mt = llvm::dyn_cast<MemRefType>(arg.value().getType());
    if (!mt)
      continue;
    std::string nm = std::string(addName(arg.value(), /*isPtr=*/false).str());
    std::string ct = std::string(getStreamPayloadTypeName(mt.getElementType(), linkPayloadUnsigned(arg.value())).str());
    int64_t total = 1;
    for (auto s : mt.getShape())
      total *= s;
    // Random-access array -> internal memory (no top-level port): INPUT preloads
    // from input<k>.data, OUTPUT reads out to output<k>.data, BOTH does both.
    char mp = regArgMemPort(func, arg.value());
    if (mp == 'i' || mp == 'o' || mp == 'b') {
      unsigned aw = scAddrW(total), dw = scDataW(mt.getElementType());
      int ii = (mp == 'i' || mp == 'b') ? inCount++ : -1;
      int oo = (mp == 'o' || mp == 'b') ? outCount++ : -1;
      memArrays.push_back({nm, ct, total, aw, dw, mp, ii, oo});
      continue;
    }
    char d = argDir(func, arg.index());
    int fidx = (d == 'o') ? outCount++ : inCount++;
    indent();
    os << (d == 'o' ? "Connections::Out< " : "Connections::In< ") << ct
       << " > " << nm << ";\n";
    ioArrays.push_back({nm, ct, total, d, fidx});
  }

  // The top body is stream_construct(s) + call(s). Collect them.
  SmallVector<StreamConstructOp, 4> channels;
  SmallVector<ChannelConstructOp, 4> chanOps; // handshake channels (combinational)
  SmallVector<WireConstructOp, 4> wireOps;    // raw combinational wires (sc_signal)
  SmallVector<func::CallOp, 4> calls;
  for (auto &op : func.front()) {
    if (auto sc = llvm::dyn_cast<StreamConstructOp>(&op))
      channels.push_back(sc);
    else if (auto cc = llvm::dyn_cast<ChannelConstructOp>(&op))
      chanOps.push_back(cc);
    else if (auto wc = llvm::dyn_cast<WireConstructOp>(&op))
      wireOps.push_back(wc);
    else if (auto call = llvm::dyn_cast<func::CallOp>(&op))
      calls.push_back(call);
  }
  numKernelInsts = calls.size(); // single-shot tb waits for this many completions

  // Channel members. A Stream's depth (from its type) picks the flavor:
  //   depth 0  -> a bare Connections::Combinational (combinational wire)
  //   depth>=1 -> an AlloFifo<T,depth> between two _in/_out wires (buffered)
  for (auto sc : channels) {
    // Skip a DANGLING stream (declared but no PE puts/gets it -- e.g. an unused
    // cell of a Stream[T,d][P0,P1] systolic grid). Emitting its Combinational +
    // AlloFifo leaves the channel's producer/consumer end unbound and unreset, which
    // aborts RTL cosim (Connections CONNECTIONS-101 "wasn't reset" -> CONNECTIONS-125
    // "unable to resolve clock"). No PE binds it, so dropping it changes nothing.
    // MUST match the skip in the ctor-init and self-wiring loops below.
    if (sc.getResult().use_empty())
      continue;
    // A self-FIFO (one kernel produces+queries) is a NORMAL buffered stream here:
    // its _in/_out Combinational wires + AlloFifo are declared just like any other
    // depth>=1 stream; the one kernel simply binds BOTH ends (see the bind loop).
    auto st = llvm::dyn_cast<StreamType>(sc.getResult().getType());
    std::string T = std::string(getStreamPayloadTypeName(st.getBaseType(), linkPayloadUnsigned(sc.getResult())).str());
    std::string nm = std::string(addName(sc.getResult(), /*isPtr=*/false).str());
    if (st.getDepth() == 0) {
      indent();
      os << "Connections::Combinational< " << T << " > " << nm << ";\n";
    } else {
      indent();
      os << "Connections::Combinational< " << T << " > " << nm << "_in;\n";
      indent();
      os << "Connections::Combinational< " << T << " > " << nm << "_out;\n";
      indent();
      if (streamValueQueried(sc.getResult())) {
        // some consumer polls empty()/full(): AlloFifoC exposes the sideband.
        os << "AlloFifoC< " << T << ", " << st.getDepth() << " > " << nm
           << "_fifo;\n";
        indent();
        os << "sc_signal<bool> " << nm << "_empty_sig;\n";
        indent();
        os << "sc_signal<bool> " << nm << "_full_sig;\n";
      } else {
        // nobody queries occupancy: plain vendor FIFO, no status ports/wires.
        os << "Connections::Fifo< " << T << ", " << st.getDepth() << " > " << nm
           << "_fifo;\n";
      }
    }
  }
  // Channel members: a Channel is always a combinational link, but its PROTOCOL
  // picks the realization -- valid_ready gets a Connections transactor, valid_only
  // gets a bare data+valid signal pair with no ready path at all.
  for (auto cc : chanOps) {
    auto ct = llvm::dyn_cast<ChannelType>(cc.getResult().getType());
    std::string T = std::string(getStreamPayloadTypeName(ct.getBaseType(), linkPayloadUnsigned(cc.getResult())).str());
    std::string nm = std::string(addName(cc.getResult(), /*isPtr=*/false).str());
    indent();
    if (isValidOnlyChannel(cc.getResult()))
      os << "sc_signal< " << T << " > " << nm << "_dat;\n"
         << "  sc_signal<bool> " << nm << "_vld;\n";
    else
      os << "Connections::Combinational< " << T << " > " << nm << ";\n";
  }
  // Wire members: a raw combinational wire is an sc_signal<T>.
  for (auto wc : wireOps) {
    auto wt = llvm::dyn_cast<WireType>(wc.getResult().getType());
    std::string T = std::string(getStreamPayloadTypeName(wt.getBaseType(), linkPayloadUnsigned(wc.getResult())).str());
    std::string nm = std::string(addName(wc.getResult(), /*isPtr=*/false).str());
    indent();
    os << "sc_signal< " << T << " > " << nm << ";\n";
  }
  // Submodule instance members: <callee> uN;
  SmallVector<std::string, 4> instNames;
  for (auto it : llvm::enumerate(calls)) {
    std::string inst = "u" + std::to_string(it.index());
    instNames.push_back(inst);
    indent();
    os << it.value().getCallee() << " " << inst << ";\n";
  }
  // Per-kernel completion signals; an SC_METHOD ANDs them into the top `done` port.
  for (auto &inst : instNames) {
    indent(); os << "sc_signal<bool> " << inst << "_done;\n";
  }
  // Discover one physical memory per (call, memory-port arg). A grid replica
  // that uses a shared boundary array gets its OWN memory (replication), keyed
  // uniquely by mp<call>_<arg>; its file index comes from the region array.
  for (auto it : llvm::enumerate(calls)) {
    auto callee = parent.lookupSymbol<func::FuncOp>(it.value().getCallee());
    for (auto opnd : llvm::enumerate(it.value().getOperands())) {
      Value ov = opnd.value();
      if (!llvm::isa<MemRefType>(ov.getType()))
        continue;
      Value carg = callee.getArgument(opnd.index());
      char mp = memPortArgDir(carg);
      if (!mp)
        continue;
      auto mt = llvm::cast<MemRefType>(carg.getType());
      int64_t total = 1;
      for (auto s : mt.getShape())
        total *= s;
      int ii = -1, oo = -1;
      for (auto &m : memArrays)
        if (m.base == std::string(getName(ov).str())) {
          ii = m.inIdx;
          oo = m.outIdx;
        }
      memInsts.push_back(
          {"mp" + std::to_string(it.index()) + "_" +
               std::to_string(opnd.index()),
           instNames[it.index()], std::string(getName(carg).str()),
           std::string(getStreamPayloadTypeName(mt.getElementType(), linkPayloadUnsigned(carg)).str()), total,
           scAddrW(total), scDataW(mt.getElementType()), mp, ii, oo,
           std::string(getName(ov).str()), /*exposed=*/false});
    }
  }
  // Decide which memories can become TOP-LEVEL PORTS.
  //
  // A boundary array touched by N kernels is REPLICATED into N memories (see the MemInst
  // comment). Inside the design that is a private trick the testbench compensates for:
  // reads preload every replica from the same file, writes are summed across replicas at
  // readout. At the BOUNDARY it stops being private and becomes the interface contract --
  // an array shared by 15 kernels would present 15 request ports carrying identical reads,
  // and a shared output would present 10 write ports whose values the integrator is
  // somehow expected to sum. Measured on the current suite: test_hierachical replicates 4
  // input arrays into 60 memories and 2 outputs into 20. That is not an interface anyone
  // can wire.
  //
  // So only a SINGLE-CLIENT array is exposed. A replicated one keeps the old behaviour --
  // memory inside the design -- which is still wrong for area but at least honest and
  // wireable, and keeps the existing designs passing. Exposing those properly needs one
  // memory with an arbiter behind a single port pair (the AlloMemShared component), which
  // is separate work.
  {
    llvm::StringMap<unsigned> clients;
    for (auto &mi : memInsts)
      clients[mi.base]++;
    for (auto &mi : memInsts)
      mi.exposed = (clients[mi.base] == 1);

    // A replicated array whose clients disagree about DIRECTION cannot work. Each
    // client gets a private copy, so one kernel's writes are invisible to another
    // kernel's reads. Measured: a writer kernel storing 100..107 and a reader kernel
    // reading the same array emitted, compiled, synthesized and RAN -- and returned
    // all zeros, with no diagnostic anywhere. That is the worst failure mode we have.
    //
    // Reject ONLY the mixed read/write case. Replication itself is not the bug and is
    // relied upon: several clients that all WRITE are the disjoint-element pattern the
    // testbench merges at readout (test_hierachical replicates 2 outputs into 20
    // memories), and several clients that all READ share an identically preloaded copy.
    // Both stay legal.
    // PURE reader / PURE writer only. A client that both reads and writes ('b') is
    // self-contained: it reads back what it wrote into its own replica, plus the
    // preloaded values it never touched. test_hierachical does exactly that with 16
    // clients on one array and is correct -- an earlier, broader version of this check
    // flagged it and turned a passing test into an error.
    llvm::StringMap<bool> anyPureRead, anyPureWrite;
    for (auto &mi : memInsts) {
      if (clients[mi.base] < 2)
        continue;
      if (mi.dir == 'i')
        anyPureRead[mi.base] = true;
      if (mi.dir == 'o')
        anyPureWrite[mi.base] = true;
    }
    llvm::StringMap<bool> reported;   // StringMap, not StringSet: no StringSet.h here
    for (auto &mi : memInsts) {
      if (clients[mi.base] < 2 || !anyPureRead.lookup(mi.base) ||
          !anyPureWrite.lookup(mi.base) || !reported.insert({mi.base, true}).second)
        continue;
      emitError(func, "array '" + mi.base + "' is accessed at arbitrary indices by " +
                          std::to_string(clients[mi.base]) +
                          " kernels, and at least one WRITES it while another READS "
                          "it. Such an array is replicated per client, so the writes "
                          "would be invisible to the reader and the result silently "
                          "wrong. Pass the values through a stream instead, or keep "
                          "the array private to one kernel.");
    }
  }
  // Memory-port arrays -> TOP-LEVEL PORTS, not internal storage.
  //
  // These used to be an internal Connections::Combinational + AlloMem instance, so the
  // array's storage lived INSIDE the synthesized design: a region whose boundary arrays
  // are all random-access produced `module top(clk, rst, done)` with every workload
  // array turned into registers/RAM in the DUT. That silently inflates every area
  // number and is not a shape anyone can integrate -- the Vitis/Catapult backend
  // exposes the same arrays as memory interfaces (`_rsc_radr/_re/_q`, `_rsc_d/_we`).
  //
  // So the DUT now only carries the ACCESS PORTS and the memory moves to the testbench
  // (see the tb, which instantiates AlloMemPins and binds it). The request
  // encoding, the kernel-side client ports and AlloMem itself are all unchanged -- only
  // where the memory is instantiated moves. Latency-insensitive Connections rather than
  // raw RAM pins keeps the existing handshake and imposes no memory-timing assumption.
  // Ports are named after the ARRAY (`A_req`), not the internal replica key
  // (`mp0_1_req`) -- an integrator reading the port list should recognise the design's
  // own names. Safe because only single-client arrays get here, so the name is unique.
  for (auto &mi : memInsts) {
    if (!mi.exposed)
      continue;
    std::string aT = "ac_int<" + std::to_string(mi.addrw) + ", false>";
    if (mi.dir != 'o') { // read side
      indent(); os << "sc_out< " << aT << " > " << mi.base << "_radr;\n";
      indent(); os << "sc_out<bool> " << mi.base << "_re;\n";
      indent(); os << "sc_in< " << mi.ctype << " > " << mi.base << "_q;\n";
      indent(); os << "sc_in<bool> " << mi.base << "_rrdy;\n";
    }
    if (mi.dir != 'i') { // write side
      indent(); os << "sc_out< " << aT << " > " << mi.base << "_wadr;\n";
      indent(); os << "sc_out< " << mi.ctype << " > " << mi.base << "_d;\n";
      indent(); os << "sc_out<bool> " << mi.base << "_we;\n";
      indent(); os << "sc_in<bool> " << mi.base << "_wrdy;\n";
    }
  }
  // Replicated (multi-client) arrays keep their memory INSIDE the design.
  for (auto &mi : memInsts) {
    if (mi.exposed)
      continue;
    // Same AlloMemPins the testbench uses for exposed arrays -- only the location
    // differs. sc_signals stand in for what the Connections channel used to be.
    std::string aT = "ac_int<" + std::to_string(mi.addrw) + ", false>";
    indent(); os << "sc_signal< " << aT << " > " << mi.chan << "_radr, " << mi.chan
                 << "_wadr;\n";
    indent(); os << "sc_signal<bool> " << mi.chan << "_re, " << mi.chan << "_we, "
                 << mi.chan << "_rrdy, " << mi.chan << "_wrdy;\n";
    indent(); os << "sc_signal< " << mi.ctype << " > " << mi.chan << "_q, " << mi.chan
                 << "_d;\n";
    indent();
    os << "AlloMemPins< " << mi.ctype << ", " << mi.total << ", " << mi.addrw
       << " > " << mi.chan << "_mem;  // replicated across clients: cannot be a port\n";
  }

  // Constructor: init list (channel names + instance names) + bindings.
  indent();
  os << "SC_CTOR(" << func.getName() << ")";
  std::string sep = " : ";
  for (auto &a : ioArrays) {
    os << sep << a.member << "(\"" << a.member << "\")";
    sep = ", ";
  }
  for (auto sc : channels) {
    if (sc.getResult().use_empty()) // dangling stream: skipped above, no member to init
      continue;
    auto st = llvm::dyn_cast<StreamType>(sc.getResult().getType());
    std::string nm = std::string(getName(sc.getResult()).str());
    if (st.getDepth() == 0) {
      os << sep << nm << "(\"" << nm << "\")";
    } else {
      os << sep << nm << "_in(\"" << nm << "_in\")";
      os << ", " << nm << "_out(\"" << nm << "_out\")";
      os << ", " << nm << "_fifo(\"" << nm << "_fifo\")";
    }
    sep = ", ";
  }
  for (auto cc : chanOps) {
    std::string nm = std::string(getName(cc.getResult()).str());
    if (isValidOnlyChannel(cc.getResult()))
      os << sep << nm << "_dat(\"" << nm << "_dat\"), " << nm << "_vld(\"" << nm
         << "_vld\")";
    else
      os << sep << nm << "(\"" << nm << "\")";
    sep = ", ";
  }
  for (auto wc : wireOps) {
    std::string nm = std::string(getName(wc.getResult()).str());
    os << sep << nm << "(\"" << nm << "\")";
    sep = ", ";
  }
  for (auto it : llvm::enumerate(calls)) {
    os << sep << instNames[it.index()] << "(\"" << instNames[it.index()]
       << "\")";
    sep = ", ";
  }
  for (auto &mi : memInsts) {
    if (mi.exposed) {
      // raw sc_in/sc_out: no name argument, nothing to add to the init list
      continue;
    } else {
      os << sep << mi.chan << "_mem(\"" << mi.chan << "_mem\")";
    }
    sep = ", ";
  }
  os << " {\n";
  addIndent();
  // Fan clk/rst into each instance; bind each STREAM operand to its channel;
  // record each MEMREF operand as a top-level I/O array for the testbench.
  for (auto it : llvm::enumerate(calls)) {
    auto call = it.value();
    auto callee = parent.lookupSymbol<func::FuncOp>(call.getCallee());
    indent(); os << instNames[it.index()] << ".clk(clk);\n";
    indent(); os << instNames[it.index()] << ".rst(rst);\n";
    indent(); os << instNames[it.index()] << ".done(" << instNames[it.index()]
                 << "_done);\n";
    // Bind each operand to the port THIS callee actually emitted for it. A grid
    // shares one boundary array across all replicas, but only the replica that
    // uses it (feeder/body/drain) has a port; unused args are internal members
    // and must NOT be bound. Decide per-(call,arg) from the callee's OWN arg.
    for (auto opnd : llvm::enumerate(call.getOperands())) {
      Value ov = opnd.value();
      Value carg = callee.getArgument(opnd.index());
      // Duplicate stream/channel arg: the primary already bound the shared port,
      // and getName(carg) now resolves to it -- binding again would double-bind.
      if (aliasOf(carg))
        continue;
      if (auto sty = llvm::dyn_cast<StreamType>(ov.getType())) {
        if (localStreamConstructs.count(ov)) {
          // self-FIFO: this ONE kernel is both producer and consumer. Bind its
          // enq (Out) to the fifo input wire and its deq (In) to the output wire
          // (depth 0 -> both bind the bare Combinational).
          std::string base = std::string(getName(ov).str());
          std::string cin = base, cout = base;
          if (sty.getDepth() != 0) {
            cin += "_in";
            cout += "_out";
          }
          std::string cargn = std::string(getName(carg).str());
          indent();
          os << instNames[it.index()] << "." << cargn << "_enq(" << cin << ");\n";
          indent();
          os << instNames[it.index()] << "." << cargn << "_deq(" << cout
             << ");\n";
          continue;
        }
        // stream operand -> stream channel. For a buffered stream (depth>=1) the
        // producer (Out) binds the FIFO's input wire, the consumer (In) its
        // output wire; a depth-0 stream is the bare Combinational.
        std::string chan = std::string(getName(ov).str());
        if (sty.getDepth() != 0)
          chan += (streamDir(callee, opnd.index()) == 'o') ? "_in" : "_out";
        indent();
        os << instNames[it.index()] << "." << getName(carg) << "(" << chan
           << ");\n";
        // Bind the occupancy sidebands the callee declared (usage-gated, mirrors
        // the port-decl condition), to this stream's top-level status signals.
        if (sty.getDepth() != 0) {
          std::string sbase = std::string(getName(ov).str());
          if (streamArgQueried(callee, opnd.index(), /*wantEmpty=*/true)) {
            indent();
            os << instNames[it.index()] << "." << getName(carg) << "_empty("
               << sbase << "_empty_sig);\n";
          }
          if (streamArgQueried(callee, opnd.index(), /*wantEmpty=*/false)) {
            indent();
            os << instNames[it.index()] << "." << getName(carg) << "_full("
               << sbase << "_full_sig);\n";
          }
        }
      } else if (llvm::isa<ChannelType>(ov.getType())) {
        std::string chan = std::string(getName(ov).str());
        indent();
        if (isValidOnlyChannel(ov)) {
          // valid_only: two raw signals to bind instead of one transactor.
          os << instNames[it.index()] << "." << getName(carg) << "_dat(" << chan
             << "_dat);\n";
          indent();
          os << instNames[it.index()] << "." << getName(carg) << "_vld(" << chan
             << "_vld);\n";
        } else {
          // valid_ready: bare Combinational (combinational, no _in/_out)
          os << instNames[it.index()] << "." << getName(carg) << "(" << chan
             << ");\n";
        }
      } else if (llvm::isa<WireType>(ov.getType())) {
        // wire operand -> sc_signal (raw combinational)
        std::string sig = std::string(getName(ov).str());
        indent();
        os << instNames[it.index()] << "." << getName(carg) << "(" << sig
           << ");\n";
      } else if (llvm::isa<MemRefType>(ov.getType())) {
        char mp = memPortArgDir(carg);
        std::string ca = std::string(getName(carg).str());
        std::string rb = std::string(getName(ov).str());
        if (mp) {
          // random-access memory port: bind req (+ rsp) straight through to THIS
          // client's TOP-LEVEL port (mp<call>_<arg>). The memory itself is outside
          // the design now, so there is no internal channel to land on -- the kernel
          // port and the top port are the same wire.
          std::string chan = "mp" + std::to_string(it.index()) + "_" +
                             std::to_string(opnd.index());
          // Exposed (single-client) -> bind straight to the top port named after the
          // array. Replicated -> bind to this replica's internal channel, as before.
          std::string base;
          for (auto &mi : memInsts)
            if (mi.chan == chan && mi.exposed)
              base = mi.base;
          std::string inst = instNames[it.index()];
          if (!base.empty()) {
            // EXPOSED: the kernel's pins are the top's pins, wired straight through.
            if (mp != 'o')
              for (const char *sfx : {"_radr", "_re", "_q", "_rrdy"}) {
                indent(); os << inst << "." << ca << sfx << "(" << base << sfx << ");\n";
              }
            if (mp != 'i')
              for (const char *sfx : {"_wadr", "_d", "_we", "_wrdy"}) {
                indent(); os << inst << "." << ca << sfx << "(" << base << sfx << ");\n";
              }
          } else {
            // REPLICATED: same pins, bound to this replica's internal signals rather
            // than to a top-level port.
            if (mp != 'o')
              for (const char *sfx : {"_radr", "_re", "_q", "_rrdy"}) {
                indent(); os << inst << "." << ca << sfx << "(" << chan << sfx << ");\n";
              }
            if (mp != 'i')
              for (const char *sfx : {"_wadr", "_d", "_we", "_wrdy"}) {
                indent(); os << inst << "." << ca << sfx << "(" << chan << sfx << ");\n";
              }
          }
        } else if (streamArgDir(carg)) {
          // sequential-scan boundary -> single stream port
          indent();
          os << instNames[it.index()] << "." << ca << "(" << rb << ");\n";
        }
        // else: unused arg -> internal array member, nothing to bind.
      }
    }
  }
  // Wire each buffered-stream FIFO: clk/rst + its _in/_out wires.
  for (auto sc : channels) {
    if (sc.getResult().use_empty()) // dangling stream: no member emitted, nothing to wire
      continue;
    auto st = llvm::dyn_cast<StreamType>(sc.getResult().getType());
    if (st.getDepth() == 0)
      continue;
    std::string nm = std::string(getName(sc.getResult()).str());
    indent(); os << nm << "_fifo.clk(clk);\n";
    indent(); os << nm << "_fifo.rst(rst);\n";
    // AlloFifoC = Connections::Fifo subclass: producer wire -> enq (In),
    // consumer wire -> deq (Out). (AlloFifo's legacy ports were in/out.)
    indent(); os << nm << "_fifo.enq(" << nm << "_in);\n";
    indent(); os << nm << "_fifo.deq(" << nm << "_out);\n";
    // Only AlloFifoC (queried streams) has the occupancy ports to bind.
    if (streamValueQueried(sc.getResult())) {
      indent(); os << nm << "_fifo.empty_o(" << nm << "_empty_sig);\n";
      indent(); os << nm << "_fifo.full_o(" << nm << "_full_sig);\n";
    }
  }
  // Exposed memories are wired in the TESTBENCH (they are outside the design now).
  // Replicated ones stay here and are wired as before.
  for (auto &mi : memInsts) {
    if (mi.exposed)
      continue;
    indent(); os << mi.chan << "_mem.clk(clk);\n";
    indent(); os << mi.chan << "_mem.rst(rst);\n";
    // AlloMemPins has all six pins regardless of direction; an unused side is simply
    // tied to a signal nobody drives, which is cheaper than a second component.
    for (const char *sfx : {"_radr", "_re", "_q", "_rrdy", "_wadr", "_d", "_we", "_wrdy"}) {
      indent();
      os << mi.chan << "_mem." << (sfx + 1) << "(" << mi.chan << sfx << ");\n";
    }
  }
  // Combinational aggregator: drive the top `done` port from all kernel dones.
  indent(); os << "SC_METHOD(_agg_done); sensitive";
  for (auto &inst : instNames)
    os << " << " << inst << "_done";
  os << ";\n";
  reduceIndent();
  indent();
  os << "}\n";

  // done = AND of every kernel's completion flag (all kernels finished their pass).
  indent(); os << "void _agg_done() { done.write(";
  if (instNames.empty()) {
    os << "true";
  } else {
    std::string sep2;
    for (auto &inst : instNames) {
      os << sep2 << inst << "_done.read()";
      sep2 = " && ";
    }
  }
  os << "); }\n";

  reduceIndent();
  os << "};\n\n";
}

//===----------------------------------------------------------------------===//
// emitModule — header + dispatch each func to kernel/top emission.
//===----------------------------------------------------------------------===//

void SystemCModuleEmitter::emitModule(ModuleOp module) {  // override (base emitter)
  // The SystemC backend is a DATAFLOW backend: it emits SC_MODULEs + a self-
  // contained sc_main testbench for @df.region / @df.kernel designs. A plain
  // (customize) kernel has no dataflow region, so it would emit only as a bodiless
  // helper function with no top module and no testbench -- it compiles but cannot
  // be simulated (there is nothing to drive). Fail early with a clear message so
  // the user picks the right backend, instead of a confusing csim failure later.
  bool hasDataflow = false;
  for (auto f : module.getOps<func::FuncOp>())
    if (f->hasAttr("dataflow") || f->hasAttr("df.kernel")) {
      hasDataflow = true;
      break;
    }
  if (!hasDataflow) {
    module.emitError(
        "target=\"systemc\" requires a dataflow design (@df.region with "
        "@df.kernel); this module has no dataflow region, so no top module or "
        "testbench can be generated. Use target=\"vhls\" for non-dataflow "
        "(customize) kernels.");
    state.encounteredError = true;
    return;
  }

  // Flatten any @df.region hierarchy (sub-regions called from kernels) into a
  // flat top before emission — SystemC can't nest regions inside a thread.
  flattenHierarchy(module);

  // Argument classification (streamDir / argDir) indexes `stypes` / `arg_dirs` by
  // argument position and returns 0 -- "not a stream" / "no direction" -- for an
  // out-of-range index. So a string one char SHORT silently reclassifies the last
  // argument: a stream port becomes a plain value, a 'b' memory port loses its
  // write side. Wrong hardware, no diagnostic. Both strings are produced by the
  // frontend (dataflow.py), so a length mismatch is a frontend bug, not user
  // input: fail the build rather than emit. Checked AFTER flattening, so it sees
  // the functions actually emitted.
  for (auto f : module.getOps<func::FuncOp>())
    for (const char *an : {"stypes", "arg_dirs"})
      if (auto a = f->getAttrOfType<StringAttr>(an))
        if (a.getValue().size() != f.getNumArguments()) {
          f.emitError("`")
              << an << "` has " << a.getValue().size() << " chars but `"
              << f.getName() << "` has " << f.getNumArguments()
              << " arguments; argument classification would silently mis-assign "
                 "the trailing ones";
          state.encounteredError = true;
          return;
        }

  // A stateful variable declared at REGION scope and touched by more than one kernel
  // is shared mutable state between concurrently-running SC_MODULEs. There is no
  // correct local form for it: emitting the declaration into each kernel's reset action
  // (what the single-kernel path does) gives every kernel its OWN copy, so writes by one
  // are invisible to the others -- wrong answers, no diagnostic. Real sharing needs a
  // memory with arbitration, i.e. the AlloMem memory-port path, which is a separate
  // feature. Reject it here rather than emit plausible-looking wrong hardware.
  {
    llvm::DenseMap<StringRef, unsigned> statefulUsers;
    for (auto f : module.getOps<func::FuncOp>()) {
      if (!f->hasAttr("df.kernel"))
        continue;
      llvm::DenseSet<StringRef> seenInThisKernel;
      f.walk([&](memref::GetGlobalOp gg) {
        auto g = module.lookupSymbol<memref::GlobalOp>(gg.getName());
        if (!g || !isStatefulGlobal(g) || g->hasAttr("constant"))
          return;
        if (seenInThisKernel.insert(g.getSymName()).second)
          statefulUsers[g.getSymName()]++;
      });
    }
    for (auto &kv : statefulUsers)
      if (kv.second > 1) {
        module.emitError("stateful variable `")
            << kv.first << "` is used by " << kv.second
            << " kernels. A region-scope `@ Stateful` shared between kernels is shared "
               "mutable state between concurrent hardware modules; the SystemC backend "
               "cannot express it (each kernel would get a private copy and silently "
               "disagree). Give each kernel its own `@ Stateful` variable, or pass the "
               "state between them over a Stream/Channel so the ordering is explicit. "
               "NOTE: passing it as a region argument does NOT work either -- the "
               "memory-port path REPLICATES a shared array (one AlloMem per client, "
               "writes summed at readout), which reproduces this same failure.";
        state.encounteredError = true;
        return;
      }
  }

  // A stream passed to exactly ONE kernel call is a self-FIFO (one kernel both
  // produces and queries it) -> realize as a local ac_channel, not a directional
  // Connections port. Mark the construct result + the callee's block arg.
  for (auto topf : module.getOps<func::FuncOp>()) {
    if (!topf->hasAttr("dataflow"))
      continue;
    llvm::DenseMap<Value, int> cnt;
    SmallVector<func::CallOp> callv;
    topf.walk([&](func::CallOp c) {
      callv.push_back(c);
      for (Value o : c.getArgOperands())
        if (llvm::isa<StreamType>(o.getType()))
          cnt[o]++;
    });
    for (auto c : callv) {
      auto callee = module.lookupSymbol<func::FuncOp>(c.getCallee());
      if (!callee)
        continue;
      for (auto it : llvm::enumerate(c.getArgOperands()))
        if (llvm::isa<StreamType>(it.value().getType()) && cnt[it.value()] == 1) {
          localStreamConstructs.insert(it.value());
          localStreamArgs.insert(callee.getArgument(it.index()));
        }
    }
  }

  // A boundary array that would seq-stream but is driven/consumed by >1 kernel
  // call (fan-in output / fan-out input) must go to the memory-port path instead
  // -- binding a 2nd producer/consumer to one Combinational aborts MatchLib.
  for (auto topf : module.getOps<func::FuncOp>()) {
    if (!topf->hasAttr("dataflow"))
      continue;
    llvm::DenseMap<Value, int> useCnt; // seq-streamable directional uses per array
    SmallVector<func::CallOp> callv;
    topf.walk([&](func::CallOp c) { callv.push_back(c); });
    auto seqDirArg = [&](func::CallOp c, unsigned idx, char &d) -> Value {
      auto callee = module.lookupSymbol<func::FuncOp>(c.getCallee());
      if (!callee)
        return nullptr;
      Value carg = callee.getArgument(idx);
      d = argDir(callee, idx);
      if ((d == 'i' || d == 'o' || d == 'b') && isSeqStreamable(carg))
        return carg;
      return nullptr;
    };
    for (auto c : callv)
      for (auto it : llvm::enumerate(c.getArgOperands())) {
        char d;
        if (llvm::isa<MemRefType>(it.value().getType()) &&
            seqDirArg(c, it.index(), d))
          useCnt[it.value()]++;
      }
    for (auto c : callv)
      for (auto it : llvm::enumerate(c.getArgOperands())) {
        char d;
        Value carg;
        if (llvm::isa<MemRefType>(it.value().getType()) && useCnt[it.value()] > 1 &&
            (carg = seqDirArg(c, it.index(), d)))
          forceMemPortArgs.insert(carg);
      }
  }

  // Same stream/channel passed to MULTIPLE arg positions of one call -> alias the
  // duplicate callee block-args onto the first so they share a single port.
  for (auto topf : module.getOps<func::FuncOp>()) {
    if (!topf->hasAttr("dataflow"))
      continue;
    topf.walk([&](func::CallOp c) {
      auto callee = module.lookupSymbol<func::FuncOp>(c.getCallee());
      if (!callee)
        return;
      llvm::DenseMap<Value, Value> primaryArg; // operand -> its first callee arg
      for (auto it : llvm::enumerate(c.getArgOperands())) {
        Value ov = it.value();
        if (!llvm::isa<StreamType>(ov.getType()) &&
            !llvm::isa<ChannelType>(ov.getType()))
          continue;
        Value carg = callee.getArgument(it.index());
        if (Value first = primaryArg.lookup(ov))
          streamArgAlias[carg] = first;
        else
          primaryArg[ov] = carg;
      }
    });
  }

  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
// Automatically generated file for SystemC (Catapult HLS / MatchLib Connections).
//===----------------------------------------------------------------------===//
#include <systemc.h>
#include <mc_connections.h>   // MatchLib Connections (LI valid/ready channels)
#include <connections/connections_fifo.h>  // vendor FWFT Connections::Fifo (buffered streams)
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>     // local self-FIFO streams (one-kernel put+get+status)
#include <ac_std_float.h>   // IEEE floats: ac_ieee_float<binaryNN>
#include <cstring>          // std::memcpy for bit-reinterpret (bitcast)
#include <stdint.h>
// f16: Catapult has no native `half`; alias it to ac_ieee_float<binary16>.
// (f32 -> ac_ieee_float<binary32> is emitted directly by getCatapultTypeName.)
typedef ac_ieee_float<binary16> half;
#include <iostream>
#include <fstream>
#include <iomanip>          // std::setprecision for lossless float tb output
#include <algorithm>
// --- float support helpers (half / ac_ieee_float<binary32> / double) ---
// Floats have no implicit int/stream conversions (and half is non-trivial), so
// memory ports (which transport a raw bit pattern in an ac_int req word) and the
// testbench (text I/O + waveform trace) need these shims.
// float -> raw bits. Catapult's front end rejects memcpy/void* casts under
// __SYNTHESIS__ (CIN-71: "Invalid pointer cast from ... to void *"), which broke
// every float MEMORY-PORT design at csynth. Floats use their IEEE type's own bit
// accessor (data_ac_int(), works in csim AND synthesis); the generic template is
// a csim-only fallback for any non-float mem-port payload (memcpy guarded out of
// synthesis, where it would only be reached by an untested double mem-port).
inline unsigned long long _fbits(const half &v) {
  return (unsigned long long)v.data_ac_int().to_uint();
}
inline unsigned long long _fbits(const ac_ieee_float<binary32> &v) {
  return (unsigned long long)v.data_ac_int().to_uint();
}
template <class T> inline unsigned long long _fbits(const T &v) {
#ifdef __SYNTHESIS__
  return (unsigned long long)v;
#else
  unsigned long long b = 0; std::memcpy(&b, &v, sizeof(T)); return b;
#endif
}
// Reconstruct a memory element from the DATAW raw bits: value-cast for integers,
// set_data() bit-load for floats (a value-cast would corrupt the float; memcpy is
// rejected under synthesis as above).
// tb: data files hold float text -> read a float and convert.
inline std::istream &operator>>(std::istream &is, half &h) { float f; is >> f; h = half(f); return is; }
inline std::istream &operator>>(std::istream &is, ac_ieee_float<binary32> &h) {
  float f; is >> f; h = ac_ieee_float<binary32>(f); return is;
}
// tb/Connections waveform trace of a float (trace its raw bit pattern). Needed
// because Connections/sc_signal ports templated on these types call sc_trace.
inline void sc_trace(sc_core::sc_trace_file *tf, const half &h, const std::string &n) {
  sc_trace(tf, (unsigned short)_fbits(h), n);
}
inline void sc_trace(sc_core::sc_trace_file *tf, const ac_ieee_float<binary32> &h,
                     const std::string &n) {
  sc_trace(tf, (unsigned)_fbits(h), n);
}
// Make ac_ieee_float<Format> a valid Connections channel/Combinational payload.
// Connections' marshaller.h ships Wrapped<> specializations for ac_std_float,
// ac::bfloat16 and ac_float, but NOT ac_ieee_float -- so a Stream/Channel of
// f32 (ac_ieee_float<binary32>) fails to synthesize (marshaller.h needs
// T::width + T::Marshall()). This specialization marshals the raw IEEE bits
// (data_ac_int()/set_data(), a lossless bit copy through the standard ac_int
// AddField path), exactly as the ac_std_float specialization does. Wrapped lives
// at GLOBAL scope (marshaller.h opens no namespace), so this must be global too.
// Guarded: the SCVerify wrapper TU (sysc_sim.cpp) needs this same specialization,
// and the cosim flow re-injects an identical guarded copy into sysc_sim.h; the
// guard keeps the kernel.cpp TU (which sees both this AND, via mc_scverify ->
// sysc_sim.h, the injected copy) from double-defining it.
#ifndef ALLO_IEEE_FLOAT_MARSHALL_DEF
#define ALLO_IEEE_FLOAT_MARSHALL_DEF
template <ac_ieee_float_format Format>
class Wrapped<ac_ieee_float<Format> > {
public:
  ac_ieee_float<Format> val;
  Wrapped() : val(0.0f) {}
  Wrapped(const ac_ieee_float<Format> &v) : val(v) {}
  static const unsigned int width = ac_ieee_float<Format>::width;
  static const bool is_signed = 1;
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    ac_int<ac_ieee_float<Format>::width, true> bits = val.data_ac_int();
    m & bits;                // packs bits on marshal-out, fills bits on marshal-in
    val.set_data(bits);      // write unpacked bits back into the float
  }
};
#endif // ALLO_IEEE_FLOAT_MARSHALL_DEF
// mc_scverify.h MUST come after the float shims above: under CCS_DUT_RTL it pulls
// in the SCVerify RTL wrapper (sysc_sim.h), which instantiates the ports'
// Wrapped<ac_ieee_float<...>>::Marshall and sc_trace(ac_ieee_float) at include
// time. A C++ specialization must be visible BEFORE the first implicit
// instantiation, so both must be declared above this include -- otherwise the
// wrapper binds the primary Wrapped<T> template ("ac_ieee_float has no member
// Marshall") and float-channel cosim fails to compile.
#include <mc_scverify.h>      // SCVerify testbench macros (CCS_MAIN / CCS_DESIGN)
// Single-shot completion counter (csim/testbench ONLY). Each kernel bumps this
// once, right after its body finishes its single pass; the sc_main testbench for
// memory-mapped-output designs advances the clock until all kernels are done
// before reading the memories out. Declared unconditionally so the tb compiles
// under __SYNTHESIS__ too, but the kernel's increment is guarded out of synthesis
// (see emitKernelModule) so the synthesized logic stays side-effect free.
static long __allo_done = 0;
// The reused body emits bare max()/min() for the Allo max/min intrinsics (Vitis
// resolves them via hls::); bind them to std:: so the same body compiles here.
using std::max;
using std::min;
// The reused Vivado-emitter body prints Vitis ap_(u)int types; alias them to
// Catapult's ac_int so the same body compiles.
// TODO: REVISIT — this ap_int/ap_rng shim exists ONLY to keep the reused Vivado body
// compiling; ap_int is just ac_int + two Vitis affordances (x(hi,lo) via ap_rng, >64-bit
// operator long long() narrowing). Emitting ac_int natively everywhere (a type-name
// override like getCatapultTypeName, plus the bit-op overrides already doing slc/set_slc)
// would let this shim + ap_rng be deleted. See EmitSystemC.md.
// ap_(u)int is a thin ac_int subclass adding the two Vitis affordances the reused
// body relies on and ac_int lacks:
//  (1) the x(hi,lo) BIT-RANGE operator (packed streams unpack via v(31,16) etc) —
//      a proxy that extracts on read and inserts on write, for const/runtime hi,lo;
//  (2) for W>64 only, an implicit narrowing conversion (`int32_t x = wide;`, e.g. a
//      GEMM accumulator past 64 bits) which ac_int omits above 64 bits.
// Adding operator() doesn't disturb arithmetic (it's the call operator, not a
// conversion); the W>64 narrowing stays gated so ac_int's own operators keep
// winning overload resolution (no builtin-conversion ambiguity).
//
// CSYNTH NOTE: the subclass-of-ac_int form below is CSIM-ONLY. Catapult's front-
// end treats ac_int as a builtin, so a struct deriving from it trips
// "struct assignment from non-struct type" (CIN-15) on every `v = <ac_int expr>;`
// — breaking synthesis branch-wide. Under __SYNTHESIS__ we therefore fall back to
// a PLAIN ac_int alias, which loses the two csim affordances (the x(hi,lo) bit-
// range and the >64-bit implicit narrowing). Consequence: every non-bit-slicing
// design synthesizes; a design that actually bit-slices (packed streams) fails at
// its `(hi,lo)` site instead — a clear, local error, and those need native ac_int
// .slc emission anyway. csim keeps the full-featured shim so behavior is unchanged.
// TODO: REVISIT — packed/wide bit-slicing fails at csynth here (CIN-15 forces this bare
// alias); emit those slices as native ac_int .slc<>()/.set_slc() (cf. emitGetSlice/
// emitSetSlice) so packed streams synthesize. Empirically checked 2026-08 (see EmitSystemC.md).
#ifdef __SYNTHESIS__
template <int W> using ap_int = ac_int<W, true>;
template <int W> using ap_uint = ac_int<W, false>;
#else
template <class AC> struct ap_rng {
  AC &r;
  int hi, lo;
  ap_rng(AC &x, int h, int l) : r(x), hi(h), lo(l) {}
  operator long long() const { // read bits [hi:lo]
    AC m = (AC(1) << (hi - lo + 1)) - 1;
    return ((r >> lo) & m).to_int64();
  }
  template <class V> ap_rng &operator=(V v) { // write bits [hi:lo] = v
    AC m = (AC(1) << (hi - lo + 1)) - 1;
    r = (r & ~(m << lo)) | ((AC(v) & m) << lo);
    return *this;
  }
};
// `using ac_int<W,S>::ac_int;` inherits ac_int's ctors, but C++ never inherits the
// copy-shaped ctor (parameter = the base type), so building the wrapper from an ac_int
// expression -- what every arith/logic op yields -- needs this explicit converting ctor.
// Without it construction fails "non-scalar conversion" at EVERY width (verified 16/22/34/
// 64/68); standard widths 8/16/32/64 dodge it only because they emit as native (u)intN_t,
// not this shim. (>64 additionally has no native-int narrowing -- see getSCTypeName.)
template <int W, bool Big = (W > 64)> struct ap_sel {
  struct s : ac_int<W, true> {
    using ac_int<W, true>::ac_int;
    s() = default;
    s(const ac_int<W, true> &v) : ac_int<W, true>(v) {}
    ap_rng<ac_int<W, true>> operator()(int hi, int lo) { return {*this, hi, lo}; }
  };
  struct u : ac_int<W, false> {
    using ac_int<W, false>::ac_int;
    u() = default;
    u(const ac_int<W, false> &v) : ac_int<W, false>(v) {}
    ap_rng<ac_int<W, false>> operator()(int hi, int lo) { return {*this, hi, lo}; }
  };
};
template <int W> struct ap_sel<W, true> {
  struct s : ac_int<W, true> {
    using ac_int<W, true>::ac_int;
    s() = default;
    s(const ac_int<W, true> &v) : ac_int<W, true>(v) {}
    ap_rng<ac_int<W, true>> operator()(int hi, int lo) { return {*this, hi, lo}; }
    operator long long() const { return this->to_int64(); }
  };
  struct u : ac_int<W, false> {
    using ac_int<W, false>::ac_int;
    u() = default;
    u(const ac_int<W, false> &v) : ac_int<W, false>(v) {}
    ap_rng<ac_int<W, false>> operator()(int hi, int lo) { return {*this, hi, lo}; }
    operator unsigned long long() const { return this->to_uint64(); }
  };
};
template <int W> using ap_int = typename ap_sel<W>::s;
template <int W> using ap_uint = typename ap_sel<W>::u;
#endif

// A DEFINED zero for any memory element type. `T()` is not usable: ac_ieee_float's
// default constructor is `{}`, which leaves the payload uninitialised, and a plain
// literal 0 does not convert (sc_out<ac_ieee_float>::write takes const T& and there is
// no implicit int->T conversion). Only needed for the reset action; the float cases get
// an explicit construction from 0.0f.
template <typename T> inline T _mem_zero() { return (T)0; }
template <> inline half _mem_zero<half>() { return half(0.0f); }
template <>
inline ac_ieee_float<binary32> _mem_zero<ac_ieee_float<binary32> >() {
  return ac_ieee_float<binary32>(0.0f);
}

// Pin-interface memory: the counterpart of the RAM pins a kernel presents. Used in BOTH
// places, which is the point -- a kernel cannot tell whether its memory sits inside the
// design (a replicated, multi-client array) or in the testbench (a single-client array
// exposed at the boundary). Only the binding differs.
//
// Synchronous read: the address is captured on the clock edge and `q` is presented on the
// NEXT one, which is why the kernel-side _rd() accessor waits twice.
template <typename T, int SIZE, int ADDRW>
SC_MODULE(AlloMemPins) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_in< ac_int<ADDRW, false> > radr;
  sc_in<bool> re;
  sc_out<T> q;
  sc_out<bool> rrdy;
  sc_in< ac_int<ADDRW, false> > wadr;
  sc_in<T> d;
  sc_in<bool> we;
  sc_out<bool> wrdy;
  T mem[SIZE];
  SC_HAS_PROCESS(AlloMemPins);
  AlloMemPins(sc_module_name n) : sc_module(n) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    // NOTE: mem[] is deliberately NOT cleared here -- the testbench preloads it before
    // reset is released, exactly as it did with AlloMem, and clearing would wipe that.
    // This memory is ALWAYS READY: single-cycle, unarbitrated, no bank conflicts. The
    // ready lines exist for the integrator's memory, which may not be -- here they are
    // simply held high, so the accessors' stall loops fall through in one edge.
    rrdy.write(true);
    wrdy.write(true);
    // `q` is a driven sc_out, so Catapult requires it written in the reset action too
    // (CIN-233). Only reachable when this memory is INSIDE the design -- a replicated,
    // multi-client array -- since a testbench-side instance is never synthesized.
    //
    // NOT `T()`: ac_ieee_float's default constructor is `{}`, so that would leave the
    // payload UNINITIALISED and the reset value of a float data pin indeterminate.
    q.write(_mem_zero<T>());
    wait();
    while (1) {
      // Both accesses are GUARDED BY THEIR ENABLE and bounds-checked. Not defensive
      // padding: a write-only array leaves `radr` an undriven sc_signal, and reading
      // mem[] at that uninitialized address indexes out of bounds and segfaults --
      // which is exactly how this first showed up (mem_port_scatter).
      unsigned wa = wadr.read().to_uint();
      if (we.read() && wa < (unsigned)SIZE)
        mem[wa] = d.read();
      unsigned ra = radr.read().to_uint();
      if (re.read() && ra < (unsigned)SIZE)
        q.write(mem[ra]);
      wait();
    }
  }
};

// Depth-N buffered stream channel (Stream[T, N>=1]) — a TWO-THREAD ring-buffer
// FIFO that Catapult can synthesize AND schedule.
//
// Single-thread ring FIFO. This was historically TWO threads (enq/deq) because a
// single SC_THREAD doing BOTH in.PopNB() and out.PushNB() couples the two
// handshakes' sc_signal writes, and Catapult's DEFAULT -IO_MODE fixed can't place
// them at the fixed cycle offsets its iomode requires -> the loop won't close at
// II=1 (SCHD-30). The systemc flow now sets -IO_MODE super (catapult.py), which
// lets the scheduler place BOTH handshakes within the loop window -- the same fix
// that makes multi-handshake router kernels schedule -- so the bidirectional FIFO
// schedules in ONE thread. That collapses everything the split needed: the
// cross-thread head/tail sc_signals, the shared sc_signal register file (the
// HIER-41 dodge for a plain array shared across threads), and the sacrificed N+1
// slot -> plain single-owner locals + a count. PushNB runs before PopNB so an
// empty FIFO does not forward an arriving element the same cycle (preserves the
// >=1-cycle buffer latency of the two-thread version).
template <typename T, int N>
SC_MODULE(AlloFifo) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In<T> in;
  Connections::Out<T> out;
  // Occupancy sidebands: regular Connections In/Out ports cannot report Empty/Full
  // in HLS (In::Empty() reads a sim-only latched-data flag our channel never sets),
  // so a Stream.empty()/full() query reads these plain status wires instead. `count`
  // mirrors the thread-local `cnt`; empty_o/full_o are driven COMBINATIONALLY from it
  // (registered-pointers + combinational-empty, matching RTL FIFO semantics -- a
  // registered flag would be a cycle stale and shift arbitration traces).
  sc_signal<int> count;
  sc_out<bool> empty_o;
  sc_out<bool> full_o;
  SC_HAS_PROCESS(AlloFifo);
  AlloFifo(sc_module_name nm) : sc_module(nm), in("in"), out("out") {
    SC_THREAD(run); sensitive << clk.pos(); async_reset_signal_is(rst, false);
    SC_METHOD(set_flags); sensitive << count;
  }
  void set_flags() {
    empty_o.write(count.read() == 0);
    full_o.write(count.read() == N);
  }
  void run() {
    in.Reset();
    out.Reset();
    T buf[N];                         // single owner -> plain local, no HIER-41
    int head = 0, tail = 0, cnt = 0;  // full N slots (cnt distinguishes full/empty)
    count.write(0);
    wait();
    while (1) {
      if (cnt > 0 && out.PushNB(buf[head])) { head = (head + 1) % N; cnt = cnt - 1; }
      if (cnt < N) {
        T v;
        if (in.PopNB(v)) { buf[tail] = v; tail = (tail + 1) % N; cnt = cnt + 1; }
      }
      count.write(cnt);               // mirror cnt out for the combinational flags
      wait();
    }
  }
};

// AlloFifoC -- the ACTIVE buffered-stream FIFO. Subclasses the vendor's
// first-word-fall-through Connections::Fifo (enq/deq In/Out ports) and adds
// coherent occupancy flags from the port signals (deq.vld == !empty,
// enq.rdy == !full, the vendor Fifo_with_idle pattern). Because Connections::Fifo
// is FWFT (the head word is presented combinationally and a slot is only freed
// AFTER the consumer's Pop handshake completes), empty_o agrees with what the
// consumer can read THIS cycle -- fixing the last-item drop that AlloFifo's eager
// PushNB + early decrement caused for empty()/full()-polling consumers. It is
// also smaller/faster (~49% area, higher Fmax) since it uses nbits<N> pointers
// and a 1-bit full instead of a 32-bit count. AlloFifo above is kept as an
// unused reference/backup.
template <typename T, int N>
struct AlloFifoC : public Connections::Fifo<T, N> {
  SC_HAS_PROCESS(AlloFifoC);
  typedef Connections::Fifo<T, N> Base;
  using Base::enq;   // In<T>  -- the top binds this via .enq(<stream>_in)
  using Base::deq;   // Out<T> -- the top binds this via .deq(<stream>_out)
  using Base::sensitive;
  sc_out<bool> empty_o;
  sc_out<bool> full_o;
  AlloFifoC(sc_module_name nm) : Base(nm), empty_o("empty_o"), full_o("full_o") {
    SC_METHOD(gen_status);
    sensitive << enq._RDYNAME_ << deq._VLDNAME_;
  }
  void gen_status() {
    empty_o.write(!deq._VLDNAME_.read()); // deq.vld == !empty (FWFT-coherent)
    full_o.write(!enq._RDYNAME_.read());  // enq.rdy == !full
  }
};

)XXX";
  if (std::getenv("ALLO_SYNC_RESET")) {
    // swap the async reset in the module templates (AlloMemPins/AlloFifo) to sync
    std::string dh(device_header);
    for (size_t p; (p = dh.find("async_reset_signal_is")) != std::string::npos;)
      dh.replace(p, /*len("async_reset_signal_is")=*/21, "reset_signal_is");
    os << dh;
  } else {
    os << device_header;
  }

  // Helper functions (pure compute — no `top`/`df.kernel`/`dataflow` attr) are
  // emitted first as plain C++ free functions (reusing the base emitter's
  // return-by-pointer convention, matching the `helper(a,b,&r)` call sites the
  // kernel bodies already emit). They must precede the modules that call them.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!func->hasAttr("top") && !func->hasAttr("df.kernel") &&
        !func->hasAttr("dataflow"))
      VhlsModuleEmitter::emitFunction(func);
  }

  StringRef topName;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr("top")) {
      topName = func.getName();
      emitTopModule(func);
    } else if (func->hasAttr("df.kernel")) {
      emitKernelModule(func);
    }
    // else: sub-region (`dataflow` attr) — hierarchy, TODO (structural).
  }

  // Testbench: a Combinational channel per top stream port + clocked src/sink
  // threads. src Pushes a known pattern into every INPUT port; sink Pops + prints
  // every OUTPUT port then sc_stop()s. (Ignored by synthesis; only for csim.)
  if (!topName.empty()) {
    os << "SC_MODULE(tb) {\n";
    addIndent();
    indent(); os << "sc_clock clk;\n";
    indent(); os << "sc_signal<bool> rst;\n";
    indent(); os << topName << " dut;\n";
    indent(); os << "sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)\n";
    for (auto &a : ioArrays) {
      indent();
      os << "Connections::Combinational< " << a.ctype << " > ch_" << a.member
         << ";\n";
    }
    // The random-access memories live HERE, not in the DUT: a boundary array is
    // storage the design ACCESSES, not storage it CONTAINS. Keeping them inside made
    // every workload array synthesize into the design (a region whose arrays are all
    // random-access emitted `module top(clk, rst, done)` with the arrays as registers),
    // which inflates every area number and is not an integratable interface.
    for (auto &mi : memInsts) {
      if (!mi.exposed)
        continue; // replicated: still inside the DUT
      std::string aT = "ac_int<" + std::to_string(mi.addrw) + ", false>";
      indent(); os << "sc_signal< " << aT << " > " << mi.chan << "_radr, " << mi.chan
                   << "_wadr;\n";
      indent(); os << "sc_signal<bool> " << mi.chan << "_re, " << mi.chan << "_we, "
                   << mi.chan << "_rrdy, " << mi.chan << "_wrdy;\n";
      indent(); os << "sc_signal< " << mi.ctype << " > " << mi.chan << "_q, "
                   << mi.chan << "_d;\n";
      indent();
      os << "AlloMemPins< " << mi.ctype << ", " << mi.total << ", " << mi.addrw
         << " > " << mi.chan << "_mem;\n";
    }
    indent(); os << "SC_HAS_PROCESS(tb);\n";
    indent();
    os << "tb(sc_module_name n) : sc_module(n), clk(\"clk\", 1, SC_NS), dut(\"dut\")";
    for (auto &a : ioArrays)
      os << ", ch_" << a.member << "(\"ch_" << a.member << "\")";
    for (auto &mi : memInsts) {
      if (!mi.exposed)
        continue;
      os << ", " << mi.chan << "_mem(\"" << mi.chan << "_mem\")";
    }
    os << " {\n";
    addIndent();
    indent(); os << "dut.clk(clk); dut.rst(rst); dut.done(done_sig);\n";
    for (auto &a : ioArrays) {
      indent();
      os << "dut." << a.member << "(ch_" << a.member << ");\n";
    }
    // Bind each exposed memory to the DUT's matching port pair, and clock it here.
    for (auto &mi : memInsts) {
      if (!mi.exposed)
        continue;
      indent();
      os << mi.chan << "_mem.clk(clk); " << mi.chan << "_mem.rst(rst);\n";
      // The memory has all six pins and EVERY one must be bound or SystemC rejects
      // elaboration; the DUT only has the pins for its direction. So always bind the
      // memory, and bind the DUT side only where that pin exists. An unused side ends
      // up on a signal nobody drives, which is harmless.
      for (const char *sfx : {"_radr", "_re", "_q", "_rrdy", "_wadr", "_d", "_we", "_wrdy"}) {
        // _radr/_re/_q/_rrdy are the read side; _wadr/_d/_we/_wrdy the write side.
        bool isRead = (sfx[1] == 'r' || sfx[1] == 'q');
        bool dutHas = isRead ? (mi.dir != 'o') : (mi.dir != 'i');
        indent();
        if (dutHas)
          os << "dut." << mi.base << sfx << "(" << mi.chan << sfx << "); ";
        os << mi.chan << "_mem." << (sfx + 1) << "(" << mi.chan << sfx << ");\n";
      }
    }
    indent(); os << "SC_THREAD(src); sensitive << clk.posedge_event(); "
                    "async_reset_signal_is(rst, false);\n";
    indent(); os << "SC_THREAD(snk); sensitive << clk.posedge_event(); "
                    "async_reset_signal_is(rst, false);\n";
    reduceIndent();
    indent(); os << "}\n";
    // src: drive each INPUT port from input<k>.data (written by hls.py from A)
    indent(); os << "void src() {\n";
    addIndent();
    for (auto &a : ioArrays)
      if (a.dir == 'i') { indent(); os << "ch_" << a.member << ".ResetWrite();\n"; }
    indent(); os << "wait();\n";
    for (auto &a : ioArrays)
      if (a.dir == 'i') {
        indent();
        // Wide read temp so a char-width element (int8_t/uint8_t) parses as an
        // integer, not a single character (see the memory-port preload note).
        bool isF = (a.ctype == "half" || a.ctype == "double" ||
                    a.ctype.find("ieee_float") != std::string::npos);
        std::string rt = isF ? a.ctype : std::string("long long");
        os << "{ std::ifstream _f(\"input" << a.fileIdx << ".data\"); " << rt
           << " _v; for (int f = 0; f < " << a.total << "; ++f) { _f >> _v; ch_"
           << a.member << ".Push((" << a.ctype << ")_v); } }\n";
      }
    reduceIndent();
    indent(); os << "}\n";
    // A stream OUTPUT (snk drains it) drives sc_stop; if there are only
    // memory-port outputs, fall back to a time-based run (below).
    bool hasStreamOut = false;
    for (auto &a : ioArrays)
      if (a.dir == 'o')
        hasStreamOut = true;
    int64_t maxTotal = 1;
    for (auto &a : ioArrays)
      maxTotal = std::max(maxTotal, a.total);
    for (auto &m : memArrays)
      maxTotal = std::max(maxTotal, m.total);

    // snk: write each OUTPUT port to output<k>.data (read back into B by hls.py)
    indent(); os << "void snk() {\n";
    addIndent();
    for (auto &a : ioArrays)
      if (a.dir == 'o') { indent(); os << "ch_" << a.member << ".ResetRead();\n"; }
    indent(); os << "wait();\n";
    for (auto &a : ioArrays)
      if (a.dir == 'o') {
        indent();
        // Symmetric to the input read: cast a char-width element to a wide int so
        // operator<< prints its NUMERIC value, not a character; floats keep full
        // round-trippable precision.
        bool isF = (a.ctype == "half" || a.ctype == "double" ||
                    a.ctype.find("ieee_float") != std::string::npos);
        if (isF)
          os << "{ std::ofstream _f(\"output" << a.fileIdx
             << ".data\"); for (int f = 0; f < " << a.total
             << "; ++f) _f << std::setprecision(9) << ch_" << a.member
             << ".Pop() << \"\\n\"; }\n";
        else
          os << "{ std::ofstream _f(\"output" << a.fileIdx
             << ".data\"); for (int f = 0; f < " << a.total
             << "; ++f) _f << (long long)(ch_" << a.member
             << ".Pop()) << \"\\n\"; }\n";
      }
    if (hasStreamOut) { indent(); os << "sc_stop();\n"; }
    reduceIndent();
    indent(); os << "}\n";
    reduceIndent();
    os << "};\n\n";

    os << "int sc_main(int, char *[]) {\n";
    addIndent();
    // static (not a stack local): the tb inlines the whole design (every
    // sub-module + AlloFifo buf[] + AlloMem mem[]), which at large mesh sizes
    // (e.g. EVA 8x8) overflows the ~8MB stack. Static storage has no such cap;
    // sc_main runs once so the single construction is unchanged.
    indent(); os << "static tb t(\"t\");\n";
    // RTL cosim (Catapult SCVerify on Xcelium/NCSC) compiles with
    // -DCONNECTIONS_ACCURATE_SIM, under which the Connections ConManager
    // requires the sim clock be registered before sc_start() -- else
    // connections.h asserts "call Connections::set_sim_clk(&clk)". Plain csim
    // (OSCI, no such define) does not need it, so this is compiled out there.
    indent(); os << "#ifdef CONNECTIONS_ACCURATE_SIM\n";
    indent(); os << "Connections::set_sim_clk(&t.clk);\n";
    indent(); os << "#endif\n";
    // Preload every INPUT memory (each shared-read replica gets its own copy)
    // from the array's input file (csim only: direct hierarchical poke of
    // AlloMem.mem[], done before reset is released).
    for (auto &mi : memInsts)
      if (mi.dir != 'o') { // 'i' and 'b' preload from their input file
        indent();
        // Read integers into a WIDE temp: a char-width element (int8_t/uint8_t is
        // signed/unsigned char) would otherwise trigger operator>>'s FORMATTED
        // CHARACTER extraction (reads '6','5' from "65"), not integer parsing.
        // Floats keep their own operator>> overload.
        bool isF = (mi.ctype == "half" || mi.ctype == "double" ||
                    mi.ctype.find("ieee_float") != std::string::npos);
        std::string rt = isF ? mi.ctype : std::string("long long");
        // Exposed memories live in the tb; replicated ones are still inside the DUT.
        std::string owner = mi.exposed ? "t." : "t.dut.";
        os << "{ std::ifstream _f(\"input" << mi.inIdx << ".data\"); " << rt
           << " _v; for (int f = 0; f < " << mi.total << "; ++f) { _f >> _v; "
           << owner << mi.chan << "_mem.mem[f] = (" << mi.ctype << ")_v; } }\n";
      }
    indent(); os << "t.rst = 0; sc_start(1, SC_NS);\n";
    // A stream output stops the sim via sc_stop (self-synchronizing: the sink
    // drains exactly N tokens). Memory-mapped outputs have no such token, so we
    // SINGLE-SHOT: advance the clock until every kernel has finished its one pass
    // (__allo_done == numKernelInsts), then settle any in-flight STORE handshakes,
    // then read the memories. This replaces the old fixed maxTotal*8 guess, which
    // under-ran deep/tiled designs (late tiles caught mid-compute) and, with the
    // former free-running body, over-accumulated `both` outputs.
    indent(); os << "t.rst = 1;\n";
    if (hasStreamOut) {
      indent(); os << "sc_start();\n";
    } else {
      // Poll the DUT's hardware `done` port (AND of all kernels' completion) --
      // valid in BOTH csim and RTL cosim, unlike the C-side __allo_done counter
      // (which the synthesized DUT can't touch). The cap only bounds wall-clock if
      // the design never asserts done (deadlock / missing token).
      int64_t capCycles = maxTotal * 2000 + 200000;
      indent();
      os << "for (long long _c = 0; _c < " << capCycles
         << "LL && !t.done_sig.read(); ++_c) sc_start(1, SC_NS); // until DUT done\n";
      indent();
      os << "sc_start(" << (memInsts.empty() ? 64 : 256)
         << ", SC_NS); // settle in-flight memory writes\n";
    }
    // Read each OUTPUT array out to output<k>.data (into B by hls.py). Shared-
    // write arrays are SPLIT across replicas that each wrote disjoint elements
    // (zero-initialized elsewhere), so sum the replicas element-wise.
    for (auto &m : memArrays)
      if (m.dir != 'i') { // 'o' and 'b' read out to their output file
        indent();
        os << "{ std::ofstream _f(\"output" << m.outIdx << ".data\");\n";
        indent();
        os << "  for (int f = 0; f < " << m.total << "; ++f) {\n";
        // float memories accumulate/write as float; integers as before.
        bool isFloat = (m.ctype == "half" || m.ctype == "double" ||
                        m.ctype.find("ieee_float") != std::string::npos);
        indent();
        os << (isFloat ? "    float _s = 0;\n" : "    long long _s = 0;\n");
        for (auto &mi : memInsts)
          if (mi.dir != 'i' && mi.outIdx == m.outIdx) {
            // Exposed memories live in the tb; replicated ones inside the DUT.
            std::string owner = mi.exposed ? "t." : "t.dut.";
            indent();
            if (isFloat)
              os << "    _s += " << owner << mi.chan << "_mem.mem[f].to_float();\n";
            else
              os << "    _s += (long long) " << owner << mi.chan
                 << "_mem.mem[f];\n";
          }
        indent();
        // float outputs: write full round-trippable precision (float32 needs 9
        // significant digits) so the data-file text doesn't lose bits.
        if (isFloat)
          os << "    _f << std::setprecision(9) << _s << \"\\n\";\n";
        else
          os << "    _f << _s << \"\\n\";\n";
        indent();
        os << "  } }\n";
      }
    indent(); os << "return 0;\n";
    reduceIndent();
    os << "}\n";
  }
}

//===----------------------------------------------------------------------===//
// Registration
// - entry point that actually runs emitter
//===----------------------------------------------------------------------===//

LogicalResult allo::emitSystemC(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os); // creates shared emitter state around output stream os
  SystemCModuleEmitter(state).emitModule(module); // constructs emitter and walks whole module, printing SystemC
  return failure(state.encounteredError); // reports success/failure
}

void allo::registerEmitSystemCTranslation() { // registers emitter as an MLIR translation named emit-systemc
  static TranslateFromMLIRRegistration toSystemC(
      "emit-systemc", "Emit SystemC", emitSystemC,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::allo::AlloDialect,
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::scf::SCFDialect,
          mlir::affine::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect,
          mlir::linalg::LinalgDialect
        >();
        // clang-format on
      });
}

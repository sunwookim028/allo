/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Based on EmitVivadoHLS.cpp for Catapult HLS support
 */

#include "allo/Translation/EmitCatapultHLS.h"
#include "allo/Dialect/Visitor.h"
#include "allo/Support/Utils.h"
#include "allo/Translation/EmitVivadoHLS.h" // Include Vivado emitter
#include "allo/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// used for determine whether to generate C++ default types or ac_(u)int
static bool BIT_FLAG = false;

// Set by getCatapultTypeName() when it meets a type it cannot spell; turned
// into an MLIR error (not a SIGABRT) by emitCatapultHLS().
static std::string CATAPULT_UNSUPPORTED_TYPE;

std::string mlir::allo::takeCatapultUnsupportedType() {
  std::string res = CATAPULT_UNSUPPORTED_TYPE;
  CATAPULT_UNSUPPORTED_TYPE.clear();
  return res;
}

llvm::SmallString<16> mlir::allo::getCatapultTypeName(Type valType) {
  if (auto arrayType = llvm::dyn_cast<ShapedType>(valType))
    valType = arrayType.getElementType();

  // Handle float types.
  // nangate-45nm_beh does not support native IEEE-754 float arithmetic.
  // Use ac_ieee_float<binary32> (from ac_std_float.h) for synthesizable f32.
  if (llvm::isa<BFloat16Type>(valType))
    // ac_std_float.h (shipped with Catapult, $MGC_HOME/shared/include) provides
    // a real bfloat16: ac::bfloat16 is ac_std_float<16, 8>. Its *arithmetic*
    // rounding mode is AC_TRN_ZERO by default; the generated header redefines
    // AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE to AC_RND_CONV so that add/mul round
    // to nearest-even like arith.addf/mulf do. Its float->bf16 CONSTRUCTOR
    // hard-codes AC_TRN_ZERO and cannot be overridden -- see
    // docs/source/developer/limitations.rst.
    return SmallString<16>("ac::bfloat16");
  else if (llvm::isa<Float16Type>(valType))
    // `half` is a Vitis spelling that no ac_types header defines; the emit
    // gate is what made this visible. ac_ieee_float<binary16> is the
    // counterpart of the binary32 case below.
    return SmallString<16>("ac_ieee_float<binary16>");
  else if (llvm::isa<Float32Type>(valType))
    return SmallString<16>("ac_ieee_float<binary32>");
  else if (llvm::isa<Float64Type>(valType))
    return SmallString<16>("double");

  // Handle integer types.
  else if (llvm::isa<IndexType>(valType))
    return SmallString<16>("int");
  else if (auto intType = llvm::dyn_cast<IntegerType>(valType)) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ac_int<1, false>");
    } else {
      std::string signedness = "";
      bool is_signed = (intType.getSignedness() !=
                        IntegerType::SignednessSemantics::Unsigned);
      if (!BIT_FLAG) {
        switch (intType.getWidth()) {
        case 8:
        case 16:
        case 32:
        case 64:
          if (!is_signed)
            signedness = "u";
          return SmallString<16>(signedness + "int" +
                                 std::to_string(intType.getWidth()) + "_t");
        default:
          return SmallString<16>("ac_int<" +
                                 std::to_string(intType.getWidth()) + ", " +
                                 (is_signed ? "true" : "false") + ">");
        }
      } else {
        return SmallString<16>("ac_int<" + std::to_string(intType.getWidth()) +
                               ", " + (is_signed ? "true" : "false") + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = llvm::dyn_cast<allo::FixedType>(valType))
    return SmallString<16>(
        "ac_fixed<" + std::to_string(fixedType.getWidth()) + ", " +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ", true>");

  else if (auto ufixedType = llvm::dyn_cast<allo::UFixedType>(valType))
    return SmallString<16>(
        "ac_fixed<" + std::to_string(ufixedType.getWidth()) + ", " +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) +
        ", false>");

  else if (auto streamType = llvm::dyn_cast<StreamType>(valType))
    return SmallString<16>(
        "ac_channel< " +
        std::string(getCatapultTypeName(streamType.getBaseType()).c_str()) +
        " >");

  else {
    // Never abort: an assert(1 == 0) here is a SIGABRT that kills the calling
    // process, so a user cannot tell "type unsupported" from "emitter crashed".
    // Record the type instead and let emitCatapultHLS() raise a failure.
    if (CATAPULT_UNSUPPORTED_TYPE.empty()) {
      std::string buf;
      llvm::raw_string_ostream ss(buf);
      valType.print(ss);
      CATAPULT_UNSUPPORTED_TYPE = ss.str();
    }
    return SmallString<16>("/*UNSUPPORTED-TYPE*/");
  }

  return SmallString<16>();
}

//===----------------------------------------------------------------------===//
// Catapult-specific implementations
//   (CatapultModuleEmitter is now declared in EmitCatapultHLS.h so downstream
//    emitters -- e.g. EmitCatapultHLS2 -- can inherit its C++ compute codegen.)
//===----------------------------------------------------------------------===//

// ac_ieee_float<binary32> has no constructor from double literals; float
// literals (with 'f' suffix) convert via the float constructor.
void CatapultModuleEmitter::emitFloatArrayElement(float value) {
  if (std::isfinite(value)) {
    // std::to_string gives 6 decimal places; append 'f' for float literal
    os << std::to_string(value) << "f";
  } else if (value > 0)
    os << "INFINITY";
  else
    os << "-INFINITY";
}

void CatapultModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                                      std::string name) {

  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getCatapultTypeName(val.getType()) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void CatapultModuleEmitter::emitFunctionDirectives(func::FuncOp func,
                                                   ArrayRef<Value> portList) {
  // hls_design top/block pragmas are now emitted BEFORE the function
  // declaration in emitFunction, so EDG can bind them correctly.
  // Only per-statement directives belong here (inside the function body).

  if (func->hasAttr("dataflow")) {
    indent();
    os << "#pragma hls_design dataflow\n";
  }

  // Emit array directives for function ports
  for (auto &port : portList)
    if (llvm::isa<MemRefType>(port.getType()))
      emitArrayDirectives(port);
}

void CatapultModuleEmitter::emitArrayDecl(Value array, bool isFunc,
                                          std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = llvm::cast<ShapedType>(array.getType());
  if (arrayType.hasStaticShape()) {
    auto memref = llvm::dyn_cast<MemRefType>(array.getType());
    if (memref) {
      auto attr = memref.getMemorySpace();
      // Use dyn_cast to safely check if attr is a StringAttr (it could be
      // IntegerAttr)
      auto strAttr = attr ? llvm::dyn_cast<StringAttr>(attr) : nullptr;
      if (strAttr && strAttr.getValue().str().substr(0, 6) == "stream") {
        // Value has been declared before or is a constant number.
        if (isDeclared(array)) {
          os << getName(array);
          return;
        }

        // print stream type using ac_channel instead of hls::stream
        os << "ac_channel< " << getCatapultTypeName(arrayType.getElementType())
           << " > ";

        auto attr_str = strAttr.getValue().str();
        int S_index = attr_str.find("S"); // spatial
        int T_index = attr_str.find("T"); // temporal
        if (isFunc &&
            !(((int)(arrayType.getShape().size()) > T_index - S_index) &&
              (T_index > S_index))) {
          os << "&"; // pass by reference, only non-array needs reference
        }

        // Add the new value to nameTable and emit its name.
        os << addName(array, /*isPtr=*/false, name);
        if ((int)(arrayType.getShape().size()) > T_index - S_index) {
          for (int i = 0; i < T_index - S_index; ++i)
            os << "[" << arrayType.getShape()[i] << "]";
        }
        // Add original array declaration as comment
        os << " /* ";
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
        os << " */";
      } else {
        emitValue(array, 0, false, name);
        for (auto &shape : arrayType.getShape())
          os << "[" << shape << "]";
      }
    } else { // tensor
      emitValue(array, 0, false, name);
    }
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
}

// Catapult loop pragmas must PRECEDE the loop header -- hls_pipeline_init_interval
// / hls_unroll bind to the construct that FOLLOWS them (Catapult's own matchlib
// examples place them before `while(1)`/`for`). Emitting them in the loop body (the
// Vivado convention the base for-emitter uses) makes Catapult drop them with
// CIN-319 "Cannot bind pragma to any valid construct" -- so pipelining silently
// never happened. The in-body hook is therefore a no-op; the pragmas are emitted
// from emitLoopDirectivesPreheader, which the base for-emitter calls just before
// the loop header.
void CatapultModuleEmitter::emitLoopDirectives(Operation *op) {}

void CatapultModuleEmitter::emitLoopDirectivesPreheader(Operation *op) {
  // Called at the loop's own indent level, immediately before the loop header.
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    indent();
    os << "#pragma hls_pipeline_init_interval "
       << llvm::cast<IntegerAttr>(ii).getValue() << "\n";
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    indent();
    auto val = llvm::cast<IntegerAttr>(factor).getValue();
    if (val == 0)
      os << "#pragma hls_unroll\n";
    else
      os << "#pragma hls_unroll " << val << "\n";
  }

  if (auto parallel = getLoopDirective(op, "parallel")) {
    indent();
    os << "#pragma hls_unroll\n"; // parallel implies full unroll
  }

  if (auto dataflow = getLoopDirective(op, "dataflow")) {
    indent();
    os << "#pragma hls_design dataflow\n";
  }
}

void CatapultModuleEmitter::emitStreamConstruct(allo::StreamConstructOp op) {
  indent();
  // Catapult requires local ac_channel declarations to be static, pointer, or
  // reference (HIER-6). Add 'static' so channels survive across invocations and
  // are not re-constructed each call (required for block synthesis).
  os << "static ";
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  if (auto shapedType = llvm::dyn_cast<ShapedType>(result.getType())) {
    for (auto shape : shapedType.getShape()) {
      os << "[" << shape << "]";
    }
  }
  os << ";\n";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitStreamTryGet(StreamTryGetOp op) {
  // Catapult synthesis: emit blocking read() instead of nb_read().
  // nb_read inside spin-while loops triggers Catapult go compile segfault (LOOP-19).
  // blocking read() always succeeds → spin-while exits in 1 iteration → bounded.
  // Area/timing estimates are equivalent; scheduling semantics differ only at runtime.
  Value result = op.getResult(0);
  Value success = op.getResult(1);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  auto stream = op->getOperand(0);

  // Declare the result data variable first.
  indent();
  emitValue(result);
  os << ";\n";

  // Emit: channel[idx].read(result);
  indent();
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (denseArrayAttr)
      for (int64_t v : denseArrayAttr.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".read(";
  emitValue(result);
  os << ");\n";

  // Success is always true: blocking read always returns data.
  std::string successName = std::string(addName(success, false).str());
  indent();
  os << "bool " << successName << " = true;\n";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitStreamTryPut(StreamTryPutOp op) {
  // Catapult synthesis: emit blocking write() instead of nb_write().
  // nb_write inside spin-while loops triggers Catapult go compile segfault (LOOP-19).
  // blocking write() always succeeds → spin-while exits in 1 iteration → bounded.
  // Area/timing estimates are equivalent; scheduling semantics differ only at runtime.
  Value success = op.getResult();
  auto stream = op->getOperand(0);
  auto value = op->getOperand(1);

  // Emit: channel[idx].write(value);
  // ac_channel::write(const T&) accepts both lvalues and rvalue literals.
  indent();
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (denseArrayAttr)
      for (int64_t v : denseArrayAttr.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".write(";
  os << getName(value) << ");\n";

  // Success is always true: blocking write always succeeds.
  std::string successName = std::string(addName(success, false).str());
  indent();
  os << "bool " << successName << " = true;\n";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitStreamEmpty(StreamEmptyOp op) {
  // ac_channel does NOT have .empty() in the synthesizable subset (EDG CIN-59).
  // Use !ch.available(1) instead: "no element available" == empty.
  Value result = op.getResult();
  auto stream = op->getOperand(0);

  indent();
  emitValue(result);
  os << " = !";
  emitValue(stream, 0, false);
  if (llvm::isa<ShapedType>(stream.getType())) {
    auto denseArrayAttr = op->getAttrOfType<DenseI64ArrayAttr>("indices");
    if (denseArrayAttr)
      for (int64_t v : denseArrayAttr.asArrayRef())
        os << "[" << v << "]";
  }
  os << ".available(1);\n";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitStreamFull(StreamFullOp op) {
  // ac_channel has no direct .full() API.
  // Synthesizable alternative: use nb_write() return value for backpressure.
  // Here we conservatively emit false (unbounded in Catapult sim by default).
  // Depth constraints are enforced via TCL directives at synthesis time.
  Value result = op.getResult();

  indent();
  emitValue(result);
  os << " = false;"
     << " /* ac_channel: no .full(); depth enforced via TCL directive */\n";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = llvm::cast<MemRefType>(memref.getType());

  // streaming
  auto attr = type.getMemorySpace();
  if (attr) {
    // Use dyn_cast to safely check if attr is a StringAttr (it could be
    // IntegerAttr)
    auto strAttr = llvm::dyn_cast<StringAttr>(attr);
    if (strAttr) {
      std::string attr_str = strAttr.getValue().str();
      if (attr_str.substr(0, 6) == "stream") {
        // Note: Catapult HLS doesn't need explicit stream pragmas like Vivado
        // HLS The streaming behavior is handled through ac_channel type
        return;
      }
    }
  }

  // Catapult ignores #pragma HLS array_partition, so nothing is emitted AFTER the
  // declaration. The memory-implementation directive it does understand is
  // #pragma hls_resource, which must PRECEDE the declaration -- see
  // emitArrayDirectivesPreheader below.
}

// A fully-partitioned array means REGISTERS, and Catapult spells that
//   #pragma hls_resource <name>_rsc variables="<name>" map_to_module="[Register]"
// placed immediately BEFORE the declaration (verified: Catapult acknowledges it with
// CIN-341 "Pragma 'hls_resource<..>' detected, variable = '..', module = '[Register]'").
//
// WHY THIS MATTERS. Without it Catapult maps any array big enough to a synchronous RAM
// (its generated tcl does `solution library add ccs_sample_mem`). For a flit buffer read
// once per slot that is fatal twice over: a 1R1W RAM allows ONE read per cycle, so five
// slot reads cannot be scheduled together (SCHD-4 "insufficient resources ...  5 are
// needed, but only 1 instances are available"), and Genus then treats the RAM as an
// unresolved black box whose area counts as ZERO -- an 8-deep buffer measured SMALLER
// than a 2-deep one. This used to be worked around with a TCL directive
// (`directive set /<top>/<proc>/<array>:rsc -MAP_TO_MODULE {[Register]}`), which cannot
// reach a cosim build because that flow never patches run.tcl.
//
// Reusing partition rather than inventing a primitive: s.partition(.., Partition.Complete)
// already MEANS "make this registers" -- that is exactly how the Vivado backend
// implements it -- so the Catapult spelling of the same request belongs here.
//
// SCOPE: LOCAL arrays only. emitAlloc returns early for a memref that is already declared
// (a function port), so a partitioned ARGUMENT never reaches here. Ports are a different
// Catapult concept anyway (hls_design_interface), not hls_resource -- so a partitioned
// port silently gets no directive. Not a problem for dataflow kernels, whose state arrays
// are all locals, but worth knowing before reaching for this on a port.
void CatapultModuleEmitter::emitArrayDirectivesPreheader(Value memref) {
  auto type = llvm::dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape())
    return;

  // Streams are ac_channel, not memories -- no resource directive applies.
  if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(type.getMemorySpace()))
    if (strAttr.getValue().str().substr(0, 6) == "stream")
      return;

  // Only a COMPLETE partition maps to registers. A block/cyclic partition asks for
  // several smaller memories, which is a different directive; leave those to the RAM.
  if (!getLayoutMap(type))
    return;
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (!isFullyPartitioned(type, dim))
      return;

  // The name must already exist: emitAlloc calls this before emitArrayDecl, which is
  // what ADDS the name. Resolve it the same way emitArrayDecl will, via the alloc's
  // "name" attribute, falling back to the declared name when there is no attribute.
  std::string name;
  if (auto *def = memref.getDefiningOp())
    if (auto attr = llvm::dyn_cast_or_null<StringAttr>(def->getAttr("name")))
      name = attr.getValue().str();
  if (name.empty())
    return; // unnamed temporary: nothing stable to bind the pragma to

  indent();
  os << "#pragma hls_resource " << name << "_rsc variables=\"" << name
     << "\" map_to_module=\"[Register]\"\n";
}

void CatapultModuleEmitter::emitFunction(func::FuncOp func) {
  if (func->hasAttr("bit"))
    BIT_FLAG = true;

  if (func.getBlocks().empty())
    // This is a declaration.
    return;

  if (func.getBlocks().size() > 1)
    emitError(func, "has more than one basic blocks.");

  // Emit hls_design pragma BEFORE the function declaration so EDG binds it.
  // Top functions get #pragma hls_design top.
  // Sub-functions are left without a block pragma; hierarchy is controlled
  // via TCL (solution design set -block) when needed.
  if (func->hasAttr("top")) {
    os << "/// This is top function.\n";
    os << "#pragma hls_design top\n";
  }

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        llvm::cast<StringAttr>(func->getAttr("inputs")).getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names =
        llvm::cast<StringAttr>(func->getAttr("outputs")).getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes = llvm::cast<StringAttr>(func->getAttr("itypes")).getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (llvm::isa<ShapedType>(arg.getType())) {
      if (llvm::isa<StreamType>(
              llvm::cast<ShapedType>(arg.getType()).getElementType())) {
        auto shapedType = llvm::dyn_cast<ShapedType>(arg.getType());
        // Use Catapult-specific stream type name
        os << getCatapultTypeName(arg.getType()) << " ";
        os << addName(arg, false);
        for (auto shape : shapedType.getShape())
          os << "[" << shape << "]";
      } else if (input_args.size() == 0) {
        emitArrayDecl(arg, true);
      } else {
        emitArrayDecl(arg, true, input_args[argIdx]);
      }
    } else {
      if (llvm::isa<StreamType>(arg.getType())) {
        // need to pass by reference - use Catapult-specific stream type
        os << getCatapultTypeName(arg.getType()) << "& ";
        os << addName(arg, false);
      } else if (input_args.size() == 0) {
        emitValue(arg);
      } else {
        emitValue(arg, 0, false, input_args[argIdx]);
      }
    }

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  auto args = func.getArguments();
  std::string otypes = "";
  if (func->hasAttr("otypes"))
    otypes = llvm::cast<StringAttr>(func->getAttr("otypes")).getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  if (auto funcReturn =
          dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        if (func.getArguments().size() > 0)
          os << ",\n";
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, otypes[idx] == 'u');
        if (llvm::isa<ShapedType>(result.getType())) {
          if (output_names != "")
            emitArrayDecl(result, true);
          else
            emitArrayDecl(result, true, output_names);
        } else {
          // In Catapult HLS, pointer indicates the value is an output.
          if (output_names != "")
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
          else
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
      idx += 1;
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);

  if (func->hasAttr("systolic")) {
    os << "#pragma scop\n";
  }
  emitBlock(func.front());
  if (func->hasAttr("systolic")) {
    os << "#pragma endscop\n";
  }

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

//===----------------------------------------------------------------------===//
// Native ac_int bit ops (replacing the inherited Vitis ap_int proxy forms)
//
// VhlsModuleEmitter emits bit indexing and slicing in the Vitis idiom:
//     ap_int<64> v16_tmp = v15;   v16 = v16_tmp(15, 0);
// Catapult has no ap_int at all -- `go analyze` stops at CRD-20 ("identifier
// ap_int is undefined"), 88 times on TinyTPU. ac_int spells the same thing as
// a member call: `t.slc<W>(lo)` reads W bits starting at lo, `t.set_slc(lo, v)`
// writes them. The wrapper temp is still needed because the emitted `num` may
// be a plain C int (int32_t) with no .slc/.set_slc/operator[] at all.
//
// The temp's SIGNEDNESS is taken from the source type, not fixed to signed:
// slc<W>(lo) returns ac_int<W, S> with S inherited from the object, so a
// signed temp over an unsigned source would sign-extend a slice whose result
// is wider than W. That failure is silent -- wrong numbers, not a compile
// error -- which is why it is worth spelling out here.
//
// Widths: GetIntSliceOp's result type is built as UInt(upper - lower) by the
// frontend (allo/ir/infer.py) and the emitted hi/lo are upper-1/lower, so the
// result's own width IS hi-lo+1 and can be used as the compile-time W that
// slc<> requires. (hi and lo are Values and need not be constants, so W cannot
// be recovered from them.)
//===----------------------------------------------------------------------===//

// "ac_int<W, S>" for the bit-op temp wrapping `val`, S from val's own type.
static std::string acIntTempType(Value val) {
  unsigned w = val.getType().getIntOrFloatBitWidth();
  bool isSigned = true;
  if (auto it = llvm::dyn_cast<IntegerType>(val.getType()))
    isSigned = it.getSignedness() != IntegerType::SignednessSemantics::Unsigned;
  return "ac_int<" + std::to_string(w) + ", " + (isSigned ? "true" : "false") +
         ">";
}

void CatapultModuleEmitter::emitGetBit(allo::GetIntBitOp op) {
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  indent();
  emitValue(result); // declares "<T> <res>"
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << acIntTempType(op.getNum()) << " _bs_" << rn << " = ";
  emitValue(op.getNum());
  os << ";\n";
  indent();
  os << rn << " = _bs_" << rn << "[";
  emitValue(op.getIndex());
  os << "];";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitSetBit(allo::SetIntBitOp op) {
  Value result = op.getResult();
  indent();
  emitValue(result);
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << acIntTempType(op.getNum()) << " _bs_" << rn << " = ";
  emitValue(op.getNum());
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

void CatapultModuleEmitter::emitGetSlice(allo::GetIntSliceOp op) {
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  unsigned w = result.getType().getIntOrFloatBitWidth();
  indent();
  emitValue(result);
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << acIntTempType(op.getNum()) << " _bs_" << rn << " = ";
  emitValue(op.getNum());
  os << ";\n";
  indent();
  os << rn << " = _bs_" << rn << ".slc<" << w << ">(";
  emitValue(op.getLo());
  os << ");";
  emitInfoAndNewLine(op);
}

void CatapultModuleEmitter::emitSetSlice(allo::SetIntSliceOp op) {
  Value result = op.getResult();
  // set_slc(lo, v) writes exactly v's width, so the value has to be wrapped in
  // an ac_int of the VALUE's width -- a plain C int would write 32 bits.
  unsigned vw = op.getVal().getType().getIntOrFloatBitWidth();
  indent();
  emitValue(result);
  os << ";\n";
  std::string rn = std::string(getName(result).str());
  indent();
  os << acIntTempType(op.getNum()) << " _bs_" << rn << " = ";
  emitValue(op.getNum());
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

// Narrowing a >64-bit ac_int to a native int/index needs an EXPLICIT
// .to_int64()/.to_uint64(): ac_int defines no implicit conversion to a C
// integer beyond 64 bits, so `int v36 = v35;` with v35 an ac_int<65, true>
// is Catapult CRD-413 ("no suitable conversion function ... exists"), 12
// times on TinyTPU. It also fails in plain g++ against hlslibs ac_types,
// which is what makes the pre-handoff gate possible without a licence.
// `index` is emitted as `int`, so it takes the signed form.
void CatapultModuleEmitter::emitNarrowCastSuffix(Value src, Value dst) {
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

void CatapultModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for Catapult High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
// ac::bfloat16's operators otherwise round toward zero (AC_TRN_ZERO); MLIR's
// arith.addf/mulf on bf16 round to nearest-even, so line the two up. Must
// precede ac_std_float.h, which reads the macro.
#ifndef AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE
#define AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE AC_RND_CONV
#endif
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <ac_std_float.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// catapult hls headers
#include "kernel.h"
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().has_value() && module.getName().value() == "host") {
    os << host_header;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunction(func);
      else if (auto cst = dyn_cast<memref::GlobalOp>(op))
        emitGlobal(cst);
      else
        emitError(&op, "is unsupported operation.");
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of allo-translate
//===----------------------------------------------------------------------===//

LogicalResult allo::emitCatapultHLS(ModuleOp module, llvm::raw_ostream &os) {
  AlloEmitterState state(os);
  takeCatapultUnsupportedType();
  CatapultModuleEmitter(state).emitModule(module);
  if (std::string bad = takeCatapultUnsupportedType(); !bad.empty()) {
    module.emitError("Catapult HLS emitter has no C++ spelling for type '")
        << bad << "'.";
    return failure();
  }
  return failure(state.encounteredError);
}

void allo::registerEmitCatapultHLSTranslation() {
  static TranslateFromMLIRRegistration toCatapultHLS(
      "emit-catapult-hls", "Emit Catapult HLS", emitCatapultHLS,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::allo::AlloDialect,
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::tensor::TensorDialect,
          mlir::scf::SCFDialect,
          mlir::affine::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect,
          mlir::linalg::LinalgDialect
        >();
        // clang-format on
      });
}
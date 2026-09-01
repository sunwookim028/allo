/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ISA_OPS_TD
#define ALLO_ISA_OPS_TD

include "allo/IR/Allo.td"
include "allo/IR/AlloISATypes.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Buffer access op interfaces
//
// Declared now so the interface .inc is generated and compiled. The access
// pattern ops that implement these (strided/tiled/expand_shape/...) land in
// Phase 1 — interfaces compile fine with zero implementors.
//===----------------------------------------------------------------------===//

def BufferAccessOpInterface : OpInterface<"BufferAccessOpInterface"> {
  let description = [{
    Interface for buffer access pattern operations (e.g. strided, tiled).
    Each access pattern can verify its compatibility with a buffer and
    materialize itself into concrete memory access IR.
  }];
  let cppNamespace = "::mlir::allo";
  let methods = [
    InterfaceMethod<"Verify that this access pattern is compatible with the buffer's slot type and address extents",
    "mlir::LogicalResult", "verifyCompatibility", (ins "BufferTypeInterface":$slotType, "llvm::ArrayRef<int64_t>":$extents)>,

    InterfaceMethod<"Materialize a read: extract a slice from the buffer according to this access pattern",
    "llvm::FailureOr<mlir::Value>", "materialize",
    (ins "OpBuilder&":$builder, "Location":$loc, "Value":$buffer)>,

    InterfaceMethod<"Materialize a write: insert a value into the buffer according to this access pattern",
    "llvm::FailureOr<mlir::Value>", "materialize",
    (ins "OpBuilder&":$builder, "Location":$loc, "Value":$value, "Value":$buffer)>
  ];
}

def BufferRelayoutOpInterface : OpInterface<"BufferRelayoutOpInterface"> {
  let cppNamespace = "::mlir::allo";
  let methods = [
    InterfaceMethod<"get the source access pattern operand",
    "mlir::Value", "getSource", (ins)>
  ];
}

//===----------------------------------------------------------------------===//
// Resource declaration ops
//===----------------------------------------------------------------------===//

def Allo_DeclareBufferOp : Allo_Op<"buffer", [Symbol]> {
  let summary = "Declare a named buffer in a hardware memory space";
  let description = [{
    Declares a buffer as an **address space times a slot**: `extents` is the
    array of addressable positions, and the buffer element type describes the
    access granularity of one position. For example,
    `allo.buffer @vecreg extents(8) : !allo.vector<16xf32>` declares 8 slots of
    16 f32 lanes each, and `allo.buffer @dram extents(1024, 64) : !allo.scalar<f32>`
    declares a row-major off-chip array addressed by two indices.

    An access pattern on this buffer carries one component per extent — that is
    the only dimensionality rule, and it applies to on-chip and off-chip memory
    alike (there is no separate "HBM" kind: off-chip memory is simply a buffer
    whose extents are multi-dimensional and whose slot is a scalar).
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DenseI64ArrayAttr:$extents,
    TypeAttrOf<BufferTypeInterface>:$bufferType);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Allo_DeclareStateOp : Allo_Op<"state", [Symbol]> {
  let summary = "Declare a named hardware state register with enumerated values";
  let description = [{
    Declares a state register that can hold one of a fixed set of enumerated
    values. For example, `allo.state @mode enums ["os", "ws"] : !allo.state<i32>`.
    An optional default value can be specified. Enum values are assigned integer
    indices by declaration order.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$enums,
    OptionalAttr<StrAttr>:$defaultState,
    TypeAttrOf<Allo_State>:$stateType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `enums` $enums (`default` $defaultState^)? attr-dict `:` $stateType
  }];
  let hasVerifier = 1;
}

def Allo_WriteStateOp : Allo_Op<"state.write",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Write an enumerated value to a state register";
  let description = [{
    Writes a statically-known enum value to a state register. The value must be
    one of the enums declared in the corresponding `allo.state` declaration.
  }];
  let arguments = (ins FlatSymbolRefAttr:$state, StrAttr:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `to` $state attr-dict
  }];
}

def Allo_ReadStateOp : Allo_Op<"state.read",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Read the current value of a state register";
  let description = [{
    Reads the current integer value of a state register. The result type must
    match the element type of the state declaration.
  }];
  let arguments = (ins FlatSymbolRefAttr:$state);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $state attr-dict `:` type($value)
  }];
}

def Allo_DeclareDescOp : Allo_Op<"desc", [Symbol]> {
  let summary = "Declare a named descriptor array with typed fields";
  let description = [{
    Declares a descriptor array with a symbol name, a number of slots, and a
    descriptor type specifying the named fields. For example,
    `allo.desc @tma size(8) : !allo.desc<["offset":i8, "stride":i8]>`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$size,
    TypeAttrOf<Allo_Descriptor>:$descType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `size` `(` $size `)` attr-dict `:` $descType
  }];
}

def Allo_WriteDescFieldOp : Allo_Op<"desc.write",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Write a value to a named field of a descriptor";
  let description = [{
    Writes a value to a specific field of a descriptor. The value type must match
    the field type declared in the descriptor type.
  }];
  let arguments = (ins FlatSymbolRefAttr:$desc, StrAttr:$field, AnyType:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `to` $field `in` $desc attr-dict `:` type($value)
  }];
}

def Allo_ReadDescFieldOp : Allo_Op<"desc.read",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Read a value from a named field of a descriptor";
  let description = [{
    Reads a value from a specific field of a descriptor. The result type must
    match the field type declared in the descriptor type.
  }];
  let arguments = (ins FlatSymbolRefAttr:$desc, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $field `from` $desc attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Instruction sequence container
//===----------------------------------------------------------------------===//

def Allo_SequenceOp : Allo_Op<"sequence",
    [AutomaticAllocationScope, IsolatedFromAbove, Symbol, NoTerminator]> {
  let summary = "Define a flat sequence of instructions";
  let description = [{
    Defines a named sequence of instructions. The body is a flat instruction
    stream; control flow is rejected until explicit support is added.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
  let extraClassDeclaration = [{
    Block *addEntryBlock() {
      return &getBody().emplaceBlock();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Buffer access pattern operations
//===----------------------------------------------------------------------===//

class Allo_BufferAccessOpBase<string mnemonic, list<Trait> traits = []> :
  Allo_Op<mnemonic, !listconcat(traits, [
  DeclareOpInterfaceMethods<BufferAccessOpInterface>, Pure,
  HasParent<"allo::DefineOp">])> {
  let results = (outs Allo_Pattern); // token representing the access pattern
}

def Allo_StridedOp : Allo_BufferAccessOpBase<"patterns.strided",
    [AttrSizedOperandSegments]> {
  let summary = "Strided buffer access pattern";
  let description = [{
    Defines a strided access pattern over a buffer. Parameters (basis, counts,
    strides) can be a mixture of static and dynamic values. For each dimension i,
    the accessed indices are: basis[i] + j * strides[i] for j in 0..counts[i]-1.
    On-chip buffers must be accessed with 1-D patterns.
  }];
  let arguments = (ins
    Variadic<Index>:$basis,
    Variadic<Index>:$counts,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$staticBasis,
    DenseI64ArrayAttr:$staticCounts,
    DenseI64ArrayAttr:$staticStrides);
  let assemblyFormat = [{
    `basis```custom<DynamicIndexList>(
      $basis, $staticBasis, "{}", "mlir::AsmParser::Delimiter::Paren")
    `counts```custom<DynamicIndexList>(
      $counts, $staticCounts, "{}", "mlir::AsmParser::Delimiter::Paren")
    `strides```custom<DynamicIndexList>(
      $strides, $staticStrides, "{}", "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
  let hasVerifier = 1;
}

def Allo_TiledOp : Allo_BufferAccessOpBase<"patterns.tiled",
    [AttrSizedOperandSegments]> {
  let summary = "Tiled buffer access pattern";
  let description = [{
    Extends strided access with a tile_size per dimension: each stride step
    accesses a contiguous tile of tile_size elements. Tile sizes must not exceed
    strides (no overlapping tiles). On-chip buffers must be accessed with 1-D
    patterns, and the tile size must evenly divide the buffer element size.
  }];
  let arguments = (ins
    Variadic<Index>:$basis,
    Variadic<Index>:$counts,
    Variadic<Index>:$strides,
    Variadic<Index>:$tileSizes,
    DenseI64ArrayAttr:$staticBasis,
    DenseI64ArrayAttr:$staticCounts,
    DenseI64ArrayAttr:$staticStrides,
    DenseI64ArrayAttr:$staticTileSizes);
  let assemblyFormat = [{
    `basis```custom<DynamicIndexList>(
      $basis, $staticBasis, "{}", "mlir::AsmParser::Delimiter::Paren")
    `counts```custom<DynamicIndexList>(
      $counts, $staticCounts, "{}", "mlir::AsmParser::Delimiter::Paren")
    `strides```custom<DynamicIndexList>(
      $strides, $staticStrides, "{}", "mlir::AsmParser::Delimiter::Paren")
    `tile_sizes```custom<DynamicIndexList>(
      $tileSizes, $staticTileSizes, "{}", "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
  let hasVerifier = 1;
}

def Allo_ExpandShapeOp : Allo_BufferAccessOpBase<"patterns.expand_shape",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Shape expansion access pattern (inverse of collapse_shape)";
  let description = [{}];
  let arguments = (ins
    Allo_Pattern:$source,
    IndexListArrayAttr:$reassociation,
    Variadic<Index>:$outputShape,
    DenseI64ArrayAttr:$staticOutputShape);
  let assemblyFormat = [{
    $source $reassociation `output_shape`
    custom<DynamicIndexList>($outputShape, $staticOutputShape)
    attr-dict
  }];
  let extraClassDeclaration = [{
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    }
  }];
}

def Allo_CollapseShapeOp : Allo_BufferAccessOpBase<"patterns.collapse_shape",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Shape collapse access pattern (inverse of expand_shape)";
  let description = [{}];
  let arguments = (ins Allo_Pattern:$source, IndexListArrayAttr:$reassociation);
  let assemblyFormat = [{
    $source $reassociation attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    }
  }];
}

def Allo_TransposeOp : Allo_BufferAccessOpBase<"patterns.transpose",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Transpose access pattern (permutes dimensions)";
  let description = [{
    Permutes the dimensions of the accessed data. During materialization, this
    generates a `linalg.transpose` on the read path and an inverse transpose on
    the write path.
  }];
  let arguments = (ins Allo_Pattern:$source, DenseI64ArrayAttr:$permutation);
  let assemblyFormat = [{
    $source `permutation` `=` $permutation attr-dict
  }];
  let extraClassDeclaration = [{
    SmallVector<int64_t> getInversePermutation() {
      auto perm = getPermutation();
      SmallVector<int64_t> inv(perm.size());
      for (unsigned i = 0; i < perm.size(); ++i)
        inv[perm[i]] = i;
      return inv;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Instruction operations
//===----------------------------------------------------------------------===//

def Allo_YieldOp : Allo_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "Yield values from an addr or compute region";
  let description = [{
    Terminates an addr region (yielding buffer access patterns) or a compute
    region (yielding destination buffer values).
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

def Allo_DefineOp : Allo_Op<"define", [Symbol,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  IsolatedFromAbove]> {
  let summary = "Define a hardware instruction with access patterns and compute semantics";
  let description = [{
    Defines a named instruction template with source/destination buffer
    references, an `addr` region for buffer access patterns, and a `compute`
    region for computation semantics. The addr region's block arguments are
    address parameters (index type); the compute region's first N block
    arguments correspond to source+destination buffers (ranked tensors), with
    any additional arguments being compute parameters (int/index).
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefArrayAttr:$sources,
    FlatSymbolRefArrayAttr:$destinations);
  let results = (outs);
  let regions = (region SizedRegion<1>:$access, SizedRegion<1>:$semantics);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    Block &getAccessBlock() { return getRegion(0).front(); }
    Block &getSemanticsBlock() { return getRegion(1).front(); }
    ArrayRef<BlockArgument> getExtraComputeArgs() {
      unsigned nBuffers = getSources().size() + getDestinations().size();
      return ArrayRef<BlockArgument>(getSemanticsBlock().getArguments()).drop_front(nBuffers);
    }
  }];
}

def Allo_EmitOp : Allo_Op<"emit", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments]> {
  let summary = "Emit (invoke) a defined instruction at the current program point";
  let description = [{
    Invokes an instruction defined by `allo.define`. The `addr` parameters are
    mapped to the addr region's block arguments to materialize buffer access
    patterns. The `compute` parameters are mapped to the extra (non-buffer)
    block arguments of the compute region.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$instruction,
    Variadic<Index>:$addrParams,
    Variadic<Index>:$computeParams,
    DenseI64ArrayAttr:$staticAddrParams,
    DenseI64ArrayAttr:$staticComputeParams);
  let results = (outs);
  let assemblyFormat = [{
    $instruction
    `addr```custom<DynamicIndexList>(
      $addrParams, $staticAddrParams, "{}", "mlir::AsmParser::Delimiter::Paren")
    `compute```custom<DynamicIndexList>(
      $computeParams, $staticComputeParams, "{}", "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
}

def Allo_LaunchOp : Allo_Op<"launch",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Launch a sequence of instructions";
  let description = [{
    Launches a sequence defined by `allo.sequence` to execute.
  }];
  let arguments = (ins FlatSymbolRefAttr:$sequence, UnitAttr:$async);
  let results = (outs);
  let assemblyFormat = [{
    $sequence (`async` $async^)? attr-dict
  }];
}

#endif // ALLO_ISA_OPS_TD

/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ISA_TYPES_TD
#define ALLO_ISA_TYPES_TD

include "allo/IR/Allo.td"

//===----------------------------------------------------------------------===//
// Buffer element type interface
//===----------------------------------------------------------------------===//

def BufferTypeInterface : TypeInterface<"BufferTypeInterface"> {
  let description = [{
    Common interface for all buffer element types (scalar, vector, tile). Each
    describes the access granularity of *one addressable position* in a hardware
    scratchpad or memory space; how many such positions there are, and along how
    many axes, is the buffer's `extents` (see `allo.buffer`).
  }];
  let cppNamespace = "::mlir::allo";
  let methods = [
    InterfaceMethod<"Get the scalar element type (e.g. f32, bf16)",
    "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Get the rank of a single buffer element (0 for scalar, 1 for vector, etc.)",
    "unsigned", "getRank", (ins)>,
    InterfaceMethod<"Get the shape of a single buffer element",
    "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
  ];
}

class Allo_BufferTypeBase<string mnemonic> :
    Allo_Type<mnemonic, [BufferTypeInterface]>;

//===----------------------------------------------------------------------===//
// Buffer element types
//===----------------------------------------------------------------------===//

def Allo_ScalarBuffer : Allo_BufferTypeBase<"ScalarBuffer"> {
  let summary = "On-chip scalar buffer element type";
  let description = [{
    Element type for on-chip buffers accessed at scalar granularity. Each buffer
    slot holds a single scalar value. Syntax: `!allo.scalar<f32>`.
  }];
  let parameters = (ins "Type":$elementType);
  let mnemonic = "scalar";
  let assemblyFormat = "`<` $elementType `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return 0; }
    ArrayRef<int64_t> getShape() const { return {}; }
  }];
  let genVerifyDecl = 1;
}

def Allo_VectorBuffer : Allo_BufferTypeBase<"VectorBuffer"> {
  let summary = "On-chip vector buffer element type";
  let description = [{
    Element type for on-chip buffers accessed at vector granularity. Each buffer
    slot holds a 1-D vector of elements. Syntax: `!allo.vector<64xbf16>`.
  }];
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape);
  let mnemonic = "vector";
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return 1; }
  }];
  let genVerifyDecl = 1;
}

def Allo_TileBuffer : Allo_BufferTypeBase<"TileBuffer"> {
  let summary = "On-chip tile buffer element type";
  let description = [{
    Element type for on-chip buffers accessed at tile (2-D) granularity. Each
    buffer slot holds a 2-D tile of elements. Syntax: `!allo.tile<4x4xf32>`.
  }];
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape);
  let mnemonic = "tile";
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
  }];
  let genVerifyDecl = 1;
}

// There is deliberately no `hbm` buffer *type*. Off-chip memory used to be one,
// because it was the only buffer allowed a multi-dimensional access — but that
// was a property of the addressing, not of the element. It is now spelled as an
// ordinary buffer with multi-dimensional `extents` and a scalar slot.

//===----------------------------------------------------------------------===//
// Descriptor and state types
//===----------------------------------------------------------------------===//

def Allo_Descriptor : Allo_Type<"Descriptor"> {
  let summary = "Hardware descriptor with named typed fields";
  let description = [{
    A hardware descriptor type with named fields, such as a TMA descriptor with
    "offset" and "stride" fields. Each field has a name and a primitive integer
    type. Syntax: `!allo.desc<["offset":i8, "stride":i8]>`.
  }];
  let parameters = (ins
    ArrayRefParameter<"StringAttr">:$fields,
    ArrayRefParameter<"Type">:$fieldTypes);
  let mnemonic = "desc";
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Allo_State : Allo_Type<"State"> {
  let summary = "Hardware state register with enumerated values";
  let description = [{
    The type of a hardware state register that holds one of a fixed set of
    enumerated values. The element type is the hardware primitive integer type
    used to store the state. Syntax: `!allo.state<i32>`.
  }];
  let parameters = (ins "Type":$elementType);
  let mnemonic = "state";
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Access pattern token type
//===----------------------------------------------------------------------===//

def Allo_Pattern : Allo_Type<"Pattern"> {
  let summary = "A token type representing an access pattern";
  let description = [{
    Token type that encodes a specific access pattern. Syntax: `!allo.pattern`.
  }];
  let mnemonic = "pattern";
}

#endif // ALLO_ISA_TYPES_TD

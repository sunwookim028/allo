/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TYPES
#define ALLO_TYPES

include "allo/IR/Allo.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def Fixed : Allo_Type<"Fixed", [MemRefElementTypeInterface]> {
  let summary = "fixed point";
  let mnemonic = "Fixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def UFixed : Allo_Type<"UFixed", [MemRefElementTypeInterface]> {
  let summary = "unsigned fixed point";
  let mnemonic = "UFixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let assemblyFormat = "`<` $width `,` $frac `>`";
}

def Stream : Allo_Type<"Stream"> {
  let summary = "stream type";
  let mnemonic = "stream";
  let parameters = (ins
    AnyType:$baseType,
    "std::size_t":$depth,
    ArrayRefParameter<"int64_t">:$shape);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// ACT buffer and affine-access pattern types share the Allo dialect.
include "allo/IR/AlloISATypes.td"

#endif // ALLO_TYPES

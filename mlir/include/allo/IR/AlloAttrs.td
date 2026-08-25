/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ATTRS
#define ALLO_ATTRS

include "allo/IR/Allo.td"
include "mlir/IR/EnumAttr.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
def CompletePartition: I32EnumAttrCase<"CompletePartition", 0, "Complete">;
def BlockPartition: I32EnumAttrCase<"BlockPartition", 1, "Block">;
def CyclicPartition: I32EnumAttrCase<"CyclicPartition", 2, "Cyclic">;
// A SKEWED partition: the bank is the sum of ALL subscripts modulo the factor,
// where block and cyclic each read one subscript. `dim` names the distribution
// dimension, the one divided down by the factor to make room. No HLS pragma
// spells this; it is the standard answer to an array read both by row and by
// column, where no per-axis layout can serve both.
def SkewPartition: I32EnumAttrCase<"SkewPartition", 3, "Skew">;

def PartitionKindEnum: I32EnumAttr<"PartitionKindEnum",
        "An example partition enum",
        [CompletePartition,BlockPartition,CyclicPartition,SkewPartition]> {
    let cppNamespace = "mlir::allo";
    let stringToSymbolFnName = "ConvertToPartitionEnum";
    let symbolToStringFnName = "ConvertToPartitionString";
}

def PartitionAxis : Allo_Attr<"PartitionAxis"> {
  let mnemonic = "part_axis";
  let parameters = (ins
    "allo::PartitionKindEnum":$kind,
    "int64_t":$factor,
    "int64_t":$dim);
  let assemblyFormat = [{
    `(` $dim `,` `` $kind `,` `` $factor`)`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
}

def PartitionAttr : Allo_Attr<"Partition"> {
  let mnemonic = "partition";
  let parameters = (ins ArrayRefParameter<"allo::PartitionAxisAttr">:$partitions);
  let assemblyFormat = [{
    `<` `[` $partitions `]` `>`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Dependence-hint enums (`#pragma HLS dependence`) for allo.assume.nodep
//===----------------------------------------------------------------------===//

def AssumeDepInter : I32EnumAttrCase<"Inter", 0, "inter">;
def AssumeDepIntra : I32EnumAttrCase<"Intra", 1, "intra">;
def AssumeDepTypeEnum : I32EnumAttr<"AssumeDepTypeEnum",
    "Dependence class of an assume.nodep hint: across vs. within iterations",
    [AssumeDepInter, AssumeDepIntra]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def AssumeDepTypeAttr : EnumAttr<Allo_Dialect, AssumeDepTypeEnum, "dep_type">;

def AssumeDirRAW : I32EnumAttrCase<"RAW", 0, "raw">;
def AssumeDirWAR : I32EnumAttrCase<"WAR", 1, "war">;
def AssumeDirWAW : I32EnumAttrCase<"WAW", 2, "waw">;
def AssumeDepDirEnum : I32EnumAttr<"AssumeDepDirEnum",
    "Dependence direction of an assume.nodep hint (read/write ordering)",
    [AssumeDirRAW, AssumeDirWAR, AssumeDirWAW]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def AssumeDepDirAttr : EnumAttr<Allo_Dialect, AssumeDepDirEnum, "dep_dir">;

//===----------------------------------------------------------------------===//
// Memory-model enums: the RTL backend's own storage vocabulary, decoupled from
// the Vitis `bind_storage` pragma strings. A bridge util maps those pragma
// strings (bram / ram_s2p / rom_1p / ...) onto these. Two axes are enums here,
// port topology and read-only-ness, because the compiler switches
// exhaustively on both. The third, WHICH STRUCTURE holds the array, is not an
// enum at all: it is a `dcp.storage` realization the device declares, since a
// part with different primitives has different names for them.
//===----------------------------------------------------------------------===//

// Port topology: how many concurrent read/write ports, and whether reads and
// writes contend. SinglePort = 1 shared R/W; SimpleDualPort = 1 read + 1 write
// (dedicated, no contention); TrueDualPort = 2 shared R/W.
def MemPortSingle     : I32EnumAttrCase<"SinglePort", 0, "1p">;
def MemPortSimpleDual : I32EnumAttrCase<"SimpleDualPort", 1, "s2p">;
def MemPortTrueDual   : I32EnumAttrCase<"TrueDualPort", 2, "t2p">;
def MemoryPortEnum : I32EnumAttr<"MemoryPortEnum",
    "Port topology of a memref (concurrent read/write ports)",
    [MemPortSingle, MemPortSimpleDual, MemPortTrueDual]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}

// RAM vs ROM: a ROM has no write ports (a store to it is a design error).
def MemKindRAM : I32EnumAttrCase<"RAM", 0, "ram">;
def MemKindROM : I32EnumAttrCase<"ROM", 1, "rom">;
def MemoryKindEnum : I32EnumAttr<"MemoryKindEnum",
    "Whether a memref is writable (RAM) or read-only (ROM)",
    [MemKindRAM, MemKindROM]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def MemoryKindAttr : EnumAttr<Allo_Dialect, MemoryKindEnum, "mem_kind">;

//===----------------------------------------------------------------------===//
// Determinacy: the scheduling-determinacy class of a dcp region or kernel. It
// is the discriminant that selects the controller regime and the composition
// wiring (time-triggered vs handshake vs broadcast). The reifier declares it in
// the IR so that no consumer has to re-derive it.
//   * counted_static: exact compile-time latency (static trip / length);
//   * conditional:    a while loop or data-dependent guard (flushing exit);
//   * indeterminate:  dynamic-trip, bounded or unknown-length (no exact span);
//   * concurrent:     a dataflow container (async spawns; self-timed).
//===----------------------------------------------------------------------===//

def DetCountedStatic : I32EnumAttrCase<"CountedStatic", 0, "counted_static">;
def DetConditional   : I32EnumAttrCase<"Conditional",   1, "conditional">;
def DetIndeterminate : I32EnumAttrCase<"Indeterminate", 2, "indeterminate">;
def DetConcurrent    : I32EnumAttrCase<"Concurrent",    3, "concurrent">;
def DeterminacyEnum : I32EnumAttr<"DeterminacyEnum",
    "Scheduling-determinacy class of a dcp region / kernel",
    [DetCountedStatic, DetConditional, DetIndeterminate, DetConcurrent]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def DeterminacyAttr : EnumAttr<Allo_Dialect, DeterminacyEnum, "determinacy">;

//===----------------------------------------------------------------------===//
// OpKind: the ABSTRACT operator vocabulary, independent of MLIR op names. What
// a device characterizes a row against (`dcp.comb`, and a `dcp.operator` whose
// `kind` names one of these), and what the scheduler classifies an operation
// into to find that row.
//
// Operators only. An array or stream ACCESS is not here: it is timed by its
// storage (`dcp.storage`, `dcp.stream_timing`) and builds no functional unit,
// so it has no row to name. `unknown` is an operation no abstract row covers
// (`math.sqrt`, an `affine.apply`), which keeps the classification total.
//===----------------------------------------------------------------------===//

def OpKindAdd         : I32EnumAttrCase<"Add",         0,  "add">;
def OpKindSub         : I32EnumAttrCase<"Sub",         1,  "sub">;
def OpKindMul         : I32EnumAttrCase<"Mul",         2,  "mul">;
def OpKindDiv         : I32EnumAttrCase<"Div",         3,  "div">;
def OpKindRem         : I32EnumAttrCase<"Rem",         4,  "rem">;
// maximumf / maxsi / maxui (NaN-propagating or integer maximum)
def OpKindMax         : I32EnumAttrCase<"Max",         5,  "max">;
// minimumf / minsi / minui
def OpKindMin         : I32EnumAttrCase<"Min",         6,  "min">;
// maxnumf (maxNum: returns the non-NaN operand)
def OpKindMaxNum      : I32EnumAttrCase<"MaxNum",      7,  "maxnum">;
def OpKindMinNum      : I32EnumAttrCase<"MinNum",      8,  "minnum">;
// ceildivsi / ceildivui
def OpKindCeilDiv     : I32EnumAttrCase<"CeilDiv",     9,  "ceildiv">;
// floordivsi
def OpKindFloorDiv    : I32EnumAttrCase<"FloorDiv",    10, "floordiv">;
def OpKindNeg         : I32EnumAttrCase<"Neg",         11, "neg">;
def OpKindCmp         : I32EnumAttrCase<"Cmp",         12, "cmp">;
def OpKindAnd         : I32EnumAttrCase<"And",         13, "and">;
def OpKindOr          : I32EnumAttrCase<"Or",          14, "or">;
def OpKindXor         : I32EnumAttrCase<"Xor",         15, "xor">;
def OpKindShl         : I32EnumAttrCase<"Shl",         16, "shl">;
def OpKindShr         : I32EnumAttrCase<"Shr",         17, "shr">;
def OpKindSelect      : I32EnumAttrCase<"Select",      18, "select">;
// The three cast kinds are separate because their hardware timing differs.
// Integer resize (sext / zext / trunc / index_cast)
def OpKindICastI      : I32EnumAttrCase<"ICastI",      19, "icast">;
// int <-> float conversion (si/ui-to-fp, fp-to-si/ui)
def OpKindFCastI      : I32EnumAttrCase<"FCastI",      20, "ifcast">;
// float resize (extf / truncf)
def OpKindFCastF      : I32EnumAttrCase<"FCastF",      21, "fcast">;
def OpKindUnknown     : I32EnumAttrCase<"Unknown",     22, "unknown">;

def OpKindEnum : I32EnumAttr<"OpKindEnum",
    "The abstract operator kind a device characterizes timing against",
    [OpKindAdd, OpKindSub, OpKindMul, OpKindDiv, OpKindRem, OpKindMax,
     OpKindMin, OpKindMaxNum, OpKindMinNum, OpKindCeilDiv, OpKindFloorDiv,
     OpKindNeg, OpKindCmp, OpKindAnd, OpKindOr, OpKindXor, OpKindShl,
     OpKindShr, OpKindSelect, OpKindICastI, OpKindFCastI, OpKindFCastF,
     OpKindUnknown]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def OpKindAttr : EnumAttr<Allo_Dialect, OpKindEnum, "op_kind">;

//===----------------------------------------------------------------------===//
// CombOpKind: the combinational op a `dcp.compute` realizes when it takes the
// comb (not IP) path.
//===----------------------------------------------------------------------===//

def CombAddi      : I32EnumAttrCase<"Addi",      0,  "addi">;
def CombSubi      : I32EnumAttrCase<"Subi",      1,  "subi">;
def CombMuli      : I32EnumAttrCase<"Muli",      2,  "muli">;
def CombDivsi     : I32EnumAttrCase<"Divsi",     3,  "divsi">;
def CombDivui     : I32EnumAttrCase<"Divui",     4,  "divui">;
def CombRemsi     : I32EnumAttrCase<"Remsi",     5,  "remsi">;
def CombRemui     : I32EnumAttrCase<"Remui",     6,  "remui">;
def CombAndi      : I32EnumAttrCase<"Andi",      7,  "andi">;
def CombOri       : I32EnumAttrCase<"Ori",       8,  "ori">;
def CombXori      : I32EnumAttrCase<"Xori",      9,  "xori">;
def CombShli      : I32EnumAttrCase<"Shli",      10, "shli">;
def CombShrsi     : I32EnumAttrCase<"Shrsi",     11, "shrsi">;
def CombShrui     : I32EnumAttrCase<"Shrui",     12, "shrui">;
def CombCmpi      : I32EnumAttrCase<"Cmpi",      13, "cmpi">;
def CombSelect    : I32EnumAttrCase<"Select",    14, "select">;
def CombExtsi     : I32EnumAttrCase<"Extsi",     15, "extsi">;
def CombExtui     : I32EnumAttrCase<"Extui",     16, "extui">;
def CombTrunci    : I32EnumAttrCase<"Trunci",    17, "trunci">;
def CombIndexCast : I32EnumAttrCase<"IndexCast", 18, "index_cast">;
def CombApply     : I32EnumAttrCase<"Apply",     19, "apply">;
def CombNegf      : I32EnumAttrCase<"Negf",      20, "negf">;
def CombMinsi     : I32EnumAttrCase<"Minsi",     21, "minsi">;
def CombMaxsi     : I32EnumAttrCase<"Maxsi",     22, "maxsi">;
def CombMinui     : I32EnumAttrCase<"Minui",     23, "minui">;
def CombMaxui     : I32EnumAttrCase<"Maxui",     24, "maxui">;
def CombIndexCastUi : I32EnumAttrCase<"IndexCastUi", 25, "index_castui">;

def CombOpKindEnum : I32EnumAttr<"CombOpKindEnum",
    "The combinational op a comb-path dcp.compute realizes",
    [CombAddi, CombSubi, CombMuli, CombDivsi, CombDivui, CombRemsi, CombRemui,
     CombAndi, CombOri, CombXori, CombShli, CombShrsi, CombShrui, CombCmpi,
     CombSelect, CombExtsi, CombExtui, CombTrunci, CombIndexCast, CombApply,
     CombNegf, CombMinsi, CombMaxsi, CombMinui, CombMaxui, CombIndexCastUi]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def CombOpKindAttr : EnumAttr<Allo_Dialect, CombOpKindEnum, "comb_kind">;

//===----------------------------------------------------------------------===//
// StallContract: how a realized IP operator participates in the datapath's
// stall protocol
//===----------------------------------------------------------------------===//

// (data.., clk, ce) -> data   the FPGA hard-IP default (ce freezes the pipe)
def StallCe   : I32EnumAttrCase<"Ce",   0, "ce">;
// (data.., clk) -> data        no enable; the pipeline always advances
def StallFree : I32EnumAttrCase<"Free", 1, "free">;
// (s_*, m_*, clk) -> ...        a valid/ready handshake (variable latency)
def StallElastic : I32EnumAttrCase<"Elastic", 2, "elastic">;
def StallContractEnum : I32EnumAttr<"StallContractEnum",
    "How a realized IP operator participates in the stall protocol",
    [StallCe, StallFree, StallElastic]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def StallContractAttr : EnumAttr<Allo_Dialect, StallContractEnum, "stall">;

//===----------------------------------------------------------------------===//
// Resource cost: what one realization spends of one device resource.
//
// A resource (`dcp.resource`) is a pure counter the compiler only adds and
// multiplies, so its NAME is the device's vocabulary and stays open. A cost is
// how much of one a realization spends, as a function of the parameters that
// realization's kind carries (an operator's operand width, a multiplexer's
// fan-in and width, a delay chain's depth and width).
//
// Each form is a function of one parameter, a multi-parameter term is the
// PRODUCT of one form per parameter, and a cost is the SUM of the terms naming
// one resource (see `ResourceUse`). That keeps the evaluator a one-parameter
// switch and still covers every measured shape: `0.4*k` per bit is linear in k
// times linear in width, and a sum of two such terms is a sum.
//===----------------------------------------------------------------------===//

// c
def CostConst     : I32EnumAttrCase<"Const",     0, "const">;
// coeffs[0] + coeffs[1] * p
def CostLinear    : I32EnumAttrCase<"Linear",    1, "linear">;
// coeffs[0] * p * p
def CostQuadratic : I32EnumAttrCase<"Quadratic", 2, "quadratic">;
// p < coeffs[0] ? coeffs[1] * p : coeffs[2], the shape of a shift-register
// cliff: a chain deeper than three stops being flip-flops and stops growing.
def CostStep      : I32EnumAttrCase<"Step",      3, "step">;
// Measured point by point: coeffs is [p0, v0, p1, v1, ...] ascending in p, and
// a parameter between two points takes the lower one's value. This staircase
// fits a piecewise constant quantity, such as the ceil(depth/32) sites an SRL
// chain occupies. A continuous quantity sampled into a table is under-stated
// between its points.
def CostTable     : I32EnumAttrCase<"Table",     4, "table">;
def CostTiled     : I32EnumAttrCase<"Tiled",     5, "tiled">;
// The same [p0, v0, p1, v1, ...] points as `table`, read continuously: a
// parameter between two points takes their linear interpolation. This fits a
// continuous quantity measured at a few points, such as an operator's delay.
def CostInterp    : I32EnumAttrCase<"Interp",    6, "interp">;
// coeffs[0] < p ? arms[1](p) : arms[0](p), with both arms costs of any form.
def CostPiecewise : I32EnumAttrCase<"Piecewise", 7, "piecewise">;
def CostFormEnum : I32EnumAttr<"CostFormEnum",
    "Shape of a resource cost as a function of one parameter",
    [CostConst, CostLinear, CostQuadratic, CostStep, CostTable, CostTiled,
     CostInterp, CostPiecewise]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def CostFormAttr : EnumAttr<Allo_Dialect, CostFormEnum, "cost_form">;

def CostAttr : Allo_Attr<"Cost"> {
  let mnemonic = "cost";
  let summary = "One resource cost as a function of one parameter";
  let description = [{
    The SHAPE comes from hardware structure and only the coefficients are
    measured: an N-bit AND is N LUT6s, an adder adds a carry chain, a divider
    is quadratic. A shape that is not structural belongs in a `table`, which
    is what the measurement produced in the first place.

    A shape holds at every parameter. A `table` and an `interp` answer only
    between their first and last measured point, and a caller reports the gap
    rather than extrapolating outside it.
  }];
  let parameters = (ins
    "allo::CostFormEnum":$form,
    "::mlir::DenseF64ArrayAttr":$coeffs,
    // The two arms of a `piecewise`, and empty for every other form.
    OptionalArrayRefParameter<"allo::CostAttr">:$arms);
  let assemblyFormat = [{
    `<` $form `,` $coeffs (`,` `[` $arms^ `]`)? `>`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// This cost at \p param, unrounded so a product of factors rounds once at
    /// the end rather than once per factor. A `tiled` factor reads \p param
    /// alone; its whole-tuple reading is `evaluateResourceUse`'s.
    ///
    /// Nullopt where \p param falls outside a `table` or an `interp`'s measured
    /// points; `unmeasuredAt` names the cost.
    std::optional<double> evaluate(int64_t param) const;

    /// The `table` or `interp` this cost reads at \p param whose measured
    /// points do not cover it, null wherever `evaluate` answers. A `piecewise`
    /// answers for the arm it selects.
    CostAttr unmeasuredAt(int64_t param) const;

    /// The first and last point a `table` or an `interp` was measured at.
    std::pair<int64_t, int64_t> measuredDomain() const;
  }];
}

def ResourceUseAttr : Allo_Attr<"ResourceUse"> {
  let mnemonic = "res_use";
  let summary = "One term of what a realization spends of one resource";
  let description = [{
    `factors` holds one cost per parameter of the realization's kind, and the
    term's value is their PRODUCT: factor `i` is evaluated at parameter `i`.
    A one-parameter kind carries one factor.

    A resource may be named by SEVERAL entries of one `uses`, and what it spends
    is their SUM, so a cost is a sum of products: a polynomial in the
    realization's parameters, in a basis the forms below span. One product is
    not enough because measured shapes are not all separable: a delay chain's
    `2*width + depth - 1` flip-flops are a per-bit term plus a per-stage term,
    and no product of one factor per parameter is a sum. Summing costs nothing
    else, since the forms, the arity rule and the evaluator are unchanged.

    A single `tiled` factor at arity greater than one is the exception to the
    arity rule: it consumes the whole tuple at once, `ceil((product+c1)/c0)`. A
    `tiled` among a full set of factors reads its own parameter like any other,
    `ceil((p_i+c1)/c0)`. The offset `c1` is optional and defaults to zero: it
    is how many of the parameter's items the tiles do not hold.
  }];
  let parameters = (ins
    "mlir::SymbolRefAttr":$resource,
    ArrayRefParameter<"allo::CostAttr">:$factors);
  let assemblyFormat = [{
    `<` $resource `,` `[` $factors `]` `>`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
}

#endif // ALLO_ATTRS

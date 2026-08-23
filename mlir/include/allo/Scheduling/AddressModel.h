/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_ADDRESS_MODEL_H
#define ALLO_SCHEDULING_ADDRESS_MODEL_H

#include "allo/Scheduling/OperatorLibrary.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::allo {

struct BankLayout;

/// The address expressions an access's hardware computes, and the width they
/// carry at. Not the flat row-major index: a banked access addresses one bank
/// at the in-bank offset, usually cheaper (`A[2*i]` under cyclic-2 has offset
/// `i`, no hardware, against flat address `2*i`, a shift-add). An access whose
/// bank varies at runtime builds a second cone for the digit, a real divider
/// whenever the factor is not a power of two.
struct AddressExprs {
  AffineExpr offset;  // the element index inside the bank this access reaches
  AffineExpr bank;    // which bank, or null when it is decided at compile time
  unsigned width = 0; // bits `offset` is carried at (one bank's, not the whole)
};

/// \p map's address expressions over a memref of \p shape banked as \p layout,
/// given the bank `assign-banks` assigned the access (nullopt when it roams).
///
/// Uniform over banked and unbanked: an unpartitioned memref is a one-bank
/// layout whose `offset` is the flat row-major index and whose `bank` is the
/// constant 0 nothing builds.
AddressExprs addressExprsOf(const BankLayout &layout, AffineMap map,
                            llvm::ArrayRef<int64_t> shape,
                            std::optional<unsigned> assignedBank);

/// What an address expression costs as hardware: the critical path through it,
/// plus the operators it instantiates as an area proxy.
struct AddressCost {
  double delay = 0.0;  // ns, longest path from an operand to the address
  unsigned adders = 0; // carry chains, including a coefficient's shift-adds
  unsigned multipliers = 0; // generic multipliers (a non-constant coefficient)
  unsigned dividers = 0;    // dividers / remainder units
  /// Constant-divisor division sites, realized as a reciprocal multiply whose
  /// shift-adds are already counted in `adders`.
  unsigned reciprocals = 0;
  /// Register-carried pieces (`SplitAddress` terms and reads). Zero means the
  /// cone is entirely residual and no register can shorten it.
  unsigned carried = 0;

  /// Nothing is instantiated: the address is wiring off an existing value.
  bool trivial() const {
    return adders == 0 && multipliers == 0 && dividers == 0;
  }
};

/// The rounded-up reciprocal for a w-bit non-negative dividend:
/// `floor(n / d) == (n * magic) >> shift` for every `n` below `2^w`, with `d`
/// not a power of two and below `2^w` itself. The multiplier fits `w+1` bits,
/// so the product takes `2w+1`. Shared by the pricing here, the address
/// emitter and `legalize-arith`.
uint64_t magicMultiplier(uint64_t d, unsigned w, unsigned &shift);

/// \p e simplified, unless simplifying made it worse to build.
///
/// `simplifyAffineExpr` is a canonicalizer, not a cost function: it flattens
/// `x mod k` into `x - (x floordiv k) * k`, three operators where the residue
/// was a mask, though it also does real work no rewrite replaces. Several
/// candidates are built and the cheapest kept, ranked on device-independent
/// weights so every layer picks the same form.
AffineExpr simplifiedForHardware(AffineExpr e, unsigned numDims,
                                 unsigned numSymbols);

/// The device delays an address cone is priced against, read from the operator
/// library's combinational rows.
struct AddressDelays {
  double add = 0.0; // one carry-chain adder / subtractor
  double mul = 0.0; // a generic multiplier
  double div = 0.0; // a divider / remainder unit

  /// The width those numbers are characterized at, which is the one width an
  /// `index` is built at everywhere (`datapathWidth`, `combParamWidth`). A
  /// narrower cone scales linearly off this: an FPGA carry chain's delay tracks
  /// width, while a shift or mask costs no logic at any width.
  static constexpr unsigned refWidth = kIndexWidth;
};

/// Read the comb rows an address cone can be built from.
AddressDelays addressDelaysOf(const OperatorLibrary &lib);

/// The one hardware form of a standalone apply's single-result map:
/// `simplifiedForHardware` over its result. What the scheduler prices, the
/// pre-schedule gates walk, and `emitCompute` builds; kept in one place so the
/// layers cannot drift onto different forms.
AffineExpr applyExprOf(AffineMap map);

/// The cost of \p e when its arithmetic is carried at \p width bits.
///
/// Prices what synthesis actually builds, not the ops emitted:
/// * A constant coefficient is a signed-digit shift-add network, not a
///   multiplier (`x * 15` is `(x << 4) - x`, one adder).
/// * `floordiv`/`mod` do not commute with truncation mod `2^width`, so a
///   divider and everything feeding it stay at `refWidth` regardless of
///   `width`; only the divider's result may be truncated. `+`, `-`, `*` do
///   commute and may be carried narrow, as may the subtree under a
///   power-of-two `mod`, which is a mask.
AddressCost addressCost(AffineExpr e, const AddressDelays &delays,
                        unsigned width);

/// The cost of \p map composed with \p shape's row-major strides, i.e. the flat
/// element index (not what a banked access builds; see `addressExprsOf`). A
/// null
/// \p map prices as zero, the stream / non-access case.
AddressCost addressCost(AffineMap map, llvm::ArrayRef<int64_t> shape,
                        const AddressDelays &delays, unsigned width);

/// Whether a register can follow an operand, and its per-iteration step when
/// one can. A digit of a counter is maintained by wrapping a register once per
/// iteration, so a step that could carry it past two multiples of the divisor
/// is not maintainable, and the pricing and the build must refuse the same
/// ones.
using CarriedFn = llvm::function_ref<std::optional<int64_t>(unsigned)>;

/// An address as `base + sum(coeff * digit-of-operand) + residual`, where
/// `operand` indexes the access map's operands (dims, then symbols).
///
/// A term is what a register can carry: either a scaled counter (constant
/// per-iteration difference, advanced rather than rebuilt) or a digit of one,
/// `(x floordiv D) mod K`, which advances by a comparator/wrap rather than a
/// constant but is just as cheap a register. The residual is everything else,
/// in the operands' own numbering, null when nothing is left.
///
/// The split is partial by design: `A[i,j]` with `i` a counter and `j`
/// data-dependent has a row stride a register can follow and a column it
/// cannot, so reducing both together would rebuild `i*N` every cycle just to
/// add `j`. A `floordiv`/`mod` over anything but a counter lands in the
/// residual for the same reason.
struct SplitAddress {
  /// One term a register can carry: `coeff * digit(scale * operand + offset)`,
  /// where `digit(x)` is `(x floordiv divisor) mod modulus`.
  ///
  /// `divisor == 1` and no modulus is the plain scaled counter, advancing by a
  /// constant. A digit is periodic: it advances by nothing most iterations and
  /// wraps when its argument crosses a multiple of `divisor`, which a register
  /// (`RegionBlock::AddrStride`) maintains as cheaply as a `floordiv`/`mod` on
  /// the address path costs every cycle.
  struct Term {
    unsigned operand;
    int64_t coeff;
    int64_t scale = 1;   // the counter's own coefficient, inside the digit
    int64_t offset = 0;  // the counter's own constant, inside the digit
    int64_t divisor = 1; // 1: no `floordiv`
    int64_t modulus = 0; // 0: no `mod`
    bool isDigit() const { return divisor != 1 || modulus != 0; }
  };
  llvm::SmallVector<Term, 4> terms;
  int64_t base = 0;
  AffineExpr residual;
  /// Digits the residual reads rather than the address sums: an operator cheap
  /// on a register but expensive on a counter belongs on top of one.
  /// `(x mod 5) floordiv 2` is the shape: the residue is a register and the
  /// `floordiv 2` over it a shift, where evaluated together the pair is two
  /// real dividers.
  ///
  /// Named as symbols numbered from the map's own `numSymbols`, so no existing
  /// leaf is renumbered and the emitter appends their values to the operand
  /// list it already passes.
  llvm::SmallVector<Term, 2> reads;
};

/// Split \p e, an address expression over \p numDims dims then symbols, with
/// \p carried naming the operands a register can follow.
///
/// Both the scheduler and the emitter pass `addressExprsOf(...).offset`, so a
/// banked access is split on the expression its bank is actually addressed
/// through. A subtree holding nothing carried is residual whole, never
/// redistributed.
SplitAddress splitAddress(AffineExpr e, unsigned numDims, unsigned numSymbols,
                          CarriedFn carried);

/// What \p addr costs once every term arrives from a register that advances
/// with its operand: only the network summing the terms with the residual is
/// left.
///
/// Priced in the order `buildAddr` writes it, one input per term and the
/// residual last, so the count is the emitter's actual chain, not an optimal
/// adder tree; the residual's own cone runs beside the registers' adders, not
/// under them. The base costs nothing, absorbed into the first register's start
/// value.
AddressCost splitAddressCost(const SplitAddress &addr,
                             const AddressDelays &delays, unsigned width);

/// The width an address over \p shape is carried at: enough bits to index it,
/// what `DatapathEmitter::addrWidth` narrows to. Stated once here so the
/// pricing and the emitted datapath, decided in different passes, agree.
/// `addressExprsOf` applies it to the per-bank shape, one bank's address port
/// width.
unsigned addressWidthOf(llvm::ArrayRef<int64_t> shape);

/// The cost of \p op's address as the emitter will build it. Zero for a stream
/// or non-access; every array access is priced, subscript affine or not.
///
/// Both cones are charged: the in-bank offset and, when the access roams, the
/// bank digit. Strength reduction is decided once here for both the scheduler
/// and the emitter: a term following an enclosing counter with constant bounds
/// is carried in a register that advances with it
/// (`DatapathBuilder::planAddressGenerators`), so only the summing network and
/// whatever did not reduce are charged. The emitter also knows whether the
/// counter's bounds resolved to constants and whether the term is wanted in the
/// same cycle, so it may send more terms to the residual: this pricing is
/// optimistic on that gap, not pessimistic.
///
/// A banked access is priced on its `AddressExprs`, not the flat index: a
/// runtime bank digit is a second cone off the same operands, so delay is the
/// max of the two cones while operator counts add.
AddressCost addressCostOf(Operation *op, const OperatorLibrary &lib);

/// `addressCostOf`'s delay, quantized to a hundredth of a nanosecond: the
/// caller names an operator type after this number, so two sites whose names
/// agree must carry the same delay or whichever registers last silently
/// redefines the other.
double addressDelayOf(Operation *op, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_ADDRESS_MODEL_H

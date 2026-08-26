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

/// The address expressions an access's hardware computes, and their width. Not
/// the flat row-major index: a banked access addresses one bank at the in-bank
/// offset, usually cheaper. A runtime-varying bank builds a second cone for the
/// digit, a real divider when the factor is not a power of two.
struct AddressExprs {
  AffineExpr offset;  // the element index inside the bank this access reaches
  AffineExpr bank;    // which bank, or null when it is decided at compile time
  unsigned width = 0; // bits `offset` is carried at (one bank's, not the whole)
};

/// \p map's address expressions over a memref of \p shape banked as \p layout,
/// given the bank assigned to the access (nullopt when it roams). An
/// unpartitioned memref is a one-bank layout: flat row-major offset, constant
/// bank 0.
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
  /// Constant-divisor division realized as a reciprocal multiply; its
  /// shift-adds are counted in `adders`.
  unsigned reciprocals = 0;
  /// Register-carried pieces. Zero means the cone is entirely residual.
  unsigned carried = 0;

  /// Nothing is instantiated: the address is wiring off an existing value.
  bool trivial() const {
    return adders == 0 && multipliers == 0 && dividers == 0;
  }
};

/// Rounded-up reciprocal for a w-bit non-negative dividend:
/// `floor(n/d) == (n*magic) >> shift` for every `n < 2^w`, with `d` not a power
/// of two and below `2^w`. The multiplier fits `w+1` bits, the product `2w+1`.
/// Shared by the pricing here, the address emitter, and `legalize-arith`.
uint64_t magicMultiplier(uint64_t d, unsigned w, unsigned &shift);

/// \p e simplified, unless simplifying made it worse to build.
///
/// `simplifyAffineExpr` canonicalizes, not costs: it flattens `x mod k` into
/// three operators where the residue was a mask. Several candidates are built
/// and the cheapest kept on device-independent weights, so every layer picks
/// the same form.
AffineExpr simplifiedForHardware(AffineExpr e, unsigned numDims,
                                 unsigned numSymbols);

/// The device delays an address cone is priced against, read from the operator
/// library's combinational rows.
struct AddressDelays {
  double add = 0.0; // one carry-chain adder / subtractor
  double mul = 0.0; // a generic multiplier
  double div = 0.0; // a divider / remainder unit

  /// The width those numbers are characterized at, the one width an `index` is
  /// built at everywhere. A narrower cone scales linearly off this: a carry
  /// chain's delay tracks width, a shift or mask costs nothing at any width.
  static constexpr unsigned refWidth = kIndexWidth;
};

/// Read the comb rows an address cone can be built from.
AddressDelays addressDelaysOf(const OperatorLibrary &lib);

/// The one hardware form of a standalone apply's single-result map:
/// `simplifiedForHardware` over its result. Kept in one place so the scheduler,
/// the pre-schedule gates, and `emitCompute` cannot drift onto different forms.
AffineExpr applyExprOf(AffineMap map);

/// The cost of \p e carried at \p width bits. Prices what synthesis builds, not
/// the ops emitted:
/// * A constant coefficient is a signed-digit shift-add network, not a
///   multiplier (`x * 15` is `(x << 4) - x`, one adder).
/// * `floordiv`/`mod` do not commute with truncation mod `2^width`, so a
///   divider and everything feeding it stay at `refWidth`; only its result may
///   be truncated. `+`, `-`, `*` and a subtree under a power-of-two `mod` (a
///   mask) may be carried narrow.
AddressCost addressCost(AffineExpr e, const AddressDelays &delays,
                        unsigned width);

/// The cost of \p map composed with \p shape's row-major strides, the flat
/// element index (not what a banked access builds; see `addressExprsOf`). A
/// null \p map prices as zero, the stream / non-access case.
AddressCost addressCost(AffineMap map, llvm::ArrayRef<int64_t> shape,
                        const AddressDelays &delays, unsigned width);

/// Whether a register can follow an operand, and its per-iteration step. A
/// counter digit is maintained by wrapping a register once per iteration, so a
/// step that could carry it past two multiples of the divisor is not
/// maintainable; pricing and build must refuse the same ones.
using CarriedFn = llvm::function_ref<std::optional<int64_t>(unsigned)>;

/// An address as `base + sum(coeff * digit-of-operand) + residual`, operands
/// numbered as the map's dims then symbols. A term is what a register can
/// carry: a scaled counter or a digit of one, `(x floordiv D) mod K`. The
/// residual is everything else, null when nothing is left.
///
/// The split is partial by design: reducing a counter row and a data-dependent
/// column together would rebuild `i*N` every cycle just to add the column, so a
/// `floordiv`/`mod` over anything but a counter also lands in the residual.
struct SplitAddress {
  /// One term a register can carry: `coeff * digit(scale * operand + offset)`,
  /// `digit(x) = (x floordiv divisor) mod modulus`. `divisor == 1` with no
  /// modulus is a plain scaled counter. A digit is periodic: it wraps only when
  /// its argument crosses a multiple of `divisor`, maintained by a register as
  /// cheaply as the `floordiv`/`mod` would cost every cycle.
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
  /// `(x mod 5) floordiv 2` evaluated together is two real dividers. Numbered
  /// as symbols from the map's own `numSymbols`, so no leaf is renumbered and
  /// the emitter appends their values to the operand list.
  llvm::SmallVector<Term, 2> reads;
};

/// Split \p e over \p numDims dims then symbols, with \p carried naming the
/// operands a register can follow. Both scheduler and emitter pass
/// `addressExprsOf(...).offset`, so a banked access is split on the expression
/// its bank is addressed through. A subtree holding nothing carried stays
/// residual whole.
SplitAddress splitAddress(AffineExpr e, unsigned numDims, unsigned numSymbols,
                          CarriedFn carried);

/// What \p addr costs once every term arrives from a register advancing with
/// its operand: only the network summing the terms with the residual is left.
/// Priced in `buildAddr`'s write order, one input per term and residual last,
/// so the count matches the emitter's chain rather than an optimal adder tree.
/// The base costs nothing, absorbed into the first register's start value.
AddressCost splitAddressCost(const SplitAddress &addr,
                             const AddressDelays &delays, unsigned width);

/// The width an address over \p shape is carried at, matching
/// `DatapathEmitter::addrWidth`. Stated once so pricing and the emitted
/// datapath agree. `addressExprsOf` applies it to the per-bank shape.
unsigned addressWidthOf(llvm::ArrayRef<int64_t> shape);

/// The cost of \p op's address as the emitter builds it. Zero for a stream or
/// non-access; every array access is priced, subscript affine or not.
///
/// Both cones are charged: the in-bank offset and, when the access roams, the
/// bank digit. Strength reduction is decided once here for scheduler and
/// emitter: a term following an enclosing counter with constant bounds is
/// carried in a register that advances with it, so only the summing network and
/// unreduced terms are charged. The emitter may send more terms to the
/// residual, so this pricing is optimistic on that gap. A banked access is
/// priced on its `AddressExprs`: delay is the max of the two cones, operator
/// counts add.
AddressCost addressCostOf(Operation *op, const OperatorLibrary &lib);

/// `addressCostOf`'s delay quantized to a hundredth of a nanosecond: the caller
/// names an operator type after this number, so two sites whose names agree
/// must carry the same delay or one silently redefines the other.
double addressDelayOf(Operation *op, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_ADDRESS_MODEL_H

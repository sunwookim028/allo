# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The matrix engine: systolic chain or adder tree behind one interface (Q2 / H11).

The interface is the unit's declaration: it reads ``me_lhs`` (one packed row
of ``DIM`` operands), addresses the weight tile ``W`` (``W[r*DIM + c]`` at PE
(r, c)), and writes ``me_out`` (one packed row of ``DIM`` results), with the
MAC engine's names bound as parameters. What the two engines share is that
declaration; what they do NOT share is the function they compute:

* ``systolic_engine`` accumulates sequentially in ascending ``r`` in the
  engine's ``ACC`` type, rounding once per term, then packs once --
  MiniTPU's array (``ARITHMETIC.md`` §8, the Phase 0 contract reference
  ``ref_mxu.mxu_row``).
* ``tree_engine`` sums the ``DIM`` products as a balanced binary tree of
  ``ACC`` adds (left child first), rounding once per level, then packs once
  -- a DotTree-shaped MXU.

At bf16 these differ in the last bits of their output on ordinary data and
agree only where no rounding happens, so the ``order`` is part of the
engine's declared interface (``MatrixEngine.order``) and the contract
reference below takes it as an argument. A swap that changes the order is a
different function, checked against the reference evaluated with that
order; a swap that keeps it (another systolic implementation) is checked
against the same reference. ``latency_model`` is a MODEL beside the
declaration -- the number the composition would book -- and never a
constant the body consumes (D-10): the systolic one is held to Phase 0's
measurement in ``legality.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import allo.dataflow as df
from allo.compose import Architecture, Channel, Instance, Memory, Unit, unit

from allo.compose import Engine
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32


def _pow2(p):
    dim = p["DIM"]
    assert dim >= 2 and dim & (dim - 1) == 0, (
        f"DIM={dim}: the tree engine is a BALANCED binary tree over the "
        f"contraction, which needs a power of two; the systolic engine has "
        f"no such condition, so the two engines are not interchangeable at "
        f"this DIM")


@unit(
    memories=("W",),
    reads=("me_lhs",),
    writes=("me_out",),
    parameters=("N_ROWS", "DIM"),
    engines=("MAC_IN", "MAC_IN_BITS", "MAC_ACC", "MAC_OUT_BITS", "MAC_MUL",
             "MAC_ADD", "MAC_PACK"),
    order="sequential",
)
def systolic_engine(w_tile: UInt(32)[DIM * DIM]):
    for work in range(N_ROWS):
        packed: UInt(DIM * MAC_IN_BITS) = me_lhs.get()
        out_word: UInt(DIM * MAC_OUT_BITS) = 0
        psum: MAC_ACC[DIM]
        with allo.meta_for(DIM) as c:
            psum[c] = 0  # row 0's psum_i is '0 (the adder's zero bypass)
            with allo.meta_for(DIM) as r:
                a: MAC_IN = packed[MAC_IN_BITS * r : MAC_IN_BITS * (r + 1)]
                w_word: UInt(32) = w_tile[r * DIM + c]
                w: MAC_IN = w_word
                product: MAC_ACC = MAC_MUL(a, w)
                psum[c] = MAC_ADD(product, psum[c])
            out_word[MAC_OUT_BITS * c : MAC_OUT_BITS * (c + 1)] = MAC_PACK(psum[c])
        me_out.put(out_word)


@unit(
    memories=("W",),
    reads=("me_lhs",),
    writes=("me_out",),
    parameters=("N_ROWS", "DIM"),
    engines=("MAC_IN", "MAC_IN_BITS", "MAC_ACC", "MAC_OUT_BITS", "MAC_MUL",
             "MAC_ADD", "MAC_PACK"),
    order="tree",
    legality=_pow2,
)
def tree_engine(w_tile: UInt(32)[DIM * DIM]):
    for work in range(N_ROWS):
        packed: UInt(DIM * MAC_IN_BITS) = me_lhs.get()
        out_word: UInt(DIM * MAC_OUT_BITS) = 0
        node: MAC_ACC[2 * DIM - 1]
        with allo.meta_for(DIM) as c:
            with allo.meta_for(DIM) as r:
                a: MAC_IN = packed[MAC_IN_BITS * r : MAC_IN_BITS * (r + 1)]
                w_word: UInt(32) = w_tile[r * DIM + c]
                w: MAC_IN = w_word
                node[r] = MAC_MUL(a, w)
            # node[DIM + n]'s children are already written when n is reached:
            # one ascending loop is the whole balanced tree (reduction_tree.py).
            with allo.meta_for(DIM - 1) as n:
                node[DIM + n] = MAC_ADD(node[2 * n], node[2 * n + 1])
            out_word[MAC_OUT_BITS * c : MAC_OUT_BITS * (c + 1)] = MAC_PACK(node[2 * DIM - 2])
        me_out.put(out_word)


@dataclass(frozen=True)
class MatrixEngine:
    """What a matrix engine declares beyond its MAC engine: its accumulate
    order (part of the function: the unit's ``order=``, which the composite
    holds to its own, README D-15), and a latency model (a booking, D-10)."""

    name: str
    unit: Unit
    latency_model: callable   # (DIM, mac: Engine) -> edges, push -> result

    @property
    def order(self) -> str:
        return self.unit.order


SYSTOLIC = MatrixEngine(
    "systolic", systolic_engine,
    # Phase 0: push -> output_valid = 2 + DIM*(PE + 1), PE = 1 + add_latency
    latency_model=lambda dim, mac: 2 + dim * (1 + mac.add_latency + 1))
TREE = MatrixEngine(
    "tree", tree_engine,
    # products in one stage, log2(DIM) adder levels, one pack stage
    latency_model=lambda dim, mac: 1 + max(1, mac.mul_latency)
    + mac.add_latency * int(math.log2(dim)) + 1)
MATRIX_ENGINES = {m.name: m for m in (SYSTOLIC, TREE)}


# --- the rig ----------------------------------------------------------------

@unit(
    memories=("A",),
    writes=("me_lhs",),
    parameters=("N_ROWS", "DIM"),
    engines=("MAC_IN", "MAC_IN_BITS"),
)
def me_feed(a_mem: UInt(32)[N_ROWS * DIM]):
    for work in range(N_ROWS):
        packed: UInt(DIM * MAC_IN_BITS) = 0
        with allo.meta_for(DIM) as r:
            a_word: UInt(32) = a_mem[work * DIM + r]
            a: MAC_IN = a_word
            packed[MAC_IN_BITS * r : MAC_IN_BITS * (r + 1)] = a
        me_lhs.put(packed)


@unit(
    memories=("OUT",),
    reads=("me_out",),
    parameters=("N_ROWS", "DIM"),
    engines=("MAC_OUT_BITS",),
)
def me_sink(out_mem: UInt(32)[N_ROWS * DIM]):
    for work in range(N_ROWS):
        out_word: UInt(DIM * MAC_OUT_BITS) = me_out.get()
        with allo.meta_for(DIM) as c:
            # The lane stays an unsigned pattern: `lane: int32 = <UInt(32)
            # slice>` lowers to `arith.trunci i32 -> i32` and fails to build
            # (a same-width signedness conversion is emitted as a truncation).
            word: UInt(32) = out_word[MAC_OUT_BITS * c : MAC_OUT_BITS * (c + 1)]
            out_mem[work * DIM + c] = word


def mxu_rig(matrix: MatrixEngine, mac: Engine, dim: int, n_rows: int,
            name=None, order="sequential", accepts=("tree",)) -> Architecture:
    """The composite's contract reference takes ``order`` (MiniTPU's:
    sequential); this rig also ``accepts`` the tree, as a DIFFERENT function
    verified against the reference with that order (``reference_order``).
    ``accepts=()`` is MiniTPU's instance, which refuses the tree."""
    params = {"N_ROWS": n_rows, "DIM": dim, "QD": 4}
    return Architecture(
        name=name or f"mxu_{matrix.name}_{mac.name}_d{dim}", parameters=params,
        engines={"MAC": mac}, order=order, accepts=accepts,
        memories=(Memory("A", "UInt(32)[N_ROWS * DIM]"),
                  Memory("W", "UInt(32)[DIM * DIM]"),
                  Memory("OUT", "UInt(32)[N_ROWS * DIM]")),
        channels=(Channel("me_lhs", depth="QD", lanes="DIM", lane_bits="MAC_IN_BITS",
                          carries="one row of DIM operands"),
                  Channel("me_out", depth="QD", lanes="DIM", lane_bits="MAC_OUT_BITS",
                          carries="one row of DIM results")),
        units=(me_feed, Instance(matrix.unit, "matrix_engine"), me_sink))


# --- the contract reference, with the order as an argument -----------------

def matrix_rows(A, W, mac: Engine, order: str):
    """Every row of ``A`` (``[n, DIM]``) against ``W`` (``[DIM, DIM]``) in the
    MAC engine's own arithmetic, accumulated in ``order``, packed once.
    ``order="sequential"`` at the bf16 engine is ``ref_mxu.mxu_row``. Now
    ``compose.Engine.dot``; kept as the name the gate imports."""
    if order not in ("sequential", "tree"):
        raise ValueError(order)
    return mac.dot(A, W, order)


def _matrix_rows_prototype(A, W, mac: Engine, order: str):
    """The prototype's loop, kept to hold ``Engine.dot`` to it."""
    A = np.asarray(A, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    n, D = A.shape
    out = np.zeros((n, D), dtype=np.int64)
    for i in range(n):
        prods = [mac.ref_mul(np.full(D, A[i, r]), W[r]) for r in range(D)]
        if order == "sequential":
            psum = np.zeros(D, dtype=np.int64)
            for r in range(D):
                psum = mac.ref_add(prods[r], psum)
            out[i] = mac.ref_pack(psum)
        elif order == "tree":
            level = list(prods)
            while len(level) > 1:
                level = [mac.ref_add(level[2 * k], level[2 * k + 1])
                         for k in range(len(level) // 2)]
            out[i] = mac.ref_pack(level[0])
        else:
            raise ValueError(order)
    return out


def stimulus(mac: Engine, dim: int, n_rows: int, seed=0, exact=False):
    """``exact=True``: small integers, whose products and partial sums are
    exact in every type here, so every accumulate order gives one answer."""
    rng = np.random.default_rng(seed)
    if mac is BF16_ACC24:
        import ml_dtypes
        if exact:
            vals = rng.integers(-8, 9, (n_rows + dim) * dim).astype(np.float32)
        else:
            vals = rng.normal(0, 1, (n_rows + dim) * dim).astype(np.float32)
        bits = vals.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int64)
    else:
        lo, hi = (-4, 5) if exact else (-128, 128)
        bits = rng.integers(lo, hi, (n_rows + dim) * dim).astype(np.int8).astype(np.uint8).astype(np.int64)
    A = bits[: n_rows * dim].reshape(n_rows, dim)
    W = bits[n_rows * dim:].reshape(dim, dim)
    return A, W


def _signed_in(mac, x):
    if mac is INT8_INT32:
        return np.asarray(x, dtype=np.int64).astype(np.uint8).astype(np.int8).astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def run_rig(matrix: MatrixEngine, mac: Engine, dim: int, n_rows: int, seed=0, exact=False):
    arch = mxu_rig(matrix, mac, dim, n_rows)
    mod = df.build(arch.region(), target="simulator")
    A, W = stimulus(mac, dim, n_rows, seed, exact)
    out = np.zeros(n_rows * dim, dtype=np.uint32)
    mod(A.reshape(-1).astype(np.uint32), W.reshape(-1).astype(np.uint32), out)
    got = out.reshape(n_rows, dim).astype(np.int64)
    # The composite's verdict order: its own, or the accepted one the bound
    # matrix engine changed it to (README D-15).
    want = matrix_rows(_signed_in(mac, A), _signed_in(mac, W), mac, arch.reference_order)
    mask = (1 << mac.OUT_BITS) - 1
    return got, want & mask

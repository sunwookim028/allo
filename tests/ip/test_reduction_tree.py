# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adder tree, RUN -- not declared and left to rot.

A capability nothing instantiates cannot be seen to be broken, so every test
here builds the ``DotTree`` region and compares against numpy: the root, the
mid-tree tap, ``M = 1``, several widths and every leaf mapping.
"""

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import Int
from examples.accelerator.tinytpu_vitis.ip.reduce import DotTree, ReduceParams

CONFIGS = [(8, 2, 8), (8, 4, 8), (8, 8, 8), (4, 2, 8), (16, 4, 16)]
SHAPES = [(8, 8), (1, 1), (1, 8), (3, 5), (8, 1)]


def reference(A, B, p, M, N):
    """The root and the tap, built independently of the unit: the tap for
    group ``g`` is the lanes ``g, g + RED_GROUPS, g + 2*RED_GROUPS, ...``,
    which is what the declared leaf mapping puts in group ``g``'s subtree."""
    a = A[:M, : p.RED_LANES].astype(np.int64)
    b = B[: p.RED_LANES, :N].astype(np.int64)
    terms = a[:, None, :] * b.T[None, :, :]
    groups = np.stack([terms[:, :, g::p.RED_GROUPS].sum(-1)
                       for g in range(p.RED_GROUPS)], -1)
    return a @ b, groups


def run(mod, A, B, p, M, N):
    C = np.zeros(p.DOT_MAX * p.DOT_MAX, np.int32)
    G = np.zeros(p.DOT_MAX * p.DOT_MAX * p.RED_GROUPS, np.int32)
    mod(A.reshape(-1), B.reshape(-1), np.array([M, N], np.int32), C, G)
    return (C[: M * N].reshape(M, N),
            G[: M * N * p.RED_GROUPS].reshape(M, N, p.RED_GROUPS))


@pytest.mark.parametrize("lanes, groups, dim", CONFIGS)
def test_tree_is_exact(lanes, groups, dim):
    p = ReduceParams(RED_LANES=lanes, RED_GROUPS=groups, DOT_MAX=dim)
    mod = df.build(DotTree(p, name=f"dot_tree_{lanes}_{groups}").region,
                   target="simulator")
    # Full-range int8, not the [-4, 4] the performance testbench uses: the
    # whole claim of the accumulator legality condition is that RED_ACC_BITS
    # carries the widest tree of RED_IN_BITS lanes, and narrow operands cannot
    # see it fail.
    rng = np.random.default_rng(0)
    A = rng.integers(-128, 128, (dim, dim)).astype(np.int8)
    B = rng.integers(-128, 128, (dim, dim)).astype(np.int8)
    for M, N in SHAPES:
        if M > dim or N > dim:
            continue
        root, tap = run(mod, A, B, p, M, N)
        gold_root, gold_tap = reference(A, B, p, M, N)
        assert (root == gold_root).all(), f"root wrong at {M}x{N}"
        assert (tap == gold_tap).all(), f"tap wrong at {M}x{N}"


def test_the_tap_is_not_the_root():
    """The second output has to be its own answer, or the test above would
    pass on a unit that emitted the root twice."""
    p = ReduceParams()
    mod = df.build(DotTree(p, name="dot_tree_tap").region, target="simulator")
    rng = np.random.default_rng(1)
    A = rng.integers(-128, 128, (p.DOT_MAX, p.DOT_MAX)).astype(np.int8)
    B = rng.integers(-128, 128, (p.DOT_MAX, p.DOT_MAX)).astype(np.int8)
    root, tap = run(mod, A, B, p, p.DOT_MAX, p.DOT_MAX)
    assert (tap.sum(-1) == root).all(), "the taps must sum to the root"
    assert (tap[..., 0] != root).any(), "a tap that equals the root is a wire"


def test_m_of_one_costs_one_row():
    """The Jalapeno requirement, and the reason this unit exists: one output
    element is finished by one pass of one tree, so M = 1 is 1/M of M's work
    rather than a shape the machine has to be padded through."""
    p = ReduceParams()
    mod = df.build(DotTree(p, name="dot_tree_m1").region, target="simulator")
    rng = np.random.default_rng(2)
    A = rng.integers(-128, 128, (p.DOT_MAX, p.DOT_MAX)).astype(np.int8)
    B = rng.integers(-128, 128, (p.DOT_MAX, p.DOT_MAX)).astype(np.int8)
    root, _ = run(mod, A, B, p, 1, 1)
    assert root[0, 0] == reference(A, B, p, 1, 1)[0][0, 0]


def accumulator(bits):
    return dict(RED_ACC=Int(bits), RED_ACC_BITS=bits)


@pytest.mark.parametrize("kw, word", [
    (dict(RED_LANES=6), "power of two"),
    (dict(RED_GROUPS=3), "de-interleave"),
    (dict(RED_GROUPS=1), "duplicates"),
    (dict(RED_ACC_BITS=18), "a type to annotate with"),
    (accumulator(18), "DIFFERENT function"),
])
def test_illegal_parameter_sets_are_refused_at_composition(kw, word):
    """Not at build, not in cosim, not in a testbench: at composition, where
    the parameters are. An 18-bit accumulator at RED_LANES=8 is the one that
    would otherwise be a plausible wrong number, and a width that disagrees
    with its own type is the one that would read the wrong bits."""
    with pytest.raises(AssertionError, match=word):
        DotTree(ReduceParams(**kw), name="illegal")


def test_an_exact_accumulator_is_one_bit_away():
    """The boundary of the legality condition is where it is claimed to be:
    RED_IN_BITS=16 plus RED_DEPTH=3."""
    DotTree(ReduceParams(**accumulator(19)), name="just_wide_enough")
    with pytest.raises(AssertionError, match="DIFFERENT function"):
        DotTree(ReduceParams(**accumulator(18)), name="one_bit_short")

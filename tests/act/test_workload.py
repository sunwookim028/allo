# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A spec is its own reference: the einsum gold against hand-written numpy."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from act import workloads  # noqa: E402
from act.workload import SpecError, Tensor, Workload  # noqa: E402

EXTENTS = {"M": 8, "K": 4, "N": 4}


def operands(workload, seed=0):
    rng = np.random.default_rng(seed)
    return {t.name: rng.integers(-4, 5, t.extent(EXTENTS)).astype(np.int8)
            for t in workload.operands}


def test_every_registered_workload_matches_hand_written_numpy():
    hand = {
        "gemm": lambda v: v["A"].astype(np.int64) @ v["B"].astype(np.int64),
        "gemm.relu": lambda v: np.maximum(
            v["A"].astype(np.int64) @ v["B"].astype(np.int64), 0),
        "gemm.sum": lambda v: (v["A"].astype(np.int64) @ v["B"].astype(np.int64)
                               + v["A"].astype(np.int64)
                               @ v["B2"].astype(np.int64)),
        "gemm.sum.relu": lambda v: np.maximum(
            v["A"].astype(np.int64) @ v["B"].astype(np.int64)
            + v["A"].astype(np.int64) @ v["B2"].astype(np.int64), 0),
    }
    assert set(hand) == set(workloads.WORKLOADS)
    for name, gold in hand.items():
        workload = workloads.get(name)
        values = operands(workload)
        assert np.array_equal(workload.evaluate(values), gold(values))


def test_free_ranks_are_the_result_ranks():
    for workload in workloads.WORKLOADS.values():
        assert set(workload.free) == set(workload.result.ranks)


def test_a_contraction_must_cover_the_iteration_space():
    with pytest.raises(SpecError):
        Workload(name="bad", ranks=("M", "K", "N"), reduce=("K",),
                 operands=(Tensor("A", ("M", "K")), Tensor("B", ("K",))),
                 result=Tensor("C", ("M", "N")),
                 contractions=(("A", "B"),))


def test_an_unknown_epilogue_op_is_named():
    with pytest.raises(SpecError) as caught:
        Workload(name="bad", ranks=("M", "K", "N"), reduce=("K",),
                 operands=(Tensor("A", ("M", "K")), Tensor("B", ("K", "N"))),
                 result=Tensor("C", ("M", "N")),
                 contractions=(("A", "B"),), epilogue=("gelu",))
    assert "POINTWISE" in str(caught.value)


def test_unknown_workload_lists_what_exists():
    with pytest.raises(KeyError) as caught:
        workloads.get("conv2d")
    assert "gemm" in str(caught.value)

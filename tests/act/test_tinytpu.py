# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TinyTPU-isa target, anchored on the shipped hand-verified program."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from act import workloads  # noqa: E402
from act.nest import INTRINSIC, OUTER, Loop, Refused  # noqa: E402
from act.search import Problem, search  # noqa: E402

pytest.importorskip("allo._mlir", reason="the target needs the bindings")

from examples.accelerator.tinytpu_vitis.act_target import (  # noqa: E402
    TINYTPU, roles_of,
)
from examples.accelerator.tinytpu_vitis.bench_isa import SHAPES  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    gemm_program,
)
from examples.accelerator.tinytpu_vitis.microarch_isa import T  # noqa: E402


def shipped_nest(M, K, N):
    return (Loop("N", N // T, OUTER), Loop("K", K // T, OUTER),
            Loop("M", M, INTRINSIC), Loop("K", T, INTRINSIC),
            Loop("N", T, INTRINSIC))


def extents(workload, shape):
    return dict(zip(workload.ranks, shape))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", ["gemm", "gemm.relu"])
def test_the_shipped_mapping_re_emits_the_shipped_program_word_for_word(
        name, shape):
    workload = workloads.get(name)
    got = TINYTPU.lower(workload, extents(workload, shape),
                        shipped_nest(*shape))
    assert got == gemm_program(*shape, relu=name.endswith("relu"))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", sorted(workloads.WORKLOADS))
def test_every_encodable_mapping_computes_the_spec(name, shape):
    workload = workloads.get(name)
    size = extents(workload, shape)
    result = search(Problem(workload, size), TINYTPU)
    if not result.candidates:
        pytest.skip(f"{name} {shape}: every mapping refused")
    for candidate in result.candidates:
        assert TINYTPU.verify(workload, size, candidate.program), (
            f"{name} {shape} {candidate.label} computes the wrong thing")


def test_refusals_carry_a_cause_and_not_a_message_to_match_on():
    workload = workloads.get("gemm.relu")
    size = {"M": 16, "K": 16, "N": 16}
    result = search(Problem(workload, size), TINYTPU)
    assert result.considered == sum(
        result.census.counts.values()) + len(result.candidates)
    assert set(result.census.counts) <= {
        "acc-peel", "ar-distance", "AGU_TERMS", "LOOP_DEPTH", "intrinsic",
        "capacity", "machine", "resources", "nest", "trip-count", "coverage",
        "spatial", "shape"}


def test_the_acc_field_is_what_refuses_most_of_the_mapspace():
    workload = workloads.get("gemm.relu")
    result = search(Problem(workload, {"M": 16, "K": 16, "N": 16}), TINYTPU)
    assert result.census.counts["acc-peel"] > result.considered // 2


def test_a_split_reduction_is_refused_by_the_peel_and_not_by_accident():
    workload = workloads.get("gemm")
    nest = (Loop("K", 2, OUTER), Loop("N", 4, OUTER), Loop("K", 2, OUTER),
            Loop("M", 16, INTRINSIC), Loop("K", 4, INTRINSIC),
            Loop("N", 4, INTRINSIC))
    with pytest.raises(Refused) as caught:
        TINYTPU.lower(workload, {"M": 16, "K": 16, "N": 16}, nest)
    assert caught.value.cause == "acc-peel"


def test_a_spatial_loop_is_refused_because_there_is_one_array():
    workload = workloads.get("gemm")
    nest = (Loop("N", 4, OUTER, True), Loop("K", 4, OUTER),
            Loop("M", 16, INTRINSIC), Loop("K", 4, INTRINSIC),
            Loop("N", 4, INTRINSIC))
    with pytest.raises(Refused) as caught:
        TINYTPU.lower(workload, {"M": 16, "K": 16, "N": 16}, nest)
    assert caught.value.cause == "spatial"


def test_roles_come_from_the_spec_and_not_from_the_rank_names():
    workload = workloads.get("gemm")
    roles = roles_of(workload)
    assert (roles.row, roles.reduce, roles.column) == ("M", "K", "N")


def test_the_cost_is_a_makespan_and_an_emit_count():
    workload = workloads.get("gemm.relu")
    result = search(Problem(workload, {"M": 16, "K": 16, "N": 16}), TINYTPU)
    for candidate in result.candidates:
        makespan, emits = candidate.cost
        plan = candidate.priced.schedule
        assert makespan == plan.makespan
        assert emits == TINYTPU.emits(candidate.program)
        assert makespan >= max(plan.unit_load.values())

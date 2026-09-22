# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import replace

import numpy as np
import pytest

import allo
from allo.dependence import (
    DependenceError,
    check,
    UndeclaredPremise,
    recorded_claims,
)
from allo.ir.types import int32

BECAUSE = "test premise"


def carried_read_after_write(A: int32[10]):
    for i in range(1, 10):
        A[i] = A[i - 1] + 1


def carried_at_four(A: int32[10]):
    for i in range(4, 10):
        A[i] = A[i - 4] + 1


def indirect(idx: int32[10], A: int32[10]):
    for i in range(10):
        A[idx[i]] = A[idx[i]] + 1


def non_uniform_stride(A: int32[20]):
    for i in range(10):
        A[2 * i] = A[i] + 1


def elementwise(A: int32[10], B: int32[10]):
    for i in range(10):
        B[i] = A[i] + 1


def read_modify_write(A: int32[10]):
    for i in range(10):
        A[i] = A[i] + 1


def row_carried(A: int32[10, 10]):
    for i in range(1, 10):
        for j in range(10):
            A[i, j] = A[i - 1, j] + 1


def guarded_write(A: int32[10], n: int32[1]):
    for i in range(1, 10):
        if n[0] > 0:
            A[i] = A[i - 1] + 1


# ---------------------------------------------------------------- the rule


def test_refuses_a_provably_false_inter_claim():
    s = allo.customize(carried_read_after_write)
    with pytest.raises(DependenceError) as excinfo:
        s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)
    message = str(excinfo.value)
    assert "variable=A inter false" in message
    assert "carried_read_after_write:i" in message
    assert "RAW" in message and "distance 1" in message
    assert "write A[i]" in message
    assert "read  A[i - 1]" in message
    assert "dependent=True, distance=1" in message


def test_refuses_a_distance_longer_than_the_provable_one():
    s = allo.customize(carried_read_after_write)
    with pytest.raises(DependenceError) as excinfo:
        s.dependence(
            "i",
            "A",
            direction="RAW",
            distance=4,
            dependent=True,
            because=BECAUSE,
        )
    assert "distance 1" in str(excinfo.value)


def test_accepts_the_provable_distance_itself():
    s = allo.customize(carried_read_after_write)
    s.dependence("i", "A", direction="RAW", distance=1, dependent=True, because=BECAUSE)
    assert len(s.dependence_obligations) == 1


def test_accepts_a_real_dependence_it_cannot_prove():
    """The trap. `A[idx[i]]` carries a RAW at distance 1 whenever `idx`
    repeats, and this asserts the answer really does change -- yet the claim is
    accepted, because the subscript is not affine in `i` and no proof exists.
    Refusing here for lack of proof would remove the only reason the primitive
    exists."""
    s = allo.customize(indirect)
    s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)

    repeated = np.zeros(10, dtype=np.int32)
    A = np.zeros(10, dtype=np.int32)
    allo.customize(indirect).build()(repeated, A)
    assert A[0] == 10, "the unprovable dependence is a real one"


def test_accepts_a_dependence_whose_distance_is_not_uniform():
    """`A[2*i] = A[i]` aliases only for some iteration pairs and at no single
    distance, so nothing is provable at a denied distance."""
    s = allo.customize(non_uniform_stride)
    s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)
    assert len(s.dependence_obligations) == 1


def test_accepts_when_no_dependence_is_there_at_all():
    s = allo.customize(elementwise)
    s.dependence("i", "B", dep_type="inter", dependent=False, because=BECAUSE)


def test_accepts_an_inter_claim_when_only_an_intra_dependence_is_provable():
    """`A[i] = A[i] + 1` reads and writes one element per iteration and carries
    nothing across iterations, so `inter false` is true and accepted."""
    s = allo.customize(read_modify_write)
    s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)


def test_refuses_a_false_intra_claim():
    s = allo.customize(read_modify_write)
    with pytest.raises(DependenceError) as excinfo:
        s.dependence("i", "A", dep_type="intra", dependent=False, because=BECAUSE)
    assert "WAR" in str(excinfo.value) and "distance 0" in str(excinfo.value)


def test_a_direction_the_claim_does_not_deny_is_not_a_refusal():
    """The kernel's only provable dependence is a RAW; a claim about WAR denies
    nothing that is provable, so it stands."""
    s = allo.customize(carried_read_after_write)
    s.dependence("i", "A", direction="WAR", dependent=False, because=BECAUSE)


def test_refuses_on_the_axis_that_carries_the_dependence_only():
    s = allo.customize(row_carried)
    s.dependence("j", "A", dep_type="inter", dependent=False, because=BECAUSE)
    with pytest.raises(DependenceError) as excinfo:
        s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)
    assert "axis i" in str(excinfo.value)


def test_a_guarded_access_is_not_a_witness():
    """A store under an `if` may never run, so it proves nothing. The rule
    accepts, and the obligation records what that acceptance rests on."""
    s = allo.customize(guarded_write)
    s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)


# ------------------------------------------------------- the standing check


def test_a_claim_is_read_back_from_the_ir():
    s = allo.customize(carried_at_four)
    s.dependence(
        "i", "A", direction="RAW", distance=4, dependent=True, because=BECAUSE
    )
    claims = list(recorded_claims(s.module))
    assert len(claims) == 1
    where, _, _, claim = claims[0]
    assert where == "carried_at_four:i"
    assert claim.distance == 4 and claim.dependent and claim.direction == "RAW"
    assert claim.because == BECAUSE


def test_the_standing_check_survives_a_later_primitive():
    """The claim is re-checked after every later primitive, from the loop
    attribute rather than from anything the schedule remembers, so a primitive
    that rewrote the accesses under a surviving claim could not leave it
    standing. `s.pipeline` keeps the claim and changes no access."""
    s = allo.customize(carried_at_four)
    s.dependence(
        "i", "A", direction="RAW", distance=4, dependent=True, because=BECAUSE
    )
    s.pipeline("i")
    assert len(list(recorded_claims(s.module))) == 1


def test_the_standing_check_blames_the_primitive_that_broke_the_claim():
    s = allo.customize(carried_at_four)
    s.dependence(
        "i", "A", direction="RAW", distance=4, dependent=True, because=BECAUSE
    )
    where, loop, memref, claim = next(iter(recorded_claims(s.module)))
    with s.module.context:
        with pytest.raises(DependenceError, match=r"after s\.reorder\(\)"):
            check(loop, memref, replace(claim, distance=8), where, after="reorder")


def test_a_loop_transformation_drops_the_claim_rather_than_moving_it():
    """`s.split` rewrites the band onto two new loops and carries the
    `dependence` attribute onto neither, so the pragma silently disappears.
    That is a defect in `split`, not in the rule, and it is why no primitive in
    this tree can currently falsify a standing claim. See the limitations
    register, item 21."""
    s = allo.customize(carried_at_four)
    s.dependence(
        "i", "A", direction="RAW", distance=4, dependent=True, because=BECAUSE
    )
    s.split("i", 2)
    assert list(recorded_claims(s.module)) == []
    assert "HLS dependence" not in str(s.build(target="vhls"))


# --------------------------------------------------------- the obligation


def test_an_accepted_claim_leaves_an_obligation_naming_its_premise():
    s = allo.customize(indirect)
    s.dependence(
        "i",
        "A",
        dep_type="inter",
        dependent=False,
        because="the caller never repeats an index within one iteration",
    )
    (obligation,) = s.dependence_obligations
    assert obligation.where == "indirect:i"
    assert obligation.pragma == "variable=A inter false"
    assert "never repeats" in obligation.premise
    assert "never repeats" in str(obligation)


def test_a_claim_without_a_premise_warns_at_the_point_of_use():
    s = allo.customize(indirect)
    with pytest.warns(UndeclaredPremise, match="accepted but not proven"):
        s.dependence("i", "A", dep_type="inter", dependent=False)
    assert s.dependence_obligations[0].premise is None
    assert "premise not declared" in str(s.dependence_obligations[0])


def test_a_declared_premise_does_not_warn():
    s = allo.customize(indirect)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndeclaredPremise)
        s.dependence("i", "A", dep_type="inter", dependent=False, because=BECAUSE)


def test_the_premise_is_emitted_beside_the_pragma():
    s = allo.customize(indirect)
    s.dependence(
        "i",
        "A",
        dep_type="inter",
        dependent=False,
        because="check_program() rejects a repeated index",
    )
    code = str(s.build(target="vhls"))
    lines = [line.strip() for line in code.splitlines()]
    pragma = next(i for i, line in enumerate(lines) if "HLS dependence" in line)
    assert lines[pragma - 1] == (
        "// dependence obligation, checked by no tool: "
        "check_program() rejects a repeated index"
    )


def test_no_premise_emits_no_comment():
    s = allo.customize(indirect)
    with pytest.warns(UndeclaredPremise):
        s.dependence("i", "A", dep_type="inter", dependent=False)
    code = str(s.build(target="vhls"))
    assert "dependence obligation" not in code


if __name__ == "__main__":
    pytest.main([__file__])

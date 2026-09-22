# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import allo
from allo.encoding import (
    Encoding,
    EncodingError,
    TINYTPU_ISA,
    UNCONSTRAINED,
    violations,
)
from allo.ir.types import int8, int32

M = K = N = 8


def reduction_with_predicated_accumulator(A: int8[M, K], B: int8[K, N], C: int32[M, N]):
    for i, j in allo.grid(M, N):
        for k in range(K):
            if k == 0:
                C[i, j] = A[i, k] * B[k, j]
            else:
                C[i, j] = C[i, j] + A[i, k] * B[k, j]


def reduction_with_peeled_accumulator(A: int8[M, K], B: int8[K, N], C: int32[M, N]):
    for i, j in allo.grid(M, N):
        C[i, j] = A[i, 0] * B[0, j]
    for i0, j0 in allo.grid(M, N):
        for k in range(K - 1):
            C[i0, j0] = C[i0, j0] + A[i0, k + 1] * B[k + 1, j0]


def four_term_address(X: int32[4096], Y: int32[4096]):
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    Y[a * 512 + b * 64 + c * 8 + d] = X[a * 512 + b * 64 + c * 8 + d]


def five_deep(X: int32[32], Y: int32[32]):
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        Y[a * 16 + b * 8 + c * 4 + d * 2 + e] = 1


def indirect_address(idx: int32[8], X: int32[8], Y: int32[8]):
    for i in range(8):
        Y[i] = X[idx[i]]


def elementwise(A: int32[10], B: int32[10]):
    for i in range(10):
        B[i] = A[i] + 1


def rules_of(module, encoding):
    return sorted({v.rule for v in violations(module, encoding)})


def test_unconstrained_encoding_refuses_nothing():
    for kernel in (
        reduction_with_predicated_accumulator,
        four_term_address,
        five_deep,
        indirect_address,
    ):
        assert violations(allo.customize(kernel).module, UNCONSTRAINED) == []


def test_predicated_field_refuses_an_accumulator_guarded_by_its_reduction_index():
    s = allo.customize(reduction_with_predicated_accumulator)
    found = violations(s.module, TINYTPU_ISA)
    assert [v.rule for v in found] == ["predicated-field"]
    assert found[0].where.endswith("i/j/k")
    assert "index-set split" in found[0].repair


def test_predicated_field_accepts_the_peeled_form():
    s = allo.customize(reduction_with_peeled_accumulator)
    assert "predicated-field" not in rules_of(s.module, TINYTPU_ISA)


def test_address_terms_counts_distinct_induction_variables():
    s = allo.customize(four_term_address)
    budget_of_three = Encoding(name="three-term", address_terms=3)
    found = violations(s.module, budget_of_three)
    assert {v.rule for v in found} == {"address-terms"}
    assert "needs 4 address terms" in found[0].found
    assert violations(s.module, Encoding(address_terms=4)) == []


def test_loop_depth_counts_the_enclosing_nest():
    s = allo.customize(five_deep)
    found = violations(s.module, Encoding(name="four-deep", loop_depth=4))
    assert {v.rule for v in found} == {"loop-depth"}
    assert "nested 5 deep" in found[0].found
    assert violations(s.module, Encoding(loop_depth=5)) == []


def test_affine_addressing_refuses_an_index_computed_outside_the_map():
    s = allo.customize(indirect_address)
    found = violations(s.module, Encoding(requires_affine_addressing=True))
    assert {v.rule for v in found} == {"affine-addressing"}


def test_static_trip_count_refuses_the_min_bound_a_ragged_split_introduces():
    s = allo.customize(elementwise)
    s.split("i", 3)
    found = violations(s.module, Encoding(requires_static_trip_counts=True))
    assert {v.rule for v in found} == {"static-trip-count"}
    assert "are not constant" in found[0].found


def test_static_trip_count_accepts_a_split_that_divides_the_extent():
    s = allo.customize(elementwise)
    s.split("i", 5)
    assert violations(s.module, Encoding(requires_static_trip_counts=True)) == []


def test_encodable_on_refuses_at_the_primitive_that_broke_it():
    s = allo.customize(elementwise)
    s.encodable_on(Encoding(name="static-sequencer", requires_static_trip_counts=True))
    s.split("i", 5)
    with pytest.raises(EncodingError) as excinfo:
        s.split("i.inner", 3)
    assert excinfo.value.after == "split"
    assert [v.rule for v in excinfo.value.violations] == ["static-trip-count"]


def test_encodable_on_refuses_at_the_declaration_when_already_illegal():
    s = allo.customize(reduction_with_predicated_accumulator)
    with pytest.raises(EncodingError) as excinfo:
        s.encodable_on(TINYTPU_ISA)
    assert excinfo.value.after == "encodable_on"
    assert "tinytpu-isa" in str(excinfo.value)


def test_encodable_on_is_inert_when_the_encoding_constrains_nothing():
    s = allo.customize(reduction_with_predicated_accumulator)
    s.encodable_on(UNCONSTRAINED)
    s.unroll("k")


def test_a_declared_encoding_does_not_disturb_codegen():
    s = allo.customize(elementwise)
    s.encodable_on(TINYTPU_ISA)
    s.split("i", 5)
    assert "affine.for" in str(s.module)


if __name__ == "__main__":
    pytest.main([__file__])

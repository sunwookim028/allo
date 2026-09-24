# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PyTorch suite: what the extractor promises before any hardware runs."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
pytest.importorskip("torch", reason="the suite's front end is torch.fx")
pytest.importorskip("allo._mlir", reason="the specs are validated against the build")

from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program  # noqa: E402
from examples.accelerator.tinytpu_vitis.workloads import (  # noqa: E402
    burst, extract, models,
)


@pytest.mark.parametrize("name", models.MAPPABLE)
def test_every_layer_maps(name):
    ex = extract.of(name)
    assert ex.layers, f"{name} extracted no layers"
    assert not ex.refusals, [r.why for r in ex.refusals]


@pytest.mark.parametrize("name", models.MAPPABLE)
def test_specs_satisfy_the_corpus_schema(name):
    for sp in extract.specs(extract.of(name)):
        spec_mod.validate(sp)
        assert spec_mod.fits_build(sp) is None
        assert sp["einsum"] == "mk,kn->mn"
        assert sp["epilogue"][-1] == "saturate"


@pytest.mark.parametrize("name", models.MAPPABLE)
def test_the_corpus_reference_submission_consumes_the_specs(name):
    """The convention claim, tested rather than asserted: `act.baseline`, which
    exists to map `act/corpus/`, maps every spec the extractor emits with no
    change to it."""
    from examples.accelerator.tinytpu_vitis.act import baseline
    for sp in extract.specs(extract.of(name)):
        assert baseline.program(sp)


def test_the_probe_model_is_refused_rather_than_approximated():
    ex = extract.of("mlp_bias")
    why = " ".join(r.why for r in ex.refusals)
    assert "bias" in why and "sigmoid" in why.lower() or "not an int8 GEMM" in why
    assert len(ex.refusals) == 2


def test_relu_is_fused_only_into_the_linear_it_consumes():
    layers = extract.of("mlp_tiny").layers
    assert [l.relu for l in layers] == [True, False]


def test_the_widening_law_reproduces_both_measured_points():
    for shape, measured, predicted, every in burst.check_measured(16):
        assert predicted == measured, shape
        assert every != measured, (
            f"{shape}: charging every removed iteration would also fit, so "
            f"the two laws are not distinguished by this point")


def test_burst_spans_come_from_the_assembled_header():
    assert burst.spans(gemm_program(64, 64, 64)) == (64, 64)
    assert burst.spans(gemm_program(4, 16, 16)) == (4, 16)


if __name__ == "__main__":
    pytest.main([__file__])

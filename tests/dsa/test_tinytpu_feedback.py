# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from examples.accelerator.tinytpu.feedback import evaluate


def test_tinytpu_access_objective_is_trace_derived():
    result = evaluate()

    assert result["status"] == "pass"
    assert result["total_access_cost"] > 0
    assert result["cost_model"]["vreg_word_cost"] == 1
    assert result["cost_model"]["vmem_word_cost"] == 4
    for benchmark in result["benchmarks"]:
        assert benchmark["vreg_words"] > 0
        assert benchmark["vmem_words"] > 0
        assert benchmark["access_cost"] == (
            benchmark["vreg_words"] + 4 * benchmark["vmem_words"]
        )

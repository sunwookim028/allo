# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``@I.expand``: one layer-level match lowered to a run of tile instructions.

MiniNPU's MXU is the case the mechanism is for -- *fixed-size and repeated*. It holds
one 16x16 stationary tile and consumes one activation row per push, so an ``M``-row
layer is necessarily one ``vmatload`` plus ``M`` push rounds. ``matmul_layer``
(``examples/accelerator/mininpu/isa.py``) matches the whole layer; its ``@I.expand``
emits that run, after allocation, from the solved ``M`` and the allocated offsets.

(Re-homed here from the quarantined FEATHER example; see
``drafts/plan-access-first-class.md`` §2.4.)
"""

import inspect
from collections import Counter

import numpy as np
import pytest

from allo.exp.dsa.errors import AllocationError

from examples.accelerator.mininpu.isa import MXU_DIM, npu


def source(M: int) -> str:
    """``Z[M,16] = X[M,16] @ W^T`` as batched TOSA -- the form torch_mlir emits.
    Names no memory: every load, push, pop and store below is the backend's."""

    def t(*dims):
        return f"tensor<{'x'.join(map(str, dims))}xf32>"

    sq = t(1, MXU_DIM, MXU_DIM)
    sig = f"({t(1, M, MXU_DIM)}, {sq}, tensor<1xf32>, tensor<1xf32>) -> {t(1, M, MXU_DIM)}"
    return f"""
func.func @main(%x: {t(1, M, MXU_DIM)}, %w: {sq}) -> {t(1, M, MXU_DIM)} {{
  %zp = "tosa.const"() {{values = dense<0.000000e+00> : tensor<1xf32>}} : () -> tensor<1xf32>
  %wt = tosa.transpose %w {{perms = array<i32: 0, 2, 1>}} : ({sq}) -> {sq}
  %z = tosa.matmul %x, %wt, %zp, %zp : {sig}
  return %z : {t(1, M, MXU_DIM)}
}}
"""


def run(M: int, seed: int = 0):
    """Compile, run on the functional simulator, diff against NumPy."""
    prog = npu.compile_program(source(M))
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, M, MXU_DIM)).astype(np.float32)
    w = rng.standard_normal((1, MXU_DIM, MXU_DIM)).astype(np.float32)
    got = np.asarray(prog(x, w), np.float32).reshape(1, M, MXU_DIM)
    np.testing.assert_allclose(got, x @ w.transpose(0, 2, 1), rtol=1e-4, atol=1e-4)
    return prog


@pytest.mark.parametrize("M", [2, 4, 8])
def test_a_whole_layer_compiles_expands_and_computes(M):
    """One match per layer, lowered to the full row loop, checked against NumPy."""
    counts = Counter(e.name for e in run(M, seed=M).emits)
    assert counts["vmatload"] == 1  # the stationary tile, hoisted out of the loop
    for name in ("vld", "vmatpush", "vmatpop", "vst"):
        assert counts[name] == M, (name, counts)
    assert "matmul_layer" not in counts  # the macro itself never reaches the stream


def test_the_weight_load_is_hoisted_not_repeated():
    """The point of matching a *layer* rather than a row: `vmatload` is paid once,
    not once per row. A per-row selection could not express that reuse."""
    for M in (2, 8):
        counts = Counter(e.name for e in run(M, seed=M).emits)
        assert counts["vmatload"] == 1 and counts["vmatpush"] == M


def test_expansion_is_parametric_in_the_row_count():
    """Doubling the layer doubles the run; the loop bound is the *solved* M, not a
    constant baked into the ISA."""
    base = len(run(4).emits)
    grown = len(run(8).emits)
    assert grown - base == 4 * (8 - 4)  # 4 instructions per extra row


def test_one_row_still_selects_the_plain_mxu_sequence():
    """The layer macro costs its own emit count (`1 + 4*M`), so at M == 1 it loses to
    `vmatpush` (cost 1) and a single row compiles to the three-step sequence with no
    expansion at all. The cost model, not the structure, draws this line."""
    counts = Counter(e.name for e in run(1).emits)
    assert counts == Counter(
        vmemld=2, vld=1, vmatload=1, vmatpush=1, vmatpop=1, vst=1, vmemst=1
    ), counts


def test_layer_cost_equals_the_length_of_its_expansion():
    """`cost` must be a function of the shape or the tree-DP could not compare one
    layer op against the instructions it lowers into -- and here it is not merely
    shaped like the expansion, it *is* the expansion's length."""
    spec = next(s for s in npu.instructions if s.name == "matmul_layer")
    m_param = list(inspect.signature(spec.access_fn).parameters).index("M")
    for M in (2, 4, 8):
        emitted = len(
            [
                e
                for e in run(M, seed=M).emits
                if e.name in ("vmatload", "vld", "vmatpush", "vmatpop", "vst")
            ]
        )
        assert spec.cost_of({m_param: M}) == emitted


def test_the_source_names_no_memory():
    """Selection, placement, staging and the row loop are all the backend's: the
    source is one `tosa.matmul` over whole matrices."""
    src = source(8)
    assert src.count("tosa.matmul") == 1
    assert "vmem" not in src and "vmat" not in src and "dram" not in src


def test_a_layer_too_tall_for_the_scratchpad_is_refused():
    """The expansion is unbounded, but the operands it reads are not: a layer whose
    three VMEM regions exceed the scratchpad fails loudly during allocation."""
    with pytest.raises(AllocationError):
        npu.compile_program(source(4096))

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scope map's shape, and the one column of it that can be wrong quietly.

``scope.py --check`` is the measurement and costs about a minute; that is a
gate to run, not a unit test. What is here is the part that can rot without
anything noticing: ``chain_sound`` reports ``True`` for all five committed
models, and a predicate that is always ``True`` is indistinguishable from one
that is broken. So it is shown refusing.

Nothing here asserts a measured value. The map is ``scope.py``'s own output
and the only claim made about it is that it is a map.
"""

import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

torch = pytest.importorskip("torch")
MAP = os.path.join(REPO, "examples", "tinytpu", "workloads", "scope_map.json")


def test_map_is_a_map():
    """Every recorded configuration carries all four sections."""
    with open(MAP) as fh:
        book = json.load(fh)
    assert book, f"{MAP} records no configuration"
    for key, data in book.items():
        for section in ("config", "ops", "shapes", "entries", "assumptions"):
            assert data.get(section), f"{key} has no {section!r} section"
        assert set(data["config"]) >= {"T", "MAXDIM", "QD", "DMA_WORDS"}


def test_dataflow_describes_a_chain():
    """A chain is reported as one, and its sources are the chain."""
    from torch import nn
    from examples.tinytpu.workloads import extract, scope

    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16, bias=False)
            self.fc2 = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model, xs = Chain().eval(), (torch.zeros(4, 16),)
    ex = extract.extract("chain", model, xs)
    sources, chain, why = scope.dataflow(extract.trace(model, xs), ex)
    assert chain, why
    assert sources == [-1, 0], sources


def test_dataflow_reads_a_fan_out_instead_of_chaining_it():
    """The finding this column exists for, in its post-fix form.

    Two bias-free Linears off one input: the extractor refuses nothing and
    both layers map. Before the fix, ``run.py`` fed layer 0's output into
    layer 1 on BOTH sides of its comparison, so the comparison agreed with
    itself while the machine computed something the model does not -- 0 of 128
    bytes differing against its own reference, 62 of 128 against ``model(x)``.

    ``chain`` is now a DESCRIPTION, not a soundness condition, because
    ``run.layer_sources`` reads the graph. So the assertion that matters is no
    longer 'the map calls this unsound' but 'the dataflow recorded is the
    graph's': both layers take the model input, and neither is fed the other.
    A test that only checked ``not chain`` would still pass if the sources
    silently went back to being chained.
    """
    from torch import nn
    from examples.tinytpu.workloads import extract, scope

    class FanOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16, bias=False)
            self.fc2 = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc1(x), self.fc2(x)

    model, xs = FanOut().eval(), (torch.zeros(4, 16),)
    ex = extract.extract("fanout", model, xs)
    assert not ex.refusals, "the extractor is expected to accept this graph"
    assert len(ex.layers) == 2
    sources, chain, why = scope.dataflow(extract.trace(model, xs), ex)
    assert not chain, why
    assert sources == [-1, -1], (
        f"both Linears read the model input; run.py recorded {sources}. "
        "If this is [-1, 0] the chaining regressed and the end-to-end check "
        "is comparing two things that are both wrong the same way.")


def test_dataflow_handles_a_reused_module():
    """A module called twice is two layers with the second fed by the first."""
    from torch import nn
    from examples.tinytpu.workloads import extract, scope

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc(self.fc(x))

    model, xs = Twice().eval(), (torch.zeros(4, 16),)
    ex = extract.extract("twice", model, xs)
    assert len(ex.layers) == 2, "one module called twice is two layers"
    sources, chain, why = scope.dataflow(extract.trace(model, xs), ex)
    assert chain, why
    assert sources == [-1, 0], sources

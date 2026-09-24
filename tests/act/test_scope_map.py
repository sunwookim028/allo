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


def test_chain_sound_accepts_a_chain():
    from torch import nn
    from examples.tinytpu.workloads import extract, scope

    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16, bias=False)
            self.fc2 = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model, xs = Chain().eval(), (torch.zeros(4, 16),)
    ex = extract.extract("chain", model, xs)
    sound, why = scope.chain_sound(extract.trace(model, xs), ex)
    assert sound, why


def test_chain_sound_refuses_a_fan_out():
    """The finding this column exists for.

    Two bias-free Linears off one input: the extractor refuses nothing, both
    layers map, and ``run.py`` feeds layer 0's output into layer 1 on BOTH
    sides of its comparison -- so the comparison agrees with itself while the
    machine computes something the model does not. The map must call that
    unsound.
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
    sound, why = scope.chain_sound(extract.trace(model, xs), ex)
    assert not sound, ("a fan-out graph was called a chain; the map's "
                       "topology_sound column no longer means anything")
    assert "not a chain" in why


def test_mapped_nodes_handles_a_reused_module():
    """A module called twice is two layers and two nodes, not one node twice."""
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
    nodes = scope.mapped_nodes(extract.trace(model, xs), ex)
    assert len(nodes) == len(ex.layers) == 2
    assert nodes[0] is not nodes[1]

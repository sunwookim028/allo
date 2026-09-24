# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benchmark suite: ordinary `nn.Module`s, nothing Allo-specific in them.

Four MLPs a reader recognises, sized so every layer fits one TinyTPU build
(M, K, N <= MAXDIM), plus one probe model that deliberately does not map.
`mlp_deep` and `mlp_wide` are the two ends of the same axis: a run of small
layers where everything is fixed cost, and two layers at the steady-state
shape the GEMM table is quoted at. Prose:
docs/source/designs/workload_suite.rst."""

import torch
from torch import nn


class MlpTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MlpDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        self.fc3 = nn.Linear(16, 12, bias=False)
        self.fc4 = nn.Linear(12, 8, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)


class MlpSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MlpWide(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MlpBias(nn.Module):
    """The probe: a bias the ISA has no epilogue for and a sigmoid it has no
    unit for. It exists so the report names refusals instead of implying the
    suite covers everything an MLP can contain."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=True)
        self.fc2 = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)))


MODELS = {
    "mlp_tiny": (MlpTiny, 4, 16),
    "mlp_deep": (MlpDeep, 4, 16),
    "mlp_small": (MlpSmall, 8, 32),
    "mlp_wide": (MlpWide, 64, 64),
    "mlp_bias": (MlpBias, 4, 16),
}

MAPPABLE = ("mlp_tiny", "mlp_deep", "mlp_small", "mlp_wide")


def build(name, seed=0):
    """`(model, example_inputs)`, with int8-valued weights already in place.

    The weights are drawn in [-8, 8) so that quantising the module for the
    end-to-end check is a cast and not a calibration: the suite measures a
    mapping, and inventing a quantisation scheme here would put a second
    unvalidated thing between the model and the number."""
    cls, batch, features = MODELS[name]
    torch.manual_seed(seed)
    model = cls()
    model.eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randint(-8, 8, p.shape).to(p.dtype))
    return model, (torch.randint(-8, 8, (batch, features)).float(),)


def names():
    return list(MODELS)

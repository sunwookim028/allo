# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A PyTorch module in, `act/corpus/`'s workload specs out, one per layer.

`AlloTracer` plus `torch.fx`'s `ShapeProp` give a typed graph with concrete
shapes, and that is the whole of what a spec needs: M, K, N, the dtype and
whether a ReLU is fused. Nothing here calls `from_pytorch`, because its back
half synthesises a bespoke Allo design *for* the model, which is the opposite
of mapping the model *onto* a fixed machine. Prose:
docs/source/designs/workload_suite.rst.

Every node the rules below do not name becomes a `Refusal`, never an
approximation."""

import inspect
import json
import os
import sys
from dataclasses import dataclass

import torch
from torch import nn
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..", "..")))
from allo.frontend.tracer import AlloTracer  # noqa: E402
from examples.accelerator.tinytpu_vitis.act import spec as spec_mod  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import MAXDIM  # noqa: E402

SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
RELU_FUNCTIONS = (torch.relu, torch.nn.functional.relu, torch.relu_)


@dataclass(frozen=True)
class Layer:
    name: str
    module: str
    m: int
    k: int
    n: int
    relu: bool


@dataclass(frozen=True)
class Refusal:
    where: str
    why: str


@dataclass(frozen=True)
class Extraction:
    model: str
    layers: tuple
    refusals: tuple
    graph: str

    @property
    def mappable(self):
        return not self.refusals


def trace(model, example_inputs):
    """`AlloTracer` + `ShapeProp`: a graph whose every node carries its shape."""
    signature = inspect.signature(model.forward)
    named = [p.name for i, p in enumerate(signature.parameters.values())
             if i < len(example_inputs)]
    concrete = {p.name: p.default for p in signature.parameters.values()
                if p.name not in named}
    tracer = AlloTracer(model, concrete_args=concrete)
    graph = tracer.trace()
    gm = GraphModule(tracer.root, graph, model.__class__.__name__)
    ShapeProp(gm).propagate(*example_inputs, *concrete.values())
    return gm


def shape_of(node):
    meta = node.meta.get("tensor_meta")
    return None if meta is None else tuple(meta.shape)


def is_relu(gm, node):
    if node.op == "call_module":
        return isinstance(gm.get_submodule(node.target), nn.ReLU)
    if node.op == "call_function":
        return node.target in RELU_FUNCTIONS
    return node.op == "call_method" and node.target in ("relu", "relu_")


def is_linear(gm, node):
    return (node.op == "call_module"
            and isinstance(gm.get_submodule(node.target), nn.Linear))


def rows_of(shape):
    rows = 1
    for extent in shape[:-1]:
        rows *= extent
    return rows


def extract(name, model, example_inputs):
    """Every `nn.Linear` as a workload spec, every other node as a refusal."""
    gm = trace(model, example_inputs)
    layers, refusals, fused, refused = [], [], set(), set()
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr") or node in fused:
            continue
        if not is_linear(gm, node):
            if is_relu(gm, node):
                source = node.args[0] if node.args else None
                refusals.append(Refusal(
                    node.name,
                    f"the Linear it consumes was itself refused ({source.name})"
                    if source in refused else
                    "a ReLU that does not consume a Linear result directly, or "
                    "is not its only consumer, has no fused epilogue to ride on"))
            else:
                refusals.append(Refusal(
                    node.name, f"{node.op} {node.target} is not an int8 GEMM "
                               f"or a fused ReLU; this build computes "
                               f"int8 x int8 -> int32 with a ReLU epilogue"))
            continue
        linear = gm.get_submodule(node.target)
        if linear.bias is not None:
            refusals.append(Refusal(
                node.name, "nn.Linear with bias: the ISA has vadd but no "
                           "mapping that broadcasts a bias row into the "
                           "accumulator, so the baseline cannot express it"))
            refused.add(node)
            continue
        shape = shape_of(node.args[0])
        if shape is None:
            refusals.append(Refusal(node.name, "ShapeProp gave the activation "
                                               "no tensor_meta"))
            refused.add(node)
            continue
        users = list(node.users)
        relu = len(users) == 1 and is_relu(gm, users[0])
        if relu:
            fused.add(users[0])
        m, k, n = rows_of(shape), linear.in_features, linear.out_features
        if max(m, k, n) > MAXDIM:
            refusals.append(Refusal(
                node.name, f"{m}x{k}x{n} does not fit this build's "
                           f"MAXDIM={MAXDIM}"))
            refused.add(node)
            continue
        layers.append(Layer(f"{name}_l{len(layers)}", node.target, m, k, n, relu))
    return Extraction(name, tuple(layers), tuple(refusals), str(gm.graph))


def to_spec(extraction, layer, index):
    """One layer in the convention `act/corpus/` already uses, validated by
    the same `act.spec.validate` the corpus is held to."""
    epilogue = ["relu", "saturate"] if layer.relu else ["saturate"]
    tail = " with a fused ReLU" if layer.relu else " with no epilogue"
    return spec_mod.validate({
        "name": layer.name,
        "stresses": (f"{extraction.model} layer {index + 1} of "
                     f"{len(extraction.layers)}: nn.Linear({layer.k}, "
                     f"{layer.n}, bias=False) on a batch of {layer.m}{tail}"),
        "einsum": "mk,kn->mn",
        "dims": {"m": layer.m, "k": layer.k, "n": layer.n},
        "inputs": [
            {"name": "X", "subscript": "mk", "dtype": "int8", "buffer": "A",
             "origin": [0, 0]},
            {"name": "W", "subscript": "kn", "dtype": "int8", "buffer": "B",
             "origin": [0, 0]},
        ],
        "constants": [],
        "output": {"name": "Y", "subscript": "mn", "dtype": "int8",
                   "buffer": "C", "origin": [0, 0], "write_window": "exact"},
        "accumulator": "int32",
        "epilogue": epilogue,
        "operand_pad": "arbitrary",
        "known_gap": None,
        "source": {"model": extraction.model, "module": layer.module,
                   "index": index},
    })


def specs(extraction):
    return [to_spec(extraction, layer, i)
            for i, layer in enumerate(extraction.layers)]


def emit(extraction, directory=SPECS):
    os.makedirs(directory, exist_ok=True)
    written = []
    for sp in specs(extraction):
        path = os.path.join(directory, f"{sp['name']}.json")
        with open(path, "w") as f:
            json.dump(sp, f, indent=2)
            f.write("\n")
        written.append(path)
    return written


def of(name, seed=0):
    from examples.accelerator.tinytpu_vitis.workloads import models
    model, example_inputs = models.build(name, seed)
    return extract(name, model, example_inputs)


if __name__ == "__main__":
    from examples.accelerator.tinytpu_vitis.workloads import models
    for name in models.names():
        ex = of(name)
        print(f"{name}: {len(ex.layers)} mappable layers, "
              f"{len(ex.refusals)} refusals")
        for layer in ex.layers:
            print(f"    {layer.name:16s} {layer.m:3d}x{layer.k:3d}x{layer.n:3d}"
                  f"  {'relu' if layer.relu else '-'}   ({layer.module})")
        for refusal in ex.refusals:
            print(f"    REFUSED {refusal.where}: {refusal.why}")

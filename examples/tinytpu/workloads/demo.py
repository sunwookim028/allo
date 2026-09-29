#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One PyTorch model through the whole flow, a stage at a time.

    python workloads/demo.py                  mlp_small on the Allo simulator
    python workloads/demo.py mlp_deep --top 5 another model, five mappings
    python workloads/demo.py --no-sim         stop after compilation
    python workloads/demo.py --spec           print each layer's spec in full

The stages: the PyTorch source, the traced graph, the per-layer workload
spec, the ACT mapspace search that compiles each layer onto the TinyTPU, the
chosen program, the design built by `df.build(target="simulator")`, and the
model run on it layer by layer against PyTorch. Each stage is a function that
prints its section and returns what the next one needs, so a script can reuse
any of them. The output has no timings, so a captured run is reproducible.

`run.py` is the suite's report over every model; this is the walk-through.
Tutorial: docs/source/designs/tinytpu_tutorial.rst."""

import argparse
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu import isa_ref  # noqa: E402
from examples.tinytpu.act import correctness, cycles  # noqa: E402
from examples.tinytpu.disasm import disassemble  # noqa: E402
from examples.tinytpu.microarch_isa import DMA_WORDS, MAXDIM, T  # noqa: E402
from examples.tinytpu.workloads import extract, models  # noqa: E402
from examples.tinytpu.workloads.run import (  # noqa: E402
    map_layer, quantized_reference, run_on_machine,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def stage(n, title):
    print(f"\n[{n}] {title}\n{'-' * (len(title) + 4)}")


def show_model(name, model, x):
    stage(1, f"PyTorch: {name}, input {tuple(x.shape)}")
    source = inspect.getsource(type(model))
    path = os.path.relpath(inspect.getsourcefile(type(model)),
                           os.path.dirname(HERE))
    print(f"# {path}\n{source.rstrip()}")


def trace(name):
    """`AlloTracer` + `ShapeProp`: the graph, and the layers it maps to."""
    stage(2, "Trace: AlloTracer + torch.fx ShapeProp")
    extraction = extract.of(name)
    print(extraction.graph.rstrip())
    for refusal in extraction.refusals:
        print(f"  REFUSED {refusal.where}: {refusal.why}")
    return extraction


def show_specs(extraction, full):
    """One workload spec per layer, in `act/corpus/`'s JSON convention."""
    specs = extract.specs(extraction)
    stage(3, f"Workload specs: {len(specs)} layers")
    for sp in specs:
        d = sp["dims"]
        print(f"  {sp['name']:14s} {sp['einsum']}  m={d['m']} k={d['k']} "
              f"n={d['n']}  epilogue={'+'.join(sp['epilogue'])}")
        if full:
            print(json.dumps(sp, indent=1))
    return specs


def compile_layers(specs, top):
    """ACT: every loop nest of each layer, lowered to a TinyTPU program or
    refused, ranked by the cost model; the best is checked against the spec.
    Returns `[(spec, best candidate)]`, or None if a layer does not compile."""
    stage(4, "Compile with ACT: search the mapspace onto the TinyTPU")
    picked = []
    for sp in specs:
        result = map_layer(sp)
        print(f"\n  {sp['name']}: {result.considered} loop nests, "
              f"{len(result.candidates)} legal, {result.census.total} refused")
        for cause, count in result.census.rows():
            print(f"      refused {count:5d}  {cause}")
        if not result.best:
            print("  every nest refused: the layer does not compile")
            return None
        print(f"    {'rank':>4s}  {'mapping':14s} {'instrs':>6s} "
              f"{'est. cycles':>11s}")
        for i, c in enumerate(result.ranked(top)):
            print(f"    {i + 1:>4d}  {c.label:14s} {len(c.program):>6d} "
                  f"{cycles.estimate(c.program):>11.0f}")
        fails = correctness.check(sp, result.best.program)
        print(f"    picked {result.best.label}; the program against the "
              f"spec's einsum (isa_ref, 4 operand distributions): "
              f"{'BIT-EXACT' if not fails else 'WRONG: ' + fails[0]}")
        if fails:
            return None
        picked.append((sp, result.best))
    return picked


def show_program(sp, candidate):
    stage(5, f"The program ACT emitted for {sp['name']} "
             f"({len(candidate.program)} instructions)")
    print(disassemble(candidate.program))


def build_design():
    stage(6, "Build the design: df.build(tinytpu_isa, target='simulator')")
    module = correctness.build_module()
    print(f"  built: T={T} ({T}x{T} array), MAXDIM={MAXDIM}, "
          f"DMA_WORDS={DMA_WORDS}")
    return module


def run_model(model, extraction, programs, x, module):
    """Every layer's program in turn, each output the next layer's input.
    With a `module`, on the built design, each layer also checked against
    `isa_ref` over the whole of `C`; without one, on `isa_ref` alone."""
    if module is None:
        stage(7, "Run the model on isa_ref (the ISA in numpy; --no-sim)")
        return run_on_machine(model, extraction, programs, x)
    from examples.tinytpu.stress_isa import execute

    def check(layer, prog, A, B, C, out):
        differ = int((out != isa_ref.run(prog, A, B, C)).sum())
        print(f"  {layer.name:14s} {layer.m}x{layer.k}x{layer.n}   design vs "
              f"isa_ref over all {out.size} bytes of C: {differ} differ")

    stage(7, "Run the model on the design, one program per layer")
    return run_on_machine(model, extraction, programs, x,
                          execute=lambda prog, A, B, C: execute(
                              module, prog, A, B, C),
                          on_layer=check)


def compare(model, extraction, x, got):
    """The machine's output against PyTorch's own evaluation of the modules
    with the machine's epilogue: ReLU where fused, then the clip to int8."""
    want = quantized_reference(model, extraction, x)
    bad = sum(int((a != b).sum()) for a, b in zip(got, want))
    total = sum(a.size for a in want)
    stage("✓" if bad == 0 else "✗", "Against PyTorch")
    print(f"  {bad} of {total} output bytes differ")
    print(f"  torch  {want[-1][0].tolist()}\n  tpu    {got[-1][0].tolist()}")
    return bad == 0


def summary(picked):
    print("\nSummary (cycles are the cost model's estimate; "
          "`make mlp-cosim` measures)")
    print(f"  {'layer':14s} {'mapping':14s} {'instrs':>6s} {'est. cycles':>11s}")
    total = 0
    for sp, c in picked:
        est = cycles.estimate(c.program)
        total += est
        print(f"  {sp['name']:14s} {c.label:14s} {len(c.program):>6d} "
              f"{est:>11.0f}")
    print(f"  {'model':14s} {'':14s} {'':6s} {total:>11.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", default="mlp_small",
                    choices=models.names())
    ap.add_argument("--top", type=int, default=3,
                    help="mappings to list per layer (default 3)")
    ap.add_argument("--no-sim", action="store_true",
                    help="run on isa_ref instead of building the design")
    ap.add_argument("--spec", action="store_true",
                    help="print each layer's spec in full")
    args = ap.parse_args(argv)

    model, (x,) = models.build(args.model)
    show_model(args.model, model, x)
    extraction = trace(args.model)
    if extraction.refusals:
        print("\nThe ISA has no instruction for the refused nodes, so the "
              "model does not compile.")
        return 1
    specs = show_specs(extraction, args.spec)
    picked = compile_layers(specs, args.top)
    if picked is None:
        return 1
    show_program(*picked[0])
    module = None if args.no_sim else build_design()
    got = run_model(model, extraction, [c.program for _, c in picked], x,
                    module)
    exact = compare(model, extraction, x, got)
    summary(picked)
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())

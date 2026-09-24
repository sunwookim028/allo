#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The suite, end to end: a PyTorch MLP in, per-layer verified cycles out.

    python workloads/run.py                    every model, static
    python workloads/run.py --emit             write specs/*.json as well
    python workloads/run.py --simulator        also check the built design
    python workloads/run.py --cosim mlp_tiny   the RTL number, one csynth
    python workloads/run.py --burst            what the widening is worth

A model's cycle figure is a SUM OVER LAYERS. The machine runs one layer per
program, from DRAM and back to DRAM, with no inter-layer fusion and nothing
resident between layers, so the sum is the whole of what it does -- not a
simplification of a fused schedule. Prose:
docs/source/designs/workload_suite.rst."""

import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from act.target import get as machine_for  # noqa: E402
from act import workloads  # noqa: E402
from act.search import Problem, search  # noqa: E402
from examples.tinytpu import act_target  # noqa: E402,F401  -- registers tinytpu-isa
from examples.tinytpu import isa_ref  # noqa: E402
from examples.tinytpu.act import correctness, cycles  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    DMA_WORDS, MAXDIM, T,
)
from examples.tinytpu.workloads import burst, extract, models  # noqa: E402

TARGET = "tinytpu-isa"
HERE = os.path.dirname(os.path.abspath(__file__))


def map_layer(sp):
    """The ACT mapper's pick for one layer, onto the fixed TinyTPU."""
    name = "gemm.relu" if "relu" in sp["epilogue"] else "gemm"
    workload = workloads.get(name)
    dims = sp["dims"]
    extents = dict(zip(workload.ranks, (dims["m"], dims["k"], dims["n"])))
    machine = machine_for(TARGET)
    result = search(Problem(workload, extents), machine)
    return result


def quantized_reference(model, extraction, x):
    """What PyTorch says the machine should produce, layer by layer.

    The epilogue is the machine's, applied in torch: ReLU on the int32
    accumulator where it was fused, then the narrowing clip to int8. Nothing
    is re-implemented in numpy, so a disagreement is the mapping's."""
    act, outs = x, []
    with torch.no_grad():
        for layer in extraction.layers:
            acc = model.get_submodule(layer.module)(act)
            if layer.relu:
                acc = torch.relu(acc)
            act = torch.clamp(acc, -128, 127)
            outs.append(act)
    return [o.numpy().astype(np.int8) for o in outs]


def run_on_machine(model, extraction, programs, x):
    """The model executed as the machine executes it: one program per layer,
    DRAM in and DRAM out, nothing carried over between layers."""
    act = x.numpy().astype(np.int8)
    outs = []
    for layer, prog in zip(extraction.layers, programs):
        A = np.zeros((MAXDIM, MAXDIM), np.int8)
        B = np.zeros((MAXDIM, MAXDIM), np.int8)
        C = np.zeros((MAXDIM, MAXDIM), np.int8)
        A[:layer.m, :layer.k] = act
        weight = model.get_submodule(layer.module).weight.detach().numpy()
        B[:layer.k, :layer.n] = weight.T.astype(np.int8)
        out = isa_ref.run(prog, A.reshape(-1), B.reshape(-1), C.reshape(-1))
        act = out.reshape(MAXDIM, MAXDIM)[:layer.m, :layer.n].copy()
        outs.append(act)
    return outs


def compile_model(name, emit, simulator_module):
    extraction = extract.of(name)
    if emit:
        extract.emit(extraction)
    rows, programs = [], []
    for sp in extract.specs(extraction):
        result = map_layer(sp)
        if not result.best:
            rows.append((sp, None, None, "every nest refused"))
            continue
        prog = result.best.program
        fails = correctness.check(sp, prog, simulator_module)
        rows.append((sp, prog, result, fails[0] if fails else None))
        programs.append(prog)
    return extraction, rows, programs


def show_model(name, extraction, rows, programs, model, example_inputs):
    print(f"\n{name}: {len(extraction.layers)} mappable layers, "
          f"{len(extraction.refusals)} refusals")
    for refusal in extraction.refusals:
        print(f"  UNMAPPABLE {refusal.where}: {refusal.why}")
    print(f"  {'layer':16s} {'shape':>12s} {'epi':>5s} {'mapping':>14s} "
          f"{'static':>6s} {'dyn':>5s} {'critical':>16s} {'estimate':>8s} "
          f"{'burst':>6s}  isa_ref")
    total = 0.0
    for sp, prog, result, bad in rows:
        d = sp["dims"]
        shape = f"{d['m']}x{d['k']}x{d['n']}"
        epi = "relu" if "relu" in sp["epilogue"] else "-"
        if prog is None:
            print(f"  {sp['name']:16s} {shape:>12s} {epi:>5s}  {bad}")
            continue
        work = cycles.work(prog)
        est = cycles.estimate(prog)
        total += est
        widest = burst.predict(prog, _widest())
        print(f"  {sp['name']:16s} {shape:>12s} {epi:>5s} "
              f"{result.best.label:>14s} {work['static']:>6d} "
              f"{work['dynamic']:>5d} "
              f"{cycles.critical_unit(prog) + ' ' + str(cycles.critical_work(prog)):>16s} "
              f"{est:>8.0f} {'-' + str(widest):>6s}  "
              f"{'BIT-EXACT' if bad is None else 'WRONG: ' + bad}")
    print(f"  model total {total:.0f} cycles, summed over "
          f"{len(programs)} layers with no fusion and no residency")
    if len(programs) == len(extraction.layers) and programs:
        want = quantized_reference(model, extraction, example_inputs[0])
        got = run_on_machine(model, extraction, programs, example_inputs[0])
        bad = sum(int((a != b).sum()) for a, b in zip(got, want))
        print(f"  end to end: the chained programs reproduce PyTorch's own "
              f"int8 epilogue chain, {bad} byte{'' if bad == 1 else 's'} "
              f"differ over {sum(a.size for a in want)}")
    return total


def _widest():
    from examples.tinytpu.ip.params import TpuParams
    return TpuParams.widest_burst(T, MAXDIM)


def cmd_static(args):
    module = correctness.build_module() if args.simulator else None
    chosen = args.models or models.MAPPABLE
    print(f"TinyTPU build: T={T} MAXDIM={MAXDIM} DMA_WORDS={DMA_WORDS}; "
          f"the burst column is the widening's predicted saving at "
          f"DMA_WORDS={_widest()}")
    grand = 0.0
    for name in chosen:
        model, example_inputs = models.build(name)
        extraction, rows, programs = compile_model(name, args.emit, module)
        grand += show_model(name, extraction, rows, programs, model,
                            example_inputs)
    print(f"\nsuite total {grand:.0f} estimated cycles over "
          f"{len(chosen)} models")
    print("An estimate is not a measurement: cycles.estimate is fitted to the "
          "published\nMAXDIM=16 points and its per-unit work counts cannot "
          "see the DMA burst at all,\nwhich is exactly the term the widening "
          "moves. Use --cosim for a real number.")
    return 0


def cmd_cosim(args):
    """One csynth of this build, then one bounded cosim per layer.

    Every model named shares the synthesis, so the RTL under test is identical
    across layers and across models, which is what makes a sum over layers a
    figure for the model rather than a sum of different machines."""
    from examples.tinytpu.act import measure
    chosen = args.models or ["mlp_tiny"]
    prj = os.path.abspath(args.project or os.path.join(HERE, "workload.prj"))
    work = []
    for name in chosen:
        _, rows, _ = compile_model(name, args.emit, None)
        if any(prog is None for _, prog, _, _ in rows):
            raise SystemExit(f"{name}: a layer did not map; nothing to measure")
        work.append((name, rows))
    print(f"synthesizing once into {prj} at DMA_WORDS={DMA_WORDS} ...",
          flush=True)
    measure.synthesize(prj)
    measured, complete = {}, True
    try:
        for name, rows in work:
            print(f"\n{name}  (DMA_WORDS={DMA_WORDS})")
            print(f"  {'layer':16s} {'estimate':>9s} {'cosim':>7s} "
                  f"{'error':>8s}   testbench")
            total = 0
            for sp, prog, _, _ in rows:
                est = cycles.estimate(prog)
                n, line = measure.measure(sp, prog, prj, timeout=args.timeout)
                complete = complete and n is not None
                err = "" if not n else f"{(est - n) / n * 100:+7.1f}%"
                total += n or 0
                measured[sp["name"]] = n
                print(f"  {sp['name']:16s} {est:9.0f} {str(n):>7s} {err:>8s}"
                      f"   {line}", flush=True)
            measured[name] = total
            print(f"  {name} on RTL: {total} cycles over {len(rows)} layers, "
                  f"summed with no fusion and no residency")
    finally:
        shutil.rmtree(prj, ignore_errors=True)
    if not complete:
        print("  A layer that did not finish is limitations item 24, and "
              "Kt >= QD deadlocks:\n  re-run it with TPU_QD=16 before "
              "concluding anything from the hang.")
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump({"dma_words": DMA_WORDS, "cycles": measured}, f, indent=2)
    return 0 if complete else 1


def cmd_burst(args):
    print(f"the widening law against the two shapes the grid measured, at "
          f"DMA_WORDS={_widest()}")
    print(f"  {'shape':>12s} {'measured':>9s} {'predicted':>10s} "
          f"{'if every iteration counted':>28s}")
    for shape, measured, pred, every in burst.check_measured(_widest()):
        print(f"  {'x'.join(str(s) for s in shape):>12s} {measured:>9d} "
              f"{pred:>10d} {every:>28d}")
    for name in args.models or models.MAPPABLE:
        model, example_inputs = models.build(name)
        extraction, rows, programs = compile_model(name, False, None)
        print(f"\n{name}")
        print(f"  {'layer':16s} {'A,B spans':>12s} {'iters@1':>13s} "
              f"{'iters@w':>13s} {'predicted saving':>17s}")
        total = 0
        for sp, prog, _, _ in rows:
            before, after = (burst.iterations(prog, 1),
                             burst.iterations(prog, _widest()))
            saving = burst.predict(prog, _widest())
            total += saving
            print(f"  {sp['name']:16s} {str(burst.spans(prog)):>12s} "
                  f"{str(before):>13s} {str(after):>13s} {saving:>17d}")
        print(f"  {name} predicted saving {total} cycles over "
              f"{len(rows)} layers")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--simulator", action="store_true")
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--project")
    ap.add_argument("--json")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    if args.cosim:
        return cmd_cosim(args)
    if args.burst:
        return cmd_burst(args)
    return cmd_static(args)


if __name__ == "__main__":
    sys.exit(main())

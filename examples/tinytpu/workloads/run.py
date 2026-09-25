#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The suite, end to end: a PyTorch MLP in, per-layer verified cycles out.

    python workloads/run.py                    every model, static
    python workloads/run.py --emit             write specs/*.json as well
    python workloads/run.py --simulator        also check the built design
    python workloads/run.py --cosim mlp_tiny   the RTL number, one csynth
    python workloads/run.py --burst            what the widening is worth
    python workloads/run.py --verify           the PyTorch oracle, as a GATE

`--verify` is the one check in this repository with something OUTSIDE it on
one side: `torch`'s own `nn.Linear` forward, on the model's real weights, with
the machine's epilogue applied in torch. Everything else compares two things
this repository wrote -- `isa_ref` against `act.spec`'s `gold`, both ours -- and
that is not evidence when the ISA itself is under search, because a candidate
authors its own reference model. `chia_agent/evaluate.py` runs this as a gate
before anything expensive, and a byte that differs is a refusal.

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
from allo.act.target import get as machine_for  # noqa: E402
from allo.act import workloads  # noqa: E402
from allo.act.search import Problem, search  # noqa: E402
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


class Unrepresentable(Exception):
    """This graph is not a sequence of layers fed one from another.

    Raised rather than approximated. Both sides of the end-to-end comparison
    below read the SAME dataflow off the fx graph, so a graph they cannot
    represent -- a layer whose activation is neither the model input nor an
    earlier mapped layer's result -- has to stop the check instead of being
    chained anyway. `gate.py` and `--verify` turn it into a refusal.
    """


def mapped_nodes(gm, extraction):
    """The graph node behind each extracted layer, in the extractor's order.

    Matched greedily by module target down the graph, not by looking the
    target up: a module called twice is two nodes and two layers, and a lookup
    would return the first one both times. `workloads/scope.py` imports this
    one rather than keeping a second copy.
    """
    pending = [l.module for l in extraction.layers]
    out, at = [], 0
    for node in gm.graph.nodes:
        if at >= len(pending):
            break
        if node.op == "call_module" and node.target == pending[at]:
            out.append(node)
            at += 1
    return out + [None] * (len(pending) - len(out))


def layer_sources(gm, extraction):
    """Where each mapped layer's activation comes from, READ OFF THE GRAPH.

    `-1` is the model input; `j` is mapped layer `j`'s int8 result. This is
    the fix for a measured bug, and the bug is worth stating because it is the
    shape of bug this whole suite exists to catch. Both `quantized_reference`
    and `run_on_machine` used to chain layer *i* into *i+1* and never look at
    the graph. On `forward(x) -> fc1(x), fc2(x)` -- a fan-out every node of
    which the extractor maps and refuses nothing in -- the two chains agreed on
    128 of 128 bytes while the machine differed from what `model(x)` actually
    returns in 62 of 128. They agreed because they were wrong the SAME way.
    Every committed model is a chain, so no published number moves; what moves
    is that the check is now evidence for graphs that are not chains, which is
    what makes it safe to gate on (`workloads/scope.py`,
    `assumptions/chained_verification`).
    """
    nodes = mapped_nodes(gm, extraction)
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    #: A fused ReLU is the same value as its Linear as far as the machine is
    #: concerned -- the epilogue rides on the same program -- so a consumer of
    #: the ReLU is a consumer of that layer.
    produces = {}
    for i, node in enumerate(nodes):
        if node is None:
            raise Unrepresentable(f"no graph node for module "
                                  f"{extraction.layers[i].module}")
        produces[node] = i
        if extraction.layers[i].relu:
            for user in node.users:
                if extract.is_relu(gm, user):
                    produces[user] = i
    out = []
    for i, node in enumerate(nodes):
        source = node.args[0] if node.args else None
        if source is not None and getattr(source, "op", None) == "placeholder":
            if placeholders and source is not placeholders[0]:
                raise Unrepresentable(
                    f"layer {extraction.layers[i].name} consumes "
                    f"{source.name!r}, a second model input; the machine is "
                    f"handed one activation")
            out.append(-1)
            continue
        j = produces.get(source)
        if j is None or j >= i:
            raise Unrepresentable(
                f"layer {extraction.layers[i].name} consumes "
                f"{getattr(source, 'name', source)!r}, which is neither the "
                f"model input nor an earlier mapped layer's result")
        out.append(j)
    return out


def _graph_of(model, extraction, x, gm=None):
    return gm if gm is not None else extract.trace(model, (x,))


def quantized_reference(model, extraction, x, gm=None):
    """What PyTorch says the machine should produce, layer by layer.

    The epilogue is the machine's, applied in torch: ReLU on the int32
    accumulator where it was fused, then the narrowing clip to int8. Nothing
    is re-implemented in numpy, so a disagreement is the mapping's.

    The dataflow is the GRAPH's (`layer_sources`), not the layer order: a
    layer that consumes the model input is given the model input.
    """
    gm = _graph_of(model, extraction, x, gm)
    sources = layer_sources(gm, extraction)
    done, outs = {}, []
    with torch.no_grad():
        for i, layer in enumerate(extraction.layers):
            act = x if sources[i] < 0 else done[sources[i]]
            acc = model.get_submodule(layer.module)(act)
            if layer.relu:
                acc = torch.relu(acc)
            done[i] = torch.clamp(acc, -128, 127)
            outs.append(done[i])
    return [o.numpy().astype(np.int8) for o in outs]


def run_on_machine(model, extraction, programs, x, gm=None):
    """The model executed as the machine executes it: one program per layer,
    DRAM in and DRAM out, nothing carried over between layers.

    Which activation each program is handed is the graph's answer, the same
    `layer_sources` the reference above uses."""
    gm = _graph_of(model, extraction, x, gm)
    sources = layer_sources(gm, extraction)
    first = x.numpy().astype(np.int8)
    done, outs = {}, []
    for i, (layer, prog) in enumerate(zip(extraction.layers, programs)):
        act = first if sources[i] < 0 else done[sources[i]]
        A = np.zeros((MAXDIM, MAXDIM), np.int8)
        B = np.zeros((MAXDIM, MAXDIM), np.int8)
        C = np.zeros((MAXDIM, MAXDIM), np.int8)
        A[:layer.m, :layer.k] = act
        weight = model.get_submodule(layer.module).weight.detach().numpy()
        B[:layer.k, :layer.n] = weight.T.astype(np.int8)
        out = isa_ref.run(prog, A.reshape(-1), B.reshape(-1), C.reshape(-1))
        done[i] = out.reshape(MAXDIM, MAXDIM)[:layer.m, :layer.n].copy()
        outs.append(done[i])
    return outs


def vs_pytorch(name, seed=0, programs=None):
    """One model against torch's own forward: (bytes compared, bytes differing).

    Maps the layers when `programs` is not supplied. The mapping depends only
    on the SHAPES, so a caller sweeping seeds maps once and pays only the
    reference run per draw -- which is what makes a wider corpus cheap.
    """
    model, xs = models.build(name, seed)
    extraction = extract.of(name, seed)
    if programs is None:
        programs = []
        for sp in extract.specs(extraction):
            result = map_layer(sp)
            if not result.best:
                raise Unrepresentable(f"{sp['name']}: every nest was refused")
            programs.append(result.best.program)
    want = quantized_reference(model, extraction, xs[0])
    got = run_on_machine(model, extraction, programs, xs[0])
    return (sum(int(a.size) for a in want),
            sum(int((a != b).sum()) for a, b in zip(got, want)), programs)


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
        try:
            want = quantized_reference(model, extraction, example_inputs[0])
            got = run_on_machine(model, extraction, programs, example_inputs[0])
            bad = sum(int((a != b).sum()) for a, b in zip(got, want))
            print(f"  end to end: the programs reproduce PyTorch's own int8 "
                  f"epilogue on the graph's own dataflow, "
                  f"{bad} byte{'' if bad == 1 else 's'} differ over "
                  f"{sum(a.size for a in want)}")
        except Unrepresentable as why:
            print(f"  end to end: NOT CHECKED -- {why}")
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
    work, skipped = [], {}
    for name in chosen:
        _, rows, _ = compile_model(name, args.emit, None)
        bad = [sp["name"] for sp, prog, _, _ in rows if prog is None]
        if bad:
            # Out of scope at this configuration, not a failure: at T=8 the
            # mapper refuses a 16x16x12 layer because 12 is not a multiple of
            # T. Recorded by name so the caller knows the term is short a
            # model rather than guessing from its absence.
            skipped[name] = f"{bad[0]}: every nest was refused"
            print(f"  SKIPPING {name}: {skipped[name]}", flush=True)
            continue
        work.append((name, rows))
    if not work:
        raise SystemExit(f"no model of {chosen} maps at this configuration; "
                         f"nothing to measure")
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
            json.dump({"dma_words": DMA_WORDS, "cycles": measured,
                       "skipped": skipped}, f, indent=2)
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


#: The floor on `--verify`'s corpus, in bytes compared against torch.
#:
#: A floor and not a list of required models, because WHICH models are in scope
#: is a function of the build and of the frozen mapper, not of the candidate:
#: `mlp_small` needs MAXDIM >= 32 and `mlp_wide` 64, and at T=8 the mapper
#: refuses `mlp_deep`'s 16x16x12 layer outright because 12 is not a multiple of
#: T (`workloads/scope_map.json` records that boundary as
#: `refused-without-a-cause`). Requiring `mlp_deep` refused every T=8 candidate
#: for something the candidate had no part in.
#:
#: What is NOT relaxed: a model whose layers all map must be bit-exact, and the
#: corpus must be at least this many bytes, so the gate cannot quietly shrink
#: to nothing. At the published row it is 2,688 bytes, at T=8 MAXDIM=32 it is
#: 4,096, and the run prints which models it had and why it skipped the rest.
VERIFY_MIN_BYTES = 1024
#: Input/weight draws per model. The mapping depends only on the shapes, so
#: extra draws cost one `isa_ref` run and one torch forward each and multiply
#: the bytes compared by eight. They widen the VALUES, not the shapes: this is
#: a deeper corpus, not a broader one, and the shape coverage is still whatever
#: the suite's four MLPs contain.
VERIFY_SEEDS = 8


def cmd_verify(args):
    """The PyTorch oracle as a gate: every mappable model, byte for byte.

    The only check in the system with something outside this repository on one
    side. Returns 0 only if every draw of every model in the corpus is exact
    and every REQUIRED model mapped.
    """
    chosen = args.models or list(models.MAPPABLE)
    print(f"verifying against torch at T={T} MAXDIM={MAXDIM} "
          f"DMA_WORDS={DMA_WORDS}: {VERIFY_SEEDS} draw(s) of "
          f"{len(chosen)} candidate model(s)")
    print(f"  {'model':12s} {'layers':>6s} {'draws':>6s} {'bytes':>8s} "
          f"{'differ':>7s}   verdict")
    fails, total_bytes, verified = [], 0, []
    for name in chosen:
        extraction = extract.of(name)
        if extraction.refusals or not extraction.layers:
            why = (extraction.refusals[0].why if extraction.refusals
                   else "no mappable layer")
            # The extractor refused something -- a shape that does not fit
            # this MAXDIM, or an op the machine has no unit for. Out of scope
            # at this configuration; the byte floor below is what keeps the
            # corpus from shrinking to nothing.
            print(f"  {name:12s} {'--':>6s} {'--':>6s} {'--':>8s} {'--':>7s}   "
                  f"not in scope here: {why}")
            continue
        programs, bad, nbytes = None, 0, 0
        try:
            for seed in range(VERIFY_SEEDS):
                n, differ, programs = vs_pytorch(name, seed, programs)
                nbytes += n
                bad += differ
        except Unrepresentable as why:
            # A layer the MAPPER refuses is out of scope at this
            # configuration, the same as a layer that does not fit MAXDIM. A
            # graph the end-to-end check cannot represent is not: that is
            # `layer_sources` refusing, and it is a failure.
            scope = "every nest was refused" in str(why)
            print(f"  {name:12s} {len(extraction.layers):>6d} {'--':>6s} "
                  f"{'--':>8s} {'--':>7s}   "
                  f"{'not mappable here' if scope else 'UNREPRESENTABLE'}: {why}")
            if not scope:
                fails.append(f"{name}: {why}")
            continue
        total_bytes += nbytes
        verified.append(name)
        print(f"  {name:12s} {len(extraction.layers):>6d} {VERIFY_SEEDS:>6d} "
              f"{nbytes:>8d} {bad:>7d}   "
              f"{'BIT-EXACT against torch' if not bad else 'WRONG'}")
        if bad:
            fails.append(f"{name}: {bad} of {nbytes} bytes differ from "
                         f"torch's own forward")
    if total_bytes < VERIFY_MIN_BYTES:
        fails.append(f"the corpus is {total_bytes} bytes, under the "
                     f"{VERIFY_MIN_BYTES}-byte floor: at this configuration "
                     f"too little of the suite is in scope for this gate to be "
                     f"evidence of anything")
    for line in fails:
        print(f"  VERIFY FAIL {line}")
    print(f"  VERIFY {'OK' if not fails else 'FAILED'}: "
          f"{len(verified)}/{len(chosen)} model(s) bit-exact against torch "
          f"over {total_bytes} bytes ({verified})")
    if not fails:
        print(f"  The corpus is what it is: {len(verified)} of the suite's "
              f"{len(models.MAPPABLE)} MLPs at MAXDIM={MAXDIM}, "
              f"{VERIFY_SEEDS} draws each,\n  {total_bytes} bytes, and no shape "
              f"the GEMM sweep does not already cover. What it adds is an "
              f"ORACLE\n  OUTSIDE this repository, not breadth: the seeds "
              f"deepen the values, not the shapes.")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--simulator", action="store_true")
    ap.add_argument("--cosim", action="store_true")
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="the PyTorch oracle as a gate (no Vitis, no RTL)")
    ap.add_argument("--project")
    ap.add_argument("--json")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)
    if args.verify:
        return cmd_verify(args)
    if args.cosim:
        return cmd_cosim(args)
    if args.burst:
        return cmd_burst(args)
    return cmd_static(args)


if __name__ == "__main__":
    sys.exit(main())

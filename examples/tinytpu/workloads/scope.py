#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The front half's scope, MEASURED: which PyTorch maps, which shapes map,
and what each suite entry's bit-exactness is evidence *of*.

    python examples/tinytpu/workloads/scope.py            # print the map (~50 s)
    python examples/tinytpu/workloads/scope.py --emit     # rewrite scope_map.json
    python examples/tinytpu/workloads/scope.py --check    # diff against it

Why it exists. ``gate.py`` says five models pass; it does not say what the
front half would do with a sixth. This walks the boundary instead of the
suite: every op two reference graphs (a transformer block and a CNN) emit,
every shape on each axis across the ceilings, and -- per suite entry -- which
of its checks compares against **PyTorch** and which compares two things this
repository wrote.

**There is no hand-maintained expected-value list.** ``scope_map.json`` is
this program's own output, committed so a later run can diff it; ``--check``
regenerates and compares. A cell can only drift by the measurement changing.

Three statuses are load-bearing and the rest are bookkeeping:

``mapped``
    the mapper produced a program AND ``act.correctness.check`` found it
    bit-exact against ``spec.gold`` on four operand distributions.
``silent-wrong``
    a program came out and it does not compute the spec. **A bug**, not a
    scope boundary: report it, do not tune around it.
``refused-without-a-cause``
    the mapper produced nothing and named no constraint, because the mapspace
    generated no nest at all. The shape is out of scope and the user is not
    told why. Distinguished from ``refused``, which names its cause.

Every cell is run, never inferred. A shape too slow for this program to
carry is absent rather than guessed: the search is exponential in the nest
(a 48x64x64 GEMM took 52 s to enumerate on 2026-09-24, against 0.9 s for
64x16x16), so the map probes axes and corners and leaves the interior to
``dev/records/2026-09-24-front-half-scope.rst``.
"""

import argparse
import json
import os
import re
import sys


def _repo_root(start=None):
    d = os.path.dirname(os.path.abspath(start or __file__))
    while True:
        if (os.path.exists(os.path.join(d, "pyproject.toml"))
                and os.path.isdir(os.path.join(d, "allo"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(os.path.abspath(start or __file__))
        d = parent


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _repo_root()
MAP = os.path.join(HERE, "scope_map.json")
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------- graphs ---
def reference_graphs():
    """Two graphs chosen for the ops they emit, not for mapping well.

    A pre-norm transformer block and a small CNN: between them they emit
    LayerNorm, softmax, two free matmuls, a transpose, residual adds, a
    scalar divide, Conv2d, BatchNorm2d, MaxPool2d and a flatten. Seeded, so
    the graph -- and therefore every row below -- is the same every run.
    """
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            for name in ("q", "k", "v", "o"):
                setattr(self, name, nn.Linear(d, d, bias=False))
            self.f1 = nn.Linear(d, 4 * d, bias=False)
            self.f2 = nn.Linear(4 * d, d, bias=False)

        def forward(self, x):
            h = self.ln(x)
            a = torch.softmax(self.q(h) @ self.k(h).transpose(-1, -2) / 4.0,
                              dim=-1)
            x = x + self.o(a @ self.v(h))
            return x + self.f2(torch.relu(self.f1(self.ln(x))))

    class Cnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 4, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(4)
            self.p = nn.MaxPool2d(2)
            self.fc = nn.Linear(4 * 4 * 4, 8, bias=False)

        def forward(self, x):
            return self.fc(self.p(torch.relu(self.bn(self.c1(x)))).flatten(1))

    torch.manual_seed(0)
    return {"transformer_block": (Block().eval(), (torch.randn(8, 16),)),
            "cnn": (Cnn().eval(), (torch.randn(4, 1, 8, 8),))}


def stable(text):
    """Python prints a builtin method as `<... at 0x7f...>`, and the address
    is different every process. A map whose cells move because the loader
    moved cannot be diffed, so the address is normalised out -- here, in the
    reader, rather than by editing the message the extractor produces."""
    return re.sub(r"0x[0-9a-f]{6,}", "0x...", text)


def node_key(gm, node):
    """A stable name for the KIND of op, so two `add`s are one row."""
    if node.op == "call_module":
        return f"nn.{type(gm.get_submodule(node.target)).__name__}"
    if node.op == "call_method":
        return f"Tensor.{node.target}"
    target = node.target
    name = getattr(target, "__name__", None) or stable(str(target))
    return f"{node.op}:{name}"


def linear_key(gm, node):
    mod = gm.get_submodule(node.target)
    return ("nn.Linear(bias=True)" if mod.bias is not None
            else "nn.Linear(bias=False)")


_MAPPED = {}


def mapping(sp):
    """``runner.map_layer`` memoised on the shape and epilogue.

    The same (M, K, N) is reached from the op table, the shape sweep and the
    entry table, and enumerating a 64x64x64 nest costs ~20 s. The mapper is a
    pure function of the workload name and the extents, so one call answers
    all three.
    """
    key = (sp["dims"]["m"], sp["dims"]["k"], sp["dims"]["n"],
           "relu" in sp["epilogue"])
    if key not in _MAPPED:
        from examples.tinytpu.workloads import run as runner
        _MAPPED[key] = runner.map_layer(sp)
    return _MAPPED[key]


# ------------------------------------------------------------------ ops ---
def op_rows():
    """One row per op KIND, over the suite's models and the two reference
    graphs. The status is what the whole front half did with it, so an op the
    extractor accepts and the mapper then refuses is its own status."""
    from examples.tinytpu.workloads import extract, models

    cases = {n: models.build(n) for n in models.names()}
    cases.update(reference_graphs())
    rows = {}

    def note(key, status, why, where):
        row = rows.setdefault(key, {"status": status, "why": why,
                                    "counts": {}, "seen_in": []})
        if where not in row["seen_in"]:
            row["seen_in"].append(where)
        row["counts"][status] = row["counts"].get(status, 0) + 1
        # A worse verdict wins -- one graph mapping it does not make it
        # mapped -- but `counts` keeps how often each happened, because
        # "nn.Linear is refused" and "one nn.Linear in seven graphs is
        # refused" are different facts and the worst-case status alone
        # cannot tell them apart. At T=8 this row is the difference.
        rank = {"maps": 0, "fused-epilogue": 0, "refused": 1,
                "refused-without-a-cause": 2, "silent-wrong": 3}
        if rank[status] > rank[row["status"]]:
            row["status"], row["why"] = status, why

    for name, (model, xs) in cases.items():
        ex = extract.extract(name, model, xs)
        gm = extract.trace(model, xs)
        by_name = {n.name: n for n in gm.graph.nodes}
        for r in ex.refusals:
            node = by_name[r.where]
            key = (linear_key(gm, node) if extract.is_linear(gm, node)
                   else node_key(gm, node))
            note(key, "refused", stable(r.why), name)
        for layer, sp in zip(ex.layers, extract.specs(ex)):
            key = "nn.Linear(bias=False)"
            result = mapping(sp)
            if not result.best:
                causes = result.census.rows()
                note(key, "refused-without-a-cause" if not causes else "refused",
                     (f"the mapper enumerated {result.considered} nests and "
                      f"refused every one: "
                      + "; ".join(f"{c} x{k}" for c, k in causes)) if causes
                     else ("the mapper enumerated NO nest and named no "
                           "constraint; the extractor had accepted the layer"),
                     name)
                continue
            note(key, "maps", "one nn.Linear becomes one GEMM program", name)
            if layer.relu:
                note("relu (fused on a Linear)", "fused-epilogue",
                     "the sole consumer of a Linear becomes the vrelu "
                     "epilogue rather than a second pass", name)
    return rows


# ---------------------------------------------------------------- shapes ---
def spec_for(m, k, n, relu=False):
    from examples.tinytpu.act import spec as spec_mod
    return spec_mod.validate({
        "name": f"scope_{m}x{k}x{n}",
        "stresses": "a swept shape, from workloads/scope.py",
        "einsum": "mk,kn->mn", "dims": {"m": m, "k": k, "n": n},
        "inputs": [
            {"name": "X", "subscript": "mk", "dtype": "int8", "buffer": "A",
             "origin": [0, 0]},
            {"name": "W", "subscript": "kn", "dtype": "int8", "buffer": "B",
             "origin": [0, 0]}],
        "constants": [],
        "output": {"name": "Y", "subscript": "mn", "dtype": "int8",
                   "buffer": "C", "origin": [0, 0], "write_window": "exact"},
        "accumulator": "int32",
        "epilogue": ["relu", "saturate"] if relu else ["saturate"],
        "operand_pad": "arbitrary", "known_gap": None})


def shape_cell(m, k, n, relu=False):
    """One swept shape, MAPPED AND VERIFIED, or the reason it is not."""
    from examples.tinytpu.act import correctness
    try:
        sp = spec_for(m, k, n, relu)
    except Exception as exc:                      # noqa: BLE001
        return {"status": "spec-rejected",
                "why": stable(f"{type(exc).__name__}: {exc}")}
    try:
        result = mapping(sp)
    except Exception as exc:                      # noqa: BLE001
        return {"status": "mapper-raised",
                "why": stable(f"{type(exc).__name__}: {exc}")}
    if not result.best:
        causes = result.census.rows()
        if not causes:
            return {"status": "refused-without-a-cause", "nests": 0,
                    "why": "the mapspace generated no nest and no constraint "
                           "was named"}
        top = causes[0][0]
        return {"status": "refused", "nests": result.considered,
                "why": top, "detail": stable(result.census.examples[top][1])}
    fails = correctness.check(sp, result.best.program, None)
    if fails:
        return {"status": "silent-wrong", "label": result.best.label,
                "why": fails[0]}
    return {"status": "mapped", "label": result.best.label,
            "nests": len(result.candidates), "static": len(result.best.program)}


def shape_rows():
    """Each axis swept independently past its ceiling, plus corners.

    Cheap by construction: the nest enumeration is exponential, so the other
    two extents are held at 16 while one moves. The interior of the box is
    measured once in dev/records/, not on every run.
    """
    from examples.tinytpu.microarch_isa import MAXDIM, T
    cells, seen = {}, set()

    def add(m, k, n, relu=False):
        key = f"{m}x{k}x{n}" + (".relu" if relu else "")
        if key in seen:
            return
        seen.add(key)
        cells[key] = shape_cell(m, k, n, relu)

    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, MAXDIM - 1, MAXDIM, MAXDIM + 1):
        add(m, 16, 16)
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, MAXDIM - T, MAXDIM,
              MAXDIM + T):
        add(4, k, 16)
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, MAXDIM - T, MAXDIM,
              MAXDIM + T):
        add(4, 16, n)
    for corner in ((4, 4, 4), (4, MAXDIM, 16), (4, 16, MAXDIM), (8, 32, 32),
                   (16, 16, 16), (5, 12, 20), (MAXDIM, 4, 4)):
        add(*corner)
    add(4, 16, 16, relu=True)
    add(16, 16, 16, relu=True)
    return cells


# -------------------------------------------------------------- evidence ---
def dataflow(gm, extraction):
    """How ``run.py`` resolves each mapped layer's activation, and whether the
    graph is one chain.

    -> ``(sources, chain, why)``. ``sources`` is ``run.layer_sources``'s own
    answer -- ``-1`` for the model input, ``j`` for mapped layer ``j`` -- so
    this row records the dataflow the END-TO-END CHECK ACTUALLY USED rather
    than a second opinion about it. ``chain`` says whether that dataflow
    happens to be the straight chain ``i -> i+1``; it is now a DESCRIPTION,
    not a soundness condition. It used to be one: both sides of the comparison
    chained the layers regardless of the graph, so a fan-out was compared
    against a PyTorch evaluation that was not the model either and the two
    agreed. ``run.layer_sources`` reads the graph now, and a graph it cannot
    represent raises instead of being chained.
    """
    from examples.tinytpu.workloads import run as runner
    try:
        sources = runner.layer_sources(gm, extraction)
    except runner.Unrepresentable as why:
        return None, False, str(why)
    chain = sources == [i - 1 for i in range(len(sources))]
    named = [("input" if j < 0 else extraction.layers[j].name) for j in sources]
    return sources, chain, ("the mapped Linears form one chain from the model "
                            "input" if chain else
                            f"not a chain: each layer's activation is {named}, "
                            f"and run.py feeds it exactly that")


def entry_rows():
    """Per suite entry: its tier, and what each of its checks is evidence OF.

    The distinction the owner asked for, computed rather than declared:
    ``vs_spec`` holds two things this repository wrote against each other;
    ``vs_pytorch`` is the only check with PyTorch on one side, and since
    ``run.layer_sources`` reads the graph it is evidence for a graph of any
    shape -- ``dataflow`` records which shape it was.
    """
    from examples.tinytpu.act import correctness
    from examples.tinytpu.workloads import extract, gate, models, run as runner

    claims = json.load(open(os.path.join(HERE, "claims.json")))["models"]
    out = {}
    for name in models.names():
        entry = claims.get(name, {})
        tier = entry.get("tier")
        ex = extract.of(name)
        gm = extract.trace(*models.build(name))
        row = {"tier": tier,
               "layers": len(ex.layers),
               "refusals": [r.where for r in ex.refusals],
               "vs_spec": None, "vs_pytorch": None, "rtl": None}
        programs, spec_bytes = [], 0
        for sp in extract.specs(ex):
            result = mapping(sp)
            if not result.best:
                row["vs_spec"] = {"ok": False, "why": "a layer did not map"}
                break
            fails = correctness.check(sp, result.best.program, None)
            if fails:
                row["vs_spec"] = {"ok": False, "why": stable(fails[0][:160])}
                break
            programs.append(result.best.program)
            spec_bytes += sp["dims"]["m"] * sp["dims"]["n"]
        else:
            row["vs_spec"] = {
                "ok": True, "result_bytes": spec_bytes,
                "reference": "spec.gold, an int64 numpy einsum in this "
                             "repository, against isa_ref, a numpy model of "
                             "the ISA in this repository",
                "evidence_for": "the mapper agrees with our own semantics; "
                                "PyTorch is not on either side"}
        if row["vs_spec"]["ok"] and tier != "probe":
            _, chain, why = dataflow(gm, ex)
            model, xs = models.build(name)
            want = runner.quantized_reference(model, ex, xs[0])
            got = runner.run_on_machine(model, ex, programs, xs[0])
            bad = sum(int((a != b).sum()) for a, b in zip(got, want))
            row["vs_pytorch"] = {
                "ok": bad == 0, "bytes": sum(int(a.size) for a in want),
                "differ": bad, "chain": chain, "dataflow": why,
                "reference": "torch's own nn.Linear forward on the model's "
                             "real weights, with the machine's epilogue "
                             "applied in torch",
                "evidence_for": "the machine computes what this model's "
                                "PyTorch modules compute, on the dataflow the "
                                "fx graph declares"}
        elif tier == "probe":
            row["vs_pytorch"] = {
                "ok": None,
                "evidence_for": "not run: gate.py skips the end-to-end check "
                                "for a probe, so no PyTorch comparison exists "
                                "for this entry at all"}
        run, reasons = gate.pick_measurement(entry, gate.live_config())
        row["rtl"] = {
            "declared_for_this_config": run is not None,
            "total": (run or {}).get("total"),
            "evidence_for": "a TRANSCRIBED record that xsim once printed these "
                            "cycles at this configuration; gate.py checks it "
                            "is self-consistent and config-matched, and does "
                            "not re-measure it",
            "why_not": reasons if run is None else None}
        out[name] = row
    return out


# ----------------------------------------------------------- assumptions ---
def assumption_rows():
    """What the suite's own verification ASSUMES, probed rather than trusted.

    **FIXED, and this row is the regression test.** ``run.py`` used to compare
    two chains -- each layer's program fed the previous layer's int8 output,
    and PyTorch's ``nn.Linear`` fed the same -- with neither side reading the
    fx graph's dataflow. For a chain of Linears that is the model; for a
    fan-out it is not, and both sides were wrong the same way, so they agreed:
    this probe measured 0 of 128 bytes differing against ``run.py``'s own
    reference while the machine differed from ``model(x)`` in **62 of 128**.
    Both sides now take the dataflow from ``run.layer_sources``, so the two
    numbers are the same number, and a graph that cannot be represented raises
    rather than being chained anyway.

    The probe stays because the row is a MEASUREMENT, not an assertion: it is
    what would move if the harness regressed. The gate that depends on it is
    ``run.py --verify``, which ``chia_agent/evaluate.py`` runs on every
    candidate -- and it is load-bearing now that the ISA is editable, because
    it is the only comparison with something outside this repository on one
    side.
    """
    import numpy as np
    import torch
    from torch import nn
    from examples.tinytpu.workloads import extract, run as runner

    class Parallel(nn.Module):
        """Two bias-free Linears off one input, returned as a pair. Every
        node is something the extractor maps, and the graph is a fan-out."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16, bias=False)
            self.fc2 = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc1(x), self.fc2(x)

    torch.manual_seed(0)
    model = Parallel().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randint(-8, 8, p.shape).to(p.dtype))
    xs = (torch.randint(-8, 8, (4, 16)).float(),)

    ex = extract.extract("fanout_probe", model, xs)
    gm = extract.trace(model, xs)
    sources, chain, why = dataflow(gm, ex)
    programs = []
    for sp in extract.specs(ex):
        result = mapping(sp)
        if not result.best:
            return {"chained_verification": {"status": "not-measured",
                                             "why": "a probe layer did not map"}}
        programs.append(result.best.program)
    want = runner.quantized_reference(model, ex, xs[0])
    got = runner.run_on_machine(model, ex, programs, xs[0])
    agreed = sum(int((a != b).sum()) for a, b in zip(got, want))
    with torch.no_grad():
        truth = [torch.clamp(t, -128, 127).numpy().astype(np.int8)
                 for t in model(xs[0])]
    real = sum(int((a != b).sum()) for a, b in zip(got, truth))
    total = sum(int(a.size) for a in truth)
    return {"chained_verification": {
        "probe": "two bias-free nn.Linears off one input, returned as a pair",
        "extractor_refusals": len(ex.refusals),
        "layers_mapped": len(programs),
        "chain": chain, "dataflow": why, "sources": sources,
        "bytes": total,
        "differ_vs_run_py_reference": agreed,
        "differ_vs_real_forward": real,
        "status": ("sound" if real == agreed else "silent-wrong"),
        "why": (f"the extractor refuses nothing, both layers map, and "
                f"run.py's own reference agrees on {total - agreed} of "
                f"{total} bytes -- while the machine's output differs from "
                f"what model(x) actually returns in {real} of {total}. A "
                f"fan-out graph is being chained on one side of the comparison "
                f"or the other: this is the bug the row exists for, and it was "
                f"fixed once already."
                if real != agreed else
                f"run.py's reference and the real forward() agree, on a graph "
                f"that is NOT a chain ({why}): both sides take each layer's "
                f"activation from the fx graph, so the agreement is evidence")}}


# ------------------------------------------------------------------ main ---
def config():
    from examples.tinytpu.microarch_isa import (
        AR_RAW_DIST, DMA_WORDS, MAXDIM, MAXROWS, NAR, NVR, QD, SPAD_ROWS, T)
    return {"T": T, "MAXDIM": MAXDIM, "QD": QD, "DMA_WORDS": DMA_WORDS,
            "MAXROWS": MAXROWS, "AR_RAW_DIST": AR_RAW_DIST, "NAR": NAR,
            "NVR": NVR, "SPAD_ROWS": SPAD_ROWS}


def key_of(cfg):
    return " ".join(f"{k}={cfg[k]}" for k in ("T", "MAXDIM", "QD", "DMA_WORDS"))


def measure():
    cfg = config()
    return {"config": cfg, "ops": op_rows(), "shapes": shape_rows(),
            "entries": entry_rows(), "assumptions": assumption_rows()}


def load_map():
    try:
        with open(MAP) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"{MAP} is unreadable ({exc}). Re-create it with --emit; a "
                 f"map with nothing to diff against cannot check anything.")


def diff(old, new):
    """Every cell that moved, named. Not a count -- a count cannot be acted on."""
    bad = []
    for section in ("ops", "shapes", "entries", "assumptions"):
        a, b = old.get(section, {}), new[section]
        for key in sorted(set(a) | set(b)):
            if key not in a:
                bad.append(f"{section}/{key}: NEW, {summarise(b[key])}")
            elif key not in b:
                bad.append(f"{section}/{key}: GONE (was {summarise(a[key])})")
            elif a[key] != b[key]:
                bad.append(f"{section}/{key}: was {summarise(a[key])}, "
                           f"now {summarise(b[key])}")
    return bad


def summarise(cell):
    if "status" in cell:
        return f"{cell['status']}" + (f" ({cell.get('why','')[:70]})"
                                      if cell["status"] != "mapped" else "")
    if "differ_vs_real_forward" in cell:
        return (f"{cell['status']} agreed={cell['differ_vs_run_py_reference']} "
                f"real={cell['differ_vs_real_forward']}/{cell['bytes']}")
    if "tier" in cell:
        p = cell.get("vs_pytorch") or {}
        return (f"tier={cell['tier']} layers={cell['layers']} "
                f"vs_spec={(cell.get('vs_spec') or {}).get('ok')} "
                f"vs_pytorch={p.get('ok')} chain={p.get('chain')} "
                f"rtl={(cell.get('rtl') or {}).get('declared_for_this_config')}")
    return json.dumps(cell, sort_keys=True)[:90]


def show(data):
    print(f"front half scope at {key_of(data['config'])}")
    print("\n  OPS (what a torch.fx graph may contain)")
    for key, row in sorted(data["ops"].items(),
                           key=lambda kv: (kv[1]["status"], kv[0])):
        tally = " ".join(f"{k}={v}" for k, v in sorted(row["counts"].items()))
        print(f"    {row['status']:26s} {key:34s} {tally}")
        if row["status"] != "maps":
            print(f"        {row['why'][:110]}")
    print("\n  SHAPES (M x K x N, each mapped cell VERIFIED against spec.gold)")
    for key, row in data["shapes"].items():
        extra = "" if row["status"] == "mapped" else f"  {row.get('why','')[:70]}"
        print(f"    {key:16s} {row['status']}{extra}")
    print("\n  ENTRIES (what each suite entry's bit-exactness is evidence of)")
    for name, row in data["entries"].items():
        p = row.get("vs_pytorch") or {}
        print(f"    {name:12s} tier={row['tier']:10s} "
              f"vs_spec(ours-vs-ours)={(row.get('vs_spec') or {}).get('ok')}  "
              f"vs_pytorch={p.get('ok')} over {p.get('bytes')} bytes  "
              f"chain={p.get('chain')}  "
              f"rtl_declared={(row.get('rtl') or {}).get('declared_for_this_config')}")
    print("\n  ASSUMPTIONS the suite's verification makes, probed")
    for name, row in data.get("assumptions", {}).items():
        print(f"    {row['status']:14s} {name}")
        print(f"        {row['why'][:400]}")
    silent = [k for k, v in data["shapes"].items() if v["status"] == "silent-wrong"]
    blind = [k for k, v in data["shapes"].items()
             if v["status"] == "refused-without-a-cause"]
    # A graph run.py could not represent at all: `dataflow` returns no sources
    # and the end-to-end check did not run. NOT "chain is False" -- a fan-out
    # is checked correctly now, and counting one as unsound would report the
    # fix as the defect.
    unsound = [n for n, v in data["entries"].items()
               if (v.get("vs_pytorch") or {}).get("ok") is None]
    print(f"\n  {len(data['shapes'])} shape cells, "
          f"{sum(1 for v in data['shapes'].values() if v['status'] == 'mapped')} "
          f"mapped and bit-exact, {len(silent)} SILENT-WRONG {silent}, "
          f"{len(blind)} refused with no cause named {blind}")
    broken = [n for n, v in data.get("assumptions", {}).items()
              if v["status"] == "silent-wrong"]
    print(f"  {len(unsound)} entr(ies) whose vs-PyTorch check did not run "
          f"{unsound}; {len(broken)} harness assumption(s) that fail silently "
          f"{broken}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true",
                    help="rewrite scope_map.json for THIS configuration")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff; exit 1 if any cell moved")
    args = ap.parse_args(argv)
    try:
        import torch                                   # noqa: F401
    except ImportError:
        print("SKIPPED: torch is absent, and the front half IS torch.fx.")
        return 0

    new = measure()
    key = key_of(new["config"])
    if args.emit:
        book = {}
        if os.path.exists(MAP):
            book = load_map()
        book[key] = new
        with open(MAP, "w") as fh:
            json.dump(book, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {os.path.relpath(MAP, REPO)} for {key}")
        show(new)
        return 0
    if args.check:
        book = load_map()
        if key not in book:
            print(f"SCOPE MAP: no measurement recorded for {key}; recorded "
                  f"configurations are {sorted(book)}. A configuration this "
                  f"map has never seen is 'cannot compare', not 'agrees'.")
            return 1
        moved = diff(book[key], new)
        show(new)
        for line in moved:
            print(f"  MOVED {line}")
        print(f"\nSCOPE MAP {'CHANGED' if moved else 'UNCHANGED'} at {key}: "
              f"{len(moved)} cell(s) moved")
        return 1 if moved else 0
    show(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce what the workload suite is allowed to claim, model by model.

Why this exists: ``run.py`` runs, and nothing checked it. The suite's one real
distinction -- **which models are verified against a reference, and which are
only executed** -- lived in prose on
``docs/source/designs/workload_suite.rst``, where a model could be promoted
from "correct" to "confirmed" by editing a table.

What it does, in one pass and at zero cost (no Vitis, no licence, no RTL):

1. Reads the declared claims (``claims.json`` beside this file, or ``--claims``).
   Every model in ``models.MODELS`` must have an entry and every entry must name
   a model -- a model added without a claim is a failure, not a skip.
2. Extracts each model and checks the layers and the **refusals** against the
   declaration. ``mlp_bias`` is in the suite to *not* map: a run in which it
   maps cleanly fails here.
3. Maps every layer through the ACT search onto the fixed TinyTPU and runs
   ``act.correctness.check`` -- ``isa_ref`` against the spec's ``gold`` over
   four operand distributions, with the write window enforced.
4. Runs the **model**, not the layers: each layer's program in turn through
   ``isa_ref``, each fed the activation the **fx graph** says it consumes
   (``run.layer_sources``), against PyTorch's own evaluation of the same
   dataflow with the machine's epilogue. Bit-exact on every byte or it fails.
   A graph whose layers cannot be fed that way is a failure, not an
   approximation.
5. Enforces the **tier**. ``confirmed`` requires a declared RTL measurement
   whose configuration is the one this process is running -- T, MAXDIM, QD and
   DMA_WORDS, all four. A measurement from another configuration is refused as
   unpairable rather than quoted. ``correct`` requires that there is NO
   measurement and that the reason is stated.
6. Checks that every string the claims declare the page must contain is still
   in it. The association is declared, never inferred: a checker that guessed
   which figure in prose a cycle count referred to would produce false
   failures, and a checker that cries wolf is one nobody runs.

It fails closed. An unreadable claims file, a missing model, a missing
reference, a configuration with no declared measurement -- each is a failure,
not a skip. The one legitimate skip is ``torch`` being absent, because the
suite's front end is ``torch.fx`` and the requirement is deliberately not in
``requirements.txt``.

Run::

    python examples/tinytpu/workloads/gate.py
    python examples/tinytpu/workloads/gate.py --simulator   # + the built design
    python examples/tinytpu/workloads/gate.py --claims FILE

Exit 0 if every claim holds, 1 otherwise.
"""

import argparse
import json
import os
import sys
import tempfile


def _repo_root(start=None):
    """The checkout root, found by searching upward for a marker.

    Not by counting directories up from ``__file__``: that has been wrong here
    four times, including in a fix for itself, and the repository has been
    reorganised under these files. A path that encodes tree shape is a latent
    break.
    """
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
CLAIMS = os.path.join(HERE, "claims.json")

#: The knobs a cycle count is only meaningful with respect to. TPU_MAXDIM
#: defaults to 64 and not 16; TPU_QD defaulted to 8 until 63ee6ec7 and to 16
#: after it; DMA_WORDS is the burst-widening candidate. "I ran cosim" names no
#: configuration, so all four travel with every number.
CONFIG_KEYS = ("T", "MAXDIM", "QD", "DMA_WORDS")

TIERS = ("correct", "confirmed", "probe")


class Fail(Exception):
    """A claim that does not hold. Carries its own explanation."""


def load_claims(path):
    """Fails closed: unreadable or malformed is a failure, never a skip."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"claims file unreadable at {path}: {exc}\n"
                 f"This gate has nothing to enforce without it, and an "
                 f"unenforceable gate must fail, not pass.")
    if not isinstance(data.get("models"), dict) or not data["models"]:
        sys.exit(f"{path} declares no models")
    return data


def live_config():
    """What this process is actually building, read from the build itself."""
    from examples.tinytpu.microarch_isa import DMA_WORDS, MAXDIM, QD, T
    return {"T": T, "MAXDIM": MAXDIM, "QD": QD, "DMA_WORDS": DMA_WORDS}


def pick_measurement(entry, live):
    """The declared run for THIS configuration, or why there is not one.

    A configuration key the declaration omits is "cannot pair", never
    "matches": an older row carries no QD field at all, and the absence dates
    it to before the parameter existed. Treating a missing field as agreement
    is how a number from one build comes to be quoted beside another.
    """
    reasons = []
    for run in entry.get("measured") or []:
        config = run.get("config") or {}
        missing = [k for k in CONFIG_KEYS if k not in config]
        if missing:
            reasons.append(f"a declared run omits {', '.join(missing)}, so it "
                           f"cannot be paired with any configuration")
            continue
        differs = [f"{k}={config[k]} not {live[k]}"
                   for k in CONFIG_KEYS if config[k] != live[k]]
        if differs:
            reasons.append("a declared run is " + ", ".join(differs))
            continue
        return run, None
    return None, reasons


def check_extraction(name, entry, extraction):
    """Layers and refusals as declared -- including that they ARE refused."""
    want = entry.get("layers", [])
    got = [{"name": l.name, "module": l.module,
            "shape": [l.m, l.k, l.n], "relu": l.relu}
           for l in extraction.layers]
    if got != want:
        raise Fail(f"{name}: the extraction is not what is declared.\n"
                   f"      declared {json.dumps(want)}\n"
                   f"      extracted {json.dumps(got)}")
    want_refusals = entry.get("refusals", [])
    if len(extraction.refusals) != len(want_refusals):
        raise Fail(
            f"{name}: {len(want_refusals)} refusal(s) declared, "
            f"{len(extraction.refusals)} seen: "
            + "; ".join(f"{r.where}: {r.why}" for r in extraction.refusals)
            + ("\n      A model declared as a probe that now maps cleanly is "
               "the failure this line exists for: the suite must name what it "
               "cannot do, not quietly approximate it."
               if want_refusals and not extraction.refusals else ""))
    for declared, seen in zip(want_refusals, extraction.refusals):
        if declared["where"] != seen.where:
            raise Fail(f"{name}: refusal expected at {declared['where']}, "
                       f"seen at {seen.where}")
        if declared["why_contains"] not in seen.why:
            raise Fail(f"{name}: the refusal at {seen.where} no longer says "
                       f"{declared['why_contains']!r}; it says {seen.why!r}. "
                       f"The suite refuses for a stated reason, and the reason "
                       f"is part of the claim.")


def check_quotes(entry, name, docs_root):
    """Every string the claims say a page must contain is still in it.

    Declared, not inferred. Matching a figure against a set of committed
    totals is decidable; deciding which cycle count a figure in prose refers
    to is not, so the page is told what it must carry rather than parsed for
    what it might mean.
    """
    for rel, needle in entry.get("quoted_in", []):
        path = os.path.join(docs_root, rel)
        try:
            with open(path, errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise Fail(f"{name}: the page it is quoted in is unreadable "
                       f"({rel}: {exc})")
        if needle not in text:
            raise Fail(f"{name}: {rel} no longer contains {needle!r}. Either "
                       f"the page was edited away from the measurement or the "
                       f"measurement changed; the two must move together.")


def check_specs_on_disk(extraction, committed):
    """The committed specs are what the extractor emits today, byte for byte."""
    from examples.tinytpu.workloads import extract
    with tempfile.TemporaryDirectory() as tmp:
        for path in extract.emit(extraction, tmp):
            base = os.path.basename(path)
            ref = os.path.join(committed, base)
            if not os.path.isfile(ref):
                raise Fail(f"{extraction.model}: {base} is not committed "
                           f"under {os.path.relpath(committed, REPO)}")
            with open(path) as a, open(ref) as b:
                if a.read() != b.read():
                    raise Fail(
                        f"{extraction.model}: the committed {base} is not what "
                        f"the extractor emits now. Re-run `run.py --emit`, and "
                        f"if the spec really changed, the measurements that "
                        f"were taken on the old one no longer describe it.")


def run_model(name, entry, module, docs_root, specs_dir, live, report):
    from examples.tinytpu.act import correctness
    from examples.tinytpu.workloads import extract, models, run as runner

    tier = entry.get("tier")
    if tier not in TIERS:
        raise Fail(f"{name}: tier {tier!r} is not one of {TIERS}")

    extraction = extract.of(name)
    check_extraction(name, entry, extraction)

    programs = []
    for sp in extract.specs(extraction):
        result = runner.map_layer(sp)
        if not result.best:
            raise Fail(f"{name}: {sp['name']} -- every nest was refused, but "
                       f"the layer is declared mappable")
        fails = correctness.check(sp, result.best.program, module)
        if fails:
            raise Fail(f"{name}: {sp['name']} is NOT bit-exact\n      "
                       + "\n      ".join(fails))
        programs.append(result.best.program)

    chained = None
    if tier != "probe":
        check_specs_on_disk(extraction, specs_dir)
        model, example_inputs = models.build(name)
        try:
            want = runner.quantized_reference(model, extraction,
                                              example_inputs[0])
            got = runner.run_on_machine(model, extraction, programs,
                                        example_inputs[0])
        except runner.Unrepresentable as why:
            raise Fail(f"{name}: the suite cannot feed this graph one layer "
                       f"from another -- {why}. A graph the end-to-end check "
                       f"cannot represent is a failure here; it used to be "
                       f"chained anyway, and the comparison then said nothing "
                       f"(workloads/scope.py, assumptions/"
                       f"chained_verification).") from why
        bad = sum(int((a != b).sum()) for a, b in zip(got, want))
        total = sum(a.size for a in want)
        if bad:
            raise Fail(f"{name}: the chained programs do NOT reproduce "
                       f"PyTorch's int8 epilogue chain -- {bad} of {total} "
                       f"bytes differ")
        chained = total

    run, reasons = pick_measurement(entry, live)
    if tier == "confirmed":
        if run is None:
            raise Fail(
                f"{name} is declared CONFIRMED and no declared measurement "
                f"belongs to the configuration this gate is running "
                f"({_fmt(live)}).\n      "
                + ("\n      ".join(reasons) if reasons else
                   "No measurement is declared at all.")
                + "\n      A missing or differing configuration key is "
                  "'cannot pair', not 'matches'. Run the gate at the declared "
                  "configuration, or measure this one.")
        declared_total = run.get("total")
        summed = sum(run.get("per_layer", {}).values())
        if summed != declared_total:
            raise Fail(f"{name}: the per-layer RTL cycles sum to {summed}, "
                       f"and the model total is declared as {declared_total}. "
                       f"A model figure here is a sum over layers with no "
                       f"fusion and no residency, so the two must agree.")
        names = {l.name for l in extraction.layers}
        if set(run.get("per_layer", {})) != names:
            raise Fail(f"{name}: the measurement covers "
                       f"{sorted(run.get('per_layer', {}))}, the extraction "
                       f"has {sorted(names)}. A model is confirmed when every "
                       f"layer of it ran, not when some did.")
        check_quotes(entry, name, docs_root)
        report.append((name, "confirmed", declared_total, chained,
                       len(extraction.layers), len(extraction.refusals)))
    elif tier == "correct":
        if run is not None or (entry.get("measured") or []):
            raise Fail(
                f"{name} is declared CORRECT, which is the claim that RTL has "
                f"never produced a cycle count for it -- and a measurement is "
                f"declared. Move the tier to `confirmed` or remove the "
                f"measurement; the ladder is not decoration.")
        if not entry.get("not_confirmed"):
            raise Fail(f"{name} is declared CORRECT and does not say why it is "
                       f"not confirmed. 'Not measured' without a reason is "
                       f"indistinguishable from 'not attempted'.")
        report.append((name, "correct", None, chained,
                       len(extraction.layers), len(extraction.refusals)))
    else:
        if entry.get("measured"):
            raise Fail(f"{name} is the probe and carries a measurement")
        report.append((name, "probe", None, None,
                       len(extraction.layers), len(extraction.refusals)))


def _fmt(config):
    return " ".join(f"{k}={config[k]}" for k in CONFIG_KEYS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default=CLAIMS,
                    help="the declared claims (default: beside this file)")
    ap.add_argument("--docs", default=REPO,
                    help="root the claims' quoted_in paths are relative to")
    ap.add_argument("--specs", default=os.path.join(HERE, "specs"),
                    help="the committed per-layer specs")
    ap.add_argument("--simulator", action="store_true",
                    help="also run every program on the built dataflow design")
    ap.add_argument("--models", nargs="*",
                    help="only these (the gate still fails if any is unknown)")
    args = ap.parse_args(argv)

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIPPED: torch is absent. The suite's front end is torch.fx and "
              "torch is deliberately not in requirements.txt, so this is the "
              "one legitimate skip -- every other missing input is a failure.")
        return 0

    claims = load_claims(args.claims)
    from examples.tinytpu.workloads import models

    declared = set(claims["models"])
    known = set(models.MODELS)
    if declared != known:
        for name in sorted(known - declared):
            print(f"  FAIL {name} is in the suite and declares no claim. A "
                  f"model that runs with nothing said about it is exactly "
                  f"what this gate exists to prevent.")
        for name in sorted(declared - known):
            print(f"  FAIL {name} is claimed and is not in the suite")
        print("\nWORKLOAD GATE FAILED: the claims and the suite disagree "
              "about which models exist")
        return 1

    live = live_config()
    print(f"live build: {_fmt(live)}")
    print(f"claims:     {os.path.relpath(os.path.abspath(args.claims), REPO)}")

    module = None
    if args.simulator:
        from examples.tinytpu.act import correctness
        print("building the dataflow simulator once ...", flush=True)
        module = correctness.build_module()

    chosen = args.models or list(claims["models"])
    bad, report = [], []
    for name in chosen:
        if name not in claims["models"]:
            bad.append(f"{name}: not declared")
            continue
        try:
            run_model(name, claims["models"][name], module, args.docs,
                      args.specs, live, report)
        except Fail as exc:
            bad.append(str(exc))
        except Exception as exc:   # noqa: BLE001 -- a fault is a verdict
            bad.append(f"{name}: {type(exc).__name__}: {exc}")

    print(f"\n  {'model':12s} {'tier':10s} {'layers':>6s} {'RTL cycles':>11s} "
          f"  what it is")
    for name, tier, total, chained, count, refused in report:
        if tier == "confirmed":
            what = f"VERIFIED and measured; {chained} bytes match PyTorch"
            cycles = f"{total}"
        elif tier == "correct":
            what = (f"VERIFIED, only executed in software; "
                    f"{chained} bytes match PyTorch")
            cycles = "--"
        else:
            what = (f"the probe: {refused} refusal(s), as declared")
            cycles = "--"
        print(f"  {name:12s} {tier:10s} {count:>6d} {cycles:>11s}   {what}")

    for line in bad:
        print(f"  FAIL {line}")
    if bad:
        print(f"\nWORKLOAD GATE FAILED: {len(bad)} claim(s) do not hold")
        return 1
    confirmed = [r for r in report if r[1] == "confirmed"]
    correct = [r for r in report if r[1] == "correct"]
    probes = [r for r in report if r[1] == "probe"]
    print(f"\nWORKLOAD GATE OK at {_fmt(live)}: "
          f"{len(confirmed)} model(s) verified AND confirmed on RTL, "
          f"{len(correct)} verified and only executed, "
          f"{len(probes)} probe(s) still refusing.")
    print("  'verified' is bit-exact against isa_ref over four operand "
          "distributions AND\n  against PyTorch's own int8 epilogue chain over "
          "every byte of the model.\n  'confirmed' additionally means RTL "
          "produced the cycle count, at THIS configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

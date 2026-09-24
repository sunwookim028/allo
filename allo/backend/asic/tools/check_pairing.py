#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refuse to pair a cycle count with an area figure from a different configuration.

Why this exists: every error in this work has lived in the **seams between**
flows -- a configuration claimed but not run, a row measured on a different
design, an area figure placed beside cycles from other RTL. No single-flow gate
can see any of them, because each flow is internally consistent. This one looks
only at the join.

A pairing is admissible on exactly one of two bases, and the pairings file says
which:

``same-export``
    Both sides cite the **same RTL export**: the cycles were measured on it and
    the area was synthesised from it. Their configuration then agrees *by
    construction*, whatever it was, because the export is the configuration.
    This is identity, not inference -- so it is checked as identity: the
    ``manifest_md5`` the synthesis run recorded must still equal the md5 of the
    manifest in the tree. An export that has been re-emitted since the run is
    **not** the RTL that area describes, and this is the check that says so.

``declared-config``
    The two sides come from different runs, so every configuration key must be
    present on both sides and equal. **A key that is absent is "cannot pair",
    never "matches."** An older row carries no ``QD`` field at all, and that
    absence dates it to before the parameter existed; reading absence as
    agreement is precisely how a number from one build comes to be printed
    beside another.

The file also carries a ``refused`` list: pairs deliberately **not** made, each
with the reason. Those are checked too, and in both directions -- a refused
pair must still be inadmissible, and it must still be inadmissible *for the
stated reason*. A refusal whose cause has been repaired is reported as
promotable rather than left standing, and one that has started failing for a
different reason is reported as mis-stated. An allow-list nobody re-derives
becomes a place to hide failures.

Nothing here is inferred from prose. Associating a figure in a sentence with
the cycle count nearest it is not decidable, and a checker that produces false
failures stops being run; so a page that states a pairing **declares** the run
it quotes, and this tool checks that the declared string is still on the page.

The reports, the exports and the pairings are all given on the command line:
the results belong to a design, this checker does not.

Run::

    python allo/backend/asic/tools/check_pairing.py \
        --reports examples/tinytpu/asic_synthesis/reports \
        --exports examples/tinytpu/rtl_handoff \
        --pairings examples/tinytpu/asic_synthesis/pairings.json

Exit 0 if every declared pairing holds and every declared refusal still stands,
1 otherwise. It fails closed: a missing or unreadable input is a failure.
"""

import argparse
import hashlib
import json
import os
import sys


def _repo_root():
    """The checkout root, found by searching upward for the vendored flow.

    Not by counting directories up from ``__file__``: that has been wrong here
    four times, including in a fix for itself, and the repository has been
    reorganised under these files.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "allo", "backend", "asic", "nodes")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(os.path.abspath(__file__))
        d = parent


REPO = _repo_root()

#: Reasons a pair is inadmissible. A ``refused`` entry names the one it expects,
#: so a refusal that starts failing differently is caught rather than absorbed.
CAUSES = {
    "no-report": "the area run has no results.json",
    "no-settings": "the area run recorded no settings snapshot, so nothing "
                   "identifies the RTL it read",
    "no-export": "the RTL export it cites is not in the tree",
    "export-superseded": "the export in the tree is not the one the area was "
                         "synthesised from -- its manifest md5 has changed",
    "missing-key": "a configuration key is absent on one side, which is "
                   "'cannot pair', not 'matches'",
    "config-differs": "the two sides name different configurations",
    "not-our-design": "the cycles and the area belong to different designs",
    "cycles-not-in-export": "the declared cycle row is not in the export's "
                            "own record",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, what):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"{what} unreadable at {path}: {exc}\n"
                 f"This gate has nothing to enforce without it, and an "
                 f"unenforceable gate must fail, not pass.")


def area_of(reports, variant):
    """``(results, None)`` or ``(None, (cause, detail))``. Fails closed."""
    rj = os.path.join(reports, variant, "results.json")
    if not os.path.isfile(rj):
        return None, ("no-report", f"{variant}: no results.json under "
                                   f"{os.path.relpath(reports, REPO)}")
    try:
        with open(rj) as fh:
            return json.load(fh), None
    except (OSError, ValueError) as exc:
        return None, ("no-report", f"{variant}: results.json unreadable ({exc})")


def normalise(config):
    """``{"TPU_MAXDIM": "64"}`` and ``{"MAXDIM": 64}`` are the same statement."""
    out = {}
    for key, value in (config or {}).items():
        name = key[4:] if key.startswith("TPU_") else key
        try:
            out[name] = int(value)
        except (TypeError, ValueError):
            out[name] = value
    return out


def export_config(exports, name):
    """Every configuration key the export itself records.

    The ``config`` block is the environment the export was emitted with, and
    ``params`` -- present only on exports emitted after the exporter began
    recording it -- is the resolved parameter set, which is where ``QD`` and
    ``DMA_WORDS`` live. Whatever is absent stays absent: this function never
    supplies a default, because a default is exactly the thing that was not
    recorded.
    """
    path = os.path.join(exports, name, "MANIFEST.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    config = normalise(manifest.get("config"))
    config.update(normalise(manifest.get("params")))
    return config


def admit(pairing, reports, exports, keys):
    """``(None, detail)`` if admissible, else ``(cause, detail)``.

    One function decides both lists, so a declared pairing and a declared
    refusal are judged by exactly the same rule.
    """
    basis = pairing.get("basis")
    variant = (pairing.get("area") or {}).get("variant")
    results, bad = area_of(reports, variant)
    if bad:
        return bad

    settings = (results.get("settings") or {}).get("rtl")
    if not settings:
        return ("no-settings",
                f"{variant}: results.json carries no settings.rtl, so nothing "
                f"says which RTL produced this area")

    if basis == "not-our-design":
        return ("not-our-design",
                f"{variant}: {pairing.get('why', 'a different design')}")

    name = pairing.get("export")
    if not name:
        return ("no-export", f"{variant}: the pairing names no export")
    directory = os.path.join(exports, name)
    if not os.path.isdir(directory):
        return ("no-export",
                f"{variant}: no export {name} under "
                f"{os.path.relpath(exports, REPO)}")

    manifest = settings.get("manifest")
    local = os.path.join(directory, os.path.basename(manifest or ""))
    if not manifest or not os.path.isfile(local):
        return ("no-export",
                f"{variant}: the manifest it read ({manifest}) is not in "
                f"export {name}")
    here, there = md5(local), settings.get("manifest_md5")
    if here != there:
        return ("export-superseded",
                f"{variant}: synthesised from {manifest} md5 {there}, and the "
                f"{name} export in the tree is md5 {here}. The RTL was "
                f"re-emitted after the run, so this area does not describe the "
                f"export a reader would find")

    # The manifest is a FILE LIST, so its md5 identifies the set of compile
    # units and not the RTL in them: T4_MAXDIM16_shipped_baseline and
    # T4_MAXDIM64_shipped emit the same 145 filenames and hash identically.
    # The checksum therefore proves the export has not been re-emitted; it does
    # NOT prove which export was read. The run's own recorded config and the
    # directory it read are what establish that, and both are checked.
    mine = export_config(exports, name) or {}
    theirs = normalise(settings.get("config"))
    clash = [f"{k}: export {name} says {mine[k]}, the run recorded {theirs[k]}"
             for k in sorted(set(mine) & set(theirs)) if mine[k] != theirs[k]]
    if clash:
        return ("config-differs",
                f"{variant}: the export named is not the one the run read; "
                + "; ".join(clash))
    read = os.path.basename((settings.get("dir") or "").rstrip("/"))
    alias = pairing.get("export_path_alias")
    if read and read != name and alias != read:
        return ("no-export",
                f"{variant}: the run read {read!r} and the pairing names "
                f"{name!r}. If they are the same export under another path, "
                f"declare it as export_path_alias with the reason")

    if basis == "same-export":
        # Identity, so the configuration needs no comparison -- but the cycle
        # row must really be the one this export records, not one carried over.
        readme = os.path.join(directory, "README.md")
        if not os.path.isfile(readme):
            return ("cycles-not-in-export",
                    f"{variant}: export {name} has no README.md to check the "
                    f"declared cycles against")
        with open(readme, errors="replace") as fh:
            text = fh.read()
        for row in pairing.get("cycles", {}).get("rows", []):
            if row not in text:
                return ("cycles-not-in-export",
                        f"{variant}: {row!r} is not in {name}/README.md. A "
                        f"cycle row that the export does not record was "
                        f"measured somewhere else")
        return (None, f"{variant}: same export {name}, manifest md5 {here[:8]}")

    if basis == "declared-config":
        known = dict(mine)
        known.update(theirs)
        ours = normalise((pairing.get("cycles") or {}).get("config"))
        missing = ([f"the cycles declare no {k}" for k in keys if k not in ours]
                   + [f"export {name} records no {k}" for k in keys
                      if k not in known])
        if missing:
            return ("missing-key", f"{variant}: " + "; ".join(missing))
        differs = [f"{k}: cycles {ours[k]}, area {known[k]}"
                   for k in keys if ours[k] != known[k]]
        if differs:
            return ("config-differs", f"{variant}: " + "; ".join(differs))
        return (None, f"{variant}: configuration agrees on "
                      + " ".join(f"{k}={ours[k]}" for k in keys))

    return ("no-settings", f"{variant}: unknown basis {basis!r}")


def check_quotes(pairing, root):
    """A page that states a pairing declares the run it quotes."""
    bad = []
    for rel, needle in pairing.get("quoted_in", []):
        path = os.path.join(root, rel)
        try:
            with open(path, errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            bad.append(f"{rel} unreadable ({exc})")
            continue
        if needle not in text:
            bad.append(f"{rel} no longer contains {needle!r}")
    return bad


def total_area(results):
    value = (results.get("area") or {}).get("total_cell")
    return f"{value:,.0f}" if isinstance(value, (int, float)) else "?"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True,
                    help="a design's committed reports directory")
    ap.add_argument("--exports", required=True,
                    help="the design's committed RTL exports")
    ap.add_argument("--pairings", required=True,
                    help="the declared pairings and refusals")
    ap.add_argument("--docs", default=REPO,
                    help="root the pairings' quoted_in paths are relative to")
    args = ap.parse_args(argv)

    for path, what in ((args.reports, "reports directory"),
                       (args.exports, "exports directory")):
        if not os.path.isdir(path):
            sys.exit(f"no {what} at {path}")
    declared = load_json(args.pairings, "pairings file")
    keys = tuple(declared.get("config_keys") or ())
    if not keys:
        sys.exit(f"{args.pairings} declares no config_keys -- without them "
                 f"'same configuration' means nothing")

    bad, admitted = [], []
    for pairing in declared.get("pairings", []):
        name = pairing.get("name", "?")
        cause, detail = admit(pairing, args.reports, args.exports, keys)
        if cause:
            bad.append(f"CLAIMED PAIR REFUSED  {name}\n      {detail}\n"
                       f"      ({CAUSES.get(cause, cause)})")
            continue
        for line in check_quotes(pairing, args.docs):
            bad.append(f"CLAIMED PAIR  {name}: {line}")
        results, _ = area_of(args.reports, pairing["area"]["variant"])
        admitted.append((name, pairing, total_area(results), detail))

    promotable = []
    for pairing in declared.get("refused", []):
        name = pairing.get("name", "?")
        expected = pairing.get("expect")
        if expected not in CAUSES:
            bad.append(f"REFUSAL  {name}: expects cause {expected!r}, which is "
                       f"not one this checker can produce")
            continue
        if not pairing.get("why"):
            bad.append(f"REFUSAL  {name}: carries no reason. A refusal without "
                       f"one is indistinguishable from an oversight.")
            continue
        cause, detail = admit(pairing, args.reports, args.exports, keys)
        if cause is None:
            promotable.append(f"{name}: this pair is now ADMISSIBLE ({detail}). "
                              f"The reason it was refused has been repaired, so "
                              f"move it into `pairings`.")
        elif cause != expected:
            bad.append(f"REFUSAL  {name}: refused for {cause!r}, and the file "
                       f"says {expected!r}.\n      {detail}\n"
                       f"      The refusal still stands, and its stated reason "
                       f"is wrong, which makes the record misleading.")

    covered = {p.get("area", {}).get("variant")
               for p in (declared.get("pairings", [])
                         + declared.get("refused", []))}
    for variant in sorted(os.listdir(args.reports)):
        if not os.path.isfile(os.path.join(args.reports, variant,
                                           "results.json")):
            continue
        if variant not in covered:
            bad.append(f"COVERAGE  {variant} has a committed area figure and "
                       f"no declared pairing status. Every area figure is "
                       f"either paired with cycles or explicitly not; silence "
                       f"is how an unpaired figure comes to be quoted.")

    print(f"pairings: {os.path.relpath(os.path.abspath(args.pairings), REPO)}"
          f"   keys: {' '.join(keys)}")
    if admitted:
        print("\nADMITTED -- a cycle count and an area figure from one "
              "configuration:")
        for name, pairing, area, detail in admitted:
            rows = pairing.get("cycles", {}).get("rows", [])
            print(f"  {name}")
            print(f"      area   {area} um2   ({detail})")
            print(f"      cycles {', '.join(rows) if rows else '--'}")
    for line in promotable:
        print(f"\n  PROMOTABLE  {line}")
    if declared.get("refused"):
        print("\nREFUSED -- declared, still standing, for the stated reason:")
        for pairing in declared["refused"]:
            if any(pairing.get("name", "?") in p for p in promotable):
                continue
            print(f"  {pairing.get('name')}: {CAUSES[pairing['expect']]}")

    if bad:
        print("\nPAIRING GATE FAILED:")
        for line in bad:
            print(f"  FAIL {line}")
    if promotable:
        print("\nPAIRING GATE FAILED -- a declared refusal no longer applies:")
        for line in promotable:
            print(f"  FAIL {line}")
    if bad or promotable:
        return 1
    print(f"\nPAIRING OK: {len(admitted)} pairing(s) admitted, "
          f"{len(declared.get('refused', []))} refused for stated reasons, "
          f"every committed area figure accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

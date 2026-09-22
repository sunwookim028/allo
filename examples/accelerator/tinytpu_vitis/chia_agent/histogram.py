# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The refusal histogram, per hardware variant. The co-design signal, tabulated.

`$0`, no model, no Vitis, about a second per variant. For each named hardware
variant this composes an evaluation tree exactly as `evaluate.py` does -- frozen
files from `git show`, the variant's `microarch_isa.py` on top -- and runs the
FROZEN mapper (`mapspace.py`) over it. What comes out is, per shape: how many of
the enumerated loop nests that hardware can encode, which constraint refused the
rest, and which nest the mapper would choose.

This is the artefact the co-design claim rests on, and it is meant to be read
side by side. A hardware field change widens the software's expressible space,
and **the binding refusal moves to the next constraint** -- which is what makes
the story credible, because a change that fixed everything at once would not be.

**The histogram is FIRST-CAUSE, and that understates the problem.** The encoder
raises at the first constraint a nest violates, so a nest counted under
`acc-peel` may violate two others as well, and "1,150 refused by `acc`" must not
be read as "1,150 nests a fixed `acc` would free". `--second-cause` measures the
rest of the distribution the only honest way available here: it removes one check
and re-censuses. The programs that then get through are known-incorrect -- test
k3 in `test_codesign.py` is exactly that mutant being rejected -- so
`--second-cause` COUNTS ONLY and its "encodable" figure is not a claim about
anything runnable.

It reports NO cycle count. Cycles come from `evaluate.py --codesign` (RTL cosim)
and nothing else; the variants here are candidate machines, and whether the
nests they unlock are worth their area is a separate, measured question.

    python histogram.py                       # every variant, rst table
    python histogram.py --variants shipped,agu4 --format text

The variants are the same edits `test_codesign.py` applies, kept in one place so
the table and the test cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
#: The checkout whose HEAD the frozen files come from. Overridable so this
#: script can be run from outside the tree while the tree is busy (the
#: evaluator re-hashes the checkout after every stage and a new file under
#: `examples/` mid-evaluation reads as tamper).
REPO = Path(os.environ.get("TINYTPU_REPO", "")) or AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"
ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")

SHAPES = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]
#: Files the mapper's import chain needs, from git.
NEEDED = ["examples/__init__.py", f"{PKG}/isa_ref.py", f"{PKG}/isa_dsl.py",
          f"{PKG}/microarch_isa.py", f"{PKG}/chia_agent/mapspace.py"]

#: `AGU_TERMS` 3 -> 4. The AGU word is 64 bits and three terms use 19 each
#: (target 4, level 3, stride 12), so a fourth has to come out of the field
#: widths: four terms of 16 (target 4, level 3, stride 9, 8 usable). Five exact
#: edits, one per place the 19 was written as a literal -- the encoder, the
#: sequencer's Allo kernel, `expand`, and `check_program`.
AGU4 = [
    ("AGU_TERMS = 3                  # address terms per instruction\n",
     "AGU_TERMS = 4                  # address terms per instruction\n"
     "#: Bits per AGU term. The word is 64 bits; four terms leave 16 each, so\n"
     "#: the stride field narrows from 12 bits to 9 (8 usable after the sign\n"
     "#: bit). Programs at MAXDIM=16 use strides of at most MAXDIM.\n"
     "AGU_W = 64 // AGU_TERMS        # target 4 bits, level 3, stride AGU_W - 7\n"),
    ('        assert 0 <= stride < (1 << 11), f"stride {stride} does not fit"\n'
     "        base = 19 * i\n",
     '        assert 0 <= stride < (1 << (AGU_W - 8)), f"stride {stride} does not fit"\n'
     "        base = AGU_W * i\n"),
    ("                    tw: int32 = w1[19 * _t : 19 * _t + 4]\n"
     "                    lw: int32 = w1[19 * _t + 4 : 19 * _t + 7]\n"
     "                    sw: int32 = w1[19 * _t + 7 : 19 * _t + 19]\n",
     "                    tw: int32 = w1[AGU_W * _t : AGU_W * _t + 4]\n"
     "                    lw: int32 = w1[AGU_W * _t + 4 : AGU_W * _t + 7]\n"
     "                    sw: int32 = w1[AGU_W * _t + 7 : AGU_W * _t + AGU_W]\n"),
    ("                base = 19 * t\n"
     "                tw = (w1 >> base) & 0xF\n"
     "                lw = (w1 >> (base + 4)) & 0x7\n"
     "                st = (w1 >> (base + 7)) & 0xFFF\n",
     "                base = AGU_W * t\n"
     "                tw = (w1 >> base) & 0xF\n"
     "                lw = (w1 >> (base + 4)) & 0x7\n"
     "                st = (w1 >> (base + 7)) & ((1 << (AGU_W - 7)) - 1)\n"),
    ("            tw = (w1 >> (19 * t)) & 0xF\n"
     "            lw = (w1 >> (19 * t + 4)) & 0x7\n",
     "            tw = (w1 >> (AGU_W * t)) & 0xF\n"
     "            lw = (w1 >> (AGU_W * t + 4)) & 0x7\n"),
]
#: The loop stack from 4 frames to 6. One constant: every array it sizes
#: (`lp_start`, `lp_iv`, `lp_trip`, `iv_now`) follows, and the AGU word's
#: 3-bit level field already reaches 7.
DEPTH6 = [("LOOP_DEPTH = 4                 # nesting levels, as MiniTPU's "
           "loop stack\n",
           "LOOP_DEPTH = 6                 # nesting levels, as MiniTPU's "
           "loop stack\n")]
#: Twice the instruction memory. `IMEM_SIZE = NHDR + IWORDS * _MAX_STATIC`.
IMEM48 = [("_MAX_STATIC = 24               # longest program shipped, plus "
           "headroom\n",
           "_MAX_STATIC = 48               # longest program shipped, plus "
           "headroom\n")]

VARIANTS = {
    "shipped": ([], "the design as it ships: AGU_TERMS=3, LOOP_DEPTH=4, "
                    "IMEM_SIZE=56"),
    "agu4": (AGU4, "AGU_TERMS=4, the AGU word repacked to four 16-bit terms"),
    "agu4+depth6": (AGU4 + DEPTH6, "AGU_TERMS=4 and a 6-frame loop stack"),
    "agu4+depth6+imem": (AGU4 + DEPTH6 + IMEM48,
                         "AGU_TERMS=4, a 6-frame loop stack, IMEM_SIZE=104"),
    "depth6": (DEPTH6, "a 6-frame loop stack alone -- the control for agu4, "
                       "since LOOP_DEPTH refuses nothing on the shipped design"),
}


def git_show(path: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO,
                          capture_output=True, check=True).stdout


def compose(tree: Path, edits) -> None:
    for rel in NEEDED:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(git_show(rel))
    path = tree / PKG / "microarch_isa.py"
    text = path.read_text()
    for old, new in edits:
        assert text.count(old) == 1, f"anchor occurs {text.count(old)} times"
        text = text.replace(old, new, 1)
    path.write_text(text)


#: The encoder's `acc-peel` POSITION check, removed for `--second-cause`. Not a
#: candidate: the nests it admits overwrite the accumulator on every k-tile.
DROP_POSITION_CHECK = """    if ks and ks[0] != len(emitted) - 1:
        raise Unencodable(
            f"acc-peel: the emitted order is "
            f"{'>'.join(l.rank for l in emitted)}, but the k=0 tile must be a "
            f"peelable prefix, which needs K innermost")
"""


def measure(name: str, shapes, drop_position_check=False) -> dict:
    """The mapper's report for one variant, in its own process and own tree."""
    edits, _ = VARIANTS[name]
    with tempfile.TemporaryDirectory(prefix="tinytpu-histogram-") as tmp:
        tree = Path(tmp)
        compose(tree, edits)
        if drop_position_check:
            dsl = tree / PKG / "isa_dsl.py"
            text = dsl.read_text()
            assert text.count(DROP_POSITION_CHECK) == 1, "position check moved"
            dsl.write_text(text.replace(DROP_POSITION_CHECK, "", 1))
        code = (
            "import json, os, sys\n"
            f"sys.path.insert(0, {str(tree / PKG / 'chia_agent')!r})\n"
            "import mapspace\n"
            "from examples.accelerator.tinytpu_vitis import microarch_isa as u\n"
            f"shapes = {shapes!r}\n"
            "out = mapspace.report(shapes, out=lambda *a: None)\n"
            "print(json.dumps({'shapes': out, 'params': "
            "{'AGU_TERMS': u.AGU_TERMS, 'LOOP_DEPTH': u.LOOP_DEPTH,\n"
            "  'IMEM_SIZE': u.IMEM_SIZE, 'T': u.T, 'MAXDIM': u.MAXDIM,\n"
            "  'AR_RAW_DIST': u.AR_RAW_DIST}}))\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(PYTHONPATH=f"{tree}:{REPO}", PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([ALLO_PYTHON, "-c", code], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=900)
        if p.returncode:
            raise RuntimeError(f"{name}: {(p.stdout + p.stderr)[-2000:]}")
        return json.loads(p.stdout.strip().splitlines()[-1])


CAUSES = ("acc-peel", "emitter", "agu-terms", "loop-depth",
          "accumulator-raw-distance", "imem", "intrinsic", "coverage")


def rows(results, shape_tag):
    """One row per variant: the counts, then every cause in a fixed order."""
    out = []
    for name, data in results.items():
        d = data["shapes"].get(shape_tag, {})
        ref = d.get("refused", {})
        out.append({
            "variant": name,
            "params": f"AGU_TERMS={data['params']['AGU_TERMS']} "
                      f"LOOP_DEPTH={data['params']['LOOP_DEPTH']} "
                      f"IMEM={data['params']['IMEM_SIZE']}",
            "encodable": d.get("encodable"),
            "total": d.get("total"),
            "chosen": d.get("chosen"),
            "words": d.get("words"),
            "dynamic": d.get("dynamic"),
            **{c: ref.get(c, 0) for c in CAUSES},
            "other": sum(n for c, n in ref.items() if c not in CAUSES),
        })
    return out


def as_text(results, shapes):
    lines = []
    for (M, K, N) in shapes:
        tag = f"{M}x{K}x{N}"
        lines.append(f"\n=== {tag} " + "=" * 56)
        head = ["variant", "enc/total", "chosen nest"] + list(CAUSES)
        widths = [20, 10, 22] + [max(len(c), 6) for c in CAUSES]
        lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for r in rows(results, tag):
            cells = [r["variant"], f"{r['encodable']}/{r['total']}",
                     str(r["chosen"])] + [str(r[c]) or "." for c in CAUSES]
            lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def as_rst(results, shapes):
    """A reST list-table per shape, for docs/source/records/."""
    out = []
    for (M, K, N) in shapes:
        tag = f"{M}x{K}x{N}"
        present = [c for c in CAUSES
                   if any(r[c] for r in rows(results, tag))]
        out.append(f"\n{tag}\n{'-' * len(tag)}\n")
        out.append(".. list-table::\n   :header-rows: 1\n")
        head = ["variant", "encodable", "chosen nest", "words"] + present
        out.append("\n".join([f"   * - {head[0]}"]
                             + [f"     - {h}" for h in head[1:]]))
        for r in rows(results, tag):
            cells = [r["variant"], f"{r['encodable']} / {r['total']}",
                     str(r["chosen"]), str(r["words"])] + [
                        (str(r[c]) if r[c] else "--") for c in present]
            out.append("\n".join([f"   * - {cells[0]}"]
                                 + [f"     - {c}" for c in cells[1:]]))
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--shapes", default="4x4x4,16x16x16")
    ap.add_argument("--format", choices=("text", "rst", "json"), default="text")
    ap.add_argument("--second-cause", action="store_true",
                    help="also remove the encoder's acc-peel POSITION check and "
                         "re-census, to show what the first-cause histogram "
                         "hides. Counting only: the programs it admits are "
                         "known-incorrect (test k3)")
    a = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x"))
              for s in a.shapes.split(",") if s]
    results = {}
    for name in a.variants.split(","):
        name = name.strip()
        if name not in VARIANTS:
            print(f"unknown variant {name!r}; one of {sorted(VARIANTS)}")
            return 2
        results[name] = measure(name, shapes)
        if a.format == "text":
            print(f"  measured {name}: {VARIANTS[name][1]}", flush=True)
        if a.second_cause:
            results[name + " (no position check)"] = measure(
                name, shapes, drop_position_check=True)
    if a.format == "json":
        print(json.dumps(results, indent=1, sort_keys=True))
    elif a.format == "rst":
        print(as_rst(results, shapes))
    else:
        print(as_text(results, shapes))
        print("\nNo cycle count appears above. Cycles come from "
              "`evaluate.py --codesign` (RTL cosim) only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

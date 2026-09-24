# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation testing of the ACTION MODEL: what a wrong Action is caught by.

`mutate.py` breaks the design and asks which level notices. This breaks the
DECLARATION -- one action of one instruction in `isa_spec.json` -- and asks
the same question, because an abstraction whose failure modes are unstated is
worth nothing. A layer that only ever sees a correct declaration has never
been shown to refuse a wrong one.

Each mutant is applied to a COPY of the spec in a sandbox under `.mutants-
actions/`, never in place. The sandbox holds a copy of this directory and of
the ISA doc, and sits in front of the checkout on `PYTHONPATH`, so
`gen_isa.py` regenerates inside it and every consumer imports the mutated
spec while `allo` still comes from the checkout. Three levels run, in the
order a person would reach for them:

    rule         `allo.actions` refusing the composition outright, at the
                 moment the machine is built. The cheapest, and the only one
                 that needs no program and no hardware.
    spec_check   `gen_isa.py --check` after regenerating: the design's own
                 assembler, sequencer and unit declarations against what the
                 actions now imply. Catches anything that moves a WORK COUNT,
                 because the design states those independently.
    stress       `stress_isa.py`: the reference model, which is derived from
                 the actions, against the hardware, which is not. Catches
                 anything that moves WHICH ELEMENT is touched or WHAT
                 ARITHMETIC is applied -- none of which any work count sees.

The split between the last two is the finding. A wrong action is caught by
`--check` alone exactly when it changes what a unit does per issue, and needs
a program and the hardware when it changes only what the instruction means.
A mutant nothing catches is a hole, and the table has to say so.

    python mutate_actions.py                 # every mutant (~3 min)
    python mutate_actions.py --quick         # skip the stress level (~40 s)
    python mutate_actions.py vadd_one_read   # just these
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(HERE, ".mutants-actions")
DOC = os.path.join("docs", "source", "designs", "tinytpu_isa_spec.rst")


def _opcode(spec, name):
    return next(o for o in spec["opcodes"] if o["name"] == name)


def _action(spec, name, index):
    return _opcode(spec, name)["actions"][index]


def _port(spec, unit, port):
    u = next(u for u in spec["units"]["list"] if u["name"] == unit)
    return next(p for p in u["ports"] if p["name"] == port)


# (name, what it models, mutation, what SHOULD catch it and why)
MUTANTS = [
    ("none", "control: the declaration unmodified",
     lambda s: None, None),

    # --- the composition is not a machine: the rule alone settles these ---
    ("accu_unknown_port", "vadd reads through a port `accu` does not have",
     lambda s: _action(s, "vadd", 0).__setitem__("port", "ar.read2"),
     "rule"),
    ("vadd_value_from_nowhere", "vadd writes a value nothing produces",
     lambda s: _action(s, "vadd", 3).__setitem__("args", ["nothing"]),
     "rule"),
    ("mm_skips_the_channel", "accu takes the array's product without receiving it",
     lambda s: _action(s, "mm", 11).__setitem__("args", ["carried", "psum"]),
     "rule"),
    ("mvout_write_unowned", "mvout writes a memory the machine does not declare",
     lambda s: _action(s, "mvout", 4).__setitem__("state", "D"),
     "rule"),

    # --- the cost moved: the design states it independently ---
    ("vadd_one_read", "vadd declared with one source read, not two",
     lambda s: _opcode(s, "vadd")["actions"].pop(1),
     "rule"),

    # --- the cost moved: the design states it independently ---
    ("mm_weights_one_short", "spm puts T words down wcol for an mm, not T + 1",
     lambda s: _action(s, "mm", 1).__setitem__("count", "T"),
     "spec_check"),
    # NOT CAUGHT, and it used to be: correcting the Action cost model removed
    # this catch, because the old catch was an artefact. The ALU's width is
    # the one hardware fact `vaddrelu` rests on that no work count can see --
    # measurable against synthesis and nowhere else. Entry 6 of
    # dev/records/tinytpu/measured_negatives.rst has the reasoning.
    ("accu_alu_narrowed", "accu's ALU chains one lane op a step, not two -- "
                          "no work count can see it; only synthesis can",
     lambda s: _port(s, "accu", "alu").__setitem__("physical", 1),
     None),
    ("ar_dual_ported", "the accumulator file and its port both read twice a cycle",
     lambda s: (next(m for m in s["memories"] if m["name"] == "ar")
                .__setitem__("read_ports", 2),
                _port(s, "accu", "ar.read").__setitem__("physical", 2)),
     "spec_check"),
    ("mvout_never_reaches_dram", "mvout declared to stop at the accumulator",
     lambda s: [_opcode(s, "mvout")["actions"].pop() for _ in range(2)],
     "spec_check"),

    # --- the meaning moved, and no count can see it ---
    ("vadd_src2_is_src1", "vadd's second source declared as its first",
     lambda s: _action(s, "vadd", 1).__setitem__("base", "ar_s1"),
     "stress"),
    ("vadd_subtracts", "vadd declared to subtract",
     lambda s: _action(s, "vadd", 2).__setitem__("compute", "sub"),
     "stress"),
    ("vaddrelu_no_rectify", "the fused instruction declared without its rectify",
     lambda s: _action(s, "vaddrelu", 3).__setitem__("compute", "add"),
     "stress"),
    ("mm_acc_unpredicated", "mm declared to read its accumulate base always",
     lambda s: _action(s, "mm", 10).pop("when"),
     "stress"),
    ("dma_ld_no_col_block", "dma_ld declared to take lane 0 of the DRAM row",
     lambda s: _action(s, "dma_ld", 0).__setitem__("offset", "0"),
     "stress"),
    ("mvout_dst_is_src", "mvout declared to write DRAM row ar0, not dram_row0",
     lambda s: _action(s, "mvout", 4).__setitem__("base", "ar0"),
     "stress"),
]

LEVELS = ("rule", "spec_check", "stress")


def sandbox(name, mutate):
    """A copy of this directory and the ISA doc, with the spec mutated."""
    root = os.path.join(WORK, name)
    shutil.rmtree(root, ignore_errors=True)
    here = os.path.join(root, "examples", "tinytpu")
    os.makedirs(os.path.dirname(here), exist_ok=True)
    shutil.copytree(HERE, here, ignore=shutil.ignore_patterns(
        ".mutants*", ".scratch", "__pycache__", "*.log", "logs", "gemmini",
        "act", "csynth_reports", "asic_synthesis", "rtl_handoff", "impact"))
    shutil.copy(os.path.join(REPO, "examples", "__init__.py"),
                os.path.join(root, "examples", "__init__.py"))
    os.makedirs(os.path.join(root, os.path.dirname(DOC)), exist_ok=True)
    shutil.copy(os.path.join(REPO, DOC), os.path.join(root, DOC))
    path = os.path.join(here, "isa_spec.json")
    with open(path) as f:
        spec = json.load(f)
    mutate(spec)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    return root, here


def run(root, here, script, args, marker, timeout):
    """-> (True if the marker appeared, the combined output)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, REPO])
    env.setdefault("TPU_MAXDIM", "16")
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(here, script)] + args,
            capture_output=True, text=True, timeout=timeout, env=env,
            cwd=here)
    except subprocess.TimeoutExpired:
        return False, "TIMED OUT"
    return marker in out.stdout, out.stdout + out.stderr


def verdict(name, mutate, quick):
    root, here = sandbox(name, mutate)
    caught = {}
    # 1. the rule, before anything is generated: does the machine build?
    ok, log = run(root, here, "gen_isa.py", ["--write"], "wrote", 300)
    caught["rule"] = (not ok) and "ActionError" in log
    if not ok and not caught["rule"]:
        return "GENERATOR ERROR", log
    if caught["rule"]:
        return "rule", log
    # 2. the design, against what the actions now imply.
    ok, log = run(root, here, "gen_isa.py", ["--check"], "ISA OK", 900)
    caught["spec_check"] = not ok
    if caught["spec_check"]:
        return "spec_check", log
    if quick:
        return "not caught (stress not run)", log
    # 3. the hardware, against a reference model derived from the actions.
    ok, log = run(root, here, "stress_isa.py", [], "STRESS OK", 900)
    caught["stress"] = not ok
    return ("stress" if caught["stress"] else "NOT CAUGHT"), log


def main(argv):
    quick = "--quick" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    table = [m for m in MUTANTS if not wanted or m[0] in wanted]
    os.makedirs(WORK, exist_ok=True)
    print(f"{'mutant':26s} {'expected':12s} {'caught by':12s} what it models")
    rows, bad = [], 0
    for name, what, mutate, expect in table:
        got, log = verdict(name, mutate, quick)
        if expect is None:
            # The control, and any mutant nothing static can catch. A hole
            # that is named is a result; a hole that is discovered is a bug.
            ok = got in ("NOT CAUGHT", "not caught (stress not run)")
        else:
            ok = (got == expect) or (quick and expect == "stress"
                                     and got.startswith("not caught"))
        mark = " " if ok else "  <-- "
        bad += 0 if ok else 1
        rows.append((name, expect or "nothing", got, what, ok))
        print(f"{name:26s} {str(expect or 'nothing'):12s} {got:12s}{mark}{what}")
        with open(os.path.join(WORK, f"{name}.log"), "w") as f:
            f.write(log)
    print()
    for level in LEVELS:
        n = sum(1 for r in rows if r[2] == level)
        print(f"  {level:12s} caught {n}")
    if bad:
        print(f"\n  MUTATE ACTIONS FAILED: {bad} mutant(s) not caught by the "
              f"level that should catch them")
        return 1
    blind = sum(1 for r in rows if r[0] != "none" and r[2].startswith("NOT"))
    print(f"\n  MUTATE ACTIONS OK: {len(rows) - 1} wrong declarations, "
          f"{len(rows) - 1 - blind} caught by the level the table predicts "
          f"and {blind} that no static level can catch, named as such"
          + (" (stress level skipped)" if quick else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

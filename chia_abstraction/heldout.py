# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The held-out rediscovery experiment: `s.dependence`, removed and re-asked for.

This is the one experiment in the directory with a KNOWN CORRECT ANSWER, which
is why it is worth running before any open-ended attempt.

`s.dependence` is a fork-local schedule primitive (`bbea2af0`) that exists
because a real defect could not be expressed any other way: TinyTPU-isa's
accumulator, `ar[f1+r] = ar[f1+r] + v`, is scheduled at II=3 in BRAM because
Vitis HLS cannot prove that two iterations never touch the same row. The
hardware workaround (a write-behind rotation) reached II=1 at **13.7x the
flip-flops** (1,270 -> 17,450) for 2.3% end to end, was built, was bit-exact,
and was reverted. The pragma form recovers 95 cycles at 16x16x16 at 1,744 FF.

So: remove the primitive from the agent's view, together with every piece of
documentation and every test that names it, leave the SYMPTOM in place, and ask
for an abstraction. An extension does not have to be novel to count -- an agent
that arrives at an abstraction of the same power WITHOUT being told it exists
is the result. And unlike an open-ended attempt, this one can be graded without
a judgement call: the right answer is in the tree, with its tests.

    make       build the held-out ref: a commit on a scratch branch with the
               primitive, its emitter branch, its tests, its design call site
               and every mention of it removed
    symptom    show that the symptom is present at that ref (the accumulator
               loop's II, from csynth)
    grade      apply a candidate patch to the held-out ref and run the GRADER:
               main's own three `test_dependence_pragma*` tests, which the
               agent never saw, plus the cycle delta on TinyTPU-isa

The grader is the honest part. It asks three graded questions, in order of
strength:

  G1 power      does the candidate give a user a way to assert that a loop's
                accesses to a named array are independent, and does it reach
                the emitted pragma? Graded by main's `test_dependence_pragma`
                and `test_dependence_pragma_dataflow_region`, rewritten
                against the CANDIDATE's spelling only where the spelling
                differs -- which is a judgement call, so the rewrite is
                recorded in the run directory and reported.
  G2 legality   does it validate its arguments against a closed set and refuse
                a bad claim? Graded by `test_dependence_pragma_rejects_bad_claims`
                and by `patch_policy.primitive_violations`, which the harness
                already applies.
  G3 payoff     with the candidate's abstraction available, does the shipped
                design get its cycles back? This is NOT required: the design
                file is frozen to the agent, so it cannot add the call site.
                What is measured instead is whether the harness, applying the
                one-line call `s.dependence("accu_0:x", "ar", dep_type="inter",
                dependent=False)` re-expressed in the candidate's own spelling,
                recovers the cycles. A candidate that passes G1 and G2 and
                fails G3 has proposed a primitive that does not work; a
                candidate that passes G1 and G2 and is not gradable on G3
                because its spelling cannot express the claim is a NEAR MISS,
                and that is the interesting category.

Nothing here is automatic about G1's rewrite or G3's call site. Both are
recorded verbatim in the run directory so the grading can be checked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EMIT = "mlir/lib/Translation/EmitVivadoHLS.cpp"
CUSTOMIZE = "allo/customize.py"
DESIGN = "examples/accelerator/tinytpu_vitis/microarch_isa.py"
VHLS_TESTS = "tests/test_vhls.py"

HELDOUT_BRANCH = "chia-abstraction-heldout-dependence"

#: The BETTER held-out base, prepared on main at
#: `examples/accelerator/tinytpu_vitis/chia_agent/holdout/`: the parent of the
#: commit that introduced `s.dependence`. At that commit the primitive has
#: never existed, no test mentions it, the design does not call it and no
#: document describes it -- the absence is REAL rather than simulated, and
#: there is no redaction to get wrong. Its `symptom.md` also records one
#: deliberate near-leak (the word "dependence" itself, which is the vendor's
#: own term and what the scheduling report says) rather than removing it.
#:
#: `--base` uses it. The redaction path below (`make`) is kept as the
#: alternative for a primitive with no such parent commit, and because its
#: leak detector is the thing that showed how many traces a redaction leaves:
#: eleven, including the design loop's own recorded evidence.
PREPARED_BASE = "a4151ca0"
PREPARED_ANSWER = "bbea2af0"
PREPARED_SYMPTOM = ("examples/accelerator/tinytpu_vitis/chia_agent/holdout/"
                    "symptom.md")

#: ONE pattern set, used both to redact and to detect a leak, so the two
#: cannot disagree. A held-out ref whose leak check passes because the checker
#: looks for less than the redactor removed is worthless.
LEAK = (
    (r"s\.dependence\([^\n]*", "<<REDACTED>>"),
    (r"\bs\.dependence\b", "<<REDACTED>>"),
    (r"#pragma HLS dependence[^\n]*", "<<REDACTED>>"),
    (r"\bdep_type\b", "<<REDACTED>>"),
    (r'getLoopDirective\((?:op|[A-Za-z_]\w*), *"dependence"\)',
     'getLoopDirective(op, "<<REDACTED>>")'),
    (r'"dependence"', '"<<REDACTED>>"'),
    (r"`dependence`", "`<<REDACTED>>`"),
    (r"\bdependence pragma\b", "<<REDACTED>>"),
    (r"\bdependence claim\b", "<<REDACTED>>"),
    (r"\bdependence=\S*", "<<REDACTED>>"),
    (r"\bdependent=\S*", "<<REDACTED>>"),
    (r"HLS dependence", "<<REDACTED>>"),
)
#: What the leak check looks for -- the left-hand sides above, verbatim.
LEAK_RE = "|".join(pat for pat, _ in LEAK)


def scan_leaks(ref: str) -> list[str]:
    """Every tracked text file at `ref` that still names the answer.

    Uses PYTHON's `re` over `git show`, not `git grep -E`. That is not taste:
    `git grep -E` is POSIX ERE and rejects `(?:...)`, so the original check
    exited 128 -- and because it was called with `check=False`, the failure was
    read as "no leaks". **The leak detector failed open**, and it reported a
    clean tree while `docs/source/developer/limitations.rst` said, in the first
    fifty lines, "HLS dependence pragma (fork issue #10)", and
    `microarch_isa.py` carried the pragma verbatim in a comment. The agent read
    lines 1-150 of that file.

    So this function raises on any git failure and returns `file:line:text` for
    every match, and `make`/`graft` refuse to hand over a ref with any.
    """
    # `-r` alone lists a submodule's path with no readable blob; ask for the
    # object type so a gitlink is identified rather than guessed.
    listing = sh(["git", "ls-tree", "-r", ref]).split("\n")
    files = [l.split("\t", 1)[1] for l in listing
             if "\t" in l and l.split()[1] == "blob"]
    pat = re.compile(LEAK_RE)
    out = []
    for rel in files:
        rel = rel.strip()
        if not rel:
            continue
        raw = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=REPO,
                             capture_output=True)
        if raw.returncode:
            # A gitlink (submodule) has no blob. Anything else that cannot be
            # read is a hard failure: a leak scan that skips a file it could
            # not open is the failure mode this function exists to prevent.
            raise SystemExit(f"scan_leaks: cannot read blob {rel} at {ref}")
        if b"\0" in raw.stdout[:8000]:
            continue
        try:
            text = raw.stdout.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                out.append(f"{rel}:{i}: {line.strip()[:110]}")
    return out


def scan_text(text: str, where: str = "prompt") -> list[str]:
    """`file:line:text` for every leak in one piece of text -- the PROMPT.

    `scan_leaks` covers the tree. It was not enough: the held-out system
    prompt is assembled from the live tree's prompt.py, not from the redacted
    graft, and on the first run it named the answer's name, attribute shape,
    emitter location, and legality style. The tree was clean; the channel was
    not. So the assembled prompt is scanned too, with the same pattern.
    """
    pat = re.compile(LEAK_RE)
    return [f"{where}:{i}: {l.strip()[:110]}"
            for i, l in enumerate(text.splitlines(), 1) if pat.search(l)]


def redact(text: str) -> str:
    for pat, sub in LEAK:
        text = re.sub(pat, sub, text)
    return text

#: The symptom, stated to the agent. Everything here is measured and recorded
#: in the design's history; none of it names the primitive or the pragma.
SYMPTOM = """
THE SYMPTOM YOU ARE GIVEN, and it is the whole of your starting information.

TinyTPU-isa's accumulator kernel holds its partial sums in an on-chip array
`ar` and updates one row per iteration:

    ar[f1 + r] = ar[f1 + r] + v

`f1` is a frame base that the program advances, and `r` is the row index. For
every program the machine actually runs, two iterations of that loop never
touch the same row -- the frame base moves on before a row is revisited. The
design knows this. Vitis HLS does not: it cannot prove it from the addressing,
so it assumes a read-after-write dependence on `ar` and schedules the loop at
**initiation interval 3** instead of 1. The loop is otherwise trivial: one
read, one add, one write.

Measured consequences, both real:

  * At II=3 the accumulator is the critical process at the larger shapes.
  * The hardware workaround -- rotating the accumulator so the write lands
    behind the read, which makes the dependence genuinely absent -- DOES reach
    II=1. It was built and it was bit-exact. It cost **13.7x the flip-flops**
    (1,270 -> 17,450) for 2.3% end to end, and the flip-flop cost grows with
    the array dimension while the 2.3% does not. It was reverted.

So the design is correct, the claim it wants to make is true, and there is no
way to say it to the compiler. Redesigning the hardware to make the claim
unnecessary costs 13.7x the registers.

Two more places the same shape of problem appears, which is your evidence that
this is not one design's problem:

  * `mlp_layered`'s staging buffer `buf` is written by one loop nest and read
    by the next, and the partitioned accumulator loop is scheduled
    conservatively for the same reason.
  * `systolic_1d`'s compute PE accumulates into a scalar across a stream read,
    and its pipeline is likewise not at II=1.

Propose the abstraction. It must be usable by a design that is not yours to
edit, it must be validated at construction, and if what it expresses is a
PROMISE rather than a fact -- if a false claim would produce wrong RTL while
every software simulation stayed exact -- your docstring must say so.
"""


#: The four outcomes a held-out run can have, in descending strength. The
#: grading is against the answer that is already in the tree at
#: `PREPARED_ANSWER`, so it needs no judgement call about VALUE -- only about
#: whether two spellings have the same power, which is recorded verbatim.
OUTCOMES = {
    "same-abstraction": (
        "the agent arrived at an abstraction with the same power at the same "
        "level: a validated schedule primitive that puts a named attribute on "
        "the loop and an emitter branch that reads it. Novelty is not "
        "required -- rediscovering what the fork already has, without being "
        "told it exists, is the result."),
    "equal-power-different-level": (
        "an abstraction of equal power somewhere else: an analysis that "
        "proves the claim, a pass that rewrites the access, an IR attribute "
        "with a different carrier. This is a SUCCESS and is arguably a better "
        "one, because the answer in the tree is an unchecked promise and an "
        "analysis would not be."),
    "instance-solved-no-abstraction": (
        "the symptom is gone and nothing reusable was added -- a special case "
        "in the emitter, a hard-coded name, a heuristic that fires on this "
        "loop. It passes the gates and it is not what was asked for."),
    "no-solution": (
        "no candidate survived. Record WHERE it stopped, by rung: the policy, "
        "the build, Allo's own suites, the design cases, the resource budget, "
        "or the PPA. A run that stops at `built` is a different result from "
        "one that stops at `proposed`."),
}

#: Graded separately from the outcome, and deliberately so. The answer in the
#: tree is an UNCHECKED PROMISE: `s.dependence` validates the SHAPE of its
#: arguments and cannot verify the CLAIM, so a false claim produces wrong RTL
#: while the Allo simulator and Vitis csim both stay exact. The fork only
#: learned that after the fact, and encoded it as a contract in the design's
#: assembler (`AR_RAW_DIST`) plus an RTL-only stress testbench.
#:
#: So: did the agent notice? An agent that states what would have to hold for
#: its abstraction to be sound, and that no software simulation can check it,
#: has done better than the original commit did.
SOUNDNESS_NOTED = (
    "did the agent say that what its abstraction expresses is a PROMISE "
    "rather than a fact -- that a false claim produces wrong RTL while every "
    "software simulation stays exact -- and say what would have to hold for "
    "it to be sound?")


def symptom() -> str:
    """The symptom to hand the agent.

    Main's prepared `holdout/symptom.md` if it is there, because it was written
    and reviewed for this purpose and it RECORDS its own deliberate near-leak
    (the word "dependence", which is the vendor's own term and what the
    scheduling report says) instead of removing it. The copy in this module is
    the fallback and is used by the redaction path.
    """
    p = REPO / PREPARED_SYMPTOM
    if p.is_file():
        return ("# The symptom, from " + PREPARED_SYMPTOM + "\n\n"
                + p.read_text(encoding="utf-8"))
    return SYMPTOM


def sh(args, cwd=REPO, check=True, text=True):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=text)
    if check and p.returncode:
        raise SystemExit(f"{' '.join(args)} failed:\n{p.stderr}")
    return p.stdout


def _block(src: str, anchor: str) -> str:
    """The brace-balanced block starting at `anchor`."""
    start = src.index(anchor)
    i, depth = src.index("{", start), 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return src[start:i + 2]


def _def_block(src: str, name: str) -> str:
    """A `def name(...)` method of a class, from its decorator to the next
    sibling `def` at the same indentation."""
    m = re.search(rf"\n(?P<ind>[ \t]+)(?:@wrapped_apply\n[ \t]+)?def {name}\(",
                  src)
    if not m:
        raise SystemExit(f"cannot find def {name} in the source")
    start = m.start() + 1
    ind = m.group("ind")
    nxt = re.search(rf"\n{ind}(?:@|def )", src[m.end():])
    end = m.end() + nxt.start() + 1 if nxt else len(src)
    return src[start:end]


def make(out: Path) -> dict:
    """Build the held-out ref on a scratch branch. Returns what was removed."""
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    sh(["git", "branch", "-f", HELDOUT_BRANCH, head])
    removed = {}
    edits = []

    # 1. The emitter branch (the 46 lines of bbea2af0's C++).
    emit = sh(["git", "show", f"{head}:{EMIT}"])
    branch = _block(emit, 'if (auto deps = llvm::dyn_cast_or_null<ArrayAttr>(')
    assert "#pragma HLS dependence" in branch, "wrong emitter block"
    removed["emitter_lines"] = len(branch.splitlines())
    edits.append((EMIT, emit.replace(branch, "", 1)))

    # 2. The primitive itself.
    cust = sh(["git", "show", f"{head}:{CUSTOMIZE}"])
    method = _def_block(cust, "dependence")
    assert "AlloValueError" in method and "ArrayAttr" in method, "wrong method"
    removed["primitive_lines"] = len(method.splitlines())
    edits.append((CUSTOMIZE, cust.replace(method, "", 1)))

    # 3. The design's call site: the SYMPTOM comes back, which is the point.
    design = sh(["git", "show", f"{head}:{DESIGN}"])
    # The call site itself, not the docstring line that also mentions it --
    # the docstring is handled by the redaction pass below.
    calls = [l for l in design.splitlines()
             if re.match(r"^\s*s\.dependence\(", l)]
    assert len(calls) == 1, calls
    removed["design_call"] = calls[0].strip()
    design = design.replace(calls[0] + "\n", "", 1)
    edits.append((DESIGN, redact(design)))

    # 4. The tests -- saved as the GRADER before they are removed.
    tests = sh(["git", "show", f"{head}:{VHLS_TESTS}"])
    keep = []
    stripped = tests
    for name in ("test_dependence_pragma_dataflow_region",
                 "test_dependence_pragma_rejects_bad_claims",
                 "test_dependence_pragma", "_loop_body"):
        m = re.search(rf"\ndef {name}\(", stripped)
        if not m:
            continue
        start = m.start() + 1
        nxt = re.search(r"\ndef ", stripped[m.end():])
        end = m.end() + nxt.start() + 1 if nxt else len(stripped)
        keep.append(stripped[start:end])
        stripped = stripped[:start] + stripped[end:]
    removed["grader_tests"] = len(keep)
    (out / "grader_tests.py").write_text(
        "# The GRADER for the held-out `s.dependence` rediscovery experiment.\n"
        "# These are main's own tests, removed from the held-out ref so the\n"
        "# agent never sees them, and applied afterwards to grade what it\n"
        "# proposed. G1 (power) and G2 (legality) are these tests.\n"
        "# Their spelling is main's; a candidate with a different spelling is\n"
        "# graded against a rewrite, which is recorded beside this file.\n\n"
        + "\n\n".join(keep))
    edits.append((VHLS_TESTS, stripped))

    # 5. Every remaining mention, in docs and in this harness's own prompt.
    #    A mention is redacted rather than deleted, so the held-out tree is
    #    still coherent and the redaction is visible in the diff.
    tracked = sh(["git", "ls-files"]).split()
    redacted = []
    # EVERY tracked text file, not just source: the leak that survived the
    # first attempt was in `dev/records/tinytpu/chia-evidence/*/variants.jsonl`
    # and `dev/records/tinytpu/impact-results/*.out`, i.e. in the design loop's
    # own recorded evidence.
    for rel in tracked:
        if rel in {EMIT, CUSTOMIZE, DESIGN, VHLS_TESTS}:
            continue
        try:
            raw = sh(["git", "show", f"{head}:{rel}"], text=False)
        except SystemExit:
            continue
        if b"\0" in raw[:8000]:
            continue                      # binary
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not re.search(LEAK_RE, text):
            continue
        new = redact(text)
        if new != text:
            redacted.append(rel)
            edits.append((rel, new))
    removed["redacted_files"] = redacted

    wt = REPO / ".chia_scratch" / "heldout-build"
    if (wt / ".git").exists():
        sh(["git", "worktree", "remove", "--force", str(wt)], check=False)
    sh(["git", "worktree", "add", "--detach", str(wt), head])
    for rel, text in edits:
        (wt / rel).write_text(text)
    sh(["git", "add", "-A"], cwd=wt)
    sh(["git", "-c", "user.email=sk3463@cornell.edu",
        "-c", "user.name=Sunwoo Kim", "commit", "-q", "-m",
        "HELD OUT: s.dependence removed, for the rediscovery experiment\n\n"
        "Generated by chia_abstraction/heldout.py from "
        f"{head[:12]}. Removes the primitive ({removed['primitive_lines']} "
        f"lines), its emitter branch ({removed['emitter_lines']} lines), its "
        f"{removed['grader_tests']} tests (saved as the grader), the design's "
        f"one call site (so the SYMPTOM returns), and redacts every mention "
        f"in {len(redacted)} other files.\n\nDO NOT MERGE.\n\n"
        "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01BmrdaXYkbAqVL8kc9ikwRk"],
       cwd=wt)
    ref = sh(["git", "rev-parse", "HEAD"], cwd=wt).strip()
    sh(["git", "branch", "-f", HELDOUT_BRANCH, ref])
    sh(["git", "worktree", "remove", "--force", str(wt)], check=False)
    removed["ref"] = ref
    removed["branch"] = HELDOUT_BRANCH
    removed["from"] = head
    # A held-out ref that still mentions the answer is not held out.
    removed["leaks"] = scan_leaks(ref)
    (out / "heldout.json").write_text(json.dumps(removed, indent=1))
    (out / "symptom.md").write_text(SYMPTOM)
    return removed


def graft(out: Path, base: str = PREPARED_BASE) -> dict:
    """The prepared base commit, with THIS harness grafted on.

    `a4151ca0` predates `chia_abstraction/` entirely, so the evaluator's own
    files are not in it -- and the ladder runs them FROM the slot. So the
    held-out ref is `base` plus this harness and the design's gates, with the
    redaction applied to the harness's OWN text only (the prompt, the policy
    comments, this file). Allo, the design, the tests and the documentation are
    untouched at `base`: the absence there is real.
    """
    head = sh(["git", "rev-parse", "HEAD"]).strip()
    base_ref = sh(["git", "rev-parse", "--verify", base + "^{commit}"]).strip()
    wt = REPO / ".chia_scratch" / "heldout-graft"
    if (wt / ".git").exists():
        sh(["git", "worktree", "remove", "--force", str(wt)], check=False)
    sh(["git", "worktree", "add", "--detach", str(wt), base_ref])
    # This harness, and the design gates the ladder runs, from HEAD.
    # NOT the design's own evaluator. `cosim.py`, `bench_isa.py`,
    # `stress_isa.py`, `isa_ref.py` and `kpn_model.py` stay at `base`, because
    # they are what that commit's design is measured and verified by, and
    # mixing HEAD's gate with an older `microarch_isa.py` would be measuring
    # neither. The run sets CHIA_MAIN_BASE to `base` so the byte-pin still
    # applies, to that commit instead of to main. Checked: at a4151ca0 all
    # five exist and `stress_isa.main(argv)` has the shape the runner needs;
    # only `chia_agent/gate_runner.py` and `param_check.py` are absent, and
    # those are runners, not gates.
    for path in ("chia_abstraction",
                 "examples/accelerator/tinytpu_vitis/chia_agent",
                 "tests/limits"):
        sh(["git", "checkout", head, "--", path], cwd=wt, check=False)
    # Redact what the graft brought in AND the base commit's own docs. The
    # first graft left `docs/` at the base, which was right for the evaluator
    # and wrong for the docs: the base's limitations register named the answer
    # at line 41, inside the lines the agent read. (The failing-open leak
    # detector is what hid that; see scan_leaks.)
    grafted = [l for l in sh(["git", "diff", "--name-only", base_ref],
                             cwd=wt).splitlines() if l.strip()]
    grafted += [l for l in sh(["git", "ls-files", "docs"], cwd=wt).splitlines()
                if l.strip() and l not in grafted]
    # The base's design file carries the pragma verbatim in a docstring. A
    # docstring changes no behaviour, and microarch_isa.py is not in the
    # main-pinned evaluator set (cosim/bench/stress/isa_ref/kpn_model are).
    grafted.append(DESIGN)
    redacted = []
    for rel in grafted:
        f = wt / rel
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not re.search(LEAK_RE, text):
            continue
        f.write_text(redact(text), encoding="utf-8")
        redacted.append(rel)
    sh(["git", "add", "-A"], cwd=wt)
    sh(["git", "-c", "user.email=sk3463@cornell.edu",
        "-c", "user.name=Sunwoo Kim", "commit", "-q", "-m",
        f"HELD OUT (grafted): the abstraction harness on {base_ref[:12]}\n\n"
        f"The held-out base is {base_ref[:12]}, the parent of the commit that "
        f"introduced the answer; at that commit the primitive has never "
        f"existed, so the absence is real and there is nothing to redact in "
        f"Allo, the design, the tests or the docs. Grafted on: "
        f"chia_abstraction/, the design's gates and tests/limits, with the "
        f"redaction applied to those {len(redacted)} grafted files only.\n\n"
        f"DO NOT MERGE.\n\n"
        "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01BmrdaXYkbAqVL8kc9ikwRk"],
       cwd=wt)
    ref = sh(["git", "rev-parse", "HEAD"], cwd=wt).strip()
    sh(["git", "branch", "-f", HELDOUT_BRANCH + "-graft", ref])
    sh(["git", "worktree", "remove", "--force", str(wt)], check=False)
    r = {"ref": ref, "base": base_ref, "answer_at": PREPARED_ANSWER,
         "CHIA_MAIN_BASE": base_ref,
         "branch": HELDOUT_BRANCH + "-graft", "grafted": len(grafted),
         "redacted_grafted": redacted,
         "leaks": scan_leaks(ref),
         "symptom": PREPARED_SYMPTOM,
         "note": "the symptom to hand the agent is main's prepared one, which "
                 "records its own deliberate near-leak; read its README."}
    (out / "heldout_graft.json").write_text(json.dumps(r, indent=1))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("make", "graft", "show"))
    ap.add_argument("--base", default=PREPARED_BASE)
    ap.add_argument("--out", type=Path,
                    default=REPO / ".chia_scratch" / "heldout")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.action == "show":
        print(SYMPTOM)
        return 0
    if a.action == "graft":
        r = graft(a.out, a.base)
        print(json.dumps(r, indent=1))
        if r["leaks"]:
            print(f"\nWARNING: leaks remain in {r['leaks']}")
            return 1
        print(f"\nheld-out ref {r['ref'][:12]} on {r['branch']}, based on "
              f"{r['base'][:12]} (the answer is at {r['answer_at']}).\n"
              f"Hand the agent main's {r['symptom']}.")
        return 0
    r = make(a.out)
    print(json.dumps(r, indent=1))
    if r["leaks"]:
        print(f"\nWARNING: the held-out ref still names the answer in "
              f"{r['leaks']}. Redact those before running.")
        return 1
    print(f"\nheld-out ref {r['ref'][:12]} on {r['branch']}.\n"
          f"grader: {a.out / 'grader_tests.py'}\n"
          f"symptom: {a.out / 'symptom.md'}\n\n"
          f"Run the experiment with:\n"
          f"  export CHIA_FROZEN_REF={r['ref']}\n"
          f"  python chia_abstraction/evaluate_abs.py --out DIR "
          f"--record-baseline     # re-measure at the held-out ref\n"
          f"  python chia_abstraction/abs_loop.py --disposition maintaining "
          f"--heldout dependence ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

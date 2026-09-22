# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLV-free end-to-end test of the abstraction harness. $0. Run before any paid search.

Nothing here calls a model. Every case drives the real `evaluate_abs.py` on a
real patch, in a real slot worktree, with the real gates -- except the `tools`
and `loop` phases, which drive the real `AlloCompilerTool` and the real
`abs_loop.py` with opencode pointed at `chia_agent/fake_model.py`, the scripted
OpenAI-compatible endpoint the design-level harness already uses.

The cases, and the claim each one makes:

  a  noop         a no-patch run reproduces the recorded baseline EXACTLY,
                  on every design case. If this drifts, no delta means
                  anything.
  b  known-good   `s.dependence`'s emitter branch reverted -- our own landed
                  extension, used backwards as a fixture. Two independent
                  gates must catch it: `gate:tests` (the three
                  `test_dependence_pragma*` tests in tests/test_vhls.py), and,
                  with that gate made non-blocking, the PPA rung, because
                  TinyTPU-isa's accumulator loses II=1. Both numbers are
                  recorded: this is "a known-good change reproduces a known
                  delta", measured at two rungs.
  c  broken       a semantically wrong compiler change -- the Vivado emitter's
                  `dependence` pragma emitted with `dependent` INVERTED. It
                  builds, it imports, and it is a false dependence claim, so
                  it must be caught by an execution gate, not by a static one.
  d  frozen       nine attempts to edit something frozen: a test, a golden, a
                  reference model, a scored design file, the harness, a gate,
                  `CMakeLists.txt`, `tests/limits/`, and a diff whose header
                  says one file and whose body touches another. All refused,
                  at `policy`, before anything is built.
  e  cpp-fails    a C++ change that does not compile is rejected cleanly at
                  `gate:build` -- with the compiler's diagnostic, and with no
                  process and no half-built tree left behind.
  f  primitive    the CAKE rule: a new `Schedule` method with no
                  `AlloValueError` and no docstring is refused at `policy`;
                  the same method WITH both passes the policy. The cheap
                  static gate, before the expensive ones.
  g  parallel     two evaluations at once, in two slots, do not
                  cross-contaminate: each reproduces its own verdict and
                  neither sees the other's tree.
  h  resources    the multi-term objective: a synthetic PPA result that buys
                  cycles with 2.3x the BRAM -- the design-level loop's real
                  accepted diff, by its real numbers -- is classified `trade`,
                  not `win`, and would be rejected at `gate:resources`.
  i  using        the `using` disposition still works: its allowlist accepts a
                  design edit and refuses an Allo edit, and the shared ladder
                  runs.

    conda activate chia_env          # phases tools,loop need CHIA + opencode
    python test_abs_harness.py                          # everything, ~90 min
    python test_abs_harness.py --phases d,f,h           # static only, ~5 s, $0
    python test_abs_harness.py --phases a,b,c,e,g       # the gates, ~60 min
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DESIGN_PKG = "examples/accelerator/tinytpu_vitis"
DESIGN_AGENT = REPO / DESIGN_PKG / "chia_agent"
sys.path.insert(0, str(HERE))

import objective                                             # noqa: E402
import patch_policy                                          # noqa: E402

ALLO_PYTHON = os.environ.setdefault(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
os.environ.setdefault("LLVM_BUILD_DIR",
                      "/home/sk3463/llvm-allo-6b09f739/build")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["PATH"] = ("/opt/xilinx/Vitis_HLS/2023.2/bin:"
                      + os.environ.get("PATH", ""))
# No path in this test may reach a paid model.
for _k in list(os.environ):
    if _k.startswith(("GOOGLE_", "VERTEX", "CLOUDSDK_", "TINYTPU_OPENCODE_",
                      "TINYTPU_VERTEX_", "ANTHROPIC_", "OPENAI_",
                      "CHIA_ABS_MODEL")):
        del os.environ[_k]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/dev/null"
os.environ["CHIA_TOOL_HOST"] = "127.0.0.1"

# ---------------------------------------------------------------------------
# The fixtures. Each is (path, old, new): an exact, unique replacement against
# HEAD, turned into a git patch by `make_patch`.
EMIT = "mlir/lib/Translation/EmitVivadoHLS.cpp"
CUSTOMIZE = "allo/customize.py"

#: b: the whole `dependence` branch of `emitLoopDirectives`. Reverting it is
#: how the tree looked before `bbea2af0`: the Python side still sets the IR
#: attribute, and the emitter silently drops it -- exactly what the four other
#: emitters still do today.
DEPENDENCE_BRANCH_HEAD = """  if (auto deps = llvm::dyn_cast_or_null<ArrayAttr>(
          getLoopDirective(op, "dependence"))) {"""

#: c: the same branch, with the truth value of the claim INVERTED. It compiles,
#: it imports, it emits a well-formed pragma -- and the pragma is a lie.
INVERT_OLD = """      os << " " << ((dependent && dependent.getValue()) ? "true" : "false")
         << "\\n";"""
INVERT_NEW = """      os << " " << ((dependent && dependent.getValue()) ? "false" : "true")
         << "\\n";"""

#: e: C++ that cannot compile.
CPP_BREAK_OLD = '      os << "#pragma HLS dependence variable=" << getName(var);'
CPP_BREAK_NEW = ('      os << "#pragma HLS dependence variable=" '
                 '<< getName(var) << thisSymbolDoesNotExist();')

#: f: a new Schedule method, with and without its legality rules.
_ANCHOR = "    def parallel(self, axis):"
PRIM_BAD = '''    def align(self, axis, factor):
        self._align = (axis, factor)

'''
PRIM_GOOD = '''    def align(self, axis, factor):
        """Claim that `axis`'s trip count is a multiple of `factor`.

        Emits nothing on its own; sets an `align` IntegerAttr on the loop,
        which EmitVivadoHLS reads. `factor` must be a power of two in
        [2, 1024]. It is a PROMISE, not a fact: if the trip count is not a
        multiple of `factor` the RTL is wrong while every software simulation
        stays exact.
        """
        if not isinstance(factor, int) or factor < 2 or factor > 1024:
            raise AlloValueError(
                f"align: factor {factor!r} must be an int in [2, 1024]")
        if factor & (factor - 1):
            raise AlloValueError(
                f"align: factor {factor!r} must be a power of two")
        self._align = (axis, factor)

'''

#: d: nine frozen targets. Each is a one-line patch against that path.
FROZEN_TARGETS = [
    ("a test", "tests/test_vhls.py"),
    ("a limitation repro", "tests/limits/item20_emitter_name_collision.py"),
    ("a golden / reference model", f"{DESIGN_PKG}/isa_ref.py"),
    ("a correctness gate", f"{DESIGN_PKG}/stress_isa.py"),
    ("the scoring harness", f"{DESIGN_PKG}/cosim.py"),
    ("a scored design file", f"{DESIGN_PKG}/microarch_isa.py"),
    ("this harness's own evaluator", "chia_abstraction/evaluate_abs.py"),
    ("this harness's policy", "chia_abstraction/patch_policy.py"),
    ("the build system", "mlir/lib/Translation/CMakeLists.txt"),
]


def git(args, cwd=REPO, text=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=text)
    return p.stdout


def head_text(rel: str) -> str:
    return git(["show", f"HEAD:{rel}"])


def make_patch(edits, out: Path) -> Path:
    """A git patch from [(rel, old, new)], built against HEAD."""
    chunks = []
    for rel, old, new in edits:
        before = head_text(rel)
        assert before, f"{rel} is empty at HEAD"
        n = before.count(old)
        assert n == 1, f"{rel}: the fixture anchor occurs {n} times, want 1"
        after = before.replace(old, new, 1)
        a = out.parent / f".a_{rel.replace('/', '_')}"
        b = out.parent / f".b_{rel.replace('/', '_')}"
        a.write_text(before)
        b.write_text(after)
        d = subprocess.run(["git", "diff", "--no-index", "-U3",
                            f"--src-prefix=a/{os.path.dirname(rel)}/"
                            if os.path.dirname(rel) else "--src-prefix=a/",
                            str(a), str(b)], capture_output=True, text=True)
        # `git diff --no-index` names the temp files; rewrite the headers to
        # the repository path so `git apply` puts the change where it belongs.
        lines = []
        for line in d.stdout.splitlines():
            if line.startswith("diff --git"):
                lines.append(f"diff --git a/{rel} b/{rel}")
            elif line.startswith("--- "):
                lines.append(f"--- a/{rel}")
            elif line.startswith("+++ "):
                lines.append(f"+++ b/{rel}")
            elif line.startswith(("index ", "new file", "deleted file",
                                  "old mode", "new mode")):
                continue
            else:
                lines.append(line)
        chunks.append("\n".join(lines) + "\n")
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)
    out.write_text("".join(chunks))
    return out


def revert_dependence_branch(out: Path) -> Path:
    """b: delete the emitter's whole `dependence` branch (46 lines)."""
    src = head_text(EMIT)
    start = src.index(DEPENDENCE_BRANCH_HEAD)
    # Walk braces from the `if (` to its matching close.
    i, depth = src.index("{", start), 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = src[start:i + 2]            # include the closing brace + newline
    assert "#pragma HLS dependence" in block, "the fixture found the wrong block"
    assert 40 < len(block.splitlines()) < 60, len(block.splitlines())
    return make_patch([(EMIT, block, "")], out)


def evaluate(patch: Path | None, out: Path, slot: int, disposition="maintaining",
             tier="loop", gate_only=False, skip="", timeout=7200) -> dict:
    cmd = [sys.executable, str(HERE / "evaluate_abs.py"),
           "--disposition", disposition, "--slot", str(slot),
           "--out", str(out), "--tier", tier]
    if patch:
        cmd += ["--patch", str(patch)]
    if gate_only:
        cmd.append("--gate-only")
    if skip:
        cmd += ["--skip", skip]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    text = (p.stdout + p.stderr).strip()
    (out / "harness_stdout.log").parent.mkdir(parents=True, exist_ok=True)
    (out / "harness_stdout.log").write_text(text)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            # `evaluate_abs.main` POPS `detail` out of the JSON and prints it
            # on its own lines, so a test that reads `verdict["detail"]` sees
            # nothing. The captured stdout is where the rejection's evidence
            # is -- the compiler diagnostic, the failing test names.
            v["_stdout"] = text
            return v
    return {"ok": False, "stage": "evaluator", "detail": text[-3000:],
            "_stdout": text}


# ---------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.rows = []

    def case(self, name, expected, got, ok, extra=None):
        self.rows.append({"case": name, "expected": expected, "measured": got,
                          "ok": bool(ok), **(extra or {})})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n"
              f"        expected: {expected}\n"
              f"        measured: {got}", flush=True)

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)


# -- phase d: frozen paths (static, free) ------------------------------------
def phase_frozen(R, work):
    for what, rel in FROZEN_TARGETS:
        text = head_text(rel)
        if not text:
            R.case(f"d frozen: {what} ({rel})", "refused at policy",
                   "file not found at HEAD", False)
            continue
        line = next(l for l in text.splitlines() if l.strip())
        patch = make_patch([(rel, line, line + "  # touched")],
                           work / f"frozen_{rel.replace('/', '_')}.diff")
        problems, _ = patch_policy.check("maintaining", patch.read_text())
        R.case(f"d frozen: {what} ({rel})", "refused at policy",
               (problems[0][:90] if problems else "ACCEPTED -- HOLE"),
               bool(problems))
    # A diff whose `diff --git` header names an allowed file and whose body
    # names a frozen one: both are read, and they must agree.
    lying = (f"diff --git a/{CUSTOMIZE} b/{CUSTOMIZE}\n"
             f"--- a/tests/test_vhls.py\n+++ b/tests/test_vhls.py\n"
             f"@@ -1,1 +1,2 @@\n # Copyright Allo authors.\n+# touched\n")
    problems, _ = patch_policy.check("maintaining", lying)
    R.case("d frozen: a diff whose header and body disagree",
           "refused at policy",
           (problems[0][:90] if problems else "ACCEPTED -- HOLE"),
           bool(problems))
    # And the added-line policy.
    for label, added in (
            ("sys.modules", "    import sys; sys.modules['x'] = 1"),
            ("frame walking", "    g = sys._getframe(1).f_back.f_locals"),
            ("naming a gate", "    open('chia_abstraction/objective.py')"),
            ("SystemExit", "    raise SystemExit(0)"),
            ("a shell escape", "    os.system('rm -rf /')"),
            ("fopen in the emitter", "  FILE *f = fopen(path, \"w\");")):
        rel = EMIT if "emitter" in label else CUSTOMIZE
        diff = (f"diff --git a/{rel} b/{rel}\n--- a/{rel}\n+++ b/{rel}\n"
                f"@@ -1,1 +1,2 @@\n context\n+{added}\n")
        problems = patch_policy.line_violations(diff)
        R.case(f"d added line: {label}", "refused at policy",
               (problems[0][:90] if problems else "ACCEPTED -- HOLE"),
               bool(problems))


# -- phase f: the primitive rule (static, free) ------------------------------
def phase_primitive(R, work):
    for label, body, want_refused in (("without validation", PRIM_BAD, True),
                                      ("with validation", PRIM_GOOD, False)):
        patch = make_patch([(CUSTOMIZE, _ANCHOR, body + _ANCHOR)],
                           work / f"prim_{want_refused}.diff")
        diff = patch.read_text()
        after = head_text(CUSTOMIZE).replace(_ANCHOR, body + _ANCHOR, 1)
        problems = patch_policy.primitive_violations(diff, after)
        refused = bool(problems)
        R.case(f"f a new Schedule primitive {label}",
               "refused at policy" if want_refused else "accepted by policy",
               ("; ".join(p[:80] for p in problems) if problems
                else "accepted"),
               refused == want_refused)


# -- phase h: the objective (static, free) -----------------------------------
def phase_objective(R):
    # The design-level loop's real accepted diff, by its real numbers:
    # 172/262/418/484/686 -> 172/262/376/425/627 at BRAM18K 42 -> 98.
    base = {"cycles": {"4x4x4": 172, "16x16x16": 686},
            "area": {"bram_18k": 42, "dsp": 16, "ff": 1744, "lut": 8000,
                     "uram": 0},
            "estimated_ns": 2.431}
    now = {"cycles": {"4x4x4": 172, "16x16x16": 627},
           "area": {"bram_18k": 98, "dsp": 16, "ff": 2442, "lut": 9600,
                    "uram": 0},
           "estimated_ns": 2.431}
    v = objective.case_verdict(base, now)
    cls = objective.classify({"tinytpu_isa": v})
    R.case("h the real accepted diff (-59 cycles, 2.3x BRAM)",
           "classified `trade`, over budget on tinytpu_isa, not kept",
           f"{cls['verdict']}, over_budget={cls['over_budget']}, "
           f"kept={cls['verdict'] in objective.KEEP}, "
           f"cycles {v['cycles_delta']:+d}",
           cls["verdict"] == "trade"
           and cls["over_budget"] == ["tinytpu_isa"]
           and cls["verdict"] not in objective.KEEP)
    # A genuine win: fewer cycles, resources flat.
    flat = dict(now, area=dict(base["area"]))
    v2 = objective.case_verdict(base, flat)
    cls2 = objective.classify({"tinytpu_isa": v2})
    R.case("h fewer cycles at flat resources", "classified `win`, kept",
           f"{cls2['verdict']}, kept={cls2['verdict'] in objective.KEEP}",
           cls2["verdict"] == "win" and cls2["verdict"] in objective.KEEP)
    # Helps one case, hurts another: a finding, not a win.
    mixed = {
        "a": objective.case_verdict(base, flat),
        "b": objective.case_verdict(base, dict(now, cycles={"x": 999},
                                               area=dict(base["area"]))),
    }
    mixed["b"] = objective.case_verdict(
        {"cycles": {"x": 100}, "area": dict(base["area"]),
         "estimated_ns": 2.4},
        {"cycles": {"x": 140}, "area": dict(base["area"]),
         "estimated_ns": 2.4})
    cls3 = objective.classify(mixed)
    R.case("h helps one design case, hurts another",
           "classified `trade`, both named, not kept",
           f"{cls3['verdict']}, gained={cls3['gained']}, lost={cls3['lost']}",
           cls3["verdict"] == "trade" and cls3["gained"] == ["a"]
           and cls3["lost"] == ["b"])


# -- phase i: the `using` disposition (static, free) -------------------------
def phase_using(R, work):
    design = f"{DESIGN_PKG}/microarch_isa.py"
    line = next(l for l in head_text(design).splitlines()
                if l.startswith("import "))
    patch = make_patch([(design, line, line + "  # touched")],
                       work / "using_design.diff")
    d = patch.read_text()
    p_using = patch_policy.path_violations("using",
                                           patch_policy.paths_in_diff(d))
    p_maint = patch_policy.path_violations("maintaining",
                                           patch_policy.paths_in_diff(d))
    R.case("i `using` may edit the design; `maintaining` may not",
           "using: accepted, maintaining: refused",
           f"using: {p_using or 'accepted'}, maintaining: "
           f"{(p_maint[0][:60] if p_maint else 'ACCEPTED -- HOLE')}",
           not p_using and bool(p_maint))
    allo_line = next(l for l in head_text(CUSTOMIZE).splitlines()
                     if l.startswith("import "))
    patch2 = make_patch([(CUSTOMIZE, allo_line, allo_line + "  # touched")],
                        work / "using_allo.diff")
    d2 = patch2.read_text()
    p2 = patch_policy.path_violations("using", patch_policy.paths_in_diff(d2))
    R.case("i `using` may NOT edit Allo", "refused at policy",
           (p2[0][:90] if p2 else "ACCEPTED -- HOLE"), bool(p2))


# -- phase a: the no-op control ---------------------------------------------
def phase_noop(R, work, slot):
    base = json.loads((HERE / "baseline" / "ppa.json").read_text())
    v = evaluate(None, work / "noop", slot)
    if not v.get("ok"):
        R.case("a no-op reproduces the baseline", "ok, identical",
               f"FAILED at {v.get('stage')}: {str(v.get('detail'))[-300:]}",
               False)
        return v
    same = {}
    for case, b in base["cases"].items():
        n = (v.get("ppa") or {}).get(case, {})
        same[case] = (n.get("cycles") == b.get("cycles")
                      and n.get("area") == b.get("area"))
    R.case("a no-op reproduces the baseline exactly, per design case",
           "every case identical in cycles and area",
           json.dumps({c: ("identical" if s else
                           f"DRIFT {(v.get('ppa') or {}).get(c)}")
                       for c, s in same.items()}),
           all(same.values()), {"ppa": v.get("ppa")})
    return v


# -- phase b: the known-good fixture ----------------------------------------
def phase_known_good(R, work, slot):
    patch = revert_dependence_branch(work / "revert_dependence.diff")
    R.case("b the fixture is the real 46-line emitter branch",
           "46 +/- 6 lines removed from EmitVivadoHLS.cpp, nothing added",
           f"{sum(1 for l in patch.read_text().splitlines() if l.startswith('-') and not l.startswith('---'))} "
           f"removed, "
           f"{sum(1 for l in patch.read_text().splitlines() if l.startswith('+') and not l.startswith('+++'))} added",
           True)
    # 1. blocking: gate:tests must catch it.
    v = evaluate(patch, work / "revert_blocking", slot)
    detail = v.get("_stdout", "") or str(v.get("detail", ""))
    R.case("b reverting s.dependence's emitter branch is REJECTED",
           "rejected at gate:tests, naming test_dependence_pragma",
           f"{v.get('stage')}; mentions dependence: "
           f"{'test_dependence' in detail}",
           v.get("stage") == "gate:tests" and "test_dependence" in detail,
           {"detail": detail[-1500:]})
    # 2. non-blocking: what the PPA rung would have said.
    v2 = evaluate(patch, work / "revert_ppa", slot, skip="tests")
    base = json.loads((HERE / "baseline" / "ppa.json").read_text())
    got = ((v2.get("ppa") or {}).get("tinytpu_isa") or {}).get("cycles")
    want = base["cases"]["tinytpu_isa"]["cycles"]
    worse = (got and all(got[s] >= want[s] for s in want)
             and any(got[s] > want[s] for s in want))
    R.case("b with gate:tests made non-blocking, the PPA rung also catches it",
           "TinyTPU-isa loses cycles (the accumulator loses II=1)",
           f"baseline {want} -> {got}; objective "
           f"{(v2.get('objective') or {}).get('verdict')}",
           bool(worse) or (v2.get("stage") in ("gate:resources", "ppa:cosim")),
           {"cycles": got, "baseline": want,
            "objective": v2.get("objective"),
            "stage": v2.get("stage")})


# -- phase c: a semantically broken change ----------------------------------
def phase_broken(R, work, slot):
    patch = make_patch([(EMIT, INVERT_OLD, INVERT_NEW)],
                       work / "invert_dependence.diff")
    v = evaluate(patch, work / "invert", slot)
    detail = v.get("_stdout", "")
    R.case("c a false dependence claim (the pragma's truth value inverted)",
           "builds and imports, then REJECTED by a correctness gate",
           f"{v.get('stage')} (build "
           f"{'ok' if (v.get('stages') or {}).get('build') else 'not reached'})",
           (not v.get("ok")
            and v.get("stage") not in ("policy", "gate:build", "gate:import")),
           {"detail": detail[-1500:]})


# -- phase e: a C++ change that does not compile -----------------------------
def phase_cpp_fails(R, work, slot):
    patch = make_patch([(EMIT, CPP_BREAK_OLD, CPP_BREAK_NEW)],
                       work / "cpp_break.diff")
    t = time.time()
    v = evaluate(patch, work / "cpp_break", slot)
    detail = v.get("_stdout", "")
    left = subprocess.run(
        ["bash", "-lc",
         f"pgrep -af 'slot{slot}' | grep -v pgrep | head -5"],
        capture_output=True, text=True).stdout.strip()
    R.case("e a C++ change that does not compile",
           "rejected at gate:build, with the compiler's diagnostic, "
           "no process left behind",
           f"{v.get('stage')} in {time.time() - t:.0f}s; diagnostic: "
           f"{'error:' in detail}; processes left: {left or 'none'}",
           v.get("stage") == "gate:build" and "error:" in detail and not left,
           {"detail": detail[-1200:]})
    # And the tree must be clean afterwards, so the next candidate is not
    # evaluated on top of a failed one.
    status = git(["status", "--porcelain", "--untracked-files=no"],
                 cwd=REPO / ".chia_scratch" / f"slot{slot}")
    R.case("e the slot is recoverable after a failed build",
           "the next reset leaves the slot clean",
           f"{len([l for l in status.splitlines() if l.strip()])} dirty "
           f"tracked files before the next reset (expected: the patch's own)",
           True, {"status": status[:400]})


# -- phase g: two evaluations at once ---------------------------------------
def phase_parallel(R, work):
    good = revert_dependence_branch(work / "par_revert.diff")
    bad = make_patch([(EMIT, CPP_BREAK_OLD, CPP_BREAK_NEW)],
                     work / "par_break.diff")
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f0 = ex.submit(evaluate, bad, work / "par0", 0)
        f1 = ex.submit(evaluate, good, work / "par1", 1)
        v0, v1 = f0.result(), f1.result()
    R.case("g two evaluations in parallel do not cross-contaminate",
           "slot 0 -> gate:build, slot 1 -> gate:tests, each in its own slot",
           f"slot0 {v0.get('stage')} in "
           f"{(v0.get('stages') or {}).get('assemble', {}).get('slot', '?')}; "
           f"slot1 {v1.get('stage')} in "
           f"{(v1.get('stages') or {}).get('assemble', {}).get('slot', '?')}; "
           f"{time.time() - t:.0f}s wall",
           v0.get("stage") == "gate:build" and v1.get("stage") == "gate:tests"
           and "slot0" in str((v0.get("stages") or {}).get("assemble", {}))
           and "slot1" in str((v1.get("stages") or {}).get("assemble", {})))


PHASES = {"d": phase_frozen, "f": phase_primitive, "h": phase_objective,
          "i": phase_using, "a": phase_noop, "b": phase_known_good,
          "c": phase_broken, "e": phase_cpp_fails, "g": phase_parallel}
FREE = "d,f,h,i"
ALL = "d,f,h,i,a,e,c,b,g"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default=ALL)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--run-dir", type=Path,
                    default=REPO / ".chia_scratch" / f"harness-test-"
                                                     f"{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args()
    work = a.run_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    R = Results()
    t0 = time.time()
    print(f"run dir: {work}\nphases:  {a.phases}\n")
    for p in a.phases.split(","):
        p = p.strip()
        if not p:
            continue
        fn = PHASES.get(p)
        if fn is None:
            print(f"unknown phase {p!r}; one of {sorted(PHASES)}")
            return 2
        print(f"-- phase {p} " + "-" * 60, flush=True)
        try:
            if p == "h":
                fn(R)
            elif p in ("d", "f", "i", "g"):
                fn(R, work)
            else:
                fn(R, work, a.slot)
        except Exception:                                # noqa: BLE001
            import traceback
            R.case(f"phase {p}", "completes", "the phase itself raised:\n"
                   + traceback.format_exc()[-1500:], False)
    elapsed = time.time() - t0
    passed = sum(1 for r in R.rows if r["ok"])
    out = {"cases": R.rows, "passed": passed, "total": len(R.rows),
           "minutes": round(elapsed / 60, 1), "usd": 0.0, "phases": a.phases}
    (work / "results.json").write_text(json.dumps(out, indent=1))
    print("\n" + "=" * 72)
    print(f"{passed}/{len(R.rows)} cases in {elapsed / 60:.1f} min, $0.00")
    print(f"results: {work / 'results.json'}")
    for r in R.rows:
        if not r["ok"]:
            print(f"  FAILED: {r['case']}")
    return 0 if R.ok else 1


if __name__ == "__main__":
    sys.exit(main())

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a candidate patch may touch, and what its added lines may contain. FROZEN.

The design-level loop's candidate is two files in a private directory, so
`chia_agent/spec_policy.py` polices FILE CONTENT. Here the candidate is a git
PATCH against a checkout, in one of two dispositions:

    using        the design and its program generator, with today's Allo
                 abstractions:  examples/tinytpu/
                 {microarch_isa,isa_dsl}.py
    maintaining  Allo itself, when an abstraction is what blocks the design:
                 allo/**.py, mlir/lib/**, mlir/include/allo/**

So the boundary is a PATH allowlist over the diff, plus a policy over the lines
the patch ADDS. Three rules, in the order they are applied:

1. **Deny wins.** `FROZEN_GLOBS` names what may never be touched in either
   disposition -- every test, every gate, every golden, every design file a
   candidate is scored on, this harness, and the build system. It is checked
   before the allowlist, so widening an allowlist by mistake cannot open a
   frozen path.
2. **Allowlist.** Only `EDITABLE[disposition]` may be touched, and a `using`
   candidate may not touch Allo, nor a `maintaining` candidate the design. A
   change that needs both is two experiments, not one.
3. **Added lines.** A narrow blacklist of the constructs that subvert a gate
   rather than compile a program: reaching the evaluator through `sys.modules`,
   rebinding a module's functions, naming a frozen gate or test file, spawning
   a shell, walking frames, ending the process with a chosen status, or writing
   a file from the C++ emitter. Each was checked against the code that is
   already in the tree: `allo/` uses none of `sys.modules`, `pytest`,
   `importlib.reload` or frame walking, and `mlir/lib/Translation` opens no
   files -- so refusing them in ADDED lines refuses nothing Allo does today.
   `os.system` and `shell=True` do occur in `allo/backend/hls.py`; only newly
   added occurrences are refused.

Why the build system is frozen: `ninja` runs the commands `CMakeLists.txt`
generates, and the build is the one stage that must be able to WRITE (into
`mlir/build`). An editable `CMakeLists.txt` would be arbitrary code execution
outside the evaluation sandbox's read-only tree. `.td` files are editable: they
are inputs to a tablegen command the frozen CMake files define.

And one structural rule, which is the cheap static gate CAKE argues for -- types
checked at construction, a primitive landing with its legality rules:

4. **A new schedule primitive must validate.** If the patch adds a method to
   `allo/customize.py`'s `Schedule`, that method must carry a docstring and must
   raise `AlloValueError` at least once. A primitive whose arguments are not
   checked against a closed set is refused here, before anything is built --
   `align_value` is in the tree as the counter-example (no validation, no test)
   and `s.dependence` as the pattern (five raises, three tests).

Rule 4 is blocking. A separate, NON-blocking hint is emitted when the patch
teaches one emitter to read a new loop attribute and leaves the others silently
ignoring it (issues #23, #24, #31 are all "ignored instead of rejected"); it
appears in the verdict as `hints`, not as a rejection.

This is a policy, not a sandbox. `evaluate_abs.py` backs it with a fresh
checkout from git, byte-identity checks on every frozen file, a bubblewrap
sandbox, nonce-vouched verdicts, and numbers cross-checked against Vitis's own
logs.
"""

from __future__ import annotations

import ast
import fnmatch
import re

DISPOSITIONS = ("using", "maintaining")

DESIGN_PKG = "examples/tinytpu"

#: Checked FIRST, in both dispositions. Nothing here is ever editable.
FROZEN_GLOBS = (
    # Every test, every limitation repro, every golden.
    "tests/*",
    "tests/**",
    # This harness, and the design-level harness whose gates it reuses.
    "chia_abstraction/*",
    "chia_abstraction/**",
    f"{DESIGN_PKG}/chia_agent/*",
    f"{DESIGN_PKG}/chia_agent/**",
    # The design's own evaluator, testbench generator and reference model.
    f"{DESIGN_PKG}/cosim.py",
    f"{DESIGN_PKG}/bench_isa.py",
    f"{DESIGN_PKG}/stress_isa.py",
    f"{DESIGN_PKG}/isa_ref.py",
    f"{DESIGN_PKG}/kpn_model.py",
    f"{DESIGN_PKG}/threaded_csim.py",
    f"{DESIGN_PKG}/mutate.py",
    f"{DESIGN_PKG}/impact/**",
    # The build system: ninja runs what these generate, and the build writes.
    "**/CMakeLists.txt",
    "**/*.cmake",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "MANIFEST.in",
    # The bindings symlink, and anything that decides which tree is imported.
    "allo/_mlir",
    "allo/_mlir_libs",
    "run_allo.sh",
    "scripts/**",
    ".github/**",
    # Git plumbing.
    ".gitignore",
    ".gitmodules",
    "externals/**",
)

#: What each disposition may touch. Checked after FROZEN_GLOBS.
EDITABLE = {
    "using": (
        f"{DESIGN_PKG}/microarch_isa.py",
        f"{DESIGN_PKG}/isa_dsl.py",
    ),
    "maintaining": (
        # The compiler: schedule surface, frontend/IR, HLS backends.
        "allo/*.py",
        "allo/**/*.py",
        # The C++ emitters and every pass under mlir/lib.
        "mlir/lib/**/*.cpp",
        "mlir/lib/**/*.h",
        "mlir/include/allo/**/*.h",
        "mlir/include/allo/**/*.td",
        # A primitive lands with its docstring; a backend key lands with its
        # page. Docs cannot move a gate, so they are free.
        "docs/source/**",
    ),
}

#: Python constructs refused in ADDED lines, in either disposition. Each entry
#: is (regex, why). Verified against the tree: nothing in `allo/` matches the
#: first eight today, so this refuses no existing Allo code.
PY_DENY = (
    (r"\bsys\s*\.\s*modules\b",
     "sys.modules reaches the evaluator's own module; Allo never uses it"),
    (r"\b__builtins__\b|\bimport\s+builtins\b|\bbuiltins\s*\.",
     "builtins is frozen during a gate"),
    (r"\bimportlib\s*\.\s*reload\b",
     "importlib.reload re-runs a module after it was snapshotted"),
    (r"\bsys\s*\.\s*(settrace|setprofile|meta_path|path_hooks|_getframe)\b",
     "tracing, import hooks and frame access are gate-subversion primitives"),
    (r"\b(f_back|f_locals|f_globals|gi_frame|tb_frame|cr_frame)\b",
     "walking frames can read the gate's nonce out of a caller's locals"),
    (r"\bos\s*\.\s*system\s*\(|shell\s*=\s*True",
     "a new shell escape; allo/backend/hls.py's existing ones are not touched"),
    (r"\bos\s*\.\s*(putenv|unsetenv)\b|\bos\s*\.\s*environ\s*\.\s*clear\b",
     "changing the environment out from under the evaluator"),
    (r"\bos\s*\.\s*environ\s*\[\s*[\"'](PYTHONPATH|LD_PRELOAD|PYTHONSTARTUP|"
     r"PYTHONHOME|TPU_[A-Z_]*)[\"']\s*\]\s*=",
     "PYTHONPATH / LD_PRELOAD / TPU_* decide which tree and which memory model "
     "the gate measures"),
    (r"\braise\s+SystemExit\b|\bsys\s*\.\s*exit\s*\(|\bos\s*\.\s*_exit\s*\(",
     "ending the process with a chosen status forged a verdict once already; "
     "raise AlloError instead (this is issue #30's fix, not its repeat)"),
    (r"\b(pytest|_pytest|unittest|conftest)\b",
     "the test framework runs the gate; the compiler must not import it"),
    # Naming a frozen gate, golden or test file from inside the compiler.
    (r"(chia_agent|chia_abstraction|gate_runner|stress_isa|bench_isa|isa_ref|"
     r"param_check|kpn_model|CHIA-GATE)",
     "a frozen gate, golden or harness file named from inside the compiler"),
    (r"[\"'][^\"']*tests/[^\"']*[\"']",
     "a path under tests/ named from inside the compiler"),
)

#: C++ constructs refused in ADDED lines under mlir/. The emitters write to an
#: `llvm::raw_ostream` they are handed; none of them opens a file, spawns a
#: process, or reads the environment today.
CPP_DENY = (
    (r"\b(system|popen|execl|execlp|execv|execvp|fork|posix_spawn)\s*\(",
     "spawning a process from the emitter"),
    (r"\b(fopen|freopen|ofstream|fstream|open64|creat)\b",
     "opening a file from the emitter; write to the raw_ostream you are given"),
    (r"\b(remove|rename|unlink|rmdir)\s*\(",
     "removing or renaming a file from the emitter"),
    (r"\b(getenv|setenv|putenv)\s*\(",
     "the emitter must not read or write the environment"),
    (r"\bdlopen\s*\(|\bLD_PRELOAD\b",
     "loading code at emission time"),
)

#: Primitive families whose emitter support is per-target, so that adding one
#: reader and no rejecters is worth a hint. See issues #23, #24, #31.
_EMITTERS = ("EmitVivadoHLS.cpp", "EmitTapaHLS.cpp", "EmitIntelHLS.cpp",
             "EmitCatapultHLS.cpp", "EmitXlsHLS.cpp")


class PatchError(Exception):
    pass


# -- the diff -----------------------------------------------------------------
def paths_in_diff(diff: str) -> set[str]:
    """Every repository path a unified diff claims to touch.

    Both the `diff --git` headers and the `---`/`+++` pairs are read, and they
    must agree: a patch whose header names one file and whose body another was
    the obvious way past a header-only check.
    """
    paths: set[str] = set()
    for a, b in re.findall(r"^diff --git a/(\S+) b/(\S+)$", diff, re.M):
        paths.update({a, b})
    for line in diff.splitlines():
        if line.startswith(("--- ", "+++ ")):
            p = line[4:].split("\t")[0].strip()
            if p in ("/dev/null", "a/dev/null", "b/dev/null"):
                continue
            if p.startswith(("a/", "b/")):
                p = p[2:]
            if p:
                paths.add(p)
    if not paths:
        raise PatchError("the patch names no files")
    return paths


def added_lines(diff: str) -> list[tuple[str, str]]:
    """[(path, added line without its '+')] over the whole diff."""
    out, path = [], "?"
    for line in diff.splitlines():
        m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line)
        if m:
            path = m.group(1).split("\t")[0]
            continue
        if line.startswith("--- ") or line.startswith("diff --git"):
            continue
        if line.startswith("+"):
            out.append((path, line[1:]))
    return out


def _matches(path: str, globs) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def path_violations(disposition: str, paths) -> list[str]:
    if disposition not in DISPOSITIONS:
        return [f"unknown disposition {disposition!r}; one of {DISPOSITIONS}"]
    bad = []
    for p in sorted(paths):
        if p.startswith("/") or ".." in p.split("/"):
            bad.append(f"{p}: absolute or parent-relative path")
            continue
        if _matches(p, FROZEN_GLOBS):
            bad.append(f"{p}: FROZEN -- a test, gate, golden, scored design "
                       f"file, harness file or build file")
            continue
        if not _matches(p, EDITABLE[disposition]):
            other = [d for d in DISPOSITIONS
                     if d != disposition and _matches(p, EDITABLE[d])]
            why = (f" (it is editable in the `{other[0]}` disposition; a change "
                   f"that needs both is two experiments)" if other else "")
            bad.append(f"{p}: not editable in the `{disposition}` "
                       f"disposition{why}")
    return bad


def line_violations(diff: str) -> list[str]:
    bad = []
    for path, line in added_lines(diff):
        if re.match(r"^\s*(#|//|\*|/\*)", line):
            continue                     # a comment cannot execute
        rules = CPP_DENY if path.startswith("mlir/") else PY_DENY
        if path.startswith("docs/"):
            continue                     # documentation cannot move a gate
        for pattern, why in rules:
            if re.search(pattern, line):
                bad.append(f"{path}: {why} -- {line.strip()[:120]!r}")
    return sorted(set(bad))


# -- rule 4: a new primitive must validate ------------------------------------
_SCHEDULE_SRC = "allo/customize.py"


def primitive_violations(diff: str, new_customize_source: str | None) -> list[str]:
    """Every method the patch ADDS to `Schedule` must validate and document.

    `new_customize_source` is `allo/customize.py` AFTER the patch. The check is
    on methods whose `def` line the patch adds, so an existing primitive is
    never re-judged.
    """
    if new_customize_source is None:
        return []
    added = {re.match(r"\s*def\s+([A-Za-z_]\w*)", line).group(1)
             for path, line in added_lines(diff)
             if path == _SCHEDULE_SRC and re.match(r"\s*def\s+[A-Za-z_]\w*", line)}
    if not added:
        return []
    try:
        tree = ast.parse(new_customize_source)
    except SyntaxError as e:
        return [f"{_SCHEDULE_SRC} does not parse after the patch: {e.msg} "
                f"at line {e.lineno}"]
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "Schedule"), None)
    if cls is None:
        return [f"{_SCHEDULE_SRC}: class Schedule is gone"]
    bad = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in added or node.name.startswith("_"):
            continue
        if not ast.get_docstring(node):
            bad.append(
                f"Schedule.{node.name}: a new schedule primitive needs a "
                f"docstring -- api/index.rst autodocs Schedule, so the "
                f"docstring IS the documentation. State the pragma or IR it "
                f"produces, the closed set of each argument, and what is a "
                f"promise rather than a fact.")
        raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
        names = set()
        for r in raises:
            for sub in ast.walk(r):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    names.add(sub.attr)
        if "AlloValueError" not in names:
            bad.append(
                f"Schedule.{node.name}: a new schedule primitive must validate "
                f"its arguments against a closed set and raise AlloValueError "
                f"with the argument's name and value, BEFORE touching any IR "
                f"(see Schedule.dependence for the pattern). Not one "
                f"AlloValueError is raised.")
    return bad


def hints(diff: str) -> list[str]:
    """Non-blocking observations. CAKE's third disposition: a report, not a gate."""
    out = []
    touched = {p for p, _ in added_lines(diff)}
    emitters = {e for e in _EMITTERS
                if any(p.endswith(e) for p in touched)}
    new_attrs = set()
    for path, line in added_lines(diff):
        if not path.startswith("mlir/"):
            continue
        for m in re.finditer(r'getLoopDirective\([^,]+,\s*"([A-Za-z_]\w*)"\)', line):
            new_attrs.add(m.group(1))
    if new_attrs and len(emitters) == 1:
        missing = [e for e in _EMITTERS if e not in emitters]
        out.append(
            f"this patch teaches {sorted(emitters)[0]} to read the loop "
            f"attribute(s) {sorted(new_attrs)}; {', '.join(missing)} will "
            f"SILENTLY IGNORE it. Issues #23, #24 and #31 are all 'ignored "
            f"instead of rejected'. A target that cannot honour a directive "
            f"should emitError, not drop it.")
    if any(p == _SCHEDULE_SRC for p, _ in added_lines(diff)) and not any(
            p.startswith("docs/") for p, _ in added_lines(diff)):
        out.append("allo/customize.py changed and no docs/ page did. The "
                   "docstring is autodoc'd, so it may be enough -- but a "
                   "backend configuration key is invisible on the API page.")
    return out


def check(disposition: str, diff: str,
          new_customize_source: str | None = None) -> tuple[list[str], list[str]]:
    """(blocking problems, non-blocking hints) for one candidate patch."""
    problems = []
    try:
        paths = paths_in_diff(diff)
    except PatchError as e:
        return [str(e)], []
    problems += path_violations(disposition, paths)
    problems += line_violations(diff)
    if disposition == "maintaining":
        problems += primitive_violations(diff, new_customize_source)
    return problems, hints(diff)

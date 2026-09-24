# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a TinyTPU-isa spec file may contain.

The agent's two writable files, `microarch_isa.py` and `isa_dsl.py`, are Python
modules that the evaluator *imports*: `cosim.py`, `bench_isa.py` and
`stress_isa.py` all import them, so anything at their import path runs inside the
process that produces the score. That makes "which edits are allowed" a
property of the harness, not of the prompt, and this module is where it is
decided. `evaluate.py` re-runs it from git on every candidate, and both edit
tools run it before writing.

It is a policy check, not a sandbox. It refuses the constructs an agent reaches
for when it edits its way around the harness -- file writes, process spawning,
dynamic execution, reaching the evaluator's module through `sys.modules` or an
`examples.*` import -- and is narrow enough to accept every construct the two
shipped files use. `evaluate.py` backs it with checks that do not trust the
candidate's process at all (per-shape cosim logs, simulated time, csynth XML).
"""

from __future__ import annotations

import ast

#: Import roots a TinyTPU-isa spec needs. Anything else is refused.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "allo",
        "numpy",
        "math",
        "typing",
        "dataclasses",
        "__future__",
        "functools",
        "itertools",
        "collections",
        "enum",
        "contextlib",
        "os",
        "sys",
    }
)
#: The only `examples.*` modules a spec may import: each other, and the frozen
#: machinery of the same package (`ip.params`, and the reduce IP that
#: `ip/__init__.py` pulls in). In particular not `cosim`, `bench_isa` or
#: anything in `chia_agent` -- the evaluator. A literal rather than an import
#: of `design.IMPORTABLE_MODULES`, because `evaluate.compose()` execs this
#: module out of git and a candidate must not be able to widen it;
#: test_harness's phase `s` asserts the two agree.
ALLOWED_EXAMPLES = frozenset(
    {
        "examples.tinytpu.microarch_isa",
        "examples.tinytpu.isa_dsl",
        "examples.tinytpu.ip.isa",
        "examples.tinytpu.ip.tinytpu",
        "examples.tinytpu.ip.assembler",
        "examples.tinytpu.ip.programs",
        "examples.tinytpu.ip.params",
        "examples.tinytpu.ip.reduce",
        "examples.tinytpu.ip.units.sequencer",
        "examples.tinytpu.ip.units.dma_load",
        "examples.tinytpu.ip.units.scratchpad",
        "examples.tinytpu.ip.units.vector_regs",
        "examples.tinytpu.ip.units.weight_loader",
        "examples.tinytpu.ip.units.pe",
        "examples.tinytpu.ip.units.accumulator",
        "examples.tinytpu.ip.units.dma_store",
        "examples.tinytpu.ip.units.reduction_tree",
    }
)
#: `os` / `sys` are needed for exactly `os.environ.get`, `os.path.*` and
#: `sys.path.insert`; every other attribute of theirs is refused.
MODULE_ATTRS = {"os": {"environ", "path"}, "sys": {"path"}}

#: Builtins that turn "describe hardware" into "do anything".
DENIED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__builtins__",
        "__loader__",
        "__spec__",
        # Ending the process early with a chosen status: a module that prints
        # `STRESS OK: ...` and raises SystemExit(0) at import time passed the
        # old stdout-reading gate (shown on this branch, 2026-09-19). The
        # verdict now comes from `gate_runner.py`, which this cannot fool;
        # refusing it here makes the attempt visible at edit time.
        "SystemExit",
        "KeyboardInterrupt",
        "exit",
        "quit",
    }
)

#: Attribute names that write to, or execute outside, the process.
DENIED_ATTRS = frozenset(
    {
        "write_text",
        "write_bytes",
        "unlink",
        "rmdir",
        "chmod",
        "rename",
        "replace",
        "system",
        "popen",
        "remove",
        "removedirs",
        "spawnv",
        "spawnl",
        "modules",
        "meta_path",
        "path_hooks",
        "settrace",
        "setprofile",
        "putenv",
        "unsetenv",
        # numpy's file writers. `numpy` is an allowed import, and the harness
        # test showed `np.savetxt(<tree>/chia_agent/stress.py, ...)` at import
        # time rewriting the stress gate before it ran. The evaluator's sandbox
        # is what stops a writer this list does not name.
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "dump",
        "memmap",
        "open_memmap",
        "lib",
        # ...and its file readers: a spec has no business reading files, and
        # /proc/self/* is a file.
        "fromfile",
        "fromregex",
        "loadtxt",
        "genfromtxt",
        "load",
        # Frame and traceback walking: how code reaches the locals of the
        # frozen check that called it (gate_runner's nonce lives in one).
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "tb_frame",
        "tb_next",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
    }
)
#: Dunder attributes are how Python reaches past a module's surface
#: (`__globals__`, `__subclasses__`, `__dict__`, ...). These two are harmless.
ALLOWED_DUNDER_ATTRS = frozenset({"__name__", "__doc__"})


#: Read-only, and only under `if __name__ == "__main__":`, which the evaluator
#: never executes: isa_dsl's self-test reads bench_isa.SHAPES there.
MAIN_GUARD_EXAMPLES = frozenset({"examples.tinytpu.bench_isa"})


def _main_guard_nodes(tree):
    inside = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"):
            inside.update(ast.walk(node))
    return inside


def _parents(tree):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _root(node):
    """The Name an attribute/subscript chain hangs off, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _bound_names(target):
    """Names a binding target binds (not the base of `x.a = ...`)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _bound_names(e)]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


def _module_names(tree) -> set[str]:
    """Names bound by an import, plus every name assigned from an expression
    that mentions one (`r = np.random`, `m = [allo][0]`), to a fixpoint."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs += [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value:
            pairs.append((node.target, node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            pairs.append((node.target, node.iter))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            pairs.append((node.optional_vars, node.context_expr))
    grew = True
    while grew:
        grew = False
        for target, value in pairs:
            if any(isinstance(n, ast.Name) and n.id in bound for n in ast.walk(value)):
                for n in _bound_names(target):
                    if n not in bound:
                        bound.add(n)
                        grew = True
    return bound


def policy_violations(name: str, source: str) -> list[str]:
    """Constructs a hardware spec has no business containing."""
    tree = ast.parse(source, filename=name)
    parent = _parents(tree)
    guarded = _main_guard_nodes(tree)
    problems = []
    modules = _module_names(tree)
    for node in ast.walk(tree):
        # Monkeypatching: rebinding an attribute of an imported module (or of
        # anything derived from one) changes code the frozen checks call --
        # numpy's RNG, allo's build -- from inside the candidate. Attribute
        # stores are allowed on the spec's own objects only. `gate_runner.py`
        # also refuses the rebinding at run time, which covers the spellings
        # this static rule cannot see.
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))):
            root = _root(node)
            if root is None or root in modules:
                problems.append(f"{name}: assigns to '{ast.unparse(node)}', an "
                                f"attribute of an imported module")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "examples":
                    problems.append(f"{name}: imports '{alias.name}'; import the "
                                    f"spec modules with 'from ... import'")
                elif root not in ALLOWED_IMPORT_ROOTS:
                    problems.append(f"{name}: imports '{root}', which is not allowed")
                elif root in MODULE_ATTRS and alias.asname:
                    problems.append(f"{name}: aliases '{root}'")
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if node.level:
                problems.append(f"{name}: relative import")
            elif root == "examples":
                if mod not in ALLOWED_EXAMPLES and not (
                        node in guarded and mod in MAIN_GUARD_EXAMPLES):
                    problems.append(f"{name}: imports '{mod}'; only the two spec "
                                    f"modules are importable from examples")
            elif root in MODULE_ATTRS:
                problems.append(f"{name}: 'from {root} import ...' is not allowed")
            elif root not in ALLOWED_IMPORT_ROOTS:
                problems.append(f"{name}: imports '{root}', which is not allowed")
        if isinstance(node, ast.Name):
            if node.id in DENIED_NAMES:
                problems.append(f"{name}: uses '{node.id}'")
            if node.id in MODULE_ATTRS:
                up = parent.get(node)
                if not (isinstance(up, ast.Attribute) and up.value is node):
                    problems.append(f"{name}: uses '{node.id}' other than as "
                                    f"'{node.id}.<attr>'")
                elif up.attr not in MODULE_ATTRS[node.id]:
                    problems.append(f"{name}: uses '{node.id}.{up.attr}'")
                elif node.id == "os" and up.attr == "environ":
                    get = parent.get(up)
                    call = parent.get(get)
                    if not (isinstance(get, ast.Attribute) and get.attr == "get"
                            and isinstance(call, ast.Call) and call.func is get):
                        problems.append(f"{name}: os.environ is read-only here; "
                                        f"use os.environ.get(...)")
        if isinstance(node, ast.Attribute):
            if node.attr in DENIED_ATTRS:
                problems.append(f"{name}: uses '.{node.attr}'")
            if (node.attr.startswith("__") and node.attr.endswith("__")
                    and node.attr not in ALLOWED_DUNDER_ATTRS):
                problems.append(f"{name}: uses '.{node.attr}'")
    # Module-level context managers and try blocks are how self-rewriting code
    # hides; nothing in a legitimate spec needs either at import time.
    for node in tree.body:
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try)):
            problems.append(
                f"{name}: has a module-level "
                f"{'try' if isinstance(node, ast.Try) else 'with'} block, which "
                f"would run on import"
            )
    problems += parameter_violations(name, source)
    return sorted(set(problems))


# -- parametricity ---------------------------------------------------------
#: The design's size parameters and the only form their definition may take in
#: microarch_isa.py: `NAME = int(os.environ.get("TPU_NAME", <int>))`. The gate
#: scores T=4, MAXDIM=16, so a literal `T = 4` or `MAXDIM = 16` specialises the
#: design to the evaluator's configuration; `param_check.py` rebuilds at other
#: MAXDIMs and needs the parameter honoured. (The first paid run's accepted
#: diff replaced the T definition with `T = 4`.)
PARAMETERS = {"T": "TPU_T", "MAXDIM": "TPU_MAXDIM"}


def _is_env_param(value, env_name) -> bool:
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "int" and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.Call)
            and ast.unparse(value.args[0].func) == "os.environ.get"
            and len(value.args[0].args) == 2
            and isinstance(value.args[0].args[0], ast.Constant)
            and value.args[0].args[0].value == env_name
            and isinstance(value.args[0].args[1], ast.Constant)
            and isinstance(value.args[0].args[1].value, int))


def parameter_violations(name: str, source: str) -> list[str]:
    """T and MAXDIM must stay environment parameters, defined once, in
    microarch_isa.py, and must not be rebound anywhere in either file."""
    tree = ast.parse(source, filename=name)
    defs = {p: [] for p in PARAMETERS}
    problems = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [(node.target, node.value if not isinstance(node, ast.AugAssign)
                        else None)]
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets = [(node.target, None)]
        elif isinstance(node, ast.NamedExpr):
            targets = [(node.target, None)]
        for t, value in targets:
            for n in _bound_names(t):
                if n in PARAMETERS:
                    defs[n].append((node, value))
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                if n in PARAMETERS:
                    problems.append(f"{name}: declares '{n}' global/nonlocal; "
                                    f"it is a design parameter")
    for p, env in PARAMETERS.items():
        good = [d for d in defs[p] if d[1] is not None and _is_env_param(d[1], env)
                and d[0] in tree.body]
        if name == "microarch_isa.py":
            if len(defs[p]) != 1 or len(good) != 1:
                problems.append(
                    f"{name}: '{p}' must be defined exactly once, at module level, "
                    f"as {p} = int(os.environ.get(\"{env}\", <default>)) -- it is a "
                    f"design parameter, and the gate also builds the design at "
                    f"other values of it")
        elif defs[p]:
            problems.append(f"{name}: assigns '{p}', a design parameter defined in "
                            f"microarch_isa.py")
    return problems


# -- documentation -----------------------------------------------------------
#: How many lines of comment + docstring text a candidate may remove from a
#: spec file, net, relative to the file at the frozen ref. Additions and edits
#: are free; wholesale deletion is not (the first paid run's accepted diff
#: deleted the 260-line design docstring of microarch_isa.py).
DOC_LOSS_MAX = 15


def doc_lines(source: str) -> list[str]:
    """Non-blank lines of comments and docstrings (module, class, function)."""
    import io
    import tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                txt = tok.string.lstrip("#").strip()
                if txt:
                    out.append(txt)
    except (tokenize.TokenError, IndentationError):
        pass
    tree = ast.parse(source)
    for node in [tree, *ast.walk(tree)]:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                out += [l.strip() for l in doc.splitlines() if l.strip()]
    return out


def doc_loss(base_source: str, source: str) -> int:
    """Net comment/docstring lines removed; negative means documentation was
    added."""
    return len(doc_lines(base_source)) - len(doc_lines(source))


def doc_violations(name: str, base_source: str, source: str) -> list[str]:
    """Refuse a net loss of more than DOC_LOSS_MAX comment/docstring lines."""
    before, after = len(doc_lines(base_source)), len(doc_lines(source))
    if before - after > DOC_LOSS_MAX:
        return [f"{name}: removes {before - after} lines of comments/docstrings "
                f"(from {before} to {after}; at most {DOC_LOSS_MAX} may go). "
                f"Edit or add documentation; do not delete it"]
    return []


def doc_violations_total(losses: dict[str, int]) -> list[str]:
    """Refuse a net loss of more than DOC_LOSS_MAX lines across the WHOLE
    candidate.

    `DOC_LOSS_MAX` is per file, and the design used to be one file, so the two
    were the same number. It is fourteen files now
    (`docs/source/designs/tinytpu_library.rst`), and a per-file budget alone
    would let a candidate delete fourteen times as much documentation as the
    guard was written to allow. The total restores exactly the original
    guarantee. The edit tool keeps checking per file, for fast feedback; the
    evaluator is where the total is enforced.
    """
    net = sum(losses.values())
    if net > DOC_LOSS_MAX:
        worst = ", ".join(f"{n} (-{d})" for n, d in
                          sorted(losses.items(), key=lambda kv: -kv[1]) if d > 0)
        return [f"the candidate removes {net} lines of comments/docstrings "
                f"across the design (at most {DOC_LOSS_MAX} may go in total): "
                f"{worst}. Edit or add documentation; do not delete it"]
    return []

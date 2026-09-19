# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a TinyTPU-isa spec file may contain.

The agent's two writable files, `microarch_isa.py` and `isa_dsl.py`, are Python
modules that the evaluator *imports*: `cosim.py`, `bench_isa.py` and
`stress.py` all import them, so anything at their import path runs inside the
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
#: The only `examples.*` modules a spec may import: each other. In particular
#: not `cosim`, `bench_isa` or anything in `chia_agent` -- the evaluator.
ALLOWED_EXAMPLES = frozenset(
    {
        "examples.accelerator.tinytpu_vitis.microarch_isa",
        "examples.accelerator.tinytpu_vitis.isa_dsl",
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
    }
)
#: Dunder attributes are how Python reaches past a module's surface
#: (`__globals__`, `__subclasses__`, `__dict__`, ...). These two are harmless.
ALLOWED_DUNDER_ATTRS = frozenset({"__name__", "__doc__"})


#: Read-only, and only under `if __name__ == "__main__":`, which the evaluator
#: never executes: isa_dsl's self-test reads bench_isa.SHAPES there.
MAIN_GUARD_EXAMPLES = frozenset({"examples.accelerator.tinytpu_vitis.bench_isa"})


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


def policy_violations(name: str, source: str) -> list[str]:
    """Constructs a hardware spec has no business containing."""
    tree = ast.parse(source, filename=name)
    parent = _parents(tree)
    guarded = _main_guard_nodes(tree)
    problems = []
    for node in ast.walk(tree):
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
    return sorted(set(problems))

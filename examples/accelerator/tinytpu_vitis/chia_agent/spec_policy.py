# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a TinyTPU spec file may contain.

The agent's two writable files are Python modules that the evaluator *imports*,
so anything at their import path runs inside the scoring process. That makes
"which edits are allowed" a property of the harness, not of the prompt, and
this module is where it is decided.

It is a policy check, not a sandbox. It refuses the constructs an agent reaches
for when it edits its way around the harness -- file writes, process spawning,
dynamic execution -- and is narrow enough to accept every construct the real
spec uses. Genuine isolation would mean evaluating inside a container.
"""

from __future__ import annotations

import ast

#: Import roots a legitimate TinyTPU spec needs. Anything else is refused: the
#: writable files are *imported* by the evaluator, so whatever they contain runs
#: inside the scoring process.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "allo",
        "numpy",
        "math",
        "typing",
        "dataclasses",
        "__future__",
        "argparse",
        "pathlib",
        "functools",
        "itertools",
        "collections",
        "enum",
    }
)

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
        "system",
        "popen",
        "remove",
        "removedirs",
        "spawnv",
        "spawnl",
    }
)


def _import_roots(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        # A relative import stays inside the example package.
        if node.level:
            return []
        return [(node.module or "").split(".")[0]]
    return []


def policy_violations(name: str, source: str) -> list[str]:
    """Constructs a hardware spec has no business containing.

    This is a policy check, not a sandbox: it refuses the file-writing,
    process-spawning and dynamic-execution constructs an agent reaches for when
    it tries to edit its way around the harness, and it is deliberately narrow
    enough to accept every construct the real spec uses (module-level schedule
    and ``tpu.*`` declaration calls, the bind loop, an ``if __name__`` guard).
    Real isolation would mean evaluating in a container.
    """
    tree = ast.parse(source, filename=name)
    problems = []
    for node in ast.walk(tree):
        for root in _import_roots(node):
            if root and root not in ALLOWED_IMPORT_ROOTS:
                problems.append(f"{name}: imports '{root}', which is not allowed here")
        if isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            problems.append(f"{name}: uses '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRS:
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

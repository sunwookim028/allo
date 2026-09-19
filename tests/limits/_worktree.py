# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import shim for tests/limits repros.

Each repro must test the tree it is committed in. Run from the primary
checkout, `allo` already resolves there and this shim does nothing. Run from a
separate worktree, the `allo` conda env's editable install (which points at the
primary checkout and outranks PYTHONPATH) is stripped and this checkout's
`allo` is imported instead; that worktree needs its own built `allo/_mlir`.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolves_here():
    """True when `import allo` would already load this checkout's package, e.g.
    when run from the primary checkout the editable install points at."""
    import importlib.util

    spec = importlib.util.find_spec("allo")
    if spec is None or not spec.origin:
        return False
    here = os.path.realpath(os.path.join(ROOT, "allo", "__init__.py"))
    return os.path.realpath(spec.origin) == here


if not _resolves_here():
    # A separate worktree: the editable finder would load the primary
    # checkout instead, so drop it and put this checkout first.
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if "editable" not in type(f).__module__.lower()
        and "editable" not in getattr(f, "__name__", "").lower()
    ]
    sys.path.insert(0, ROOT)
import allo  # noqa: E402

assert os.path.realpath(allo.__file__) == os.path.realpath(
    os.path.join(ROOT, "allo", "__init__.py")
), (allo.__file__, ROOT)


def verdict(item, reproduces, detail="", absent="FIXED"):
    """`absent` labels a non-reproduction: FIXED when a commit removed it,
    CANNOT-REPRODUCE when no fix is identified or it was never broken."""
    tag = "REPRODUCES" if reproduces else absent
    print(f"[item {item}] {tag}" + (f": {detail}" if detail else ""))


def run_guarded(item, main):
    """Run `main` in a child process so a C++ abort (MLIR assertion) still
    yields a verdict line instead of killing the repro silently."""
    import subprocess

    if os.environ.get("ALLO_LIMITS_CHILD") == "1":
        main()
        return
    env = dict(os.environ, ALLO_LIMITS_CHILD="1")
    p = subprocess.run(
        [sys.executable, sys.argv[0], *sys.argv[1:]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if f"[item {item}]" not in p.stdout:
        tail = [l for l in p.stderr.splitlines() if l.strip()][-1:] or ["<no output>"]
        verdict(item, True, f"child died (rc={p.returncode}): {tail[0][:200]}")

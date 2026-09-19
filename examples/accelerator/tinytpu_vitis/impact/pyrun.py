"""Run a script with `allo` forced to resolve to THIS worktree.

The conda env carries an editable finder pointing at /home/sk3463/allo; strip it
so a variant built here cannot silently import main's allo.
    python pyrun.py <script.py> [args...]
"""
import os, runpy, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.meta_path[:] = [f for f in sys.meta_path
                    if "editable" not in type(f).__module__.lower()
                    and "editable" not in getattr(f, "__name__", "").lower()]
sys.path.insert(0, ROOT)
import allo  # noqa: E402
assert allo.__file__.startswith(ROOT + "/"), allo.__file__
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

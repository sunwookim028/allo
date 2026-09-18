# Run a script against the allo checkout at $ALLO_ROOT, bypassing the conda env's
# editable finder (it outranks PYTHONPATH and may point at another checkout).
import os, runpy, sys
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()
                    and "editable" not in getattr(f, "__name__", "").lower()]
root = os.path.abspath(os.environ["ALLO_ROOT"])
sys.path.insert(0, root)
import allo
assert os.path.dirname(os.path.abspath(allo.__file__)) == os.path.join(root, "allo"), allo.__file__
print(f"[run_sc] allo from {allo.__file__}", flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

# usage: python run_sc.py emit.py <outdir> [variant ...]
# Writes a csyn Catapult project per dot-product-link variant (default wire stream channel).
import os, sys
# The design lives with the other designs. Find the repo root
# by searching UPWARD for a marker -- never by counting levels (dev/roadmap.md).
_D = os.path.dirname(os.path.abspath(__file__))
while not (os.path.exists(os.path.join(_D, "pyproject.toml"))
           and os.path.isdir(os.path.join(_D, "allo"))):
    _P = os.path.dirname(_D)
    if _P == _D:
        raise RuntimeError("no repository root above " + __file__)
    _D = _P
sys.path.insert(0, os.path.join(_D, "examples", "systemc"))
import allo.dataflow as df
import dot_product_four_links as pe_split  # examples/systemc/dot_product_four_links.py

out = os.path.abspath(sys.argv[1])
for v in sys.argv[2:] or ["wire", "stream", "channel"]:
    prj = os.path.join(out, f"pe_{v}")
    os.makedirs(prj, exist_ok=True)
    df.build(pe_split.VARIANTS[v], target="systemc", mode="csyn", project=prj)
    print(f"[emit] pe_{v} -> {prj}", flush=True)

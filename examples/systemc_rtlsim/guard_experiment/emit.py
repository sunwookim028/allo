# usage: python run_sc.py emit.py <outdir> [variant ...]
# Writes a csyn Catapult project per pe_split variant (default wire stream channel).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import allo.dataflow as df
import pe_split  # ../pe_split.py

out = os.path.abspath(sys.argv[1])
for v in sys.argv[2:] or ["wire", "stream", "channel"]:
    prj = os.path.join(out, f"pe_{v}")
    os.makedirs(prj, exist_ok=True)
    df.build(pe_split.VARIANTS[v], target="systemc", mode="csyn", project=prj)
    print(f"[emit] pe_{v} -> {prj}", flush=True)

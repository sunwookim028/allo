# sha256 of the TinyTPU emission text (vhls, catapult, systemc) under one recipe.
# usage: python unrst_hash.py <tree> <scratch>
import hashlib, os, sys
tree, scr = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
os.chdir(tree)
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
import allo
print("allo from", allo.__file__)
for tgt in ("vhls", "catapult", "systemc"):
    s = customize(tinytpu_isa)
    schedule(s)
    prj = os.path.join(scr, f"h_{tgt}.prj")
    kw = {}
    if tgt != "systemc":
        prj = None
    if tgt == "systemc":
        kw["mode"] = "csim"
    try:
        m = s.build(target=tgt, project=prj, **kw) if prj else s.build(target=tgt)
        code = m.hls_code.replace(prj, "<PRJ>") if prj else m.hls_code
        print(tgt, hashlib.sha256(code.encode()).hexdigest(), len(code))
    except Exception as e:
        print(tgt, "ERR", type(e).__name__, str(e)[:200])

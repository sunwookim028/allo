import hashlib, os, sys, tempfile, shutil
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
tmp = tempfile.mkdtemp(prefix="hash_tinytpu.")
for tgt in ("vhls", "catapult"):
    s = customize(tinytpu_isa)
    schedule(s)
    prj = os.path.join(tmp, tgt + ".prj")
    mod = s.build(target=tgt)
    text = str(mod)
    print(tgt, hashlib.sha256(text.encode()).hexdigest(), len(text))
    with open(os.path.join(tmp, tgt + ".cpp"), "w") as f:
        f.write(text)
print("dir", tmp)

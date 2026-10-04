# sha256 of every MiniTPU unit variant's emission (systemc, vhls), one line each.
# usage: python emit_units.py <tree> <outdir>
import hashlib, importlib, os, sys, inspect
tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree); os.chdir(tree)
os.makedirs(out, exist_ok=True)
import allo.dataflow as df
units = sorted(f[:-3] for f in os.listdir("examples/minitpu/units") if f.endswith(".py") and f != "__init__.py")
for u in units:
    m = importlib.import_module(f"examples.minitpu.units.{u}")
    for v, (make, _) in getattr(m, "VARIANTS", {}).items():
        for tgt in ("systemc", "vhls"):
            try:
                code = df.build(make(8), target=tgt).hls_code
                h = hashlib.sha256(code.encode()).hexdigest()[:16]
                open(os.path.join(out, f"{u}.{v}.{tgt}.cpp"), "w").write(code)
            except BaseException as e:
                h = "ERR " + type(e).__name__ + " " + str(e).splitlines()[0][:100] if str(e) else "ERR " + type(e).__name__
            print(u, v, tgt, h, flush=True)

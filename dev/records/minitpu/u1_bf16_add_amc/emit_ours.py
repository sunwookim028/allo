# Route (b), step 1: emit our frontend's MLIR text for the bits kernel.
# Run in OUR env: source examples/minitpu/harness/env-zhang21.sh; cd <worktree>;
#   $ALLO_PYTHON dev/records/minitpu/u1_bf16_add_amc/emit_ours.py <out_dir> [N]
# Writes: region.mlir (df.region form, as examples/minitpu/units/bf16_add.py
# builds it), plain.mlir (kernel_bits.py, the same body as a plain function),
# plain_amcedits.mlir (kernel_bits_amc.py, the body with A-n edits), and
# region_amcedits.mlir (that body wrapped back into the df.region/df.kernel
# form). The plain forms fail our own verifier (finding O1), so only the
# region forms are handed to AMC.
import os, sys, importlib.util, traceback
import allo
import allo.dataflow as df

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 16  # elements per kernel
os.makedirs(out, exist_ok=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


from examples.minitpu.units import bf16_add

try:
    s = df.customize(bf16_add.bits(NN))
    open(f"{out}/region.mlir", "w").write(str(s.module))
    print("region.mlir written")
except Exception:
    traceback.print_exc()

for name, fn in [("plain", "bf16_add_bits"), ("plain_amcedits", "bf16_add_bits_amc")]:
    K = load(name, f"{HERE}/kernel_bits{'_amc' if 'amc' in name else ''}.py")
    try:
        s = allo.customize(getattr(K, fn))
        open(f"{out}/{name}.mlir", "w").write(str(s.module))
        print(f"{name}.mlir written")
    except Exception:
        traceback.print_exc()

# region_amcedits: the A-edited body inside the same df.region wrapper.
src = open(f"{HERE}/kernel_bits_amc.py").read()
body = src[src.index("def bf16_add_bits_amc("):].split("\n", 1)[1]
body = "\n".join(("        " + l) if l.strip() else "" for l in body.splitlines())
wrapped = f'''import allo.dataflow as df
from allo.ir.types import UInt, uint1, uint16
N = {NN}


@df.region()
def top(A: uint16[N], B: uint16[N], C: uint16[N]):
    @df.kernel(mapping=[1], args=[A, B, C])
    def add(av: uint16[N], bv: uint16[N], cv: uint16[N]):
{body}
'''
wpath = f"{out}/region_amcedits_src.py"
open(wpath, "w").write(wrapped)
try:
    W = load("region_amcedits_src", wpath)
    s = df.customize(W.top)
    open(f"{out}/region_amcedits.mlir", "w").write(str(s.module))
    print("region_amcedits.mlir written")
except Exception:
    traceback.print_exc()

# region_amcedits_unroll / region_unroll: the leading-zero loop fully
# unrolled by our own schedule (AMC asserts on loops inside ifs, finding A9);
# for region_unroll the nested function is inlined first.
for nm, mk in [("region_amcedits_unroll", lambda: W.top), ("region_unroll", lambda: bf16_add.bits(NN))]:
    try:
        s = df.customize(mk())
        if nm == "region_unroll":
            s.inline("leading_zeros17")
        loops = s.get_loops("add_0")
        print(nm, loops)
        try:
            s.unroll(loops["S_i_0"]["offset"])
        except Exception:
            traceback.print_exc()
        s.pipeline(loops["S_i_0"]["i"])
        open(f"{out}/{nm}.mlir", "w").write(str(s.module))
        print(f"{nm}.mlir written")
    except Exception:
        traceback.print_exc()

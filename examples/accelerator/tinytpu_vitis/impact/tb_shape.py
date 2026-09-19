"""Write <run>/isa_sweep.prj/tb.cpp for one shape of one variant (for profile.sh).
    python pyrun.py tb_shape.py <variant|base> <run dir> M K N"""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples.accelerator.tinytpu_vitis import cosim as C  # noqa: E402
name, run, M, K, N = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
if name == "base":
    from examples.accelerator.tinytpu_vitis import microarch_isa as V
    from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program
    V.gemm_program = gemm_program
else:
    V = importlib.import_module(name)
for k in ("assemble", "gemm_program", "IMEM_SIZE", "MAXDIM", "T"):
    setattr(C, k, getattr(V, k))
open(os.path.join(run, "isa_sweep.prj", "tb.cpp"), "w").write(C.testbench(M, K, N))

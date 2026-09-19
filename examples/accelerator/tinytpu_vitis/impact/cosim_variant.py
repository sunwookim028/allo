"""Cosim one design VARIANT on the same flow `cosim.py` uses, in its own dir.

    python pyrun.py cosim_variant.py <variant module> <run dir>
    e.g.  TPU_SHAPES=4x4x4,16x16x16 python pyrun.py cosim_variant.py base runs/base

`<variant module>` is a file in this directory, or `base` = the shipped
`microarch_isa` -- since the landing, the design the stack became; the
pre-landing baseline the attribution table measures against is `v_base`. It must export `tinytpu_isa`, `assemble`, `schedule`,
`gemm_program`, `IMEM_SIZE`, `MAXDIM`, `T`. Everything else -- testbench, m_axi
depth patch, Tcl, `-B/usr/bin`, align_value 64, widen 512 -- is `cosim.py`'s,
unchanged, so the only thing that differs between two runs is the design.
"""
import importlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from examples.accelerator.tinytpu_vitis import cosim as C  # noqa: E402

name, run = sys.argv[1], os.path.abspath(sys.argv[2])
if name == "base":
    from examples.accelerator.tinytpu_vitis import microarch_isa as V
    from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program
    V.gemm_program = gemm_program
else:
    V = importlib.import_module(name)
for k in ("tinytpu_isa", "assemble", "schedule", "gemm_program",
          "IMEM_SIZE", "MAXDIM", "T"):
    setattr(C, k, getattr(V, k))
if hasattr(V, "patch_kernel"):
    _depths = C.patch_axi_depths
    def _patched(prj):
        _depths(prj)
        V.patch_kernel(prj)
    C.patch_axi_depths = _patched
os.makedirs(run, exist_ok=True)
os.chdir(run)
print(f"variant={name} module={V.__file__} run={run}", flush=True)
sys.exit(C.main())

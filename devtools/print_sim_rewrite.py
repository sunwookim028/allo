# The KEY simulator intermediate: the module AFTER build_dataflow_simulator
# rewrites streams into ring buffers + OpenMP, but BEFORE the LLVM passes.
# This is what LLVMOMPModule.__init__ produces at simulator.py:1532, captured
# before the PassManager lowering runs.
#
#   !allo.stream<i32,4>   ->  struct{ memref<5xi32> buffer, memref<i32> head, memref<i32> tail }
#   stream_put            ->  ring store + scf.while spin-until-not-full + omp.critical tail bump
#   stream_get            ->  ring load  + scf.while spin-until-not-empty + omp.critical head bump
#   the two kernels       ->  omp.parallel > omp.sections > omp.section  (concurrent threads)
#
# Usage:  OMP_NUM_THREADS=4 conda run -n allo python print_sim_rewrite.py
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")

from allo._mlir.ir import Context, Module
from allo._mlir.dialects import allo as allo_d
from allo.backend.simulator import build_dataflow_simulator
from allo.passes import decompose_library_function
import allo.dataflow as df
from stream_producer_consumer import top

s = df.customize(top)
with Context() as ctx:
    allo_d.register_dialect(ctx)
    m = Module.parse(str(s.module), ctx)
    m = decompose_library_function(m)
    build_dataflow_simulator(m, "top")     # stream -> ring buffer + OMP  (no LLVM yet)
    print(m)

# Minimal dataflow stream example: producer --pipe--> consumer.
# One int32 FIFO of depth 4 carries M*N elements from producer to consumer.
#
# Backends (run this file, or import `top` and call the emitters below):
#   MLIR (frontend)   : allo.dataflow.customize(top).module
#   Vitis/Vivado HLS  : allo.dataflow.build(top, target="vitis_hls" | "vivado_hls")
#   Catapult (ASIC)   : allo.dataflow.build(top, target="catapult")
#   SystemC           : HLSModule(..., platform="systemc")  (see driver script)
#   TAPA / Intel HLS  : allo.dataflow.build(top, target="tapa" | "ihls")
#   XLS               : allo.dataflow.build(top, target="xls")
#   LLVM CPU simulator: allo.dataflow.build(top, target="simulator")
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

M, N = 2, 2
Ty = int32


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            pipe.put(local_A[i, j])

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            local_B[i, j] = pipe.get()


if __name__ == "__main__":
    import numpy as np

    mod = df.build(top, target="simulator")
    a = np.ones((M, N), dtype=np.int32)
    b = np.zeros((M, N), dtype=np.int32)
    mod(a, b)
    print("simulator output:\n", b)

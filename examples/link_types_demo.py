# One region exercising all three dataflow link types side by side:
#   fifo : Stream[int32, 4]          FIFO (buffered)          -> !allo.stream<i32, 4>
#   wire : Wire[int32]               combinational (no buffer)-> !allo.wire<i32>
#   chan : Channel[int32, valid_ready] handshake (no buffer)  -> !allo.channel<i32, valid_ready>
#
# The producer writes each incoming value to all three links; the consumer reads
# all three back and sums them. Run this file to print the lowered MLIR.
import allo
from allo.ir.types import int32, Stream, Wire, Channel, valid_ready
import allo.dataflow as df

M, N = 2, 2
Ty = int32


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    fifo: Stream[Ty, 4]
    wire: Wire[Ty]
    chan: Channel[Ty, valid_ready]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            fifo.put(local_A[i, j])
            wire.put(local_A[i, j])
            chan.put(local_A[i, j])

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            x: Ty = fifo.get()
            y: Ty = wire.get()
            z: Ty = chan.get()
            local_B[i, j] = x + y + z


if __name__ == "__main__":
    # full pipeline (frontend + interface lifting)
    print(df.customize(top).module)

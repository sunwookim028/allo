# Minimal dataflow WIRE example: producer --wire--> consumer.
# A combinational wire (unbuffered, no handshake) carries M*N int32 values
# from producer to consumer.
#
# Wire is a first-class dataflow link type (alongside Stream / Channel):
#   pipe: Wire[int32]         -> !allo.wire<i32>
#   pipe.put(x) / pipe.get()  -> allo.wire_put / allo.wire_get
#
# STATUS: this lowers to frontend MLIR (wire_construct/wire_put/wire_get).
# End-to-end lowering -- interface lifting (move_stream_to_interface/_build_top)
# and the Catapult ac_signal emitter -- is still in progress. Inspect the
# frontend MLIR with:
#     from allo.dataflow import _customize
#     print(_customize(top).module)
import allo
from allo.ir.types import int32, Wire
import allo.dataflow as df

M, N = 2, 2
Ty = int32


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    pipe: Wire[Ty]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            pipe.put(local_A[i, j])

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            local_B[i, j] = pipe.get()


if __name__ == "__main__":
    from allo.dataflow import _customize

    # raw builder output (before interface lifting)
    print(_customize(top).module)

import os, shutil, sys
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

N = 8
@df.region()
def top(A: int32[N], B: int32[N]):
    fifo: Stream[int32, 4][1]
    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[N]):
        for i in range(N):
            fifo[0].put(a[i])
    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[N]):
        for i in range(N):
            b[i] = fifo[0].get() + 1

d = sys.argv[1]
if os.path.exists(d): shutil.rmtree(d)
os.makedirs(d)
df.build(top, target="systemc", mode="csyn", project=d)
with open(os.path.join(d, "input0.data"), "w") as f:
    f.write("".join(f"{i}\n" for i in range(N)))
with open(os.path.join(d, "golden_output0.data"), "w") as f:
    f.write("".join(f"{i+1}\n" for i in range(N)))
print(sorted(os.listdir(d)))

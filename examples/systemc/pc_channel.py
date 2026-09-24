# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal producer/consumer over a valid_ready Channel — a thesis-sized example.

A `Channel[int32, valid_ready]` is a buffer-free link with a full valid/ready
handshake. It lowers to a MatchLib `Connections::Combinational<T>` bound to a
`Connections::Out` on the producer and a `Connections::In` on the consumer;
`put` -> `.Push()`, `get` -> `.Pop()`, both blocking on the handshake.

The region `pc_channel` is importable so the Catapult helper can synthesize it:

    python csyn_subdir.py pc_channel pc_channel        # csynth (RTL)

Standalone usage (dump / simulate):
    conda activate allo
    export OMP_NUM_THREADS=8

    python pc_channel.py mlir      # print the frontend MLIR
    python pc_channel.py systemc   # print the generated SystemC
    python pc_channel.py csim      # build + run csim, check B == A   (default)

`csim` additionally needs a SystemC library to link against. Catapult ships one, but its
libsystemc wants a newer libstdc++ than the system one -- without the second line here the
build succeeds and then dies at run time with GLIBCXX_3.4.26 (see notes/ALLO_GOTCHAS.md):

    export SYSTEMC_HOME=$MGC_HOME/shared
    export ALLO_CXX_EXTRA="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"

Or skip both and run the project's self-contained `csim.sh`, which uses Catapult's own g++
and bundled SystemC and needs only MGC_HOME.
"""
import io
import os
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")   # edit-this-checkout, not the installed pkg
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import allo
import allo.dataflow as df
from allo._mlir.dialects import allo as allo_d
from allo.ir.types import int32, Channel, valid_ready

N = 4


@df.region()
def pc_channel(A: int32[N], B: int32[N]):
    ch: Channel[int32, valid_ready]          # handshake link, no buffer

    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[N]):
        for i in range(N):
            ch.put(a[i])

    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[N]):
        for i in range(N):
            b[i] = ch.get()


def print_mlir():
    print(df.customize(pc_channel).module)


def print_systemc():
    buf = io.StringIO()
    ok = allo_d.emit_systemc(df.customize(pc_channel).module, buf)
    buf.seek(0)
    print(buf.read() if ok else "<emit_systemc returned False>")


def run_csim(prj="pc_prj"):
    a = np.arange(N, dtype=np.int32)
    b = np.zeros(N, dtype=np.int32)
    mod = df.build(pc_channel, target="systemc", mode="csim", project=prj)
    mod(a, b)
    ok = bool((b == a).all())
    print(f"  pc_channel  B={b}  expected={a}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "csim"
    if what == "mlir":
        print_mlir()
    elif what == "systemc":
        print_systemc()
    else:
        raise SystemExit(0 if run_csim() else 1)

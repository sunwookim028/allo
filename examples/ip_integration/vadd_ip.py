# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A hand-written HLS IP as the datapath of a small programmable accelerator.

The design is a tiny instruction-driven vector machine:

    sequencer (Allo)  --a_s/b_s-->  vadd (HLS C++ IP)  --out_s-->  sequencer

The sequencer is written in Allo. It holds a four-instruction memory, decodes
one instruction per cycle of its loop, and for a ``VADD`` streams both operand
vectors out and reads the sums back. The datapath is *not* written in Allo: it
is the hand-written ``vadd.cpp`` next to this file, integrated with
``allo.IPModule``, and its ports are ``hls::stream<T>`` rather than arrays.

That last part is the point of this example. A stream-ported IP participates in
the dataflow region as a peer of the Allo kernels: under
``target="simulator"`` it runs in its own thread and its ``read()``/``write()``
calls drive Allo's FIFOs directly, and under ``target="vitis_hls"`` it is
stitched in through Vitis's own ``hls::stream``. The same source works for both.
See ``docs/IP_STREAM_SIM_SHIM.md`` and ``docs/IP_STREAM_INTEGRATION.md``.

Run it with::

    OMP_NUM_THREADS=8 python3 vadd_datapath_ip.py

``OMP_NUM_THREADS`` must be at least the number of kernels in the region: the
simulator gives each kernel its own thread, and the IP blocks on its ports.
"""

from pathlib import Path

import numpy as np

import allo
import allo.dataflow as df
from allo.backend import hls
from allo.ir.types import int1, int8, int32, Stateful, Stream

N = 32  # Vector length
IMEM_SIZE = 4  # Length of program
INSTR_W = 4  # Size of one instruction
NEXT = 2  # Number of external memory slots
NIBUF = 2  # Number of operands/input buffer size

# OPcodes
LOAD, STORE, VADD, NOP = 0, 1, 2, 3
RUN_PROG, LOAD_INSTR = 0, 1
HALT, GO = 0, 1

# The datapath, as a hand-written HLS IP. `input_idx`/`output_idx` declare the
# direction of each port: an `hls::stream<T>&` looks identical in C++ whether
# the IP reads or writes it, so the direction has to be stated here.
vadd = allo.IPModule(
    top="vadd",
    impl=Path(__file__).resolve().parent / "vadd.cpp",
    input_idx=[0, 1, 2],  # enable_s, a_s, b_s
    output_idx=[3],  # out_s
)

# The sequencer, as a hand-written HLS IP. Ports 0-4 are arrays; ports 5-8 are
# streams, whose direction cannot be read off the C++ signature and so is
# declared here: the IP writes enable_s/a_s/b_s and reads out_s.
sequencer_ip = allo.IPModule(
    top="sequencer",
    impl=Path(__file__).resolve().parent / "sequencer.cpp",
    input_idx=[8],  # out_s
    output_idx=[5, 6, 7],  # enable_s, a_s, b_s
)


@df.region()
def top(
    ctrl: int8[1],  # Tells the sequencer whether to load instructions or send them
    d_addr: int32[1],  # Destination address (instructions)
    prog_in: int8[INSTR_W],  # External input (program)
    ext_in: int32[N * NEXT],  # External input (data)
    ext_out: int32[N],
):  # External output (to host as memory)

    a_s: Stream[int32, N]  # input stream 1
    b_s: Stream[int32, N]  # input stream 2
    out_s: Stream[int32, N]  # output stream
    enable_s: Stream[int1, 1]  # Enable for the datapath


    # Sequencer: the hand-written IP, in its own kernel so that it gets its own
    # concurrent process. Note the instruction memory lives inside the IP (as a
    # C `static`), not in the Allo region.
    @df.kernel(mapping=[1], args=[ctrl, d_addr, prog_in, ext_in, ext_out])
    def sequencer(
        ctrl_in: int8[1],
        d_addr_in: int32[1],
        instr_in: int8[INSTR_W],
        ext_in_p: int32[NEXT * N],
        ext_out_p: int32[N],
    ):
        sequencer_ip(
            ctrl_in, d_addr_in, instr_in, ext_in_p, ext_out_p, enable_s, a_s, b_s, out_s
        )

    # Datapath: the hand-written IP, in its own kernel so that it gets its own
    # concurrent process. It blocks on its stream ports, so it must run
    # alongside the sequencer rather than before or after it.
    @df.kernel(mapping=[1])
    def datapath():
        vadd(enable_s, a_s, b_s, out_s)


def test_vadd():
    """Load a four-instruction program, run it, and check the result.

    The program loads two vectors from external memory, adds them with the IP,
    and stores the sum back out. Instructions are fed in one at a time, each in
    its own invocation, and land in the region's stateful instruction memory.
    """
    A = np.random.randint(-10, 10, N, dtype=np.int32)
    B = np.random.randint(-10, 10, N, dtype=np.int32)
    C = np.zeros(N, dtype=np.int32)
    dmem = np.concatenate([A, B])

    program = [
        [LOAD, 0, 0, 0],
        [LOAD, 1, 1, 0],
        [VADD, 0, 0, 1],
        [STORE, 0, 0, 0],
    ]

    print("Starting simulation")
    sim_mod = df.build(top, target="simulator")

    print("Loading program")
    for pc, instr in enumerate(program):  # Load the instructions into local imem
        ctrl = np.array([LOAD_INSTR], dtype=np.int8)
        d_addr = np.array([pc], dtype=np.int32)
        prog_in = np.array(instr, dtype=np.int8)
        sim_mod(ctrl, d_addr, prog_in, dmem, C)
        print("Instruction", pc, "loaded:", instr)

    # Run the program
    print("Running the program")
    ctrl = np.array([RUN_PROG], dtype=np.int8)
    d_addr = np.array([0], dtype=np.int32)
    prog_in = np.zeros(INSTR_W, dtype=np.int8)
    sim_mod(ctrl, d_addr, prog_in, dmem, C)
    print("Run complete")

    np.testing.assert_allclose(C, np.add(A, B), atol=1e-5)
    print("A:", A, "\nB:", B, "\nC:", C)
    print("Simulation passed")

    # The same design also builds for the FPGA flow, where the IP is stitched in
    # through Vitis's hls::stream instead of the simulator's ring buffers.
    if hls.is_available("vitis_hls"):
        print("Building HLS project")
        df.build(top, target="vitis_hls", mode="csim", project="vadd_ip.prj")
        print("HLS project built")
    else:
        print("vitis_hls not found on PATH; skipping the HLS build")


if __name__ == "__main__":
    test_vadd()

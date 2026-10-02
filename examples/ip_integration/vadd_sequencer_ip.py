# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same accelerator with the split the other way round: the IP is the control.

    sequencer (HLS C++ IP)  --a_s/b_s-->  datapath (Allo)  --out_s-->  sequencer

This is the mirror of ``vadd_datapath_ip.py``. There, the Allo kernel drove a
hand-written datapath; here the hand-written ``sequencer.cpp`` holds the
instruction memory and the decode loop, and the vector add is the Allo kernel.

It is worth having both, because this direction exercises something the other
does not: an IP with a **mixed** interface. ``sequencer`` takes five ordinary
array ports (control, address, program, and the two external memories) *and*
four ``hls::stream<T>&`` ports. Allo passes the arrays as pointers and the
streams as ring buffers, in one call.

Run it with::

    OMP_NUM_THREADS=8 python3 vadd_sequencer_ip.py

See ``docs/IP_STREAM_SIM_SHIM.md`` for how the stream ports are wired up on the
CPU, and ``docs/IP_STREAM_INTEGRATION.md`` for the FPGA path.
"""

from pathlib import Path

import numpy as np

import allo
import allo.dataflow as df
from allo.backend import hls
from allo.ir.types import int1, int8, int32, Stream

N = 32  # Vector length
IMEM_SIZE = 4  # Length of program
INSTR_W = 4  # Size of one instruction
NEXT = 2  # Number of external memory slots
NIBUF = 2  # Number of operands/input buffer size

# OPcodes
LOAD, STORE, VADD, NOP = 0, 1, 2, 3
RUN_PROG, LOAD_INSTR = 0, 1
HALT, GO = 0, 1

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

    # Datapath (pure vector addition)
    @df.kernel(mapping=[1])
    def datapath():
        a: int32 = 0
        b: int32 = 0
        enable: int1 = enable_s.get()
        if enable == GO:
            for i in range(N):
                a = a_s.get()
                b = b_s.get()
                out_s.put(a + b)


def test_vadd():
    """Load a four-instruction program, run it, and check the result.

    Identical driver to ``vadd_datapath_ip.py`` -- the two designs are
    interchangeable from the host's point of view, which is the whole idea:
    which side is hand-written C++ and which side is Allo does not change the
    interface.
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

    if hls.is_available("vitis_hls"):
        print("Building HLS project")
        df.build(top, target="vitis_hls", mode="csim", project="vadd_sequencer_ip.prj")
        print("HLS project built")
    else:
        print("vitis_hls not found on PATH; skipping the HLS build")


if __name__ == "__main__":
    test_vadd()

import allo
from allo.ir.types import int8, int32, Stream, Stateful
import allo.dataflow as df
import numpy as np
print("Imports finished")

N = 32
IMEM_SIZE = 4 # Length of program
CMD_W = 4
INSTR_W = 4 # Size of one instruction
NEXT = 2 # Number of external memory slots
NIBUF = 2 # Number of operands/input buffer size
LOAD, STORE, VADD, NOP = 0, 1, 2, 3
LOAD_INSTR = 1
RUN_PROG = 0

@df.region()
def top(ctrl: int8[1], # Tells the sequencer whether to load instructions or send them
        d_addr: int32[1], # 
        prog_in: int32[INSTR_W], # External input (program)
        ext_in:  int32[N * NEXT], # External input (data)
        ext_out: int32[N]): # External output (to host as memory)

    imem: int32[IMEM_SIZE * INSTR_W] @ Stateful = 0
    cmd_s: Stream[int32[CMD_W], INSTR_W] 

    # Sequencer (PC loop, extract OPcodes and progress through the program)
    @df.kernel(mapping=[1], args=[ctrl, prog_in, d_addr])
    def sequencer(ctrl_in: int8[1],
                  instr_in: int32[INSTR_W],
                  d_addr_in: int32[1]): 
        cmd: int32[CMD_W] = 0
        if ctrl_in[0] == LOAD_INSTR:
            base: int32 = d_addr_in[0] * INSTR_W
            for w in range(INSTR_W):
                imem[base + w] = instr_in[w]
            cmd[0] = NOP
            cmd[1] = 0
            cmd[2] = 0
            cmd[3] = 0
            for _pc in range(IMEM_SIZE):   
                cmd_s.put(cmd)
        else:
            for pc in range(IMEM_SIZE):
                base2: int32 = pc * INSTR_W
                cmd[0] = imem[base2]
                cmd[1] = imem[base2 + 1]
                cmd[2] = imem[base2 + 2]
                cmd[3] = imem[base2 + 3]
                cmd_s.put(cmd)
            
        


    # Datapath (fetch, decode, execute)
    @df.kernel(mapping=[1], args=[ext_in, ext_out, d_addr])
    def datapath(ext_in_p: int32[NEXT * N], 
                 ext_out_p: int32[N],
                 d_addr_in: int32[1]):
        in_buf: int32[NIBUF * N] = 0
        out_buf: int32[N] = 0
        for _ in range(IMEM_SIZE):
            cmd: int32[CMD_W] = cmd_s.get()
            op: int32 = cmd[0]
            f1: int32 = cmd[1]
            f2: int32 = cmd[2]
            f3: int32 = cmd[3]

            if op == LOAD:
                for i in range(N):
                    in_buf[f1 * N + i] = ext_in_p[f2 * N + i]
            elif op == STORE:
                for i in range(N):
                    ext_out_p[f1 * N + i + d_addr_in] = out_buf[f2 * N + i ]
            elif op == VADD:
                for i in range(N):
                    out_buf[f1 * N + i] = in_buf[f2 * N + i] + in_buf[f3 * N + i]


def test_vadd():
    np_type_A = np.int32
    np_type_B = np.int32
    np_type_dmem = np.int32

    A = np.random.randint(-2, 2, N, dtype=np.int32)
    B = np.random.randint(-2, 2, N, dtype=np.int32)
    C = np.zeros(N, dtype=np.int32)
    C2 = np.zeros(N, dtype=np.int32)
    dmem = np.concatenate([A, B])
    
    program = [
        [LOAD, 0, 0, 0],
        [LOAD, 1, 1, 0],
        [VADD, 0, 0, 1],
        [STORE, 0, 0, 0]
    ]

    print("Starting simulation")
    sim_mod = df.build(top, target="simulator")

    for pc, instr in enumerate(program): # Load all of the instructions into imem
        ctrl = np.array([LOAD_INSTR], dtype=np.int8)
        d_addr = np.array([pc], dtype=np.int32)
        prog_in = np.array(instr, dtype=np.int32)
        sim_mod(ctrl, d_addr, prog_in, dmem, C)
    
    # Run the program
    ctrl = np.array([RUN_PROG], dtype=np.int8)
    d_addr = np.array([0], dtype=np.int32)
    prog_in = np.zeros(INSTR_W, dtype=np.int32)
    sim_mod(ctrl, d_addr, prog_in, dmem, C)

    np.testing.assert_allclose(C, np.add(A, B), atol=1e-5)
    print(f"A: ", A, "\n" "B: ", B, "\n", "C: ", C) 
    print("Simulation passed")

    # print("Running csim")
    # mod = df.build(top, target="vitis_hls", mode="csim", project="vadd.prj")
    # mod(imem, dmem, C2)
    # np.testing.assert_allclose(C2, np.add(A, B), atol=1e-5)
    # print("csim passed")

if __name__ == "__main__":
    test_vadd()